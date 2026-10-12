"""Independent model arithmetic and explicit row-membership contracts.

This module does not import production metrics, recipes, scoring or tuning.
Undefined evaluation is None; a fitted population is never a held-out test.
"""
from bisect import bisect_right
from collections import defaultdict
import hashlib
import math
import statistics

import numpy as np

from .runtime_archive_reader import _Unsupported


def binary_reference(scores, labels, *, direction_agnostic=False):
    if len(scores) != len(labels):
        raise ValueError("score/label row mismatch")
    groups = defaultdict(lambda: [0, 0])
    for score, label in zip(scores, labels, strict=True):
        if not math.isfinite(float(score)) or label not in (0, 1):
            raise ValueError("binary reference requires finite scores and binary labels")
        groups[float(score)][int(label)] += 1
    good = sum(v[0] for v in groups.values())
    bad = sum(v[1] for v in groups.values())
    if not good or not bad:
        return {"ks": 0.0, "auc": .5}
    cumulative_good = cumulative_bad = 0
    pairs = ks = 0.0
    for score in sorted(groups):
        g, b = groups[score]
        pairs += b * (cumulative_good + g / 2)
        cumulative_good += g
        cumulative_bad += b
        ks = max(ks, abs(cumulative_good / good - cumulative_bad / bad))
    auc = pairs / (good * bad)
    return {"ks": ks, "auc": max(auc, 1 - auc) if direction_agnostic else auc}


def quantile_reference(values, probability):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("empty quantile population")
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def edges_reference(values, bins=10):
    values = [float(v) for v in values if math.isfinite(float(v))]
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return [-math.inf, math.inf]
    if len(distinct) == 2:
        return [-math.inf, sum(distinct) / 2, math.inf]
    edges = sorted(set(quantile_reference(values, i / bins) for i in range(bins + 1)))
    return [-math.inf, *edges[1:-1], math.inf]


def distribution_reference(values, edges):
    if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("invalid fixed bucket edges")
    counts = [0] * (len(edges) - 1)
    for value in values:
        if math.isfinite(float(value)):
            counts[bisect_right(edges[1:-1], float(value))] += 1
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def psi_reference(expected, observed):
    if len(expected) != len(observed) or not expected:
        raise ValueError("PSI bucket contract mismatch")
    if any(not math.isfinite(float(v)) or v < 0 for v in [*expected, *observed]):
        raise ValueError("invalid bucket proportions")
    left = [max(float(v), 1e-6) for v in expected]
    right = [max(float(v), 1e-6) for v in observed]
    left = [v / sum(left) for v in left]
    right = [v / sum(right) for v in right]
    return max(0.0, math.fsum((b - a) * math.log(b / a) for a, b in zip(left, right, strict=True)))


def regression_reference(predictions, labels):
    if not predictions or len(predictions) != len(labels):
        raise ValueError("regression population mismatch")
    if any(not math.isfinite(float(v)) for v in [*predictions, *labels]):
        raise ValueError("regression requires finite observations")
    residual = [float(y) - float(p) for y, p in zip(labels, predictions, strict=True)]
    squared = math.fsum(v * v for v in residual)
    mean = statistics.mean(labels)
    variance = math.fsum((v - mean) ** 2 for v in labels)
    return {"rmse": math.sqrt(squared / len(labels)), "mae": statistics.mean(abs(v) for v in residual),
            "r2": 1 - squared / variance if variance > 0 else 0.0}


def multiclass_reference(probabilities, labels, classes):
    classes = list(classes)
    if len(classes) < 3 or len(set(classes)) != len(classes) or len(probabilities) != len(labels):
        raise ValueError("multiclass class/row contract mismatch")
    if any(label not in classes for label in labels):
        raise ValueError("multiclass unknown label")
    for row in probabilities:
        if (len(row) != len(classes) or any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in row)
                or not math.isclose(sum(row), 1, abs_tol=1e-6)):
            raise ValueError("multiclass invalid probability vector")
    probabilities = [[float(v) / math.fsum(row) for v in row] for row in probabilities]
    if not labels:
        return {"macro_auc": None, "logloss": None, "accuracy": None}
    aucs = [binary_reference([row[i] for row in probabilities], [int(v == cls) for v in labels])["auc"]
            for i, cls in enumerate(classes)]
    # A one-vs-rest AUC is undefined when any declared class is absent.
    macro = statistics.mean(aucs) if set(labels) == set(classes) else None
    eps = np.finfo(np.float64).eps
    loss = -statistics.mean(math.log(max(eps, min(1 - eps, row[classes.index(label)])))
                            for row, label in zip(probabilities, labels, strict=True))
    accuracy = statistics.mean(classes[max(range(len(classes)), key=lambda i: row[i])] == label
                               for row, label in zip(probabilities, labels, strict=True))
    return {"macro_auc": macro, "logloss": loss, "accuracy": accuracy}


def membership_reference(frame, config, *, development=False):
    column, values, target = config["split_col"], config["split_values"], config["target_col"]
    if set(values) != {"train", "test", "oot"} or len(set(values.values())) != 3:
        raise ValueError("split contract must declare distinct train/test/OOT values")
    if column not in frame or target not in frame:
        raise ValueError("split/target column missing")
    target_type = "regression" if config.get("target_type") == "continuous" else config.get("target_type", "binary")
    labels = frame[target].tolist()
    masks = {}
    for role, literal in values.items():
        mask = [v == literal for v in frame[column].tolist()]
        if role != "oot":
            for i, chosen in enumerate(mask):
                if not chosen:
                    continue
                label = labels[i]
                missing = label is None or isinstance(label, (float, np.floating)) and math.isnan(label)
                if missing and config.get("drop_nan_labels", False):
                    mask[i] = False
                elif missing:
                    raise ValueError("missing development label without approved exclusion")
                elif target_type == "binary" and label not in (0, 1):
                    raise ValueError("invalid development binary label")
                elif target_type == "regression" and not math.isfinite(float(label)):
                    raise ValueError("invalid regression development label")
        masks[role] = mask
    if not any(masks["train"]) or not any(masks["test"]):
        raise ValueError("development train/test population required")
    if development:
        masks["development"] = [a or b for a, b in zip(masks["train"], masks["test"], strict=True)]
    return masks


def lift_reference(scores, labels):
    keys = {f"lift_{end}_{pct}": None for pct in (5, 10) for end in ("head", "tail")}
    if len(labels) < 20 or not sum(labels):
        return keys
    sx, sy = statistics.mean(scores), statistics.mean(labels)
    covariance = math.fsum((x - sx) * (y - sy) for x, y in zip(scores, labels, strict=True))
    sign = -1 if covariance < 0 else 1
    order = sorted(range(len(labels)), key=lambda i: sign * scores[i])
    for pct in (5, 10):
        n = max(1, math.floor(pct * len(labels) / 100))
        for end, positions in (("head", order[-n:]), ("tail", order[:n])):
            keys[f"lift_{end}_{pct}"] = statistics.mean(labels[i] for i in positions) / statistics.mean(labels)
    return keys


def bootstrap_reference(scores, labels, seed):
    point = binary_reference(scores, labels)["ks"]
    bad, good = np.flatnonzero(np.asarray(labels) == 1), np.flatnonzero(np.asarray(labels) == 0)
    if len(bad) < 2 or len(good) < 2:
        return {"low": point, "high": point, "std": 0.0, "n_boot": 0}
    first = int(hashlib.sha256(f"{seed}:ks_ci".encode()).hexdigest()[:8], 16) % 2_147_483_647
    derived = int(hashlib.sha256(f"{first}:bootstrap_ks".encode()).hexdigest()[:8], 16) % 2_147_483_647
    rng = np.random.RandomState(derived)
    draws = []
    for _ in range(200):
        positions = [*bad[rng.randint(0, len(bad), size=len(bad))], *good[rng.randint(0, len(good), size=len(good))]]
        draws.append(binary_reference([scores[i] for i in positions], [labels[i] for i in positions])["ks"])
    return {"low": quantile_reference(draws, .025), "high": quantile_reference(draws, .975),
            "std": statistics.stdev(draws), "n_boot": 200}


def metrics_reference(frame, config, predict, *, development=False, classes=()):
    masks = membership_reference(frame, config, development=development)
    profiles = {}
    for role in ("development",) if development else ("train", "test", "oot"):
        positions = [i for i, yes in enumerate(masks[role]) if yes]
        if not positions:
            profiles[role] = {"scores": [], "labels": None, "metrics": None}
            continue
        part = frame.iloc[positions]
        labels = part[config["target_col"]].tolist()
        scores = np.asarray(predict(part)).tolist()
        eligible = all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in labels)
        kind = "regression" if config.get("target_type") == "continuous" else config.get("target_type", "binary")
        if role == "oot" and not eligible:
            metrics = None
        elif kind == "binary":
            metrics = binary_reference(scores, labels)
        elif kind == "regression":
            metrics = regression_reference(scores, labels)
        elif kind == "multiclass":
            metrics = multiclass_reference(scores, labels, classes)
        else:
            raise _Unsupported("model target profile requires an independent reference")
        profiles[role] = {"scores": scores, "labels": labels if eligible else None, "metrics": metrics}
    kind = "regression" if config.get("target_type") == "continuous" else config.get("target_type", "binary")
    result = {}
    for role, values in profiles.items():
        prefix = "train" if role == "development" else role
        for metric in ({"binary": ("ks", "auc"), "regression": ("rmse", "mae", "r2"),
                        "multiclass": ("macro_auc", "logloss", "accuracy")}[kind]):
            result[f"{prefix}_{metric}"] = None if values["metrics"] is None else values["metrics"][metric]
    if development:
        result.update({key: None for key in ("test_ks", "test_auc", "oot_ks", "oot_auc", "psi_test_vs_train",
            "psi_oot_vs_train", "overfit_train_test_gap", "overfit_train_oot_gap", "overfit_flag")})
    elif config.get("target_type", "binary") == "binary":
        train, test, oot = [profiles[r] for r in ("train", "test", "oot")]
        edges = edges_reference(train["scores"])
        expected = distribution_reference(train["scores"], edges)
        result["psi_test_vs_train"] = psi_reference(expected, distribution_reference(test["scores"], edges))
        result["psi_oot_vs_train"] = psi_reference(expected, distribution_reference(oot["scores"], edges)) if oot["scores"] else None
        gap = abs(result["train_ks"] - result["test_ks"]) / abs(result["train_ks"]) if abs(result["train_ks"]) > 1e-12 else 0.0
        gap_oot = abs(result["train_ks"] - result["oot_ks"]) if result["oot_ks"] is not None else None
        result.update(overfit_train_test_gap=gap, overfit_train_oot_gap=gap_oot,
                      overfit_flag=gap > .1 or gap_oot is not None and gap_oot > .05)
        for role in ("test", "oot"):
            item = profiles[role]
            lifts = lift_reference(item["scores"], item["labels"]) if item["labels"] is not None else {
                f"lift_{end}_{pct}": None for pct in (5, 10) for end in ("head", "tail")}
            result.update({f"{role}_{key}": value for key, value in lifts.items()})
            ci = bootstrap_reference(item["scores"], item["labels"], config["seed"]) if item["labels"] is not None else None
            result.update({f"{role}_ks_ci_{key}": None if ci is None else ci[key] for key in ("low", "high", "std")})
            if role == "test":
                result["ks_ci_n_boot"] = ci["n_boot"]
    else:
        key = "rmse" if kind == "regression" else "macro_auc"
        train, test, oot = [result.get(f"{role}_{key}") for role in ("train", "test", "oot")]
        def gap(observed):
            if observed is None or train is None:
                return None
            return observed - train if kind == "regression" else train - observed
        result.update(train_ks=None, test_ks=None, oot_ks=None, train_auc=None, test_auc=None, oot_auc=None,
            psi_test_vs_train=None, psi_oot_vs_train=None, overfit_train_test_gap=gap(test),
            overfit_train_oot_gap=gap(oot), overfit_flag=False)
    return result, masks
