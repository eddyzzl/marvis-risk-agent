"""Independent references for calibration, ensembles, CatBoost, MLP, scorecards,
sample weights and the calibration fold.

No ``marvis.packs``/``marvis.feature``/``marvis.validation`` code is imported and
no learner ``predict``/``predict_proba`` is called: fitted parameters are read as
numbers and evaluated here. numpy is used only to reproduce the declared
``RandomState`` membership stream.
"""
import hashlib
import math
import struct
from itertools import pairwise

from .runtime_archive_reader import _Unsupported


def _sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


# --- calibration -----------------------------------------------------------

def calibration_rows_reference(labels, scores, *, n_bins, score_type=""):
    p = [min(1.0, max(0.0, float(s))) for s in scores]
    y = [float(v) for v in labels]
    rows = []
    for index in range(n_bins):
        # numpy.linspace edges: i * (1 / n) with the last edge forced to 1.0.
        lower = index * (1.0 / n_bins)
        upper = 1.0 if index == n_bins - 1 else (index + 1) * (1.0 / n_bins)
        last = index == n_bins - 1
        members = [i for i, value in enumerate(p) if lower <= value and (value <= upper if last else value < upper)]
        if not members:
            continue
        avg = math.fsum(p[i] for i in members) / len(members)
        rate = math.fsum(y[i] for i in members) / len(members)
        rows.append({"score_type": score_type, "bin": index + 1, "prob_lower": lower, "prob_upper": upper,
                     "sample_count": len(members), "positive_count": sum(y[i] == 1.0 for i in members),
                     "avg_predicted_pd": avg, "observed_bad_rate": rate,
                     "calibration_gap": avg - rate, "abs_gap": abs(avg - rate)})
    return rows


def calibration_metrics_reference(labels, scores, *, n_bins):
    p = [min(1.0, max(0.0, float(s))) for s in scores]
    brier = math.fsum((a - float(b)) ** 2 for a, b in zip(p, labels, strict=True)) / len(p)
    rows = calibration_rows_reference(labels, p, n_bins=n_bins)
    total = sum(r["sample_count"] for r in rows)
    ece = math.fsum(r["sample_count"] / total * abs(r["calibration_gap"]) for r in rows) if total else 0.0
    return {"brier": brier, "ece": ece}


def apply_sigmoid_reference(coefficient, intercept, scores):
    return [min(1.0, max(0.0, _sigmoid(coefficient * float(s) + intercept))) for s in scores]


def apply_isotonic_reference(x_thresholds, y_thresholds, scores):
    """Linear interpolation between fitted thresholds, clipped at the ends."""
    xs, ys = [float(v) for v in x_thresholds], [float(v) for v in y_thresholds]
    if len(xs) != len(ys) or not xs or any(b < a for a, b in pairwise(xs)):
        raise ValueError("isotonic thresholds must be aligned and increasing")
    result = []
    for raw in scores:
        s = float(raw)
        if s <= xs[0]:
            value = ys[0]
        elif s >= xs[-1]:
            value = ys[-1]
        else:
            hi = next(i for i, x in enumerate(xs) if x >= s)
            lo = hi - 1
            value = ys[hi] if xs[hi] == s else ys[lo] + (ys[hi] - ys[lo]) * (s - xs[lo]) / (xs[hi] - xs[lo])
        result.append(min(1.0, max(0.0, value)))
    return result


def calibration_fold_membership(row_count, seed, *, fraction=0.15):
    """Row positions of the default calibration fit fold (per-row groups)."""
    import numpy as np

    def derive(value, name):
        return int(hashlib.sha256(f"{int(value)}:{name}".encode()).hexdigest()[:8], 16) % 2_147_483_647

    groups = np.arange(row_count)
    np.random.RandomState(derive(derive(seed, "calibration_fold"), "valid_fold")).shuffle(groups)
    target, chosen = round(row_count * fraction), []
    for group in groups.tolist():
        if len(chosen) >= target or len(chosen) >= row_count - 1:
            break
        chosen.append(group)
    if not chosen or len(chosen) == row_count:
        raise _Unsupported("calibration fold carve is empty")
    return sorted(chosen)


# --- ensembles, CatBoost, MLP ---------------------------------------------

def ensemble_reference(member_predictions, weights=None):
    if not member_predictions or len({len(m) for m in member_predictions}) != 1:
        raise ValueError("ensemble members must score the same rows")
    weights = [1.0 / len(member_predictions)] * len(member_predictions) if weights is None else [float(w) for w in weights]
    total = math.fsum(weights)
    if len(weights) != len(member_predictions) or total <= 0:
        raise ValueError("invalid ensemble weights")
    return [math.fsum(w * float(m[i]) for w, m in zip(weights, member_predictions, strict=True)) / total
            for i in range(len(member_predictions[0]))]


def catboost_json_reference(model, rows):
    """Evaluate a CatBoost ``format='json'`` export of an oblivious float-feature model."""
    trees = model.get("oblivious_trees")
    features = (model.get("features_info") or {}).get("float_features")
    if not trees or features is None or (model.get("features_info") or {}).get("categorical_features"):
        raise _Unsupported("CatBoost profile requires float-only oblivious trees")
    scale, bias = model.get("scale_and_bias", [1.0, [0.0]])
    if isinstance(bias, list) and len(bias) != 1:
        raise _Unsupported("multi-dimensional CatBoost output unsupported")
    bias = bias[0] if isinstance(bias, list) else bias
    if len(trees[0]["leaf_values"]) != 2 ** len(trees[0].get("splits") or []):
        raise _Unsupported("multi-dimensional CatBoost output unsupported")
    by_index = {f["feature_index"]: f for f in features}
    for f in features:
        if f.get("nan_value_treatment", "AsIs") not in {"AsIs", "AsFalse", "AsTrue", "Min", "Max"}:
            raise _Unsupported("CatBoost NaN treatment outside independent subset")
    result = []
    for row in rows:
        margin = []
        for tree in trees:
            index = 0
            for depth, split in enumerate(tree.get("splits") or []):
                if split.get("split_type", "FloatFeature") != "FloatFeature":
                    raise _Unsupported("non-float CatBoost split")
                value = float(row[split["float_feature_index"]])
                if math.isnan(value):
                    treatment = by_index[split["float_feature_index"]].get("nan_value_treatment", "AsIs")
                    if treatment == "AsIs":
                        raise _Unsupported("CatBoost AsIs NaN requires engine semantics")
                    passed = treatment in {"Max", "AsTrue"}
                else:
                    # CatBoost compares float32 feature values with float32 borders.
                    passed = struct.unpack("f", struct.pack("f", value))[0] > split["border"]
                index |= int(passed) << depth
            margin.append(tree["leaf_values"][index])
        result.append(_sigmoid(scale * math.fsum(margin) + bias))
    return result


_ACTIVATIONS = {"relu": lambda v: max(0.0, v), "tanh": math.tanh, "logistic": _sigmoid, "identity": lambda v: v}


def mlp_reference(params, rows):
    """``params``: medians, means, scales, coefs (layers of [in][out]), intercepts, activation."""
    activation = _ACTIVATIONS.get(params["activation"])
    if activation is None or params.get("out_activation", "logistic") != "logistic":
        raise _Unsupported("MLP activation outside independent subset")
    result = []
    for row in rows:
        values = []
        for value, median, mean, scale in zip(row, params["medians"], params["means"], params["scales"], strict=True):
            value = median if value is None or math.isnan(float(value)) else float(value)
            values.append((value - mean) / (scale or 1.0))
        for layer, (weights, intercepts) in enumerate(zip(params["coefs"], params["intercepts"], strict=True)):
            out = [math.fsum([b] + [values[i] * weights[i][j] for i in range(len(values))]) for j, b in enumerate(intercepts)]
            values = out if layer == len(params["coefs"]) - 1 else [activation(v) for v in out]
        if len(values) != 1:
            raise _Unsupported("multi-output MLP unsupported")
        result.append(_sigmoid(values[0]))
    return result


def sklearn_mlp_params(pipeline):
    """Read fitted numbers from an imputer/scaler/MLP pipeline without predicting."""
    impute, scale, mlp = (pipeline.named_steps[k] for k in ("impute", "scale", "mlp"))
    if impute.strategy != "median" or getattr(mlp, "n_outputs_", 1) != 1:
        raise _Unsupported("MLP preprocessing profile unsupported")
    return {"medians": [float(v) for v in impute.statistics_], "means": [float(v) for v in scale.mean_],
            "scales": [float(v) for v in scale.scale_], "activation": mlp.activation, "out_activation": mlp.out_activation_,
            "coefs": [w.tolist() for w in mlp.coefs_], "intercepts": [b.tolist() for b in mlp.intercepts_]}


# --- scorecard and weights ------------------------------------------------

def scorecard_points_reference(intercept, coefficients, woe_rows, *, base_score=600, pdo=50, base_odds=50.0):
    """``woe_rows``: [(feature, bin_index, woe)] -> base points and per-bin points."""
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    return {"factor": factor, "offset": offset, "base_points": offset - factor * float(intercept),
            "points": [{"feature": f, "bin_index": b, "points": -factor * float(coefficients[f]) * float(w)}
                       for f, b, w in woe_rows]}


def weighted_binary_reference(scores, labels, weights):
    groups = {}
    for s, y, w in zip(scores, labels, weights, strict=True):
        s, w = float(s), float(w)
        if not (math.isfinite(s) and math.isfinite(w)) or y not in (0, 1) or w < 0:
            continue
        cell = groups.setdefault(s, [0.0, 0.0])
        cell[int(y)] += w
    good = math.fsum(g for g, _ in groups.values())
    bad = math.fsum(b for _, b in groups.values())
    if good <= 0 or bad <= 0:
        return {"ks": 0.0, "auc": 0.5}
    pairs, seen_good, seen_bad, ks = 0.0, 0.0, 0.0, 0.0
    for score in sorted(groups):
        g, b = groups[score]
        pairs += b * seen_good + 0.5 * b * g
        seen_good += g
        seen_bad += b
        ks = max(ks, abs(seen_bad / bad - seen_good / good))
    return {"ks": ks, "auc": pairs / (bad * good)}


def _weighted_distribution(scores, weights, edges):
    from bisect import bisect_right

    counts = [0.0] * (len(edges) - 1)
    for score, weight in zip(scores, weights, strict=True):
        score, weight = float(score), float(weight)
        if math.isfinite(score) and math.isfinite(weight):
            counts[min(bisect_right(edges, score, 1, len(edges) - 1), len(edges) - 1) - 1] += max(weight, 0.0)
    total = math.fsum(counts)
    return [c / total for c in counts] if total > 0 else [0.0] * len(counts)


def weighted_split_metrics_reference(profiles, weights, edges):
    """``profiles``/``weights``: {role: (scores, labels or None)} / {role: weights}; train edges."""
    from .runtime_modeling_reference import psi_reference

    result = {}
    for role in ("train", "test", "oot"):
        scores, labels = profiles[role]
        values = None if labels is None or not scores else weighted_binary_reference(scores, labels, weights[role])
        result[f"weighted_{role}_ks"] = None if values is None else values["ks"]
        result[f"weighted_{role}_auc"] = None if values is None else values["auc"]
    expected = _weighted_distribution(profiles["train"][0], weights["train"], edges)
    for role in ("test", "oot"):
        scores = profiles[role][0]
        result[f"weighted_psi_{role}_vs_train"] = (None if not scores else
            psi_reference(expected, _weighted_distribution(scores, weights[role], edges)))
    return result
