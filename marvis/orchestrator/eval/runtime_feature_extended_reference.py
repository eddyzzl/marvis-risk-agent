"""Independent feature PSI, VIF, model importance and bin-detail references.

No ``marvis.feature``/``marvis.packs`` calculator is imported. PSI/VIF/binning
are standard-library arithmetic. Importance must retrain the same third-party
learner (LightGBM) and is marked ``library_dependent``: gains are summed from the
dumped tree structure, never read from ``Booster.feature_importance``.
"""
from bisect import bisect_right
import math
import re

from .runtime_archive_reader import _Unsupported
from .runtime_validation_independent import equal_frequency_edges, psi

VIF_CAP = 1e9
IMPORTANCE_PROFILE = {"objective": "binary", "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31,
                      "random_state": 42, "n_jobs": 1, "deterministic": True, "force_col_wise": True,
                      "verbose": -1, "train_frac": 0.7}
_SPLIT_ROLES = {"train": {"train", "training", "dev", "develop", "development", "build"},
                "test": {"test", "testing", "valid", "validation", "val", "holdout"},
                "oot": {"oot", "ootest", "out_of_time", "oos", "time_oot"}}


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _distribution(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        if math.isfinite(value):
            counts[min(bisect_right(edges, value, 1, len(edges) - 1), len(edges) - 1) - 1] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def psi_reference(base, compare, *, bins=10):
    base, compare = [_float(v) for v in base], [_float(v) for v in compare]
    edges = equal_frequency_edges(base, bins)
    expected, actual = _distribution(base, edges), _distribution(compare, edges)
    return {"psi": psi(expected, actual), "edges": edges,
            "bin_distributions": {"base": expected, "compare": actual}}


def split_role(value):
    text = str(value).strip().lower()
    return next((role for role, names in _SPLIT_ROLES.items() if text in names), None)


def psi_pairs(labels, view):
    groups = sorted({str(v) for v in labels if v is not None})
    if len(groups) < 2:
        return []
    if view == "psi_split":
        return [] if "train" not in groups else [("train", r) for r in ("test", "oot") if r in groups]
    if view == "psi_month_first":
        return [(groups[0], g) for g in groups[1:]]
    if view == "psi_month_last":
        return [(groups[-1], g) for g in groups[:-1]]
    if view in {"psi_mom", "psi_month_previous"}:
        return list(zip(groups[:-1], groups[1:], strict=True))
    raise _Unsupported("PSI view requires its independent pairing rule")


def psi_series_reference(values, labels, view, *, bins=10):
    """Per-pair PSI and the max headline; month labels are supplied pre-normalized."""
    values = [_float(v) for v in values]
    if view == "psi_split":
        labels = [split_role(v) for v in labels]
    rows = []
    for base_label, compare_label in psi_pairs(labels, view):
        base = [v for v, label in zip(values, labels, strict=True) if label == base_label]
        compare = [v for v, label in zip(values, labels, strict=True) if label == compare_label]
        if sum(map(math.isfinite, base)) < 2 or sum(map(math.isfinite, compare)) < 2:
            continue
        rows.append({"base": base_label, "compare": compare_label, "psi": psi_reference(base, compare, bins=bins)["psi"]})
    return {"series": rows, "psi": max((r["psi"] for r in rows), default=None)}


def month_labels_reference(values):
    """Normalize numeric/ISO calendar values without the producer's pandas helper.

    The feature contract preserves compact YYYYMM labels for six/eight digit
    dates, including separated dates. Other nonempty labels remain literal.
    """
    import pandas as pd

    labels = []
    parsed = pd.to_datetime(pd.Series(values), errors="coerce")
    for value, date in zip(values, parsed, strict=True):
        if pd.isna(value):
            labels.append(None)
            continue
        text = str(value).strip()
        digits = re.sub(r"\D+", "", text)
        labels.append(digits[:6] if len(digits) in {6, 8}
                      else date.strftime("%Y-%m") if not pd.isna(date) else text or None)
    return labels


def pearson_reference(values, target):
    pairs = [(_float(v), _float(y)) for v, y in zip(values, target, strict=True)]
    pairs = [(v, y) for v, y in pairs if math.isfinite(v) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    xbar = math.fsum(v for v, _ in pairs) / len(pairs)
    ybar = math.fsum(y for _, y in pairs) / len(pairs)
    cross = math.fsum((v - xbar) * (y - ybar) for v, y in pairs)
    xx = math.fsum((v - xbar) ** 2 for v, _ in pairs)
    yy = math.fsum((y - ybar) ** 2 for _, y in pairs)
    return cross / math.sqrt(xx * yy) if xx and yy else 0.0


def head_tail_lift_reference(values, target, *, min_rows=1):
    pairs = [(_float(v), _float(y)) for v, y in zip(values, target, strict=True)]
    pairs = [(v, y) for v, y in pairs if math.isfinite(v) and y in {0.0, 1.0}]
    n = len(pairs)
    base = sum(y for _, y in pairs) / n if n else 0.0
    result = {f"lift_{end}_{pct}": None for pct in (5, 10) for end in ("head", "tail")}
    if n >= min_rows and base:
        sign = -1 if pearson_reference([v for v, _ in pairs], [y for _, y in pairs]) < 0 else 1
        ordered = sorted(pairs, key=lambda pair: sign * pair[0])
        for pct in (5, 10):
            count = max(1, math.floor(pct * n / 100))
            result[f"lift_head_{pct}"] = sum(y for _, y in ordered[-count:]) / count / base
            result[f"lift_tail_{pct}"] = sum(y for _, y in ordered[:count]) / count / base
    result["lift_reason"] = (f"有效标注样本仅 {n} 行，5%/10% 极端分位切片很小，结果仅供方向参考。" if n < 20 else "")
    return result


def meaning_consistency_reference(values, target, *, feature, meaning, decision):
    """Compare the retained semantic decision with independent sample direction.

    This verifies deterministic consistency, never the truth of an LLM's
    business interpretation. The caller authenticates the dictionary and binds
    the original decision separately.
    """
    corr = pearson_reference(values, target)
    expected = str(decision.get("expected_direction") or "uncertain")
    if expected not in {"positive", "negative", "u_shape", "uncertain"}:
        expected = "uncertain"
    actual = "positive" if corr > .03 else "negative" if corr < -.03 else "uncertain"
    evidence, shape_reason = None, ""
    if expected == "u_shape":
        analysis = bin_analysis_reference(values, target, feature=feature, requested_bins=7)
        rows = [row for row in analysis["rows"] if row["interval"] != "缺失值"]
        rates, counts = [r["bad_rate"] for r in rows], [r["count"] for r in rows]
        evidence = {"bin_bad_rates": rates, "bin_counts": counts}
        actual = "uncertain"
        minimum = max(3, math.ceil(sum(counts) * .02))
        if len(rates) < 3:
            shape_reason = "有效分箱少于 3 个，无法判断 U 型走势。"
        elif any(count < minimum for count in counts):
            shape_reason = f"存在样本量低于 {minimum} 的分箱，U 型判断不稳定。"
        else:
            trough = min(range(len(rates)), key=rates.__getitem__)
            amplitude = min(rates[0] - rates[trough], rates[-1] - rates[trough])
            left = [b - a for a, b in zip(rates[:trough], rates[1:trough + 1])]
            right = [b - a for a, b in zip(rates[trough:-1], rates[trough + 1:])]
            shaped = (bool(left) and bool(right) and amplitude >= .03
                      and sum(v <= .01 for v in left) / len(left) >= .67
                      and sum(v >= -.01 for v in right) / len(right) >= .67)
            actual = "u_shape" if shaped else "uncertain"
            evidence.update(trough_bin=trough + 1, minimum_end_gap=amplitude,
                            requested_bins=7, actual_bins=analysis["actual_bins"])
            shape_reason = "分箱坏率呈两端高、中间低的 U 型走势。" if shaped else "分箱坏率未形成两端高、中间低的稳定 U 型走势。"
    rationale = str(decision.get("rationale") or "").strip()
    if expected == "uncertain" and not rationale:
        rationale = "没有可用的受约束 LLM 语义判向，按保守策略不作方向判断。"
    return {"business_meaning": meaning, "expected_direction": expected, "actual_direction": actual,
            "direction_correlation": corr,
            "meaning_consistency": "需人工看" if "uncertain" in {expected, actual} else "一致" if expected == actual else "不一致",
            "meaning_consistency_reason": shape_reason or (f"业务预期={expected}，实际方向={actual}。" if expected != "uncertain" else rationale),
            "meaning_judgement_source": str(decision.get("judgement_source") or "no_llm_fallback"),
            "meaning_direction_confidence": str(decision.get("confidence") or "low"),
            "meaning_direction_rationale": rationale, "meaning_direction_model": str(decision.get("model") or ""),
            "meaning_direction_prompt": {"name": str(decision.get("prompt_name") or "feature_meaning_direction"),
                                         "version": int(decision.get("prompt_version") or 1)},
            "u_shape_evidence": evidence}


def _least_squares_r2(columns, y):
    """Householder QR least squares with intercept; dependent columns are dropped."""
    n = len(y)
    mean = math.fsum(y) / n
    total = math.fsum((v - mean) ** 2 for v in y)
    if total == 0:
        return 0.0
    matrix = [[1.0] * n] + [list(c) for c in columns]
    residual = list(y)
    basis = []
    for column in matrix:
        vector = list(column)
        for q in basis:
            dot = math.fsum(a * b for a, b in zip(q, vector, strict=True))
            vector = [a - dot * b for a, b in zip(vector, q, strict=True)]
        for q in basis:  # second pass: re-orthogonalize for stability
            dot = math.fsum(a * b for a, b in zip(q, vector, strict=True))
            vector = [a - dot * b for a, b in zip(vector, q, strict=True)]
        norm = math.sqrt(math.fsum(v * v for v in vector))
        scale = math.sqrt(math.fsum(v * v for v in column)) or 1.0
        if norm <= 1e-10 * scale:
            continue
        q = [v / norm for v in vector]
        basis.append(q)
        dot = math.fsum(a * b for a, b in zip(q, residual, strict=True))
        residual = [a - dot * b for a, b in zip(residual, q, strict=True)]
    resid = math.fsum(v * v for v in residual)
    return max(0.0, min(1.0, 1 - resid / total))


def vif_reference(rows, features):
    result = {f: 0.0 for f in features}
    table = {f: [_float(r.get(f)) for r in rows] for f in features}
    usable = [f for f in features if any(not math.isnan(v) for v in table[f])]
    complete = [i for i in range(len(rows)) if all(not math.isnan(table[f][i]) for f in usable)]
    if not complete or len(usable) < 2:
        return result
    if len(complete) < max(30, 2 * len(usable)):
        return {**result, **{f: None for f in usable}}
    for feature in usable:
        y = [table[feature][i] for i in complete]
        others = [[table[o][i] for i in complete] for o in usable if o != feature]
        r2 = _least_squares_r2(others, y)
        result[feature] = VIF_CAP if r2 >= 1 else min(VIF_CAP, 1.0 / (1.0 - r2))
    return result


def importance_reference(frame, features, target_col):
    """Library-dependent retrain; returns (importance, profile_scope)."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import train_test_split

    target = frame[target_col].to_numpy(dtype=float)
    labelled = np.isfinite(target)
    design, y = frame.loc[labelled, features], target[labelled].astype(int)
    if design.empty or len(set(y.tolist())) < 2:
        return {f: None for f in features}
    profile = dict(IMPORTANCE_PROFILE)
    fraction = profile.pop("train_frac")
    try:
        x_train, _, y_train, _ = train_test_split(design, y, train_size=fraction,
                                                  random_state=profile["random_state"], stratify=y)
    except ValueError:
        x_train, y_train = design, y
    if len(set(y_train.tolist())) < 2:
        return {f: None for f in features}
    model = lgb.LGBMClassifier(**profile).fit(x_train, y_train)
    dump = model.booster_.dump_model()
    names = dump["feature_names"]
    gains = {name: [] for name in names}

    def visit(node):
        if "split_index" in node:
            gains[names[node["split_feature"]]].append(float(node["split_gain"]))
            visit(node["left_child"])
            visit(node["right_child"])

    for tree in dump["tree_info"]:
        visit(tree["tree_structure"])
    totals = {name: math.fsum(values) for name, values in gains.items()}
    total = math.fsum(totals.values())
    if total <= 0:
        return {f: 0.0 for f in features}
    return {f: totals.get(f, 0.0) / total for f in features}


def _number_text(value):
    if value == -math.inf:
        return "-inf"
    if value == math.inf:
        return "inf"
    return str(int(value)) if float(value).is_integer() else f"{float(value):.6g}"


def _pearson(x, y):
    n = len(x)
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return math.nan
    return math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(sxx * syy)


def bin_analysis_reference(values, target, *, feature, requested_bins):
    values, target = [_float(v) for v in values], [_float(v) for v in target]
    pairs = [(v, int(t)) for v, t in zip(values, target, strict=True) if math.isfinite(t)]
    if not pairs or any(t not in (0, 1) for _, t in pairs) or len({t for _, t in pairs}) < 2:
        raise ValueError("bin analysis target must contain both binary classes")
    edges = equal_frequency_edges(values, requested_bins)
    groups = list(range(len(edges) - 1))
    assigned = [(min(bisect_right(edges, v, 1, len(edges) - 1), len(edges) - 1) - 1) if math.isfinite(v) else -1
                for v, _ in pairs]
    if -1 in assigned:
        groups.append(-1)
    total_bad = sum(t for _, t in pairs)
    total_good = len(pairs) - total_bad
    cells, total_iv = [], []
    for group in groups:
        labels = [t for (_, t), g in zip(pairs, assigned, strict=True) if g == group]
        bad, count = sum(labels), len(labels)
        good = count - bad
        bad_dist = (bad + .5) / (total_bad + .5 * len(groups))
        good_dist = (good + .5) / (total_good + .5 * len(groups))
        woe = math.log(good_dist / bad_dist)
        iv = (good_dist - bad_dist) * woe
        total_iv.append(iv)
        cells.append({"group": group, "count": count, "bad": bad, "good": good, "bad_rate": bad / count if count else 0.0,
                      "woe": woe, "iv": iv,
                      "lower": math.nan if group == -1 else edges[group], "upper": math.nan if group == -1 else edges[group + 1]})
    overall = total_bad / len(pairs)
    order = sorted(range(len(cells)), key=lambda i: (-cells[i]["bad_rate"], i))
    cumulative, cum_bad, cum_good = {}, 0, 0
    for rank, index in enumerate(order, start=1):
        cum_bad += cells[index]["bad"]
        cum_good += cells[index]["good"]
        rate = cum_bad / (cum_bad + cum_good) if cum_bad + cum_good else 0.0
        cumulative[index] = {"risk_rank": rank, "cumulative_bad_rate": rate,
                             "cumulative_lift": rate / overall if overall else 0.0,
                             "ks": abs(cum_bad / total_bad - cum_good / total_good)}
    rows = []
    for index, cell in enumerate(cells):
        if cell["group"] == -1:
            interval = "缺失值"
        else:
            interval = f"{'(' if cell['lower'] == -math.inf else '['}{_number_text(cell['lower'])}, {_number_text(cell['upper'])})"
        rows.append({"bin_index": index + 1, "risk_rank": cumulative[index]["risk_rank"], "interval": interval,
                     "count": cell["count"], "bad_count": cell["bad"], "good_count": cell["good"],
                     "bad_rate": cell["bad_rate"], "cumulative_bad_rate": cumulative[index]["cumulative_bad_rate"],
                     "lift": cell["bad_rate"] / overall if overall else 0.0,
                     "cumulative_lift": cumulative[index]["cumulative_lift"], "ks": cumulative[index]["ks"],
                     "woe": cell["woe"], "iv_contribution": cell["iv"]})
    real = [c["bad_rate"] for c in cells if c["group"] != -1]
    monotonic = len(real) <= 2 or all(a <= b for a, b in zip(real, real[1:])) or all(a >= b for a, b in zip(real, real[1:]))
    finite = [(v, float(t)) for v, t in zip(values, target, strict=True) if math.isfinite(v) and math.isfinite(t)]
    direction = "unknown"
    if len(finite) >= 2 and len({v for v, _ in finite}) >= 2 and len({t for _, t in finite}) >= 2:
        corr = _pearson([v for v, _ in finite], [t for _, t in finite])
        if math.isfinite(corr):
            direction = "risk_up" if corr >= 0 else "risk_down"
    actual = len(edges) - 1
    degraded = "" if actual >= requested_bins else (
        f"分箱降级:请求 {requested_bins} 个箱,实际只得到 {actual} 个(单一取值或重复值折叠了分位边界)。")
    return {"feature": str(feature), "requested_bins": int(requested_bins), "actual_bins": actual,
            "direction": direction, "total_iv": round(math.fsum(total_iv), 6), "monotonic": monotonic,
            "degraded_reason": degraded, "rows": rows}


def correlation_reference(rows, features, *, threshold=0.8):
    """Pairwise-complete Pearson matrix (undefined -> 0, diagonal 1), pairs and VIF."""
    table = {f: [_float(r.get(f)) for r in rows] for f in features}
    matrix = []
    for left in features:
        line = []
        for right in features:
            if left == right:
                line.append(1.0)
                continue
            pairs = [(a, b) for a, b in zip(table[left], table[right], strict=True) if not (math.isnan(a) or math.isnan(b))]
            value = _pearson([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 2 else math.nan
            line.append(value if math.isfinite(value) else 0.0)
        matrix.append(line)
    pairs = [[features[i], features[j], matrix[i][j]] for i in range(len(features)) for j in range(i + 1, len(features))
             if abs(matrix[i][j]) >= threshold]
    return {"features": list(features), "matrix": matrix, "collinear_pairs": pairs, "vif": vif_reference(rows, features)}
