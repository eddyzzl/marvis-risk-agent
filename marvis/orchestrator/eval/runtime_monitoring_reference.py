"""Independent fixed-window model monitoring arithmetic and threshold policy.

Unknown labels remain n/a. Thresholds and baselines are supplied explicitly;
this module does not read platform monitoring calculators or policy defaults.
"""
import math

from .runtime_modeling_reference import binary_reference, distribution_reference, psi_reference

CHECK_IDS = {"score_psi", "feature_csi_max", "ks_drop", "auc_drop"}


def validate_thresholds(thresholds):
    if set(thresholds) != CHECK_IDS:
        raise ValueError("complete explicit monitoring thresholds required")
    for key, item in thresholds.items():
        if (set(item) != {"label", "metric", "direction", "warn", "fail"}
                or item["metric"] != key or item["direction"] not in {"min", "max"}
                or not isinstance(item["label"], str) or not item["label"]):
            raise ValueError("monitoring threshold schema mismatch")
        for name in ("warn", "fail"):
            value = item[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)):
                raise ValueError("monitoring threshold must be finite")
        if item["warn"] is not None and item["fail"] is not None:
            if item["direction"] == "max" and item["warn"] > item["fail"] or item["direction"] == "min" and item["warn"] < item["fail"]:
                raise ValueError("monitoring warn/fail order invalid")
    return thresholds


def threshold_reference(value, spec):
    if value is None or not math.isfinite(float(value)):
        return "amber", "missing"
    if spec["warn"] is None and spec["fail"] is None:
        return "amber", "needs_policy"
    predicate = (lambda x, bound: x < bound) if spec["direction"] == "min" else (lambda x, bound: x > bound)
    for name, level in (("fail", "red"), ("warn", "amber")):
        if spec[name] is not None and predicate(value, spec[name]):
            return level, name
    return "green", "pass"


def monitoring_reference(frame, scores, *, features, target, baseline, development_metrics, thresholds):
    validate_thresholds(thresholds)
    if len(scores) != len(frame) or any(not math.isfinite(float(v)) for v in scores):
        raise ValueError("monitoring source/prediction row contract mismatch")
    rows, drifted = [], []
    def check(key, value, **extra):
        level, status = threshold_reference(value, thresholds[key])
        return {"id": key, "label": thresholds[key]["label"], "metric": key,
                "value": value, "level": level, "status": status, **extra}
    edges, train = baseline.get("score_edges", []), ((baseline.get("score_distribution") or {}).get("train") or {}).get("bin_proportions")
    if len(edges) >= 2 and train and scores:
        psi = psi_reference(train, distribution_reference(scores, edges))
        rows.append(check("score_psi", psi, sample_count=len(scores)))
    else:
        rows.append(check("score_psi", None))
    for feature in features:
        stored = (baseline.get("feature_distributions") or {}).get(feature)
        if not isinstance(stored, dict) or feature not in frame:
            continue
        edges = stored.get("quantile_edges", [])
        if len(edges) < 2:
            continue
        values = []
        for v in frame[feature].tolist():
            try:
                number = float(v)
                if math.isfinite(number):
                    values.append(number)
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        expected = stored.get("bin_proportions")
        if not isinstance(expected, list) or len(expected) != len(edges) - 1 or sum(expected) <= 0:
            expected = [1 / (len(edges) - 1)] * (len(edges) - 1)
        csi = psi_reference(expected, distribution_reference(values, edges))
        drifted.append({"feature": feature, "csi": csi, "sample_count": len(values)})
    drifted.sort(key=lambda r: (-r["csi"], r["feature"]))
    rows.append(check("feature_csi_max", drifted[0]["csi"] if drifted else None,
                      top_feature=drifted[0]["feature"] if drifted else None))
    pairs = []
    if target is not None and target in frame:
        for score, label in zip(scores, frame[target].tolist(), strict=True):
            try:
                number = float(label)
                if math.isfinite(number) and number in (0, 1):
                    pairs.append((score, int(number)))
            except (TypeError, ValueError):
                continue
    if {p[1] for p in pairs} != {0, 1}:
        rows.extend({"id": key, "label": thresholds[key]["label"], "metric": key,
            "value": None, "level": "n/a", "status": "n/a"} for key in ("ks_drop", "auc_drop"))
    else:
        live = binary_reference([p[0] for p in pairs], [p[1] for p in pairs], direction_agnostic=True)
        for key, metric in (("ks_drop", "ks"), ("auc_drop", "auc")):
            dev = development_metrics.get(f"train_{metric}")
            value = None if dev is None else dev - live[metric]
            rows.append(check(key, value, live_value=live[metric], dev_value=dev, sample_count=len(pairs)))
    levels = {r["level"] for r in rows}
    overall = "red" if "red" in levels else "amber" if "amber" in levels else "green"
    recommendation = {"green": "分布稳定,可继续按监控计划执行。",
        "red": "存在显著漂移或性能下降,建议触发模型风险复核或重训评估。",
        "amber": "存在轻中度漂移或指标缺口,建议加强观察频率并复核相关特征。"}[overall]
    return {"checks": rows, "top_drifted_features": drifted[:10], "overall_level": overall,
            "recommendation": recommendation, "row_count": len(frame)}
