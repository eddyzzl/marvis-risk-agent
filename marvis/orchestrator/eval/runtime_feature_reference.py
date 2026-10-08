"""Independent arithmetic for the default numeric feature-analysis contract.

Quantiles use explicit interpolation, AUC counts ordered good/bad pairs by
score group, KS compares complete tied-score groups, and IV counts source rows.
No production feature calculator, NumPy quantile, rankdata or WOE kernel is used.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import math
import statistics

from .runtime_archive_reader import _Unsupported


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _finite(value):
    return value is not None and math.isfinite(float(value))


def feature_reference(values, target, *, feature, selected=("iv", "ks", "auc", "coverage"), bins=10):
    selected = set(selected)
    if not selected or selected - {"iv", "ks", "auc", "coverage"}:
        raise _Unsupported("selected feature metric requires its independent calculator")
    if type(bins) is not int or not 2 <= bins <= 100:
        raise ValueError("invalid feature bin count")
    data = [float(value) if _finite(value) else None for value in values]
    if target is not None and (len(target) != len(data) or any(not _finite(v) or float(v) not in {0, 1} for v in target)):
        raise ValueError("feature target must be complete binary labels")
    labels = None if target is None else [int(v) for v in target]
    finite = sorted(value for value in data if value is not None)
    row = {"feature": feature}
    count = len(finite)
    if "coverage" in selected:
        row.update(coverage=count / len(data) if data else 0.0,
            missing_rate=(len(data) - count) / len(data) if data else 0.0,
            valid_count=count, unique_count=len(set(finite)), unique_rate=len(set(finite)) / count if count else 0.0,
            mode_rate=max(Counter(finite).values()) / count if count else 0.0,
            zero_rate=finite.count(0) / count if count else 0.0,
            mean=statistics.mean(finite) if count else None,
            std=statistics.stdev(finite) if count > 1 else 0.0 if count else None,
            min=finite[0] if count else None, max=finite[-1] if count else None,
            q25=_quantile(finite, .25) if count else None,
            median=_quantile(finite, .5) if count else None, q75=_quantile(finite, .75) if count else None)
    if selected & {"iv", "ks", "auc"} and labels is None:
        raise _Unsupported("missing-target feature recommendations require a separate reference")
    groups = defaultdict(lambda: [0, 0])
    if labels is not None:
        for value, label in zip(data, labels, strict=True):
            if value is not None:
                groups[value][label] += 1
        good = sum(counts[0] for counts in groups.values())
        bad = sum(counts[1] for counts in groups.values())
        cumulative_good, cumulative_bad, auc_pairs, ks = 0, 0, 0.0, 0.0
        for value in sorted(groups):
            group_good, group_bad = groups[value]
            auc_pairs += group_bad * (cumulative_good + group_good / 2)
            cumulative_good += group_good
            cumulative_bad += group_bad
            if good and bad:
                ks = max(ks, abs(cumulative_good / good - cumulative_bad / bad))
        if "ks" in selected:
            row["ks"] = ks
        if "auc" in selected:
            auc = auc_pairs / (good * bad) if good and bad else .5
            row["auc"] = max(auc, 1 - auc)
        if "iv" in selected:
            if len(set(labels)) != 2:
                raise _Unsupported("single-class IV requires explicit unavailable-result verification")
            unique = sorted(set(finite))
            if len(unique) == 2:
                breaks = [(unique[0] + unique[1]) / 2]
            elif len(unique) > 2:
                quantiles = sorted(set(_quantile(finite, i / bins) for i in range(bins + 1)))
                breaks = quantiles[1:-1]
            else:
                breaks = []
            cells = [[0, 0] for _ in range(len(breaks) + 1 + int(None in data))]
            for value, label in zip(data, labels, strict=True):
                index = len(cells) - 1 if value is None else bisect_right(breaks, value)
                cells[index][label] += 1
            good, bad = labels.count(0), labels.count(1)
            contributions = []
            for group_good, group_bad in cells:
                good_share = (group_good + .5) / (good + .5 * len(cells))
                bad_share = (group_bad + .5) / (bad + .5 * len(cells))
                contributions.append((good_share - bad_share) * math.log(good_share / bad_share))
            row["iv"] = round(math.fsum(contributions), 6)
    return row


def feature_recommendation_reference(metric):
    """Check the default no-PSI policy separately from feature arithmetic."""
    evidence = [{"metric": key, "value": metric[key]} for key in
                ("iv", "ks", "auc", "missing_rate", "mode_rate", "zero_rate", "unique_count") if key in metric]
    if "coverage" in metric and metric["valid_count"] == 0:
        label, reason = "不推荐", "没有有效数值，无法用于分析。"
    elif "coverage" in metric and metric["missing_rate"] >= .8:
        label, reason = "不推荐", f"缺失率 {metric['missing_rate']:.1%} 过高。"
    elif "coverage" in metric and (metric["unique_count"] <= 1 or metric["mode_rate"] >= .95):
        label, reason = "不推荐", f"单一值率 {metric['mode_rate']:.1%}，变量几乎没有区分信息。"
    elif not ({"iv", "ks", "auc"} & metric.keys()):
        label, reason = "待评估", "本次未选择 IV、KS 或 AUC，无法判断单变量区分力。"
    elif not any(metric.get(key, -1) >= threshold for key, threshold in (("iv", .1), ("ks", .1), ("auc", .6))):
        label, reason = "暂不推荐", "本次所选区分力指标未显示出足够信号。"
    else:
        prefix = "样本量偏小，建议扩大样本复核；" if metric.get("valid_count", 0) < 100 else ""
        label, reason = "候选", prefix + "有区分力；本次没有可用 PSI 结果，需另行验证稳定性。"
    confidence = "none" if label == "待评估" else "medium" if ({"iv", "ks", "auc"} & metric.keys() or label in {"不推荐", "暂不推荐"}) else "low"
    return {"recommendation": label, "recommendation_reason": reason,
            "recommendation_state": {"候选": "candidate", "待评估": "unevaluated", "不推荐": "not_recommended", "暂不推荐": "not_recommended"}[label],
            "recommendation_evidence": evidence, "recommendation_confidence": confidence}
