"""Independent portfolio stability trend and Vintage references.

No ``marvis.packs.analysis``/``marvis.packs.strategy``/``marvis.validation``
code is imported; PSI, levels, month gaps and Vintage accumulation are written
out from the declared contracts.
"""
from bisect import bisect_right
import math
import re

from .runtime_validation_independent import psi

PSI_WARN, PSI_FAIL = 0.10, 0.25
SNAPSHOT_FLAG = (
    "per-MOB bad_count non-decreasing across all cohorts -- data looks like a "
    "SNAPSHOT flag (每列/行=截至该MOB是否曾坏, 已单调); declared=incremental, "
    "so cum_bad_rate re-accumulates and may double-count. Confirm label_semantics."
)


def level(value):
    if value is None:
        return "n/a"
    return "red" if value >= PSI_FAIL else "amber" if value >= PSI_WARN else "green"


def _finite(values):
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _distribution(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        counts[min(bisect_right(edges, value, 1, len(edges) - 1), len(edges) - 1) - 1] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def month_gap_flags(months):
    parsed = []
    for month in months:
        text = str(month)
        if not (len(text) >= 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit()):
            return []
        parsed.append((int(text[:4]), int(text[5:7])))
    gaps = [f"{a[0]:04d}-{a[1]:02d} -> {b[0]:04d}-{b[1]:02d}"
            for a, b in zip(parsed, parsed[1:]) if (b[0] * 12 + b[1]) - (a[0] * 12 + a[1]) != 1]
    return [{"kind": "month_gap", "gaps": gaps, "message": f"趋势月份不连续：{'; '.join(gaps)}。"}] if gaps else []


def score_trend_reference(score_edges, train_proportions, months):
    """``months``: [(month, scores)] already ordered by the caller's declaration."""
    if (len(score_edges) < 2 or len(train_proportions) != len(score_edges) - 1
            or any(math.isnan(float(v)) for v in score_edges)
            or any(b <= a for a, b in zip(score_edges, score_edges[1:]))
            or any(not math.isfinite(float(v)) or v < 0 for v in train_proportions)
            or not math.isclose(sum(train_proportions), 1.0, abs_tol=1e-8)):
        raise ValueError("baseline score edges and train distribution required")
    trend = []
    for month, scores in months:
        finite = _finite(scores)
        if not finite:
            trend.append({"month": month, "metric": None, "level": "n/a", "sample_count": 0})
            continue
        value = psi(list(train_proportions), _distribution(finite, list(score_edges)))
        trend.append({"month": month, "metric": value, "level": level(value), "sample_count": len(finite)})
    return {"metric_name": "psi", "trend": trend, "per_feature_trend": [],
            "red_flags": month_gap_flags([m for m, _ in months])}


def csi_trend_reference(feature_baselines, months, *, feature_cols=None, top_features=10):
    """``months``: [(month, {feature: values})]; CSI vs the uniform baseline bins."""
    if not feature_baselines:
        raise ValueError("feature baselines required")
    features = list(feature_cols) if feature_cols else list(feature_baselines)
    trend, per_feature = [], []
    for month, frame in months:
        csis = {}
        for feature in features:
            edges = list((feature_baselines.get(feature) or {}).get("quantile_edges") or [])
            if feature not in frame or len(edges) < 2:
                continue
            finite = _finite(frame[feature])
            if not finite:
                continue
            bins = len(edges) - 1
            csis[feature] = (psi([1.0 / bins] * bins, _distribution(finite, edges)), len(finite))
        if not csis:
            trend.append({"month": month, "metric": None, "level": "n/a", "sample_count": 0})
            continue
        worst = max(csis.values(), key=lambda item: item[0])
        trend.append({"month": month, "metric": worst[0], "level": level(worst[0]), "sample_count": worst[1]})
        for feature, (value, count) in sorted(csis.items(), key=lambda item: (-item[1][0], item[0]))[:top_features]:
            per_feature.append({"feature": feature, "month": month, "csi": value, "sample_count": count})
    return {"metric_name": "max_csi", "trend": trend, "per_feature_trend": per_feature,
            "red_flags": month_gap_flags([m for m, _ in months])}


def cohort_key(value):
    text = str(value).strip()
    for pattern in (r"^(\d{4})(\d{2})$", r"^(\d{4})-(\d{2})"):
        match = re.match(pattern, text)
        if match:
            year, month = int(match[1]), int(match[2])
            if year < 1 or not 1 <= month <= 12:
                raise ValueError("cohort month must be valid")
            return f"{year:04d}-{month:02d}"
    raise ValueError("cohort format outside independent Vintage subset")


def vintage_reference(rows, *, cohort_col, mob_col, bad_col, mob_max=12, label_semantics="incremental"):
    if label_semantics not in {"incremental", "snapshot"} or mob_max < 1:
        raise ValueError("invalid Vintage declaration")
    cells = {}
    for row in rows:
        try:
            mob = float(row[mob_col])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(mob):
            continue
        if mob < 0 or mob % 1:
            raise ValueError("MOB must be non-negative integers")
        bad = float(row[bad_col])
        if bad not in (0.0, 1.0):
            raise ValueError("target must be binary 0/1")
        key = (cohort_key(row[cohort_col]), int(mob))
        count, bads = cells.get(key, (0, 0))
        cells[key] = (count + 1, bads + int(bad))
    points, snapshot_shaped, strict_increase = [], True, False
    cohorts = sorted({c for c, _ in cells})
    for cohort in cohorts:
        mobs = sorted(m for c, m in cells if c == cohort)
        counts = [cells[cohort, m] for m in mobs]
        bads = [b for _, b in counts]
        if not (len(bads) >= 3 and all(b >= a for a, b in zip(bads, bads[1:]))):
            snapshot_shaped = False
        strict_increase |= any(b > a for a, b in zip(bads, bads[1:]))
        denominator = float(max(c for c, _ in counts))
        cumulative = 0.0
        for mob, (count, bad) in zip(mobs, counts, strict=True):
            rate = bad / count if count else 0.0
            warnings = []
            if label_semantics == "snapshot":
                cum = rate
            else:
                cumulative += bad
                raw = cumulative / denominator if denominator else 0.0
                cum = min(raw, 1.0)
                if raw > 1.0 + 1e-9:
                    warnings.append(f"cum_bad_rate clipped for cohort {cohort} at mob {mob}: "
                                    f"raw ratio {raw:.4f} exceeds cohort denominator")
            points.append({"cohort": cohort, "mob": mob, "sample_count": count, "bad_count": bad,
                           "bad_rate": rate, "cum_bad_rate": cum, "warnings": warnings})
    metric = "bad_rate" if label_semantics == "snapshot" else "cum_bad_rate"
    axis = sorted({p["mob"] for p in points})
    curves = {}
    for cohort in cohorts:
        by_mob = {p["mob"]: p[metric] for p in points if p["cohort"] == cohort}
        values = [by_mob.get(m) for m in axis][:mob_max]
        curves[cohort] = values + [None] * (mob_max - len(values))
    first = {}
    for p in points:
        if p["cohort"] not in first or p["mob"] < first[p["cohort"]][0]:
            first[p["cohort"]] = (p["mob"], p["sample_count"])
    flagged = bool(label_semantics == "incremental" and cohorts and snapshot_shaped and strict_increase)
    warnings = [w for p in points for w in (*p["warnings"], *((SNAPSHOT_FLAG,) if flagged else ()))]
    return {"cohorts": cohorts, "curves": curves, "counts": {c: v[1] for c, v in first.items()},
            "mob_axis": axis[:mob_max], "mob_max": mob_max, "snapshot_shaped_warning": flagged,
            "warnings": warnings, "points": points}


def vintage_summary_reference(curve, *, ref_mob=6):
    if ref_mob < 1:
        raise ValueError("ref_mob must be positive")
    axis, cohorts = curve["mob_axis"], curve["cohorts"]
    if axis:
        index = axis.index(ref_mob) if ref_mob in axis else (len(curve["curves"][cohorts[0]]) if cohorts else 0)
    else:
        index = ref_mob - 1
    at_ref = {c: float(curve["curves"][c][index]) for c in cohorts
              if index < len(curve["curves"][c]) and curve["curves"][c][index] is not None}
    ordered = [at_ref[c] for c in cohorts if c in at_ref]
    delta = ordered[-1] - ordered[0] if len(ordered) >= 2 else 0.0
    trend = "deteriorating" if delta > 1e-12 else "improving" if delta < -1e-12 else "stable"
    return {"at_ref": at_ref, "trend": trend}
