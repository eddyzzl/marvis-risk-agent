"""Standard-library recomputation of V2 validation evidence.

Deliberately standalone: no relative imports and no production metric, scoring,
binning, PMML or time-period code (``marvis.validation*``, ``marvis.feature``,
``marvis.packs``, ``marvis.output``), and no numpy/pandas/scipy/sklearn. The
platform conventions are written out again here so a producer change cannot
silently change the adjudicator. Shapes outside the declared subset raise
``IndependentUnsupported`` instead of being guessed.
"""
from bisect import bisect_right
import csv
import io
import math
import re

__all__ = [
    "IndependentUnsupported", "independent_effectiveness", "independent_basic_distribution",
    "independent_pmml_regression_scores", "read_validation_rows",
]

_PSI_SMOOTHING = 1e-6
_INDEPENDENT_QUANTILE_BINS = 10
_SPLITS = ("train", "test", "oot")


class IndependentUnsupported(Exception):
    """The input is outside the independently implemented subset."""


# --- primitives -----------------------------------------------------------

def _linspace_quantiles(count):
    step = 1.0 / count
    return [0.0 if i == 0 else 1.0 if i == count else i * step for i in range(count + 1)]


def _quantile(ordered, q):
    # numpy "linear": virtual index (n - 1) * q, lerp from the upper end at t >= .5.
    n = len(ordered)
    virtual = (n - 1) * q
    lo = math.floor(virtual)
    gamma = virtual - lo
    lo = min(max(lo, 0), n - 1)
    hi = min(lo + 1, n - 1)
    if virtual <= 0:
        return ordered[0]
    if virtual >= n - 1:
        return ordered[-1]
    a, b = ordered[lo], ordered[hi]
    diff = b - a
    return b - diff * (1 - gamma) if gamma >= 0.5 else a + diff * gamma


def equal_frequency_edges(values, bin_count):
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return [-math.inf, math.inf]
    distinct = sorted(set(finite))
    if len(distinct) == 2:
        return [-math.inf, (distinct[0] + distinct[1]) / 2, math.inf]
    edges = sorted(set(_quantile(finite, q) for q in _linspace_quantiles(bin_count)))
    if len(edges) <= 1:
        return [-math.inf, math.inf]
    edges[0], edges[-1] = -math.inf, math.inf
    return edges


def _bin(value, edges):
    """1-based bin, 0 for non-finite (searchsorted right over interior edges)."""
    if not math.isfinite(value):
        return 0
    return min(bisect_right(edges, value, 1, len(edges) - 1), len(edges) - 1)


def _counts(values, edges):
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = _bin(value, edges)
        if index:
            counts[index - 1] += 1
    return counts


def _distribution(values, edges):
    counts = _counts(values, edges)
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def _smoothed(dist):
    values = [_PSI_SMOOTHING if d <= 0 else float(d) for d in dist]
    total = math.fsum(values)
    return [v / total for v in values] if total > 0 else None


def psi(expected, actual):
    e, a = _smoothed(expected), _smoothed(actual)
    if e is None or a is None:
        return 0.0
    return max(0.0, math.fsum((y - x) * math.log(y / x) for x, y in zip(e, a, strict=True)))


def _psi_contributions(expected, actual):
    e, a = _smoothed(expected), _smoothed(actual)
    if e is None or a is None:
        return [0.0] * len(expected)
    return [max(0.0, (y - x) * math.log(y / x)) for x, y in zip(e, a, strict=True)]


def _pairs(scores, labels):
    return [(s, y) for s, y in zip(scores, labels, strict=True) if math.isfinite(s) and y in (0, 1)]


def ks(scores, labels):
    rows = sorted(_pairs(scores, labels), key=lambda row: row[0])
    bad = sum(y for _, y in rows)
    good = len(rows) - bad
    if not bad or not good:
        return 0.0
    best = cum_bad = cum_good = 0
    best = 0.0
    for index, (score, label) in enumerate(rows):
        cum_bad += label
        cum_good += 1 - label
        if index == len(rows) - 1 or rows[index + 1][0] != score:
            best = max(best, abs(cum_bad / bad - cum_good / good))
    return best


def auc(scores, labels):
    groups = {}
    for score, label in _pairs(scores, labels):
        groups.setdefault(score, [0, 0])[label] += 1
    good = sum(g for g, _ in groups.values())
    bad = sum(b for _, b in groups.values())
    if not good or not bad:
        return 0.5
    pairs, seen_good = 0.0, 0
    for score in sorted(groups):
        g, b = groups[score]
        pairs += b * (seen_good + g / 2)
        seen_good += g
    return pairs / (good * bad)


def _pearson(x, y):
    rows = [(a, b) for a, b in zip(x, y, strict=True) if math.isfinite(a) and math.isfinite(b)]
    if len(rows) < 2:
        return 0.0
    mx = math.fsum(a for a, _ in rows) / len(rows)
    my = math.fsum(b for _, b in rows) / len(rows)
    sxx = math.fsum((a - mx) ** 2 for a, _ in rows)
    syy = math.fsum((b - my) ** 2 for _, b in rows)
    if sxx == 0 or syy == 0:
        return 0.0
    value = math.fsum((a - mx) * (b - my) for a, b in rows) / math.sqrt(sxx * syy)
    return value if math.isfinite(value) else 0.0


def head_tail_lift(scores, labels, fraction=0.05):
    """(validation head = low-risk end, validation tail = high-risk end)."""
    rows = _pairs(scores, labels)
    n = len(rows)
    base = sum(y for _, y in rows) / n if n else 0.0
    if n < 1 or base <= 0:
        return None, None
    corr = _pearson([s for s, _ in rows], [float(y) for _, y in rows])
    sign = 1.0 if corr > 0 else -1.0 if corr < 0 else 1.0
    ordered = [y for _, y in sorted(rows, key=lambda row: sign * row[0])]
    count = max(1, math.floor(fraction * n))
    risk_head = (sum(ordered[-count:]) / count) / base
    risk_tail = (sum(ordered[:count]) / count) / base
    return risk_tail, risk_head


def _ratio(a, b):
    return a / b if b else 0.0


def bin_table(scores, labels, edges, *, reverse):
    marginals = []
    for index in range(1, len(edges)):
        members = [y for s, y in zip(scores, labels, strict=True) if _bin(s, edges) == index]
        marginals.append((edges[index - 1], edges[index], len(members), sum(members)))
    if reverse:
        marginals.reverse()
    total = sum(m[2] for m in marginals)
    total_bad = sum(m[3] for m in marginals)
    total_good = total - total_bad
    overall = _ratio(total_bad, total)
    rows, cum_count, cum_bad = [], 0, 0
    for walk, (lower, upper, count, bad) in enumerate(marginals, start=1):
        rate = _ratio(bad, count)
        cum_count += count
        cum_bad += bad
        cum_bad_pct = _ratio(cum_bad, total_bad)
        rows.append({"bin_index": walk, "score_lower": float(lower), "score_upper": float(upper),
                     "sample_count": count, "bad_count": bad, "bad_rate": rate,
                     "cum_sample_pct": _ratio(cum_count, total), "cum_bad_pct": cum_bad_pct,
                     "lift": _ratio(rate, overall),
                     "ks": abs(cum_bad_pct - _ratio(cum_count - cum_bad, total_good))})
    return rows


def _reverse(scores, labels):
    if len(scores) < 2:
        return False
    return _pearson(scores, [float(y) for y in labels]) < 0


def roc_ks_curve(split, scores, labels):
    rows = [(s, y) for s, y in zip(scores, labels, strict=True) if math.isfinite(s)]
    positives = sum(y for _, y in rows)
    negatives = len(rows) - positives
    if not rows or not positives or not negatives:
        return {"split": split, "fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "ks_curve": [0.0, 0.0],
                "ks": 0.0, "population_at_ks": 0.0}
    # numpy argsort(...)[::-1] of an ascending quicksort is not stable; ties only
    # matter inside a score group, whose cumulative endpoint is order independent.
    ordered = sorted(rows, key=lambda row: row[0], reverse=True)
    tpr, fpr, population = [0.0], [0.0], [0.0]
    cum_bad = cum_good = 0
    for index, (score, label) in enumerate(ordered):
        cum_bad += label
        cum_good += 1 - label
        if index == len(ordered) - 1 or ordered[index + 1][0] != score:
            tpr.append(cum_bad / positives)
            fpr.append(cum_good / negatives)
            population.append((index + 1) / len(ordered))
    curve = [t - f for t, f in zip(tpr, fpr, strict=True)]
    best = max(range(len(curve)), key=lambda i: (abs(curve[i]), -i))
    return {"split": split, "fpr": fpr, "tpr": tpr, "ks_curve": curve,
            "ks": ks([s for s, _ in rows], [y for _, y in rows]), "population_at_ks": population[best]}


def _interval(lower, upper):
    def text(value):
        if value == -math.inf:
            return "-inf"
        if value == math.inf:
            return "inf"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"[{text(lower)},{text(upper)}]"


def _month_ordinal(month):
    if len(month) != 6 or not month.isdigit() or not 1 <= int(month[4:]) <= 12:
        return None
    return int(month[:4]) * 12 + int(month[4:])


# --- inputs ---------------------------------------------------------------

_DATE = (
    (re.compile(r"^(\d{4})(\d{2})(?:\.0+)?$"), lambda m: (m[1], m[2], "01")),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})(?:\.0+)?$"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"^(\d{4})/(\d{2})/(\d{2})$"), lambda m: (m[1], m[2], m[3])),
)


def date_key(value):
    text = str(value).strip()
    for pattern, parts in _DATE:
        match = pattern.match(text)
        if match:
            year, month, day = parts(match)
            if not 1 <= int(month) <= 12 or not 1 <= int(day) <= 31:
                raise IndependentUnsupported("time value outside independent calendar subset")
            return f"{year}{month}{day}"
    raise IndependentUnsupported("time format outside independent subset")


def _label(raw, positive, negative):
    def matches(label):
        if isinstance(label, str):
            return raw == label
        if isinstance(label, bool) or not isinstance(label, (int, float)):
            raise IndependentUnsupported("confirmed label type outside independent subset")
        try:
            return float(raw) == float(label)
        except ValueError:
            return False
    is_positive, is_negative = matches(positive), matches(negative)
    if is_positive == is_negative:
        raise ValueError("target value outside confirmed labels")
    return 1 if is_positive else 0


def read_validation_rows(sample_bytes, *, encoding, confirmed):
    """Decode only target/split/time from raw CSV bytes with the frozen choices."""
    if encoding not in {"utf-8", "utf-8-sig"}:
        raise IndependentUnsupported("sample encoding outside independent subset")
    mapping = confirmed.get("split_value_mapping")
    if (not isinstance(mapping, dict) or set(mapping) != set(_SPLITS)
            or any(type(v) is not str for v in mapping.values()) or len(set(mapping.values())) != 3):
        raise IndependentUnsupported("split mapping outside independent subset")
    by_raw = {raw: canonical for canonical, raw in mapping.items()}
    reader = csv.DictReader(io.StringIO(sample_bytes.decode(encoding), newline=""))
    target, split, time = confirmed["target_col"], confirmed["split_col"], confirmed["time_col"]
    rows = []
    for record in reader:
        if record.get(split) not in by_raw:
            raise ValueError("split value outside confirmed mapping")
        rows.append({"label": _label(record[target], confirmed["positive_label"], confirmed["negative_label"]),
                     "split": by_raw[record[split]], "date": date_key(record[time]), "raw": record})
    return rows


def independent_pmml_regression_scores(pmml_bytes, records, output_field):
    """Evaluate the declared PMML RegressionModel subset without the PMML engine."""
    from defusedxml.ElementTree import fromstring

    root = fromstring(pmml_bytes, forbid_dtd=True, forbid_entities=True, forbid_external=True)

    def local(element):
        return element.tag.rsplit("}", 1)[-1]

    if local(root) != "PMML":
        raise IndependentUnsupported("not a PMML document")
    models = [child for child in root if local(child).endswith("Model")]
    if (len(models) != 1 or local(models[0]) != "RegressionModel"
            or any(local(e) in {"TransformationDictionary", "LocalTransformations", "Extension", "Targets"}
                   for e in root.iter())):
        raise IndependentUnsupported("PMML structure outside independent RegressionModel subset")
    model = models[0]
    if model.get("functionName") != "classification" or model.get("normalizationMethod") != "logit":
        raise IndependentUnsupported("PMML regression link outside independent subset")
    tables = [child for child in model if local(child) == "RegressionTable"]
    outputs = [e for e in model.iter() if local(e) == "OutputField"]
    chosen = [e for e in outputs if e.get("name") == output_field]
    if (len(tables) != 2 or len(chosen) != 1 or chosen[0].get("feature") != "probability"):
        raise IndependentUnsupported("PMML output outside independent binary probability subset")
    target = chosen[0].get("value")
    active = [t for t in tables if t.get("targetCategory") == target]
    reference = [t for t in tables if t.get("targetCategory") != target]
    if len(active) != 1 or len(reference) != 1:
        raise IndependentUnsupported("PMML target category outside independent subset")

    def terms(table):
        result = []
        for child in table:
            if local(child) != "NumericPredictor" or child.get("exponent", "1") not in {"1", "1.0"}:
                raise IndependentUnsupported("PMML predictor outside independent numeric subset")
            result.append((child.get("name"), float(child.get("coefficient"))))
        return float(table.get("intercept", "0")), result

    def linear(table, record):
        intercept, coefficients = table
        total = [intercept]
        for name, coefficient in coefficients:
            raw = record.get(name)
            if raw is None or str(raw).strip() == "":
                raise IndependentUnsupported("missing PMML input requires engine missing-value semantics")
            value = float(raw)
            if not math.isfinite(value):
                raise IndependentUnsupported("non-finite PMML input")
            total.append(coefficient * value)
        return math.fsum(total)

    on, off = terms(active[0]), terms(reference[0])
    scores = []
    for record in records:
        # Two-class logit: p = 1 / (1 + exp(y_ref - y_target)).
        margin = linear(on, record) - linear(off, record)
        scores.append(1.0 / (1.0 + math.exp(-margin)) if margin >= 0 else math.exp(margin) / (1.0 + math.exp(margin)))
    return scores


# --- assembled results ----------------------------------------------------

def _by_split(rows, scores):
    return {split: ([s for r, s in zip(rows, scores, strict=True) if r["split"] == split],
                    [r["label"] for r in rows if r["split"] == split]) for split in _SPLITS}


def independent_basic_distribution(rows):
    split_summary = []
    for split in _SPLITS:
        members = [r for r in rows if r["split"] == split]
        bad = sum(r["label"] for r in members)
        dates = [r["date"] for r in members]
        split_summary.append({"split": split, "sample_count": len(members), "bad_count": bad,
                              "bad_rate": _ratio(bad, len(members)),
                              "period_start": min(dates) if dates else "", "period_end": max(dates) if dates else ""})
    monthly = []
    for month in sorted({r["date"][:6] for r in rows}):
        members = [r for r in rows if r["date"][:6] == month]
        bad = sum(r["label"] for r in members)
        monthly.append({"month": month, "sample_count": len(members), "bad_count": bad, "bad_rate": _ratio(bad, len(members))})
    dates = [r["date"] for r in rows]
    return {"sample_period": [min(dates), max(dates)] if dates else ["", ""],
            "split_summary": split_summary, "monthly_distribution": monthly}


def independent_effectiveness(rows, scores, *, bin_count):
    if len(rows) != len(scores) or any(not math.isfinite(s) for s in scores):
        raise ValueError("independent score population differs")
    splits = _by_split(rows, scores)
    if any(not splits[s][0] for s in _SPLITS):
        raise IndependentUnsupported("empty required split")
    train_scores = splits["train"][0]
    edges = equal_frequency_edges(train_scores, bin_count)
    train_distribution = _distribution(train_scores, edges)
    overall = []
    for split in _SPLITS:
        s, y = splits[split]
        head, tail = head_tail_lift(s, y)
        overall.append({"split": split, "ks": ks(s, y),
                        "psi_vs_train": 0.0 if split == "train" else psi(train_distribution, _distribution(s, edges)),
                        "sample_count": len(s), "bad_rate": _ratio(sum(y), len(y)), "bad_count": sum(y),
                        "auc": auc(s, y), "head_lift_5pct": head, "tail_lift_5pct": tail})
    tables, independent = {}, {}
    for split in _SPLITS:
        s, y = splits[split]
        reverse = _reverse(s, y)
        tables[split] = bin_table(s, y, edges, reverse=reverse)
        independent[split] = bin_table(s, y, equal_frequency_edges(s, _INDEPENDENT_QUANTILE_BINS), reverse=reverse)
    months = sorted({r["date"][:6] for r in rows})
    by_month = {m: ([s for r, s in zip(rows, scores, strict=True) if r["date"][:6] == m],
                    [r["label"] for r in rows if r["date"][:6] == m]) for m in months}
    monthly_ks = []
    for month in months:
        s, y = by_month[month]
        head, tail = head_tail_lift(s, y)
        monthly_ks.append({"month": month, "ks": ks(s, y), "sample_count": len(s), "bad_count": sum(y),
                           "bad_rate": _ratio(sum(y), len(y)), "auc": auc(s, y),
                           "head_lift_5pct": head, "tail_lift_5pct": tail})
    monthly_psi, previous, previous_month = [], None, ""
    first = _distribution(by_month[months[0]][0], edges)
    last = _distribution(by_month[months[-1]][0], edges)
    for month in months:
        current = _distribution(by_month[month][0], edges)
        a, b = _month_ordinal(previous_month), _month_ordinal(month)
        monthly_psi.append({"month": month, "psi_vs_train": psi(train_distribution, current),
                            "psi_first_month": 0.0 if month == months[0] else psi(first, current),
                            "psi_last_month": 0.0 if month == months[-1] else psi(last, current),
                            "psi_mom": None if previous is None else psi(previous, current),
                            "psi_mom_reference_month": previous_month,
                            "psi_mom_has_calendar_gap": bool(a is not None and b is not None and b - a > 1)})
        previous, previous_month = current, month
    expected_scores = splits["train"][0] + splits["test"][0]
    stability_edges = equal_frequency_edges(expected_scores, bin_count)
    expected_counts = _counts(expected_scores, stability_edges)
    actual_counts = _counts(splits["oot"][0], stability_edges)
    e_total, a_total = sum(expected_counts), sum(actual_counts)
    e_dist = [c / e_total if e_total else 0.0 for c in expected_counts]
    a_dist = [c / a_total if a_total else 0.0 for c in actual_counts]
    contributions = _psi_contributions(e_dist, a_dist)
    stability = [{"bin_label": _interval(stability_edges[i], stability_edges[i + 1]),
                  "expected_count": expected_counts[i], "expected_pct": _ratio(expected_counts[i], e_total),
                  "actual_count": actual_counts[i], "actual_pct": _ratio(actual_counts[i], a_total),
                  "psi": contributions[i]} for i in range(len(expected_counts))]
    curves = {split: roc_ks_curve(split, *splits[split]) for split in _SPLITS}
    return {"overall": overall, "bin_tables": tables, "monthly_ks": monthly_ks, "monthly_psi": monthly_psi,
            "psi_stability_table": stability, "roc_ks_curves": curves,
            "independent_quantile_bin_tables": independent}
