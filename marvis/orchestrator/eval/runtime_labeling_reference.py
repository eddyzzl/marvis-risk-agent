"""Independent label arithmetic for acceptance, with no producer-kernel calls.

Rows describe observed repayment state, not independent proof of external truth.
As in the published contract, an early bad hit is final; a non-bad loan is good
only after an observation at or beyond the requested MOB. Cohort maturity is
based on the cohort's maximum observed MOB, not every member's maturity.
"""
from __future__ import annotations

from datetime import date
import math
import numbers
import re

import pandas as pd


def _cohort(value):
    if pd.isna(value):
        raise ValueError("missing cohort")
    text = str(value).strip()
    if re.fullmatch(r"\d{6}", text):
        parsed = date(int(text[:4]), int(text[4:]), 1)
    elif re.match(r"^\d{4}-\d{2}(?:$|[- T])", text):
        parsed = date(int(text[:4]), int(text[5:7]), 1)
    else:
        parsed = pd.Timestamp(value)
    return f"{parsed.year:04d}-{parsed.month:02d}"


def _identity(value):
    if isinstance(value, str) and value.strip():
        return ("text", value)
    if isinstance(value, numbers.Real) and not isinstance(value, (bool,)) and math.isfinite(value):
        return ("number", value)
    raise ValueError("ambiguous loan identity")


def reference_labels(frame, request):
    """Rebuild rows/counts using a separate per-entity accumulator.

    Input parsing and the explicit request schema are shared data contracts.
    Neither construct_label, check_cohort_maturity nor the producer's receipt
    replay function supplies any expected label, count or maturity value here.
    """
    required = [request.id_col, request.mob_col, request.cohort_col, request.date_col, request.value_col]
    if len(set(required)) != len(required) or not set(required) <= set(frame.columns):
        raise ValueError("ambiguous or missing label columns")
    if request.target_col in frame.columns:
        raise ValueError("label target already exists")
    dates = pd.to_datetime(frame[request.date_col], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("unknown source observation date")
    cutoff = pd.Timestamp(date.fromisoformat(request.as_of_date), tz="UTC")
    selected = frame.loc[dates.dt.normalize() <= cutoff, required]
    if selected.empty:
        raise ValueError("no observed rows before cutoff")
    mobs = pd.to_numeric(selected[request.mob_col], errors="coerce")
    loans, cohorts = {}, {}
    order = {value: index for index, value in enumerate(request.states or ())}
    threshold_rank = order.get(request.threshold_status)
    for row, mob in zip(selected.to_dict("records"), mobs.tolist(), strict=True):
        # Missing MOB cannot locate a row in the contract's window. It does not
        # manufacture evidence that an entity reached the maturity threshold.
        if pd.isna(mob):
            continue
        if not math.isfinite(mob) or mob < 0 or mob != int(mob):
            raise ValueError("invalid observed MOB")
        identity = _identity(row[request.id_col])
        cohort = _cohort(row[request.cohort_col])
        loan = loans.setdefault(identity, {"id": row[request.id_col], "cohort": row[request.cohort_col],
                                           "cohort_key": cohort, "max_mob": -1, "hit": False})
        if loan["cohort_key"] != cohort:
            raise ValueError("loan belongs to inconsistent cohorts")
        loan["max_mob"] = max(loan["max_mob"], int(mob))
        group = cohorts.setdefault(cohort, {"ids": set(), "max_mob": -1})
        group["ids"].add(identity)
        group["max_mob"] = max(group["max_mob"], int(mob))
        if request.observation_window < mob <= request.at_mob:
            value = row[request.value_col]
            if request.rule_kind == "dpd":
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError):
                    numeric = float("nan")
                hit = math.isfinite(numeric) and numeric >= request.threshold_dpd
            else:
                rank = None if pd.isna(value) else order.get(str(value))
                hit = rank is not None and rank >= threshold_rank
            loan["hit"] |= hit
    records, bad, good, unknown = [], 0, 0, 0
    for loan in loans.values():
        if loan["hit"]:
            label, bad = 1.0, bad + 1
        elif loan["max_mob"] >= request.at_mob:
            label, good = 0.0, good + 1
        else:
            label, unknown = None, unknown + 1
        records.append([loan["id"], loan["cohort"], label])
    total = len(records)
    quality = {"n_loans": total, "n_bad": bad, "n_good": good, "n_unmatured": unknown,
               "label_coverage": (bad + good) / total if total else None,
               "bad_rate": bad / (bad + good) if bad + good else None}
    cohort_rows = [{"cohort": cohort, "n_loans": len(group["ids"]),
                    "max_observed_mob": group["max_mob"], "required_mob": request.at_mob,
                    "matured": group["max_mob"] >= request.at_mob}
                   for cohort, group in sorted(cohorts.items())]
    immature = [row["cohort"] for row in cohort_rows if not row["matured"]]
    maturity = {"required_mob": request.at_mob, "cohorts": cohort_rows,
                "immature_cohorts": immature, "all_matured": not immature}
    threshold = request.threshold_dpd if request.rule_kind == "dpd" else request.threshold_status
    if request.rule_kind == "dpd":
        value = float(threshold)
        threshold_text = str(int(value)) if value.is_integer() else str(value)
        head = f"DPD{threshold_text}"
    else:
        head = threshold
    definition = {"threshold_kind": request.rule_kind, "threshold": threshold,
                  "observation_window": request.observation_window, "performance_window": request.performance_window,
                  "at_mob": request.at_mob, "hit_rule": "ever",
                  "label": f"{head}+@mob{request.at_mob} (obs={request.observation_window}, perf={request.performance_window})"}
    labels = pd.DataFrame(records, columns=[request.id_col, request.cohort_col, request.target_col])
    return {"frame": labels, "quality": quality, "maturity": maturity, "bad_definition": definition,
            "rows_at_as_of": len(selected), "rows_excluded_after_as_of": len(frame) - len(selected)}
