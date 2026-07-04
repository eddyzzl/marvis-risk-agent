from __future__ import annotations

import pandas as pd

from marvis.packs.strategy.contracts import VintageCurve
from marvis.validation.vintage import compute_vintage_curve, vintage_curve_wide


def vintage_curve(
    df: pd.DataFrame,
    *,
    cohort_col: str,
    mob_col: str,
    bad_col: str,
    mob_max: int = 12,
    label_semantics: str = "incremental",
) -> VintageCurve:
    if mob_max < 1:
        raise ValueError("mob_max must be positive")

    points = compute_vintage_curve(
        df,
        cohort_col=cohort_col,
        mob_col=mob_col,
        target_col=bad_col,
        label_semantics=label_semantics,
    )
    # A1: snapshot flags are already cumulative per loan -- read the per-MOB marginal
    # rate directly (metric='bad_rate') so accumulation never double-counts. Incremental
    # events keep the accumulated cum_bad_rate.
    metric = "bad_rate" if label_semantics == "snapshot" else "cum_bad_rate"
    wide = vintage_curve_wide(points, metric=metric)
    mob_axis = tuple(sorted({point.mob for point in points})[:mob_max])
    curves = {
        cohort: _truncate_or_pad(values, mob_max)
        for cohort, values in wide.items()
    }
    warnings = tuple(
        warning for point in points for warning in point.data_quality_warnings
    )
    return VintageCurve(
        cohort_col=cohort_col,
        mob_max=mob_max,
        cohorts=tuple(sorted(curves)),
        curves=curves,
        counts=_cohort_counts_at_first_mob(points),
        mob_axis=mob_axis,
        warnings=warnings,
    )


def vintage_summary(curve: VintageCurve, *, ref_mob: int = 6) -> dict:
    if ref_mob < 1:
        raise ValueError("ref_mob must be positive")
    at_ref = {}
    for cohort in curve.cohorts:
        values = curve.curves.get(cohort, [])
        index = _mob_index(curve, ref_mob)
        if index < len(values) and values[index] is not None:
            at_ref[cohort] = float(values[index])
    ordered_values = [at_ref[cohort] for cohort in curve.cohorts if cohort in at_ref]
    return {"at_ref": at_ref, "trend": _trend(ordered_values)}


def _truncate_or_pad(values: list[float | None], length: int) -> list[float | None]:
    trimmed = list(values[:length])
    if len(trimmed) < length:
        trimmed.extend([None] * (length - len(trimmed)))
    return trimmed


def _mob_index(curve: VintageCurve, ref_mob: int) -> int:
    if curve.mob_axis:
        try:
            return curve.mob_axis.index(int(ref_mob))
        except ValueError:
            return len(curve.curves.get(curve.cohorts[0], [])) if curve.cohorts else 0
    return int(ref_mob) - 1


def _cohort_counts_at_first_mob(points) -> dict[str, int]:
    counts: dict[str, int] = {}
    first_mobs: dict[str, int] = {}
    for point in points:
        current = first_mobs.get(point.cohort)
        if current is None or point.mob < current:
            first_mobs[point.cohort] = int(point.mob)
            counts[point.cohort] = int(point.sample_count)
    return counts


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "stable"
    delta = values[-1] - values[0]
    if delta > 1e-12:
        return "deteriorating"
    if delta < -1e-12:
        return "improving"
    return "stable"


__all__ = ["vintage_curve", "vintage_summary"]
