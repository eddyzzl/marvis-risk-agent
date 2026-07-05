from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


DEFAULT_OVERDUE_BUCKETS = ("current", "1-30", "31-60", "61-90", "90+")


@dataclass(frozen=True)
class VintagePoint:
    cohort: str
    mob: int
    sample_count: int
    bad_count: int
    bad_rate: float
    cum_bad_rate: float
    balance_sum: float | None
    denominator: str
    data_quality_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RollRatePoint:
    from_bucket: str
    to_bucket: str
    count: int
    rate: float


def compute_vintage_curve(
    dataframe: pd.DataFrame,
    *,
    cohort_col: str,
    mob_col: str,
    target_col: str,
    balance_col: str | None = None,
    denominator: str = "count",
    label_semantics: str = "incremental",
) -> list[VintagePoint]:
    if denominator not in {"count", "balance"}:
        raise ValueError("denominator must be count or balance")
    if label_semantics not in {"incremental", "snapshot"}:
        raise ValueError("label_semantics must be incremental or snapshot")
    required = [cohort_col, mob_col, target_col]
    if denominator == "balance":
        if not balance_col:
            raise ValueError("balance_col is required for balance denominator")
        required.append(balance_col)
    elif balance_col:
        required.append(balance_col)
    _assert_columns(dataframe, required)

    frame = dataframe[required].copy()
    frame["_cohort"] = frame[cohort_col].map(_cohort_key)
    frame["_mob"] = _parse_mob(frame[mob_col])
    frame = frame[frame["_mob"].notna()].copy()
    if frame.empty:
        return []
    frame["_mob"] = frame["_mob"].astype(int)
    frame["_target"] = _parse_target(frame[target_col])
    if balance_col:
        frame["_balance"] = pd.to_numeric(frame[balance_col], errors="raise").fillna(0.0).astype(float)
    else:
        frame["_balance"] = np.nan

    points: list[VintagePoint] = []
    # A1: red-flag heuristic. A snapshot/ever-bad flag produces a per-MOB bad_count
    # that never DECREASES within a cohort (a loan once bad stays bad). If EVERY
    # cohort is non-decreasing yet the caller declared "incremental", the data most
    # likely carries snapshot flags and the accumulation double-counts -- advisory
    # only, it never mutates the curve (mirrors RollRateMatrix.data_quality_warnings).
    # Conservative to avoid false positives on genuinely-incremental small cohorts:
    # every cohort must span >=3 MOBs, be non-decreasing, and at least one cohort must
    # strictly increase somewhere (an all-flat / single-2-point sequence is too weak a
    # signal to call snapshot).
    all_cohorts_snapshot_shaped = True
    any_strict_increase = False
    cohort_count = 0
    for cohort, cohort_frame in frame.groupby("_cohort", sort=True):
        cohort_count += 1
        # First pass: per-(cohort, mob) marginal metrics (unchanged semantics).
        rows: list[dict] = []
        for mob, group in cohort_frame.groupby("_mob", sort=True):
            target = group["_target"].to_numpy(dtype=int)
            sample_count = int(len(group))
            bad_count = int(np.sum(target == 1))
            balance_sum = None
            if balance_col:
                balances = group["_balance"].to_numpy(dtype=float)
                balance_sum = float(np.sum(balances))
                bad_balance = float(np.sum(balances[target == 1]))
            else:
                bad_balance = 0.0
            if denominator == "balance":
                bad_numerator = bad_balance
                denominator_value = balance_sum if balance_sum is not None else 0.0
            else:
                bad_numerator = float(bad_count)
                denominator_value = float(sample_count)

            rows.append(
                {
                    "mob": int(mob),
                    "sample_count": sample_count,
                    "bad_count": bad_count,
                    "bad_rate": _ratio(bad_numerator, denominator_value),
                    "balance_sum": balance_sum,
                    "bad_numerator": bad_numerator,
                }
            )

        # Per-cohort snapshot heuristic: >=3 MOBs and bad_count non-decreasing across
        # ascending MOB. Fewer than 3 MOBs is too weak a signal (a 2-point rise is
        # perfectly ordinary incremental data), so it disqualifies the whole dataset.
        bad_counts = [row["bad_count"] for row in rows]
        pairs = list(zip(bad_counts, bad_counts[1:]))
        cohort_snapshot_shaped = len(bad_counts) >= 3 and all(
            later >= earlier for earlier, later in pairs
        )
        if not cohort_snapshot_shaped:
            all_cohorts_snapshot_shaped = False
        if any(later > earlier for earlier, later in pairs):
            any_strict_increase = True

        # Fixed cohort-level denominator: max observed base across the cohort's MOBs.
        if denominator == "balance":
            cohort_denominator = max(
                (row["balance_sum"] or 0.0 for row in rows),
                default=0.0,
            )
        else:
            cohort_denominator = float(max((row["sample_count"] for row in rows), default=0))

        # Second pass. incremental: accumulate the bad numerator across MOBs in
        # ascending order (current behaviour). snapshot: the per-MOB marginal rate
        # IS the true cumulative rate -- the flag is already cumulative per loan, so
        # accumulating would double-count (exactly report_compute's metric='bad_rate').
        cumulative_bad = 0.0
        for row in rows:
            warnings: tuple[str, ...] = ()
            if label_semantics == "snapshot":
                cum_bad_rate = row["bad_rate"]
            else:
                cumulative_bad += row["bad_numerator"]
                if cohort_denominator == 0:
                    raw_cum_ratio = 0.0
                else:
                    raw_cum_ratio = cumulative_bad / cohort_denominator
                cum_bad_rate = min(raw_cum_ratio, 1.0)
                if raw_cum_ratio > 1.0 + 1e-9:
                    warnings = (
                        f"cum_bad_rate clipped for cohort {cohort} at mob {row['mob']}: "
                        f"raw ratio {raw_cum_ratio:.4f} exceeds cohort denominator",
                    )
            points.append(
                VintagePoint(
                    cohort=str(cohort),
                    mob=row["mob"],
                    sample_count=row["sample_count"],
                    bad_count=row["bad_count"],
                    bad_rate=row["bad_rate"],
                    cum_bad_rate=cum_bad_rate,
                    balance_sum=row["balance_sum"],
                    denominator=denominator,
                    data_quality_warnings=warnings,
                )
            )

    # Advisory snapshot red flag: attach once (to every point) when the data looks
    # cumulative but was declared incremental, so the accumulation is likely wrong.
    if (
        label_semantics == "incremental"
        and cohort_count > 0
        and all_cohorts_snapshot_shaped
        and any_strict_increase
    ):
        flag = (
            "per-MOB bad_count non-decreasing across all cohorts -- data looks like a "
            "SNAPSHOT flag (每列/行=截至该MOB是否曾坏, 已单调); declared=incremental, "
            "so cum_bad_rate re-accumulates and may double-count. Confirm label_semantics."
        )
        points = [
            dataclasses.replace(
                point,
                data_quality_warnings=(*point.data_quality_warnings, flag),
            )
            for point in points
        ]
    return points


def vintage_curve_wide(
    points: Sequence[VintagePoint],
    *,
    metric: str = "cum_bad_rate",
) -> dict[str, list[float | None]]:
    if metric not in {"cum_bad_rate", "bad_rate"}:
        raise ValueError("metric must be cum_bad_rate or bad_rate")
    cohorts = sorted({point.cohort for point in points})
    mobs = sorted({point.mob for point in points})
    by_key = {(point.cohort, point.mob): float(getattr(point, metric)) for point in points}
    return {
        cohort: [by_key.get((cohort, mob)) for mob in mobs]
        for cohort in cohorts
    }


def compute_roll_rate(
    dataframe: pd.DataFrame,
    *,
    from_bucket_col: str,
    to_bucket_col: str,
    buckets: Sequence[str] = DEFAULT_OVERDUE_BUCKETS,
) -> list[RollRatePoint]:
    _assert_columns(dataframe, [from_bucket_col, to_bucket_col])
    bucket_order = tuple(str(bucket) for bucket in buckets)
    if not bucket_order:
        raise ValueError("buckets must not be empty")
    valid = set(bucket_order)
    from_values = dataframe[from_bucket_col].dropna().map(str)
    to_values = dataframe[to_bucket_col].dropna().map(str)
    unknown = sorted((set(from_values) | set(to_values)) - valid)
    if unknown:
        raise ValueError(f"unknown bucket: {', '.join(unknown)}")

    counts = (
        dataframe.assign(
            _from=dataframe[from_bucket_col].map(lambda value: str(value) if pd.notna(value) else None),
            _to=dataframe[to_bucket_col].map(lambda value: str(value) if pd.notna(value) else None),
        )
        .dropna(subset=["_from", "_to"])
        .groupby(["_from", "_to"], sort=False)
        .size()
        .to_dict()
    )
    totals = {bucket: sum(int(counts.get((bucket, to_bucket), 0)) for to_bucket in bucket_order) for bucket in bucket_order}
    points = []
    for from_bucket in bucket_order:
        total = totals[from_bucket]
        for to_bucket in bucket_order:
            count = int(counts.get((from_bucket, to_bucket), 0))
            points.append(
                RollRatePoint(
                    from_bucket=from_bucket,
                    to_bucket=to_bucket,
                    count=count,
                    rate=_ratio(count, total),
                )
            )
    return points


def vintage_summary_payload(
    vintage_points: Sequence[VintagePoint],
    roll_rate_points: Sequence[RollRatePoint] | None = None,
) -> dict:
    warnings: list[str] = []
    for point in vintage_points:
        warnings.extend(str(warning) for warning in point.data_quality_warnings)
    return {
        "vintage": [_jsonable(asdict(point)) for point in vintage_points],
        "roll_rate": [_jsonable(asdict(point)) for point in (roll_rate_points or [])],
        "warnings": warnings,
    }


def _assert_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


def _cohort_key(value) -> str:
    if pd.isna(value):
        raise ValueError("cohort contains missing values")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if hasattr(value, "strftime") and not isinstance(value, str):
        return pd.Timestamp(value).strftime("%Y-%m")
    text = str(value).strip()
    if len(text) == 6 and text.isdigit():
        year = int(text[:4])
        month = int(text[4:])
        _validate_month(year, month)
        return f"{year:04d}-{month:02d}"
    if len(text) >= 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
        year = int(text[:4])
        month = int(text[5:7])
        _validate_month(year, month)
        return f"{year:04d}-{month:02d}"
    parsed = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(parsed).strftime("%Y-%m")


def _validate_month(year: int, month: int) -> None:
    if year < 1 or not 1 <= month <= 12:
        raise ValueError("cohort month must be valid")


def _parse_mob(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    finite = values.dropna()
    if ((finite < 0) | (finite % 1 != 0)).any():
        raise ValueError("MOB must be non-negative integers")
    return values


def _parse_target(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("target must be binary 0/1")
    return values.astype(int)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "DEFAULT_OVERDUE_BUCKETS",
    "RollRatePoint",
    "VintagePoint",
    "compute_roll_rate",
    "compute_vintage_curve",
    "vintage_curve_wide",
    "vintage_summary_payload",
]
