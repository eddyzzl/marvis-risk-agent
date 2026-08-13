"""Deterministic leakage and selection-bias diagnostics (evidence only).

This module is persistence-free and aggregate-only: it never stores or returns
per-row customer data, only counts, rates, and column summaries. Both entry
points are pure pandas and deterministic; neither mutates its inputs, adopts a
strategy, or blocks a workflow. Every result is *evidence* -- it never renders a
"leaked / biased" verdict; downstream governance decides what to do with it.

Two diagnostics are provided:

* :func:`time_leakage_diagnostics` -- per feature, compares the feature's
  observation timestamp against the label's performance window and reports the
  overlap counts plus red-flag markers.
* :func:`selection_bias_diagnostics` -- compares an approval population against
  a risk (through-the-door) population with counts, bad rates, and numeric
  column summaries.

Fail-closed contract: invalid inputs (missing columns, unparseable timestamps,
non-numeric summary columns, inconsistent window specifications) raise
:class:`LeakageDiagnosticsError` instead of silently degrading to empty output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from marvis.packs.strategy.errors import StrategyError


class LeakageDiagnosticsError(StrategyError):
    """Leakage/selection-bias diagnostics input failed closed (never silent)."""


#: Levels shared by the red-flag / evidence-flag markers. ``red`` marks a
#: temporal-leakage overlap; ``info``/``amber`` mark non-blocking observations.
DIAGNOSTIC_FLAG_LEVELS = frozenset({"info", "amber", "red"})


@dataclass(frozen=True)
class DiagnosticFlag:
    """One structured marker. Evidence only -- never an automatic block."""

    code: str
    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "level": self.level, "message": self.message}


@dataclass(frozen=True)
class LabelWindow:
    """The label performance window against which feature times are compared.

    ``mode`` is ``"scalar"`` (one fixed window for every row, described by ISO
    ``start``/``end``) or ``"columns"`` (a per-row window read from
    ``start_col``/``end_col``). Exactly one mode is populated; the other pair of
    fields is ``None``.
    """

    mode: str
    start: str | None
    end: str | None
    start_col: str | None
    end_col: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "mode": self.mode,
            "start": self.start,
            "end": self.end,
            "start_col": self.start_col,
            "end_col": self.end_col,
        }


@dataclass(frozen=True)
class FeatureTimeLeakage:
    """Per-feature temporal-overlap evidence.

    Buckets partition the *observable* rows (those with a non-null observation
    time and a non-null window) into three mutually exclusive, exhaustive
    classes against the label window ``[start, end]`` (inclusive):

    * ``before_window_count`` -- observation time strictly before ``start``.
    * ``in_window_count`` -- ``start <= observation_time <= end``.
    * ``after_window_count`` -- observation time strictly after ``end``.

    ``observed_rows = before_window_count + in_window_count +
    after_window_count``; ``missing_rows`` are the remaining rows that cannot be
    classified because the observation time or the window is null.
    ``overlap_rate = in_window_count / observed_rows`` (0.0 when unobservable).
    """

    feature: str
    time_col: str
    observed_rows: int
    missing_rows: int
    before_window_count: int
    in_window_count: int
    after_window_count: int
    overlap: str
    overlap_rate: float
    red_flags: tuple[DiagnosticFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "time_col": self.time_col,
            "observed_rows": self.observed_rows,
            "missing_rows": self.missing_rows,
            "before_window_count": self.before_window_count,
            "in_window_count": self.in_window_count,
            "after_window_count": self.after_window_count,
            "overlap": self.overlap,
            "overlap_rate": self.overlap_rate,
            "red_flags": [flag.to_dict() for flag in self.red_flags],
        }


@dataclass(frozen=True)
class TimeLeakageDiagnostics:
    """Aggregate-only temporal-leakage evidence for a feature list."""

    window: LabelWindow
    features: tuple[FeatureTimeLeakage, ...]
    red_flags: tuple[DiagnosticFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "features": [feature.to_dict() for feature in self.features],
            "red_flags": [flag.to_dict() for flag in self.red_flags],
        }


@dataclass(frozen=True)
class ColumnSummary:
    """Numeric column summary computed over the population's own rows.

    ``mean``/``min``/``median``/``max`` are computed over non-null values;
    ``std`` is the *population* standard deviation (``ddof=0``) and is ``None``
    when fewer than two non-null values exist. ``missing_rate =
    missing_count / total_rows``. All statistics are ``None`` when the column is
    entirely null.
    """

    column: str
    count: int
    missing_count: int
    missing_rate: float
    mean: float | None
    std: float | None
    min: float | None
    median: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_rate": self.missing_rate,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "median": self.median,
            "max": self.max,
        }


@dataclass(frozen=True)
class PopulationEvidence:
    """One population's aggregate evidence.

    ``labeled_count`` counts non-null target values; ``bad_count`` counts rows
    where ``target == target_bad_value``; ``bad_rate = bad_count /
    labeled_count`` (0.0 when unlabeled); ``label_coverage = labeled_count /
    count`` (0.0 when empty).
    """

    population: str
    count: int
    labeled_count: int
    bad_count: int
    bad_rate: float
    label_coverage: float
    columns: tuple[ColumnSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "count": self.count,
            "labeled_count": self.labeled_count,
            "bad_count": self.bad_count,
            "bad_rate": self.bad_rate,
            "label_coverage": self.label_coverage,
            "columns": [summary.to_dict() for summary in self.columns],
        }


@dataclass(frozen=True)
class SelectionBiasDiagnostics:
    """Aggregate-only approval-vs-risk distribution evidence.

    No verdict is drawn: differing counts are reported as an ``info`` flag, and
    the caller keeps full responsibility for any "biased / unbiased" reading.
    """

    target: str
    target_bad_value: int | str
    approval: PopulationEvidence
    risk: PopulationEvidence
    flags: tuple[DiagnosticFlag, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_bad_value": self.target_bad_value,
            "approval": self.approval.to_dict(),
            "risk": self.risk.to_dict(),
            "flags": [flag.to_dict() for flag in self.flags],
        }


def time_leakage_diagnostics(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    feature_time_cols: Mapping[str, str] | None = None,
    observation_time_col: str | None = None,
    label_window_start_col: str | None = None,
    label_window_end_col: str | None = None,
    label_window_start: str | int | pd.Timestamp | datetime | date | None = None,
    label_window_end: str | int | pd.Timestamp | datetime | date | None = None,
) -> TimeLeakageDiagnostics:
    """Diagnose feature observation time against the label performance window.

    ``feature_cols`` are the features to diagnose (each must exist in ``df``).
    Their observation time comes from exactly one of:

    * ``feature_time_cols`` -- a mapping ``{feature: time_column}`` that must
      cover exactly the requested features; or
    * ``observation_time_col`` -- one shared observation-time column.

    The label performance window comes from exactly one of:

    * ``label_window_start_col`` + ``label_window_end_col`` (per-row window); or
    * ``label_window_start`` + ``label_window_end`` (one fixed window).

    Timestamps may be datetime-like columns, parseable date strings, or integral
    ``YYYYMM`` / ``YYYYMMDD`` labels (all-non-null values of a numeric column
    must share one width). A row whose observation time falls inside the closed
    window ``[start, end]`` is an in-window timestamp; any such row raises a
    ``time_leakage_overlap`` red flag, and any observation after ``end`` raises
    a ``time_leakage_after_window`` red flag. The function only returns evidence
    and never blocks the pipeline.
    """
    _validate_dataframe(df, "df")
    features = _validate_features(df, feature_cols)
    time_by_feature = _resolve_feature_time_cols(
        df,
        features=features,
        feature_time_cols=feature_time_cols,
        observation_time_col=observation_time_col,
    )
    start, end, window = _resolve_label_window(
        df,
        start_col=label_window_start_col,
        end_col=label_window_end_col,
        start_ts=label_window_start,
        end_ts=label_window_end,
    )
    rows = tuple(
        _feature_time_leakage(
            df,
            feature=feature,
            time_col=time_by_feature[feature],
            start=start,
            end=end,
        )
        for feature in features
    )
    flags = tuple(flag for row in rows for flag in row.red_flags)
    return TimeLeakageDiagnostics(window=window, features=rows, red_flags=flags)


def selection_bias_diagnostics(
    approval_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    *,
    target_bad_value: int | str = 1,
) -> SelectionBiasDiagnostics:
    """Compare the approval population against the risk population (evidence).

    Each population is summarized independently over its own rows -- no row
    alignment is assumed, so differing population sizes are handled naturally
    and reported as an ``info`` flag rather than an error. ``columns`` must be
    numeric columns present in both DataFrames; ``target`` must exist in both.
    ``target_bad_value`` (default ``1``) selects the "bad" class for the bad
    rate. The result is evidence only and draws no bias verdict.
    """
    _validate_dataframe(approval_df, "approval_df")
    _validate_dataframe(risk_df, "risk_df")
    _assert_columns(approval_df, [target])
    _assert_columns(risk_df, [target])
    summary_columns = _validate_numeric_columns(approval_df, risk_df, columns)
    approval = _population_evidence(
        approval_df,
        population="approval",
        target=target,
        target_bad_value=target_bad_value,
        columns=summary_columns,
    )
    risk = _population_evidence(
        risk_df,
        population="risk",
        target=target,
        target_bad_value=target_bad_value,
        columns=summary_columns,
    )
    flags: list[DiagnosticFlag] = []
    if approval.count != risk.count:
        flags.append(
            DiagnosticFlag(
                code="population_count_mismatch",
                level="info",
                message=(
                    f"approval population has {approval.count} rows and risk "
                    f"population has {risk.count} rows"
                ),
            )
        )
    return SelectionBiasDiagnostics(
        target=target,
        target_bad_value=target_bad_value,
        approval=approval,
        risk=risk,
        flags=tuple(flags),
    )


def _feature_time_leakage(
    df: pd.DataFrame,
    *,
    feature: str,
    time_col: str,
    start: pd.Series,
    end: pd.Series,
) -> FeatureTimeLeakage:
    obs = _parse_time_series(df[time_col], name=time_col)
    observed_rows = int((obs.notna() & start.notna() & end.notna()).sum())
    missing_rows = int(len(df) - observed_rows)
    before = int((obs < start).sum())
    in_window = int(((obs >= start) & (obs <= end)).sum())
    after = int((obs > end).sum())
    if before + in_window + after != observed_rows:
        raise LeakageDiagnosticsError(
            "internal overlap buckets do not conserve observed rows"
        )
    if observed_rows == 0:
        overlap = "undetermined"
    else:
        overlap = "overlap" if in_window > 0 else "no_overlap"
    overlap_rate = (in_window / observed_rows) if observed_rows else 0.0
    flags: list[DiagnosticFlag] = []
    if in_window > 0:
        flags.append(
            DiagnosticFlag(
                code="time_leakage_overlap",
                level="red",
                message=(
                    f"feature {feature!r}: {in_window} of {observed_rows} "
                    "observation timestamps fall inside the label window"
                ),
            )
        )
    if after > 0:
        flags.append(
            DiagnosticFlag(
                code="time_leakage_after_window",
                level="red",
                message=(
                    f"feature {feature!r}: {after} of {observed_rows} "
                    "observation timestamps fall after the label window end"
                ),
            )
        )
    return FeatureTimeLeakage(
        feature=feature,
        time_col=time_col,
        observed_rows=observed_rows,
        missing_rows=missing_rows,
        before_window_count=before,
        in_window_count=in_window,
        after_window_count=after,
        overlap=overlap,
        overlap_rate=overlap_rate,
        red_flags=tuple(flags),
    )


def _population_evidence(
    df: pd.DataFrame,
    *,
    population: str,
    target: str,
    target_bad_value: int | str,
    columns: Sequence[str],
) -> PopulationEvidence:
    count = int(len(df))
    target_series = df[target]
    labeled_count = int(target_series.notna().sum())
    bad_count = int((target_series == target_bad_value).sum())
    bad_rate = (bad_count / labeled_count) if labeled_count else 0.0
    label_coverage = (labeled_count / count) if count else 0.0
    summaries = tuple(
        _column_summary(df[column], total_rows=count, column=column)
        for column in columns
    )
    return PopulationEvidence(
        population=population,
        count=count,
        labeled_count=labeled_count,
        bad_count=bad_count,
        bad_rate=bad_rate,
        label_coverage=label_coverage,
        columns=summaries,
    )


def _column_summary(
    series: pd.Series,
    *,
    total_rows: int,
    column: str,
) -> ColumnSummary:
    nonnull = series.dropna()
    count = int(len(nonnull))
    missing_count = total_rows - count
    missing_rate = (missing_count / total_rows) if total_rows else 0.0
    if count == 0:
        return ColumnSummary(
            column=column,
            count=0,
            missing_count=missing_count,
            missing_rate=missing_rate,
            mean=None,
            std=None,
            min=None,
            median=None,
            max=None,
        )
    values = nonnull.astype("float64")
    mean = _finite_or_none(values.mean())
    std = _finite_or_none(values.std(ddof=0)) if count > 1 else None
    minimum = _finite_or_none(values.min())
    median = _finite_or_none(values.quantile(0.5))
    maximum = _finite_or_none(values.max())
    return ColumnSummary(
        column=column,
        count=count,
        missing_count=missing_count,
        missing_rate=missing_rate,
        mean=mean,
        std=std,
        min=minimum,
        median=median,
        max=maximum,
    )


def _resolve_label_window(
    df: pd.DataFrame,
    *,
    start_col: str | None,
    end_col: str | None,
    start_ts: str | int | pd.Timestamp | datetime | date | None,
    end_ts: str | int | pd.Timestamp | datetime | date | None,
) -> tuple[pd.Series, pd.Series, LabelWindow]:
    has_columns = start_col is not None or end_col is not None
    has_scalars = start_ts is not None or end_ts is not None
    if has_columns and has_scalars:
        raise LeakageDiagnosticsError(
            "label window must be given either by columns or by scalar "
            "timestamps, not both"
        )
    if has_columns:
        if start_col is None or end_col is None:
            raise LeakageDiagnosticsError(
                "label window columns require both start and end columns"
            )
        _assert_columns(df, [start_col, end_col])
        start = _parse_time_series(df[start_col], name=start_col)
        end = _parse_time_series(df[end_col], name=end_col)
        inverted = int((start > end).sum())
        if inverted:
            raise LeakageDiagnosticsError(
                f"label window has {inverted} rows with start after end"
            )
        window = LabelWindow(
            mode="columns",
            start=None,
            end=None,
            start_col=start_col,
            end_col=end_col,
        )
        return start, end, window
    if has_scalars:
        if start_ts is None or end_ts is None:
            raise LeakageDiagnosticsError(
                "label window timestamps require both start and end"
            )
        start_value = _parse_scalar_time(start_ts, name="label_window_start")
        end_value = _parse_scalar_time(end_ts, name="label_window_end")
        if start_value > end_value:
            raise LeakageDiagnosticsError(
                "label_window_start must not be after label_window_end"
            )
        start = pd.Series([start_value] * len(df), index=df.index).astype(
            "datetime64[ns]"
        )
        end = pd.Series([end_value] * len(df), index=df.index).astype(
            "datetime64[ns]"
        )
        window = LabelWindow(
            mode="scalar",
            start=start_value.isoformat(),
            end=end_value.isoformat(),
            start_col=None,
            end_col=None,
        )
        return start, end, window
    raise LeakageDiagnosticsError(
        "a label window is required: pass window columns or scalar timestamps"
    )


def _resolve_feature_time_cols(
    df: pd.DataFrame,
    *,
    features: Sequence[str],
    feature_time_cols: Mapping[str, str] | None,
    observation_time_col: str | None,
) -> dict[str, str]:
    if feature_time_cols is None and observation_time_col is None:
        raise LeakageDiagnosticsError(
            "feature observation time is required: pass feature_time_cols or "
            "observation_time_col"
        )
    if feature_time_cols is not None and observation_time_col is not None:
        raise LeakageDiagnosticsError(
            "pass either per-feature feature_time_cols or one "
            "observation_time_col, not both"
        )
    if observation_time_col is not None:
        _assert_columns(df, [observation_time_col])
        return {feature: observation_time_col for feature in features}
    assert feature_time_cols is not None
    if not isinstance(feature_time_cols, Mapping):
        raise LeakageDiagnosticsError(
            "feature_time_cols must be a mapping of feature -> time column"
        )
    missing = sorted(set(features) - set(feature_time_cols))
    unexpected = sorted(set(feature_time_cols) - set(features))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing time column for: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected features: " + ", ".join(unexpected))
        raise LeakageDiagnosticsError(
            "feature_time_cols must map exactly the requested features ("
            + "; ".join(details)
            + ")"
        )
    result = {
        feature: str(feature_time_cols[feature]) for feature in features
    }
    _assert_columns(df, sorted(set(result.values())))
    return result


def _validate_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(feature_cols, (str, bytes)) or not isinstance(
        feature_cols, Sequence
    ):
        raise LeakageDiagnosticsError(
            "feature_cols must be a non-empty sequence of column names"
        )
    features = tuple(feature_cols)
    if not features:
        raise LeakageDiagnosticsError("feature_cols must not be empty")
    if any(not isinstance(feature, str) or not feature.strip() for feature in features):
        raise LeakageDiagnosticsError(
            "feature_cols entries must be non-empty strings"
        )
    if len(set(features)) != len(features):
        raise LeakageDiagnosticsError("feature_cols must not contain duplicates")
    _assert_columns(df, list(features))
    return features


def _validate_numeric_columns(
    approval_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise LeakageDiagnosticsError(
            "columns must be a sequence of column names"
        )
    result: list[str] = []
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise LeakageDiagnosticsError(
                "columns entries must be non-empty strings"
            )
        if column not in approval_df.columns or column not in risk_df.columns:
            raise LeakageDiagnosticsError(
                f"column {column!r} must exist in both approval_df and risk_df"
            )
        if _is_non_numeric(approval_df[column]):
            raise LeakageDiagnosticsError(
                f"column {column!r} must be numeric in approval_df"
            )
        if _is_non_numeric(risk_df[column]):
            raise LeakageDiagnosticsError(
                f"column {column!r} must be numeric in risk_df"
            )
        result.append(column)
    if len(set(result)) != len(result):
        raise LeakageDiagnosticsError("columns must not contain duplicates")
    return tuple(result)


def _is_non_numeric(series: pd.Series) -> bool:
    return bool(is_bool_dtype(series)) or not bool(is_numeric_dtype(series))


def _parse_time_series(values: pd.Series, *, name: str) -> pd.Series:
    if is_datetime64_any_dtype(values):
        return _as_naive(pd.to_datetime(values, errors="raise"))
    if is_numeric_dtype(values) and not is_bool_dtype(values):
        numeric = pd.to_numeric(values, errors="raise").astype("float64")
        nonnull = numeric[numeric.notna()]
        if nonnull.empty:
            return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        if not nonnull.map(lambda value: float(value).is_integer()).all():
            raise LeakageDiagnosticsError(
                f"{name} numeric values must be integral YYYYMM/YYYYMMDD labels"
            )
        labels = nonnull.astype("int64")
        widths = {len(str(int(label))) for label in labels}
        if widths not in ({6}, {8}):
            raise LeakageDiagnosticsError(
                f"{name} numeric labels must all be YYYYMM (6 digits) or "
                "YYYYMMDD (8 digits)"
            )
        fmt = "%Y%m" if widths == {6} else "%Y%m%d"
        try:
            parsed = pd.to_datetime(labels.astype(str), format=fmt, errors="raise")
        except (TypeError, ValueError) as exc:
            raise LeakageDiagnosticsError(
                f"{name} contains invalid YYYYMM/YYYYMMDD labels"
            ) from exc
        return _as_naive(parsed.reindex(values.index))
    try:
        return _as_naive(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise LeakageDiagnosticsError(
            f"{name} must contain parseable timestamps"
        ) from exc


def _parse_scalar_time(
    value: str | int | pd.Timestamp | datetime | date,
    *,
    name: str,
) -> pd.Timestamp:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
    elif isinstance(value, Integral) and not isinstance(value, bool):
        timestamp = _parse_numeric_label(int(value), name=name)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() and len(stripped) in (6, 8):
            timestamp = _parse_numeric_label(int(stripped), name=name)
        else:
            try:
                timestamp = pd.Timestamp(stripped)
            except (TypeError, ValueError) as exc:
                raise LeakageDiagnosticsError(
                    f"{name} is not a parseable timestamp: {value!r}"
                ) from exc
    else:
        raise LeakageDiagnosticsError(
            f"{name} must be a timestamp, date, or YYYYMM/YYYYMMDD label"
        )
    if pd.isna(timestamp):
        raise LeakageDiagnosticsError(f"{name} must not be null")
    return timestamp


def _parse_numeric_label(value: int, *, name: str) -> pd.Timestamp:
    if value < 0:
        raise LeakageDiagnosticsError(f"{name} numeric labels must be non-negative")
    digits = len(str(value))
    if digits == 6:
        fmt = "%Y%m"
    elif digits == 8:
        fmt = "%Y%m%d"
    else:
        raise LeakageDiagnosticsError(
            f"{name} numeric labels must be YYYYMM (6 digits) or YYYYMMDD "
            "(8 digits)"
        )
    try:
        return pd.to_datetime(str(value), format=fmt, errors="raise")
    except (TypeError, ValueError) as exc:
        raise LeakageDiagnosticsError(
            f"{name} is not a valid YYYYMM/YYYYMMDD label: {value!r}"
        ) from exc


def _as_naive(values: pd.Series) -> pd.Series:
    if getattr(values.dt, "tz", None) is not None:
        return values.dt.tz_localize(None)
    return values


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise LeakageDiagnosticsError(f"{name} must be a pandas DataFrame")


def _assert_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted({column for column in columns if column not in df.columns})
    if missing:
        raise LeakageDiagnosticsError(
            "missing columns: " + ", ".join(missing)
        )


__all__ = [
    "DIAGNOSTIC_FLAG_LEVELS",
    "ColumnSummary",
    "DiagnosticFlag",
    "FeatureTimeLeakage",
    "LabelWindow",
    "LeakageDiagnosticsError",
    "PopulationEvidence",
    "SelectionBiasDiagnostics",
    "TimeLeakageDiagnostics",
    "selection_bias_diagnostics",
    "time_leakage_diagnostics",
]
