"""Deterministic as-of historical replay (逐月重放) for one strategy.

Semantics
=========

"Historical backtest" means: assume the strategy had been live since some month,
then replay its first-match waterfall on the data snapshot that existed *at each
historical month*, and report per-month volume, pass rate, bad rate, and
optionally amount.  Every result is explicitly labelled as a historical replay
(``evidence_stage == "backtested"`` and ``validation_status == "unvalidated"``)
and carries a fixed human-readable declaration that this is *not* real
performance.

This module is deliberately persistence-free and side-effect-free: it never
mutates the supplied Strategy Pool snapshot, canonical strategy spec, or
DataFrames, and it never creates, adopts, promotes, or deploys a strategy.

Formulas and definitions
------------------------

For one month snapshot with ``N`` rows:

* Evaluate the canonical Strategy DSL once with the vectorized first-match
  evaluator (:func:`marvis.packs.strategy.evaluator.evaluate_strategy_frame`),
  which is the exact apply primitive used by ``pool_apply``.  Each row receives
  one typed action from ``{approval, reject, review}`` plus the default action
  for unmatched rows.

* ``count`` (件数)          = ``N``.

* ``pass`` (通过)           = rows whose resolved action type is ``approval``.
  ``pass_count``            = ``|pass|``.
  ``pass_rate``             = ``pass_count / count`` (``0.0`` when ``count == 0``).

* ``bad``                   = a passed row whose target label equals the
  normalized bad value.  Target values must be ``0``/``1``/missing; when
  ``target_bad_value == 0`` the labels are inverted so "bad" is always ``1``
  (mirroring ``pool_impact`` polarity normalization).

* ``labelled_pass_count``   = passed rows with a non-missing target.
  ``label_coverage``        = ``labelled_pass_count / pass_count``
  (``0.0`` when ``pass_count == 0``).
  ``bad_count``             = passed rows with ``target == 1``.
  ``bad_rate``              = ``bad_count / labelled_pass_count``, or ``None``
  when ``labelled_pass_count == 0`` (never a fabricated ``0.0``).

* Amount (only when ``amount_col`` is provided; ``amount is None`` otherwise):
  ``coverage_count``        = passed rows with a non-missing amount.
  ``coverage_rate``         = ``coverage_count / pass_count``.
  ``sum``                   = ``sum(amount)`` over passed non-missing amounts.
  Amount values must be non-negative finite reals (booleans and complex values
  fail closed).

Inputs
------

* ``strategy_ref``: either a Strategy Pool snapshot (``strategy.candidate-pool.v2``)
  or a canonical Strategy DSL spec (``StrategySpec`` / its ``to_dict`` mapping).
  See :func:`resolve_strategy_spec`.

* ``month_snapshots``: an iterable of ``MonthSnapshot`` or ``(month, frame)``
  pairs.  These are the already-resolved as-of data snapshots; resolve them with
  :func:`resolve_month_snapshots` (from one authenticated frame) or
  :func:`resolve_month_snapshots_from_registry` (from the dataset registry).

Fail-closed guarantees
----------------------

* A requested as-of month with no resolvable data snapshot raises
  :class:`MissingMonthSnapshotsError` listing **every** missing month; data is
  never fabricated.
* Row, month, and source-column budgets are enforced before evaluation.
* Strategy types other than ``approval``/``reject`` are rejected because
  "pass"/"bad rate" are not defined for limit/pricing/segmentation semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Real
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.dsl import (
    StrategySpec,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.pool import (
    POOL_SCHEMA_VERSION,
    compile_strategy_pool,
    validate_strategy_pool,
)
from marvis.validation.time_periods import month_key_series


HISTORICAL_BACKTEST_SCHEMA_VERSION = "strategy.historical-backtest.v1"
HISTORICAL_BACKTEST_PRODUCER_VERSION = "marvis.strategy.historical-backtest/1"
HISTORICAL_BACKTEST_MAX_ROWS = 1_000_000
HISTORICAL_BACKTEST_MAX_MONTHS = 240
HISTORICAL_BACKTEST_MAX_SOURCE_COLUMNS = 500

EVIDENCE_STAGE_BACKTESTED = "backtested"
VALIDATION_STATUS_UNVALIDATED = "unvalidated"
HISTORICAL_REPLAY_DECLARATION = (
    "历史重放，非真实业绩：本结果为 as-of 逐月重放（backtested / unvalidated），"
    "不代表策略真实上线业绩，不得作为真实业绩宣称。"
)

_SUPPORTED_STRATEGY_TYPES = frozenset({"approval", "reject"})
_ACTION_NAMES = {"approval": "approve", "reject": "reject", "review": "review"}
_MONTH_RE = re.compile(r"^\d{6}$")


class HistoricalBacktestError(StrategyError):
    """Historical replay inputs or immutable evidence failed closed."""


class MissingMonthSnapshotsError(HistoricalBacktestError):
    """One or more requested as-of months have no resolvable data snapshot."""

    def __init__(self, missing_months: Sequence[str]) -> None:
        normalized = _normalize_month_keys(missing_months, allow_empty=False)
        self.missing_months: tuple[str, ...] = normalized
        rendered = ", ".join(normalized)
        super().__init__(
            "historical replay failed closed: no data snapshot for as-of month(s): "
            + rendered
        )


@dataclass(frozen=True)
class MonthSnapshot:
    """One immutable as-of data snapshot resolved for a single YYYYMM month."""

    month: str
    frame: pd.DataFrame = field(compare=False, repr=False)


@dataclass(frozen=True)
class AmountObservation:
    """Amount evidence over the passed rows of one month."""

    column: str
    coverage_count: int
    coverage_rate: float
    sum: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "coverage_count": self.coverage_count,
            "coverage_rate": self.coverage_rate,
            "sum": self.sum,
        }


@dataclass(frozen=True)
class MonthReplayResult:
    """Deterministic per-month replay metrics.

    ``amount is None`` means no ``amount_col`` was requested; otherwise it
    carries coverage and sum over the passed rows (coverage may be zero).
    """

    month: str
    count: int
    pass_count: int
    pass_rate: float
    reject_count: int
    review_count: int
    labelled_pass_count: int
    label_coverage: float
    bad_count: int
    bad_rate: float | None
    amount: AmountObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "count": self.count,
            "pass_count": self.pass_count,
            "pass_rate": self.pass_rate,
            "reject_count": self.reject_count,
            "review_count": self.review_count,
            "labelled_pass_count": self.labelled_pass_count,
            "label_coverage": self.label_coverage,
            "bad_count": self.bad_count,
            "bad_rate": self.bad_rate,
            "amount": None if self.amount is None else self.amount.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedStrategy:
    """A canonical first-match spec plus its source reference."""

    spec: StrategySpec = field(compare=False, repr=False)
    spec_hash: str
    ref: Mapping[str, Any]


@dataclass(frozen=True)
class HistoricalBacktestResult:
    """Immutable historical replay evidence with explicit evidence markers."""

    strategy_ref: Mapping[str, Any]
    strategy_spec_hash: str
    target_col: str
    target_bad_value: int
    amount_col: str | None
    row_budget: int
    month_count: int
    total_count: int
    total_pass_count: int
    total_pass_rate: float
    total_reject_count: int
    total_review_count: int
    total_labelled_pass_count: int
    total_bad_count: int
    total_bad_rate: float | None
    months: tuple[MonthReplayResult, ...]
    evidence_stage: str = EVIDENCE_STAGE_BACKTESTED
    validation_status: str = VALIDATION_STATUS_UNVALIDATED
    declaration: str = HISTORICAL_REPLAY_DECLARATION
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy_ref",
            MappingProxyType(dict(self.strategy_ref)),
        )
        body = self._body_dict()
        object.__setattr__(self, "content_hash", _sha256(_canonical_json(body)))

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORICAL_BACKTEST_SCHEMA_VERSION,
            "producer_version": HISTORICAL_BACKTEST_PRODUCER_VERSION,
            "evidence_stage": self.evidence_stage,
            "validation_status": self.validation_status,
            "declaration": self.declaration,
            "strategy_ref": dict(self.strategy_ref),
            "strategy_spec_hash": self.strategy_spec_hash,
            "bindings": {
                "target_col": self.target_col,
                "target_bad_value": self.target_bad_value,
                "amount_col": self.amount_col,
            },
            "row_budget": self.row_budget,
            "month_count": self.month_count,
            "total": {
                "count": self.total_count,
                "pass_count": self.total_pass_count,
                "pass_rate": self.total_pass_rate,
                "reject_count": self.total_reject_count,
                "review_count": self.total_review_count,
                "labelled_pass_count": self.total_labelled_pass_count,
                "bad_count": self.total_bad_count,
                "bad_rate": self.total_bad_rate,
            },
            "months": [month.to_dict() for month in self.months],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body_dict(), "content_hash": self.content_hash}


def resolve_strategy_spec(
    strategy_ref: StrategySpec | Mapping[str, Any],
) -> ResolvedStrategy:
    """Resolve a Strategy Pool snapshot or canonical strategy spec to a spec.

    ``strategy_ref`` may be:

    * a :class:`~marvis.packs.strategy.dsl.StrategySpec`,
    * a canonical Strategy DSL spec mapping (``strategy_type``/``default_action``/
      ``rules``), or
    * a Strategy Pool snapshot (``strategy.candidate-pool.v2``).

    Pool snapshots are validated and compiled exactly like ``pool_apply``, and
    the resolved reference records the pool ``pool_ref`` (``pool_id``, ``task_id``,
    ``strategy_type``, ``revision``, ``revision_id``, ``snapshot_hash``) plus the
    compiled ``design_hash``.  Canonical strategies record their DSL
    ``strategy_spec_hash``.
    """

    if isinstance(strategy_ref, StrategySpec):
        spec = strategy_ref
        return ResolvedStrategy(
            spec=spec,
            spec_hash=strategy_spec_hash(spec),
            ref={
                "kind": "canonical_strategy",
                "schema_version": spec.schema_version,
                "strategy_spec_hash": strategy_spec_hash(spec),
            },
        )
    if not isinstance(strategy_ref, Mapping):
        raise HistoricalBacktestError(
            "strategy_ref must be a StrategySpec, a canonical strategy spec "
            "mapping, or a Strategy Pool snapshot"
        )
    if strategy_ref.get("schema_version") == POOL_SCHEMA_VERSION or (
        "entries" in strategy_ref and "strategy_type" in strategy_ref
    ):
        try:
            pool = validate_strategy_pool(strategy_ref)
            design = compile_strategy_pool(pool)
            spec = parse_strategy_spec(design["strategy_spec"])
        except StrategyError as exc:
            raise HistoricalBacktestError(
                f"strategy_ref is not a valid Strategy Pool snapshot: {exc}"
            ) from exc
        return ResolvedStrategy(
            spec=spec,
            spec_hash=strategy_spec_hash(spec),
            ref={
                "kind": "strategy_pool",
                "pool_ref": dict(design["pool_ref"]),
                "design_hash": str(design["design_hash"]),
            },
        )
    try:
        spec = parse_strategy_spec(strategy_ref)
    except StrategyError as exc:
        raise HistoricalBacktestError(
            "strategy_ref is neither a valid Strategy Pool snapshot nor a "
            f"canonical strategy spec: {exc}"
        ) from exc
    return ResolvedStrategy(
        spec=spec,
        spec_hash=strategy_spec_hash(spec),
        ref={
            "kind": "canonical_strategy",
            "schema_version": spec.schema_version,
            "strategy_spec_hash": strategy_spec_hash(spec),
        },
    )


def resolve_month_snapshots(
    frame: pd.DataFrame,
    *,
    month_col: str,
    requested_months: Sequence[str],
    max_rows: int = HISTORICAL_BACKTEST_MAX_ROWS,
    max_columns: int = HISTORICAL_BACKTEST_MAX_SOURCE_COLUMNS,
) -> tuple[MonthSnapshot, ...]:
    """Split one authenticated frame into ordered as-of month snapshots.

    ``requested_months`` is normalized to a sorted, duplicate-free sequence of
    canonical ``YYYYMM`` keys.  Every requested month must be present in
    ``frame[month_col]``; any missing month raises
    :class:`MissingMonthSnapshotsError` with the complete sorted list of missing
    months.  No month's data is ever synthesized.
    """

    working = _require_frame(
        frame,
        required_columns=(month_col,),
        max_rows=max_rows,
        max_columns=max_columns,
    )
    requested = _normalize_month_keys(requested_months, allow_empty=False)
    try:
        keys = month_key_series(working[month_col], column_name=month_col)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HistoricalBacktestError(
            f"month column {month_col!r} is invalid: {exc}"
        ) from exc
    keys = keys.reset_index(drop=True)
    present = set(keys.unique().tolist())
    missing = sorted(set(requested) - present)
    if missing:
        raise MissingMonthSnapshotsError(missing)

    snapshots: list[MonthSnapshot] = []
    for month in requested:
        mask = (keys == month).to_numpy(dtype=bool)
        sub = working[mask].reset_index(drop=True)
        snapshots.append(MonthSnapshot(month=month, frame=sub))
    return tuple(snapshots)


def resolve_month_snapshots_from_registry(
    registry,
    dataset_id: str,
    *,
    month_col: str,
    requested_months: Sequence[str],
    max_rows: int = HISTORICAL_BACKTEST_MAX_ROWS,
    max_columns: int = HISTORICAL_BACKTEST_MAX_SOURCE_COLUMNS,
) -> tuple[MonthSnapshot, ...]:
    """Read an authenticated dataset snapshot and split it per as-of month.

    The registry must expose ``read_authenticated_parquet_snapshot(dataset_id)``
    (as :class:`~marvis.data.registry.DatasetRegistry` does), which verifies the
    registered content hash and fails closed on drift.  A missing dataset raises
    a typed :class:`HistoricalBacktestError`; a missing month raises
    :class:`MissingMonthSnapshotsError`.
    """

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise HistoricalBacktestError("dataset_id must be a non-empty string")
    read = getattr(registry, "read_authenticated_parquet_snapshot", None)
    if not callable(read):
        raise HistoricalBacktestError(
            "registry must expose read_authenticated_parquet_snapshot(dataset_id)"
        )
    try:
        frame = read(dataset_id)
    except KeyError as exc:
        raise HistoricalBacktestError(
            f"dataset is not resolvable: {dataset_id}"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise HistoricalBacktestError(
            "registry read_authenticated_parquet_snapshot must return a DataFrame"
        )
    return resolve_month_snapshots(
        frame,
        month_col=month_col,
        requested_months=requested_months,
        max_rows=max_rows,
        max_columns=max_columns,
    )


def replay_historical_months(
    strategy_ref: StrategySpec | Mapping[str, Any],
    month_snapshots: Iterable[MonthSnapshot | tuple[str, pd.DataFrame]],
    *,
    target_col: str,
    target_bad_value: int = 1,
    amount_col: str | None = None,
    row_budget: int = HISTORICAL_BACKTEST_MAX_ROWS,
    max_months: int = HISTORICAL_BACKTEST_MAX_MONTHS,
    max_columns: int = HISTORICAL_BACKTEST_MAX_SOURCE_COLUMNS,
) -> HistoricalBacktestResult:
    """Replay one strategy deterministically over as-of month snapshots.

    Returns immutable :class:`HistoricalBacktestResult` evidence explicitly
    marked ``backtested``/``unvalidated`` with a "历史重放、非真实业绩"
    declaration.  Inputs are never mutated.
    """

    resolved = resolve_strategy_spec(strategy_ref)
    if resolved.spec.strategy_type not in _SUPPORTED_STRATEGY_TYPES:
        raise HistoricalBacktestError(
            "historical pass/bad-rate replay supports approval/reject strategies "
            f"only; got strategy_type {resolved.spec.strategy_type!r}"
        )
    bad_value = _target_bad_value(target_bad_value)
    target = _column(target_col, "target_col")
    amount = _optional_column(amount_col, "amount_col")
    if amount is not None and amount == target:
        raise HistoricalBacktestError("target_col and amount_col must be different")

    ordered = _normalize_month_snapshots(month_snapshots)
    if len(ordered) > max_months:
        raise HistoricalBacktestError(
            f"historical replay exceeds the {max_months}-month limit"
        )
    total_rows = sum(len(frame) for _, frame in ordered)
    if total_rows > row_budget:
        raise HistoricalBacktestError(
            f"historical replay exceeds the {row_budget}-row budget"
        )

    months: list[MonthReplayResult] = []
    for month, frame in ordered:
        working = _require_frame(
            frame,
            required_columns=tuple(
                column for column in (target, amount) if column is not None
            ),
            max_rows=row_budget,
            max_columns=max_columns,
        )
        months.append(
            _replay_month(
                working,
                month=month,
                spec=resolved.spec,
                target_col=target,
                bad_value=bad_value,
                amount_col=amount,
            )
        )

    totals = _rollup(months)
    return HistoricalBacktestResult(
        strategy_ref=dict(resolved.ref),
        strategy_spec_hash=resolved.spec_hash,
        target_col=target,
        target_bad_value=bad_value,
        amount_col=amount,
        row_budget=row_budget,
        month_count=len(months),
        total_count=totals["count"],
        total_pass_count=totals["pass_count"],
        total_pass_rate=totals["pass_rate"],
        total_reject_count=totals["reject_count"],
        total_review_count=totals["review_count"],
        total_labelled_pass_count=totals["labelled_pass_count"],
        total_bad_count=totals["bad_count"],
        total_bad_rate=totals["bad_rate"],
        months=tuple(months),
    )


def canonical_historical_backtest_json(result: HistoricalBacktestResult) -> str:
    """Return the sole byte-stable JSON representation for one result."""

    if not isinstance(result, HistoricalBacktestResult):
        raise HistoricalBacktestError("result must be a HistoricalBacktestResult")
    return _canonical_json(result.to_dict())


def _replay_month(
    frame: pd.DataFrame,
    *,
    month: str,
    spec: StrategySpec,
    target_col: str,
    bad_value: int,
    amount_col: str | None,
) -> MonthReplayResult:
    evaluation = evaluate_strategy_frame(frame, spec)
    action_types = evaluation.action_type.reset_index(drop=True).to_numpy(
        dtype=object
    )
    unknown = [
        str(value)
        for value in np.unique(action_types)
        if value not in _ACTION_NAMES
    ]
    if unknown:
        raise HistoricalBacktestError(
            "strategy produced unsupported action(s): " + ", ".join(unknown)
        )

    pass_mask = action_types == "approval"
    count = int(len(frame))
    pass_count = int(np.count_nonzero(pass_mask))
    reject_count = int(np.count_nonzero(action_types == "reject"))
    review_count = int(np.count_nonzero(action_types == "review"))
    if pass_count + reject_count + review_count != count:
        raise HistoricalBacktestError(
            "approval/reject/review actions do not cover the month population"
        )

    target = _target_series(frame[target_col], bad_value=bad_value)
    labelled = pass_mask & ~np.isnan(target)
    labelled_pass_count = int(np.count_nonzero(labelled))
    bad_count = int(np.count_nonzero(labelled & (target == 1.0)))

    amount_obs: AmountObservation | None = None
    if amount_col is not None:
        values = _amount_series(frame[amount_col], column=amount_col)
        covered = pass_mask & ~np.isnan(values)
        coverage_count = int(np.count_nonzero(covered))
        amount_obs = AmountObservation(
            column=amount_col,
            coverage_count=coverage_count,
            coverage_rate=_ratio(coverage_count, pass_count),
            sum=float(np.nansum(values[covered])),
        )

    return MonthReplayResult(
        month=month,
        count=count,
        pass_count=pass_count,
        pass_rate=_ratio(pass_count, count),
        reject_count=reject_count,
        review_count=review_count,
        labelled_pass_count=labelled_pass_count,
        label_coverage=_ratio(labelled_pass_count, pass_count),
        bad_count=bad_count,
        bad_rate=(
            None if labelled_pass_count == 0 else _ratio(bad_count, labelled_pass_count)
        ),
        amount=amount_obs,
    )


def _rollup(months: Sequence[MonthReplayResult]) -> dict[str, Any]:
    count = sum(int(month.count) for month in months)
    pass_count = sum(int(month.pass_count) for month in months)
    reject_count = sum(int(month.reject_count) for month in months)
    review_count = sum(int(month.review_count) for month in months)
    labelled_pass_count = sum(int(month.labelled_pass_count) for month in months)
    bad_count = sum(int(month.bad_count) for month in months)
    return {
        "count": count,
        "pass_count": pass_count,
        "pass_rate": _ratio(pass_count, count),
        "reject_count": reject_count,
        "review_count": review_count,
        "labelled_pass_count": labelled_pass_count,
        "bad_count": bad_count,
        "bad_rate": (
            None if labelled_pass_count == 0 else _ratio(bad_count, labelled_pass_count)
        ),
    }


def _normalize_month_snapshots(
    month_snapshots: Iterable[MonthSnapshot | tuple[str, pd.DataFrame]],
) -> tuple[tuple[str, pd.DataFrame], ...]:
    items: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for item in month_snapshots:
        if isinstance(item, MonthSnapshot):
            month, frame = item.month, item.frame
        elif isinstance(item, tuple) and len(item) == 2:
            month, frame = item
        else:
            raise HistoricalBacktestError(
                "month_snapshots must contain MonthSnapshot or (month, frame) pairs"
            )
        month = _month(month, "month")
        if month in seen:
            raise HistoricalBacktestError(f"duplicate as-of month: {month}")
        seen.add(month)
        if not isinstance(frame, pd.DataFrame):
            raise HistoricalBacktestError(
                "month snapshot frame must be a pandas DataFrame"
            )
        items.append((month, frame))
    if not items:
        raise HistoricalBacktestError("month_snapshots must not be empty")
    items.sort(key=lambda pair: pair[0])
    return tuple(items)


def _require_frame(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    max_rows: int,
    max_columns: int,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise HistoricalBacktestError("rows must be a pandas DataFrame")
    if len(frame) > max_rows:
        raise HistoricalBacktestError(
            f"source exceeds the {max_rows}-row budget"
        )
    if len(frame.columns) > max_columns:
        raise HistoricalBacktestError(
            f"source exceeds the {max_columns}-column budget"
        )
    if frame.columns.duplicated().any():
        raise HistoricalBacktestError("source DataFrame has duplicate columns")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise HistoricalBacktestError(
            "source is missing columns: " + ", ".join(missing)
        )
    return frame.reset_index(drop=True)


def _normalize_month_keys(value: object, *, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise HistoricalBacktestError("months must be a sequence of YYYYMM keys")
    if not value and not allow_empty:
        raise HistoricalBacktestError("months must not be empty")
    normalized = [_month(item, "month") for item in value]
    return tuple(sorted(set(normalized)))


def _month(value: object, name: str = "month") -> str:
    if not isinstance(value, str):
        raise HistoricalBacktestError(f"{name} must be a YYYYMM string")
    text = value.strip()
    if _MONTH_RE.fullmatch(text) is None:
        raise HistoricalBacktestError(f"{name} must use the YYYYMM format")
    month_number = int(text[4:])
    if not 1 <= month_number <= 12:
        raise HistoricalBacktestError(f"{name} must be a valid YYYYMM")
    return text


def _column(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoricalBacktestError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_column(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _column(value, name)


def _target_bad_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise HistoricalBacktestError("target_bad_value must be integer 0 or 1")
    return value


def _target_series(values: pd.Series, *, bad_value: int) -> np.ndarray:
    raw = values.reset_index(drop=True)
    missing = raw.isna()
    numeric = pd.to_numeric(raw, errors="coerce")
    coerced = numeric.to_numpy(dtype=float)
    non_missing = ~missing.to_numpy(dtype=bool)
    if bool((non_missing & (~np.isfinite(coerced))).any()):
        raise HistoricalBacktestError("target must contain only 0, 1, or missing")
    if bool((non_missing & ~np.isin(coerced, [0.0, 1.0])).any()):
        raise HistoricalBacktestError("target must contain only 0, 1, or missing")
    if bad_value == 0:
        coerced = np.where(non_missing, 1.0 - coerced, np.nan)
    return coerced


def _amount_series(values: pd.Series, *, column: str) -> np.ndarray:
    raw = values.reset_index(drop=True)
    unsupported = raw.notna() & raw.map(
        lambda value: not _is_supported_amount_scalar(value)
    )
    if bool(unsupported.any()):
        raise HistoricalBacktestError(
            f"amount column {column!r} must contain non-negative finite real "
            "numbers or missing"
        )
    numeric = pd.to_numeric(raw, errors="coerce")
    coerced = numeric.to_numpy(dtype=float)
    non_missing = raw.notna().to_numpy(dtype=bool)
    if bool((non_missing & ~np.isfinite(coerced)).any()):
        raise HistoricalBacktestError(
            f"amount column {column!r} must contain non-negative finite real "
            "numbers or missing"
        )
    if bool((non_missing & (coerced < 0.0)).any()):
        raise HistoricalBacktestError(
            f"amount column {column!r} must contain non-negative finite real "
            "numbers or missing"
        )
    return coerced


def _is_supported_amount_scalar(value: object) -> bool:
    if isinstance(value, bool | np.bool_ | complex | np.complexfloating):
        return False
    return isinstance(value, str | Real)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HistoricalBacktestError(
            "historical replay evidence is not canonical JSON"
        ) from exc


__all__ = [
    "EVIDENCE_STAGE_BACKTESTED",
    "HISTORICAL_BACKTEST_MAX_MONTHS",
    "HISTORICAL_BACKTEST_MAX_ROWS",
    "HISTORICAL_BACKTEST_MAX_SOURCE_COLUMNS",
    "HISTORICAL_BACKTEST_PRODUCER_VERSION",
    "HISTORICAL_BACKTEST_SCHEMA_VERSION",
    "HISTORICAL_REPLAY_DECLARATION",
    "VALIDATION_STATUS_UNVALIDATED",
    "AmountObservation",
    "HistoricalBacktestError",
    "HistoricalBacktestResult",
    "MissingMonthSnapshotsError",
    "MonthReplayResult",
    "MonthSnapshot",
    "ResolvedStrategy",
    "canonical_historical_backtest_json",
    "replay_historical_months",
    "resolve_month_snapshots",
    "resolve_month_snapshots_from_registry",
    "resolve_strategy_spec",
]
