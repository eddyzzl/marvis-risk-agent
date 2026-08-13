"""B-10: typed limit/pricing impact measurement (development evidence).

This module complements :mod:`marvis.packs.strategy.impact_cube` (which covers
approval/reject) with deterministic, side-effect-free impact computation for the
two *value* strategy families: ``limit`` and ``pricing``.  It is persistence-free
and does not touch the Strategy Pool, create/adopt/promote/deploy any strategy,
or write artifacts.  Every result is explicitly development-only measurement
evidence (``effect_stage="backtested"``, ``validation_status="unvalidated"``).

It reuses the same fail-closed conventions as the rest of the strategy pack:

* Missing economic inputs never silently become ``0``.  A field that cannot be
  computed is emitted as a *typed* ``{"availability": "unavailable", "reason":
  ..., "value": None}`` value, or the whole call raises :class:`StrategyError`
  for structurally invalid inputs (negative limits, misaligned indexes,
  non-binary targets, out-of-range PD/LGD).
* Indexes must match exactly; no silent realignment.
* Results are JSON-safe and deterministic (no random state, no persistence).

Formulas and business assumptions
---------------------------------

limit_impact
~~~~~~~~~~~~
Let ``L_i`` be the assigned (post-change) limit, ``L0_i`` the baseline limit,
``E_i`` the post-change exposure/EAD, ``E0_i`` the baseline exposure, ``PD_i``
the probability of default and ``LGD_i`` the loss given default.

* Exposure resolution: ``E_i`` is the supplied ``exposure`` column, or
  ``L_i * utilization_i`` when only ``utilization`` is supplied.  ``E0_i`` is
  ``baseline_exposure``, or ``L0_i * utilization_i``.
* Total exposure: ``after = sum(E_i)``, ``before = sum(E0_i)``,
  ``delta = after - before``.
* Action buckets (direction of the exposure change, consistent with
  ``impact_cube._transition_direction`` for limit strategies):

  * ``up``: ``E_i > E0_i``
  * ``down``: ``E_i < E0_i``
  * ``unchanged``: ``E_i == E0_i``

  Each bucket reports its row count and its ``exposure_delta =
  sum_{i in bucket}(E_i - E0_i)``.  The three bucket deltas sum to the total
  exposure delta.
* Expected loss:

  * ``EL_after = sum(E_i * PD_i * LGD_i)``
  * ``EL_before = sum(E0_i * PD_i * LGD_i)``
  * ``EL_delta = EL_after - EL_before``

* PD source: a supplied ``pd`` column/scalar is used directly.  Otherwise, if a
  binary ``target`` (1 = bad, 0 = good, missing allowed) is supplied, the
  population empirical bad rate is used as a coarse PD proxy and flagged with
  ``pd_proxy_used``.  EL is ``unavailable`` when neither PD nor a labelled
  target is supplied, or when ``lgd`` is missing.

pricing_impact
~~~~~~~~~~~~~~
Let ``E_i`` be the exposure/EAD of the target population, ``dr_i`` the annual
decimal rate delta, ``term_i`` the term in months, ``dfee_i`` the per-loan fee
delta, ``PD_i`` the probability of default and ``LGD_i`` the loss given default.

* Incremental revenue:

  * rate delta: ``revenue = sum(E_i * dr_i * term_i / 12)``
  * fee delta:  ``revenue = sum(dfee_i)``

  Exactly one of ``rate_delta``/``fee_delta`` must be supplied.  A rate delta
  requires ``term_months``.

* Incremental expected bad-debt cost:

  ``expected_loss = sum(E_i * PD_i * LGD_i)``

* Net impact: ``net_impact = revenue - expected_loss`` (available only when the
  expected-loss chain is available).

Business assumptions
~~~~~~~~~~~~~~~~~~~~
* Repricing does **not** shift PD, LGD, or EAD (no demand-elasticity or
  adverse-selection response is modelled); the increment is a first-order,
  point-estimate development estimate.
* ``target`` encodes ``1 = bad``, ``0 = good``; missing labels are retained in
  counts but excluded from bad-rate denominators.
* The segment x month matrix is aggregate evidence only, with one cell per
  observed (segment, month) combination; it is not a privacy boundary (that is
  :mod:`impact_cube`'s governed concern).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from typing import Any

import pandas as pandas

from marvis.packs.strategy.errors import StrategyError


LIMIT_PRICING_IMPACT_SCHEMA_VERSION = "strategy.limit-pricing-impact.v1"

_EFFECT_STAGE = "backtested"
_VALIDATION_STATUS = "unvalidated"

_LIFECYCLE = {
    "mutates_pool": False,
    "creates_strategy": False,
    "adopts_strategy": False,
    "promotes_strategy": False,
    "deploys_strategy": False,
}

_ACTION_DIRECTIONS = ("up", "down", "unchanged")


def _typed_field(
    availability: str,
    value: Any,
    reason: str | None,
) -> dict[str, Any]:
    """Build a typed availability field, mirroring the ImpactCube convention."""

    if availability == "present":
        if value is None or reason is not None:
            raise StrategyError("present impact fields require a value")
    elif availability == "unavailable":
        if value is not None or not isinstance(reason, str) or not reason:
            raise StrategyError(
                "unavailable impact fields require a reason and null value"
            )
    else:
        raise StrategyError("impact availability is invalid")
    return {
        "availability": availability,
        "reason": reason,
        "value": value,
    }


def _check_bounds(
    value: float,
    *,
    name: str,
    lower: float | None,
    upper: float | None,
    lower_inclusive: bool,
) -> None:
    if lower is not None:
        invalid = value < lower if lower_inclusive else value <= lower
        if invalid:
            operator = ">=" if lower_inclusive else ">"
            raise StrategyError(f"{name} must be {operator} {lower:g}")
    if upper is not None and value > upper:
        raise StrategyError(f"{name} must be <= {upper:g}")


def _coerce_numeric_series(
    value: pandas.Series,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> pandas.Series:
    try:
        numeric = pandas.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError(f"{name} must contain numeric values") from exc
    if numeric.isna().any() or not numeric.map(math.isfinite).all():
        raise StrategyError(f"{name} must contain finite non-missing values")
    if lower is not None:
        invalid = numeric.lt(lower) if lower_inclusive else numeric.le(lower)
        if bool(invalid.any()):
            operator = ">=" if lower_inclusive else ">"
            raise StrategyError(f"{name} values must be {operator} {lower:g}")
    if upper is not None and bool(numeric.gt(upper).any()):
        raise StrategyError(f"{name} values must be <= {upper:g}")
    return numeric


def _required_series(
    value: pandas.Series,
    *,
    name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> pandas.Series:
    if not isinstance(value, pandas.Series):
        raise StrategyError(f"{name} must be a pandas Series")
    return _coerce_numeric_series(value, name=name, lower=lower, upper=upper)


def _optional_series(
    value: pandas.Series | None,
    *,
    name: str,
    index: pandas.Index,
    lower: float | None = None,
    upper: float | None = None,
) -> pandas.Series | None:
    if value is None:
        return None
    if not isinstance(value, pandas.Series):
        raise StrategyError(f"{name} must be a pandas Series")
    if not value.index.equals(index):
        raise StrategyError(f"{name} index must exactly match assigned values")
    return _coerce_numeric_series(value, name=name, lower=lower, upper=upper)


def _numeric_input(
    value: pandas.Series | Real,
    *,
    name: str,
    index: pandas.Index,
    lower: float | None = None,
    upper: float | None = None,
    lower_inclusive: bool = True,
) -> pandas.Series:
    if isinstance(value, pandas.Series):
        if not value.index.equals(index):
            raise StrategyError(f"{name} index must exactly match assigned values")
        return _coerce_numeric_series(
            value,
            name=name,
            lower=lower,
            upper=upper,
            lower_inclusive=lower_inclusive,
        )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"{name} must be a numeric scalar or pandas Series")
    number = float(value)
    if not math.isfinite(number):
        raise StrategyError(f"{name} must be finite")
    _check_bounds(
        number,
        name=name,
        lower=lower,
        upper=upper,
        lower_inclusive=lower_inclusive,
    )
    return pandas.Series([number] * len(index), index=index, dtype=float)


def _target_series(
    value: pandas.Series | None,
    *,
    index: pandas.Index,
) -> pandas.Series | None:
    if value is None:
        return None
    if not isinstance(value, pandas.Series):
        raise StrategyError("target must be a pandas Series")
    if not value.index.equals(index):
        raise StrategyError("target index must exactly match assigned values")
    try:
        numeric = pandas.to_numeric(value, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise StrategyError(
            "target must contain binary values or missing labels"
        ) from exc
    finite = numeric.dropna()
    if not finite.map(math.isfinite).all() or not finite.isin([0.0, 1.0]).all():
        raise StrategyError("target must contain only 0, 1, or missing labels")
    return numeric


def _object_series(
    value: pandas.Series | None,
    *,
    name: str,
    index: pandas.Index,
) -> pandas.Series | None:
    if value is None:
        return None
    if not isinstance(value, pandas.Series):
        raise StrategyError(f"{name} must be a pandas Series")
    if not value.index.equals(index):
        raise StrategyError(f"{name} index must exactly match assigned values")
    return value.reset_index(drop=True)


def _finite_sum(values: list[float], *, name: str) -> float:
    total = math.fsum(float(value) for value in values)
    if not math.isfinite(total):
        raise StrategyError(f"{name} produced a non-finite value")
    return 0.0 if total == 0.0 else float(total)


def _checked_sum(values: list[float], *, name: str) -> float:
    total = math.fsum(values)
    if not math.isfinite(total):
        raise StrategyError(f"{name} produced a non-finite value")
    return 0.0 if total == 0.0 else float(total)


def _resolve_ead(
    limit: pandas.Series,
    exposure: pandas.Series | Real | None,
    utilization: pandas.Series | Real | None,
    *,
    name: str,
    index: pandas.Index,
) -> pandas.Series:
    if exposure is not None:
        return _numeric_input(exposure, name=name, index=index, lower=0.0)
    if utilization is not None:
        u = _numeric_input(
            utilization,
            name="utilization",
            index=index,
            lower=0.0,
            upper=1.0,
        )
        product = (limit * u).astype(float)
        if product.isna().any() or not product.map(math.isfinite).all():
            raise StrategyError(f"{name} derived exposure must be finite")
        return product
    raise StrategyError(
        f"limit_impact requires {name} or utilization to derive exposure"
    )


def _cell_value(value: Any) -> Any:
    try:
        missing = bool(pandas.isna(value)) if not isinstance(value, str) else False
    except (TypeError, ValueError):
        missing = False
    if value is None or missing:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, str):
        return value
    return str(value)


def _cell_token(value: Any) -> str:
    cell = _cell_value(value)
    if cell is None:
        return "null"
    return json.dumps(
        cell,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _cell_groups(
    segment: pandas.Series,
    month: pandas.Series,
) -> list[dict[str, Any]]:
    tokens = [
        (_cell_token(s), _cell_token(m))
        for s, m in zip(segment, month, strict=True)
    ]
    seen: dict[tuple[str, str], tuple[Any, Any]] = {}
    for token, seg_value, month_value in zip(
        tokens, segment, month, strict=True
    ):
        seen.setdefault(
            token, (_cell_value(seg_value), _cell_value(month_value))
        )
    return [
        {
            "segment": seen[token][0],
            "month": seen[token][1],
            "positions": [
                index for index, item in enumerate(tokens) if item == token
            ],
        }
        for token in sorted(seen)
    ]


def _cell_label_evidence(
    target: pandas.Series | None,
    positions: list[int],
) -> dict[str, Any]:
    if target is None:
        return {"labeled_count": 0, "bad_count": 0, "bad_rate": None}
    cell_target = pandas.Series(
        [target.iloc[position] for position in positions], dtype=float
    )
    labeled_count = int(cell_target.notna().sum())
    bad_count = int(cell_target.eq(1.0).sum())
    return {
        "labeled_count": labeled_count,
        "bad_count": bad_count,
        "bad_rate": (
            None if labeled_count == 0 else float(bad_count / labeled_count)
        ),
    }


def _limit_segment_month_cells(
    *,
    segment: pandas.Series | None,
    month: pandas.Series | None,
    after: pandas.Series,
    before: pandas.Series | None,
    pd_series: pandas.Series | None,
    lgd_series: pandas.Series | None,
    target: pandas.Series | None,
    el_available: bool,
    baseline_bound: bool,
) -> dict[str, Any]:
    if segment is None or month is None:
        missing = [
            name
            for name, present in (
                ("segment", segment is not None),
                ("month", month is not None),
            )
            if not present
        ]
        return _typed_field(
            "unavailable", None, "_and_".join(missing) + "_field_not_bound"
        )
    after_values = after.tolist()
    before_values = None if before is None else before.tolist()
    pd_values = None if pd_series is None else pd_series.tolist()
    lgd_values = None if lgd_series is None else lgd_series.tolist()
    rows: list[dict[str, Any]] = []
    for group in _cell_groups(segment, month):
        positions = group["positions"]
        row = {
            "segment": group["segment"],
            "month": group["month"],
            "count": len(positions),
            **_cell_label_evidence(target, positions),
            "exposure_delta": None,
            "expected_loss_after": None,
            "expected_loss_delta": None,
        }
        if baseline_bound:
            assert before_values is not None
            row["exposure_delta"] = _finite_sum(
                [
                    float(after_values[position])
                    - float(before_values[position])
                    for position in positions
                ],
                name="cell exposure_delta",
            )
        if el_available:
            assert pd_values is not None and lgd_values is not None
            row["expected_loss_after"] = _finite_sum(
                [
                    float(after_values[position])
                    * float(pd_values[position])
                    * float(lgd_values[position])
                    for position in positions
                ],
                name="cell expected_loss_after",
            )
            if baseline_bound:
                assert before_values is not None
                expected_loss_before = _finite_sum(
                    [
                        float(before_values[position])
                        * float(pd_values[position])
                        * float(lgd_values[position])
                        for position in positions
                    ],
                    name="cell expected_loss_before",
                )
                row["expected_loss_delta"] = _checked_sum(
                    [row["expected_loss_after"], -expected_loss_before],
                    name="cell expected_loss_delta",
                )
        rows.append(row)
    return _typed_field("present", {"rows": rows}, None)


def _pricing_segment_month_cells(
    *,
    segment: pandas.Series | None,
    month: pandas.Series | None,
    exposure_values: list[float],
    pd_series: pandas.Series | None,
    lgd_series: pandas.Series | None,
    target: pandas.Series | None,
    el_available: bool,
    revenue_per_position: list[float],
) -> dict[str, Any]:
    if segment is None or month is None:
        missing = [
            name
            for name, present in (
                ("segment", segment is not None),
                ("month", month is not None),
            )
            if not present
        ]
        return _typed_field(
            "unavailable", None, "_and_".join(missing) + "_field_not_bound"
        )
    pd_values = None if pd_series is None else pd_series.tolist()
    lgd_values = None if lgd_series is None else lgd_series.tolist()
    rows: list[dict[str, Any]] = []
    for group in _cell_groups(segment, month):
        positions = group["positions"]
        row = {
            "segment": group["segment"],
            "month": group["month"],
            "count": len(positions),
            **_cell_label_evidence(target, positions),
            "incremental_revenue": _finite_sum(
                [revenue_per_position[position] for position in positions],
                name="cell incremental_revenue",
            ),
            "expected_loss": None,
            "net_impact": None,
        }
        if el_available:
            assert pd_values is not None and lgd_values is not None
            row["expected_loss"] = _finite_sum(
                [
                    float(exposure_values[position])
                    * float(pd_values[position])
                    * float(lgd_values[position])
                    for position in positions
                ],
                name="cell expected_loss",
            )
            row["net_impact"] = _checked_sum(
                [row["incremental_revenue"], -row["expected_loss"]],
                name="cell net_impact",
            )
        rows.append(row)
    return _typed_field("present", {"rows": rows}, None)


@dataclass(frozen=True)
class LimitImpactResult:
    """Deterministic, development-only typed impact evidence for a limit change."""

    schema_version: str
    strategy_type: str
    count: int
    labeled_count: int
    total_limit: float
    total_exposure_after: float
    exposure: dict[str, Any]
    action_buckets: dict[str, Any]
    expected_loss: dict[str, Any]
    by_segment_month: dict[str, Any]
    lifecycle: dict[str, bool]
    red_flags: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_type": self.strategy_type,
            "effect_stage": _EFFECT_STAGE,
            "validation_status": _VALIDATION_STATUS,
            "count": self.count,
            "labeled_count": self.labeled_count,
            "total_limit": self.total_limit,
            "total_exposure_after": self.total_exposure_after,
            "exposure": self.exposure,
            "action_buckets": self.action_buckets,
            "expected_loss": self.expected_loss,
            "by_segment_month": self.by_segment_month,
            "lifecycle": self.lifecycle,
            "red_flags": list(self.red_flags),
        }


def limit_impact(
    *,
    assigned_limit: pandas.Series,
    exposure: pandas.Series | Real | None = None,
    baseline_limit: pandas.Series | None = None,
    baseline_exposure: pandas.Series | Real | None = None,
    pd: pandas.Series | Real | None = None,
    lgd: pandas.Series | Real | None = None,
    target: pandas.Series | None = None,
    utilization: pandas.Series | Real | None = None,
    segment: pandas.Series | None = None,
    month: pandas.Series | None = None,
) -> LimitImpactResult:
    """Measure the typed impact of a limit strategy change.

    ``assigned_limit`` is the post-change strategy application result (one row per
    population member).  ``baseline_limit`` is the pre-change limit; without it the
    before/after comparison, action buckets, and EL delta are reported as
    ``unavailable`` (never as fabricated zeros).
    """

    assigned = _required_series(
        assigned_limit, name="assigned_limit", lower=0.0
    )
    index = assigned.index
    count = len(assigned)

    baseline = _optional_series(
        baseline_limit, name="baseline_limit", index=index, lower=0.0
    )
    baseline_bound = baseline is not None
    target_series = _target_series(target, index=index)
    labeled_count = 0 if target_series is None else int(target_series.notna().sum())
    segment_series = _object_series(segment, name="segment", index=index)
    month_series = _object_series(month, name="month", index=index)

    after = _resolve_ead(
        assigned,
        exposure,
        utilization,
        name="exposure",
        index=index,
    )
    before: pandas.Series | None = None
    if baseline_bound:
        before = _resolve_ead(
            baseline,
            baseline_exposure,
            utilization,
            name="baseline_exposure",
            index=index,
        )

    pd_series: pandas.Series | None = None
    pd_proxy_used = False
    missing_el_inputs: list[str] = []
    if pd is not None:
        pd_series = _numeric_input(pd, name="pd", index=index, lower=0.0, upper=1.0)
    elif target_series is not None:
        labeled = target_series.notna()
        if int(labeled.sum()) == 0:
            missing_el_inputs.append("pd")
        else:
            bad_count = int(target_series.loc[labeled].eq(1.0).sum())
            proxy = float(bad_count / int(labeled.sum()))
            pd_series = pandas.Series([proxy] * count, index=index, dtype=float)
            pd_proxy_used = True
    else:
        missing_el_inputs.append("pd")

    lgd_series: pandas.Series | None = None
    if lgd is not None:
        lgd_series = _numeric_input(
            lgd, name="lgd", index=index, lower=0.0, upper=1.0
        )
    else:
        missing_el_inputs.append("lgd")

    el_available = not missing_el_inputs

    total_limit = _finite_sum(assigned.tolist(), name="total_limit")
    total_exposure_after = _finite_sum(after.tolist(), name="total_exposure_after")

    exposure_field: dict[str, Any]
    action_buckets_field: dict[str, Any]
    if baseline_bound:
        assert before is not None
        total_exposure_before = _finite_sum(
            before.tolist(), name="total_exposure_before"
        )
        exposure_delta = _checked_sum(
            [total_exposure_after, -total_exposure_before],
            name="exposure_delta",
        )
        exposure_field = _typed_field(
            "present",
            {
                "after": total_exposure_after,
                "before": total_exposure_before,
                "delta": exposure_delta,
            },
            None,
        )
        bucket_counts = {direction: 0 for direction in _ACTION_DIRECTIONS}
        bucket_deltas = {direction: 0.0 for direction in _ACTION_DIRECTIONS}
        for a, b in zip(after.tolist(), before.tolist(), strict=True):
            if a > b:
                bucket_counts["up"] += 1
                bucket_deltas["up"] += a - b
            elif a < b:
                bucket_counts["down"] += 1
                bucket_deltas["down"] += a - b
            else:
                bucket_counts["unchanged"] += 1
        action_buckets_field = _typed_field(
            "present",
            {
                "rows": [
                    {
                        "direction": direction,
                        "count": bucket_counts[direction],
                        "exposure_delta": _checked_sum(
                            [bucket_deltas[direction]], name="bucket exposure_delta"
                        ),
                    }
                    for direction in _ACTION_DIRECTIONS
                ]
            },
            None,
        )
    else:
        exposure_field = _typed_field(
            "unavailable", None, "baseline_limit_not_bound"
        )
        action_buckets_field = _typed_field(
            "unavailable", None, "baseline_limit_not_bound"
        )

    expected_loss_field: dict[str, Any]
    if el_available:
        assert pd_series is not None and lgd_series is not None
        pd_values = pd_series.tolist()
        lgd_values = lgd_series.tolist()
        el_after = _finite_sum(
            [
                float(a) * float(pd_value) * float(lgd_value)
                for a, pd_value, lgd_value in zip(
                    after.tolist(), pd_values, lgd_values, strict=True
                )
            ],
            name="expected_loss_after",
        )
        el_before: float | None = None
        el_delta: float | None = None
        if baseline_bound:
            assert before is not None
            el_before = _finite_sum(
                [
                    float(b) * float(pd_value) * float(lgd_value)
                    for b, pd_value, lgd_value in zip(
                        before.tolist(), pd_values, lgd_values, strict=True
                    )
                ],
                name="expected_loss_before",
            )
            el_delta = _checked_sum(
                [el_after, -el_before], name="expected_loss_delta"
            )
        expected_loss_field = _typed_field(
            "present",
            {
                "after": el_after,
                "before": el_before,
                "delta": el_delta,
                "pd_proxy_used": pd_proxy_used,
            },
            None,
        )
    else:
        expected_loss_field = _typed_field(
            "unavailable",
            None,
            "missing_economics_inputs:" + ",".join(sorted(set(missing_el_inputs))),
        )

    by_segment_month = _limit_segment_month_cells(
        segment=segment_series,
        month=month_series,
        after=after,
        before=before,
        pd_series=pd_series,
        lgd_series=lgd_series,
        target=target_series,
        el_available=el_available,
        baseline_bound=baseline_bound,
    )

    red_flags = _limit_red_flags(
        baseline_bound=baseline_bound,
        pd_proxy_used=pd_proxy_used,
        el_available=el_available,
        labeled_count=labeled_count,
    )

    return LimitImpactResult(
        schema_version=LIMIT_PRICING_IMPACT_SCHEMA_VERSION,
        strategy_type="limit",
        count=count,
        labeled_count=labeled_count,
        total_limit=total_limit,
        total_exposure_after=total_exposure_after,
        exposure=exposure_field,
        action_buckets=action_buckets_field,
        expected_loss=expected_loss_field,
        by_segment_month=by_segment_month,
        lifecycle=dict(_LIFECYCLE),
        red_flags=tuple(red_flags),
    )


def _limit_red_flags(
    *,
    baseline_bound: bool,
    pd_proxy_used: bool,
    el_available: bool,
    labeled_count: int,
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if not baseline_bound:
        flags.append(
            {
                "code": "baseline_not_bound",
                "level": "amber",
                "message": (
                    "Baseline limit was not bound; before/after comparison, "
                    "action buckets, and EL delta are unavailable."
                ),
            }
        )
    if pd_proxy_used:
        flags.append(
            {
                "code": "pd_proxy_used",
                "level": "amber",
                "message": (
                    "No PD column was supplied; the population empirical bad "
                    "rate was used as a coarse PD proxy."
                ),
            }
        )
    if not el_available:
        flags.append(
            {
                "code": "economics_unavailable",
                "level": "amber",
                "message": (
                    "Expected-loss impact is unavailable because deterministic "
                    "PD/LGD inputs are incomplete."
                ),
            }
        )
    if labeled_count == 0:
        flags.append(
            {
                "code": "labels_unavailable",
                "level": "red",
                "message": "No labeled rows; bad-rate evidence is unavailable.",
            }
        )
    return flags


@dataclass(frozen=True)
class PricingImpactResult:
    """Deterministic, development-only typed impact evidence for a pricing change."""

    schema_version: str
    strategy_type: str
    count: int
    labeled_count: int
    total_exposure: float
    incremental_revenue: float
    revenue_source: str
    expected_loss: dict[str, Any]
    net_impact: dict[str, Any]
    by_segment_month: dict[str, Any]
    lifecycle: dict[str, bool]
    red_flags: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy_type": self.strategy_type,
            "effect_stage": _EFFECT_STAGE,
            "validation_status": _VALIDATION_STATUS,
            "count": self.count,
            "labeled_count": self.labeled_count,
            "total_exposure": self.total_exposure,
            "incremental_revenue": self.incremental_revenue,
            "revenue_source": self.revenue_source,
            "expected_loss": self.expected_loss,
            "net_impact": self.net_impact,
            "by_segment_month": self.by_segment_month,
            "lifecycle": self.lifecycle,
            "red_flags": list(self.red_flags),
        }


def pricing_impact(
    *,
    exposure: pandas.Series,
    rate_delta: pandas.Series | Real | None = None,
    fee_delta: pandas.Series | Real | None = None,
    term_months: pandas.Series | Real | None = None,
    pd: pandas.Series | Real | None = None,
    lgd: pandas.Series | Real | None = None,
    target: pandas.Series | None = None,
    segment: pandas.Series | None = None,
    month: pandas.Series | None = None,
) -> PricingImpactResult:
    """Measure the typed impact of a pricing change on a target population.

    ``exposure`` is the row-aligned EAD of the target population.  Exactly one of
    ``rate_delta`` (annual decimal rate change) or ``fee_delta`` (per-loan fee
    change) must be supplied.  A rate delta requires ``term_months``.
    """

    if rate_delta is None and fee_delta is None:
        raise StrategyError("pricing_impact requires rate_delta or fee_delta")
    if rate_delta is not None and fee_delta is not None:
        raise StrategyError(
            "pricing_impact requires exactly one of rate_delta or fee_delta"
        )

    exposure_series = _required_series(exposure, name="exposure", lower=0.0)
    index = exposure_series.index
    count = len(exposure_series)

    target_series = _target_series(target, index=index)
    labeled_count = 0 if target_series is None else int(target_series.notna().sum())
    segment_series = _object_series(segment, name="segment", index=index)
    month_series = _object_series(month, name="month", index=index)

    revenue_source = "rate_delta" if rate_delta is not None else "fee_delta"
    if revenue_source == "rate_delta":
        if term_months is None:
            raise StrategyError("pricing_impact rate_delta requires term_months")
        rate_values = _numeric_input(
            rate_delta,
            name="rate_delta",
            index=index,
            lower=-1.0,
            upper=1.0,
        ).tolist()
        term_values = _numeric_input(
            term_months,
            name="term_months",
            index=index,
            lower=0.0,
            lower_inclusive=False,
        ).tolist()
        revenue_per_position = [
            float(e) * float(r) * (float(t) / 12.0)
            for e, r, t in zip(
                exposure_series.tolist(), rate_values, term_values, strict=True
            )
        ]
    else:
        fee_values = _numeric_input(fee_delta, name="fee_delta", index=index).tolist()
        revenue_per_position = [float(value) for value in fee_values]
    incremental_revenue = _finite_sum(
        revenue_per_position, name="incremental_revenue"
    )

    pd_series: pandas.Series | None = None
    pd_proxy_used = False
    missing_el_inputs: list[str] = []
    if pd is not None:
        pd_series = _numeric_input(pd, name="pd", index=index, lower=0.0, upper=1.0)
    elif target_series is not None:
        labeled = target_series.notna()
        if int(labeled.sum()) == 0:
            missing_el_inputs.append("pd")
        else:
            bad_count = int(target_series.loc[labeled].eq(1.0).sum())
            proxy = float(bad_count / int(labeled.sum()))
            pd_series = pandas.Series([proxy] * count, index=index, dtype=float)
            pd_proxy_used = True
    else:
        missing_el_inputs.append("pd")

    lgd_series: pandas.Series | None = None
    if lgd is not None:
        lgd_series = _numeric_input(
            lgd, name="lgd", index=index, lower=0.0, upper=1.0
        )
    else:
        missing_el_inputs.append("lgd")

    el_available = not missing_el_inputs
    total_exposure = _finite_sum(
        exposure_series.tolist(), name="total_exposure"
    )

    expected_loss_field: dict[str, Any]
    net_impact_field: dict[str, Any]
    if el_available:
        assert pd_series is not None and lgd_series is not None
        pd_values = pd_series.tolist()
        lgd_values = lgd_series.tolist()
        expected_loss = _finite_sum(
            [
                float(e) * float(pd_value) * float(lgd_value)
                for e, pd_value, lgd_value in zip(
                    exposure_series.tolist(), pd_values, lgd_values, strict=True
                )
            ],
            name="expected_loss",
        )
        expected_loss_field = _typed_field(
            "present",
            {"value": expected_loss, "pd_proxy_used": pd_proxy_used},
            None,
        )
        net_impact_field = _typed_field(
            "present",
            {
                "incremental_revenue": incremental_revenue,
                "expected_loss": expected_loss,
                "net": _checked_sum(
                    [incremental_revenue, -expected_loss], name="net_impact"
                ),
            },
            None,
        )
    else:
        expected_loss_field = _typed_field(
            "unavailable",
            None,
            "missing_economics_inputs:" + ",".join(sorted(set(missing_el_inputs))),
        )
        net_impact_field = _typed_field(
            "unavailable", None, "expected_loss_unavailable"
        )

    by_segment_month = _pricing_segment_month_cells(
        segment=segment_series,
        month=month_series,
        exposure_values=exposure_series.tolist(),
        pd_series=pd_series,
        lgd_series=lgd_series,
        target=target_series,
        el_available=el_available,
        revenue_per_position=revenue_per_position,
    )

    red_flags = _pricing_red_flags(
        pd_proxy_used=pd_proxy_used,
        el_available=el_available,
        labeled_count=labeled_count,
    )

    return PricingImpactResult(
        schema_version=LIMIT_PRICING_IMPACT_SCHEMA_VERSION,
        strategy_type="pricing",
        count=count,
        labeled_count=labeled_count,
        total_exposure=total_exposure,
        incremental_revenue=incremental_revenue,
        revenue_source=revenue_source,
        expected_loss=expected_loss_field,
        net_impact=net_impact_field,
        by_segment_month=by_segment_month,
        lifecycle=dict(_LIFECYCLE),
        red_flags=tuple(red_flags),
    )


def _pricing_red_flags(
    *,
    pd_proxy_used: bool,
    el_available: bool,
    labeled_count: int,
) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    if pd_proxy_used:
        flags.append(
            {
                "code": "pd_proxy_used",
                "level": "amber",
                "message": (
                    "No PD column was supplied; the population empirical bad "
                    "rate was used as a coarse PD proxy."
                ),
            }
        )
    if not el_available:
        flags.append(
            {
                "code": "economics_unavailable",
                "level": "amber",
                "message": (
                    "Expected bad-debt cost is unavailable because deterministic "
                    "PD/LGD inputs are incomplete."
                ),
            }
        )
    if labeled_count == 0:
        flags.append(
            {
                "code": "labels_unavailable",
                "level": "red",
                "message": "No labeled rows; bad-rate evidence is unavailable.",
            }
        )
    return flags


__all__ = [
    "LIMIT_PRICING_IMPACT_SCHEMA_VERSION",
    "LimitImpactResult",
    "PricingImpactResult",
    "limit_impact",
    "pricing_impact",
]
