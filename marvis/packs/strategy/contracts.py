from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VintageCurve:
    cohort_col: str
    mob_max: int
    cohorts: tuple[str, ...]
    curves: dict[str, list[float | None]]
    counts: dict[str, int]
    mob_axis: tuple[int, ...] = ()


@dataclass(frozen=True)
class RollRateMatrix:
    states: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    period: str
    base_counts: dict[str, float]
    #: DOM-8: id-level adjacent-observation month gaps (e.g. 202601 -> 202603), one
    #: dict per id with a gap, informational only -- never mutates the matrix itself.
    data_quality_warnings: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ProfitResult:
    segment: str
    count: int
    revenue: float
    expected_loss: float
    funding_cost: float
    operating_cost: float
    net_profit: float
    roa: float


@dataclass(frozen=True)
class StrategyRule:
    condition: str
    decision: str
    value: Any


@dataclass(frozen=True)
class Strategy:
    id: str
    strategy_type: str
    rules: tuple[StrategyRule, ...]
    score_col: str | None
    default_decision: Any
    description: str


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    approval_rate: float
    approved_count: int
    approved_bad_rate: float
    rejected_bad_rate: float
    expected_profit: float
    swap_in_count: int
    swap_out_count: int
    swap_in_bad_rate: float | None
    swap_out_bad_rate: float | None
    by_segment: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TradeoffPoint:
    cutoff: float
    approval_rate: float
    bad_rate: float
    expected_profit: float


__all__ = [
    "BacktestResult",
    "ProfitResult",
    "RollRateMatrix",
    "Strategy",
    "StrategyRule",
    "TradeoffPoint",
    "VintageCurve",
]
