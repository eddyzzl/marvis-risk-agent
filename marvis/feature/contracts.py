from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bin:
    index: int
    lower: float
    upper: float
    count: int
    bad_count: int
    good_count: int
    bad_rate: float
    woe: float
    iv_contribution: float


@dataclass(frozen=True)
class BinningResult:
    feature: str
    method: str
    bins: tuple[Bin, ...]
    edges: tuple[float, ...]
    total_iv: float
    monotonic: bool
    na_bin: Bin | None


@dataclass(frozen=True)
class FeatureMetrics:
    feature: str
    iv: float
    ks: float
    auc: float
    psi: float | None
    missing_rate: float
    unique_count: int
    lift_top_bin: float


@dataclass(frozen=True)
class WOEResult:
    feature: str
    edges: tuple[float, ...]
    woe_by_bin: tuple[float, ...]
    na_woe: float | None


@dataclass(frozen=True)
class CorrelationReport:
    features: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    collinear_pairs: tuple[tuple[str, str, float], ...]
    vif: dict[str, float]


__all__ = [
    "Bin",
    "BinningResult",
    "CorrelationReport",
    "FeatureMetrics",
    "WOEResult",
]
