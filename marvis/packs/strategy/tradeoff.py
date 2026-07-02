from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.data.direction import ScoreDirection, check_score_direction
from marvis.data.errors import ScoreDirectionConflictError
from marvis.packs.strategy.contracts import TradeoffPoint
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams, profit_calc


# S1a: default matches the pre-existing hard-coded behavior (L38's `scores >= cutoff`
# treats higher score as "should approve"), so callers that never pass score_direction
# see zero behavior change.
_DEFAULT_TRADEOFF_DIRECTION: ScoreDirection = "higher_is_better"


def tradeoff_view(
    df: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    cutoffs: list[float] | None = None,
    profit_params: ProfitParams | None = None,
    ead_col: str | None = None,
    pd_col: str | None = None,
    score_direction: ScoreDirection | None = None,
    confirm_direction_conflict: bool = False,
) -> list[TradeoffPoint]:
    _assert_columns(df, [score_col, target_col])
    scores = pd.to_numeric(df[score_col], errors="raise")
    # NaN labels must never be coerced to 0; callers gate/drop them upstream (tool boundary).
    target = pd.to_numeric(df[target_col], errors="raise").astype(int)
    effective_direction = score_direction or _DEFAULT_TRADEOFF_DIRECTION

    if not confirm_direction_conflict:
        _raise_on_direction_conflict(
            tool="tradeoff_view",
            scores=scores.to_numpy(dtype=float),
            target=target.to_numpy(dtype=float),
            score_col=score_col,
            target_col=target_col,
            declared_direction=effective_direction,
        )

    approve = _approval_mask_fn(effective_direction)
    return [
        TradeoffPoint(
            cutoff=cutoff,
            approval_rate=_ratio(float(approved.sum()), float(len(df))),
            bad_rate=_bad_rate(target.loc[approved]),
            expected_profit=_strategy_profit(
                df.loc[approved],
                profit_params=profit_params,
                ead_col=ead_col,
                pd_col=pd_col,
            ),
        )
        for cutoff in _cutoff_values(scores, cutoffs)
        for approved in [approve(scores, cutoff)]
    ]


def _approval_mask_fn(direction: ScoreDirection):
    if direction == "higher_is_better":
        return lambda scores, cutoff: scores >= cutoff  # current behavior, unchanged
    return lambda scores, cutoff: scores < cutoff  # higher_is_riskier -> low score approved


def _raise_on_direction_conflict(
    *,
    tool: str,
    scores: np.ndarray,
    target: np.ndarray,
    score_col: str,
    target_col: str,
    declared_direction: ScoreDirection,
) -> None:
    result = check_score_direction(scores, target, declared_direction=declared_direction)
    if result.status == "conflict":
        raise ScoreDirectionConflictError(
            tool=tool,
            score_col=score_col,
            target_col=target_col,
            declared_direction=declared_direction,
            implied_direction=result.implied_direction,
            corr=result.corr,
            n_labeled=result.n,
        )


def recommend_operating_point(
    points: list[TradeoffPoint],
    *,
    objective: str = "max_profit",
    max_bad_rate: float | None = None,
) -> TradeoffPoint:
    if objective not in {"max_profit", "max_approval"}:
        raise ValueError("objective must be max_profit or max_approval")
    if not points:
        raise ValueError("points must not be empty")
    feasible = [
        point
        for point in points
        if max_bad_rate is None or point.bad_rate <= float(max_bad_rate)
    ]
    if not feasible:
        return min(points, key=lambda point: point.bad_rate)
    if objective == "max_profit":
        return max(feasible, key=lambda point: point.expected_profit)
    return max(feasible, key=lambda point: point.approval_rate)


def _cutoff_values(scores: pd.Series, cutoffs: list[float] | None) -> list[float]:
    if cutoffs is not None:
        return sorted({float(cutoff) for cutoff in cutoffs})
    clean = scores.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return []
    return sorted({float(cutoff) for cutoff in np.quantile(clean, np.linspace(0.05, 0.95, 19))})


def _strategy_profit(
    approved: pd.DataFrame,
    *,
    profit_params: ProfitParams | None,
    ead_col: str | None,
    pd_col: str | None,
) -> float:
    if profit_params is None:
        return 0.0
    if not ead_col or not pd_col:
        raise StrategyError("ead_col and pd_col are required for profit tradeoff")
    return profit_calc(
        approved,
        segment_col=None,
        ead_col=ead_col,
        pd_col=pd_col,
        params=profit_params,
    )[0].net_profit


def _bad_rate(target: pd.Series) -> float:
    if target.empty:
        return 0.0
    return float((target == 1).mean())


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = ["recommend_operating_point", "tradeoff_view"]
