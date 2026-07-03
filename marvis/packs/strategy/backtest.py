from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.packs.strategy.contracts import BacktestResult, Strategy
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.strategy import apply_strategy


@dataclass(frozen=True)
class _SwapStats:
    in_count: int
    out_count: int
    in_bad_rate: float | None
    out_bad_rate: float | None


def backtest_strategy(
    df: pd.DataFrame,
    strategy: Strategy,
    *,
    target_col: str,
    baseline: Strategy | None = None,
    profit_params: ProfitParams | None = None,
    ead_col: str | None = None,
    pd_col: str | None = None,
) -> BacktestResult:
    _assert_columns(df, [target_col])
    decision = apply_strategy(df, strategy)
    approved = decision != "reject"
    target = _target_series(df, target_col)
    swap = _swap_analysis(df, approved, baseline, target_col) if baseline else _zero_swap()
    profit_value, profit_note = _strategy_profit(
        df.loc[approved],
        profit_params=profit_params,
        ead_col=ead_col,
        pd_col=pd_col,
    )
    return BacktestResult(
        strategy_id=strategy.id,
        approval_rate=_ratio(float(approved.sum()), float(len(df))),
        approved_count=int(approved.sum()),
        approved_bad_rate=_bad_rate(target.loc[approved]),
        rejected_bad_rate=_bad_rate(target.loc[~approved]),
        expected_profit=profit_value,
        swap_in_count=swap.in_count,
        swap_out_count=swap.out_count,
        swap_in_bad_rate=swap.in_bad_rate,
        swap_out_bad_rate=swap.out_bad_rate,
        by_segment=_segment_breakdown(decision, target),
        profit_note=profit_note,
    )


def _swap_analysis(
    df: pd.DataFrame,
    new_approved: pd.Series,
    baseline: Strategy | None,
    target_col: str,
) -> _SwapStats:
    if baseline is None:
        return _zero_swap()
    old_approved = apply_strategy(df, baseline) != "reject"
    target = _target_series(df, target_col)
    swap_in = new_approved & ~old_approved
    swap_out = ~new_approved & old_approved
    return _SwapStats(
        in_count=int(swap_in.sum()),
        out_count=int(swap_out.sum()),
        in_bad_rate=_bad_rate_optional(target.loc[swap_in]),
        out_bad_rate=_bad_rate_optional(target.loc[swap_out]),
    )


def _segment_breakdown(decision: pd.Series, target: pd.Series) -> tuple[dict, ...]:
    frame = pd.DataFrame({"decision": decision.map(str), "target": target})
    rows = []
    for decision_value, group in frame.groupby("decision", sort=True, dropna=False):
        bad_count = int((group["target"] == 1).sum())
        rows.append(
            {
                "decision": str(decision_value),
                "count": int(len(group)),
                "bad_count": bad_count,
                "bad_rate": _ratio(float(bad_count), float(len(group))),
            }
        )
    return tuple(rows)


def _strategy_profit(
    approved: pd.DataFrame,
    *,
    profit_params: ProfitParams | None,
    ead_col: str | None,
    pd_col: str | None,
) -> tuple[float | None, str | None]:
    """Return ``(expected_profit, note)`` for the approved rows.

    * No profit backtest requested (``profit_params is None``) -> ``(0.0, None)``.
    * Profit requested but the expected-loss chain inputs are missing (``pd_col`` /
      ``ead_col``) -> FIN-3 #4: degrade gracefully to ``(None, note)`` instead of
      raising or fabricating a misleading 0.0, so the EL chain never silently
      produces a fake profit and the caller can surface the reason as a red flag.
    * Otherwise -> ``(net_profit, None)``.
    """
    if profit_params is None:
        return 0.0, None
    if not ead_col or not pd_col:
        return None, (
            "已请求利润回测，但缺少 pd_col/ead_col，无法计算预期损失链，"
            "expected_profit 记为不可用（未用 0 冒充）。"
        )
    net_profit = profit_calc(
        approved,
        segment_col=None,
        ead_col=ead_col,
        pd_col=pd_col,
        params=profit_params,
    )[0].net_profit
    return net_profit, None


def _target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    # NaN labels must never be coerced to 0; callers gate/drop them upstream (tool boundary).
    return pd.to_numeric(df[target_col], errors="raise").astype(int)


def _bad_rate(target: pd.Series) -> float:
    if target.empty:
        return 0.0
    return float((target == 1).mean())


def _bad_rate_optional(target: pd.Series) -> float | None:
    # Swap sets (swap-in/swap-out) can be legitimately empty -- an empty set has no
    # defined bad rate, so this returns None instead of the misleading 0.0 (DOM-11).
    if target.empty:
        return None
    return float((target == 1).mean())


def _zero_swap() -> _SwapStats:
    return _SwapStats(
        in_count=0,
        out_count=0,
        in_bad_rate=None,
        out_bad_rate=None,
    )


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = ["backtest_strategy"]
