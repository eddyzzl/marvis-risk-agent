from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import pandas as pd

from marvis.packs.strategy.contracts import BacktestResult, Strategy
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import FrameEvaluation, evaluate_strategy_frame
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.typed_backtest import (
    StrategyBacktestResult,
    run_typed_backtest,
)
from marvis.packs.strategy.economics import NumericInput


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
    economics_inputs: Mapping[str, NumericInput] | None = None,
) -> BacktestResult | StrategyBacktestResult:
    if strategy.strategy_type not in {"approval", "reject"}:
        if profit_params is not None or ead_col is not None or pd_col is not None:
            raise StrategyError(
                "profit_params/ead_col/pd_col are approval compatibility inputs; "
                "use economics_inputs for limit or pricing"
            )
        return run_typed_backtest(
            df,
            strategy.spec or legacy_strategy_to_spec(strategy),
            target_col=target_col,
            strategy_id=strategy.id,
            baseline=(
                None
                if baseline is None
                else baseline.spec or legacy_strategy_to_spec(baseline)
            ),
            economics_inputs=economics_inputs,
        )
    if economics_inputs is not None:
        raise StrategyError(
            "economics_inputs are only valid for limit or pricing strategies"
        )
    _assert_columns(df, [target_col])
    evaluation = evaluate_strategy_frame(
        df,
        strategy.spec or legacy_strategy_to_spec(strategy),
    )
    approved, rejected, reviewed = _decision_masks(strategy, evaluation)
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
        rejected_bad_rate=_bad_rate(target.loc[rejected]),
        expected_profit=profit_value,
        swap_in_count=swap.in_count,
        swap_out_count=swap.out_count,
        swap_in_bad_rate=swap.in_bad_rate,
        swap_out_bad_rate=swap.out_bad_rate,
        by_segment=_segment_breakdown(strategy, evaluation, target),
        profit_note=profit_note,
        rejected_count=int(rejected.sum()),
        review_count=int(reviewed.sum()),
        review_rate=_ratio(float(reviewed.sum()), float(len(df))),
        review_bad_rate=_bad_rate_optional(target.loc[reviewed]),
    )


def _swap_analysis(
    df: pd.DataFrame,
    new_approved: pd.Series,
    baseline: Strategy | None,
    target_col: str,
) -> _SwapStats:
    if baseline is None:
        return _zero_swap()
    old_approved = strategy_approval_mask(df, baseline)
    target = _target_series(df, target_col)
    swap_in = new_approved & ~old_approved
    swap_out = ~new_approved & old_approved
    return _SwapStats(
        in_count=int(swap_in.sum()),
        out_count=int(swap_out.sum()),
        in_bad_rate=_bad_rate_optional(target.loc[swap_in]),
        out_bad_rate=_bad_rate_optional(target.loc[swap_out]),
    )


def _segment_breakdown(
    strategy: Strategy, evaluation: FrameEvaluation, target: pd.Series
) -> tuple[dict, ...]:
    if strategy.strategy_type in {"approval", "reject"}:
        canonical = evaluation.action_type.map(
            {"approval": "approve", "reject": "reject", "review": "review"}
        )
    else:
        canonical = evaluation.decisions.map(str)
    frame = pd.DataFrame(
        {
            "decision": canonical,
            "output_value": evaluation.decisions.map(str),
            "target": target,
        }
    )
    rows = []
    for (decision_value, output_value), group in frame.groupby(
        ["decision", "output_value"], sort=True, dropna=False
    ):
        bad_count = int((group["target"] == 1).sum())
        row = {
            "decision": str(decision_value),
            "count": int(len(group)),
            "bad_count": bad_count,
            "bad_rate": _ratio(float(bad_count), float(len(group))),
        }
        if str(output_value) != str(decision_value):
            row["legacy_output_value"] = str(output_value)
        rows.append(row)
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


def strategy_approval_mask(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    """Return the approved population without treating review as approval.

    Non-approval strategies fail closed until their dedicated typed backtest and
    comparison contracts are available; they must never masquerade as 100% approved.
    """

    evaluation = evaluate_strategy_frame(
        df,
        strategy.spec or legacy_strategy_to_spec(strategy),
    )
    return _decision_masks(strategy, evaluation)[0]


def _decision_masks(
    strategy: Strategy, evaluation: FrameEvaluation
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if strategy.strategy_type in {"approval", "reject"}:
        approved = evaluation.action_type.eq("approval")
        rejected = evaluation.action_type.eq("reject")
        reviewed = evaluation.action_type.eq("review")
        if not bool((approved | rejected | reviewed).all()):
            raise StrategyError(
                "approval strategy produced an unsupported decision action"
            )
        return approved, rejected, reviewed
    raise StrategyError(
        f"approval mask is not defined for strategy type {strategy.strategy_type}"
    )


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


__all__ = ["backtest_strategy", "strategy_approval_mask"]
