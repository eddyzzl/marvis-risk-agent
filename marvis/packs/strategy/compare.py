from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.packs.strategy.backtest import backtest_strategy
from marvis.packs.strategy.contracts import Strategy
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.strategy import apply_strategy


@dataclass(frozen=True)
class CompareCell:
    count: int
    bad_rate: float


@dataclass(frozen=True)
class CompareResult:
    matrix_2x2: dict[str, CompareCell]
    deltas: dict[str, float]
    summary_text: str
    red_flags: tuple[dict, ...]


def compare_strategies(
    df: pd.DataFrame,
    strategy: Strategy,
    baseline: Strategy,
    *,
    target_col: str,
    profit_params: ProfitParams | None = None,
    ead_col: str | None = None,
    pd_col: str | None = None,
) -> CompareResult:
    """Deterministic 2x2 approve/decline agreement matrix between a candidate and
    a baseline strategy, plus approval/bad-rate/profit deltas. Reuses the shared
    backtest core for the aggregate deltas so numbers match backtest_strategy
    exactly; the 2x2 cells are computed here from the two decision vectors."""
    _assert_columns(df, [target_col])
    target = pd.to_numeric(df[target_col], errors="raise").astype(int)
    new_approved = apply_strategy(df, strategy) != "reject"
    base_approved = apply_strategy(df, baseline) != "reject"

    matrix = {
        "both_approve": _cell(target, new_approved & base_approved),
        "only_new": _cell(target, new_approved & ~base_approved),
        "only_baseline": _cell(target, ~new_approved & base_approved),
        "both_decline": _cell(target, ~new_approved & ~base_approved),
    }

    new_result = backtest_strategy(
        df, strategy, target_col=target_col, baseline=baseline,
        profit_params=profit_params, ead_col=ead_col, pd_col=pd_col,
    )
    base_result = backtest_strategy(
        df, baseline, target_col=target_col,
        profit_params=profit_params, ead_col=ead_col, pd_col=pd_col,
    )
    deltas = {
        "approval_rate": float(new_result.approval_rate - base_result.approval_rate),
        "approved_bad_rate": float(new_result.approved_bad_rate - base_result.approved_bad_rate),
        "expected_profit": float(new_result.expected_profit - base_result.expected_profit),
    }

    red_flags: list[dict] = []
    swap_in = matrix["only_new"]
    swap_out = matrix["only_baseline"]
    if swap_in.count and swap_out.count and swap_in.bad_rate > swap_out.bad_rate:
        red_flags.append(
            {
                "code": "swap_in_worse",
                "level": "red",
                "message": (
                    f"swap-in 坏率 {swap_in.bad_rate:.4f} 高于 swap-out 坏率 "
                    f"{swap_out.bad_rate:.4f}，新策略换入的客群更差。"
                ),
            }
        )
    if deltas["expected_profit"] < 0:
        red_flags.append(
            {
                "code": "profit_negative_delta",
                "level": "amber",
                "message": (
                    f"预期利润较基线下降 {abs(deltas['expected_profit']):.2f}。"
                ),
            }
        )

    summary_text = (
        f"新策略审批率较基线{_delta_word(deltas['approval_rate'])}"
        f"{abs(deltas['approval_rate']) * 100:.1f}pp，"
        f"通过客群坏率{_delta_word(deltas['approved_bad_rate'])}"
        f"{abs(deltas['approved_bad_rate']) * 100:.2f}pp，"
        f"预期利润{_delta_word(deltas['expected_profit'])}{abs(deltas['expected_profit']):.2f}。"
    )
    return CompareResult(
        matrix_2x2=matrix,
        deltas=deltas,
        summary_text=summary_text,
        red_flags=tuple(red_flags),
    )


def _cell(target: pd.Series, mask: pd.Series) -> CompareCell:
    count = int(mask.sum())
    bad_rate = float((target.loc[mask] == 1).mean()) if count else 0.0
    return CompareCell(count=count, bad_rate=bad_rate)


def _delta_word(value: float) -> str:
    if value > 0:
        return "上升"
    if value < 0:
        return "下降"
    return "持平"


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


__all__ = ["CompareCell", "CompareResult", "compare_strategies"]
