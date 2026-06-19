from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.packs.strategy.contracts import ProfitResult


@dataclass(frozen=True)
class ProfitParams:
    annual_rate: float
    funding_rate: float
    lgd: float
    operating_cost_per_loan: float
    term_months: int


def profit_calc(
    df: pd.DataFrame,
    *,
    segment_col: str | None,
    ead_col: str,
    pd_col: str,
    params: ProfitParams,
) -> list[ProfitResult]:
    required = [ead_col, pd_col]
    if segment_col:
        required.append(segment_col)
    _assert_columns(df, required)

    groups = [("all", df)] if not segment_col else df.groupby(segment_col, sort=True, dropna=False)
    return [
        _profit_result(str(segment), group, ead_col=ead_col, pd_col=pd_col, params=params)
        for segment, group in groups
    ]


def vintage_profit(
    df: pd.DataFrame,
    *,
    cohort_col: str,
    ead_col: str,
    pd_col: str,
    params: ProfitParams,
) -> dict[str, ProfitResult]:
    return {
        result.segment: result
        for result in profit_calc(
            df,
            segment_col=cohort_col,
            ead_col=ead_col,
            pd_col=pd_col,
            params=params,
        )
    }


def _profit_result(
    segment: str,
    group: pd.DataFrame,
    *,
    ead_col: str,
    pd_col: str,
    params: ProfitParams,
) -> ProfitResult:
    ead = pd.to_numeric(group[ead_col], errors="raise").astype(float)
    pd_values = pd.to_numeric(group[pd_col], errors="raise").astype(float)
    term_factor = float(params.term_months) / 12.0
    ead_sum = float(ead.sum())
    revenue = float((ead * float(params.annual_rate) * term_factor).sum())
    expected_loss = float((ead * pd_values * float(params.lgd)).sum())
    funding_cost = float((ead * float(params.funding_rate) * term_factor).sum())
    operating_cost = float(len(group) * float(params.operating_cost_per_loan))
    net_profit = revenue - expected_loss - funding_cost - operating_cost
    return ProfitResult(
        segment=segment,
        count=int(len(group)),
        revenue=revenue,
        expected_loss=expected_loss,
        funding_cost=funding_cost,
        operating_cost=operating_cost,
        net_profit=net_profit,
        roa=net_profit / ead_sum if ead_sum else 0.0,
    )


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")


__all__ = ["ProfitParams", "profit_calc", "vintage_profit"]
