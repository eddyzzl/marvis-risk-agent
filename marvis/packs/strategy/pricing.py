"""S6 (A3) limit x pricing matrix: deterministic band x limit x rate profit grid.

Each cell's expected-profit/EL uses the SAME per-loan formula as the shared profit
kernel (marvis.packs.strategy.profit) -- a double-manual lock rather than importing
it, because a pricing cell prices a HYPOTHETICAL limit (EAD = the grid limit, not an
observed ead_col) at a HYPOTHETICAL annual rate, so profit_calc's per-row EAD/rate
columns do not apply. Keeping the arithmetic identical (revenue = EAD*rate*term,
EL = EAD*PD*LGD, funding = EAD*funding_rate*term, op = cost_per_loan) means a change
to the profit convention only has to be mirrored in one small place, and the unit
tests hand-check every cell against this exact formula.

PD source: pd_col mean over the band's rows when a PD column is supplied, else the
band's empirical bad rate as a PD proxy (surfaced as a ``pd_proxy_used`` red flag --
an empirical bad rate is a coarse stand-in for a calibrated PD).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.packs.strategy.bands import _resolve_edges
from marvis.packs.strategy.errors import StrategyError

# A cell is feasible only if it clears roa>=0 AND its per-unit expected loss stays
# within this fraction of EAD. Overridable per call; the default keeps a positive-ROA
# cell whose EL/EAD is nonetheless alarmingly high out of the recommended set.
_DEFAULT_EL_EAD_MAX = 0.20


@dataclass(frozen=True)
class PricingCell:
    band: str
    limit: float
    rate: float
    count: int
    pd: float
    el: float
    ead: float
    expected_profit: float
    roa: float
    feasible: bool


@dataclass(frozen=True)
class PricingParams:
    lgd: float
    funding_rate: float
    term_months: int
    cost_per_loan: float
    el_ead_max: float = _DEFAULT_EL_EAD_MAX


@dataclass(frozen=True)
class LimitPricingResult:
    matrix: tuple[PricingCell, ...]
    recommended: tuple[dict, ...]
    band_edges: tuple[float, ...]
    red_flags: tuple[dict, ...]


def limit_pricing_matrix(
    df: pd.DataFrame,
    *,
    score_col: str,
    limit_grid: list[float],
    rate_grid: list[float],
    params: PricingParams,
    target_col: str | None = None,
    pd_col: str | None = None,
    band_edges: list[float] | None = None,
    n_bands: int = 5,
) -> LimitPricingResult:
    _assert_columns(df, [score_col])
    if not limit_grid:
        raise StrategyError("limit_pricing_matrix requires at least one limit")
    if not rate_grid:
        raise StrategyError("limit_pricing_matrix requires at least one rate")
    if pd_col is None and target_col is None:
        raise StrategyError("limit_pricing_matrix requires pd_col or target_col")

    scores = pd.to_numeric(df[score_col], errors="raise").astype(float)
    edges = _resolve_edges(scores, n_bands=n_bands, band_edges=band_edges)
    pd_proxy_used = pd_col is None

    red_flags: list[dict] = []
    cells: list[PricingCell] = []
    recommended: list[dict] = []
    for band_index in range(len(edges) - 1):
        lo = float(edges[band_index])
        hi = float(edges[band_index + 1])
        if band_index == len(edges) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        band_df = df.loc[mask]
        count = int(len(band_df))
        band_label = f"[{_fmt_edge(lo)},{_fmt_edge(hi)})"
        band_pd = _band_pd(band_df, pd_col=pd_col, target_col=target_col)

        band_cells: list[PricingCell] = []
        for limit in limit_grid:
            for rate in rate_grid:
                cell = _price_cell(
                    band_label, float(limit), float(rate), count, band_pd, params
                )
                cells.append(cell)
                band_cells.append(cell)

        feasible_cells = [cell for cell in band_cells if cell.feasible]
        if feasible_cells:
            # Max expected profit, ties broken by lower rate then lower limit for a
            # stable, borrower-friendlier deterministic pick.
            best = max(
                feasible_cells,
                key=lambda c: (c.expected_profit, -c.rate, -c.limit),
            )
            recommended.append({"band": best.band, "limit": best.limit, "rate": best.rate})
        elif count > 0:
            red_flags.append({
                "code": "negative_profit_band",
                "level": "red",
                "message": f"分数带 {band_label} 无可行（roa≥0 且 EL/EAD 达标）的额度/定价档。",
            })

    if pd_proxy_used:
        red_flags.insert(0, {
            "code": "pd_proxy_used",
            "level": "amber",
            "message": "未提供 PD 列，已用各分数带经验坏率作为 PD 代理，EL 仅供参考。",
        })

    return LimitPricingResult(
        matrix=tuple(cells),
        recommended=tuple(recommended),
        band_edges=tuple(edges),
        red_flags=tuple(red_flags),
    )


def _price_cell(
    band: str,
    limit: float,
    rate: float,
    count: int,
    band_pd: float,
    params: PricingParams,
) -> PricingCell:
    term_factor = float(params.term_months) / 12.0
    ead_per_loan = float(limit)
    # Per-loan components, identical in form to profit._profit_result.
    revenue = ead_per_loan * float(rate) * term_factor
    el_per_loan = ead_per_loan * float(band_pd) * float(params.lgd)
    funding = ead_per_loan * float(params.funding_rate) * term_factor
    op = float(params.cost_per_loan)
    profit_per_loan = revenue - el_per_loan - funding - op

    total_ead = ead_per_loan * count
    total_el = el_per_loan * count
    expected_profit = profit_per_loan * count
    roa = expected_profit / total_ead if total_ead else 0.0
    el_ead_ratio = el_per_loan / ead_per_loan if ead_per_loan else 0.0
    feasible = bool(roa >= 0.0 and el_ead_ratio <= float(params.el_ead_max) and count > 0)
    return PricingCell(
        band=band,
        limit=float(limit),
        rate=float(rate),
        count=count,
        pd=float(band_pd),
        el=total_el,
        ead=total_ead,
        expected_profit=expected_profit,
        roa=roa,
        feasible=feasible,
    )


def _band_pd(band_df: pd.DataFrame, *, pd_col: str | None, target_col: str | None) -> float:
    if len(band_df) == 0:
        return 0.0
    column = pd_col or target_col
    values = pd.to_numeric(band_df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    if pd_col is not None:
        return float(values.mean())
    # PD proxy: the band's empirical bad rate (share of target==1).
    return float((values == 1).mean())


def _fmt_edge(value: float) -> str:
    return f"{value:g}"


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


__all__ = [
    "LimitPricingResult",
    "PricingCell",
    "PricingParams",
    "limit_pricing_matrix",
]
