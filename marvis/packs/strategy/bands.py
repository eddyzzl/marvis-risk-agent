from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from marvis.data.direction import ScoreDirection
from marvis.feature.binning import equal_frequency_edges
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.profit import ProfitParams, profit_calc


# Bands with fewer than this share of the population are flagged `sparse_band`.
_SPARSE_BAND_THRESHOLD = 0.02


@dataclass(frozen=True)
class ScoreBand:
    lo: float
    hi: float
    count: int
    pop_pct: float
    bad_rate: float
    cum_approval_rate: float
    cum_bad_rate: float
    expected_profit: float
    decision: str


@dataclass(frozen=True)
class RedFlag:
    code: str
    level: str
    message: str


@dataclass(frozen=True)
class CutoffBandsResult:
    bands: tuple[ScoreBand, ...]
    band_edges: tuple[float, ...]
    recommended_rules: tuple[dict, ...]
    red_flags: tuple[RedFlag, ...]


def design_cutoff_bands(
    df: pd.DataFrame,
    *,
    score_col: str,
    target_col: str,
    score_direction: ScoreDirection,
    n_bands: int = 5,
    band_edges: list[float] | None = None,
    objective: str = "max_profit",
    max_bad_rate: float | None = None,
    min_approval_rate: float | None = None,
    profit_params: ProfitParams | None = None,
    ead_col: str | None = None,
    pd_col: str | None = None,
) -> CutoffBandsResult:
    """Deterministic (INV-1) score banding + monotone approve/review/decline cut.

    Bands are `[lo, hi)` intervals in ascending score order (final band inclusive
    of the max). Cumulative metrics accumulate in *approval order*: for
    higher_is_better we approve from the highest band down, for higher_is_riskier
    from the lowest band up, so cum_approval_rate/cum_bad_rate read as "if we
    approve down to this band's boundary". Band edges come from the platform's
    equal_frequency_edges (open +/-inf endpoints) unless an explicit band_edges
    override is supplied; no randomness anywhere."""
    if objective not in {"max_profit", "max_approval"}:
        raise StrategyError("objective must be max_profit or max_approval")
    _assert_columns(df, [score_col, target_col])
    scores = pd.to_numeric(df[score_col], errors="raise").astype(float)
    target = pd.to_numeric(df[target_col], errors="raise").astype(int)
    total = int(len(df))
    if total == 0:
        raise StrategyError("design_cutoff_bands requires a non-empty labeled frame")

    edges = _resolve_edges(scores, n_bands=n_bands, band_edges=band_edges)
    raw_bands = _band_stats(
        df,
        scores=scores,
        target=target,
        edges=edges,
        total=total,
        profit_params=profit_params,
        ead_col=ead_col,
        pd_col=pd_col,
    )
    approve_order = _approve_order(len(raw_bands), score_direction)
    cum = _cumulative(raw_bands, approve_order, total)
    approve_count = _best_approve_prefix(
        raw_bands,
        approve_order,
        cum,
        objective=objective,
        max_bad_rate=max_bad_rate,
        min_approval_rate=min_approval_rate,
        total=total,
    )
    decisions = _assign_decisions(len(raw_bands), approve_order, approve_count)
    bands = tuple(
        ScoreBand(
            lo=raw["lo"],
            hi=raw["hi"],
            count=raw["count"],
            pop_pct=raw["pop_pct"],
            bad_rate=raw["bad_rate"],
            cum_approval_rate=cum[idx]["cum_approval_rate"],
            cum_bad_rate=cum[idx]["cum_bad_rate"],
            expected_profit=raw["expected_profit"],
            decision=decisions[idx],
        )
        for idx, raw in enumerate(raw_bands)
    )
    red_flags = _red_flags(
        bands,
        approve_order=approve_order,
        approve_count=approve_count,
        max_bad_rate=max_bad_rate,
        min_approval_rate=min_approval_rate,
        cum=cum,
    )
    rules = _recommended_rules(bands, score_col=score_col, score_direction=score_direction)
    return CutoffBandsResult(
        bands=bands,
        band_edges=tuple(edges),
        recommended_rules=rules,
        red_flags=red_flags,
    )


def _resolve_edges(scores: pd.Series, *, n_bands: int, band_edges: list[float] | None) -> list[float]:
    if band_edges is not None:
        # Explicit user-supplied boundaries are honoured verbatim (finite, no
        # equal_frequency): the operator has chosen the cut points on purpose.
        edges = sorted({float(edge) for edge in band_edges})
        if len(edges) < 2:
            raise StrategyError("band_edges must yield at least one band (>=2 edges)")
        return edges
    if n_bands < 1:
        raise StrategyError("n_bands must be >= 1")
    clean = scores.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise StrategyError("design_cutoff_bands requires non-null scores")
    # T2-3: the auto/quantile path reuses the platform's equal_frequency_edges so
    # degenerate/repeated scores segment identically everywhere in MARVIS -- open
    # (+/-inf) endpoints (every score lands in a band) and the 2-unique midpoint
    # special case, instead of the old bespoke np.quantile that produced finite
    # endpoints and spurious float-noise interior splits.
    edges = [float(edge) for edge in equal_frequency_edges(clean, n_bands)]
    return edges


def _band_stats(
    df: pd.DataFrame,
    *,
    scores: pd.Series,
    target: pd.Series,
    edges: list[float],
    total: int,
    profit_params: ProfitParams | None,
    ead_col: str | None,
    pd_col: str | None,
) -> list[dict]:
    rows: list[dict] = []
    for idx in range(len(edges) - 1):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == len(edges) - 2:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        band_target = target.loc[mask]
        bad_rate = float((band_target == 1).mean()) if count else 0.0
        expected_profit = _band_profit(
            df.loc[mask], profit_params=profit_params, ead_col=ead_col, pd_col=pd_col
        )
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "count": count,
                "pop_pct": float(count / total) if total else 0.0,
                "bad_rate": bad_rate,
                "expected_profit": expected_profit,
            }
        )
    return rows


def _band_profit(
    band: pd.DataFrame,
    *,
    profit_params: ProfitParams | None,
    ead_col: str | None,
    pd_col: str | None,
) -> float:
    if profit_params is None:
        return 0.0
    if not ead_col or not pd_col:
        raise StrategyError("ead_col and pd_col are required for profit banding")
    if len(band) == 0:
        return 0.0
    return profit_calc(
        band, segment_col=None, ead_col=ead_col, pd_col=pd_col, params=profit_params
    )[0].net_profit


def _approve_order(n: int, score_direction: ScoreDirection) -> list[int]:
    # higher_is_better -> approve from the top band down; higher_is_riskier ->
    # approve from the bottom band up. Band indices are ascending by score.
    if score_direction == "higher_is_better":
        return list(range(n - 1, -1, -1))
    return list(range(n))


def _cumulative(raw_bands: list[dict], approve_order: list[int], total: int) -> list[dict]:
    cum = [dict() for _ in raw_bands]
    running_count = 0
    running_bad = 0.0
    for band_idx in approve_order:
        band = raw_bands[band_idx]
        running_count += band["count"]
        running_bad += band["bad_rate"] * band["count"]
        cum[band_idx] = {
            "cum_approval_rate": float(running_count / total) if total else 0.0,
            "cum_bad_rate": float(running_bad / running_count) if running_count else 0.0,
        }
    return cum


def _best_approve_prefix(
    raw_bands: list[dict],
    approve_order: list[int],
    cum: list[dict],
    *,
    objective: str,
    max_bad_rate: float | None,
    min_approval_rate: float | None,
    total: int,
) -> int:
    # Candidate cuts: approve the first k bands in approval order (k in 0..n).
    # Score each feasible cut by the objective; if none feasible, fall back to
    # the cut whose approved-population bad rate is lowest (closest to feasible).
    n = len(raw_bands)
    candidates = []
    for k in range(0, n + 1):
        approved_idx = approve_order[:k]
        approved_count = sum(raw_bands[i]["count"] for i in approved_idx)
        approval_rate = float(approved_count / total) if total else 0.0
        approved_bad = sum(raw_bands[i]["bad_rate"] * raw_bands[i]["count"] for i in approved_idx)
        approved_bad_rate = float(approved_bad / approved_count) if approved_count else 0.0
        profit = sum(raw_bands[i]["expected_profit"] for i in approved_idx)
        feasible = True
        if max_bad_rate is not None and approved_bad_rate > float(max_bad_rate):
            feasible = False
        if min_approval_rate is not None and approval_rate < float(min_approval_rate):
            feasible = False
        candidates.append(
            {
                "k": k,
                "approval_rate": approval_rate,
                "approved_bad_rate": approved_bad_rate,
                "profit": profit,
                "feasible": feasible,
            }
        )
    feasible = [c for c in candidates if c["feasible"] and c["k"] > 0]
    if feasible:
        if objective == "max_profit":
            best = max(feasible, key=lambda c: (c["profit"], c["approval_rate"], c["k"]))
        else:
            best = max(feasible, key=lambda c: (c["approval_rate"], c["profit"], c["k"]))
        return int(best["k"])
    # Infeasible: pick the non-empty cut with the lowest approved bad rate (the
    # "closest" approvable set); the tool surfaces infeasible_constraints.
    nonempty = [c for c in candidates if c["k"] > 0]
    best = min(nonempty, key=lambda c: (c["approved_bad_rate"], -c["approval_rate"], c["k"]))
    return int(best["k"])


def _assign_decisions(n: int, approve_order: list[int], approve_count: int) -> list[str]:
    # Monotone three-segment cut: the first `approve_count` bands in approval
    # order are approved, the rest declined. review is reserved for manual
    # overrides and stays empty here (spec: "review 段可空").
    decisions = ["decline"] * n
    for band_idx in approve_order[:approve_count]:
        decisions[band_idx] = "approve"
    return decisions


def _red_flags(
    bands: tuple[ScoreBand, ...],
    *,
    approve_order: list[int],
    approve_count: int,
    max_bad_rate: float | None,
    min_approval_rate: float | None,
    cum: list[dict],
) -> tuple[RedFlag, ...]:
    flags: list[RedFlag] = []
    # nonmonotonic_bad_rate: adjacent ascending-score bands whose bad rate rises
    # in the wrong direction relative to the approval ordering.
    for idx in range(len(bands) - 1):
        lower = bands[idx].bad_rate
        upper = bands[idx + 1].bad_rate
        if lower > 0 or upper > 0:
            if approve_order[0] == 0:
                # higher_is_riskier: bad rate should be non-decreasing with score.
                inverted = upper < lower
            else:
                # higher_is_better: bad rate should be non-increasing with score.
                inverted = upper > lower
            if inverted:
                flags.append(
                    RedFlag(
                        code="nonmonotonic_bad_rate",
                        level="amber",
                        message=(
                            f"相邻分数带坏率逆序：[{bands[idx].lo:g},{bands[idx].hi:g}) "
                            f"坏率{lower:.4f} 与 [{bands[idx + 1].lo:g},{bands[idx + 1].hi:g}) "
                            f"坏率{upper:.4f} 方向不单调。"
                        ),
                    )
                )
                break
    # sparse_band: any band below the population threshold.
    for band in bands:
        if band.pop_pct < _SPARSE_BAND_THRESHOLD:
            flags.append(
                RedFlag(
                    code="sparse_band",
                    level="amber",
                    message=(
                        f"稀疏分数带 [{band.lo:g},{band.hi:g}) 样本占比 "
                        f"{band.pop_pct * 100:.2f}% 低于 2%，指标不稳。"
                    ),
                )
            )
            break
    # infeasible_constraints: no cut satisfies the risk/approval constraints.
    if _is_infeasible(bands, approve_count, max_bad_rate, min_approval_rate, cum, approve_order):
        flags.append(
            RedFlag(
                code="infeasible_constraints",
                level="red",
                message=(
                    "在给定 max_bad_rate/min_approval_rate 约束下无可行切法，"
                    "已返回最接近的近似解。"
                ),
            )
        )
    return tuple(flags)


def _is_infeasible(
    bands: tuple[ScoreBand, ...],
    approve_count: int,
    max_bad_rate: float | None,
    min_approval_rate: float | None,
    cum: list[dict],
    approve_order: list[int],
) -> bool:
    if max_bad_rate is None and min_approval_rate is None:
        return False
    if approve_count == 0:
        return True
    approved_idx = approve_order[:approve_count]
    approved_count = sum(bands[i].count for i in approved_idx)
    approval_rate = float(approved_count / sum(b.count for b in bands)) if bands else 0.0
    approved_bad = sum(bands[i].bad_rate * bands[i].count for i in approved_idx)
    approved_bad_rate = float(approved_bad / approved_count) if approved_count else 0.0
    if max_bad_rate is not None and approved_bad_rate > float(max_bad_rate):
        return True
    if min_approval_rate is not None and approval_rate < float(min_approval_rate):
        return True
    return False


def _recommended_rules(
    bands: tuple[ScoreBand, ...],
    *,
    score_col: str,
    score_direction: ScoreDirection,
) -> tuple[dict, ...]:
    # Emit a single reject rule at the approve/decline boundary, expressed against
    # score_col in the direction implied by score_direction. build_strategy then
    # consumes this directly.
    declined = [band for band in bands if band.decision == "decline"]
    if not declined:
        return ()
    if score_direction == "higher_is_better":
        # decline the low scores; reject below the lowest approved band's lo.
        approved = [band for band in bands if band.decision == "approve"]
        if not approved:
            return ()
        cutoff = min(band.lo for band in approved)
        condition = f"{score_col} < {cutoff:g}"
    else:
        # higher_is_riskier: decline the high scores; reject at/above the boundary.
        approved = [band for band in bands if band.decision == "approve"]
        if not approved:
            return ()
        cutoff = max(band.hi for band in approved)
        condition = f"{score_col} >= {cutoff:g}"
    return ({"condition": condition, "decision": "reject"},)


def _assert_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise StrategyError(f"missing columns: {', '.join(missing)}")


__all__ = [
    "CutoffBandsResult",
    "RedFlag",
    "ScoreBand",
    "design_cutoff_bands",
]
