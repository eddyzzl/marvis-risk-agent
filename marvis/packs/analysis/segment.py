"""细分画像内核 (segment_profile).

按 segment_col 分组统计计数/占比/审批率/坏率/均分/净利润，并给出集中度
(top1/top5/HHI) 与红旗 (high_concentration / sparse_segment 归并「其他」)。

利润列复用 strategy 包 profit 内核的同款公式，但这里本地实现（不跨包 import）——
公式简单，两包各自测试锁同一手算值，注释互指 (strategy/profit.py profit_calc)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from marvis.packs.analysis.errors import AnalysisError

#: top_k 之外的小细分归并到该标签。
OTHER_SEGMENT = "其他"
#: top1 占比超过该阈值 -> high_concentration 红旗。
HIGH_CONCENTRATION_TOP1 = 0.40
#: HHI 超过该阈值 -> high_concentration 红旗。
HIGH_CONCENTRATION_HHI = 0.25


@dataclass(frozen=True)
class SegmentRow:
    segment: str
    count: int
    pop_pct: float
    approval_rate: float | None
    bad_rate: float | None
    avg_score: float | None
    net_profit: float | None


@dataclass(frozen=True)
class Concentration:
    top1_pct: float
    top5_pct: float
    hhi: float


@dataclass(frozen=True)
class SegmentProfileResult:
    segments: list[SegmentRow]
    concentration: Concentration
    red_flags: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ProfitParams:
    """本地利润参数（与 strategy.profit.ProfitParams 同款字段/公式）。"""

    annual_rate: float
    funding_rate: float
    lgd: float
    operating_cost_per_loan: float
    term_months: int


def segment_profile(
    df: pd.DataFrame,
    *,
    segment_col: str,
    target_col: str | None = None,
    score_col: str | None = None,
    approved_col: str | None = None,
    profit_params: ProfitParams | None = None,
    ead_col: str | None = None,
    pd_col: str | None = None,
    top_k: int = 20,
) -> SegmentProfileResult:
    required = [segment_col]
    for optional in (target_col, score_col, approved_col, ead_col, pd_col):
        if optional:
            required.append(optional)
    missing = [column for column in dict.fromkeys(required) if column not in df.columns]
    if missing:
        raise AnalysisError(f"segment_profile 缺少列：{', '.join(missing)}")

    total = int(len(df))
    if total == 0:
        raise AnalysisError("segment_profile: 空数据集")

    # concentration is computed over the *full* segment cardinality (before 归并),
    # so 归并到「其他」不会掩盖真实集中度。
    raw_counts = df[segment_col].astype(str).value_counts()
    concentration = _concentration(raw_counts, total)

    ranked_segments = list(raw_counts.index)
    kept = ranked_segments[:top_k]
    merged = set(ranked_segments[top_k:])

    rows: list[SegmentRow] = []
    grouped: dict[str, pd.DataFrame] = {}
    for segment in kept:
        grouped[segment] = df[df[segment_col].astype(str) == segment]
    if merged:
        grouped[OTHER_SEGMENT] = df[df[segment_col].astype(str).isin(merged)]

    for segment, group in grouped.items():
        rows.append(
            _segment_row(
                segment,
                group,
                total=total,
                target_col=target_col,
                score_col=score_col,
                approved_col=approved_col,
                profit_params=profit_params,
                ead_col=ead_col,
                pd_col=pd_col,
            )
        )

    red_flags: list[dict] = []
    if concentration.top1_pct > HIGH_CONCENTRATION_TOP1 or concentration.hhi > HIGH_CONCENTRATION_HHI:
        red_flags.append(
            {
                "kind": "high_concentration",
                "top1_pct": concentration.top1_pct,
                "hhi": concentration.hhi,
                "message": (
                    f"细分高度集中：top1 占比 {concentration.top1_pct:.1%}，HHI {concentration.hhi:.3f}"
                    f"（阈值 top1>{HIGH_CONCENTRATION_TOP1:.0%} 或 HHI>{HIGH_CONCENTRATION_HHI}）。"
                ),
            }
        )
    if merged:
        red_flags.append(
            {
                "kind": "sparse_segment",
                "merged_count": len(merged),
                "message": f"{len(merged)} 个小细分已归并为「{OTHER_SEGMENT}」（top_k={top_k}）。",
            }
        )

    return SegmentProfileResult(segments=rows, concentration=concentration, red_flags=red_flags)


def _segment_row(
    segment: str,
    group: pd.DataFrame,
    *,
    total: int,
    target_col: str | None,
    score_col: str | None,
    approved_col: str | None,
    profit_params: ProfitParams | None,
    ead_col: str | None,
    pd_col: str | None,
) -> SegmentRow:
    count = int(len(group))
    pop_pct = count / total if total else 0.0
    approval_rate = None
    if approved_col:
        approved = pd.to_numeric(group[approved_col], errors="coerce")
        approval_rate = float(approved.mean()) if count else None
    bad_rate = None
    if target_col:
        target = pd.to_numeric(group[target_col], errors="coerce")
        valid = target.dropna()
        bad_rate = float(valid.mean()) if len(valid) else None
    avg_score = None
    if score_col:
        score = pd.to_numeric(group[score_col], errors="coerce")
        valid = score.dropna()
        avg_score = float(valid.mean()) if len(valid) else None
    net_profit = None
    if profit_params is not None and ead_col and pd_col:
        net_profit = _net_profit(group, profit_params, ead_col=ead_col, pd_col=pd_col)
    return SegmentRow(
        segment=str(segment),
        count=count,
        pop_pct=pop_pct,
        approval_rate=approval_rate,
        bad_rate=bad_rate,
        avg_score=avg_score,
        net_profit=net_profit,
    )


def _net_profit(group: pd.DataFrame, params: ProfitParams, *, ead_col: str, pd_col: str) -> float:
    """本地净利润公式，逐字对应 strategy/profit.py::_profit_result。

    net = revenue - expected_loss - funding_cost - operating_cost，其中
    revenue = sum(ead * annual_rate * term/12), expected_loss = sum(ead * pd * lgd),
    funding_cost = sum(ead * funding_rate * term/12), operating_cost = n * op_cost。
    """
    ead = pd.to_numeric(group[ead_col], errors="coerce").fillna(0.0).astype(float)
    pd_values = pd.to_numeric(group[pd_col], errors="coerce").fillna(0.0).astype(float)
    term_factor = float(params.term_months) / 12.0
    revenue = float((ead * float(params.annual_rate) * term_factor).sum())
    expected_loss = float((ead * pd_values * float(params.lgd)).sum())
    funding_cost = float((ead * float(params.funding_rate) * term_factor).sum())
    operating_cost = float(len(group) * float(params.operating_cost_per_loan))
    return revenue - expected_loss - funding_cost - operating_cost


def _concentration(counts: pd.Series, total: int) -> Concentration:
    shares = [int(value) / total for value in counts.tolist()] if total else []
    top1 = shares[0] if shares else 0.0
    top5 = float(sum(shares[:5]))
    hhi = float(sum(share * share for share in shares))
    return Concentration(top1_pct=float(top1), top5_pct=top5, hhi=hhi)


__all__ = [
    "HIGH_CONCENTRATION_HHI",
    "HIGH_CONCENTRATION_TOP1",
    "OTHER_SEGMENT",
    "Concentration",
    "ProfitParams",
    "SegmentProfileResult",
    "SegmentRow",
    "segment_profile",
]
