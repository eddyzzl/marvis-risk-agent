"""细分画像内核 (segment_profile).

按 segment_col 分组统计计数/占比/审批率/坏率/均分/净利润，并给出集中度
(top1/top5/HHI) 与红旗 (high_concentration / sparse_segment 归并「其他」)。

利润列复用 strategy 包 profit 内核的同款公式，但这里本地实现（不跨包 import）——
公式简单，两包各自测试锁同一手算值，注释互指 (strategy/profit.py profit_calc)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pandas as pd

from marvis.packs.analysis.errors import AnalysisError

#: top_k 之外的小细分归并到该标签。
OTHER_SEGMENT = "其他"
#: top1 占比阈值下限（另需超过均匀占比的 HIGH_CONCENTRATION_EVEN_MULTIPLE 倍）。
HIGH_CONCENTRATION_TOP1 = 0.40
#: HHI 阈值下限（另需超过均匀分布 HHI=1/n 的 HIGH_CONCENTRATION_EVEN_MULTIPLE 倍）。
HIGH_CONCENTRATION_HHI = 0.25
#: n 个细分均匀分布时 top1 = HHI = 1/n；阈值至少取其该倍数，均匀分布永不触发。
HIGH_CONCENTRATION_EVEN_MULTIPLE = 1.5


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
    concentration_basis: str = "count"
    ead_concentration: Concentration | None = None
    ead_concentration_basis: str | None = None


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
    ead_concentration = None
    ead_concentration_basis = None
    if ead_col:
        ead = pd.to_numeric(df[ead_col], errors="coerce")
        if ead.isna().any() or any(not math.isfinite(float(value)) for value in ead):
            raise AnalysisError(f"segment_profile: EAD 列 `{ead_col}` 含非有限或非数值。")
        if (ead < 0).any():
            raise AnalysisError(f"segment_profile: EAD 列 `{ead_col}` 不能为负数。")
        total_ead = float(ead.sum())
        if total_ead <= 0:
            raise AnalysisError(f"segment_profile: EAD 列 `{ead_col}` 合计必须大于 0。")
        ead_by_segment = (
            pd.DataFrame({"segment": df[segment_col].astype(str), "ead": ead})
            .groupby("segment", sort=False)["ead"]
            .sum()
            .sort_values(ascending=False)
        )
        ead_concentration = _concentration(ead_by_segment, total_ead)
        ead_concentration_basis = "ead"

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
    segment_count = int(len(raw_counts))
    top1_threshold, hhi_threshold = concentration_thresholds(segment_count)
    if concentration.top1_pct > top1_threshold or concentration.hhi > hhi_threshold:
        red_flags.append(
            {
                "kind": "high_concentration",
                "top1_pct": concentration.top1_pct,
                "hhi": concentration.hhi,
                "segment_count": segment_count,
                "top1_threshold": top1_threshold,
                "hhi_threshold": hhi_threshold,
                "message": (
                    f"细分高度集中：{segment_count} 个细分，top1 占比 {concentration.top1_pct:.1%}，"
                    f"HHI {concentration.hhi:.3f}（阈值 top1>{top1_threshold:.1%} 或 HHI>{hhi_threshold:.3f}；"
                    f"按细分个数调整：至少为均匀分布 1/{segment_count} 的 "
                    f"{HIGH_CONCENTRATION_EVEN_MULTIPLE:g} 倍，且不低于 {HIGH_CONCENTRATION_TOP1:.0%} / "
                    f"{HIGH_CONCENTRATION_HHI:g}）。"
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

    return SegmentProfileResult(
        segments=rows,
        concentration=concentration,
        concentration_basis="count",
        ead_concentration=ead_concentration,
        ead_concentration_basis=ead_concentration_basis,
        red_flags=red_flags,
    )


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


def concentration_thresholds(segment_count: int) -> tuple[float, float]:
    """(top1, HHI) 阈值随细分个数调整。

    n 个细分均匀分布时 top1 = HHI = 1/n：n 很小时固定阈值（top1>40% / HHI>0.25）
    对均匀分布也会触发（两个各 50% 的细分 HHI=0.5）。因此阈值取
    max(固定下限, EVEN_MULTIPLE / n)：均匀分布永不触发；n>=6 时退回固定阈值。
    """
    even_share = 1.0 / segment_count if segment_count > 0 else 1.0
    floor = HIGH_CONCENTRATION_EVEN_MULTIPLE * even_share
    return max(HIGH_CONCENTRATION_TOP1, floor), max(HIGH_CONCENTRATION_HHI, floor)


def _concentration(weights: pd.Series, total: float) -> Concentration:
    shares = [float(value) / total for value in weights.tolist()] if total else []
    top1 = shares[0] if shares else 0.0
    top5 = float(sum(shares[:5]))
    hhi = float(sum(share * share for share in shares))
    return Concentration(top1_pct=float(top1), top5_pct=top5, hhi=hhi)


__all__ = [
    "HIGH_CONCENTRATION_EVEN_MULTIPLE",
    "HIGH_CONCENTRATION_HHI",
    "HIGH_CONCENTRATION_TOP1",
    "OTHER_SEGMENT",
    "Concentration",
    "ProfitParams",
    "SegmentProfileResult",
    "SegmentRow",
    "concentration_thresholds",
    "segment_profile",
]
