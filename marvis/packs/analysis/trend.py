"""稳定性趋势内核 (score_stability_trend + feature_csi_trend).

复用 modeling monitor_run 的同一 PSI/CSI 内核 (marvis.validation.binning.
bin_distribution + compute_psi, INV-1) 与训练期 baseline 快照 (score_edges /
score_distribution.train.bin_proportions / feature_distributions[f].quantile_edges)，
把"某月一张打分表 vs baseline"的 PSI/CSI 沿时间轴排成趋势，逐月给 green/amber/red。

阈值默认沿用 modeling 的 MONITOR_RUN_THRESHOLDS（score_psi / feature_csi_max：
<0.10 green, 0.10-0.25 amber, >=0.25 red）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from marvis.packs.analysis.errors import AnalysisError
from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS
from marvis.validation.binning import bin_distribution, compute_psi

#: green/amber 边界与 amber/red 边界（沿用 monitor_run score_psi 档位）。
_PSI_WARN = float(MONITOR_RUN_THRESHOLDS["score_psi"]["warn"])
_PSI_FAIL = float(MONITOR_RUN_THRESHOLDS["score_psi"]["fail"])


@dataclass(frozen=True)
class TrendPoint:
    month: str
    metric: float | None
    level: str  # green | amber | red | n/a
    sample_count: int


@dataclass(frozen=True)
class FeatureTrendPoint:
    feature: str
    month: str
    csi: float | None
    sample_count: int


@dataclass(frozen=True)
class StabilityTrendResult:
    metric_name: str  # "psi" or "max_csi"
    trend: list[TrendPoint]
    per_feature_trend: list[FeatureTrendPoint]
    red_flags: list[dict] = field(default_factory=list)


def level_for(value: float | None, *, warn: float = _PSI_WARN, fail: float = _PSI_FAIL) -> str:
    if value is None:
        return "n/a"
    if value >= fail:
        return "red"
    if value >= warn:
        return "amber"
    return "green"


def score_stability_trend(
    baseline: dict,
    month_frames: list[tuple[str, pd.DataFrame]],
    *,
    score_col: str,
    warn: float = _PSI_WARN,
    fail: float = _PSI_FAIL,
) -> StabilityTrendResult:
    """逐月 score PSI 趋势（每月一张打分表 vs baseline train 分布）。

    ``month_frames`` 是 (month, frame) 列表，已按月排好；每个 frame 至少含 ``score_col``。
    与 monitor_run 的 score PSI 完全同款：bin_distribution(finite_scores, score_edges)
    vs baseline train bin_proportions，再 compute_psi。
    """
    edges = np.asarray(baseline.get("score_edges") or [], dtype=float)
    train_dist = ((baseline.get("score_distribution") or {}).get("train") or {}).get("bin_proportions")
    if edges.size < 2 or not train_dist:
        raise AnalysisError("baseline 缺少 score_edges 或 train 分布，无法计算 score PSI 趋势。")
    expected = np.asarray(train_dist, dtype=float)

    trend: list[TrendPoint] = []
    for month, frame in month_frames:
        if score_col not in frame.columns:
            raise AnalysisError(f"打分表缺少评分列 `{score_col}`（月 {month}）。")
        scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            trend.append(TrendPoint(month=month, metric=None, level="n/a", sample_count=0))
            continue
        actual = bin_distribution(finite, edges)
        psi = float(compute_psi(expected, actual))
        trend.append(TrendPoint(month=month, metric=psi, level=level_for(psi, warn=warn, fail=fail), sample_count=int(finite.size)))

    red_flags = _month_gap_flags([month for month, _ in month_frames])
    return StabilityTrendResult(metric_name="psi", trend=trend, per_feature_trend=[], red_flags=red_flags)


def feature_csi_trend(
    baseline: dict,
    month_frames: list[tuple[str, pd.DataFrame]],
    *,
    feature_cols: list[str] | None = None,
    warn: float = _PSI_WARN,
    fail: float = _PSI_FAIL,
    top_features: int = 10,
) -> StabilityTrendResult:
    """逐月 max feature CSI 趋势 + 前 top_features 特征逐月 CSI。

    与 monitor_run 的 feature CSI 完全同款：对每个特征
    bin_distribution(finite_values, quantile_edges) vs 均匀分布 (1/bin_count)，
    再 compute_psi。每月取所有特征 CSI 的最大值作为该月趋势点。
    """
    feature_baselines = baseline.get("feature_distributions") or {}
    if not feature_baselines:
        raise AnalysisError("baseline 缺少 feature_distributions，无法计算 CSI 趋势。")
    features = list(feature_cols) if feature_cols else list(feature_baselines.keys())

    trend: list[TrendPoint] = []
    per_feature: list[FeatureTrendPoint] = []
    for month, frame in month_frames:
        month_csis: dict[str, tuple[float, int]] = {}
        for feature in features:
            feature_baseline = feature_baselines.get(feature)
            if not isinstance(feature_baseline, dict) or feature not in frame.columns:
                continue
            edges = np.asarray(feature_baseline.get("quantile_edges") or [], dtype=float)
            if edges.size < 2:
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            actual = bin_distribution(finite, edges)
            bin_count = edges.size - 1
            expected = np.full(bin_count, 1.0 / bin_count, dtype=float)
            csi = float(compute_psi(expected, actual))
            month_csis[feature] = (csi, int(finite.size))
        if not month_csis:
            trend.append(TrendPoint(month=month, metric=None, level="n/a", sample_count=0))
            continue
        worst = max(month_csis.values(), key=lambda item: item[0])
        trend.append(
            TrendPoint(
                month=month,
                metric=worst[0],
                level=level_for(worst[0], warn=warn, fail=fail),
                sample_count=worst[1],
            )
        )
        # per-feature rows for the top_features most-drifted features this month
        ranked = sorted(month_csis.items(), key=lambda item: (-item[1][0], item[0]))[:top_features]
        for feature, (csi, count) in ranked:
            per_feature.append(FeatureTrendPoint(feature=feature, month=month, csi=csi, sample_count=count))

    red_flags = _month_gap_flags([month for month, _ in month_frames])
    return StabilityTrendResult(
        metric_name="max_csi", trend=trend, per_feature_trend=per_feature, red_flags=red_flags
    )


def _month_gap_flags(months: list[str]) -> list[dict]:
    """趋势月份不连续 -> month_gap 红旗（月份需为 YYYY-MM）。"""
    parsed = []
    for month in months:
        text = str(month)
        if len(text) >= 7 and text[4] == "-" and text[:4].isdigit() and text[5:7].isdigit():
            parsed.append((int(text[:4]), int(text[5:7])))
        else:
            return []  # non YYYY-MM labels: skip gap detection rather than false-flag
    gaps: list[str] = []
    for prev, current in zip(parsed[:-1], parsed[1:], strict=False):
        prev_index = prev[0] * 12 + prev[1]
        current_index = current[0] * 12 + current[1]
        if current_index - prev_index != 1:
            gaps.append(f"{prev[0]:04d}-{prev[1]:02d} -> {current[0]:04d}-{current[1]:02d}")
    if gaps:
        return [
            {
                "kind": "month_gap",
                "gaps": gaps,
                "message": f"趋势月份不连续：{'; '.join(gaps)}。",
            }
        ]
    return []


__all__ = [
    "FeatureTrendPoint",
    "StabilityTrendResult",
    "TrendPoint",
    "feature_csi_trend",
    "level_for",
    "score_stability_trend",
]
