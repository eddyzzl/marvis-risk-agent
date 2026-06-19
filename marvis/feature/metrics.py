from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from marvis.feature.binning import assign_bins, equal_frequency_edges
from marvis.feature.contracts import FeatureMetrics
from marvis.feature.errors import FeatureError
from marvis.feature.iv import compute_woe_iv


def feature_ks(scores: np.ndarray, target: np.ndarray) -> float:
    scores_arr, target_arr = _finite_binary_pairs(scores, target)
    if scores_arr.size == 0:
        return 0.0
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    sorted_target = target_arr[order]
    total_bad = int(np.sum(sorted_target == 1))
    total_good = int(np.sum(sorted_target == 0))
    if total_bad == 0 or total_good == 0:
        return 0.0
    cum_bad = np.cumsum(sorted_target == 1) / total_bad
    cum_good = np.cumsum(sorted_target == 0) / total_good
    change_points = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
    return float(np.max(np.abs(cum_bad[change_points] - cum_good[change_points])))


def feature_auc(scores: np.ndarray, target: np.ndarray) -> float:
    scores_arr, target_arr = _finite_binary_pairs(scores, target)
    pos = scores_arr[target_arr == 1]
    neg = scores_arr[target_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = rankdata(scores_arr)
    pos_ranks = ranks[target_arr == 1]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def feature_lift(scores: np.ndarray, target: np.ndarray, *, bins: int = 10) -> list[float]:
    scores_arr, target_arr = _finite_binary_pairs(scores, target)
    if scores_arr.size == 0:
        return []
    base_rate = float(np.mean(target_arr == 1))
    edges = equal_frequency_edges(scores_arr, bins)
    assigned = assign_bins(scores_arr, edges)
    lifts = []
    for group in reversed(range(len(edges) - 1)):
        mask = assigned == group
        if not np.any(mask) or base_rate <= 0:
            lifts.append(0.0)
        else:
            lifts.append(float(np.mean(target_arr[mask] == 1) / base_rate))
    return lifts


def compute_psi(
    expected_dist: np.ndarray,
    actual_dist: np.ndarray,
    *,
    smoothing: float = 1e-6,
) -> float:
    expected = np.asarray(expected_dist, dtype=float)
    actual = np.asarray(actual_dist, dtype=float)
    if expected.shape != actual.shape:
        raise FeatureError("PSI distributions must have the same shape")
    if np.any(expected < 0) or np.any(actual < 0):
        raise FeatureError("PSI distributions must be non-negative")
    expected = np.where(expected <= 0, smoothing, expected)
    actual = np.where(actual <= 0, smoothing, actual)
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    return max(0.0, psi)


def feature_psi(
    base_values: np.ndarray,
    compare_values: np.ndarray,
    edges: np.ndarray,
    *,
    smoothing: float = 1e-6,
) -> float:
    return compute_psi(
        _bin_distribution(base_values, edges),
        _bin_distribution(compare_values, edges),
        smoothing=smoothing,
    )


def feature_metrics(
    values: np.ndarray,
    target: np.ndarray,
    *,
    feature: str,
    bins: int = 10,
    compare_values: np.ndarray | None = None,
) -> FeatureMetrics:
    values_arr = np.asarray(values, dtype=float)
    edges = equal_frequency_edges(values_arr, bins)
    binning = compute_woe_iv(values_arr, target, edges, feature=feature)
    lift = feature_lift(values_arr, target, bins=bins)
    psi = feature_psi(values_arr, compare_values, edges) if compare_values is not None else None
    return FeatureMetrics(
        feature=feature,
        iv=binning.total_iv,
        ks=feature_ks(values_arr, target),
        auc=feature_auc(values_arr, target),
        psi=psi,
        missing_rate=float(np.mean(~np.isfinite(values_arr))),
        unique_count=int(np.unique(values_arr[np.isfinite(values_arr)]).size),
        lift_top_bin=lift[0] if lift else 0.0,
    )


def _bin_distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    assigned = assign_bins(values, edges)
    valid = assigned >= 0
    if not np.any(valid):
        return np.zeros(len(edges) - 1, dtype=float)
    counts = np.bincount(assigned[valid], minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def _finite_binary_pairs(scores: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if scores_arr.shape != target_arr.shape:
        raise FeatureError("scores and target must have the same shape")
    mask = np.isfinite(scores_arr) & np.isfinite(target_arr)
    scores_arr = scores_arr[mask]
    target_arr = target_arr[mask]
    if target_arr.size and not np.all(np.isin(target_arr, [0, 1])):
        raise FeatureError("target must be binary 0/1")
    return scores_arr, target_arr.astype(int)


__all__ = [
    "compute_psi",
    "feature_auc",
    "feature_ks",
    "feature_lift",
    "feature_metrics",
    "feature_psi",
]
