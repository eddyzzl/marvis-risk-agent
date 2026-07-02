from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from marvis.feature.binning import assign_bins, equal_frequency_edges
from marvis.feature.contracts import FeatureMetrics
from marvis.feature.correlation import safe_correlation
from marvis.feature.errors import FeatureError
from marvis.feature.iv import compute_woe_iv


# FS-9: the equal-frequency IV binning convention used by feature_metrics() by default.
DEFAULT_IV_BINS = 10


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


def weighted_feature_ks(scores: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    scores_arr, target_arr, weight_arr = _finite_binary_weighted_triples(scores, target, weights)
    if scores_arr.size == 0:
        return 0.0
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    sorted_target = target_arr[order]
    sorted_weight = weight_arr[order]
    bad_weight = np.where(sorted_target == 1, sorted_weight, 0.0)
    good_weight = np.where(sorted_target == 0, sorted_weight, 0.0)
    total_bad = float(bad_weight.sum())
    total_good = float(good_weight.sum())
    if total_bad <= 0 or total_good <= 0:
        return 0.0
    cum_bad = np.cumsum(bad_weight) / total_bad
    cum_good = np.cumsum(good_weight) / total_good
    change_points = np.r_[np.where(np.diff(sorted_scores) != 0)[0], len(sorted_scores) - 1]
    return float(np.max(np.abs(cum_bad[change_points] - cum_good[change_points])))


def feature_auc(scores: np.ndarray, target: np.ndarray, *, direction_agnostic: bool = False) -> float:
    scores_arr, target_arr = _finite_binary_pairs(scores, target)
    pos = scores_arr[target_arr == 1]
    neg = scores_arr[target_arr == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = rankdata(scores_arr)
    pos_ranks = ranks[target_arr == 1]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    if direction_agnostic:
        auc = max(auc, 1 - auc)
    return float(auc)


def weighted_feature_auc(scores: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    scores_arr, target_arr, weight_arr = _finite_binary_weighted_triples(scores, target, weights)
    total_bad = float(weight_arr[target_arr == 1].sum())
    total_good = float(weight_arr[target_arr == 0].sum())
    if total_bad <= 0 or total_good <= 0:
        return 0.5
    order = np.argsort(scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    sorted_target = target_arr[order]
    sorted_weight = weight_arr[order]
    numerator = 0.0
    cumulative_good = 0.0
    start = 0
    n = sorted_scores.size
    while start < n:
        end = start + 1
        while end < n and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_target = sorted_target[start:end]
        group_weight = sorted_weight[start:end]
        group_bad = float(group_weight[group_target == 1].sum())
        group_good = float(group_weight[group_target == 0].sum())
        numerator += group_bad * cumulative_good + 0.5 * group_bad * group_good
        cumulative_good += group_good
        start = end
    return float(numerator / (total_bad * total_good))


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


def head_tail_lift(
    values: np.ndarray,
    target: np.ndarray,
    *,
    fractions: tuple[float, ...] = (0.05, 0.10),
    min_rows: int = 20,
) -> dict[str, float | None]:
    """Risk-direction-aware lift at the head (highest-risk) and tail (lowest-risk)
    extremes of a feature, at the given row fractions (spec §2 头尾 lift).

    The head is the slice of rows at the feature's high-bad-rate end and the tail the
    low-bad-rate end. Which end is which is set by the SIGN of the feature↔target
    correlation, so a positively- and a negatively-associated feature both report
    head = high-risk-end. ``lift`` = bad rate within the slice ÷ overall bad rate.

    Deterministic: stable mergesort + exact integer row counts, no RNG and no model,
    so it preserves the deterministic-metric invariant. Returns ``None`` for every key
    when there are too few labelled rows (< ``min_rows``) or the overall bad rate is 0.
    """
    keys = [f"lift_{end}_{int(round(frac * 100))}" for frac in fractions for end in ("head", "tail")]
    scores_arr, target_arr = _finite_binary_pairs(values, target)
    n = scores_arr.size
    base_rate = float(np.mean(target_arr == 1)) if n else 0.0
    if n < min_rows or base_rate <= 0:
        return {key: None for key in keys}
    # risk_sign: +1 → larger feature value = higher risk; -1 → inverted. A flat or
    # U-shaped feature yields corr≈0 → deterministic fallback to +1 (its slices land
    # at lift≈1, so the head/tail labelling is moot).
    risk_sign = float(np.sign(safe_correlation(scores_arr, target_arr.astype(float)))) or 1.0
    order = np.argsort(risk_sign * scores_arr, kind="mergesort")  # ascending risk
    result: dict[str, float | None] = {}
    for frac in fractions:
        count = max(1, int(np.floor(frac * n)))
        pct = int(round(frac * 100))
        result[f"lift_head_{pct}"] = float(np.mean(target_arr[order[-count:]] == 1) / base_rate)
        result[f"lift_tail_{pct}"] = float(np.mean(target_arr[order[:count]] == 1) / base_rate)
    return result


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
    expected_total = float(expected.sum())
    actual_total = float(actual.sum())
    if expected_total <= 0 or actual_total <= 0:
        return 0.0
    expected = expected / expected_total
    actual = actual / actual_total
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


def weighted_feature_psi(
    base_values: np.ndarray,
    compare_values: np.ndarray,
    edges: np.ndarray,
    *,
    base_weights: np.ndarray,
    compare_weights: np.ndarray,
    smoothing: float = 1e-6,
) -> float:
    return compute_psi(
        _weighted_bin_distribution(base_values, edges, base_weights),
        _weighted_bin_distribution(compare_values, edges, compare_weights),
        smoothing=smoothing,
    )


def feature_metrics(
    values: np.ndarray,
    target: np.ndarray,
    *,
    feature: str,
    bins: int = DEFAULT_IV_BINS,
    compare_values: np.ndarray | None = None,
) -> FeatureMetrics:
    values_arr = np.asarray(values, dtype=float)
    edges = equal_frequency_edges(values_arr, bins)
    try:
        iv = compute_woe_iv(values_arr, target, edges, feature=feature).total_iv
    except FeatureError:
        iv = 0.0
    lift = feature_lift(values_arr, target, bins=bins)
    psi = feature_psi(values_arr, compare_values, edges) if compare_values is not None else None
    return FeatureMetrics(
        feature=feature,
        iv=iv,
        ks=feature_ks(values_arr, target),
        auc=feature_auc(values_arr, target, direction_agnostic=True),
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


def _weighted_bin_distribution(values: np.ndarray, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if values_arr.shape != weight_arr.shape:
        raise FeatureError("values and weights must have the same shape")
    assigned = assign_bins(values_arr, edges)
    valid = (assigned >= 0) & np.isfinite(weight_arr)
    if not np.any(valid):
        return np.zeros(len(edges) - 1, dtype=float)
    clipped_weights = np.clip(weight_arr[valid], 0.0, None)
    counts = np.bincount(
        assigned[valid],
        weights=clipped_weights,
        minlength=len(edges) - 1,
    ).astype(float)
    total = float(counts.sum())
    return counts / total if total > 0 else np.zeros(len(edges) - 1, dtype=float)


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


def _finite_binary_weighted_triples(
    scores: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    if scores_arr.shape != target_arr.shape or scores_arr.shape != weight_arr.shape:
        raise FeatureError("scores, target, and weights must have the same shape")
    mask = np.isfinite(scores_arr) & np.isfinite(target_arr) & np.isfinite(weight_arr) & (weight_arr >= 0)
    scores_arr = scores_arr[mask]
    target_arr = target_arr[mask]
    weight_arr = weight_arr[mask]
    if target_arr.size and not np.all(np.isin(target_arr, [0, 1])):
        raise FeatureError("target must be binary 0/1")
    return scores_arr, target_arr.astype(int), weight_arr


__all__ = [
    "DEFAULT_IV_BINS",
    "compute_psi",
    "feature_auc",
    "feature_ks",
    "feature_lift",
    "feature_metrics",
    "feature_psi",
    "head_tail_lift",
    "weighted_feature_auc",
    "weighted_feature_ks",
    "weighted_feature_psi",
]
