from __future__ import annotations

import numpy as np

from marvis.feature.errors import BinningError


def equal_frequency_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    _validate_bin_count(bin_count)
    arr = _finite_values(values)
    if arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0, 1, int(bin_count) + 1)
    edges = np.unique(np.quantile(arr, quantiles))
    if edges.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def equal_width_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    _validate_bin_count(bin_count)
    arr = _finite_values(values)
    if arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    if lower == upper:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = np.linspace(lower, upper, int(bin_count) + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def manual_edges(breakpoints: list[float]) -> np.ndarray:
    if not breakpoints:
        raise BinningError("manual binning requires at least one breakpoint")
    points = _finite_values(np.asarray(breakpoints, dtype=float))
    if points.size == 0:
        raise BinningError("manual binning breakpoints must be finite")
    return np.array([-np.inf, *np.unique(points), np.inf], dtype=float)


def chimerge_edges(
    values: np.ndarray,
    target: np.ndarray,
    *,
    max_bins: int,
    min_pvalue: float = 0.05,
    init_bins: int = 100,
) -> np.ndarray:
    _validate_bin_count(max_bins)
    arr, tgt = _finite_pairs(values, target)
    if arr.size == 0 or np.unique(arr).size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = equal_frequency_edges(arr, min(init_bins, max_bins * 20, arr.size))
    stats = _bin_stats(arr, tgt, edges)
    edges = _edges_without_empty_bins(edges, stats)
    stats = [stat for stat in stats if stat["count"] > 0]
    if len(stats) <= 1:
        return np.array([-np.inf, np.inf], dtype=float)

    while len(stats) > max_bins:
        index = _least_distinct_adjacent_pair(stats)
        stats[index] = _merge_stats(stats[index], stats[index + 1])
        del stats[index + 1]
        edges = np.delete(edges, index + 1)

    while len(stats) > 2:
        pvalues = [_adjacent_pvalue(stats[index], stats[index + 1]) for index in range(len(stats) - 1)]
        max_index = int(np.argmax(pvalues))
        if pvalues[max_index] <= min_pvalue:
            break
        stats[max_index] = _merge_stats(stats[max_index], stats[max_index + 1])
        del stats[max_index + 1]
        edges = np.delete(edges, max_index + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(float)


def tree_edges(
    values: np.ndarray,
    target: np.ndarray,
    *,
    max_bins: int,
    min_samples_leaf: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    _validate_bin_count(max_bins)
    arr, tgt = _finite_pairs(values, target)
    if arr.size == 0 or np.unique(arr).size <= 1 or np.unique(tgt).size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    from sklearn.tree import DecisionTreeClassifier, _tree

    tree = DecisionTreeClassifier(
        max_leaf_nodes=int(max_bins),
        min_samples_leaf=min_samples_leaf,
        random_state=int(seed),
    )
    tree.fit(arr.reshape(-1, 1), tgt)
    thresholds = sorted(
        float(value)
        for value in tree.tree_.threshold
        if value != _tree.TREE_UNDEFINED
    )
    if not thresholds:
        return np.array([-np.inf, np.inf], dtype=float)
    return np.array([-np.inf, *np.unique(thresholds), np.inf], dtype=float)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise BinningError("edges must be a one-dimensional array with at least 2 values")
    if np.any(np.diff(edges) <= 0):
        raise BinningError("edges must be strictly increasing")
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=int)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    assigned = np.searchsorted(edges[1:-1], arr[mask], side="right")
    out[mask] = np.clip(assigned, 0, edges.size - 2)
    return out


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _finite_pairs(values: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if arr.shape != tgt.shape:
        raise BinningError("values and target must have the same shape")
    mask = np.isfinite(arr) & np.isfinite(tgt)
    return arr[mask], tgt[mask].astype(int)


def _bin_stats(values: np.ndarray, target: np.ndarray, edges: np.ndarray) -> list[dict[str, int]]:
    assigned = assign_bins(values, edges)
    stats = []
    for index in range(len(edges) - 1):
        mask = assigned == index
        bad = int(np.sum(target[mask] == 1))
        count = int(np.sum(mask))
        stats.append({"good": count - bad, "bad": bad, "count": count})
    return stats


def _edges_without_empty_bins(edges: np.ndarray, stats: list[dict[str, int]]) -> np.ndarray:
    kept = [edges[0]]
    for index, stat in enumerate(stats):
        if stat["count"] > 0:
            kept.append(edges[index + 1])
    return np.asarray(kept, dtype=float)


def _least_distinct_adjacent_pair(stats: list[dict[str, int]]) -> int:
    chi2_values = [
        _adjacent_chi2(stats[index], stats[index + 1])
        for index in range(len(stats) - 1)
    ]
    return int(np.argmin(chi2_values))


def _adjacent_chi2(left: dict[str, int], right: dict[str, int]) -> float:
    from scipy.stats import chi2_contingency

    chi2, _pvalue, _dof, _expected = chi2_contingency(
        _smoothed_contingency(left, right),
        correction=False,
    )
    return float(chi2)


def _adjacent_pvalue(left: dict[str, int], right: dict[str, int]) -> float:
    from scipy.stats import chi2_contingency

    _chi2, pvalue, _dof, _expected = chi2_contingency(
        _smoothed_contingency(left, right),
        correction=False,
    )
    return float(pvalue)


def _smoothed_contingency(left: dict[str, int], right: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [[left["good"], left["bad"]], [right["good"], right["bad"]]],
        dtype=float,
    ) + 0.5


def _merge_stats(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {
        "good": left["good"] + right["good"],
        "bad": left["bad"] + right["bad"],
        "count": left["count"] + right["count"],
    }


def _validate_bin_count(bin_count: int) -> None:
    if int(bin_count) < 1:
        raise BinningError("bin_count must be at least 1")


__all__ = [
    "assign_bins",
    "chimerge_edges",
    "equal_frequency_edges",
    "equal_width_edges",
    "manual_edges",
    "tree_edges",
]
