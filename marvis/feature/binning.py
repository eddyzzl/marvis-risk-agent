from __future__ import annotations

import numpy as np

from marvis.feature.errors import BinningError


def equal_frequency_edges(
    values: np.ndarray,
    bin_count: int,
    *,
    min_bin_pct: float = 0.0,
) -> np.ndarray:
    _validate_bin_count(bin_count)
    arr = _finite_values(values)
    if arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    unique_values = np.unique(arr)
    if unique_values.size == 2:
        midpoint = float((unique_values[0] + unique_values[1]) / 2)
        return np.array([-np.inf, midpoint, np.inf], dtype=float)
    quantiles = np.linspace(0, 1, int(bin_count) + 1)
    edges = np.unique(np.quantile(arr, quantiles))
    if edges.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    if min_bin_pct > 0:
        edges = _merge_small_count_bins(arr, edges, min_bin_pct)
    return edges


def degraded_bin_diagnostic(
    edges: np.ndarray,
    requested_bin_count: int,
    *,
    feature: str | None = None,
) -> dict | None:
    """FIN-3 #3: opt-in diagnostic for silent bin degradation.

    :func:`equal_frequency_edges` (and the other edge builders) silently return
    fewer bins than requested when a single/near-constant column or repeated values
    collapse the quantile edges -- the return-value behaviour many callers depend on
    is intentionally unchanged. This companion lets a screening / CSI caller detect
    that degradation and surface it through its own data_quality / warnings channel:
    returns a warning dict when the actual bin count (``len(edges) - 1``) is below
    ``requested_bin_count``, else ``None`` (no degradation -> nothing to report).
    """
    edge_arr = np.asarray(edges, dtype=float)
    actual_bins = max(0, edge_arr.size - 1)
    requested = int(requested_bin_count)
    if actual_bins >= requested:
        return None
    warning = {
        "kind": "degraded_binning",
        "requested_bin_count": requested,
        "actual_bin_count": actual_bins,
        "message": (
            f"分箱降级:请求 {requested} 个箱,实际只得到 {actual_bins} 个"
            "(单一取值或重复值折叠了分位边界)。"
        ),
    }
    if feature is not None:
        warning["feature"] = str(feature)
    return warning


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
    min_bin_pct: float = 0.0,
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

    if min_bin_pct > 0:
        stats, edges = _merge_small_share_stats(stats, edges, min_bin_pct)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(float)


def monotonic_direction(
    values: np.ndarray,
    target: np.ndarray,
    edges: np.ndarray,
    *,
    direction: str = "auto",
) -> str:
    """Resolve the bad-rate monotonic direction for a binned binary target."""
    arr, tgt = _finite_pairs(values, target)
    edge_arr = np.asarray(edges, dtype=float)
    if arr.size == 0:
        return _normalize_direction(direction, default="increasing")
    stats = [stat for stat in _bin_stats(arr, tgt, edge_arr) if stat["count"] > 0]
    return _resolve_monotonic_direction(direction, arr, tgt, stats)


def monotonic_edges(
    values: np.ndarray,
    target: np.ndarray,
    edges: np.ndarray,
    *,
    direction: str = "auto",
) -> np.ndarray:
    """Merge adjacent bins until bin bad rates are monotonic.

    The merge is a post-processing step over an existing edge proposal. It keeps
    the original open endpoints and only removes inner split points.
    """
    arr, tgt = _finite_pairs(values, target)
    edge_arr = np.asarray(edges, dtype=float)
    if edge_arr.ndim != 1 or edge_arr.size < 2:
        raise BinningError("edges must be a one-dimensional array with at least 2 values")
    if np.any(np.diff(edge_arr) <= 0):
        raise BinningError("edges must be strictly increasing")
    if arr.size == 0:
        return edge_arr.astype(float)

    stats = _bin_stats(arr, tgt, edge_arr)
    edge_arr = _edges_without_empty_bins(edge_arr, stats)
    stats = [stat for stat in stats if stat["count"] > 0]
    if len(stats) <= 1:
        return np.array([-np.inf, np.inf], dtype=float)

    resolved_direction = _resolve_monotonic_direction(direction, arr, tgt, stats)
    while len(stats) > 1 and not _stats_monotonic(stats, resolved_direction):
        index = _least_distinct_violating_pair(stats, resolved_direction)
        stats[index] = _merge_stats(stats[index], stats[index + 1])
        del stats[index + 1]
        edge_arr = np.delete(edge_arr, index + 1)

    edge_arr[0] = -np.inf
    edge_arr[-1] = np.inf
    return edge_arr.astype(float)


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


def _merge_small_count_bins(values: np.ndarray, edges: np.ndarray, min_bin_pct: float) -> np.ndarray:
    """Merge any bin whose share of ``values`` is below ``min_bin_pct`` into its
    smaller neighbor (PREP-9), for binning methods with no target/good-bad split to
    drive a chi2-based merge (:func:`equal_frequency_edges`). Repeats until every
    surviving bin clears the threshold or only one bin remains."""
    edges = np.asarray(edges, dtype=float)
    total = values.size
    if total == 0 or edges.size <= 2:
        return edges
    min_count = min_bin_pct * total
    while edges.size > 2:
        counts = _bin_counts(values, edges)
        small = [index for index, count in enumerate(counts) if count < min_count]
        if not small:
            break
        index = small[0]
        # Merge with whichever neighbor is smaller (ties -> the left/previous bin),
        # so a small edge bin doesn't get stranded with no smaller neighbor to join.
        if index == 0:
            merge_with = 1
        elif index == len(counts) - 1:
            merge_with = index - 1
        else:
            merge_with = index - 1 if counts[index - 1] <= counts[index + 1] else index + 1
        drop_index = max(index, merge_with)
        edges = np.delete(edges, drop_index)
    return edges


def _bin_counts(values: np.ndarray, edges: np.ndarray) -> list[int]:
    assigned = assign_bins(values, edges)
    return [int(np.sum(assigned == index)) for index in range(len(edges) - 1)]


def _merge_small_share_stats(
    stats: list[dict[str, int]], edges: np.ndarray, min_bin_pct: float
) -> tuple[list[dict[str, int]], np.ndarray]:
    """Merge any bin whose share of the fit rows is below ``min_bin_pct`` (PREP-9)
    into its statistically-nearest neighbor (chi2-least-distinct), same merge rule
    :func:`chimerge_edges` already uses for its ``max_bins`` reduction. Runs after
    the p-value convergence loop so it never fights that loop's own merges; repeats
    until every surviving bin clears the threshold or only one bin remains."""
    stats = list(stats)
    edges = np.asarray(edges, dtype=float)
    total = sum(stat["count"] for stat in stats)
    if total == 0:
        return stats, edges
    min_count = min_bin_pct * total
    while len(stats) > 1:
        shares = [stat["count"] for stat in stats]
        if min(shares) >= min_count:
            break
        index = _least_distinct_adjacent_pair_containing_smallest(stats)
        stats[index] = _merge_stats(stats[index], stats[index + 1])
        del stats[index + 1]
        edges = np.delete(edges, index + 1)
    return stats, edges


def _least_distinct_adjacent_pair_containing_smallest(stats: list[dict[str, int]]) -> int:
    """Like :func:`_least_distinct_adjacent_pair`, but restricted to adjacent pairs
    that include the smallest bin -- so a below-threshold bin always gets merged
    (into whichever neighbor is statistically closest), rather than some unrelated
    pair of already-large bins."""
    smallest = min(range(len(stats)), key=lambda index: stats[index]["count"])
    candidates = [index for index in (smallest - 1, smallest) if 0 <= index < len(stats) - 1]
    chi2_values = [(index, _adjacent_chi2(stats[index], stats[index + 1])) for index in candidates]
    return min(chi2_values, key=lambda item: item[1])[0]


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


def _resolve_monotonic_direction(
    direction: str,
    values: np.ndarray,
    target: np.ndarray,
    stats: list[dict[str, int]],
) -> str:
    normalized = _normalize_direction(direction, default="")
    if normalized:
        return normalized
    rates = [_bad_rate(stat) for stat in stats]
    increasing_violation = _total_direction_violation(rates, "increasing")
    decreasing_violation = _total_direction_violation(rates, "decreasing")
    if increasing_violation < decreasing_violation:
        return "increasing"
    if decreasing_violation < increasing_violation:
        return "decreasing"
    if np.std(values) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(values, target)[0, 1])
        if np.isfinite(corr) and corr < 0:
            return "decreasing"
    return "increasing"


def _normalize_direction(direction: str, *, default: str) -> str:
    value = str(direction or "auto").strip().lower()
    aliases = {
        "auto": "",
        "asc": "increasing",
        "up": "increasing",
        "increasing": "increasing",
        "desc": "decreasing",
        "down": "decreasing",
        "decreasing": "decreasing",
    }
    if value not in aliases:
        raise BinningError("direction must be auto, increasing, or decreasing")
    return aliases[value] or default


def _stats_monotonic(stats: list[dict[str, int]], direction: str) -> bool:
    rates = [_bad_rate(stat) for stat in stats]
    if direction == "increasing":
        return all(left <= right for left, right in zip(rates, rates[1:]))
    return all(left >= right for left, right in zip(rates, rates[1:]))


def _least_distinct_violating_pair(stats: list[dict[str, int]], direction: str) -> int:
    rates = [_bad_rate(stat) for stat in stats]
    if direction == "increasing":
        violating = [index for index, (left, right) in enumerate(zip(rates, rates[1:])) if left > right]
    else:
        violating = [index for index, (left, right) in enumerate(zip(rates, rates[1:])) if left < right]
    if not violating:
        return _least_distinct_adjacent_pair(stats)
    return min(violating, key=lambda index: _adjacent_chi2(stats[index], stats[index + 1]))


def _total_direction_violation(rates: list[float], direction: str) -> float:
    if direction == "increasing":
        return float(sum(max(0.0, left - right) for left, right in zip(rates, rates[1:])))
    return float(sum(max(0.0, right - left) for left, right in zip(rates, rates[1:])))


def _bad_rate(stat: dict[str, int]) -> float:
    return float(stat["bad"] / stat["count"]) if stat["count"] else 0.0


def _validate_bin_count(bin_count: int) -> None:
    if int(bin_count) < 1:
        raise BinningError("bin_count must be at least 1")


__all__ = [
    "assign_bins",
    "chimerge_edges",
    "degraded_bin_diagnostic",
    "equal_frequency_edges",
    "equal_width_edges",
    "manual_edges",
    "monotonic_direction",
    "monotonic_edges",
    "tree_edges",
]
