import numpy as np
import pytest

from marvis.feature.binning import (
    assign_bins,
    chimerge_edges,
    degraded_bin_diagnostic,
    equal_frequency_edges,
    equal_width_edges,
    manual_edges,
    monotonic_direction,
    monotonic_edges,
    tree_edges,
)
from marvis.feature.errors import BinningError


def test_equal_frequency_edges_filter_nan_inf_and_dedupe_quantiles():
    edges = equal_frequency_edges(np.array([1, 1, 1, 2, 3, np.nan, np.inf]), 4)

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert np.all(np.diff(edges) > 0)
    assert len(edges) <= 5


def test_equal_frequency_edges_preserves_imbalanced_binary_split():
    values = np.array([0] * 95 + [1] * 5, dtype=float)

    edges = equal_frequency_edges(values, 10)

    assert edges.tolist() == [float("-inf"), 0.5, float("inf")]


def test_equal_width_edges_handle_empty_and_single_value_columns():
    empty = equal_width_edges(np.array([np.nan, np.inf]), 5)
    single = equal_width_edges(np.array([7, 7, np.nan]), 5)
    normal = equal_width_edges(np.array([0, 5, 10]), 2)

    assert empty.tolist() == [float("-inf"), float("inf")]
    assert single.tolist() == [float("-inf"), float("inf")]
    assert normal.tolist() == [float("-inf"), 5.0, float("inf")]


def test_manual_edges_sort_dedupe_and_reject_invalid_breakpoints():
    edges = manual_edges([3, 1, 3, 2])

    assert edges.tolist() == [float("-inf"), 1.0, 2.0, 3.0, float("inf")]
    with pytest.raises(BinningError):
        manual_edges([])
    with pytest.raises(BinningError):
        manual_edges([float("nan"), float("inf")])


def test_assign_bins_maps_nan_to_missing_bin_and_clips_extremes():
    edges = np.array([float("-inf"), 0.0, 10.0, float("inf")])
    assigned = assign_bins(np.array([float("nan"), -5.0, 0.0, 9.9, 10.0, 999.0]), edges)

    assert assigned.tolist() == [-1, 0, 1, 1, 2, 2]


def test_assign_bins_rejects_invalid_edges():
    with pytest.raises(BinningError):
        assign_bins(np.array([1, 2]), np.array([0.0]))
    with pytest.raises(BinningError):
        assign_bins(np.array([1, 2]), np.array([float("-inf"), 1.0, 1.0, float("inf")]))


def test_chimerge_edges_reduce_to_max_bins_and_keep_open_endpoints():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    target = np.array([0, 0, 0, 1, 1, 1, 1, 1])

    edges = chimerge_edges(values, target, max_bins=3, init_bins=8)

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert len(edges) <= 4
    assert np.all(np.diff(edges) > 0)


def test_equal_frequency_edges_min_bin_pct_merges_small_bins():
    """PREP-9: a bin below min_bin_pct share must be merged with a neighbor so every
    surviving bin clears the threshold."""
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.uniform(0, 90, 900), rng.uniform(95, 96, 100)])

    edges = equal_frequency_edges(values, 8)
    assigned = assign_bins(values, edges)
    counts = [int(np.sum(assigned == index)) for index in range(len(edges) - 1)]
    assert min(counts) / values.size < 0.15  # unconstrained: at least one small bin

    edges2 = equal_frequency_edges(values, 8, min_bin_pct=0.15)
    assigned2 = assign_bins(values, edges2)
    counts2 = [int(np.sum(assigned2 == index)) for index in range(len(edges2) - 1)]
    assert min(counts2) / values.size >= 0.15 - 1e-9
    assert edges2[0] == float("-inf")
    assert edges2[-1] == float("inf")


def test_chimerge_edges_min_bin_pct_merges_small_bins():
    """PREP-9: chimerge can leave a statistically-significant but tiny bin; min_bin_pct
    merges it into its chi2-nearest neighbor."""
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.uniform(0, 100, 950), rng.uniform(50, 51, 50)])
    target = (values > 50).astype(int)

    edges = chimerge_edges(values, target, max_bins=10, min_pvalue=0.001)
    assigned = assign_bins(values, edges)
    counts = [int(np.sum(assigned == index)) for index in range(len(edges) - 1)]
    assert min(counts) / values.size < 0.05  # unconstrained: the 50-row bin survives

    edges2 = chimerge_edges(values, target, max_bins=10, min_pvalue=0.001, min_bin_pct=0.05)
    assigned2 = assign_bins(values, edges2)
    counts2 = [int(np.sum(assigned2 == index)) for index in range(len(edges2) - 1)]
    assert min(counts2) / values.size >= 0.05 - 1e-9
    assert edges2[0] == float("-inf")
    assert edges2[-1] == float("inf")


def test_min_bin_pct_defaults_to_no_op_and_collapses_to_single_bin_when_too_high():
    values = np.arange(1, 13, dtype=float)
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])

    # Default (min_bin_pct=0.0) matches the pre-PREP-9 behavior exactly.
    assert equal_frequency_edges(values, 4).tolist() == equal_frequency_edges(values, 4, min_bin_pct=0.0).tolist()
    assert chimerge_edges(values, target, max_bins=4).tolist() == chimerge_edges(
        values, target, max_bins=4, min_bin_pct=0.0
    ).tolist()

    # An unreasonably high min_bin_pct collapses everything to a single bin, not an error.
    collapsed = equal_frequency_edges(values, 4, min_bin_pct=0.9)
    assert collapsed.tolist() == [float("-inf"), float("inf")]


def test_monotonic_edges_merge_adjacent_bad_rate_violations():
    values = np.arange(1, 13, dtype=float)
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1])
    edges = np.array([float("-inf"), 2.5, 4.5, 6.5, float("inf")])

    direction = monotonic_direction(values, target, edges, direction="auto")
    adjusted = monotonic_edges(values, target, edges, direction=direction)
    assigned = assign_bins(values, adjusted)
    bad_rates = [
        float(np.mean(target[assigned == index]))
        for index in range(len(adjusted) - 1)
    ]

    assert direction == "increasing"
    assert adjusted.tolist() == [float("-inf"), 2.5, 6.5, float("inf")]
    assert all(left <= right for left, right in zip(bad_rates, bad_rates[1:]))


def test_tree_edges_find_supervised_split_and_handle_constant_target():
    values = np.arange(10, dtype=float)
    target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    edges = tree_edges(values, target, max_bins=2, seed=7)
    constant = tree_edges(values, np.zeros_like(values), max_bins=3, seed=7)

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert len(edges) == 3
    assert 4.0 < edges[1] < 5.0
    assert constant.tolist() == [float("-inf"), float("inf")]



# -- FIN-3 #3: opt-in diagnostic for silent bin degradation ---------------------
def test_degraded_bin_diagnostic_flags_single_value_column():
    # A single-valued column collapses equal_frequency_edges to one bin -- the return
    # value is unchanged (still the [-inf, inf] fallback), but the diagnostic detects
    # that the requested bin count was not met so a caller can warn.
    single_value = np.full(500, 3.14, dtype=float)
    edges = equal_frequency_edges(single_value, 10)
    assert edges.tolist() == [float("-inf"), float("inf")]

    warning = degraded_bin_diagnostic(edges, 10, feature="score")
    assert warning is not None
    assert warning["kind"] == "degraded_binning"
    assert warning["requested_bin_count"] == 10
    assert warning["actual_bin_count"] == 1
    assert warning["feature"] == "score"


def test_degraded_bin_diagnostic_none_when_bins_met():
    # A well-distributed column reaches the requested bin count -- no degradation, so
    # the diagnostic is None (nothing to surface).
    spread = np.linspace(0.0, 1.0, 1000, dtype=float)
    edges = equal_frequency_edges(spread, 10)
    assert edges.size - 1 == 10
    assert degraded_bin_diagnostic(edges, 10) is None
