import numpy as np
import pytest

from marvis.feature.binning import (
    assign_bins,
    equal_frequency_edges,
    equal_width_edges,
    manual_edges,
)
from marvis.feature.errors import BinningError


def test_equal_frequency_edges_filter_nan_inf_and_dedupe_quantiles():
    edges = equal_frequency_edges(np.array([1, 1, 1, 2, 3, np.nan, np.inf]), 4)

    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert np.all(np.diff(edges) > 0)
    assert len(edges) <= 5


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
