import math
import numpy as np
import pandas as pd
import pytest

from riskmodel_checker.validation.binning import (
    equal_frequency_bin_edges,
    assign_bins,
    bin_table,
    compute_ks,
    compute_psi,
)


def test_equal_frequency_bin_edges_uses_quantiles():
    scores = np.arange(0, 100)
    edges = equal_frequency_bin_edges(scores, bin_count=10)
    assert len(edges) == 11
    assert edges[0] == 0
    assert edges[-1] == 99
    assert edges[5] == pytest.approx(49.5, abs=1.0)


def test_equal_frequency_bin_edges_dedupes_when_many_ties():
    scores = np.array([0.0] * 50 + [1.0] * 50)
    edges = equal_frequency_bin_edges(scores, bin_count=10)
    assert len(set(edges)) < 11


def test_assign_bins_maps_scores_to_indices():
    edges = np.array([0.0, 0.2, 0.5, 1.0])
    scores = np.array([0.1, 0.3, 0.7, 0.9, 0.0, 1.0])
    indices = assign_bins(scores, edges)
    assert list(indices) == [1, 2, 3, 3, 1, 3]


def test_compute_ks_known_values():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ks = compute_ks(scores, labels)
    assert ks == pytest.approx(1.0)


def test_compute_ks_zero_when_no_signal():
    scores = np.array([0.5] * 10)
    labels = np.array([0, 1] * 5)
    assert compute_ks(scores, labels) == pytest.approx(0.0)


def test_compute_psi_zero_when_distributions_match():
    expected = np.array([0.1] * 10)
    actual = np.array([0.1] * 10)
    assert compute_psi(expected, actual) == pytest.approx(0.0)


def test_compute_psi_handles_zero_buckets():
    expected = np.array([0.5, 0.5, 0.0])
    actual = np.array([0.5, 0.3, 0.2])
    psi = compute_psi(expected, actual)
    assert psi > 0
    assert math.isfinite(psi)


def test_bin_table_basic_shape():
    df = pd.DataFrame({
        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "y":     [0,   0,   0,   0,   1,   1,   1,   1,   1,   0],
    })
    edges = np.array([0.0, 0.5, 1.0])
    rows = bin_table(df, edges, score_col="score", target_col="y")
    assert len(rows) == 2
    assert rows[0].bin_index == 1
    assert rows[0].sample_count == 5
    assert rows[0].bad_count == 1
    assert rows[0].bad_rate == pytest.approx(0.2)
    assert rows[1].sample_count == 5
    assert rows[1].bad_count == 4
