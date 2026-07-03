import numpy as np
import pandas as pd
import pytest

from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import compute_psi as feature_compute_psi
from marvis.feature.metrics import feature_ks
from marvis.validation.binning import (
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
)
from marvis.validation.effectiveness import _should_reverse_eval_bins


def test_validation_equal_frequency_edges_reuse_feature_core():
    scores = np.array([1.0, 2.0, 2.0, 3.0, np.nan, np.inf])

    assert equal_frequency_bin_edges(scores, 4).tolist() == equal_frequency_edges(scores, 4).tolist()


def test_validation_ks_and_psi_reuse_feature_core():
    scores = np.array([0.1, 0.2, 0.8, 0.9, np.nan])
    labels = np.array([0, 0, 1, 1, 1])
    expected = np.array([0.5, 0.5, 0.0])
    actual = np.array([0.2, 0.3, 0.5])

    assert compute_ks(scores, labels) == pytest.approx(feature_ks(scores, labels))
    assert compute_psi(expected, actual) == pytest.approx(feature_compute_psi(expected, actual))


def test_validation_auto_sort_uses_safe_correlation_zero_as_no_reverse():
    sample = pd.DataFrame({
        "sample_score": [0.1, 0.2, np.nan, 0.4],
        "y": [0, 1, 0, 1],
    })
    zero_corr = pd.DataFrame({
        "sample_score": [-1.0, -1.0, 1.0, 1.0],
        "y": [0, 1, 0, 1],
    })

    assert _should_reverse_eval_bins(sample, score_col="sample_score", target_col="y") is False
    assert _should_reverse_eval_bins(zero_corr, score_col="sample_score", target_col="y") is False
