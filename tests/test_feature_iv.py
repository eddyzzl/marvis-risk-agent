import math

import numpy as np
import pytest

from marvis.feature.errors import FeatureError
from marvis.feature.iv import _smoothed_woe_iv, compute_woe_iv, woe_result_from_binning


def _inline_woe_iv(bad, good, total_bad, total_good, n_groups, smoothing):
    """The Laplace-smoothed WOE/IV formula as it was inlined at every call site
    before T2-5, kept here as the ground truth the shared kernel must reproduce."""
    bad_dist = (bad + smoothing) / (total_bad + smoothing * n_groups)
    good_dist = (good + smoothing) / (total_good + smoothing * n_groups)
    woe = float(np.log(good_dist / bad_dist))
    return woe, float((good_dist - bad_dist) * woe)


def test_smoothed_woe_iv_kernel_matches_inline_formula():
    """T2-5 two-sided consistency: the shared _smoothed_woe_iv kernel used by both
    compute_woe_iv (numeric bins) and categorical_woe_encode (raw categories) must
    reproduce the previously-inlined formula bit-for-bit for every input shape."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        total_bad = int(rng.integers(1, 5000))
        total_good = int(rng.integers(1, 5000))
        n_groups = int(rng.integers(1, 20))
        bad = int(rng.integers(0, total_bad + 1))
        good = int(rng.integers(0, total_good + 1))
        smoothing = float(rng.choice([0.5, 1.0, 0.1, 2.0]))
        assert _smoothed_woe_iv(
            bad, good, total_bad, total_good, n_groups, smoothing=smoothing
        ) == _inline_woe_iv(bad, good, total_bad, total_good, n_groups, smoothing)


def test_compute_woe_iv_matches_known_smoothed_distribution():
    result = compute_woe_iv(
        np.array([1, 2, 3, 4], dtype=float),
        np.array([0, 0, 1, 1]),
        np.array([float("-inf"), 2.5, float("inf")]),
        feature="score",
        smoothing=0.5,
    )

    assert len(result.bins) == 2
    assert result.bins[0].good_count == 2
    assert result.bins[0].bad_count == 0
    assert result.bins[0].woe == pytest.approx(math.log(5))
    assert result.bins[1].woe == pytest.approx(-math.log(5))
    assert result.total_iv == pytest.approx(2.145917, abs=1e-6)
    assert result.monotonic is True


def test_compute_woe_iv_tracks_missing_bin_and_woe_result():
    result = compute_woe_iv(
        np.array([np.nan, 1, 2], dtype=float),
        np.array([1, 0, 1]),
        np.array([float("-inf"), 1.5, float("inf")]),
        feature="age",
    )
    woe = woe_result_from_binning(result)

    assert result.na_bin is not None
    assert result.na_bin.index == -1
    assert result.na_bin.count == 1
    assert woe.feature == "age"
    assert woe.woe_by_bin == tuple(item.woe for item in result.bins)
    assert woe.na_woe == result.na_bin.woe


def test_compute_woe_iv_rejects_invalid_targets():
    with pytest.raises(FeatureError):
        compute_woe_iv(
            np.array([1, 2, 3]),
            np.array([0, 2, 1]),
            np.array([float("-inf"), float("inf")]),
            feature="x",
        )
    with pytest.raises(FeatureError):
        compute_woe_iv(
            np.array([1, 2, 3]),
            np.array([0.0, 0.5, 1.0]),
            np.array([float("-inf"), float("inf")]),
            feature="x",
        )
    with pytest.raises(FeatureError):
        compute_woe_iv(
            np.array([1, 2, 3]),
            np.array([0, 0, 0]),
            np.array([float("-inf"), float("inf")]),
            feature="x",
        )
