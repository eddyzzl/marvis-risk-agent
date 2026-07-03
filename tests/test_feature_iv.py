import math

import numpy as np
import pytest

from marvis.feature.errors import FeatureError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning


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
