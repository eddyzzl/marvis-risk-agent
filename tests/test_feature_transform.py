import numpy as np
import pandas as pd
import pytest

from marvis.feature.errors import FeatureError
from marvis.feature.transform import (
    apply_scaler,
    cap_outliers,
    impute_missing,
    minmax_normalize,
    zscore_standardize,
)


def test_minmax_normalize_returns_reusable_params_and_preserves_missing():
    values = np.array([2.0, 4.0, 6.0, np.nan])

    scaled, params = minmax_normalize(values, feature_range=(-1, 1))
    applied = apply_scaler(np.array([4.0, 8.0, np.nan]), params, kind="minmax")

    assert scaled.tolist()[:3] == [-1.0, 0.0, 1.0]
    assert np.isnan(scaled[3])
    assert params == {"min": 2.0, "max": 6.0, "feature_range": (-1.0, 1.0)}
    assert applied.tolist()[:2] == [0.0, 2.0]
    assert np.isnan(applied[2])


def test_zscore_standardize_zero_variance_returns_zero_for_finite_values():
    values = np.array([3.0, 3.0, np.nan])

    scaled, params = zscore_standardize(values)
    applied = apply_scaler(np.array([3.0, 4.0, np.nan]), params, kind="zscore")

    assert scaled.tolist()[:2] == [0.0, 0.0]
    assert np.isnan(scaled[2])
    assert params == {"mean": 3.0, "std": 0.0}
    assert applied.tolist()[:2] == [0.0, 0.0]
    assert np.isnan(applied[2])


def test_impute_missing_returns_fill_value_for_reuse():
    series = pd.Series([1.0, np.nan, 3.0], name="amount")

    filled, value = impute_missing(series, strategy="median")

    assert value == 2.0
    assert filled.name == "amount"
    assert filled.tolist() == [1.0, 2.0, 3.0]


def test_cap_outliers_quantile_returns_clip_bounds():
    values = np.array([1.0, 2.0, 100.0, np.nan])

    capped, params = cap_outliers(values, method="quantile", lower_q=0.0, upper_q=0.5)

    assert capped.tolist()[:3] == [1.0, 2.0, 2.0]
    assert np.isnan(capped[3])
    assert params == {"lower": 1.0, "upper": 2.0, "method": "quantile"}


def test_transform_rejects_invalid_options():
    with pytest.raises(FeatureError, match="kind"):
        apply_scaler(np.array([1.0]), {}, kind="unknown")

    with pytest.raises(FeatureError, match="feature_range"):
        minmax_normalize(np.array([1.0]), feature_range=(1, 1))

    with pytest.raises(FeatureError, match="strategy"):
        impute_missing(pd.Series([1.0]), strategy="unknown")

    with pytest.raises(FeatureError, match="fill_value"):
        impute_missing(pd.Series([np.nan]), strategy="constant")

    with pytest.raises(FeatureError, match="quantiles"):
        cap_outliers(np.array([1.0]), method="quantile", lower_q=0.9, upper_q=0.1)
