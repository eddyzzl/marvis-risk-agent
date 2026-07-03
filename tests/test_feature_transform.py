import numpy as np
import pandas as pd
import pytest

from marvis.feature.errors import FeatureError
from marvis.feature.transform import (
    apply_scaler,
    cap_outliers,
    detect_sentinel_values,
    impute_missing,
    mask_sentinel_values,
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


def test_detect_sentinel_values_flags_isolated_extreme_peak():
    # 200 normal values in [1, 100], plus a -999 "no hit" sentinel on 5% of rows,
    # isolated from the real distribution by a huge gap.
    rng = np.concatenate([np.linspace(1.0, 100.0, 190), np.full(10, -999.0)])

    hits = detect_sentinel_values(rng)

    assert hits == [(-999.0, pytest.approx(10 / 200))]


def test_detect_sentinel_values_ignores_low_share_and_non_extreme_and_close_values():
    # -1 sits below 1% share -> not flagged.
    low_share = np.concatenate([np.linspace(1.0, 100.0, 199), np.full(1, -1.0)])
    assert detect_sentinel_values(low_share) == []

    # 9999 present but not at an extreme (larger real values exist above it) -> not flagged.
    not_extreme = np.concatenate([np.linspace(1.0, 20000.0, 190), np.full(10, 9999.0)])
    assert detect_sentinel_values(not_extreme) == []

    # -1 sits right next to the rest of a [-1, 100] uniform-ish distribution -> not isolated.
    close = np.concatenate([np.linspace(-1.0, 100.0, 190), np.full(10, -1.0)])
    assert detect_sentinel_values(close) == []


def test_mask_sentinel_values_treats_configured_values_as_missing():
    series = pd.Series([1.0, -999.0, 3.0, np.nan])

    masked = mask_sentinel_values(series, [-999.0])

    assert masked.tolist()[:1] == [1.0]
    assert np.isnan(masked.iloc[1])
    assert masked.tolist()[2] == 3.0
    assert np.isnan(masked.iloc[3])
    # No-op when sentinel_values is empty/None.
    assert mask_sentinel_values(series, None) is series
    assert mask_sentinel_values(series, []) is series


def test_impute_missing_and_cap_outliers_treat_sentinel_values_as_missing():
    series = pd.Series([1.0, 2.0, 3.0, -999.0])
    filled, value = impute_missing(series, strategy="median", sentinel_values=[-999.0])
    assert value == 2.0  # median of [1, 2, 3], -999 excluded from the fit
    assert filled.iloc[3] == 2.0  # sentinel row masked to NaN then filled like any other missing row

    values = np.array([1.0, 2.0, 100.0, -999.0])
    capped, params = cap_outliers(
        values, method="quantile", lower_q=0.0, upper_q=0.5, sentinel_values=[-999.0]
    )
    assert params == {"lower": 1.0, "upper": 2.0, "method": "quantile"}
    assert np.isnan(capped[3])
