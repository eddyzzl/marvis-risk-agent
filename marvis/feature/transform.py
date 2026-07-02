from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError


# Sentinel/special-value candidates commonly used by credit-bureau and third-party
# score feeds to mean "no hit / not covered / over limit" rather than a genuine
# numeric observation (PREP-4). Checked in addition to whatever the caller passes.
_DEFAULT_SENTINEL_CANDIDATES: tuple[float, ...] = (-999.0, -99.0, -1.0, 9999.0, 99999.0)


def detect_sentinel_values(
    values: np.ndarray,
    *,
    candidates: tuple[float, ...] = _DEFAULT_SENTINEL_CANDIDATES,
    min_share: float = 0.01,
    gap_factor: float = 3.0,
) -> list[tuple[float, float]]:
    """Deterministically flag suspected sentinel/special values in ``values``.

    A candidate value is flagged when *all* hold on the finite observations:

    - it is actually present in the column;
    - its share of finite observations is >= ``min_share`` (default 1%) -- an
      isolated peak, not routine noise;
    - it sits at an extreme of the observed range (the min or the max) -- sentinels
      like -999/9999 are placed outside the plausible business range by design;
    - it is separated from the rest of the distribution by a gap at least
      ``gap_factor`` times the median gap between other consecutive distinct
      values -- i.e. the peak is genuinely isolated, not just the tail of a
      continuous distribution.

    Returns ``[(sentinel_value, share), ...]`` sorted by descending share.
    Deterministic; no randomness.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    unique_values, counts = np.unique(finite, return_counts=True)
    if unique_values.size < 2:
        return []
    shares = counts / finite.size
    min_value = float(unique_values[0])
    max_value = float(unique_values[-1])
    gaps = np.diff(unique_values)
    other_gaps = gaps[gaps > 0]
    typical_gap = float(np.median(other_gaps)) if other_gaps.size else 0.0

    flagged: list[tuple[float, float]] = []
    for candidate in candidates:
        index = np.searchsorted(unique_values, float(candidate))
        if index >= unique_values.size or unique_values[index] != float(candidate):
            continue
        share = float(shares[index])
        if share < min_share:
            continue
        is_extreme = unique_values[index] in (min_value, max_value)
        if not is_extreme:
            continue
        if index == 0:
            neighbor_gap = float(gaps[0]) if gaps.size else 0.0
        else:
            neighbor_gap = float(gaps[index - 1])
        is_isolated = typical_gap <= 0 or neighbor_gap >= gap_factor * typical_gap
        if not is_isolated:
            continue
        flagged.append((float(candidate), share))
    flagged.sort(key=lambda item: item[1], reverse=True)
    return flagged


def minmax_normalize(
    values: np.ndarray,
    *,
    feature_range: tuple[float, float] = (0, 1),
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(values, dtype=float)
    lower, upper = _validate_feature_range(feature_range)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, np.nan, dtype=float), {
            "min": float("nan"),
            "max": float("nan"),
            "feature_range": (lower, upper),
        }
    params = {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "feature_range": (lower, upper),
    }
    return apply_scaler(arr, params, kind="minmax"), params


def zscore_standardize(values: np.ndarray) -> tuple[np.ndarray, dict]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, np.nan, dtype=float), {"mean": float("nan"), "std": float("nan")}
    params = {"mean": float(np.mean(finite)), "std": float(np.std(finite))}
    return apply_scaler(arr, params, kind="zscore"), params


def apply_scaler(values: np.ndarray, params: dict, *, kind: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if kind == "minmax":
        lower, upper = _validate_feature_range(tuple(params.get("feature_range", (0, 1))))
        min_value = float(params["min"])
        max_value = float(params["max"])
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            return out
        if max_value == min_value:
            out[mask] = lower
        else:
            scaled = (arr[mask] - min_value) / (max_value - min_value)
            out[mask] = scaled * (upper - lower) + lower
        return out
    if kind == "zscore":
        mean = float(params["mean"])
        std = float(params["std"])
        if not np.isfinite(mean) or not np.isfinite(std):
            return out
        out[mask] = 0.0 if std == 0 else (arr[mask] - mean) / std
        return out
    raise FeatureError("kind must be 'minmax' or 'zscore'")


def mask_sentinel_values(series: pd.Series, sentinel_values: list[float] | None) -> pd.Series:
    """Treat ``sentinel_values`` (PREP-4) as missing: return ``series`` with those
    values replaced by NaN, leaving already-missing rows untouched. A no-op when
    ``sentinel_values`` is empty/None."""
    if not sentinel_values:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.isin([float(value) for value in sentinel_values])
    return series.mask(mask)


def impute_missing(
    series: pd.Series,
    *,
    strategy: str = "median",
    fill_value=None,
    sentinel_values: list[float] | None = None,
) -> tuple[pd.Series, object]:
    series = mask_sentinel_values(series, sentinel_values)
    if strategy == "constant":
        if fill_value is None:
            raise FeatureError("fill_value is required for constant imputation")
        value = fill_value
    elif strategy == "mean":
        value = float(series.mean())
    elif strategy == "median":
        value = float(series.median())
    elif strategy == "mode":
        modes = series.dropna().mode()
        if modes.empty and fill_value is None:
            raise FeatureError("fill_value is required when mode is empty")
        value = modes.iloc[0] if not modes.empty else fill_value
    else:
        raise FeatureError("strategy must be mean, median, mode, or constant")
    return series.fillna(value), value


def cap_outliers(
    values: np.ndarray,
    *,
    method: str = "iqr",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    sentinel_values: list[float] | None = None,
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(values, dtype=float)
    if sentinel_values:
        arr = arr.copy()
        arr[np.isin(arr, [float(value) for value in sentinel_values])] = np.nan
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr.copy(), {"lower": float("nan"), "upper": float("nan"), "method": method}
    if method == "iqr":
        q1 = float(np.quantile(finite, 0.25))
        q3 = float(np.quantile(finite, 0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    elif method == "quantile":
        if not 0 <= lower_q <= upper_q <= 1:
            raise FeatureError("quantiles must satisfy 0 <= lower_q <= upper_q <= 1")
        lower = float(np.quantile(finite, lower_q))
        upper = float(np.quantile(finite, upper_q))
    else:
        raise FeatureError("method must be 'iqr' or 'quantile'")
    out = arr.copy()
    mask = np.isfinite(out)
    out[mask] = np.clip(out[mask], lower, upper)
    return out, {"lower": lower, "upper": upper, "method": method}


def _validate_feature_range(feature_range: tuple[float, float]) -> tuple[float, float]:
    if len(feature_range) != 2:
        raise FeatureError("feature_range must contain lower and upper bounds")
    lower = float(feature_range[0])
    upper = float(feature_range[1])
    if upper <= lower:
        raise FeatureError("feature_range upper bound must be greater than lower bound")
    return lower, upper


__all__ = [
    "apply_scaler",
    "cap_outliers",
    "detect_sentinel_values",
    "impute_missing",
    "mask_sentinel_values",
    "minmax_normalize",
    "zscore_standardize",
]
