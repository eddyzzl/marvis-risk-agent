from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError


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


def impute_missing(
    series: pd.Series,
    *,
    strategy: str = "median",
    fill_value=None,
) -> tuple[pd.Series, object]:
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
) -> tuple[np.ndarray, dict]:
    arr = np.asarray(values, dtype=float)
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
    "impute_missing",
    "minmax_normalize",
    "zscore_standardize",
]
