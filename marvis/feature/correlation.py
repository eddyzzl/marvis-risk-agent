from __future__ import annotations

import numpy as np
import pandas as pd

from marvis.feature.contracts import CorrelationReport


def correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
    *,
    method: str = "pearson",
) -> np.ndarray:
    matrix = np.eye(len(features), dtype=float)
    for i, left in enumerate(features):
        for j, right in enumerate(features):
            if i == j:
                continue
            if method == "spearman":
                corr = safe_correlation(
                    df[left].rank().to_numpy(dtype=float),
                    df[right].rank().to_numpy(dtype=float),
                )
            else:
                corr = safe_correlation(
                    df[left].to_numpy(dtype=float),
                    df[right].to_numpy(dtype=float),
                )
            matrix[i, j] = corr
    return matrix


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_arr, y_arr = _finite_pairs(x, y)
    if x_arr.size < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


# VIF of perfectly-collinear features is mathematically infinite; cap at a large finite
# sentinel so the value stays JSON-safe (Starlette serializes responses with
# allow_nan=False, and a browser JSON.parse rejects `Infinity`). VIF > ~10 already means
# "severe collinearity", so the cap only flattens the already-unusable extreme.
_VIF_CAP = 1e9


def vif(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    result = {feature: 0.0 for feature in features}
    # Exclude entirely-NaN columns from the design matrix; otherwise a single all-NaN
    # feature empties the listwise-dropped frame and would zero EVERY feature's VIF.
    usable = [feature for feature in features if df[feature].notna().any()]
    clean = df[usable].dropna()
    if clean.empty or len(usable) < 2:
        return result
    for feature in usable:
        others = [item for item in usable if item != feature]
        r2 = _ols_r2(clean[others].to_numpy(dtype=float), clean[feature].to_numpy(dtype=float))
        result[feature] = _VIF_CAP if r2 >= 1 else min(_VIF_CAP, float(1.0 / (1.0 - r2)))
    return result


def find_collinear_pairs(
    matrix: np.ndarray,
    features: list[str],
    *,
    threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    pairs = []
    for i, left in enumerate(features):
        for j in range(i + 1, len(features)):
            corr = float(matrix[i, j])
            if abs(corr) >= threshold:
                pairs.append((left, features[j], corr))
    return pairs


def correlation_report(
    df: pd.DataFrame,
    features: list[str],
    *,
    method: str = "pearson",
    threshold: float = 0.8,
) -> CorrelationReport:
    matrix = correlation_matrix(df, features, method=method)
    return CorrelationReport(
        features=tuple(features),
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        collinear_pairs=tuple(find_collinear_pairs(matrix, features, threshold=threshold)),
        vif=vif(df, features),
    )


def _finite_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or np.std(y) == 0:
        return 0.0
    design = np.column_stack([np.ones(x.shape[0]), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ coefficients
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    if ss_total == 0:
        return 0.0
    ss_resid = float(np.sum((y - predicted) ** 2))
    return max(0.0, min(1.0, 1 - ss_resid / ss_total))


__all__ = [
    "correlation_matrix",
    "correlation_report",
    "find_collinear_pairs",
    "safe_correlation",
    "vif",
]
