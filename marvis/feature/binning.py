from __future__ import annotations

import numpy as np

from marvis.feature.errors import BinningError


def equal_frequency_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    _validate_bin_count(bin_count)
    arr = _finite_values(values)
    if arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0, 1, int(bin_count) + 1)
    edges = np.unique(np.quantile(arr, quantiles))
    if edges.size <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def equal_width_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    _validate_bin_count(bin_count)
    arr = _finite_values(values)
    if arr.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    lower = float(np.min(arr))
    upper = float(np.max(arr))
    if lower == upper:
        return np.array([-np.inf, np.inf], dtype=float)
    edges = np.linspace(lower, upper, int(bin_count) + 1, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def manual_edges(breakpoints: list[float]) -> np.ndarray:
    if not breakpoints:
        raise BinningError("manual binning requires at least one breakpoint")
    points = _finite_values(np.asarray(breakpoints, dtype=float))
    if points.size == 0:
        raise BinningError("manual binning breakpoints must be finite")
    return np.array([-np.inf, *np.unique(points), np.inf], dtype=float)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise BinningError("edges must be a one-dimensional array with at least 2 values")
    if np.any(np.diff(edges) <= 0):
        raise BinningError("edges must be strictly increasing")
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, -1, dtype=int)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out
    assigned = np.searchsorted(edges[1:-1], arr[mask], side="right")
    out[mask] = np.clip(assigned, 0, edges.size - 2)
    return out


def _finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _validate_bin_count(bin_count: int) -> None:
    if int(bin_count) < 1:
        raise BinningError("bin_count must be at least 1")


__all__ = [
    "assign_bins",
    "equal_frequency_edges",
    "equal_width_edges",
    "manual_edges",
]
