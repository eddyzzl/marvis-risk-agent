from __future__ import annotations

import numpy as np

from marvis.feature.binning import assign_bins
from marvis.feature.contracts import Bin, BinningResult, WOEResult
from marvis.feature.errors import FeatureError


def compute_woe_iv(
    values: np.ndarray,
    target: np.ndarray,
    edges: np.ndarray,
    *,
    feature: str,
    smoothing: float = 0.5,
    na_as_bin: bool = True,
) -> BinningResult:
    """Compute WOE/IV with a single platform-wide convention.

    WOE_i = ln(good_i_dist / bad_i_dist), with Laplace smoothing.
    IV_i = (good_i_dist - bad_i_dist) * WOE_i.

    Higher WOE means the bin is more good-customer heavy.
    """
    arr = np.asarray(values, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if arr.shape != tgt.shape:
        raise FeatureError("values and target must have the same shape")
    valid_target = np.isfinite(tgt)
    target_values = tgt[valid_target]
    if target_values.size == 0 or not np.all(np.isin(target_values, [0, 1])):
        raise FeatureError("target must be binary 0/1")
    unique_target = set(np.unique(target_values).astype(int).tolist())
    if len(unique_target) < 2:
        raise FeatureError("target must contain both good and bad classes")

    bin_index = assign_bins(arr[valid_target], np.asarray(edges, dtype=float))
    target_int = target_values.astype(int)
    if na_as_bin:
        groups = list(range(len(edges) - 1))
        if np.any(bin_index == -1):
            groups.append(-1)
        denominator_mask = np.ones_like(target_int, dtype=bool)
    else:
        groups = list(range(len(edges) - 1))
        denominator_mask = bin_index != -1

    denominator_target = target_int[denominator_mask]
    total_bad = int(np.sum(denominator_target == 1))
    total_good = int(np.sum(denominator_target == 0))
    if total_bad == 0 or total_good == 0:
        raise FeatureError("target must contain both good and bad classes")

    bins = []
    total_iv = 0.0
    group_count = len(groups)
    for group in groups:
        mask = bin_index == group
        count = int(np.sum(mask))
        bad = int(np.sum(target_int[mask] == 1))
        good = count - bad
        bad_dist = (bad + smoothing) / (total_bad + smoothing * group_count)
        good_dist = (good + smoothing) / (total_good + smoothing * group_count)
        woe = float(np.log(good_dist / bad_dist))
        iv_contribution = float((good_dist - bad_dist) * woe)
        total_iv += iv_contribution
        bins.append(
            Bin(
                index=group,
                lower=_lower_edge(edges, group),
                upper=_upper_edge(edges, group),
                count=count,
                bad_count=bad,
                good_count=good,
                bad_rate=(bad / count if count else 0.0),
                woe=woe,
                iv_contribution=iv_contribution,
            )
        )

    na_bin = next((item for item in bins if item.index == -1), None)
    real_bins = tuple(item for item in bins if item.index != -1)
    return BinningResult(
        feature=feature,
        method="given",
        bins=real_bins,
        edges=tuple(float(value) for value in edges),
        total_iv=round(float(total_iv), 6),
        monotonic=_is_monotonic([item.bad_rate for item in real_bins]),
        na_bin=na_bin,
    )


def woe_result_from_binning(binning: BinningResult) -> WOEResult:
    return WOEResult(
        feature=binning.feature,
        edges=binning.edges,
        woe_by_bin=tuple(item.woe for item in binning.bins),
        na_woe=binning.na_bin.woe if binning.na_bin is not None else None,
    )


def _lower_edge(edges: np.ndarray, group: int) -> float:
    if group == -1:
        return float("nan")
    return float(edges[group])


def _upper_edge(edges: np.ndarray, group: int) -> float:
    if group == -1:
        return float("nan")
    return float(edges[group + 1])


def _is_monotonic(values: list[float]) -> bool:
    if len(values) <= 2:
        return True
    increasing = all(left <= right for left, right in zip(values, values[1:]))
    decreasing = all(left >= right for left, right in zip(values, values[1:]))
    return increasing or decreasing


__all__ = ["compute_woe_iv", "woe_result_from_binning"]
