"""Optional per-feature bin-detail analysis for the feature-analysis workflow."""

from __future__ import annotations

import math

import numpy as np

from marvis.feature.binning import degraded_bin_diagnostic, equal_frequency_edges
from marvis.feature.iv import compute_woe_iv


def feature_bin_analysis(
    values: np.ndarray,
    target: np.ndarray,
    *,
    feature: str,
    requested_bins: int,
) -> dict:
    """Return a get_eval_table-style equal-frequency bin detail payload.

    The platform WOE/IV kernel remains authoritative. Lift, cumulative lift and
    KS are derived from the same bin counts so the Excel report never mixes
    independently calculated statistics.
    """
    arr = np.asarray(values, dtype=float)
    tgt = np.asarray(target, dtype=float)
    edges = equal_frequency_edges(arr, int(requested_bins))
    result = compute_woe_iv(arr, tgt, edges, feature=str(feature))
    all_bins = [*result.bins]
    if result.na_bin is not None:
        all_bins.append(result.na_bin)

    total_bad = sum(int(item.bad_count) for item in all_bins)
    total_good = sum(int(item.good_count) for item in all_bins)
    total = total_bad + total_good
    overall_bad_rate = total_bad / total if total else 0.0
    # Cumulative risk statistics are business "top-risk first" quantities, not
    # feature-value-order quantities. Preserve the original interval order for
    # display, but calculate cumulative bad rate/lift/KS in descending observed
    # bad-rate order and expose the corresponding rank explicitly.
    risk_order = sorted(
        range(len(all_bins)),
        key=lambda index: (-float(all_bins[index].bad_rate), index),
    )
    cumulative_bad = 0
    cumulative_good = 0
    cumulative_by_display_index: dict[int, dict[str, float | int]] = {}
    for risk_rank, display_index in enumerate(risk_order, start=1):
        item = all_bins[display_index]
        cumulative_bad += int(item.bad_count)
        cumulative_good += int(item.good_count)
        cumulative_total = cumulative_bad + cumulative_good
        cumulative_bad_rate = cumulative_bad / cumulative_total if cumulative_total else 0.0
        bad_cdf = cumulative_bad / total_bad if total_bad else 0.0
        good_cdf = cumulative_good / total_good if total_good else 0.0
        cumulative_by_display_index[display_index] = {
            "risk_rank": risk_rank,
            "cumulative_bad_rate": float(cumulative_bad_rate),
            "cumulative_lift": (
                float(cumulative_bad_rate / overall_bad_rate) if overall_bad_rate else 0.0
            ),
            "ks": float(abs(bad_cdf - good_cdf)),
        }

    rows: list[dict] = []
    for display_offset, item in enumerate(all_bins):
        cumulative = cumulative_by_display_index[display_offset]
        rows.append({
            "bin_index": display_offset + 1,
            "risk_rank": cumulative["risk_rank"],
            "interval": _interval_text(item.lower, item.upper, missing=item.index == -1),
            "count": int(item.count),
            "bad_count": int(item.bad_count),
            "good_count": int(item.good_count),
            "bad_rate": float(item.bad_rate),
            "cumulative_bad_rate": cumulative["cumulative_bad_rate"],
            "lift": float(item.bad_rate / overall_bad_rate) if overall_bad_rate else 0.0,
            "cumulative_lift": cumulative["cumulative_lift"],
            "ks": cumulative["ks"],
            "woe": float(item.woe),
            "iv_contribution": float(item.iv_contribution),
        })

    diagnostic = degraded_bin_diagnostic(edges, int(requested_bins), feature=str(feature))
    return {
        "feature": str(feature),
        "requested_bins": int(requested_bins),
        "actual_bins": len(result.bins),
        "direction": _risk_direction(arr, tgt),
        "total_iv": float(result.total_iv),
        "monotonic": bool(result.monotonic),
        "degraded_reason": str(diagnostic.get("message") or "") if diagnostic else "",
        "rows": rows,
    }


def _risk_direction(values: np.ndarray, target: np.ndarray) -> str:
    mask = np.isfinite(values) & np.isfinite(target)
    if int(mask.sum()) < 2:
        return "unknown"
    x = values[mask]
    y = target[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return "unknown"
    correlation = float(np.corrcoef(x, y)[0, 1])
    if not math.isfinite(correlation):
        return "unknown"
    return "risk_up" if correlation >= 0 else "risk_down"


def _interval_text(lower: float, upper: float, *, missing: bool = False) -> str:
    if missing:
        return "缺失值"
    return f"({_number_text(lower)}, {_number_text(upper)}]"


def _number_text(value: float) -> str:
    if value == float("-inf"):
        return "-inf"
    if value == float("inf"):
        return "inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


__all__ = ["feature_bin_analysis"]
