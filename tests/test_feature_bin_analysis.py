import numpy as np
import pytest

from marvis.feature.bin_analysis import feature_bin_analysis


def test_bin_analysis_cumulative_metrics_follow_high_risk_order_for_risk_down():
    values = np.arange(1_000, dtype=float)
    # Lower values are riskier, with non-identical bin bad rates.
    target = (values < 500).astype(float)

    result = feature_bin_analysis(
        values,
        target,
        feature="risk_down_feature",
        requested_bins=5,
    )

    assert result["direction"] == "risk_down"
    rows = result["rows"]
    by_risk = sorted(rows, key=lambda row: row["risk_rank"])
    assert [row["bad_rate"] for row in by_risk] == sorted(
        (row["bad_rate"] for row in rows),
        reverse=True,
    )
    assert by_risk[0]["cumulative_bad_rate"] == pytest.approx(by_risk[0]["bad_rate"])
    assert by_risk[-1]["cumulative_lift"] == pytest.approx(1.0)


def test_bin_analysis_keeps_interval_order_but_ranks_risk_up_from_high_values():
    values = np.arange(1_000, dtype=float)
    target = np.zeros(1_000, dtype=float)
    for bin_index in range(5):
        start = bin_index * 200
        target[start : start + ((bin_index + 1) * 20)] = 1.0

    result = feature_bin_analysis(
        values,
        target,
        feature="risk_up_feature",
        requested_bins=5,
    )

    assert result["direction"] == "risk_up"
    rows = result["rows"]
    assert [row["bin_index"] for row in rows] == [1, 2, 3, 4, 5]
    highest_value_bin = rows[-1]
    assert highest_value_bin["risk_rank"] == 1
    assert highest_value_bin["cumulative_bad_rate"] == pytest.approx(
        highest_value_bin["bad_rate"]
    )
