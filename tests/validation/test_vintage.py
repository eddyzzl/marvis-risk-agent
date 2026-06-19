import json

import pandas as pd
import pytest

from marvis.validation.vintage import (
    DEFAULT_OVERDUE_BUCKETS,
    compute_roll_rate,
    compute_vintage_curve,
    vintage_curve_wide,
    vintage_summary_payload,
)


def test_compute_vintage_curve_parses_cohort_and_sorts_mob_as_integer():
    frame = pd.DataFrame({
        "cohort": ["202501", "2025-01", pd.Timestamp("2025-01-20"), "202502", "202502", "202502"],
        "mob": ["0", "0", "1", "10", "2", None],
        "bad": [1, 0, 1, 1, 0, 1],
        "balance": [100.0, 900.0, 100.0, 50.0, 50.0, 999.0],
    })

    points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
    )

    assert [(point.cohort, point.mob) for point in points] == [
        ("2025-01", 0),
        ("2025-01", 1),
        ("2025-02", 2),
        ("2025-02", 10),
    ]
    jan_mob0 = points[0]
    assert jan_mob0.sample_count == 2
    assert jan_mob0.bad_count == 1
    assert jan_mob0.bad_rate == pytest.approx(0.5)
    assert all(not hasattr(point, "customer_id") for point in points)


def test_vintage_count_and_balance_denominators_are_distinct_and_monotonic():
    frame = pd.DataFrame({
        "cohort": ["202501", "202501", "202501"],
        "mob": [0, 0, 1],
        "bad": [1, 0, 1],
        "balance": [100.0, 900.0, 100.0],
    })

    count_points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
    )
    balance_points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
        balance_col="balance",
        denominator="balance",
    )

    assert count_points[0].bad_rate == pytest.approx(0.5)
    assert balance_points[0].bad_rate == pytest.approx(0.1)
    assert balance_points[0].balance_sum == pytest.approx(1000.0)
    for points in (count_points, balance_points):
        rates = [point.cum_bad_rate for point in points]
        assert rates == sorted(rates)


def test_vintage_curve_wide_aligns_mob_axis_and_preserves_missing():
    frame = pd.DataFrame({
        "cohort": ["202501", "202501", "202502", "202502"],
        "mob": [0, 1, 1, 3],
        "bad": [0, 1, 1, 1],
    })
    points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")

    wide = vintage_curve_wide(points)
    bad_rate = vintage_curve_wide(points, metric="bad_rate")

    assert wide["2025-01"] == [0.0, 0.5, None]
    assert wide["2025-02"] == [None, 1.0, 1.0]
    assert bad_rate["2025-02"] == [None, 1.0, 1.0]
    with pytest.raises(ValueError, match="metric"):
        vintage_curve_wide(points, metric="unknown")


def test_compute_roll_rate_returns_complete_normalized_matrix():
    frame = pd.DataFrame({
        "from": ["current", "current", "1-30"],
        "to": ["current", "1-30", "31-60"],
    })
    buckets = ("current", "1-30", "31-60")

    points = compute_roll_rate(frame, from_bucket_col="from", to_bucket_col="to", buckets=buckets)
    by_pair = {(point.from_bucket, point.to_bucket): point for point in points}

    assert len(points) == 9
    assert by_pair[("current", "current")].count == 1
    assert by_pair[("current", "current")].rate == pytest.approx(0.5)
    assert by_pair[("current", "1-30")].rate == pytest.approx(0.5)
    assert by_pair[("31-60", "current")].rate == 0.0
    for from_bucket in buckets:
        assert sum(point.rate for point in points if point.from_bucket == from_bucket) == pytest.approx(
            1.0 if from_bucket in {"current", "1-30"} else 0.0
        )


def test_roll_rate_rejects_unknown_bucket_and_empty_data_returns_zero_matrix():
    empty = pd.DataFrame({"from": [], "to": []})
    points = compute_roll_rate(empty, from_bucket_col="from", to_bucket_col="to")

    assert len(points) == len(DEFAULT_OVERDUE_BUCKETS) ** 2
    assert all(point.count == 0 and point.rate == 0.0 for point in points)

    with pytest.raises(ValueError, match="unknown bucket"):
        compute_roll_rate(
            pd.DataFrame({"from": ["current"], "to": ["bad"]}),
            from_bucket_col="from",
            to_bucket_col="to",
        )


def test_vintage_summary_payload_is_json_serializable():
    vintage = compute_vintage_curve(
        pd.DataFrame({"cohort": ["202501"], "mob": [0], "bad": [0]}),
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
    )
    roll = compute_roll_rate(
        pd.DataFrame({"from": ["current"], "to": ["current"]}),
        from_bucket_col="from",
        to_bucket_col="to",
    )

    payload = vintage_summary_payload(vintage, roll)

    assert set(payload) == {"vintage", "roll_rate", "warnings"}
    assert payload["warnings"] == []
    json.dumps(payload)
