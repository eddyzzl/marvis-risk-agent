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


def test_vintage_curve_uses_snapshot_cohort_denominator_without_pooling():
    frame = pd.DataFrame({
        "cohort": ["202601"] * 8,
        "mob": [0] * 4 + [1] * 4,
        "bad": [0, 0, 0, 1, 0, 1, 1, 1],
        "balance": [100.0] * 8,
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

    count_by_mob = {point.mob: point for point in count_points}
    balance_by_mob = {point.mob: point for point in balance_points}
    # cum_bad_rate is genuinely cumulative: cohort denominator = max sample_count
    # (4) across the cohort's MOBs; mob=1 accumulates bad from mob=0 (1) plus
    # mob=1 (3) = 4/4 = 1.0, not the old bug's bare per-MOB bad_rate (0.75).
    assert count_by_mob[0].cum_bad_rate == pytest.approx(0.25)
    assert count_by_mob[1].cum_bad_rate == pytest.approx(1.0)
    assert balance_by_mob[1].cum_bad_rate == pytest.approx(1.0)


def test_cum_bad_rate_is_genuinely_cumulative_across_two_cohorts_and_three_mobs():
    frame = pd.DataFrame({
        "cohort": ["202601"] * 12 + ["202602"] * 9,
        "mob": [0] * 5 + [1] * 4 + [2] * 3 + [0] * 4 + [1] * 3 + [2] * 2,
        "bad": (
            [1, 0, 0, 0, 0] + [1, 1, 0, 0] + [1, 0, 0]
            + [0, 0, 0, 1] + [1, 0, 0] + [0, 1]
        ),
    })

    points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")
    by_key = {(point.cohort, point.mob): point for point in points}

    # cohort 2026-01: denominator = max(5, 4, 3) = 5; cumulative bad 1, 3, 4.
    assert by_key[("2026-01", 0)].cum_bad_rate == pytest.approx(1 / 5)
    assert by_key[("2026-01", 1)].cum_bad_rate == pytest.approx(3 / 5)
    assert by_key[("2026-01", 2)].cum_bad_rate == pytest.approx(4 / 5)

    # cohort 2026-02: denominator = max(4, 3, 2) = 4; cumulative bad 1, 2, 3.
    assert by_key[("2026-02", 0)].cum_bad_rate == pytest.approx(1 / 4)
    assert by_key[("2026-02", 1)].cum_bad_rate == pytest.approx(2 / 4)
    assert by_key[("2026-02", 2)].cum_bad_rate == pytest.approx(3 / 4)

    for cohort in ("2026-01", "2026-02"):
        rates = [by_key[(cohort, mob)].cum_bad_rate for mob in (0, 1, 2)]
        assert rates == sorted(rates)


def test_cum_bad_rate_uses_max_sample_count_when_cohort_first_observed_mid_life():
    # Cohort is only observed starting at mob=1 (no mob=0 rows), and mob=2 has
    # MORE samples than mob=1. Using the first-observed MOB's count (2) as the
    # denominator would give cum_bad_rate = 3/2 = 1.5 at mob=2 -- nonsensical.
    # The fixed cohort denominator must be max(sample_count across MOBs) = 5.
    frame = pd.DataFrame({
        "cohort": ["202601"] * 2 + ["202601"] * 5,
        "mob": [1, 1] + [2] * 5,
        "bad": [1, 0] + [1, 1, 0, 0, 0],
    })

    points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")
    by_mob = {point.mob: point for point in points}

    assert by_mob[1].sample_count == 2
    assert by_mob[2].sample_count == 5
    assert by_mob[1].cum_bad_rate == pytest.approx(1 / 5)
    assert by_mob[2].cum_bad_rate == pytest.approx(3 / 5)
    assert by_mob[2].cum_bad_rate <= 1.0
    assert by_mob[1].cum_bad_rate <= by_mob[2].cum_bad_rate
    assert by_mob[1].data_quality_warnings == ()
    assert by_mob[2].data_quality_warnings == ()


def test_cum_bad_rate_clips_to_one_and_records_data_quality_warning_when_denominator_exceeded():
    # bad_numerator across the cohort's rows sums to more than the fixed
    # cohort denominator (only possible when the same "bad" flag is summed
    # across MOBs from data that isn't truly incremental) -- the kernel must
    # clip to 1.0 and surface a warning rather than silently emitting >1.
    frame = pd.DataFrame({
        "cohort": ["202601"] * 2 + ["202601"] * 2,
        "mob": [0, 0, 1, 1],
        "bad": [1, 1, 1, 0],
    })

    points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")
    by_mob = {point.mob: point for point in points}

    assert by_mob[0].cum_bad_rate == pytest.approx(1.0)
    assert by_mob[0].data_quality_warnings == ()
    assert by_mob[1].cum_bad_rate == pytest.approx(1.0)
    assert len(by_mob[1].data_quality_warnings) == 1
    assert "clipped" in by_mob[1].data_quality_warnings[0]

    payload = vintage_summary_payload(points)
    assert payload["warnings"] == list(by_mob[1].data_quality_warnings)


def _snapshot_flag_frame() -> pd.DataFrame:
    # 4 loans in one cohort. Snapshot/ever-bad semantics: a loan flagged bad at an
    # earlier MOB STAYS bad at every later MOB. Loan 1 goes bad at mob0 (stays bad
    # mob1/mob2); loan 2 goes bad at mob1 (stays bad mob2). True cumulative bad
    # rate: mob0=1/4, mob1=2/4, mob2=2/4. The incremental accumulator would sum the
    # per-row bads (1 + 2 + 2 = 5 > 4) and blow past 1.0 -- exactly the double-count.
    return pd.DataFrame({
        "cohort": ["202601"] * 12,
        "mob": [0, 0, 0, 0] + [1, 1, 1, 1] + [2, 2, 2, 2],
        "bad": [1, 0, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0],
    })


def test_snapshot_semantics_does_not_accumulate_bad():
    frame = _snapshot_flag_frame()

    snapshot_points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
        label_semantics="snapshot",
    )
    by_mob = {point.mob: point for point in snapshot_points}

    # cum_bad_rate under snapshot semantics IS the per-MOB marginal bad_rate: the
    # already-cumulative flag is read directly, never re-accumulated.
    assert by_mob[0].cum_bad_rate == pytest.approx(1 / 4)
    assert by_mob[1].cum_bad_rate == pytest.approx(2 / 4)
    assert by_mob[2].cum_bad_rate == pytest.approx(2 / 4)
    for point in snapshot_points:
        assert point.cum_bad_rate == pytest.approx(point.bad_rate)
        assert point.cum_bad_rate <= 1.0

    # Equivalence with the report_compute self-protection path (metric='bad_rate').
    snapshot_wide = vintage_curve_wide(snapshot_points, metric="cum_bad_rate")
    bad_rate_wide = vintage_curve_wide(snapshot_points, metric="bad_rate")
    assert snapshot_wide == bad_rate_wide


def test_incremental_semantics_is_backward_compatible():
    frame = pd.DataFrame({
        "cohort": ["202601"] * 12 + ["202602"] * 9,
        "mob": [0] * 5 + [1] * 4 + [2] * 3 + [0] * 4 + [1] * 3 + [2] * 2,
        "bad": (
            [1, 0, 0, 0, 0] + [1, 1, 0, 0] + [1, 0, 0]
            + [0, 0, 0, 1] + [1, 0, 0] + [0, 1]
        ),
    })

    default_points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")
    explicit_points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
        label_semantics="incremental",
    )

    # Default is incremental and passing it explicitly is byte-identical.
    assert [p.cum_bad_rate for p in explicit_points] == [p.cum_bad_rate for p in default_points]
    by_key = {(p.cohort, p.mob): p for p in explicit_points}
    assert by_key[("2026-01", 0)].cum_bad_rate == pytest.approx(1 / 5)
    assert by_key[("2026-01", 1)].cum_bad_rate == pytest.approx(3 / 5)
    assert by_key[("2026-01", 2)].cum_bad_rate == pytest.approx(4 / 5)


def test_label_semantics_rejects_unknown_value():
    frame = pd.DataFrame({"cohort": ["202601"], "mob": [0], "bad": [0]})
    with pytest.raises(ValueError, match="label_semantics"):
        compute_vintage_curve(
            frame,
            cohort_col="cohort",
            mob_col="mob",
            target_col="bad",
            label_semantics="cumulative",
        )


def test_monotone_bad_count_across_all_cohorts_flags_snapshot_red_flag():
    # Per-MOB bad_count is non-decreasing in EVERY cohort (looks like snapshot
    # flags) but the caller declared 'incremental' -- the kernel must attach an
    # advisory red flag naming 'snapshot', never mutating the curve.
    frame = _snapshot_flag_frame()

    points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
        label_semantics="incremental",
    )

    all_warnings = [w for point in points for w in point.data_quality_warnings]
    assert any("snapshot" in w.lower() or "快照" in w for w in all_warnings)


def test_snapshot_declaration_suppresses_the_monotone_red_flag():
    # Same monotone data, but declared snapshot -> the monotone heuristic is
    # satisfied, so no snapshot red flag is emitted.
    frame = _snapshot_flag_frame()

    points = compute_vintage_curve(
        frame,
        cohort_col="cohort",
        mob_col="mob",
        target_col="bad",
        label_semantics="snapshot",
    )

    all_warnings = [w for point in points for w in point.data_quality_warnings]
    assert not any("snapshot" in w.lower() or "快照" in w for w in all_warnings)


def test_vintage_curve_wide_aligns_mob_axis_and_preserves_missing():
    frame = pd.DataFrame({
        "cohort": ["202501", "202501", "202502", "202502"],
        "mob": [0, 1, 1, 3],
        "bad": [0, 1, 1, 1],
    })
    points = compute_vintage_curve(frame, cohort_col="cohort", mob_col="mob", target_col="bad")

    wide = vintage_curve_wide(points)
    bad_rate = vintage_curve_wide(points, metric="bad_rate")

    assert wide["2025-01"] == [0.0, 1.0, None]
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
