import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.packs.strategy import VintageCurve, vintage_curve, vintage_summary
from marvis.validation.vintage import compute_vintage_curve, vintage_curve_wide


def test_vintage_curve_wraps_phase4v_wide_curve_and_counts():
    frame = pd.DataFrame({
        "loan_month": ["202601", "202601", "202601", "202602", "202602"],
        "mob": [0, 0, 1, 0, 1],
        "bad": [0, 1, 1, 0, 0],
    })
    expected_points = compute_vintage_curve(
        frame,
        cohort_col="loan_month",
        mob_col="mob",
        target_col="bad",
    )
    expected_wide = vintage_curve_wide(expected_points, metric="cum_bad_rate")

    curve = vintage_curve(
        frame,
        cohort_col="loan_month",
        mob_col="mob",
        bad_col="bad",
        mob_max=3,
    )

    assert isinstance(curve, VintageCurve)
    assert curve.cohort_col == "loan_month"
    assert curve.mob_max == 3
    assert curve.cohorts == ("2026-01", "2026-02")
    assert curve.curves["2026-01"] == [*expected_wide["2026-01"], None]
    assert curve.curves["2026-02"] == [*expected_wide["2026-02"], None]
    assert curve.counts == {"2026-01": 2, "2026-02": 1}
    assert curve.mob_axis == (0, 1)


def test_vintage_curve_truncates_to_mob_max_and_handles_empty_input():
    frame = pd.DataFrame({
        "loan_month": ["202601", "202601", "202601"],
        "mob": [0, 1, 2],
        "bad": [0, 1, 1],
    })

    curve = vintage_curve(
        frame,
        cohort_col="loan_month",
        mob_col="mob",
        bad_col="bad",
        mob_max=2,
    )
    empty_curve = vintage_curve(
        pd.DataFrame({"loan_month": [], "mob": [], "bad": []}),
        cohort_col="loan_month",
        mob_col="mob",
        bad_col="bad",
        mob_max=4,
    )

    assert curve.curves["2026-01"] == [0.0, 1.0]
    assert empty_curve == VintageCurve(
        cohort_col="loan_month",
        mob_max=4,
        cohorts=(),
        curves={},
        counts={},
    )


def test_vintage_summary_identifies_deteriorating_and_improving_trends():
    deteriorating = VintageCurve(
        cohort_col="loan_month",
        mob_max=2,
        cohorts=("2026-01", "2026-02", "2026-03"),
        curves={
            "2026-01": [0.01, 0.02],
            "2026-02": [0.02, 0.04],
            "2026-03": [0.03, 0.07],
        },
        counts={"2026-01": 100, "2026-02": 100, "2026-03": 100},
    )
    improving = VintageCurve(
        cohort_col="loan_month",
        mob_max=2,
        cohorts=("2026-01", "2026-02", "2026-03"),
        curves={
            "2026-01": [0.03, 0.08],
            "2026-02": [0.02, 0.05],
            "2026-03": [0.01, 0.03],
        },
        counts={"2026-01": 100, "2026-02": 100, "2026-03": 100},
    )

    assert vintage_summary(deteriorating, ref_mob=2) == {
        "at_ref": {"2026-01": 0.02, "2026-02": 0.04, "2026-03": 0.07},
        "trend": "deteriorating",
    }
    assert vintage_summary(improving, ref_mob=2)["trend"] == "improving"


def test_vintage_summary_skips_missing_reference_mob_and_defaults_stable():
    curve = VintageCurve(
        cohort_col="loan_month",
        mob_max=3,
        cohorts=("2026-01", "2026-02"),
        curves={"2026-01": [0.01, None, 0.03], "2026-02": [0.01]},
        counts={"2026-01": 100, "2026-02": 80},
    )

    assert vintage_summary(curve, ref_mob=2) == {"at_ref": {}, "trend": "stable"}
    with pytest.raises(ValueError, match="ref_mob"):
        vintage_summary(curve, ref_mob=0)


def test_vintage_summary_uses_actual_mob_axis_for_non_contiguous_mobs():
    curve = VintageCurve(
        cohort_col="loan_month",
        mob_max=2,
        cohorts=("2026-01", "2026-02"),
        curves={"2026-01": [0.01, 0.03], "2026-02": [0.02, 0.04]},
        counts={"2026-01": 100, "2026-02": 100},
        mob_axis=(1, 3),
    )

    assert vintage_summary(curve, ref_mob=3) == {
        "at_ref": {"2026-01": 0.03, "2026-02": 0.04},
        "trend": "deteriorating",
    }


def test_strategy_package_exports_vintage_functions():
    assert strategy_pack.vintage_curve is vintage_curve
    assert strategy_pack.vintage_summary is vintage_summary
