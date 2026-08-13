"""Tests for marvis.packs.strategy.leakage_diagnostics.

Golden numbers below are hand-computed from the synthetic inputs; none are
re-derived by calling the function under test's own formulas.
"""

import math

import pandas as pd
import pytest

from marvis.packs.strategy.leakage_diagnostics import (
    LeakageDiagnosticsError,
    selection_bias_diagnostics,
    time_leakage_diagnostics,
)


def _summary(evidence, column: str):
    return next(summary for summary in evidence.columns if summary.column == column)


# --- time leakage ---


def test_time_leakage_no_overlap_reports_no_red_flags():
    df = pd.DataFrame(
        {
            "f_a": [1.0, 2.0, 3.0],
            "f_b": [0.1, 0.2, 0.3],
            "obs": ["2026-01-01", "2026-01-15", "2026-01-31"],
        }
    )
    result = time_leakage_diagnostics(
        df,
        feature_cols=["f_a", "f_b"],
        observation_time_col="obs",
        label_window_start="2026-02-01",
        label_window_end="2026-03-31",
    )

    assert result.window.mode == "scalar"
    assert [feature.feature for feature in result.features] == ["f_a", "f_b"]
    assert result.red_flags == ()

    for feature in result.features:
        assert feature.time_col == "obs"
        assert feature.observed_rows == 3
        assert feature.missing_rows == 0
        assert feature.before_window_count == 3
        assert feature.in_window_count == 0
        assert feature.after_window_count == 0
        assert feature.overlap == "no_overlap"
        assert feature.overlap_rate == 0.0
        assert feature.red_flags == ()


def test_time_leakage_overlap_flags_red():
    df = pd.DataFrame(
        {
            "f": [1.0, 2.0, 3.0, 4.0],
            "obs": [
                "2026-01-01",
                "2026-02-15",
                "2026-03-31",
                "2026-05-01",
            ],
        }
    )
    result = time_leakage_diagnostics(
        df,
        feature_cols=["f"],
        observation_time_col="obs",
        label_window_start="2026-02-01",
        label_window_end="2026-03-31",
    )

    feature = result.features[0]
    # 1 before, 2 inside (inclusive boundaries), 1 after.
    assert feature.observed_rows == 4
    assert feature.missing_rows == 0
    assert feature.before_window_count == 1
    assert feature.in_window_count == 2
    assert feature.after_window_count == 1
    assert feature.overlap == "overlap"
    assert feature.overlap_rate == 0.5

    codes = {(flag.code, flag.level) for flag in result.red_flags}
    assert ("time_leakage_overlap", "red") in codes
    assert ("time_leakage_after_window", "red") in codes
    assert any(
        flag.code == "time_leakage_overlap" and "f" in flag.message
        for flag in result.red_flags
    )


def test_time_leakage_missing_time_column_fails_closed():
    df = pd.DataFrame({"f": [1.0, 2.0], "obs": ["2026-01-01", "2026-01-02"]})
    with pytest.raises(LeakageDiagnosticsError, match="missing columns"):
        time_leakage_diagnostics(
            df,
            feature_cols=["f"],
            observation_time_col="does_not_exist",
            label_window_start="2026-02-01",
            label_window_end="2026-03-31",
        )


def test_time_leakage_missing_window_fails_closed():
    df = pd.DataFrame({"f": [1.0, 2.0], "obs": ["2026-01-01", "2026-01-02"]})
    with pytest.raises(LeakageDiagnosticsError, match="label window is required"):
        time_leakage_diagnostics(df, feature_cols=["f"], observation_time_col="obs")


def test_time_leakage_missing_feature_column_fails_closed():
    df = pd.DataFrame({"f": [1.0, 2.0], "obs": ["2026-01-01", "2026-01-02"]})
    with pytest.raises(LeakageDiagnosticsError, match="missing columns"):
        time_leakage_diagnostics(
            df,
            feature_cols=["ghost"],
            observation_time_col="obs",
            label_window_start="2026-02-01",
            label_window_end="2026-03-31",
        )


def test_time_leakage_per_feature_time_columns():
    df = pd.DataFrame(
        {
            "f_a": [1.0, 2.0],
            "f_b": [3.0, 4.0],
            "ta": ["2026-01-01", "2026-01-02"],
            "tb": ["2026-02-15", "2026-03-31"],
        }
    )
    result = time_leakage_diagnostics(
        df,
        feature_cols=["f_a", "f_b"],
        feature_time_cols={"f_a": "ta", "f_b": "tb"},
        label_window_start="2026-02-01",
        label_window_end="2026-03-31",
    )

    by_feature = {feature.feature: feature for feature in result.features}
    assert by_feature["f_a"].time_col == "ta"
    assert by_feature["f_a"].in_window_count == 0
    assert by_feature["f_a"].overlap == "no_overlap"
    assert by_feature["f_b"].time_col == "tb"
    assert by_feature["f_b"].in_window_count == 2
    assert by_feature["f_b"].overlap == "overlap"
    assert [flag.code for flag in result.red_flags] == ["time_leakage_overlap"]


def test_time_leakage_numeric_yyyymmdd_labels():
    df = pd.DataFrame({"f": [1, 2], "obs": [20260101, 20260115]})
    result = time_leakage_diagnostics(
        df,
        feature_cols=["f"],
        observation_time_col="obs",
        label_window_start=20260201,
        label_window_end=20260331,
    )
    assert result.features[0].observed_rows == 2
    assert result.features[0].before_window_count == 2
    assert result.features[0].in_window_count == 0
    assert result.features[0].overlap == "no_overlap"
    assert result.red_flags == ()


# --- selection bias ---


def test_selection_bias_golden_numbers():
    approval_df = pd.DataFrame(
        {
            "target": [1, 0, 1, 0],
            "score": [0.0, 2.0, 0.0, 2.0],
            "income": [100.0, 200.0, 300.0, None],
        }
    )
    risk_df = pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 0, 1],
            "score": [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            "income": [100.0, 200.0, 300.0, 100.0, 200.0, 300.0],
        }
    )

    result = selection_bias_diagnostics(
        approval_df, risk_df, columns=["score", "income"], target="target"
    )

    # Population counts / bad rates (hand-computed).
    assert result.target == "target"
    assert result.target_bad_value == 1
    assert result.approval.count == 4
    assert result.approval.labeled_count == 4
    assert result.approval.bad_count == 2
    assert result.approval.bad_rate == 0.5
    assert result.approval.label_coverage == 1.0
    assert result.risk.count == 6
    assert result.risk.labeled_count == 6
    assert result.risk.bad_count == 3
    assert result.risk.bad_rate == 0.5
    assert result.risk.label_coverage == 1.0

    # Approval score summary: [0, 2, 0, 2].
    approval_score = _summary(result.approval, "score")
    assert approval_score.count == 4
    assert approval_score.missing_count == 0
    assert approval_score.missing_rate == 0.0
    assert approval_score.mean == 1.0
    assert approval_score.std == 1.0
    assert approval_score.min == 0.0
    assert approval_score.median == 1.0
    assert approval_score.max == 2.0

    # Risk score summary: [1, 1, 1, 3, 3, 3].
    risk_score = _summary(result.risk, "score")
    assert risk_score.count == 6
    assert risk_score.missing_count == 0
    assert risk_score.missing_rate == 0.0
    assert risk_score.mean == 2.0
    assert risk_score.std == 1.0
    assert risk_score.min == 1.0
    assert risk_score.median == 2.0
    assert risk_score.max == 3.0

    # Approval income summary: [100, 200, 300, None].
    approval_income = _summary(result.approval, "income")
    assert approval_income.count == 3
    assert approval_income.missing_count == 1
    assert approval_income.missing_rate == 0.25
    assert approval_income.mean == 200.0
    assert approval_income.std == pytest.approx(math.sqrt(20000 / 3))
    assert approval_income.min == 100.0
    assert approval_income.median == 200.0
    assert approval_income.max == 300.0

    # Risk income summary: [100, 200, 300, 100, 200, 300].
    risk_income = _summary(result.risk, "income")
    assert risk_income.count == 6
    assert risk_income.missing_count == 0
    assert risk_income.missing_rate == 0.0
    assert risk_income.mean == 200.0
    assert risk_income.std == pytest.approx(math.sqrt(20000 / 3))
    assert risk_income.min == 100.0
    assert risk_income.median == 200.0
    assert risk_income.max == 300.0


def test_selection_bias_count_mismatch_is_evidence_not_error():
    approval_df = pd.DataFrame({"target": [1, 0, 1, 0], "score": [1.0, 2.0, 3.0, 4.0]})
    risk_df = pd.DataFrame(
        {"target": [1, 0, 1, 0, 0, 1], "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )

    result = selection_bias_diagnostics(
        approval_df, risk_df, columns=["score"], target="target"
    )

    # Differing population sizes are reported, not rejected.
    assert result.approval.count == 4
    assert result.risk.count == 6
    assert result.approval.bad_count == 2
    assert result.risk.bad_count == 3
    assert [flag.code for flag in result.flags] == ["population_count_mismatch"]
    assert result.flags[0].level == "info"

    equal = selection_bias_diagnostics(
        approval_df, approval_df, columns=["score"], target="target"
    )
    assert equal.flags == ()


def test_selection_bias_missing_target_fails_closed():
    approval_df = pd.DataFrame({"target": [1, 0], "score": [1.0, 2.0]})
    risk_df = pd.DataFrame({"target": [1, 0], "score": [1.0, 2.0]})
    with pytest.raises(LeakageDiagnosticsError, match="missing columns"):
        selection_bias_diagnostics(
            approval_df, risk_df, columns=["score"], target="does_not_exist"
        )


def test_selection_bias_non_numeric_column_fails_closed():
    approval_df = pd.DataFrame({"target": [1, 0], "score": ["a", "b"]})
    risk_df = pd.DataFrame({"target": [1, 0], "score": ["a", "b"]})
    with pytest.raises(LeakageDiagnosticsError, match="must be numeric"):
        selection_bias_diagnostics(
            approval_df, risk_df, columns=["score"], target="target"
        )


def test_selection_bias_null_target_uses_labeled_denominator():
    approval_df = pd.DataFrame(
        {"target": [1, 0, None, None], "score": [1.0, 2.0, 3.0, 4.0]}
    )
    risk_df = approval_df.copy()

    result = selection_bias_diagnostics(
        approval_df, risk_df, columns=["score"], target="target"
    )

    assert result.approval.count == 4
    assert result.approval.labeled_count == 2
    assert result.approval.bad_count == 1
    assert result.approval.bad_rate == 0.5
    assert result.approval.label_coverage == 0.5
