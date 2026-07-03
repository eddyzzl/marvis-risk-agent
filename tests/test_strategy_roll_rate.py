import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.packs.strategy import RollRateMatrix, roll_rate_matrix


def test_roll_rate_matrix_builds_adjacent_pairs_per_customer_in_time_order():
    frame = pd.DataFrame({
        "customer_id": ["A", "B", "A", "C", "A", "B", "C"],
        "month": ["2026-02", "2026-01", "2026-01", "2026-01", "2026-03", "2026-02", "2026-02"],
        "status": ["M1", "C", "C", "M3+", "M2", "C", "M3+"],
    })

    result = roll_rate_matrix(
        frame,
        id_col="customer_id",
        time_col="month",
        status_col="status",
        states=["C", "M1", "M2", "M3+"],
    )

    assert isinstance(result, RollRateMatrix)
    assert result.states == ("C", "M1", "M2", "M3+")
    assert result.period == "month"
    assert result.base_counts == {"C": 2, "M1": 1, "M2": 0, "M3+": 1}
    assert result.matrix == (
        (0.5, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_roll_rate_matrix_sorts_by_parsed_dates_not_lexical_strings():
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "A"],
        "month": ["2026-1", "2026-10", "2026-2"],
        "status": ["C", "M2", "M1"],
    })

    result = roll_rate_matrix(
        frame,
        id_col="customer_id",
        time_col="month",
        status_col="status",
        states=["C", "M1", "M2"],
    )

    assert result.base_counts == {"C": 1, "M1": 1, "M2": 0}
    assert result.matrix == (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
    )


def test_roll_rate_matrix_rejects_unparseable_time_values():
    frame = pd.DataFrame({
        "customer_id": ["A", "A"],
        "month": ["2026-01", "not-a-date"],
        "status": ["C", "M1"],
    })

    with pytest.raises(ValueError, match="time_col"):
        roll_rate_matrix(
            frame,
            id_col="customer_id",
            time_col="month",
            status_col="status",
            states=["C", "M1"],
        )


def test_roll_rate_matrix_returns_complete_zero_matrix_for_no_transitions():
    frame = pd.DataFrame({
        "customer_id": ["A", "B"],
        "month": ["2026-01", "2026-01"],
        "status": ["C", "M1"],
    })

    result = roll_rate_matrix(
        frame,
        id_col="customer_id",
        time_col="month",
        status_col="status",
        states=["C", "M1"],
    )

    assert result.base_counts == {"C": 0, "M1": 0}
    assert result.matrix == ((0.0, 0.0), (0.0, 0.0))


def test_roll_rate_matrix_rejects_unknown_status_and_empty_states():
    frame = pd.DataFrame({
        "customer_id": ["A", "A"],
        "month": ["2026-01", "2026-02"],
        "status": ["C", "UNKNOWN"],
    })

    with pytest.raises(ValueError, match="unknown status"):
        roll_rate_matrix(
            frame,
            id_col="customer_id",
            time_col="month",
            status_col="status",
            states=["C", "M1"],
        )
    with pytest.raises(ValueError, match="states"):
        roll_rate_matrix(
            pd.DataFrame({"customer_id": [], "month": [], "status": []}),
            id_col="customer_id",
            time_col="month",
            status_col="status",
            states=[],
        )


def test_roll_rate_matrix_flags_missing_month_gap_without_changing_matrix():
    # DOM-8: id A skips 202602 (202601 -> 202603, a 1-month gap); id B is a normal
    # consecutive pair. The gap only produces a data_quality_warnings entry -- the
    # matrix/base_counts computation is untouched (informational only).
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B"],
        "month": ["202601", "202603", "202601", "202602"],
        "status": ["C", "M1", "C", "C"],
    })

    result = roll_rate_matrix(
        frame, id_col="customer_id", time_col="month", status_col="status", states=["C", "M1"],
    )

    assert result.matrix == ((0.5, 0.5), (0.0, 0.0))
    assert result.base_counts == {"C": 2, "M1": 0}
    assert len(result.data_quality_warnings) == 1
    warning = result.data_quality_warnings[0]
    assert warning["code"] == "missing_month"
    assert warning["id"] == "A"
    assert warning["gap_months"] == 1


def test_roll_rate_matrix_no_gap_produces_no_warnings():
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B"],
        "month": ["202601", "202602", "202601", "202602"],
        "status": ["C", "M1", "C", "C"],
    })

    result = roll_rate_matrix(
        frame, id_col="customer_id", time_col="month", status_col="status", states=["C", "M1"],
    )

    assert result.data_quality_warnings == ()


def test_roll_rate_matrix_balance_col_weights_transitions_hand_computed():
    # DOM-8: 3 ids, one transition each. A: C->C, balance 100. B: C->M1, balance
    # 300. C: M1->M1, balance 50. Balance-weighted: from C, weighted "C->C" = 100,
    # "C->M1" = 300, base[C] = 400 -> C->C = 0.25, C->M1 = 0.75 (hand-computed).
    # This diverges from the unweighted 0.5/0.5 count-based result below.
    frame = pd.DataFrame({
        "customer_id": ["A", "A", "B", "B", "C", "C"],
        "month": ["202601", "202602", "202601", "202602", "202601", "202602"],
        "status": ["C", "C", "C", "M1", "M1", "M1"],
        "balance": [100.0, 100.0, 300.0, 300.0, 50.0, 50.0],
    })

    weighted = roll_rate_matrix(
        frame, id_col="customer_id", time_col="month", status_col="status",
        states=["C", "M1"], balance_col="balance",
    )
    assert weighted.matrix == ((0.25, 0.75), (0.0, 1.0))
    assert weighted.base_counts == {"C": 400.0, "M1": 50.0}

    unweighted = roll_rate_matrix(
        frame, id_col="customer_id", time_col="month", status_col="status", states=["C", "M1"],
    )
    assert unweighted.matrix == ((0.5, 0.5), (0.0, 1.0))
    assert unweighted.base_counts == {"C": 2, "M1": 1}


def test_roll_rate_matrix_default_behavior_unchanged_without_balance_col():
    # DOM-8: omitting balance_col must reproduce the pre-existing byte-identical
    # result (default count-based path untouched).
    frame = pd.DataFrame({
        "customer_id": ["A", "B", "A", "C", "A", "B", "C"],
        "month": ["2026-02", "2026-01", "2026-01", "2026-01", "2026-03", "2026-02", "2026-02"],
        "status": ["M1", "C", "C", "M3+", "M2", "C", "M3+"],
    })

    result = roll_rate_matrix(
        frame, id_col="customer_id", time_col="month", status_col="status",
        states=["C", "M1", "M2", "M3+"],
    )

    assert result.base_counts == {"C": 2, "M1": 1, "M2": 0, "M3+": 1}
    assert result.matrix == (
        (0.5, 0.5, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert result.data_quality_warnings == ()


def test_strategy_package_exports_roll_rate_matrix():
    assert strategy_pack.roll_rate_matrix is roll_rate_matrix
