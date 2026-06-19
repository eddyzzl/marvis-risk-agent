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


def test_strategy_package_exports_roll_rate_matrix():
    assert strategy_pack.roll_rate_matrix is roll_rate_matrix
