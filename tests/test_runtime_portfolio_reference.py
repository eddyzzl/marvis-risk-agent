"""Hand-calculated portfolio cases, independent of production calculators."""

import copy

import pytest

from marvis.orchestrator.eval.runtime_portfolio_reference import transition_reference, segment_reference


def calculate(rows, **kwargs):
    return transition_reference(rows, id_col="loan", snapshot_col="month", bucket_col="state",
                                balance_col="ead", states=["good", "loss"], loss_state="loss",
                                lgd=0.5, horizon_months=2, **kwargs)


@pytest.fixture
def rows():
    return [
        {"loan": "a", "month": "2025-12", "state": "good", "ead": 100},
        {"loan": "b", "month": "2025-12", "state": "good", "ead": 1},
        {"loan": "a", "month": "2026-01", "state": "good", "ead": 80},
        {"loan": "b", "month": "2026-01", "state": "loss", "ead": 1},
    ]


def test_balance_migration_count_probability_and_latest_snapshot(rows):
    result = calculate(rows)
    assert result["migration"]["avg_matrix"][0] == pytest.approx([100 / 101, 1 / 101, 0])
    # Count-based monthly loss is 1/2, hence two-month absorption is 3/4.
    assert result["expected_loss"]["chain"] == [
        {"from_state": "good", "p_to_loss": 0.75}, {"from_state": "loss", "p_to_loss": 1.0},
    ]
    assert result["expected_loss"]["total_el"] == 30.5
    assert result["expected_loss"]["el_by_month"][0]["expected_loss"] == 37.875
    assert result["flow"]["net_flows"] == [{"month": "2025-12", "into_bad": 1.0, "out_of_bad": 0.0}]


def test_exit_not_loss_and_latest_panel_edge_not_exit(rows):
    rows.pop()
    result = calculate(rows)
    assert result["flow"]["matrix_by_month"][0]["from_to_matrix"][0] == pytest.approx([100 / 101, 0, 1 / 101])
    assert result["expected_loss"]["total_el"] == 0.0
    assert len(result["flow"]["matrix_by_month"]) == 1


def test_segment_exposure_concentration_differs_from_population_count():
    result = segment_reference([
        {"group": "many", "ead": 1}, {"group": "many", "ead": 1},
        {"group": "large", "ead": 98},
    ], segment_col="group", balance_col="ead")
    assert result["concentration"]["top1_pct"] == 2 / 3
    assert result["concentration"]["hhi"] == pytest.approx(5 / 9)
    assert result["ead_concentration"]["top1_pct"] == 0.98
    assert result["ead_concentration"]["hhi"] == pytest.approx(0.9608)
    assert result["segments"][0]["count"] == 2
    assert result["segments"][0]["bad_rate"] is None


def test_segment_concentration_precedes_other_group_aggregation():
    result = segment_reference([{"group": f"g{i}", "ead": 1} for i in range(22)],
                               segment_col="group", balance_col="ead")
    assert len(result["segments"]) == 21
    assert result["segments"][-1]["segment"] == "其他" and result["segments"][-1]["count"] == 2
    assert result["concentration"]["hhi"] == pytest.approx(1 / 22)
    assert result["concentration"]["top1_pct"] == 1 / 22
    assert result["red_flags"] == [{"kind": "sparse_segment", "merged_count": 2}]


@pytest.mark.parametrize("problem", ["duplicate", "mixed_id", "null_id", "negative", "nan", "gap", "window", "single_month"])
def test_no_independent_pass_for_ambiguous_or_unobserved_input(rows, problem):
    extra = {}
    if problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "mixed_id":
        rows[0]["loan"] = 1
    elif problem == "null_id":
        rows[0]["loan"] = None
    elif problem == "negative":
        rows[0]["ead"] = -1
    elif problem == "nan":
        rows[0]["ead"] = float("nan")
    elif problem == "gap":
        for row in rows[2:]:
            row["month"] = "2026-02"
    elif problem == "window":
        extra["window"] = ["2026-01"]
    else:
        rows = rows[:2]
    with pytest.raises(ValueError):
        calculate(rows, **extra)
