import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.packs.strategy import ProfitParams, profit_calc, vintage_profit


def _params() -> ProfitParams:
    return ProfitParams(
        annual_rate=0.12,
        funding_rate=0.03,
        lgd=0.5,
        operating_cost_per_loan=10.0,
        term_months=6,
    )


def test_profit_calc_matches_manual_formula_by_segment():
    frame = pd.DataFrame({
        "segment": ["A", "A", "B"],
        "ead": [1000.0, 2000.0, 500.0],
        "pd": [0.02, 0.05, 0.10],
    })

    results = profit_calc(frame, segment_col="segment", ead_col="ead", pd_col="pd", params=_params())
    by_segment = {result.segment: result for result in results}

    assert by_segment["A"].count == 2
    assert by_segment["A"].revenue == pytest.approx(180.0)
    assert by_segment["A"].expected_loss == pytest.approx(60.0)
    assert by_segment["A"].funding_cost == pytest.approx(45.0)
    assert by_segment["A"].operating_cost == pytest.approx(20.0)
    assert by_segment["A"].net_profit == pytest.approx(55.0)
    assert by_segment["A"].roa == pytest.approx(55.0 / 3000.0)

    assert by_segment["B"].revenue == pytest.approx(30.0)
    assert by_segment["B"].expected_loss == pytest.approx(25.0)
    assert by_segment["B"].funding_cost == pytest.approx(7.5)
    assert by_segment["B"].operating_cost == pytest.approx(10.0)
    assert by_segment["B"].net_profit == pytest.approx(-12.5)
    assert by_segment["B"].roa == pytest.approx(-12.5 / 500.0)


def test_profit_calc_supports_overall_and_zero_ead():
    frame = pd.DataFrame({
        "ead": [0.0, 0.0],
        "pd": [0.10, 0.20],
    })

    result = profit_calc(frame, segment_col=None, ead_col="ead", pd_col="pd", params=_params())[0]

    assert result.segment == "all"
    assert result.count == 2
    assert result.revenue == 0.0
    assert result.expected_loss == 0.0
    assert result.funding_cost == 0.0
    assert result.operating_cost == 20.0
    assert result.net_profit == -20.0
    assert result.roa == 0.0


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("ead", [100.0, -1.0], "EAD"),
        ("ead", [100.0, float("inf")], "finite"),
        ("pd", [0.1, 1.1], "PD"),
        ("pd", [0.1, float("nan")], "finite"),
    ],
)
def test_profit_calc_rejects_invalid_economic_inputs(column, values, message):
    frame = pd.DataFrame({"ead": [100.0, 200.0], "pd": [0.1, 0.2]})
    frame[column] = values

    with pytest.raises(ValueError, match=message):
        profit_calc(frame, segment_col=None, ead_col="ead", pd_col="pd", params=_params())


@pytest.mark.parametrize(
    "params",
    [
        ProfitParams(-0.01, 0.03, 0.5, 10.0, 6),
        ProfitParams(0.12, 1.01, 0.5, 10.0, 6),
        ProfitParams(0.12, 0.03, 1.01, 10.0, 6),
        ProfitParams(0.12, 0.03, 0.5, -1.0, 6),
        ProfitParams(0.12, 0.03, 0.5, 10.0, 0),
        ProfitParams(float("inf"), 0.03, 0.5, 10.0, 6),
    ],
)
def test_profit_calc_rejects_invalid_bounded_parameters(params):
    frame = pd.DataFrame({"ead": [100.0], "pd": [0.1]})

    with pytest.raises(ValueError, match="profit parameter"):
        profit_calc(frame, segment_col=None, ead_col="ead", pd_col="pd", params=params)


def test_profit_calc_rejects_empty_dataset():
    frame = pd.DataFrame({"ead": pd.Series(dtype=float), "pd": pd.Series(dtype=float)})

    with pytest.raises(ValueError, match="non-empty"):
        profit_calc(frame, segment_col=None, ead_col="ead", pd_col="pd", params=_params())


def test_vintage_profit_returns_profit_result_by_cohort():
    frame = pd.DataFrame({
        "cohort": ["2026-01", "2026-01", "2026-02"],
        "ead": [1000.0, 2000.0, 500.0],
        "pd": [0.02, 0.05, 0.10],
    })

    results = vintage_profit(
        frame,
        cohort_col="cohort",
        ead_col="ead",
        pd_col="pd",
        params=_params(),
    )

    assert set(results) == {"2026-01", "2026-02"}
    assert results["2026-01"].segment == "2026-01"
    assert results["2026-01"].net_profit == pytest.approx(55.0)
    assert results["2026-02"].net_profit == pytest.approx(-12.5)


def test_strategy_package_exports_profit_functions():
    assert strategy_pack.ProfitParams is ProfitParams
    assert strategy_pack.profit_calc is profit_calc
    assert strategy_pack.vintage_profit is vintage_profit
