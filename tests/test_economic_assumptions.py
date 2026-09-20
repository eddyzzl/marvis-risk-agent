from dataclasses import replace

import pandas as pd
import pytest

from marvis.domain import StrategyProfitInput
from marvis.packs.strategy.economic_assumptions import assess_profit_assumptions


def contract(**changes):
    base = StrategyProfitInput(
        "ead",
        "pd",
        0.12,
        0.03,
        0.5,
        10.0,
        6,
        currency="CNY",
        assumption_sources={
            key: "owner-approved:2026-09"
            for key in (
                "ead",
                "pd",
                "annual_rate",
                "funding_rate",
                "lgd",
                "operating_cost_per_loan",
                "term_months",
                "conversion_rate",
                "utilization_rate",
            )
        },
    )
    return replace(base, **changes)


def frame():
    return pd.DataFrame({"ead": [1000.0, 2000.0], "pd": [0.02, 0.05]})


def test_hand_calculated_profit_and_sensitivity_reversal():
    result = assess_profit_assumptions(
        frame(), contract(), scenarios={"high_cost": {"operating_cost_per_loan": 50.0}}
    )
    values = result["economics"]
    assert values["revenue"] == 180.0
    assert values["expected_loss"] == 60.0
    assert values["funding_cost"] == 45.0
    assert values["operating_cost"] == 20.0
    assert values["profit"] == 55.0
    assert result["sensitivity"][0]["economics"]["profit"] == -25.0
    assert result["sensitivity"][0]["profit_sign_reversed"] is True
    assert result["effect_stage"] == "estimated"
    assert result["causal_gain_verified"] is False
    assert "by_row" not in values


def test_currency_scale_normalizes_both_exposure_and_cost():
    smaller = frame().assign(ead=lambda f: f.ead / 1000)
    result = assess_profit_assumptions(
        smaller,
        contract(exposure_unit="currency_thousands", operating_cost_per_loan=0.01),
    )
    assert result["economics"]["profit"] == pytest.approx(55.0)
    assert result["currency"] == "CNY"
    assert result["output_unit"] == "currency_units"


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": None},
        {"assumption_sources": {}},
        {"pd_col": "missing"},
        {"exposure_basis": "offered_limit"},
        {"exposure_basis": "offered_limit", "conversion_rate": 0.5},
    ],
)
def test_missing_economic_contract_never_invents_profit(changes):
    result = assess_profit_assumptions(frame(), contract(**changes))
    assert result["status"] == "insufficient_evidence"
    assert result["economics"] is None
    assert result["missing"]


def test_offered_limit_uses_conversion_utilization_and_per_loan_cost_consistently():
    result = assess_profit_assumptions(
        frame(),
        contract(
            exposure_basis="offered_limit",
            conversion_rate=0.5,
            utilization_rate=0.8,
            behavior_response_ref="unverified-paper",
        ),
    )
    assert result["economics"]["profit"] == pytest.approx(20.0)
    assert result["behavior_response_status"] == "not_validated"
    assert result["interpretation"] == "fixed_condition_scenario"


def test_invalid_sensitivity_and_unknown_denominator_rejected():
    for changes in (
        {"lgd": True},
        {"term_months": 1.5},
        {"funding_rate": float("nan")},
        {"invent_profit": 999},
    ):
        with pytest.raises(ValueError):
            assess_profit_assumptions(frame(), contract(), scenarios={"bad": changes})
    result = assess_profit_assumptions(frame().assign(ead=0.0), contract())
    assert result["economics"]["roa"] is None
