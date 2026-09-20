"""Explicit, fixed-condition economics using the existing row-level engine."""

from dataclasses import asdict, replace

import pandas as pd

from marvis.domain import StrategyProfitInput
from marvis.orchestrator.evidence import payload_hash
from marvis.packs.strategy.economics import pricing_metrics


_SOURCE_FIELDS = (
    "ead",
    "pd",
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
)
_SCALES = {
    "currency_units": 1.0,
    "currency_thousands": 1000.0,
    "currency_millions": 1_000_000.0,
}
_SCENARIO_FIELDS = frozenset(
    {
        "annual_rate",
        "funding_rate",
        "lgd",
        "operating_cost_per_loan",
        "term_months",
        "conversion_rate",
        "utilization_rate",
    }
)


def assess_profit_assumptions(
    frame, contract: StrategyProfitInput | None, *, scenarios=None
):
    """Return no economic value when assumptions or source declarations are absent.

    Source declarations are attribution, not empirical validation. Every number
    here remains an estimated fixed-condition scenario, even when a caller names
    a behavior-response document. No response-validation adapter exists here.
    Currency scale applies to both exposure and per-loan monetary cost.
    """
    result = {
        "schema_version": "economic-assumptions.v1",
        "status": "insufficient_evidence",
        "effect_stage": "estimated",
        "interpretation": "fixed_condition_scenario",
        "causal_gain_verified": False,
        "behavior_response_status": "not_validated",
        "currency": None,
        "output_unit": "currency_units",
        "assumptions": None,
        "assumptions_hash": None,
        "missing": [],
        "economics": None,
        "sensitivity": [],
    }
    if contract is None:
        result["missing"] = ["profit_contract"]
        return result
    assumptions = asdict(contract)
    result.update(
        currency=contract.currency,
        assumptions=assumptions,
        assumptions_hash=payload_hash(assumptions),
    )
    required_sources = list(_SOURCE_FIELDS)
    if contract.exposure_basis == "offered_limit":
        required_sources += ["conversion_rate", "utilization_rate"]
        for name in ("conversion_rate", "utilization_rate"):
            if getattr(contract, name) is None:
                result["missing"].append(name)
    if not contract.currency:
        result["missing"].append("currency")
    for name in required_sources:
        if not contract.assumption_sources.get(name):
            result["missing"].append(f"source:{name}")
    for name in (contract.ead_col, contract.pd_col):
        if name not in frame:
            result["missing"].append(f"column:{name}")
    if result["missing"]:
        return result
    result["economics"] = _calculate(frame, contract)
    result["status"] = "estimated"
    if scenarios is None:
        scenarios = {
            "lower_revenue_rate": {"annual_rate": contract.annual_rate * 0.8},
            "higher_loss_severity": {"lgd": min(1.0, contract.lgd * 1.2)},
            "higher_funding_cost": {
                "funding_rate": min(1.0, contract.funding_rate * 1.2)
            },
        }
    if not isinstance(scenarios, dict) or len(scenarios) > 20:
        raise ValueError("sensitivity requires at most 20 named scenarios")
    for name, changes in scenarios.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(changes, dict)
            or not changes
            or set(changes) - _SCENARIO_FIELDS
        ):
            raise ValueError("invalid sensitivity scenario")
        scenario = replace(contract, **changes)
        if scenario.exposure_basis == "offered_limit" and (
            scenario.conversion_rate is None or scenario.utilization_rate is None
        ):
            raise ValueError("sensitivity cannot omit conversion or utilization")
        value = _calculate(frame, scenario)
        profit = result["economics"]["profit"]
        result["sensitivity"].append(
            {
                "name": name,
                "changes": changes,
                "economics": value,
                "profit_delta": value["profit"] - profit,
                "profit_sign_reversed": profit * value["profit"] < 0,
                "effect_stage": "estimated",
                "interpretation": "fixed_condition_scenario",
            }
        )
    return result


def _calculate(frame, contract):
    scale = _SCALES[contract.exposure_unit]
    ead = frame[contract.ead_col] * scale
    cost = contract.operating_cost_per_loan * scale
    if contract.exposure_basis == "offered_limit":
        ead = ead * contract.conversion_rate * contract.utilization_rate
        # Per-loan operating cost is incurred only on converted loans.
        cost *= contract.conversion_rate
    measured = pricing_metrics(
        pd.Series(contract.annual_rate, index=frame.index, dtype=float),
        pd.Series(float("nan"), index=frame.index, dtype=float),
        ead=ead,
        pd=frame[contract.pd_col],
        lgd=contract.lgd,
        funding_rate=contract.funding_rate,
        term_months=contract.term_months,
        operating_cost_per_loan=cost,
    )["economics"]
    # Customer-level values are not needed for this reusable summary contract.
    return {key: value for key, value in measured.items() if key != "by_row"}
