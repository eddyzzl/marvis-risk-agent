from dataclasses import replace

import pytest

from marvis.business_acceptance import (
    BusinessCriterion,
    BusinessEvidence,
    BusinessObjective,
    acceptance_display_rows,
    evaluate_business_acceptance,
)
from marvis.orchestrator.evidence import payload_hash


def objective(**changes):
    return replace(
        BusinessObjective(
            business_line="consumer_credit",
            decision_node="approval",
            population="existing_customers",
            period_start="2026-01-01",
            period_end="2026-03-31",
            target_kind="model",
            responsibility_source="business-owner:review-17",
            target_id="model-b",
            target_version="v2",
            criteria=(
                BusinessCriterion("oot_ks", "ratio", "mature_loans", minimum=0.3),
            ),
        ),
        **changes,
    )


def evidence(**changes):
    return replace(
        BusinessEvidence(
            target_kind="model",
            target_id="model-b",
            target_version="v2",
            source_ref="metrics:selection:v1",
            source_hash=payload_hash({"oot_ks": 0.4}),
            metrics={"oot_ks": 0.4},
            metric_units={"oot_ks": "ratio"},
            denominators={"oot_ks": "mature_loans"},
            business_line="consumer_credit",
            decision_node="approval",
            population="existing_customers",
            period_start="2026-01-01",
            period_end="2026-03-31",
            effect_stage="oot_validated",
            labels_mature=True,
            label_origin="observed",
        ),
        **changes,
    )


def test_execution_is_not_an_input_to_business_acceptance_and_empty_goals_never_pass():
    assert evaluate_business_acceptance(None, evidence())["status"] == "not_configured"
    assert evaluate_business_acceptance(objective(), evidence())["status"] == "passed"
    assert (
        evaluate_business_acceptance(objective(), evidence(metrics={"oot_ks": 0.2}))[
            "status"
        ]
        == "failed"
    )
    assert (
        evaluate_business_acceptance(objective(), None)["status"]
        == "insufficient_evidence"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"target_id": "candidate-a"},
        {"target_version": "v1"},
        {"population": "other"},
        {"period_start": None},
        {"period_end": "2025-03-31"},
        {"labels_mature": False},
        {"label_origin": "reject_inferred"},
        {"label_origin": "assumed"},
        {"effect_stage": "estimated"},
        {"metric_units": {"oot_ks": "percent"}},
        {"denominators": {"oot_ks": "all_applicants"}},
        {"metrics": {"oot_ks": float("nan")}},
        {"metrics": {"oot_ks": float("inf")}},
        {"metrics": {"oot_ks": True}},
        {"business_line": None},
        {"business_line": "auto_finance"},
        {"decision_node": "collection"},
    ],
)
def test_mismatched_or_missing_evidence_never_passes(change):
    result = evaluate_business_acceptance(objective(), evidence(**change))
    assert result["status"] == "insufficient_evidence"
    assert result["reasons"]


def test_not_applicable_requires_explicit_contract_permission_and_reason():
    with pytest.raises(ValueError, match="permission"):
        objective(applicable=False)
    result = evaluate_business_acceptance(
        objective(
            applicable=False,
            allow_not_applicable=True,
            not_applicable_reason="探索任务不执行模型验收",
        ),
        None,
    )
    assert result["status"] == "not_applicable"


def test_observed_difference_is_not_a_causal_effect_without_identification():
    result = evaluate_business_acceptance(
        objective(claim="causal"), evidence(effect_stage="post_launch_observed")
    )
    assert result["status"] == "insufficient_evidence"
    assert any("因果" in reason for reason in result["reasons"])


def test_delta_requires_bound_baseline_and_does_not_choose_candidate_maximum():
    goal = objective(
        criteria=(
            BusinessCriterion(
                "oot_ks",
                "ratio",
                "mature_loans",
                minimum=0.05,
                comparison="delta",
                baseline_ref="champion-v1",
            ),
        )
    )
    assert (
        evaluate_business_acceptance(goal, evidence())["status"]
        == "insufficient_evidence"
    )
    result = evaluate_business_acceptance(
        goal,
        evidence(
            baseline_ref="champion-v1",
            baseline_metrics={"oot_ks": 0.37},
            baseline_binding_hash=payload_hash({"oot_ks": 0.37}),
            baseline_context=baseline_context(),
        ),
    )
    assert result["status"] == "failed"
    assert result["criteria"][0]["value"] == pytest.approx(0.03)
    assert acceptance_display_rows(result)[0]["status"] == "failed"


def test_contract_round_trip_and_strict_thresholds():
    goal = objective()
    assert BusinessObjective.from_dict(goal.to_dict()) == goal
    for value in (True, "0.3", float("inf"), float("nan")):
        with pytest.raises(ValueError):
            BusinessCriterion("oot_ks", "ratio", "mature_loans", minimum=value)
    with pytest.raises(ValueError):
        BusinessObjective.from_dict({**goal.to_dict(), "force_pass": True})


def baseline_context(**changes):
    return {
        "business_line": "consumer_credit",
        "decision_node": "approval",
        "population": "existing_customers",
        "period_start": "2026-01-01",
        "period_end": "2026-03-31",
        "currency": None,
        "metric_units": {"oot_ks": "ratio"},
        "denominators": {"oot_ks": "mature_loans"},
        **changes,
    }


def test_delta_overflow_cannot_pass_and_baseline_context_must_match():
    goal = objective(
        criteria=(
            BusinessCriterion(
                "oot_ks",
                "ratio",
                "mature_loans",
                minimum=0.1,
                comparison="delta",
                baseline_ref="base",
            ),
        )
    )
    measured = evidence(
        metrics={"oot_ks": 1e308},
        baseline_metrics={"oot_ks": -1e308},
        baseline_ref="base",
        baseline_binding_hash=payload_hash({}),
        baseline_context=baseline_context(),
    )
    assert (
        evaluate_business_acceptance(goal, measured)["status"]
        == "insufficient_evidence"
    )
    wrong = replace(
        measured,
        metrics={"oot_ks": 0.8},
        baseline_metrics={"oot_ks": 0.2},
        baseline_context=baseline_context(decision_node="collection"),
    )
    assert (
        evaluate_business_acceptance(goal, wrong)["status"] == "insufficient_evidence"
    )


def test_multiple_delta_metrics_use_their_own_units_and_denominators():
    goal = objective(
        criteria=(
            BusinessCriterion(
                "oot_ks",
                "ratio",
                "mature_loans",
                minimum=0.1,
                comparison="delta",
                baseline_ref="base",
            ),
            BusinessCriterion(
                "population",
                "count",
                "all_rows",
                minimum=10,
                comparison="delta",
                baseline_ref="base",
            ),
        )
    )
    units = {"oot_ks": "ratio", "population": "count"}
    denominators = {"oot_ks": "mature_loans", "population": "all_rows"}
    measured = evidence(
        metrics={"oot_ks": 0.6, "population": 120},
        metric_units=units,
        denominators=denominators,
        baseline_ref="base",
        baseline_binding_hash=payload_hash({}),
        baseline_metrics={"oot_ks": 0.3, "population": 100},
        baseline_context=baseline_context(
            metric_units=units, denominators=denominators
        ),
    )
    assert evaluate_business_acceptance(goal, measured)["status"] == "passed"


@pytest.mark.parametrize("field", ["target_kind", "minimum_effect_stage", "claim"])
def test_malformed_contract_enums_raise_value_error(field):
    with pytest.raises(ValueError):
        BusinessObjective.from_dict({**objective().to_dict(), field: []})
