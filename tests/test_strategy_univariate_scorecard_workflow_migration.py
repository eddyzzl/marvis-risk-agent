from __future__ import annotations

from types import MappingProxyType

import pytest

from marvis.agent.strategy_workflows import (
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    is_migrated_workflow,
    migrated_workflow_requirements,
    prepare_strategy_plan,
    resolve_strategy_request,
)
from marvis.agent.strategy_workflows._catalog import build_catalog


FAMILY_IDS = (
    "univariate_candidate_analysis",
    "univariate_candidate_refinement",
    "candidate_monthly_stability",
    "scorecard_model_score_evidence_build",
    "scorecard_band_build",
    "scorecard_cutoff_selection",
)


def _fresh_refinement_inputs() -> dict[str, object]:
    return {
        "feature": "score",
        "method": "equal_width",
        "selection": {
            "risk_threshold": {"operator": ">=", "value": 0.2}
        },
    }


def _source_refinement_inputs() -> dict[str, object]:
    return {
        "source_candidate_id": "candidate-" + "a" * 32,
        "feature": "score",
        "method": "equal_width",
        "selection": {"source_bin_ids": ["regular:1"]},
    }


def test_six_family_specs_are_fully_migrated_with_declared_templates() -> None:
    specs = {spec.workflow_id: spec for spec in build_catalog()}

    assert all(is_migrated_workflow(workflow_id) for workflow_id in FAMILY_IDS)
    assert {
        workflow_id: specs[workflow_id].template_ids
        for workflow_id in FAMILY_IDS
    } == {
        "univariate_candidate_analysis": (
            "strategy_univariate_candidate_analysis",
        ),
        "univariate_candidate_refinement": (
            "strategy_univariate_candidate_refinement",
            "strategy_univariate_candidate_refinement_existing",
        ),
        "candidate_monthly_stability": (
            "strategy_candidate_monthly_stability",
        ),
        "scorecard_model_score_evidence_build": (
            "strategy_scorecard_model_score_evidence_build",
        ),
        "scorecard_band_build": ("strategy_scorecard_band_build",),
        "scorecard_cutoff_selection": (
            "strategy_scorecard_cutoff_selection",
        ),
    }


def test_refinement_requirements_resolve_from_normalized_inputs() -> None:
    assert migrated_workflow_requirements(
        "univariate_candidate_refinement",
        _fresh_refinement_inputs(),
    ) == (True, True, True)
    assert migrated_workflow_requirements(
        "univariate_candidate_refinement",
        _source_refinement_inputs(),
    ) == (False, False, False)

    with pytest.raises(RuntimeError, match="normalized workflow_inputs"):
        migrated_workflow_requirements("univariate_candidate_refinement")
    with pytest.raises(RuntimeError, match="normalized inputs"):
        migrated_workflow_requirements(
            "univariate_candidate_refinement",
            {},
        )


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            _fresh_refinement_inputs(),
            StrategyWorkflowRequirements(True, True, True),
        ),
        (
            _source_refinement_inputs(),
            StrategyWorkflowRequirements(False, False, False),
        ),
    ],
)
def test_resolved_refinement_carries_its_input_specific_requirements(
    inputs: dict[str, object],
    expected: StrategyWorkflowRequirements,
) -> None:
    resolved = resolve_strategy_request(
        "univariate_candidate_refinement",
        inputs,
        context=StrategyWorkflowResolutionContext(
            allowed_columns=("score",),
            target_col="bad",
        ),
    )

    assert resolved.requirements == expected
    assert isinstance(resolved.workflow_inputs, MappingProxyType)


def test_evidence_binding_is_required_copied_and_deep_frozen() -> None:
    inputs = {
        "features": ["score"],
        "seed": 42,
        "max_iter": 100,
        "scorecard_max_bins": 8,
    }
    with pytest.raises(StrategyWorkflowValidationError) as missing:
        prepare_strategy_plan(
            "scorecard_model_score_evidence_build",
            inputs,
            context=StrategyWorkflowPreparationContext(dataset_id=None),
        )
    assert missing.value.code == "strategy_workflow_evidence_binding_required"

    mutable_evidence = {
        "sample_design_ref": {"bundle_chain": ["bundle-1"]}
    }
    seen_inputs: list[MappingProxyType] = []

    def bind(_workflow_id, normalized_inputs):
        seen_inputs.append(normalized_inputs)
        return mutable_evidence

    prepared = prepare_strategy_plan(
        "scorecard_model_score_evidence_build",
        inputs,
        context=StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_workflow_evidence=bind,
        ),
    )
    mutable_evidence["sample_design_ref"]["bundle_chain"].append("mutated")

    assert isinstance(seen_inputs[0], MappingProxyType)
    assert prepared.to_runtime_slots()["sample_design_ref"] == {
        "bundle_chain": ["bundle-1"]
    }


def test_evidence_binding_cannot_override_canonical_user_slots() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as conflict:
        prepare_strategy_plan(
            "scorecard_model_score_evidence_build",
            {
                "features": ["score"],
                "seed": 42,
                "max_iter": 100,
                "scorecard_max_bins": 8,
            },
            context=StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda *_args: {
                    "features": ["platform-overwrite"]
                },
            ),
        )

    assert conflict.value.code == "strategy_workflow_evidence_conflict"
    assert conflict.value.fields == ("features",)
