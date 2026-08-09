"""Characterize the foundation/delivery workflows before shadow deletion.

These tests intentionally import the new specs directly instead of registering
them in the global catalog.  The legacy compiler remains the comparison oracle
until the main migration integrates the specs and removes its shadow branches.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

import pytest

from marvis.agent.strategy_request_compiler import validate_strategy_request
from marvis.agent.strategy_workflows._foundation_delivery import (
    FOUNDATION_DELIVERY_SPECS,
    LEGACY_SAMPLE_DESIGN_REPLAY_SPEC,
    is_sample_design_v2_fresh_partition_selector,
    validate_sample_design_v2_replay_inputs,
)
from marvis.agent.strategy_workflows.contracts import (
    StrategyWorkflowPreparationContext,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
    deep_thaw,
)


_COLUMNS = (
    "sample_role",
    "customer_id",
    "apply_date",
    "apply_month",
    "weight",
    "loan_amount",
    "overdue_amount",
    "legacy_score",
    "channel",
)
_RESOLUTION_CONTEXT = StrategyWorkflowResolutionContext(
    allowed_columns=_COLUMNS,
    target_col="bad",
)


def _eq(column: str, value: object) -> dict[str, object]:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _project_context_inputs() -> dict[str, object]:
    return {
        "as_of": "2026-06-30",
        "scope": "自营存量提额项目",
        "business_context": {
            "project.background": "优化存量客户提额策略",
        },
        "explicit_unavailable": ["current.status_fields.economics"],
        "external_report_filenames": ["历史评审.xlsx"],
    }


def _sample_v2_inputs() -> dict[str, object]:
    return {
        "target_bad_value": 1,
        "drop_nan_labels": False,
        "relationship": "nested_same_cohort",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_role", "dev"),
                "validation": _eq("sample_role", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": None,
            "month_field": "apply_month",
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "legacy_score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
    }


def _legacy_sample_inputs() -> dict[str, object]:
    return {
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 30,
        "observation_window_status": "provided",
        "observation_start": "2026-01-01",
        "observation_end": "2026-04-30",
        "maturity_status": "confirmed_matured",
        "split_col": "sample_role",
        "development_values": ["dev"],
        "validation_values": ["valid"],
        "oot_values": ["oot"],
        "month_col": "apply_month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "drop_nan_labels": False,
    }


def _inputs_by_workflow() -> dict[str, dict[str, object]]:
    return {
        "strategy_project_context": _project_context_inputs(),
        "strategy_sample_design_v2": _sample_v2_inputs(),
        "strategy_model_evidence_v2": {},
        "strategy_dsl_delivery": {"strategy_id": "strategy-candidate_1"},
        "strategy_report_bundle_v2": {},
        "strategy_sample_design": _legacy_sample_inputs(),
    }


def _legacy_result(
    workflow_id: str,
    inputs: dict[str, object],
):
    return validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": workflow_id,
            "workflow_inputs": inputs,
        },
        allowed_columns=_COLUMNS,
        target_col="bad",
        allow_legacy_replay=workflow_id == "strategy_sample_design",
    )


def _specs_by_id():
    return {spec.workflow_id: spec for spec in FOUNDATION_DELIVERY_SPECS}


def test_specs_capture_current_exposure_requirements_and_templates() -> None:
    expected = {
        "strategy_project_context": (
            True,
            True,
            True,
            False,
            False,
            False,
            ("strategy_project_context",),
        ),
        "strategy_sample_design_v2": (
            True,
            True,
            True,
            True,
            True,
            True,
            ("strategy_sample_design_v2", "strategy_sample_design_v2_native"),
        ),
        "strategy_model_evidence_v2": (
            True,
            True,
            False,
            False,
            False,
            False,
            ("strategy_model_evidence_v2",),
        ),
        "strategy_dsl_delivery": (
            True,
            True,
            True,
            True,
            False,
            False,
            ("strategy_dsl_delivery",),
        ),
        "strategy_report_bundle_v2": (
            True,
            True,
            True,
            False,
            False,
            False,
            ("strategy_report_bundle_v2",),
        ),
        # Retained only so persisted V1 requests can replay.  It must never
        # return to fresh/manual exposure while V2 is the public sample surface.
        "strategy_sample_design": (
            False,
            True,
            False,
            True,
            True,
            True,
            ("strategy_sample_design",),
        ),
    }
    actual = {}
    for spec in FOUNDATION_DELIVERY_SPECS:
        requirements = spec.requirements
        assert not callable(requirements)
        actual[spec.workflow_id] = (
            spec.fresh,
            spec.replayable,
            spec.manual,
            requirements.dataset,
            requirements.target,
            requirements.complete_labels,
            spec.template_ids,
        )

    assert actual == expected
    assert LEGACY_SAMPLE_DESIGN_REPLAY_SPEC.fresh is False


@pytest.mark.parametrize(
    "workflow_id",
    [
        "strategy_project_context",
        "strategy_sample_design_v2",
        "strategy_model_evidence_v2",
        "strategy_dsl_delivery",
        "strategy_report_bundle_v2",
        "strategy_sample_design",
    ],
)
def test_validators_and_confirmations_match_the_legacy_compiler(
    workflow_id: str,
) -> None:
    inputs = _inputs_by_workflow()[workflow_id]
    legacy = _legacy_result(workflow_id, inputs)
    spec = _specs_by_id()[workflow_id]

    assert legacy.draft is not None
    assert spec.validator is not None
    assert spec.confirmation is not None
    normalized = spec.validator(inputs, _RESOLUTION_CONTEXT)

    assert normalized == legacy.draft.to_dict()["workflow_inputs"]
    assert spec.confirmation(normalized) == legacy.confirmation


def test_sample_v2_population_dto_normalizes_like_the_legacy_compiler() -> None:
    inputs = _sample_v2_inputs()
    inputs["approval_population"]["inclusion"] = {
        "match": "all",
        "conditions": [
            {"column": "channel", "operator": "eq", "value": "app"},
        ],
    }
    legacy = _legacy_result("strategy_sample_design_v2", inputs)
    validator = _specs_by_id()["strategy_sample_design_v2"].validator

    assert legacy.draft is not None
    assert validator is not None
    assert (
        validator(inputs, _RESOLUTION_CONTEXT)
        == legacy.draft.to_dict()["workflow_inputs"]
    )


def test_sample_v2_explicit_replay_adapter_preserves_historical_ast_surface() -> None:
    """Replay needs a separate mode while fresh intake stays bounded.

    A normalized persisted request can contain a raw population AST and a
    recursive partition AST.  The current StrategyWorkflowSpec validator
    signature has no replay-mode argument, so main migration must route replay
    to this explicit adapter before deleting the legacy branch.
    """

    inputs = _sample_v2_inputs()
    inputs["approval_population"]["inclusion"] = _eq("channel", "app")
    inputs["partitioning"]["selectors"]["development"] = {
        "op": "or",
        "args": [
            {
                "op": "and",
                "args": [_eq("sample_role", "dev"), _eq("channel", "app")],
            },
            _eq("customer_id", "customer-a"),
        ],
    }
    legacy = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": inputs,
        },
        allowed_columns=_COLUMNS,
        target_col="bad",
        allow_legacy_replay=True,
    )
    fresh_validator = _specs_by_id()["strategy_sample_design_v2"].validator

    assert legacy.draft is not None
    assert fresh_validator is not None
    with pytest.raises(StrategyWorkflowValidationError) as fresh_error:
        fresh_validator(inputs, _RESOLUTION_CONTEXT)
    assert fresh_error.value.code == (
        "strategy_sample_design_v2_population_dto_required"
    )
    assert (
        validate_sample_design_v2_replay_inputs(
            inputs,
            _RESOLUTION_CONTEXT,
        )
        == legacy.draft.to_dict()["workflow_inputs"]
    )


@pytest.mark.parametrize(
    ("workflow_id", "field", "value"),
    [
        ("strategy_project_context", "expected_revision", 1),
        ("strategy_sample_design_v2", "dataset_id", "dataset-user"),
        ("strategy_model_evidence_v2", "artifact_id", "artifact-user"),
        ("strategy_dsl_delivery", "dataset_ref", {}),
        ("strategy_report_bundle_v2", "generated_at", "user-time"),
        ("strategy_sample_design", "dataset_id", "dataset-user"),
    ],
)
def test_unknown_or_platform_fields_fail_like_the_legacy_compiler(
    workflow_id: str,
    field: str,
    value: object,
) -> None:
    inputs = _inputs_by_workflow()[workflow_id]
    inputs[field] = value
    legacy = _legacy_result(workflow_id, inputs)
    validator = _specs_by_id()[workflow_id].validator

    assert legacy.draft is None
    assert validator is not None
    with pytest.raises(StrategyWorkflowValidationError) as raised:
        validator(inputs, _RESOLUTION_CONTEXT)

    assert str(raised.value) == legacy.clarification
    assert raised.value.code == legacy.clarification_code
    assert raised.value.fields == legacy.clarification_fields


@pytest.mark.parametrize(
    ("workflow_id", "mutate"),
    [
        ("strategy_project_context", lambda value: value.update(as_of="2026-6-30")),
        (
            "strategy_sample_design_v2",
            lambda value: value.update(target_bad_value=True),
        ),
        (
            "strategy_dsl_delivery",
            lambda value: value.update(strategy_id="candidate-1"),
        ),
        (
            "strategy_report_bundle_v2",
            lambda value: value.update(status="published"),
        ),
        (
            "strategy_sample_design",
            lambda value: value.update(observation_end="2025-12-31"),
        ),
    ],
)
def test_domain_invalid_inputs_fail_like_the_legacy_compiler(
    workflow_id: str,
    mutate,
) -> None:
    inputs = _inputs_by_workflow()[workflow_id]
    mutate(inputs)
    legacy = _legacy_result(workflow_id, inputs)
    validator = _specs_by_id()[workflow_id].validator

    assert legacy.draft is None
    assert validator is not None
    with pytest.raises(StrategyWorkflowValidationError) as raised:
        validator(inputs, _RESOLUTION_CONTEXT)

    assert str(raised.value) == legacy.clarification
    assert raised.value.code == legacy.clarification_code
    assert raised.value.fields == legacy.clarification_fields


_PROJECT_EVIDENCE = {
    "expected_revision": 3,
    "expected_revision_id": "context-revision-3",
    "expected_state_hash": "context-state-3",
    "user_message_ref": {
        "message_id": "message-1",
        "content_hash": "a" * 64,
    },
}
_SAMPLE_IDENTITY_EVIDENCE = {
    "dataset_id": "dataset-1",
    "expected_dataset_content_hash": "b" * 64,
    "workspace_revision": 4,
    "workspace_generation": 2,
    "semantic_mapping_hash": "c" * 64,
    "target_col": "bad",
}
_SAMPLE_V2_COMPATIBILITY_EVIDENCE = {
    **_SAMPLE_IDENTITY_EVIDENCE,
    "scope": "strategy_development",
    "policy": {"diagnostic_severities": {}},
    "compatibility_performance_window_status": "provided",
    "compatibility_performance_window_days": 30,
    "compatibility_observation_window_status": "provided",
    "compatibility_observation_start": "2026-01-01",
    "compatibility_observation_end": "2026-04-30",
    "compatibility_maturity_status": "confirmed_matured",
    "compatibility_split_col": "sample_role",
    "compatibility_development_values": ["dev"],
    "compatibility_validation_values": ["valid"],
    "compatibility_oot_values": ["oot"],
    "compatibility_month_col": "apply_month",
    "compatibility_weight_col": "weight",
    "compatibility_loan_amount_col": "loan_amount",
    "compatibility_overdue_amount_col": "overdue_amount",
}
_MODEL_EVIDENCE = {
    "sample_design_ref": {"bundle_artifact_id": "sample-bundle"},
    "univariate_sources": [{"artifact_id": "candidate-1"}],
    "expected_registry_token": "d" * 64,
}
_DSL_EVIDENCE = {
    "strategy_ref": {"strategy_id": "strategy-candidate_1"},
    "dataset_ref": {
        "dataset_id": "dataset-1",
        "expected_content_hash": "b" * 64,
    },
    "workspace_ref": {"revision": 4},
    "maximum_equivalence_rows": 4096,
}
_REPORT_EVIDENCE = {
    "project_context_ref": {"artifact_id": "context-1"},
    "sample_design_ref": {"bundle_artifact_id": "sample-1"},
    "candidate_pool_ref": {"strategy_type": "approval"},
    "pool_validation_refs": [],
    "candidate_stability_ref": None,
    "pool_stability_ref": None,
    "voting_candidate_search_ref": None,
    "cross_candidate_search_ref": None,
    "cross_rule_search_ref": None,
    "pool_impact_ref": {"artifact_id": "impact-1"},
    "impact_cube_ref": None,
    "report_revision": 2,
    "previous_report_id": "report-1",
    "previous_report_content_hash": "e" * 64,
    "generated_at": "2026-08-01T00:00:00+00:00",
    "strategy_identity": {"strategy_id": "strategy-1"},
    "model_evidence_ref": None,
    "training_evidence_ref": None,
    "score_evidence_ref": None,
}


def _evidence_by_workflow() -> dict[str, dict[str, object]]:
    return {
        "strategy_project_context": deepcopy(_PROJECT_EVIDENCE),
        "strategy_sample_design_v2": deepcopy(_SAMPLE_V2_COMPATIBILITY_EVIDENCE),
        "strategy_model_evidence_v2": deepcopy(_MODEL_EVIDENCE),
        "strategy_dsl_delivery": deepcopy(_DSL_EVIDENCE),
        "strategy_report_bundle_v2": deepcopy(_REPORT_EVIDENCE),
        "strategy_sample_design": deepcopy(_SAMPLE_IDENTITY_EVIDENCE),
    }


@pytest.mark.parametrize(
    "workflow_id",
    [
        "strategy_project_context",
        "strategy_sample_design_v2",
        "strategy_model_evidence_v2",
        "strategy_dsl_delivery",
        "strategy_report_bundle_v2",
        "strategy_sample_design",
    ],
)
def test_preparers_bind_only_platform_evidence_and_preserve_legacy_slots(
    workflow_id: str,
) -> None:
    spec = _specs_by_id()[workflow_id]
    inputs = _inputs_by_workflow()[workflow_id]
    assert spec.validator is not None
    normalized = spec.validator(inputs, _RESOLUTION_CONTEXT)
    evidence = _evidence_by_workflow()[workflow_id]
    calls: list[tuple[str, object]] = []

    def bind(requested_workflow: str, frozen_inputs):
        calls.append((requested_workflow, frozen_inputs))
        assert isinstance(frozen_inputs, MappingProxyType)
        return evidence

    assert spec.preparer is not None
    prepared = spec.preparer(
        normalized,
        StrategyWorkflowPreparationContext(
            dataset_id="must-not-be-read-directly",
            drop_nan_labels=True,
            bind_workflow_evidence=bind,
        ),
    )
    slots = prepared.to_runtime_slots()

    assert len(calls) == 1
    called_workflow, frozen_inputs = calls[0]
    assert called_workflow == workflow_id
    assert deep_thaw(frozen_inputs) == normalized
    assert prepared.success_criteria == ()
    assert prepared.template_id in spec.template_ids
    assert all(slots[key] == value for key, value in evidence.items())
    if workflow_id == "strategy_project_context":
        assert slots == {
            **evidence,
            **normalized,
        }
    elif workflow_id == "strategy_sample_design":
        assert slots == {
            **evidence,
            **{
                key: value
                for key, value in normalized.items()
                if key != "drop_nan_labels"
            },
            "drop_nan_labels": True,
        }
    elif workflow_id == "strategy_sample_design_v2":
        assert prepared.template_id == "strategy_sample_design_v2"
        assert slots == {
            **evidence,
            **{
                key: value
                for key, value in normalized.items()
                if key != "drop_nan_labels"
            },
            "drop_nan_labels": True,
        }
    elif workflow_id == "strategy_report_bundle_v2":
        assert slots == {**evidence, **normalized}
    elif workflow_id in {
        "strategy_model_evidence_v2",
        "strategy_dsl_delivery",
    }:
        assert slots == evidence


def test_sample_v2_native_semantics_select_the_native_template() -> None:
    spec = _specs_by_id()["strategy_sample_design_v2"]
    inputs = _sample_v2_inputs()
    inputs["relationship"] = "parallel_time_cohorts"
    assert spec.validator is not None
    normalized = spec.validator(inputs, _RESOLUTION_CONTEXT)
    native_evidence = {
        **_SAMPLE_IDENTITY_EVIDENCE,
        "scope": "strategy_development",
        "policy": {"diagnostic_severities": {}},
    }
    assert spec.preparer is not None
    prepared = spec.preparer(
        normalized,
        StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_workflow_evidence=lambda _workflow, _inputs: native_evidence,
        ),
    )

    assert prepared.template_id == "strategy_sample_design_v2_native"


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (_eq("sample_role", "dev"), True),
        (
            {
                "op": "and",
                "args": [
                    _eq("sample_role", "dev"),
                    {"op": "is_not_null", "arg": {"column": "customer_id"}},
                ],
            },
            True,
        ),
        ({"op": "not", "arg": _eq("sample_role", "dev")}, False),
        (
            {
                "op": "and",
                "args": [
                    _eq("sample_role", "dev"),
                    {
                        "op": "or",
                        "args": [
                            _eq("channel", "app"),
                            _eq("customer_id", "a"),
                        ],
                    },
                ],
            },
            False,
        ),
        (
            {
                "op": "eq",
                "left": {"column": "sample_role"},
                "right": {"column": "channel"},
            },
            False,
        ),
    ],
)
def test_fresh_sample_v2_partition_shape_has_one_canonical_interface(
    selector: object,
    expected: bool,
) -> None:
    assert is_sample_design_v2_fresh_partition_selector(selector) is expected


@pytest.mark.parametrize("workflow_id", list(_inputs_by_workflow()))
def test_preparers_fail_closed_without_the_platform_binding_callback(
    workflow_id: str,
) -> None:
    spec = _specs_by_id()[workflow_id]
    assert spec.validator is not None
    assert spec.preparer is not None
    normalized = spec.validator(
        _inputs_by_workflow()[workflow_id],
        _RESOLUTION_CONTEXT,
    )

    with pytest.raises(StrategyWorkflowValidationError) as raised:
        spec.preparer(
            normalized,
            StrategyWorkflowPreparationContext(dataset_id="dataset-ignored"),
        )

    assert raised.value.code == "strategy_workflow_evidence_binding_required"


def test_platform_evidence_cannot_override_user_owned_slots() -> None:
    spec = _specs_by_id()["strategy_report_bundle_v2"]
    assert spec.validator is not None
    assert spec.preparer is not None
    normalized = spec.validator({}, _RESOLUTION_CONTEXT)

    with pytest.raises(StrategyWorkflowValidationError) as raised:
        spec.preparer(
            normalized,
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda _workflow, _inputs: {
                    **_REPORT_EVIDENCE,
                    "title": "平台覆盖用户标题",
                },
            ),
        )

    assert raised.value.code == "strategy_workflow_evidence_conflict"
    assert raised.value.fields == ("title",)


def test_prepared_slots_are_frozen_and_do_not_alias_callback_objects() -> None:
    evidence = deepcopy(_MODEL_EVIDENCE)
    spec = _specs_by_id()["strategy_model_evidence_v2"]
    assert spec.preparer is not None
    prepared = spec.preparer(
        {},
        StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_workflow_evidence=lambda _workflow, _inputs: evidence,
        ),
    )

    evidence["univariate_sources"][0]["artifact_id"] = "mutated"
    assert deep_thaw(prepared.slots)["univariate_sources"] == [
        {"artifact_id": "candidate-1"}
    ]
