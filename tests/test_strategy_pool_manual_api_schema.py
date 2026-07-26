"""Strict public API contracts for manual Strategy Pool operations."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import validate_strategy_request
from marvis.agent.turn_handlers import _MANUAL_STRATEGY_WORKFLOWS
from marvis.api_schemas import ManualStrategyRequest


ENTRY_A = "pool-entry-" + "a" * 32
ENTRY_B = "pool-entry-" + "b" * 32


def _request(workflow: str, workflow_inputs: dict) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": workflow,
        "workflow_inputs": workflow_inputs,
    }


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        ("strategy_pool_compile", {"strategy_type": "approval"}),
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "reject",
                "entry_id": ENTRY_A,
                "reason": "移除重复候选",
            },
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "pricing",
                "ordered_ids": [ENTRY_B, ENTRY_A],
                "reason": "按已确认优先级重排",
            },
        ),
    ],
)
def test_manual_pool_operation_accepts_only_user_owned_controls(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    request = ManualStrategyRequest.model_validate(
        _request(workflow, workflow_inputs),
        strict=True,
    )

    assert request.workflow == workflow
    assert request.workflow_inputs == workflow_inputs
    assert workflow in _MANUAL_STRATEGY_WORKFLOWS


@pytest.mark.parametrize(
    ("strategy_type", "action"),
    [
        ("approval", {"type": "approval"}),
        ("approval", {"type": "reject"}),
        ("approval", {"type": "review"}),
        ("reject", {"type": "approval"}),
        ("reject", {"type": "reject"}),
        ("reject", {"type": "review"}),
        ("limit", {"type": "limit", "value": 0}),
        ("limit", {"type": "limit", "value": 25_000.5}),
        ("pricing", {"type": "pricing", "value": 0}),
        ("pricing", {"type": "pricing", "value": 1.0}),
        ("segmentation", {"type": "segment", "value": "tier-a"}),
        ("segmentation", {"type": "segment", "value": 7}),
        ("segmentation", {"type": "segment", "value": 7.5}),
    ],
)
def test_manual_pool_set_action_accepts_only_type_compatible_typed_actions(
    strategy_type: str,
    action: dict,
) -> None:
    workflow_inputs = {
        "strategy_type": strategy_type,
        "entry_id": ENTRY_A,
        "action": action,
        "reason": "人工确认动作",
    }

    request = ManualStrategyRequest.model_validate(
        _request("strategy_pool_set_action", workflow_inputs),
        strict=True,
    )

    assert request.workflow_inputs == workflow_inputs
    assert request.workflow in _MANUAL_STRATEGY_WORKFLOWS


def test_manual_pool_set_action_shape_normalizes_through_existing_compiler() -> None:
    request = ManualStrategyRequest.model_validate(
        _request(
            "strategy_pool_set_action",
            {
                "strategy_type": "limit",
                "entry_id": ENTRY_A,
                "action": {"type": "limit", "value": 10_000},
            },
        ),
        strict=True,
    )

    result = validate_strategy_request(
        request.model_dump(mode="python"),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "strategy_type": "limit",
        "entry_id": ENTRY_A,
        "action": {
            "type": "limit",
            "value": 10_000,
            "reason_code": None,
            "stop": True,
        },
    }


@pytest.mark.parametrize(
    ("strategy_type", "action"),
    [
        ("approval", {"type": "limit", "value": 100}),
        ("reject", {"type": "segment", "value": "tier-a"}),
        ("limit", {"type": "reject"}),
        ("pricing", {"type": "limit", "value": 0.2}),
        ("segmentation", {"type": "pricing", "value": 0.2}),
    ],
)
def test_manual_pool_set_action_rejects_cross_type_actions(
    strategy_type: str,
    action: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(
                "strategy_pool_set_action",
                {
                    "strategy_type": strategy_type,
                    "entry_id": ENTRY_A,
                    "action": action,
                },
            ),
            strict=True,
        )


@pytest.mark.parametrize(
    ("strategy_type", "action"),
    [
        ("limit", {"type": "limit"}),
        ("limit", {"type": "limit", "value": -1}),
        ("limit", {"type": "limit", "value": math.inf}),
        ("limit", {"type": "limit", "value": math.nan}),
        ("pricing", {"type": "pricing"}),
        ("pricing", {"type": "pricing", "value": -0.01}),
        ("pricing", {"type": "pricing", "value": 1.01}),
        ("pricing", {"type": "pricing", "value": math.inf}),
        ("segmentation", {"type": "segment"}),
        ("segmentation", {"type": "segment", "value": ""}),
        ("segmentation", {"type": "segment", "value": "   "}),
        ("segmentation", {"type": "segment", "value": math.nan}),
    ],
)
def test_manual_pool_set_action_rejects_invalid_typed_values(
    strategy_type: str,
    action: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(
                "strategy_pool_set_action",
                {
                    "strategy_type": strategy_type,
                    "entry_id": ENTRY_A,
                    "action": action,
                },
            ),
            strict=True,
        )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "rule_id": "candidate-rule-" + "a" * 32},
        ),
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "approval",
                "entry_id": ENTRY_A,
                "revision": 4,
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "entry_id": ENTRY_A,
                "action": {"type": "reject"},
                "expected_content_hash": "a" * 64,
            },
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "approval",
                "ordered_ids": [ENTRY_A],
                "dataset_id": "forged",
            },
        ),
        (
            "strategy_pool_compile",
            {
                "strategy_type": "approval",
                "sample_design_ref": {"artifact_id": "forged"},
            },
        ),
        (
            "strategy_pool_compile",
            {
                "strategy_type": "approval",
                "requirements": [{"artifact_id": "forged"}],
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "entry_id": ENTRY_A,
                "action": {
                    "type": "reject",
                    "reason_code": "NOT_A_MANUAL_FORM_CONTROL",
                },
            },
        ),
    ],
)
def test_manual_pool_operations_reject_platform_owned_or_unsupported_fields(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        ("strategy_pool_remove_entry", {"strategy_type": "approval"}),
        (
            "strategy_pool_set_action",
            {"strategy_type": "approval", "entry_id": ENTRY_A},
        ),
        (
            "strategy_pool_set_action",
            {"strategy_type": "approval", "action": {"type": "reject"}},
        ),
        ("strategy_pool_reorder", {"strategy_type": "approval"}),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": []},
        ),
    ],
)
def test_manual_pool_operations_reject_partial_shapes(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )


def test_manual_pool_reorder_rejects_duplicate_entry_ids() -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(
                "strategy_pool_reorder",
                {
                    "strategy_type": "approval",
                    "ordered_ids": [ENTRY_A, ENTRY_A],
                },
            ),
            strict=True,
        )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "strategy_pool_remove_entry",
            {
                "strategy_type": "approval",
                "entry_id": ENTRY_A,
                "reason": None,
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "approval",
                "entry_id": ENTRY_A,
                "action": {"type": "reject"},
                "reason": None,
            },
        ),
        (
            "strategy_pool_reorder",
            {
                "strategy_type": "approval",
                "ordered_ids": [ENTRY_A],
                "reason": None,
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "limit",
                "entry_id": ENTRY_A,
                "action": {"type": "limit", "value": None},
            },
        ),
    ],
)
def test_manual_pool_operations_reject_explicit_null_optional_fields(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "strategy_pool_remove_entry",
            {"strategy_type": "approval", "entry_id": ENTRY_A, "reason": 7},
        ),
        (
            "strategy_pool_reorder",
            {"strategy_type": "approval", "ordered_ids": (ENTRY_A,)},
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "limit",
                "entry_id": ENTRY_A,
                "action": {"type": "limit", "value": "100"},
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "pricing",
                "entry_id": ENTRY_A,
                "action": {"type": "pricing", "value": True},
            },
        ),
        (
            "strategy_pool_set_action",
            {
                "strategy_type": "segmentation",
                "entry_id": ENTRY_A,
                "action": {"type": "segment", "value": False},
            },
        ),
    ],
)
def test_manual_pool_operations_reject_non_strict_types(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )
