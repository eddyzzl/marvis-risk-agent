"""Strict public manual API contract for adding one authenticated Pool source."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import validate_strategy_request
from marvis.agent.turn_handlers import _MANUAL_STRATEGY_WORKFLOWS
from marvis.api_schemas import ManualStrategyRequest


ASSET_ID = "candidate-asset-" + "a" * 32
SELECTION_IDS = (
    "automatic-tree-leaf-selection-" + "b" * 32,
    "interactive-tree-frontier-selection-" + "c" * 32,
    "interactive-tree-frontier-group-selection-" + "d" * 32,
    "cross-matrix-cell-selection-" + "e" * 32,
    "scorecard-cutoff-selection-" + "f" * 32,
)


def _request(inputs: dict) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize(
    ("strategy_type", "default_action", "action"),
    [
        ("approval", {"type": "approval"}, {"type": "reject"}),
        ("reject", {"type": "review"}, {"type": "reject"}),
        (
            "limit",
            {"type": "limit", "value": 0},
            {"type": "limit", "value": 50_000},
        ),
        (
            "pricing",
            {"type": "pricing", "value": 0.1},
            {"type": "pricing", "value": 0.24},
        ),
        (
            "segmentation",
            {"type": "segment", "value": "fallback"},
            {"type": "segment", "value": "tier-a"},
        ),
    ],
)
def test_manual_pool_add_accepts_only_minimal_typed_user_controls(
    strategy_type: str,
    default_action: dict,
    action: dict,
) -> None:
    inputs = {
        "candidate_asset_id": ASSET_ID,
        "strategy_type": strategy_type,
        "default_action": default_action,
        "action": action,
        "reason": "人工确认加入候选池",
    }

    request = ManualStrategyRequest.model_validate(_request(inputs), strict=True)

    assert request.workflow_inputs == inputs
    assert request.workflow in _MANUAL_STRATEGY_WORKFLOWS
    compiled = validate_strategy_request(
        request.model_dump(mode="python"),
        allowed_columns=(),
    )
    assert compiled.draft is not None
    normalized = compiled.draft.to_dict()["workflow_inputs"]
    assert normalized["strategy_type"] == strategy_type
    assert normalized["candidate_asset_id"] == ASSET_ID
    assert normalized["default_action"]["type"] == default_action["type"]
    assert normalized["action"]["type"] == action["type"]


@pytest.mark.parametrize("selection_id", SELECTION_IDS)
def test_manual_pool_add_accepts_each_supported_materialized_selection(
    selection_id: str,
) -> None:
    inputs = {
        "selection_id": selection_id,
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
    }

    request = ManualStrategyRequest.model_validate(_request(inputs), strict=True)

    assert request.workflow_inputs == inputs


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
        {
            "candidate_asset_id": ASSET_ID,
            "selection_id": SELECTION_IDS[0],
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
        {
            "candidate_asset_id": ASSET_ID,
            "strategy_type": "approval",
            "action": {"type": "reject"},
        },
        {
            "candidate_asset_id": ASSET_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
        },
    ],
)
def test_manual_pool_add_requires_one_source_and_both_actions(inputs: dict) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(_request(inputs), strict=True)


@pytest.mark.parametrize(
    ("strategy_type", "field", "action"),
    [
        ("approval", "default_action", {"type": "limit", "value": 100}),
        ("reject", "action", {"type": "segment", "value": "tier-a"}),
        ("limit", "action", {"type": "reject"}),
        ("pricing", "default_action", {"type": "pricing", "value": 1.01}),
        ("segmentation", "action", {"type": "segment", "value": math.nan}),
    ],
)
def test_manual_pool_add_rejects_incompatible_or_invalid_actions(
    strategy_type: str,
    field: str,
    action: dict,
) -> None:
    inputs = {
        "candidate_asset_id": ASSET_ID,
        "strategy_type": strategy_type,
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
    }
    if strategy_type == "limit":
        inputs["default_action"] = {"type": "limit", "value": 0}
        inputs["action"] = {"type": "limit", "value": 1}
    elif strategy_type == "pricing":
        inputs["default_action"] = {"type": "pricing", "value": 0.1}
        inputs["action"] = {"type": "pricing", "value": 0.2}
    elif strategy_type == "segmentation":
        inputs["default_action"] = {"type": "segment", "value": "fallback"}
        inputs["action"] = {"type": "segment", "value": "tier-a"}
    inputs[field] = action

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(_request(inputs), strict=True)


@pytest.mark.parametrize(
    "forbidden",
    [
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_hash",
        "dataset_id",
        "sample_design_ref",
        "revision",
        "workspace_revision",
        "requirements",
        "rule_id",
    ],
)
def test_manual_pool_add_rejects_every_platform_binding(forbidden: str) -> None:
    inputs = {
        "candidate_asset_id": ASSET_ID,
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
        forbidden: "forged",
    }

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(_request(inputs), strict=True)


def test_manual_pool_add_allows_reviewed_voting_placement_only_for_asset_pointer() -> None:
    for placement_mode in (
        "before_selected_members",
        "replace_selected_members",
    ):
        request = ManualStrategyRequest.model_validate(
            _request(
                {
                    "candidate_asset_id": ASSET_ID,
                    "strategy_type": "approval",
                    "default_action": {"type": "approval"},
                    "action": {"type": "reject"},
                    "placement_mode": placement_mode,
                }
            ),
            strict=True,
        )
        assert request.workflow_inputs["placement_mode"] == placement_mode

    for invalid in ("append", "", None, "before_all_members"):
        with pytest.raises(ValidationError):
            ManualStrategyRequest.model_validate(
                _request(
                    {
                        "candidate_asset_id": ASSET_ID,
                        "strategy_type": "approval",
                        "default_action": {"type": "approval"},
                        "action": {"type": "reject"},
                        "placement_mode": invalid,
                    }
                ),
                strict=True,
            )

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(
                {
                    "selection_id": SELECTION_IDS[0],
                    "strategy_type": "approval",
                    "default_action": {"type": "approval"},
                    "action": {"type": "reject"},
                    "placement_mode": "before_selected_members",
                }
            ),
            strict=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_asset_id", "candidate-asset-" + "A" * 32),
        ("selection_id", "interactive-tree-frontier-selection-" + "x" * 32),
        ("reason", None),
        ("strategy_type", 1),
        ("default_action", {"type": "reject", "reason_code": "FORGED"}),
        ("action", {"type": "reject", "stop": False}),
    ],
)
def test_manual_pool_add_rejects_malformed_non_strict_or_expanded_controls(
    field: str,
    value: object,
) -> None:
    inputs = {
        "candidate_asset_id": ASSET_ID,
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
    }
    if field == "selection_id":
        inputs.pop("candidate_asset_id")
    inputs[field] = value

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(_request(inputs), strict=True)
