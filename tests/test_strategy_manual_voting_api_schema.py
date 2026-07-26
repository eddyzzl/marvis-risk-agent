"""Narrow API-schema contracts for manual Voting Candidate Lab requests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from marvis.api_schemas import ManualStrategyRequest


RULE_A = "candidate-rule-" + "a" * 32
RULE_B = "candidate-rule-" + "b" * 32
RULE_C = "candidate-rule-" + "c" * 32
SEARCH_ID = "voting-search-" + "d" * 32
COMBO_ID = "voting-combo-" + "e" * 32


def _search_inputs() -> dict:
    return {
        "strategy_type": "approval",
        "member_count": 3,
        "n": 2,
        "objective": {
            "metric": "bad_rate",
            "direction": "minimize",
        },
        "constraints": [
            {
                "metric": "hit_share",
                "operator": "gte",
                "value": 0.1,
            },
            {
                "metric": "bad_capture_rate",
                "operator": "gte",
                "value": 0.4,
            },
        ],
        "include_rule_ids": [RULE_A],
        "exclude_rule_ids": [RULE_B],
        "max_combinations": 500,
    }


def _request(workflow: str, workflow_inputs: dict) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": workflow,
        "workflow_inputs": workflow_inputs,
    }


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        ("voting_candidate_search", _search_inputs()),
        (
            "voting_candidate_search",
            {
                "strategy_type": "segmentation",
                "member_count": 2,
                "n": 1,
                "objective": {
                    "metric": "lift",
                    "direction": "maximize",
                },
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
                "strategy_type": "approval",
            },
        ),
    ],
)
def test_manual_voting_schema_accepts_only_valid_user_owned_inputs(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    request = ManualStrategyRequest.model_validate(
        _request(workflow, workflow_inputs),
        strict=True,
    )

    assert request.request_kind == "standard_workflow"
    assert request.workflow == workflow
    assert request.workflow_inputs == workflow_inputs


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "voting_candidate_search",
            {**_search_inputs(), "artifact_id": "forged-artifact"},
        ),
        (
            "voting_candidate_search",
            {**_search_inputs(), "dataset_ref": {"dataset_id": "forged"}},
        ),
        (
            "voting_candidate_search",
            {**_search_inputs(), "expected_content_hash": "f" * 64},
        ),
        (
            "voting_candidate_search",
            {**_search_inputs(), "unknown_control": True},
        ),
        (
            "voting_candidate_search",
            {
                **_search_inputs(),
                "objective": {
                    "metric": "bad_rate",
                    "direction": "minimize",
                    "rank": 1,
                },
            },
        ),
        (
            "voting_candidate_search",
            {
                **_search_inputs(),
                "constraints": [
                    {
                        "metric": "hit_share",
                        "operator": "gte",
                        "value": 0.1,
                        "artifact_id": "forged-artifact",
                    }
                ],
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
                "source_artifact_id": "forged-artifact",
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
                "unknown_control": True,
            },
        ),
    ],
)
def test_manual_voting_schema_rejects_extra_or_platform_owned_fields(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )


@pytest.mark.parametrize(
    "workflow_inputs",
    [
        {
            **_search_inputs(),
            "n": 4,
        },
        {
            **_search_inputs(),
            "member_count": 2,
            "include_rule_ids": [RULE_A, RULE_B, RULE_C],
            "exclude_rule_ids": [],
        },
        {
            **_search_inputs(),
            "include_rule_ids": [RULE_A],
            "exclude_rule_ids": [RULE_A],
        },
        {
            **_search_inputs(),
            "constraints": [
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": 0.1,
                },
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": 0.2,
                },
            ],
        },
        {
            **_search_inputs(),
            "include_rule_ids": [RULE_A, RULE_A],
            "exclude_rule_ids": [],
        },
        {
            **_search_inputs(),
            "constraints": [
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": 1.01,
                },
            ],
        },
        {
            **_search_inputs(),
            "constraints": [
                {
                    "metric": "hit_share",
                    "operator": "gte",
                    "value": 0.0,
                },
            ],
        },
        {
            **_search_inputs(),
            "constraints": [
                {
                    "metric": "hit_count",
                    "operator": "gte",
                    "value": float("inf"),
                },
            ],
        },
    ],
)
def test_manual_voting_search_schema_rejects_conflicting_controls(
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request("voting_candidate_search", workflow_inputs),
            strict=True,
        )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "voting_candidate_search",
            {
                **_search_inputs(),
                "include_rule_ids": ["candidate-rule-" + "a" * 31],
            },
        ),
        (
            "voting_candidate_search",
            {
                **_search_inputs(),
                "exclude_rule_ids": ["candidate-rule-" + "A" * 32],
            },
        ),
        (
            "voting_candidate_search",
            {
                **_search_inputs(),
                "include_rule_ids": [SEARCH_ID],
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": "voting-search-" + "d" * 31,
                "combo_id": COMBO_ID,
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": "voting-combo-" + "E" * 32,
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": COMBO_ID,
                "combo_id": SEARCH_ID,
            },
        ),
        (
            "voting_candidate_build_from_search",
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
                "strategy_type": None,
            },
        ),
    ],
)
def test_manual_voting_schema_rejects_invalid_or_forged_ids(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            _request(workflow, workflow_inputs),
            strict=True,
        )


def test_manual_voting_request_envelope_forbids_extra_fields() -> None:
    payload = _request("voting_candidate_search", deepcopy(_search_inputs()))
    payload["model_id"] = "forged-model"

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)
