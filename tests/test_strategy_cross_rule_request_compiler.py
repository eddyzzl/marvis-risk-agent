"""Typed and natural-language contracts for bounded 2D/3D Cross rule search."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


SEARCH_ID = "cross-rule-search-" + "a" * 32
RULE_ID = "cross-rule-" + "b" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply

    def complete(self, **_kwargs):
        return self.reply


def _search_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "features": ["age", "score", "income"],
        "dimension": 3,
        "constraints": {
            "min_lift": 1.5,
            "min_bad_count": 20,
            "max_hit_share": 0.3,
            "min_amount_lift": None,
        },
        "max_trials": 500,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_rule_search",
        "workflow_inputs": inputs,
    }


def _build_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "search_id": SEARCH_ID,
        "rule_id": RULE_ID,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_rule_candidate_build_from_search",
        "workflow_inputs": inputs,
    }


def test_cross_rule_search_validates_only_human_controls() -> None:
    result = validate_strategy_request(
        _search_payload(),
        allowed_columns=("age", "score", "income"),
        target_col="bad",
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _search_payload()
    assert "cross_rule_search" in STANDARD_STRATEGY_WORKFLOWS
    assert "2D/3D Cross 阈值规则搜索" in result.confirmation
    assert "不会自动选择" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "source_artifact_id",
        "expected_evidence_hash",
        "dataset_id",
        "target_col",
        "thresholds",
        "directions",
        "rule_id",
        "rank",
        "winner",
    ],
)
def test_cross_rule_search_rejects_platform_or_derived_fields(
    forbidden: str,
) -> None:
    result = validate_strategy_request(
        _search_payload(**{forbidden: "forged"}),
        allowed_columns=("age", "score", "income"),
    )

    assert result.draft is None
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "inputs",
    [
        _search_payload(dimension=4)["workflow_inputs"],
        _search_payload(max_trials=0)["workflow_inputs"],
        _search_payload(max_trials=5001)["workflow_inputs"],
        _search_payload(features=["age", "age"])["workflow_inputs"],
        _search_payload(
            constraints={
                "min_lift": -1.0,
                "min_bad_count": 20,
                "max_hit_share": 0.3,
                "min_amount_lift": None,
            }
        )["workflow_inputs"],
        _search_payload(
            constraints={
                "min_lift": 1.5,
                "min_bad_count": 20,
                "max_hit_share": 1.1,
                "min_amount_lift": None,
            }
        )["workflow_inputs"],
    ],
)
def test_cross_rule_search_rejects_out_of_contract_controls(
    inputs: dict,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_rule_search",
            "workflow_inputs": inputs,
        },
        allowed_columns=("age", "score", "income"),
    )

    assert result.draft is None


def test_cross_rule_search_compiles_exact_current_turn_controls() -> None:
    payload = _search_payload()
    result = compile_strategy_request(
        (
            "搜索 3D Cross 阈值规则：features=[age, score, income]，"
            "dimension=3，min_lift=1.5，min_bad_count=20，"
            "max_hit_share=0.3，min_amount_lift=null，max_trials=500。"
            "只搜索，不构建候选、不入池。"
        ),
        allowed_columns=("age", "score", "income"),
        target_col="bad",
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload


def test_cross_rule_search_rejects_llm_changed_constraint() -> None:
    payload = _search_payload()
    payload["workflow_inputs"]["constraints"]["min_lift"] = 2.0

    result = compile_strategy_request(
        (
            "搜索 3D Cross 阈值规则：features=[age, score, income]，"
            "dimension=3，min_lift=1.5，min_bad_count=20，"
            "max_hit_share=0.3，min_amount_lift=null，max_trials=500。"
        ),
        allowed_columns=("age", "score", "income"),
        llm=_FakeLLM(payload),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_rule_search_controls_not_grounded"


def test_cross_rule_search_cannot_route_to_cross_matrix_pair_search() -> None:
    result = compile_strategy_request(
        (
            "搜索 2D Cross 阈值规则：features=[age, score]，dimension=2，"
            "min_lift=1.2，min_bad_count=10，max_hit_share=0.4，"
            "min_amount_lift=null，max_trials=100。"
        ),
        allowed_columns=("age", "score"),
        llm=_FakeLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "cross_matrix_candidate_search",
                "workflow_inputs": {
                    "features": ["age", "score"],
                    "max_pairs": 1,
                },
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_rule_search_workflow_required"


def test_cross_rule_candidate_build_accepts_exact_ids_and_reason() -> None:
    payload = _build_payload(selection_reason="业务确认该规则用于后续验证。")
    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "cross_rule_candidate_build_from_search" in (
        STANDARD_STRATEGY_WORKFLOWS
    )
    assert SEARCH_ID in result.confirmation
    assert RULE_ID in result.confirmation


def test_cross_rule_candidate_build_rejects_heuristic_selection() -> None:
    result = compile_strategy_request(
        (
            f"从 {SEARCH_ID} 构建最好的一条 Cross 规则候选，"
            f"rule_id={RULE_ID}。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_build_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_rule_selection_explicit_ids_required"


def test_manual_cross_rule_requests_share_the_strict_contract() -> None:
    assert ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_rule_search",
            "workflow_inputs": _search_payload()["workflow_inputs"],
        }
    ).workflow == "cross_rule_search"
    assert ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_rule_candidate_build_from_search",
            "workflow_inputs": {
                "search_id": SEARCH_ID,
                "rule_id": RULE_ID,
                "selection_reason": "人工风险评审。",
            },
        }
    ).workflow == "cross_rule_candidate_build_from_search"

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "cross_rule_candidate_build_from_search",
                "workflow_inputs": {
                    "search_id": SEARCH_ID,
                    "rule_id": RULE_ID,
                    "rank": 1,
                },
            }
        )
