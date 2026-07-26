"""Natural-language compiler contract for current-Pool draft materialization."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _compile(
    utterance: str,
    workflow_inputs: dict,
    *,
    workflow: str = "strategy_pool_materialize",
):
    return compile_strategy_request(
        utterance,
        allowed_columns=(),
        target_col=None,
        llm=_PayloadLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": workflow,
                "workflow_inputs": workflow_inputs,
            }
        ),
    )


@pytest.mark.parametrize(
    ("strategy_type", "utterance"),
    [
        ("approval", "把当前审批策略池物化为 draft Strategy"),
        ("reject", "从当前拒绝策略池创建草案策略"),
        ("limit", "把当前额度策略池固化成 draft Strategy"),
        ("pricing", "materialize current pricing pool as a draft strategy"),
        ("segmentation", "create a draft strategy from the current segmentation pool"),
    ],
)
def test_pool_materialize_accepts_one_explicit_type_as_draft_only(
    strategy_type: str,
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": strategy_type})

    assert "strategy_pool_materialize" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "strategy_pool_materialize"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_materialize",
        "workflow_inputs": {"strategy_type": strategy_type},
    }
    assert "draft Strategy" in result.confirmation
    assert "不采纳" in result.confirmation
    assert "不部署" in result.confirmation


@pytest.mark.parametrize(
    "platform_field",
    [
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "expected_pool_artifact_id",
        "expected_pool_artifact_content_hash",
        "expected_design_hash",
        "pool_id",
        "artifact_id",
        "strategy_spec",
        "requirements",
        "metrics",
    ],
)
def test_pool_materialize_rejects_platform_owned_inputs(
    platform_field: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_materialize",
            "workflow_inputs": {
                "strategy_type": "approval",
                platform_field: "forged",
            },
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert platform_field in result.clarification


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "不要把当前审批策略池物化为 draft Strategy",
            "strategy_pool_materialize_positive_command_required",
        ),
        (
            "能把当前审批策略池物化为 draft Strategy 吗？",
            "strategy_pool_materialize_positive_command_required",
        ),
        (
            "之前把当前审批策略池物化为了 draft Strategy",
            "strategy_pool_materialize_positive_command_required",
        ),
        (
            "以后把当前审批策略池物化为 draft Strategy",
            "strategy_pool_materialize_positive_command_required",
        ),
        (
            "把当前策略池物化为 draft Strategy",
            "strategy_pool_materialize_controls_not_grounded",
        ),
        (
            "把当前审批和拒绝策略池物化为 draft Strategy",
            "strategy_pool_materialize_controls_not_grounded",
        ),
        (
            "把当前审批策略池物化为 draft Strategy，expected_design_hash="
            + "a" * 64,
            "strategy_pool_materialize_platform_binding_forbidden",
        ),
    ],
)
def test_pool_materialize_rejects_noncurrent_ambiguous_or_forged_commands(
    utterance: str,
    code: str,
) -> None:
    result = _compile(utterance, {"strategy_type": "approval"})

    assert result.draft is None
    assert result.clarification_code == code


@pytest.mark.parametrize(
    "follow_up",
    [
        "并采纳",
        "并部署",
        "并回测",
        "并生成报告",
        "并开始监控",
        "并导出 DSL",
    ],
)
def test_pool_materialize_must_be_the_only_operation(follow_up: str) -> None:
    result = _compile(
        f"把当前审批策略池物化为 draft Strategy，{follow_up}",
        {"strategy_type": "approval"},
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "strategy_pool_materialize_single_operation_required"
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "把当前审批策略池物化为 draft Strategy，不要采纳或部署",
        "materialize current approval pool as a draft strategy; "
        "do not adopt or deploy",
    ],
)
def test_pool_materialize_accepts_tightly_scoped_negative_lifecycle_disclaimer(
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": "approval"})

    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.workflow == "strategy_pool_materialize"


def test_explicit_pool_materialize_cannot_be_rerouted_by_llm() -> None:
    result = _compile(
        "把当前审批策略池物化为 draft Strategy",
        {"strategy_type": "approval"},
        workflow="strategy_pool_compile",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_materialize_workflow_required"
