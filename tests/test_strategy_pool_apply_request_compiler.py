"""Natural-language compiler contract for governed current-Pool application."""

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
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_apply",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


class _RawPayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _compile(utterance: str, workflow_inputs: dict):
    return compile_strategy_request(
        utterance,
        allowed_columns=("customer_id", "score"),
        target_col="bad",
        llm=_PayloadLLM(workflow_inputs),
    )


@pytest.mark.parametrize(
    ("strategy_type", "utterance"),
    [
        ("approval", "把当前审批策略池应用到当前样本"),
        ("reject", "把当前拒绝策略池写回当前样本"),
        ("limit", "把当前额度策略池应用到当前样本"),
        ("pricing", "把当前定价策略池写回当前样本"),
        ("segmentation", "apply current segmentation pool to the current sample"),
    ],
)
def test_pool_apply_accepts_each_explicit_type_as_one_reversible_step(
    strategy_type: str,
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": strategy_type})

    assert "strategy_pool_apply" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "strategy_pool_apply"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_apply",
        "workflow_inputs": {"strategy_type": strategy_type},
    }
    assert "不可变派生数据集" in result.confirmation
    assert "不激活" in result.confirmation
    assert "不采纳" in result.confirmation
    assert "不部署" in result.confirmation


def test_pool_apply_accepts_one_explicit_ascii_output_prefix() -> None:
    llm = _PayloadLLM(
        {"strategy_type": "approval", "output_prefix": "decision_"}
    )
    result = compile_strategy_request(
        "把当前审批策略池应用到当前样本，输出前缀 decision_",
        allowed_columns=("customer_id", "score"),
        target_col="bad",
        llm=llm,
    )

    assert result.clarification is None
    assert result.draft.workflow_inputs == {
        "strategy_type": "approval",
        "output_prefix": "decision_",
    }
    assert "decision_" in result.confirmation
    assert llm.calls[0]["prompt_version"] == 47
    assert "strategy_pool_apply" in llm.calls[0]["system_prompt"]
    assert "Pool revision/snapshot hash" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    "platform_field",
    [
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "pool_id",
        "pool_artifact_id",
        "artifact_id",
        "dataset_id",
        "dataset_content_hash",
        "sample_design_ref",
        "requirements",
        "requirements_hash",
        "strategy_spec",
        "design_hash",
        "action_counts",
        "activated",
        "adopted",
        "deployed",
    ],
)
def test_pool_apply_rejects_every_platform_owned_input(
    platform_field: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_apply",
            "workflow_inputs": {
                "strategy_type": "approval",
                platform_field: "forged",
            },
        },
        allowed_columns=("score",),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert platform_field in result.clarification


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({}, "strategy_type"),
        ({"strategy_type": "other"}, "strategy_type"),
        ({"strategy_type": "approval", "output_prefix": "../escape"}, "output_prefix"),
        ({"strategy_type": "approval", "output_prefix": "1decision_"}, "output_prefix"),
        ({"strategy_type": "approval", "output_prefix": ""}, "output_prefix"),
        ({"strategy_type": "approval", "output_prefix": None}, "output_prefix"),
        (
            {"strategy_type": "approval", "output_prefix": "x" * 49},
            "output_prefix",
        ),
    ],
)
def test_pool_apply_rejects_invalid_user_controls(
    inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_apply",
            "workflow_inputs": inputs,
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert message in result.clarification


@pytest.mark.parametrize(
    ("utterance", "inputs", "code"),
    [
        (
            "不要把当前审批策略池应用到当前样本",
            {"strategy_type": "approval"},
            "strategy_pool_apply_positive_command_required",
        ),
        (
            "能把当前审批策略池应用到当前样本吗？",
            {"strategy_type": "approval"},
            "strategy_pool_apply_positive_command_required",
        ),
        (
            "之前把当前审批策略池应用到了当前样本",
            {"strategy_type": "approval"},
            "strategy_pool_apply_positive_command_required",
        ),
        (
            "把当前策略池应用到当前样本",
            {"strategy_type": "approval"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批和拒绝策略池应用到当前样本",
            {"strategy_type": "approval"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本，输出前缀 safe_",
            {"strategy_type": "approval", "output_prefix": "invented_"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本，输出前缀 safe_",
            {"strategy_type": "approval"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本",
            {"strategy_type": "approval", "output_prefix": "invented_"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本，output_prefix=../escape",
            {"strategy_type": "approval"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本，output_prefix=safe-evil",
            {"strategy_type": "approval", "output_prefix": "safe"},
            "strategy_pool_apply_controls_not_grounded",
        ),
        (
            "把当前审批策略池应用到当前样本，expected_pool_revision=7",
            {"strategy_type": "approval"},
            "strategy_pool_apply_platform_binding_forbidden",
        ),
    ],
)
def test_pool_apply_rejects_noncurrent_ambiguous_or_ungrounded_commands(
    utterance: str,
    inputs: dict,
    code: str,
) -> None:
    result = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == code


@pytest.mark.parametrize(
    "follow_up",
    [
        "并采纳",
        "并部署",
        "并激活派生数据集",
        "并导出",
        "再删除一条 Pool 规则",
        "然后修改策略池",
        "并上线",
    ],
)
def test_pool_apply_requires_application_to_be_the_only_operation(
    follow_up: str,
) -> None:
    result = _compile(
        f"把当前审批策略池应用到当前样本，{follow_up}",
        {"strategy_type": "approval"},
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_apply_single_operation_required"


def test_explicit_pool_apply_cannot_be_rerouted_by_llm() -> None:
    result = compile_strategy_request(
        "把当前审批策略池应用到当前样本",
        allowed_columns=("score",),
        target_col="bad",
        llm=_RawPayloadLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_compile",
                "workflow_inputs": {"strategy_type": "approval"},
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_apply_workflow_required"
