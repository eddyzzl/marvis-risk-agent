"""Natural-language contract for independent Strategy Pool replay evidence."""

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
    def __init__(self, workflow_inputs: dict, *, workflow: str = "strategy_pool_validation"):
        self.workflow_inputs = workflow_inputs
        self.workflow = workflow
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": self.workflow,
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


def _compile(utterance: str, workflow_inputs: dict, *, workflow: str = "strategy_pool_validation"):
    llm = _PayloadLLM(workflow_inputs, workflow=workflow)
    return (
        compile_strategy_request(
            utterance,
            allowed_columns=("customer_id", "score"),
            target_col="bad",
            llm=llm,
        ),
        llm,
    )


@pytest.mark.parametrize(
    ("strategy_type", "partition", "utterance"),
    [
        (
            "approval",
            "validation",
            "对当前审批策略池执行 validation 独立样本回放验证",
        ),
        (
            "reject",
            "oot",
            "run independent replay validation for the current reject pool on OOT",
        ),
        (
            "reject",
            "validation",
            "在 validation 上验证当前拒绝策略池",
        ),
        (
            "approval",
            "validation",
            "在验证集上回放当前审批策略池",
        ),
        (
            "reject",
            "oot",
            "validate the current reject pool on OOT",
        ),
        (
            "limit",
            "validation",
            "在验证集上回放当前额度策略池",
        ),
        (
            "pricing",
            "oot",
            "在 OOT 上验证当前定价策略池",
        ),
        (
            "segmentation",
            "validation",
            "在验证集上复核当前分群策略池",
        ),
    ],
)
def test_pool_validation_accepts_only_explicit_type_and_independent_partition(
    strategy_type: str,
    partition: str,
    utterance: str,
) -> None:
    result, llm = _compile(
        utterance,
        {"strategy_type": strategy_type, "partition": partition},
    )

    assert "strategy_pool_validation" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "strategy_pool_validation"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_validation",
        "workflow_inputs": {
            "strategy_type": strategy_type,
            "partition": partition,
        },
    }
    assert "独立样本回放验证" in result.confirmation
    assert partition in result.confirmation
    assert "不会修改 Pool" in result.confirmation
    assert "不晋级、不采纳、不部署" in result.confirmation
    assert llm.calls[0]["prompt_version"] == 50
    assert "strategy_pool_validation" in llm.calls[0]["system_prompt"]
    assert "independent replay evidence" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({}, "strategy_type"),
        ({"strategy_type": "approval"}, "partition"),
        (
            {"strategy_type": "approval", "partition": "development"},
            "partition",
        ),
        (
            {"strategy_type": "approval", "partition": "train"},
            "partition",
        ),
    ],
)
def test_pool_validation_rejects_unsupported_user_controls(
    inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_validation",
            "workflow_inputs": inputs,
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


@pytest.mark.parametrize(
    "platform_field",
    [
        "pool_ref",
        "sample_design_ref",
        "population",
        "comparison_mode",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "artifact_id",
        "dataset_id",
        "workspace_revision",
        "target_col",
        "requirements",
        "requirements_hash",
        "metrics",
        "validation_status",
    ],
)
def test_pool_validation_rejects_platform_owned_inputs(
    platform_field: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_validation",
            "workflow_inputs": {
                "strategy_type": "approval",
                "partition": "validation",
                platform_field: "forged",
            },
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert platform_field in result.clarification


@pytest.mark.parametrize(
    ("utterance", "inputs", "code"),
    [
        (
            "不要对当前审批策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "能对当前审批策略池执行 validation 独立样本回放验证吗？",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "以后对当前审批策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "上一版审批策略池曾在 OOT 上做独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "审批策略池已经在 OOT 上完成独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "审批策略池在 OOT 完成了独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "审批策略池在 OOT 做了独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "历史上审批策略池在 OOT 做过独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "明天对当前审批策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "下次对当前审批策略池执行 OOT 独立样本回放验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "等会儿对当前审批策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "样本成熟后对当前审批策略池执行 OOT 独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "现在要对当前审批策略池执行 OOT 独立验证吗",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "可不可以对当前审批策略池执行 OOT 独立验证",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "the approval pool was already independently validated on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "tomorrow run independent replay validation for the current "
            "reject pool on OOT",
            {"strategy_type": "reject", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "should we run independent replay validation for the approval "
            "pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "can we validate the current approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "could we validate the current approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "do we need to validate the current approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "tell me whether to run independent replay validation for the "
            "approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "is it possible to validate the current approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "please explain how to replay the current approval pool on OOT",
            {"strategy_type": "approval", "partition": "oot"},
            "strategy_pool_validation_positive_command_required",
        ),
        (
            "对当前策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_controls_not_grounded",
        ),
        (
            "对当前审批和拒绝策略池执行 validation 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_controls_not_grounded",
        ),
        (
            "对当前审批策略池执行独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_controls_not_grounded",
        ),
        (
            "对当前审批策略池执行 validation 和 OOT 独立样本回放验证",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_controls_not_grounded",
        ),
        (
            "对当前审批策略池执行 development 独立样本回放验证",
            {"strategy_type": "approval", "partition": "development"},
            "invalid_strategy_request",
        ),
        (
            "对当前审批策略池执行 validation 独立样本回放验证并计算 PSI 稳定性",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_evidence_scope_forbidden",
        ),
        (
            "对当前审批策略池执行 validation 独立样本回放验证，pool_ref 用这个值",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_platform_binding_forbidden",
        ),
        (
            "对当前审批策略池执行 validation 独立样本回放验证并采纳",
            {"strategy_type": "approval", "partition": "validation"},
            "strategy_pool_validation_single_operation_required",
        ),
    ],
)
def test_pool_validation_rejects_ambiguous_or_expansive_commands(
    utterance: str,
    inputs: dict,
    code: str,
) -> None:
    result, _llm = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == code


def test_explicit_pool_validation_cannot_be_rerouted_by_llm() -> None:
    result, _llm = _compile(
        "对当前审批策略池执行 validation 独立样本回放验证",
        {"strategy_type": "approval"},
        workflow="strategy_pool_compile",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_validation_workflow_required"
