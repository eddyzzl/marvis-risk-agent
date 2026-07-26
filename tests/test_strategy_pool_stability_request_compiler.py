"""Natural-language contract for current-Pool cross-partition stability."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    utterance_targets_strategy_pool_stability,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(
        self,
        workflow_inputs: dict,
        *,
        workflow: str = "strategy_pool_stability",
    ) -> None:
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


def _compile(
    utterance: str,
    workflow_inputs: dict,
    *,
    workflow: str = "strategy_pool_stability",
):
    llm = _PayloadLLM(workflow_inputs, workflow=workflow)
    return (
        compile_strategy_request(
            utterance,
            allowed_columns=("apply_month", "channel", "score"),
            target_col="bad",
            llm=llm,
        ),
        llm,
    )


@pytest.mark.parametrize(
    ("strategy_type", "utterance"),
    [
        ("approval", "测量当前 approval 审批策略池的跨分区 PSI 稳定性"),
        ("reject", "分析当前 reject 拒绝策略池的跨样本稳定性"),
        ("limit", "测量当前 limit 额度策略池的分布漂移"),
        ("pricing", "measure current pricing pool cross-partition stability"),
        (
            "segmentation",
            "calculate PSI stability for the current segmentation pool",
        ),
    ],
)
def test_pool_stability_accepts_all_five_explicit_current_pool_types(
    strategy_type: str,
    utterance: str,
) -> None:
    result, llm = _compile(utterance, {"strategy_type": strategy_type})

    assert "strategy_pool_stability" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "strategy_pool_stability"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_stability",
        "workflow_inputs": {"strategy_type": strategy_type},
    }
    assert "跨分区稳定性" in result.confirmation
    assert "development" in result.confirmation
    assert "validation/OOT" in result.confirmation
    assert "PSI" in result.confirmation
    assert "不会修改 Pool" in result.confirmation
    assert "不采纳、不晋级、不部署" in result.confirmation
    assert "strategy_pool_stability" in llm.calls[0]["system_prompt"]
    assert "exact ImpactCube" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({}, "strategy_type"),
        ({"strategy_type": "unknown"}, "strategy_type"),
        ({"strategy_type": "approval", "partitions": ["oot"]}, "partitions"),
        ({"strategy_type": "approval", "artifact_id": "a" * 64}, "artifact_id"),
        (
            {
                "strategy_type": "approval",
                "impact_cube_ref": {"artifact_id": "a" * 64},
            },
            "impact_cube_ref",
        ),
        ({"strategy_type": "approval", "psi_threshold": 0.25}, "psi_threshold"),
        ({"strategy_type": "approval", "metrics": {"psi": 0.1}}, "metrics"),
    ],
)
def test_pool_stability_rejects_missing_or_platform_owned_controls(
    inputs: dict,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_stability",
            "workflow_inputs": inputs,
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


@pytest.mark.parametrize(
    ("utterance", "inputs", "expected_code"),
    [
        (
            "不要测量当前审批策略池的跨分区 PSI 稳定性",
            {"strategy_type": "approval"},
            "strategy_pool_stability_positive_command_required",
        ),
        (
            "当前审批策略池的跨分区 PSI 稳定性怎么样？",
            {"strategy_type": "approval"},
            "strategy_pool_stability_positive_command_required",
        ),
        (
            "昨天测量过当前审批策略池的跨分区 PSI 稳定性",
            {"strategy_type": "approval"},
            "strategy_pool_stability_positive_command_required",
        ),
        (
            "明天测量当前审批策略池的跨分区 PSI 稳定性",
            {"strategy_type": "approval"},
            "strategy_pool_stability_positive_command_required",
        ),
        (
            "测量当前审批策略池的跨分区 PSI 稳定性并采纳策略",
            {"strategy_type": "approval"},
            "strategy_pool_stability_single_operation_required",
        ),
        (
            "测量当前审批策略池的跨分区 PSI 稳定性",
            {"strategy_type": "pricing"},
            "strategy_pool_stability_controls_not_grounded",
        ),
        (
            "测量当前审批和拒绝策略池的跨分区 PSI 稳定性",
            {"strategy_type": "approval"},
            "strategy_pool_stability_controls_not_grounded",
        ),
    ],
)
def test_pool_stability_requires_one_grounded_positive_operation(
    utterance: str,
    inputs: dict,
    expected_code: str,
) -> None:
    result, _llm = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == expected_code


def test_explicit_pool_stability_cannot_be_misrouted_to_impact_cube() -> None:
    result, _llm = _compile(
        "测量当前 pricing 定价策略池的跨分区 PSI 稳定性",
        {"strategy_type": "pricing"},
        workflow="strategy_impact_cube",
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_stability_workflow_required"


def test_pool_stability_reservation_is_disjoint_from_other_measurements() -> None:
    assert utterance_targets_strategy_pool_stability(
        "测量当前 pricing 定价策略池的跨分区 PSI 稳定性"
    )
    assert not utterance_targets_strategy_pool_stability(
        "分析 candidate-rule-1234567890abcdef1234567890abcdef 的逐月稳定性"
    )
    assert not utterance_targets_strategy_pool_stability(
        "测算当前 pricing 定价策略池的统一影响"
    )
    assert not utterance_targets_strategy_pool_stability(
        "生成当前 pricing 定价策略池的稳定性报告"
    )
