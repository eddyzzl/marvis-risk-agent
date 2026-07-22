"""Natural-language compiler contract for read-only Strategy Pool impact."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_impact",
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
        allowed_columns=("month", "loan_amount", "overdue_amount", "other_month"),
        target_col="bad",
        llm=_PayloadLLM(workflow_inputs),
    )


def test_pool_impact_absolute_accepts_exact_optional_columns_and_nan_authorization() -> None:
    result = _compile(
        "请测算审批策略池影响；月份列 month，放款金额列 loan_amount，"
        "逾期金额列 overdue_amount，并明确允许丢弃空标签。",
        {
            "strategy_type": "approval",
            "month_col": "month",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": True,
        },
    )

    assert "strategy_pool_impact" in STANDARD_STRATEGY_WORKFLOWS
    assert result.clarification is None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "strategy_type": "approval",
        "comparison_mode": "absolute",
        "month_col": "month",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "drop_nan_labels": True,
    }
    assert "只读影响证据" in result.confirmation
    assert "不会创建、修改、采纳或部署策略" in result.confirmation
    assert "保留样本行、仅从风险分母排除" in result.confirmation


def test_pool_impact_accepts_calculate_as_positive_english_command() -> None:
    result = _compile(
        "calculate approval pool impact",
        {"strategy_type": "approval"},
    )

    assert result.clarification is None
    assert result.draft is not None
    assert result.draft.workflow == "strategy_pool_impact"


def test_pool_impact_vs_baseline_requires_the_exact_user_id() -> None:
    baseline_id = "strategy-baseline-001"
    result = _compile(
        f"测算拒绝策略池相对基线策略 {baseline_id} 的影响。",
        {
            "strategy_type": "reject",
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": baseline_id,
        },
    )

    assert result.clarification is None
    assert result.draft.workflow_inputs["baseline_strategy_id"] == baseline_id
    assert result.draft.workflow_inputs["drop_nan_labels"] is False


@pytest.mark.parametrize(
    ("utterance", "inputs", "missing"),
    [
        (
            "测算审批策略池相对基线 strategy-real 的影响",
            {
                "strategy_type": "approval",
                "comparison_mode": "vs_baseline",
                "baseline_strategy_id": "strategy-invented",
            },
            "strategy-invented",
        ),
        (
            "测算审批策略池影响，月份列 month",
            {"strategy_type": "approval", "month_col": "other_month"},
            "month_col other_month",
        ),
        (
            "测算拒绝策略池影响",
            {"strategy_type": "approval"},
            "strategy_type approval",
        ),
    ],
)
def test_pool_impact_rejects_ungrounded_id_column_and_type(
    utterance: str,
    inputs: dict,
    missing: str,
) -> None:
    result = _compile(utterance, inputs)

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert missing in result.clarification_fields


@pytest.mark.parametrize(
    "utterance",
    [
        "不要测算审批策略池影响。",
        "do not calculate approval pool impact.",
        "昨天测算过审批策略池影响吗？",
        "只生成审批策略池影响报告即可。",
        "测算审批策略池影响，然后部署。",
    ],
)
def test_pool_impact_negated_question_report_only_or_multi_operation_clarifies(
    utterance: str,
) -> None:
    result = _compile(utterance, {"strategy_type": "approval"})

    assert result.draft is None
    assert result.clarification_code in {
        "strategy_pool_impact_positive_command_required",
        "strategy_pool_impact_single_operation_required",
    }


def test_pool_impact_multi_operation_cannot_be_misrouted_to_another_workflow() -> None:
    result = compile_strategy_request(
        "先测算审批策略池影响，然后编译策略池。",
        allowed_columns=("bad",),
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
    assert result.clarification_code == "strategy_pool_impact_workflow_required"


def test_pool_impact_rejects_llm_owned_platform_fields_and_staged_types() -> None:
    forbidden = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_impact",
            "workflow_inputs": {
                "strategy_type": "approval",
                "dataset_id": "dataset-invented",
                "target_col": "bad",
                "metrics": {"approval_rate": 0.9},
            },
        },
        allowed_columns=("month",),
    )
    staged = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_impact",
            "workflow_inputs": {"strategy_type": "pricing"},
        },
        allowed_columns=("month",),
    )

    assert forbidden.draft is None
    assert "dataset_id" in forbidden.clarification
    assert staged.draft is None
    assert "V2" in staged.clarification
    assert "pricing" in staged.clarification


def test_pool_impact_negated_nan_drop_cannot_authorize_true() -> None:
    result = _compile(
        "请测算审批策略池影响，但不丢弃空标签。",
        {"strategy_type": "approval", "drop_nan_labels": True},
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert "drop_nan_labels=true" in result.clarification_fields


@pytest.mark.parametrize("selected_type", ["approval", "reject"])
def test_pool_impact_conflicting_pool_types_require_clarification(
    selected_type: str,
) -> None:
    result = _compile(
        "请测算审批策略池和拒绝策略池的影响。",
        {"strategy_type": selected_type},
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert f"strategy_type {selected_type}" in result.clarification_fields


@pytest.mark.parametrize("selected_id", ["strategy-base-a", "strategy-base-b"])
def test_pool_impact_alternative_baseline_ids_require_clarification(
    selected_id: str,
) -> None:
    result = _compile(
        "请测算审批策略池相对基线 strategy-base-a 或 strategy-base-b 的影响。",
        {
            "strategy_type": "approval",
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": selected_id,
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert selected_id in result.clarification_fields


def test_pool_impact_negated_baseline_is_rejected_and_positive_replacement_is_used() -> None:
    utterance = (
        "请测算审批策略池相对基线的影响，不要用 strategy-base-a，"
        "用 strategy-base-b。"
    )
    rejected = _compile(
        utterance,
        {
            "strategy_type": "approval",
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": "strategy-base-a",
        },
    )
    accepted = _compile(
        utterance,
        {
            "strategy_type": "approval",
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": "strategy-base-b",
        },
    )

    assert rejected.draft is None
    assert rejected.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert "strategy-base-a" in rejected.clarification_fields
    assert accepted.clarification is None
    assert accepted.draft.workflow_inputs["baseline_strategy_id"] == "strategy-base-b"


def test_negated_impact_followed_by_compile_is_not_hijacked_by_impact_guard() -> None:
    result = compile_strategy_request(
        "不要回测审批策略池，只编译预览。",
        allowed_columns=("bad",),
        target_col="bad",
        llm=_RawPayloadLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_compile",
                "workflow_inputs": {"strategy_type": "approval"},
            }
        ),
    )

    assert result.clarification is None
    assert result.draft.workflow == "strategy_pool_compile"


def test_pool_rate_calculation_forces_impact_workflow_guard() -> None:
    result = compile_strategy_request(
        "计算审批策略池的通过率和坏账率。",
        allowed_columns=("bad",),
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
    assert result.clarification_code == "strategy_pool_impact_workflow_required"


def test_pool_impact_cannot_omit_an_explicit_optional_column_binding() -> None:
    result = _compile(
        "请测算审批策略池影响，月份列 month。",
        {"strategy_type": "approval"},
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert "month_col month" in result.clarification_fields


def test_pool_impact_cannot_swap_explicit_loan_and_overdue_columns() -> None:
    result = _compile(
        "请测算审批策略池影响，放款金额列 loan_amount，"
        "逾期金额列 overdue_amount。",
        {
            "strategy_type": "approval",
            "loan_amount_col": "overdue_amount",
            "overdue_amount_col": "loan_amount",
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_impact_controls_not_grounded"
    assert "loan_amount_col loan_amount" in result.clarification_fields
    assert "overdue_amount_col overdue_amount" in result.clarification_fields
