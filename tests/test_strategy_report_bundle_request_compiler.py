"""Natural-language compiler contract for StrategyReportBundle V2."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    utterance_targets_strategy_report_bundle_v2,
    validate_strategy_request,
)


class _PayloadLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_report_bundle_v2",
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
        allowed_columns=None,
        target_col=None,
        llm=_PayloadLLM(workflow_inputs),
    )


def test_report_command_uses_fixed_defaults_and_no_dataset_controls() -> None:
    utterance = "请生成当前策略迭代评审报告。"

    result = _compile(utterance, {})

    assert utterance_targets_strategy_report_bundle_v2(utterance) is True
    assert "strategy_report_bundle_v2" in STANDARD_STRATEGY_WORKFLOWS
    assert result.clarification is None
    assert result.draft.to_dict() == {
        "request_kind": "standard_workflow",
        "workflow": "strategy_report_bundle_v2",
        "workflow_inputs": {
            "title": "策略迭代评审报告",
            "status": "partial",
        },
    }
    assert "不会创建策略" in result.confirmation
    assert "不代表采纳或部署" in result.confirmation


def test_report_accepts_only_exact_grounded_title_and_status() -> None:
    result = _compile(
        "请生成当前审批策略评审报告，报告标题为《7月准入策略复盘》，状态 final。",
        {"title": "7月准入策略复盘", "status": "final"},
    )

    assert result.clarification is None
    assert result.draft.workflow_inputs == {
        "title": "7月准入策略复盘",
        "status": "final",
    }


@pytest.mark.parametrize(
    "utterance",
    [
        "不要生成当前策略评审报告。",
        "不生成当前策略评审报告。",
        "能否生成当前策略评审报告？",
        "要不要生成当前策略评审报告",
        "假设生成当前策略评审报告做演示。",
        "昨天生成了策略评审报告。",
        "已经生成当前策略评审报告。",
        "请生成当前策略评审报告，然后训练模型。",
        "请生成当前策略评审报告，同时评分。",
        "请生成当前策略评审报告并评分、采纳后部署上线。",
        "请生成当前策略评审报告，同时构建候选并测算影响。",
    ],
)
def test_report_negation_question_demo_history_or_chaining_clarifies(
    utterance: str,
) -> None:
    result = _compile(utterance, {})

    assert result.draft is None
    assert result.clarification_code in {
        "strategy_report_bundle_v2_intent_negated",
        "strategy_report_bundle_v2_positive_command_required",
        "strategy_report_bundle_v2_single_operation_required",
    }


def test_report_command_cannot_be_misrouted_to_generic_lifecycle() -> None:
    result = compile_strategy_request(
        "请生成当前策略迭代评审报告。",
        allowed_columns=None,
        target_col=None,
        llm=_RawPayloadLLM(
            {
                "operation": "report",
                "strategy_type": "approval",
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_report_bundle_v2_workflow_required"


@pytest.mark.parametrize(
    "utterance",
    [
        "生成已有审批策略报告",
        "生成分群策略报告",
        "generate an existing approval strategy report",
    ],
)
def test_report_bundle_target_does_not_hijack_existing_strategy_reports(
    utterance: str,
) -> None:
    assert utterance_targets_strategy_report_bundle_v2(utterance) is False


@pytest.mark.parametrize(
    ("utterance", "workflow_inputs", "fields"),
    [
        (
            "请生成当前策略评审报告。",
            {"title": "模型补写标题", "status": "final"},
            {"title", "status"},
        ),
        (
            "请生成当前策略评审报告，标题为《用户标题》，状态 final。",
            {"title": "另一标题", "status": "draft"},
            {"title", "status"},
        ),
    ],
)
def test_report_rejects_ungrounded_title_or_status(
    utterance: str,
    workflow_inputs: dict,
    fields: set[str],
) -> None:
    result = _compile(utterance, workflow_inputs)

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_controls_not_grounded"
    )
    assert set(result.clarification_fields) == fields


def test_report_rejects_every_llm_owned_platform_binding() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_report_bundle_v2",
            "workflow_inputs": {
                "title": "策略迭代评审报告",
                "status": "partial",
                "project_context_ref": {"artifact_id": "invented"},
                "report_revision": 99,
                "generated_at": "2026-07-23T00:00:00Z",
                "metrics": {"approval_rate": 0.99},
            },
        },
        allowed_columns=None,
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_platform_binding_forbidden"
    )
    assert set(result.clarification_fields) == {
        "generated_at",
        "metrics",
        "project_context_ref",
        "report_revision",
    }


def test_report_utterance_cannot_override_platform_cas() -> None:
    result = _compile(
        "请生成当前策略评审报告，report revision: 7。",
        {},
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_platform_binding_forbidden"
    )


def test_report_utterance_cannot_supply_metrics_or_strategy_identity() -> None:
    result = _compile(
        "请生成当前策略评审报告，strategy_id=strategy-1，通过率为 95%。",
        {},
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_platform_binding_forbidden"
    )
