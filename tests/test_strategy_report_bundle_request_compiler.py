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
    "strategy_type",
    ["审批", "拒绝", "额度", "定价", "分群"],
)
def test_report_accepts_all_five_types_without_exposing_platform_bindings(
    strategy_type: str,
) -> None:
    result = _compile(
        f"请生成当前{strategy_type}策略迭代评审报告。",
        {},
    )

    assert result.clarification is None
    assert result.draft.workflow_inputs == {
        "title": "策略迭代评审报告",
        "status": "partial",
    }


@pytest.mark.parametrize(
    ("utterance", "status"),
    [
        ("status=final，请生成当前审批策略评审报告。", "final"),
        ("请生成当前审批策略评审报告，设为 draft。", "draft"),
        ("请生成阶段性审批策略评审报告。", "partial"),
    ],
)
def test_report_accepts_explicit_assignment_or_clear_positive_status_phrase(
    utterance: str,
    status: str,
) -> None:
    result = _compile(utterance, {"status": status})

    assert result.clarification is None
    assert result.draft.workflow_inputs["status"] == status


@pytest.mark.parametrize(
    ("utterance", "workflow_inputs"),
    [
        (
            "请生成当前审批策略评审报告，标题为《最终版拒绝 Pool 复盘》。",
            {"title": "最终版拒绝 Pool 复盘", "status": "final"},
        ),
        (
            "请生成当前审批策略评审报告，不要 final 状态。",
            {"status": "final"},
        ),
        (
            "请生成当前审批策略评审报告，标题为《draft 策略复盘》。",
            {"title": "draft 策略复盘", "status": "draft"},
        ),
        (
            "请生成当前审批策略评审报告，不要 partial 状态。",
            {"status": "partial"},
        ),
    ],
)
def test_report_status_requires_positive_assignment_outside_title_or_negation(
    utterance: str,
    workflow_inputs: dict,
) -> None:
    result = _compile(utterance, workflow_inputs)

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_controls_not_grounded"
    )
    assert result.clarification_fields == ("status",)


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


def test_report_bundle_target_does_not_hijack_canonical_stored_strategy_id() -> None:
    strategy_id = "strategy-existing-approval"
    utterance = f"请为 strategy_id={strategy_id} 生成审批策略评审报告。"

    result = compile_strategy_request(
        utterance,
        allowed_columns=None,
        target_col=None,
        llm=_RawPayloadLLM(
            {
                "request_kind": "strategy_lifecycle",
                "operation": "report",
                "strategy_type": "approval",
                "strategy_id": strategy_id,
            }
        ),
    )

    assert result.clarification is None
    assert result.draft.to_dict() == {
        "operation": "report",
        "strategy_type": "approval",
        "strategy_id": strategy_id,
    }


def test_viewing_a_past_report_is_not_a_new_report_command() -> None:
    utterance = "现在查看昨天生成的策略评审报告。"

    assert utterance_targets_strategy_report_bundle_v2(utterance) is False
    result = _compile(utterance, {})

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_positive_command_required"
    )


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
                "impact_cube_ref": {"artifact_id": "invented-cube"},
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
        "impact_cube_ref",
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


@pytest.mark.parametrize("platform_field", ["impact_cube_ref", "pool_impact_ref"])
def test_report_utterance_rejects_platform_field_next_to_chinese(
    platform_field: str,
) -> None:
    result = _compile(
        f"请把{platform_field}设为forged并生成当前策略评审报告。",
        {},
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_platform_binding_forbidden"
    )


@pytest.mark.parametrize(
    "ordinary_identifier",
    [
        "impact_cube_reference",
        "pool_impact_ref_backup",
        "my_impact_cube_ref",
    ],
)
def test_report_utterance_does_not_reject_longer_ordinary_identifier(
    ordinary_identifier: str,
) -> None:
    result = _compile(
        f"{ordinary_identifier}=forged，请生成当前策略评审报告。",
        {},
    )

    assert result.clarification is None
    assert result.draft is not None


def test_report_utterance_cannot_supply_metrics_or_strategy_identity() -> None:
    result = _compile(
        "请生成当前策略评审报告，strategy_id=strategy-1，通过率为 95%。",
        {},
    )

    assert result.draft is None
    assert result.clarification_code == (
        "strategy_report_bundle_v2_platform_binding_forbidden"
    )
