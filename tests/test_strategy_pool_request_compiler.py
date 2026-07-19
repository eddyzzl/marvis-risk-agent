"""Natural-language compiler contracts for task-scoped Strategy Pool edits."""

from __future__ import annotations

import json

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_ID = "candidate-asset-" + "a" * 32
RULE_1 = "candidate-rule-" + "1" * 32
RULE_2 = "candidate-rule-" + "2" * 32
ENTRY_1 = "pool-entry-" + "3" * 32


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _compile(utterance: str, workflow: str, workflow_inputs: dict):
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": workflow,
            "workflow_inputs": workflow_inputs,
        }
    )
    return compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=llm,
    ), llm


def test_pool_workflows_are_explicit_standard_workflows() -> None:
    assert {
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_compile",
    } <= set(STANDARD_STRATEGY_WORKFLOWS)


def test_add_candidate_requires_typed_actions_and_canonicalizes_them() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "candidate_asset_id": ASSET_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject", "reason_code": "HIGH_RISK"},
                "reason": "把已查看的高风险候选加入草案池",
            },
        },
        allowed_columns=(),
    )

    assert result.clarification is None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["candidate_asset_id"] == ASSET_ID
    assert inputs["default_action"] == {
        "type": "approval",
        "value": "approve",
        "reason_code": None,
        "stop": True,
    }
    assert inputs["action"] == {
        "type": "reject",
        "value": "reject",
        "reason_code": "HIGH_RISK",
        "stop": True,
    }
    assert "development / unvalidated" in result.confirmation
    assert "自动写入可逆 draft Pool revision" in result.confirmation
    assert "请求人工确认" not in result.confirmation
    assert "不会采纳或部署" in result.confirmation


def test_add_candidate_controls_must_be_present_in_the_user_utterance() -> None:
    result, llm = _compile(
        "把刚才那个候选加入策略池",
        "strategy_pool_add_candidate",
        {
            "candidate_asset_id": ASSET_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_controls_not_grounded"
    assert ASSET_ID in result.clarification
    assert len(llm.calls) == 1


def test_remove_accepts_a_full_entry_id_but_never_an_ungrounded_id() -> None:
    grounded, _ = _compile(
        f"从审批策略池删除 {ENTRY_1}",
        "strategy_pool_remove_entry",
        {
            "strategy_type": "approval",
            "entry_id": ENTRY_1,
        },
    )
    assert grounded.draft is not None

    invented, _ = _compile(
        "从审批策略池删除刚才那条",
        "strategy_pool_remove_entry",
        {
            "strategy_type": "approval",
            "rule_id": RULE_1,
        },
    )
    assert invented.draft is None
    assert invented.clarification_code == "strategy_pool_controls_not_grounded"


def test_pool_entry_controls_reject_noncanonical_short_ids() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_remove_entry",
            "workflow_inputs": {
                "strategy_type": "approval",
                "rule_id": "rule-1",
            },
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert "完整" in result.clarification


def test_set_action_requires_the_rule_id_and_typed_action_in_original_text() -> None:
    result, _ = _compile(
        f"把审批策略池中 {RULE_1} 的动作改成人工复核 review",
        "strategy_pool_set_action",
        {
            "strategy_type": "approval",
            "rule_id": RULE_1,
            "action": {"type": "review"},
        },
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["action"]["type"] == "review"


def test_pool_mutation_rejects_llm_invented_reason_and_action_reason_code() -> None:
    result, _ = _compile(
        f"把 {RULE_1} 的动作改成人工复核 review",
        "strategy_pool_set_action",
        {
            "strategy_type": "approval",
            "rule_id": RULE_1,
            "action": {"type": "review", "reason_code": "MANUAL_REVIEW"},
            "reason": "模型建议转人工",
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_controls_not_grounded"
    assert "MANUAL_REVIEW" in result.clarification
    assert "模型建议转人工" in result.clarification


def test_reorder_requires_an_explicit_full_order_and_rejects_heuristic_sorting() -> None:
    explicit, _ = _compile(
        f"按完整顺序重排审批策略池：{RULE_2}，{RULE_1}",
        "strategy_pool_reorder",
        {
            "strategy_type": "approval",
            "ordered_ids": [RULE_2, RULE_1],
        },
    )
    assert explicit.draft is not None

    partial, _ = _compile(
        f"把 {RULE_1} 放前面",
        "strategy_pool_reorder",
        {
            "strategy_type": "approval",
            "ordered_ids": [RULE_1],
        },
    )
    assert partial.draft is None
    assert partial.clarification_code == "strategy_pool_full_order_required"

    heuristic, _ = _compile(
        "把审批策略池按效果最好自动排序",
        "strategy_pool_reorder",
        {
            "strategy_type": "approval",
            "ordered_ids": [RULE_2, RULE_1],
        },
    )
    assert heuristic.draft is None
    assert heuristic.clarification_code == "strategy_pool_full_order_required"


def test_reorder_validation_rejects_duplicate_ids() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_reorder",
            "workflow_inputs": {
                "strategy_type": "approval",
                "ordered_ids": [RULE_1, RULE_1],
            },
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert "重复" in result.clarification


def test_compile_pool_is_read_only_and_needs_only_strategy_type() -> None:
    result, _ = _compile(
        "编译并预览审批策略池草案，不要采纳或部署",
        "strategy_pool_compile",
        {"strategy_type": "approval"},
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "strategy_type": "approval"
    }
    assert "只读" in result.confirmation
    assert "不会采纳或部署" in result.confirmation

    invented_type, _ = _compile(
        "编译并预览这个策略池草案",
        "strategy_pool_compile",
        {"strategy_type": "approval"},
    )
    assert invented_type.draft is None
    assert invented_type.clarification_code == "strategy_pool_controls_not_grounded"
