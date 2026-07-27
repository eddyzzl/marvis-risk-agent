"""Natural-language compiler contracts for task-scoped Strategy Pool edits."""

from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_ID = "candidate-asset-" + "a" * 32
SELECTION_ID = "automatic-tree-leaf-selection-" + "b" * 32
OTHER_SELECTION_ID = "automatic-tree-leaf-selection-" + "c" * 32
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


class _RawLLM:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.raw


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


def test_pool_add_fails_closed_for_deep_model_payload_and_oversized_utterance() -> None:
    nested: object = "ROUTE"
    for _ in range(80):
        nested = [nested]
    raw = json.dumps(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval", "output_value": nested},
                "action": {"type": "reject"},
            },
        },
        ensure_ascii=False,
    )
    llm = _RawLLM(raw)
    deep = compile_strategy_request(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        allowed_columns=(),
        llm=llm,
    )

    assert deep.draft is None
    assert deep.clarification_code == "strategy_request_too_complex"
    assert len(llm.calls) == 1

    oversized, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        f"命中动作：reject；备注：{'复核' * 4100}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )
    assert oversized.draft is None
    assert oversized.clarification_code == "strategy_pool_add_request_too_large"


def test_pool_add_prompt_locks_selection_and_independent_label_contracts() -> None:
    _, llm = _compile(
        "把选择结果加入审批策略池",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    prompt = llm.calls[0]["system_prompt"]
    assert "candidate_asset_id 与 selection_id 严格二选一" in prompt
    assert "Pool 默认动作和命中动作必须分别从显式标签子句" in prompt
    assert "不得从动作词反推策略池类型" in prompt
    assert "入池与采纳、部署、执行、投入使用、删除" in prompt
    assert "唯一 source ID 必须与该正向入池命令位于同一授权子句" in prompt


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


def test_add_exact_tree_leaf_selection_accepts_one_canonical_source() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject", "reason_code": "HIGH_RISK"},
                "reason": "人工风险复核确认",
            },
        },
        allowed_columns=(),
    )

    assert result.clarification is None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["selection_id"] == SELECTION_ID
    assert "candidate_asset_id" not in inputs
    assert inputs["default_action"]["type"] == "approval"
    assert inputs["action"]["type"] == "reject"
    assert inputs["action"]["reason_code"] == "HIGH_RISK"


def test_add_source_requires_exactly_one_candidate_asset_or_selection() -> None:
    common = {
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
    }
    neither = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": common,
        },
        allowed_columns=(),
    )
    both = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                **common,
                "candidate_asset_id": ASSET_ID,
                "selection_id": SELECTION_ID,
            },
        },
        allowed_columns=(),
    )

    assert neither.draft is None
    assert "二选一" in neither.clarification
    assert both.draft is None
    assert "二选一" in both.clarification


def test_add_rejects_noncanonical_tree_leaf_selection_id() -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "selection_id": "automatic-tree-leaf-selection-" + "A" * 32,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        },
        allowed_columns=(),
    )

    assert result.draft is None
    assert "32 位小写十六进制" in result.clarification


def test_add_selection_compiles_only_with_exact_labeled_controls() -> None:
    result, _ = _compile(
        f"把选择结果 {SELECTION_ID} 加入 Strategy Pool；"
        "策略池类型：approval；Pool 默认动作：approval；命中动作：reject；"
        "入池理由：人工风险复核确认；命中原因码：HIGH_RISK",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject", "reason_code": "HIGH_RISK"},
            "reason": "人工风险复核确认",
        },
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["selection_id"] == SELECTION_ID
    assert inputs["reason"] == "人工风险复核确认"


def test_existing_candidate_asset_add_remains_supported_with_labeled_controls() -> None:
    result, _ = _compile(
        f"把 {ASSET_ID} 加入审批 Strategy Pool；默认动作 approval，"
        "命中动作 reject 拒绝",
        "strategy_pool_add_candidate",
        {
            "candidate_asset_id": ASSET_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is not None


def test_existing_candidate_asset_add_accepts_established_intent_synonyms() -> None:
    for operation in (
        f"把 {ASSET_ID} 放进审批策略池",
        f"将 {ASSET_ID} 纳入审批策略池",
        f"把 {ASSET_ID} 写到审批策略池",
        f"把 {ASSET_ID} 新增到审批策略池",
        f"put {ASSET_ID} into the approval strategy pool",
        f"append {ASSET_ID} to the approval strategy pool",
    ):
        result, _ = _compile(
            f"{operation}；默认动作：approval；命中动作：reject",
            "strategy_pool_add_candidate",
            {
                "candidate_asset_id": ASSET_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is not None, operation


def test_add_selection_id_is_unique_and_bidirectionally_grounded() -> None:
    mismatched, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": OTHER_SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )
    ambiguous, _ = _compile(
        f"把 {SELECTION_ID} 或 {OTHER_SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert mismatched.draft is None
    assert mismatched.clarification_code == "strategy_pool_add_source_not_grounded"
    assert ambiguous.draft is None
    assert ambiguous.clarification_code == "strategy_pool_add_source_required"


def test_add_source_must_be_inside_the_positive_authorization_clause() -> None:
    cases = (
        (
            f"不要使用 {SELECTION_ID}；把刚才那个加入审批策略池；"
            "默认动作：approval；命中动作：reject",
            None,
        ),
        (
            "把刚才那个加入审批策略池；默认动作：approval；"
            f"命中动作：reject；入池理由：仅参考 {SELECTION_ID}",
            f"仅参考 {SELECTION_ID}",
        ),
        (
            f"不要使用 {SELECTION_ID} 但把刚才那个加入审批策略池；"
            "默认动作：approval；命中动作：reject",
            None,
        ),
        (
            f"do not use {SELECTION_ID} but add the previous one to the "
            "approval strategy pool; default action: approval; hit action: reject",
            None,
        ),
    )
    for utterance, reason in cases:
        workflow_inputs = {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        }
        if reason is not None:
            workflow_inputs["reason"] = reason
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            workflow_inputs,
        )

        assert result.draft is None, utterance
        assert result.clarification_code == "strategy_pool_add_source_required"


def test_add_reason_label_cannot_impersonate_an_add_command() -> None:
    reason = f"使用 {SELECTION_ID}"
    for label, labeled_reason in (
        ("入池理由", reason),
        ("添加理由", f"{reason} 参考审批策略池"),
    ):
        result, _ = _compile(
            "策略池类型：approval；默认动作：approval；命中动作：reject；"
            f"{label}：{labeled_reason}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
                "reason": labeled_reason,
            },
        )

        assert result.draft is None, label
        assert result.clarification_code == "strategy_pool_add_intent_required"


def test_add_source_rejects_malformed_or_uppercase_alternatives() -> None:
    invalid_alternatives = (
        "automatic-tree-leaf-selection-" + "C" * 32,
        "automatic-tree-leaf-selection-" + "d" * 31,
        "candidate-asset-" + "E" * 32,
        "candidate-asset-" + "f" * 31,
        "Candidate-asset-" + "e" * 32,
        "automatic-tree-leaf-selection-" + "e" * 128,
        "candidate-asset-",
        "candidate-asset-" + "１" * 32,
        "automatic-tree-leaf-selection-🚫",
    )
    for invalid_source in invalid_alternatives:
        result, _ = _compile(
            f"把 {SELECTION_ID} 或 {invalid_source} 加入审批策略池；"
            "默认动作：approval；命中动作：reject",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None, invalid_source
        assert result.clarification_code == "strategy_pool_add_source_required"


def test_add_requires_positive_non_negated_pool_intent() -> None:
    for utterance in (
        f"不要把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"不把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"暂不将 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"不用把 {SELECTION_ID} 加到审批策略池；默认动作：approval；命中动作：reject",
        f"不加入 {SELECTION_ID} 到审批策略池；默认动作：approval；命中动作：reject",
        f"取消把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"不要部署但也不要把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"do not add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject",
        f"can't add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject",
    ):
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_intent_negated"


@pytest.mark.parametrize(
    "utterance",
    (
        f"我昨天已经把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"请把‘把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject’改写得更专业",
        f"failed to add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject",
        f"{SELECTION_ID} 加入不了审批策略池；"
        "默认动作：approval；命中动作：reject",
    ),
)
def test_add_rejects_narrative_rewrite_and_failed_operation_text(
    utterance: str,
) -> None:
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_intent_negated"


def test_add_rejects_postposed_cancellation_and_hypothetical_explanation() -> None:
    for utterance in (
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；等等，不要入池",
        f"不要执行，只告诉我如果把 {SELECTION_ID} 加入审批策略池会怎样；"
        "默认动作：approval；命中动作：reject",
        f"what if I add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject",
        f"假设把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject，会发生什么？",
        f"评估把 {SELECTION_ID} 加入审批策略池的影响；"
        "默认动作：approval；命中动作：reject",
        f"模拟一下把 {SELECTION_ID} 加入审批策略池后的效果；"
        "默认动作：approval；命中动作：reject",
        f"能否把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"以后再把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；不，取消入池",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject；算了",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；别做了",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；算了，这次先不做了",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；取消吧",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；取消吧，谢谢",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；刚才那句作废",
        f"明天把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"审批通过后把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"明早把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"两天后把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"等审批通过就把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池吗？默认动作：approval；命中动作：reject",
        f"如何把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        f"请说明如何把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"演示一下把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"测试一下把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池是不允许的；"
        "默认动作：approval；命中动作：reject",
        f"文档写着‘把 {SELECTION_ID} 加入审批策略池’；"
        "默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；不要用这个 source；"
        "默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；但不要用这个ID；"
        "默认动作：approval；命中动作：reject",
        f"add {SELECTION_ID} to the approval strategy pool tomorrow; "
        "default action: approval; hit action: reject",
        f"after approval add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject",
        f"add {SELECTION_ID} to the approval strategy pool; "
        "but do not use this ID; default action: approval; hit action: reject",
        f"add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject; actually no",
        f"add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject; stop",
    ):
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_intent_negated"


@pytest.mark.parametrize(
    "cancellation",
    (
        "等一下还是先不要了",
        "等等先不加",
        "停一下不要了",
        "我反悔了",
    ),
)
def test_add_rejects_additional_postposed_cancellation_phrases(
    cancellation: str,
) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        f"命中动作：reject；{cancellation}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_intent_negated"


def test_add_pool_type_cannot_be_inferred_from_action_words() -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入 Strategy Pool；"
        "Pool 默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert "strategy_type" in result.clarification_fields


def test_add_rejects_negated_strategy_pool_type_label() -> None:
    for utterance in (
        f"把 {SELECTION_ID} 加入 Strategy Pool；策略池类型：不要 approval；"
        "默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；策略池类型：不要 reject；"
        "默认动作：approval；命中动作：reject",
        f"add {SELECTION_ID} to the strategy pool; "
        "pool type: do not use approval; default action: approval; "
        "hit action: reject",
    ):
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
        assert result.clarification_fields == ("strategy_type",)


def test_add_labeled_pool_type_default_and_hit_actions_are_not_swappable() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入 Strategy Pool；策略池类型：approval；"
        "Pool 默认动作：approval；命中动作：reject"
    )
    swapped, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "reject"},
            "action": {"type": "approval"},
        },
    )
    wrong_pool, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "reject",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert swapped.draft is None
    assert swapped.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert set(swapped.clarification_fields) == {"default_action", "action"}
    assert wrong_pool.draft is None
    assert wrong_pool.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert wrong_pool.clarification_fields == ("strategy_type",)


def test_add_rejects_negated_or_false_positive_action_labels() -> None:
    cases = (
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：不要 approval；"
            "命中动作：reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            "命中动作：不要 reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: approval; unmatched action: reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；这不是默认动作：approval；"
            "命中动作：reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；未命中动作：review",
            {"type": "approval"},
            {"type": "review"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            "不要默认动作：reject；命中动作：reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: do not use approval; hit action: reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "do not use default action: approval; hit action: reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: approval; never matched action: reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；"
            "默认动作：approval 但不要 approval；命中动作：reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            "命中动作：reject 但不要 reject",
            {"type": "approval"},
            {"type": "reject"},
        ),
    )
    for utterance, default_action, action in cases:
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": default_action,
                "action": action,
            },
        )

        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"


def test_add_reason_text_is_isolated_from_executable_action_labels() -> None:
    reason = "不要命中动作 review"
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；入池理由：{reason}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "review"},
            "reason": reason,
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert result.clarification_fields == ("action",)


def test_add_rejects_unmatched_reason_code_and_output_value_labels() -> None:
    result, _ = _compile(
        f"add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval; hit action: reject; "
        "unmatched reason code: BASELINE; unmatched output value: FALLBACK",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {
                "type": "reject",
                "reason_code": "BASELINE",
                "output_value": "FALLBACK",
            },
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert result.clarification_fields == ("action",)


def test_add_action_reason_codes_cannot_cross_default_and_hit_labels() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；Pool 默认原因码：BASELINE；"
        "命中动作：reject；命中原因码：HIGH_RISK"
    )
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval", "reason_code": "HIGH_RISK"},
            "action": {"type": "reject", "reason_code": "BASELINE"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert set(result.clarification_fields) == {"default_action", "action"}


def test_add_dotted_and_quoted_controls_require_the_complete_value() -> None:
    cases = (
        (
            "命中原因码：HIGH.RISK",
            {"type": "reject", "reason_code": "HIGH"},
        ),
        (
            "命中输出值：ROUTE.V2",
            {"type": "reject", "output_value": "ROUTE"},
        ),
        (
            '命中输出值："A,B"',
            {"type": "reject", "output_value": "A"},
        ),
        (
            '命中输出值：{"route":"A}B","values":[1,2]}',
            {"type": "reject", "output_value": "A}B"},
        ),
        (
            '命中输出值：{"route":"A","route":"ROUTE"}',
            {"type": "reject", "output_value": "ROUTE"},
        ),
    )
    for labeled_control, action in cases:
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            f"命中动作：reject；{labeled_control}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": action,
            },
        )

        assert result.draft is None, labeled_control
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
        assert result.clarification_fields == ("action",)

    exact_output = "A,B.V2"
    exact, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；命中原因码：HIGH.RISK；"
        '命中输出值："A,B.V2"',
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {
                "type": "reject",
                "reason_code": "HIGH.RISK",
                "output_value": exact_output,
            },
        },
    )
    assert exact.draft is not None


def test_add_cannot_omit_explicit_scoped_reason_codes_or_output_values() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；Pool 默认原因码：BASELINE；"
        "Pool 默认输出值：DEFAULT_ROUTE；"
        "命中动作：reject；命中原因码：HIGH_RISK；命中输出值：HIT_ROUTE"
    )
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert set(result.clarification_fields) == {"default_action", "action"}


def test_add_accepts_exact_scoped_reason_codes_and_output_values() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；Pool 默认原因码：BASELINE；"
        "Pool 默认输出值：DEFAULT_ROUTE；"
        "命中动作：reject；命中原因码：HIGH_RISK；命中输出值：HIT_ROUTE"
    )
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {
                "type": "approval",
                "reason_code": "BASELINE",
                "output_value": "DEFAULT_ROUTE",
            },
            "action": {
                "type": "reject",
                "reason_code": "HIGH_RISK",
                "output_value": "HIT_ROUTE",
            },
        },
    )

    assert result.draft is not None


def test_add_value_actions_require_exact_values_not_numeric_substrings() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入额度策略池；默认动作：limit 1000；命中动作：limit 2000"
    )
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 100},
            "action": {"type": "limit", "value": 200},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert set(result.clarification_fields) == {"default_action", "action"}

    exact, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 1000},
            "action": {"type": "limit", "value": 2000},
        },
    )
    assert exact.draft is not None


def test_add_decimal_action_values_are_not_truncated_at_the_decimal_point() -> None:
    utterance = (
        f"把 {SELECTION_ID} 加入定价策略池；"
        "默认动作：pricing 0.1；命中动作：pricing 0.2"
    )
    truncated, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "pricing",
            "default_action": {"type": "pricing", "value": 0},
            "action": {"type": "pricing", "value": 0},
        },
    )
    exact, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "pricing",
            "default_action": {"type": "pricing", "value": 0.1},
            "action": {"type": "pricing", "value": 0.2},
        },
    )

    assert truncated.draft is None
    assert truncated.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert set(truncated.clarification_fields) == {"default_action", "action"}
    assert exact.draft is not None


def test_add_value_controls_reject_grouping_substrings_and_duplicate_labels() -> None:
    cases = (
        (
            f"把 {SELECTION_ID} 加入额度策略池；"
            "默认动作：limit 1,000；命中动作：limit 2,000",
            "limit",
            {"type": "limit", "value": 1},
            {"type": "limit", "value": 2},
        ),
        (
            f"把 {SELECTION_ID} 加入分群策略池；"
            "默认动作：segment 高风险客群；命中动作：segment 低风险客群",
            "segmentation",
            {"type": "segment", "value": "高"},
            {"type": "segment", "value": "低"},
        ),
        (
            f"把 {SELECTION_ID} 加入额度策略池；"
            "默认动作：limit 1000；默认动作：limit 2000；"
            "命中动作：limit 3000",
            "limit",
            {"type": "limit", "value": 1000},
            {"type": "limit", "value": 3000},
        ),
    )
    for utterance, strategy_type, default_action, action in cases:
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": strategy_type,
                "default_action": default_action,
                "action": action,
            },
        )

        assert result.draft is None, utterance
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"


def test_add_rejects_negated_or_excluded_pool_and_action_labels() -> None:
    for utterance in (
        f"把 {SELECTION_ID} 加入非审批策略池；默认动作：approval；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval 除外；命中动作：reject",
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：排除 approval；命中动作：reject",
        f"add {SELECTION_ID} to the approval strategy pool; "
        "default action: approval excluded; hit action: reject",
    ):
        result, _ = _compile(
            utterance,
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None, utterance
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"


@pytest.mark.parametrize(
    ("utterance", "expected_field"),
    (
        (
            f"把 {SELECTION_ID} 加入 Strategy Pool；策略池类型：并不是 approval；"
            "默认动作：approval；命中动作：reject",
            "strategy_type",
        ),
        (
            f"把 {SELECTION_ID} 加入 Strategy Pool；策略池类型：绝非approval；"
            "默认动作：approval；命中动作：reject",
            "strategy_type",
        ),
        (
            f"add {SELECTION_ID} to the strategy pool; "
            "pool type: anything but approval; default action: approval; "
            "hit action: reject",
            "strategy_type",
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：并不是 approval；"
            "命中动作：reject",
            "default_action",
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：绝非approval；"
            "命中动作：reject",
            "default_action",
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: anything but approval; hit action: reject",
            "default_action",
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval 以外；"
            "命中动作：reject",
            "default_action",
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval 之外；"
            "命中动作：reject",
            "default_action",
        ),
        (
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：除了 approval；"
            "命中动作：reject",
            "default_action",
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: other than approval; hit action: reject",
            "default_action",
        ),
        (
            f"add {SELECTION_ID} to the approval strategy pool; "
            "default action: anything other than approval; hit action: reject",
            "default_action",
        ),
    ),
)
def test_add_rejects_strongly_negated_pool_type_and_default_action_labels(
    utterance: str,
    expected_field: str,
) -> None:
    result, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert result.clarification_fields == (expected_field,)


def test_add_rejects_action_controls_quoted_from_reference_text() -> None:
    for controls in (
        "文档写着‘默认动作 approval’，示例里说‘命中动作 reject’",
        'documentation says "default action approval" and "hit action reject"',
    ):
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；{controls}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None, controls
        assert result.clarification_code == "strategy_pool_add_intent_negated"


def test_add_exact_very_large_limit_does_not_overflow_grounding() -> None:
    value = 10**400
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入额度策略池；默认动作：limit {value}；"
        f"命中动作：limit {value}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": value},
            "action": {"type": "limit", "value": value},
        },
    )

    assert result.draft is not None


def test_add_large_integer_comparison_does_not_round_through_float() -> None:
    user_value = 9007199254740993
    mismatched, _ = _compile(
        f"把 {SELECTION_ID} 加入额度策略池；默认动作：limit {user_value}；"
        f"命中动作：limit {user_value}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 9007199254740992},
            "action": {"type": "limit", "value": 9007199254740992},
        },
    )

    assert mismatched.draft is None
    assert mismatched.clarification_code == "strategy_pool_add_controls_not_grounded"


def test_add_output_value_rejects_structured_values_without_substring_fallback() -> (
    None
):
    utterance = (
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "Pool 默认输出值：[1,2]；命中动作：reject"
    )
    truncated, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval", "output_value": 1},
            "action": {"type": "reject"},
        },
    )
    exact, _ = _compile(
        utterance,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval", "output_value": [1, 2]},
            "action": {"type": "reject"},
        },
    )

    assert truncated.draft is None
    assert truncated.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert exact.draft is not None
    assert exact.draft.to_dict()["workflow_inputs"]["default_action"][
        "output_value"
    ] == [1, 2]


def test_add_value_action_output_alias_remains_compatible_and_exact() -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入额度策略池；默认动作：limit 1000；"
        "Pool 默认输出值：900；命中动作：limit 2000；命中输出值：1900",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 1000, "output_value": 900},
            "action": {"type": "limit", "value": 2000, "output_value": 1900},
        },
    )

    assert result.draft is not None


def test_add_malformed_json_looking_output_never_rebinds_as_a_string() -> None:
    for raw_value in (
        '{"route":"approve","route":"reject"}',
        "[1,]",
        "{route:reject}",
        "NaN",
        "Infinity",
        "-Infinity",
        '"ROUTE',
        "'ROUTE",
        "001",
    ):
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            f"Pool 默认输出值：{raw_value}；命中动作：reject",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {
                    "type": "approval",
                    "output_value": raw_value,
                },
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None, raw_value
        assert result.clarification_code == "strategy_pool_add_controls_not_grounded"

    rounded, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；命中输出值：9007199254740993.0",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject", "output_value": 9007199254740992.0},
        },
    )
    assert rounded.draft is None
    assert rounded.clarification_code == "strategy_pool_add_controls_not_grounded"


def test_add_reason_is_optional_but_strictly_bidirectional_and_verbatim() -> None:
    base = (
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        "命中动作：reject；入池理由：人工风险复核确认"
    )
    omitted, _ = _compile(
        base,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )
    invented, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            "reason": "加入审批策略池",
        },
    )
    rewritten, _ = _compile(
        base,
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            "reason": "人工复核确认",
        },
    )

    for result in (omitted, invented, rewritten):
        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_reason_not_grounded"


def test_add_and_adopt_or_deploy_must_be_separate_requests() -> None:
    for follow_up in (
        "并采纳",
        "然后部署",
        "随后上线",
        "未验证候选并部署",
        "and then deploy",
        "then activated it",
        "然后投入生产",
        "然后上生产",
        "随后投用",
        "发布到线上",
        "推到线上",
        "正式运行",
        "然后落地执行",
        "然后立即执行",
        "然后执行它",
        "然后投入使用",
        "然后开始使用",
        "然后推生产",
        "then promoted it",
        "then enabled it",
        "then shipped it",
        "then pushed it to prod",
        "then released it",
        "then published it",
        "then launched it",
        "then productionized it",
        "then entered production",
        "then execute it",
        "then run it",
        "then use it in production",
        "then put it into production",
        "then take it live",
        f"然后删除 {ENTRY_1}",
        f"and then remove {ENTRY_1}",
        f"然后把 {RULE_1} 动作改成 review",
        f"然后按 {RULE_2}、{RULE_1} 完整重排审批策略池",
        f"然后把 {RULE_1} 放到前面",
        "然后编译预览审批策略池",
        "然后把审批策略池编译预览",
        f"然后把 {ENTRY_1} 从策略池删掉",
        "然后删除它",
        "然后把它删了",
        "然后把它踢出策略池",
        "然后把它拿掉",
        f"完成后把 {ENTRY_1} 撤掉",
        "然后撤回规则",
        f"然后调整 {RULE_1} 为 review",
        f"然后把 {RULE_1} 挪到第一位",
        f"然后把 {RULE_1} 移到第二位",
        f"然后把 {RULE_1} 移到末尾",
        f"然后把 {RULE_1} 设成 review",
        "再设成 review",
        f"然后把 {RULE_1} 置为 review",
        f"然后把 {RULE_1} 切换为 review",
        f"然后把 {RULE_1} 放在第二位",
        f"然后把 {RULE_1} 排到第二位",
        f"then move {RULE_1} to second place",
        f"然后把 {RULE_1} 下移一位",
        f"然后把 {RULE_1} 放后面",
        f"然后交换 {RULE_1} 和 {RULE_2} 的顺序",
        f"然后让 {RULE_1} 排第一",
        "然后重新编译",
    ):
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            f"命中动作：reject；{follow_up}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is None
        assert result.clarification_code == "strategy_pool_add_single_step_required"


@pytest.mark.parametrize(
    "follow_up",
    (
        "然后回测这条规则",
        "然后应用到当前样本",
        "然后生成效果报告",
        "然后提交审批",
    ),
)
def test_add_rejects_chained_evaluation_reporting_and_approval_operations(
    follow_up: str,
) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        f"命中动作：reject；{follow_up}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_single_step_required"


@pytest.mark.parametrize(
    "confusable_source",
    (
        "automatic－tree－leaf－selection－" + "d" * 32,
        "automatic-tree-leaf-\u200bselection-" + "d" * 32,
        "automatic\u00ad-tree-leaf-selection-" + "d" * 32,
        "automatic\u034f-tree-leaf-selection-" + "d" * 32,
        "аutomatic-tree-leaf-selection-" + "d" * 32,
    ),
)
def test_add_rejects_confusable_or_invisible_alternative_source_ids(
    confusable_source: str,
) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 或 {confusable_source} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_source_required"


@pytest.mark.parametrize(
    "prefix",
    (
        "操作日志显示我把",
        "系统记录：把",
        "这句话的意思是把",
        "用户说把",
        "我考虑把",
    ),
)
def test_add_rejects_unconsumed_narrative_prefixes(prefix: str) -> None:
    result, _ = _compile(
        f"{prefix} {SELECTION_ID} 加入审批策略池；"
        "默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_command_not_explicit"


@pytest.mark.parametrize(
    "tail",
    (
        "还是算了吧",
        "先别加了",
        "不要了",
        "先暂停",
        "这次作罢",
        "先放一放",
        "hold on, do not add it",
        "然后批准该策略",
        "然后审核通过",
        "然后导出结果",
        "然后下载报告",
        "然后保存为正式策略",
        "then approve it",
    ),
)
def test_add_rejects_any_unconsumed_tail_clause(tail: str) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        f"命中动作：reject；{tail}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_command_not_explicit"


@pytest.mark.parametrize("reason", ("其实不要入池", "算了"))
def test_add_reason_cannot_hide_a_cancellation(reason: str) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
        f"命中动作：reject；入池理由：{reason}",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
            "reason": reason,
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_reason_not_passive"


@pytest.mark.parametrize(
    "strategy_type_body",
    ("approval 然后批准该策略", "approval 但其实不要了"),
)
def test_add_strategy_type_label_body_must_be_fully_consumed(
    strategy_type_body: str,
) -> None:
    result, _ = _compile(
        f"把 {SELECTION_ID} 加入 Strategy Pool；"
        f"策略池类型：{strategy_type_body}；默认动作：approval；命中动作：reject",
        "strategy_pool_add_candidate",
        {
            "selection_id": SELECTION_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_add_controls_not_grounded"
    assert result.clarification_fields == ("strategy_type",)


def test_add_reason_cannot_hide_a_second_pool_operation() -> None:
    for reason in (
        f"完成后删除 {ENTRY_1}",
        f"完成后把 {ENTRY_1} 踢出策略池",
    ):
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            f"命中动作：reject；入池理由：{reason}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
                "reason": reason,
            },
        )

        assert result.draft is None, reason
        assert result.clarification_code == "strategy_pool_add_single_step_required"


def test_add_allows_explicitly_negated_lifecycle_follow_up() -> None:
    for negated_follow_up in (
        "不要采纳或部署",
        f"不要把 {RULE_1} 移到第二位",
        f"不要把 {RULE_1} 设成 review",
    ):
        result, _ = _compile(
            f"把 {SELECTION_ID} 加入审批策略池；默认动作：approval；"
            f"命中动作：reject；{negated_follow_up}",
            "strategy_pool_add_candidate",
            {
                "selection_id": SELECTION_ID,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        )

        assert result.draft is not None, negated_follow_up


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


@pytest.mark.parametrize(
    ("workflow", "utterance", "workflow_inputs"),
    (
        (
            "strategy_pool_remove_entry",
            f"不要从审批策略池删除 {ENTRY_1}",
            {"strategy_type": "approval", "entry_id": ENTRY_1},
        ),
        (
            "strategy_pool_remove_entry",
            f"能否从审批策略池删除 {ENTRY_1}？",
            {"strategy_type": "approval", "entry_id": ENTRY_1},
        ),
        (
            "strategy_pool_set_action",
            f"不要把审批策略池中 {RULE_1} 的动作改成 review",
            {
                "strategy_type": "approval",
                "rule_id": RULE_1,
                "action": {"type": "review"},
            },
        ),
        (
            "strategy_pool_set_action",
            f"可以把审批策略池中 {RULE_1} 的动作改成 review 吗？",
            {
                "strategy_type": "approval",
                "rule_id": RULE_1,
                "action": {"type": "review"},
            },
        ),
        (
            "strategy_pool_reorder",
            f"不要按完整顺序重排审批策略池：{RULE_2}，{RULE_1}",
            {
                "strategy_type": "approval",
                "ordered_ids": [RULE_2, RULE_1],
            },
        ),
        (
            "strategy_pool_reorder",
            f"是否按完整顺序重排审批策略池：{RULE_2}，{RULE_1}？",
            {
                "strategy_type": "approval",
                "ordered_ids": [RULE_2, RULE_1],
            },
        ),
    ),
)
def test_pool_mutations_require_current_positive_non_question_authorization(
    workflow: str,
    utterance: str,
    workflow_inputs: dict,
) -> None:
    result, _ = _compile(utterance, workflow, workflow_inputs)

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_mutation_intent_required"


@pytest.mark.parametrize(
    ("workflow", "utterance", "workflow_inputs"),
    (
        (
            "strategy_pool_remove_entry",
            f"从审批策略池删除 {RULE_1}；算了",
            {"strategy_type": "approval", "rule_id": RULE_1},
        ),
        (
            "strategy_pool_remove_entry",
            f"从审批策略池删除不了 {RULE_1}",
            {"strategy_type": "approval", "rule_id": RULE_1},
        ),
        (
            "strategy_pool_remove_entry",
            f"操作日志显示从审批策略池删除 {RULE_1}",
            {"strategy_type": "approval", "rule_id": RULE_1},
        ),
        (
            "strategy_pool_remove_entry",
            f"我想知道从审批策略池删除 {RULE_1} 会怎样",
            {"strategy_type": "approval", "rule_id": RULE_1},
        ),
        (
            "strategy_pool_set_action",
            f"把审批策略池 {RULE_1} 的动作改成 review；算了",
            {
                "strategy_type": "approval",
                "rule_id": RULE_1,
                "action": {"type": "review"},
            },
        ),
    ),
)
def test_pool_mutations_reject_unconsumed_or_failed_command_text(
    workflow: str,
    utterance: str,
    workflow_inputs: dict,
) -> None:
    result, _ = _compile(utterance, workflow, workflow_inputs)

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_mutation_intent_required"


@pytest.mark.parametrize("reason", ("其实不要删除", "算了"))
def test_pool_mutation_reason_cannot_hide_a_cancellation(reason: str) -> None:
    result, _ = _compile(
        f"从审批策略池删除 {RULE_1}；理由：{reason}",
        "strategy_pool_remove_entry",
        {
            "strategy_type": "approval",
            "rule_id": RULE_1,
            "reason": reason,
        },
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_reason_not_passive"


@pytest.mark.parametrize(
    ("workflow", "utterance", "workflow_inputs"),
    (
        (
            "strategy_pool_set_action",
            f"把审批策略池 {RULE_1} 的动作改成 approval 或 review",
            {
                "strategy_type": "approval",
                "rule_id": RULE_1,
                "action": {"type": "review"},
            },
        ),
        (
            "strategy_pool_set_action",
            f"把审批策略池 {RULE_1} 的动作改成通过或人工复核",
            {
                "strategy_type": "approval",
                "rule_id": RULE_1,
                "action": {"type": "review"},
            },
        ),
        (
            "strategy_pool_remove_entry",
            f"从审批策略池删除 {RULE_1} 或人工复核",
            {"strategy_type": "approval", "rule_id": RULE_1},
        ),
        (
            "strategy_pool_reorder",
            f"按完整顺序重排审批策略池或拒绝策略池：{RULE_2}，{RULE_1}",
            {
                "strategy_type": "approval",
                "ordered_ids": [RULE_2, RULE_1],
            },
        ),
    ),
)
def test_pool_mutations_reject_disjunctive_types_and_actions(
    workflow: str,
    utterance: str,
    workflow_inputs: dict,
) -> None:
    result, _ = _compile(utterance, workflow, workflow_inputs)

    assert result.draft is None
    assert result.clarification_code == "strategy_pool_mutation_intent_required"


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


def test_reorder_requires_an_explicit_full_order_and_rejects_heuristic_sorting() -> (
    None
):
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
    assert result.draft.to_dict()["workflow_inputs"] == {"strategy_type": "approval"}
    assert "只读" in result.confirmation
    assert "不会采纳或部署" in result.confirmation

    invented_type, _ = _compile(
        "编译并预览这个策略池草案",
        "strategy_pool_compile",
        {"strategy_type": "approval"},
    )
    assert invented_type.draft is None
    assert invented_type.clarification_code == "strategy_pool_controls_not_grounded"
