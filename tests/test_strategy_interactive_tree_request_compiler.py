"""Compiler contract for one immutable interactive-tree prune revision."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    validate_strategy_request,
)


AUTOMATIC_SOURCE = "candidate-asset-" + "a" * 32
REVISION_SOURCE = "interactive-tree-revision-" + "b" * 32
OTHER_SOURCE = "candidate-asset-" + "c" * 32
NODE_A = "node-" + "1" * 20
NODE_B = "node-" + "2" * 20
REASON = "人工确认该子树颗粒度过细"


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(
    *,
    source_tree_id: str = AUTOMATIC_SOURCE,
    node_id: str = NODE_A,
    operation: str = "prune_subtree",
    reason: object = ...,
    **extra: object,
) -> dict:
    inputs: dict[str, object] = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": operation,
    }
    if reason is not ...:
        inputs["reason"] = reason
    inputs.update(extra)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_revision",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize("source_tree_id", [AUTOMATIC_SOURCE, REVISION_SOURCE])
def test_interactive_tree_revision_accepts_one_exact_source_and_node(
    source_tree_id: str,
) -> None:
    payload = _payload(source_tree_id=source_tree_id, reason=REASON)

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "interactive_tree_revision" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "interactive_tree_revision"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "交互式树" in result.confirmation
    assert source_tree_id in result.confirmation
    assert NODE_A in result.confirmation
    assert "prune_subtree" in result.confirmation
    assert "不可变" in result.confirmation
    assert "不会加入 Strategy Pool" in result.confirmation
    assert "不会采纳或部署" in result.confirmation


def test_interactive_tree_revision_normalizes_only_the_optional_user_reason() -> None:
    result = validate_strategy_request(
        _payload(reason="  人工\t确认  e\u0301 子树\n颗粒度过细  "),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "source_tree_id": AUTOMATIC_SOURCE,
        "node_id": NODE_A,
        "operation": "prune_subtree",
        "reason": "人工 确认 é 子树 颗粒度过细",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_tree_id": "candidate-asset-short"}, "source_tree_id"),
        ({"source_tree_id": "candidate-asset-" + "A" * 32}, "source_tree_id"),
        (
            {"source_tree_id": "interactive-tree-revision-short"},
            "source_tree_id",
        ),
        ({"node_id": "node-short"}, "node_id"),
        ({"node_id": "node-" + "A" * 20}, "node_id"),
        ({"operation": "split_leaf"}, "prune_subtree"),
        ({"reason": "   \n  "}, "reason"),
        ({"reason": "contains\x00nul"}, "reason"),
        ({"reason": "x" * 501}, "500"),
    ],
)
def test_interactive_tree_revision_rejects_invalid_ids_operation_or_reason(
    overrides: dict[str, object],
    message: str,
) -> None:
    payload = _payload(**overrides)

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


@pytest.mark.parametrize(
    "caller_field",
    [
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "revision_hash",
        "tree_hash",
        "tree",
        "frontier_node_ids",
        "visible_node_ids",
        "condition",
        "metrics",
        "dataset_id",
        "workspace_revision",
        "sample_design_ref",
        "pool_id",
        "action",
        "adopt",
        "deploy",
    ],
)
def test_interactive_tree_revision_rejects_every_caller_owned_platform_field(
    caller_field: str,
) -> None:
    result = validate_strategy_request(
        _payload(**{caller_field: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    assert caller_field in result.clarification


def test_manual_and_natural_language_compile_to_the_same_canonical_request() -> None:
    expected = _payload(reason=REASON)
    manual = validate_strategy_request(expected, allowed_columns=())
    llm = _FakeLLM(expected)
    natural = compile_strategy_request(
        (
            f"对交互式树 {AUTOMATIC_SOURCE} 的节点 {NODE_A} 执行 "
            f"prune_subtree；理由：{REASON}。"
        ),
        allowed_columns=(),
        llm=llm,
    )

    assert manual.draft is not None
    assert natural.draft is not None
    assert manual.draft.to_dict() == natural.draft.to_dict() == expected
    assert len(llm.calls) == 1
    prompt = llm.calls[0]["system_prompt"]
    assert "interactive_tree_revision" in prompt
    assert "source_tree_id" in prompt
    assert "node_id" in prompt
    assert "prune_subtree" in prompt
    assert "Strategy Pool" in prompt


@pytest.mark.parametrize(
    ("utterance", "reply", "code", "fields"),
    [
        (
            f"对刚才那棵树的节点 {NODE_A} 执行 prune_subtree",
            _payload(),
            "interactive_tree_revision_explicit_ids_required",
            {"source_tree_id"},
        ),
        (
            f"对交互式树 {AUTOMATIC_SOURCE} 的这个节点执行 prune_subtree",
            _payload(),
            "interactive_tree_revision_explicit_ids_required",
            {"node_id"},
        ),
        (
            (
                f"对 {AUTOMATIC_SOURCE} 或 {OTHER_SOURCE} 的节点 {NODE_A} "
                "执行 prune_subtree"
            ),
            _payload(),
            "interactive_tree_revision_explicit_ids_required",
            {"source_tree_id"},
        ),
        (
            (
                f"对 {AUTOMATIC_SOURCE} 的节点 {NODE_A} 或 {NODE_B} "
                "执行 prune_subtree"
            ),
            _payload(),
            "interactive_tree_revision_explicit_ids_required",
            {"node_id"},
        ),
        (
            f"对 {AUTOMATIC_SOURCE} 的节点 {NODE_A} 执行 prune_subtree",
            _payload(source_tree_id=OTHER_SOURCE),
            "interactive_tree_revision_controls_not_grounded",
            {"source_tree_id"},
        ),
        (
            f"对 {AUTOMATIC_SOURCE} 的节点 {NODE_A} 执行 prune_subtree",
            _payload(node_id=NODE_B),
            "interactive_tree_revision_controls_not_grounded",
            {"node_id"},
        ),
    ],
)
def test_interactive_tree_revision_requires_one_bidirectionally_grounded_pointer(
    utterance: str,
    reply: dict,
    code: str,
    fields: set[str],
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == code
    assert set(result.clarification_fields) == fields


@pytest.mark.parametrize(
    "utterance",
    [
        f"从 {AUTOMATIC_SOURCE} 自动选择风险最高节点并剪枝",
        f"从 {AUTOMATIC_SOURCE} 自动选择最差节点执行 prune_subtree",
        f"prune the best node from {AUTOMATIC_SOURCE}",
        (
            f"对 {AUTOMATIC_SOURCE} 的 {NODE_A} 执行 prune_subtree，"
            "因为它是风险最高节点"
        ),
    ],
)
def test_interactive_tree_revision_rejects_heuristic_node_selection(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "interactive_tree_revision_node_selection_ambiguous"
    )
    assert "明确" in result.clarification
    assert "node ID" in result.clarification


@pytest.mark.parametrize(
    "next_action",
    [
        "然后加入 Strategy Pool",
        "并设置通过动作",
        "然后采纳这个策略",
        "然后部署到生产",
        "and then apply it to the dataset",
    ],
)
def test_interactive_tree_revision_rejects_compound_mutations(
    next_action: str,
) -> None:
    result = compile_strategy_request(
        (
            f"对 {AUTOMATIC_SOURCE} 的 {NODE_A} 执行 prune_subtree，"
            f"{next_action}"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "interactive_tree_revision_single_step_required"
    assert result.clarification_fields == ("next_action",)


def test_interactive_tree_revision_never_executes_an_explicitly_negated_prune() -> (
    None
):
    result = compile_strategy_request(
        f"不要对 {AUTOMATIC_SOURCE} 的 {NODE_A} 执行 prune_subtree",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "interactive_tree_revision_intent_negated"


@pytest.mark.parametrize(
    "reply",
    [
        _payload(reason="人工确认该子树需要简化"),
        _payload(),
    ],
)
def test_interactive_tree_revision_reason_is_bidirectionally_grounded(
    reply: dict,
) -> None:
    result = compile_strategy_request(
        (
            f"对 {AUTOMATIC_SOURCE} 的 {NODE_A} 执行 prune_subtree；"
            f"理由：{REASON}。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == "interactive_tree_revision_reason_not_grounded"
    assert result.clarification_fields == ("reason",)


def test_interactive_tree_revision_rejects_an_invented_reason() -> None:
    result = compile_strategy_request(
        f"对 {AUTOMATIC_SOURCE} 的 {NODE_A} 执行 prune_subtree",
        allowed_columns=(),
        llm=_FakeLLM(_payload(reason=REASON)),
    )

    assert result.draft is None
    assert result.clarification_code == "interactive_tree_revision_reason_not_grounded"
    assert result.clarification_fields == ("reason",)
