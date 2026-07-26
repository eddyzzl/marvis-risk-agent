"""Compiler and Manual API contract for one explicit revision-frontier pointer."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


REVISION_A = "interactive-tree-revision-" + "a" * 32
REVISION_B = "interactive-tree-revision-" + "b" * 32
NODE_A = "node-" + "1" * 20
NODE_B = "node-" + "2" * 20
LEAF_A = "leaf-" + "3" * 20
SELECTION_REASON = "人工确认该前沿节点用于下一轮策略评审"
FRONTIER_SELECTION = "interactive-tree-frontier-selection-" + "4" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(
    *,
    revision_id: str = REVISION_A,
    source_node_id: str = NODE_A,
    selection_reason: object = ...,
    **extra: object,
) -> dict:
    inputs: dict[str, object] = {
        "revision_id": revision_id,
        "source_node_id": source_node_id,
    }
    if selection_reason is not ...:
        inputs["selection_reason"] = selection_reason
    inputs.update(extra)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_frontier_materialization",
        "workflow_inputs": inputs,
    }


@pytest.mark.parametrize("source_node_id", [NODE_A, LEAF_A])
def test_frontier_materialization_accepts_one_exact_revision_and_frontier_node(
    source_node_id: str,
) -> None:
    payload = _payload(
        source_node_id=source_node_id,
        selection_reason=SELECTION_REASON,
    )

    result = validate_strategy_request(payload, allowed_columns=())
    manual = ManualStrategyRequest.model_validate(payload, strict=True)

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert manual.workflow == "interactive_tree_frontier_materialization"
    assert "interactive_tree_frontier_materialization" in (
        STANDARD_STRATEGY_WORKFLOWS
    )
    assert "interactive_tree_frontier_materialization" in (
        STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "交互树前沿" in result.confirmation
    assert REVISION_A in result.confirmation
    assert source_node_id in result.confirmation
    assert "不会加入 Strategy Pool" in result.confirmation


def test_frontier_materialization_normalizes_only_the_optional_reason() -> None:
    result = validate_strategy_request(
        _payload(selection_reason="  人工\t确认  e\u0301 前沿\n节点  "),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "revision_id": REVISION_A,
        "source_node_id": NODE_A,
        "selection_reason": "人工 确认 é 前沿 节点",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision_id": "interactive-tree-revision-short"}, "revision_id"),
        ({"revision_id": "interactive-tree-revision-" + "A" * 32}, "revision_id"),
        ({"source_node_id": "node-short"}, "source_node_id"),
        ({"source_node_id": "leaf-" + "A" * 20}, "source_node_id"),
        ({"selection_reason": "   \n  "}, "selection_reason"),
        ({"selection_reason": "contains\x00nul"}, "selection_reason"),
        ({"selection_reason": "x" * 501}, "500"),
    ],
)
def test_frontier_materialization_rejects_invalid_user_controls(
    overrides: dict[str, object],
    message: str,
) -> None:
    result = validate_strategy_request(
        _payload(**overrides),
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


@pytest.mark.parametrize(
    "platform_field",
    [
        "source_artifact_id",
        "expected_artifact_content_hash",
        "artifact_id",
        "artifact_hash",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
        "condition",
        "metrics",
        "action",
        "default_action",
        "dataset_id",
        "workspace_revision",
    ],
)
def test_frontier_materialization_rejects_all_platform_owned_fields(
    platform_field: str,
) -> None:
    payload = _payload(**{platform_field: "forged"})

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)


def test_frontier_materialization_nl_is_exactly_grounded_and_bumps_prompt() -> None:
    expected = _payload(selection_reason=SELECTION_REASON)
    llm = _FakeLLM(expected)

    result = compile_strategy_request(
        (
            f"从交互树修订 {REVISION_A} 物化前沿节点 {NODE_A}；"
            f"选择理由：{SELECTION_REASON}。"
        ),
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == expected
    assert llm.calls[0]["prompt_version"] == 40
    assert "interactive_tree_frontier_materialization" in (
        llm.calls[0]["system_prompt"]
    )


@pytest.mark.parametrize(
    ("utterance", "code", "fields"),
    [
        (
            f"从刚才的交互树修订物化前沿节点 {NODE_A}",
            "interactive_tree_frontier_explicit_ids_required",
            {"revision_id"},
        ),
        (
            f"从交互树修订 {REVISION_A} 物化这个前沿节点",
            "interactive_tree_frontier_explicit_ids_required",
            {"source_node_id"},
        ),
        (
            f"从 {REVISION_A} 或 {REVISION_B} 物化前沿节点 {NODE_A}",
            "interactive_tree_frontier_explicit_ids_required",
            {"revision_id"},
        ),
        (
            f"从 {REVISION_A} 物化前沿节点 {NODE_A} 或 {NODE_B}",
            "interactive_tree_frontier_explicit_ids_required",
            {"source_node_id"},
        ),
        (
            f"从 {REVISION_B} 物化前沿节点 {NODE_A}",
            "interactive_tree_frontier_controls_not_grounded",
            {"revision_id"},
        ),
        (
            f"从 {REVISION_A} 物化前沿节点 {NODE_B}",
            "interactive_tree_frontier_controls_not_grounded",
            {"source_node_id"},
        ),
    ],
)
def test_frontier_materialization_rejects_pronouns_multiple_or_rewritten_ids(
    utterance: str,
    code: str,
    fields: set[str],
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == code
    assert set(result.clarification_fields) == fields


@pytest.mark.parametrize(
    "utterance",
    [
        f"从 {REVISION_A} 自动挑选风险最高的前沿节点并物化",
        f"materialize the best frontier node from {REVISION_A}",
        f"从 {REVISION_A} 物化前沿节点 {NODE_A}，因为它是最差节点",
    ],
)
def test_frontier_materialization_rejects_heuristic_selection(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_selection_ambiguous"
    )


@pytest.mark.parametrize(
    "next_action",
    [
        "然后加入 Strategy Pool",
        "并设置拒绝动作",
        "然后采纳这个策略",
        "and then deploy it",
    ],
)
def test_frontier_materialization_rejects_same_turn_pool_or_lifecycle_action(
    next_action: str,
) -> None:
    result = compile_strategy_request(
        f"从 {REVISION_A} 物化前沿节点 {NODE_A}，{next_action}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_single_step_required"
    )
    assert result.clarification_fields == ("next_action",)


@pytest.mark.parametrize(
    "platform_claim",
    [
        f"expected_artifact_content_hash={'f' * 64}",
        f"artifact hash: {'f' * 64}",
        f"树哈希：{'f' * 64}",
    ],
)
def test_frontier_materialization_rejects_user_authored_artifact_hashes_in_nl(
    platform_claim: str,
) -> None:
    result = compile_strategy_request(
        f"从 {REVISION_A} 物化前沿节点 {NODE_A}，{platform_claim}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_platform_controls_forbidden"
    )


def _pool_payload(source_id: str) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "selection_id": source_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    }


def test_frontier_selection_is_a_supported_cross_turn_pool_source() -> None:
    expected = _pool_payload(FRONTIER_SELECTION)
    validated = validate_strategy_request(expected, allowed_columns=())
    compiled = compile_strategy_request(
        (
            f"把选择结果 {FRONTIER_SELECTION} 加入 Strategy Pool；"
            "策略池类型：approval；Pool 默认动作：approval；命中动作：reject"
        ),
        allowed_columns=(),
        llm=_FakeLLM(expected),
    )

    assert validated.draft is not None
    assert compiled.draft is not None
    assert compiled.draft.to_dict()["workflow_inputs"]["selection_id"] == (
        FRONTIER_SELECTION
    )


def test_full_interactive_tree_revision_cannot_enter_pool_directly() -> None:
    result = validate_strategy_request(
        _pool_payload(REVISION_A),
        allowed_columns=(),
    )

    assert result.draft is None
    assert "interactive-tree-frontier-selection" in result.clarification
