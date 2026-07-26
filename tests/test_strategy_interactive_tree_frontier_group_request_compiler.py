"""Compiler and Manual API contract for one explicit frontier OR group."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    utterance_targets_interactive_tree_frontier_group_materialization,
    utterance_targets_interactive_tree_frontier_materialization,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


REVISION_A = "interactive-tree-revision-" + "a" * 32
REVISION_B = "interactive-tree-revision-" + "b" * 32
NODE_A = "node-" + "1" * 20
NODE_B = "node-" + "2" * 20
LEAF_C = "leaf-" + "3" * 20
NODE_D = "node-" + "4" * 20
SELECTION_REASON = "业务评审确认任一成员命中即进入候选规则"
GROUP_SELECTION = "interactive-tree-frontier-group-selection-" + "5" * 32


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
    source_node_ids: object = ...,
    selection_reason: object = ...,
    **extra: object,
) -> dict:
    inputs: dict[str, object] = {
        "revision_id": revision_id,
        "source_node_ids": (
            [NODE_A, NODE_B] if source_node_ids is ... else source_node_ids
        ),
    }
    if selection_reason is not ...:
        inputs["selection_reason"] = selection_reason
    inputs.update(extra)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_frontier_group_materialization",
        "workflow_inputs": inputs,
    }


def test_frontier_group_accepts_two_to_fifty_exact_unique_node_ids() -> None:
    fifty = [
        f"{'node' if index % 2 == 0 else 'leaf'}-{index:020x}"
        for index in range(50)
    ]

    for source_node_ids in ([NODE_A, NODE_B], fifty):
        payload = _payload(
            source_node_ids=source_node_ids,
            selection_reason=SELECTION_REASON,
        )
        result = validate_strategy_request(payload, allowed_columns=())
        manual = ManualStrategyRequest.model_validate(payload, strict=True)

        assert result.draft is not None
        assert result.draft.to_dict() == payload
        assert manual.workflow == (
            "interactive_tree_frontier_group_materialization"
        )
        assert "interactive_tree_frontier_group_materialization" in (
            STANDARD_STRATEGY_WORKFLOWS
        )
        assert "interactive_tree_frontier_group_materialization" in (
            STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"][
                "enum"
            ]
        )
        assert "OR" in result.confirmation
        assert REVISION_A in result.confirmation
        assert source_node_ids[0] in result.confirmation
        assert source_node_ids[-1] in result.confirmation
        assert "不会加入 Strategy Pool" in result.confirmation


def test_frontier_group_normalizes_only_the_optional_reason() -> None:
    result = validate_strategy_request(
        _payload(selection_reason="  人工\t确认  e\u0301 OR\n组合  "),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {
        "revision_id": REVISION_A,
        "source_node_ids": [NODE_A, NODE_B],
        "selection_reason": "人工 确认 é OR 组合",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"revision_id": "interactive-tree-revision-short"}, "revision_id"),
        ({"revision_id": "interactive-tree-revision-" + "A" * 32}, "revision_id"),
        ({"source_node_ids": [NODE_A]}, "2 到 50"),
        ({"source_node_ids": [NODE_A, NODE_B] * 26}, "2 到 50"),
        ({"source_node_ids": [NODE_A, NODE_A]}, "重复"),
        ({"source_node_ids": [NODE_A, "node-short"]}, "source_node_ids"),
        ({"source_node_ids": [NODE_A, "leaf-" + "A" * 20]}, "source_node_ids"),
        ({"source_node_ids": "not-a-list"}, "source_node_ids"),
        ({"selection_reason": "   \n  "}, "selection_reason"),
        ({"selection_reason": "contains\x00nul"}, "selection_reason"),
        ({"selection_reason": "x" * 501}, "500"),
    ],
)
def test_frontier_group_rejects_invalid_user_controls(
    overrides: dict[str, object],
    message: str,
) -> None:
    payload = _payload(**overrides)
    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)


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
        "group_id",
        "selection_id",
        "selection_hash",
        "fragment_id",
        "rule_id",
        "effect_id",
        "condition",
        "metrics",
        "action",
        "dataset_id",
        "workspace_revision",
    ],
)
def test_frontier_group_rejects_all_platform_owned_fields(
    platform_field: str,
) -> None:
    payload = _payload(**{platform_field: "forged"})

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)


def test_frontier_group_nl_is_exactly_grounded_and_routes_before_singleton() -> (
    None
):
    expected = _payload(selection_reason=SELECTION_REASON)
    llm = _FakeLLM(expected)
    utterance = (
        f"从交互树修订 {REVISION_A} 把前沿节点 {NODE_A} 和 {NODE_B} "
        f"按 OR 组合物化；选择理由：{SELECTION_REASON}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == expected
    assert utterance_targets_interactive_tree_frontier_group_materialization(
        utterance
    )
    assert not utterance_targets_interactive_tree_frontier_materialization(
        utterance
    )
    assert llm.calls[0]["prompt_version"] == 47
    assert "interactive_tree_frontier_group_materialization" in (
        llm.calls[0]["system_prompt"]
    )


@pytest.mark.parametrize(
    ("utterance", "code", "fields"),
    [
        (
            f"把 {NODE_A} 和 {NODE_B} 按 OR 组合物化",
            "interactive_tree_frontier_group_explicit_ids_required",
            {"revision_id"},
        ),
        (
            f"从 {REVISION_A} 把这个节点和 {NODE_B} 按 OR 组合物化",
            "interactive_tree_frontier_group_explicit_ids_required",
            {"source_node_ids"},
        ),
        (
            f"从 {REVISION_A} 或 {REVISION_B} 把 {NODE_A} 和 {NODE_B} "
            "按 OR 组合物化",
            "interactive_tree_frontier_group_explicit_ids_required",
            {"revision_id"},
        ),
        (
            f"从 {REVISION_A} 把 {NODE_A} 按 OR 组合物化",
            "interactive_tree_frontier_group_explicit_ids_required",
            {"source_node_ids"},
        ),
        (
            f"从 {REVISION_A} 把 {NODE_A}、{NODE_A} 按 OR 组合物化",
            "interactive_tree_frontier_group_explicit_ids_required",
            {"source_node_ids"},
        ),
        (
            f"从 {REVISION_B} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化",
            "interactive_tree_frontier_group_controls_not_grounded",
            {"revision_id"},
        ),
        (
            f"从 {REVISION_A} 把 {NODE_A} 和 {LEAF_C} 按 OR 组合物化",
            "interactive_tree_frontier_group_controls_not_grounded",
            {"source_node_ids"},
        ),
    ],
)
def test_frontier_group_rejects_missing_duplicate_or_rewritten_ids(
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


def test_frontier_group_grounding_treats_member_order_as_nonsemantic() -> None:
    result = compile_strategy_request(
        f"从 {REVISION_A} 把 {NODE_B} 和 {NODE_A} 按 OR 组合物化",
        allowed_columns=(),
        llm=_FakeLLM(_payload(source_node_ids=[NODE_A, NODE_B])),
    )

    assert result.draft is not None
    assert set(result.draft.workflow_inputs["source_node_ids"]) == {
        NODE_A,
        NODE_B,
    }


@pytest.mark.parametrize(
    "utterance",
    [
        f"从 {REVISION_A} 把最好的前沿节点按 OR 组合物化",
        f"从 {REVISION_A} 把全部前沿节点按 OR 组合物化",
        f"从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 组合物化",
    ],
)
def test_frontier_group_rejects_heuristic_or_missing_explicit_or_semantics(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_group_selection_ambiguous"
    )


@pytest.mark.parametrize(
    "utterance",
    [
        f"能否从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化？",
        f"不要从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化",
        f"如果以后从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化",
    ],
)
def test_frontier_group_rejects_question_negation_and_hypothesis(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_group_intent_negated"
    )


@pytest.mark.parametrize(
    "next_action",
    [
        "然后加入 Strategy Pool",
        "并设置拒绝动作",
        "然后应用到样本",
        "然后采纳这个策略",
        "and then deploy it",
    ],
)
def test_frontier_group_rejects_same_turn_pool_or_lifecycle_action(
    next_action: str,
) -> None:
    result = compile_strategy_request(
        (
            f"从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化，"
            f"{next_action}"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_group_single_step_required"
    )
    assert result.clarification_fields == ("next_action",)


@pytest.mark.parametrize(
    "platform_claim",
    [
        f"expected_artifact_content_hash={'f' * 64}",
        f"group_id=interactive-tree-frontier-group-{'f' * 32}",
        f"树哈希：{'f' * 64}",
    ],
)
def test_frontier_group_rejects_user_authored_platform_fields_in_nl(
    platform_claim: str,
) -> None:
    result = compile_strategy_request(
        (
            f"从 {REVISION_A} 把 {NODE_A} 和 {NODE_B} 按 OR 组合物化，"
            f"{platform_claim}"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_frontier_group_platform_controls_forbidden"
    )


def _pool_payload(selection_id: str) -> dict:
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "selection_id": selection_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    }


def test_frontier_group_selection_is_a_supported_cross_turn_pool_source() -> None:
    expected = _pool_payload(GROUP_SELECTION)

    validated = validate_strategy_request(expected, allowed_columns=())
    compiled = compile_strategy_request(
        (
            f"把选择结果 {GROUP_SELECTION} 加入 Strategy Pool；"
            "策略池类型：approval；Pool 默认动作：approval；命中动作：reject"
        ),
        allowed_columns=(),
        llm=_FakeLLM(expected),
    )

    assert validated.draft is not None
    assert compiled.draft is not None
    assert compiled.draft.to_dict()["workflow_inputs"]["selection_id"] == (
        GROUP_SELECTION
    )
