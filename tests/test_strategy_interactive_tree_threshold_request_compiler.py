"""Public request contracts for one exact interactive-tree threshold edit."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


SOURCE_ID = "candidate-asset-" + "a" * 32
NODE_ID = "node-" + "1" * 20
OTHER_NODE_ID = "node-" + "2" * 20
REASON = "人工复核后调整根节点风险切分"


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(
    *,
    source_tree_id: object = SOURCE_ID,
    node_id: object = NODE_ID,
    operation: object = "adjust_split_threshold",
    threshold: object = 1.5,
    reason: object = ...,
    **extra: object,
) -> dict:
    inputs = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": operation,
    }
    if threshold is not ...:
        inputs["threshold"] = threshold
    if reason is not ...:
        inputs["reason"] = reason
    inputs.update(extra)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_revision",
        "workflow_inputs": inputs,
    }


def test_typed_threshold_revision_accepts_only_the_exact_user_controls() -> None:
    payload = _payload(reason=REASON)

    result = validate_strategy_request(payload, allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "阈值" in result.confirmation
    assert SOURCE_ID in result.confirmation
    assert NODE_ID in result.confirmation
    assert "1.5" in result.confirmation
    assert "不会加入 Strategy Pool" in result.confirmation


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold": ...}, "threshold"),
        ({"threshold": True}, "finite"),
        ({"threshold": "1.5"}, "finite"),
        ({"threshold": math.nan}, "finite"),
        ({"threshold": math.inf}, "finite"),
        (
            {"operation": "prune_subtree", "threshold": 1.5},
            "prune_subtree",
        ),
    ],
)
def test_typed_threshold_revision_rejects_missing_or_nonfinite_threshold(
    overrides: dict[str, object],
    message: str,
) -> None:
    result = validate_strategy_request(_payload(**overrides), allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


def test_manual_threshold_dto_contains_no_platform_evidence_fields() -> None:
    request = ManualStrategyRequest.model_validate(
        _payload(reason=REASON),
        strict=True,
    )

    assert request.workflow_inputs == _payload(reason=REASON)["workflow_inputs"]

    for field in (
        "revision_hash",
        "tree_hash",
        "visible_node_ids",
        "metrics",
    ):
        with pytest.raises(ValidationError):
            ManualStrategyRequest.model_validate(
                _payload(**{field: "forged"}),
                strict=True,
            )


def test_exact_natural_language_threshold_edit_compiles_to_the_same_request() -> (
    None
):
    expected = _payload(reason=REASON)
    llm = _FakeLLM(expected)

    result = compile_strategy_request(
        (
            f"把交互式树 {SOURCE_ID} 当前可见分裂节点 {NODE_ID} 的"
            f"新阈值调整为 1.5；理由：{REASON}。"
        ),
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == expected
    prompt = llm.calls[0]["system_prompt"]
    assert "adjust_split_threshold" in prompt
    assert "threshold" in prompt
    assert "最佳阈值" in prompt


def test_exact_natural_language_feature_replacement_is_bidirectionally_grounded() -> (
    None
):
    expected = _payload(
        operation="replace_split_feature",
        feature="z",
        threshold=1.5,
        reason=REASON,
    )
    llm = _FakeLLM(expected)

    result = compile_strategy_request(
        (
            f"把交互式树 {SOURCE_ID} 当前可见分裂节点 {NODE_ID} 的"
            f"新分裂特征改为 z，新阈值调整为 1.5；理由：{REASON}。"
        ),
        allowed_columns=("x", "z"),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == expected
    assert "换分裂特征" in result.confirmation
    assert "z" in result.confirmation


@pytest.mark.parametrize(
    ("reply", "field"),
    [
        (
            _payload(
                operation="replace_split_feature",
                feature="x",
                threshold=1.5,
            ),
            "feature",
        ),
        (
            _payload(
                operation="replace_split_feature",
                feature="z",
                threshold=2.5,
            ),
            "threshold",
        ),
    ],
)
def test_feature_replacement_rejects_llm_control_drift(
    reply: dict,
    field: str,
) -> None:
    result = compile_strategy_request(
        (
            f"把交互式树 {SOURCE_ID} 当前可见分裂节点 {NODE_ID} 的"
            "新分裂特征改为 z，新阈值调整为 1.5"
        ),
        allowed_columns=("x", "z"),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_revision_controls_not_grounded"
    )
    assert field in result.clarification_fields


@pytest.mark.parametrize(
    "utterance",
    [
        f"把 {SOURCE_ID} 的 {NODE_ID} 阈值调好一点",
        f"把 {SOURCE_ID} 的 {NODE_ID} 调成最佳阈值",
        f"自动优化 {SOURCE_ID} 的 {NODE_ID} 阈值",
        f"调整 {SOURCE_ID} 的全部节点阈值为 1.5",
    ],
)
def test_natural_language_threshold_edit_rejects_vague_or_bulk_optimization(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_revision_threshold_ambiguous"
    )
    assert set(result.clarification_fields) <= {"node_id", "threshold"}


def test_natural_language_threshold_edit_requires_an_explicit_new_value() -> None:
    result = compile_strategy_request(
        f"调整交互式树 {SOURCE_ID} 当前可见节点 {NODE_ID} 的阈值",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_revision_explicit_threshold_required"
    )
    assert result.clarification_fields == ("threshold",)


@pytest.mark.parametrize(
    ("reply", "field"),
    [
        (_payload(threshold=2.5), "threshold"),
        (_payload(node_id=OTHER_NODE_ID), "node_id"),
        (_payload(operation="prune_subtree", threshold=...), "operation"),
    ],
)
def test_natural_language_threshold_controls_are_bidirectionally_grounded(
    reply: dict,
    field: str,
) -> None:
    result = compile_strategy_request(
        (
            f"把交互式树 {SOURCE_ID} 当前可见分裂节点 {NODE_ID} 的"
            "新阈值调整为 1.5"
        ),
        allowed_columns=(),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_revision_controls_not_grounded"
    )
    assert field in result.clarification_fields


@pytest.mark.parametrize(
    "next_action",
    [
        "然后物化前沿节点",
        "然后加入 Strategy Pool",
        "然后采纳",
        "然后部署到生产",
    ],
)
def test_threshold_edit_rejects_compound_mutations(next_action: str) -> None:
    result = compile_strategy_request(
        (
            f"把交互式树 {SOURCE_ID} 当前可见分裂节点 {NODE_ID} 的"
            f"新阈值调整为 1.5，{next_action}"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == (
        "interactive_tree_revision_single_step_required"
    )
    assert result.clarification_fields == ("next_action",)
