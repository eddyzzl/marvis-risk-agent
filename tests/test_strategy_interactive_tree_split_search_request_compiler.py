from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


SOURCE_ID = "candidate-asset-" + "a" * 32
NODE_ID = "node-" + "1" * 20


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "source_tree_id": SOURCE_ID,
        "node_id": NODE_ID,
        "mode": "all_features",
        "max_thresholds_per_feature": 5,
        "max_row_evaluations": 2000,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_split_search",
        "workflow_inputs": inputs,
    }


def test_manual_split_search_accepts_exact_bounded_all_feature_controls() -> None:
    payload = _payload()

    compiled = validate_strategy_request(payload, allowed_columns=())
    manual = ManualStrategyRequest.model_validate(payload, strict=True)

    assert compiled.draft is not None
    assert compiled.draft.to_dict() == payload
    assert manual.workflow_inputs == payload["workflow_inputs"]
    assert "interactive_tree_split_search" in STANDARD_STRATEGY_WORKFLOWS
    assert "排名只用于浏览" in compiled.confirmation
    assert "不会选择胜者" in compiled.confirmation


def test_selected_feature_search_canonicalizes_feature_order() -> None:
    compiled = validate_strategy_request(
        _payload(
            mode="selected_features",
            features=["score", "income"],
        ),
        allowed_columns=("income", "score"),
    )

    assert compiled.draft is not None
    assert compiled.draft.to_dict()["workflow_inputs"]["features"] == [
        "income",
        "score",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "all_features", "features": ["score"]},
        {"mode": "selected_features"},
        {"mode": "selected_features", "features": ["score", "score"]},
        {"max_thresholds_per_feature": 21},
        {"max_row_evaluations": 20_000_001},
        {"max_row_evaluations": True},
        {"winner": "score"},
        {"dataset_id": "forged"},
    ],
)
def test_split_search_rejects_ambiguous_unbounded_or_platform_inputs(
    overrides: dict[str, object],
) -> None:
    payload = _payload(**overrides)

    compiled = validate_strategy_request(payload, allowed_columns=())
    assert compiled.draft is None
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)


def test_natural_language_all_feature_search_is_bidirectionally_grounded() -> None:
    expected = _payload()
    llm = _FakeLLM(expected)

    compiled = compile_strategy_request(
        (
            f"搜索交互树 {SOURCE_ID} 的节点 {NODE_ID} 全部特征分裂候选，"
            "每特征最多 5 个阈值，总行评估预算 2000。"
        ),
        allowed_columns=(),
        llm=llm,
    )

    assert compiled.draft is not None
    assert compiled.draft.to_dict() == expected
    assert "interactive_tree_split_search" in llm.calls[0]["system_prompt"]


def test_natural_language_search_rejects_hidden_default_or_chained_edit() -> None:
    missing_budget = compile_strategy_request(
        f"搜索交互树 {SOURCE_ID} 的节点 {NODE_ID} 全部特征分裂候选。",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )
    chained = compile_strategy_request(
        (
            f"搜索交互树 {SOURCE_ID} 的节点 {NODE_ID} 全部特征分裂候选，"
            "每特征最多 5 个阈值，总行评估预算 2000，然后自动选择最佳候选"
            "继续建树。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert missing_budget.draft is None
    assert missing_budget.clarification_code == (
        "interactive_tree_split_search_controls_not_grounded"
    )
    assert chained.draft is None
    assert chained.clarification_code == (
        "interactive_tree_split_search_single_step_required"
    )
