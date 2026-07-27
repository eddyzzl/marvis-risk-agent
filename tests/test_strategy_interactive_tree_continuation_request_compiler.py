from __future__ import annotations

import pytest
from pydantic import ValidationError

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.api_schemas import ManualStrategyRequest


SEARCH_ID = "interactive-tree-split-search-" + "a" * 32
CANDIDATE_ID = "interactive-tree-split-candidate-" + "b" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply

    def complete(self, **_kwargs):
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "search_id": SEARCH_ID,
        "candidate_id": CANDIDATE_ID,
        "max_additional_depth": 3,
        "min_gini_gain": 0.01,
        "max_generated_nodes": 31,
        "max_thresholds_per_feature": 10,
        "max_row_evaluations": 2000000,
        "objective": "max_gini_gain",
        "tie_break": "eligible_gain_feature_threshold_candidate_id",
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_auto_continuation",
        "workflow_inputs": inputs,
    }


def test_manual_continuation_accepts_only_explicit_bounded_controls() -> None:
    payload = _payload()

    compiled = validate_strategy_request(payload, allowed_columns=())
    manual = ManualStrategyRequest.model_validate(payload, strict=True)

    assert compiled.draft is not None
    assert compiled.draft.to_dict() == payload
    assert manual.workflow_inputs == payload["workflow_inputs"]
    assert "interactive_tree_auto_continuation" in STANDARD_STRATEGY_WORKFLOWS
    assert "不会自动挑选种子候选" in compiled.confirmation


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_additional_depth": 7},
        {"min_gini_gain": 0.51},
        {"max_generated_nodes": 2},
        {"max_thresholds_per_feature": 21},
        {"max_row_evaluations": 20_000_001},
        {"objective": "first_candidate"},
        {"tie_break": "rank"},
        {"source_tree_id": "forged"},
    ],
)
def test_continuation_rejects_unbounded_or_platform_inputs(
    overrides: dict[str, object],
) -> None:
    payload = _payload(**overrides)

    assert validate_strategy_request(payload, allowed_columns=()).draft is None
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(payload, strict=True)


def test_natural_language_continuation_is_bidirectionally_grounded() -> None:
    expected = _payload()
    compiled = compile_strategy_request(
        (
            f"从搜索 {SEARCH_ID} 明确选择候选 {CANDIDATE_ID} 自动续建子树，"
            "最大追加深度 3，最小 Gini 增益 0.01，最大生成节点数 31，"
            "每特征最大阈值数 10，总行评估预算 2000000，"
            "objective=max_gini_gain，"
            "tie_break=eligible_gain_feature_threshold_candidate_id。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(expected),
    )

    assert compiled.draft is not None
    assert compiled.draft.to_dict() == expected


def test_natural_language_continuation_rejects_defaults_or_chaining() -> None:
    hidden = compile_strategy_request(
        f"从 {SEARCH_ID} 的 {CANDIDATE_ID} 自动续建子树。",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )
    chained = compile_strategy_request(
        (
            f"从 {SEARCH_ID} 的 {CANDIDATE_ID} 自动续建子树，最大追加深度 3，"
            "最小 Gini 增益 0.01，最大生成节点数 31，每特征最大阈值数 10，"
            "总行评估预算 2000000，objective=max_gini_gain，"
            "tie_break=eligible_gain_feature_threshold_candidate_id，然后入池。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert hidden.clarification_code == (
        "interactive_tree_auto_continuation_controls_not_grounded"
    )
    assert chained.clarification_code == (
        "interactive_tree_auto_continuation_single_step_required"
    )
