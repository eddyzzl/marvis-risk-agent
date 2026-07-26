"""Manual Candidate Lab and natural-language Voting requests share one compiler."""

from __future__ import annotations

from marvis.agent.strategy_request_compiler import (
    compile_strategy_request,
    validate_strategy_request,
)
from marvis.agent.turn_handlers import _MANUAL_STRATEGY_WORKFLOWS


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **_kwargs):
        return self.payload


def test_manual_and_natural_language_voting_search_compile_identically() -> None:
    rule_a = "candidate-rule-" + "a" * 32
    rule_b = "candidate-rule-" + "b" * 32
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_search",
        "workflow_inputs": {
            "strategy_type": "approval",
            "member_count": 3,
            "n": 2,
            "objective": {"metric": "bad_rate", "direction": "minimize"},
            "constraints": [
                {"metric": "hit_share", "operator": "gte", "value": 0.1}
            ],
            "include_rule_ids": [rule_a],
            "exclude_rule_ids": [rule_b],
            "max_combinations": 500,
        },
    }

    manual = validate_strategy_request(payload, allowed_columns=())
    natural = compile_strategy_request(
        (
            "搜索审批 Strategy Pool 的 Voting 组合：K=3，n=2；"
            "目标最小化 bad_rate；约束 hit_share >= 10%；"
            f"必须包含 {rule_a}；排除 {rule_b}；max_combinations=500。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert manual.draft is not None
    assert natural.draft is not None
    assert manual.draft.to_dict() == natural.draft.to_dict() == payload
    assert payload["workflow"] in _MANUAL_STRATEGY_WORKFLOWS


def test_manual_and_natural_language_voting_build_compile_identically() -> None:
    search_id = "voting-search-" + "c" * 32
    combo_id = "voting-combo-" + "d" * 32
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_build_from_search",
        "workflow_inputs": {
            "search_id": search_id,
            "combo_id": combo_id,
            "strategy_type": "approval",
        },
    }

    manual = validate_strategy_request(payload, allowed_columns=())
    natural = compile_strategy_request(
        (
            "从 Voting 搜索证据精确构建一个候选："
            f"search_id={search_id}，combo_id={combo_id}，"
            "来源为 approval Strategy Pool。只构建候选，不入池。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert manual.draft is not None
    assert natural.draft is not None
    assert manual.draft.to_dict() == natural.draft.to_dict() == payload
    assert payload["workflow"] in _MANUAL_STRATEGY_WORKFLOWS
