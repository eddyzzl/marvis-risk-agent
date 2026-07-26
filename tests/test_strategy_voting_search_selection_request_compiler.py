"""Natural-language contract for exact Voting search-result materialization."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


SEARCH_ID = "voting-search-" + "a" * 32
COMBO_ID = "voting-combo-" + "b" * 32
OTHER_COMBO_ID = "voting-combo-" + "c" * 32
RULE_ID = "candidate-rule-" + "d" * 32
OTHER_RULE_ID = "candidate-rule-" + "f" * 32
ENTRY_ID = "pool-entry-" + "e" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_build_from_search",
        "workflow_inputs": inputs,
    }


def test_voting_search_selection_accepts_only_exact_user_pointers() -> None:
    result = validate_strategy_request(_payload(), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()
    assert "voting_candidate_build_from_search" in STANDARD_STRATEGY_WORKFLOWS
    assert SEARCH_ID in result.confirmation
    assert COMBO_ID in result.confirmation
    assert "不会加入" in result.confirmation
    assert "Pool" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "artifact_id",
        "artifact_hash",
        "content_hash",
        "rule_ids",
        "member_rule_ids",
        "entry_ids",
        "selected_entry_ids",
        "n",
        "rank",
        "winner",
        "champion",
    ],
)
def test_voting_search_selection_rejects_derived_or_heuristic_fields(
    forbidden: str,
) -> None:
    result = validate_strategy_request(
        _payload(**{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "overrides",
    [
        {"search_id": "voting-search-short"},
        {"combo_id": "voting-combo-short"},
        {"strategy_type": "collections"},
    ],
)
def test_voting_search_selection_rejects_invalid_pointer_or_type(
    overrides: dict[str, object],
) -> None:
    result = validate_strategy_request(_payload(**overrides), allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


def test_voting_search_selection_compiles_exact_current_turn_pointers() -> None:
    llm = _FakeLLM(_payload(strategy_type="approval"))
    utterance = (
        "从 Voting 搜索证据精确构建一个候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}，"
        "来源为 approval Strategy Pool。只构建候选，不入池。"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(strategy_type="approval")
    assert llm.calls[0]["prompt_version"] == 43
    assert "voting_candidate_build_from_search" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    "selector",
    [
        "第一名",
        "最好的",
        "冠军",
        "winner",
        "champion",
        "first",
        "Top 1",
        "刚才那个",
    ],
)
def test_voting_search_selection_rejects_heuristic_selector_even_with_exact_ids(
    selector: str,
) -> None:
    utterance = (
        f"从 Voting 搜索结果构建 {selector} 候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_selection_explicit_ids_required"


@pytest.mark.parametrize(
    "injected",
    [
        "artifact_id=" + "f" * 64,
        "artifact_hash=" + "f" * 64,
        "content_hash=" + "f" * 64,
        "pool_revision=7",
        "pool_snapshot_hash=" + "f" * 64,
        "pool_id=strategy-pool-approval",
        "revision_id=revision-7",
        "expected_pool_revision=7",
        "expected_pool_snapshot_hash=" + "f" * 64,
        "expected_pool_id=strategy-pool-approval",
        "expected_revision_id=revision-7",
        "pool_ref={forged}",
        "dataset_id=dataset-1",
        "dataset_content_hash=" + "f" * 64,
        "expected_content_hash=" + "f" * 64,
        "dataset_binding={forged}",
        "target_col=bad",
        "target_polarity=one_is_bad",
        "target_semantics={forged}",
        "target_binding={forged}",
        "polarity=one_is_bad",
        "sample_design_ref=sample-design-1",
        "sample_design_id=sample-design-1",
        "sample_design_hash=" + "f" * 64,
        "sample_design_content_hash=" + "f" * 64,
        "sample_design_partition=development",
        "partition=development",
        "workspace_revision=12",
        "workspace_generation=4",
        "semantic_mapping_hash=" + "f" * 64,
        "semantic_mapping=bad_is_one",
        "requirement_bindings=minimum_share",
        "observation_bindings={forged}",
        "provenance={forged}",
        "pool revision=7",
        "dataset content hash=" + "f" * 64,
        "workspace generation=4",
        "semantic mapping=bad_is_one",
        "requirement bindings=minimum_share",
        "策略池版本=7",
        "策略池快照哈希=" + "f" * 64,
        "策略池ID=strategy-pool-approval",
        "Pool 版本=7",
        "Pool 快照哈希=" + "f" * 64,
        "版本ID=revision-7",
        "修订ID=revision-7",
        "数据集ID=dataset-1",
        "数据集内容哈希=" + "f" * 64,
        "数据内容哈希=" + "f" * 64,
        "目标列=bad",
        "目标极性=1为坏",
        "标签列=bad",
        "坏标签方向=1为坏",
        "样本设计引用=sample-design-1",
        "样本设计ID=sample-design-1",
        "样本设计哈希=" + "f" * 64,
        "样本设计内容哈希=" + "f" * 64,
        "样本设计分区=development",
        "工作区版本=12",
        "工作区代次=4",
        "工作区revision=12",
        "工作区generation=4",
        "语义映射=bad_is_one",
        "语义映射哈希=" + "f" * 64,
        "需求绑定=minimum_share",
        "规则需求绑定=minimum_share",
        f"rule_ids=[{RULE_ID}]",
        f"entry_ids=[{ENTRY_ID}]",
        "member_ids=[candidate-rule-" + "f" * 32 + "]",
        "n=2",
        "rank=1",
    ],
)
def test_voting_search_selection_rejects_platform_or_derived_injection(
    injected: str,
) -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}，{injected}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert (
        result.clarification_code
        == "voting_search_selection_platform_binding_forbidden"
    )


@pytest.mark.parametrize(
    "context_note",
    [
        (
            "平台在构建前自行恢复当前策略池、数据集、目标语义、样本设计、"
            "工作区、语义映射和需求约束"
        ),
        (
            "The platform rehydrates the current pool, dataset, target semantics, "
            "sample design, workspace context, semantic mapping, and requirement "
            "bindings"
        ),
    ],
)
def test_voting_search_selection_allows_unbound_platform_context_description(
    context_note: str,
) -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。{context_note}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()


@pytest.mark.parametrize(
    "follow_up",
    [
        "并加入 Strategy Pool",
        "并修改 Pool",
        "并设置拒绝动作",
        "并应用到当前样本",
        "并采纳",
        "并部署",
        "并写回数据集",
    ],
)
def test_voting_search_selection_is_one_build_step_only(follow_up: str) -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}，{follow_up}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_selection_single_step_required"


def test_voting_search_selection_accepts_explicit_negative_lifecycle_disclaimer() -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
        "只构建候选，不入池、不修改 Pool、不设置动作、不应用、不采纳、不部署、不写回。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None


def test_voting_search_selection_rejects_same_turn_research() -> None:
    utterance = (
        "先重新搜索审批 Voting 组合，再从搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload(strategy_type="approval")),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_selection_single_step_required"


@pytest.mark.parametrize(
    "follow_up",
    [
        "然后再搜索一遍",
        "然后找更好的组合",
        "then search again for a better combination",
        "then find a better combination",
    ],
)
def test_voting_search_selection_rejects_chained_research_without_voting_subject(
    follow_up: str,
) -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。{follow_up}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_selection_single_step_required"


def test_voting_search_selection_allows_explicit_no_research_disclaimer() -> None:
    utterance = (
        "不重新搜索 Voting 组合，只从现有搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    "disclaimer",
    [
        "不要再搜索一遍，也不要找更好的组合",
        "do not search again or find a better combination",
        "without searching again for a better combination",
    ],
)
def test_voting_search_selection_allows_negated_research_without_voting_subject(
    disclaimer: str,
) -> None:
    utterance = (
        "从 Voting 搜索结果精确构建候选："
        f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。{disclaimer}。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "能否从 Voting 搜索结果构建候选？"
            f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
        ),
        (
            "不要从 Voting 搜索结果构建候选："
            f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
        ),
        (
            "昨天从 Voting 搜索结果构建了候选："
            f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
        ),
        (
            "以后从 Voting 搜索结果构建候选："
            f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
        ),
    ],
)
def test_voting_search_selection_requires_current_positive_command(
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
        == "voting_search_selection_positive_command_required"
    )


@pytest.mark.parametrize(
    ("utterance", "reply", "expected_code"),
    [
        (
            (
                "从 Voting 搜索结果精确构建候选："
                f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
            ),
            {
                "request_kind": "standard_workflow",
                "workflow": "voting_candidate_search",
                "workflow_inputs": {
                    "strategy_type": "approval",
                    "member_count": 2,
                    "n": 1,
                    "objective": {
                        "metric": "bad_capture_rate",
                        "direction": "maximize",
                    },
                },
            },
            "voting_search_selection_workflow_required",
        ),
        (
            (
                "搜索审批 Strategy Pool 的 Voting 组合：K=2，n=1；"
                "目标最大化 bad_capture_rate。"
            ),
            _payload(),
            "voting_candidate_search_workflow_required",
        ),
        (
            (
                "构建审批 Strategy Pool 的 Voting 候选，2 选 1："
                f"{RULE_ID}、{OTHER_RULE_ID}。"
            ),
            _payload(),
            "voting_candidate_workflow_required",
        ),
    ],
)
def test_voting_search_selection_search_and_free_rule_routes_are_disjoint(
    utterance: str,
    reply: dict,
    expected_code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == expected_code


@pytest.mark.parametrize(
    ("utterance", "reply", "expected_code"),
    [
        (
            (
                "从 Voting 搜索结果精确构建候选："
                f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
            ),
            _payload(combo_id=OTHER_COMBO_ID),
            "voting_search_selection_controls_not_grounded",
        ),
        (
            (
                "从 Voting 搜索结果精确构建候选："
                f"search_id={SEARCH_ID}，search_id={SEARCH_ID}，"
                f"combo_id={COMBO_ID}。"
            ),
            _payload(),
            "voting_search_selection_controls_not_grounded",
        ),
        (
            (
                "从 approval Strategy Pool 的 Voting 搜索结果精确构建候选："
                f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
            ),
            _payload(),
            "voting_search_selection_strategy_type_not_grounded",
        ),
        (
            (
                "从 Voting 搜索结果精确构建候选："
                f"search_id={SEARCH_ID}，combo_id={COMBO_ID}。"
            ),
            _payload(strategy_type="approval"),
            "voting_search_selection_strategy_type_not_grounded",
        ),
    ],
)
def test_voting_search_selection_does_not_rewrite_pointers_or_optional_type(
    utterance: str,
    reply: dict,
    expected_code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == expected_code


def test_voting_search_selection_compiles_exact_english_pointer_command() -> None:
    utterance = (
        "Build one Voting candidate from "
        f"search_id={SEARCH_ID} and combo_id={COMBO_ID}. "
        "Only build the candidate; do not add it to the Strategy Pool, "
        "apply, adopt, deploy, or write it back."
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()
