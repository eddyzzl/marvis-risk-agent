"""Natural-language contract for bounded, read-only Voting combination search."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


RULE_A = "candidate-rule-" + "a" * 32
RULE_B = "candidate-rule-" + "b" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "strategy_type": "approval",
        "member_count": 3,
        "n": 2,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_search",
        "workflow_inputs": inputs,
    }


def test_voting_search_validates_required_controls_and_safe_defaults() -> None:
    result = validate_strategy_request(_payload(), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(
        constraints=[],
        include_rule_ids=[],
        exclude_rule_ids=[],
        max_combinations=10_000,
    )
    assert "voting_candidate_search" in STANDARD_STRATEGY_WORKFLOWS
    assert "Voting 组合搜索" in result.confirmation
    assert "K=3" in result.confirmation
    assert "n=2" in result.confirmation
    assert "bad_capture_rate / maximize" in result.confirmation
    assert "10,000" in result.confirmation
    assert "只搜索" in result.confirmation
    assert "不会构建候选" in result.confirmation
    assert "不会修改 Pool" in result.confirmation


@pytest.mark.parametrize(
    ("metric", "required_share"),
    [
        ("bad_rate", "hit_share"),
        ("weighted_bad_rate", "weighted_hit_share"),
        ("bad_amount_rate", "hit_amount_share"),
    ],
)
def test_voting_search_minimized_rate_requires_positive_share_constraint(
    metric: str,
    required_share: str,
) -> None:
    missing = validate_strategy_request(
        _payload(
            objective={"metric": metric, "direction": "minimize"},
            constraints=[],
        ),
        allowed_columns=(),
    )
    absolute = validate_strategy_request(
        _payload(
            objective={"metric": metric, "direction": "minimize"},
            constraints=[
                {
                    "metric": {
                        "bad_rate": "hit_count",
                        "weighted_bad_rate": "weighted_hit_total",
                        "bad_amount_rate": "hit_amount",
                    }[metric],
                    "operator": "gte",
                    "value": 1,
                }
            ],
        ),
        allowed_columns=(),
    )
    valid = validate_strategy_request(
        _payload(
            objective={"metric": metric, "direction": "minimize"},
            constraints=[
                {
                    "metric": required_share,
                    "operator": "gte",
                    "value": 0.1,
                }
            ],
        ),
        allowed_columns=(),
    )

    assert missing.draft is None
    assert missing.clarification_code == "voting_search_minimum_share_required"
    assert absolute.draft is None
    assert absolute.clarification_code == "voting_search_minimum_share_required"
    assert valid.draft is not None


@pytest.mark.parametrize(
    "forbidden",
    [
        "pool_ref",
        "dataset_id",
        "target_col",
        "hit_matrix",
        "weights",
        "amounts",
        "search_result",
        "winner",
        "champion",
    ],
)
def test_voting_search_rejects_platform_and_result_fields(forbidden: str) -> None:
    result = validate_strategy_request(
        _payload(**{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert forbidden in result.clarification


def test_voting_search_compiles_explicit_include_exclude_and_constraint() -> None:
    payload = _payload(
        objective={"metric": "bad_rate", "direction": "minimize"},
        constraints=[
            {"metric": "hit_share", "operator": "gte", "value": 0.1},
        ],
        include_rule_ids=[RULE_A],
        exclude_rule_ids=[RULE_B],
        max_combinations=500,
    )
    llm = _FakeLLM(payload)
    utterance = (
        "搜索审批 Strategy Pool 的 Voting 组合：K=3，n=2；"
        "目标最小化 bad_rate；约束 hit_share >= 10%；"
        f"必须包含 {RULE_A}；排除 {RULE_B}；max_combinations=500。"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert llm.calls[0]["prompt_version"] == 47
    assert "voting_candidate_search" in llm.calls[0]["system_prompt"]


def test_voting_search_compiles_explicit_chinese_metric_aliases() -> None:
    payload = _payload(
        objective={"metric": "bad_capture_rate", "direction": "maximize"},
        constraints=[
            {"metric": "hit_share", "operator": "gte", "value": 0.1},
        ],
    )
    utterance = (
        "搜索审批 Strategy Pool 的 Voting 组合：K=3，n=2；"
        "目标最大化坏样本捕获率；约束命中占比不少于10%。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(
        objective={"metric": "bad_capture_rate", "direction": "maximize"},
        constraints=[
            {"metric": "hit_share", "operator": "gte", "value": 0.1},
        ],
        include_rule_ids=[],
        exclude_rule_ids=[],
        max_combinations=10_000,
    )


def test_voting_search_compiles_common_risk_business_metric_aliases() -> None:
    payload = _payload(
        objective={"metric": "bad_rate", "direction": "minimize"},
        constraints=[
            {"metric": "hit_share", "operator": "gte", "value": 0.1},
        ],
    )
    utterance = (
        "搜索审批 Strategy Pool 的 Voting 组合：K=3，n=2；"
        "目标最小化坏账率；约束命中率不少于10%。"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["objective"] == {
        "metric": "bad_rate",
        "direction": "minimize",
    }
    assert result.draft.to_dict()["workflow_inputs"]["constraints"] == [
        {"metric": "hit_share", "operator": "gte", "value": 0.1},
    ]


def test_voting_search_is_reserved_before_explicit_build_routing() -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "voting_candidate_build",
            "workflow_inputs": {
                "strategy_type": "approval",
                "rule_ids": [RULE_A, RULE_B],
                "n": 1,
            },
        }
    )
    utterance = (
        "搜索审批 Strategy Pool 的 Voting 组合：K=2，n=1，"
        "目标最大化 bad_capture_rate；"
        f"必须包含 {RULE_A}，排除 {RULE_B}。"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_search_workflow_required"


def test_voting_build_with_compare_word_is_not_misrouted_to_search() -> None:
    payload = {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_build",
        "workflow_inputs": {
            "strategy_type": "approval",
            "rule_ids": [RULE_A, RULE_B],
            "n": 1,
        },
    }
    utterance = (
        "构建并比较审批 Voting 组合效果："
        f"使用 {RULE_A}、{RULE_B}，2 选 1"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload


@pytest.mark.parametrize(
    ("utterance", "include_rule_ids"),
    [
        (
            (
                "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate；"
                f"必须包含 {RULE_A}。{RULE_B} 请忽略。"
            ),
            [RULE_A, RULE_B],
        ),
        (
            (
                "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate；"
                f"必须包含 {RULE_A}，不要 {RULE_B}。"
            ),
            [RULE_A, RULE_B],
        ),
        (
            (
                "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate；"
                f"不包含 {RULE_B}。"
            ),
            [RULE_B],
        ),
    ],
)
def test_voting_search_rejects_unlabeled_or_negated_include_ids(
    utterance: str,
    include_rule_ids: list[str],
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload(include_rule_ids=include_rule_ids)),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_rule_controls_not_grounded"


@pytest.mark.parametrize(
    ("utterance", "payload", "code"),
    [
        (
            "搜索审批 Voting 组合：n=2，目标最大化 bad_capture_rate。",
            _payload(),
            "voting_search_member_count_not_grounded",
        ),
        (
            "搜索审批 Voting 组合：K=3，目标最大化 bad_capture_rate。",
            _payload(),
            "voting_search_n_not_grounded",
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2。",
            _payload(),
            "voting_search_objective_not_grounded",
        ),
        (
            (
                "搜索审批 Voting 组合：K=3，n=2，"
                "目标最小化 bad_rate；约束 hit_count >= 10。"
            ),
            _payload(
                objective={"metric": "bad_rate", "direction": "minimize"},
                constraints=[{"metric": "hit_count", "operator": "gte", "value": 10}],
            ),
            "voting_search_minimum_share_required",
        ),
        (
            (
                "搜索审批 Voting 组合：K=3，n=2，"
                f"目标最大化 bad_capture_rate；必须包含 {RULE_A}。"
            ),
            _payload(),
            "voting_search_rule_controls_not_grounded",
        ),
    ],
)
def test_voting_search_fails_closed_when_controls_are_not_grounded(
    utterance: str,
    payload: dict,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(payload),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_voting_search_rejects_question_or_chained_candidate_build() -> None:
    question = compile_strategy_request(
        "能不能搜索审批 Voting 组合？K=3，n=2，目标最大化 bad_capture_rate。",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )
    chained = compile_strategy_request(
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate；"
            "然后构建第一名并加入策略池。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert question.draft is None
    assert question.clarification_code == "voting_search_positive_command_required"
    assert chained.draft is None
    assert chained.clarification_code == "voting_search_single_step_required"


def test_voting_search_rejects_unconnected_follow_up_sentences() -> None:
    result = compile_strategy_request(
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "构建第一名。加入策略池。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_single_step_required"


@pytest.mark.parametrize(
    "utterance",
    [
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate，"
            "构建第一名，加入策略池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "不构建候选，加入策略池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "不选择第一名，部署第二名。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名加入规则池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名放入策略池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名写入策略池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名纳入策略池。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "采用第一名。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "直接使用第一名。"
        ),
        (
            "Search approval Voting combinations: K=3, n=2, "
            "objective maximize bad_capture_rate. "
            "Add the top result to the strategy pool."
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "应用第一名。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名应用到当前样本。"
        ),
        (
            "Search approval Voting combinations: K=3, n=2, "
            "objective maximize bad_capture_rate. Apply the top result."
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "选择第二名。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "构建第二名。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名设置为拒绝动作。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "把第一名加入 Pool。"
        ),
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "修改规则池。"
        ),
    ],
)
def test_voting_search_rejects_connectorless_or_mixed_polarity_follow_up(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "voting_search_single_step_required"


def test_voting_search_allows_explicitly_negated_follow_up_boundaries() -> None:
    result = compile_strategy_request(
        (
            "搜索审批 Voting 组合：K=3，n=2，目标最大化 bad_capture_rate。"
            "不构建候选，不选择组合，不加入策略池，也不部署。"
        ),
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None
    assert result.draft.workflow == "voting_candidate_search"
