"""Natural-language compiler contract for explicit Voting candidates."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


RULE_A = "candidate-rule-" + "a" * 32
RULE_B = "candidate-rule-" + "b" * 32
RULE_C = "candidate-rule-" + "c" * 32
POOL_ENTRY = "pool-entry-" + "d" * 32
VOTING_ASSET = "candidate-asset-" + "e" * 32


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
        "rule_ids": [RULE_A, RULE_B, RULE_C],
        "n": 2,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "voting_candidate_build",
        "workflow_inputs": inputs,
    }


def _pool_add_payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "candidate_asset_id": VOTING_ASSET,
        "strategy_type": "approval",
        "default_action": {"type": "approval"},
        "action": {"type": "reject"},
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": inputs,
    }


def test_voting_candidate_validates_only_human_controls() -> None:
    result = validate_strategy_request(_payload(), allowed_columns=())

    assert result.draft is not None
    assert result.draft.to_dict() == _payload()
    assert "voting_candidate_build" in STANDARD_STRATEGY_WORKFLOWS
    assert "Voting / n-of-k" in result.confirmation
    assert "至少命中 2 条" in result.confirmation
    assert "尚未入池" not in result.confirmation
    assert "不会入池" in result.confirmation
    assert "不会" in result.confirmation and "部署" in result.confirmation


@pytest.mark.parametrize(
    "forbidden",
    [
        "selected_entry_ids",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "dataset_id",
        "target_col",
        "condition",
        "metrics",
        "action",
        "recommendation",
    ],
)
def test_voting_candidate_rejects_platform_owned_fields(forbidden: str) -> None:
    result = validate_strategy_request(
        _payload(**{forbidden: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "overrides",
    [
        {"rule_ids": [RULE_A]},
        {"rule_ids": [RULE_A, RULE_A]},
        {"rule_ids": [RULE_A, "candidate-rule-short"]},
        {"n": 0},
        {"n": 4},
        {"n": True},
        {"strategy_type": "collection"},
    ],
)
def test_voting_candidate_rejects_invalid_rule_set_threshold_or_type(
    overrides: dict[str, object],
) -> None:
    result = validate_strategy_request(_payload(**overrides), allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


def test_voting_candidate_compiles_exact_rule_set_and_n() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_C, RULE_A, RULE_B]))
    utterance = (
        "构建审批策略池的 Voting 候选："
        f"{RULE_A}、{RULE_B}、{RULE_C}，3 选 2"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert set(result.draft.to_dict()["workflow_inputs"]["rule_ids"]) == {
        RULE_A,
        RULE_B,
        RULE_C,
    }
    assert result.draft.to_dict()["workflow_inputs"]["n"] == 2
    assert llm.calls[0]["prompt_version"] == 52
    assert "voting_candidate_build" in llm.calls[0]["system_prompt"]


@pytest.mark.parametrize(
    "reply",
    [
        {"operation": "analyze", "strategy_type": "approval"},
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_compile",
            "workflow_inputs": {"strategy_type": "approval"},
        },
    ],
)
def test_explicit_voting_request_cannot_route_to_another_workflow(
    reply: dict,
) -> None:
    llm = _FakeLLM(reply)
    utterance = (
        "构建审批策略池 Voting 候选，2 选 1："
        f"{RULE_A}、{RULE_B}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_workflow_required"
    assert result.clarification_fields == ("workflow",)


def test_viewing_voting_state_is_not_a_build_command() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B], n=1))
    utterance = (
        "查看审批策略池 Voting 规则的现状："
        f"{RULE_A}、{RULE_B}，n=1"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_build_intent_required"


def test_voting_requires_one_positive_command_clause() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B, RULE_C], n=1))
    utterance = (
        f"构建审批策略池 Voting 候选，n=1：{RULE_A}、{RULE_B}；"
        f"再构建审批策略池 Voting 候选，n=1：{RULE_B}、{RULE_C}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_single_command_required"


def test_voting_rejects_controls_outside_the_authorized_clause() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B, RULE_C], n=1))
    utterance = (
        f"上下文 ID：{RULE_A}；"
        f"现在构建审批策略池 Voting 候选，n=1：{RULE_B}、{RULE_C}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert (
        result.clarification_code
        == "voting_candidate_controls_outside_command"
    )


def test_voting_rejects_context_id_before_current_command_in_same_sentence() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B, RULE_C], n=1))
    utterance = (
        f"上下文 ID：{RULE_A}，"
        f"现在构建审批策略池 Voting 候选，n=1：{RULE_B}、{RULE_C}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert (
        result.clarification_code
        == "voting_candidate_controls_outside_command"
    )


@pytest.mark.parametrize(
    "utterance",
    [
        f"审批策略池；现在构建 Voting 候选，n=1：{RULE_A}、{RULE_B}",
        f"n=1；现在构建审批策略池 Voting 候选：{RULE_A}、{RULE_B}",
    ],
)
def test_voting_requires_type_and_n_in_the_authorized_clause(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B], n=1))

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert (
        result.clarification_code
        == "voting_candidate_controls_outside_command"
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "先重排审批策略池，再",
        "编译审批策略池并",
        f"先从策略池删除 {POOL_ENTRY}，再",
    ],
)
def test_voting_rejects_compound_pool_operations(prefix: str) -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B], n=1))
    utterance = (
        f"{prefix}构建审批策略池 Voting 候选，"
        f"n=1：{RULE_A}、{RULE_B}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_single_step_required"


@pytest.mark.parametrize(
    "utterance",
    [
        (
            f"不要使用 {RULE_A}；现在构建审批策略池 Voting 候选，"
            f"n=1：{RULE_B}、{RULE_C}"
        ),
        (
            f"构建审批策略池 Voting 候选，n=1：{RULE_A}、{RULE_B}、{RULE_C}；"
            f"不要 {RULE_A}，只用 {RULE_B}、{RULE_C}"
        ),
        (
            f"构建审批策略池 Voting 候选，排除 {RULE_A}，"
            f"选择 {RULE_B}、{RULE_C}，n=1"
        ),
        (
            f"构建审批策略池 Voting 候选，不要 n=1，改成2："
            f"{RULE_A}、{RULE_B}、{RULE_C}"
        ),
    ],
)
def test_voting_rejects_negated_or_replaced_controls(utterance: str) -> None:
    llm = _FakeLLM(_payload())

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_negated_control"


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            f"构建审批策略池 Voting 候选：{RULE_A}、{RULE_B}",
            "voting_candidate_n_not_grounded",
        ),
        (
            f"用刚才这些规则构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}、{RULE_C}",
            "voting_candidate_explicit_rules_required",
        ),
        (
            f"构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}，然后加入策略池",
            "voting_candidate_single_step_required",
        ),
        (
            f"构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}，入池",
            "voting_candidate_single_step_required",
        ),
        (
            f"不要构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}",
            "voting_candidate_build_intent_negated",
        ),
        (
            f"构建 Voting 候选，n=2：{RULE_A}、{RULE_B}",
            "voting_candidate_strategy_type_not_grounded",
        ),
        (
            f"构建审批策略池 Voting 候选，n=2：{RULE_A}",
            "voting_candidate_rules_not_grounded",
        ),
        (
            f"能不能构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}？",
            "voting_candidate_positive_command_required",
        ),
        (
            f"如果下周通过评审，就构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}",
            "voting_candidate_positive_command_required",
        ),
        (
            f"构建审批策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}；算了，先不做了",
            "voting_candidate_positive_command_required",
        ),
        (
            f"构建审批或拒绝策略池 Voting 候选，n=2：{RULE_A}、{RULE_B}",
            "voting_candidate_strategy_type_not_grounded",
        ),
        (
            f"构建审批策略池 Voting 候选，n=1 还是 n=2：{RULE_A}、{RULE_B}",
            "voting_candidate_n_not_grounded",
        ),
        (
            f"构建审批策略池 Voting 候选，3 选 2：{RULE_A}、{RULE_B}",
            "voting_candidate_n_not_grounded",
        ),
    ],
)
def test_voting_candidate_fails_closed_when_controls_are_not_grounded(
    utterance: str,
    code: str,
) -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B]))

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == code


def test_voting_candidate_rejects_historical_rule_ids_as_current_controls() -> None:
    llm = _FakeLLM(_payload(rule_ids=[RULE_A, RULE_B, RULE_C]))
    utterance = (
        f"之前的规则是 {RULE_A}、{RULE_B}；"
        f"现在构建审批策略池 Voting 候选，n=2：{RULE_B}、{RULE_C}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert result.clarification_code == "voting_candidate_positive_command_required"


@pytest.mark.parametrize(
    "placement_mode",
    ["before_selected_members", "replace_selected_members"],
)
def test_pool_add_accepts_only_reviewed_voting_placement_modes(
    placement_mode: str,
) -> None:
    result = validate_strategy_request(
        _pool_add_payload(placement_mode=placement_mode),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert (
        result.draft.to_dict()["workflow_inputs"]["placement_mode"]
        == placement_mode
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("placement_mode", "before_pool_position_2"),
        ("position", 2),
    ],
)
def test_pool_add_rejects_unreviewed_or_numeric_placement_controls(
    field: str,
    value: object,
) -> None:
    result = validate_strategy_request(
        _pool_add_payload(**{field: value}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert field in result.clarification


@pytest.mark.parametrize(
    ("placement_text", "placement_mode"),
    [
        ("放置方式: before_selected_members", "before_selected_members"),
        (
            "放置方式: before_selected_members，"
            "保留成员作为未达 n 时的后续规则",
            "before_selected_members",
        ),
        ("放置方式: replace_selected_members", "replace_selected_members"),
        ("保留成员作为回退并放在成员前", "before_selected_members"),
        ("由 Voting 替代成员", "replace_selected_members"),
    ],
)
def test_pool_add_grounds_explicit_voting_placement(
    placement_text: str,
    placement_mode: str,
) -> None:
    llm = _FakeLLM(_pool_add_payload(placement_mode=placement_mode))
    utterance = (
        f"把 {VOTING_ASSET} 加入审批策略池；"
        f"默认动作：approval；命中动作：reject；{placement_text}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is not None
    assert (
        result.draft.to_dict()["workflow_inputs"]["placement_mode"]
        == placement_mode
    )
    assert placement_mode in result.confirmation


def test_pool_add_without_placement_can_reach_turn_handler_follow_up() -> None:
    llm = _FakeLLM(_pool_add_payload())
    utterance = (
        f"把 {VOTING_ASSET} 加入审批策略池；"
        "默认动作：approval；命中动作：reject"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is not None
    assert "placement_mode" not in result.draft.to_dict()["workflow_inputs"]


@pytest.mark.parametrize(
    ("utterance_suffix", "placement_mode"),
    [
        ("放置方式：默认", "before_selected_members"),
        ("放置方式：before_selected_members", "replace_selected_members"),
        ("保留成员作为回退并放在成员前；由 Voting 替代成员", "before_selected_members"),
    ],
)
def test_pool_add_rejects_ambiguous_or_forged_voting_placement(
    utterance_suffix: str,
    placement_mode: str,
) -> None:
    llm = _FakeLLM(_pool_add_payload(placement_mode=placement_mode))
    utterance = (
        f"把 {VOTING_ASSET} 加入审批策略池；"
        f"默认动作：approval；命中动作：reject；{utterance_suffix}"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert (
        result.clarification_code
        == "strategy_pool_add_placement_mode_not_grounded"
    )


def test_pool_add_does_not_silently_drop_explicit_placement() -> None:
    llm = _FakeLLM(_pool_add_payload())
    utterance = (
        f"把 {VOTING_ASSET} 加入审批策略池；"
        "默认动作：approval；命中动作：reject；"
        "放置方式：before_selected_members"
    )

    result = compile_strategy_request(utterance, allowed_columns=(), llm=llm)

    assert result.draft is None
    assert (
        result.clarification_code
        == "strategy_pool_add_placement_mode_not_grounded"
    )
