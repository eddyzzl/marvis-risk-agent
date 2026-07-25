"""Natural-language compiler contract for governed full-tree writeback."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    validate_strategy_request,
)


ASSET_A = "candidate-asset-" + "a" * 32
ASSET_B = "candidate-asset-" + "b" * 32


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {"tree_asset_id": ASSET_A}
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_apply",
        "workflow_inputs": inputs,
    }


def test_automatic_tree_apply_validates_only_user_owned_controls() -> None:
    result = validate_strategy_request(
        _payload(
            leaf_id_column="tree_leaf_bucket",
            rule_id_column="tree_rule_bucket",
        ),
        allowed_columns=("score", "income"),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(
        leaf_id_column="tree_leaf_bucket",
        rule_id_column="tree_rule_bucket",
    )
    assert "automatic_tree_apply" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "automatic_tree_apply"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "自动树全量写回" in result.confirmation
    assert ASSET_A in result.confirmation
    assert "tree_leaf_bucket" in result.confirmation
    assert "tree_rule_bucket" in result.confirmation
    assert "development / unvalidated" in result.confirmation
    assert "不会入池、采纳或部署" in result.confirmation


@pytest.mark.parametrize(
    "platform_field",
    [
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "activate_result",
        "artifact_id",
        "asset_hash",
        "tree_result_hash",
        "result_dataset_id",
        "metrics",
        "action",
    ],
)
def test_automatic_tree_apply_rejects_platform_owned_fields(
    platform_field: str,
) -> None:
    result = validate_strategy_request(
        _payload(**{platform_field: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    assert platform_field in result.clarification


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tree_asset_id": "candidate-asset-short"}, "tree_asset_id"),
        ({"tree_asset_id": "candidate-asset-" + "A" * 32}, "tree_asset_id"),
        ({"leaf_id_column": "bad-name"}, "leaf_id_column"),
        ({"rule_id_column": "1rule"}, "rule_id_column"),
        (
            {"leaf_id_column": "Tree_Node", "rule_id_column": "tree_node"},
            "必须不同",
        ),
    ],
)
def test_automatic_tree_apply_rejects_invalid_user_controls(
    overrides: dict[str, object],
    message: str,
) -> None:
    result = validate_strategy_request(_payload(**overrides), allowed_columns=())

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"
    assert message in result.clarification


def test_automatic_tree_apply_compiles_one_exact_asset_and_explicit_columns() -> None:
    reply = _payload(
        leaf_id_column="tree_leaf_bucket",
        rule_id_column="tree_rule_bucket",
    )
    llm = _FakeLLM(reply)

    result = compile_strategy_request(
        f"把自动树资产 {ASSET_A} 应用到当前样本，"
        "叶节点输出列 tree_leaf_bucket，规则输出列 tree_rule_bucket。",
        allowed_columns=("score", "income"),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == reply
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt_version"] == 35
    assert "automatic_tree_apply" in call["system_prompt"]
    assert "tree_asset_id" in call["system_prompt"]
    assert "artifact hash" in call["system_prompt"]


@pytest.mark.parametrize(
    ("utterance", "code", "fields"),
    [
        (
            "把刚才那棵自动树应用到当前样本",
            "automatic_tree_apply_explicit_asset_required",
            {"tree_asset_id"},
        ),
        (
            f"把 {ASSET_A} 和 {ASSET_B} 应用到当前样本",
            "automatic_tree_apply_explicit_asset_required",
            {"tree_asset_id"},
        ),
        (
            f"把 {ASSET_A} 与 {ASSET_A} 应用到当前样本",
            "automatic_tree_apply_explicit_asset_required",
            {"tree_asset_id"},
        ),
        (
            f"把 {ASSET_B} 应用到当前样本",
            "automatic_tree_apply_controls_not_grounded",
            {"tree_asset_id"},
        ),
    ],
)
def test_automatic_tree_apply_requires_one_bidirectionally_grounded_full_asset_id(
    utterance: str,
    code: str,
    fields: set[str],
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=("score",),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == code
    assert set(result.clarification_fields) == fields


@pytest.mark.parametrize(
    ("utterance", "reply", "fields"),
    [
        (
            f"把 {ASSET_A} 应用到当前样本，叶节点输出列 tree_leaf_bucket",
            _payload(),
            {"leaf_id_column"},
        ),
        (
            f"把 {ASSET_A} 应用到当前样本",
            _payload(leaf_id_column="invented_leaf"),
            {"leaf_id_column"},
        ),
        (
            f"把 {ASSET_A} 应用到当前样本，规则输出列 actual_rule",
            _payload(rule_id_column="other_rule"),
            {"rule_id_column"},
        ),
    ],
)
def test_automatic_tree_apply_output_columns_are_bidirectionally_grounded(
    utterance: str,
    reply: dict,
    fields: set[str],
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=("score",),
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_apply_controls_not_grounded"
    assert set(result.clarification_fields) == fields


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            f"不要把 {ASSET_A} 应用到当前样本",
            "automatic_tree_apply_intent_not_authorized",
        ),
        (
            f"可以把 {ASSET_A} 应用到当前样本吗？",
            "automatic_tree_apply_intent_not_authorized",
        ),
        (
            f"以后把 {ASSET_A} 应用到当前样本",
            "automatic_tree_apply_intent_not_authorized",
        ),
        (
            f"把 {ASSET_A} 应用到当前样本并加入策略池",
            "automatic_tree_apply_single_step_required",
        ),
        (
            f"把 {ASSET_A} 应用到当前样本然后采纳并部署",
            "automatic_tree_apply_single_step_required",
        ),
        (
            f"把 {ASSET_A} 应用到当前样本并生成报告",
            "automatic_tree_apply_single_step_required",
        ),
        (
            f"把 {ASSET_A} 应用到当前样本，输出列 result_column",
            "automatic_tree_apply_output_column_ambiguous",
        ),
    ],
)
def test_automatic_tree_apply_requires_one_positive_unambiguous_command(
    utterance: str,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=("score",),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == code


def test_explicit_automatic_tree_apply_cannot_be_rerouted_by_llm() -> None:
    result = compile_strategy_request(
        f"把自动树资产 {ASSET_A} 应用到当前样本",
        allowed_columns=("score",),
        llm=_FakeLLM(
            {
                "operation": "apply",
                "strategy_type": "approval",
                "strategy_id": "strategy-forged",
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_apply_workflow_required"
    assert result.clarification_fields == ("workflow",)
