"""Natural-language compiler contract for exact automatic-tree leaf pointers."""

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
LEAF_A = "leaf-" + "1" * 20
LEAF_B = "leaf-" + "2" * 20


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "tree_asset_id": ASSET_A,
        "leaf_id": LEAF_A,
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_leaf_materialization",
        "workflow_inputs": inputs,
    }


def test_leaf_materialization_validates_exact_pointer_and_echoes_non_action_scope() -> (
    None
):
    result = validate_strategy_request(
        _payload(selection_reason="人工确认该叶用于下一轮评审"),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict() == _payload(
        selection_reason="人工确认该叶用于下一轮评审"
    )
    assert "automatic_tree_leaf_materialization" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "automatic_tree_leaf_materialization"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "精确叶节点物化" in result.confirmation
    assert ASSET_A in result.confirmation
    assert LEAF_A in result.confirmation
    assert "pointer" in result.confirmation
    assert "不复制规则、条件、指标或业务动作" in result.confirmation
    assert "不会加入 Strategy Pool" in result.confirmation
    assert "不会采纳或部署" in result.confirmation


def test_leaf_materialization_canonicalizes_reason_with_existing_unicode_whitespace_contract() -> (
    None
):
    result = validate_strategy_request(
        _payload(selection_reason="  人工\t确认  e\u0301 风险\n复核  "),
        allowed_columns=(),
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"]["selection_reason"] == (
        "人工 确认 é 风险 复核"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tree_asset_id": "candidate-asset-short"}, "tree_asset_id"),
        ({"tree_asset_id": "candidate-asset-" + "A" * 32}, "tree_asset_id"),
        ({"leaf_id": "leaf-short"}, "leaf_id"),
        ({"leaf_id": "leaf-" + "A" * 20}, "leaf_id"),
        ({"selection_reason": "   \n  "}, "selection_reason"),
        ({"selection_reason": "contains\x00nul"}, "selection_reason"),
    ],
)
def test_leaf_materialization_rejects_invalid_ids_or_reason(
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
        "expected_asset_id",
        "expected_asset_hash",
        "expected_tree_result_hash",
        "artifact_id",
        "artifact_hash",
        "asset_hash",
        "tree_result_hash",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
        "condition",
        "metrics",
        "action",
        "default_action",
        "dataset_id",
        "target_col",
    ],
)
def test_leaf_materialization_rejects_every_platform_owned_field(
    platform_field: str,
) -> None:
    result = validate_strategy_request(
        _payload(**{platform_field: "forged"}),
        allowed_columns=(),
    )

    assert result.draft is None
    assert "不支持的字段" in result.clarification
    assert platform_field in result.clarification


def test_leaf_materialization_compiles_one_exact_asset_and_leaf_with_verbatim_reason() -> (
    None
):
    reply = _payload(selection_reason="人工 确认 é 风险 复核")
    llm = _FakeLLM(reply)

    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，选择理由：  人工\t确认  e\u0301 风险\n复核",
        allowed_columns=(),
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == reply
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt_version"] == 41
    assert "automatic_tree_leaf_materialization" in call["system_prompt"]
    assert "pointer" in call["system_prompt"]
    assert "leaf_id" in call["user_prompt"]


@pytest.mark.parametrize(
    ("utterance", "code", "fields"),
    [
        (
            f"从刚才那棵树物化 {LEAF_A}",
            "automatic_tree_leaf_explicit_ids_required",
            {"tree_asset_id"},
        ),
        (
            f"从 {ASSET_A} 物化这个叶子",
            "automatic_tree_leaf_explicit_ids_required",
            {"leaf_id"},
        ),
        (
            f"从 {ASSET_A} 物化 {LEAF_A}，另一个资产是 {ASSET_B}",
            "automatic_tree_leaf_explicit_ids_required",
            {"tree_asset_id"},
        ),
        (
            f"从 {ASSET_A} 物化 {LEAF_A} 或 {LEAF_B}",
            "automatic_tree_leaf_explicit_ids_required",
            {"leaf_id"},
        ),
        (
            f"从 {ASSET_B} 物化 {LEAF_A}",
            "automatic_tree_leaf_controls_not_grounded",
            {"tree_asset_id"},
        ),
        (
            f"从 {ASSET_A} 物化 {LEAF_B}",
            "automatic_tree_leaf_controls_not_grounded",
            {"leaf_id"},
        ),
    ],
)
def test_leaf_materialization_requires_one_bidirectionally_grounded_full_id_each(
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


def test_leaf_materialization_accepts_repeated_mentions_of_the_same_exact_ids() -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}；再次确认资产 {ASSET_A}、叶子 {LEAF_A}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    "utterance",
    [
        f"从 {ASSET_A} 自动选择坏率最高的叶子",
        f"从 {ASSET_A} 自动选择风险最高叶节点",
        f"select the best leaf from {ASSET_A}",
        f"pick the highest risk leaf from {ASSET_A}",
        f"从 {ASSET_A} 物化 {LEAF_A}，它是最优叶子",
    ],
)
def test_leaf_materialization_rejects_heuristic_or_superlative_selection(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_selection_ambiguous"
    assert "明确" in result.clarification
    assert "leaf ID" in result.clarification


def test_leaf_materialization_rejects_llm_rewritten_selection_reason() -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，理由是用于风险复核",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason="用于风险人工复核")),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_reason_not_grounded"
    assert result.clarification_fields == ("selection_reason",)


@pytest.mark.parametrize(
    "workflow_inputs",
    [
        {"tree_asset_id": ASSET_A, "leaf_id": LEAF_A},
        {
            "tree_asset_id": ASSET_A,
            "leaf_id": LEAF_A,
            "selection_reason": "物化",
        },
    ],
)
def test_leaf_materialization_reason_is_bidirectionally_and_semantically_grounded(
    workflow_inputs: dict[str, object],
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，选择理由：风险复核",
        allowed_columns=(),
        llm=_FakeLLM(
            {
                "request_kind": "standard_workflow",
                "workflow": "automatic_tree_leaf_materialization",
                "workflow_inputs": workflow_inputs,
            }
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_reason_not_grounded"
    assert result.clarification_fields == ("selection_reason",)


def test_leaf_materialization_never_persists_an_explicitly_negated_reason() -> None:
    utterance = (
        f"不要使用选择理由：风险复核，只物化 {ASSET_A} 的 {LEAF_A}"
    )

    invented = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason="风险复核")),
    )
    omitted = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert invented.draft is None
    assert invented.clarification_code == "automatic_tree_leaf_reason_not_grounded"
    assert omitted.draft is not None
    assert "selection_reason" not in omitted.draft.to_dict()["workflow_inputs"]


def test_leaf_materialization_reparses_reason_after_negated_reason_contrast() -> None:
    utterance = (
        "不要使用理由：旧理由但选择理由：风险复核，只物化 "
        f"{ASSET_A} 的 {LEAF_A}"
    )

    omitted = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )
    grounded = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason="风险复核")),
    )

    assert omitted.draft is None
    assert omitted.clarification_code == "automatic_tree_leaf_reason_not_grounded"
    assert grounded.draft is not None
    assert grounded.draft.to_dict()["workflow_inputs"]["selection_reason"] == (
        "风险复核"
    )


def test_leaf_materialization_rejects_nested_reason_replacement_syntax() -> None:
    utterance = (
        "不要使用理由：旧理由改为选择理由：风险复核，只物化 "
        f"{ASSET_A} 的 {LEAF_A}"
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_reason_not_grounded"


@pytest.mark.parametrize(
    "utterance",
    [
        f"不要物化自动树资产 {ASSET_A} 的叶节点 {LEAF_A}",
        f"不物化自动树资产 {ASSET_A} 的叶节点 {LEAF_A}",
        f"do not materialize leaf {LEAF_A} from {ASSET_A}",
        f"不要选择叶节点 {LEAF_A}，资产是 {ASSET_A}",
        f"自动树资产 {ASSET_A}，叶节点 {LEAF_A}，选择理由：物化用于复核",
        f"自动树资产 {ASSET_A}，叶节点 {LEAF_A}，原因：select this later",
    ],
)
def test_leaf_materialization_requires_positive_materialization_intent(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_intent_negated"


@pytest.mark.parametrize(
    "follow_up",
    [
        "并加入策略池",
        "并写入 Strategy Pool",
        "并设置为拒绝动作",
        "并拒绝命中该叶子的客户",
        "并让命中客户通过审批",
        "并转人工复核",
        "并采纳这条规则",
        "并部署上线",
        "并把叶ID写回数据集",
        "拒绝命中该叶子的客户",
        "采纳这条规则",
        "部署上线",
        "action=reject",
        "不要加入策略池但要部署上线",
        "do not add it to the pool but deploy it",
        "and add it to the strategy pool",
        "and set the action to reject",
        "and adopt it",
        "and deploy it",
        "and write back the leaf id",
    ],
)
def test_leaf_materialization_rejects_chained_pool_action_lifecycle_or_writeback(
    follow_up: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，{follow_up}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_single_step_required"
    assert "只创建叶节点指针" in result.clarification


@pytest.mark.parametrize(
    "follow_up",
    [
        "并加入规则池",
        "然后加到策略池",
        "并加入 pool",
        "命中后拒绝客户",
        "把动作改成拒绝",
        "随后投产",
        "把叶ID回填到数据集",
    ],
)
def test_leaf_materialization_rejects_every_unconsumed_follow_up_clause(
    follow_up: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，{follow_up}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_single_step_required"


@pytest.mark.parametrize(
    "follow_up",
    [
        "无需入池直接部署上线",
        "不要入池随后部署上线",
        "without adding it to the pool deploy it",
    ],
)
def test_leaf_materialization_negation_cannot_swallow_a_later_positive_action(
    follow_up: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，{follow_up}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_single_step_required"


@pytest.mark.parametrize(
    "reason",
    [
        "risk review then deploy it",
        "风险复核随后投产",
        "风险复核后启用这条规则",
        "风险复核后发布",
        "risk review and activate this rule",
        "risk review before putting it into production",
        "风险复核后执行该规则",
        "风险复核后应用该规则",
        "risk review and execute this rule",
        "risk review before applying this rule",
        "命中客户拒绝",
        "命中客户走人工复核",
        "reject matching customers",
        "approve matching customers",
        "route matching customers to manual review",
        "go live with this rule",
        "roll out this rule",
        "拒绝",
        "拒绝他们",
        "reject",
        "reject them",
        "approve them",
        "route them to manual review",
    ],
)
def test_leaf_materialization_reason_cannot_hide_a_follow_up_operation(
    reason: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，选择理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason=reason)),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_single_step_required"


@pytest.mark.parametrize(
    "reason",
    [
        "批准命中客户",
        "放行命中客户",
        "命中后转人工审核",
        "批准",
        "put this into effect",
        "风险评审后批准",
        "人工复核后放行",
    ],
)
def test_leaf_materialization_reason_requires_positive_audit_rationale(
    reason: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，选择理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason=reason)),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_reason_not_grounded"


@pytest.mark.parametrize(
    "selection_text",
    [
        "坏账率最高的叶节点",
        "最差叶子",
        "worst leaf",
        "highest bad rate leaf",
    ],
)
def test_leaf_materialization_rejects_any_metric_or_extreme_selection_semantics(
    selection_text: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化{selection_text} {LEAF_A}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_selection_ambiguous"


@pytest.mark.parametrize(
    "reason",
    [
        "该叶是违约率最高的叶节点",
        "this is the top-risk leaf",
        "该叶预期损失最大",
        "该叶KS最大",
        "该叶排名第1",
        "this leaf has the largest expected loss",
        "this leaf is the most risky one",
        "this leaf is Top1 by expected loss",
        "该叶风险第2名",
        "该叶风险次高",
        "this leaf is number one by loss",
        "this leaf has the greatest loss",
        "this is the riskiest leaf",
        "该叶风险高于所有其他叶节点",
        "this leaf has higher risk than every other leaf",
        "该叶风险NO.1",
        "该叶风险前1名",
        "所有其他叶节点风险都低于该叶",
        "every other leaf has lower risk than this leaf",
    ],
)
def test_leaf_materialization_reason_cannot_hide_extreme_selection(
    reason: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason=reason)),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_leaf_selection_ambiguous"


def test_leaf_materialization_preserves_pointer_only_review_reason() -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，理由是后续由人工评审",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason="后续由人工评审")),
    )

    assert result.draft is not None


def test_leaf_materialization_accepts_complete_business_review_rationale() -> None:
    reason = "业务评审依据"
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，理由：{reason}",
        allowed_columns=(),
        llm=_FakeLLM(_payload(selection_reason=reason)),
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    "negated_follow_up",
    [
        "不要加入策略池，也不要采纳或部署",
        "do not add it to the strategy pool and do not deploy it",
        "不要自动选择最好叶子，只物化这个精确 ID",
    ],
)
def test_leaf_materialization_allows_explicitly_negated_follow_up_operations(
    negated_follow_up: str,
) -> None:
    result = compile_strategy_request(
        f"从 {ASSET_A} 物化 {LEAF_A}，{negated_follow_up}",
        allowed_columns=(),
        llm=_FakeLLM(_payload()),
    )

    assert result.draft is not None
