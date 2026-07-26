"""Natural-language compiler contract for automatic-tree candidate builds."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    STRATEGY_REQUEST_JSON_SCHEMA,
    compile_strategy_request,
    validate_strategy_request,
)


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {"features": ["age", "income"]}
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_candidate_build",
        "workflow_inputs": inputs,
    }


def test_automatic_tree_build_validates_user_owned_inputs_and_echoes_scope() -> None:
    payload = _payload(
        sample_weight_col="weight",
        directions={"age": "increasing", "income": "decreasing"},
        max_depth=5,
        min_leaf_count=120,
        min_weight_fraction_leaf=0.05,
        seed=42,
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
    )

    result = validate_strategy_request(
        payload,
        allowed_columns=[
            "age",
            "income",
            "weight",
            "loan_amount",
            "overdue_amount",
            "bad",
        ],
        target_col="bad",
    )

    assert result.draft is not None
    assert result.draft.to_dict() == payload
    assert "automatic_tree_candidate_build" in STANDARD_STRATEGY_WORKFLOWS
    assert (
        "automatic_tree_candidate_build"
        in STRATEGY_REQUEST_JSON_SCHEMA["schema"]["properties"]["workflow"]["enum"]
    )
    assert "自动决策树候选构建" in result.confirmation
    assert "age、income" in result.confirmation
    assert "权重列 weight" in result.confirmation
    assert "age=递增" in result.confirmation
    assert "income=递减" in result.confirmation
    assert "风险方向诊断期望" in result.confirmation
    assert "方向约束" not in result.confirmation
    assert "最大深度 5" in result.confirmation
    assert "最小叶样本数 120" in result.confirmation
    assert "最小叶权重占比 5.00%" in result.confirmation
    assert "随机种子 42" in result.confirmation
    assert "放款金额列 loan_amount" in result.confirmation
    assert "逾期金额列 overdue_amount" in result.confirmation
    assert "不会自动选择叶子" in result.confirmation
    assert "Strategy Pool" in result.confirmation


def test_automatic_tree_build_keeps_omitted_tool_defaults_omitted() -> None:
    result = validate_strategy_request(
        _payload(features=["age"]),
        allowed_columns=["age"],
        target_col="bad",
    )

    assert result.draft is not None
    assert result.draft.to_dict()["workflow_inputs"] == {"features": ["age"]}
    for forbidden_default in (
        "directions",
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "sample_weight_col",
        "seed",
    ):
        assert forbidden_default not in result.confirmation


@pytest.mark.parametrize(
    "utterance",
    [
        "features: age, income",
        "用 age, income 构建自动决策树候选",
    ],
)
def test_automatic_tree_build_grounds_comma_separated_feature_lists(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age", "income"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age", "income"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    ("inputs", "allowed_columns", "target_col", "message"),
    [
        ({}, ["age"], "bad", "features"),
        ({"features": []}, ["age"], "bad", "features"),
        ({"features": ["age", "age"]}, ["age"], "bad", "重复"),
        ({"features": ["ghost"]}, ["age"], "bad", "ghost"),
        ({"features": ["bad"]}, ["bad"], "bad", "目标列"),
        (
            {"features": ["age"], "directions": {}},
            ["age"],
            "bad",
            "风险方向诊断期望",
        ),
        (
            {"features": ["age"], "directions": {"income": "increasing"}},
            ["age", "income"],
            "bad",
            "未选择",
        ),
        (
            {"features": ["age"], "directions": {"age": "up"}},
            ["age"],
            "bad",
            "increasing",
        ),
        (
            {"features": ["age"], "sample_weight_col": "age"},
            ["age"],
            "bad",
            "不同字段",
        ),
        (
            {
                "features": ["age"],
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "loan_amount",
            },
            ["age", "loan_amount"],
            "bad",
            "不同字段",
        ),
        (
            {"features": ["age"], "sample_weight_col": "bad"},
            ["age", "bad"],
            "bad",
            "目标列",
        ),
        ({"features": ["age"], "max_depth": 9}, ["age"], "bad", "1 到 8"),
        (
            {"features": ["age"], "min_leaf_count": 0},
            ["age"],
            "bad",
            "正整数",
        ),
        (
            {"features": ["age"], "min_weight_fraction_leaf": 0.51},
            ["age"],
            "bad",
            "0 到 0.5",
        ),
        (
            {"features": ["age"], "seed": 4_294_967_296},
            ["age"],
            "bad",
            "4294967295",
        ),
        (
            {"features": ["age"], "dataset_id": "llm-owned"},
            ["age"],
            "bad",
            "不支持的字段",
        ),
        (
            {"features": ["age"], "metrics": {"ks": 0.8}},
            ["age"],
            "bad",
            "不支持的字段",
        ),
    ],
)
def test_automatic_tree_build_rejects_invalid_or_platform_owned_inputs(
    inputs: dict,
    allowed_columns: list[str],
    target_col: str | None,
    message: str,
) -> None:
    result = validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": inputs,
        },
        allowed_columns=allowed_columns,
        target_col=target_col,
    )

    assert result.draft is None
    assert message in result.clarification


def test_automatic_tree_build_compilation_requires_all_explicit_controls_in_source() -> (
    None
):
    reply = _payload(
        sample_weight_col="weight",
        directions={"age": "increasing", "income": "decreasing"},
        max_depth=5,
        min_leaf_count=120,
        min_weight_fraction_leaf=0.05,
        seed=42,
        loan_amount_col="loan_amount",
        overdue_amount_col="overdue_amount",
    )
    llm = _FakeLLM(reply)

    result = compile_strategy_request(
        "用特征 age 和 income 构建自动决策树候选；权重列 weight；"
        "age 单调递增，income 单调递减；最大深度 5；最小叶样本数 120；"
        "最小叶权重占比 5%；随机种子 42；放款金额列 loan_amount；"
        "逾期金额列 overdue_amount。",
        allowed_columns=[
            "age",
            "income",
            "weight",
            "loan_amount",
            "overdue_amount",
        ],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.to_dict() == reply
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["prompt_version"] == 44
    assert "automatic_tree_candidate_build" in call["system_prompt"]
    assert "不能串联" in call["system_prompt"]
    assert "最好叶子" in call["system_prompt"]
    assert "dataset_id" in call["system_prompt"]
    assert "budgets" in call["system_prompt"]


@pytest.mark.parametrize(
    ("utterance", "reply_inputs", "missing"),
    [
        ("用 age 构建自动树候选", {"features": ["age", "income"]}, "income"),
        (
            "用 age 构建自动树候选，权重列 weight",
            {"features": ["age"], "sample_weight_col": "other_weight"},
            "sample_weight_col=other_weight",
        ),
        (
            "用特征 age 构建自动树候选，权重列 weight 放款金额列 loan_amount",
            {
                "features": ["age"],
                "sample_weight_col": "loan_amount",
                "loan_amount_col": "weight",
            },
            "sample_weight_col=loan_amount",
        ),
        (
            "用 age 构建自动树候选，放款金额字段 loan_amount",
            {"features": ["age", "loan_amount"]},
            "loan_amount",
        ),
        (
            "用 age 构建自动树候选，age 单调递增",
            {"features": ["age"], "directions": {"age": "decreasing"}},
            "age=decreasing",
        ),
        (
            "用 age 和 income 构建自动树候选，age 递增 income 递减",
            {
                "features": ["age", "income"],
                "directions": {"age": "decreasing", "income": "increasing"},
            },
            "age=decreasing",
        ),
        (
            "用 age 构建自动树候选，最大深度 4",
            {"features": ["age"], "max_depth": 5},
            "max_depth=5",
        ),
        (
            "用 age 构建自动树候选，最小叶权重占比 5%",
            {"features": ["age"], "min_weight_fraction_leaf": 0.08},
            "min_weight_fraction_leaf=0.08",
        ),
    ],
)
def test_automatic_tree_build_rejects_llm_controls_not_grounded_in_user_text(
    utterance: str,
    reply_inputs: dict,
    missing: str,
) -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": reply_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=[
            "age",
            "income",
            "weight",
            "other_weight",
            "loan_amount",
        ],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_controls_not_grounded"
    assert missing in result.clarification_fields
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    ("utterance", "reply_inputs", "missing"),
    [
        (
            "用特征 age 和 income 构建自动树候选",
            {"features": ["age"]},
            "features includes income",
        ),
        (
            "用 age 构建自动树候选，样本权重列 weight",
            {"features": ["age"]},
            "sample_weight_col=weight",
        ),
        (
            "用 age 构建自动树候选，age 递增",
            {"features": ["age"]},
            "directions.age=increasing",
        ),
        (
            "用 age 构建自动树候选，min_leaf_count=500",
            {"features": ["age"]},
            "min_leaf_count=500",
        ),
    ],
)
def test_automatic_tree_build_rejects_llm_omission_of_explicit_user_controls(
    utterance: str,
    reply_inputs: dict,
    missing: str,
) -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": reply_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age", "income", "weight"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_controls_not_grounded"
    assert missing in result.clarification_fields


@pytest.mark.parametrize(
    ("utterance", "reply_inputs", "missing"),
    [
        (
            "用 age 构建自动树候选，候选变量不要 loan_amount",
            {"features": ["age", "loan_amount"]},
            "loan_amount",
        ),
        (
            "用 age 构建自动树候选，排除 loan_amount 和 overdue_amount",
            {"features": ["age", "loan_amount", "overdue_amount"]},
            "loan_amount",
        ),
        (
            "除了 age 和 income，用 score 构建自动树候选",
            {"features": ["age", "income", "score"]},
            "age",
        ),
        (
            "用 age 构建自动树候选，age 不要递增",
            {"features": ["age"], "directions": {"age": "increasing"}},
            "age=increasing",
        ),
        (
            "用 feature_seed42 构建自动树候选",
            {"features": ["feature_seed42"], "seed": 42},
            "seed=42",
        ),
        (
            "用 age_increasing_flag 构建自动树候选",
            {
                "features": ["age_increasing_flag"],
                "directions": {"age_increasing_flag": "increasing"},
            },
            "age_increasing_flag=increasing",
        ),
    ],
)
def test_automatic_tree_build_grounding_honors_negation_and_token_boundaries(
    utterance: str,
    reply_inputs: dict,
    missing: str,
) -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": reply_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=[
            "age",
            "income",
            "score",
            "loan_amount",
            "overdue_amount",
            "feature_seed42",
            "age_increasing_flag",
        ],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_controls_not_grounded"
    assert missing in result.clarification_fields


def test_automatic_tree_build_treats_chinese_additive_chule_as_inclusion() -> None:
    llm = _FakeLLM(_payload(features=["age", "income", "score"]))

    result = compile_strategy_request(
        "除了 age 和 income，还用 score 构建自动树候选",
        allowed_columns=["age", "income", "score"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    ("utterance", "reply_inputs", "accepted"),
    [
        (
            "用 age 构建自动树候选，样本权重列 w 改为 w2",
            {"features": ["age"], "sample_weight_col": "w"},
            False,
        ),
        (
            "用 age 构建自动树候选，样本权重列 w 改为 w2",
            {"features": ["age"], "sample_weight_col": "w2"},
            True,
        ),
        (
            "用 age 构建自动树候选，最大深度 4 改为 5",
            {"features": ["age"], "max_depth": 4},
            False,
        ),
        (
            "用 age 构建自动树候选，最大深度 4 改为 5",
            {"features": ["age"], "max_depth": 5},
            True,
        ),
    ],
)
def test_automatic_tree_build_grounding_uses_replacement_value(
    utterance: str,
    reply_inputs: dict,
    accepted: bool,
) -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": reply_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age", "w", "w2"],
        target_col="bad",
        llm=llm,
    )

    assert (result.draft is not None) is accepted


def test_automatic_tree_build_does_not_accept_build_select_pool_chain() -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        "用 age 建树，然后自动选最好叶子并加入策略池",
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"
    assert "单独完成候选树构建" in result.clarification


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树，并把叶子 leaf-1 作为拒绝规则",
        "用 age 建树，采用坏率最高叶节点",
        "用 age 建树，设置叶节点为拒绝动作",
        "用 age 建树后直接拒绝高风险叶子",
        "用 age 建树，并对高风险叶子执行拒绝",
        "用 age 建树，leaf-1 拒绝",
        "用 age 建树，让风险最高叶子走人工复核",
        "用 age 建树，依据叶节点给出审批动作",
        "build a tree with age, then reject high-risk leaves",
        "build a tree with age, route leaf-1 to manual review",
    ],
)
def test_automatic_tree_build_rejects_implicit_leaf_follow_up_actions(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树，不要自动选最好叶子但把 leaf-1 设为拒绝规则",
        "build a tree with age, do not auto-pick best leaf but use leaf-1 as reject",
    ],
)
def test_automatic_tree_build_rejects_positive_action_after_negated_clause(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树不要使用 income 后直接拒绝高风险叶子",
        "用 age 建树，不要自动选最好叶子且把 leaf-1 设为拒绝规则",
        "用 age 建树，不用 income 直接拒绝高风险叶子",
        "用 age 建树，不要使用 income 并直接拒绝高风险叶子",
        "用 age 建树，不要使用 income 之后直接拒绝高风险叶子",
        "用 age 建树，不要使用 income 接着让 leaf-1 走人工复核",
        "用 age 建树，不要使用 income 可是依据叶节点给出审批动作",
        "用 age 建树，不要自动选最好叶子再把 leaf-1 设为拒绝规则",
        "build a tree with age, do not use income yet route leaf-1 to manual review",
        "build a tree with age, do not use income afterwards reject high-risk leaves",
        "用 age 建树，不要考虑收益直接拒绝高风险叶子",
        "用 age 建树，不要生成报告直接拒绝高风险叶子",
        "用 age 建树，无需权重直接拒绝高风险叶子",
        "用 age 建树，不要使用金额口径直接拒绝高风险叶子",
    ],
)
def test_automatic_tree_build_rejects_action_after_local_negation(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age", "income"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"


@pytest.mark.parametrize(
    ("utterance", "reply_inputs"),
    [
        (
            "用 age 建树，最小叶样本数 100",
            {"features": ["age"], "min_leaf_count": 100},
        ),
        ("用 age 建树，仅展示完整叶子列表", {"features": ["age"]}),
        (
            "用 age 建树，不要把任何叶子作为拒绝规则，也不要设置叶节点动作",
            {"features": ["age"]},
        ),
        (
            "build a tree with age, do not auto-pick best leaf and do not use "
            "leaf-1 as reject",
            {"features": ["age"]},
        ),
        (
            "用 age 建树，不要直接拒绝高风险叶子，也不要让叶节点走人工复核",
            {"features": ["age"]},
        ),
        (
            "用 age 建树，不要自动选最好叶子且不要把 leaf-1 设为拒绝规则",
            {"features": ["age"]},
        ),
        (
            "build a tree with age, do not route leaf-1 to manual review",
            {"features": ["age"]},
        ),
        (
            "build a tree with age, do not automatically select best leaf",
            {"features": ["age"]},
        ),
        (
            "用 age 建树，不要再直接拒绝高风险叶子",
            {"features": ["age"]},
        ),
        (
            "用 age 建树，无需再让 leaf-1 走人工复核",
            {"features": ["age"]},
        ),
        ("build a tree with age, only display complete leaves", {"features": ["age"]}),
    ],
)
def test_automatic_tree_build_follow_up_guard_preserves_safe_leaf_phrasing(
    utterance: str,
    reply_inputs: dict,
) -> None:
    llm = _FakeLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": reply_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


def test_automatic_tree_build_rejects_reversed_best_leaf_request() -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        "用 age 建树，然后把最好叶子自动选出来",
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"


def test_automatic_tree_build_allows_explicitly_negated_follow_up_actions() -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        "只用 age 建树，不要自动选最好叶子，也不要加入策略池",
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树并采用坏率最高的节点作为拒绝规则",
        "用 age 建树并让风险最高节点走人工复核",
        "用 age 建树并直接生成拒绝策略",
        "用 age 建树后自动形成审批规则",
        "用 age 建树并按坏率给叶子排名",
        "build a tree with age and rank leaves by bad rate",
        "用 age 建树后采用 leaf-1",
        "用 age 建树后保留 leaf-1",
        "用 age 建树并提取高风险叶子",
        "build a tree with age and retain leaf-1",
        "用 age 建树并采纳这棵树",
        "用 age 建树然后部署上线",
        "build a tree with age and adopt it",
        "用 age 建树并写回叶ID",
        "用 age 建树并自动形成拒绝规则",
    ],
)
def test_automatic_tree_build_rejects_request_level_forbidden_intents(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_single_step_required"


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树并生成叶节点规则",
        "用 age 建树并导出树规则",
        "用 age 建树，展示完整叶列表、节点指标和路径规则，导出 "
        "JSON、Python、SQL、SVG、PNG、XLSX",
        "用 age 建树，不要按坏率给叶子排名，也不要部署上线",
        "build a tree with age, do not retain leaf-1 and do not adopt it",
    ],
)
def test_automatic_tree_build_allows_build_deliverables_and_negated_forbidden_intents(
    utterance: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    "utterance",
    [
        "不要用 age 建树",
        "don't build a tree with age",
    ],
)
def test_automatic_tree_build_rejects_negated_build_intent(utterance: str) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_intent_negated"


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "dataset_id=dataset-other，用 age 建树",
            "automatic_tree_build_dataset_context_required",
        ),
        (
            "用另一个样本 dataset-other 的 age 建树",
            "automatic_tree_build_dataset_context_required",
        ),
        (
            "target_col=other_bad，用 age 建树",
            "automatic_tree_build_target_context_required",
        ),
        (
            "drop_nan_labels=true，用 age 建树",
            "automatic_tree_build_label_policy_not_overridable",
        ),
        (
            "budgets.max_rows=1000，用 age 建树",
            "automatic_tree_build_platform_budget_not_overridable",
        ),
        (
            "用 age 建树，最多只处理1000行",
            "automatic_tree_build_platform_budget_not_overridable",
        ),
        (
            "用 age 建树前先告诉我 max_rows 默认是多少？",
            "automatic_tree_build_platform_budget_not_overridable",
        ),
    ],
)
def test_automatic_tree_build_rejects_platform_owned_controls_omitted_by_llm(
    utterance: str,
    code: str,
) -> None:
    llm = _FakeLLM(_payload(features=["age"]))

    result = compile_strategy_request(
        utterance,
        allowed_columns=["age", "other_bad"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == code


@pytest.mark.parametrize(
    ("short_column", "long_column"),
    [
        ("收入", "月收入"),
        ("金额", "放款金额"),
        ("年龄", "用户年龄"),
    ],
)
def test_automatic_tree_build_prefers_longest_overlapping_column_mention(
    short_column: str,
    long_column: str,
) -> None:
    llm = _FakeLLM(_payload(features=[long_column]))

    result = compile_strategy_request(
        f"用 {long_column} 构建自动树",
        allowed_columns=[short_column, long_column],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None


@pytest.mark.parametrize(
    ("short_column", "long_column"),
    [
        ("收入", "月收入"),
        ("金额", "放款金额"),
        ("年龄", "用户年龄"),
    ],
)
def test_automatic_tree_build_keeps_independent_short_column_mention(
    short_column: str,
    long_column: str,
) -> None:
    utterance = f"用 {long_column} 和 {short_column} 构建自动树"
    accepted = compile_strategy_request(
        utterance,
        allowed_columns=[short_column, long_column],
        target_col="bad",
        llm=_FakeLLM(_payload(features=[long_column, short_column])),
    )
    omitted = compile_strategy_request(
        utterance,
        allowed_columns=[short_column, long_column],
        target_col="bad",
        llm=_FakeLLM(_payload(features=[long_column])),
    )

    assert accepted.draft is not None
    assert omitted.draft is None
    assert omitted.clarification_code == "automatic_tree_build_controls_not_grounded"
    assert f"features includes {short_column}" in omitted.clarification_fields


@pytest.mark.parametrize("llm_feature", ["甲乙", "乙丙"])
def test_automatic_tree_build_rejects_crossing_column_mentions_as_ambiguous(
    llm_feature: str,
) -> None:
    result = compile_strategy_request(
        "用 甲乙丙 建树",
        allowed_columns=["甲乙", "乙丙"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=[llm_feature])),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_column_mention_ambiguous"
    assert set(result.clarification_fields) == {"甲乙", "乙丙"}


def test_automatic_tree_build_prefers_unique_exact_case_independent_of_whitelist_order() -> (
    None
):
    accepted = compile_strategy_request(
        "用 Score 建树",
        allowed_columns=["score", "Score"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=["Score"])),
    )
    wrong_case = compile_strategy_request(
        "用 Score 建树",
        allowed_columns=["score", "Score"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=["score"])),
    )

    assert accepted.draft is not None
    assert wrong_case.draft is None
    assert wrong_case.clarification_code == "automatic_tree_build_controls_not_grounded"


def test_automatic_tree_build_rejects_case_collision_without_exact_case() -> None:
    result = compile_strategy_request(
        "用 SCORE 建树",
        allowed_columns=["score", "Score"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=["Score"])),
    )

    assert result.draft is None
    assert result.clarification_code == "automatic_tree_build_column_mention_ambiguous"
    assert set(result.clarification_fields) == {"score", "Score"}


def test_automatic_tree_direction_uses_resolved_longest_column_mentions() -> None:
    utterance = "用 收入 和 月收入 建树，月收入 递减"
    accepted = compile_strategy_request(
        utterance,
        allowed_columns=["收入", "月收入"],
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                features=["收入", "月收入"],
                directions={"月收入": "decreasing"},
            )
        ),
    )
    forged_short_direction = compile_strategy_request(
        utterance,
        allowed_columns=["收入", "月收入"],
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                features=["收入", "月收入"],
                directions={"收入": "decreasing", "月收入": "decreasing"},
            )
        ),
    )

    assert accepted.draft is not None
    assert forged_short_direction.draft is None
    assert (
        forged_short_direction.clarification_code
        == "automatic_tree_build_controls_not_grounded"
    )
    assert "收入=decreasing" in forged_short_direction.clarification_fields


def test_automatic_tree_column_role_uses_resolved_longest_column_mentions() -> None:
    utterance = "用 age 建树，放款金额 作为放款金额列"
    accepted = compile_strategy_request(
        utterance,
        allowed_columns=["age", "金额", "放款金额"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=["age"], loan_amount_col="放款金额")),
    )
    forged_short_role = compile_strategy_request(
        utterance,
        allowed_columns=["age", "金额", "放款金额"],
        target_col="bad",
        llm=_FakeLLM(_payload(features=["age"], loan_amount_col="金额")),
    )

    assert accepted.draft is not None
    assert forged_short_role.draft is None
    assert (
        forged_short_role.clarification_code
        == "automatic_tree_build_controls_not_grounded"
    )
    assert "loan_amount_col=金额" in forged_short_role.clarification_fields
