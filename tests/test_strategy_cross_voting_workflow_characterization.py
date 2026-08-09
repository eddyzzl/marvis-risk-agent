"""Characterize legacy Voting/Cross behavior at the canonical module seam."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from marvis.agent.strategy_workflows._cross_voting import (
    CROSS_VOTING_WORKFLOW_SPECS,
    cross_matrix_analysis_confirmation,
    cross_matrix_candidate_build_from_search_confirmation,
    cross_matrix_candidate_search_confirmation,
    cross_matrix_cell_selection_confirmation,
    cross_rule_candidate_build_from_search_confirmation,
    cross_rule_search_confirmation,
    prepare_cross_matrix_analysis,
    prepare_cross_matrix_candidate_build_from_search,
    prepare_cross_matrix_candidate_search,
    prepare_cross_matrix_cell_selection,
    prepare_cross_rule_candidate_build_from_search,
    prepare_cross_rule_search,
    prepare_voting_candidate_build,
    prepare_voting_candidate_build_from_search,
    prepare_voting_candidate_search,
    validate_voting_candidate_build_from_search_inputs,
    validate_voting_candidate_build_inputs,
    validate_voting_candidate_search_inputs,
    validate_cross_matrix_analysis_inputs,
    validate_cross_matrix_candidate_build_from_search_inputs,
    validate_cross_matrix_candidate_search_inputs,
    validate_cross_matrix_cell_selection_inputs,
    validate_cross_rule_candidate_build_from_search_inputs,
    validate_cross_rule_search_inputs,
    voting_candidate_build_confirmation,
    voting_candidate_build_from_search_confirmation,
    voting_candidate_search_confirmation,
)
from marvis.agent.strategy_workflows.contracts import (
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowValidationError,
)


RULE_A = "candidate-rule-" + "a" * 32
RULE_B = "candidate-rule-" + "b" * 32
RULE_C = "candidate-rule-" + "c" * 32
SEARCH_ID = "voting-search-" + "d" * 32
COMBO_ID = "voting-combo-" + "e" * 32
CROSS_SEARCH_ID = "cross-search-" + "1" * 32
CROSS_PAIR_ID = "cross-pair-" + "2" * 32
CROSS_RULE_SEARCH_ID = "cross-rule-search-" + "3" * 32
CROSS_RULE_ID = "cross-rule-" + "4" * 32
CROSS_ASSET_ID = "candidate-asset-" + "5" * 32
CROSS_CELL_A = "cross-cell-" + "6" * 32
CROSS_CELL_B = "cross-cell-" + "7" * 32

NO_REQUIREMENTS = StrategyWorkflowRequirements(
    dataset=False,
    target=False,
    complete_labels=False,
)
CONFIRMATION_SUFFIX = (
    "；请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
)


def _resolution_context() -> StrategyWorkflowResolutionContext:
    return StrategyWorkflowResolutionContext(
        allowed_columns=("age", "score", "income", "bad"),
        target_col="bad",
    )


def test_voting_candidate_build_normalizes_only_explicit_user_controls() -> None:
    normalized = validate_voting_candidate_build_inputs(
        {
            "strategy_type": " approval ",
            "rule_ids": [f" {RULE_B} ", RULE_A],
            "n": 2,
        },
        _resolution_context(),
    )

    assert normalized == {
        "strategy_type": "approval",
        "rule_ids": [RULE_B, RULE_A],
        "n": 2,
    }


def test_voting_candidate_search_normalizes_safe_defaults_and_order() -> None:
    normalized = validate_voting_candidate_search_inputs(
        {
            "strategy_type": "approval",
            "member_count": 3,
            "n": 2,
            "objective": {"metric": "bad_rate", "direction": "minimize"},
            "constraints": [
                {"metric": "lift", "operator": "gte", "value": 1.2},
                {"metric": "hit_share", "operator": "gte", "value": 0.1},
            ],
            "include_rule_ids": [RULE_B, RULE_A],
            "exclude_rule_ids": [RULE_C],
        },
        _resolution_context(),
    )

    assert normalized == {
        "strategy_type": "approval",
        "member_count": 3,
        "n": 2,
        "objective": {"metric": "bad_rate", "direction": "minimize"},
        "constraints": [
            {"metric": "hit_share", "operator": "gte", "value": 0.1},
            {"metric": "lift", "operator": "gte", "value": 1.2},
        ],
        "include_rule_ids": [RULE_A, RULE_B],
        "exclude_rule_ids": [RULE_C],
        "max_combinations": 10_000,
    }


def test_voting_search_minimization_requires_positive_denominator_share() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        validate_voting_candidate_search_inputs(
            {
                "strategy_type": "approval",
                "member_count": 3,
                "n": 2,
                "objective": {
                    "metric": "bad_rate",
                    "direction": "minimize",
                },
            },
            _resolution_context(),
        )

    assert str(captured.value) == (
        "最小化 bad_rate 必须提供正数 hit_share gte 约束，"
        "绝对命中量不能替代占比下限。"
    )
    assert captured.value.code == "voting_search_minimum_share_required"
    assert captured.value.fields == ("constraints", "hit_share")


def test_voting_candidate_build_from_search_normalizes_exact_pointers() -> None:
    normalized = validate_voting_candidate_build_from_search_inputs(
        {
            "search_id": f" {SEARCH_ID} ",
            "combo_id": COMBO_ID,
            "strategy_type": "approval",
        },
        _resolution_context(),
    )

    assert normalized == {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
        "strategy_type": "approval",
    }


@pytest.mark.parametrize(
    ("confirmation", "inputs", "expected"),
    [
        (
            voting_candidate_search_confirmation,
            {
                "strategy_type": "approval",
                "member_count": 3,
                "n": 2,
                "objective": {"metric": "bad_rate", "direction": "minimize"},
                "constraints": [
                    {"metric": "hit_share", "operator": "gte", "value": 0.1}
                ],
                "include_rule_ids": [RULE_A],
                "exclude_rule_ids": [RULE_C],
                "max_combinations": 500,
            },
            "已识别为〔Voting 组合搜索 Workflow〕；来源 Strategy Pool 类型：approval；"
            "组合参数：K=3，n=2；排序目标：bad_rate / minimize；"
            "确定性评估预算：最多 500 个组合；资格约束：hit_share gte 0.1；"
            f"必须包含：{RULE_A}；排除：{RULE_C}；"
            "平台将在计划创建前绑定当前 Pool 身份与受治理 development 样本；"
            "无需用户或模型提供 dataset pointer；本步骤只搜索并发布聚合证据；"
            "不会构建候选或选择冠军，不会修改 Pool、入池、应用、采纳或部署",
        ),
        (
            voting_candidate_build_from_search_confirmation,
            {
                "search_id": SEARCH_ID,
                "combo_id": COMBO_ID,
                "strategy_type": "approval",
            },
            "已识别为〔Voting 搜索结果精确构建 Workflow〕；"
            f"搜索证据 pointer：{SEARCH_ID}；组合 pointer：{COMBO_ID}；"
            "来源 Strategy Pool 类型：approval；平台将在计划开始前重新校验"
            "搜索证据、组合与当前 Pool；不会采用排名、最好或冠军等启发式选择；"
            "本步骤只构建 development/backtested/unvalidated Voting 候选；"
            "不会加入或修改 Pool，不会设置动作、应用、采纳或部署",
        ),
        (
            voting_candidate_build_confirmation,
            {"strategy_type": "approval", "rule_ids": [RULE_B, RULE_A], "n": 2},
            "已识别为〔Voting / n-of-k 候选构建 Workflow〕；"
            "来源 Strategy Pool 类型：approval；"
            f"精确成员规则：{RULE_B}、{RULE_A}；"
            "组合条件：2 条规则中至少命中 2 条；平台将绑定当前 Pool revision/hash "
            "和原始样本，逐行计算命中数与风险效果；本步骤只生成 "
            "development/backtested/unvalidated 候选；不会入池、设置业务动作、"
            "采纳或部署",
        ),
    ],
)
def test_voting_confirmation_text_is_characterized(
    confirmation,
    inputs: dict[str, object],
    expected: str,
) -> None:
    assert confirmation(inputs) == expected + CONFIRMATION_SUFFIX


def test_voting_preparers_separate_user_pointers_from_platform_evidence() -> None:
    calls: list[tuple[str, MappingProxyType]] = []
    evidence = {
        "voting_candidate_search": {
            "pool_ref": {
                "artifact_id": "a" * 64,
                "expected_snapshot_hash": "b" * 64,
            }
        },
        "voting_candidate_build_from_search": {},
        "voting_candidate_build": {
            "expected_pool_revision": 7,
            "expected_pool_snapshot_hash": "c" * 64,
            "selected_entry_ids": [
                "pool-entry-" + "1" * 32,
                "pool-entry-" + "2" * 32,
            ],
        },
    }

    def bind(workflow_id, frozen_inputs):
        assert isinstance(frozen_inputs, MappingProxyType)
        calls.append((workflow_id, frozen_inputs))
        return evidence[workflow_id]

    context = StrategyWorkflowPreparationContext(
        dataset_id=None,
        bind_workflow_evidence=bind,
    )
    search_inputs = {
        "strategy_type": "approval",
        "member_count": 3,
        "n": 2,
        "objective": {"metric": "lift", "direction": "maximize"},
        "constraints": [],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 500,
    }

    search = prepare_voting_candidate_search(search_inputs, context)
    selection = prepare_voting_candidate_build_from_search(
        {"search_id": SEARCH_ID, "combo_id": COMBO_ID},
        context,
    )
    direct = prepare_voting_candidate_build(
        {"strategy_type": "approval", "rule_ids": [RULE_B, RULE_A], "n": 2},
        context,
    )

    assert search.template_id == "strategy_voting_candidate_search"
    assert search.to_runtime_slots() == {
        **search_inputs,
        **evidence["voting_candidate_search"],
    }
    assert selection.template_id == "strategy_voting_candidate_build_from_search"
    assert selection.to_runtime_slots() == {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
    }
    assert direct.template_id == "strategy_voting_candidate_build"
    assert direct.to_runtime_slots() == {
        "strategy_type": "approval",
        "n": 2,
        **evidence["voting_candidate_build"],
    }
    assert [workflow_id for workflow_id, _inputs in calls] == [
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "voting_candidate_build",
    ]


def test_voting_specs_preserve_legacy_metadata() -> None:
    specs = {
        spec.workflow_id: spec
        for spec in CROSS_VOTING_WORKFLOW_SPECS
        if spec.workflow_id.startswith("voting_")
    }

    assert set(specs) == {
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "voting_candidate_build",
    }
    assert all(spec.fresh and spec.replayable and spec.migrated for spec in specs.values())
    assert specs["voting_candidate_search"].manual is True
    assert specs["voting_candidate_build_from_search"].manual is True
    assert specs["voting_candidate_build"].manual is False
    assert all(spec.requirements == NO_REQUIREMENTS for spec in specs.values())


def test_cross_pair_search_and_selection_normalize_only_explicit_controls() -> None:
    search = validate_cross_matrix_candidate_search_inputs(
        {"features": [" score ", "age", "income"], "max_pairs": 3},
        _resolution_context(),
    )
    selection = validate_cross_matrix_candidate_build_from_search_inputs(
        {"search_id": f" {CROSS_SEARCH_ID} ", "pair_id": CROSS_PAIR_ID},
        _resolution_context(),
    )

    assert search == {
        "features": ["score", "age", "income"],
        "max_pairs": 3,
    }
    assert selection == {
        "search_id": CROSS_SEARCH_ID,
        "pair_id": CROSS_PAIR_ID,
    }


def test_cross_rule_search_and_selection_normalize_hard_controls() -> None:
    search = validate_cross_rule_search_inputs(
        {
            "features": ["score", "age", "income"],
            "dimension": 3,
            "constraints": {
                "min_lift": 1.5,
                "min_bad_count": 20,
                "max_hit_share": 0.3,
                "min_amount_lift": None,
            },
            "max_trials": 500,
        },
        _resolution_context(),
    )
    selection = validate_cross_rule_candidate_build_from_search_inputs(
        {
            "search_id": CROSS_RULE_SEARCH_ID,
            "rule_id": CROSS_RULE_ID,
            "selection_reason": "  人工   风险复核。 ",
        },
        _resolution_context(),
    )

    assert search == {
        "features": ["score", "age", "income"],
        "dimension": 3,
        "constraints": {
            "min_lift": 1.5,
            "min_bad_count": 20,
            "max_hit_share": 0.3,
            "min_amount_lift": None,
        },
        "max_trials": 500,
    }
    assert selection == {
        "search_id": CROSS_RULE_SEARCH_ID,
        "rule_id": CROSS_RULE_ID,
        "selection_reason": "人工 风险复核。",
    }


def test_cross_searches_reject_target_columns_and_invalid_current_pointers() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as target_error:
        validate_cross_matrix_candidate_search_inputs(
            {"features": ["score", "bad"], "max_pairs": 1},
            _resolution_context(),
        )
    with pytest.raises(StrategyWorkflowValidationError) as pointer_error:
        validate_cross_rule_candidate_build_from_search_inputs(
            {"search_id": "cross-rule-search-stale", "rule_id": CROSS_RULE_ID},
            _resolution_context(),
        )

    assert str(target_error.value) == (
        "cross_matrix_candidate_search features 不能包含当前目标列「bad」。"
    )
    assert str(pointer_error.value) == (
        "cross_rule_candidate_build_from_search search_id 必须是完整的 "
        "cross-rule-search ID。"
    )


@pytest.mark.parametrize(
    ("confirmation", "inputs", "expected"),
    [
        (
            cross_matrix_candidate_search_confirmation,
            {"features": ["score", "age", "income"], "max_pairs": 3},
            "已识别为〔Cross Matrix 自动组合搜索 Workflow〕；"
            "显式候选字段：score、age、income；确定性评估预算：最多 3 个特征对；"
            "平台将在计划创建时绑定最新精确单变量候选证据，并仅使用其 "
            "risk/development 样本；每个字段的轴方法由受控 Tool 从父证据中"
            "选择最高排名的可用方法；本步骤只发布聚合搜索证据；不会构建候选"
            "或选择候选，不会入池、应用、采纳或部署",
        ),
        (
            cross_matrix_candidate_build_from_search_confirmation,
            {"search_id": CROSS_SEARCH_ID, "pair_id": CROSS_PAIR_ID},
            "已识别为〔Cross 搜索结果精确构建 Workflow〕；"
            f"搜索证据 pointer：{CROSS_SEARCH_ID}；特征对 pointer：{CROSS_PAIR_ID}；"
            "平台将在计划开始前重新认证完整搜索证据、父候选、数据与 "
            "risk/development 样本，并重新计算精确 Cross 资产；不会采用排名、"
            "最好或冠军等启发式选择；本步骤只构建一个 "
            "development/backtested/unvalidated Cross 候选；不会加入或修改 Pool，"
            "不会设置动作、应用、采纳或部署",
        ),
        (
            cross_rule_search_confirmation,
            {
                "features": ["score", "age", "income"],
                "dimension": 3,
                "constraints": {
                    "min_lift": 1.5,
                    "min_bad_count": 20,
                    "max_hit_share": 0.3,
                    "min_amount_lift": None,
                },
                "max_trials": 500,
            },
            "已识别为〔2D/3D Cross 阈值规则搜索 Workflow〕；"
            "显式候选字段：score、age、income；组合维度：3D；最多评估 500 条"
            "确定性试验；约束：min_lift=1.5，min_bad_count=20，"
            "max_hit_share=0.3，min_amount_lift=null；平台将绑定最新精确单变量"
            "证据与 risk/development 样本，从认证分箱边界和风险方向生成有预算的 "
            "2D/3D 阈值组合；本步骤只发布全部已评估规则的聚合证据与排序；"
            "不会自动选择、构建候选、入池、应用、采纳或部署",
        ),
        (
            cross_rule_candidate_build_from_search_confirmation,
            {
                "search_id": CROSS_RULE_SEARCH_ID,
                "rule_id": CROSS_RULE_ID,
                "selection_reason": "人工风险复核。",
            },
            "已识别为〔Cross 阈值规则精确候选构建 Workflow〕；"
            f"搜索证据 pointer：{CROSS_RULE_SEARCH_ID}；规则 pointer：{CROSS_RULE_ID}；"
            "平台将重新认证并完整重放搜索、数据、样本和规则条件；不会采用排名、"
            "最好或冠军等启发式选择；本步骤只构建一个 development/unvalidated "
            "候选；不会自动入池、应用、采纳或部署；用户原话选择说明：人工风险复核。",
        ),
    ],
)
def test_cross_search_confirmation_text_is_characterized(
    confirmation,
    inputs: dict[str, object],
    expected: str,
) -> None:
    assert confirmation(inputs) == expected + CONFIRMATION_SUFFIX


def test_cross_search_preparers_bind_currentness_without_copying_ranked_facts() -> None:
    source = {
        "source_artifact_id": "a" * 64,
        "expected_artifact_content_hash": "b" * 64,
        "expected_candidate_id": "candidate-" + "c" * 32,
        "expected_evidence_hash": "d" * 64,
    }
    calls: list[str] = []

    def bind(workflow_id, _inputs):
        calls.append(workflow_id)
        return (
            source
            if workflow_id
            in {"cross_matrix_candidate_search", "cross_rule_search"}
            else {}
        )

    context = StrategyWorkflowPreparationContext(
        dataset_id=None,
        bind_workflow_evidence=bind,
    )
    pair_search_inputs = {"features": ["score", "age"], "max_pairs": 1}
    rule_search_inputs = {
        "features": ["score", "age"],
        "dimension": 2,
        "constraints": {
            "min_lift": 1.2,
            "min_bad_count": 10,
            "max_hit_share": 0.4,
            "min_amount_lift": None,
        },
        "max_trials": 100,
    }

    pair_search = prepare_cross_matrix_candidate_search(
        pair_search_inputs,
        context,
    )
    pair_build = prepare_cross_matrix_candidate_build_from_search(
        {"search_id": CROSS_SEARCH_ID, "pair_id": CROSS_PAIR_ID},
        context,
    )
    rule_search = prepare_cross_rule_search(rule_search_inputs, context)
    rule_build = prepare_cross_rule_candidate_build_from_search(
        {"search_id": CROSS_RULE_SEARCH_ID, "rule_id": CROSS_RULE_ID},
        context,
    )

    assert pair_search.to_runtime_slots() == {**source, **pair_search_inputs}
    assert pair_build.to_runtime_slots() == {
        "search_id": CROSS_SEARCH_ID,
        "pair_id": CROSS_PAIR_ID,
    }
    assert rule_search.to_runtime_slots() == {**source, **rule_search_inputs}
    assert rule_build.to_runtime_slots() == {
        "search_id": CROSS_RULE_SEARCH_ID,
        "rule_id": CROSS_RULE_ID,
        "selection_reason": None,
    }
    assert calls == [
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
    ]


def test_cross_search_specs_preserve_legacy_metadata() -> None:
    workflow_ids = {
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
    }
    specs = {
        spec.workflow_id: spec
        for spec in CROSS_VOTING_WORKFLOW_SPECS
        if spec.workflow_id in workflow_ids
    }

    assert set(specs) == workflow_ids
    assert all(spec.fresh and spec.replayable and spec.manual for spec in specs.values())
    assert all(spec.requirements == NO_REQUIREMENTS for spec in specs.values())
    assert all(spec.migrated for spec in specs.values())


def test_cross_matrix_analysis_derives_axis_controls_and_manual_breakpoints() -> None:
    normalized = validate_cross_matrix_analysis_inputs(
        {
            "x_feature": "age",
            "x_method": "equal_frequency",
            "y_feature": "score",
            "y_method": "manual",
            "bin_count": 5,
            "min_bin_pct": 0.02,
            "sentinel_values": [-999],
            "manual_breakpoints": {"score": [300, 500]},
            "loan_amount_col": "income",
        },
        _resolution_context(),
    )

    assert normalized == {
        "features": ["age", "score"],
        "methods": ["equal_frequency", "manual"],
        "bin_count": 5,
        "min_bin_pct": 0.02,
        "sentinel_values": [-999],
        "manual_breakpoints": {"score": [300.0, 500.0]},
        "loan_amount_col": "income",
        "x_feature": "age",
        "x_method": "equal_frequency",
        "y_feature": "score",
        "y_method": "manual",
    }


def test_cross_matrix_analysis_rejects_forged_derived_axis_controls() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        validate_cross_matrix_analysis_inputs(
            {
                "x_feature": "age",
                "x_method": "equal_frequency",
                "y_feature": "score",
                "y_method": "equal_width",
                "features": ["score", "age"],
            },
            _resolution_context(),
        )

    assert str(captured.value) == (
        "cross_matrix_analysis features 只能是平台派生的有序轴字段。"
    )


def test_cross_matrix_cell_selection_normalizes_exact_pointers_and_reason() -> None:
    normalized = validate_cross_matrix_cell_selection_inputs(
        {
            "cross_asset_id": f" {CROSS_ASSET_ID} ",
            "cell_ids": [f" {CROSS_CELL_B} ", CROSS_CELL_A],
            "selection_reason": "  人工   风险复核。 ",
        },
        _resolution_context(),
    )

    assert normalized == {
        "cross_asset_id": CROSS_ASSET_ID,
        "cell_ids": [CROSS_CELL_B, CROSS_CELL_A],
        "selection_reason": "人工 风险复核。",
    }


@pytest.mark.parametrize(
    ("confirmation", "inputs", "expected"),
    [
        (
            cross_matrix_analysis_confirmation,
            {
                "features": ["age", "score"],
                "methods": ["equal_frequency", "manual"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [-999],
                "manual_breakpoints": {"score": [300.0, 500.0]},
                "loan_amount_col": "income",
                "x_feature": "age",
                "x_method": "equal_frequency",
                "y_feature": "score",
                "y_method": "manual",
            },
            "已识别为〔二维 Cross Matrix 候选分析 Workflow〕；"
            "X 轴：age / equal_frequency；Y 轴：score / manual；"
            "数值目标箱数 5，最小箱占比 2.00%；平台会先生成两个轴的不可变"
            "单变量证据，再逐行重放完整二维矩阵；只生成 "
            "development/backtested/unvalidated Cross evidence；不会选择格子、"
            "入池、采纳或部署；放款金额列：income；独立哨兵值：-999；"
            "手工轴切点：score=[300、500]",
        ),
        (
            cross_matrix_cell_selection_confirmation,
            {
                "cross_asset_id": CROSS_ASSET_ID,
                "cell_ids": [CROSS_CELL_B, CROSS_CELL_A],
                "selection_reason": "人工风险复核。",
            },
            "已识别为〔Cross Matrix 精确单元格选择 Workflow〕；"
            f"完整 Cross 候选资产 pointer：{CROSS_ASSET_ID}；"
            f"精确 cell pointers：{CROSS_CELL_B}、{CROSS_CELL_A}；"
            "多个 cell 按确定性 OR 语义组成一个不可变选择；平台按源矩阵顺序归一化；"
            "本步骤不排名、不推荐、不生成业务动作，也不会入池、采纳或部署；"
            "用户原话选择说明：人工风险复核。",
        ),
    ],
)
def test_cross_matrix_confirmation_text_is_characterized(
    confirmation,
    inputs: dict[str, object],
    expected: str,
) -> None:
    assert confirmation(inputs) == expected + CONFIRMATION_SUFFIX


def test_cross_matrix_analysis_preparer_binds_dataset_and_drop_nan() -> None:
    evidence = {
        "dataset_id": "dataset-1",
        "expected_content_hash": "a" * 64,
        "workspace_revision": 7,
        "analysis_generation": 3,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "bad",
        "sample_design_ref": {"artifact_id": "sample-1"},
    }
    context = StrategyWorkflowPreparationContext(
        dataset_id="dataset-1",
        drop_nan_labels=True,
        bind_workflow_evidence=lambda _workflow_id, _inputs: evidence,
    )
    inputs = {
        "features": ["age", "score"],
        "methods": ["equal_frequency", "equal_width"],
        "bin_count": 5,
        "min_bin_pct": 0.02,
        "sentinel_values": [],
        "x_feature": "age",
        "x_method": "equal_frequency",
        "y_feature": "score",
        "y_method": "equal_width",
    }

    prepared = prepare_cross_matrix_analysis(inputs, context)

    assert prepared.template_id == "strategy_cross_matrix_analysis"
    assert prepared.to_runtime_slots() == {
        **inputs,
        **evidence,
        "drop_nan_labels": True,
    }
    assert prepared.success_criteria == ()


def test_cross_matrix_cell_preparer_uses_authenticated_source_order() -> None:
    evidence = {
        "source_artifact_id": "artifact-1",
        "expected_artifact_content_hash": "a" * 64,
        "expected_asset_id": CROSS_ASSET_ID,
        "expected_asset_hash": "b" * 64,
        "expected_candidate_id": "candidate-" + "c" * 32,
        "expected_evidence_hash": "d" * 64,
        "cell_ids": [CROSS_CELL_A, CROSS_CELL_B],
    }
    context = StrategyWorkflowPreparationContext(
        dataset_id=None,
        bind_workflow_evidence=lambda _workflow_id, _inputs: evidence,
    )

    prepared = prepare_cross_matrix_cell_selection(
        {
            "cross_asset_id": CROSS_ASSET_ID,
            "cell_ids": [CROSS_CELL_B, CROSS_CELL_A],
            "selection_reason": "人工风险复核。",
        },
        context,
    )

    assert prepared.template_id == "strategy_cross_matrix_cell_selection"
    assert prepared.to_runtime_slots() == {
        **evidence,
        "selection_reason": "人工风险复核。",
    }
    assert "cross_asset_id" not in prepared.slots


def test_cross_matrix_cell_preparer_rejects_mismatched_authenticated_cells() -> None:
    context = StrategyWorkflowPreparationContext(
        dataset_id=None,
        bind_workflow_evidence=lambda _workflow_id, _inputs: {
            "source_artifact_id": "artifact-1",
            "expected_artifact_content_hash": "a" * 64,
            "expected_asset_id": CROSS_ASSET_ID,
            "expected_asset_hash": "b" * 64,
            "expected_candidate_id": "candidate-" + "c" * 32,
            "expected_evidence_hash": "d" * 64,
            "cell_ids": [CROSS_CELL_A]
        },
    )

    with pytest.raises(StrategyWorkflowValidationError) as captured:
        prepare_cross_matrix_cell_selection(
            {
                "cross_asset_id": CROSS_ASSET_ID,
                "cell_ids": [CROSS_CELL_B, CROSS_CELL_A],
            },
            context,
        )

    assert captured.value.code == "strategy_workflow_evidence_invalid"
    assert captured.value.fields == ("cell_ids",)


def test_all_nine_specs_preserve_requirements_templates_and_manual_flags() -> None:
    specs = {spec.workflow_id: spec for spec in CROSS_VOTING_WORKFLOW_SPECS}
    expected_ids = {
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "voting_candidate_build",
        "cross_matrix_candidate_search",
        "cross_matrix_candidate_build_from_search",
        "cross_rule_search",
        "cross_rule_candidate_build_from_search",
        "cross_matrix_analysis",
        "cross_matrix_cell_selection",
    }

    assert set(specs) == expected_ids
    assert specs["cross_matrix_analysis"].requirements == (
        StrategyWorkflowRequirements(
            dataset=True,
            target=True,
            complete_labels=True,
        )
    )
    assert all(
        spec.requirements == NO_REQUIREMENTS
        for workflow_id, spec in specs.items()
        if workflow_id != "cross_matrix_analysis"
    )
    assert specs["cross_matrix_analysis"].manual is True
    assert specs["cross_matrix_cell_selection"].manual is False
    assert all(
        spec.template_ids == (f"strategy_{workflow_id}",)
        for workflow_id, spec in specs.items()
    )
    assert all(spec.migrated for spec in specs.values())


def test_preparation_fails_closed_without_or_with_conflicting_evidence() -> None:
    inputs = {"search_id": CROSS_SEARCH_ID, "pair_id": CROSS_PAIR_ID}
    with pytest.raises(StrategyWorkflowValidationError) as missing:
        prepare_cross_matrix_candidate_build_from_search(
            inputs,
            StrategyWorkflowPreparationContext(dataset_id=None),
        )
    with pytest.raises(StrategyWorkflowValidationError) as conflict:
        prepare_cross_matrix_candidate_build_from_search(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda _workflow_id, _inputs: {
                    "pair_id": "forged"
                },
            ),
        )

    assert missing.value.code == "strategy_workflow_evidence_binding_required"
    assert conflict.value.code == "strategy_workflow_evidence_conflict"
    assert conflict.value.fields == ("pair_id",)


def test_preparation_copies_and_deep_freezes_platform_evidence() -> None:
    evidence = {
        "source_artifact_id": "a" * 64,
        "expected_artifact_content_hash": "b" * 64,
        "expected_candidate_id": "candidate-" + "c" * 32,
        "expected_evidence_hash": "d" * 64,
        "source": {
            "artifact_id": "artifact-1",
            "lineage": ["candidate-1"],
        }
    }
    prepared = prepare_cross_matrix_candidate_search(
        {"features": ["score", "age"], "max_pairs": 1},
        StrategyWorkflowPreparationContext(
            dataset_id=None,
            bind_workflow_evidence=lambda _workflow_id, _inputs: evidence,
        ),
    )

    evidence["source"]["artifact_id"] = "tampered"
    evidence["source"]["lineage"].append("candidate-2")

    assert prepared.to_runtime_slots()["source"] == {
        "artifact_id": "artifact-1",
        "lineage": ["candidate-1"],
    }
    with pytest.raises(TypeError):
        prepared.slots["source"]["artifact_id"] = "forged"


@pytest.mark.parametrize(
    ("prepare", "inputs"),
    [
        (
            prepare_voting_candidate_search,
            {
                "strategy_type": "approval",
                "member_count": 2,
                "n": 1,
                "objective": {"metric": "lift", "direction": "maximize"},
                "constraints": [],
                "include_rule_ids": [],
                "exclude_rule_ids": [],
                "max_combinations": 100,
            },
        ),
        (
            prepare_voting_candidate_build,
            {"strategy_type": "approval", "rule_ids": [RULE_A, RULE_B], "n": 1},
        ),
        (
            prepare_cross_matrix_candidate_search,
            {"features": ["score", "age"], "max_pairs": 1},
        ),
        (
            prepare_cross_rule_search,
            {
                "features": ["score", "age"],
                "dimension": 2,
                "constraints": {
                    "min_lift": 1.0,
                    "min_bad_count": 1,
                    "max_hit_share": 0.5,
                    "min_amount_lift": None,
                },
                "max_trials": 10,
            },
        ),
        (
            prepare_cross_matrix_analysis,
            {
                "features": ["age", "score"],
                "methods": ["equal_frequency", "equal_width"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
                "x_feature": "age",
                "x_method": "equal_frequency",
                "y_feature": "score",
                "y_method": "equal_width",
            },
        ),
    ],
)
def test_preparers_reject_empty_evidence_when_runtime_slots_require_it(
    prepare,
    inputs: dict[str, object],
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        prepare(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id="dataset-1",
                bind_workflow_evidence=lambda _workflow_id, _inputs: {},
            ),
        )

    assert captured.value.code == "strategy_workflow_evidence_invalid"


def test_cell_preparer_rejects_evidence_for_a_different_asset() -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        prepare_cross_matrix_cell_selection(
            {"cross_asset_id": CROSS_ASSET_ID, "cell_ids": [CROSS_CELL_A]},
            StrategyWorkflowPreparationContext(
                dataset_id=None,
                bind_workflow_evidence=lambda _workflow_id, _inputs: {
                    "source_artifact_id": "artifact-1",
                    "expected_artifact_content_hash": "a" * 64,
                    "expected_asset_id": "candidate-asset-" + "f" * 32,
                    "expected_asset_hash": "b" * 64,
                    "expected_candidate_id": "candidate-" + "c" * 32,
                    "expected_evidence_hash": "d" * 64,
                    "cell_ids": [CROSS_CELL_A],
                },
            ),
        )

    assert captured.value.code == "strategy_workflow_evidence_invalid"
    assert captured.value.fields == ("expected_asset_id",)


@pytest.mark.parametrize(
    ("prepare", "inputs", "forged_field"),
    [
        (
            prepare_voting_candidate_build,
            {"strategy_type": "approval", "rule_ids": [RULE_A, RULE_B], "n": 1},
            "rule_ids",
        ),
        (
            prepare_cross_matrix_analysis,
            {
                "features": ["age", "score"],
                "methods": ["equal_frequency", "equal_width"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
                "x_feature": "age",
                "x_method": "equal_frequency",
                "y_feature": "score",
                "y_method": "equal_width",
            },
            "drop_nan_labels",
        ),
    ],
)
def test_platform_evidence_cannot_take_ownership_of_canonical_controls(
    prepare,
    inputs: dict[str, object],
    forged_field: str,
) -> None:
    with pytest.raises(StrategyWorkflowValidationError) as captured:
        prepare(
            inputs,
            StrategyWorkflowPreparationContext(
                dataset_id="dataset-1",
                bind_workflow_evidence=lambda _workflow_id, _inputs: {
                    forged_field: "forged"
                },
            ),
        )

    assert captured.value.code == "strategy_workflow_evidence_conflict"
    assert captured.value.fields == (forged_field,)
