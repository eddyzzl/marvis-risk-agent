from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.packs.strategy.candidate_design import CANDIDATE_POLICY_VERSION
from marvis.plugins.manifest import ToolRef


STRATEGY_ANALYSIS = WorkflowTemplate(
    id="strategy_analysis",
    title="快速策略分析与回测",
    goal_patterns=(
        "快速策略分析",
        "快速策略回测",
        "quick strategy analysis",
        "quick strategy backtest",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec("score_col", True, "task_context", "Score column"),
        SlotSpec("strategy_type", True, "user", "Strategy type"),
        SlotSpec("rules", True, "user", "Ordered strategy rules"),
        SlotSpec("default_decision", True, "user", "Fallback decision"),
    ),
    steps=(
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "rules": "{slot:rules}",
                "score_col": "{slot:score_col}",
                "default_decision": "{slot:default_decision}",
                "description": "Workflow generated strategy candidate",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck(
                    "range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "expected_profit", "allow_null": True}
                ),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成策略权衡视图",
            tool_ref=ToolRef("strategy", "tradeoff_view"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
            },
            depends_on_titles=("回测策略",),
            post_checks=(PostCheck("nonempty", {"field": "points"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_PROFIT_ANALYSIS = WorkflowTemplate(
    id="strategy_profit_analysis",
    title="分群利润分析",
    goal_patterns=("分群利润分析", "利润测算", "profit analysis", "profit calculation"),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec("ead_col", True, "user", "Exposure at default column"),
        SlotSpec("pd_col", True, "user", "Probability of default column"),
        SlotSpec(
            "profit_params",
            True,
            "user",
            "Pricing, funding, LGD, cost, and term assumptions",
        ),
        SlotSpec("segment_col", False, "user", "Optional segment column"),
    ),
    steps=(
        StepTemplate(
            title="测算分群利润",
            tool_ref=ToolRef("strategy", "profit_calc"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "segment_col": "{slot:segment_col}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "params": "{slot:profit_params}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "results"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_ROLL_RATE_ANALYSIS = WorkflowTemplate(
    id="strategy_roll_rate_analysis",
    title="Roll-rate 迁徙分析",
    goal_patterns=("roll rate 分析", "迁徙率分析", "roll-rate analysis"),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec("id_col", True, "user", "Entity identifier column"),
        SlotSpec("time_col", True, "user", "Observation time column"),
        SlotSpec("status_col", True, "user", "Status bucket column"),
        SlotSpec("states", True, "user", "Ordered status bucket values"),
        SlotSpec(
            "balance_col", False, "user", "Optional from-observation balance weight"
        ),
        SlotSpec(
            "observation_semantics",
            False,
            "user",
            "Must be adjacent_observation; snapshot panels use bucket_migration",
        ),
    ),
    steps=(
        StepTemplate(
            title="计算相邻观测迁徙矩阵",
            tool_ref=ToolRef("strategy", "roll_rate_matrix"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "id_col": "{slot:id_col}",
                "time_col": "{slot:time_col}",
                "status_col": "{slot:status_col}",
                "states": "{slot:states}",
                "balance_col": "{slot:balance_col}",
                "observation_semantics": "{slot:observation_semantics}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "matrix"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS = WorkflowTemplate(
    id="strategy_univariate_candidate_analysis",
    title="单变量候选分析",
    goal_patterns=(
        "单变量候选分析",
        "单变量效果分析",
        "比较分箱方法",
        "univariate candidate analysis",
    ),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec(
            "expected_content_hash",
            True,
            "task_context",
            "Confirmed immutable dataset hash",
        ),
        SlotSpec(
            "workspace_revision",
            False,
            "task_context",
            "Confirmed data-workspace revision; zero is valid",
        ),
        SlotSpec(
            "analysis_generation",
            False,
            "task_context",
            "Confirmed active dataset generation; zero is valid",
        ),
        SlotSpec(
            "semantic_mapping_hash",
            True,
            "task_context",
            "Confirmed semantic mapping hash",
        ),
        SlotSpec(
            "target_col", True, "task_context", "Server-bound binary target column"
        ),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec(
            "features", False, "user", "Explicit fields or [] for semantic candidates"
        ),
        SlotSpec(
            "methods", False, "user", "Ordered methods or [] for type-aware defaults"
        ),
        SlotSpec("bin_count", True, "user", "Requested bins per numeric method"),
        SlotSpec("min_bin_pct", True, "user", "Minimum desired bin population share"),
        SlotSpec("loan_amount_col", False, "user", "Optional disbursed amount column"),
        SlotSpec("overdue_amount_col", False, "user", "Optional overdue amount column"),
        SlotSpec(
            "sentinel_values",
            False,
            "user",
            "Explicit special values kept separate; [] is valid",
        ),
    ),
    steps=(
        StepTemplate(
            title="分析单变量候选",
            tool_ref=ToolRef("strategy", "analyze_univariate_candidates"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "features": "{slot:features}",
                "methods": "{slot:methods}",
                "bin_count": "{slot:bin_count}",
                "min_bin_pct": "{slot:min_bin_pct}",
                "loan_amount_col": "{slot:loan_amount_col}",
                "overdue_amount_col": "{slot:overdue_amount_col}",
                "sentinel_values": "{slot:sentinel_values}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "candidate_id"}),
                PostCheck("nonempty", {"field": "evidence_hash"}),
                PostCheck("nonempty", {"field": "artifacts"}),
                PostCheck("range", {"field": "rankings.0.iv", "min": 0.0}),
                PostCheck(
                    "range",
                    {"field": "rankings.0.ks", "min": 0.0, "max": 1.0},
                ),
                PostCheck(
                    "range",
                    {"field": "rankings.0.auc", "min": 0.0, "max": 1.0},
                ),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_CROSS_MATRIX_ANALYSIS = WorkflowTemplate(
    id="strategy_cross_matrix_analysis",
    title="二维 Cross Matrix 候选分析",
    goal_patterns=(
        "二维交叉矩阵",
        "二维 Cross Matrix",
        "2D cross matrix",
    ),
    slots=(
        *STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS.slots,
        SlotSpec("x_feature", True, "user", "Explicit X-axis feature"),
        SlotSpec("x_method", True, "user", "Explicit X-axis binning method"),
        SlotSpec("y_feature", True, "user", "Explicit Y-axis feature"),
        SlotSpec("y_method", True, "user", "Explicit Y-axis binning method"),
    ),
    steps=(
        STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS.steps[0],
        StepTemplate(
            title="构建二维 Cross Matrix 候选",
            tool_ref=ToolRef("strategy", "build_cross_matrix_candidate"),
            inputs_template={
                "source_artifact_id": (
                    "$ref:分析单变量候选.output.artifacts.0.artifact_id"
                ),
                "expected_artifact_content_hash": (
                    "$ref:分析单变量候选.output.artifacts.0.content_hash"
                ),
                "expected_candidate_id": "$ref:分析单变量候选.output.candidate_id",
                "expected_evidence_hash": "$ref:分析单变量候选.output.evidence_hash",
                "x_feature": "{slot:x_feature}",
                "x_method": "{slot:x_method}",
                "y_feature": "{slot:y_feature}",
                "y_method": "{slot:y_method}",
            },
            depends_on_titles=("分析单变量候选",),
            post_checks=(
                PostCheck("nonempty", {"field": "asset_id"}),
                PostCheck("nonempty", {"field": "asset_hash"}),
                PostCheck("nonempty", {"field": "cell_count"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_CROSS_MATRIX_CELL_SELECTION = WorkflowTemplate(
    id="strategy_cross_matrix_cell_selection",
    title="Cross Matrix 精确单元格选择",
    goal_patterns=(
        "选择 Cross Matrix 精确单元格",
        "物化 Cross Matrix 指定格子",
        "select exact cross matrix cells",
        "materialize cross matrix cell selection",
    ),
    slots=(
        SlotSpec(
            "source_artifact_id",
            True,
            "task_context",
            "Verified task-owned Cross Matrix artifact id",
        ),
        SlotSpec(
            "expected_artifact_content_hash",
            True,
            "task_context",
            "Verified Cross Matrix artifact content hash",
        ),
        SlotSpec(
            "expected_asset_id",
            True,
            "task_context",
            "Verified Cross Matrix asset id",
        ),
        SlotSpec(
            "expected_asset_hash",
            True,
            "task_context",
            "Verified Cross Matrix asset hash",
        ),
        SlotSpec(
            "expected_candidate_id",
            True,
            "task_context",
            "Verified source candidate id",
        ),
        SlotSpec(
            "expected_evidence_hash",
            True,
            "task_context",
            "Verified source candidate evidence hash",
        ),
        SlotSpec(
            "cell_ids",
            True,
            "user",
            "Explicit Cross Matrix cell ids; normalized in source order",
        ),
        SlotSpec(
            "selection_reason",
            False,
            "user",
            "Optional user-owned exact-cell selection rationale",
        ),
    ),
    steps=(
        StepTemplate(
            title="物化 Cross Matrix 精确单元格选择",
            tool_ref=ToolRef("strategy", "materialize_cross_matrix_cell_selection"),
            inputs_template={
                "source_artifact_id": "{slot:source_artifact_id}",
                "expected_artifact_content_hash": (
                    "{slot:expected_artifact_content_hash}"
                ),
                "expected_asset_id": "{slot:expected_asset_id}",
                "expected_asset_hash": "{slot:expected_asset_hash}",
                "expected_candidate_id": "{slot:expected_candidate_id}",
                "expected_evidence_hash": "{slot:expected_evidence_hash}",
                "cell_ids": "{slot:cell_ids}",
                "selection_reason": "{slot:selection_reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "selection_id"}),
                PostCheck("nonempty", {"field": "selection_hash"}),
                PostCheck("nonempty", {"field": "group_id"}),
                PostCheck("nonempty", {"field": "cell_ids"}),
                PostCheck("nonempty", {"field": "source_asset_id"}),
                PostCheck("nonempty", {"field": "source_asset_hash"}),
                PostCheck("nonempty", {"field": "source_candidate_id"}),
                PostCheck("nonempty", {"field": "source_evidence_hash"}),
                PostCheck("nonempty", {"field": "fragment_id"}),
                PostCheck("nonempty", {"field": "rule_id"}),
                PostCheck("nonempty", {"field": "effect_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_AUTOMATIC_TREE_CANDIDATE_BUILD = WorkflowTemplate(
    id="strategy_automatic_tree_candidate_build",
    title="自动决策树候选构建",
    goal_patterns=(
        "自动决策树候选",
        "自动构建决策树",
        "构建完整决策树",
        "automatic tree candidate",
        "build automatic strategy tree",
    ),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec(
            "expected_content_hash",
            True,
            "task_context",
            "Confirmed immutable dataset hash",
        ),
        SlotSpec(
            "workspace_revision",
            False,
            "task_context",
            "Confirmed data-workspace revision; zero is valid",
        ),
        SlotSpec(
            "analysis_generation",
            False,
            "task_context",
            "Confirmed active dataset generation; zero is valid",
        ),
        SlotSpec(
            "semantic_mapping_hash",
            True,
            "task_context",
            "Confirmed semantic mapping hash",
        ),
        SlotSpec(
            "target_col", True, "task_context", "Server-bound binary target column"
        ),
        SlotSpec("features", True, "user", "Explicit ordered tree feature fields"),
        SlotSpec(
            "drop_nan_labels",
            False,
            "user",
            "Confirmed target null-row exclusion",
        ),
        SlotSpec("sample_weight_col", False, "user", "Optional sample weight column"),
        SlotSpec(
            "directions",
            False,
            "user",
            "Optional per-feature risk directions",
        ),
        SlotSpec("max_depth", False, "user", "Optional maximum tree depth"),
        SlotSpec("min_leaf_count", False, "user", "Optional minimum rows per leaf"),
        SlotSpec(
            "min_weight_fraction_leaf",
            False,
            "user",
            "Optional minimum weighted share per leaf",
        ),
        SlotSpec("seed", False, "user", "Optional deterministic tree seed"),
        SlotSpec("loan_amount_col", False, "user", "Optional disbursed amount column"),
        SlotSpec("overdue_amount_col", False, "user", "Optional overdue amount column"),
    ),
    steps=(
        StepTemplate(
            title="构建自动决策树候选",
            tool_ref=ToolRef("strategy", "build_automatic_tree_candidate"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "semantic_mapping_hash": "{slot:semantic_mapping_hash}",
                "target_col": "{slot:target_col}",
                "features": "{slot:features}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "sample_weight_col": "{slot:sample_weight_col}",
                "directions": "{slot:directions}",
                "max_depth": "{slot:max_depth}",
                "min_leaf_count": "{slot:min_leaf_count}",
                "min_weight_fraction_leaf": "{slot:min_weight_fraction_leaf}",
                "seed": "{slot:seed}",
                "loan_amount_col": "{slot:loan_amount_col}",
                "overdue_amount_col": "{slot:overdue_amount_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "summary.asset_id"}),
                PostCheck("nonempty", {"field": "summary.asset_hash"}),
                PostCheck("nonempty", {"field": "summary.tree_id"}),
                PostCheck("nonempty", {"field": "summary.tree_result_hash"}),
                PostCheck("nonempty", {"field": "leaf_index"}),
                PostCheck("nonempty", {"field": "artifacts"}),
                PostCheck(
                    "schema",
                    {
                        "schema": {
                            "type": "object",
                            "properties": {"report_info_gaps": {"type": "array"}},
                            "required": ["report_info_gaps"],
                        }
                    },
                ),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_VOTING_CANDIDATE_BUILD = WorkflowTemplate(
    id="strategy_voting_candidate_build",
    title="Voting n-of-k 策略候选构建",
    goal_patterns=(
        "构建 Voting 策略候选",
        "构建投票组合候选",
        "生成 n-of-k 策略候选",
        "build voting strategy candidate",
        "build n-of-k candidate",
    ),
    slots=(
        SlotSpec("strategy_type", True, "user", "Explicit Strategy Pool type"),
        SlotSpec(
            "expected_pool_revision",
            True,
            "task_context",
            "Current Strategy Pool CAS revision",
        ),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Strategy Pool CAS snapshot hash",
        ),
        SlotSpec(
            "selected_entry_ids",
            True,
            "task_context",
            "Platform-resolved duplicate-free current Pool entry ids",
        ),
        SlotSpec("n", True, "user", "Required hits in the n-of-k condition"),
    ),
    steps=(
        StepTemplate(
            title="构建 Voting n-of-k 策略候选",
            tool_ref=ToolRef("strategy", "build_voting_candidate"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": (
                    "{slot:expected_pool_snapshot_hash}"
                ),
                "selected_entry_ids": "{slot:selected_entry_ids}",
                "n": "{slot:n}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "asset_id"}),
                PostCheck("nonempty", {"field": "asset_hash"}),
                PostCheck("nonempty", {"field": "candidate_id"}),
                PostCheck("nonempty", {"field": "evidence_hash"}),
                PostCheck("nonempty", {"field": "fragment_id"}),
                PostCheck("nonempty", {"field": "effect_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_AUTOMATIC_TREE_LEAF_MATERIALIZATION = WorkflowTemplate(
    id="strategy_automatic_tree_leaf_materialization",
    title="自动树精确叶节点物化",
    goal_patterns=(
        "物化自动树指定叶节点",
        "选择自动树精确叶节点",
        "materialize exact automatic tree leaf",
        "materialize an automatic tree leaf",
    ),
    slots=(
        SlotSpec(
            "source_artifact_id",
            True,
            "task_context",
            "Verified task-owned automatic-tree artifact id",
        ),
        SlotSpec(
            "expected_artifact_content_hash",
            True,
            "task_context",
            "Verified automatic-tree artifact content hash",
        ),
        SlotSpec(
            "expected_asset_id",
            True,
            "task_context",
            "Verified automatic-tree asset id",
        ),
        SlotSpec(
            "expected_asset_hash",
            True,
            "task_context",
            "Verified automatic-tree asset hash",
        ),
        SlotSpec(
            "expected_tree_result_hash",
            True,
            "task_context",
            "Verified deterministic tree result hash",
        ),
        SlotSpec("leaf_id", True, "user", "Explicit automatic-tree leaf id"),
        SlotSpec(
            "selection_reason",
            False,
            "user",
            "Optional user-owned leaf selection rationale",
        ),
    ),
    steps=(
        StepTemplate(
            title="物化自动树精确叶节点",
            tool_ref=ToolRef("strategy", "materialize_automatic_tree_leaf_fragment"),
            inputs_template={
                "source_artifact_id": "{slot:source_artifact_id}",
                "expected_artifact_content_hash": (
                    "{slot:expected_artifact_content_hash}"
                ),
                "expected_asset_id": "{slot:expected_asset_id}",
                "expected_asset_hash": "{slot:expected_asset_hash}",
                "expected_tree_result_hash": "{slot:expected_tree_result_hash}",
                "leaf_id": "{slot:leaf_id}",
                "selection_reason": "{slot:selection_reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "selection_id"}),
                PostCheck("nonempty", {"field": "selection_hash"}),
                PostCheck("nonempty", {"field": "tree_asset_id"}),
                PostCheck("nonempty", {"field": "tree_asset_hash"}),
                PostCheck("nonempty", {"field": "tree_result_hash"}),
                PostCheck("nonempty", {"field": "leaf_id"}),
                PostCheck("nonempty", {"field": "fragment_id"}),
                PostCheck("nonempty", {"field": "fragment_hash"}),
                PostCheck("nonempty", {"field": "rule_id"}),
                PostCheck("nonempty", {"field": "effect_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT = WorkflowTemplate(
    id="strategy_univariate_candidate_refinement",
    title="单变量候选选择与合并",
    goal_patterns=(
        "单变量候选选择",
        "合并候选分箱",
        "筛选策略规则",
        "univariate candidate refinement",
    ),
    slots=(
        *STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS.slots,
        SlotSpec("feature", True, "user", "Feature to refine from candidate evidence"),
        SlotSpec("method", True, "user", "Binning method to refine"),
        SlotSpec(
            "merge_groups",
            False,
            "user",
            "Explicit groups of source bin ids; [] keeps source bins unchanged",
        ),
        SlotSpec(
            "selection",
            True,
            "user",
            "Explicit source bin ids or an observed-risk threshold",
        ),
        SlotSpec(
            "selection_reason",
            False,
            "user",
            "Optional user-owned rationale, never a calculated result",
        ),
    ),
    steps=(
        STRATEGY_UNIVARIATE_CANDIDATE_ANALYSIS.steps[0],
        StepTemplate(
            title="选择并合并单变量候选",
            tool_ref=ToolRef("strategy", "refine_univariate_candidate"),
            inputs_template={
                "source_artifact_id": (
                    "$ref:分析单变量候选.output.artifacts.0.artifact_id"
                ),
                "expected_artifact_content_hash": (
                    "$ref:分析单变量候选.output.artifacts.0.content_hash"
                ),
                "expected_candidate_id": ("$ref:分析单变量候选.output.candidate_id"),
                "expected_evidence_hash": ("$ref:分析单变量候选.output.evidence_hash"),
                "feature": "{slot:feature}",
                "method": "{slot:method}",
                "merge_groups": "{slot:merge_groups}",
                "selection": "{slot:selection}",
                "selection_reason": "{slot:selection_reason}",
            },
            depends_on_titles=("分析单变量候选",),
            post_checks=(
                PostCheck("nonempty", {"field": "asset_id"}),
                PostCheck("nonempty", {"field": "asset_hash"}),
                PostCheck("nonempty", {"field": "effect_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_UNIVARIATE_CANDIDATE_REFINEMENT_EXISTING = WorkflowTemplate(
    id="strategy_univariate_candidate_refinement_existing",
    title="已有单变量证据候选选择与合并",
    goal_patterns=(
        "选择已有候选箱",
        "合并已有候选箱",
        "refine existing univariate candidate",
    ),
    slots=(
        SlotSpec(
            "source_artifact_id", True, "task_context", "Bound source JSON artifact"
        ),
        SlotSpec(
            "expected_artifact_content_hash",
            True,
            "task_context",
            "Bound source artifact content hash",
        ),
        SlotSpec(
            "expected_candidate_id",
            True,
            "user",
            "Candidate id explicitly copied from the user's request",
        ),
        SlotSpec(
            "expected_evidence_hash",
            True,
            "task_context",
            "Bound parent evidence hash",
        ),
        SlotSpec("feature", True, "user", "Feature to refine from candidate evidence"),
        SlotSpec("method", True, "user", "Binning method to refine"),
        SlotSpec("merge_groups", False, "user", "Explicit source bin id merge groups"),
        SlotSpec("selection", True, "user", "Explicit bins or risk threshold"),
        SlotSpec("selection_reason", False, "user", "Optional user-owned rationale"),
    ),
    steps=(
        StepTemplate(
            title="选择并合并已有单变量候选",
            tool_ref=ToolRef("strategy", "refine_univariate_candidate"),
            inputs_template={
                "source_artifact_id": "{slot:source_artifact_id}",
                "expected_artifact_content_hash": (
                    "{slot:expected_artifact_content_hash}"
                ),
                "expected_candidate_id": "{slot:expected_candidate_id}",
                "expected_evidence_hash": "{slot:expected_evidence_hash}",
                "feature": "{slot:feature}",
                "method": "{slot:method}",
                "merge_groups": "{slot:merge_groups}",
                "selection": "{slot:selection}",
                "selection_reason": "{slot:selection_reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "asset_id"}),
                PostCheck("nonempty", {"field": "asset_hash"}),
                PostCheck("nonempty", {"field": "effect_id"}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_POOL_ADD_CANDIDATE = WorkflowTemplate(
    id="strategy_pool_add_candidate",
    title="候选资产加入 Strategy Pool",
    goal_patterns=("候选入池", "添加候选规则", "add candidate to strategy pool"),
    slots=(
        SlotSpec("source_artifact_id", True, "task_context", "Bound candidate asset artifact"),
        SlotSpec(
            "expected_artifact_content_hash",
            True,
            "task_context",
            "Bound candidate artifact content hash",
        ),
        SlotSpec("expected_asset_id", True, "task_context", "Verified candidate asset id"),
        SlotSpec("expected_asset_hash", True, "task_context", "Verified candidate asset hash"),
        SlotSpec("strategy_type", True, "user", "Typed Strategy Pool kind"),
        SlotSpec("default_action", True, "user", "Explicit typed default action"),
        SlotSpec("action", True, "user", "Explicit typed action for the candidate rule"),
        SlotSpec(
            "placement_mode",
            True,
            "user",
            "Voting placement semantics or platform-bound append for ordinary candidates",
        ),
        # Planner's required-slot check treats the valid absent-pool revision
        # ``0`` as falsy.  Keep this Planner-optional while the template and
        # Tool schema still require and carry the platform-bound CAS value.
        SlotSpec("expected_pool_revision", False, "task_context", "Current Pool CAS revision"),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Pool CAS snapshot hash",
        ),
        SlotSpec("reason", False, "user", "Optional user-owned edit rationale"),
    ),
    steps=(
        StepTemplate(
            title="候选资产加入策略池",
            tool_ref=ToolRef("strategy", "add_candidate_to_pool"),
            inputs_template={
                "source_artifact_id": "{slot:source_artifact_id}",
                "expected_artifact_content_hash": "{slot:expected_artifact_content_hash}",
                "expected_asset_id": "{slot:expected_asset_id}",
                "expected_asset_hash": "{slot:expected_asset_hash}",
                "strategy_type": "{slot:strategy_type}",
                "default_action": "{slot:default_action}",
                "action": "{slot:action}",
                "placement_mode": "{slot:placement_mode}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
                "reason": "{slot:reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "pool_id"}),
                PostCheck("nonempty", {"field": "snapshot_hash"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_POOL_REMOVE_ENTRY = WorkflowTemplate(
    id="strategy_pool_remove_entry",
    title="从 Strategy Pool 删除条目",
    goal_patterns=("策略池删除规则", "移除池条目", "remove strategy pool entry"),
    slots=(
        SlotSpec("strategy_type", True, "user", "Typed Strategy Pool kind"),
        SlotSpec("rule_id", True, "task_context", "Verified rule id from the current Pool"),
        SlotSpec("expected_pool_revision", True, "task_context", "Current Pool CAS revision"),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Pool CAS snapshot hash",
        ),
        SlotSpec("reason", False, "user", "Optional user-owned edit rationale"),
    ),
    steps=(
        StepTemplate(
            title="删除策略池条目",
            tool_ref=ToolRef("strategy", "remove_pool_entry"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "rule_id": "{slot:rule_id}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
                "reason": "{slot:reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "pool_id"}),
                PostCheck("nonempty", {"field": "snapshot_hash"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_POOL_SET_ACTION = WorkflowTemplate(
    id="strategy_pool_set_action",
    title="修改 Strategy Pool 条目动作",
    goal_patterns=("修改池规则动作", "策略池动作", "set strategy pool entry action"),
    slots=(
        SlotSpec("strategy_type", True, "user", "Typed Strategy Pool kind"),
        SlotSpec("rule_id", True, "task_context", "Verified rule id from the current Pool"),
        SlotSpec("action", True, "user", "Explicit typed replacement action"),
        SlotSpec("expected_pool_revision", True, "task_context", "Current Pool CAS revision"),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Pool CAS snapshot hash",
        ),
        SlotSpec("reason", False, "user", "Optional user-owned edit rationale"),
    ),
    steps=(
        StepTemplate(
            title="修改策略池条目动作",
            tool_ref=ToolRef("strategy", "set_pool_entry_action"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "rule_id": "{slot:rule_id}",
                "action": "{slot:action}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
                "reason": "{slot:reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "pool_id"}),
                PostCheck("nonempty", {"field": "snapshot_hash"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_POOL_REORDER = WorkflowTemplate(
    id="strategy_pool_reorder",
    title="完整重排 Strategy Pool",
    goal_patterns=("策略池完整排序", "完整重排规则", "reorder strategy pool"),
    slots=(
        SlotSpec("strategy_type", True, "user", "Typed Strategy Pool kind"),
        SlotSpec("ordered_rule_ids", True, "task_context", "Verified complete rule-id order"),
        SlotSpec("expected_pool_revision", True, "task_context", "Current Pool CAS revision"),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Pool CAS snapshot hash",
        ),
        SlotSpec("reason", False, "user", "Optional user-owned edit rationale"),
    ),
    steps=(
        StepTemplate(
            title="完整重排策略池",
            tool_ref=ToolRef("strategy", "reorder_strategy_pool"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "ordered_rule_ids": "{slot:ordered_rule_ids}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
                "reason": "{slot:reason}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "pool_id"}),
                PostCheck("nonempty", {"field": "snapshot_hash"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_POOL_COMPILE = WorkflowTemplate(
    id="strategy_pool_compile",
    title="编译预览 Strategy Pool",
    goal_patterns=("预览策略池", "编译策略池草案", "compile strategy pool"),
    slots=(
        SlotSpec("strategy_type", True, "user", "Typed Strategy Pool kind"),
        SlotSpec("expected_pool_revision", True, "task_context", "Current Pool revision"),
        SlotSpec(
            "expected_pool_snapshot_hash",
            True,
            "task_context",
            "Current Pool snapshot hash",
        ),
    ),
    steps=(
        StepTemplate(
            title="编译策略池草案",
            tool_ref=ToolRef("strategy", "compile_strategy_pool"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "expected_pool_revision": "{slot:expected_pool_revision}",
                "expected_pool_snapshot_hash": "{slot:expected_pool_snapshot_hash}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "design_hash"}),
                PostCheck("nonempty", {"field": "strategy_spec"}),
            ),
            needs_confirmation=False,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


_LIMIT_PRICING_INPUTS = {
    "dataset_id": "{slot:dataset_id}",
    "score_col": "{slot:score_col}",
    "target_col": "{slot:target_col}",
    "pd_col": "{slot:pd_col}",
    "band_edges": "{slot:band_edges}",
    "n_bands": "{slot:n_bands}",
    "limit_grid": "{slot:limit_grid}",
    "rate_grid": "{slot:rate_grid}",
    "lgd": "{slot:lgd}",
    "funding_rate": "{slot:funding_rate}",
    "term_months": "{slot:term_months}",
    "cost_per_loan": "{slot:cost_per_loan}",
    "el_ead_max": "{slot:el_ead_max}",
    "strategy_id": "{slot:strategy_id}",
    "drop_nan_labels": "{slot:drop_nan_labels}",
}


STRATEGY_LIMIT_PRICING_ANALYSIS = WorkflowTemplate(
    id="strategy_limit_pricing_analysis",
    title="额度与定价矩阵分析",
    goal_patterns=("额度定价矩阵", "额度与定价分析", "limit pricing matrix"),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec("score_col", True, "user", "Score column"),
        SlotSpec(
            "pd_col",
            False,
            "user",
            "Calibrated PD column; mutually exclusive with target_col",
        ),
        SlotSpec(
            "target_col",
            False,
            "user",
            "Binary target used as PD proxy; mutually exclusive with pd_col",
        ),
        SlotSpec("limit_grid", True, "user", "Candidate limits"),
        SlotSpec("rate_grid", True, "user", "Candidate annual rates"),
        SlotSpec("funding_rate", True, "user", "Annual funding rate"),
        SlotSpec("term_months", True, "user", "Term in months"),
        SlotSpec("cost_per_loan", True, "user", "Operating cost per loan"),
        SlotSpec("band_edges", False, "user", "Optional explicit score band edges"),
        SlotSpec(
            "n_bands", False, "user", "Number of score bands when edges are omitted"
        ),
        SlotSpec("lgd", False, "user", "Loss given default; default 0.6"),
        SlotSpec("el_ead_max", False, "user", "Maximum feasible EL/EAD ratio"),
        SlotSpec(
            "strategy_id",
            False,
            "task_context",
            "Optional task-owned limit/pricing strategy id",
        ),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
    ),
    steps=(
        StepTemplate(
            title="计算额度定价矩阵",
            tool_ref=ToolRef("strategy", "limit_pricing_matrix"),
            inputs_template={**_LIMIT_PRICING_INPUTS, "confirm": False},
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "matrix"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="导出额度定价矩阵",
            tool_ref=ToolRef("strategy", "limit_pricing_matrix"),
            inputs_template={
                **_LIMIT_PRICING_INPUTS,
                "expected_source_hash": (
                    "$ref:计算额度定价矩阵.output.source_dataset_content_hash"
                ),
                "confirm": True,
            },
            depends_on_titles=("计算额度定价矩阵",),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


DETERMINISTIC_STRATEGY_CANDIDATE_DEVELOPMENT = WorkflowTemplate(
    id="deterministic_strategy_candidate_development",
    title="非审批策略确定性候选开发",
    goal_patterns=(
        "额度候选策略开发",
        "定价候选策略开发",
        "分群候选策略开发",
        "deterministic strategy candidate development",
    ),
    slots=(
        SlotSpec(
            "dataset_id", True, "task_context", "Registered task-owned dataset id"
        ),
        SlotSpec(
            "target_col", True, "task_context", "Server-bound binary target column"
        ),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec(
            "strategy_type",
            True,
            "task_context",
            "One of limit, pricing, or segmentation",
        ),
        SlotSpec(
            "candidate_design",
            True,
            "user",
            "Validated candidate search space without rules, metrics, or recommendations",
        ),
        SlotSpec(
            "economics_inputs",
            False,
            "user",
            "Complete limit/pricing economics; omitted only for segmentation",
        ),
        SlotSpec(
            "baseline_strategy_id",
            False,
            "user",
            "Optional task-owned baseline of the same strategy type",
        ),
    ),
    steps=(
        StepTemplate(
            title="确定性设计策略候选",
            tool_ref=ToolRef("strategy", "design_strategy_candidate"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "strategy_type": "{slot:strategy_type}",
                "candidate_design": "{slot:candidate_design}",
                "economics_inputs": "{slot:economics_inputs}",
                # The LLM/user cannot choose the algorithm version.  The platform
                # pins it here and the deterministic kernel rejects all others.
                "candidate_policy_version": CANDIDATE_POLICY_VERSION,
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "strategy_spec"}),
                PostCheck("nonempty", {"field": "strategy_effect_hash"}),
                PostCheck("nonempty", {"field": "design_evidence"}),
            ),
        ),
        StepTemplate(
            title="构造确定性候选策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "$ref:确定性设计策略候选.output.strategy_spec",
                "description": "Platform-designed deterministic strategy candidate",
            },
            depends_on_titles=("确定性设计策略候选",),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测确定性候选策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造确定性候选策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                # Reuse the normalized bundle emitted by design so candidate
                # selection and backtest cannot silently diverge in economics.
                "economics_inputs": ("$ref:确定性设计策略候选.output.economics_inputs"),
            },
            depends_on_titles=("确定性设计策略候选", "构造确定性候选策略"),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {"field": "expected_profit", "allow_null": True},
                ),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成确定性候选策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造确定性候选策略.output.strategy_id",
            },
            depends_on_titles=("构造确定性候选策略", "回测确定性候选策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
        StepTemplate(
            title="采纳确定性候选策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造确定性候选策略.output.strategy_id",
                "backtest_id": "$ref:回测确定性候选策略.output.backtest_id",
                "adoption_reason": "",
            },
            depends_on_titles=(
                "构造确定性候选策略",
                "回测确定性候选策略",
                "生成确定性候选策略文档",
            ),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            # This is the only mandatory human governance decision in the flow.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="生成本地采纳策略最终文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:采纳确定性候选策略.output.strategy_id",
            },
            depends_on_titles=("采纳确定性候选策略",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


TYPED_STRATEGY_BUILD = WorkflowTemplate(
    id="typed_strategy_build",
    title="类型化策略开发",
    goal_patterns=("类型化策略开发", "typed strategy build"),
    slots=(
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
    ),
    steps=(
        StepTemplate(
            title="构造类型化策略草案",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled typed strategy",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="生成类型化策略草案文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造类型化策略草案.output.strategy_id",
            },
            depends_on_titles=("构造类型化策略草案",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


TYPED_STRATEGY_EVALUATION = WorkflowTemplate(
    # Natural-language requests compile into a canonical StrategySpec before this
    # workflow is instantiated. The shared build/backtest/doc chain therefore
    # evaluates all five strategy types without copying approval-only cutoff or
    # tradeoff steps into limit, pricing or segmentation flows.
    id="typed_strategy_evaluation",
    title="类型化策略评估",
    goal_patterns=(
        "类型化策略评估",
        "typed strategy evaluation",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
        SlotSpec(
            "baseline_strategy_id", False, "user", "Optional baseline strategy id"
        ),
        SlotSpec(
            "economics_inputs", False, "user", "Typed limit/pricing economics inputs"
        ),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="构造类型化策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled typed strategy",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测类型化策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造类型化策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=("构造类型化策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                # The shared Tool retains flat approval aliases only for
                # approval/reject compatibility. Missing aliases on the other
                # three typed envelopes are valid, hence ``allow_null``.
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成类型化策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造类型化策略.output.strategy_id",
            },
            depends_on_titles=("构造类型化策略", "回测类型化策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


TYPED_STRATEGY_APPLY = WorkflowTemplate(
    id="typed_strategy_apply",
    title="构造并应用类型化策略",
    goal_patterns=("构造并应用策略", "build and apply typed strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Task-owned input dataset id"),
        SlotSpec("strategy_spec", True, "user", "Validated canonical Strategy DSL"),
    ),
    steps=(
        StepTemplate(
            title="构造待应用策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_spec": "{slot:strategy_spec}",
                "description": "Natural-language compiled strategy for application",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="应用类型化策略并生成逐行结果",
            tool_ref=ToolRef("strategy", "apply_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造待应用策略.output.strategy_id",
            },
            depends_on_titles=("构造待应用策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("range", {"field": "population_count", "min": 0}),
                PostCheck("nonempty", {"field": "evidence"}),
            ),
        ),
        StepTemplate(
            title="生成已应用策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造待应用策略.output.strategy_id",
            },
            depends_on_titles=("构造待应用策略", "应用类型化策略并生成逐行结果"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_EVALUATION = WorkflowTemplate(
    id="stored_strategy_evaluation",
    title="已有策略评估",
    goal_patterns=(
        "回测已有策略",
        "分析已有策略",
        "对比已有策略",
        "stored strategy evaluation",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec("strategy_id", True, "user", "Task-owned strategy id"),
        SlotSpec(
            "baseline_strategy_id", False, "user", "Optional same-type baseline id"
        ),
        SlotSpec(
            "economics_inputs", False, "user", "Typed limit/pricing economics inputs"
        ),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="回测已有策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="生成已有策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=("回测已有策略",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_REPORT = WorkflowTemplate(
    id="stored_strategy_report",
    title="已有策略报告",
    goal_patterns=("生成已有策略报告", "stored strategy report"),
    slots=(SlotSpec("strategy_id", True, "user", "Task-owned strategy id"),),
    steps=(
        StepTemplate(
            title="生成已有策略报告",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_APPLY = WorkflowTemplate(
    id="stored_strategy_apply",
    title="应用已有策略",
    goal_patterns=("应用已有策略", "执行已有策略", "apply stored strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Task-owned input dataset id"),
        SlotSpec("strategy_id", True, "user", "Task-owned persisted strategy id"),
    ),
    steps=(
        StepTemplate(
            title="应用已有策略并生成逐行结果",
            tool_ref=ToolRef("strategy", "apply_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("range", {"field": "population_count", "min": 0}),
                PostCheck("nonempty", {"field": "output_columns"}),
                PostCheck("nonempty", {"field": "evidence"}),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STORED_STRATEGY_ADOPTION = WorkflowTemplate(
    id="stored_strategy_adoption",
    title="已有策略采纳",
    goal_patterns=("采纳已有策略", "adopt stored strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec("strategy_id", True, "user", "Task-owned draft strategy id"),
        SlotSpec("adoption_reason", True, "user", "Human supplied adoption reason"),
        SlotSpec(
            "economics_inputs", False, "user", "Typed limit/pricing economics inputs"
        ),
        SlotSpec("profit_params", False, "user", "Approval/reject profit parameters"),
        SlotSpec("ead_col", False, "user", "Approval/reject EAD column"),
        SlotSpec("pd_col", False, "user", "Approval/reject PD column"),
    ),
    steps=(
        StepTemplate(
            title="采纳前回测",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "{slot:strategy_id}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "economics_inputs": "{slot:economics_inputs}",
                "profit_params": "{slot:profit_params}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("nonempty", {"field": "schema_version"}),
                PostCheck("nonempty", {"field": "metrics"}),
                PostCheck(
                    "range",
                    {
                        "field": "approval_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "approved_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "range",
                    {
                        "field": "rejected_bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck("range", {"field": "expected_profit", "allow_null": True}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="采纳已有策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "backtest_id": "$ref:采纳前回测.output.backtest_id",
                "adoption_reason": "{slot:adoption_reason}",
            },
            depends_on_titles=("采纳前回测",),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            needs_confirmation=True,
        ),
        StepTemplate(
            title="生成采纳策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={"strategy_id": "{slot:strategy_id}"},
            depends_on_titles=("采纳已有策略",),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


VINTAGE_ANALYSIS = WorkflowTemplate(
    id="vintage_analysis",
    title="Vintage 风险分析",
    goal_patterns=("风险分析", "vintage", "vintage analysis", "账龄分析"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered vintage dataset id"),
        SlotSpec("cohort_col", True, "task_context", "Cohort/month column"),
        SlotSpec("mob_col", True, "task_context", "Month-on-book column"),
        SlotSpec("bad_col", True, "task_context", "Binary bad/default target column"),
        SlotSpec("mob_max", False, "task_context", "Maximum MOB to render"),
        SlotSpec("ref_mob", False, "task_context", "Reference MOB for trend summary"),
        # A1: label_semantics has no default slot value on purpose -- an undeclared
        # basis makes tool_vintage_curve raise LabelSemanticsNotDeclaredError so the
        # user is forced to pick incremental vs snapshot; drop_nan_labels threads the
        # NaN-label confirmation through the same gate.
        SlotSpec(
            "label_semantics",
            False,
            "user",
            "Bad-column cumulation basis: incremental or snapshot",
        ),
        SlotSpec("drop_nan_labels", False, "user", "Confirm dropping NaN-label rows"),
    ),
    steps=(
        StepTemplate(
            title="计算 Vintage 曲线",
            tool_ref=ToolRef("strategy", "vintage_curve"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "cohort_col": "{slot:cohort_col}",
                "mob_col": "{slot:mob_col}",
                "bad_col": "{slot:bad_col}",
                "mob_max": "{slot:mob_max}",
                "ref_mob": "{slot:ref_mob}",
                # Literal null default (not {slot:label_semantics}), mirroring
                # design_cutoff_bands' band_edges: apply_adjust's gate override only
                # reaches a key already present in the instantiated inputs, and an
                # omitted slot would be dropped by planner._fill_inputs. Baking null
                # here (a valid ["string","null"] per the manifest) is what lets the
                # gate write the user's incremental/snapshot choice onto this step;
                # an unanswered gate leaves null -> the tool raises the semantics gate.
                "label_semantics": None,
                # Baked False (a valid boolean per the manifest) for the same reason,
                # so a "drop the NaN rows" confirmation can be written onto the step.
                "drop_nan_labels": False,
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "cohorts"}),),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

SLICE_AGGREGATE = WorkflowTemplate(
    # S6 ad-hoc 问数 entry: a single deterministic group-by aggregate over a
    # ready dataset. The LLM only produced a validated SliceSpec (INV-1); the
    # 口径确认门 is handled turn-side (turn_handlers) BEFORE this plan is built,
    # so — like vintage_analysis — the one step just runs to DONE and renders its
    # table (no needs_confirmation gate). Every slice_aggregate input is an
    # optional slot; turn_handlers fills exactly SliceSpec.tool_inputs(dataset_id),
    # and omitted slots drop out via _fill_inputs' _OMIT handling.
    id="slice_aggregate",
    title="即席问数",
    goal_patterns=("问数", "即席分析", "slice aggregate", "ad-hoc query"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Ready dataset id to aggregate"),
        SlotSpec(
            "metrics", True, "task_context", "Validated aggregate metrics (op/col)"
        ),
        SlotSpec("group_by", False, "task_context", "Optional group-by columns"),
        SlotSpec("filters", False, "task_context", "Optional filter conditions"),
        SlotSpec("month_col", False, "task_context", "Optional month column"),
        SlotSpec("months", False, "task_context", "Optional month range"),
        SlotSpec("sort_by", False, "task_context", "Optional sort column/metric label"),
    ),
    steps=(
        StepTemplate(
            title="计算聚合结果",
            tool_ref=ToolRef("data_ops", "slice_aggregate"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "metrics": "{slot:metrics}",
                "group_by": "{slot:group_by}",
                "filters": "{slot:filters}",
                "month_col": "{slot:month_col}",
                "months": "{slot:months}",
                "sort_by": "{slot:sort_by}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "columns"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_DEVELOPMENT = WorkflowTemplate(
    # S2 conversational strategy-development template. It is the standard entry;
    # strategy_analysis remains only as an explicit quick-analysis compatibility path.
    # Flow: tradeoff scan (direction self-check) -> design cutoff bands -> build
    # strategy from the recommended rules -> backtest -> [optional] compare vs
    # baseline -> [mandatory confirm] adopt -> render doc. goal_patterns are disjoint from
    # strategy_analysis so keyword routing never crosses the two.
    id="strategy_development",
    title="策略开发",
    goal_patterns=(
        "策略开发",
        "开发策略",
        "策略分析",
        "策略回测",
        "策略权衡",
        "设计cutoff",
        "分数带策略",
        "strategy development",
        "strategy analysis",
        "strategy backtest",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec("score_col", True, "task_context", "Score column"),
        SlotSpec(
            "score_direction",
            False,
            "task_context",
            "Score direction if a model artifact injected one",
        ),
        SlotSpec("objective", False, "user", "max_profit or max_approval"),
        SlotSpec("max_bad_rate", False, "user", "Max approved bad rate constraint"),
        SlotSpec("min_approval_rate", False, "user", "Min approval rate constraint"),
        SlotSpec(
            "ead_col", False, "user", "Exposure-at-default column for profit evaluation"
        ),
        SlotSpec(
            "pd_col",
            False,
            "user",
            "Probability-of-default column for profit evaluation",
        ),
        SlotSpec(
            "profit_params", False, "user", "Profit parameters for expected-profit"
        ),
        SlotSpec(
            "strategy_type", True, "task_context", "Approval or reject strategy type"
        ),
        SlotSpec(
            "baseline_strategy_id",
            False,
            "user",
            "Baseline strategy id for the optional compare step",
        ),
    ),
    steps=(
        StepTemplate(
            title="权衡扫描",
            tool_ref=ToolRef("strategy", "tradeoff_view"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "score_direction": "{slot:score_direction}",
                "objective": "{slot:objective}",
                "max_bad_rate": "{slot:max_bad_rate}",
                "min_approval_rate": "{slot:min_approval_rate}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "points"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="设计分数带",
            tool_ref=ToolRef("strategy", "design_cutoff_bands"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "score_direction": "{slot:score_direction}",
                "objective": "{slot:objective}",
                "max_bad_rate": "{slot:max_bad_rate}",
                "min_approval_rate": "{slot:min_approval_rate}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
                # Keep an explicit null key so a validated structured adjustment
                # can supply manual edges before execution. Omitted optional slot
                # keys are otherwise removed by planner._fill_inputs.
                "band_edges": None,
            },
            depends_on_titles=("权衡扫描",),
            post_checks=(PostCheck("nonempty", {"field": "bands"}),),
        ),
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                "strategy_type": "{slot:strategy_type}",
                "rules": "$ref:设计分数带.output.recommended_rules",
                "score_col": "{slot:score_col}",
                "default_decision": "approve",
                "description": "Strategy development generated candidate",
            },
            depends_on_titles=("设计分数带",),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck(
                    "range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "expected_profit", "allow_null": True}
                ),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="对比基线",
            tool_ref=ToolRef("strategy", "compare_strategies"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "baseline_strategy_id": "{slot:baseline_strategy_id}",
                "ead_col": "{slot:ead_col}",
                "pd_col": "{slot:pd_col}",
                "profit_params": "{slot:profit_params}",
            },
            depends_on_titles=("构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "status"}),),
            decision_point=True,
        ),
        StepTemplate(
            # S6: optional challenger report step, sits right after the compare step.
            # planner has no step-pruning, so degradation is at the TOOL level: with no
            # champion (baseline_strategy_id slot omitted) render_challenger_report
            # returns status='no_baseline' + a 「未提供基线」 markdown and writes NO
            # artifact -- exactly the compare_strategies no-op precedent, so the step
            # never fails the plan. The renderer accepts only the persisted challenger
            # backtest receipt; it reloads and recomputes task-owned champion/challenger
            # evidence instead of trusting caller-supplied metrics or adoption flags.
            title="挑战者报告",
            tool_ref=ToolRef("strategy", "render_challenger_report"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "champion_strategy_id": "{slot:baseline_strategy_id}",
                "challenger_backtest": "$ref:回测策略.output",
            },
            depends_on_titles=("构造策略", "对比基线", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "status"}),),
        ),
        StepTemplate(
            title="采纳策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "backtest_id": "$ref:回测策略.output.backtest_id",
                # The evidence-bound final gate writes the operator's reason into this
                # explicit override target.  Task setup must never pre-authorize adoption.
                "adoption_reason": "",
                "band_stats": "$ref:设计分数带.output",
            },
            depends_on_titles=("设计分数带", "构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            # Mandatory adoption gate: auto-accept must not pass it through
            # (delivery-gate precedent), so the driver always pauses here.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "band_stats": "$ref:设计分数带.output",
            },
            depends_on_titles=("设计分数带", "构造策略", "采纳策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


RULE_STRATEGY = WorkflowTemplate(
    # S4 conversational rule-mining strategy template (new id; disjoint
    # goal_patterns from strategy_analysis/strategy_development so keyword routing
    # never crosses them). Flow: mine candidate reject rules -> select an ordered
    # subset -> evaluate the chosen set (waterfall/overlap) -> build a reject
    # strategy from the selected rules -> backtest -> [mandatory confirm] adopt
    # (S2 forced gate reused) -> render doc. Adoption/doc/memory reuse S2 unchanged.
    id="rule_strategy",
    title="规则策略开发",
    goal_patterns=("规则挖掘", "拒绝规则", "规则策略", "rule mining", "rule strategy"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec(
            "drop_nan_labels", False, "user", "Confirmed target null-row exclusion"
        ),
        SlotSpec(
            "feature_cols",
            False,
            "user",
            "Candidate feature columns (default: numeric columns)",
        ),
        SlotSpec(
            "score_col",
            False,
            "user",
            "Score column, if the rules should carry score-band rules",
        ),
        SlotSpec("max_depth", False, "user", "Decision-tree depth for rule mining"),
        SlotSpec("min_support", False, "user", "Minimum rule support"),
        SlotSpec("min_lift", False, "user", "Minimum rule lift"),
        SlotSpec("top_k", False, "user", "Maximum candidate rules to return"),
    ),
    steps=(
        StepTemplate(
            title="挖掘规则",
            tool_ref=ToolRef("strategy", "mine_rules"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "feature_cols": "{slot:feature_cols}",
                "max_depth": "{slot:max_depth}",
                "min_support": "{slot:min_support}",
                "min_lift": "{slot:min_lift}",
                "top_k": "{slot:top_k}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "candidate_rules"}),),
        ),
        StepTemplate(
            title="选择规则集",
            tool_ref=ToolRef("strategy", "select_rule_set"),
            inputs_template={
                "candidate_rules": "$ref:挖掘规则.output.candidate_rules",
                # None deterministically keeps all mined candidates. A structured
                # request compiler may supply an explicit subset before plan run;
                # selection itself is reversible and is not a governance gate.
                "selection": None,
            },
            depends_on_titles=("挖掘规则",),
            post_checks=(PostCheck("nonempty", {"field": "selected_rules"}),),
        ),
        StepTemplate(
            title="评估规则集",
            tool_ref=ToolRef("strategy", "evaluate_rule_set"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
                "rules": "$ref:选择规则集.output.selected_rules",
            },
            depends_on_titles=("选择规则集",),
            post_checks=(PostCheck("nonempty", {"field": "waterfall"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="构造策略",
            tool_ref=ToolRef("strategy", "build_strategy"),
            inputs_template={
                # Pin the literal reject-strategy defaults (SlotSpec has no
                # default-value mechanism): a rule strategy is an approval-type
                # strategy whose selected rules reject, defaulting to approve.
                "strategy_type": "approval",
                "rules": "$ref:选择规则集.output.selected_rules",
                # score_col flows only when the optional slot is filled; when a
                # score column is present build_strategy's rule-direction
                # self-check (S1a) fires automatically on any score-band rules.
                "score_col": "{slot:score_col}",
                "default_decision": "approve",
                "description": "Rule strategy generated candidate",
            },
            depends_on_titles=("选择规则集",),
            post_checks=(PostCheck("nonempty", {"field": "strategy_id"}),),
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
                "drop_nan_labels": "{slot:drop_nan_labels}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck(
                    "range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}
                ),
                PostCheck(
                    "range", {"field": "expected_profit", "allow_null": True}
                ),  # FIN-3 #4: None when profit requested w/o pd_col (graceful EL degradation)
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="采纳策略",
            tool_ref=ToolRef("strategy", "adopt_strategy"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "backtest_id": "$ref:回测策略.output.backtest_id",
                "adoption_reason": "",
            },
            depends_on_titles=("构造策略", "回测策略"),
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            # Mandatory adoption gate (S2 forced-gate precedent): auto-accept must
            # not pass it through, so the driver always pauses here.
            needs_confirmation=True,
        ),
        StepTemplate(
            title="策略文档",
            tool_ref=ToolRef("strategy", "render_strategy_doc"),
            inputs_template={
                "strategy_id": "$ref:构造策略.output.strategy_id",
            },
            depends_on_titles=("构造策略", "采纳策略"),
            post_checks=(PostCheck("nonempty", {"field": "doc_path"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
