from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
    _register_builtin_template,
)
from marvis.plugins.manifest import ToolRef


SAMPLE_ECHO = WorkflowTemplate(
    id="sample_echo",
    title="Sample Echo Workflow",
    goal_patterns=("echo", "sample echo", "测试编排"),
    slots=(
        SlotSpec("message", True, "user", "Message to echo"),
    ),
    steps=(
        StepTemplate(
            title="Echo",
            tool_ref=ToolRef("_sample", "echo"),
            inputs_template={"message": "{slot:message}"},
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "echoed"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

MODEL_VALIDATION = WorkflowTemplate(
    id="model_validation",
    title="模型验证",
    goal_patterns=("模型验证", "验证模型", "model validation", "run validation"),
    slots=(
        SlotSpec("task_id", True, "task_context", "Current validation task id"),
    ),
    steps=(
        StepTemplate(
            title="扫描材料",
            tool_ref=ToolRef("v1_compat", "scan_materials"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=(),
            post_checks=(
                PostCheck("one_of", {"field": "status", "values": ["scanned"]}),
                PostCheck("nonempty", {"field": "materials"}),
            ),
        ),
        StepTemplate(
            title="执行 Notebook",
            tool_ref=ToolRef("v1_compat", "run_notebook"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=("扫描材料",),
            post_checks=(
                PostCheck("one_of", {"field": "status", "values": ["executed"]}),
                PostCheck("nonempty", {"field": "evidence_ref"}),
            ),
        ),
        StepTemplate(
            title="计算验证指标",
            tool_ref=ToolRef("v1_compat", "compute_validation_metrics"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=("执行 Notebook",),
            post_checks=(
                PostCheck("one_of", {"field": "status", "values": ["writing_artifacts", "review_required"]}),
                PostCheck("range", {"field": "ks", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "auc", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "psi", "min": 0.0, "allow_null": True}),
            ),
        ),
        StepTemplate(
            title="生成报告",
            tool_ref=ToolRef("v1_compat", "render_reports"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=("计算验证指标",),
            post_checks=(
                PostCheck("one_of", {"field": "status", "values": ["succeeded", "review_required"]}),
                PostCheck("nonempty", {"field": "artifacts"}),
            ),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

STANDARD_MODELING = WorkflowTemplate(
    id="standard_modeling",
    title="标准建模",
    goal_patterns=("标准建模", "建模", "训练模型", "模型开发", "build model", "train model", "standard modeling"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered modeling dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("feature_cols", True, "task_context", "Candidate feature columns"),
        SlotSpec("split_col", True, "task_context", "Train/test/oot split column"),
        SlotSpec("split_values", True, "task_context", "Split value mapping"),
        SlotSpec("recipe", True, "user", "Modeling recipe id"),
        SlotSpec("seed", True, "task_context", "Reproducibility seed"),
        SlotSpec("business_columns", False, "task_context", "Optional model report business-column mapping"),
        SlotSpec("feature_dictionary_id", False, "task_context", "Optional feature dictionary dataset id"),
        SlotSpec("project_meta", False, "user", "Optional model report project metadata"),
    ),
    steps=(
        StepTemplate(
            title="检查数据质量",
            tool_ref=ToolRef("modeling", "check_data_quality"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck(
                    "schema",
                    {
                        "type": "object",
                        "properties": {"issues": {"type": "array"}},
                        "required": ["issues"],
                    },
                ),
            ),
        ),
        StepTemplate(
            title="评估建模就绪度",
            tool_ref=ToolRef("modeling", "modeling_readiness"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
            },
            depends_on_titles=("检查数据质量",),
            post_checks=(PostCheck("one_of", {"field": "ready", "values": [True]}),),
        ),
        StepTemplate(
            title="准备建模样本",
            tool_ref=ToolRef("modeling", "prepare_modeling_frame"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "feature_cols": "{slot:feature_cols}",
                "split_col": "{slot:split_col}",
                "split_config": {},
                "seed": "{slot:seed}",
            },
            depends_on_titles=("评估建模就绪度",),
            post_checks=(PostCheck("nonempty", {"field": "result_dataset_id"}),),
        ),
        StepTemplate(
            title="筛选特征",
            tool_ref=ToolRef("modeling", "select_features"),
            inputs_template={
                "dataset_id": "$ref:准备建模样本.output.result_dataset_id",
                "features": "{slot:feature_cols}",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("准备建模样本",),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
        ),
        StepTemplate(
            title="训练模型",
            tool_ref=ToolRef("modeling", "train_model"),
            inputs_template={
                "dataset_id": "$ref:准备建模样本.output.result_dataset_id",
                "recipe": "{slot:recipe}",
                "features": "$ref:筛选特征.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "split_values": "{slot:split_values}",
                "params": {},
                "seed": "{slot:seed}",
            },
            depends_on_titles=("准备建模样本", "筛选特征"),
            post_checks=(
                PostCheck("nonempty", {"field": "experiment_id"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
            ),
        ),
        StepTemplate(
            title="对比实验",
            tool_ref=ToolRef("modeling", "compare_experiments"),
            inputs_template={"experiment_ids": ["$ref:训练模型.output.experiment_id"]},
            depends_on_titles=("训练模型",),
            post_checks=(PostCheck("nonempty", {"field": "experiments"}),),
        ),
        StepTemplate(
            title="生成模型开发报告",
            tool_ref=ToolRef("modeling", "generate_model_report"),
            inputs_template={
                "experiment_id": "$ref:训练模型.output.experiment_id",
                "dataset_id": "{slot:dataset_id}",
                "business_columns": "{slot:business_columns}",
                "feature_dictionary_id": "{slot:feature_dictionary_id}",
                "project_meta": "{slot:project_meta}",
            },
            depends_on_titles=("训练模型", "对比实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "section_status"}),
            ),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

DATA_JOIN = WorkflowTemplate(
    # V2 data-join template (foundation module). The driver instantiates it BY ID
    # for task_type="data_join". The sample table is the anchor (rows preserved
    # 1:1); feature tables are left-joined. Forced-confirmation is honored at two
    # layers: (a) the engine — execute_join hard-blocks until every spec is
    # confirmed (JoinNotConfirmedError); (b) the plan — execute_join is the
    # needs_confirmation human gate (validator INV-3), so the executor pauses
    # BEFORE the data-mutating join and the driver shows propose_join's
    # match/fan-out/fingerprint diagnostics. confirm_join runs between them to mark
    # the engine specs confirmed (per-feature, with optional dedup strategy).
    id="data_join",
    title="数据拼接",
    goal_patterns=("数据拼接", "拼数据", "拼接样本", "join data", "data join"),
    slots=(
        SlotSpec("anchor_id", True, "task_context", "Anchor (sample) dataset id — rows preserved 1:1"),
        SlotSpec("feature_ids", True, "task_context", "Feature dataset ids to left-join onto the anchor"),
        SlotSpec("dedup_strategies", False, "task_context", "Optional per-feature dedup strategy map (feature_id -> first|last)"),
    ),
    steps=(
        StepTemplate(
            title="拼接诊断",
            tool_ref=ToolRef("data_ops", "propose_join"),
            inputs_template={
                "anchor_id": "{slot:anchor_id}",
                "feature_ids": "{slot:feature_ids}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "join_plan_id"}),),
            # spec §2/§10: propose_join is a decision point — in agent mode the executor may
            # adapt the remaining plan from the diagnostics (match/fan-out/fingerprint). The
            # execute_join INV-3 confirmation gate + engine JoinNotConfirmedError backstop keep
            # the 1:1 anchor invariant regardless of any replan.
            decision_point=True,
            phase="数据准备",
        ),
        StepTemplate(
            title="确认拼接",
            tool_ref=ToolRef("data_ops", "confirm_join"),
            inputs_template={
                "join_plan_id": "$ref:拼接诊断.output.join_plan_id",
                "dedup_strategies": "{slot:dedup_strategies}",
            },
            depends_on_titles=("拼接诊断",),
            # needs_dedup is a valid state (a non-unique key still awaiting a strategy): let
            # the plan reach the C2 gate to surface it rather than hard-failing here.
            post_checks=(PostCheck("one_of", {"field": "status", "values": ["confirmed", "needs_dedup"]}),),
            phase="数据准备",
        ),
        StepTemplate(
            title="执行拼接",
            tool_ref=ToolRef("data_ops", "execute_join"),
            inputs_template={
                "join_plan_id": "$ref:拼接诊断.output.join_plan_id",
            },
            depends_on_titles=("拼接诊断", "确认拼接"),
            post_checks=(PostCheck("nonempty", {"field": "result_dataset_id"}),),
            # 强制确认门(INV-3):execute_join 必须 needs_confirmation。执行器暂停在真正左连接之前,
            # 驱动展示拼接诊断(命中率/膨胀/键指纹),用户确认后才执行;引擎层另有 JoinNotConfirmedError 兜底。
            needs_confirmation=True,
            phase="数据准备",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

MODELING = WorkflowTemplate(
    # V2 conversational model-development template. The plan-conversation driver
    # instantiates it BY ID for task_type="modeling"; it mirrors the proven
    # ModelingSession prototype flow on the real PlanExecutor:
    #   leakage-aware screen -> [confirm features] -> tune -> train -> compare
    #   -> [confirm model] -> report.
    # phase tags drive right-rail big-step grouping; needs_confirmation marks the
    # two human gates (the executor pauses BEFORE the gated step, so the driver
    # shows the *prior* step's just-computed result at each pause). goal_patterns
    # are intentionally narrow so generic goal routing still resolves the common
    # "模型开发"/"建模" goals to standard_modeling (legacy select-based flow).
    id="modeling",
    title="模型开发",
    goal_patterns=("对话式模型开发", "conversational model development"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered modeling dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("feature_cols", True, "task_context", "Candidate feature columns"),
        SlotSpec("split_col", True, "task_context", "Train/test/oot split column"),
        SlotSpec("split_values", True, "task_context", "Split value mapping"),
        SlotSpec("recipe", True, "task_context", "Primary recipe to tune (lgb if among recipes)"),
        SlotSpec("recipes", True, "task_context", "Recipe ids to train + compare (≥1)"),
        SlotSpec("seed", True, "task_context", "Reproducibility seed"),
        SlotSpec("split_config", False, "task_context", "Split rules/config for the G1 make_split gate (passthrough when empty)"),
        SlotSpec("target_type", False, "task_context", "Derived target type: binary (default) or continuous"),
        SlotSpec("holdout_values", False, "task_context", "OOT split value(s) held out of the leakage screen"),
        SlotSpec("business_columns", False, "task_context", "Optional model report business-column mapping"),
        SlotSpec("feature_dictionary_id", False, "task_context", "Optional feature dictionary dataset id"),
        SlotSpec("project_meta", False, "user", "Optional model report project metadata"),
    ),
    steps=(
        StepTemplate(
            title="切分样本",
            tool_ref=ToolRef("modeling", "make_split"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "target_col": "{slot:target_col}",
                "feature_cols": "{slot:feature_cols}",
                "split_col": "{slot:split_col}",
                "split_config": "{slot:split_config}",
                "seed": "{slot:seed}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "result_dataset_id"}),),
            phase="特征",
        ),
        StepTemplate(
            title="特征筛选",
            tool_ref=ToolRef("modeling", "screen_features"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "{slot:feature_cols}",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "holdout_values": "{slot:holdout_values}",
                "target_type": "{slot:target_type}",
            },
            depends_on_titles=("切分样本",),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
            # G1 门:确认切分(train/test/oot 计数、按月/渠道分布)后再筛特征
            #(执行器暂停在本步前,驱动展示"切分样本"产出的样本分析)
            needs_confirmation=True,
            phase="特征",
        ),
        StepTemplate(
            title="调参",
            tool_ref=ToolRef("modeling", "tune_hyperparameters"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:特征筛选.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "split_values": "{slot:split_values}",
                "recipe": "{slot:recipe}",
                "seed": "{slot:seed}",
                # Bounded random search so the synchronous driver turn stays
                # responsive; users can request a wider search later (G3). Non-lgb
                # recipes skip the search and train with their own defaults.
                "n_trials": 12,
            },
            depends_on_titles=("切分样本", "特征筛选"),
            # best_params must be present + a dict, but MAY be empty: only lgb runs the random
            # search; every other recipe (lr/xgb/scorecard/mlp/regressor/multiclass) skips it
            # and trains with its own defaults ({}), so "nonempty" would wrongly fail them.
            post_checks=(PostCheck("schema", {
                "type": "object",
                "properties": {"best_params": {"type": "object"}},
                "required": ["best_params"],
            }),),
            # 门:确认筛选出的特征集后再花算力调参(执行器暂停在本步前,驱动展示"特征筛选"产出)
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="训练模型",
            tool_ref=ToolRef("modeling", "train_models"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "recipes": "{slot:recipes}",
                "features": "$ref:特征筛选.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "split_values": "{slot:split_values}",
                "params": "$ref:调参.output.best_params",
                "seed": "{slot:seed}",
                "target_type": "{slot:target_type}",
            },
            depends_on_titles=("切分样本", "特征筛选", "调参"),
            post_checks=(PostCheck("nonempty", {"field": "best_experiment_id"}),),
            phase="建模",
        ),
        StepTemplate(
            title="对比实验",
            tool_ref=ToolRef("modeling", "compare_experiments"),
            inputs_template={"experiment_ids": "$ref:训练模型.output.experiment_ids"},
            depends_on_titles=("训练模型",),
            post_checks=(PostCheck("nonempty", {"field": "experiments"}),),
            phase="建模",
        ),
        StepTemplate(
            title="生成模型开发报告",
            tool_ref=ToolRef("modeling", "generate_model_report"),
            inputs_template={
                "experiment_id": "$ref:训练模型.output.best_experiment_id",
                "dataset_id": "{slot:dataset_id}",
                "business_columns": "{slot:business_columns}",
                "feature_dictionary_id": "{slot:feature_dictionary_id}",
                "project_meta": "{slot:project_meta}",
            },
            # depends on 调参 too so the model gate shows the trials leaderboard (G4)
            # alongside the trained-model metrics before the report is finalized.
            depends_on_titles=("调参", "训练模型", "对比实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "section_status"}),
            ),
            # 门:确认训练指标/trials 后再定稿报告(执行器暂停在本步前,驱动展示 train+compare)
            needs_confirmation=True,
            phase="报告",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

FEATURE_ANALYSIS = WorkflowTemplate(
    # V2 standalone feature-analysis template (spec §1 form A): compute the selected
    # per-feature metrics over a single dataset; the wide table is the report, no
    # screening gate. The driver instantiates it BY ID for task_type="feature_analysis".
    id="feature_analysis",
    title="特征分析",
    goal_patterns=("独立特征分析", "feature analysis report"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Dataset to analyse"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("features", True, "task_context", "Candidate feature columns"),
        SlotSpec("metrics", False, "task_context", "Selected optional metrics (e.g. vif)"),
    ),
    steps=(
        StepTemplate(
            title="特征指标",
            tool_ref=ToolRef("feature", "compute_feature_metrics"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "features": "{slot:features}",
                "target_col": "{slot:target_col}",
                "metrics": "{slot:metrics}",
                "bins": 10,
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "metrics"}),),
            phase="特征分析",
        ),
        StepTemplate(
            title="生成特征分析报告",
            tool_ref=ToolRef("feature", "generate_feature_report"),
            inputs_template={
                "metrics": "$ref:特征指标.output.metrics",
                "collinear": "$ref:特征指标.output.collinear",
            },
            depends_on_titles=("特征指标",),
            post_checks=(PostCheck("nonempty", {"field": "report_path"}),),
            phase="特征分析",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

FEATURE_DERIVATION = WorkflowTemplate(
    id="feature_derivation",
    title="特征衍生与筛选",
    goal_patterns=("特征衍生", "特征交叉", "衍生变量", "feature derivation", "feature crosses"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered feature dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
        SlotSpec("feature_cols", True, "task_context", "Base feature columns"),
        SlotSpec("derivation_recipe", True, "user", "Explicit feature derivation recipe"),
    ),
    steps=(
        StepTemplate(
            title="计算基础特征指标",
            tool_ref=ToolRef("feature", "compute_feature_metrics"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "features": "{slot:feature_cols}",
                "target_col": "{slot:target_col}",
                "bins": 10,
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "metrics"}),),
        ),
        StepTemplate(
            title="衍生特征",
            tool_ref=ToolRef("feature", "cross_features"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "recipe": "{slot:derivation_recipe}",
            },
            depends_on_titles=("计算基础特征指标",),
            post_checks=(
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("nonempty", {"field": "new_columns"}),
            ),
            decision_point=True,
        ),
        StepTemplate(
            title="分析衍生特征",
            tool_ref=ToolRef("feature", "compute_feature_metrics"),
            inputs_template={
                "dataset_id": "$ref:衍生特征.output.result_dataset_id",
                "features": "$ref:衍生特征.output.new_columns",
                "target_col": "{slot:target_col}",
                "bins": 10,
            },
            depends_on_titles=("衍生特征",),
            post_checks=(PostCheck("nonempty", {"field": "metrics"}),),
        ),
        StepTemplate(
            # spec form B §4: leakage-aware screening yields the selected feature set the
            # downstream model should use (the title's "筛选" was previously unimplemented).
            title="特征筛选",
            tool_ref=ToolRef("feature", "screen_features"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "features": "{slot:feature_cols}",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("分析衍生特征",),
            post_checks=(),
            phase="特征分析",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

STRATEGY_ANALYSIS = WorkflowTemplate(
    id="strategy_analysis",
    title="策略分析与回测",
    goal_patterns=("策略分析", "策略回测", "策略权衡", "strategy analysis", "strategy backtest"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered strategy dataset id"),
        SlotSpec("target_col", True, "task_context", "Binary target column"),
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
            needs_confirmation=True,
        ),
        StepTemplate(
            title="回测策略",
            tool_ref=ToolRef("strategy", "backtest_strategy"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "strategy_id": "$ref:构造策略.output.strategy_id",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("构造策略",),
            post_checks=(
                PostCheck("nonempty", {"field": "backtest_id"}),
                PostCheck("range", {"field": "approval_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "approved_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "rejected_bad_rate", "min": 0.0, "max": 1.0}),
                PostCheck("range", {"field": "expected_profit"}),
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
            },
            depends_on_titles=("回测策略",),
            post_checks=(PostCheck("nonempty", {"field": "points"}),),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

_register_builtin_template(SAMPLE_ECHO)
_register_builtin_template(MODEL_VALIDATION)
_register_builtin_template(STANDARD_MODELING)
_register_builtin_template(DATA_JOIN)
_register_builtin_template(MODELING)
_register_builtin_template(FEATURE_ANALYSIS)
_register_builtin_template(FEATURE_DERIVATION)
_register_builtin_template(STRATEGY_ANALYSIS)
