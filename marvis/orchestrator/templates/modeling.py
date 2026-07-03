from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.orchestrator.templates._shared import (
    BINARY_MODELING_SUCCESS_CRITERIA,
    JOIN_EXECUTE_POST_CHECKS,
)
from marvis.plugins.manifest import ToolRef


STANDARD_MODELING = WorkflowTemplate(
    id="standard_modeling",
    title="标准建模",
    goal_patterns=("标准建模", "建模", "训练模型", "模型开发", "build model", "train model", "standard modeling"),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered modeling dataset id"),
        SlotSpec("target_col", True, "task_context", "Model target column"),
        SlotSpec("feature_cols", True, "task_context", "Candidate feature columns"),
        SlotSpec("split_col", True, "task_context", "Train/test/oot split column"),
        SlotSpec("split_values", True, "task_context", "Split value mapping"),
        SlotSpec("recipe", True, "user", "Modeling recipe id"),
        SlotSpec("seed", True, "task_context", "Reproducibility seed"),
        SlotSpec("business_columns", False, "task_context", "Optional model report business-column mapping"),
        SlotSpec("feature_dictionary_id", False, "task_context", "Optional feature dictionary dataset id"),
        SlotSpec("project_meta", False, "user", "Optional model report project metadata"),
        SlotSpec("champion_reference", False, "task_context", "Optional prior Champion reference for post-training comparison"),
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
                "split_col": "{slot:split_col}",
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
            title="选择实验",
            tool_ref=ToolRef("modeling", "select_experiment"),
            inputs_template={
                "experiment_ids": ["$ref:训练模型.output.experiment_id"],
                "target_type": "binary",
                "selection_policy": {
                    "require_pmml": True,
                    "require_handoff": True,
                },
            },
            depends_on_titles=("训练模型", "对比实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "selected_experiment_id"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
            ),
            needs_confirmation=True,
        ),
        StepTemplate(
            title="生成模型开发报告",
            tool_ref=ToolRef("modeling", "generate_model_report"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "dataset_id": "{slot:dataset_id}",
                "business_columns": "{slot:business_columns}",
                "feature_dictionary_id": "{slot:feature_dictionary_id}",
                "project_meta": "{slot:project_meta}",
            },
            depends_on_titles=("选择实验",),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "section_status"}),
            ),
            needs_confirmation=True,
        ),
        StepTemplate(
            title="模型交付动作",
            tool_ref=ToolRef("modeling", "post_training_action"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "sample_dataset_id": "{slot:dataset_id}",
                "actions": ["export_pmml", "handoff_to_validation", "create_challenger_backtest"],
                "selection_policy_decision": "$ref:选择实验.output.policy_decision",
                "champion_reference": "{slot:champion_reference}",
            },
            depends_on_titles=("选择实验", "生成模型开发报告"),
            post_checks=(
                PostCheck("nonempty", {"field": "artifact_id"}),
                PostCheck("nonempty", {"field": "actions"}),
            ),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
    success_criteria=BINARY_MODELING_SUCCESS_CRITERIA,
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
        SlotSpec("target_type", False, "task_context", "Target type: binary, continuous, or multiclass"),
        SlotSpec("holdout_values", False, "task_context", "OOT split value(s) held out of the leakage screen"),
        SlotSpec("sample_weight_col", False, "task_context", "Optional sample-weight column for fit/sample weighting"),
        SlotSpec("sample_weight_candidates", False, "task_context", "Detected sample-weight candidate columns"),
        SlotSpec("sample_weight_diagnostics", False, "task_context", "Sample-weight quality diagnostics"),
        SlotSpec("tuning_params", False, "task_context", "Optional fixed tuning/training params chosen by the user or agent"),
        SlotSpec("passthrough_cols", False, "task_context", "Non-feature columns to preserve in the modeling frame"),
        SlotSpec("business_columns", False, "task_context", "Optional model report business-column mapping"),
        SlotSpec("feature_dictionary_id", False, "task_context", "Optional feature dictionary dataset id"),
        SlotSpec("project_meta", False, "user", "Optional model report project metadata"),
        SlotSpec("champion_reference", False, "task_context", "Optional prior Champion reference for post-training comparison"),
        SlotSpec("selection_policy", False, "task_context", "Final model selection delivery policy"),
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
                "passthrough_cols": "{slot:passthrough_cols}",
                "seed": "{slot:seed}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "result_dataset_id"}),),
            phase="特征",
        ),
        StepTemplate(
            title="选择建模规格",
            tool_ref=ToolRef("modeling", "choose_modeling_spec"),
            inputs_template={
                "target_col": "{slot:target_col}",
                "features": "$ref:切分样本.output.feature_cols",
                "target_type": "{slot:target_type}",
                "recipe": "{slot:recipe}",
                "recipes": "{slot:recipes}",
                "sample_weight_col": "{slot:sample_weight_col}",
                "sample_weight_candidates": "{slot:sample_weight_candidates}",
                "sample_weight_diagnostics": "{slot:sample_weight_diagnostics}",
                "n_trials": 40,
                "params": "{slot:tuning_params}",
                "seed": "{slot:seed}",
            },
            depends_on_titles=("切分样本",),
            post_checks=(
                PostCheck("nonempty", {"field": "recipe"}),
                PostCheck("nonempty", {"field": "recipes"}),
            ),
            phase="建模",
        ),
        StepTemplate(
            title="特征筛选",
            tool_ref=ToolRef("modeling", "screen_features"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:选择建模规格.output.feature_cols",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "holdout_values": "{slot:holdout_values}",
                "target_type": "$ref:选择建模规格.output.target_type",
                "leakage_ks": 0.4,
                "max_missing_rate": 0.95,
                # Loose top_k backstop (FS-1 decision #3): a wide raw table (hundreds/
                # thousands of clean columns) must not flow straight into multivariate
                # refinement unbounded — 200 is far above any real feature count but
                # caps the pathological case. iv_min/corr_max in "精选特征" below do the
                # real narrowing.
                "top_k": 200,
            },
            depends_on_titles=("切分样本", "选择建模规格"),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
            # G1 门:确认切分(train/test/oot 计数、按月/渠道分布)后再筛特征
            #(执行器暂停在本步前,驱动展示"切分样本"产出的样本分析)
            needs_confirmation=True,
            phase="特征",
        ),
        StepTemplate(
            # FS-1: multivariate refinement funnel between the sanity-level screen and
            # tuning — IV floor + correlation dedup narrow the screen's clean-but-
            # unranked candidate set before it reaches the model. VIF stays off by
            # default (vif_max=1e9 below never trips; tree recipes don't need
            # multicollinearity control the way linear scorecards do) but iv_min/
            # corr_max are adjustable via the gate's adjust path (same generic
            # mechanism as the screen gate's leakage_ks/max_missing_rate) to loosen or
            # effectively bypass the funnel.
            title="精选特征",
            tool_ref=ToolRef("modeling", "select_features"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:特征筛选.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "holdout_values": "{slot:holdout_values}",
                "target_type": "$ref:选择建模规格.output.target_type",
                "space": "raw",
                "iv_min": 0.02,
                "corr_max": 0.95,
                # VIF off by default (tree recipes don't need multicollinearity control
                # the way linear scorecards do): 1e9 matches correlation.py's own VIF
                # cap sentinel, so the >vif_max check never trips. Set a real threshold
                # (e.g. 10.0) via adjust/template override to opt in.
                "vif_max": 1e9,
            },
            depends_on_titles=("切分样本", "特征筛选", "选择建模规格"),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
            # 门:确认精选后的特征漏斗(进/出特征数、IV 底线与相关去冗余各自淘汰数)后再配置调参
            #(执行器暂停在本步前,驱动展示"特征筛选"产出;确认后才进入调参配置)
            needs_confirmation=True,
            phase="特征",
        ),
        StepTemplate(
            title="配置调参",
            tool_ref=ToolRef("modeling", "configure_tuning"),
            inputs_template={
                "recipe": "$ref:选择建模规格.output.recipe",
                "recipes": "$ref:选择建模规格.output.recipes",
                "target_type": "$ref:选择建模规格.output.target_type",
                "sample_weight_col": "$ref:选择建模规格.output.sample_weight_col",
                "n_trials_by_recipe": "$ref:选择建模规格.output.n_trials_by_recipe",
                "params": "$ref:选择建模规格.output.params",
                "seed": "$ref:选择建模规格.output.seed",
            },
            depends_on_titles=("选择建模规格", "精选特征"),
            post_checks=(
                PostCheck("nonempty", {"field": "reason"}),
            ),
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="调参",
            tool_ref=ToolRef("modeling", "tune_hyperparameters"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:精选特征.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "split_values": "{slot:split_values}",
                "recipe": "$ref:配置调参.output.recipe",
                "recipes": "$ref:配置调参.output.recipes",
                "sample_weight_col": "$ref:配置调参.output.sample_weight_col",
                "seed": "$ref:配置调参.output.seed",
                "params": "$ref:配置调参.output.params",
                # Bounded two-stage random search per recipe so the synchronous
                # driver turn stays responsive; users can request a wider search
                # later (G3). Every BINARY_MODELING_RECIPES family now tunes
                # (TUNE-1/SEL-2) — each with its own budget from n_trials_by_recipe.
                "n_trials_by_recipe": "$ref:配置调参.output.n_trials_by_recipe",
            },
            depends_on_titles=("切分样本", "精选特征", "配置调参"),
            # best_params must be present + a dict: single recipe -> flat params
            # dict (possibly empty for a non-tunable family); multiple recipes ->
            # a dict keyed by recipe id, each value itself the tuned params dict.
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
                "recipes": "$ref:选择建模规格.output.recipes",
                "features": "$ref:精选特征.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "{slot:split_col}",
                "split_values": "{slot:split_values}",
                "params": "$ref:调参.output.best_params",
                "sample_weight_col": "$ref:选择建模规格.output.sample_weight_col",
                "seed": "$ref:选择建模规格.output.seed",
                "target_type": "$ref:选择建模规格.output.target_type",
            },
            depends_on_titles=("切分样本", "选择建模规格", "精选特征", "调参"),
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
            title="选择实验",
            tool_ref=ToolRef("modeling", "select_experiment"),
            inputs_template={
                "experiment_ids": "$ref:训练模型.output.experiment_ids",
                "target_type": "$ref:选择建模规格.output.target_type",
                "selection_policy": "{slot:selection_policy}",
            },
            depends_on_titles=("选择建模规格", "调参", "训练模型", "对比实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "selected_experiment_id"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
            ),
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="生成模型开发报告",
            tool_ref=ToolRef("modeling", "generate_model_report"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "dataset_id": "{slot:dataset_id}",
                "business_columns": "{slot:business_columns}",
                "feature_dictionary_id": "{slot:feature_dictionary_id}",
                "project_meta": "{slot:project_meta}",
            },
            # depends on 调参 too so the model gate shows the trials leaderboard (G4)
            # alongside the trained-model metrics before the report is finalized.
            depends_on_titles=("切分样本", "调参", "训练模型", "选择实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "section_status"}),
            ),
            # 门:确认训练指标/trials 后再定稿报告(执行器暂停在本步前,驱动展示 train+compare)
            needs_confirmation=True,
            phase="报告",
        ),
        StepTemplate(
            title="模型交付动作",
            tool_ref=ToolRef("modeling", "post_training_action"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "sample_dataset_id": "{slot:dataset_id}",
                "actions": ["export_pmml", "handoff_to_validation", "create_challenger_backtest"],
                "selection_policy_decision": "$ref:选择实验.output.policy_decision",
                "champion_reference": "{slot:champion_reference}",
            },
            depends_on_titles=("选择实验", "生成模型开发报告"),
            post_checks=(
                PostCheck("nonempty", {"field": "artifact_id"}),
                PostCheck("nonempty", {"field": "actions"}),
            ),
            needs_confirmation=True,
            phase="交付",
        ),
    ),
    default_autonomy=1,
    success_criteria=BINARY_MODELING_SUCCESS_CRITERIA,
    source="builtin",
)

MODELING_WITH_JOIN = WorkflowTemplate(
    id="modeling_with_join",
    title="多表模型开发",
    goal_patterns=("多表模型开发", "join then modeling"),
    slots=(
        SlotSpec("anchor_id", True, "task_context", "Anchor sample dataset id"),
        SlotSpec("feature_ids", True, "task_context", "Feature dataset ids to join"),
        SlotSpec("target_col", True, "task_context", "Target column on the anchor/joined sample"),
        SlotSpec("feature_cols", False, "task_context", "Candidate feature columns; empty means infer after join"),
        SlotSpec("split_col", False, "task_context", "Existing split column, if any"),
        SlotSpec("split_values", False, "task_context", "Existing split value mapping, if any"),
        SlotSpec("recipe", True, "task_context", "Primary recipe to tune (lgb if among recipes)"),
        SlotSpec("recipes", True, "task_context", "Recipe ids to train + compare"),
        SlotSpec("seed", True, "task_context", "Reproducibility seed"),
        SlotSpec("split_config", False, "task_context", "Split rules/config for the G1 make_split gate"),
        SlotSpec("target_type", False, "task_context", "Target type: binary, continuous, or multiclass"),
        SlotSpec("holdout_values", False, "task_context", "OOT split value(s) held out of the leakage screen"),
        SlotSpec("sample_weight_col", False, "task_context", "Optional sample-weight column for fit/sample weighting"),
        SlotSpec("sample_weight_candidates", False, "task_context", "Detected sample-weight candidate columns"),
        SlotSpec("sample_weight_diagnostics", False, "task_context", "Sample-weight quality diagnostics"),
        SlotSpec("tuning_params", False, "task_context", "Optional fixed tuning/training params chosen by the user or agent"),
        SlotSpec("passthrough_cols", False, "task_context", "Non-feature columns to preserve in the modeling frame"),
        SlotSpec("business_columns", False, "task_context", "Optional model report business-column mapping"),
        SlotSpec("feature_dictionary_id", False, "task_context", "Optional feature dictionary dataset id"),
        SlotSpec("project_meta", False, "user", "Optional model report project metadata"),
        SlotSpec("champion_reference", False, "task_context", "Optional prior Champion reference for post-training comparison"),
        SlotSpec("dedup_strategies", False, "task_context", "Optional per-feature dedup strategy map"),
        SlotSpec("selection_policy", False, "task_context", "Final model selection delivery policy"),
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
            post_checks=(PostCheck("one_of", {"field": "status", "values": ["confirmed", "needs_dedup"]}),),
            phase="数据准备",
        ),
        StepTemplate(
            title="执行拼接",
            tool_ref=ToolRef("data_ops", "execute_join"),
            inputs_template={"join_plan_id": "$ref:拼接诊断.output.join_plan_id"},
            depends_on_titles=("拼接诊断", "确认拼接"),
            post_checks=JOIN_EXECUTE_POST_CHECKS,
            needs_confirmation=True,
            phase="数据准备",
        ),
        StepTemplate(
            title="切分样本",
            tool_ref=ToolRef("modeling", "make_split"),
            inputs_template={
                "dataset_id": "$ref:执行拼接.output.result_dataset_id",
                "target_col": "{slot:target_col}",
                "feature_cols": "{slot:feature_cols}",
                "split_col": "{slot:split_col}",
                "split_config": "{slot:split_config}",
                "passthrough_cols": "{slot:passthrough_cols}",
                "seed": "{slot:seed}",
            },
            depends_on_titles=("执行拼接",),
            post_checks=(PostCheck("nonempty", {"field": "result_dataset_id"}),),
            phase="特征",
        ),
        StepTemplate(
            title="选择建模规格",
            tool_ref=ToolRef("modeling", "choose_modeling_spec"),
            inputs_template={
                "target_col": "{slot:target_col}",
                "features": "$ref:切分样本.output.feature_cols",
                "target_type": "{slot:target_type}",
                "recipe": "{slot:recipe}",
                "recipes": "{slot:recipes}",
                "sample_weight_col": "{slot:sample_weight_col}",
                "sample_weight_candidates": "{slot:sample_weight_candidates}",
                "sample_weight_diagnostics": "{slot:sample_weight_diagnostics}",
                "n_trials": 40,
                "params": "{slot:tuning_params}",
                "seed": "{slot:seed}",
            },
            depends_on_titles=("切分样本",),
            post_checks=(
                PostCheck("nonempty", {"field": "recipe"}),
                PostCheck("nonempty", {"field": "recipes"}),
            ),
            phase="建模",
        ),
        StepTemplate(
            title="特征筛选",
            tool_ref=ToolRef("modeling", "screen_features"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:选择建模规格.output.feature_cols",
                "target_col": "{slot:target_col}",
                "split_col": "$ref:切分样本.output.split_col",
                "holdout_values": "$ref:切分样本.output.holdout_values",
                "target_type": "$ref:选择建模规格.output.target_type",
                "leakage_ks": 0.4,
                "max_missing_rate": 0.95,
                # Loose top_k backstop (FS-1 decision #3), same rationale as MODELING.
                "top_k": 200,
            },
            depends_on_titles=("切分样本", "选择建模规格"),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
            needs_confirmation=True,
            phase="特征",
        ),
        StepTemplate(
            # FS-1 multivariate refinement funnel — see MODELING template for rationale.
            title="精选特征",
            tool_ref=ToolRef("modeling", "select_features"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:特征筛选.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "$ref:切分样本.output.split_col",
                "holdout_values": "$ref:切分样本.output.holdout_values",
                "target_type": "$ref:选择建模规格.output.target_type",
                "space": "raw",
                "iv_min": 0.02,
                "corr_max": 0.95,
                # VIF off by default — see MODELING template for rationale.
                "vif_max": 1e9,
            },
            depends_on_titles=("切分样本", "特征筛选", "选择建模规格"),
            post_checks=(PostCheck("nonempty", {"field": "selected"}),),
            needs_confirmation=True,
            phase="特征",
        ),
        StepTemplate(
            title="配置调参",
            tool_ref=ToolRef("modeling", "configure_tuning"),
            inputs_template={
                "recipe": "$ref:选择建模规格.output.recipe",
                "recipes": "$ref:选择建模规格.output.recipes",
                "target_type": "$ref:选择建模规格.output.target_type",
                "sample_weight_col": "$ref:选择建模规格.output.sample_weight_col",
                "n_trials_by_recipe": "$ref:选择建模规格.output.n_trials_by_recipe",
                "params": "$ref:选择建模规格.output.params",
                "seed": "$ref:选择建模规格.output.seed",
            },
            depends_on_titles=("选择建模规格", "精选特征"),
            post_checks=(
                PostCheck("nonempty", {"field": "reason"}),
            ),
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="调参",
            tool_ref=ToolRef("modeling", "tune_hyperparameters"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "features": "$ref:精选特征.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "$ref:切分样本.output.split_col",
                "split_values": "$ref:切分样本.output.split_values",
                "recipe": "$ref:配置调参.output.recipe",
                "recipes": "$ref:配置调参.output.recipes",
                "sample_weight_col": "$ref:配置调参.output.sample_weight_col",
                "seed": "$ref:配置调参.output.seed",
                "params": "$ref:配置调参.output.params",
                "n_trials_by_recipe": "$ref:配置调参.output.n_trials_by_recipe",
            },
            depends_on_titles=("切分样本", "精选特征", "配置调参"),
            post_checks=(PostCheck("schema", {
                "type": "object",
                "properties": {"best_params": {"type": "object"}},
                "required": ["best_params"],
            }),),
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="训练模型",
            tool_ref=ToolRef("modeling", "train_models"),
            inputs_template={
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "recipes": "$ref:选择建模规格.output.recipes",
                "features": "$ref:精选特征.output.selected",
                "target_col": "{slot:target_col}",
                "split_col": "$ref:切分样本.output.split_col",
                "split_values": "$ref:切分样本.output.split_values",
                "params": "$ref:调参.output.best_params",
                "sample_weight_col": "$ref:选择建模规格.output.sample_weight_col",
                "seed": "$ref:选择建模规格.output.seed",
                "target_type": "$ref:选择建模规格.output.target_type",
            },
            depends_on_titles=("切分样本", "选择建模规格", "精选特征", "调参"),
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
            title="选择实验",
            tool_ref=ToolRef("modeling", "select_experiment"),
            inputs_template={
                "experiment_ids": "$ref:训练模型.output.experiment_ids",
                "target_type": "$ref:选择建模规格.output.target_type",
                "selection_policy": "{slot:selection_policy}",
            },
            depends_on_titles=("选择建模规格", "调参", "训练模型", "对比实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "selected_experiment_id"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
            ),
            needs_confirmation=True,
            phase="建模",
        ),
        StepTemplate(
            title="生成模型开发报告",
            tool_ref=ToolRef("modeling", "generate_model_report"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "dataset_id": "$ref:切分样本.output.result_dataset_id",
                "business_columns": "{slot:business_columns}",
                "feature_dictionary_id": "{slot:feature_dictionary_id}",
                "project_meta": "{slot:project_meta}",
            },
            depends_on_titles=("切分样本", "调参", "训练模型", "选择实验"),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "section_status"}),
            ),
            needs_confirmation=True,
            phase="报告",
        ),
        StepTemplate(
            title="模型交付动作",
            tool_ref=ToolRef("modeling", "post_training_action"),
            inputs_template={
                "experiment_id": "$ref:选择实验.output.selected_experiment_id",
                "sample_dataset_id": "$ref:切分样本.output.result_dataset_id",
                "actions": ["export_pmml", "handoff_to_validation", "create_challenger_backtest"],
                "selection_policy_decision": "$ref:选择实验.output.policy_decision",
                "champion_reference": "{slot:champion_reference}",
            },
            depends_on_titles=("切分样本", "选择实验", "生成模型开发报告"),
            post_checks=(
                PostCheck("nonempty", {"field": "artifact_id"}),
                PostCheck("nonempty", {"field": "actions"}),
            ),
            needs_confirmation=True,
            phase="交付",
        ),
    ),
    default_autonomy=1,
    success_criteria=BINARY_MODELING_SUCCESS_CRITERIA,
    source="builtin",
)
