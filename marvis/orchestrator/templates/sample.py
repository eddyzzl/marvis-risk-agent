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
                "business_columns": {},
                "project_meta": {},
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
            post_checks=(PostCheck("nonempty", {"field": "backtest_id"}),),
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
_register_builtin_template(FEATURE_DERIVATION)
_register_builtin_template(STRATEGY_ANALYSIS)
