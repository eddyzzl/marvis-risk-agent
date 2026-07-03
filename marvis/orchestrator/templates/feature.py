from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.orchestrator.templates._shared import JOIN_EXECUTE_POST_CHECKS
from marvis.plugins.manifest import ToolRef


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

FEATURE_ANALYSIS_WITH_JOIN = WorkflowTemplate(
    id="feature_analysis_with_join",
    title="多表特征分析",
    goal_patterns=("多表特征分析", "join then feature analysis"),
    slots=(
        SlotSpec("anchor_id", True, "task_context", "Anchor sample dataset id"),
        SlotSpec("feature_ids", True, "task_context", "Feature dataset ids to join"),
        SlotSpec("target_col", True, "task_context", "Target column on the anchor/joined sample"),
        SlotSpec("features", False, "task_context", "Candidate features; empty means infer after join"),
        SlotSpec("metrics", False, "task_context", "Selected optional metrics"),
        SlotSpec("dedup_strategies", False, "task_context", "Optional per-feature dedup strategy map"),
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
            title="特征指标",
            tool_ref=ToolRef("feature", "compute_feature_metrics"),
            inputs_template={
                "dataset_id": "$ref:执行拼接.output.result_dataset_id",
                "features": "{slot:features}",
                "target_col": "{slot:target_col}",
                "metrics": "{slot:metrics}",
                "bins": 10,
            },
            depends_on_titles=("执行拼接",),
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
            # spec form B §4 + FS-5: leakage-aware screening runs on the DERIVED dataset over
            # the union of base + newly derived columns, so the derived columns actually enter
            # the leakage/redundancy screen (previously it re-screened the original dataset and
            # base columns, so the derived columns never reached selection).
            title="特征筛选",
            tool_ref=ToolRef("feature", "screen_features"),
            inputs_template={
                "dataset_id": "$ref:衍生特征.output.result_dataset_id",
                "features": ["{slot:feature_cols}", "$ref:衍生特征.output.new_columns"],
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=("衍生特征", "分析衍生特征"),
            post_checks=(),
            phase="特征分析",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
