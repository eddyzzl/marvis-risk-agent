"""S3 组合分析模板 (PORTFOLIO_ANALYSIS).

七步：1 流量 ∥ 2 迁徙热力 ∥ 3 细分画像 ∥ 4 稳定性趋势（有 experiment_id 才在），
5 损失估计（依赖迁徙热力），6 组合分析汇总（依赖全部前步，decision_point +
needs_confirmation，门文案聚合各步 red_flags=门 checklist），7 生成组合报告
（依赖汇总，post nonempty report_path）。

步骤 1-4 无互依赖，executor 就绪即跑语义天然并行。experiment_id 缺省时用不含
趋势步的 PORTFOLIO_ANALYSIS_NO_TREND 变体（剪步语义：planner from_template 不丢步，
所以由 setup 选变体实现"无 experiment_id 去掉趋势步"）。
"""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.plugins.manifest import ToolRef

_GOAL_PATTERNS = ("组合分析", "组合报告", "资产质量", "portfolio analysis")


def _base_slots() -> tuple[SlotSpec, ...]:
    return (
        SlotSpec("performance_dataset_id", True, "task_context", "Registered performance snapshot dataset id"),
        SlotSpec("id_col", True, "task_context", "Loan id column"),
        SlotSpec("snapshot_col", True, "task_context", "Snapshot month column"),
        SlotSpec("bucket_col", True, "task_context", "Delinquency bucket column"),
        SlotSpec("states", True, "task_context", "Ordered bucket states (worst last), human-confirmed"),
        SlotSpec("balance_col", False, "task_context", "Balance column (count basis when absent)"),
        SlotSpec("segment_col", False, "user", "Segment column for the profile step"),
        SlotSpec("score_col", False, "user", "Score column for the trend step"),
        SlotSpec("experiment_id", False, "user", "Experiment id for the stability-trend step"),
    )


def _flow_step() -> StepTemplate:
    return StepTemplate(
        title="流量分析",
        tool_ref=ToolRef("analysis", "flow_rate"),
        inputs_template={
            "dataset_id": "{slot:performance_dataset_id}",
            "id_col": "{slot:id_col}",
            "snapshot_col": "{slot:snapshot_col}",
            "bucket_col": "{slot:bucket_col}",
            "states": "{slot:states}",
            "balance_col": "{slot:balance_col}",
        },
        depends_on_titles=(),
        post_checks=(PostCheck("nonempty", {"field": "months"}),),
    )


def _migration_step() -> StepTemplate:
    return StepTemplate(
        title="迁徙热力",
        tool_ref=ToolRef("analysis", "bucket_migration"),
        inputs_template={
            "dataset_id": "{slot:performance_dataset_id}",
            "id_col": "{slot:id_col}",
            "snapshot_col": "{slot:snapshot_col}",
            "bucket_col": "{slot:bucket_col}",
            "states": "{slot:states}",
            "balance_col": "{slot:balance_col}",
        },
        depends_on_titles=(),
        post_checks=(PostCheck("nonempty", {"field": "avg_matrix"}),),
    )


def _segment_step() -> StepTemplate:
    return StepTemplate(
        title="细分画像",
        tool_ref=ToolRef("analysis", "segment_profile"),
        inputs_template={
            "dataset_id": "{slot:performance_dataset_id}",
            "segment_col": "{slot:segment_col}",
        },
        depends_on_titles=(),
        post_checks=(PostCheck("nonempty", {"field": "segments"}),),
    )


def _trend_step() -> StepTemplate:
    return StepTemplate(
        title="稳定性趋势",
        tool_ref=ToolRef("analysis", "score_stability_trend"),
        inputs_template={
            "experiment_id": "{slot:experiment_id}",
            "dataset_id": "{slot:performance_dataset_id}",
            "month_col": "{slot:snapshot_col}",
            "score_col": "{slot:score_col}",
        },
        depends_on_titles=(),
        post_checks=(PostCheck("nonempty", {"field": "trend"}),),
    )


def _el_step() -> StepTemplate:
    return StepTemplate(
        title="损失估计",
        tool_ref=ToolRef("analysis", "expected_loss_estimate"),
        inputs_template={
            "dataset_id": "{slot:performance_dataset_id}",
            "id_col": "{slot:id_col}",
            "snapshot_col": "{slot:snapshot_col}",
            "bucket_col": "{slot:bucket_col}",
            "states": "{slot:states}",
            "balance_col": "{slot:balance_col}",
        },
        depends_on_titles=("迁徙热力",),
        post_checks=(PostCheck("nonempty", {"field": "chain"}),),
    )


def _gate_step(*, with_trend: bool) -> StepTemplate:
    inputs = {
        "flow": "$ref:流量分析.output",
        "migration": "$ref:迁徙热力.output",
        "segment": "$ref:细分画像.output",
        "expected_loss": "$ref:损失估计.output",
    }
    depends = ["流量分析", "迁徙热力", "细分画像", "损失估计"]
    if with_trend:
        inputs["trend"] = "$ref:稳定性趋势.output"
        depends.insert(3, "稳定性趋势")
    return StepTemplate(
        title="组合分析汇总",
        tool_ref=ToolRef("analysis", "portfolio_gate_summary"),
        inputs_template=inputs,
        depends_on_titles=tuple(depends),
        post_checks=(PostCheck("nonempty", {"field": "checklist"}),),
        decision_point=True,
        needs_confirmation=True,
    )


def _report_step(*, with_trend: bool) -> StepTemplate:
    inputs = {
        "flow": "$ref:流量分析.output",
        "migration": "$ref:迁徙热力.output",
        "segment": "$ref:细分画像.output",
        "expected_loss": "$ref:损失估计.output",
        "project_meta": "{slot:project_meta}",
    }
    # Every $ref needs a dependency edge (PlanValidator), so the report step
    # depends on each analysis step it carries numbers from -- plus the gate,
    # which serializes it after the confirmation.
    depends = ["流量分析", "迁徙热力", "细分画像", "损失估计", "组合分析汇总"]
    if with_trend:
        inputs["trend"] = "$ref:稳定性趋势.output"
        depends.insert(3, "稳定性趋势")
    return StepTemplate(
        title="生成组合报告",
        tool_ref=ToolRef("analysis", "portfolio_report"),
        inputs_template=inputs,
        depends_on_titles=tuple(depends),
        post_checks=(PostCheck("nonempty", {"field": "report_path"}),),
    )


PORTFOLIO_ANALYSIS = WorkflowTemplate(
    id="portfolio_analysis",
    title="组合分析",
    goal_patterns=_GOAL_PATTERNS,
    slots=(
        *_base_slots(),
        SlotSpec("project_meta", False, "task_context", "Project metadata for the report overview"),
    ),
    steps=(
        _flow_step(),
        _migration_step(),
        _segment_step(),
        _trend_step(),
        _el_step(),
        _gate_step(with_trend=True),
        _report_step(with_trend=True),
    ),
    default_autonomy=1,
    source="builtin",
)

# Pruned variant used when no experiment_id is supplied: the stability-trend
# step (4) is dropped and the gate/report no longer inject its $ref. Same id
# family/goal patterns are NOT reused -- this is a distinct id the setup selects.
PORTFOLIO_ANALYSIS_NO_TREND = WorkflowTemplate(
    id="portfolio_analysis_no_trend",
    title="组合分析（无趋势）",
    goal_patterns=(),
    slots=(
        *_base_slots(),
        SlotSpec("project_meta", False, "task_context", "Project metadata for the report overview"),
    ),
    steps=(
        _flow_step(),
        _migration_step(),
        _segment_step(),
        _el_step(),
        _gate_step(with_trend=False),
        _report_step(with_trend=False),
    ),
    default_autonomy=1,
    source="builtin",
)


__all__ = ["PORTFOLIO_ANALYSIS", "PORTFOLIO_ANALYSIS_NO_TREND"]
