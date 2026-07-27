from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.plugins.manifest import ToolRef


MONITORING_RUN = WorkflowTemplate(
    # S1b/DOM-3: closes the monitoring-policy execution gap -- score a fresh
    # dataset against a trained experiment's artifact, then run PSI/CSI/KS/AUC
    # checks against the training-time baseline snapshot. The result remains an
    # evidence-bearing decision point, but an ordinary monitoring run has no
    # side effect that warrants a local confirmation gate.
    id="monitoring_run",
    title="模型监控运行",
    goal_patterns=("模型监控", "监控运行", "monitoring run", "model monitoring"),
    slots=(
        SlotSpec("experiment_id", True, "task_context", "Trained modeling experiment id"),
        SlotSpec("dataset_id", True, "task_context", "New dataset id to score and monitor"),
        SlotSpec("target_col", False, "task_context", "Optional label column if the new sample is labeled"),
        SlotSpec("monitoring_policy", False, "task_context", "Optional monitor_run threshold overrides"),
    ),
    steps=(
        StepTemplate(
            title="打分",
            tool_ref=ToolRef("modeling", "score_dataset"),
            inputs_template={
                "experiment_id": "{slot:experiment_id}",
                "dataset_id": "{slot:dataset_id}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("range", {"field": "score_missing_rate", "min": 0.0, "max": 1.0}),
            ),
        ),
        StepTemplate(
            title="监控运行",
            tool_ref=ToolRef("modeling", "monitor_run"),
            inputs_template={
                "experiment_id": "{slot:experiment_id}",
                "scored_dataset_id": "$ref:打分.output.result_dataset_id",
                "score_col": "$ref:打分.output.score_col",
                "target_col": "{slot:target_col}",
                "monitoring_policy": "{slot:monitoring_policy}",
            },
            depends_on_titles=("打分",),
            post_checks=(
                PostCheck("nonempty", {"field": "overall_level"}),
                PostCheck("nonempty", {"field": "checks"}),
            ),
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_MONITORING = WorkflowTemplate(
    # S5: strategy monitoring closure. Runs one monitoring pass off an adopted
    # strategy's immutable monitoring plan (model PSI/CSI via the monitor_run kernel when
    # model-backed, plus strategy-facing approval/bad-rate drift vs the adoption
    # baseline), pausing at an alarm confirmation gate whose copy names the
    # red/amber flags. The governed disposition step binds the exact immutable
    # plan/run receipt the user reviewed. On a red verdict it offers three real
    # dispositions (维持并观察 / 调阈值重跑 / 起新版本策略): observe records an
    # immutable decision, threshold adjustment appends a plan revision and reruns,
    # and new-version creates a fresh draft strategy task. The final step renders
    # the resulting receipt; it no longer substitutes a suggestion for execution.
    id="strategy_monitoring",
    title="策略监控",
    goal_patterns=("策略监控", "跑监控", "monitoring run 策略", "strategy monitoring"),
    slots=(
        SlotSpec("strategy_id", True, "task_context", "Adopted strategy id to monitor"),
        SlotSpec("dataset_id", True, "user", "New-period performance/application dataset id"),
        SlotSpec("score_col", False, "task_context", "Score column when the strategy is model-backed"),
        SlotSpec("target_col", False, "task_context", "Optional label column if the new sample is labeled"),
    ),
    steps=(
        StepTemplate(
            title="执行策略监控",
            tool_ref=ToolRef("strategy", "run_strategy_monitoring"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "dataset_id": "{slot:dataset_id}",
                "score_col": "{slot:score_col}",
                "target_col": "{slot:target_col}",
            },
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "overall_level"}),
                PostCheck("nonempty", {"field": "checks"}),
                PostCheck("nonempty", {"field": "monitoring_plan_id"}),
                PostCheck("nonempty", {"field": "monitoring_run_id"}),
            ),
        ),
        StepTemplate(
            title="处置监控结果",
            tool_ref=ToolRef("strategy", "apply_monitoring_disposition"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "monitoring_run_id": "$ref:执行策略监控.output.monitoring_run_id",
                "expected_plan_id": "$ref:执行策略监控.output.monitoring_plan_id",
                "expected_plan_revision": "$ref:执行策略监控.output.monitoring_plan_revision",
                "expected_plan_hash": "$ref:执行策略监控.output.monitoring_plan_hash",
                # Literal None defaults are filled only through the governed gate.
                # A red run may never turn bare "确认" into implicit observe.
                "disposition": None,
                "reason": None,
                "threshold_patch": None,
            },
            depends_on_titles=("执行策略监控",),
            post_checks=(
                PostCheck("nonempty", {"field": "status"}),
                PostCheck("nonempty", {"field": "resolved_monitoring_run_id"}),
            ),
            # 告警确认门：文案渲染自 run_strategy_monitoring 的证据。红灯必须
            # 明确三选一；绿/黄灯允许确认知悉。该门执行真实处置，不只是生成
            # next_action 提示。
            needs_confirmation=True,
            decision_point=True,
        ),
        StepTemplate(
            title="生成监控报告",
            tool_ref=ToolRef("strategy", "render_monitoring_report"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "source_monitoring_run_id": (
                    "$ref:处置监控结果.output.source_monitoring_run_id"
                ),
            },
            depends_on_titles=("处置监控结果",),
            post_checks=(
                PostCheck("nonempty", {"field": "report_path"}),
                PostCheck("nonempty", {"field": "artifact_id"}),
            ),
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
