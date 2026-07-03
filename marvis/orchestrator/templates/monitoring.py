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
    # checks against the training-time baseline snapshot, ending in an alert
    # confirmation gate whose copy names any red/amber flags and the suggested
    # action (see marvis.agent.renderers._render_monitor_run).
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
            # 告警确认门:门文案（渲染自 monitor_run 输出）列出红/黄旗名目与建议动作，
            # 需人工确认后才算本次监控运行收口（S-8 红旗 checklist 精神的监控侧落地）。
            needs_confirmation=True,
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


STRATEGY_MONITORING = WorkflowTemplate(
    # S5: strategy monitoring closure. Runs one monitoring pass off an adopted
    # strategy's monitoring plan (model PSI/CSI via the monitor_run kernel when
    # model-backed, plus strategy-facing approval/bad-rate drift vs the adoption
    # baseline), pausing at an alarm confirmation gate whose copy names the
    # red/amber flags. On a red verdict the gate offers three dispositions
    # (维持并观察 / 调阈值重跑 / 起新版本策略); the driver parses the reply and, for
    # 起新版本, surfaces a next_action pointing at STRATEGY_DEVELOPMENT (it never
    # auto-creates a task). The second step renders a monitoring report.
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
            ),
        ),
        StepTemplate(
            title="生成监控报告",
            tool_ref=ToolRef("strategy", "render_monitoring_report"),
            inputs_template={
                "strategy_id": "{slot:strategy_id}",
                "overall_level": "$ref:执行策略监控.output.overall_level",
                "checks": "$ref:执行策略监控.output.checks",
                # Literal None default: the red-light gate's parsed disposition
                # (observe/adjust_threshold/new_version) is written onto this gate
                # step's own `disposition` input through the reset_step channel the
                # driver uses at this gate (the band_edges/selection precedent), so
                # the report surfaces next_action. None -> no disposition chosen
                # (green/amber runs skip the checklist).
                "disposition": None,
            },
            depends_on_titles=("执行策略监控",),
            post_checks=(PostCheck("nonempty", {"field": "report_path"}),),
            # 告警确认门:门文案(渲染自其依赖 run_strategy_monitoring 的输出,见
            # renderers._render_run_strategy_monitoring)分级列出红/黄旗;red 时列出
            # 「维持并观察 / 调阈值重跑 / 起新版本策略」三选项处置建议,门回复解析
            # 三关键词。需人工确认才算本次监控收口(执行到报告落盘)。
            needs_confirmation=True,
            decision_point=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
