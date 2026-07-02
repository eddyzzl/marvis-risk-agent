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
