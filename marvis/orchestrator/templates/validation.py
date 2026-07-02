from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.plugins.manifest import ToolRef


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
