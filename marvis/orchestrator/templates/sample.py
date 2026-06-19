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
            post_checks=(PostCheck("nonempty", {"field": "materials"}),),
        ),
        StepTemplate(
            title="执行 Notebook",
            tool_ref=ToolRef("v1_compat", "run_notebook"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=("扫描材料",),
            post_checks=(PostCheck("nonempty", {"field": "evidence_ref"}),),
        ),
        StepTemplate(
            title="计算验证指标",
            tool_ref=ToolRef("v1_compat", "compute_validation_metrics"),
            inputs_template={"task_id": "{slot:task_id}"},
            depends_on_titles=("执行 Notebook",),
            post_checks=(
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
            post_checks=(PostCheck("nonempty", {"field": "artifacts"}),),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)

_register_builtin_template(SAMPLE_ECHO)
_register_builtin_template(MODEL_VALIDATION)
