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

_register_builtin_template(SAMPLE_ECHO)
