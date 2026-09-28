"""Collection effects remain ordinary governed Workflow templates."""

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef


def _workflow(operation, title, goal):
    return WorkflowTemplate(
        id="collection_" + operation,
        title=title,
        goal_patterns=(goal, "collection " + operation),
        slots=tuple(
            SlotSpec("collection_" + name, True, "task_context", label)
            for name, label in (
                ("batch_id", "Frozen native collection batch"),
                ("request_hash", "Authenticated proposal input hash"),
                ("preview_hash", "Reviewed deterministic preview hash"),
            )
        ),
        steps=(
            StepTemplate(
                title=title,
                tool_ref=ToolRef("collection", operation),
                inputs_template={
                    name: "{slot:collection_" + name + "}"
                    for name in ("batch_id", "request_hash", "preview_hash")
                },
                depends_on_titles=(),
                post_checks=(PostCheck("nonempty", {"field": "artifact_id"}),),
                needs_confirmation=True,
            ),
        ),
        default_autonomy=1,
    )


COLLECTION_QUEUE = _workflow("queue_batch", "确认催收参考排队", "催收参考排队")
COLLECTION_EXECUTE = _workflow("execute_reference", "确认催收参考执行", "催收参考执行")
COLLECTION_CANCEL = _workflow("cancel_batch", "取消催收参考批次", "取消催收参考批次")
