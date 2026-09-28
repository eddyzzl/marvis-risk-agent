from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef

RISK_SOURCE_QUERY = WorkflowTemplate(
    id="risk_source_query",
    title="授权来源查询与恢复",
    goal_patterns=("授权来源查询", "征信来源查询", "KYC来源查询", "risk source query"),
    slots=(
        SlotSpec(
            "source_request_id",
            True,
            "task_context",
            "Server-issued source request ID with task/subject/purpose grant",
        ),
    ),
    steps=(
        StepTemplate(
            title="执行已授权查询或读取原查询结果",
            tool_ref=ToolRef("risk_context", "query_source"),
            inputs_template={"request_id": "{slot:source_request_id}"},
            depends_on_titles=(),
            post_checks=(
                PostCheck("nonempty", {"field": "artifact_id"}),
                PostCheck(
                    "one_of",
                    {"field": "schema_version", "values": ["risk_source.summary.v1"]},
                ),
            ),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
)
