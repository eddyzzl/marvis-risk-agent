from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef

EVENT_FEATURE_REPLAY = WorkflowTemplate(
    id="event_feature_replay",
    title="事件窗口特征回放",
    goal_patterns=("事件窗口特征回放", "event feature replay"),
    slots=(
        SlotSpec(
            "event_request_id", True, "task_context", "Server-issued event request ID"
        ),
        SlotSpec(
            "event_proposal_hash",
            True,
            "task_context",
            "Authenticated event proposal hash",
        ),
        SlotSpec(
            "event_contract",
            True,
            "user",
            "Source, knowledge cutoff, window and relation contract",
        ),
    ),
    steps=(
        StepTemplate(
            title="确认事件时间窗与关系口径并冻结特征",
            tool_ref=ToolRef("risk_context", "replay_events"),
            inputs_template={
                "request_id": "{slot:event_request_id}",
                "proposal_hash": "{slot:event_proposal_hash}",
                "contract": "{slot:event_contract}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "artifact_id"}),),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
)
