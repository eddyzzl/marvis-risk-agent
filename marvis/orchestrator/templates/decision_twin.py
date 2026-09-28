from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import ToolRef


HISTORICAL_DECISION_REPLAY = WorkflowTemplate(
    id="historical_decision_replay",
    title="历史决策回放",
    goal_patterns=("历史决策回放", "决策孪生回放", "historical decision replay"),
    slots=(
        SlotSpec(
            "replay_contract",
            True,
            "user",
            "Explicit source, time, package and optional economic assumptions",
        ),
        SlotSpec("proposal_hash", True, "task_context", "Authenticated proposal hash"),
    ),
    steps=(
        StepTemplate(
            title="确认历史口径并回放冻结决策包",
            tool_ref=ToolRef("decision_twin", "replay_history"),
            inputs_template={
                "contract": "{slot:replay_contract}",
                "proposal_hash": "{slot:proposal_hash}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "artifact_id"}),),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
)

HISTORICAL_OUTCOME_RECONCILIATION = WorkflowTemplate(
    id="historical_outcome_reconciliation",
    title="历史决策现金流对账",
    goal_patterns=("历史决策现金流对账", "historical outcome reconciliation"),
    slots=(
        SlotSpec(
            "reconciliation_contract",
            True,
            "user",
            "Replay receipt and exact imported mature outcome source",
        ),
        SlotSpec(
            "proposal_hash", True, "user", "Confirmed reconciliation contract hash"
        ),
    ),
    steps=(
        StepTemplate(
            title="确认现金流口径并对账已观察的批准人群",
            tool_ref=ToolRef("decision_twin", "reconcile_history"),
            inputs_template={
                "contract": "{slot:reconciliation_contract}",
                "proposal_hash": "{slot:proposal_hash}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "artifact_id"}),),
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
)
