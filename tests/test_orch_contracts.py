import pytest

from marvis.orchestrator.contracts import (
    AgentStatus,
    OutputRef,
    Plan,
    PlanStatus,
    PlanStep,
    PostCheck,
    ReviewVerdict,
    StepStatus,
    SubAgent,
    format_output_ref,
    parse_output_ref,
    plan_from_dict,
    plan_to_dict,
)
from marvis.orchestrator.harness_state import (
    IllegalPlanTransition,
    IllegalStepTransition,
    assert_plan_transition,
    assert_step_transition,
)
from marvis.plugins.manifest import ToolRef


def _plan() -> Plan:
    step = PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="Echo",
        tool_ref=ToolRef("_sample", "echo", "0.1.0"),
        inputs={"message": "hi"},
        depends_on=[],
        post_checks=[PostCheck(kind="schema", spec={"required": ["echoed"]})],
        needs_confirmation=True,
        decision_point=True,
        sub_agent_scope="summarize",
        granted_tools=[ToolRef("_sample", "echo")],
        status=StepStatus.AWAITING_CONFIRM,
        sub_agent_id="agent-1",
        output_ref="value:echo",
        review_verdicts=[
            ReviewVerdict(
                reviewer="deterministic",
                passed=True,
                reasons=[],
                at="2026-06-19T00:00:00+00:00",
            )
        ],
    )
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="run echo",
        source="template",
        template_id="sample.echo",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        created_at="2026-06-19T00:00:00+00:00",
        updated_at="2026-06-19T00:01:00+00:00",
        novel_mode="explore",
        tier="autonomous",
        replan_count=2,
    )


def test_plan_contract_round_trips_to_json_safe_dict():
    plan = _plan()

    payload = plan_to_dict(plan)
    reparsed = plan_from_dict(payload)

    assert payload["status"] == "validated"
    assert payload["novel_mode"] == "explore"
    assert payload["tier"] == "autonomous"
    assert payload["replan_count"] == 2
    assert payload["steps"][0]["status"] == "awaiting_confirm"
    assert payload["steps"][0]["decision_point"] is True
    assert payload["steps"][0]["tool_ref"] == {
        "plugin": "_sample",
        "tool": "echo",
        "version": "0.1.0",
    }
    assert reparsed == plan


def test_adaptive_contract_defaults_do_not_change_plain_dag_behavior():
    step = PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="Echo",
        tool_ref=ToolRef("_sample", "echo"),
        inputs={},
        depends_on=[],
        post_checks=[],
    )
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="run echo",
        source="template",
        template_id="sample_echo",
        steps=[step],
        autonomy_level=1,
    )

    assert step.decision_point is False
    assert plan.novel_mode == "plan_ahead"
    assert plan.tier == "balanced"
    assert plan.replan_count == 0


def test_subagent_contract_defaults_and_tool_refs():
    subagent = SubAgent(
        id="agent-1",
        parent_task_id="task-1",
        parent_step_id="step-1",
        scope="Only summarize the echo output",
        granted_tools=[ToolRef("_sample", "echo")],
        context_budget=4096,
    )

    assert subagent.status == AgentStatus.SPAWNED
    assert subagent.granted_tools[0].label() == "_sample.echo"


def test_output_ref_parse_and_format():
    parsed = parse_output_ref("metrics:step-1")

    assert parsed == OutputRef(kind="metrics", value="step-1")
    assert format_output_ref("artifact", "reports/final.docx") == "artifact:reports/final.docx"


@pytest.mark.parametrize("raw", ["", "metrics", "unknown:x", "dataset:"])
def test_output_ref_rejects_invalid_values(raw):
    with pytest.raises(ValueError):
        parse_output_ref(raw)


def test_plan_state_machine_accepts_and_rejects_transitions():
    assert_plan_transition(PlanStatus.DRAFT, PlanStatus.VALIDATED)
    assert_plan_transition(PlanStatus.AWAITING_CONFIRM, PlanStatus.RUNNING)

    with pytest.raises(IllegalPlanTransition) as exc_info:
        assert_plan_transition(PlanStatus.DONE, PlanStatus.RUNNING)
    assert exc_info.value.current == PlanStatus.DONE
    assert exc_info.value.target == PlanStatus.RUNNING


def test_step_state_machine_accepts_retry_and_confirmation_flow():
    assert_step_transition(StepStatus.PENDING, StepStatus.AWAITING_CONFIRM)
    assert_step_transition(StepStatus.AWAITING_CONFIRM, StepStatus.RUNNING)
    assert_step_transition(StepStatus.FAILED, StepStatus.PENDING)

    with pytest.raises(IllegalStepTransition) as exc_info:
        assert_step_transition(StepStatus.DONE, StepStatus.RUNNING)
    assert exc_info.value.current == StepStatus.DONE
    assert exc_info.value.target == StepStatus.RUNNING
