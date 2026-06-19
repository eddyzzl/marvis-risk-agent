import pytest

from marvis.db import PlanRepository, connect, init_db
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    PlanStep,
    PostCheck,
    ReviewVerdict,
    StepStatus,
    SubAgent,
)
from marvis.orchestrator.errors import IllegalPlanTransition, PlanNotFoundError
from marvis.plugins.manifest import ToolRef


def _plan() -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="run sample workflow",
        source="template",
        template_id="sample.echo",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        created_at="2026-06-19T00:00:00+00:00",
        updated_at="2026-06-19T00:00:00+00:00",
        steps=[
            PlanStep(
                id="step-1",
                plan_id="plan-1",
                index=0,
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs={"message": "hi"},
                depends_on=[],
                post_checks=[PostCheck("schema", {"required": ["echoed"]})],
                needs_confirmation=True,
                granted_tools=[ToolRef("_sample", "echo")],
            ),
            PlanStep(
                id="step-2",
                plan_id="plan-1",
                index=1,
                title="Sleep",
                tool_ref=ToolRef("_sample", "sleep", "0.1.0"),
                inputs={"seconds": "$ref:step-1.output.seconds"},
                depends_on=["step-1"],
                post_checks=[],
            ),
        ],
    )


def test_plan_repository_create_and_load_round_trips_plan(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()

    repo.create_plan(plan)
    loaded = repo.load_plan("plan-1")

    assert loaded == plan
    audits = repo.list_audit(kind="plan.create")
    assert audits[0]["target_ref"] == "plan-1"


def test_plan_repository_confirm_plan_uses_state_machine(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    repo.confirm_plan("plan-1")
    assert repo.load_plan("plan-1").status == PlanStatus.CONFIRMED

    with pytest.raises(IllegalPlanTransition):
        repo.confirm_plan("plan-1")


def test_plan_repository_updates_step_and_confirmation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    plan = repo.load_plan("plan-1")
    step = plan.steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    step.output_ref = "value:step-1"
    step.review_verdicts = [
        ReviewVerdict("deterministic", True, [], "2026-06-19T00:00:00+00:00")
    ]

    repo.update_step(step)
    repo.confirm_step("step-1")
    loaded = repo.load_plan("plan-1").steps[0]

    assert loaded.status == StepStatus.AWAITING_CONFIRM
    assert loaded.output_ref == "value:step-1"
    assert loaded.review_verdicts[0].reviewer == "deterministic"
    assert repo.is_step_confirmed("step-1") is True


def test_plan_repository_stores_and_loads_step_output(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})

    assert output_ref == "metrics:step-1"
    assert repo.load_step_output("step-1") == {"echoed": "hi"}


def test_plan_repository_fk_cascades_steps_and_outputs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    repo.store_step_output("step-1", {"echoed": "hi"})

    with connect(db_path) as conn:
        conn.execute("DELETE FROM plans WHERE id = ?", ("plan-1",))
        step_count = conn.execute("SELECT COUNT(*) FROM plan_steps").fetchone()[0]
        output_count = conn.execute("SELECT COUNT(*) FROM plan_step_outputs").fetchone()[0]

    assert step_count == 0
    assert output_count == 0
    with pytest.raises(PlanNotFoundError):
        repo.load_plan("plan-1")


def test_plan_repository_upserts_sub_agents(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    sub = SubAgent(
        id="agent-1",
        parent_task_id="task-1",
        parent_step_id="step-1",
        scope="summarize safely",
        granted_tools=[ToolRef("_sample", "echo")],
        context_budget=2048,
    )

    repo.upsert_sub_agent(sub)
    repo.set_sub_agent_status("agent-1", AgentStatus.RETURNED, result_ref="value:summary")

    loaded = repo.get_sub_agent("agent-1")
    assert loaded.status == AgentStatus.RETURNED
    assert loaded.result_ref == "value:summary"
