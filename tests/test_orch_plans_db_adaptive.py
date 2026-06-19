from marvis.db import PlanRepository, connect, init_db
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef


def test_plan_repository_migrates_adaptive_columns_on_existing_tables(tmp_path):
    db_path = tmp_path / "app.sqlite"
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE plans (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                source TEXT NOT NULL,
                template_id TEXT,
                autonomy_level INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE plan_steps (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                title TEXT NOT NULL,
                tool_plugin TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_version TEXT,
                inputs_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                post_checks_json TEXT NOT NULL,
                needs_confirmation INTEGER NOT NULL,
                sub_agent_scope TEXT,
                granted_tools_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL,
                sub_agent_id TEXT,
                output_ref TEXT,
                review_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    init_db(db_path)

    with connect(db_path) as conn:
        plan_columns = {row[1] for row in conn.execute("PRAGMA table_info(plans)")}
        step_columns = {row[1] for row in conn.execute("PRAGMA table_info(plan_steps)")}
    assert {"novel_mode", "tier", "replan_count"} <= plan_columns
    assert "decision_point" in step_columns


def test_plan_repository_adaptive_fields_round_trip(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.novel_mode = "explore"
    plan.tier = "autonomous"
    plan.replan_count = 2
    plan.steps[0].decision_point = True

    repo.create_plan(plan)
    loaded = repo.load_plan("plan-1")

    assert loaded.novel_mode == "explore"
    assert loaded.tier == "autonomous"
    assert loaded.replan_count == 2
    assert loaded.steps[0].decision_point is True


def test_plan_repository_replaces_remaining_steps_and_increments_replan_count(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.steps[0].status = StepStatus.DONE
    repo.create_plan(plan)
    repo.store_step_output("step-1", {"echoed": "done"})
    new_plan = _plan(
        _step("step-1", 0, status=StepStatus.DONE),
        _step("step-3", 1, title="Repaired"),
    )
    new_plan.tier = "autonomous"
    new_plan.novel_mode = "explore"

    repo.replace_remaining_steps("plan-1", new_plan)

    loaded = repo.load_plan("plan-1")
    assert [step.id for step in loaded.steps] == ["step-1", "step-3"]
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.replan_count == 1
    assert loaded.tier == "autonomous"
    assert repo.load_step_output("step-1") == {"echoed": "done"}


def test_plan_repository_appends_steps_and_lists_recent_failed_refs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.steps[1].status = StepStatus.FAILED
    repo.create_plan(plan)

    repo.append_steps("plan-1", [_step("step-3", 99, title="Next")])
    loaded = repo.load_plan("plan-1")

    assert [(step.id, step.index) for step in loaded.steps] == [
        ("step-1", 0),
        ("step-2", 1),
        ("step-3", 2),
    ]
    assert repo.recent_failed_tool_refs("plan-1", limit=4) == ["_sample.echo"]


def _plan(*steps: PlanStep) -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="run adaptive workflow",
        source="generated",
        template_id=None,
        steps=list(steps) or [_step("step-1", 0), _step("step-2", 1)],
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
    )


def _step(
    step_id: str,
    index: int,
    *,
    title: str | None = None,
    status: StepStatus = StepStatus.PENDING,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=index,
        title=title or step_id,
        tool_ref=ToolRef("_sample", "echo"),
        inputs={"message": "hi"},
        depends_on=[],
        post_checks=[],
        status=status,
    )
