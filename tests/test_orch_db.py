import json

import pytest

from marvis.db import PlanRepository, connect, init_db
import marvis.repositories.plans as plan_repo_module
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
from marvis.state_machine import ConflictError


def test_plan_repository_is_reexported_from_db_for_compatibility():
    assert PlanRepository is plan_repo_module.PlanRepository


def _plan(*, success_criteria=None) -> Plan:
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
        success_criteria=list(success_criteria or []),
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


def test_plan_repository_round_trips_plan_success_criteria(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan(success_criteria=[{"metric": "oot_ks", "min": 0.3331}])

    repo.create_plan(plan)
    loaded = repo.load_plan("plan-1")

    assert loaded.success_criteria == [{"metric": "oot_ks", "min": 0.3331}]


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


def test_plan_repository_confirm_step_rejects_non_awaiting_step(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    with pytest.raises(ConflictError, match="step is not awaiting confirmation"):
        repo.confirm_step("step-1")

    assert repo.is_step_confirmed("step-1") is False


def test_plan_repository_stores_and_loads_step_output(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})

    assert output_ref == "metrics:step-1:v1"
    assert repo.load_step_output("step-1") == {"echoed": "hi"}
    assert repo.load_step_evidence("step-1")["output_ref"] == output_ref
    assert repo.load_step_evidence("step-1")["schema_version"] == "evidence.v1"


def test_plan_repository_versions_step_outputs_and_loads_latest_by_default(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    first_ref = repo.store_step_output("step-1", {"echoed": "first"})
    second_ref = repo.store_step_output("step-1", {"echoed": "second"})

    assert first_ref == "metrics:step-1:v1"
    assert second_ref == "metrics:step-1:v2"
    assert repo.load_step_output("step-1") == {"echoed": "second"}
    assert repo.load_step_output("step-1", version=1) == {"echoed": "first"}
    assert repo.load_step_output("step-1", version=2) == {"echoed": "second"}
    assert repo.load_step_evidence("step-1", version=1)["output_ref"] == first_ref
    assert repo.load_step_evidence("step-1", version=2)["output_ref"] == second_ref


def test_plan_repository_stores_step_evidence_envelope_metadata(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    output_ref = repo.store_step_output(
        "step-1",
        {"echoed": "hi"},
        evidence={
            "tool_name": "_sample.echo",
            "input_hash": "sha256:abc",
            "source_dataset_refs": ["dataset:raw"],
            "parent_output_refs": ["metrics:upstream:v1"],
            "random_seed": 42,
        },
    )

    assert repo.load_step_output("step-1") == {"echoed": "hi"}
    evidence = repo.load_step_evidence("step-1")
    assert evidence["output_ref"] == output_ref
    assert evidence["tool_name"] == "_sample.echo"
    assert evidence["input_hash"] == "sha256:abc"
    assert evidence["source_dataset_refs"] == ["dataset:raw"]
    assert evidence["parent_output_refs"] == ["metrics:upstream:v1"]
    assert evidence["random_seed"] == 42


def test_plan_repository_redacts_step_output_and_evidence_before_persisting(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    repo.store_step_output(
        "step-1",
        {
            "stdout": '"api_key": "abc123456"',
            "customer": {"phone": "13812345678", "note": "ok"},
            "authorization": "Bearer token123456789",
            "join_plan_id": "join_plan_e0a75291365a4bfe89e1751234567890de",
        },
        evidence={
            "tool_name": "_sample.secret",
            "input_summary": {
                "password": "hunter2secret",
                "rows": ["身份证 110101199003070019"],
            },
        },
    )

    output = repo.load_step_output("step-1")
    evidence = repo.load_step_evidence("step-1")
    assert "abc123456" not in json.dumps(output, ensure_ascii=False)
    assert output["customer"]["phone"] == "[REDACTED]"
    assert output["authorization"] == "Bearer [REDACTED_SECRET]"
    assert output["join_plan_id"] == "join_plan_e0a75291365a4bfe89e1751234567890de"
    assert evidence["input_summary"]["password"] == "[REDACTED]"
    assert "110101199003070019" not in json.dumps(evidence, ensure_ascii=False)
    assert evidence["persistence_redacted_count"] >= 4


def test_plan_repository_records_step_run_lifecycle(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "hi"},
    )
    running = repo.list_running_step_runs("plan-1")
    assert len(running) == 1
    assert running[0]["id"] == run_id
    assert running[0]["input"] == {"message": "hi"}
    repo.finish_step_run(
        run_id,
        status="succeeded",
        output_ref="metrics:step-1:v1",
        duration_ms=12,
        side_effects=["artifact:report"],
    )

    runs = repo.list_step_runs("step-1")
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["attempt"] == 1
    assert runs[0]["tool_ref"] == "_sample.echo"
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["input"] == {"message": "hi"}
    assert runs[0]["output_ref"] == "metrics:step-1:v1"
    assert runs[0]["duration_ms"] == 12
    assert runs[0]["side_effects"] == ["artifact:report"]


def test_plan_repository_reset_step_clears_stale_execution_state(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    plan = repo.load_plan("plan-1")
    step = plan.steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    repo.update_step(step)
    repo.confirm_step("step-1")

    step = repo.load_plan("plan-1").steps[0]
    step.status = StepStatus.FAILED
    step.output_ref = repo.store_step_output("step-1", {"echoed": "old"})
    step.review_verdicts = [
        ReviewVerdict("deterministic", False, ["old failure"], "2026-06-19T00:00:00+00:00")
    ]
    step.error = "old failure"
    repo.update_step(step)

    repo.reset_step("step-1")

    loaded = repo.load_plan("plan-1").steps[0]
    assert loaded.status == StepStatus.PENDING
    assert loaded.output_ref is None
    assert loaded.review_verdicts == []
    assert loaded.error is None
    assert repo.is_step_confirmed("step-1") is False


def test_plan_repository_retry_failed_step_resets_downstream_and_reopens_plan(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.status = PlanStatus.RUNNING
    plan.steps[0].status = StepStatus.AWAITING_CONFIRM
    plan.steps[0].sub_agent_id = "sub-1"
    plan.steps[0].output_ref = "metrics:step-1:v1"
    plan.steps[1].status = StepStatus.DONE
    plan.steps[1].output_ref = "metrics:step-2:v1"
    repo.create_plan(plan)
    repo.confirm_step("step-1")
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plans SET status = 'failed' WHERE id = 'plan-1'"
        )
        conn.execute(
            """
            UPDATE plan_steps
               SET status = 'failed',
                   review_json = ?,
                   error = 'old failure'
             WHERE id = 'step-1'
            """,
            (
                json.dumps(
                    [
                        {
                            "reviewer": "deterministic",
                            "passed": False,
                            "reasons": ["old failure"],
                            "at": "2026-06-19T00:00:00+00:00",
                        }
                    ],
                    ensure_ascii=False,
                ),
            ),
        )

    reset_step_ids = repo.retry_failed_step("plan-1", "step-1")

    loaded = repo.load_plan("plan-1")
    assert reset_step_ids == ["step-1", "step-2"]
    assert loaded.status == PlanStatus.RUNNING
    assert [step.status for step in loaded.steps] == [StepStatus.PENDING, StepStatus.PENDING]
    assert loaded.steps[0].sub_agent_id is None
    assert loaded.steps[0].output_ref is None
    assert loaded.steps[0].review_verdicts == []
    assert loaded.steps[0].error is None
    assert repo.is_step_confirmed("step-1") is False
    assert repo.list_audit(kind="plan.step.retry")[0]["detail"]["reset_step_ids"] == [
        "step-1",
        "step-2",
    ]
    assert repo.list_audit(kind="plan.step.retry")[0]["detail"]["inputs_replaced"] is False


def test_plan_repository_retry_failed_step_can_replace_target_inputs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.status = PlanStatus.FAILED
    plan.steps[0].status = StepStatus.FAILED
    plan.steps[1].status = StepStatus.DONE
    repo.create_plan(plan)

    reset_step_ids = repo.retry_failed_step(
        "plan-1",
        "step-1",
        inputs={"message": "retry with new cutoff"},
    )

    loaded = repo.load_plan("plan-1")
    assert reset_step_ids == ["step-1", "step-2"]
    assert loaded.steps[0].inputs == {"message": "retry with new cutoff"}
    assert loaded.steps[1].inputs == {"seconds": "$ref:step-1.output.seconds"}
    assert repo.list_audit(kind="plan.step.retry")[0]["detail"]["inputs_replaced"] is True


def test_plan_repository_retry_failed_step_rejects_non_failed_plan_or_step(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    with pytest.raises(ConflictError, match="plan is not failed"):
        repo.retry_failed_step("plan-1", "step-1")

    with connect(db_path) as conn:
        conn.execute("UPDATE plans SET status = 'failed' WHERE id = 'plan-1'")

    with pytest.raises(ConflictError, match="step is not failed"):
        repo.retry_failed_step("plan-1", "step-1")


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
        output_version_count = conn.execute("SELECT COUNT(*) FROM plan_step_output_versions").fetchone()[0]

    assert step_count == 0
    assert output_count == 0
    assert output_version_count == 0
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


def test_plan_repository_upsert_sub_agent_with_audit_rolls_back_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
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

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(plan_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.upsert_sub_agent_with_audit(
            sub,
            audit={
                "kind": "subagent.spawn",
                "target_ref": sub.id,
                "outcome": "succeeded",
            },
        )

    with pytest.raises(KeyError):
        repo.get_sub_agent("agent-1")


def test_plan_repository_set_sub_agent_status_with_audit_rolls_back_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
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

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(plan_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.set_sub_agent_status_with_audit(
            "agent-1",
            AgentStatus.RETURNED,
            result_ref="value:summary",
            audit={
                "kind": "subagent.run",
                "target_ref": "agent-1",
                "outcome": "succeeded",
            },
        )

    loaded = repo.get_sub_agent("agent-1")
    assert loaded.status == AgentStatus.SPAWNED
    assert loaded.result_ref is None
