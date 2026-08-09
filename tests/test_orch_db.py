import json
import sqlite3

import pytest

from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import PlanRepository, connect, init_db
import marvis.db_schema as db_schema_module
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
    plan_fingerprint,
    plan_step_confirmation_fingerprint,
)
from marvis.orchestrator.errors import IllegalPlanTransition, PlanNotFoundError
from marvis.orchestrator.evidence import payload_hash
from marvis.plugins.manifest import ToolRef
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.tasks import TaskRepository
from marvis.state_machine import ConflictError
from tests.schema_fixture_support import install_v1_plan_step_runs_predecessor


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


def _seed_atomic_transaction_task(db_path) -> None:
    init_db(db_path)
    created_at = "2026-08-04T00:00:00+00:00"
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'atomic transaction', 'v1', 'tester', '/tmp/source',
                      'created', 'created', ?, ?)
            """,
            ("task-1", created_at, created_at),
        )


def _seed_atomic_transaction_dataset(db_path) -> str:
    dataset_hash = "a" * 64
    created_at = "2026-08-04T00:00:00+00:00"
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count,
                columns_json, has_target, target_col, created_at, content_hash
            ) VALUES (
                'dataset-1', 'task-1', 'analysis', '/tmp/dataset-1.parquet',
                'parquet', 10, ?, 1, 'bad', ?, ?
            )
            """,
            (
                json.dumps([{"name": "bad"}, {"name": "score"}]),
                created_at,
                dataset_hash,
            ),
        )
    return dataset_hash


def _initial_workspace_draft(dataset_hash: str) -> DataWorkspaceDraft:
    return DataWorkspaceDraft(
        active_dataset_id="dataset-1",
        active_dataset_content_hash=dataset_hash,
        selected_field="bad",
        semantic_mapping=DataSemanticMapping(
            target_col="bad",
            field_roles={"bad": "target"},
        ),
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


def test_create_plan_callback_failure_rolls_back_plan_steps_audit_and_message(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    _seed_atomic_transaction_task(db_path)
    plan_repo = PlanRepository(db_path)
    task_repo = TaskRepository(db_path)

    def write_message_then_fail(conn):
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="assistant",
            stage="chat",
            content="plan overview",
            metadata={"intent": "plan_overview"},
        )
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        plan_repo.create_plan(_plan(), on_connection=write_message_then_fail)

    with connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("plans", "plan_steps", "audit", "agent_messages")
        }
    assert counts == {
        "plans": 0,
        "plan_steps": 0,
        "audit": 0,
        "agent_messages": 0,
    }


def test_create_plan_commits_plan_and_callback_message_together(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_atomic_transaction_task(db_path)
    plan_repo = PlanRepository(db_path)
    task_repo = TaskRepository(db_path)

    def write_overview(conn):
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="assistant",
            stage="chat",
            content="plan overview",
            metadata={"intent": "plan_overview"},
        )

    plan_repo.create_plan(_plan(), on_connection=write_overview)

    assert plan_repo.load_plan("plan-1") == _plan()
    assert [message["metadata"]["intent"] for message in task_repo.list_agent_messages(
        "task-1"
    )] == ["plan_overview"]
    with connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("plans", "plan_steps", "audit", "agent_messages")
        }
    assert counts == {
        "plans": 1,
        "plan_steps": 2,
        "audit": 1,
        "agent_messages": 1,
    }


def test_initial_binding_callback_failure_rolls_back_workspace_audit_and_receipts(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    _seed_atomic_transaction_task(db_path)
    dataset_hash = _seed_atomic_transaction_dataset(db_path)
    workspace_repo = DataWorkspaceRepository(db_path)
    task_repo = TaskRepository(db_path)

    def write_messages_then_fail(conn):
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="user",
            stage="chat",
            content="confirm roles",
            metadata={"intent": "confirm_roles"},
        )
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="assistant",
            stage="chat",
            content="semantic authorization receipt",
            metadata={"intent": "c1_semantic_authorization"},
        )
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        workspace_repo.save_initial_binding(
            "task-1",
            _initial_workspace_draft(dataset_hash),
            expected_revision=0,
            on_connection=write_messages_then_fail,
        )

    assert workspace_repo.get_or_default("task-1").revision == 0
    with connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("data_workspaces", "audit", "agent_messages")
        }
    assert counts == {
        "data_workspaces": 0,
        "audit": 0,
        "agent_messages": 0,
    }


def test_initial_binding_commits_workspace_audit_and_receipts_together(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_atomic_transaction_task(db_path)
    dataset_hash = _seed_atomic_transaction_dataset(db_path)
    workspace_repo = DataWorkspaceRepository(db_path)
    task_repo = TaskRepository(db_path)

    def write_confirmation_and_receipt(conn):
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="user",
            stage="chat",
            content="confirm roles",
            metadata={"intent": "confirm_roles"},
        )
        task_repo.add_agent_message_on_connection(
            conn,
            "task-1",
            role="assistant",
            stage="chat",
            content="semantic authorization receipt",
            metadata={"intent": "c1_semantic_authorization"},
        )

    bound = workspace_repo.save_initial_binding(
        "task-1",
        _initial_workspace_draft(dataset_hash),
        expected_revision=0,
        on_connection=write_confirmation_and_receipt,
    )

    assert workspace_repo.get_or_default("task-1") == bound
    assert [message["metadata"]["intent"] for message in task_repo.list_agent_messages(
        "task-1"
    )] == ["confirm_roles", "c1_semantic_authorization"]
    with connect(db_path) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("data_workspaces", "audit", "agent_messages")
        }
    assert counts == {
        "data_workspaces": 1,
        "audit": 1,
        "agent_messages": 2,
    }


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


def test_plan_repository_confirm_plan_binds_reviewed_snapshot(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    reviewed = repo.load_plan("plan-1")

    repo.confirm_plan(
        "plan-1",
        expected_plan_fingerprint=plan_fingerprint(reviewed),
        expected_plan_revision=reviewed.replan_count,
        expected_plan_status=reviewed.status,
    )

    assert repo.load_plan("plan-1").status == PlanStatus.CONFIRMED


@pytest.mark.parametrize(
    ("expectation", "message"),
    [
        ({"expected_plan_revision": 1}, "revision changed"),
        ({"expected_plan_status": PlanStatus.RUNNING}, "status changed"),
    ],
)
def test_plan_repository_confirm_plan_rejects_stale_metadata(
    tmp_path,
    expectation,
    message,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    with pytest.raises(ConflictError, match=message):
        repo.confirm_plan("plan-1", **expectation)

    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED


def test_plan_repository_confirm_plan_rejects_same_revision_snapshot_mutation(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    reviewed = repo.load_plan("plan-1")
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plans SET goal = ? WHERE id = ?",
            ("mutated after review", "plan-1"),
        )

    with pytest.raises(ConflictError, match="fingerprint changed"):
        repo.confirm_plan(
            "plan-1",
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_plan_revision=reviewed.replan_count,
            expected_plan_status=reviewed.status,
        )

    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED


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


def test_plan_repository_confirm_step_binds_plan_and_step_snapshots(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    step = repo.load_plan("plan-1").steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    repo.update_step(step)
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[0]

    repo.confirm_step(
        gate.id,
        expected_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        expected_plan_fingerprint=plan_fingerprint(reviewed),
        expected_plan_revision=reviewed.replan_count,
        expected_plan_status=reviewed.status,
    )

    assert repo.is_step_confirmed(gate.id) is True


def test_plan_repository_confirm_step_rejects_stale_step_snapshot(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    step = repo.load_plan("plan-1").steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    repo.update_step(step)
    reviewed_gate = repo.load_plan("plan-1").steps[0]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET inputs_json = ? WHERE id = ?",
            (json.dumps({"message": "mutated"}), reviewed_gate.id),
        )

    with pytest.raises(ConflictError, match="step .* fingerprint changed"):
        repo.confirm_step(
            reviewed_gate.id,
            expected_step_fingerprint=plan_step_confirmation_fingerprint(reviewed_gate),
        )

    assert repo.is_step_confirmed(reviewed_gate.id) is False


def test_plan_repository_confirm_step_with_inputs_rejects_stale_plan_snapshot(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    step = repo.load_plan("plan-1").steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    repo.update_step(step)
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[0]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET title = ? WHERE id = ?",
            ("Sibling mutated after review", "step-2"),
        )

    with pytest.raises(ConflictError, match="plan .* fingerprint changed"):
        repo.confirm_step_with_inputs(
            gate.id,
            input_updates={"message": "reviewed update"},
            expected_step_fingerprint=plan_step_confirmation_fingerprint(gate),
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_plan_revision=reviewed.replan_count,
            expected_plan_status=reviewed.status,
        )

    loaded_gate = repo.load_plan("plan-1").steps[0]
    assert loaded_gate.inputs == {"message": "hi"}
    assert repo.is_step_confirmed(gate.id) is False


def test_plan_repository_confirm_step_with_inputs_binds_reviewed_snapshot(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    step = repo.load_plan("plan-1").steps[0]
    step.status = StepStatus.AWAITING_CONFIRM
    repo.update_step(step)
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[0]

    repo.confirm_step_with_inputs(
        gate.id,
        input_updates={"reason": "reviewed"},
        expected_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        expected_plan_fingerprint=plan_fingerprint(reviewed),
        expected_plan_revision=reviewed.replan_count,
        expected_plan_status=reviewed.status,
    )

    loaded_gate = repo.load_plan("plan-1").steps[0]
    assert loaded_gate.inputs == {"message": "hi", "reason": "reviewed"}
    assert repo.is_step_confirmed(gate.id) is True


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


def _mark_step_running(repo: PlanRepository, step_id: str) -> None:
    # start_step_run only opens a run for a RUNNING step (matching the executor,
    # which sets RUNNING before starting the run). Move the step there first.
    plan = repo.load_plan("plan-1")
    step = next(s for s in plan.steps if s.id == step_id)
    step.status = StepStatus.RUNNING
    repo.update_step(step)


def _finish_bound_step(
    repo: PlanRepository,
    step_id: str,
    *,
    inputs: dict,
    output: dict,
) -> str:
    _mark_step_running(repo, step_id)
    plan = repo.load_plan("plan-1")
    step = next(item for item in plan.steps if item.id == step_id)
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id=step_id,
        tool_ref=step.tool_ref.label(),
        inputs=inputs,
    )
    step.status = StepStatus.CHECKING
    repo.update_step(step)
    output_ref = repo.store_step_output(
        step_id,
        output,
        evidence={"step_run_id": run_id},
    )
    repo.finish_step_run(run_id, status="succeeded", output_ref=output_ref)
    step.output_ref = output_ref
    repo.update_step(step)
    step.status = StepStatus.DONE
    repo.update_step(step)
    return output_ref


def test_plan_repository_records_step_run_lifecycle(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    _mark_step_running(repo, "step-1")

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
    assert running[0]["progress"] == {}
    assert repo.update_step_run_progress(
        run_id,
        {"kind": "model_tuning", "algorithm": "lgb", "trial": 3},
    ) is True
    running = repo.list_running_step_runs("plan-1")
    assert running[0]["progress"] == {
        "kind": "model_tuning",
        "algorithm": "lgb",
        "trial": 3,
    }
    assert running[0]["progress_updated_at"]
    output_ref = repo.store_step_output(
        "step-1",
        {"echoed": "hi"},
        evidence={"step_run_id": run_id},
    )
    repo.finish_step_run(
        run_id,
        status="succeeded",
        output_ref=output_ref,
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
    assert runs[0]["output_ref"] == output_ref
    assert runs[0]["duration_ms"] == 12
    assert runs[0]["side_effects"] == ["artifact:report"]
    assert runs[0]["progress"]["trial"] == 3
    assert repo.update_step_run_progress(run_id, {"trial": 4}) is False


@pytest.mark.parametrize(
    ("plan_id", "tool_ref"),
    [
        ("plan-other", "_sample.echo"),
        ("plan-1", "_sample.other"),
    ],
)
def test_step_run_rejects_wrong_plan_or_tool_binding(
    tmp_path,
    plan_id,
    tool_ref,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    _mark_step_running(repo, "step-1")

    with pytest.raises(ConflictError, match="plan/tool binding"):
        repo.start_step_run(
            plan_id=plan_id,
            step_id="step-1",
            tool_ref=tool_ref,
            inputs={"message": "hi"},
        )


def test_step_presentation_binding_authenticates_exact_run_input_and_artifact(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    inputs = {"message": "first invocation"}
    output = {
        "artifact": {
            "artifact_id": "artifact-first",
            "kind": "test_json",
            "content_hash": "a" * 64,
        }
    }
    output_ref = _finish_bound_step(
        repo,
        "step-1",
        inputs=inputs,
        output=output,
    )

    binding = repo.load_step_presentation_binding("step-1", output_ref)

    assert binding["plan_id"] == "plan-1"
    assert binding["task_id"] == "task-1"
    assert binding["inputs"] == inputs
    assert binding["output"] == output
    assert binding["evidence"]["artifact_bindings"] == [
        {
            "artifact_id": "artifact-first",
            "content_hash": "a" * 64,
            "kind": "test_json",
        }
    ]
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with connect(db_path) as conn:
            conn.execute(
                """
                UPDATE plan_step_runs
                   SET output_hash = ?
                 WHERE output_ref = ?
                """,
                ("sha256:" + "f" * 64, output_ref),
            )


def test_step_presentation_binding_rejects_complete_same_task_output_and_evidence_swap(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    first_ref = _finish_bound_step(
        repo,
        "step-1",
        inputs={"message": "first invocation"},
        output={
            "artifact": {
                "artifact_id": "artifact-first",
                "kind": "test_json",
                "content_hash": "a" * 64,
            }
        },
    )
    _finish_bound_step(
        repo,
        "step-2",
        inputs={"message": "second invocation"},
        output={
            "artifact": {
                "artifact_id": "artifact-second",
                "kind": "test_json",
                "content_hash": "b" * 64,
            }
        },
    )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with connect(db_path) as conn:
            second = conn.execute(
                """
                SELECT output_json, evidence_json
                  FROM plan_step_output_versions
                 WHERE step_id = 'step-2' AND version = 1
                """
            ).fetchone()
            conn.execute(
                """
                UPDATE plan_step_output_versions
                   SET output_json = ?, evidence_json = ?
                 WHERE step_id = 'step-1' AND version = 1
                """,
                (second["output_json"], second["evidence_json"]),
            )

    assert repo.load_step_presentation_binding("step-1", first_ref)[
        "output"
    ]["artifact"]["artifact_id"] == "artifact-first"


def test_canonical_step_output_rejects_unsigned_or_mismatched_invocation_receipt(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.steps[0].tool_ref = ToolRef(
        "modeling",
        "train_model_with_evidence_v2",
    )
    repo.create_plan(plan)
    _mark_step_running(repo, "step-1")
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="modeling.train_model_with_evidence_v2",
        inputs={"sample_design_ref": {"sample_id": "sample-a"}},
    )
    output = {"experiment_id": "experiment-b"}

    with pytest.raises(ConflictError, match="verified producer receipt"):
        repo.store_step_output(
            "step-1",
            output,
            evidence={"step_run_id": run_id},
        )
    with pytest.raises(ConflictError, match="does not match"):
        repo.store_step_output(
            "step-1",
            output,
            evidence={
                "step_run_id": run_id,
                "producer_invocation_id": run_id,
                "raw_output_hash": payload_hash({"experiment_id": "experiment-a"}),
                "canonical_binding_verified": True,
                "tool_version": "2.0.0",
                "manifest_hash": f"sha256:{'1' * 64}",
            },
        )

    with pytest.raises(ConflictError, match="verified producer receipt"):
        repo.store_step_output(
            "step-1",
            output,
            evidence={
                "step_run_id": run_id,
                "producer_invocation_id": run_id,
                "raw_output_hash": payload_hash(output),
                "canonical_binding_verified": True,
            },
        )

    assert repo.latest_step_output_ref("step-1") is None
    run = repo.list_step_runs("step-1")[0]
    assert run["status"] == "running"
    assert run["output_ref"] is None


def test_step_output_rejects_unregistered_result_dataset_binding(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    _mark_step_running(repo, "step-1")
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "hi"},
    )

    with pytest.raises(ConflictError, match="not registered"):
        repo.store_step_output(
            "step-1",
            {"result_dataset_id": "dataset-from-another-envelope"},
            evidence={"step_run_id": run_id},
        )

    assert repo.latest_step_output_ref("step-1") is None


def test_bound_step_result_receipt_is_immutable_and_plan_purge_cascades(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    _finish_bound_step(
        repo,
        "step-1",
        inputs={"message": "hi"},
        output={"echoed": "hi"},
    )

    with pytest.raises(sqlite3.IntegrityError, match="receipt is immutable"):
        with connect(db_path) as conn:
            conn.execute(
                """
                UPDATE plan_step_runs
                   SET invocation_id = 'late-forgery'
                 WHERE step_id = 'step-1'
                """
            )

    with pytest.raises(sqlite3.IntegrityError, match="tool receipt is immutable"):
        with connect(db_path) as conn:
            conn.execute(
                """
                UPDATE plan_step_runs
                   SET manifest_hash = ?
                 WHERE step_id = 'step-1'
                """,
                (f"sha256:{'f' * 64}",),
            )

    with connect(db_path) as conn:
        conn.execute("DELETE FROM plans WHERE id = 'plan-1'")
        assert conn.execute(
            "SELECT 1 FROM plan_step_output_versions WHERE step_id = 'step-1'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM plan_step_runs WHERE step_id = 'step-1'"
        ).fetchone() is None


def test_migration_008_adds_step_run_progress_columns_to_v7_database(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    with connect(db_path) as conn:
        install_v1_plan_step_runs_predecessor(conn)
        conn.execute("PRAGMA user_version = 7")

    init_db(db_path)

    with connect(db_path) as conn:
        columns = {
            row[1]: row[2]
            for row in conn.execute("PRAGMA table_info(plan_step_runs)").fetchall()
        }
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert columns["progress_json"] == "TEXT"
    assert columns["progress_updated_at"] == "TEXT"
    assert version == db_schema_module.SCHEMA_VERSION


def test_start_step_run_rejects_step_not_running(tmp_path):
    """A run may only be opened for a RUNNING step. Opening one against a DONE
    step (a stale/concurrent caller) raises ConflictError and writes no run row,
    so a finished step never grows a spurious 'running' run."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())
    plan = repo.load_plan("plan-1")
    step = plan.steps[0]
    step.status = StepStatus.DONE
    repo.update_step(step)

    with pytest.raises(ConflictError, match="expected running"):
        repo.start_step_run(
            plan_id="plan-1", step_id="step-1", tool_ref="_sample.echo", inputs={}
        )
    assert repo.list_step_runs("step-1") == []


def test_start_step_run_allowed_on_retry_after_step_returns_to_running(tmp_path):
    """The retry path is not broken: retry_failed_step resets a failed step to
    pending, the executor re-runs it through RUNNING, and start_step_run then
    opens attempt 2. Simulate that flow and assert the second run is accepted
    and numbered attempt 2."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_plan())

    _mark_step_running(repo, "step-1")
    run1 = repo.start_step_run(
        plan_id="plan-1", step_id="step-1", tool_ref="_sample.echo", inputs={}
    )
    repo.finish_step_run(run1, status="failed", error="boom", error_kind="RuntimeError")

    # Retry resets the step to pending (mimicking retry_failed_step), then the
    # executor advances it back to RUNNING before the next run. retry_failed_step
    # requires a FAILED plan+step; set both directly (this test targets the
    # step-run guard, not the plan state machine).
    plan = repo.load_plan("plan-1")
    step = plan.steps[0]
    step.status = StepStatus.FAILED
    repo.update_step(step)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plans SET status = ? WHERE id = ?",
            (PlanStatus.FAILED.value, "plan-1"),
        )
    repo.retry_failed_step("plan-1", "step-1")
    _mark_step_running(repo, "step-1")

    run2 = repo.start_step_run(
        plan_id="plan-1", step_id="step-1", tool_ref="_sample.echo", inputs={}
    )
    runs = {r["id"]: r for r in repo.list_step_runs("step-1")}
    assert runs[run2]["attempt"] == 2
    assert runs[run2]["status"] == "running"


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


def _reviewable_gate_adjustment_plan() -> Plan:
    plan = _plan()
    plan.status = PlanStatus.AWAITING_CONFIRM
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].output_ref = "metrics:step-1:v1"
    plan.steps[1].status = StepStatus.AWAITING_CONFIRM
    plan.steps[1].needs_confirmation = True
    plan.steps[1].output_ref = "metrics:step-2:v1"
    return plan


def test_plan_repository_gate_adjustment_resets_batch_from_reviewed_snapshot(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_reviewable_gate_adjustment_plan())
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[1]

    repo.apply_gate_adjustment(
        "plan-1",
        target_step_id=gate.id,
        reset_step_ids=["step-1", "step-2"],
        replacement_inputs_by_step={
            "step-1": {"message": "revised"},
        },
        expected_plan_status=reviewed.status,
        expected_plan_revision=reviewed.replan_count,
        expected_plan_fingerprint=plan_fingerprint(reviewed),
        expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
    )

    loaded = repo.load_plan("plan-1")
    assert [step.status for step in loaded.steps] == [
        StepStatus.PENDING,
        StepStatus.PENDING,
    ]
    assert loaded.steps[0].inputs == {"message": "revised"}
    assert loaded.steps[1].inputs == {"seconds": "$ref:step-1.output.seconds"}
    assert all(step.output_ref is None for step in loaded.steps)
    assert sorted(
        row["target_ref"] for row in repo.list_audit(kind="plan.step.reset")
    ) == ["step-1", "step-2"]


def test_plan_repository_gate_adjustment_rejects_cancelled_reviewed_snapshot(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_reviewable_gate_adjustment_plan())
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[1]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plans SET status = ? WHERE id = ?",
            (PlanStatus.CANCELLED.value, reviewed.id),
        )

    with pytest.raises(ConflictError, match="status changed"):
        repo.apply_gate_adjustment(
            reviewed.id,
            target_step_id=gate.id,
            reset_step_ids=["step-1", "step-2"],
            replacement_inputs_by_step={"step-1": {"message": "revised"}},
            expected_plan_status=reviewed.status,
            expected_plan_revision=reviewed.replan_count,
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )

    loaded = repo.load_plan(reviewed.id)
    assert loaded.status is PlanStatus.CANCELLED
    assert [step.status for step in loaded.steps] == [
        StepStatus.DONE,
        StepStatus.AWAITING_CONFIRM,
    ]
    assert loaded.steps[0].inputs == {"message": "hi"}
    assert repo.list_audit(kind="plan.step.reset") == []


def test_plan_repository_gate_adjustment_rejects_same_revision_plan_mutation(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_reviewable_gate_adjustment_plan())
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[1]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET title = ? WHERE id = ?",
            ("mutated after review", "step-1"),
        )

    with pytest.raises(ConflictError, match="plan .* fingerprint changed"):
        repo.apply_gate_adjustment(
            reviewed.id,
            target_step_id=gate.id,
            reset_step_ids=["step-1", "step-2"],
            replacement_inputs_by_step={"step-1": {"message": "revised"}},
            expected_plan_status=reviewed.status,
            expected_plan_revision=reviewed.replan_count,
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )

    loaded = repo.load_plan(reviewed.id)
    assert [step.status for step in loaded.steps] == [
        StepStatus.DONE,
        StepStatus.AWAITING_CONFIRM,
    ]
    assert loaded.steps[0].inputs == {"message": "hi"}
    assert repo.list_audit(kind="plan.step.reset") == []


def test_plan_repository_gate_adjustment_rejects_stale_target_confirmation(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_reviewable_gate_adjustment_plan())
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[1]
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET confirmed = 1 WHERE id = ?",
            (gate.id,),
        )

    with pytest.raises(ConflictError, match="step .* fingerprint changed"):
        repo.apply_gate_adjustment(
            reviewed.id,
            target_step_id=gate.id,
            reset_step_ids=["step-1", "step-2"],
            replacement_inputs_by_step={"step-1": {"message": "revised"}},
            expected_plan_status=reviewed.status,
            expected_plan_revision=reviewed.replan_count,
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )

    loaded = repo.load_plan(reviewed.id)
    assert [step.status for step in loaded.steps] == [
        StepStatus.DONE,
        StepStatus.AWAITING_CONFIRM,
    ]
    assert loaded.steps[0].inputs == {"message": "hi"}
    assert repo.is_step_confirmed(gate.id) is True
    assert repo.list_audit(kind="plan.step.reset") == []


def test_plan_repository_gate_adjustment_rolls_back_when_later_reset_fails(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_reviewable_gate_adjustment_plan())
    reviewed = repo.load_plan("plan-1")
    gate = reviewed.steps[1]
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_second_gate_adjustment_reset
            BEFORE UPDATE ON plan_steps
            WHEN OLD.id = 'step-2' AND NEW.status = 'pending'
            BEGIN
                SELECT RAISE(ABORT, 'forced second reset failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced second reset failure"):
        repo.apply_gate_adjustment(
            reviewed.id,
            target_step_id=gate.id,
            reset_step_ids=["step-1", "step-2"],
            replacement_inputs_by_step={"step-1": {"message": "revised"}},
            expected_plan_status=reviewed.status,
            expected_plan_revision=reviewed.replan_count,
            expected_plan_fingerprint=plan_fingerprint(reviewed),
            expected_target_step_fingerprint=plan_step_confirmation_fingerprint(gate),
        )

    loaded = repo.load_plan(reviewed.id)
    assert [step.status for step in loaded.steps] == [
        StepStatus.DONE,
        StepStatus.AWAITING_CONFIRM,
    ]
    assert loaded.steps[0].inputs == {"message": "hi"}
    assert [step.output_ref for step in loaded.steps] == [
        "metrics:step-1:v1",
        "metrics:step-2:v1",
    ]
    assert repo.list_audit(kind="plan.step.reset") == []


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


def test_plan_repository_retry_cancelled_step_reopens_same_plan(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _plan()
    plan.status = PlanStatus.CANCELLED
    plan.steps[0].status = StepStatus.DONE
    plan.steps[0].output_ref = "metrics:step-1:v1"
    plan.steps[1].status = StepStatus.FAILED
    plan.steps[1].error = "用户已停止当前动作"
    repo.create_plan(plan)

    reset_step_ids = repo.retry_failed_step("plan-1", "step-2")

    loaded = repo.load_plan("plan-1")
    assert reset_step_ids == ["step-2"]
    assert loaded.status == PlanStatus.RUNNING
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].output_ref == "metrics:step-1:v1"
    assert loaded.steps[1].status == StepStatus.PENDING
    assert loaded.steps[1].error is None
    audit = repo.list_audit(kind="plan.step.retry")[0]
    assert audit["detail"]["from_plan_status"] == PlanStatus.CANCELLED.value


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
