import json
import time

import pytest

from marvis.runtime_observations import observe_runtime, confirmation_wait, end_confirmation_wait
from marvis.runtime_waits import refresh_confirmation_wait
from marvis.orchestrator.eval.runtime_process import read_backend_events, backend_timing, process_observation


def _report_task(tmp_path):
    from marvis.db import TaskRepository, init_db, connect
    from marvis.domain import TaskCreate
    from test_report_draft_persistence import _complete_draft

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = TaskRepository(db)
    task = repo.create_task(TaskCreate(model_name="fixture", model_version="1", validator="fixture", source_dir=str(tmp_path)))
    with connect(db) as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version=1, status='writing_artifacts' WHERE id=?", (task.id,))
    message = _complete_draft(repo, task.id)
    return repo, task, message


def _timing(path, scope="report_confirmation_wait"):
    return backend_timing(read_backend_events(path))["scopes"][scope]


def test_report_wait_starts_after_job_completion_and_ends_when_processing_resumes(tmp_path):
    repo, task, _ = _report_task(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        repo.mark_job_running(job)
        refresh_confirmation_wait(repo, task.id)
        assert not any(row["event"] == "waiting" for row in read_backend_events(path))
        repo.finish_job(job, status="succeeded")
        refresh_confirmation_wait(repo, task.id)
        assert sum(row["event"] == "waiting" for row in read_backend_events(path)) == 1
        repo.start_job(task.id, "report")
    timing = _timing(path)
    assert timing["complete"] and timing["measured_intervals"] == 1
    assert timing["intervals"][0]["outcome"] == "resumed"
    assert timing["known_busy_duration_ns"] > 0
    assert task.id not in path.read_text()


def test_unanswered_confirmation_is_right_censored_and_never_silently_zero(tmp_path):
    repo, task, _ = _report_task(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(repo, task.id)
    timing = _timing(path)
    assert not timing["complete"] and timing["right_censored_intervals"] == 1
    assert timing["known_busy_duration_ns"] > 0
    assert timing["intervals"][0]["outcome"] == "censored"
    result = process_observation([], [], read_backend_events(path), plan_states=[])
    assert result["formal_timing_complete"] is False and "human_wait" in result["unmeasured"]


def test_saved_draft_version_supersedes_wait_binding_without_leaking_values(tmp_path):
    repo, task, message = _report_task(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(repo, task.id)
        repo.save_agent_report_draft(task.id, message_id=message["id"], edit_revision=0,
                                     expected_revision=0, values={"TEXT:model_scope": "PRIVATE_EDIT"})
        end_confirmation_wait(task.id)
    timing = _timing(path)
    assert timing["complete"] and timing["measured_intervals"] == 2
    assert [row["outcome"] for row in timing["intervals"]] == ["superseded", "resumed"]
    assert len({row["binding_sha256"] for row in timing["intervals"]}) == 2
    assert "PRIVATE_EDIT" not in path.read_text() and message["id"] not in path.read_text()


@pytest.mark.parametrize("damage", ["incomplete", "streaming", "not_confirmable", "stale", "failed"])
def test_unconfirmable_state_does_not_start_a_wait(tmp_path, damage):
    from marvis.db import connect

    repo, task, message = _report_task(tmp_path)
    metadata = dict(message["metadata"])
    if damage == "incomplete":
        metadata["draft_values"] = {}
    elif damage == "streaming":
        metadata["streaming"] = True
    elif damage == "not_confirmable":
        metadata["confirmable"] = False
    elif damage == "stale":
        metadata["report_revision"] = 99
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE agent_messages SET metadata_json=? WHERE id=?", (json.dumps(metadata), message["id"]))
        if damage == "failed":
            conn.execute("UPDATE tasks SET status='failed' WHERE id=?", (task.id,))
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(repo, task.id)
    assert _timing(path)["observed_denominator"] == 0


def test_auto_generated_report_does_not_count_its_old_draft_as_external_wait(tmp_path):
    repo, task, message = _report_task(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        repo.mark_job_running(job)
        repo.update_agent_report_conclusions_with_audit(
            task.id, message["metadata"]["draft_values"], 0,
            audit={"kind": "report.agent_conclusions.generated", "target_ref": task.id, "outcome": "succeeded"},
        )
        repo.finish_job(job, status="succeeded")
    assert _timing(path)["observed_denominator"] == 0


def test_intermediate_auto_gate_is_not_wait_time_but_returned_gate_is(tmp_path):
    from marvis.db import PlanRepository
    from marvis.orchestrator.contracts import PlanStatus, StepStatus
    from test_orch_plans_db_adaptive import _plan

    repo, task, _ = _report_task(tmp_path)
    # Make the old report draft stale, leaving only the native plan gate.
    from marvis.db import connect
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE tasks SET report_values_revision=1 WHERE id=?", (task.id,))
    plans = PlanRepository(repo.db_path)
    plan = _plan()
    plan.task_id = task.id
    plan.status = PlanStatus.AWAITING_CONFIRM
    plan.steps[0].status = StepStatus.AWAITING_CONFIRM
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "driver")
        repo.mark_job_running(job)
        plans.create_plan(plan)
        refresh_confirmation_wait(repo, task.id)
        assert not any(row["event"] == "waiting" for row in read_backend_events(path))
        repo.finish_job(job, status="succeeded")
        repo.start_job(task.id, "driver")
    timing = _timing(path, "workflow_confirmation_wait")
    assert timing["complete"] and timing["measured_intervals"] == 1


def test_wait_state_read_error_does_not_replace_business_result(tmp_path, monkeypatch):
    repo, task, _ = _report_task(tmp_path)
    def fail(*args):
        raise RuntimeError("private-read-error")
    monkeypatch.setattr(repo, "get_task", fail)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(repo, task.id)
    assert not backend_timing(read_backend_events(path))["journal_complete"]
    assert "private-read-error" not in path.read_text()


def test_concurrent_job_cannot_get_a_late_wait_from_an_old_idle_snapshot(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    repo, task, _ = _report_task(tmp_path)
    snapshot_read = threading.Event()
    release_snapshot = threading.Event()
    job_requested = threading.Event()
    original = repo.list_agent_messages
    def held_read(task_id):
        snapshot_read.set()
        assert release_snapshot.wait(5)
        return original(task_id)
    monkeypatch.setattr(repo, "list_agent_messages", held_read)
    def start():
        job_requested.set()
        return repo.start_job(task.id, "report")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        with ThreadPoolExecutor(max_workers=2) as pool:
            waiting = pool.submit(refresh_confirmation_wait, repo, task.id)
            assert snapshot_read.wait(5)
            job = pool.submit(start)
            assert job_requested.wait(5)
            assert not job.done()
            release_snapshot.set()
            waiting.result(timeout=5)
            assert job.result(timeout=5)
    events = read_backend_events(path)
    sequence = [row["event"] for row in events]
    assert sequence.index("waiting") < sequence.index("queued") < sequence.index("resumed")
    assert _timing(path)["complete"] and _timing(path)["right_censored_intervals"] == 0


def test_batch_confirmation_rollback_does_not_end_existing_wait(tmp_path):
    repo, task, message = _report_task(tmp_path)
    confirmation = {"task_id": task.id, "text_values": message["metadata"]["draft_values"], "expected_revision": 0}
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(repo, task.id)
        with pytest.raises(KeyError):
            repo.confirm_agent_report_batch([confirmation, {**confirmation, "task_id": "missing"}])
    timing = _timing(path)
    assert not timing["complete"] and timing["right_censored_intervals"] == 1
    assert timing["observed_denominator"] == 1
    assert timing["intervals"][0]["outcome"] == "censored"


@pytest.mark.parametrize("change", ["task", "binding", "raw_content", "legacy_header"])
def test_wait_pairs_require_matching_opaque_binding_and_new_carrier(tmp_path, change):
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        confirmation_wait("private-task", "report_confirmation_wait", {"draft": "private-value"})
        end_confirmation_wait("private-task")
    events = read_backend_events(path)
    if change in {"task", "binding"}:
        events[2][f"{change}_sha256"] = "a" * 64
        timing = backend_timing(events)["scopes"]["report_confirmation_wait"]
        assert not timing["complete"] and timing["invalid_events"]
    else:
        if change == "raw_content":
            events[1]["content"] = "PRIVATE"
        else:
            events[0]["event"] = "plan_revisions_enabled"
        with pytest.raises(ValueError):
            backend_timing(events)


def _native_plan(tmp_path, *, gate=False):
    from marvis.db import PlanRepository
    from marvis.orchestrator.contracts import PlanStatus, StepStatus
    from test_runtime_backend_observation import _task_repo
    from test_orch_plans_db_adaptive import _plan

    repo, task = _task_repo(tmp_path)
    plan = _plan()
    plan.task_id = task.id
    if gate:
        plan.status = PlanStatus.AWAITING_CONFIRM
        plan.steps[0].status = StepStatus.AWAITING_CONFIRM
    return repo, PlanRepository(repo.db_path), plan


@pytest.mark.parametrize("gate", [False, True])
def test_direct_plan_creation_and_confirmation_need_no_job_or_manual_refresh(tmp_path, gate):
    repo, plans, plan = _native_plan(tmp_path, gate=gate)
    scope = "workflow_confirmation_wait" if gate else "plan_confirmation_wait"
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        assert _timing(path, scope)["observed_denominator"] == 1
        if gate:
            plans.confirm_step(plan.steps[0].id)
        else:
            plans.confirm_plan(plan.id)
    result = _timing(path, scope)
    assert result["complete"] and result["measured_intervals"] == 1
    assert result["intervals"][0]["outcome"] == "resumed"
    assert plan.goal not in path.read_text() and plan.task_id not in path.read_text()


def test_plan_becoming_validated_and_cancelled_refreshes_its_wait(tmp_path):
    from marvis.orchestrator.contracts import PlanStatus

    repo, plans, plan = _native_plan(tmp_path)
    plan.status = PlanStatus.DRAFT
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        assert _timing(path, "plan_confirmation_wait")["observed_denominator"] == 0
        plans.set_plan_status(plan.id, PlanStatus.VALIDATED)
        assert _timing(path, "plan_confirmation_wait")["observed_denominator"] == 1
        plans.set_plan_status(plan.id, PlanStatus.CANCELLED)
    assert _timing(path, "plan_confirmation_wait")["intervals"][0]["outcome"] == "withdrawn"


def test_gate_payload_change_without_revision_rebinds_wait_and_rejects_old_confirmation(tmp_path):
    from marvis.state_machine import ConflictError
    from marvis.orchestrator.contracts import plan_payload_fingerprint, plan_to_dict

    repo, plans, plan = _native_plan(tmp_path, gate=True)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        prior = plans.load_plan(plan.id)
        step = prior.steps[0]
        old = plan_payload_fingerprint(plan_to_dict(prior))
        step.inputs = {"message": "private-new-input"}
        # update_step targets the persisted step ID; a stale caller-owned
        # parent label must not redirect observation to another task.
        step.plan_id = "stale-parent-label"
        plans.update_step(step)
        with pytest.raises(ConflictError):
            plans.confirm_step(step.id, expected_plan_fingerprint=old)
        assert plans.load_plan(plan.id).replan_count == 0
        plans.confirm_step_with_inputs(step.id, input_updates={"message": "final-input"})
    result = _timing(path, "workflow_confirmation_wait")
    assert result["complete"] and result["measured_intervals"] == 2
    assert [row["outcome"] for row in result["intervals"]] == ["superseded", "resumed"]
    assert len({row["binding_sha256"] for row in result["intervals"]}) == 2
    assert "private-new-input" not in path.read_text() and "final-input" not in path.read_text()


def test_plan_confirmation_commit_failure_preserves_pending_wait(tmp_path):
    import sqlite3
    from marvis.db import connect

    repo, plans, plan = _native_plan(tmp_path)
    with connect(repo.db_path) as conn:
        conn.execute("CREATE TABLE fail_confirmation(id TEXT REFERENCES plans(id) DEFERRABLE INITIALLY DEFERRED)")
        conn.execute("""CREATE TRIGGER fail_confirmation_commit AFTER UPDATE OF status ON plans
            WHEN NEW.status='confirmed' BEGIN INSERT INTO fail_confirmation VALUES ('missing'); END""")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        with pytest.raises(sqlite3.IntegrityError):
            plans.confirm_plan(plan.id)
    result = _timing(path, "plan_confirmation_wait")
    assert result["observed_denominator"] == 1 and result["right_censored_intervals"] == 1
    assert result["intervals"][0]["outcome"] == "censored"
    assert backend_timing(read_backend_events(path))["journal_complete"]


def test_plan_state_savepoint_rollback_keeps_the_same_wait(tmp_path):
    from marvis.db import connect
    from marvis.runtime_observations import plan_state_on_commit

    repo, plans, plan = _native_plan(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        with connect(repo.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("SAVEPOINT confirmation")
            conn.execute("UPDATE plans SET status='confirmed' WHERE id=?", (plan.id,))
            plan_state_on_commit(conn, repo.db_path, plan_id=plan.id)
            conn.execute("ROLLBACK TO confirmation")
    result = _timing(path, "plan_confirmation_wait")
    assert result["observed_denominator"] == 1 and result["right_censored_intervals"] == 1
    assert backend_timing(read_backend_events(path))["journal_complete"]


def test_old_wait_carrier_keeps_old_coverage_and_new_carrier_declares_plan_states(tmp_path):
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        pass
    events = read_backend_events(path)
    latest = process_observation([], [], events)
    assert latest["schema"] == "marvis.runtime-process-observation.v6"
    assert "committed_plan_and_step_state_changes" in latest["confirmation_wait_entry_coverage"]
    assert not latest["formal_timing_complete"]
    events[0]["event"] = "confirmation_waits_enabled"
    old = process_observation([], [], events)
    assert old["schema"] == "marvis.runtime-process-observation.v5"
    assert old["confirmation_wait_entry_coverage"] == ["task_job_completion", "report_draft_publication", "report_draft_save"]


def test_native_confirmation_http_closes_wait_without_starting_a_job(tmp_path):
    from test_orch_api import _client, _confirmation_snapshot, _job_statuses, _plan
    from marvis.orchestrator.contracts import PlanStatus

    client = _client(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        client.app.state.plan_repo.create_plan(_plan(status=PlanStatus.VALIDATED))
        snapshot = _confirmation_snapshot(client, "plan-1")
        assert client.post("/api/plans/plan-1/confirm", json=snapshot).status_code == 200
        assert _job_statuses(client.app.state.plan_repo.db_path) == []
    result = _timing(path, "plan_confirmation_wait")
    assert result["complete"] and result["measured_intervals"] == 1
    assert result["intervals"][0]["outcome"] == "resumed"


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_governed_decision_closes_wait_only_after_committed_native_change(tmp_path, decision):
    from marvis.governance import GovernanceRepository, canonical_payload_hash
    from marvis.orchestrator.contracts import plan_fingerprint, plan_step_confirmation_fingerprint
    from test_orch_api import _client
    from test_governance_repository import _seed_plan_gate, _binding
    from marvis.db import TaskRepository

    client = _client(tmp_path)
    plans = client.app.state.plan_repo
    inputs = {"strategy_id": "strategy-7"}
    _seed_plan_gate(plans.db_path, inputs)
    governance = GovernanceRepository(plans.db_path)
    principal = governance.create_local_principal()
    binding = _binding(input_hash=canonical_payload_hash(inputs))
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        refresh_confirmation_wait(TaskRepository(plans.db_path), "task-1")
        plan = plans.load_plan("plan-1")
        if decision == "approve":
            governance.authorize_step(binding, principal=principal, reason="reviewed", issue_effect_approval=True)
        else:
            governance.record_decision(
                binding, principal=principal, decision="reject", reason="reviewed",
                expected_plan_revision=plan.replan_count, expected_plan_status=plan.status.value,
                expected_plan_fingerprint=plan_fingerprint(plan),
                expected_step_fingerprint=plan_step_confirmation_fingerprint(plan.steps[0]),
                expected_step_status=plan.steps[0].status.value,
            )
    result = _timing(path, "workflow_confirmation_wait")
    assert result["complete"] and result["measured_intervals"] == 1
    assert result["intervals"][0]["outcome"] == ("resumed" if decision == "approve" else "withdrawn")
    assert len(governance.list_decisions_by_step(plan.id, plan.steps[0].id)) == 1


def test_postcommit_observation_failure_does_not_fail_a_committed_plan(tmp_path, monkeypatch):
    import marvis.runtime_waits as waits

    repo, plans, plan = _native_plan(tmp_path)
    def fail(*args):
        raise RuntimeError("private-observer-error")
    monkeypatch.setattr(waits, "refresh_confirmation_wait", fail)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        plans.create_plan(plan)
        assert plans.load_plan(plan.id).status == plan.status
    assert not backend_timing(read_backend_events(path))["journal_complete"]
    assert "private-observer-error" not in path.read_text()
