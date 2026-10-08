import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from marvis.runtime_observations import measured, observe_runtime
from marvis.orchestrator.eval.runtime_process import (
    backend_timing, read_backend_events, process_observation, validate_process_observation,
)


def test_concurrent_calls_and_exceptions_keep_content_out_of_journal(tmp_path):
    path = tmp_path / "events"

    @measured("plugin")
    def call(value):
        if value == 3:
            raise ValueError("private-error")
        return "private-result"

    with observe_runtime(path, time.monotonic_ns()):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(call, [1, 2, 4, 5]))
        with pytest.raises(ValueError, match="private-error"):
            call(3)
    assert results == ["private-result"] * 4
    raw = path.read_text()
    assert "private" not in raw
    assert path.stat().st_mode & 0o777 == 0o600
    summary = backend_timing(read_backend_events(path))["scopes"]["plugin"]
    assert summary["complete"]
    assert summary["measured_intervals"] == 5
    assert sum(r["outcome"] == "raised" for r in summary["intervals"]) == 1
    assert 0 < summary["known_busy_duration_ns"] <= summary["known_summed_duration_ns"]


def test_queue_records_only_committed_success_and_cancellation_endpoint(tmp_path):
    from marvis.db import TaskRepository, init_db
    from marvis.domain import TaskCreate
    from marvis.repositories.tasks import ConflictError

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = TaskRepository(db)
    task = repo.create_task(TaskCreate(model_name="private", model_version="1", validator="private", source_dir=str(tmp_path)))
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        with pytest.raises(ConflictError):
            repo.start_job(task.id, "agent")
        assert repo.mark_job_running(job)
        assert not repo.mark_job_running(job)
        repo.finish_job(job, status="succeeded")
        cancelled = repo.start_job(task.id, "agent")
        repo.finish_job(cancelled, status="cancelled")
        assert not repo.mark_job_running(cancelled)
    source = read_backend_events(path)
    assert job not in path.read_text() and task.id not in path.read_text()
    queue = backend_timing(source)["scopes"]["queue"]
    assert queue["observed_denominator"] == 2
    assert queue["measured_intervals"] == 2 and queue["unknown_intervals"] == 0
    assert queue["complete"]
    assert [row["outcome"] for row in queue["intervals"]] == ["running", "cancelled"]


def test_real_adhoc_worker_and_runner_input_exception_are_measured(tmp_path):
    from test_plugin_runner import _runtime
    from marvis.plugins.manifest import ToolRef

    runner, _ = _runtime(tmp_path)
    module = tmp_path / "toy.py"
    module.write_text("def run(inputs, ctx):\n    return {'value': 7}\n")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        result = runner.invoke_adhoc(module=module, entrypoint="run", inputs={},
                                    input_schema={"type": "object"}, output_schema={"type": "object"},
                                    timeout_seconds=10, task_id="private-task")
        with pytest.raises(ValueError):
            runner.invoke(ToolRef("unknown", "unknown"), {}, task_id="private-task", invocation_id="")
    assert result.ok and result.output == {"value": 7}
    summary = backend_timing(read_backend_events(path))["scopes"]
    assert summary["adhoc"]["complete"] and summary["adhoc"]["measured_intervals"] == 1
    assert summary["plugin"]["intervals"][0]["outcome"] == "raised"


def test_report_batch_rollback_emits_no_queue_and_commit_emits_once(tmp_path):
    from marvis.db import TaskRepository, init_db
    from marvis.db_schema import connect
    from marvis.domain import TaskCreate

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = TaskRepository(db)
    task = repo.create_task(TaskCreate(model_name="fixture", model_version="1", validator="fixture", source_dir=str(tmp_path)))
    with connect(db) as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version=1, status='writing_artifacts' WHERE id=?", (task.id,))
    confirmation = {"task_id": task.id, "text_values": {"TEXT:final_validation_conclusion": "fixture"}, "expected_revision": 0}
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        with pytest.raises(KeyError):
            repo.confirm_agent_report_batch([confirmation, {**confirmation, "task_id": "missing"}])
        assert [row["event"] for row in read_backend_events(path)] == ["confirmation_plan_states_enabled"]
        assert not repo.task_has_active_job(task.id)
        [confirmed] = repo.confirm_agent_report_batch([confirmation])
        assert repo.mark_job_running(confirmed["job_id"])
    queue = backend_timing(read_backend_events(path))["scopes"]["queue"]
    assert queue["complete"] and queue["measured_intervals"] == 1


def _events():
    return [
        {"event": "started", "scope": "plugin", "identity": "a" * 32, "elapsed_ns": 10},
        {"event": "started", "scope": "plugin", "identity": "b" * 32, "elapsed_ns": 20},
        {"event": "returned", "scope": "plugin", "identity": "a" * 32, "elapsed_ns": 30},
        {"event": "raised", "scope": "plugin", "identity": "b" * 32, "elapsed_ns": 40},
        {"event": "closed", "valid": True, "elapsed_ns": 50},
    ]


@pytest.mark.parametrize("change", [None, "missing_end", "missing_start", "unsealed", "failed_write", "duplicate", "late_record"])
def test_overlap_and_incomplete_journals(change):
    events = _events()
    if change == "missing_end":
        events.pop(2)
    elif change == "missing_start":
        events.pop(0)
    elif change == "unsealed":
        events.pop()
    elif change == "failed_write":
        events[-1]["valid"] = False
    elif change == "duplicate":
        events.insert(1, events[0].copy())
    elif change == "late_record":
        events.append({**events[0], "elapsed_ns": 60})
    summary = backend_timing(events)["scopes"]["plugin"]
    assert summary["complete"] is (change is None)
    if change is None:
        assert summary["known_summed_duration_ns"] == 40
        assert summary["known_busy_duration_ns"] == 30


@pytest.mark.parametrize("change", ["extra_content", "negative_time", "boolean_time", "unrecognized_scope", "raw_identity", "invalid_json", "symlink"])
def test_unsafe_journal_is_not_archived(tmp_path, change):
    events = _events()
    if change == "extra_content":
        events[0]["payload"] = "private-prompt"
    elif change == "negative_time":
        events[0]["elapsed_ns"] = -1
    elif change == "boolean_time":
        events[0]["elapsed_ns"] = True
    elif change == "unrecognized_scope":
        events[0]["scope"] = "private-tool"
    elif change == "raw_identity":
        events[0]["identity"] = "private-id"
    path = tmp_path / "events"
    path.write_text("\n".join(json.dumps(row) for row in events))
    if change == "invalid_json":
        path.write_text("private-not-json")
    elif change == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    assert read_backend_events(path) == [{"event": "unavailable"}]


def test_manifest_recomputes_backend_and_rejects_future_time():
    events = _events()
    result = process_observation([], [], events)
    validate_process_observation(result, [], interventions=0, duration_ms=1, backend_events=events)
    result["backend"]["scopes"]["plugin"]["known_busy_duration_ns"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        validate_process_observation(result, [], interventions=0, duration_ms=1, backend_events=events)
    events[-1]["elapsed_ns"] = 3_000_000
    result = process_observation([], [], events)
    with pytest.raises(ValueError, match="exceeds_case"):
        validate_process_observation(result, [], interventions=0, duration_ms=1, backend_events=events)


def test_measurement_write_failure_preserves_result_but_cannot_claim_complete(tmp_path):
    import marvis.runtime_observations as observations

    @measured("plugin")
    def call():
        return 7

    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        observations._observer.file.close()
        assert call() == 7
    assert not backend_timing(read_backend_events(path))["journal_complete"]


def _task_repo(tmp_path):
    from marvis.db import TaskRepository, init_db
    from marvis.domain import TaskCreate

    db = tmp_path / "app.sqlite"
    init_db(db)
    repo = TaskRepository(db)
    task = repo.create_task(TaskCreate(model_name="fixture", model_version="1", validator="fixture", source_dir=str(tmp_path)))
    return repo, task


@pytest.mark.parametrize("status", ["failed", "cancelled", "succeeded", "interrupted"])
def test_unstarted_terminal_status_is_measured_once_and_is_not_execution(tmp_path, status):
    repo, task = _task_repo(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        repo.finish_job(job, status=status)
        repo.finish_job(job, status=status)
        assert not repo.mark_job_running(job)
    events = read_backend_events(path)
    queue = backend_timing(events)["scopes"]["queue"]
    assert queue["complete"] and queue["measured_intervals"] == 1
    assert queue["intervals"][0]["outcome"] == status
    assert process_observation([], [], events)["schema"] == "marvis.runtime-process-observation.v6"


@pytest.mark.parametrize("rollback", ["outer", "savepoint", "commit_failure"])
def test_outer_transaction_or_savepoint_rollback_does_not_end_queue(tmp_path, rollback):
    import sqlite3
    import marvis.runtime_observations as observations
    from marvis.db_schema import connect

    repo, task = _task_repo(tmp_path)
    with connect(repo.db_path) as conn:
        conn.execute("CREATE TABLE observation_commit_failure(task_id TEXT REFERENCES tasks(id) DEFERRABLE INITIALLY DEFERRED)")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        if rollback == "outer":
            with pytest.raises(RuntimeError, match="later failure"):
                with connect(repo.db_path) as conn:
                    assert repo.finish_job_on_connection(conn, job, status="cancelled")
                    assert not any(row["event"] == "cancelled" for row in read_backend_events(path))
                    raise RuntimeError("later failure")
        elif rollback == "savepoint":
            with connect(repo.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("SAVEPOINT fixture")
                assert repo.finish_job_on_connection(conn, job, status="cancelled")
                conn.execute("ROLLBACK TO fixture")
                conn.execute("RELEASE fixture")
        else:
            with pytest.raises(sqlite3.IntegrityError):
                with connect(repo.db_path) as conn:
                    assert repo.finish_job_on_connection(conn, job, status="cancelled")
                    conn.execute("INSERT INTO observation_commit_failure VALUES ('missing-task')")
        assert repo.get_job(job)["status"] == "queued"
        assert observations._observer.pending == {}
    queue = backend_timing(read_backend_events(path))["scopes"]["queue"]
    assert not queue["complete"] and queue["unknown_intervals"] == 1
    assert queue["measured_intervals"] == 0


@pytest.mark.parametrize("release", ["watchdog", "restart", "orphan"])
def test_native_recovery_paths_close_queued_wait_after_commit(tmp_path, release):
    from marvis.recovery import reclaim_stale_running_tasks, _fail_orphan_task_jobs

    repo, task = _task_repo(tmp_path)
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        job = repo.start_job(task.id, "agent")
        if release == "watchdog":
            repo.fail_heartbeat_lost_jobs(older_than_seconds=0)
        elif release == "restart":
            reclaim_stale_running_tasks(repo.db_path)
        else:
            _fail_orphan_task_jobs(repo, task.id)
        assert repo.get_job(job)["status"] == "failed"
    queue = backend_timing(read_backend_events(path))["scopes"]["queue"]
    assert queue["complete"] and queue["measured_intervals"] == 1
    assert queue["intervals"][0]["outcome"] == "failed"


def test_preexisting_queue_has_unknown_start_instead_of_invented_zero(tmp_path):
    repo, task = _task_repo(tmp_path)
    job = repo.start_job(task.id, "agent")
    path = tmp_path / "events"
    with observe_runtime(path, time.monotonic_ns()):
        repo.finish_job(job, status="cancelled")
    queue = backend_timing(read_backend_events(path))["scopes"]["queue"]
    assert not queue["complete"] and queue["unknown_intervals"] == 1
    assert queue["measured_intervals"] == 0


@pytest.mark.parametrize("change", ["duplicate", "late", "missing"])
def test_queue_terminal_coverage_marker_cannot_be_forged_or_duplicated(change):
    events = [
        {"event": "queue_terminals_enabled", "elapsed_ns": 0},
        {"event": "queued", "scope": "queue", "identity": "a" * 64, "elapsed_ns": 10},
        {"event": "cancelled", "scope": "queue", "identity": "a" * 64, "elapsed_ns": 20},
        {"event": "closed", "valid": True, "elapsed_ns": 30},
    ]
    if change == "duplicate":
        events.insert(1, events[0].copy())
    elif change == "late":
        events[0], events[1] = events[1], {**events[0], "elapsed_ns": 11}
    else:
        events.pop(0)
    with pytest.raises(ValueError, match="backend_header"):
        backend_timing(events)
