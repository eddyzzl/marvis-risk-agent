import sqlite3
import time

from marvis.db import TaskRepository, init_db
from marvis.agent.orchestrator import is_metrics_failure
from marvis.domain import TaskCreate, TaskStatus
from marvis.job_heartbeat import heartbeat_job
from marvis.job_watchdog import sweep_heartbeat_lost_jobs


def _task(repo: TaskRepository, tmp_path):
    return repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )


def test_touch_job_heartbeat_updates_column_and_only_while_active(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)

    assert repo.touch_job_heartbeat(job_id) is True

    repo.finish_job(job_id, status="succeeded")
    # A heartbeat racing a concurrent finish must not resurrect a terminal job.
    assert repo.touch_job_heartbeat(job_id) is False


def test_fail_heartbeat_lost_jobs_releases_stale_running_job_and_writes_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)

    released = repo.fail_heartbeat_lost_jobs(older_than_seconds=0)

    assert [job["id"] for job in released] == [job_id]
    assert released[0]["task_id"] == task.id
    assert released[0]["kind"] == "join"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        assert row == ("failed", "HeartbeatLost")
        audit_row = conn.execute(
            "SELECT kind, target_ref, outcome FROM audit WHERE kind = 'job.heartbeat_lost'"
        ).fetchone()
        assert audit_row == ("job.heartbeat_lost", job_id, "failed")
    assert repo.task_has_active_job(task.id) is False


def test_watchdog_releases_stale_queued_job_that_never_started(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "agent")

    released = sweep_heartbeat_lost_jobs(repo, older_than_seconds=0)

    assert [job["id"] for job in released] == [job_id]
    assert released[0]["previous_status"] == "queued"
    job = repo.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error_name"] == "JobStartLost"
    with sqlite3.connect(db_path) as conn:
        audit_row = conn.execute(
            "SELECT kind FROM audit WHERE target_ref = ?",
            (job_id,),
        ).fetchone()
    assert audit_row == ("job.start_lost",)
    assert repo.task_has_active_job(task.id) is False


def test_stuck_job_health_count_includes_stale_queued_jobs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    repo.start_job(task.id, "agent")

    assert repo.count_heartbeat_stale_running_jobs(older_than_seconds=0) == 1


def test_watchdog_closes_task_state_for_queued_notebook_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned")
    repo.update_status(task.id, TaskStatus.RUNNING, "notebook queued")
    repo.start_job(task.id, "notebook")

    repo.fail_heartbeat_lost_jobs(older_than_seconds=0)

    recovered = repo.get_task(task.id)
    assert recovered.status == TaskStatus.FAILED
    assert recovered.status_message.startswith("模型可复现性验证失败：")


def test_watchdog_closes_task_state_for_queued_metrics_job_as_retryable(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned")
    repo.update_status(task.id, TaskStatus.RUNNING, "running")
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed")
    repo.update_status(task.id, TaskStatus.COMPUTING_METRICS, "metrics queued")
    repo.start_job(task.id, "metrics")

    repo.fail_heartbeat_lost_jobs(older_than_seconds=0)

    recovered = repo.get_task(task.id)
    assert recovered.status == TaskStatus.FAILED
    assert is_metrics_failure(recovered) is True


def test_fail_heartbeat_lost_jobs_ignores_recently_touched_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)
    repo.touch_job_heartbeat(job_id)

    # A generous threshold means "just touched" is nowhere near stale.
    released = repo.fail_heartbeat_lost_jobs(older_than_seconds=3600)

    assert released == []
    assert repo.task_has_active_job(task.id) is True


def test_periodic_watchdog_mode_does_not_release_stale_running_worker(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)

    released = repo.fail_heartbeat_lost_jobs(
        older_than_seconds=0,
        include_running=False,
    )

    assert released == []
    assert repo.get_job(job_id)["status"] == "running"
    assert repo.task_has_active_job(task.id) is True


def test_sweep_heartbeat_lost_jobs_never_raises_on_repo_failure(tmp_path):
    class BoomRepo:
        def fail_heartbeat_lost_jobs(self, *, older_than_seconds, include_running=True):
            raise RuntimeError("db is on fire")

    # A watchdog sweep that itself crashes the app would be worse than a
    # missed sweep — this must swallow the error and return an empty list.
    assert sweep_heartbeat_lost_jobs(BoomRepo(), older_than_seconds=60) == []


def test_heartbeat_job_ticks_while_block_runs_and_stops_after(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)

    with heartbeat_job(repo, job_id, interval_seconds=0.05):
        time.sleep(0.2)

    job = repo.get_job(job_id)
    assert job["heartbeat_at"] is not None
    # The ticker thread must not keep running (and touching heartbeat_at)
    # after the context manager exits, or a finished job could be "revived".
    heartbeat_after_exit = job["heartbeat_at"]
    repo.finish_job(job_id, status="succeeded")
    time.sleep(0.1)
    finished_job = repo.get_job(job_id)
    assert finished_job["heartbeat_at"] == heartbeat_after_exit
    assert finished_job["status"] == "succeeded"


def test_heartbeat_job_does_not_touch_a_job_that_finished_before_the_next_tick(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(job_id)

    with heartbeat_job(repo, job_id, interval_seconds=30):
        # Finishes well before the first tick would fire; touch_job_heartbeat
        # is itself a no-op on a terminal job, so this is a belt-and-suspenders
        # check that a fast job doesn't leave a dangling heartbeat write race.
        repo.finish_job(job_id, status="succeeded")

    job = repo.get_job(job_id)
    assert job["status"] == "succeeded"
