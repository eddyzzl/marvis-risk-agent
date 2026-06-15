import sqlite3

from marvis.db import TaskRepository, init_db
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TaskCreate,
    TaskStatus,
)
from marvis.recovery import last_completed_step, reclaim_stale_running_tasks


def test_last_completed_step_returns_none_when_dir_empty(tmp_path):
    assert last_completed_step(tmp_path) is None


def test_last_completed_step_detects_notebook_step(tmp_path):
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    (execution_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    (execution_dir / "code_model_scores.csv").write_text(
        "row_index,code_model_score\n0,0.1\n",
        encoding="utf-8",
    )
    (execution_dir / "runtime_contract.json").write_text("{}", encoding="utf-8")

    assert last_completed_step(tmp_path) == "notebook"


def test_last_completed_step_detects_artifacts_step(tmp_path):
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    (execution_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    (execution_dir / "code_model_scores.csv").write_text(
        "row_index,code_model_score\n0,0.1\n",
        encoding="utf-8",
    )
    (execution_dir / "runtime_contract.json").write_text("{}", encoding="utf-8")
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "validation.xlsx").write_bytes(b"PK\x03\x04")
    (outputs_dir / "validation_report.docx").write_bytes(b"PK\x03\x04")

    assert last_completed_step(tmp_path) == "artifacts"


def test_reclaim_stale_running_tasks_marks_orphan_running_tasks_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED
    assert loaded.status_message == "reclaimed: server restart while running"
    assert loaded.status_reason_code == TASK_STATUS_REASON_SERVER_RESTART


def test_reclaim_stale_running_tasks_marks_later_active_states_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED


def test_reclaim_stale_running_tasks_preserves_agent_writing_artifacts_without_active_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    assert repo.list_agent_messages(task.id) == []


def test_reclaim_stale_running_tasks_preserves_agent_writing_artifacts_with_active_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    job_id = repo.start_job(task.id, "report")

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row == ("failed", "ServerRestart")
    messages = repo.list_agent_messages(task.id)
    assert messages[-1]["stage"] == "failure"
    assert messages[-1]["metadata"]["interrupted_by_restart"] is True


def test_reclaim_stale_running_tasks_skips_recent_active_agent_job_within_stale_window(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    repo.start_job(task.id, "report")

    # With a one-hour stale window a just-updated task is not yet stale, so it must
    # NOT be treated as interrupted: no premature "server restart" notice is
    # inserted (regression guard for the cutoff-gated active-job UNION half).
    reclaimed = reclaim_stale_running_tasks(db_path, stale_after_seconds=3600)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    assert repo.list_agent_messages(task.id) == []


def test_reclaim_stale_running_tasks_finalizes_agent_draft_message_for_writing_artifacts_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    job_id = repo.start_job(task.id, "agent")
    draft_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="word_conclusion_draft",
        content="",
        metadata={"streaming": True, "model_id": "m1"},
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row == ("failed", "ServerRestart")
    messages = {message["id"]: message for message in repo.list_agent_messages(task.id)}
    assert messages[draft_message["id"]]["metadata"]["streaming"] is False
    assert messages[draft_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[draft_message["id"]]["content"]


def test_reclaim_stale_running_tasks_marks_orphan_jobs_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(id, task_id, kind, status, created_at)
            VALUES('job-1', 'task-1', 'notebook', 'running', '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

    reclaim_stale_running_tasks(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, error_name FROM jobs WHERE id='job-1'"
        ).fetchone()
    assert row == ("failed", "ServerRestart")


def test_reclaim_stale_running_tasks_finalizes_streaming_agent_messages(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    empty_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="",
        metadata={"streaming": True, "model_id": "m1"},
    )
    partial_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="metrics",
        content="已经写入的半截分析",
        metadata={"streaming": True, "model_id": "m1"},
    )

    reclaim_stale_running_tasks(db_path)

    messages = {message["id"]: message for message in repo.list_agent_messages(task.id)}
    assert messages[empty_message["id"]]["metadata"]["streaming"] is False
    assert messages[empty_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[empty_message["id"]]["content"]
    assert messages[partial_message["id"]]["metadata"]["streaming"] is False
    assert messages[partial_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert messages[partial_message["id"]]["content"].startswith("已经写入的半截分析")
    assert "输出在此处中断" in messages[partial_message["id"]]["content"]


def test_reclaim_stale_running_tasks_adds_agent_restart_notice(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)

    reclaim_stale_running_tasks(db_path)

    messages = repo.list_agent_messages(task.id)
    assert messages[-1]["stage"] == "failure"
    assert messages[-1]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[-1]["content"]
