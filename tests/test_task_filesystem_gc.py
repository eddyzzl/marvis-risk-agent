from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import stat
import threading

import pytest
from fastapi.testclient import TestClient

import marvis.data.task_filesystem_gc as task_fs_gc_module
from marvis.app import create_app
from marvis.data.contracts import Dataset
from marvis.data.dataset_source_gc import DatasetSourceGarbageCollector
from marvis.data.task_filesystem_gc import (
    TaskFilesystemGarbageCollector,
    TaskFilesystemGcSweepReport,
    TaskFilesystemGcTarget,
    TaskFilesystemGcWatchdog,
)
from marvis.data.windows_task_tree_delete import (
    UnsafeWindowsTaskTreeDeleteError,
    WindowsTaskTreeDeleteRetryableError,
    WindowsTaskTreeDeleteUnsupportedError,
)
from marvis.db import DatasetRepository, TaskRepository, connect, init_db
from marvis.db_schema import SCHEMA_VERSION
from marvis.domain import TASK_TYPE_VINTAGE, TaskCreate
from marvis.settings import build_settings


FIXED_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _task(repo: TaskRepository, tmp_path):
    intake = tmp_path / "workspace" / "material_uploads" / "risk-intake-owned"
    intake.mkdir(parents=True)
    task = repo.create_task(
        TaskCreate(
            task_type=TASK_TYPE_VINTAGE,
            model_name="risk analysis",
            model_version="v1",
            validator="qa",
            source_dir=str(intake),
            run_mode="agent",
        )
    )
    with repo.transaction() as conn:
        repo.record_task_filesystem_provision_on_connection(
            conn,
            task,
            target_type="risk_intake_dir",
            relative_path=intake.name,
        )
    return task


def _targets(task_id: str) -> tuple[TaskFilesystemGcTarget, ...]:
    return (
        TaskFilesystemGcTarget("task_dir", task_id),
        TaskFilesystemGcTarget(
            "dataset_identity_dir",
            f"{task_id}/.source-identities",
        ),
        TaskFilesystemGcTarget("dataset_task_dir", task_id),
        TaskFilesystemGcTarget("risk_intake_dir", "risk-intake-owned"),
    )


def test_task_purge_atomically_enqueues_typed_filesystem_targets(tmp_path):
    db_path = tmp_path / "workspace" / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_task_delete_after_gc_enqueue
            BEFORE DELETE ON tasks
            BEGIN
                SELECT RAISE(ABORT, 'synthetic purge rollback');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic purge rollback"):
        repo.purge_task(task.id, task_fs_targets=_targets(task.id))

    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id = ?", (task.id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_fs_gc_queue WHERE origin_task_id = ?",
            (task.id,),
        ).fetchone()[0] == 0

    with connect(db_path) as conn:
        conn.execute("DROP TRIGGER abort_task_delete_after_gc_enqueue")
    summary = repo.purge_task(task.id, task_fs_targets=_targets(task.id))

    assert summary["task_fs_gc_candidates"] == 4
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT target_type, relative_path, state
              FROM task_fs_gc_queue
             WHERE origin_task_id = ?
             ORDER BY target_type
            """,
            (task.id,),
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("dataset_identity_dir", f"{task.id}/.source-identities", "pending"),
        ("dataset_task_dir", task.id, "pending"),
        ("risk_intake_dir", "risk-intake-owned", "pending"),
        ("task_dir", task.id, "pending"),
    ]


def test_schema_30_upgrades_a_version_29_database_with_task_fs_ledger(tmp_path):
    db_path = tmp_path / "workspace" / "marvis.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DROP TABLE task_fs_gc_queue")
        conn.execute("PRAGMA user_version = 29")

    init_db(db_path)

    with connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(task_fs_gc_queue)").fetchall()
        }
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(task_fs_gc_queue)").fetchall()
        }
    assert version == SCHEMA_VERSION
    assert {
        "target_type",
        "relative_path",
        "state",
        "attempt_count",
        "next_attempt_at",
        "origin_task_id",
        "origin_task_created_at",
    } <= columns
    assert {"idx_task_fs_gc_due", "idx_task_fs_gc_origin"} <= indexes


def test_schema_34_preserves_existing_gc_rows_and_extends_target_constraint(
    tmp_path,
):
    db_path = tmp_path / "workspace" / "marvis.sqlite"
    init_db(db_path)
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(db_path) as conn:
        conn.execute("DROP TABLE task_fs_gc_queue")
        conn.execute(
            """
            CREATE TABLE task_fs_gc_queue (
                target_type TEXT NOT NULL CHECK(target_type IN (
                    'task_dir', 'dataset_identity_dir',
                    'dataset_task_dir', 'risk_intake_dir'
                )),
                relative_path TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN (
                    'pending', 'retry_wait', 'quarantined'
                )),
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                last_error_code TEXT,
                last_error_message TEXT,
                origin_task_id TEXT NOT NULL,
                origin_task_created_at TEXT NOT NULL,
                enqueued_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(target_type, relative_path)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_task_fs_gc_due
                ON task_fs_gc_queue(
                    state, next_attempt_at, target_type, relative_path
                )
            """
        )
        conn.execute(
            """
            CREATE INDEX idx_task_fs_gc_origin
                ON task_fs_gc_queue(origin_task_id, target_type)
            """
        )
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, origin_task_id, origin_task_created_at,
                enqueued_at, updated_at
            ) VALUES ('task_dir', 'old-task', 'pending', 0, ?, 'old-task', ?, ?, ?)
            """,
            (timestamp, timestamp, timestamp, timestamp),
        )
        conn.execute("PRAGMA user_version = 33")

    init_db(db_path)

    with connect(db_path) as conn:
        preserved = conn.execute(
            """
            SELECT target_type, relative_path, origin_task_id
              FROM task_fs_gc_queue
            """
        ).fetchall()
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, origin_task_id, origin_task_created_at,
                enqueued_at, updated_at
            ) VALUES (
                'validation_batch_source_dir',
                'validation-batches/0123456789abcdef0123456789abcdef',
                'pending', 0, ?, 'batch-task', ?, ?, ?
            )
            """,
            (timestamp, timestamp, timestamp, timestamp),
        )
        indexes = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA index_list(task_fs_gc_queue)"
            ).fetchall()
        }
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert [tuple(row) for row in preserved] == [
        ("task_dir", "old-task", "old-task")
    ]
    assert {"idx_task_fs_gc_due", "idx_task_fs_gc_origin"} <= indexes
    assert version == SCHEMA_VERSION


@pytest.mark.parametrize(
    "target_constraint",
    [
        "CHECK(target_type IN ('task_dir'))",
        "CHECK(target_type IN ('task_dir', 'validation_batch_source_dir'))",
    ],
)
def test_schema_34_rebuilds_partial_queue_without_false_completion(
    tmp_path,
    target_constraint,
):
    db_path = tmp_path / "partial-v33.sqlite"
    with connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE task_fs_gc_queue (
                target_type TEXT NOT NULL {target_constraint},
                relative_path TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO task_fs_gc_queue VALUES ('task_dir', 'unproven-task')"
        )
        conn.execute("PRAGMA user_version = 33")

    init_db(db_path)

    with connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(task_fs_gc_queue)").fetchall()
        }
        table_sql = str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = 'task_fs_gc_queue'"
            ).fetchone()[0]
        )
        rows = conn.execute("SELECT * FROM task_fs_gc_queue").fetchall()
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(task_fs_gc_queue)").fetchall()
        }

    assert version == SCHEMA_VERSION
    assert {
        "target_type",
        "relative_path",
        "state",
        "attempt_count",
        "next_attempt_at",
        "last_error_code",
        "last_error_message",
        "origin_task_id",
        "origin_task_created_at",
        "enqueued_at",
        "updated_at",
    } <= columns
    assert "validation_batch_source_dir" in table_sql
    # A two-column legacy row has no task identity/timestamp provenance.  Do
    # not invent deletion authority while repairing the schema.
    assert rows == []
    assert {"idx_task_fs_gc_due", "idx_task_fs_gc_origin"} <= indexes


@pytest.mark.parametrize(
    "datasets_ddl",
    [
        None,
        "CREATE TABLE datasets (id TEXT PRIMARY KEY)",
    ],
)
def test_schema_upgrade_skips_dataset_lookup_index_for_partial_legacy_schema(
    tmp_path,
    datasets_ddl,
):
    db_path = tmp_path / "partial-v28.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DROP TABLE datasets")
        if datasets_ddl is not None:
            conn.execute(datasets_ddl)
        conn.execute("PRAGMA user_version = 28")

    init_db(db_path)

    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        indexes = (
            []
            if datasets_ddl is None
            else conn.execute("PRAGMA index_list(datasets)").fetchall()
        )
    assert "idx_datasets_source_path_task" not in {str(row[1]) for row in indexes}


def test_task_filesystem_cleanup_retries_from_persisted_state_after_restart(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    (task_dir / "result.txt").write_text("keep until durable retry", encoding="utf-8")
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )

    def sharing_violation(_root, _relative_path):
        raise PermissionError("synthetic sharing violation")

    failing = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
        delete_target=sharing_violation,
    )
    first = failing.sweep(force=True)

    assert first.deferred == 1
    assert task_dir.is_dir()
    [entry] = failing.status().entries
    assert entry.state == "retry_wait"
    assert entry.attempt_count == 1
    assert entry.last_error_code == "PermissionError"
    assert entry.next_attempt_at > FIXED_NOW

    restarted = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW + timedelta(days=1),
    )
    recovered = restarted.sweep()

    assert recovered.deleted == 1
    assert restarted.status().entries == ()
    assert not task_dir.exists()


def test_risk_intake_reuse_cannot_race_cleanup_reference_recheck(tmp_path):
    workspace = tmp_path / "workspace"
    app = create_app(workspace)
    client = TestClient(app)
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "vintage",
            "run_mode": "agent",
            "model_name": "risk intake owner",
            "validator": "qa",
        },
    )
    assert created.status_code == 200, created.text
    task = created.json()
    intake = Path(task["source_dir"])
    relative = intake.relative_to(workspace / "material_uploads").as_posix()
    TaskRepository(app.state.settings.db_path).purge_task(
        task["id"],
        task_fs_targets=(TaskFilesystemGcTarget("risk_intake_dir", relative),),
    )

    delete_entered = threading.Event()
    resume_delete = threading.Event()

    def paused_delete(root, relative_path):
        delete_entered.set()
        assert resume_delete.wait(timeout=5)
        shutil.rmtree(root / relative_path)

    collector = TaskFilesystemGarbageCollector(
        app.state.settings.db_path,
        tasks_root=app.state.settings.tasks_dir,
        datasets_root=app.state.settings.datasets_dir,
        material_uploads_root=workspace / "material_uploads",
        delete_target=paused_delete,
    )
    cleanup = threading.Thread(
        target=lambda: collector.sweep(force=True),
        daemon=True,
    )
    cleanup.start()
    assert delete_entered.wait(timeout=5)
    outcome: dict[str, object] = {}

    def reuse_intake():
        outcome["response"] = client.post(
            "/api/tasks",
            json={
                "task_type": "vintage",
                "run_mode": "manual",
                "model_name": "concurrent reuser",
                "validator": "qa",
                "source_dir": str(intake),
            },
        )

    creator = threading.Thread(target=reuse_intake, daemon=True)
    creator.start()
    try:
        resume_delete.set()
        cleanup.join(timeout=10)
        creator.join(timeout=10)
    finally:
        resume_delete.set()
    assert cleanup.is_alive() is False
    assert creator.is_alive() is False
    response = outcome["response"]
    assert response.status_code == 422
    assert not intake.exists()
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE source_dir = ?",
            (str(intake),),
        ).fetchone()[0] == 0


def test_dataset_task_directory_waits_for_dataset_source_queue_to_drain(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    relative_source = f"{task.id}/sample.parquet"
    source = workspace / "datasets" / relative_source
    source.parent.mkdir(parents=True)
    source.write_bytes(b"parquet")
    DatasetRepository(db_path).create_dataset(
        Dataset(
            id=f"dataset-{task.id}",
            task_id=task.id,
            role="sample",
            source_path=relative_source,
            format="parquet",
            sheet=None,
            row_count=1,
            columns=(),
            has_target=False,
            target_col=None,
            created_at="2026-08-01T12:00:00Z",
            content_hash="a" * 64,
        )
    )
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("dataset_task_dir", task.id),),
    )
    task_fs_gc = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
    )

    blocked = task_fs_gc.sweep(force=True)

    assert blocked.deferred == 1
    assert source.is_file()
    [entry] = task_fs_gc.status().entries
    assert entry.state == "retry_wait"
    assert entry.last_error_code == "_DatasetSourceCleanupPending"

    dataset_report = DatasetSourceGarbageCollector(
        db_path,
        workspace / "datasets",
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)
    completed = task_fs_gc.sweep(force=True)

    assert dataset_report.deleted == 1
    assert completed.deleted == 1
    assert not source.parent.exists()
    assert task_fs_gc.status().entries == ()


def test_delete_api_leaves_failed_directory_cleanup_in_durable_retry_state(tmp_path):
    workspace = tmp_path / "workspace"
    app = create_app(workspace)
    client = TestClient(app)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "durable cleanup",
            "validator": "qa",
            "source_dir": str(workspace),
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    task_dir = app.state.settings.tasks_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "result.txt").write_text("retry me", encoding="utf-8")

    def locked_directory(_root, _relative_path):
        raise PermissionError("synthetic directory sharing violation")

    app.state.task_filesystem_gc = TaskFilesystemGarbageCollector(
        app.state.settings.db_path,
        tasks_root=app.state.settings.tasks_dir,
        datasets_root=app.state.settings.datasets_dir,
        material_uploads_root=workspace / "material_uploads",
        delete_target=locked_directory,
    )

    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 204
    assert task_dir.is_dir()
    status = app.state.task_filesystem_gc.status(origin_task_id=task_id)
    assert status.pending_count == 3
    assert {entry.target_type for entry in status.entries} == {
        "task_dir",
        "dataset_identity_dir",
        "dataset_task_dir",
    }
    with connect(app.state.settings.db_path) as conn:
        delete_audit = conn.execute(
            """
            SELECT outcome FROM audit
             WHERE kind = 'task.delete' AND target_ref = ?
            """,
            (task_id,),
        ).fetchone()
        cleanup_count = conn.execute(
            """
            SELECT COUNT(*) FROM audit
             WHERE kind = 'task.cleanup' AND target_ref = ?
            """,
            (task_id,),
        ).fetchone()[0]
    assert delete_audit["outcome"] == "cleanup_pending"
    assert cleanup_count == 0


def test_delete_api_enqueues_and_removes_owned_risk_intake_directory(tmp_path):
    workspace = tmp_path / "workspace"
    app = create_app(workspace)
    client = TestClient(app)
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "vintage",
            "run_mode": "agent",
            "model_name": "risk intake cleanup",
            "validator": "qa",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    intake = Path(created.json()["source_dir"])
    (intake / "uploaded.csv").write_text("id,bad\n1,0\n", encoding="utf-8")

    response = client.delete(f"/api/tasks/{task_id}")

    assert response.status_code == 204
    assert not intake.exists()
    assert app.state.task_filesystem_gc.status(origin_task_id=task_id).entries == ()
    assert _cleanup_audit_count(app.state.settings.db_path, task_id) == 1


def test_delete_api_preserves_user_directory_that_only_looks_like_risk_intake(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    app = create_app(workspace)
    client = TestClient(app)
    user_directory = workspace / "material_uploads" / "risk-intake-user-owned"
    user_directory.mkdir(parents=True)
    sentinel = user_directory / "keep.csv"
    sentinel.write_text("id,bad\n1,0\n", encoding="utf-8")
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "vintage",
            "run_mode": "manual",
            "model_name": "user material",
            "validator": "qa",
            "source_dir": str(user_directory),
        },
    )
    assert created.status_code == 200, created.text

    response = client.delete(f"/api/tasks/{created.json()['id']}")

    assert response.status_code == 204
    assert sentinel.read_text(encoding="utf-8") == "id,bad\n1,0\n"


def test_collector_quarantines_risk_intake_target_without_exact_provenance(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    user_directory = workspace / "material_uploads" / "risk-intake-unproven"
    user_directory.mkdir(parents=True)
    sentinel = user_directory / "keep.txt"
    sentinel.write_text("user-owned", encoding="utf-8")
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, last_error_code, last_error_message,
                origin_task_id, origin_task_created_at, enqueued_at, updated_at
            ) VALUES (
                'risk_intake_dir', 'risk-intake-unproven', 'pending', 0,
                ?, NULL, NULL, 'unproven-task', ?, ?, ?
            )
            """,
            (timestamp, timestamp, timestamp, timestamp),
        )
    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
    )

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert sentinel.read_text(encoding="utf-8") == "user-owned"
    [entry] = collector.status().entries
    assert entry.state == "quarantined"
    assert "lacks exact platform-created provenance" in entry.last_error_message


def test_task_directory_symlink_is_unlinked_without_following_target(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    task_link = workspace / "tasks" / task.id
    task_link.parent.mkdir(parents=True, exist_ok=True)
    task_link.symlink_to(outside, target_is_directory=True)
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )

    report = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
    ).sweep(force=True)

    assert report.deleted == 1
    assert not task_link.exists()
    assert not task_link.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_task_id_reuse_cancels_stale_directory_cleanup(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    old = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / old.id
    task_dir.mkdir(parents=True)
    sentinel = task_dir / "new-owner.txt"
    sentinel.write_text("new task owns this", encoding="utf-8")
    repo.purge_task(
        old.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", old.id),),
    )
    replacement = repo.create_task(
        TaskCreate(
            model_name="replacement",
            model_version="v2",
            validator="qa",
            source_dir=str(workspace),
        )
    )
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE tasks SET id = ? WHERE id = ?",
            (old.id, replacement.id),
        )

    report = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
    ).sweep(force=True)

    assert report.cancelled_referenced == 1
    assert sentinel.read_text(encoding="utf-8") == "new task owns this"
    assert TaskRepository(db_path).get_task(old.id).model_name == "replacement"


def test_cleanup_audit_is_written_when_task_filesystem_queue_drains_last(tmp_path):
    workspace, db_path, task_id = _purge_with_dataset_and_task_dir(tmp_path)
    dataset_gc = DatasetSourceGarbageCollector(db_path, workspace / "datasets")
    task_fs_gc = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
    )

    assert dataset_gc.sweep(force=True).deleted == 1
    assert _cleanup_audit_count(db_path, task_id) == 0
    assert task_fs_gc.sweep(force=True).deleted == 1
    assert _cleanup_audit_count(db_path, task_id) == 1


def test_cleanup_audit_is_written_when_dataset_source_queue_drains_last(tmp_path):
    workspace, db_path, task_id = _purge_with_dataset_and_task_dir(tmp_path)
    task_fs_gc = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
    )

    assert task_fs_gc.sweep(force=True).deleted == 1
    assert _cleanup_audit_count(db_path, task_id) == 0
    assert DatasetSourceGarbageCollector(
        db_path,
        workspace / "datasets",
    ).sweep(force=True).deleted == 1
    assert _cleanup_audit_count(db_path, task_id) == 1


def test_cleanup_audit_is_immediate_and_idempotent_when_both_queues_are_empty(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)

    repo.purge_task(task.id, actor="tester")

    with connect(db_path) as conn:
        delete_outcome = conn.execute(
            """
            SELECT outcome FROM audit
             WHERE kind = 'task.delete' AND target_ref = ?
            """,
            (task.id,),
        ).fetchone()[0]
        from marvis.repositories.tasks import maybe_write_task_cleanup_succeeded

        assert maybe_write_task_cleanup_succeeded(conn, task_id=task.id) is False
    assert delete_outcome == "db_purged"
    assert _cleanup_audit_count(db_path, task.id) == 1


def _purge_with_dataset_and_task_dir(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    (task_dir / "result.txt").write_text("delete", encoding="utf-8")
    relative_source = f"{task.id}/sample.parquet"
    source = workspace / "datasets" / relative_source
    source.parent.mkdir(parents=True)
    source.write_bytes(b"parquet")
    DatasetRepository(db_path).create_dataset(
        Dataset(
            id=f"dataset-{task.id}",
            task_id=task.id,
            role="sample",
            source_path=relative_source,
            format="parquet",
            sheet=None,
            row_count=1,
            columns=(),
            has_target=False,
            target_col=None,
            created_at="2026-08-01T12:00:00Z",
            content_hash="b" * 64,
        )
    )
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )
    return workspace, db_path, task.id


def _cleanup_audit_count(db_path, task_id: str) -> int:
    with connect(db_path) as conn:
        return int(
            conn.execute(
                """
                SELECT COUNT(*) FROM audit
                 WHERE kind = 'task.cleanup'
                   AND target_ref = ?
                   AND outcome = 'succeeded'
                """,
                (task_id,),
            ).fetchone()[0]
        )


def test_unsupported_recursive_delete_is_quarantined_fail_closed(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    (task_dir / "keep.txt").write_text("keep", encoding="utf-8")
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )
    monkeypatch.setattr(
        task_fs_gc_module,
        "_supports_secure_recursive_delete",
        lambda: False,
    )
    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
    )

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert (task_dir / "keep.txt").read_text(encoding="utf-8") == "keep"
    [entry] = collector.status().entries
    assert entry.state == "quarantined"
    assert entry.last_error_code == "_UnsafeTaskFilesystemTarget"


def test_windows_platform_routes_task_tree_cleanup_to_handle_bound_delete(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )
    calls = []

    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        platform_name="nt",
        windows_delete_target=lambda root, relative_path: calls.append(
            (root, relative_path)
        ),
    )

    report = collector.sweep(force=True)

    assert report.deleted == 1
    assert calls == [(workspace / "tasks", task.id)]
    assert collector.status().entries == ()


@pytest.mark.parametrize(
    "failure",
    [
        UnsafeWindowsTaskTreeDeleteError("reparse task tree"),
        WindowsTaskTreeDeleteUnsupportedError("native API unavailable"),
    ],
)
def test_windows_unsafe_or_unsupported_tree_is_quarantined(tmp_path, failure):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    sentinel = task_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )

    def fail_delete(_root, _relative_path):
        raise failure

    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        platform_name="nt",
        windows_delete_target=fail_delete,
    )

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    [entry] = collector.status().entries
    assert entry.state == "quarantined"
    assert entry.last_error_code == "_UnsafeTaskFilesystemTarget"


@pytest.mark.parametrize(
    "failure",
    [
        WindowsTaskTreeDeleteRetryableError(32, "sharing or lock conflict"),
        OSError(145, "directory not empty"),
    ],
)
def test_windows_transient_tree_delete_failure_stays_retryable(tmp_path, failure):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    sentinel = task_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )

    def fail_delete(_root, _relative_path):
        raise failure

    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
        platform_name="nt",
        windows_delete_target=fail_delete,
    )

    report = collector.sweep(force=True)

    assert report.deferred == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    [entry] = collector.status().entries
    assert entry.state == "retry_wait"
    assert entry.attempt_count == 1
    assert entry.next_attempt_at > FIXED_NOW


def test_corrupt_case_variant_target_is_quarantined_without_deleting_tree(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    target = workspace / "tasks" / "task-a"
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, last_error_code, last_error_message,
                origin_task_id, origin_task_created_at, enqueued_at, updated_at
            ) VALUES (
                'task_dir', 'task-a', 'pending', 0, ?, NULL, NULL,
                'Task-A', ?, ?, ?
            )
            """,
            (timestamp, timestamp, timestamp, timestamp),
        )
    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
    )

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
    [entry] = collector.status().entries
    assert entry.state == "quarantined"
    assert "typed owner" in entry.last_error_message


def test_posix_delete_uses_exact_canonical_owner_spelling(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO task_fs_gc_queue(
                target_type, relative_path, state, attempt_count,
                next_attempt_at, last_error_code, last_error_message,
                origin_task_id, origin_task_created_at, enqueued_at, updated_at
            ) VALUES (
                'task_dir', 'Task-A', 'pending', 0, ?, NULL, NULL,
                'Task-A', ?, ?, ?
            )
            """,
            (timestamp, timestamp, timestamp, timestamp),
        )
    deleted_names = []
    monkeypatch.setattr(
        task_fs_gc_module,
        "_supports_secure_recursive_delete",
        lambda: True,
    )
    monkeypatch.setattr(task_fs_gc_module.os, "open", lambda *_args, **_kwargs: 10)
    monkeypatch.setattr(
        task_fs_gc_module.os,
        "stat",
        lambda *_args, **_kwargs: type("DirectoryStat", (), {"st_mode": stat.S_IFDIR})(),
    )
    monkeypatch.setattr(
        task_fs_gc_module.shutil,
        "rmtree",
        lambda name, *, dir_fd: deleted_names.append((name, dir_fd)),
    )
    monkeypatch.setattr(task_fs_gc_module.os, "fsync", lambda _fd: None)
    monkeypatch.setattr(task_fs_gc_module.os, "close", lambda _fd: None)
    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
        platform_name="posix",
    )

    report = collector.sweep(force=True)

    assert report.deleted == 1
    assert deleted_names == [("Task-A", 10)]


def test_explicit_requeue_recovers_one_legacy_windows_unsupported_target(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = _task(repo, tmp_path)
    task_dir = workspace / "tasks" / task.id
    task_dir.mkdir(parents=True)
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )
    available = False
    deleted = []

    def windows_delete(root, relative_path):
        if not available:
            raise WindowsTaskTreeDeleteUnsupportedError(
                "secure descriptor-relative recursive deletion is unavailable"
            )
        deleted.append((root, relative_path))

    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
        platform_name="nt",
        windows_delete_target=windows_delete,
    )
    first = collector.sweep(force=True)
    assert first.quarantined == 1

    available = True
    assert collector.requeue_quarantined("task_dir", task.id) is True
    [pending] = collector.status().entries
    assert pending.state == "pending"
    assert pending.attempt_count == 0
    assert pending.last_error_code is None
    assert pending.last_error_message is None

    second = collector.sweep(force=True)

    assert second.deleted == 1
    assert deleted == [(workspace / "tasks", task.id)]
    assert collector.status().entries == ()
    assert collector.requeue_quarantined("task_dir", task.id) is False


def test_requeue_quarantined_is_exact_single_target_and_rejects_unsafe_path(tmp_path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(db_path) as conn:
        for task_id in ("Task-A", "Task-B"):
            conn.execute(
                """
                INSERT INTO task_fs_gc_queue(
                    target_type, relative_path, state, attempt_count,
                    next_attempt_at, last_error_code, last_error_message,
                    origin_task_id, origin_task_created_at, enqueued_at, updated_at
                ) VALUES (
                    'task_dir', ?, 'quarantined', 3, ?, 'LegacyUnsupported',
                    'secure delete unavailable', ?, ?, ?, ?
                )
                """,
                (task_id, timestamp, task_id, timestamp, timestamp, timestamp),
            )
    collector = TaskFilesystemGarbageCollector(
        db_path,
        tasks_root=workspace / "tasks",
        datasets_root=workspace / "datasets",
        material_uploads_root=workspace / "material_uploads",
        clock=lambda: FIXED_NOW,
    )

    assert collector.requeue_quarantined("task_dir", "Task-A") is True

    with connect(db_path) as conn:
        states = {
            str(row["relative_path"]): str(row["state"])
            for row in conn.execute(
                "SELECT relative_path, state FROM task_fs_gc_queue"
            ).fetchall()
        }
    assert states == {"Task-A": "pending", "Task-B": "quarantined"}
    with pytest.raises(ValueError, match="safe canonical path"):
        collector.requeue_quarantined("task_dir", "../Task-B")
    with pytest.raises(ValueError, match="unsupported task filesystem target"):
        collector.requeue_quarantined("unknown", "Task-B")


def test_create_app_sweeps_persisted_task_filesystem_cleanup_candidates(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = _task(repo, tmp_path)
    task_dir = settings.tasks_dir / task.id
    task_dir.mkdir(parents=True)
    (task_dir / "stale.txt").write_text("stale", encoding="utf-8")
    repo.purge_task(
        task.id,
        task_fs_targets=(TaskFilesystemGcTarget("task_dir", task.id),),
    )

    app = create_app(settings)
    try:
        assert app.state.task_filesystem_gc_startup_report.deleted == 1
        assert app.state.task_filesystem_gc.status().entries == ()
        assert isinstance(
            app.state.task_filesystem_gc_watchdog,
            TaskFilesystemGcWatchdog,
        )
        assert not task_dir.exists()
    finally:
        app.state.task_filesystem_gc_watchdog.stop()
        app.state.dataset_source_gc_watchdog.stop()
        app.state.job_watchdog.stop()


def test_task_filesystem_gc_watchdog_runs_bounded_periodic_sweeps():
    called = threading.Event()
    calls: list[dict] = []

    class RecordingCollector:
        def sweep(self, **kwargs):
            calls.append(kwargs)
            called.set()
            return TaskFilesystemGcSweepReport(0, 0, 0, 0, 0)

    watchdog = TaskFilesystemGcWatchdog(
        RecordingCollector(),
        interval_seconds=0.01,
        sweep_limit=19,
    )
    try:
        watchdog.start()
        assert called.wait(timeout=1)
    finally:
        assert watchdog.stop() is True

    assert calls
    assert calls[0] == {"limit": 19}
