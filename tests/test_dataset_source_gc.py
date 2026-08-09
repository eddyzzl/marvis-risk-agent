from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
import stat
import threading

import pytest
from fastapi.testclient import TestClient

import marvis.data.dataset_source_gc as dataset_source_gc_module
from marvis.app import create_app
from marvis.data.contracts import Dataset
from marvis.data.dataset_source_gc import (
    DatasetSourceGarbageCollector,
    DatasetSourceGcSweepReport,
    DatasetSourceGcWatchdog,
)
from marvis.data.windows_safe_delete import UnsafeWindowsDatasetDeleteError
from marvis.db import DatasetRepository, TaskRepository, connect, init_db
from marvis.domain import TaskCreate
from marvis.settings import build_settings


FIXED_NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _create_task(repo: TaskRepository, *, name: str = "task"):
    return repo.create_task(
        TaskCreate(
            model_name=name,
            model_version="v1",
            validator="qa",
            source_dir=str(repo.db_path.parent),
        )
    )


def _register_source(settings, task_id: str, *, source_path: str) -> Dataset:
    path = settings.datasets_dir / source_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"parquet-bytes")
    dataset = Dataset(
        id=f"dataset-{task_id}",
        task_id=task_id,
        role="sample",
        source_path=source_path,
        format="parquet",
        sheet=None,
        row_count=1,
        columns=(),
        has_target=False,
        target_col=None,
        created_at="2026-08-01T12:00:00Z",
        content_hash="a" * 64,
    )
    DatasetRepository(settings.db_path).create_dataset(dataset)
    return dataset


def _purge(settings, task_id: str) -> dict:
    return TaskRepository(settings.db_path).purge_task(
        task_id,
        validate_dataset_source_path=lambda value: (
            settings.datasets_dir / value
        ).relative_to(settings.datasets_dir),
    )


def test_task_purge_atomically_records_dataset_source_cleanup_candidate(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )

    summary = _purge(settings, task.id)

    assert summary["dataset_source_paths"] == [dataset.source_path]
    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    status = collector.status()
    assert status.pending_count == 1
    assert status.quarantined_count == 0
    assert status.entries[0].source_path == dataset.source_path
    assert status.entries[0].state == "pending"
    assert (settings.datasets_dir / dataset.source_path).is_file()


def test_dataset_source_reference_recheck_uses_source_path_index(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)

    with connect(settings.db_path) as connection:
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(datasets)").fetchall()
        }
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM datasets WHERE source_path = ?",
            ("task/sample.parquet",),
        ).fetchall()

    assert "idx_datasets_source_path_task" in indexes
    assert any(
        "idx_datasets_source_path_task" in str(row[3])
        for row in plan
    )


def test_task_purge_rollback_does_not_leave_dataset_source_cleanup_candidate(
    tmp_path,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    with connect(settings.db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_dataset_delete_for_gc_test
            BEFORE DELETE ON datasets
            BEGIN
                SELECT RAISE(ABORT, 'synthetic purge failure');
            END
            """
        )

    with pytest.raises(Exception, match="synthetic purge failure"):
        _purge(settings, task.id)

    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    assert collector.status().entries == ()
    assert TaskRepository(settings.db_path).get_task(task.id).id == task.id


def test_gc_retries_failed_file_delete_from_persisted_state(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    path = settings.datasets_dir / dataset.source_path

    def sharing_violation(_path):
        raise PermissionError("synthetic Windows sharing violation")

    failing = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
        delete_source=sharing_violation,
    )
    first = failing.sweep(force=True)

    assert first.deferred == 1
    assert first.deleted == 0
    assert path.is_file()
    deferred = failing.status().entries[0]
    assert deferred.state == "retry_wait"
    assert deferred.attempt_count == 1
    assert deferred.last_error_code == "PermissionError"
    assert deferred.next_attempt_at > FIXED_NOW

    restarted = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW + timedelta(days=1),
    )
    second = restarted.sweep()

    assert second.deleted == 1
    assert second.deferred == 0
    assert restarted.status().entries == ()
    assert not path.exists()


def test_gc_routes_windows_cleanup_through_handle_bound_delete(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    calls = []

    monkeypatch.setattr(
        dataset_source_gc_module,
        "_is_windows_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        dataset_source_gc_module,
        "delete_dataset_file_windows",
        lambda path, root: calls.append((path, root)),
    )

    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    report = collector.sweep(force=True)

    assert report.deleted == 1
    assert calls == [
        (
            settings.datasets_dir / dataset.source_path,
            settings.datasets_dir,
        )
    ]


def test_gc_quarantines_unsafe_windows_handle_target(tmp_path, monkeypatch):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)

    def reject_unsafe_target(_path, _root):
        raise UnsafeWindowsDatasetDeleteError("synthetic reparse target")

    monkeypatch.setattr(
        dataset_source_gc_module,
        "_is_windows_platform",
        lambda: True,
    )
    monkeypatch.setattr(
        dataset_source_gc_module,
        "delete_dataset_file_windows",
        reject_unsafe_target,
    )

    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    report = collector.sweep(force=True)

    assert report.quarantined == 1
    entry = collector.status().entries[0]
    assert entry.last_error_code == "_UnsafeDatasetSourcePath"
    assert (settings.datasets_dir / dataset.source_path).is_file()


def test_gc_cancels_cleanup_when_source_is_referenced_again(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task_repo = TaskRepository(settings.db_path)
    owner = _create_task(task_repo, name="owner")
    dataset = _register_source(
        settings,
        owner.id,
        source_path=f"{owner.id}/sample.parquet",
    )
    _purge(settings, owner.id)
    consumer = _create_task(task_repo, name="consumer")
    DatasetRepository(settings.db_path).create_dataset(
        replace(
            dataset,
            id=f"dataset-{consumer.id}",
            task_id=consumer.id,
        )
    )

    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    report = collector.sweep(force=True)

    assert report.cancelled_referenced == 1
    assert report.deleted == 0
    assert collector.status().entries == ()
    assert (settings.datasets_dir / dataset.source_path).is_file()


def test_create_app_sweeps_persisted_dataset_source_cleanup_candidates(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    path = settings.datasets_dir / dataset.source_path
    assert path.is_file()

    app = create_app(settings)
    try:
        assert app.state.dataset_source_gc_startup_report.deleted == 1
        assert app.state.dataset_source_gc.status().entries == ()
        assert isinstance(app.state.dataset_source_gc_watchdog, DatasetSourceGcWatchdog)
        assert not path.exists()
    finally:
        app.state.dataset_source_gc_watchdog.stop()
        app.state.job_watchdog.stop()


def test_delete_task_api_consumes_durable_cleanup_candidate(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    app = create_app(settings)
    try:
        task = _create_task(TaskRepository(settings.db_path))
        dataset = _register_source(
            settings,
            task.id,
            source_path=f"{task.id}/sample.parquet",
        )
        path = settings.datasets_dir / dataset.source_path
        client = TestClient(app)

        response = client.delete(f"/api/tasks/{task.id}")

        assert response.status_code == 204
        assert app.state.dataset_source_gc.status().entries == ()
        assert not path.exists()
    finally:
        app.state.dataset_source_gc_watchdog.stop()
        app.state.job_watchdog.stop()


def test_dataset_source_gc_watchdog_runs_bounded_periodic_sweeps():
    called = threading.Event()
    calls: list[dict] = []

    class RecordingCollector:
        def sweep(self, **kwargs):
            calls.append(kwargs)
            called.set()

    watchdog = DatasetSourceGcWatchdog(
        RecordingCollector(),
        interval_seconds=0.01,
        sweep_limit=17,
    )
    try:
        watchdog.start()
        assert called.wait(timeout=1)
    finally:
        watchdog.stop()

    assert calls
    assert calls[0] == {"limit": 17}


def test_dataset_source_gc_watchdog_can_restart_after_clean_stop():
    called = threading.Event()
    calls = []

    class RecordingCollector:
        def sweep(self, **kwargs):
            calls.append(kwargs)
            called.set()
            return DatasetSourceGcSweepReport(0, 0, 0, 0, 0)

    watchdog = DatasetSourceGcWatchdog(
        RecordingCollector(),
        interval_seconds=0.01,
    )
    watchdog.start()
    assert called.wait(timeout=1)
    assert watchdog.stop() is True
    assert watchdog.status().alive is False

    called.clear()
    watchdog.start()
    assert called.wait(timeout=1)
    assert watchdog.stop() is True

    assert len(calls) >= 2


def test_dataset_source_gc_watchdog_status_surfaces_last_error_and_recovers():
    called = threading.Event()

    class FailingCollector:
        def sweep(self, **kwargs):
            called.set()
            raise RuntimeError("synthetic periodic failure")

    watchdog = DatasetSourceGcWatchdog(
        FailingCollector(),
        interval_seconds=0.01,
    )
    try:
        watchdog.start()
        assert called.wait(timeout=1)
        status = watchdog.status()
        for _ in range(100):
            if status.last_finished_at is not None:
                break
            threading.Event().wait(0.01)
            status = watchdog.status()
        assert status.alive is True
        assert status.last_started_at is not None
        assert status.last_finished_at is not None
        assert status.last_report is None
        assert status.last_error == "RuntimeError: synthetic periodic failure"
    finally:
        assert watchdog.stop() is True


def test_gc_rejects_invalid_source_path_items_with_public_validation_error(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )

    with pytest.raises(ValueError, match="non-empty strings"):
        collector.sweep(source_paths=[["not-hashable"]])


def test_gc_removes_empty_ordinary_dataset_directory_after_file_delete(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    source_directory = (settings.datasets_dir / dataset.source_path).parent

    report = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)

    assert report.deleted == 1
    assert not source_directory.exists()


def test_gc_quarantines_corrupt_escape_candidate_without_touching_outside_file(
    tmp_path,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    outside = settings.datasets_dir.parent / "outside.parquet"
    outside.write_bytes(b"must-survive")
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_source_gc_queue(
                source_path, state, attempt_count, next_attempt_at,
                last_error_code, last_error_message, origin_task_id,
                enqueued_at, updated_at
            ) VALUES (?, 'pending', 0, ?, NULL, NULL, ?, ?, ?)
            """,
            ("../outside.parquet", timestamp, "corrupt-row", timestamp, timestamp),
        )
    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert outside.read_bytes() == b"must-survive"
    status = collector.status()
    assert status.pending_count == 0
    assert status.quarantined_count == 1
    assert status.entries[0].state == "quarantined"
    assert status.entries[0].last_error_code == "_UnsafeDatasetSourcePath"


@pytest.mark.skipif(
    os.unlink not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor-relative symlink-race coverage requires POSIX dir_fd support",
)
def test_gc_does_not_follow_parent_swapped_to_symlink_during_delete(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    source_path = settings.datasets_dir / dataset.source_path
    original_parent = source_path.parent
    parked_parent = settings.datasets_dir / "parked-source"
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir()
    outside_file = outside_parent / source_path.name
    outside_file.write_bytes(b"must-survive-race")

    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )
    secure_delete = collector._delete_source_file

    def swap_parent_then_delete(path):
        original_parent.rename(parked_parent)
        original_parent.symlink_to(outside_parent, target_is_directory=True)
        secure_delete(path)

    collector._delete_source = swap_parent_then_delete

    report = collector.sweep(force=True)

    assert report.quarantined == 1
    assert outside_file.read_bytes() == b"must-survive-race"
    assert (parked_parent / source_path.name).read_bytes() == b"parquet-bytes"


def test_gc_quarantines_malformed_timestamp_without_blocking_valid_due_entry(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    monkeypatch.setattr("marvis.repositories.tasks._now", lambda: timestamp)
    _purge(settings, task.id)
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_source_gc_queue(
                source_path, state, attempt_count, next_attempt_at,
                last_error_code, last_error_message, origin_task_id,
                enqueued_at, updated_at
            ) VALUES (?, 'pending', 0, '!', NULL, NULL, ?, ?, ?)
            """,
            ("corrupt/timestamp.parquet", "corrupt-row", timestamp, timestamp),
        )
    collector = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    )

    report = collector.sweep(limit=10)

    assert report.examined == 2
    assert report.quarantined == 1
    assert report.deleted == 1
    assert not (settings.datasets_dir / dataset.source_path).exists()
    status = collector.status()
    assert status.pending_count == 0
    assert status.quarantined_count == 1
    assert status.entries[0].next_attempt_at == FIXED_NOW
    assert status.entries[0].last_error_code == "_CorruptDatasetSourceGcState"


def test_gc_status_tolerates_corrupt_persisted_timestamps_before_sweep(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_source_gc_queue(
                source_path, state, attempt_count, next_attempt_at,
                last_error_code, last_error_message, origin_task_id,
                enqueued_at, updated_at
            ) VALUES (?, 'pending', 0, 'not-a-time', NULL, NULL, ?, 'also-bad', ?)
            """,
            ("corrupt/status.parquet", "corrupt-row", timestamp),
        )

    entry = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).status().entries[0]

    assert entry.next_attempt_at is None
    assert entry.enqueued_at is None
    assert entry.updated_at == FIXED_NOW
    assert entry.last_error_code == "CorruptDatasetSourceGcTimestamp"
    assert "next_attempt_at" in entry.last_error_message
    assert "enqueued_at" in entry.last_error_message


@pytest.mark.skipif(
    os.unlink not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="directory fsync ordering is implemented by the POSIX safe-delete path",
)
def test_gc_fsyncs_file_parent_and_removed_directory_parent_before_queue_commit(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    fsynced_descriptors = []
    monkeypatch.setattr(
        dataset_source_gc_module.os,
        "fsync",
        lambda descriptor: fsynced_descriptors.append(descriptor),
    )

    report = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)

    assert report.deleted == 1
    assert len(fsynced_descriptors) == 2
    assert not (settings.datasets_dir / dataset.source_path).exists()


@pytest.mark.skipif(
    os.unlink not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="descriptor identity coverage requires POSIX dir_fd support",
)
def test_gc_does_not_rmdir_replacement_directory_after_open_parent_is_renamed(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = _create_task(TaskRepository(settings.db_path))
    dataset = _register_source(
        settings,
        task.id,
        source_path=f"{task.id}/sample.parquet",
    )
    _purge(settings, task.id)
    original_parent = (settings.datasets_dir / dataset.source_path).parent
    parked_parent = settings.datasets_dir / "parked-after-open"
    swapped = False

    def swap_on_first_directory_sync(_descriptor):
        nonlocal swapped
        if swapped:
            return
        swapped = True
        original_parent.rename(parked_parent)
        original_parent.mkdir()

    monkeypatch.setattr(
        dataset_source_gc_module.os,
        "fsync",
        swap_on_first_directory_sync,
    )

    report = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)

    assert report.deleted == 1
    assert original_parent.is_dir()
    assert parked_parent.is_dir()


@pytest.mark.skipif(
    os.unlink not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"),
    reason="CAS mode hardening coverage requires POSIX dir_fd support",
)
def test_gc_rehardens_surviving_cas_directory_after_interrupted_writable_mode(
    tmp_path,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    digest = "a" * 64
    source_path = f"_cas/{digest}/{digest}.parquet"
    digest_dir = settings.datasets_dir / "_cas" / digest
    digest_dir.mkdir(parents=True)
    (digest_dir / "unrelated.marker").write_bytes(b"keep")
    digest_dir.chmod(0o700)
    timestamp = FIXED_NOW.isoformat().replace("+00:00", "Z")
    with connect(settings.db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_source_gc_queue(
                source_path, state, attempt_count, next_attempt_at,
                last_error_code, last_error_message, origin_task_id,
                enqueued_at, updated_at
            ) VALUES (?, 'pending', 0, ?, NULL, NULL, ?, ?, ?)
            """,
            (source_path, timestamp, "interrupted-cas", timestamp, timestamp),
        )

    report = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)

    assert report.deleted == 1
    assert digest_dir.is_dir()
    assert stat.S_IMODE(digest_dir.stat().st_mode) == 0o555


def test_gc_treats_lexical_source_path_alias_as_an_existing_reference(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task_repo = TaskRepository(settings.db_path)
    owner = _create_task(task_repo, name="owner")
    consumer = _create_task(task_repo, name="consumer")
    canonical_path = f"{owner.id}/sample.parquet"
    owner_dataset = _register_source(
        settings,
        owner.id,
        source_path=canonical_path,
    )
    DatasetRepository(settings.db_path).create_dataset(
        replace(
            owner_dataset,
            id=f"dataset-{consumer.id}",
            task_id=consumer.id,
        )
    )
    aliased_path = f"{owner.id}/./sample.parquet"
    with connect(settings.db_path) as connection:
        connection.execute(
            "UPDATE datasets SET source_path = ? WHERE id = ?",
            (aliased_path, owner_dataset.id),
        )

    summary = _purge(settings, owner.id)
    report = DatasetSourceGarbageCollector(
        settings.db_path,
        settings.datasets_dir,
        clock=lambda: FIXED_NOW,
    ).sweep(force=True)

    assert summary["dataset_source_paths"] == [aliased_path]
    assert report.cancelled_referenced == 1
    assert report.deleted == 0
    assert (settings.datasets_dir / canonical_path).read_bytes() == b"parquet-bytes"
