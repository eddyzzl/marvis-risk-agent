from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest

from marvis.db import SCHEMA_VERSION, TaskRepository, init_db
from marvis.data.task_filesystem_gc import TaskFilesystemGcTarget
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VALIDATION_BATCH,
    TaskCreate,
)
from marvis.repositories.validation_batches import (
    ValidationBatchRepository,
    ValidationBatchStateConflict,
)


def _child_payload(tmp_path: Path, name: str) -> TaskCreate:
    source_dir = tmp_path / name
    source_dir.mkdir()
    paths = {
        "notebook_path": source_dir / f"{name}.ipynb",
        "sample_path": source_dir / f"{name}.csv",
        "pmml_path": source_dir / f"{name}.pmml",
        "dictionary_path": source_dir / f"{name}.xlsx",
    }
    for path in paths.values():
        path.touch()
    return TaskCreate(
        task_type=TASK_TYPE_VALIDATION,
        model_name=name,
        model_version="v1",
        validator="qa",
        source_dir=str(source_dir),
        run_mode="agent",
        **{key: str(value) for key, value in paths.items()},
    )


def test_validation_batch_migration_creates_durable_tables(tmp_path: Path):
    db_path = tmp_path / "marvis.sqlite"

    init_db(db_path)

    assert SCHEMA_VERSION >= 25
    with TaskRepository(db_path).transaction() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {"validation_batches", "validation_batch_items"} <= tables


def test_create_batch_is_atomic_and_hides_child_tasks_from_ui_listing(tmp_path: Path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    batch_repo = ValidationBatchRepository(db_path)
    parent_payload = TaskCreate(
        task_type=TASK_TYPE_VALIDATION_BATCH,
        model_name="2026年8月模型验证批次",
        model_version="",
        validator="qa",
        source_dir=str(tmp_path),
        run_mode="agent",
    )

    batch = batch_repo.create_batch(
        parent_payload,
        [_child_payload(tmp_path, "模型A"), _child_payload(tmp_path, "模型B")],
    )

    task_repo = TaskRepository(db_path)
    parent = task_repo.get_task(batch.parent_task_id)
    items = batch_repo.list_items(batch.parent_task_id)
    assert parent.task_type == TASK_TYPE_VALIDATION_BATCH
    assert batch.status == "created"
    assert batch.item_count == 2
    assert [item.ordinal for item in items] == [1, 2]
    assert [item.status for item in items] == ["queued", "queued"]
    assert all(
        task_repo.get_task(item.child_task_id).validation_workflow_version == 2
        for item in items
    )
    assert [task.id for task in task_repo.list_tasks(include_batch_children=False)] == [
        parent.id
    ]
    assert len(task_repo.list_tasks(include_batch_children=True)) == 3


def test_create_batch_does_not_re_read_after_committed_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    batch_repo = ValidationBatchRepository(db_path)

    def fail_post_commit_read(_parent_task_id: str):
        raise RuntimeError("synthetic post-commit batch read failure")

    monkeypatch.setattr(batch_repo, "get_batch", fail_post_commit_read)

    batch = batch_repo.create_batch(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION_BATCH,
            model_name="批次",
            model_version="",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        ),
        [_child_payload(tmp_path, "模型A")],
    )

    assert batch.status == "created"
    assert batch.item_count == 1
    assert ValidationBatchRepository(db_path).get_batch(batch.parent_task_id) == batch


def test_batch_requires_one_to_ten_standard_validation_children(tmp_path: Path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    batch_repo = ValidationBatchRepository(db_path)
    parent_payload = TaskCreate(
        task_type=TASK_TYPE_VALIDATION_BATCH,
        model_name="批次",
        model_version="",
        validator="qa",
        source_dir=str(tmp_path),
        run_mode="agent",
    )

    with pytest.raises(ValueError, match="1 to 10"):
        batch_repo.create_batch(parent_payload, [])


def test_batch_item_state_tracks_stage_metrics_and_manual_review(tmp_path: Path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    batch_repo = ValidationBatchRepository(db_path)
    batch = batch_repo.create_batch(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION_BATCH,
            model_name="批次",
            model_version="",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        ),
        [_child_payload(tmp_path, "模型A")],
    )
    [item] = batch_repo.list_items(batch.parent_task_id)

    batch_repo.update_item(
        item.id,
        status="review_required",
        stage="completed",
        oot_ks=0.31,
        oot_psi=0.12,
        pmml_status="pass",
        stress_risk="medium",
        report_complete=True,
        outcome="manual_review",
        error_message="稳定性 PSI 达到预警阈值",
    )

    updated = batch_repo.get_item(item.id)
    assert updated.status == "review_required"
    assert updated.oot_ks == pytest.approx(0.31)
    assert updated.oot_psi == pytest.approx(0.12)
    assert updated.report_complete is True
    assert updated.outcome == "manual_review"
    assert updated.error_message == "稳定性 PSI 达到预警阈值"

    reopened = batch_repo.update_item(
        item.id,
        status="running",
        stage="metrics",
        reopened=True,
        clear_results=True,
    )
    assert reopened.oot_ks is None
    assert reopened.oot_psi is None
    assert reopened.pmml_status == ""
    assert reopened.stress_risk == ""
    assert reopened.report_complete is False
    assert reopened.outcome == ""
    assert reopened.error_message == ""


def test_batch_start_claim_allows_only_one_concurrent_runner(tmp_path: Path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    repo = ValidationBatchRepository(db_path)
    batch = repo.create_batch(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION_BATCH,
            model_name="批次",
            model_version="",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        ),
        [_child_payload(tmp_path, "模型A")],
    )

    def claim() -> str:
        try:
            _record, previous = repo.claim_start(batch.parent_task_id)
        except ValidationBatchStateConflict:
            return "conflict"
        return f"claimed:{previous}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _index: claim(), range(2)))

    assert outcomes == ["claimed:created", "conflict"]
    assert repo.get_batch(batch.parent_task_id).status == "running"


def test_batch_start_claim_does_not_re_read_after_committed_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    repo = ValidationBatchRepository(db_path)
    batch = repo.create_batch(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION_BATCH,
            model_name="批次",
            model_version="",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        ),
        [_child_payload(tmp_path, "模型A")],
    )

    def fail_post_commit_read(_parent_task_id: str):
        raise RuntimeError("synthetic post-commit batch read failure")

    monkeypatch.setattr(repo, "get_batch", fail_post_commit_read)

    claimed, previous_status = repo.claim_start(batch.parent_task_id)

    assert previous_status == "created"
    assert claimed.status == "running"
    assert (
        ValidationBatchRepository(db_path).get_batch(batch.parent_task_id)
        == claimed
    )


def test_batch_family_purge_rolls_back_tasks_gc_and_audit_together(tmp_path: Path):
    workspace = tmp_path / "workspace"
    db_path = workspace / "marvis.sqlite"
    init_db(db_path)
    source = (
        workspace
        / "material_uploads"
        / "validation-batches"
        / "0123456789abcdef0123456789abcdef"
    )
    source.mkdir(parents=True)
    repo = ValidationBatchRepository(db_path)
    batch = repo.create_batch(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION_BATCH,
            model_name="批次",
            model_version="",
            validator="qa",
            source_dir=str(source),
            run_mode="agent",
        ),
        [_child_payload(tmp_path, "模型A")],
        parent_source_relative_path=(
            "validation-batches/0123456789abcdef0123456789abcdef"
        ),
    )
    [item] = repo.list_items(batch.parent_task_id)
    task_repo = TaskRepository(db_path)
    with task_repo.transaction() as conn:
        conn.execute(
            """
            CREATE TRIGGER abort_batch_task_delete
            BEFORE DELETE ON tasks
            BEGIN
                SELECT RAISE(ABORT, 'synthetic batch purge rollback');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic batch purge rollback"):
        repo.purge_batch(
            batch.parent_task_id,
            task_fs_targets_by_task={
                item.child_task_id: (
                    TaskFilesystemGcTarget("task_dir", item.child_task_id),
                ),
                batch.parent_task_id: (
                    TaskFilesystemGcTarget("task_dir", batch.parent_task_id),
                    TaskFilesystemGcTarget(
                        "validation_batch_source_dir",
                        "validation-batches/0123456789abcdef0123456789abcdef",
                    ),
                ),
            },
        )

    assert task_repo.get_task(batch.parent_task_id)
    assert task_repo.get_task(item.child_task_id)
    assert repo.get_batch(batch.parent_task_id).item_count == 1
    with task_repo.transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_fs_gc_queue"
        ).fetchone()[0] == 0
        assert conn.execute(
            """
            SELECT COUNT(*) FROM audit
             WHERE kind IN ('task.delete', 'validation_batch.delete')
            """
        ).fetchone()[0] == 0
