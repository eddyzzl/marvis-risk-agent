import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from marvis.artifacts.transactional import (
    ArtifactTransactionError,
    ArtifactUnitOfWork,
    TransactionalArtifactStore,
    TransactionalDirectoryStore,
)


def test_staged_artifact_promotes_atomically_and_commit_keeps_final(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")
    artifact = store.stage("report.txt")
    artifact.path.write_text("ok", encoding="utf-8")

    final_path = artifact.promote()
    artifact.commit()

    assert final_path == tmp_path / "artifacts" / "report.txt"
    assert final_path.read_text(encoding="utf-8") == "ok"
    assert not artifact.path.exists()
    artifact.rollback()
    assert final_path.exists()


def test_staged_artifact_rollback_removes_stage_and_promoted_file(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")
    staged = store.stage("model.pkl")
    staged.path.write_bytes(b"model")

    final_path = staged.promote()
    assert final_path.exists()

    staged.rollback()

    assert not staged.path.exists()
    assert not final_path.exists()
    assert not (tmp_path / "artifacts" / ".staging").exists()


def test_staged_artifact_rollback_restores_existing_file(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    existing = root / "model_meta.json"
    existing.write_text("old", encoding="utf-8")
    store = TransactionalArtifactStore(root)
    staged = store.stage("model_meta.json")
    staged.path.write_text("new", encoding="utf-8")

    staged.promote()
    assert existing.read_text(encoding="utf-8") == "new"

    staged.rollback()

    assert existing.read_text(encoding="utf-8") == "old"
    assert not staged.backup_path.exists()
    assert not (root / ".staging").exists()


def test_promoting_one_file_keeps_sibling_staged_files(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")
    first = store.stage("artifact.model_meta.json")
    second = store.stage("model_meta.json")
    first.path.write_text("first", encoding="utf-8")
    second.path.write_text("second", encoding="utf-8")

    first.promote()

    assert first.final_path.read_text(encoding="utf-8") == "first"
    assert second.path.exists()
    assert second.path.read_text(encoding="utf-8") == "second"
    assert (tmp_path / "artifacts" / ".staging").exists()

    second.promote()
    first.commit()
    second.commit()

    assert second.final_path.read_text(encoding="utf-8") == "second"
    assert not (tmp_path / "artifacts" / ".staging").exists()


def test_transactional_artifact_store_rejects_absolute_path_outside_root(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactTransactionError, match="must stay under"):
        store.stage(tmp_path / "outside.txt")


def test_transactional_artifact_store_rejects_relative_path_traversal(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactTransactionError, match="must stay under"):
        store.stage("../outside.txt")


def test_transactional_artifact_store_cleans_orphan_staged_files(tmp_path: Path):
    store = TransactionalArtifactStore(tmp_path / "artifacts")
    staged = store.stage("orphan.parquet")
    staged.path.write_bytes(b"partial")

    assert store.cleanup_orphans() == 1
    assert not staged.path.exists()
    assert not (tmp_path / "artifacts" / ".staging").exists()


def test_staged_directory_commit_replaces_target_and_removes_backup(tmp_path: Path):
    root = tmp_path / "plugins"
    existing = root / "sample_pack"
    existing.mkdir(parents=True)
    (existing / "tools.py").write_text("old", encoding="utf-8")
    store = TransactionalDirectoryStore(root)
    staged = store.stage("sample_pack")
    staged.path.mkdir(parents=True)
    (staged.path / "tools.py").write_text("new", encoding="utf-8")

    final_path = staged.activate()
    staged.commit()

    assert final_path == existing
    assert (existing / "tools.py").read_text(encoding="utf-8") == "new"
    assert not staged.backup_path.exists()
    assert not (root / ".staging").exists()


def test_staged_directory_rollback_restores_existing_target(tmp_path: Path):
    root = tmp_path / "plugins"
    existing = root / "sample_pack"
    existing.mkdir(parents=True)
    (existing / "tools.py").write_text("old", encoding="utf-8")
    store = TransactionalDirectoryStore(root)
    staged = store.stage("sample_pack")
    staged.path.mkdir(parents=True)
    (staged.path / "tools.py").write_text("new", encoding="utf-8")

    staged.activate()
    staged.rollback()

    assert (existing / "tools.py").read_text(encoding="utf-8") == "old"
    assert not staged.backup_path.exists()
    assert not (root / ".staging").exists()


def test_transactional_directory_store_rejects_relative_path_traversal(tmp_path: Path):
    store = TransactionalDirectoryStore(tmp_path / "plugins")

    with pytest.raises(ArtifactTransactionError, match="must stay under"):
        store.stage("../outside")


def test_artifact_unit_of_work_commits_after_callback_succeeds(tmp_path: Path):
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(tmp_path / "artifacts", "joined.parquet")
    artifact.path.write_bytes(b"joined")
    calls = []

    result = uow.finalize(lambda: calls.append("db") or "dataset-1")

    assert result == "dataset-1"
    assert calls == ["db"]
    assert (tmp_path / "artifacts" / "joined.parquet").read_bytes() == b"joined"
    artifact.rollback()
    assert (tmp_path / "artifacts" / "joined.parquet").exists()


def test_artifact_unit_of_work_commits_artifacts_after_db_context_succeeds(tmp_path: Path):
    db_path = tmp_path / "app.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE events(id TEXT PRIMARY KEY)")
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(tmp_path / "artifacts", "joined.parquet")
    artifact.path.write_bytes(b"joined")

    def insert_event(conn):
        conn.execute("INSERT INTO events(id) VALUES ('evt-1')")
        return "ok"

    result = uow.finalize_with_connection(
        lambda: sqlite3.connect(db_path),
        insert_event,
    )

    assert result == "ok"
    assert (tmp_path / "artifacts" / "joined.parquet").read_bytes() == b"joined"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM events").fetchall() == [("evt-1",)]
    artifact.rollback()
    assert (tmp_path / "artifacts" / "joined.parquet").exists()


def test_artifact_unit_of_work_rolls_back_artifact_when_db_context_fails(tmp_path: Path):
    db_path = tmp_path / "app.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE events(id TEXT PRIMARY KEY)")
    root = tmp_path / "artifacts"
    root.mkdir()
    existing = root / "joined.parquet"
    existing.write_bytes(b"old")
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(root, "joined.parquet")
    artifact.path.write_bytes(b"new")

    @contextmanager
    def failing_commit():
        conn = sqlite3.connect(db_path)
        try:
            yield conn
            conn.rollback()
            raise RuntimeError("commit failed")
        finally:
            conn.close()

    with pytest.raises(RuntimeError, match="commit failed"):
        uow.finalize_with_connection(
            failing_commit,
            lambda conn: conn.execute("INSERT INTO events(id) VALUES ('evt-1')"),
        )

    assert existing.read_bytes() == b"old"
    assert not artifact.path.exists()
    assert not artifact.backup_path.exists()
    assert not (root / ".staging").exists()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM events").fetchall() == []


def test_artifact_unit_of_work_rolls_back_promoted_file_when_callback_fails(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    existing = root / "joined.parquet"
    existing.write_bytes(b"old")
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(root, "joined.parquet")
    artifact.path.write_bytes(b"new")

    def fail_db_write():
        raise RuntimeError("audit failed")

    with pytest.raises(RuntimeError, match="audit failed"):
        uow.finalize(fail_db_write)

    assert existing.read_bytes() == b"old"
    assert not artifact.path.exists()
    assert not artifact.backup_path.exists()
    assert not (root / ".staging").exists()


def test_artifact_unit_of_work_rolls_back_removed_file_when_callback_fails(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    existing = root / "validation_report.docx"
    existing.write_bytes(b"old-report")
    uow = ArtifactUnitOfWork()
    removal = uow.remove_path(existing)

    with pytest.raises(RuntimeError, match="status failed"):
        uow.finalize(lambda: (_ for _ in ()).throw(RuntimeError("status failed")))

    assert existing.read_bytes() == b"old-report"
    assert not removal.backup_path.exists()


def test_artifact_unit_of_work_commits_removed_directory(tmp_path: Path):
    root = tmp_path / "task"
    images = root / "images"
    images.mkdir(parents=True)
    (images / "old.png").write_bytes(b"old-image")
    uow = ArtifactUnitOfWork()
    removal = uow.remove_path(images)

    uow.finalize(lambda: None)

    assert not images.exists()
    assert not removal.backup_path.exists()


def test_artifact_unit_of_work_rolls_back_removed_directory_when_callback_fails(tmp_path: Path):
    root = tmp_path / "task"
    images = root / "images"
    images.mkdir(parents=True)
    (images / "old.png").write_bytes(b"old-image")
    uow = ArtifactUnitOfWork()
    removal = uow.remove_path(images)

    with pytest.raises(RuntimeError, match="status failed"):
        uow.finalize(lambda: (_ for _ in ()).throw(RuntimeError("status failed")))

    assert (images / "old.png").read_bytes() == b"old-image"
    assert not removal.backup_path.exists()


def test_artifact_unit_of_work_rollback_is_idempotent_after_finalize_failure(tmp_path: Path):
    root = tmp_path / "artifacts"
    root.mkdir()
    existing = root / "joined.parquet"
    existing.write_bytes(b"old")
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(root, "joined.parquet")
    artifact.path.write_bytes(b"new")

    def fail_db_write():
        raise RuntimeError("audit failed")

    with pytest.raises(RuntimeError, match="audit failed"):
        uow.finalize(fail_db_write)

    uow.rollback()

    assert existing.read_bytes() == b"old"
    assert not artifact.path.exists()
    assert not artifact.backup_path.exists()
    assert not (root / ".staging").exists()
