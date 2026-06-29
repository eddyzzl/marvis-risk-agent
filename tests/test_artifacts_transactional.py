from pathlib import Path

import pytest

from marvis.artifacts.transactional import (
    ArtifactTransactionError,
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
