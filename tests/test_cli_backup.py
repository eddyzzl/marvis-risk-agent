import io
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile

import pytest

from marvis import __main__
from marvis import backup as backup_module
from marvis.backup import BackupError, create_backup, read_manifest, restore_backup
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.settings import build_settings


def _seed_workspace(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = TaskRepository(settings.db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A-card",
            model_version="v1",
            validator="validator",
            source_dir=str(settings.workspace),
            algorithm="lgb",
            run_mode="manual",
            target_col="bad_flag",
            score_col="score",
            split_col="split",
            time_col="apply_month",
            feature_columns=[],
            notebook_path=None,
            sample_path=None,
            pmml_path=None,
            dictionary_path=None,
            report_values={},
        )
    )
    task_dir = settings.tasks_dir / task.id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "note.txt").write_text("hello world", encoding="utf-8")
    # A large-ish dataset artifact that should be excluded by default.
    (settings.datasets_dir / "big.parquet").write_bytes(b"pretend-parquet-bytes")
    # A DuckDB scratch dir that should never be archived.
    duckdb_tmp = settings.workspace / ".duckdb_tmp"
    duckdb_tmp.mkdir(parents=True, exist_ok=True)
    (duckdb_tmp / "scratch.tmp").write_text("junk", encoding="utf-8")
    return settings, task


def test_create_backup_excludes_datasets_and_duckdb_tmp_by_default(tmp_path):
    settings, task = _seed_workspace(tmp_path)
    output_path = tmp_path / "backup.tar.gz"

    result = create_backup(settings.workspace, output_path)

    assert result.output_path == output_path
    assert output_path.exists()
    with tarfile.open(output_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "marvis.sqlite" in names
    assert f"tasks/{task.id}/note.txt" in names
    assert "manifest.json" in names
    assert not any(name.startswith("datasets/") for name in names)
    assert not any(".duckdb_tmp" in name for name in names)
    assert result.manifest["include_datasets"] is False


def test_create_backup_includes_datasets_when_requested(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    output_path = tmp_path / "backup.tar.gz"

    create_backup(settings.workspace, output_path, include_datasets=True)

    with tarfile.open(output_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "datasets/big.parquet" in names


def test_create_backup_rejects_output_inside_included_workspace_tree(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    output_path = settings.tasks_dir / "self-including-backup.tar.gz"

    with pytest.raises(BackupError, match="inside an included workspace directory"):
        create_backup(settings.workspace, output_path)
    assert not output_path.exists()


def test_create_backup_rejects_output_that_is_the_live_database(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    before = settings.db_path.read_bytes()

    with pytest.raises(BackupError, match="source database"):
        create_backup(settings.workspace, settings.db_path)

    assert settings.db_path.read_bytes() == before
    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]


def test_create_backup_rejects_hardlink_to_live_database(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    output_path = tmp_path / "db-hardlink.tar.gz"
    output_path.hardlink_to(settings.db_path)

    with pytest.raises(BackupError, match="source database"):
        create_backup(settings.workspace, output_path)

    with sqlite3.connect(settings.db_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchall() == [("ok",)]


def test_create_backup_rejects_symlinked_source_directory(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.txt").write_text("secret", encoding="utf-8")
    shutil.rmtree(settings.tasks_dir)
    settings.tasks_dir.symlink_to(external, target_is_directory=True)

    with pytest.raises(BackupError, match="source directory must not be a symlink"):
        create_backup(settings.workspace, tmp_path / "backup.tar.gz")


def test_create_backup_rejects_dangling_source_directory_symlink(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    shutil.rmtree(settings.tasks_dir)
    settings.tasks_dir.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(BackupError, match="source directory must not be a symlink"):
        create_backup(settings.workspace, tmp_path / "backup.tar.gz")


def test_create_backup_rejects_symlinked_source_file(tmp_path):
    settings, task = _seed_workspace(tmp_path)
    external = tmp_path / "secret.txt"
    external.write_text("secret", encoding="utf-8")
    (settings.tasks_dir / task.id / "linked.txt").symlink_to(external)

    with pytest.raises(BackupError, match="source file must not be a symlink"):
        create_backup(settings.workspace, tmp_path / "backup.tar.gz")


def test_create_backup_preserves_existing_archive_when_write_fails(
    tmp_path,
    monkeypatch,
):
    settings, _task = _seed_workspace(tmp_path)
    output_path = tmp_path / "backup.tar.gz"
    output_path.write_bytes(b"previous-backup")

    def fail_add(*_args, **_kwargs):
        raise OSError("simulated archive failure")

    monkeypatch.setattr(backup_module.tarfile.TarFile, "add", fail_add)

    with pytest.raises(OSError, match="simulated archive failure"):
        create_backup(settings.workspace, output_path)

    assert output_path.read_bytes() == b"previous-backup"

def test_create_backup_uses_sqlite_online_backup_api_captures_uncommitted_wal(tmp_path):
    """GAP-9: a naive `cp` of a WAL-mode db can miss transactions still sitting
    in the -wal file; the online backup API must not."""
    settings, _task = _seed_workspace(tmp_path)
    # Insert a second task and commit, but never checkpoint/close -- the write
    # stays in the WAL file, not the main db file, until something checkpoints it.
    conn = sqlite3.connect(settings.db_path, isolation_level="DEFERRED")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        INSERT INTO tasks(
            id, task_type, model_name, model_version, validator, source_dir,
            algorithm, run_mode, target_col, score_col, split_col, time_col,
            feature_columns_json, notebook_path, sample_path, pmml_path,
            dictionary_path, report_values_json, report_values_revision,
            status, status_message, status_reason_code, created_at, updated_at,
            target_type, recipes_json, sample_weight_col, oot_ks_min,
            metrics_json, capability_tier
        ) VALUES (
            'wal-task', 'validation', 'B', 'v1', 'v', '.', 'lgb', 'manual',
            'y', 's', 'split', 't', '[]', NULL, NULL, NULL, NULL, '{}', 0,
            'created', 'created', '', '2026-01-01', '2026-01-01', '', '[]',
            '', NULL, '[]', ''
        )
        """
    )
    conn.commit()
    try:
        output_path = tmp_path / "backup.tar.gz"
        create_backup(settings.workspace, output_path)
        restored_dir = tmp_path / "restored"
        restore_backup(output_path, restored_dir)
        restored_repo = TaskRepository(restored_dir / "marvis.sqlite")
        restored_ids = {task.id for task in restored_repo.list_tasks()}
    finally:
        conn.close()

    assert "wal-task" in restored_ids


def test_restore_backup_round_trips_tasks_and_files(tmp_path):
    settings, task = _seed_workspace(tmp_path)
    output_path = tmp_path / "backup.tar.gz"
    create_backup(settings.workspace, output_path)

    restored_dir = tmp_path / "restored"
    manifest = restore_backup(output_path, restored_dir)

    assert manifest["file_count"] == 2
    restored_repo = TaskRepository(restored_dir / "marvis.sqlite")
    restored_ids = {restored_task.id for restored_task in restored_repo.list_tasks()}
    assert task.id in restored_ids
    assert (restored_dir / "tasks" / task.id / "note.txt").read_text(encoding="utf-8") == "hello world"
    assert not (restored_dir / "datasets" / "big.parquet").exists()


def test_restore_backup_rejects_nonempty_target_without_force(tmp_path):
    settings, _task = _seed_workspace(tmp_path)
    output_path = tmp_path / "backup.tar.gz"
    create_backup(settings.workspace, output_path)

    restored_dir = tmp_path / "restored"
    restored_dir.mkdir()
    (restored_dir / "existing.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(BackupError, match="not empty"):
        restore_backup(output_path, restored_dir)

    restore_backup(output_path, restored_dir, force=True)
    assert not (restored_dir / "existing.txt").exists()


def test_restore_backup_force_preserves_existing_workspace_when_extraction_fails(tmp_path):
    archive_path = tmp_path / "broken-backup.tar.gz"
    manifest = {
        "marvis_version": "test",
        "created_at": "2026-07-10T00:00:00+00:00",
        "include_datasets": False,
        "file_count": 1,
        "files": ["tasks/replacement.txt"],
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))

        replacement_bytes = b"replacement"
        replacement_info = tarfile.TarInfo(name="tasks/replacement.txt")
        replacement_info.size = len(replacement_bytes)
        archive.addfile(replacement_info, io.BytesIO(replacement_bytes))

        invalid_link = tarfile.TarInfo(name="unsafe-link")
        invalid_link.type = tarfile.SYMTYPE
        invalid_link.linkname = "/etc/passwd"
        archive.addfile(invalid_link)

    target_workspace = tmp_path / "workspace"
    (target_workspace / "nested").mkdir(parents=True)
    (target_workspace / "existing.txt").write_bytes(b"keep me")
    (target_workspace / "nested" / "binary.bin").write_bytes(b"\x00\xfforiginal")
    before = {
        path.relative_to(target_workspace).as_posix(): path.read_bytes()
        for path in target_workspace.rglob("*")
        if path.is_file()
    }

    with pytest.raises(tarfile.AbsoluteLinkError):
        restore_backup(archive_path, target_workspace, force=True)

    after = {
        path.relative_to(target_workspace).as_posix(): path.read_bytes()
        for path in target_workspace.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_restore_backup_rejects_unlisted_directory_symlink(tmp_path):
    archive_path = tmp_path / "symlink-backup.tar.gz"
    manifest = {
        "marvis_version": "test",
        "created_at": "2026-07-10T00:00:00+00:00",
        "include_datasets": False,
        "file_count": 1,
        "files": ["replacement.txt"],
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        replacement = b"replacement"
        replacement_info = tarfile.TarInfo(name="replacement.txt")
        replacement_info.size = len(replacement)
        archive.addfile(replacement_info, io.BytesIO(replacement))
        alias = tarfile.TarInfo(name="extra-alias")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "."
        archive.addfile(alias)

    target_workspace = tmp_path / "restored"
    with pytest.raises(BackupError, match="must not contain symlinks"):
        restore_backup(archive_path, target_workspace)

    assert not target_workspace.exists()


def test_restore_backup_rejects_manifest_listed_internal_symlink(tmp_path):
    archive_path = tmp_path / "listed-symlink-backup.tar.gz"
    manifest = {
        "marvis_version": "test",
        "created_at": "2026-07-10T00:00:00+00:00",
        "include_datasets": False,
        "file_count": 2,
        "files": ["target.txt", "alias.txt"],
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        target = b"target"
        target_info = tarfile.TarInfo(name="target.txt")
        target_info.size = len(target)
        archive.addfile(target_info, io.BytesIO(target))
        alias = tarfile.TarInfo(name="alias.txt")
        alias.type = tarfile.SYMTYPE
        alias.linkname = "target.txt"
        archive.addfile(alias)

    target_workspace = tmp_path / "restored"
    with pytest.raises(BackupError, match="must not contain symlinks"):
        restore_backup(archive_path, target_workspace)

    assert not target_workspace.exists()


def test_restore_backup_force_rolls_back_if_interrupted_after_original_move(
    tmp_path,
    monkeypatch,
):
    settings, _task = _seed_workspace(tmp_path)
    archive_path = tmp_path / "backup.tar.gz"
    create_backup(settings.workspace, archive_path)
    target_workspace = tmp_path / "restored"
    target_workspace.mkdir()
    (target_workspace / "existing.txt").write_text("keep me", encoding="utf-8")

    original_replace = type(target_workspace).replace

    def interrupting_replace(self, destination):
        result = original_replace(self, destination)
        if self == target_workspace and Path(destination).name == "previous-workspace":
            raise KeyboardInterrupt("interrupted after moving original")
        return result

    monkeypatch.setattr(type(target_workspace), "replace", interrupting_replace)

    with pytest.raises(KeyboardInterrupt, match="interrupted after moving original"):
        restore_backup(archive_path, target_workspace, force=True)

    assert (target_workspace / "existing.txt").read_text(encoding="utf-8") == "keep me"


def test_restore_backup_without_force_rechecks_target_before_swap(tmp_path, monkeypatch):
    settings, _task = _seed_workspace(tmp_path)
    archive_path = tmp_path / "backup.tar.gz"
    create_backup(settings.workspace, archive_path)
    target_workspace = tmp_path / "restored"
    original_validate = backup_module._validate_staged_workspace

    def populate_target_during_staging(staged_workspace, manifest):
        original_validate(staged_workspace, manifest)
        target_workspace.mkdir()
        (target_workspace / "concurrent.txt").write_text("do not replace", encoding="utf-8")

    monkeypatch.setattr(
        backup_module,
        "_validate_staged_workspace",
        populate_target_during_staging,
    )

    with pytest.raises(BackupError, match="not empty"):
        restore_backup(archive_path, target_workspace, force=False)

    assert (target_workspace / "concurrent.txt").read_text(encoding="utf-8") == "do not replace"


def test_restore_backup_force_preserves_concurrent_target_and_original(tmp_path, monkeypatch):
    settings, _task = _seed_workspace(tmp_path)
    archive_path = tmp_path / "backup.tar.gz"
    create_backup(settings.workspace, archive_path)
    target_workspace = tmp_path / "restored"
    target_workspace.mkdir()
    (target_workspace / "existing.txt").write_text("keep original", encoding="utf-8")
    original_replace = type(target_workspace).replace

    def concurrent_replace(self, destination):
        if self.name == "workspace" and Path(destination) == target_workspace:
            target_workspace.mkdir()
            (target_workspace / "concurrent.txt").write_text(
                "keep concurrent",
                encoding="utf-8",
            )
        return original_replace(self, destination)

    monkeypatch.setattr(type(target_workspace), "replace", concurrent_replace)

    with pytest.raises(BackupError, match="original workspace remains"):
        restore_backup(archive_path, target_workspace, force=True)

    assert (target_workspace / "concurrent.txt").read_text(encoding="utf-8") == "keep concurrent"
    preserved_originals = list(
        tmp_path.glob(".restored.restore-*/previous-workspace/existing.txt")
    )
    assert len(preserved_originals) == 1
    assert preserved_originals[0].read_text(encoding="utf-8") == "keep original"
    shutil.rmtree(preserved_originals[0].parents[1])


def test_read_manifest_rejects_non_backup_archive(tmp_path):
    bogus = tmp_path / "not-a-backup.tar.gz"
    with tarfile.open(bogus, "w:gz") as archive:
        info = tarfile.TarInfo(name="hello.txt")
        data = b"hi"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    with pytest.raises(BackupError, match="manifest.json"):
        read_manifest(bogus)


def test_create_backup_raises_on_missing_workspace(tmp_path):
    with pytest.raises(BackupError, match="workspace not found"):
        create_backup(tmp_path / "does-not-exist", tmp_path / "out.tar.gz")


# --- CLI wiring ---------------------------------------------------------


def test_cli_backup_dispatches_create_backup(tmp_path, monkeypatch):
    import types

    calls = []

    def fake_create_backup(workspace, output_path, *, include_datasets=False):
        calls.append((workspace, output_path, include_datasets))
        return types.SimpleNamespace(
            output_path=output_path,
            manifest={"file_count": 3, "include_datasets": include_datasets, "marvis_version": "0.0.0"},
        )

    monkeypatch.setattr("marvis.backup.create_backup", fake_create_backup)

    __main__.main(
        [
            "backup",
            "--workspace",
            str(tmp_path / "workspace"),
            "--out",
            str(tmp_path / "out.tar.gz"),
        ]
    )

    assert len(calls) == 1
    workspace, output_path, include_datasets = calls[0]
    assert workspace == (tmp_path / "workspace").resolve()
    assert output_path == tmp_path / "out.tar.gz"
    assert include_datasets is False


def test_cli_restore_dispatches_restore_backup(tmp_path, monkeypatch):
    calls = []

    def fake_restore_backup(archive_path, target_workspace, *, force=False):
        calls.append((archive_path, target_workspace, force))
        return {"file_count": 1, "created_at": "2026-01-01", "marvis_version": "0.0.0"}

    monkeypatch.setattr("marvis.backup.restore_backup", fake_restore_backup)

    __main__.main(
        [
            "restore",
            str(tmp_path / "backup.tar.gz"),
            "--workspace",
            str(tmp_path / "workspace"),
            "--force",
        ]
    )

    assert calls == [(tmp_path / "backup.tar.gz", tmp_path / "workspace", True)]


def test_cli_help_lists_backup_and_restore_subcommands(capsys):
    with pytest.raises(SystemExit) as exc:
        __main__.main(["--help"])

    assert exc.value.code == 0
    stdout = capsys.readouterr().out
    assert "backup" in stdout
    assert "restore" in stdout
