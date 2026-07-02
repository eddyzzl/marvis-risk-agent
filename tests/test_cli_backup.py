import sqlite3
import tarfile

import pytest

from marvis import __main__
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


def test_read_manifest_rejects_non_backup_archive(tmp_path):
    bogus = tmp_path / "not-a-backup.tar.gz"
    with tarfile.open(bogus, "w:gz") as archive:
        info = tarfile.TarInfo(name="hello.txt")
        data = b"hi"
        info.size = len(data)
        import io

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
