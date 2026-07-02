from pathlib import Path

from marvis.app import create_app
from marvis.artifacts.recovery import reconcile_workspace_artifacts
from marvis.db import PluginRepository, TaskRepository, init_db
from marvis.domain import TASK_TYPE_VALIDATION, TaskCreate
from marvis.plugins.loader import compute_checksum
from marvis.plugins.manifest import PluginManifest, ToolSpec
from marvis.settings import build_settings


def test_reconcile_workspace_artifacts_removes_staging_dirs(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    staging = settings.datasets_dir / "task-1" / "joins" / ".staging"
    staging.mkdir(parents=True)
    (staging / "partial.parquet").write_bytes(b"partial")

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_staging_dirs == 1
    assert report.removed_staged_entries == 1
    assert not staging.exists()


def test_reconcile_workspace_artifacts_restores_plugin_backup_when_db_not_committed(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    final_dir = settings.plugins_dir / "sample_pack"
    backup_dir = settings.plugins_dir / ".sample_pack.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bak"
    _write_plugin_dir(final_dir, marker="new")
    _write_plugin_dir(backup_dir, marker="old")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 1
    assert (final_dir / "tools.py").read_text(encoding="utf-8") == "VERSION = 'old'\n"
    assert not backup_dir.exists()


def test_reconcile_workspace_artifacts_removes_plugin_backup_when_db_matches_final(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    final_dir = settings.plugins_dir / "sample_pack"
    backup_dir = settings.plugins_dir / ".sample_pack.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bak"
    _write_plugin_dir(final_dir, marker="new")
    _write_plugin_dir(backup_dir, marker="old")
    PluginRepository(settings.db_path).upsert_plugin(
        _plugin_manifest(checksum=compute_checksum(final_dir)),
        enabled=True,
    )

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_backups == 1
    assert (final_dir / "tools.py").read_text(encoding="utf-8") == "VERSION = 'new'\n"
    assert not backup_dir.exists()


def test_reconcile_workspace_artifacts_removes_uncommitted_handoff_backup_and_dir(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    handoff_root = settings.tasks_dir / "task-1" / "validation_handoff"
    final_dir = handoff_root / "artifact_1"
    backup_dir = handoff_root / ".artifact_1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bak"
    _write_handoff_dir(final_dir, marker="new")
    _write_handoff_dir(backup_dir, marker="old")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 1
    assert report.removed_orphan_dirs == 1
    assert not final_dir.exists()
    assert not backup_dir.exists()


def test_reconcile_workspace_artifacts_removes_handoff_backup_when_task_committed(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    handoff_root = settings.tasks_dir / "task-1" / "validation_handoff"
    final_dir = handoff_root / "artifact_1"
    backup_dir = handoff_root / ".artifact_1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bak"
    _write_handoff_dir(final_dir, marker="new")
    _write_handoff_dir(backup_dir, marker="old")
    TaskRepository(settings.db_path).create_task(
        TaskCreate(
            task_type=TASK_TYPE_VALIDATION,
            model_name="model",
            model_version="artifact_1",
            validator="MARVIS",
            source_dir=str(final_dir.resolve()),
        )
    )

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_backups == 1
    assert (final_dir / "marker.txt").read_text(encoding="utf-8") == "new\n"
    assert not backup_dir.exists()


def test_reconcile_workspace_artifacts_removes_uncommitted_handoff_dir(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    final_dir = settings.tasks_dir / "task-1" / "validation_handoff" / "artifact_1"
    _write_handoff_dir(final_dir, marker="orphan")

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_orphan_dirs == 1
    assert not final_dir.exists()


def test_create_app_runs_artifact_recovery(tmp_path: Path):
    settings = build_settings(tmp_path)
    staging = settings.tasks_dir / "task-1" / "outputs" / ".staging"
    staging.mkdir(parents=True)
    (staging / "partial.xlsx").write_bytes(b"partial")

    app = create_app(settings)

    assert app.state.artifact_recovery_report["removed_staging_dirs"] == 1
    assert not staging.exists()


# REL-7: file-level staging backups (".<name>.<hex32>.bak") and orphaned atomic
# write scratch files (".<name>.<hex32>.tmp") left under the datasets/tasks
# roots by an interrupted ArtifactUnitOfWork.promote_all()/RemovedPath.remove()
# or files.write_text_atomic() must be reconciled the same way the plugins and
# validation_handoff directory backups already are.


def test_reconcile_workspace_artifacts_restores_dataset_file_backup(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    dataset_dir = settings.datasets_dir / "task-1"
    dataset_dir.mkdir(parents=True)
    final_path = dataset_dir / "upload_abcd1234.parquet"
    backup_path = dataset_dir / ".upload_abcd1234.parquet.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.bak"
    final_path.write_bytes(b"new-bytes-not-yet-committed")
    backup_path.write_bytes(b"old-bytes-last-committed")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 1
    assert final_path.read_bytes() == b"old-bytes-last-committed"
    assert not backup_path.exists()


def test_reconcile_workspace_artifacts_restores_task_root_file_backup(tmp_path: Path):
    # Mirrors persist_model_meta()'s uow.stage_file(out_dir, "model_meta.json")
    # under tasks_dir/<task>/modeling_artifacts, i.e. outside validation_handoff.
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    out_dir = settings.tasks_dir / "task-1" / "modeling_artifacts"
    out_dir.mkdir(parents=True)
    final_path = out_dir / "model_meta.json"
    backup_path = out_dir / ".model_meta.json.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.bak"
    final_path.write_text('{"artifact_id": "new"}', encoding="utf-8")
    backup_path.write_text('{"artifact_id": "old"}', encoding="utf-8")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 1
    assert final_path.read_text(encoding="utf-8") == '{"artifact_id": "old"}'
    assert not backup_path.exists()


def test_reconcile_workspace_artifacts_restores_task_root_dir_backup(tmp_path: Path):
    # Mirrors uow.remove_path(task_dir / "outputs") in pipeline.py: RemovedPath
    # backs up a whole directory as ".outputs.<hex32>.bak" directly under the
    # per-task root, one level above validation_handoff.
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task_dir = settings.tasks_dir / "task-1"
    final_dir = task_dir / "outputs"
    backup_dir = task_dir / ".outputs.cccccccccccccccccccccccccccccccc.bak"
    final_dir.mkdir(parents=True)
    (final_dir / "validation.xlsx").write_bytes(b"new-report")
    backup_dir.mkdir(parents=True)
    (backup_dir / "validation.xlsx").write_bytes(b"old-report")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 1
    assert (final_dir / "validation.xlsx").read_bytes() == b"old-report"
    assert not backup_dir.exists()


def test_reconcile_workspace_artifacts_removes_orphan_dataset_tmp_file(tmp_path: Path):
    # Mirrors marvis.files.write_text_atomic being interrupted before the
    # final os.replace(): the scratch ".tmp" sibling must be deleted, and
    # since the real target was never touched by the interrupted write, it is
    # left completely alone.
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task_dir = settings.tasks_dir / "task-1"
    task_dir.mkdir(parents=True)
    final_path = task_dir / "execution_environment.json"
    tmp_path_file = task_dir / ".execution_environment.json.dddddddddddddddddddddddddddddddd.tmp"
    final_path.write_text('{"ok": true}', encoding="utf-8")
    tmp_path_file.write_text('{"partial": ', encoding="utf-8")

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_orphan_tmp_files == 1
    assert not tmp_path_file.exists()
    assert final_path.read_text(encoding="utf-8") == '{"ok": true}'


def test_reconcile_workspace_artifacts_removes_orphan_tmp_file_with_no_final_target(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    dataset_dir = settings.datasets_dir / "task-1"
    dataset_dir.mkdir(parents=True)
    tmp_path_file = dataset_dir / ".new_upload.parquet.eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.tmp"
    tmp_path_file.write_bytes(b"partial-parquet-bytes")

    report = reconcile_workspace_artifacts(settings)

    assert report.removed_orphan_tmp_files == 1
    assert not tmp_path_file.exists()


def test_reconcile_workspace_artifacts_datasets_tasks_backup_reconcile_is_idempotent(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    dataset_dir = settings.datasets_dir / "task-1"
    dataset_dir.mkdir(parents=True)
    final_path = dataset_dir / "upload_abcd1234.parquet"
    backup_path = dataset_dir / ".upload_abcd1234.parquet.ffffffffffffffffffffffffffffffff.bak"
    final_path.write_bytes(b"new-bytes")
    backup_path.write_bytes(b"old-bytes")
    tmp_file = dataset_dir / ".stray.parquet.11111111111111111111111111111111.tmp"
    tmp_file.write_bytes(b"scratch")

    first = reconcile_workspace_artifacts(settings)
    second = reconcile_workspace_artifacts(settings)

    assert first.restored_backups == 1
    assert first.removed_orphan_tmp_files == 1
    assert second.restored_backups == 0
    assert second.removed_orphan_tmp_files == 0
    assert final_path.read_bytes() == b"old-bytes"
    assert not backup_path.exists()
    assert not tmp_file.exists()


def test_reconcile_workspace_artifacts_leaves_committed_dataset_files_untouched(tmp_path: Path):
    # No .bak/.tmp present: an ordinary, fully-committed dataset file must not
    # be renamed, deleted, or otherwise disturbed by the new generic pass.
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    dataset_dir = settings.datasets_dir / "task-1"
    dataset_dir.mkdir(parents=True)
    final_path = dataset_dir / "upload_abcd1234.parquet"
    final_path.write_bytes(b"committed-bytes")

    report = reconcile_workspace_artifacts(settings)

    assert report.restored_backups == 0
    assert report.removed_orphan_tmp_files == 0
    assert final_path.read_bytes() == b"committed-bytes"


def _write_plugin_dir(path: Path, *, marker: str) -> None:
    path.mkdir(parents=True)
    (path / "__init__.py").write_text("", encoding="utf-8")
    (path / "tools.py").write_text(f"VERSION = {marker!r}\n", encoding="utf-8")


def _write_handoff_dir(path: Path, *, marker: str) -> None:
    path.mkdir(parents=True)
    (path / "model.pmml").write_text("<PMML />\n", encoding="utf-8")
    (path / "dictionary.csv").write_text("feature\nx1\n", encoding="utf-8")
    (path / "scoring_notebook.ipynb").write_text("{}\n", encoding="utf-8")
    (path / "marker.txt").write_text(f"{marker}\n", encoding="utf-8")


def _plugin_manifest(*, checksum: str) -> PluginManifest:
    return PluginManifest(
        name="sample_pack",
        version="0.1.0",
        display_name="Sample Pack",
        description="sample",
        module="sample_pack.tools",
        python_requires=">=3.10,<3.14",
        tools=(
            ToolSpec(
                name="echo",
                summary="echo",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                determinism="deterministic",
                timeout_seconds=5,
                failure_policy="fail",
                side_effects=(),
                entrypoint="tool_echo",
            ),
        ),
        checksum=checksum,
    )
