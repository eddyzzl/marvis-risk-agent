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
