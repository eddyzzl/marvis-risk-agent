"""Startup reconciliation for staged artifact transactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import shutil

from marvis.db import PluginRepository, TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION
from marvis.plugins.loader import compute_checksum


_BACKUP_DIR_RE = re.compile(r"^\.(?P<name>[A-Za-z_][A-Za-z0-9_-]{0,127})\.[0-9a-f]{32}\.bak$")


@dataclass
class ArtifactRecoveryReport:
    removed_staging_dirs: int = 0
    removed_staged_entries: int = 0
    removed_backups: int = 0
    restored_backups: int = 0
    removed_orphan_dirs: int = 0
    skipped_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def reconcile_workspace_artifacts(settings) -> ArtifactRecoveryReport:
    """Clean or restore artifacts left behind by interrupted staged transactions."""

    report = ArtifactRecoveryReport()
    roots = _existing_roots([
        Path(settings.datasets_dir),
        Path(settings.tasks_dir),
        Path(settings.plugins_dir),
    ])
    for root in roots:
        _remove_staging_dirs(root, report)

    _reconcile_plugin_backups(Path(settings.plugins_dir), PluginRepository(settings.db_path), report)
    _reconcile_validation_handoff_dirs(Path(settings.tasks_dir), TaskRepository(settings.db_path), report)
    return report


def _existing_roots(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        result.append(path)
    return result


def _remove_staging_dirs(root: Path, report: ArtifactRecoveryReport) -> None:
    for staging_dir in sorted(root.rglob(".staging"), key=lambda item: len(item.parts), reverse=True):
        if not staging_dir.is_dir():
            continue
        try:
            report.removed_staged_entries += sum(1 for _item in staging_dir.rglob("*"))
            shutil.rmtree(staging_dir)
            report.removed_staging_dirs += 1
        except Exception as exc:  # pragma: no cover - defensive, filesystem dependent
            report.errors.append(f"failed to remove staging dir {staging_dir}: {exc}")


def _reconcile_plugin_backups(
    plugins_dir: Path,
    repo: PluginRepository,
    report: ArtifactRecoveryReport,
) -> None:
    if not plugins_dir.exists():
        return
    for backup_dir in sorted(path for path in plugins_dir.iterdir() if path.is_dir()):
        name = _backup_final_name(backup_dir)
        if name is None:
            continue
        final_dir = plugins_dir / name
        try:
            row = repo.get_plugin(name)
            if row is not None and final_dir.is_dir() and row.get("checksum") == compute_checksum(final_dir):
                shutil.rmtree(backup_dir)
                report.removed_backups += 1
            else:
                _restore_backup(backup_dir, final_dir)
                report.restored_backups += 1
        except Exception as exc:  # pragma: no cover - defensive, filesystem dependent
            report.errors.append(f"failed to reconcile plugin backup {backup_dir}: {exc}")


def _reconcile_validation_handoff_dirs(
    tasks_dir: Path,
    task_repo: TaskRepository,
    report: ArtifactRecoveryReport,
) -> None:
    if not tasks_dir.exists():
        return
    committed_dirs = _committed_validation_source_dirs(task_repo)
    for handoff_root in sorted(tasks_dir.glob("*/validation_handoff")):
        if not handoff_root.is_dir():
            continue
        _reconcile_handoff_backups(handoff_root, committed_dirs, report)
        _remove_uncommitted_handoff_dirs(handoff_root, committed_dirs, report)


def _committed_validation_source_dirs(task_repo: TaskRepository) -> set[Path]:
    result: set[Path] = set()
    for task in task_repo.list_tasks():
        if task.task_type != TASK_TYPE_VALIDATION:
            continue
        if not task.source_dir:
            continue
        result.add(Path(task.source_dir).resolve(strict=False))
    return result


def _reconcile_handoff_backups(
    handoff_root: Path,
    committed_dirs: set[Path],
    report: ArtifactRecoveryReport,
) -> None:
    for backup_dir in sorted(path for path in handoff_root.iterdir() if path.is_dir()):
        name = _backup_final_name(backup_dir)
        if name is None:
            continue
        final_dir = handoff_root / name
        try:
            if final_dir.resolve(strict=False) in committed_dirs and final_dir.is_dir():
                shutil.rmtree(backup_dir)
                report.removed_backups += 1
            else:
                _restore_backup(backup_dir, final_dir)
                report.restored_backups += 1
        except Exception as exc:  # pragma: no cover - defensive, filesystem dependent
            report.errors.append(f"failed to reconcile validation handoff backup {backup_dir}: {exc}")


def _remove_uncommitted_handoff_dirs(
    handoff_root: Path,
    committed_dirs: set[Path],
    report: ArtifactRecoveryReport,
) -> None:
    for path in sorted(item for item in handoff_root.iterdir() if item.is_dir()):
        if path.name.startswith(".") or path.name == ".staging":
            continue
        if path.resolve(strict=False) in committed_dirs:
            continue
        if not _looks_like_handoff_material_dir(path):
            report.skipped_paths.append(str(path))
            continue
        shutil.rmtree(path)
        report.removed_orphan_dirs += 1


def _looks_like_handoff_material_dir(path: Path) -> bool:
    return (
        (path / "model.pmml").is_file()
        and (path / "dictionary.csv").is_file()
        and (path / "scoring_notebook.ipynb").is_file()
    )


def _backup_final_name(path: Path) -> str | None:
    match = _BACKUP_DIR_RE.match(path.name)
    return match.group("name") if match else None


def _restore_backup(backup_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.rename(final_dir)
