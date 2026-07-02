from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from marvis import __version__


class BackupError(RuntimeError):
    """Raised when a workspace backup or restore cannot be completed."""


# GAP-9: db_schema.py forces PRAGMA journal_mode=WAL (db_schema.py:~709), so a
# naive `cp workspace/marvis.sqlite backup/` while the service is running can
# silently miss the most recent transactions still sitting in the -wal file
# (SQLite only guarantees a plain file copy is complete once everything has
# been checkpointed back into the main db file). sqlite3.Connection.backup()
# uses SQLite's online backup API, which reads a transactionally consistent
# snapshot regardless of WAL/checkpoint state -- the correct way to copy a live
# database.
_BACKUP_DB_NAME = "marvis.sqlite"
_MANIFEST_NAME = "manifest.json"
# Directories that make up "the workspace" for backup purposes, excluding
# datasets/ (large, regenerable from source files, and explicitly opt-in via
# --include-datasets) and any cache/temp directories (.duckdb_tmp, __pycache__).
_DEFAULT_BACKUP_DIRS = ("tasks", "plugins", "report_templates", "branding", "settings")
_EXCLUDED_DIR_NAMES = {".duckdb_tmp", "__pycache__", ".excel_ingest_", ".xlsx_ingest_"}


@dataclass(frozen=True)
class BackupResult:
    output_path: Path
    manifest: dict


def _is_excluded(name: str) -> bool:
    if name in _EXCLUDED_DIR_NAMES:
        return True
    return name.startswith((".duckdb_tmp", ".excel_ingest_", ".xlsx_ingest_"))


def _backup_sqlite(db_path: Path, dest_path: Path) -> None:
    """Consistency snapshot via SQLite's online backup API (not a raw file copy)."""
    if not db_path.exists():
        raise BackupError(f"database not found: {db_path}")
    source = sqlite3.connect(db_path)
    try:
        dest = sqlite3.connect(dest_path)
        try:
            source.backup(dest)
        finally:
            dest.close()
    finally:
        source.close()


def create_backup(
    workspace: Path,
    output_path: Path,
    *,
    include_datasets: bool = False,
) -> BackupResult:
    """Create a consistent tar.gz snapshot of a workspace.

    The sqlite database is copied via the online backup API (safe to run while
    the service is live); tasks/plugins/report_templates/branding/settings are
    archived as-is. datasets/ (raw uploads + joined parquet, can be very large
    and is reproducible from source files) is excluded unless
    include_datasets=True. A manifest.json records the MARVIS version,
    timestamp, and file inventory for restore-time sanity checking.
    """
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise BackupError(f"workspace not found: {workspace}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    included_dirs = list(_DEFAULT_BACKUP_DIRS)
    if include_datasets:
        included_dirs.append("datasets")

    with tempfile.TemporaryDirectory(prefix=".marvis_backup_") as scratch:
        scratch_dir = Path(scratch)
        db_path = workspace / _BACKUP_DB_NAME
        staged_db = scratch_dir / _BACKUP_DB_NAME
        if db_path.exists():
            _backup_sqlite(db_path, staged_db)

        file_inventory: list[str] = []
        if staged_db.exists():
            file_inventory.append(_BACKUP_DB_NAME)

        with tarfile.open(output_path, "w:gz") as archive:
            if staged_db.exists():
                archive.add(staged_db, arcname=_BACKUP_DB_NAME)
            for dir_name in included_dirs:
                source_dir = workspace / dir_name
                if not source_dir.is_dir():
                    continue
                for path in sorted(source_dir.rglob("*")):
                    if path.is_dir():
                        continue
                    if any(_is_excluded(part) for part in path.relative_to(workspace).parts):
                        continue
                    arcname = path.relative_to(workspace).as_posix()
                    archive.add(path, arcname=arcname)
                    file_inventory.append(arcname)

            manifest = {
                "marvis_version": __version__,
                "created_at": datetime.now(UTC).isoformat(),
                "include_datasets": include_datasets,
                "file_count": len(file_inventory),
                "files": file_inventory,
            }
            manifest_path = scratch_dir / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            archive.add(manifest_path, arcname=_MANIFEST_NAME)

    return BackupResult(output_path=output_path, manifest=manifest)


def read_manifest(archive_path: Path) -> dict:
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(_MANIFEST_NAME)
        except KeyError as exc:
            raise BackupError(f"{archive_path} is not a MARVIS backup (missing manifest.json)") from exc
        extracted = archive.extractfile(member)
        if extracted is None:
            raise BackupError(f"{archive_path} manifest.json could not be read")
        return json.loads(extracted.read().decode("utf-8"))


def restore_backup(
    archive_path: Path,
    target_workspace: Path,
    *,
    force: bool = False,
) -> dict:
    """Unpack a backup archive into target_workspace.

    target_workspace must not already exist (or must be empty), unless
    force=True. The restored sqlite file and directory tree are extracted
    as-is; the existing startup reconcile (reconcile_workspace_artifacts) is
    responsible for cleaning up any leftover partial-write artifacts the next
    time the restored workspace is served, exactly as it does for a workspace
    that was not cleanly shut down.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise BackupError(f"backup archive not found: {archive_path}")
    manifest = read_manifest(archive_path)

    target_workspace = Path(target_workspace)
    if target_workspace.exists():
        has_contents = any(target_workspace.iterdir())
        if has_contents and not force:
            raise BackupError(
                f"target workspace {target_workspace} is not empty; pass --force to overwrite"
            )
        if has_contents and force:
            shutil.rmtree(target_workspace)
    target_workspace.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(target_workspace, filter="data")

    manifest_path = target_workspace / _MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()

    return manifest


__all__ = [
    "BackupError",
    "BackupResult",
    "create_backup",
    "read_manifest",
    "restore_backup",
]
