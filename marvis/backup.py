from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import contextmanager
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
    included_dirs = list(_DEFAULT_BACKUP_DIRS)
    if include_datasets:
        included_dirs.append("datasets")
    output_path = Path(output_path).resolve(strict=False)
    db_path = workspace / _BACKUP_DB_NAME
    sqlite_paths = [
        db_path,
        workspace / f"{_BACKUP_DB_NAME}-wal",
        workspace / f"{_BACKUP_DB_NAME}-shm",
        workspace / f"{_BACKUP_DB_NAME}-journal",
    ]
    if any(_paths_alias(output_path, source_path) for source_path in sqlite_paths):
        raise BackupError("backup output must not replace the source database or its sidecars")
    for dir_name in included_dirs:
        included_root = (workspace / dir_name).resolve(strict=False)
        try:
            output_path.relative_to(included_root)
        except ValueError:
            continue
        raise BackupError(
            "backup output must not be inside an included workspace directory: "
            f"{output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        _atomic_output_path(output_path) as staged_output_path,
        tempfile.TemporaryDirectory(prefix=".marvis_backup_") as scratch,
    ):
        scratch_dir = Path(scratch)
        staged_db = scratch_dir / _BACKUP_DB_NAME
        if db_path.exists():
            if db_path.is_symlink():
                raise BackupError("source database must not be a symlink")
            _backup_sqlite(db_path, staged_db)

        file_inventory: list[str] = []
        if staged_db.exists():
            file_inventory.append(_BACKUP_DB_NAME)

        with tarfile.open(staged_output_path, "w:gz") as archive:
            if staged_db.exists():
                archive.add(staged_db, arcname=_BACKUP_DB_NAME)
            for dir_name in included_dirs:
                source_dir = workspace / dir_name
                if source_dir.is_symlink():
                    raise BackupError(f"backup source directory must not be a symlink: {dir_name}")
                if not source_dir.is_dir():
                    continue
                for path in sorted(source_dir.rglob("*")):
                    if path.is_symlink():
                        raise BackupError(
                            f"backup source file must not be a symlink: "
                            f"{path.relative_to(workspace).as_posix()}"
                        )
                    if path.is_dir():
                        continue
                    try:
                        path.resolve().relative_to(workspace)
                    except ValueError as exc:
                        raise BackupError(
                            f"backup source file escapes workspace: {path}"
                        ) from exc
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


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second.resolve(strict=False):
        return True
    if not first.exists() or not second.exists():
        return False
    try:
        return first.samefile(second)
    except OSError:
        return False


@contextmanager
def _atomic_output_path(output_path: Path):
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    staged_path = Path(temp_name)
    try:
        yield staged_path
        with staged_path.open("rb") as staged_file:
            os.fsync(staged_file.fileno())
        os.replace(staged_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        staged_path.unlink(missing_ok=True)


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


def _validate_staged_workspace(staged_workspace: Path, manifest: dict) -> None:
    manifest_path = staged_workspace / _MANIFEST_NAME
    try:
        staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("restored manifest.json could not be read") from exc
    if staged_manifest != manifest or not isinstance(manifest, dict):
        raise BackupError("restored manifest.json does not match the backup manifest")

    files = manifest.get("files")
    file_count = manifest.get("file_count")
    if (
        not isinstance(files, list)
        or any(not isinstance(path, str) or not path for path in files)
        or len(set(files)) != len(files)
        or file_count != len(files)
    ):
        raise BackupError("backup manifest has an invalid file inventory")

    actual_files: list[str] = []
    for path in staged_workspace.rglob("*"):
        relative_path = path.relative_to(staged_workspace).as_posix()
        if path.is_symlink():
            raise BackupError("restored workspace must not contain symlinks")
        if path.is_file() and relative_path != _MANIFEST_NAME:
            actual_files.append(relative_path)
    if sorted(actual_files) != sorted(files):
        raise BackupError("restored files do not match the backup manifest inventory")

    db_path = staged_workspace / _BACKUP_DB_NAME
    if db_path.exists():
        try:
            connection = sqlite3.connect(db_path)
            try:
                check_rows = connection.execute("PRAGMA quick_check").fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise BackupError("restored marvis.sqlite failed integrity validation") from exc
        if check_rows != [("ok",)]:
            raise BackupError("restored marvis.sqlite failed integrity validation")


def _target_has_contents(path: Path) -> bool:
    if path.is_symlink():
        return True
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def restore_backup(
    archive_path: Path,
    target_workspace: Path,
    *,
    force: bool = False,
) -> dict:
    """Unpack a backup archive into target_workspace.

    target_workspace must not already exist (or must be empty), unless
    force=True. The archive is fully extracted and validated in a temporary
    directory on the target filesystem before a rename-based swap. The existing
    startup reconcile (reconcile_workspace_artifacts) remains responsible for
    cleaning up any leftover partial-write artifacts the next time the restored
    workspace is served.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise BackupError(f"backup archive not found: {archive_path}")
    manifest = read_manifest(archive_path)

    target_workspace = Path(target_workspace)
    if target_workspace.exists() or target_workspace.is_symlink():
        has_contents = _target_has_contents(target_workspace)
        if has_contents and not force:
            raise BackupError(
                f"target workspace {target_workspace} is not empty; pass --force to overwrite"
            )

    target_workspace.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_workspace.name}.restore-",
            dir=target_workspace.parent,
        )
    )
    preserve_scratch = False
    try:
        staged_workspace = scratch_dir / "workspace"
        staged_workspace.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(staged_workspace, filter="data")

        _validate_staged_workspace(staged_workspace, manifest)
        (staged_workspace / _MANIFEST_NAME).unlink()

        previous_workspace = scratch_dir / "previous-workspace"
        removed_empty_target = False
        try:
            if force:
                if target_workspace.exists() or target_workspace.is_symlink():
                    target_workspace.replace(previous_workspace)
            elif target_workspace.exists() or target_workspace.is_symlink():
                # Recheck immediately before the swap: extraction/validation can
                # take long enough for another process to create or populate the
                # destination after the initial preflight.
                if _target_has_contents(target_workspace):
                    raise BackupError(
                        f"target workspace {target_workspace} is not empty; "
                        "pass --force to overwrite"
                    )
                try:
                    target_workspace.rmdir()
                except OSError as exc:
                    raise BackupError(
                        f"target workspace {target_workspace} changed during restore"
                    ) from exc
                removed_empty_target = True
            staged_workspace.replace(target_workspace)
        except BaseException as swap_exc:
            # Infer whether the first rename happened from the filesystem rather
            # than a flag set after Path.replace() returns: KeyboardInterrupt can
            # arrive after the rename completed but before that flag assignment.
            if previous_workspace.exists() or previous_workspace.is_symlink():
                if target_workspace.exists() or target_workspace.is_symlink():
                    # The target may have been recreated or modified by another
                    # process. It is not ours to delete. Preserve both sides and
                    # surface their locations for explicit operator recovery.
                    preserve_scratch = True
                    raise BackupError(
                        "restore swap failed after the target path changed; "
                        f"the current target is preserved at {target_workspace} and "
                        f"the original workspace remains at {previous_workspace}"
                    ) from swap_exc
                try:
                    previous_workspace.replace(target_workspace)
                except BaseException as rollback_exc:
                    preserve_scratch = True
                    raise BackupError(
                        "restore swap failed and the original workspace could not be rolled back; "
                        f"it remains at {previous_workspace}"
                    ) from rollback_exc
            elif removed_empty_target and not target_workspace.exists():
                target_workspace.mkdir(parents=True, exist_ok=True)
            raise
    finally:
        if not preserve_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return manifest


__all__ = [
    "BackupError",
    "BackupResult",
    "create_backup",
    "read_manifest",
    "restore_backup",
]
