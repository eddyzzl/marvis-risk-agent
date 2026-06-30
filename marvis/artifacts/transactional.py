"""Transactional file staging for artifact-producing tools.

Tools write into a hidden staging directory, then atomically promote the final
artifact only after computation succeeds. Callers can roll back both staged and
promoted files if a later DB/audit write fails. When a repository exposes a
connection factory, ``ArtifactUnitOfWork`` can also own that DB transaction so
artifact promotion and DB commit share one boundary.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path


class ArtifactTransactionError(RuntimeError):
    """Raised when an artifact staging operation is invalid."""


@dataclass
class StagedArtifact:
    """One staged artifact with an intended final path."""

    stage_path: Path
    final_path: Path
    backup_path: Path
    _committed: bool = False
    _promoted: bool = False
    _had_backup: bool = False

    @property
    def path(self) -> Path:
        return self.stage_path

    def promote(self) -> Path:
        if self._promoted:
            return self.final_path
        if not self.stage_path.exists():
            raise ArtifactTransactionError(f"staged artifact does not exist: {self.stage_path}")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists() or self.final_path.is_symlink():
            if self.backup_path.exists() or self.backup_path.is_symlink():
                _remove_path(self.backup_path)
            self.final_path.rename(self.backup_path)
            self._had_backup = True
        try:
            self.stage_path.replace(self.final_path)
        except Exception:
            if self._had_backup and self.backup_path.exists() and not self.final_path.exists():
                self.backup_path.rename(self.final_path)
            raise
        self._promoted = True
        _remove_empty_parents(self.stage_path.parent)
        return self.final_path

    def commit(self) -> Path:
        if self.backup_path.exists() or self.backup_path.is_symlink():
            _remove_path(self.backup_path)
        self._committed = True
        return self.final_path if self._promoted else self.stage_path

    def rollback(self) -> None:
        if self._committed:
            return
        self.stage_path.unlink(missing_ok=True)
        if self._promoted:
            self.final_path.unlink(missing_ok=True)
            if self._had_backup and self.backup_path.exists():
                self.backup_path.rename(self.final_path)
        _remove_empty_parents(self.stage_path.parent)


@dataclass
class StagedDirectory:
    """One staged directory with an intended final directory path."""

    stage_path: Path
    final_path: Path
    backup_path: Path
    _committed: bool = False
    _activated: bool = False
    _had_backup: bool = False

    @property
    def path(self) -> Path:
        return self.stage_path

    def activate(self) -> Path:
        if self._activated:
            return self.final_path
        if not self.stage_path.is_dir():
            raise ArtifactTransactionError(f"staged directory does not exist: {self.stage_path}")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            if self.backup_path.exists():
                _remove_path(self.backup_path)
            self.final_path.rename(self.backup_path)
            self._had_backup = True
        try:
            self.stage_path.rename(self.final_path)
        except Exception:
            if self._had_backup and self.backup_path.exists() and not self.final_path.exists():
                self.backup_path.rename(self.final_path)
            raise
        self._activated = True
        _remove_empty_parents(self.stage_path.parent)
        return self.final_path

    def commit(self) -> Path:
        if self.backup_path.exists():
            _remove_path(self.backup_path)
        _remove_empty_parents(self.stage_path.parent)
        self._committed = True
        return self.final_path

    def rollback(self) -> None:
        if self._committed:
            return
        if self._activated:
            _remove_path(self.final_path)
            if self._had_backup and self.backup_path.exists():
                self.backup_path.rename(self.final_path)
        else:
            _remove_path(self.stage_path)
        _remove_empty_parents(self.stage_path.parent)


class TransactionalArtifactStore:
    """Stage files next to their destination and promote them atomically."""

    def __init__(self, root: Path, *, staging_dir_name: str = ".staging") -> None:
        self.root = Path(root)
        self.staging_dir = self.root / staging_dir_name

    def stage(self, final_name: str | Path) -> StagedArtifact:
        final_path = self._final_path(final_name)
        token = uuid.uuid4().hex
        stage_path = self.staging_dir / f"{final_path.stem}.{token}{final_path.suffix}"
        backup_path = self.root / f".{final_path.name}.{token}.bak"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return StagedArtifact(stage_path=stage_path, final_path=final_path, backup_path=backup_path)

    def cleanup_orphans(self) -> int:
        if not self.staging_dir.exists():
            return 0
        removed = 0
        for path in sorted(self.staging_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
                removed += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            self.staging_dir.rmdir()
        except OSError:
            pass
        return removed

    def _final_path(self, final_name: str | Path) -> Path:
        final = Path(final_name)
        candidate = final if final.is_absolute() else self.root / final
        try:
            candidate.resolve(strict=False).relative_to(self.root.resolve(strict=False))
        except ValueError as exc:
            raise ArtifactTransactionError(
                f"artifact path must stay under {self.root}: {final}"
            ) from exc
        return candidate


class TransactionalDirectoryStore:
    """Stage directories under a root and atomically activate them."""

    def __init__(self, root: Path, *, staging_dir_name: str = ".staging") -> None:
        self.root = Path(root)
        self.staging_dir = self.root / staging_dir_name

    def stage(self, final_name: str | Path) -> StagedDirectory:
        final_path = self._final_path(final_name)
        token = uuid.uuid4().hex
        stage_path = self.staging_dir / f"{final_path.name}.{token}"
        backup_path = self.root / f".{final_path.name}.{token}.bak"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return StagedDirectory(stage_path=stage_path, final_path=final_path, backup_path=backup_path)

    def _final_path(self, final_name: str | Path) -> Path:
        final = Path(final_name)
        candidate = final if final.is_absolute() else self.root / final
        try:
            candidate.resolve(strict=False).relative_to(self.root.resolve(strict=False))
        except ValueError as exc:
            raise ArtifactTransactionError(
                f"artifact directory path must stay under {self.root}: {final}"
            ) from exc
        return candidate


class ArtifactUnitOfWork:
    """Coordinate promoted artifact files/directories with a later DB/audit write.

    The plain ``finalize`` method supports older repository methods that own their
    own transaction. ``finalize_with_connection`` is the preferred path for new
    multi-resource writes: promote artifacts, run DB writes on a caller-provided
    connection context, let that context commit, then commit artifact backups. If
    the callback or DB commit raises, promoted artifacts are rolled back.
    """

    def __init__(self) -> None:
        self._items: list[StagedArtifact | StagedDirectory] = []
        self._closed = False

    def stage_file(self, root: Path, final_name: str | Path) -> StagedArtifact:
        artifact = TransactionalArtifactStore(root).stage(final_name)
        self.track(artifact)
        return artifact

    def stage_directory(self, root: Path, final_name: str | Path) -> StagedDirectory:
        directory = TransactionalDirectoryStore(root).stage(final_name)
        self.track(directory)
        return directory

    def track(self, item: StagedArtifact | StagedDirectory):
        if self._closed:
            raise ArtifactTransactionError("artifact unit of work is already closed")
        self._items.append(item)
        return item

    def promote_all(self) -> None:
        for item in self._items:
            if isinstance(item, StagedDirectory):
                item.activate()
            else:
                item.promote()

    def commit(self) -> None:
        for item in self._items:
            item.commit()
        self._closed = True

    def rollback(self) -> None:
        for item in reversed(self._items):
            item.rollback()
        self._closed = True

    def finalize(self, callback):
        self.promote_all()
        try:
            result = callback()
        except Exception:
            self.rollback()
            raise
        self.commit()
        return result

    def finalize_with_connection(self, connection_factory, callback):
        self.promote_all()
        try:
            with connection_factory() as conn:
                result = callback(conn)
        except Exception:
            self.rollback()
            raise
        self.commit()
        return result


def _remove_empty_parents(path: Path) -> None:
    current = Path(path)
    while current.name == ".staging" or current.parent.name == ".staging":
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


__all__ = [
    "ArtifactTransactionError",
    "ArtifactUnitOfWork",
    "StagedArtifact",
    "StagedDirectory",
    "TransactionalArtifactStore",
    "TransactionalDirectoryStore",
]
