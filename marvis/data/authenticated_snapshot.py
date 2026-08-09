from __future__ import annotations

import hashlib
import hmac
import os
import stat
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Sequence

import pandas as pd


class SnapshotFailureReason(StrEnum):
    PATH_OUTSIDE_ROOT = "path_outside_root"
    SOURCE_NOT_REGULAR = "source_not_regular"
    SOURCE_CHANGED_WHILE_OPENING = "source_changed_while_opening"
    SOURCE_BYTES_CHANGED = "source_bytes_changed"
    PRIVATE_SNAPSHOT_INCOMPLETE = "private_snapshot_incomplete"
    SOURCE_CHANGED_DURING_READ = "source_changed_during_read"
    READ_FAILED = "read_failed"


class AuthenticatedSnapshotError(RuntimeError):
    def __init__(self, reason: SnapshotFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def read_authenticated_parquet_snapshot(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one immutable, hash-authenticated Parquet snapshot.

    The source must be a regular file below ``root`` without symlink traversal.
    Bytes are copied from one retained descriptor into a private temporary file;
    both descriptors and the visible source path are re-authenticated before the
    parsed frame is returned.
    """

    absolute_path, resolved_root = _governed_path(path, root=root)
    source_fd = -1
    snapshot = None
    try:
        before = os.lstat(absolute_path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            _fail(
                SnapshotFailureReason.SOURCE_NOT_REGULAR,
                "authenticated snapshot source must be a regular file",
            )

        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(absolute_path, flags)
        opened = os.fstat(source_fd)
        after_open = os.lstat(absolute_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_CHANGED_WHILE_OPENING,
                "authenticated snapshot source changed while opening",
            )

        snapshot = tempfile.TemporaryFile(mode="w+b", dir=resolved_root)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            snapshot.write(chunk)
        snapshot.flush()
        if (
            _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_BYTES_CHANGED,
                "authenticated snapshot source bytes changed",
            )

        snapshot_stat = os.fstat(snapshot.fileno())
        if int(snapshot_stat.st_size) != copied:
            _fail(
                SnapshotFailureReason.PRIVATE_SNAPSHOT_INCOMPLETE,
                "authenticated private snapshot is incomplete",
            )

        snapshot.seek(0)
        selected_columns = None if columns is None else list(columns)
        frame = pd.read_parquet(snapshot, columns=selected_columns)
        try:
            current = os.lstat(absolute_path)
        except OSError as exc:
            raise AuthenticatedSnapshotError(
                SnapshotFailureReason.SOURCE_CHANGED_DURING_READ,
                "authenticated snapshot source changed during read",
            ) from exc
        if (
            _stable_file_stat(os.fstat(snapshot.fileno()))
            != _stable_file_stat(snapshot_stat)
            or _stable_file_stat(os.fstat(source_fd))
            != _stable_file_stat(opened)
            or stat.S_ISLNK(current.st_mode)
            or _stable_file_stat(current) != _stable_file_stat(opened)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_CHANGED_DURING_READ,
                "authenticated snapshot source changed during read",
            )
        return frame
    except AuthenticatedSnapshotError:
        raise
    except Exception as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.READ_FAILED,
            "authenticated snapshot could not be read",
        ) from exc
    finally:
        if snapshot is not None:
            snapshot.close()
        if source_fd >= 0:
            os.close(source_fd)


def materialize_authenticated_file_snapshot(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
    destination: Path,
) -> Path:
    """Pin authenticated bytes into a non-overwritable content-addressed file.

    The source is copied from one retained descriptor and authenticated before
    an atomic directory rename publishes it.  The final directory/file are
    read-only, and an existing object is reused only after its bytes match the
    digest encoded in its path.  Database state can therefore point at this
    immutable object instead of racing a mutable source path at commit time.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        _fail(
            SnapshotFailureReason.SOURCE_BYTES_CHANGED,
            "content-addressed snapshot requires a lowercase SHA-256 digest",
        )
    absolute_path, resolved_root = _governed_path(path, root=root)
    absolute_destination, cas_root = _content_addressed_destination(
        destination,
        root=resolved_root,
        expected_sha256=expected_sha256,
    )
    if absolute_destination.exists():
        _verify_content_addressed_file(
            absolute_destination,
            expected_sha256=expected_sha256,
        )
        # A previously published object proves only that the CAS copy is
        # intact.  A still-mutable registered source must independently match
        # the same digest; otherwise a post-review byte swap could be hidden by
        # reusing another dataset's existing CAS object.
        if absolute_path.resolve(strict=True) != absolute_destination.resolve(
            strict=True
        ):
            _verify_content_addressed_file(
                absolute_path,
                expected_sha256=expected_sha256,
            )
        _harden_content_addressed_path(absolute_destination)
        return absolute_destination.resolve(strict=True)
    if absolute_destination.parent.exists():
        _fail(
            SnapshotFailureReason.PRIVATE_SNAPSHOT_INCOMPLETE,
            "content-addressed snapshot directory exists without its object",
        )

    source_fd = -1
    staging_dir: Path | None = None
    try:
        before = os.lstat(absolute_path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            _fail(
                SnapshotFailureReason.SOURCE_NOT_REGULAR,
                "content-addressed snapshot source must be a regular file",
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_fd = os.open(absolute_path, flags)
        opened = os.fstat(source_fd)
        after_open = os.lstat(absolute_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after_open.st_mode)
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after_open)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after_open)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_CHANGED_WHILE_OPENING,
                "content-addressed snapshot source changed while opening",
            )

        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".cas-{expected_sha256[:12]}-",
                dir=cas_root,
            )
        )
        staging_file = staging_dir / absolute_destination.name
        digest = hashlib.sha256()
        copied = 0
        with staging_file.open("xb") as target:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                copied += len(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())

        try:
            current = os.lstat(absolute_path)
        except OSError as exc:
            raise AuthenticatedSnapshotError(
                SnapshotFailureReason.SOURCE_CHANGED_DURING_READ,
                "content-addressed snapshot source disappeared during copy",
            ) from exc
        if (
            _stable_file_stat(os.fstat(source_fd)) != _stable_file_stat(opened)
            or _stable_file_stat(current) != _stable_file_stat(opened)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_BYTES_CHANGED,
                "content-addressed snapshot source bytes changed",
            )

        os.chmod(staging_file, 0o444)
        os.chmod(staging_dir, 0o555)
        try:
            os.rename(staging_dir, absolute_destination.parent)
            staging_dir = None
        except OSError:
            if not absolute_destination.exists():
                raise
            _discard_snapshot_staging(staging_dir)
            staging_dir = None
        _verify_content_addressed_file(
            absolute_destination,
            expected_sha256=expected_sha256,
        )
        _harden_content_addressed_path(absolute_destination)
        _fsync_directory(cas_root)
        return absolute_destination.resolve(strict=True)
    except AuthenticatedSnapshotError:
        raise
    except Exception as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.READ_FAILED,
            "content-addressed snapshot could not be materialized",
        ) from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if staging_dir is not None:
            _discard_snapshot_staging(staging_dir)


def verify_content_addressed_file_snapshot(
    path: Path,
    *,
    root: Path,
    expected_sha256: str,
) -> Path:
    """Re-authenticate an already-published content-addressed snapshot.

    Callers use this inside their database write transaction immediately before
    binding a row to the path. That closes the materialize-to-bind race with
    task-level garbage collection: a collector that deleted the last
    unreferenced object makes this check fail before the database update.
    """

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        _fail(
            SnapshotFailureReason.SOURCE_BYTES_CHANGED,
            "content-addressed snapshot requires a lowercase SHA-256 digest",
        )
    absolute_path, _cas_root = _content_addressed_destination(
        path,
        root=Path(root).resolve(strict=True),
        expected_sha256=expected_sha256,
    )
    if not absolute_path.exists() and not absolute_path.is_symlink():
        _fail(
            SnapshotFailureReason.PRIVATE_SNAPSHOT_INCOMPLETE,
            "content-addressed snapshot is missing",
        )
    _verify_content_addressed_file(
        absolute_path,
        expected_sha256=expected_sha256,
    )
    _harden_content_addressed_path(absolute_path)
    return absolute_path.resolve(strict=True)


def _content_addressed_destination(
    destination: Path,
    *,
    root: Path,
    expected_sha256: str,
) -> tuple[Path, Path]:
    absolute_root = Path(root).resolve(strict=True)
    absolute_destination = Path(destination).absolute()
    try:
        absolute_destination.relative_to(absolute_root)
    except ValueError as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "content-addressed snapshot destination escaped governed storage",
        ) from exc
    if (
        absolute_destination.name != f"{expected_sha256}.parquet"
        or absolute_destination.parent.name != expected_sha256
    ):
        _fail(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "content-addressed snapshot destination does not encode its digest",
        )
    cas_root = absolute_destination.parent.parent
    if cas_root.parent.absolute() != absolute_root or cas_root.is_symlink():
        _fail(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "content-addressed snapshot root must be a direct governed child",
        )
    cas_root.mkdir(exist_ok=True)
    if absolute_destination.parent.is_symlink():
        _fail(
            SnapshotFailureReason.SOURCE_NOT_REGULAR,
            "content-addressed snapshot destination must not use symlinks",
        )
    current = cas_root
    while True:
        if current.is_symlink():
            _fail(
                SnapshotFailureReason.SOURCE_NOT_REGULAR,
                "content-addressed snapshot destination must not use symlinks",
            )
        if current == absolute_root:
            break
        if current == current.parent:
            _fail(
                SnapshotFailureReason.PATH_OUTSIDE_ROOT,
                "content-addressed snapshot destination escaped governed storage",
            )
        current = current.parent
    try:
        resolved_cas_root = cas_root.resolve(strict=True)
        resolved_cas_root.relative_to(absolute_root)
    except (OSError, ValueError) as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "content-addressed snapshot destination escaped governed storage",
        ) from exc
    return absolute_destination, resolved_cas_root


def _verify_content_addressed_file(path: Path, *, expected_sha256: str) -> None:
    descriptor = -1
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            _fail(
                SnapshotFailureReason.SOURCE_NOT_REGULAR,
                "content-addressed snapshot must be a regular file",
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
        after = os.lstat(path)
        if (
            _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after)
            or _stable_file_stat(before) != _stable_file_stat(opened)
            or _stable_file_stat(opened) != _stable_file_stat(after)
            or copied != int(opened.st_size)
            or not hmac.compare_digest(digest.hexdigest(), expected_sha256)
        ):
            _fail(
                SnapshotFailureReason.SOURCE_BYTES_CHANGED,
                "content-addressed snapshot bytes do not match its identity",
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _harden_content_addressed_path(path: Path) -> None:
    os.chmod(path, 0o444)
    os.chmod(path.parent, 0o555)


def _discard_snapshot_staging(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Some supported filesystems/platforms do not expose directory fsync.
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            return
    finally:
        os.close(descriptor)


def _governed_path(path: Path, *, root: Path) -> tuple[Path, Path]:
    absolute_path = path.absolute()
    absolute_root = root.absolute()
    try:
        absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "authenticated snapshot path escaped governed storage",
        ) from exc

    current = absolute_path
    while True:
        if current.is_symlink():
            raise AuthenticatedSnapshotError(
                SnapshotFailureReason.SOURCE_NOT_REGULAR,
                "authenticated snapshot path must not use symlinks",
            )
        if current == absolute_root:
            break
        if current == current.parent:
            raise AuthenticatedSnapshotError(
                SnapshotFailureReason.PATH_OUTSIDE_ROOT,
                "authenticated snapshot path escaped governed storage",
            )
        current = current.parent

    try:
        resolved_root = absolute_root.resolve(strict=True)
        resolved_path = absolute_path.resolve(strict=True)
        if not resolved_root.is_dir() or not resolved_path.is_relative_to(
            resolved_root
        ):
            raise ValueError("path escaped governed storage")
    except (OSError, ValueError) as exc:
        raise AuthenticatedSnapshotError(
            SnapshotFailureReason.PATH_OUTSIDE_ROOT,
            "authenticated snapshot path escaped governed storage",
        ) from exc
    return absolute_path, resolved_root


def _file_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
    )


def _stable_file_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _fail(reason: SnapshotFailureReason, message: str) -> None:
    raise AuthenticatedSnapshotError(reason, message)


__all__ = [
    "AuthenticatedSnapshotError",
    "SnapshotFailureReason",
    "materialize_authenticated_file_snapshot",
    "read_authenticated_parquet_snapshot",
]
