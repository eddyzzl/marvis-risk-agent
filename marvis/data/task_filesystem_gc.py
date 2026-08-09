"""Crash-safe garbage collection for task-owned filesystem trees.

Task purge owns candidate creation so deleting database state and recording
filesystem cleanup intent share one transaction.  This collector owns bounded
retries and performs every reference recheck under SQLite's writer lock before
touching a directory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import errno
import logging
import os
import shutil
import sqlite3
import stat
import threading

from marvis.db_schema import connect
from marvis.safe_paths import (
    canonical_relative_posix_path,
    conservative_relative_path_identity,
)


TASK_DIR = "task_dir"
DATASET_IDENTITY_DIR = "dataset_identity_dir"
DATASET_TASK_DIR = "dataset_task_dir"
RISK_INTAKE_DIR = "risk_intake_dir"
VALIDATION_BATCH_SOURCE_DIR = "validation_batch_source_dir"
TASK_FILESYSTEM_TARGET_TYPES = frozenset(
    {
        TASK_DIR,
        DATASET_IDENTITY_DIR,
        DATASET_TASK_DIR,
        RISK_INTAKE_DIR,
        VALIDATION_BATCH_SOURCE_DIR,
    }
)
_WATCHDOG_INTERVAL_ENV = "MARVIS_TASK_FS_GC_INTERVAL_SECONDS"
DEFAULT_WATCHDOG_INTERVAL_SECONDS = 60


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TaskFilesystemGcTarget:
    target_type: str
    relative_path: str

    def __post_init__(self) -> None:
        if self.target_type not in TASK_FILESYSTEM_TARGET_TYPES:
            raise ValueError(f"unsupported task filesystem target: {self.target_type}")
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ValueError("relative_path must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TaskFilesystemGcEntry:
    target_type: str
    relative_path: str
    state: str
    attempt_count: int
    next_attempt_at: datetime | None
    last_error_code: str | None
    last_error_message: str | None
    origin_task_id: str
    origin_task_created_at: str
    enqueued_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class TaskFilesystemGcStatus:
    entries: tuple[TaskFilesystemGcEntry, ...]
    pending_count: int
    quarantined_count: int


@dataclass(frozen=True, slots=True)
class TaskFilesystemGcSweepReport:
    examined: int
    deleted: int
    cancelled_referenced: int
    deferred: int
    quarantined: int


@dataclass(frozen=True, slots=True)
class TaskFilesystemGcWatchdogStatus:
    alive: bool
    last_started_at: datetime | None
    last_finished_at: datetime | None
    last_report: TaskFilesystemGcSweepReport | None
    last_error: str | None


class _UnsafeTaskFilesystemTarget(PermissionError):
    pass


class _CorruptTaskFilesystemGcState(ValueError):
    pass


class _DatasetSourceCleanupPending(RuntimeError):
    pass


class TaskFilesystemGarbageCollector:
    """Workspace-scoped interface for durable task-directory cleanup."""

    def __init__(
        self,
        db_path: Path,
        *,
        tasks_root: Path,
        datasets_root: Path,
        material_uploads_root: Path,
        clock: Callable[[], datetime] | None = None,
        delete_target: Callable[[Path, str], None] | None = None,
        platform_name: str | None = None,
        windows_delete_target: Callable[[Path, str], None] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.tasks_root = Path(tasks_root)
        self.datasets_root = Path(datasets_root)
        self.material_uploads_root = Path(material_uploads_root)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._platform_name = os.name if platform_name is None else platform_name
        self._windows_delete_target = windows_delete_target
        if delete_target is not None:
            self._delete_target = delete_target
        elif self._platform_name == "nt":
            self._delete_target = self._delete_target_windows
        else:
            self._delete_target = self._delete_target_descriptor_relative

    def sweep(
        self,
        *,
        limit: int = 100,
        force: bool = False,
        origin_task_id: str | None = None,
    ) -> TaskFilesystemGcSweepReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        if origin_task_id is not None and (
            not isinstance(origin_task_id, str) or not origin_task_id
        ):
            raise ValueError("origin_task_id must be a non-empty string")
        now = _utc_datetime(self._clock(), name="clock")
        clauses = ["state != 'quarantined'"]
        params: list[object] = []
        if not force:
            clauses.append(
                "(datetime(next_attempt_at) IS NULL OR next_attempt_at <= ?)"
            )
            params.append(_iso(now))
        if origin_task_id is not None:
            clauses.append("origin_task_id = ?")
            params.append(origin_task_id)
        params.append(limit)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT target_type, relative_path
                  FROM task_fs_gc_queue
                 WHERE {' AND '.join(clauses)}
                 ORDER BY
                    CASE target_type
                        WHEN 'dataset_identity_dir' THEN 0
                        WHEN 'task_dir' THEN 1
                        WHEN 'risk_intake_dir' THEN 2
                        WHEN 'validation_batch_source_dir' THEN 3
                        WHEN 'dataset_task_dir' THEN 4
                    END,
                    next_attempt_at,
                    relative_path
                 LIMIT ?
                """,  # noqa: S608 - clauses are selected from fixed literals.
                tuple(params),
            ).fetchall()
        outcomes = [
            self._sweep_one(
                str(row["target_type"]),
                str(row["relative_path"]),
                now=now,
                force=force,
            )
            for row in rows
        ]
        return TaskFilesystemGcSweepReport(
            examined=sum(outcome != "skipped" for outcome in outcomes),
            deleted=outcomes.count("deleted"),
            cancelled_referenced=outcomes.count("cancelled_referenced"),
            deferred=outcomes.count("deferred"),
            quarantined=outcomes.count("quarantined"),
        )

    def status(
        self,
        *,
        limit: int = 500,
        offset: int = 0,
        origin_task_id: str | None = None,
    ) -> TaskFilesystemGcStatus:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer between 1 and 5000")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if origin_task_id is not None and (
            not isinstance(origin_task_id, str) or not origin_task_id
        ):
            raise ValueError("origin_task_id must be a non-empty string")
        where = "" if origin_task_id is None else "WHERE origin_task_id = ?"
        filter_params: tuple[object, ...] = (
            () if origin_task_id is None else (origin_task_id,)
        )
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT target_type, relative_path, state, attempt_count,
                       next_attempt_at, last_error_code, last_error_message,
                       origin_task_id, origin_task_created_at,
                       enqueued_at, updated_at
                  FROM task_fs_gc_queue
                  {where}
                 ORDER BY CASE state WHEN 'quarantined' THEN 0 ELSE 1 END,
                          next_attempt_at, target_type, relative_path
                 LIMIT ? OFFSET ?
                """,  # noqa: S608 - where is selected from fixed literals.
                (*filter_params, limit, offset),
            ).fetchall()
            counts = conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN state != 'quarantined' THEN 1 ELSE 0 END)
                        AS pending_count,
                    SUM(CASE WHEN state = 'quarantined' THEN 1 ELSE 0 END)
                        AS quarantined_count
                  FROM task_fs_gc_queue
                  {where}
                """,  # noqa: S608 - where is selected from fixed literals.
                filter_params,
            ).fetchone()
        return TaskFilesystemGcStatus(
            entries=tuple(_entry_from_row(row) for row in rows),
            pending_count=int(counts["pending_count"] or 0),
            quarantined_count=int(counts["quarantined_count"] or 0),
        )

    def requeue_quarantined(self, target_type: str, relative_path: str) -> bool:
        """Explicitly requeue one exact, still-safe quarantined target."""

        TaskFilesystemGcTarget(target_type, relative_path)
        try:
            canonical_path = canonical_relative_posix_path(relative_path)
        except ValueError as exc:
            raise ValueError("relative_path is not a safe canonical path") from exc
        if canonical_path != relative_path:
            raise ValueError("relative_path is not in canonical POSIX form")
        now = _utc_datetime(self._clock(), name="clock")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state, origin_task_id
                  FROM task_fs_gc_queue
                 WHERE target_type = ? AND relative_path = ?
                """,
                (target_type, relative_path),
            ).fetchone()
            if row is None or str(row["state"]) != "quarantined":
                return False
            try:
                self._validate_target(
                    target_type,
                    relative_path,
                    origin_task_id=str(row["origin_task_id"]),
                )
            except _UnsafeTaskFilesystemTarget as exc:
                raise ValueError(str(exc)) from exc
            cursor = conn.execute(
                """
                UPDATE task_fs_gc_queue
                   SET state = 'pending', attempt_count = 0,
                       next_attempt_at = ?, last_error_code = NULL,
                       last_error_message = NULL, updated_at = ?
                 WHERE target_type = ? AND relative_path = ?
                   AND state = 'quarantined'
                """,
                (_iso(now), _iso(now), target_type, relative_path),
            )
            return cursor.rowcount == 1

    def _sweep_one(
        self,
        target_type: str,
        relative_path: str,
        *,
        now: datetime,
        force: bool,
    ) -> str:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state, attempt_count, next_attempt_at, origin_task_id,
                       origin_task_created_at
                  FROM task_fs_gc_queue
                 WHERE target_type = ? AND relative_path = ?
                """,
                (target_type, relative_path),
            ).fetchone()
            if row is None or str(row["state"]) == "quarantined":
                return "skipped"
            origin_task_id = str(row["origin_task_id"])
            origin_task_created_at = str(row["origin_task_created_at"])
            try:
                next_attempt_at = _parse_utc(str(row["next_attempt_at"]))
            except ValueError:
                self._record_failure(
                    conn,
                    target_type,
                    relative_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=_CorruptTaskFilesystemGcState(
                        "invalid persisted next_attempt_at"
                    ),
                    quarantined=True,
                )
                return "quarantined"
            if not force and next_attempt_at > now:
                return "skipped"
            try:
                root = self._validate_target(
                    target_type,
                    relative_path,
                    origin_task_id=origin_task_id,
                )
                reference_state = self._reference_state(
                    conn,
                    target_type,
                    relative_path,
                    origin_task_id=origin_task_id,
                    origin_task_created_at=origin_task_created_at,
                    root=root,
                )
                if reference_state == "referenced":
                    self._remove_queue_row(
                        conn,
                        target_type,
                        relative_path,
                        origin_task_id=origin_task_id,
                    )
                    return "cancelled_referenced"
                if reference_state == "dataset_cleanup_pending":
                    raise _DatasetSourceCleanupPending(
                        "dataset source cleanup remains queued below task directory"
                    )
                self._delete_target(root, relative_path)
            except _UnsafeTaskFilesystemTarget as exc:
                self._record_failure(
                    conn,
                    target_type,
                    relative_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=exc,
                    quarantined=True,
                )
                return "quarantined"
            except Exception as exc:
                self._record_failure(
                    conn,
                    target_type,
                    relative_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=exc,
                    quarantined=False,
                )
                return "deferred"
            self._remove_queue_row(
                conn,
                target_type,
                relative_path,
                origin_task_id=origin_task_id,
            )
            return "deleted"

    def _validate_target(
        self,
        target_type: str,
        relative_path: str,
        *,
        origin_task_id: str,
    ) -> Path:
        try:
            normalized = canonical_relative_posix_path(relative_path)
            task_identity = canonical_relative_posix_path(origin_task_id)
        except ValueError as exc:
            raise _UnsafeTaskFilesystemTarget(str(exc)) from exc
        if normalized != relative_path or task_identity != origin_task_id:
            raise _UnsafeTaskFilesystemTarget(
                "task filesystem target does not use its canonical owner spelling"
            )
        if "/" in task_identity or task_identity in {"", ".", ".."}:
            raise _UnsafeTaskFilesystemTarget("origin task id is not one path component")
        expected = {
            TASK_DIR: task_identity,
            DATASET_IDENTITY_DIR: f"{task_identity}/.source-identities",
            DATASET_TASK_DIR: task_identity,
        }
        if target_type == RISK_INTAKE_DIR:
            if "/" in normalized or not normalized.startswith("risk-intake-"):
                raise _UnsafeTaskFilesystemTarget(
                    "risk intake target is not a controlled directory name"
                )
            return self.material_uploads_root
        if target_type == VALIDATION_BATCH_SOURCE_DIR:
            parts = normalized.split("/")
            token = parts[-1] if parts else ""
            if (
                len(parts) != 2
                or parts[0] != "validation-batches"
                or len(token) != 32
                or token != token.lower()
                or any(character not in "0123456789abcdef" for character in token)
            ):
                raise _UnsafeTaskFilesystemTarget(
                    "validation batch target is not a controlled directory name"
                )
            return self.material_uploads_root
        if target_type not in expected or normalized != expected[target_type]:
            raise _UnsafeTaskFilesystemTarget(
                "task filesystem target does not match its typed owner"
            )
        return self.tasks_root if target_type == TASK_DIR else self.datasets_root

    def _reference_state(
        self,
        conn: sqlite3.Connection,
        target_type: str,
        relative_path: str,
        *,
        origin_task_id: str,
        origin_task_created_at: str,
        root: Path,
    ) -> str:
        if conn.execute(
            "SELECT 1 FROM tasks WHERE id = ? LIMIT 1",
            (origin_task_id,),
        ).fetchone() is not None:
            return "referenced"
        if target_type in {RISK_INTAKE_DIR, VALIDATION_BATCH_SOURCE_DIR}:
            from marvis.repositories.tasks import task_filesystem_provenance_exists

            if not task_filesystem_provenance_exists(
                conn,
                task_id=origin_task_id,
                task_created_at=origin_task_created_at,
                target_type=target_type,
                relative_path=relative_path,
            ):
                label = (
                    "risk intake"
                    if target_type == RISK_INTAKE_DIR
                    else "validation batch"
                )
                raise _UnsafeTaskFilesystemTarget(
                    f"{label} target lacks exact platform-created provenance"
                )
            absolute_target = str((root.absolute() / relative_path).absolute())
            if target_type == RISK_INTAKE_DIR:
                if conn.execute(
                    "SELECT 1 FROM tasks WHERE source_dir = ? LIMIT 1",
                    (absolute_target,),
                ).fetchone() is not None:
                    return "referenced"
            elif _any_task_source_below(conn, absolute_target):
                return "referenced"
        if target_type == DATASET_TASK_DIR:
            if _any_relative_path_below(
                conn.execute("SELECT source_path FROM datasets").fetchall(),
                prefix=relative_path,
            ):
                return "referenced"
            if _any_relative_path_below(
                conn.execute(
                    "SELECT source_path FROM dataset_source_gc_queue"
                ).fetchall(),
                prefix=relative_path,
            ):
                return "dataset_cleanup_pending"
        return "unreferenced"

    def _record_failure(
        self,
        conn: sqlite3.Connection,
        target_type: str,
        relative_path: str,
        *,
        attempt_count: int,
        now: datetime,
        exc: Exception,
        quarantined: bool,
    ) -> None:
        delay_seconds = min(5 * (2 ** min(attempt_count - 1, 10)), 3600)
        next_attempt = now if quarantined else now + timedelta(seconds=delay_seconds)
        conn.execute(
            """
            UPDATE task_fs_gc_queue
               SET state = ?, attempt_count = ?, next_attempt_at = ?,
                   last_error_code = ?, last_error_message = ?, updated_at = ?
             WHERE target_type = ? AND relative_path = ?
            """,
            (
                "quarantined" if quarantined else "retry_wait",
                attempt_count,
                _iso(next_attempt),
                exc.__class__.__name__,
                _bounded_error_message(exc),
                _iso(now),
                target_type,
                relative_path,
            ),
        )

    def _remove_queue_row(
        self,
        conn: sqlite3.Connection,
        target_type: str,
        relative_path: str,
        *,
        origin_task_id: str,
    ) -> None:
        conn.execute(
            """
            DELETE FROM task_fs_gc_queue
             WHERE target_type = ? AND relative_path = ?
            """,
            (target_type, relative_path),
        )
        from marvis.repositories.tasks import maybe_write_task_cleanup_succeeded

        maybe_write_task_cleanup_succeeded(conn, task_id=origin_task_id)

    def _delete_target_descriptor_relative(
        self,
        root: Path,
        relative_path: str,
    ) -> None:
        if not _supports_secure_recursive_delete():
            raise _UnsafeTaskFilesystemTarget(
                "secure descriptor-relative recursive deletion is unavailable"
            )
        parts = canonical_relative_posix_path(relative_path).split("/")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        descriptors: list[int] = []
        try:
            try:
                descriptors.append(os.open(root.absolute(), directory_flags))
            except FileNotFoundError:
                return
            for part in parts[:-1]:
                try:
                    descriptors.append(
                        os.open(part, directory_flags, dir_fd=descriptors[-1])
                    )
                except FileNotFoundError:
                    return
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise _UnsafeTaskFilesystemTarget(
                            "task filesystem target traverses a symlink or file"
                        ) from exc
                    raise
            parent_fd = descriptors[-1]
            name = parts[-1]
            try:
                target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            if stat.S_ISLNK(target_stat.st_mode):
                os.unlink(name, dir_fd=parent_fd)
            elif stat.S_ISDIR(target_stat.st_mode):
                shutil.rmtree(name, dir_fd=parent_fd)
            else:
                raise _UnsafeTaskFilesystemTarget(
                    "task filesystem cleanup target is not a directory or symlink"
                )
            os.fsync(parent_fd)
        finally:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _delete_target_windows(self, root: Path, relative_path: str) -> None:
        from marvis.data.windows_task_tree_delete import (
            UnsafeWindowsTaskTreeDeleteError,
            WindowsTaskTreeDeleteUnsupportedError,
            delete_task_tree_windows,
        )

        try:
            if self._windows_delete_target is not None:
                self._windows_delete_target(root, relative_path)
                return
            parts = canonical_relative_posix_path(relative_path).split("/")
            delete_task_tree_windows(root.joinpath(*parts), root)
        except (
            UnsafeWindowsTaskTreeDeleteError,
            WindowsTaskTreeDeleteUnsupportedError,
        ) as exc:
            raise _UnsafeTaskFilesystemTarget(str(exc)) from exc


class TaskFilesystemGcWatchdog:
    """Periodically retry due task-directory cleanup candidates."""

    def __init__(
        self,
        collector,
        *,
        interval_seconds: float | None = None,
        sweep_limit: int = 100,
        stop_timeout_seconds: float = 10.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(getattr(collector, "sweep", None)):
            raise ValueError("collector must provide sweep()")
        interval = (
            task_filesystem_gc_interval_seconds()
            if interval_seconds is None
            else interval_seconds
        )
        if isinstance(interval, bool) or not isinstance(interval, int | float) or interval <= 0:
            raise ValueError("interval_seconds must be a positive number")
        if (
            isinstance(sweep_limit, bool)
            or not isinstance(sweep_limit, int)
            or not 1 <= sweep_limit <= 500
        ):
            raise ValueError("sweep_limit must be an integer between 1 and 500")
        if (
            isinstance(stop_timeout_seconds, bool)
            or not isinstance(stop_timeout_seconds, int | float)
            or stop_timeout_seconds <= 0
        ):
            raise ValueError("stop_timeout_seconds must be a positive number")
        self._collector = collector
        self._interval_seconds = float(interval)
        self._sweep_limit = sweep_limit
        self._stop_timeout_seconds = float(stop_timeout_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_started_at: datetime | None = None
        self._last_finished_at: datetime | None = None
        self._last_report: TaskFilesystemGcSweepReport | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = None
            self._stop = threading.Event()
            thread = threading.Thread(
                target=self._run,
                name="marvis-task-filesystem-gc-watchdog",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self) -> bool:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return True
            self._stop.set()
        thread.join(timeout=self._stop_timeout_seconds)
        stopped = not thread.is_alive()
        with self._lifecycle_lock:
            if self._thread is thread and stopped:
                self._thread = None
        if not stopped:
            logger.error(
                "task filesystem GC watchdog did not stop within %.1f seconds",
                self._stop_timeout_seconds,
            )
        return stopped

    def status(self) -> TaskFilesystemGcWatchdogStatus:
        with self._lifecycle_lock:
            thread = self._thread
            return TaskFilesystemGcWatchdogStatus(
                alive=bool(thread is not None and thread.is_alive()),
                last_started_at=self._last_started_at,
                last_finished_at=self._last_finished_at,
                last_report=self._last_report,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            started_at = _utc_datetime(self._clock(), name="watchdog clock")
            with self._lifecycle_lock:
                self._last_started_at = started_at
            try:
                report = self._collector.sweep(limit=self._sweep_limit)
            except Exception as exc:
                with self._lifecycle_lock:
                    self._last_report = None
                    self._last_error = (
                        f"{exc.__class__.__name__}: {_bounded_error_message(exc)}"
                    )
                logger.exception("periodic task filesystem GC sweep failed")
            else:
                with self._lifecycle_lock:
                    self._last_report = report
                    self._last_error = None
            finally:
                finished_at = _utc_datetime(self._clock(), name="watchdog clock")
                with self._lifecycle_lock:
                    self._last_finished_at = finished_at


def _entry_from_row(row) -> TaskFilesystemGcEntry:
    timestamps: dict[str, datetime | None] = {}
    invalid_fields: list[str] = []
    for field in ("next_attempt_at", "enqueued_at", "updated_at"):
        try:
            timestamps[field] = _parse_utc(str(row[field]))
        except (TypeError, ValueError):
            timestamps[field] = None
            invalid_fields.append(field)
    error_code = None if row["last_error_code"] is None else str(row["last_error_code"])
    error_message = (
        None if row["last_error_message"] is None else str(row["last_error_message"])
    )
    if invalid_fields:
        corruption = "invalid timestamp fields: " + ", ".join(invalid_fields)
        error_code = error_code or "CorruptTaskFilesystemGcTimestamp"
        error_message = corruption if not error_message else f"{error_message}; {corruption}"
    return TaskFilesystemGcEntry(
        target_type=str(row["target_type"]),
        relative_path=str(row["relative_path"]),
        state=str(row["state"]),
        attempt_count=int(row["attempt_count"]),
        next_attempt_at=timestamps["next_attempt_at"],
        last_error_code=error_code,
        last_error_message=error_message,
        origin_task_id=str(row["origin_task_id"]),
        origin_task_created_at=str(row["origin_task_created_at"]),
        enqueued_at=timestamps["enqueued_at"],
        updated_at=timestamps["updated_at"],
    )


def _any_relative_path_below(rows, *, prefix: str) -> bool:
    try:
        prefix_identity = conservative_relative_path_identity(prefix)
    except ValueError:
        return False
    for row in rows:
        try:
            candidate = conservative_relative_path_identity(str(row["source_path"]))
        except ValueError:
            continue
        if candidate == prefix_identity or candidate.startswith(f"{prefix_identity}/"):
            return True
    return False


def _any_task_source_below(conn: sqlite3.Connection, absolute_prefix: str) -> bool:
    """Recheck all task material references while the caller holds writer lock."""

    prefix = Path(absolute_prefix).absolute()
    for row in conn.execute("SELECT source_dir FROM tasks").fetchall():
        raw = str(row["source_dir"] or "").strip()
        if not raw:
            continue
        candidate = Path(raw).absolute()
        if candidate == prefix or prefix in candidate.parents:
            return True
    return False


def _supports_secure_recursive_delete() -> bool:
    required = (os.open, os.stat, os.unlink)
    return (
        bool(getattr(shutil.rmtree, "avoids_symlink_attacks", False))
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(function in os.supports_dir_fd for function in required)
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("task filesystem GC timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _utc_datetime(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bounded_error_message(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


def task_filesystem_gc_interval_seconds() -> int:
    raw = os.environ.get(_WATCHDOG_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_WATCHDOG_INTERVAL_SECONDS


__all__ = [
    "DATASET_IDENTITY_DIR",
    "DATASET_TASK_DIR",
    "DEFAULT_WATCHDOG_INTERVAL_SECONDS",
    "RISK_INTAKE_DIR",
    "TASK_DIR",
    "TASK_FILESYSTEM_TARGET_TYPES",
    "VALIDATION_BATCH_SOURCE_DIR",
    "TaskFilesystemGarbageCollector",
    "TaskFilesystemGcEntry",
    "TaskFilesystemGcStatus",
    "TaskFilesystemGcSweepReport",
    "TaskFilesystemGcTarget",
    "TaskFilesystemGcWatchdog",
    "TaskFilesystemGcWatchdogStatus",
    "task_filesystem_gc_interval_seconds",
]
