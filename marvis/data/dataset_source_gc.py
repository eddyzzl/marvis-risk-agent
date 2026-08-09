"""Crash-safe garbage collection for dataset source files.

The public module is deliberately small: callers can inspect durable state and
request a bounded sweep.  Task purge owns candidate creation because the
candidate must be committed atomically with removal of the final dataset row.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import errno
import logging
import os
import sqlite3
import stat
import threading

from marvis.db_schema import connect
from marvis.data.windows_safe_delete import (
    UnsafeWindowsDatasetDeleteError,
    WindowsDatasetDeleteUnsupportedError,
    delete_dataset_file_windows,
)
from marvis.safe_paths import conservative_relative_path_identity, lexical_child_path
from marvis.repositories.tasks import maybe_write_task_cleanup_succeeded


logger = logging.getLogger(__name__)
_WATCHDOG_INTERVAL_ENV = "MARVIS_DATASET_SOURCE_GC_INTERVAL_SECONDS"
DEFAULT_WATCHDOG_INTERVAL_SECONDS = 60


@dataclass(frozen=True, slots=True)
class DatasetSourceGcEntry:
    source_path: str
    state: str
    attempt_count: int
    next_attempt_at: datetime | None
    last_error_code: str | None
    last_error_message: str | None
    origin_task_id: str
    enqueued_at: datetime | None
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class DatasetSourceGcStatus:
    entries: tuple[DatasetSourceGcEntry, ...]
    pending_count: int
    quarantined_count: int


@dataclass(frozen=True, slots=True)
class DatasetSourceGcSweepReport:
    examined: int
    deleted: int
    cancelled_referenced: int
    deferred: int
    quarantined: int


@dataclass(frozen=True, slots=True)
class DatasetSourceGcWatchdogStatus:
    alive: bool
    last_started_at: datetime | None
    last_finished_at: datetime | None
    last_report: DatasetSourceGcSweepReport | None
    last_error: str | None


class _UnsafeDatasetSourcePath(PermissionError):
    pass


class _CorruptDatasetSourceGcState(ValueError):
    pass


class DatasetSourceGarbageCollector:
    """Workspace-scoped interface for durable dataset source cleanup."""

    def __init__(
        self,
        db_path: Path,
        datasets_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        delete_source: Callable[[Path], None] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.datasets_root = Path(datasets_root)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._delete_source = delete_source or self._delete_source_file

    def sweep(
        self,
        *,
        limit: int = 100,
        force: bool = False,
        source_paths: Sequence[str] | None = None,
    ) -> DatasetSourceGcSweepReport:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        selected_paths: tuple[str, ...] | None = None
        if source_paths is not None:
            if isinstance(source_paths, str | bytes | bytearray):
                raise ValueError("source_paths must be a sequence of paths")
            raw_paths = tuple(source_paths)
            if len(raw_paths) > 500:
                raise ValueError("source_paths may contain at most 500 paths")
            if any(not isinstance(path, str) or not path for path in raw_paths):
                raise ValueError("source_paths must contain non-empty strings")
            selected_paths = tuple(dict.fromkeys(raw_paths))
            if not selected_paths:
                return DatasetSourceGcSweepReport(
                    examined=0,
                    deleted=0,
                    cancelled_referenced=0,
                    deferred=0,
                    quarantined=0,
                )
        now = _utc_datetime(self._clock(), name="clock")
        with connect(self.db_path) as conn:
            if selected_paths is not None:
                placeholders = ",".join("?" for _ in selected_paths)
                due_clause = (
                    ""
                    if force
                    else (
                        "AND (datetime(next_attempt_at) IS NULL "
                        "OR next_attempt_at <= ?)"
                    )
                )
                params: tuple[object, ...] = (*selected_paths,)
                if not force:
                    params = (*params, _iso(now))
                params = (*params, limit)
                rows = conn.execute(
                    f"""
                    SELECT source_path
                      FROM dataset_source_gc_queue
                     WHERE state != 'quarantined'
                       AND source_path IN ({placeholders})
                       {due_clause}
                     ORDER BY next_attempt_at, source_path
                     LIMIT ?
                    """,  # noqa: S608 - placeholders are generated, never supplied.
                    params,
                ).fetchall()
            elif force:
                rows = conn.execute(
                    """
                    SELECT source_path
                      FROM dataset_source_gc_queue
                     WHERE state != 'quarantined'
                     ORDER BY next_attempt_at, source_path
                     LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source_path
                      FROM dataset_source_gc_queue
                     WHERE state != 'quarantined'
                       AND (
                            datetime(next_attempt_at) IS NULL
                            OR next_attempt_at <= ?
                       )
                     ORDER BY next_attempt_at, source_path
                     LIMIT ?
                    """,
                    (_iso(now), limit),
                ).fetchall()
        outcomes = [
            self._sweep_one(str(row["source_path"]), now=now, force=force)
            for row in rows
        ]
        return DatasetSourceGcSweepReport(
            examined=sum(outcome != "skipped" for outcome in outcomes),
            deleted=outcomes.count("deleted"),
            cancelled_referenced=outcomes.count("cancelled_referenced"),
            deferred=outcomes.count("deferred"),
            quarantined=outcomes.count("quarantined"),
        )

    def status(self, *, limit: int = 500, offset: int = 0) -> DatasetSourceGcStatus:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer between 1 and 5000")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT source_path, state, attempt_count, next_attempt_at,
                       last_error_code, last_error_message, origin_task_id,
                       enqueued_at, updated_at
                  FROM dataset_source_gc_queue
                 ORDER BY CASE state WHEN 'quarantined' THEN 0 ELSE 1 END,
                          next_attempt_at, source_path
                 LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            counts = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN state != 'quarantined' THEN 1 ELSE 0 END)
                        AS pending_count,
                    SUM(CASE WHEN state = 'quarantined' THEN 1 ELSE 0 END)
                        AS quarantined_count
                  FROM dataset_source_gc_queue
                """
            ).fetchone()
        entries = tuple(_entry_from_row(row) for row in rows)
        return DatasetSourceGcStatus(
            entries=entries,
            pending_count=int(counts["pending_count"] or 0),
            quarantined_count=int(counts["quarantined_count"] or 0),
        )

    def _sweep_one(self, source_path: str, *, now: datetime, force: bool) -> str:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT state, attempt_count, next_attempt_at, origin_task_id
                  FROM dataset_source_gc_queue
                 WHERE source_path = ?
                """,
                (source_path,),
            ).fetchone()
            if row is None or str(row["state"]) == "quarantined":
                return "skipped"
            try:
                next_attempt_at = _parse_utc(str(row["next_attempt_at"]))
            except ValueError:
                self._record_failure(
                    conn,
                    source_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=_CorruptDatasetSourceGcState(
                        "invalid persisted next_attempt_at"
                    ),
                    quarantined=True,
                )
                return "quarantined"
            if not force and next_attempt_at > now:
                return "skipped"
            if _dataset_source_reference_count(conn, source_path) != 0:
                conn.execute(
                    "DELETE FROM dataset_source_gc_queue WHERE source_path = ?",
                    (source_path,),
                )
                maybe_write_task_cleanup_succeeded(
                    conn,
                    task_id=str(row["origin_task_id"]),
                )
                return "cancelled_referenced"
            try:
                candidate = self._lexical_source_path(source_path)
                self._delete_source(candidate)
            except _UnsafeDatasetSourcePath as exc:
                self._record_failure(
                    conn,
                    source_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=exc,
                    quarantined=True,
                )
                return "quarantined"
            except Exception as exc:
                self._record_failure(
                    conn,
                    source_path,
                    attempt_count=int(row["attempt_count"]) + 1,
                    now=now,
                    exc=exc,
                    quarantined=False,
                )
                return "deferred"
            conn.execute(
                "DELETE FROM dataset_source_gc_queue WHERE source_path = ?",
                (source_path,),
            )
            maybe_write_task_cleanup_succeeded(
                conn,
                task_id=str(row["origin_task_id"]),
            )
            return "deleted"

    def _record_failure(
        self,
        conn: sqlite3.Connection,
        source_path: str,
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
            UPDATE dataset_source_gc_queue
               SET state = ?, attempt_count = ?, next_attempt_at = ?,
                   last_error_code = ?, last_error_message = ?, updated_at = ?
             WHERE source_path = ?
            """,
            (
                "quarantined" if quarantined else "retry_wait",
                attempt_count,
                _iso(next_attempt),
                exc.__class__.__name__,
                _bounded_error_message(exc),
                _iso(now),
                source_path,
            ),
        )

    def _lexical_source_path(self, source_path: str) -> Path:
        try:
            return lexical_child_path(self.datasets_root, source_path)
        except PermissionError as exc:
            raise _UnsafeDatasetSourcePath(str(exc)) from exc

    def _delete_source_file(self, dataset_path: Path) -> None:
        if _is_windows_platform():
            try:
                delete_dataset_file_windows(dataset_path, self.datasets_root)
            except (
                UnsafeWindowsDatasetDeleteError,
                WindowsDatasetDeleteUnsupportedError,
            ) as exc:
                raise _UnsafeDatasetSourcePath(str(exc)) from exc
            return
        root = self.datasets_root.resolve()
        try:
            relative = dataset_path.relative_to(root)
        except ValueError as exc:  # pragma: no cover - guarded by lexical resolver
            raise _UnsafeDatasetSourcePath("dataset source escaped root") from exc
        parts = relative.parts
        digest = parts[1] if len(parts) == 3 and parts[0] == "_cas" else None
        is_content_addressed = (
            digest is not None
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            and parts[2] == f"{digest}.parquet"
        )
        if not _supports_descriptor_relative_delete():
            raise _UnsafeDatasetSourcePath(
                "secure descriptor-relative dataset deletion is unavailable"
            )
        self._delete_source_descriptor_relative(
            root,
            parts,
            is_content_addressed=is_content_addressed,
        )

    def _delete_source_descriptor_relative(
        self,
        root: Path,
        parts: tuple[str, ...],
        *,
        is_content_addressed: bool,
    ) -> None:
        """Delete below ``root`` without a check/use path-resolution gap.

        Every untrusted directory is opened relative to an already-open parent
        with ``O_NOFOLLOW``.  A concurrent rename can make cleanup a harmless
        no-op, but it cannot redirect unlink/rmdir to a symlink target.
        """

        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        descriptors: list[int] = []
        try:
            descriptors.append(os.open(root, directory_flags))
            for part in parts[:-1]:
                try:
                    descriptors.append(
                        os.open(part, directory_flags, dir_fd=descriptors[-1])
                    )
                except FileNotFoundError:
                    return
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise _UnsafeDatasetSourcePath(
                            "dataset source directory changed to a symlink or file"
                        ) from exc
                    raise

            parent_fd = descriptors[-1]
            filename = parts[-1]
            try:
                source_stat = os.stat(
                    filename,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                source_stat = None
            if source_stat is not None and not (
                stat.S_ISREG(source_stat.st_mode) or stat.S_ISLNK(source_stat.st_mode)
            ):
                raise _UnsafeDatasetSourcePath(
                    "dataset source is neither a regular file nor a symlink"
                )

            removed_parent = False
            if is_content_addressed:
                os.fchmod(parent_fd, 0o700)
            try:
                if source_stat is not None:
                    os.unlink(filename, dir_fd=parent_fd)
                os.fsync(parent_fd)
                if len(parts) >= 2:
                    opened_parent = os.fstat(parent_fd)
                    try:
                        named_parent = os.stat(
                            parts[-2],
                            dir_fd=descriptors[-2],
                            follow_symlinks=False,
                        )
                    except OSError:
                        named_parent = None
                    if (
                        named_parent is not None
                        and stat.S_ISDIR(named_parent.st_mode)
                        and opened_parent.st_dev == named_parent.st_dev
                        and opened_parent.st_ino == named_parent.st_ino
                    ):
                        try:
                            os.rmdir(parts[-2], dir_fd=descriptors[-2])
                        except OSError:
                            pass
                        else:
                            removed_parent = True
                            os.fsync(descriptors[-2])
            finally:
                if is_content_addressed and not removed_parent:
                    os.fchmod(parent_fd, 0o555)
                    os.fsync(parent_fd)
        finally:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class DatasetSourceGcWatchdog:
    """Periodically retry due durable cleanup candidates while the app runs."""

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
            dataset_source_gc_interval_seconds()
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
        self._last_report: DatasetSourceGcSweepReport | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = None
            self._stop = threading.Event()
            thread = threading.Thread(
                target=self._run,
                name="marvis-dataset-source-gc-watchdog",
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
                "dataset source GC watchdog did not stop within %.1f seconds",
                self._stop_timeout_seconds,
            )
        return stopped

    def status(self) -> DatasetSourceGcWatchdogStatus:
        with self._lifecycle_lock:
            thread = self._thread
            return DatasetSourceGcWatchdogStatus(
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
                logger.exception("periodic dataset source GC sweep failed")
            else:
                with self._lifecycle_lock:
                    self._last_report = report
                    self._last_error = None
            finally:
                finished_at = _utc_datetime(self._clock(), name="watchdog clock")
                with self._lifecycle_lock:
                    self._last_finished_at = finished_at


def _entry_from_row(row) -> DatasetSourceGcEntry:
    timestamps: dict[str, datetime | None] = {}
    invalid_timestamp_fields: list[str] = []
    for field in ("next_attempt_at", "enqueued_at", "updated_at"):
        try:
            timestamps[field] = _parse_utc(str(row[field]))
        except (TypeError, ValueError):
            timestamps[field] = None
            invalid_timestamp_fields.append(field)
    last_error_code = (
        None if row["last_error_code"] is None else str(row["last_error_code"])
    )
    last_error_message = (
        None
        if row["last_error_message"] is None
        else str(row["last_error_message"])
    )
    if invalid_timestamp_fields:
        corruption_message = (
            "invalid timestamp fields: " + ", ".join(invalid_timestamp_fields)
        )
        last_error_code = last_error_code or "CorruptDatasetSourceGcTimestamp"
        last_error_message = (
            corruption_message
            if not last_error_message
            else f"{last_error_message}; {corruption_message}"
        )
    return DatasetSourceGcEntry(
        source_path=str(row["source_path"]),
        state=str(row["state"]),
        attempt_count=int(row["attempt_count"]),
        next_attempt_at=timestamps["next_attempt_at"],
        last_error_code=last_error_code,
        last_error_message=last_error_message,
        origin_task_id=str(row["origin_task_id"]),
        enqueued_at=timestamps["enqueued_at"],
        updated_at=timestamps["updated_at"],
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("dataset GC timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _utc_datetime(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bounded_error_message(exc: Exception) -> str:
    return " ".join(str(exc).split())[:500]


def _supports_descriptor_relative_delete() -> bool:
    required = (os.open, os.stat, os.unlink, os.rmdir)
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "fchmod")
        and all(function in os.supports_dir_fd for function in required)
    )


def _is_windows_platform() -> bool:
    return os.name == "nt"


def _dataset_source_reference_count(
    conn: sqlite3.Connection,
    source_path: str,
) -> int:
    (exact_count,) = conn.execute(
        "SELECT COUNT(*) FROM datasets WHERE source_path = ?",
        (source_path,),
    ).fetchone()
    if int(exact_count) != 0:
        return int(exact_count)
    try:
        candidate_identity = conservative_relative_path_identity(source_path)
    except ValueError:
        return 0
    count = 0
    for row in conn.execute("SELECT source_path FROM datasets").fetchall():
        try:
            existing_identity = conservative_relative_path_identity(
                str(row["source_path"])
            )
        except ValueError:
            continue
        if existing_identity == candidate_identity:
            count += 1
    return count


def dataset_source_gc_interval_seconds() -> int:
    raw = os.environ.get(_WATCHDOG_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_WATCHDOG_INTERVAL_SECONDS


__all__ = [
    "DatasetSourceGarbageCollector",
    "DatasetSourceGcEntry",
    "DatasetSourceGcStatus",
    "DatasetSourceGcSweepReport",
    "DatasetSourceGcWatchdog",
    "DatasetSourceGcWatchdogStatus",
    "dataset_source_gc_interval_seconds",
]
