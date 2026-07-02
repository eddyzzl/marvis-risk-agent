from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

# A RUNNING job whose executor thread hung (unbounded DuckDB join, disk IO
# stall, GIL/lock contention) keeps idx_jobs_active_task locked forever — the
# only recourse today is restarting the whole service (REL-5). The heartbeat
# column (see marvis.job_heartbeat) turns "still working" vs. "died mid-job"
# into a simple staleness check; this module is the sweep that acts on it,
# both once at startup (covers a job whose process died outright, since a dead
# process can no longer tick the heartbeat either) and periodically while the
# app is up (covers a job that hangs after the app has been running for a
# while, not just after a restart).
_HEARTBEAT_TIMEOUT_ENV = "MARVIS_JOB_HEARTBEAT_TIMEOUT_SECONDS"
_WATCHDOG_INTERVAL_ENV = "MARVIS_JOB_WATCHDOG_INTERVAL_SECONDS"
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 600
DEFAULT_WATCHDOG_INTERVAL_SECONDS = 60


def heartbeat_timeout_seconds() -> int:
    raw = os.environ.get(_HEARTBEAT_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_HEARTBEAT_TIMEOUT_SECONDS


def watchdog_interval_seconds() -> int:
    raw = os.environ.get(_WATCHDOG_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_WATCHDOG_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_WATCHDOG_INTERVAL_SECONDS


def sweep_heartbeat_lost_jobs(task_repo, *, older_than_seconds: int | None = None) -> list[dict]:
    """Fail every RUNNING job whose heartbeat has gone stale and return the
    released job rows. Never raises — a watchdog that itself crashes the app
    would be worse than a missed sweep."""
    threshold = (
        heartbeat_timeout_seconds() if older_than_seconds is None else older_than_seconds
    )
    try:
        released = task_repo.fail_heartbeat_lost_jobs(older_than_seconds=threshold)
    except Exception:
        logger.exception("job heartbeat watchdog sweep failed")
        return []
    for job in released:
        logger.warning(
            "released hung job %s (task=%s kind=%s) after heartbeat exceeded %ss",
            job.get("id"),
            job.get("task_id"),
            job.get("kind"),
            threshold,
        )
    return released


class JobHeartbeatWatchdog:
    """Background daemon thread that periodically sweeps heartbeat-lost jobs
    while the app is running. Startup already does one sweep synchronously
    (see marvis.app.create_app); this thread covers jobs that go stale later,
    not just ones already stale at boot."""

    def __init__(
        self,
        task_repo,
        *,
        interval_seconds: float | None = None,
        heartbeat_timeout: int | None = None,
    ) -> None:
        self._task_repo = task_repo
        self._interval_seconds = (
            watchdog_interval_seconds() if interval_seconds is None else interval_seconds
        )
        self._heartbeat_timeout = heartbeat_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="marvis-job-heartbeat-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            sweep_heartbeat_lost_jobs(
                self._task_repo, older_than_seconds=self._heartbeat_timeout
            )
