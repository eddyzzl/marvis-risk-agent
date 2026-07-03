from __future__ import annotations

import contextlib
import threading
from typing import Iterator

# Long jobs (notebook/metrics/report stages, join execution, plan/driver runs)
# all execute synchronously on a single worker thread with no natural "still
# alive" signal beyond the eventual finish_job() call. If that thread hangs
# (disk IO stall, unbounded DuckDB join, GIL/lock contention) the job stays
# 'running' forever and idx_jobs_active_task wedges the task behind a 409 with
# no way out short of a full restart (REL-5). heartbeat_job wraps the blocking
# call in a background ticker thread that periodically touches jobs.heartbeat_at
# so the watchdog (marvis.job_watchdog) can tell "still working" from "died".
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 15.0


@contextlib.contextmanager
def heartbeat_job(
    task_repo,
    job_id: str,
    *,
    interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Touch ``jobs.heartbeat_at`` for ``job_id`` every ``interval_seconds``
    while the wrapped block runs. Best-effort: a heartbeat failure never
    interrupts the job itself, it just stops ticking (the watchdog will
    eventually reclaim the job, which is the correct outcome for a DB that's
    itself unhealthy)."""
    stop = threading.Event()

    def _tick() -> None:
        while not stop.wait(interval_seconds):
            try:
                task_repo.touch_job_heartbeat(job_id)
            except Exception:
                continue

    thread = threading.Thread(
        target=_tick,
        name="marvis-job-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2.0)
