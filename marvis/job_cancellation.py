from __future__ import annotations

from dataclasses import dataclass, field
import threading


class JobCancelled(Exception):
    """Raised at a cooperative checkpoint when a user requests job cancellation."""


@dataclass
class JobCancellationToken:
    job_id: str
    _cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise JobCancelled("job cancelled")


class JobCancellationRegistry:
    """Cooperative cancellation for long jobs that have no kernel/process to
    interrupt (join execution, plan/driver runs) — mirrors
    marvis.notebook_cancellation's registry, keyed by job_id instead of
    task_id since a task can only ever have one active job at a time
    (idx_jobs_active_task) but job ids are what routers/executors already
    thread through end to end (REL-5)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[str, JobCancellationToken] = {}
        self._pending: set[str] = set()

    def register(self, job_id: str) -> JobCancellationToken:
        token = JobCancellationToken(job_id=job_id)
        with self._lock:
            self._tokens[job_id] = token
            should_cancel = job_id in self._pending
            self._pending.discard(job_id)
        if should_cancel:
            token.cancel()
        return token

    def unregister(self, job_id: str, token: JobCancellationToken) -> None:
        with self._lock:
            if self._tokens.get(job_id) is token:
                self._tokens.pop(job_id, None)
                self._pending.discard(job_id)

    def request_cancel(self, job_id: str, *, allow_pending: bool = True) -> bool:
        with self._lock:
            token = self._tokens.get(job_id)
            if token is None:
                if allow_pending:
                    self._pending.add(job_id)
                return False
        token.cancel()
        return True

    def clear_pending(self, job_id: str) -> None:
        with self._lock:
            self._pending.discard(job_id)


_REGISTRY = JobCancellationRegistry()


def register_job_cancellation(job_id: str) -> JobCancellationToken:
    return _REGISTRY.register(job_id)


def unregister_job_cancellation(job_id: str, token: JobCancellationToken) -> None:
    _REGISTRY.unregister(job_id, token)


def request_job_cancellation(job_id: str) -> bool:
    return _REGISTRY.request_cancel(job_id)


def clear_pending_job_cancellation(job_id: str) -> None:
    _REGISTRY.clear_pending(job_id)
