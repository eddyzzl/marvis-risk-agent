from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any


class NotebookCancelled(Exception):
    """Raised inside notebook execution when a user requests cancellation."""


@dataclass
class NotebookCancellationToken:
    task_id: str
    job_id: str | None = None
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _client: Any | None = None

    def bind_client(self, client: Any) -> None:
        with self._lock:
            self._client = client
            cancelled = self._cancelled.is_set()
        if cancelled:
            self._interrupt_client(client)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            client = self._client
        if client is not None:
            self._interrupt_client(client)

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise NotebookCancelled("notebook execution cancelled")

    @staticmethod
    def _interrupt_client(client: Any) -> None:
        kernel_manager = getattr(client, "km", None)
        if kernel_manager is None:
            return
        try:
            kernel_manager.interrupt_kernel()
        except Exception:
            pass


class NotebookCancellationRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[str, NotebookCancellationToken] = {}
        self._pending: dict[str, str | None] = {}

    def register(
        self,
        task_id: str,
        *,
        job_id: str | None = None,
    ) -> NotebookCancellationToken:
        token = NotebookCancellationToken(task_id=task_id, job_id=job_id)
        with self._lock:
            self._tokens[task_id] = token
            # Consume any pending cancel requested during the window before this
            # token existed. A job-bound request may only cancel the same job;
            # discarding a mismatched request prevents it poisoning a retry.
            pending_missing = object()
            pending_job_id = self._pending.pop(task_id, pending_missing)
            should_cancel = pending_job_id is not pending_missing and (
                pending_job_id is None or pending_job_id == job_id
            )
        if should_cancel:
            token.cancel()
        return token

    def unregister(self, task_id: str, token: NotebookCancellationToken) -> None:
        with self._lock:
            if self._tokens.get(task_id) is token:
                self._tokens.pop(task_id, None)
                self._pending.pop(task_id, None)

    def request_cancel(
        self,
        task_id: str,
        *,
        allow_pending: bool = True,
        expected_job_id: str | None = None,
    ) -> bool:
        with self._lock:
            token = self._tokens.get(task_id)
            if token is None:
                if allow_pending:
                    self._pending[task_id] = expected_job_id
                return False
            if expected_job_id is not None and token.job_id != expected_job_id:
                return False
        token.cancel()
        return True

    def clear_pending(self, task_id: str) -> None:
        with self._lock:
            self._pending.pop(task_id, None)


_REGISTRY = NotebookCancellationRegistry()


def register_notebook_cancellation(
    task_id: str,
    *,
    job_id: str | None = None,
) -> NotebookCancellationToken:
    return _REGISTRY.register(task_id, job_id=job_id)


def unregister_notebook_cancellation(
    task_id: str, token: NotebookCancellationToken
) -> None:
    _REGISTRY.unregister(task_id, token)


def request_notebook_cancellation(task_id: str) -> bool:
    return _REGISTRY.request_cancel(task_id)


def request_active_notebook_cancellation(
    task_id: str,
    *,
    expected_job_id: str | None = None,
) -> bool:
    return _REGISTRY.request_cancel(
        task_id,
        allow_pending=False,
        expected_job_id=expected_job_id,
    )


def clear_pending_notebook_cancellation(task_id: str) -> None:
    _REGISTRY.clear_pending(task_id)
