from __future__ import annotations

import threading

from marvis.agent.service import agent_conclusions_confirmed
from marvis.db import TaskRepository
from marvis.domain import TaskRecord, TaskStatus
from marvis.pipeline import (
    METRICS_STAGE_FAILURE_PREFIX,
    NOTEBOOK_STAGE_FAILURE_PREFIX,
)


class AgentValidationCancelled(Exception):
    pass


_AGENT_CANCELLATION_LOCK = threading.Lock()
_AGENT_CANCELLATION_REQUESTS: set[tuple[str, str | None]] = set()
_ACTIVE_AGENT_JOBS: dict[str, str] = {}


def agent_next_stage(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    scan_failure_prefix: str,
) -> str | None:
    if task.status in {TaskStatus.CREATED, TaskStatus.FAILED} and (
        task.status == TaskStatus.CREATED
        or _is_scan_failure(task, scan_failure_prefix=scan_failure_prefix)
    ):
        return "scan"
    if task.status == TaskStatus.SCANNED or is_notebook_failure(task):
        return "reproducibility"
    if task.status == TaskStatus.EXECUTED or is_metrics_failure(task):
        return "metrics"
    if task.status in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        values, _ = repo.get_report_values(task.id)
        if not agent_conclusions_confirmed(values):
            return "word_conclusion_draft"
    return None


def is_notebook_failure(task: TaskRecord) -> bool:
    return task.status == TaskStatus.FAILED and task.status_message.startswith(
        NOTEBOOK_STAGE_FAILURE_PREFIX
    )


def is_metrics_failure(task: TaskRecord) -> bool:
    return task.status == TaskStatus.FAILED and (
        task.status_message.startswith(METRICS_STAGE_FAILURE_PREFIX)
        or _looks_like_metrics_failure_message(task.status_message)
    )


def register_agent_cancellation(task_id: str, job_id: str) -> None:
    with _AGENT_CANCELLATION_LOCK:
        _ACTIVE_AGENT_JOBS[task_id] = job_id
        # A task-only request belongs to streaming work that had no job lease;
        # it must not poison a later job. A job-bound request created in the DB
        # insert -> registry window is intentionally preserved for this job.
        _AGENT_CANCELLATION_REQUESTS.discard((task_id, None))


def request_agent_cancellation(task_id: str, *, job_id: str | None = None) -> None:
    with _AGENT_CANCELLATION_LOCK:
        target_job_id = job_id or _ACTIVE_AGENT_JOBS.get(task_id)
        _AGENT_CANCELLATION_REQUESTS.add((task_id, target_job_id))


def clear_agent_cancellation(task_id: str, *, job_id: str | None = None) -> None:
    with _AGENT_CANCELLATION_LOCK:
        if job_id is None:
            stale_requests = {
                request
                for request in _AGENT_CANCELLATION_REQUESTS
                if request[0] == task_id
            }
            _AGENT_CANCELLATION_REQUESTS.difference_update(stale_requests)
            _ACTIVE_AGENT_JOBS.pop(task_id, None)
            return
        _AGENT_CANCELLATION_REQUESTS.discard((task_id, job_id))
        if _ACTIVE_AGENT_JOBS.get(task_id) == job_id:
            _ACTIVE_AGENT_JOBS.pop(task_id, None)


def agent_cancellation_requested(
    task_id: str,
    *,
    job_id: str | None = None,
) -> bool:
    with _AGENT_CANCELLATION_LOCK:
        target_job_id = job_id or _ACTIVE_AGENT_JOBS.get(task_id)
        return (task_id, target_job_id) in _AGENT_CANCELLATION_REQUESTS


def raise_if_agent_cancelled(task_id: str, *, job_id: str | None = None) -> None:
    if agent_cancellation_requested(task_id, job_id=job_id):
        raise AgentValidationCancelled("agent validation cancelled")


def _is_scan_failure(task: TaskRecord, *, scan_failure_prefix: str) -> bool:
    return task.status == TaskStatus.FAILED and task.status_message.startswith(
        scan_failure_prefix
    )


METRICS_FAILURE_MESSAGE_MARKERS = (
    "live notebook kernel is not available",
    "RMC_TARGET_COL=",
    "sample column check failed",
    "data dictionary missing columns",
)


def _looks_like_metrics_failure_message(message: str) -> bool:
    normalized = message.lower()
    return any(marker.lower() in normalized for marker in METRICS_FAILURE_MESSAGE_MARKERS)
