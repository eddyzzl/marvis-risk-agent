from __future__ import annotations

import threading

from riskmodel_checker.agent.service import agent_conclusions_confirmed
from riskmodel_checker.db import TaskRepository
from riskmodel_checker.domain import TaskRecord, TaskStatus
from riskmodel_checker.pipeline import (
    METRICS_STAGE_FAILURE_PREFIX,
    NOTEBOOK_STAGE_FAILURE_PREFIX,
)


class AgentValidationCancelled(Exception):
    pass


_AGENT_CANCELLATION_LOCK = threading.Lock()
_AGENT_CANCELLATION_REQUESTS: set[str] = set()


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


def request_agent_cancellation(task_id: str) -> None:
    with _AGENT_CANCELLATION_LOCK:
        _AGENT_CANCELLATION_REQUESTS.add(task_id)


def clear_agent_cancellation(task_id: str) -> None:
    with _AGENT_CANCELLATION_LOCK:
        _AGENT_CANCELLATION_REQUESTS.discard(task_id)


def agent_cancellation_requested(task_id: str) -> bool:
    with _AGENT_CANCELLATION_LOCK:
        return task_id in _AGENT_CANCELLATION_REQUESTS


def raise_if_agent_cancelled(task_id: str) -> None:
    if agent_cancellation_requested(task_id):
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
