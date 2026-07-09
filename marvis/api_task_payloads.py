from datetime import datetime
from pathlib import Path
import re

from marvis.db import TaskRepository
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TASK_STATUS_REASON_USER_CANCELLED,
    TaskRecord,
    TaskStatus,
)
from marvis.safe_paths import safe_filename_component

_UNSET = object()


def task_payload(
    repo: TaskRepository,
    task: TaskRecord,
    tasks_dir: Path | None = None,
    *,
    active_job_kind: str | None | object = _UNSET,
) -> dict:
    """``active_job_kind`` defaults to a sentinel so callers can distinguish "not
    supplied, look it up" from "supplied, and it really is None" (PERF-6: batch
    callers like list_task_payloads precompute active job kinds for every task in
    one query and pass the result through here instead of letting each payload
    open its own connection via repo.get_active_job_kind)."""
    resolved_active_job_kind = (
        repo.get_active_job_kind(task.id)
        if active_job_kind is _UNSET
        else active_job_kind
    )
    return {
        **task_to_dict(task),
        "active_job_kind": resolved_active_job_kind,
        "failure_stage": task_failure_stage(repo, task),
        "failure_reason_code": task_failure_reason_code(task),
        "stop_reason_code": task_stop_reason_code(repo, task),
        "stopped": task_stopped(repo, task),
        "report_available": task_report_available(tasks_dir, task.id),
    }


def list_task_payloads(
    repo: TaskRepository,
    tasks: list[TaskRecord],
    tasks_dir: Path | None = None,
) -> list[dict]:
    """Batched task_payload for the polling task-list endpoint (PERF-6): resolves
    active_job_kind for all tasks with a single query instead of one connection
    per task, then reuses task_payload's per-task field derivation unchanged."""
    active_job_kinds = repo.get_active_job_kinds_for_tasks([task.id for task in tasks])
    return [
        task_payload(
            repo,
            task,
            tasks_dir,
            active_job_kind=active_job_kinds.get(task.id),
        )
        for task in tasks
    ]


def task_to_dict(task: TaskRecord) -> dict:
    from dataclasses import asdict

    return asdict(task)


def task_report_available(tasks_dir: Path | None, task_id: str) -> bool:
    if tasks_dir is None:
        return False
    return (tasks_dir / task_id / "outputs" / "validation_report.docx").exists()


def task_report_download_filename(task: TaskRecord, suffix: str) -> str:
    model_name = safe_filename_component(task.model_name, fallback="模型")
    return f"{model_name}_模型验证报告_{task_created_date_for_filename(task)}{suffix}"


def task_created_date_for_filename(task: TaskRecord) -> str:
    raw_created_at = str(task.created_at or "").strip()
    try:
        parsed = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00"))
    except ValueError:
        digits = re.sub(r"\D+", "", raw_created_at)[:8]
        return digits or "unknown_date"
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y%m%d")


def task_failure_stage(repo: TaskRepository, task: TaskRecord) -> str | None:
    if task.status != TaskStatus.FAILED:
        return None
    if task_failed_during_scan(task):
        return "scan"
    message_stage = legacy_failure_stage_from_message(task.status_message)
    job_stage = failure_stage_from_job_kind(repo.get_latest_failed_job_kind(task.id))
    return earliest_failure_stage(message_stage, job_stage)


def task_failed_during_scan(task: TaskRecord) -> bool:
    return str(task.status_message or "").startswith("材料扫描失败：")


def task_failure_reason_code(task: TaskRecord) -> str | None:
    if task.status != TaskStatus.FAILED:
        return None
    reason = normalized_status_reason(task.status_reason_code)
    if reason == TASK_STATUS_REASON_SERVER_RESTART:
        return reason
    return legacy_failure_reason_code_from_message(task.status_message)


def task_stop_reason_code(repo: TaskRepository, task: TaskRecord) -> str | None:
    reason = normalized_status_reason(task.status_reason_code)
    if reason == TASK_STATUS_REASON_USER_CANCELLED:
        return reason
    # Successful terminals are never "stopped": only the structured
    # status_reason_code may mark a completed task as user-cancelled — not the fuzzy
    # legacy message text, which could contain "已取消"/"cancelled" incidentally.
    if task.status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        return None
    return legacy_stop_reason_code_from_message(task.status_message)


def task_stopped(repo: TaskRepository, task: TaskRecord) -> bool:
    return task_stop_reason_code(repo, task) == TASK_STATUS_REASON_USER_CANCELLED


def normalized_status_reason(reason: str | None) -> str:
    value = str(reason or "")
    if value in {
        TASK_STATUS_REASON_USER_CANCELLED,
        TASK_STATUS_REASON_SERVER_RESTART,
    }:
        return value
    return ""


def failure_stage_from_job_kind(kind: str | None) -> str | None:
    return {
        "notebook": "notebook",
        "metrics": "metrics",
        "report": "report",
    }.get(str(kind or ""))


def earliest_failure_stage(*stages: str | None) -> str | None:
    stage_order = {
        "scan": 0,
        "notebook": 1,
        "metrics": 2,
        "report": 3,
    }
    ranked = [
        stage
        for stage in stages
        if stage in stage_order
    ]
    if not ranked:
        return None
    return min(ranked, key=lambda stage: stage_order[stage])


def legacy_failure_stage_from_message(message: str) -> str | None:
    text = str(message or "")
    if re.search(
        r"模型可复现性验证失败|notebook failed at cell|reproducibility",
        text,
        flags=re.IGNORECASE,
    ):
        return "notebook"
    if re.search(
        r"模型效果&稳定性验证失败|指标|metrics|notebook metrics failed|"
        r"sample column check failed|data dictionary missing columns|"
        r"live notebook kernel is not available",
        text,
        flags=re.IGNORECASE,
    ):
        return "metrics"
    if re.search(r"报告输出失败|报告|Word|report", text, flags=re.IGNORECASE):
        return "report"
    if re.search(r"notebook", text, flags=re.IGNORECASE):
        return "notebook"
    return None


def legacy_failure_reason_code_from_message(message: str) -> str | None:
    if str(message or "") == "reclaimed: server restart while running":
        return TASK_STATUS_REASON_SERVER_RESTART
    return None


def legacy_stop_reason_code_from_message(message: str) -> str | None:
    text = str(message or "")
    if "cancelled" in text.lower() or "已停止" in text or "已取消" in text:
        return TASK_STATUS_REASON_USER_CANCELLED
    return None
