from datetime import datetime
from pathlib import Path
import re
import sqlite3

from marvis.api_task_helpers import format_validation_batch_parent_name
from marvis.repositories.tasks import TaskRepository
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TASK_STATUS_REASON_USER_CANCELLED,
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_PORTFOLIO,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VALIDATION_BATCH,
    TASK_TYPE_VINTAGE,
    TaskRecord,
    TaskStatus,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.validation_batches import ValidationBatchRepository
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.safe_paths import safe_filename_component
from marvis.validation_completion import legacy_validation_delivery_complete

_UNSET = object()
_TASK_REPORT_LABELS = {
    TASK_TYPE_VALIDATION: "模型验证报告",
    TASK_TYPE_FEATURE_ANALYSIS: "特征分析报告",
    TASK_TYPE_DATA_JOIN: "数据处理报告",
    TASK_TYPE_MODELING: "模型开发报告",
    TASK_TYPE_STRATEGY: "策略分析报告",
    TASK_TYPE_VINTAGE: "风险分析报告",
    TASK_TYPE_PORTFOLIO: "组合分析报告",
}


def _latest_workflow_statuses(
    repo: TaskRepository,
    task_ids: list[str],
) -> dict[str, str]:
    """Read the optional V2 plan projection without breaking V1-only stores.

    Validation-only embeddings and a few supported migration/test fixtures own a
    task database that predates the orchestration tables.  ``workflow_status`` is
    additive metadata for those callers, so an absent ``plans`` table means
    "there is no plan state", not that the otherwise valid task API is unusable.
    Other SQLite failures still propagate.
    """

    try:
        return PlanRepository(repo.db_path).latest_workflow_statuses_for_tasks(
            task_ids
        )
    except sqlite3.OperationalError as exc:
        if "no such table: plans" not in str(exc).lower():
            raise
        return {}


def _latest_current_steps(
    repo: TaskRepository,
    task_ids: list[str],
) -> dict[str, dict]:
    """Current step of each task's latest plan (home 接着做 progress); a store
    without the plan tables has none, like ``_latest_workflow_statuses``."""

    try:
        return PlanRepository(repo.db_path).latest_current_steps_for_tasks(task_ids)
    except sqlite3.OperationalError as exc:
        if "no such table: plan" not in str(exc).lower():
            raise
        return {}


def _pending_input_confirmations(
    repo: TaskRepository,
    tasks: list[TaskRecord],
) -> set[str]:
    """Validation tasks whose input contract waits on the user's field check.

    Like the plan projection, the contract table is additive: a store that
    predates it simply has no pending confirmations.
    """

    task_ids = [task.id for task in tasks if task.task_type == TASK_TYPE_VALIDATION]
    if not task_ids:
        return set()
    try:
        return ValidationContractRepository(repo.db_path).pending_confirmation_task_ids(
            task_ids
        )
    except sqlite3.OperationalError as exc:
        if "no such table: validation_input_contracts" not in str(exc).lower():
            raise
        return set()


def task_payload(
    repo: TaskRepository,
    task: TaskRecord,
    tasks_dir: Path | None = None,
    *,
    active_job_kind: str | None | object = _UNSET,
    workflow_status: str | None | object = _UNSET,
    batch_item_count: int | None | object = _UNSET,
    batch_summary: dict | None | object = _UNSET,
    awaiting_input_confirmation: bool | object = _UNSET,
    current_step: dict | None | object = _UNSET,
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
    resolved_workflow_status = (
        _latest_workflow_statuses(repo, [task.id]).get(task.id)
        if workflow_status is _UNSET
        else workflow_status
    )
    resolved_awaiting_input_confirmation = (
        task.id in _pending_input_confirmations(repo, [task])
        if awaiting_input_confirmation is _UNSET
        else bool(awaiting_input_confirmation)
    )
    resolved_current_step = (
        _latest_current_steps(repo, [task.id]).get(task.id)
        if current_step is _UNSET
        else current_step
    )
    payload = {
        **task_to_dict(task),
        "workflow_status": resolved_workflow_status,
        "current_step": resolved_current_step,
        "awaiting_input_confirmation": resolved_awaiting_input_confirmation,
        "business_objective_locked": resolved_workflow_status is not None,
        "active_job_kind": resolved_active_job_kind,
        "failure_stage": task_failure_stage(repo, task),
        "failure_reason_code": task_failure_reason_code(task),
        "stop_reason_code": task_stop_reason_code(repo, task),
        "stopped": task_stopped(repo, task),
        "report_available": task_report_available(tasks_dir, task.id),
    }
    if legacy_validation_delivery_complete(
        repo, task, tasks_dir,
        active_job_kind=resolved_active_job_kind,
        workflow_status=resolved_workflow_status,
        awaiting_input_confirmation=resolved_awaiting_input_confirmation,
    ):
        payload["status"] = TaskStatus.SUCCEEDED
        payload["status_message"] = "pipeline succeeded"
    return _apply_validation_batch_display_fields(
        payload,
        task,
        batch_item_count=batch_item_count,
        batch_summary=batch_summary,
        repo=repo,
    )


def list_task_payloads(
    repo: TaskRepository,
    tasks: list[TaskRecord],
    tasks_dir: Path | None = None,
) -> list[dict]:
    """Batched task_payload for the polling task-list endpoint (PERF-6): resolves
    active_job_kind for all tasks with a single query instead of one connection
    per task, then reuses task_payload's per-task field derivation unchanged.
    The latest plan workflow state is batched the same way so polling does not
    introduce one plan query per task."""
    task_ids = [task.id for task in tasks]
    active_job_kinds = repo.get_active_job_kinds_for_tasks(task_ids)
    workflow_statuses = _latest_workflow_statuses(repo, task_ids)
    current_steps = _latest_current_steps(repo, task_ids)
    batch_ids = [
        task.id for task in tasks if task.task_type == TASK_TYPE_VALIDATION_BATCH
    ]
    batch_repo = ValidationBatchRepository(repo.db_path)
    batch_item_counts = batch_repo.item_counts_for_parents(batch_ids) if batch_ids else {}
    batch_summaries = batch_repo.status_summaries_for_parents(batch_ids) if batch_ids else {}
    pending_confirmations = _pending_input_confirmations(repo, tasks)
    return [
        task_payload(
            repo,
            task,
            tasks_dir,
            active_job_kind=active_job_kinds.get(task.id),
            workflow_status=workflow_statuses.get(task.id),
            batch_item_count=(
                batch_item_counts.get(task.id)
                if task.task_type == TASK_TYPE_VALIDATION_BATCH
                else None
            ),
            batch_summary=(
                batch_summaries.get(task.id)
                if task.task_type == TASK_TYPE_VALIDATION_BATCH
                else None
            ),
            awaiting_input_confirmation=task.id in pending_confirmations,
            current_step=current_steps.get(task.id),
        )
        for task in tasks
    ]


def _counted_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for label in labels:
        if label:
            counts[label] = counts.get(label, 0) + 1
    return [label if n == 1 else f"{label} ×{n}" for label, n in counts.items()]


def _short(text: str, limit: int = 24) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def purge_name_lists(names: dict) -> dict:
    """Readable names for the delete dialog (V3 UI-4), keyed like the purge
    summary. Plan goals keep only their template name ("模型开发: dataset_id=…"
    -> "模型开发"); internal ids never reach the dialog."""
    from marvis.agent.presenters.display_labels import recipe_label

    plans = []
    for goal in names.get("plan_goals") or []:
        head = re.split(r"[:：]", str(goal), maxsplit=1)[0]
        plans.append(_short(head))
    return {
        "datasets": [_short(name, 32) for name in names.get("datasets") or []],
        "derived_datasets": int(names.get("derived_datasets") or 0),
        "plans": _counted_labels(plans),
        "experiments": _counted_labels(
            [recipe_label(code) for code in names.get("experiment_recipes") or []]
        ),
        "strategies": _counted_labels(
            [_short(text) for text in names.get("strategy_descriptions") or []]
        ),
    }


def task_to_dict(task: TaskRecord) -> dict:
    from dataclasses import asdict

    return asdict(task)


def _parse_iso_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _apply_validation_batch_display_fields(
    payload: dict,
    task: TaskRecord,
    *,
    batch_item_count: int | None | object,
    batch_summary: dict | None | object = _UNSET,
    repo: TaskRepository,
) -> dict:
    if task.task_type != TASK_TYPE_VALIDATION_BATCH:
        return payload
    summary = (
        ValidationBatchRepository(repo.db_path).status_summaries_for_parents([task.id]).get(task.id)
        if batch_summary is _UNSET
        else batch_summary
    )
    if summary is not None:
        payload["batch_summary"] = summary
    if batch_item_count is _UNSET:
        counts = ValidationBatchRepository(repo.db_path).item_counts_for_parents(
            [task.id]
        )
        item_count = counts.get(task.id)
    else:
        item_count = batch_item_count
    if item_count is None:
        return payload
    count = int(item_count)
    payload["item_count"] = count
    payload["model_name"] = format_validation_batch_parent_name(
        count,
        _parse_iso_datetime(task.created_at),
    )
    return payload


def task_report_available(tasks_dir: Path | None, task_id: str) -> bool:
    if tasks_dir is None:
        return False
    return (tasks_dir / task_id / "outputs" / "validation_report.docx").exists()


def task_report_download_filename(task: TaskRecord, suffix: str) -> str:
    model_name = safe_filename_component(task.model_name, fallback="模型")
    report_label = _TASK_REPORT_LABELS.get(task.task_type, "模型验证报告")
    return (
        f"{model_name}_{report_label}_"
        f"{task_created_date_for_filename(task)}{suffix}"
    )


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
