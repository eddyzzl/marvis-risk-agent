from __future__ import annotations

from fastapi import APIRouter, Request
from marvis.errors import conflict, unprocessable

from marvis.api_scan_helpers import perform_scan_task, scan_hook_payload
from marvis.api_task_helpers import (
    dispatch_platform_hook,
    get_task_or_404,
    reject_if_task_has_active_job,
)
from marvis.db import TaskRepository
from marvis.domain import TaskStatus


router = APIRouter(prefix="/api", tags=["scans"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.post("/tasks/{task_id}/scan")
def scan_task(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    reject_if_task_has_active_job(repo, task_id)
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        raise conflict(f"cannot scan task in status {task.status.value}")
    try:
        payload = perform_scan_task(repo, task, request.app.state.settings)
        if payload.get("status") == TaskStatus.SCANNED.value:
            dispatch_platform_hook(
                getattr(request.app.state, "hook_dispatcher", None),
                "task.scanned",
                scan_hook_payload(payload),
                task_id=task_id,
            )
        return payload
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        # scan_source_dir limit and source-dir errors are client-side invalid input.
        raise unprocessable(f"source dir invalid: {exc}") from exc
