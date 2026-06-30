from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.domain import TaskStatus
from marvis.notebook_cancellation import request_notebook_cancellation
from marvis.pipeline import _metrics_cancel_marker_path


router = APIRouter(prefix="/api", tags=["stage-controls"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.post("/tasks/{task_id}/notebook/cancel", status_code=202)
def cancel_task_notebook(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.status != TaskStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel notebook in status {task.status.value}",
        )
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "notebook cancellation requested; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/metrics/cancel", status_code=202)
def cancel_task_metrics(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.status != TaskStatus.COMPUTING_METRICS:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel metrics in status {task.status.value}",
        )
    _write_metrics_cancel_marker(request.app.state.settings.tasks_dir / task_id)
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "metrics cancellation requested; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/report/cancel", status_code=202)
def cancel_task_report(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel report in status {task.status.value}",
        )
    if repo.get_active_job_kind(task_id) != "report":
        raise HTTPException(status_code=409, detail="task has no active report job")
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "report cancellation requested; poll GET /api/tasks/{task_id}",
    }


def _write_metrics_cancel_marker(task_dir: Path) -> None:
    marker_path = _metrics_cancel_marker_path(task_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("cancelled\n", encoding="utf-8")
