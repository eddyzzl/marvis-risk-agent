from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from marvis.errors import conflict

from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.domain import TaskStatus
from marvis.files import write_text_atomic
from marvis.job_cancellation import request_job_cancellation
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
        raise conflict(f"cannot cancel notebook in status {task.status.value}")
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
        raise conflict(f"cannot cancel metrics in status {task.status.value}")
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
        raise conflict(f"cannot cancel report in status {task.status.value}")
    if repo.get_active_job_kind(task_id) != "report":
        raise conflict("task has no active report job")
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "report cancellation requested; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/join/cancel", status_code=202)
def cancel_task_join(task_id: str, request: Request) -> dict:
    """Cooperative cancel for a running join execution job (REL-5). Unlike
    notebook/metrics/report (which interrupt a kernel or watch a marker file),
    execute_join_plan has no external process to signal — this flips an
    in-memory JobCancellationToken that the join engine checks between feature
    joins (marvis/data/join_engine.py), the only safe rollback boundary."""
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    job_id = _active_job_id(repo, task_id, "join")
    if job_id is None:
        raise conflict("task has no active join job")
    request_job_cancellation(job_id)
    return {
        "task_id": task_id,
        "job_id": job_id,
        "status": "accepted",
        "message": "join cancellation requested; poll GET /api/tasks/{task_id}",
    }


def _active_job_id(repo: TaskRepository, task_id: str, kind: str) -> str | None:
    job = repo.get_latest_job(task_id, kind=kind)
    if job is None or job.get("status") not in {"queued", "running"}:
        return None
    return str(job["id"])


def _write_metrics_cancel_marker(task_dir: Path) -> None:
    marker_path = _metrics_cancel_marker_path(task_dir)
    write_text_atomic(marker_path, "cancelled\n")
