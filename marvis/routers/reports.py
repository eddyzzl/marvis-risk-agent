from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from marvis.api_report_helpers import (
    latest_driver_report_path,
    require_confirmed_agent_conclusions,
)
from marvis.api_task_helpers import get_task_or_404
from marvis.api_task_payloads import task_report_download_filename
from marvis.db import TaskRepository
from marvis.domain import TaskStatus
from marvis.output.word_preview import docx_to_html_preview
from marvis.pipeline import REPORT_STAGE_FAILURE_PREFIX


router = APIRouter(prefix="/api", tags=["reports"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/report/download")
def download_task_report(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    require_confirmed_agent_conclusions(repo, task)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(status_code=404, detail="report not generated")
    report_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation_report.docx"
    )
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=task_report_download_filename(task, ".docx"),
    )


@router.get("/tasks/{task_id}/report/preview")
def preview_task_report(task_id: str, request: Request) -> HTMLResponse:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    require_confirmed_agent_conclusions(repo, task)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(status_code=404, detail="report not generated")
    report_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation_report.docx"
    )
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return HTMLResponse(docx_to_html_preview(report_path))


@router.get("/tasks/{task_id}/analysis/download")
def download_task_analysis(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.status not in {
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    } and not (
        task.status == TaskStatus.FAILED
        and task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
    ):
        raise HTTPException(status_code=404, detail="analysis not generated")
    analysis_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation.xlsx"
    )
    if not analysis_path.exists():
        raise HTTPException(status_code=404, detail="analysis not generated")
    return FileResponse(
        analysis_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=task_report_download_filename(task, ".xlsx"),
    )


@router.get("/tasks/{task_id}/driver-report/download")
def download_driver_report(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    report_path = latest_driver_report_path(request.app.state, task_id)
    if report_path is None or not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=task_report_download_filename(task, ".xlsx"),
    )
