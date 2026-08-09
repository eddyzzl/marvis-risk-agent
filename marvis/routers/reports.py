from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from marvis.errors import not_found

from marvis.api_report_helpers import (
    driver_report_artifact,
    latest_driver_report_artifacts,
    require_confirmed_agent_conclusions,
)
from marvis.api_task_helpers import get_task_or_404
from marvis.api_task_payloads import task_report_download_filename
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskStatus
from marvis.output.word_preview import docx_to_html_preview
from marvis.pipeline import REPORT_STAGE_FAILURE_PREFIX
from marvis.safe_paths import resolve_fixed_task_output


router = APIRouter(prefix="/api", tags=["reports"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/report/download")
def download_task_report(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    require_confirmed_agent_conclusions(repo, task)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise not_found("report not generated")
    report_path = resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        task_id,
        "validation_report.docx",
    )
    if report_path is None:
        raise not_found("report not generated")
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
        raise not_found("report not generated")
    report_path = resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        task_id,
        "validation_report.docx",
    )
    if report_path is None:
        raise not_found("report not generated")
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
        raise not_found("analysis not generated")
    analysis_path = resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        task_id,
        "validation.xlsx",
    )
    if analysis_path is None:
        raise not_found("analysis not generated")
    return FileResponse(
        analysis_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=task_report_download_filename(task, ".xlsx"),
    )


def _registered_report_redirect(
    *,
    task_id: str,
    artifact_id: str,
    content_hash: str,
) -> RedirectResponse:
    target = (
        f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
        f"{quote(artifact_id, safe='')}/download"
        f"?expected_content_hash={quote(content_hash, safe='')}"
    )
    return RedirectResponse(target, status_code=307)


@router.get("/tasks/{task_id}/driver-report/download")
def download_driver_report(task_id: str, request: Request) -> Response:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    artifacts = latest_driver_report_artifacts(request.app.state, task_id)
    if not artifacts:
        raise not_found("report not generated")
    artifact = artifacts[0]
    if artifact.artifact_id and artifact.content_hash:
        return _registered_report_redirect(
            task_id=task_id,
            artifact_id=artifact.artifact_id,
            content_hash=artifact.content_hash,
        )
    safe_path = resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        task_id,
        artifact.path.name,
    )
    if safe_path is None or safe_path != artifact.path:
        raise not_found("report not generated")
    return FileResponse(
        safe_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=task_report_download_filename(task, ".xlsx"),
    )


@router.get("/tasks/{task_id}/driver-reports/{report_id}/download")
def download_driver_report_artifact(
    task_id: str,
    report_id: str,
    request: Request,
) -> Response:
    get_task_or_404(_repo(request), task_id)
    artifact = driver_report_artifact(request.app.state, task_id, report_id)
    if artifact is None:
        raise not_found("report not generated")
    if artifact.artifact_id and artifact.content_hash:
        return _registered_report_redirect(
            task_id=task_id,
            artifact_id=artifact.artifact_id,
            content_hash=artifact.content_hash,
        )
    safe_path = resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        task_id,
        artifact.path.name,
    )
    if safe_path is None or safe_path != artifact.path:
        raise not_found("report not generated")
    return FileResponse(
        safe_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=safe_path.name,
    )
