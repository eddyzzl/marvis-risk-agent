from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Request
from marvis.errors import conflict, unprocessable

from marvis.agent.orchestrator import is_metrics_failure
from marvis.api_report_helpers import require_confirmed_agent_conclusions
from marvis.api_scan_helpers import SCAN_FAILURE_PREFIX, is_scan_failure
from marvis.api_schemas import ValidateRequest
from marvis.api_stage_helpers import (
    fail_queued_job,
    pipeline_settings_from_request,
    pipeline_settings_from_settings,
    run_stage_job,
    start_task_job,
)
from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION, TaskStatus
from marvis.notebooks import close_live_notebook_session, get_live_notebook_session
from marvis.pipeline import (
    LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE,
    legacy_live_notebook_execution_allowed,
    run_metrics_stage,
    run_notebook_stage,
    run_report_stage,
    run_staged_pipeline,
)
from marvis.repositories.validation_contracts import (
    ValidationContractActiveJobConflict,
    ValidationContractRepository,
)
from marvis.state_machine import ConflictError, IllegalTransition


router = APIRouter(prefix="/api", tags=["validation-stages"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _start_v2_guarded_job(
    request: Request,
    repo: TaskRepository,
    task_id: str,
    kind: str,
):
    task = get_task_or_404(repo, task_id)
    if (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
    ):
        try:
            job_id = ValidationContractRepository(
                request.app.state.settings.db_path
            ).start_ready_job(task_id, kind)
        except (ValidationContractActiveJobConflict, ConflictError) as exc:
            raise conflict("task already has an active stage") from exc
        except ValueError as exc:
            raise unprocessable(str(exc)) from exc
    else:
        job_id = start_task_job(repo, task_id, kind)
    task = repo.get_task(task_id)
    return task, job_id


@router.post("/tasks/{task_id}/notebook", status_code=202)
def run_task_notebook(
    task_id: str,
    payload: ValidateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task, job_id = _start_v2_guarded_job(
        request, repo, task_id, "notebook"
    )
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        repo.finish_job(job_id, status="failed")
        raise conflict(f"cannot run notebook in status {task.status.value}")
    if is_scan_failure(task):
        detail = task.status_message.removeprefix(SCAN_FAILURE_PREFIX)
        repo.finish_job(job_id, status="failed")
        raise conflict(f"材料扫描未完整通过：{detail}")
    try:
        repo.update_status(
            task_id,
            TaskStatus.RUNNING,
            "notebook queued",
            expected={
                TaskStatus.SCANNED,
                TaskStatus.FAILED,
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            },
        )
    except IllegalTransition as exc:
        fail_queued_job(repo, job_id, exc)
        raise conflict(f"cannot run notebook in status {exc.current.value}") from exc
    background_tasks.add_task(
        run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_notebook_stage,
        {
            "task_id": task_id,
            "settings": pipeline_settings_from_request(
                request,
                task,
                payload.feature_columns,
            ),
            "stage_claimed": True,
            "cancellation_job_id": job_id,
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        after_hook_event="notebook.completed",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "notebook stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/metrics", status_code=202)
def run_task_metrics(
    task_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task, job_id = _start_v2_guarded_job(request, repo, task_id, "metrics")
    metrics_retry = is_metrics_failure(task)
    pipeline_settings = pipeline_settings_from_request(request, task, None)
    if (
        task.status
        in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}
        and not pipeline_settings.notebook_isolated_execution
    ):
        if not legacy_live_notebook_execution_allowed(pipeline_settings):
            close_live_notebook_session(task_id)
            repo.finish_job(job_id, status="failed")
            raise conflict(LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE)
        if get_live_notebook_session(task_id) is None:
            repo.finish_job(job_id, status="failed")
            raise conflict("live notebook kernel is not available; rerun notebook stage before metrics")
    if task.status in {
        TaskStatus.CREATED,
        TaskStatus.SCANNED,
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        repo.finish_job(job_id, status="failed")
        raise conflict(f"cannot generate metrics in status {task.status.value}")
    if task.status == TaskStatus.FAILED and not metrics_retry:
        repo.finish_job(job_id, status="failed")
        raise conflict(f"cannot generate metrics in status {task.status.value}")
    try:
        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            "metrics queued",
            expected={
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
                TaskStatus.FAILED,
            }
            if metrics_retry
            else {
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            },
        )
    except IllegalTransition as exc:
        fail_queued_job(repo, job_id, exc)
        raise conflict(f"cannot generate metrics in status {exc.current.value}") from exc
    background_tasks.add_task(
        run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_metrics_stage,
        {
            "task_id": task_id,
            "settings": pipeline_settings,
            "stage_claimed": True,
            "cancellation_job_id": job_id,
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        after_hook_event="validation.completed",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "metrics stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/report", status_code=202)
def run_task_report(
    task_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task, job_id = _start_v2_guarded_job(request, repo, task_id, "report")
    try:
        require_confirmed_agent_conclusions(repo, task)
    except Exception as exc:
        fail_queued_job(repo, job_id, exc)
        raise
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        repo.finish_job(job_id, status="failed")
        raise conflict(f"cannot generate report in status {task.status.value}")
    background_tasks.add_task(
        run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_report_stage,
        {
            "task_id": task_id,
            "settings": pipeline_settings_from_request(request, task, None),
            "cancellation_job_id": job_id,
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        before_hook_event="report.before_generate",
        after_hook_event="report.after_generate",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "word report stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/validate", status_code=202)
def validate_task(
    task_id: str,
    payload: ValidateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task, job_id = _start_v2_guarded_job(request, repo, task_id, "pipeline")
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.EXECUTED,
        TaskStatus.COMPUTING_METRICS,
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    }:
        repo.finish_job(job_id, status="failed")
        raise conflict(f"cannot validate task in status {task.status.value}")
    settings = request.app.state.settings
    background_tasks.add_task(
        run_stage_job,
        job_id,
        settings.db_path,
        run_staged_pipeline,
        {
            "task_id": task_id,
            "settings": pipeline_settings_from_settings(
                settings,
                task,
                payload.feature_columns,
            ),
            "cancellation_job_id": job_id,
        },
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "pipeline dispatched; poll GET /api/tasks/{task_id} for terminal status",
    }
