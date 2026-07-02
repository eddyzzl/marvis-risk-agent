from __future__ import annotations

from pathlib import Path
import traceback

from fastapi import HTTPException, Request

from marvis.api_task_helpers import (
    ACTIVE_JOB_DETAIL,
    dispatch_platform_hook,
)
from marvis.api_task_payloads import normalized_status_reason
from marvis.db import TaskRepository
from marvis.domain import TASK_STATUS_REASON_USER_CANCELLED, TaskRecord
from marvis.execution_environment import load_execution_environment
from marvis.job_heartbeat import heartbeat_job
from marvis.pipeline import PipelineSettings
from marvis.state_machine import ConflictError


def start_task_job(repo: TaskRepository, task_id: str, kind: str) -> str:
    try:
        return repo.start_job(task_id, kind)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL) from exc


def fail_queued_job(repo: TaskRepository, job_id: str, exc: Exception) -> None:
    repo.finish_job(
        job_id,
        status="failed",
        error_name=exc.__class__.__name__,
        error_value=str(exc),
        traceback="",
    )


def run_stage_job(
    job_id: str,
    db_path: Path,
    stage_func,
    kwargs: dict,
    *,
    success_agent_notice: str | None = None,
    hook_dispatcher=None,
    before_hook_event: str | None = None,
    after_hook_event: str | None = None,
) -> None:
    repo = TaskRepository(db_path)
    repo.mark_job_running(job_id)
    task_id = kwargs.get("task_id")
    task_id_text = str(task_id) if task_id else None
    dispatch_platform_hook(
        hook_dispatcher,
        before_hook_event,
        stage_hook_payload(job_id, task_id_text),
        task_id=task_id_text,
    )
    try:
        with heartbeat_job(repo, job_id):
            stage_func(**kwargs)
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    else:
        job_status = (
            "cancelled"
            if stage_returned_cancelled_task(repo, task_id)
            else "succeeded"
        )
        repo.finish_job(job_id, status=job_status)
        if job_status == "succeeded":
            dispatch_platform_hook(
                hook_dispatcher,
                after_hook_event,
                stage_hook_payload(job_id, task_id_text, status=job_status),
                task_id=task_id_text,
            )
        if job_status == "succeeded" and success_agent_notice == "word_report_ready":
            add_agent_report_ready_message(repo, task_id)


def stage_hook_payload(
    job_id: str,
    task_id: str | None,
    *,
    status: str | None = None,
) -> dict:
    payload = {"job_id": job_id}
    if task_id:
        payload["task_id"] = task_id
    if status:
        payload["status"] = status
    return payload


def add_agent_report_ready_message(repo: TaskRepository, task_id: str | None) -> None:
    if not task_id:
        return
    task = repo.get_task(task_id)
    if task.run_mode != "agent":
        return
    messages = repo.list_agent_messages(task_id)
    latest_confirmed_index = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("stage") == "word_conclusion_confirmed"
        ),
        default=-1,
    )
    if latest_confirmed_index < 0:
        return
    if any(
        message.get("stage") == "word_report_ready"
        for message in messages[latest_confirmed_index + 1 :]
    ):
        return
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_report_ready",
        content=(
            "报告已生成。右侧步骤里的“预览”可以在线查看 Word，"
            "“下载Word”用于下载验证报告，“下载Excel”用于下载指标分析明细。"
        ),
        metadata={"report_ready": True},
    )


def stage_returned_cancelled_task(repo: TaskRepository, task_id: str | None) -> bool:
    if not task_id:
        return False
    try:
        task = repo.get_task(task_id)
    except Exception:
        return False
    return normalized_status_reason(task.status_reason_code) == (
        TASK_STATUS_REASON_USER_CANCELLED
    )


def pipeline_settings_from_request(
    request: Request,
    task: TaskRecord,
    feature_columns: list[str] | None,
) -> PipelineSettings:
    return pipeline_settings_from_settings(request.app.state.settings, task, feature_columns)


def pipeline_settings_from_settings(
    settings,
    task: TaskRecord,
    feature_columns: list[str] | None,
) -> PipelineSettings:
    return PipelineSettings(
        workspace=settings.workspace,
        db_path=settings.db_path,
        report_template_path=settings.report_template_path,
        feature_columns=feature_columns or task.feature_columns,
        notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
    )


def agent_pipeline_settings(settings, task: TaskRecord) -> PipelineSettings:
    return pipeline_settings_from_settings(settings, task, task.feature_columns)
