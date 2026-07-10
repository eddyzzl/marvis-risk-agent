from marvis.errors import conflict, unprocessable

from marvis.agent.orchestrator import (
    agent_cancellation_requested as _agent_cancellation_requested,
    clear_agent_cancellation,
    is_metrics_failure,
    raise_if_agent_cancelled as _raise_if_agent_cancelled,
    request_agent_cancellation,
)
from marvis.db import TaskRepository
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    TaskRecord,
    TaskStatus,
)
from marvis.notebook_cancellation import (
    clear_pending_notebook_cancellation,
    request_notebook_cancellation,
)
from marvis.notebooks import close_live_notebook_session
from marvis.pipeline import NOTEBOOK_STAGE_FAILURE_PREFIX, REPORT_STAGE_FAILURE_PREFIX


AGENT_STOP_ACK_CONTENT = "已停止当前动作，请问有什么指示？"
AGENT_STOP_STATUS_MESSAGE = "已停止当前动作"


def reset_agent_task_for_rerun(
    repo: TaskRepository,
    task_id: str,
    stage: str,
) -> TaskRecord:
    target_status = {
        "scan": TaskStatus.CREATED,
        "reproducibility": TaskStatus.SCANNED,
        "metrics": TaskStatus.EXECUTED,
        "word_conclusion_draft": TaskStatus.WRITING_ARTIFACTS,
    }.get(stage)
    if target_status is None:
        raise unprocessable(f"unknown rerun stage: {stage}")
    repo.reset_status_for_agent_rerun(
        task_id,
        target_status,
        f"agent rerun requested: {stage}",
        clear_agent_report_conclusions=True,
    )
    if stage in {"scan", "reproducibility"}:
        close_live_notebook_session(task_id)
    return repo.get_task(task_id)


def require_agent_rerun_stage_reached(task: TaskRecord, stage: str) -> None:
    if stage == "scan":
        return
    if agent_rerun_stage_reached(task, stage):
        return
    raise conflict("尚未执行到该阶段，不能重新执行；请先按顺序完成前置验证步骤。")


def agent_rerun_stage_reached(task: TaskRecord, stage: str) -> bool:
    status = task.status
    if stage == "reproducibility":
        return (
            status
            in {
                TaskStatus.SCANNED,
                TaskStatus.RUNNING,
                TaskStatus.EXECUTED,
                TaskStatus.COMPUTING_METRICS,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or task.status_message.startswith(NOTEBOOK_STAGE_FAILURE_PREFIX)
            or is_metrics_failure(task)
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    if stage == "metrics":
        return (
            status
            in {
                TaskStatus.EXECUTED,
                TaskStatus.COMPUTING_METRICS,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or is_metrics_failure(task)
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    if stage == "word_conclusion_draft":
        return (
            status
            in {
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    return False


def agent_has_cancellable_work(repo: TaskRepository, task_id: str) -> bool:
    if repo.get_active_job_kind(task_id) == "agent":
        return True
    return any(
        message.get("role") == "assistant"
        and bool((message.get("metadata") or {}).get("streaming"))
        for message in repo.list_agent_messages(task_id)
    )


def agent_has_stop_ack_message(repo: TaskRepository, task_id: str) -> bool:
    for message in repo.list_agent_messages(task_id):
        metadata = message.get("metadata") or {}
        if (
            message.get("role") == "assistant"
            and metadata.get("intent") == "stop"
            and metadata.get("cancel_requested") is True
        ):
            return True
    return False


def handle_agent_stop_message(repo: TaskRepository, task: TaskRecord) -> dict:
    return handle_agent_stop_message_with_callbacks(
        repo,
        task,
        request_agent_cancellation_fn=request_agent_cancellation,
        request_notebook_cancellation_fn=request_notebook_cancellation,
    )


def handle_agent_stop_message_with_callbacks(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    request_agent_cancellation_fn,
    request_notebook_cancellation_fn,
) -> dict:
    if not agent_has_cancellable_work(repo, task.id):
        message = repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="当前没有正在执行的 Agent 任务，无需停止。需要继续验证时可以重新发送指令。",
            metadata={"intent": "stop", "active_job": None},
        )
        return {
            "task_id": task.id,
            "status": "message_saved",
            "message": message["content"],
            "messages": repo.list_agent_messages(task.id),
        }
    active_job = repo.get_latest_job(task.id, kind="agent")
    active_job_id = (
        str(active_job["id"])
        if active_job is not None
        and active_job.get("status") in {"queued", "running"}
        else None
    )
    request_agent_cancellation_fn(task.id, job_id=active_job_id)
    request_notebook_cancellation_fn(task.id)
    mark_agent_cancelled(repo, task.id)
    if agent_has_stop_ack_message(repo, task.id):
        ack_content = AGENT_STOP_ACK_CONTENT
    else:
        ack_content = repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=AGENT_STOP_ACK_CONTENT,
            metadata={"intent": "stop", "cancel_requested": True},
        )["content"]
    return {
        "task_id": task.id,
        "status": "cancel_requested",
        "message": ack_content,
        "messages": repo.list_agent_messages(task.id),
    }


def clear_agent_and_notebook_cancellation(
    task_id: str,
    *,
    job_id: str | None = None,
) -> None:
    clear_agent_cancellation(task_id, job_id=job_id)
    clear_pending_notebook_cancellation(task_id)


def agent_cancellation_requested(task_id: str, *, job_id: str | None = None) -> bool:
    return _agent_cancellation_requested(task_id, job_id=job_id)


def raise_if_agent_cancelled(task_id: str, *, job_id: str | None = None) -> None:
    _raise_if_agent_cancelled(task_id, job_id=job_id)


def mark_agent_cancelled(repo: TaskRepository, task_id: str) -> None:
    try:
        task = repo.get_task(task_id)
        resume_status_by_current = {
            TaskStatus.SCANNED: TaskStatus.SCANNED,
            TaskStatus.RUNNING: TaskStatus.SCANNED,
            TaskStatus.COMPUTING_METRICS: TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS: TaskStatus.REVIEW_REQUIRED,
        }
        resume_status = resume_status_by_current.get(task.status)
        if resume_status is None:
            return
        if task.status == resume_status:
            repo.update_status_message(
                task_id,
                AGENT_STOP_STATUS_MESSAGE,
                reason_code=TASK_STATUS_REASON_USER_CANCELLED,
            )
            return
        repo.update_status(
            task_id,
            resume_status,
            AGENT_STOP_STATUS_MESSAGE,
            expected=task.status,
            reason_code=TASK_STATUS_REASON_USER_CANCELLED,
        )
    except Exception:
        pass
