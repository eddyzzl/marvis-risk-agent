from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import traceback
from typing import Any

from marvis.agent.orchestrator import AgentValidationCancelled
from marvis.db import TaskRepository


@dataclass(frozen=True)
class ValidationJobCallbacks:
    """Compatibility seams for the validation-agent execution loop."""

    agent_auto_accept: Callable[[str | None], bool]
    agent_next_stage: Callable[[TaskRepository, Any], str | None]
    raise_if_agent_cancelled: Callable[[str], None]
    open_agent_stage: Callable[..., None]
    run_scan_stage: Callable[..., bool]
    run_reproducibility_stage: Callable[..., bool]
    run_metrics_stage: Callable[..., bool]
    run_word_conclusion_stage: Callable[..., bool]
    finalize_agent_opening_message: Callable[..., None]
    mark_agent_cancelled: Callable[[TaskRepository, str], None]
    agent_has_stop_ack_message: Callable[[TaskRepository, str], bool]
    add_exception_summary: Callable[..., None]
    clear_agent_cancellation: Callable[[str], None]
    stop_ack_content: str


def run_agent_validation_job(
    job_id: str,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    opening_message_id: str | None = None,
    stage: str | None = None,
    stage_message_id: str | None = None,
    acceptance_mode: str | None = None,
    stage_instruction: str | None = None,
    callbacks: ValidationJobCallbacks,
) -> None:
    repo = TaskRepository(settings.db_path)
    repo.mark_job_running(job_id)
    auto_accept = callbacks.agent_auto_accept(acceptance_mode)
    try:
        current_stage = stage
        current_opening_message_id = opening_message_id
        current_stage_message_id = stage_message_id
        current_stage_instruction = stage_instruction
        while True:
            task = repo.get_task(task_id)
            current_stage = current_stage or callbacks.agent_next_stage(repo, task)
            if current_stage is None:
                if current_opening_message_id or not auto_accept:
                    callbacks.finalize_agent_opening_message(
                        repo,
                        task_id=task_id,
                        message_id=current_opening_message_id,
                        model_profile=model_profile,
                        content=(
                            "当前没有可继续执行的下一步。你可以继续询问已生成的验证结果，"
                            "或确认报告结论后生成 Word。"
                        ),
                    )
                repo.finish_job(job_id, status="succeeded")
                return
            callbacks.raise_if_agent_cancelled(task_id)
            callbacks.open_agent_stage(
                repo,
                task=task,
                task_id=task_id,
                stage=current_stage,
                model_profile=model_profile,
                opening_message_id=current_opening_message_id,
                auto_accept=auto_accept,
            )
            callbacks.raise_if_agent_cancelled(task_id)
            if current_stage == "scan":
                stage_succeeded = callbacks.run_scan_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "reproducibility":
                stage_succeeded = callbacks.run_reproducibility_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "metrics":
                stage_succeeded = callbacks.run_metrics_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "word_conclusion_draft":
                stage_succeeded = callbacks.run_word_conclusion_stage(
                    repo,
                    settings,
                    task_id,
                    model_profile,
                    draft_message_id=current_stage_message_id,
                    auto_accept=auto_accept,
                    rewrite_instruction=current_stage_instruction,
                )
            else:
                raise RuntimeError(f"unknown agent stage: {current_stage}")
            if not stage_succeeded:
                repo.finish_job(job_id, status="failed")
                return
            if not auto_accept:
                repo.finish_job(job_id, status="succeeded")
                return
            current_stage = callbacks.agent_next_stage(repo, repo.get_task(task_id))
            current_opening_message_id = None
            current_stage_message_id = None
            current_stage_instruction = None
            if current_stage is None:
                repo.finish_job(job_id, status="succeeded")
                return
    except AgentValidationCancelled as exc:
        callbacks.mark_agent_cancelled(repo, task_id)
        if not callbacks.agent_has_stop_ack_message(repo, task_id):
            repo.add_agent_message(
                task_id,
                role="assistant",
                stage="chat",
                content=callbacks.stop_ack_content,
                metadata={"cancelled": True, "intent": "stop", "cancel_requested": True},
            )
        repo.finish_job(
            job_id,
            status="cancelled",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback="",
        )
    except Exception as exc:
        try:
            task = repo.get_task(task_id)
            callbacks.add_exception_summary(
                repo,
                task_id=task_id,
                task=task,
                model_profile=model_profile,
                error=exc,
            )
        finally:
            repo.finish_job(
                job_id,
                status="failed",
                error_name=exc.__class__.__name__,
                error_value=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
    finally:
        callbacks.clear_agent_cancellation(task_id)


__all__ = ["ValidationJobCallbacks", "run_agent_validation_job"]
