from collections.abc import Callable
import json
import logging
from pathlib import Path
import traceback

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
)

from marvis.agent.orchestrator import (
    AgentValidationCancelled,
    agent_next_stage,
    is_metrics_failure,
    request_agent_cancellation,
)
from marvis.agent.service import (
    REQUIRED_AGENT_REPORT_KEYS,
    agent_conclusions_confirmed,
    agent_rerun_stage,  # noqa: F401 - compatibility export for validation-agent routes/tests.
    answer_chat_message,  # noqa: F401 - compatibility export for validation-agent routes/tests.
    compose_agent_start_message,
    failure_summary,
    generate_word_conclusions,
    is_agent_advance_intent,  # noqa: F401 - compatibility export for validation-agent routes.
    is_stop_validation_intent,  # noqa: F401 - compatibility export for validation-agent routes.
    summarize_stage,
)
from marvis.agent import validation_service as _validation_service
from marvis.agent_memory.api_support import (
    agent_memory_context_from_store as _agent_memory_context_from_store,
    audit_agent_memory_use_from_store as _audit_agent_memory_use_from_store,
    capture_user_preference_memory,
)
from marvis.agent_memory.extractors import extract_user_preference
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import TaskRepository
from marvis.agent.plan_driver import DriverError
from marvis.agent.turn_handlers import (
    DRIVER_AGENT_TASK_TYPES,
    DriverTurnRuntime,
    dispatch_driver_turn as dispatch_plan_driver_turn,
)
from marvis.api_task_helpers import (
    get_task_or_404 as _get_task_or_404,
    reject_if_task_has_active_job as _reject_if_task_has_active_job,  # noqa: F401
)
from marvis.api_report_field_helpers import (
    build_report_field_payload as _build_report_field_payload,
)
from marvis.api_scan_helpers import (
    SCAN_FAILURE_PREFIX,
    perform_scan_task as _perform_scan_task,
)
from marvis.api_stage_helpers import (
    add_agent_report_ready_message as _add_agent_report_ready_message,
    agent_pipeline_settings as _agent_pipeline_settings,
    fail_queued_job as _fail_queued_job,
    run_stage_job as _run_stage_job,
    start_task_job as _start_task_job,
)
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VINTAGE,
    TaskRecord,
    TaskStatus,
)
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_settings import (
    LLMSettingsError,
    resolve_llm_model,
)
from marvis.api_settings import router as settings_router
from marvis.api_task_payloads import (
    task_payload as _task_payload,  # noqa: F401 - compatibility alias for structure tests/imports.
)
from marvis.notebook_cancellation import request_notebook_cancellation
from marvis.pipeline import (
    run_metrics_stage,
    run_notebook_stage,
    run_report_stage,
)
from marvis.state_machine import ConflictError
from marvis.validation.overfitting import overfitting_check_from_validation_results


router = APIRouter(prefix="/api")
router.include_router(settings_router)
logger = logging.getLogger(__name__)

AGENT_STOP_ACK_CONTENT = _validation_service.AGENT_STOP_ACK_CONTENT
AGENT_STOP_STATUS_MESSAGE = _validation_service.AGENT_STOP_STATUS_MESSAGE
_agent_cancellation_requested = _validation_service.agent_cancellation_requested
_agent_has_cancellable_work = _validation_service.agent_has_cancellable_work
_agent_has_stop_ack_message = _validation_service.agent_has_stop_ack_message
_agent_rerun_stage_reached = _validation_service.agent_rerun_stage_reached
_clear_agent_cancellation = _validation_service.clear_agent_and_notebook_cancellation
_mark_agent_cancelled = _validation_service.mark_agent_cancelled
_raise_if_agent_cancelled = _validation_service.raise_if_agent_cancelled
_require_agent_rerun_stage_reached = (
    _validation_service.require_agent_rerun_stage_reached
)
_reset_agent_task_for_rerun = _validation_service.reset_agent_task_for_rerun


def _request_agent_cancellation(task_id: str) -> None:
    request_agent_cancellation(task_id)


def _handle_agent_stop_message(repo: TaskRepository, task: TaskRecord) -> dict:
    return _validation_service.handle_agent_stop_message_with_callbacks(
        repo,
        task,
        request_agent_cancellation_fn=request_agent_cancellation,
        request_notebook_cancellation_fn=request_notebook_cancellation,
    )


AGENT_ACCEPTANCE_NORMAL = "normal"
AGENT_ACCEPTANCE_AUTO = "auto_accept"
AGENT_ACCEPTANCE_MODES = {AGENT_ACCEPTANCE_NORMAL, AGENT_ACCEPTANCE_AUTO}


def _is_metrics_failure(task: TaskRecord) -> bool:
    return is_metrics_failure(task)


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _task_tier(request, task) -> str:
    """The capability tier name for a task's plan: the per-task pick if set, else the
    global settings default (spec §5.1). Affects only the autonomy budget
    (max_replan_iterations) — never gates/determinism/safety."""
    from marvis.orchestrator.capability import tier_from_settings

    if getattr(task, "capability_tier", ""):
        return task.capability_tier
    try:
        return tier_from_settings(request.app.state.settings).name
    except Exception:
        return "balanced"


def _confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    task: TaskRecord,
    task_id: str,
    settings,
    text_values: dict[str, str],
    expected_revision: int | None,
    background_tasks: BackgroundTasks,
    model_profile: dict | None = None,
    hook_dispatcher=None,
) -> dict:
    latest_task = _get_task_or_404(repo, task_id)
    if latest_task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate report in status {latest_task.status.value}",
        )
    if expected_revision is None:
        _, expected_revision = repo.get_report_values(task_id)
    job_id = _start_task_job(repo, task_id, "report")
    try:
        update_conclusions = getattr(repo, "update_agent_report_conclusions_with_audit", None)
        if callable(update_conclusions):
            revision = update_conclusions(
                task_id,
                text_values,
                expected_revision=expected_revision,
                audit={
                    "kind": "report.agent_conclusions.confirm",
                    "target_ref": task_id,
                    "outcome": "succeeded",
                    "detail": {
                        "keys": sorted(text_values),
                        "expected_revision": expected_revision,
                    },
                },
            )
        else:
            revision = repo.update_agent_report_conclusions(
                task_id,
                text_values,
                expected_revision=expected_revision,
            )
    except ConflictError as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metadata = {
        "revision": revision,
        "confirmed_keys": sorted(text_values),
    }
    if model_profile:
        metadata.update(_model_metadata(model_profile))
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已确认，将开始生成最终 Word 报告。",
        metadata=metadata,
    )
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        settings.db_path,
        run_report_stage,
        {
            "task_id": task_id,
            "settings": _agent_pipeline_settings(settings, latest_task),
        },
        success_agent_notice="word_report_ready",
        hook_dispatcher=hook_dispatcher,
        before_hook_event="report.before_generate",
        after_hook_event="report.after_generate",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "revision": revision,
        "message": "agent conclusions confirmed; word report stage dispatched",
        "messages": repo.list_agent_messages(task_id),
    }


def _run_agent_validation_job(
    job_id: str,
    settings,
    task_id: str,
    model_profile: dict,
    opening_message_id: str | None = None,
    stage: str | None = None,
    stage_message_id: str | None = None,
    acceptance_mode: str | None = None,
    stage_instruction: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    repo.mark_job_running(job_id)
    auto_accept = _agent_auto_accept(acceptance_mode)
    try:
        current_stage = stage
        current_opening_message_id = opening_message_id
        current_stage_message_id = stage_message_id
        current_stage_instruction = stage_instruction
        while True:
            task = repo.get_task(task_id)
            current_stage = current_stage or _agent_next_stage(repo, task)
            if current_stage is None:
                if current_opening_message_id or not auto_accept:
                    _finalize_agent_opening_message(
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
            _raise_if_agent_cancelled(task_id)
            _open_agent_stage(
                repo,
                task=task,
                task_id=task_id,
                stage=current_stage,
                model_profile=model_profile,
                opening_message_id=current_opening_message_id,
                auto_accept=auto_accept,
            )
            _raise_if_agent_cancelled(task_id)
            if current_stage == "scan":
                stage_succeeded = _run_agent_scan_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "reproducibility":
                stage_succeeded = _run_agent_reproducibility_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "metrics":
                stage_succeeded = _run_agent_metrics_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "word_conclusion_draft":
                stage_succeeded = _run_agent_word_conclusion_stage(
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
            current_stage = _agent_next_stage(repo, repo.get_task(task_id))
            current_opening_message_id = None
            current_stage_message_id = None
            current_stage_instruction = None
            if current_stage is None:
                repo.finish_job(job_id, status="succeeded")
                return
    except AgentValidationCancelled as exc:
        _mark_agent_cancelled(repo, task_id)
        if not _agent_has_stop_ack_message(repo, task_id):
            repo.add_agent_message(
                task_id,
                role="assistant",
                stage="chat",
                content=AGENT_STOP_ACK_CONTENT,
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
            error_detail = f"{exc.__class__.__name__}: {exc}"
            _add_and_stream_agent_message(
                repo,
                task_id,
                stage="failure",
                model_profile=model_profile,
                producer=lambda on_delta: failure_summary(
                    task=task,
                    stage="Agent 执行",
                    error=error_detail,
                    model_profile=model_profile,
                    on_delta=on_delta,
                ),
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
        _clear_agent_cancellation(task_id)


def _agent_next_stage(repo: TaskRepository, task: TaskRecord) -> str | None:
    return agent_next_stage(repo, task, scan_failure_prefix=SCAN_FAILURE_PREFIX)



def _open_agent_stage(
    repo: TaskRepository,
    *,
    task: TaskRecord,
    task_id: str,
    stage: str,
    model_profile: dict,
    opening_message_id: str | None,
    auto_accept: bool = False,
) -> None:
    # Scan is the entry stage; the next message is the agent's substantive
    # opening (compose_agent_start_message) so a separate "接下来开始执行..."
    # banner here is redundant chatter. The banner stays for later stages
    # where it follows the previous stage's wrap-up.
    if auto_accept and stage != "scan":
        _add_agent_auto_stage_start_message(
            repo,
            task_id=task_id,
            stage=stage,
            model_profile=model_profile,
        )
    if stage == "scan":
        if opening_message_id:
            _stream_agent_message(
                repo,
                opening_message_id,
                task_id=task_id,
                model_profile=model_profile,
                producer=lambda on_delta: compose_agent_start_message(
                    task=task,
                    model_profile=model_profile,
                    on_delta=on_delta,
                ),
            )
            return
        _add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: compose_agent_start_message(
                task=task,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
        )
        return
    if auto_accept:
        return
    _finalize_agent_opening_message(
        repo,
        task_id=task_id,
        message_id=opening_message_id,
        model_profile=model_profile,
        content=_agent_stage_opening_text(stage),
    )


def _add_agent_auto_stage_start_message(
    repo: TaskRepository,
    *,
    task_id: str,
    stage: str,
    model_profile: dict,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=f"接下来开始执行{_agent_stage_label(stage)}。",
        metadata={
            **_model_metadata(model_profile),
            "auto_accept": True,
            "auto_stage_start": stage,
            "streaming": False,
        },
    )


def _finalize_agent_opening_message(
    repo: TaskRepository,
    *,
    task_id: str,
    message_id: str | None,
    model_profile: dict,
    content: str,
) -> None:
    metadata = {**_model_metadata(model_profile), "streaming": False}
    if message_id:
        repo.update_agent_message(message_id, content=content, metadata=metadata)
        return
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=content,
        metadata=metadata,
    )


def _agent_stage_opening_text(stage: str) -> str:
    if stage == "reproducibility":
        return "收到，我将继续执行模型可复现性验证，运行 Notebook 并检查代码模型分数与提交 PMML 分数的一致性。"
    if stage == "metrics":
        return "收到，我将继续执行模型效果与稳定性验证，计算 KS、PSI、分箱和压力测试等指标。"
    if stage == "word_conclusion_draft":
        return "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。"
    return "收到，我将继续执行下一步验证。"


def _run_agent_scan_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="scan",
        content=(
            "正在调用材料识别工具 scan_materials：读取材料目录，识别 Notebook、样本数据、"
            "PMML 模型和数据字典，并检查 Notebook RMC 契约。"
        ),
        metadata={
            **_model_metadata(model_profile),
            "tool_call": {
                "name": "scan_materials",
                "stage": "scan",
            },
        },
    )
    _raise_if_agent_cancelled(task_id)
    scan_payload = _perform_scan_task(repo, task, settings)
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="材料完备性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    _add_and_stream_agent_message(
        repo,
        task_id,
        stage="scan",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="scan",
            evidence=scan_payload,
            model_profile=model_profile,
            fallback="材料扫描完成，平台已识别必需验证材料。",
            on_delta=on_delta,
        ),
    )
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(repo, task_id, model_profile, next_stage="reproducibility")
    return True


def _run_agent_reproducibility_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    repo.update_status(
        task_id,
        TaskStatus.RUNNING,
        "agent notebook queued",
        expected={TaskStatus.SCANNED, TaskStatus.FAILED},
    )
    _raise_if_agent_cancelled(task_id)
    run_notebook_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, task),
        stage_claimed=True,
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="模型可复现性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="reproducibility",
        evidence=evidence,
    )
    message = _add_and_stream_agent_message(
        repo,
        task_id,
        stage="reproducibility",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="reproducibility",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback="分数一致性阶段已完成，请查看可复现性证据明细。",
            on_delta=on_delta,
        ),
    )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(repo, task_id, model_profile, next_stage="metrics")
    return True


def _run_agent_metrics_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED and _is_metrics_failure(task):
        expected_statuses = {
            TaskStatus.FAILED,
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    else:
        expected_statuses = {
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    repo.update_status(
        task_id,
        TaskStatus.COMPUTING_METRICS,
        "agent metrics queued",
        expected=expected_statuses,
    )
    _raise_if_agent_cancelled(task_id)
    run_metrics_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, task),
        stage_claimed=True,
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="效果和稳定性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="metrics",
        evidence=evidence,
    )
    message = _add_and_stream_agent_message(
        repo,
        task_id,
        stage="metrics",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="metrics",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback="效果、稳定性和 Excel 指标产物已生成，请结合 OOT KS、PSI 和压力测试明细复核。",
            on_delta=on_delta,
        ),
    )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(
            repo, task_id, model_profile, next_stage="word_conclusion_draft"
        )
    return True


def _run_agent_word_conclusion_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    draft_message_id: str | None = None,
    *,
    auto_accept: bool = False,
    rewrite_instruction: str | None = None,
) -> bool:
    task = repo.get_task(task_id)
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
        user_message=rewrite_instruction or "",
    )
    draft_result: dict[str, object] = {}

    def produce_draft(_on_delta):
        _, report_revision = repo.get_report_values(task_id)
        values, metadata = generate_word_conclusions(
            task=task,
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            user_instruction=rewrite_instruction,
        )
        draft_result["values"] = values
        draft_result["report_revision"] = report_revision
        return (
            _format_conclusion_values(values),
            {**metadata, "draft_values": values, "report_revision": report_revision},
        )

    if draft_message_id:
        message = _stream_agent_message(
            repo,
            draft_message_id,
            task_id=task_id,
            model_profile=model_profile,
            producer=produce_draft,
        )
    else:
        message = _add_and_stream_agent_message(
            repo,
            task_id,
            stage="word_conclusion_draft",
            model_profile=model_profile,
            producer=produce_draft,
        )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    if auto_accept:
        return _auto_confirm_agent_report_conclusions(
            repo=repo,
            settings=settings,
            task_id=task_id,
            model_profile=model_profile,
            values=draft_result.get("values"),
            expected_revision=draft_result.get("report_revision"),
        )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
        metadata={**_model_metadata(model_profile), "awaiting_confirmation": True},
    )
    return True


def _auto_confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    values: object,
    expected_revision: object,
) -> bool:
    if (
        not isinstance(values, dict)
        or not agent_conclusions_confirmed(values)
        or not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
    ):
        raise RuntimeError("agent report draft is incomplete; cannot auto-confirm report")
    revision = repo.update_agent_report_conclusions(
        task_id,
        {
            key: str(values.get(key) or "").strip()
            for key in REQUIRED_AGENT_REPORT_KEYS
        },
        expected_revision=expected_revision,
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已自动确认，正在生成最终 Word 报告。",
        metadata={
            **_model_metadata(model_profile),
            "revision": revision,
            "confirmed_keys": sorted(REQUIRED_AGENT_REPORT_KEYS),
            "auto_accept": True,
        },
    )
    _raise_if_agent_cancelled(task_id)
    run_report_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, repo.get_task(task_id)),
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="报告生成",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    _add_agent_report_ready_message(repo, task_id)
    return True


def _add_agent_failure_summary(
    repo: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    stage_label: str,
    error: str,
    model_profile: dict,
) -> None:
    _add_and_stream_agent_message(
        repo,
        task_id,
        stage="failure",
        model_profile=model_profile,
        producer=lambda on_delta: failure_summary(
            task=task,
            stage=stage_label,
            error=error,
            model_profile=model_profile,
            on_delta=on_delta,
        ),
    )


def _add_agent_continue_prompt(
    repo: TaskRepository,
    task_id: str,
    model_profile: dict,
    *,
    next_stage: str,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            f"是否继续执行【{_agent_stage_label(next_stage)}】？"
            "你可以先继续提问；需要继续时，请明确回复“继续”。"
        ),
        metadata={**_model_metadata(model_profile), "awaiting_next_stage": next_stage},
    )


def _agent_stage_label(stage: str) -> str:
    if stage == "scan":
        return "模型材料完备性验证"
    if stage == "reproducibility":
        return "模型可复现性验证"
    if stage == "metrics":
        return "模型效果&稳定性验证"
    if stage == "word_conclusion_draft":
        return "报告结论草稿生成"
    return "下一步验证"


def _normalize_agent_report_command(content: str) -> str:
    return "".join(str(content or "").lower().split()).strip("。.!！?？")


def _is_agent_report_confirm_intent(content: str) -> bool:
    command = _normalize_agent_report_command(content)
    return command in {
        "确认",
        "确认写入",
        "写入报告",
        "确认生成报告",
        "生成报告",
        "生成word",
        "生成word报告",
        "可以写入",
    }


def _is_agent_report_regenerate_intent(content: str) -> bool:
    command = _normalize_agent_report_command(content)
    return command in {
        "重新生成",
        "重新生成报告",
        "重新生成word",
        "重新生成word报告",
        "重新生成草稿",
        "重新生成结论",
        "重新生成三段总结",
        "重写报告",
        "重新起草",
        "再生成",
        "再生成报告",
        "再写一版",
    }


def _latest_pending_agent_report_draft(messages: list[dict]) -> dict:
    for message in reversed(messages):
        if message.get("stage") == "word_conclusion_confirmed":
            return {}
        if message.get("role") != "assistant":
            continue
        if message.get("stage") != "word_conclusion_draft":
            continue
        metadata = message.get("metadata") or {}
        draft_values = metadata.get("draft_values")
        report_revision = metadata.get("report_revision")
        if (
            isinstance(draft_values, dict)
            and agent_conclusions_confirmed(draft_values)
            and isinstance(report_revision, int)
            and not isinstance(report_revision, bool)
        ):
            return {
                "message_id": message.get("id"),
                "report_revision": report_revision,
                "values": {
                    key: str(draft_values.get(key) or "").strip()
                    for key in REQUIRED_AGENT_REPORT_KEYS
                },
            }
    return {}



def _resolve_driver_agent_client(request, task: TaskRecord, payload):
    """Agent mode hands the manual gate-controls to an LLM, so a configured LLM is
    mandatory: returns the client, or raises HTTP 409 when none is configured (never
    silently runs the manual flow). Manual mode operates the gates by hand → None."""
    if task.run_mode != "agent":
        return None
    profile = _resolve_agent_model(
        request, getattr(payload, "model_id", None), getattr(payload, "effort", None)
    )
    return OpenAICompatibleLLMClient(profile)


def _driver_llm_client(request, task: TaskRecord):
    """The PlanDriver's LLM for agent-mode free-text gate instructions (adjust /
    replan). None in manual mode or when no LLM is configured — the driver then
    degrades to the canned gate hint (manual gates use control buttons, not text)."""
    if task.run_mode != "agent":
        return None
    try:
        return OpenAICompatibleLLMClient(resolve_llm_model(request.app.state.settings.workspace, None))
    except LLMSettingsError:
        return None



def _dispatch_driver_turn(
    request, repo: TaskRepository, task: TaskRecord, *, user_text: str | None,
    agent_client, acceptance_mode: str | None = None, selection: list | None = None,
    dedup_strategies: dict | None = None, adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    """Run one driver turn. ``acceptance_mode`` controls the agent-mode behavior at
    gates (spec §6, two 受控度): AUTO(自动审查) lets the LLM auto-drive ALL gates;
    NORMAL(默认权限) runs a single turn and STOPS at the first gate for the user to
    confirm — even with an LLM configured. Manual mode (agent_client None) always
    stops at the gate for the control button. ``selection`` carries an edited feature
    set from the §4 screening table; ``dedup_strategies`` carries the per-feature dedup
    map from the §4 join dedup picker."""
    runtime = DriverTurnRuntime(
        settings=request.app.state.settings,
        plan_repo=request.app.state.plan_repo,
        plan_executor=request.app.state.plan_executor,
        planner=request.app.state.planner,
        plan_validator=request.app.state.plan_validator,
        llm_client=_driver_llm_client(request, task),
        tier=_task_tier(request, task),
    )
    try:
        return dispatch_plan_driver_turn(
            runtime, repo, task, user_text=user_text, agent_client=agent_client,
            auto_accept_enabled=_agent_auto_accept(acceptance_mode), selection=selection,
            dedup_strategies=dedup_strategies, adjust_params=adjust_params,
            expected_step_id=expected_step_id,
        )
    except DriverError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _dispatch_agent_validation_job(
    *,
    repo: TaskRepository,
    task: TaskRecord,
    settings,
    model_profile: dict,
    acceptance_mode: str | None = None,
    background_tasks: BackgroundTasks,
    forced_stage: str | None = None,
    stage_instruction: str | None = None,
) -> dict:
    _clear_agent_cancellation(task.id)
    normalized_acceptance_mode = _normalize_agent_acceptance_mode(acceptance_mode)
    auto_accept = _agent_auto_accept(normalized_acceptance_mode)
    stage = forced_stage or _agent_next_stage(repo, task)
    job_id = _start_task_job(repo, task.id, "agent")
    should_create_opening_message = not (auto_accept and stage and stage != "scan")
    opening_message = (
        _add_streaming_agent_message(
            repo,
            task.id,
            stage="chat",
            model_profile=model_profile,
        )
        if should_create_opening_message
        else None
    )
    if opening_message and stage and stage != "scan":
        _finalize_agent_opening_message(
            repo,
            task_id=task.id,
            message_id=opening_message["id"],
            model_profile=model_profile,
            content=_agent_stage_opening_text(stage),
        )
    stage_message = None
    if stage == "word_conclusion_draft":
        stage_message = _add_streaming_agent_message(
            repo,
            task.id,
            stage="word_conclusion_draft",
            model_profile=model_profile,
        )
    background_tasks.add_task(
        _run_agent_validation_job,
        job_id,
        settings,
        task.id,
        model_profile,
        opening_message["id"] if opening_message else None,
        stage,
        stage_message["id"] if stage_message else None,
        normalized_acceptance_mode,
        stage_instruction,
    )
    return {
        "task_id": task.id,
        "status": "accepted",
        "stage": stage,
        "acceptance_mode": normalized_acceptance_mode,
        "message": "agent validation dispatched; poll task and agent messages",
        "messages": repo.list_agent_messages(task.id),
    }



def _agent_evidence(request: Request, task_id: str) -> dict:
    return _agent_evidence_from_settings(request.app.state.settings, task_id)


def _agent_chat_evidence(
    request: Request,
    repo: TaskRepository,
    task: TaskRecord,
    conversation: list[dict],
) -> dict:
    evidence = _agent_evidence(request, task.id)
    values, revision = repo.get_report_values(task.id)
    report_payload = _build_report_field_payload(request, task, values, revision)
    evidence["report_fields"] = {
        "revision": report_payload["revision"],
        "text_values": report_payload["text_values"],
        "metric_values": report_payload["metric_values"],
    }
    evidence["visible_stage_summaries"] = [
        {
            "stage": message["stage"],
            "content": message["content"],
        }
        for message in conversation[-16:]
        if message.get("role") == "assistant"
        and message.get("stage") != "chat"
        and str(message.get("content") or "").strip()
    ]
    return evidence


def _agent_memory_context(
    request: Request,
    task: TaskRecord,
    *,
    stage: str,
    user_message: str = "",
    evidence: dict | None = None,
) -> dict | None:
    return _agent_memory_context_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        task,
        stage=stage,
        user_message=user_message,
        evidence=evidence,
    )


def _capture_user_preference_memory(
    request: Request,
    task_id: str,
    message: dict,
) -> None:
    capture_user_preference_memory(
        request.app.state.settings,
        task_id,
        message,
        extractor=extract_user_preference,
    )


def _audit_agent_memory_use(request: Request, message: dict, *, task_id: str) -> None:
    _audit_agent_memory_use_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        message,
        task_id=task_id,
    )


def _agent_evidence_from_settings(settings, task_id: str) -> dict:
    task_dir = settings.tasks_dir / task_id
    validation_results = _read_json(task_dir / "outputs" / "validation_results.json")
    return {
        "scan": _read_json(task_dir / "execution" / "scan_result.json"),
        "notebook_steps": _read_json(task_dir / "execution" / "notebook_steps.json"),
        "contract": _read_json(task_dir / "execution" / "runtime_contract.json"),
        "reproducibility": _read_json(task_dir / "outputs" / "reproducibility_result.json"),
        "validation_results": _agent_validation_results_with_overfitting_check(validation_results),
    }


def _agent_validation_results_with_overfitting_check(validation_results):
    if not isinstance(validation_results, dict):
        return validation_results
    return {
        **validation_results,
        "overfitting_check": overfitting_check_from_validation_results(validation_results),
    }


# Driver-based task types run their deterministic plan flow through the agent
# endpoints in BOTH manual (user operates the controls, no LLM) and agent (an LLM
# operates them) mode. Validation, in contrast, only uses these endpoints in agent
# mode — its manual mode is the separate scan/notebook flow.
_DRIVER_AGENT_TASK_TYPES = DRIVER_AGENT_TASK_TYPES


def _require_agent_task(task: TaskRecord) -> None:
    if task.run_mode == "agent":
        return
    if task.run_mode == "manual" and task.task_type in _DRIVER_AGENT_TASK_TYPES:
        return
    raise HTTPException(status_code=409, detail="task is not in Agent mode")


# Agent task types that have a wired conversational backend today.
# validation -> validation agent (default path); driver task types share the
# PlanDriver backend.
# This is an ALLOWLIST, not a denylist: any task_type not listed here rejects
# explicitly instead of silently falling through to the validation agent on a
# goal prompt written for a different workflow. New task types must be added here
# as their agent flow is wired (see docs/plans/v2-completion-plan.md SS8 step 0).
_WIRED_AGENT_TASK_TYPES = frozenset(
    {
        TASK_TYPE_VALIDATION,
        TASK_TYPE_MODELING,
        TASK_TYPE_DATA_JOIN,
        TASK_TYPE_FEATURE_ANALYSIS,
        TASK_TYPE_STRATEGY,
        TASK_TYPE_VINTAGE,
    }
)


def _require_wired_agent_task_type(task: TaskRecord) -> None:
    if task.task_type not in _WIRED_AGENT_TASK_TYPES:
        raise HTTPException(
            status_code=501,
            detail=(
                f"任务类型 '{task.task_type}' 的 Agent 流程尚未接入"
                "（当前仅支持 模型验证 / 模型开发 / 数据拼接 / 特征分析 / 策略分析 / Vintage风险分析）"
            ),
        )


_VALID_EFFORTS = ("low", "medium", "high")


def _normalize_effort(effort: str | None) -> str:
    value = str(effort or "").strip().lower()
    return value if value in _VALID_EFFORTS else "high"


def _normalize_agent_acceptance_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower().replace("-", "_")
    return value if value in AGENT_ACCEPTANCE_MODES else AGENT_ACCEPTANCE_NORMAL


def _agent_auto_accept(mode: str | None) -> bool:
    return _normalize_agent_acceptance_mode(mode) == AGENT_ACCEPTANCE_AUTO


def _resolve_agent_model(
    request: Request, model_id: str | None, effort: str | None = None
) -> dict:
    try:
        profile = resolve_llm_model(request.app.state.settings.workspace, model_id)
    except LLMSettingsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # An explicit per-request effort wins; otherwise fall back to the value
    # persisted in the model profile (so the UI-configured effort is honored).
    if effort is not None:
        profile["reasoning_effort"] = _normalize_effort(effort)
    else:
        profile["reasoning_effort"] = _normalize_effort(profile.get("reasoning_effort"))
    return profile


def _model_metadata(model_profile: dict) -> dict:
    return {
        "model_id": model_profile.get("model_id"),
        "display_name": model_profile.get("display_name"),
        "model_name": model_profile.get("model_name"),
    }


def _add_streaming_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
) -> dict:
    return repo.add_agent_message(
        task_id,
        role="assistant",
        stage=stage,
        content="",
        metadata={**_model_metadata(model_profile), "streaming": True},
    )


def _add_and_stream_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    message = _add_streaming_agent_message(
        repo,
        task_id,
        stage=stage,
        model_profile=model_profile,
    )
    return _stream_agent_message(
        repo,
        message["id"],
        task_id=task_id,
        model_profile=model_profile,
        producer=producer,
    )


def _stream_agent_message(
    repo: TaskRepository,
    message_id: str,
    *,
    task_id: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    parts: list[str] = []
    streaming_metadata = {**_model_metadata(model_profile), "streaming": True}

    def on_delta(delta: str) -> None:
        if not delta:
            return
        _raise_if_agent_cancelled(task_id)
        parts.append(delta)
        repo.update_agent_message(
            message_id,
            content="".join(parts),
            metadata=streaming_metadata,
        )

    try:
        _raise_if_agent_cancelled(task_id)
        content, metadata = producer(on_delta)
        _raise_if_agent_cancelled(task_id)
        final_metadata = {
            **metadata,
            **_model_metadata(model_profile),
            "streaming": False,
        }
        if parts:
            final_metadata["streamed"] = True
        _raise_if_agent_cancelled(task_id)
        return repo.update_agent_message(
            message_id,
            content=content,
            metadata=final_metadata,
        )
    except AgentValidationCancelled:
        cancelled_metadata = {
            **_model_metadata(model_profile),
            "streaming": False,
            "cancelled": True,
        }
        if parts:
            cancelled_metadata["streamed"] = True
        repo.update_agent_message(
            message_id,
            content="".join(parts),
            metadata=cancelled_metadata,
        )
        raise


def _format_conclusion_values(values: dict[str, str]) -> str:
    labels = {
        "TEXT:pressure_test_summary": "压力测试总结",
        "TEXT:pressure_impact_recommendation": "压力影响建议",
        "TEXT:final_validation_conclusion": "最终验证结论",
    }
    ordered_keys = [
        "TEXT:pressure_test_summary",
        "TEXT:pressure_impact_recommendation",
        "TEXT:final_validation_conclusion",
    ]
    ordered_keys.extend(key for key in values if key not in labels)
    return "\n\n".join(
        f"{labels.get(key, key)}\n{value}"
        for key in ordered_keys
        if (value := values.get(key))
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
