from collections.abc import Callable
import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
)

from marvis.agent.orchestrator import (
    AgentValidationCancelled,  # noqa: F401 - compatibility export for tests/imports.
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
from marvis.agent.validation_runner import (
    ValidationJobCallbacks,
    run_agent_validation_job as _run_agent_validation_job_impl,
)
from marvis.agent.validation_stages import (
    ValidationStageDependencies,
    add_agent_auto_stage_start_message as _add_agent_auto_stage_start_message_impl,
    add_agent_continue_prompt as _add_agent_continue_prompt_impl,
    add_agent_failure_summary as _add_agent_failure_summary_impl,
    auto_confirm_agent_report_conclusions as _auto_confirm_agent_report_conclusions_impl,
    finalize_agent_opening_message as _finalize_agent_opening_message_impl,
    open_agent_stage as _open_agent_stage_impl,
    run_agent_metrics_stage as _run_agent_metrics_stage_impl,
    run_agent_reproducibility_stage as _run_agent_reproducibility_stage_impl,
    run_agent_scan_stage as _run_agent_scan_stage_impl,
    run_agent_word_conclusion_stage as _run_agent_word_conclusion_stage_impl,
)
from marvis.agent.validation_evidence import (
    agent_evidence_from_settings as _agent_evidence_from_settings,
)
from marvis.agent.validation_messages import (
    add_and_stream_agent_message as _add_and_stream_agent_message_impl,
    add_streaming_agent_message as _add_streaming_agent_message_impl,
    agent_stage_opening_text as _agent_stage_opening_text,
    format_conclusion_values as _format_conclusion_values,  # noqa: F401 - compatibility export for tests/imports.
    model_metadata as _model_metadata_impl,
    stream_agent_message as _stream_agent_message_impl,
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


def _validation_stage_dependencies() -> ValidationStageDependencies:
    return ValidationStageDependencies(
        perform_scan_task=_perform_scan_task,
        run_notebook_stage=run_notebook_stage,
        run_metrics_stage=run_metrics_stage,
        run_report_stage=run_report_stage,
        agent_pipeline_settings=_agent_pipeline_settings,
        agent_evidence_from_settings=_agent_evidence_from_settings,
        add_agent_report_ready_message=_add_agent_report_ready_message,
        is_metrics_failure=_is_metrics_failure,
        compose_agent_start_message=compose_agent_start_message,
        summarize_stage=summarize_stage,
        generate_word_conclusions=generate_word_conclusions,
        failure_summary=failure_summary,
    )


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


def _add_agent_job_exception_summary(
    repo: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    model_profile: dict,
    error: Exception,
) -> None:
    error_detail = f"{error.__class__.__name__}: {error}"
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
    _run_agent_validation_job_impl(
        job_id,
        settings,
        task_id,
        model_profile,
        opening_message_id=opening_message_id,
        stage=stage,
        stage_message_id=stage_message_id,
        acceptance_mode=acceptance_mode,
        stage_instruction=stage_instruction,
        callbacks=ValidationJobCallbacks(
            agent_auto_accept=_agent_auto_accept,
            agent_next_stage=_agent_next_stage,
            raise_if_agent_cancelled=_raise_if_agent_cancelled,
            open_agent_stage=_open_agent_stage,
            run_scan_stage=_run_agent_scan_stage,
            run_reproducibility_stage=_run_agent_reproducibility_stage,
            run_metrics_stage=_run_agent_metrics_stage,
            run_word_conclusion_stage=_run_agent_word_conclusion_stage,
            finalize_agent_opening_message=_finalize_agent_opening_message,
            mark_agent_cancelled=_mark_agent_cancelled,
            agent_has_stop_ack_message=_agent_has_stop_ack_message,
            add_exception_summary=_add_agent_job_exception_summary,
            clear_agent_cancellation=_clear_agent_cancellation,
            stop_ack_content=AGENT_STOP_ACK_CONTENT,
        ),
    )


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
    return _open_agent_stage_impl(
        repo,
        task=task,
        task_id=task_id,
        stage=stage,
        model_profile=model_profile,
        opening_message_id=opening_message_id,
        auto_accept=auto_accept,
        deps=_validation_stage_dependencies(),
    )


def _add_agent_auto_stage_start_message(
    repo: TaskRepository,
    *,
    task_id: str,
    stage: str,
    model_profile: dict,
) -> None:
    return _add_agent_auto_stage_start_message_impl(
        repo,
        task_id=task_id,
        stage=stage,
        model_profile=model_profile,
    )


def _finalize_agent_opening_message(
    repo: TaskRepository,
    *,
    task_id: str,
    message_id: str | None,
    model_profile: dict,
    content: str,
) -> None:
    return _finalize_agent_opening_message_impl(
        repo,
        task_id=task_id,
        message_id=message_id,
        model_profile=model_profile,
        content=content,
    )


def _run_agent_scan_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return _run_agent_scan_stage_impl(
        repo,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=_validation_stage_dependencies(),
    )


def _run_agent_reproducibility_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return _run_agent_reproducibility_stage_impl(
        repo,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=_validation_stage_dependencies(),
    )


def _run_agent_metrics_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return _run_agent_metrics_stage_impl(
        repo,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=_validation_stage_dependencies(),
    )


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
    return _run_agent_word_conclusion_stage_impl(
        repo,
        settings,
        task_id,
        model_profile,
        draft_message_id=draft_message_id,
        auto_accept=auto_accept,
        rewrite_instruction=rewrite_instruction,
        deps=_validation_stage_dependencies(),
    )


def _auto_confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    values: object,
    expected_revision: object,
) -> bool:
    return _auto_confirm_agent_report_conclusions_impl(
        repo=repo,
        settings=settings,
        task_id=task_id,
        model_profile=model_profile,
        values=values,
        expected_revision=expected_revision,
        deps=_validation_stage_dependencies(),
    )


def _add_agent_failure_summary(
    repo: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    stage_label: str,
    error: str,
    model_profile: dict,
) -> None:
    return _add_agent_failure_summary_impl(
        repo,
        task_id=task_id,
        task=task,
        stage_label=stage_label,
        error=error,
        model_profile=model_profile,
        deps=_validation_stage_dependencies(),
    )


def _add_agent_continue_prompt(
    repo: TaskRepository,
    task_id: str,
    model_profile: dict,
    *,
    next_stage: str,
) -> None:
    return _add_agent_continue_prompt_impl(
        repo,
        task_id,
        model_profile,
        next_stage=next_stage,
    )


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
    return _model_metadata_impl(model_profile)


def _add_streaming_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
) -> dict:
    return _add_streaming_agent_message_impl(
        repo,
        task_id,
        stage=stage,
        model_profile=model_profile,
    )


def _add_and_stream_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    return _add_and_stream_agent_message_impl(
        repo,
        task_id,
        stage=stage,
        model_profile=model_profile,
        producer=producer,
        raise_if_cancelled=_raise_if_agent_cancelled,
    )


def _stream_agent_message(
    repo: TaskRepository,
    message_id: str,
    *,
    task_id: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    return _stream_agent_message_impl(
        repo,
        message_id,
        task_id=task_id,
        model_profile=model_profile,
        producer=producer,
        raise_if_cancelled=_raise_if_agent_cancelled,
    )
