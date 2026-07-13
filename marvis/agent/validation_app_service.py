"""Validation-agent HTTP composition seam.

This module is the explicit service layer the ``routers/validation_agent.py``
adapter depends on: request-shaped helpers (task lookup/guards, driver-turn
dispatch, validation-job dispatch, report-conclusion confirmation, chat
evidence/memory wiring) that used to live as private ``_``-prefixed functions
inside ``marvis.api``. Moved here verbatim (same signatures, same behavior) so
the router imports a named module instead of reaching into another module's
private symbol table. ``marvis.api`` re-exports these names for backward
compatibility with existing extension points; new code should import from
here directly.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import BackgroundTasks, Request

from marvis.errors import conflict, not_implemented, unprocessable

from marvis.agent.orchestrator import (
    agent_next_stage,
    is_metrics_failure,
    register_agent_cancellation,
    request_agent_cancellation,
)
from marvis.agent.service import (
    agent_conclusions_confirmed,
    compose_agent_start_message,
    failure_summary,
    generate_word_conclusions,
    summarize_stage,
    REQUIRED_AGENT_REPORT_KEYS,
)
from marvis.agent.turn_handlers import (
    DriverTurnRuntime,
    dispatch_driver_turn as dispatch_plan_driver_turn,
)
from marvis.agent.validation_runner import (
    ValidationJobCallbacks,
    run_agent_validation_job as run_agent_validation_job_impl,
)
from marvis.agent.validation_stages import (
    ValidationStageDependencies,
    add_agent_auto_stage_start_message as add_agent_auto_stage_start_message_impl,
    add_agent_continue_prompt as add_agent_continue_prompt_impl,
    add_agent_failure_summary as add_agent_failure_summary_impl,
    add_agent_input_confirmation_prompt as add_agent_input_confirmation_prompt_impl,
    auto_confirm_agent_report_conclusions as auto_confirm_agent_report_conclusions_impl,
    finalize_agent_opening_message as finalize_agent_opening_message_impl,
    open_agent_stage as open_agent_stage_impl,
    run_agent_metrics_stage as run_agent_metrics_stage_impl,
    run_agent_reproducibility_stage as run_agent_reproducibility_stage_impl,
    run_agent_scan_stage as run_agent_scan_stage_impl,
    run_agent_word_conclusion_stage as run_agent_word_conclusion_stage_impl,
)
from marvis.agent.validation_evidence import (
    agent_evidence_from_settings as agent_evidence_from_settings_impl,
)
from marvis.agent.validation_messages import (
    add_and_stream_agent_message as add_and_stream_agent_message_impl,
    add_streaming_agent_message as add_streaming_agent_message_impl,
    agent_stage_opening_text as agent_stage_opening_text_impl,
    model_metadata as model_metadata_impl,
    stream_agent_message as stream_agent_message_impl,
)
from marvis.agent.validation_service import (
    AGENT_STOP_ACK_CONTENT,
    agent_has_stop_ack_message,
    clear_agent_and_notebook_cancellation as clear_agent_cancellation,
    handle_agent_stop_message_with_callbacks,
    mark_agent_cancelled,
    raise_if_agent_cancelled,
)
from marvis.agent.plan_driver import DriverError
from marvis.agent_memory.api_support import (
    agent_memory_context_from_store,
    audit_agent_memory_use_from_store,
    capture_user_preference_memory as capture_user_preference_memory_with_extractor,
)
from marvis.agent_memory.extractors import extract_user_preference
from marvis.agent_memory.store import AgentMemoryStore
from marvis.api_report_field_helpers import build_report_field_payload
from marvis.api_scan_helpers import SCAN_FAILURE_PREFIX, perform_scan_task
from marvis.api_stage_helpers import (
    add_agent_report_ready_message,
    agent_pipeline_settings,
    fail_queued_job,
    run_stage_job,
    start_task_job,
)
from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
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
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.notebook_cancellation import request_notebook_cancellation
from marvis.pipeline import (
    run_metrics_stage,
    run_notebook_stage,
    run_pmml_scoring_stage,
    run_report_stage,
)
from marvis.repositories.validation_contracts import (
    ValidationContractActiveJobConflict,
    ValidationContractRepository,
    require_confirmed_validation_input_contract,
)
from marvis.state_machine import ConflictError


AGENT_ACCEPTANCE_NORMAL = "normal"
AGENT_ACCEPTANCE_AUTO = "auto_accept"
AGENT_ACCEPTANCE_MODES = {AGENT_ACCEPTANCE_NORMAL, AGENT_ACCEPTANCE_AUTO}

_VALID_EFFORTS = ("low", "medium", "high")

_DRIVER_JOB_KIND = "driver"
_DRIVER_JOB_BUSY_DETAIL = "该任务正在执行上一步，请等待完成"


def repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def task_tier(request: Request, task: TaskRecord) -> str:
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


def require_agent_task(task: TaskRecord, driver_agent_task_types: frozenset[str]) -> None:
    if task.run_mode == "agent":
        return
    if task.run_mode == "manual" and task.task_type in driver_agent_task_types:
        return
    raise conflict("task is not in Agent mode")


# Agent task types that have a wired conversational backend today.
# validation -> validation agent (default path); driver task types share the
# PlanDriver backend.
# This is an ALLOWLIST, not a denylist: any task_type not listed here rejects
# explicitly instead of silently falling through to the validation agent on a
# goal prompt written for a different workflow. New task types must be added here
# as their agent flow is wired (see docs/plans/v2-completion-plan.md SS8 step 0).
WIRED_AGENT_TASK_TYPES = frozenset(
    {
        TASK_TYPE_VALIDATION,
        TASK_TYPE_MODELING,
        TASK_TYPE_DATA_JOIN,
        TASK_TYPE_FEATURE_ANALYSIS,
        TASK_TYPE_STRATEGY,
        TASK_TYPE_VINTAGE,
    }
)


def require_wired_agent_task_type(
    task: TaskRecord, wired_agent_task_types: frozenset[str] = WIRED_AGENT_TASK_TYPES
) -> None:
    if task.task_type not in wired_agent_task_types:
        raise not_implemented(
            f"任务类型 '{task.task_type}' 的 Agent 流程尚未接入"
            "（当前仅支持 模型验证 / 模型开发 / 数据拼接 / 特征分析 / 策略分析 / Vintage风险分析）"
        )


def normalize_effort(effort: str | None) -> str:
    value = str(effort or "").strip().lower()
    return value if value in _VALID_EFFORTS else "high"


def normalize_agent_acceptance_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower().replace("-", "_")
    return value if value in AGENT_ACCEPTANCE_MODES else AGENT_ACCEPTANCE_NORMAL


def agent_auto_accept(mode: str | None) -> bool:
    return normalize_agent_acceptance_mode(mode) == AGENT_ACCEPTANCE_AUTO


def resolve_agent_model(
    request: Request,
    model_id: str | None,
    effort: str | None = None,
    *,
    role: str | None = None,
) -> dict:
    try:
        profile = resolve_llm_model(request.app.state.settings.workspace, model_id, role=role)
    except LLMSettingsError as exc:
        raise conflict(str(exc)) from exc
    # An explicit per-request effort wins; otherwise fall back to the value
    # persisted in the model profile (so the UI-configured effort is honored).
    if effort is not None:
        profile["reasoning_effort"] = normalize_effort(effort)
    else:
        profile["reasoning_effort"] = normalize_effort(profile.get("reasoning_effort"))
    return profile


def resolve_driver_agent_client(request: Request, task: TaskRecord, payload):
    """Agent mode hands the manual gate-controls to an LLM, so a configured LLM is
    mandatory: returns the client, or raises HTTP 409 when none is configured (never
    silently runs the manual flow). Manual mode operates the gates by hand → None."""
    if task.run_mode != "agent":
        return None
    # LLM-4: this client drives agent_autodrive_turn -> decide_gate exclusively
    # (see marvis.agent.turn_handlers), so role="gate" — an explicit per-request
    # model_id (from the UI) still wins over any role_overrides mapping.
    profile = resolve_agent_model(
        request,
        getattr(payload, "model_id", None),
        getattr(payload, "effort", None),
        role="gate",
    )
    return OpenAICompatibleLLMClient(profile)


def driver_llm_client(request: Request, task: TaskRecord) -> OpenAICompatibleLLMClient | None:
    """The PlanDriver's LLM for agent-mode free-text gate instructions (adjust /
    replan). None in manual mode or when no LLM is configured — the driver then
    degrades to the canned gate hint (manual gates use control buttons, not text)."""
    if task.run_mode != "agent":
        return None
    try:
        # LLM-4: route_instruction (caller="router") drives this client exclusively.
        return OpenAICompatibleLLMClient(
            resolve_llm_model(request.app.state.settings.workspace, None, role="router")
        )
    except LLMSettingsError:
        return None


def dispatch_driver_turn(
    request: Request,
    repo_: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    agent_client,
    acceptance_mode: str | None = None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
) -> dict:
    """Run one driver turn. ``acceptance_mode`` controls the agent-mode behavior at
    gates (spec §6, two 受控度): AUTO(自动审查) lets the LLM auto-drive ALL gates;
    NORMAL(默认权限) runs a single turn and STOPS at the first gate for the user to
    confirm — even with an LLM configured. Manual mode (agent_client None) always
    stops at the gate for the control button. ``selection`` carries an edited feature
    set from the §4 screening table; ``dedup_strategies`` carries the per-feature dedup
    map from the §4 join dedup picker.

    REL-1: the whole turn (including any AUTO-mode multi-gate auto-drive loop) is
    wrapped in a task job so the ``idx_jobs_active_task`` unique index rejects a
    second concurrent turn (double-sent confirm / dual tabs / a retried request)
    with 409 *before* it ever reaches ``PlanExecutor.run`` a second time, instead of
    racing in and getting misjudged as a server-restart orphan. The job stays
    synchronous (it mirrors the synchronous HTTP request, not a BackgroundTasks
    job) — it exists purely as the concurrency lock + observability record
    (REL-6), so ``GET /api/tasks`` shows ``active_job_kind == "driver"`` for the
    duration of the turn. A hung driver turn is out of scope here (REL-5 covers
    job heartbeat/watchdog + a cancel endpoint separately)."""
    try:
        job_id = repo_.start_job(task.id, _DRIVER_JOB_KIND)
    except ConflictError as exc:
        raise conflict(_DRIVER_JOB_BUSY_DETAIL) from exc
    if repo_.mark_job_running(job_id) is False:
        raise conflict(_DRIVER_JOB_BUSY_DETAIL)
    runtime = DriverTurnRuntime(
        settings=request.app.state.settings,
        plan_repo=request.app.state.plan_repo,
        plan_executor=request.app.state.plan_executor,
        planner=request.app.state.planner,
        plan_validator=request.app.state.plan_validator,
        llm_client=driver_llm_client(request, task),
        tier=task_tier(request, task),
    )
    try:
        result = dispatch_plan_driver_turn(
            runtime, repo_, task, user_text=user_text, agent_client=agent_client,
            auto_accept_enabled=agent_auto_accept(acceptance_mode), selection=selection,
            dedup_strategies=dedup_strategies, adjust_params=adjust_params,
            expected_step_id=expected_step_id,
        )
    except DriverError as exc:
        repo_.finish_job(job_id, status="failed", error_name="DriverError", error_value=str(exc))
        raise conflict(str(exc)) from exc
    except Exception as exc:
        repo_.finish_job(job_id, status="failed", error_name=exc.__class__.__name__, error_value=str(exc))
        raise
    else:
        repo_.finish_job(job_id, status="succeeded")
        return result


def dispatch_agent_validation_job(
    *,
    repo_: TaskRepository,
    task: TaskRecord,
    settings,
    model_profile: dict,
    acceptance_mode: str | None = None,
    background_tasks: BackgroundTasks,
    forced_stage: str | None = None,
    stage_instruction: str | None = None,
) -> dict:
    normalized_acceptance_mode = normalize_agent_acceptance_mode(acceptance_mode)
    auto_accept = agent_auto_accept(normalized_acceptance_mode)
    stage, awaiting_contract = _agent_validation_stage_decision(
        repo_,
        task,
        requested_stage=forced_stage,
    )
    if awaiting_contract is not None:
        add_agent_input_confirmation_prompt_impl(
            repo_,
            task_id=task.id,
            model_profile=model_profile,
            contract_payload=awaiting_contract,
        )
        return {
            "task_id": task.id,
            "status": "awaiting_confirmation",
            "stage": "input_confirmation",
            "acceptance_mode": normalized_acceptance_mode,
            "validation_input_contract": awaiting_contract,
            "message": "validation input contract requires confirmation",
            "messages": repo_.list_agent_messages(task.id),
        }
    if (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
        and stage is not None
        and stage != "scan"
    ):
        try:
            job_id = ValidationContractRepository(repo_.db_path).start_ready_job(
                task.id,
                "agent",
            )
        except (ValidationContractActiveJobConflict, ConflictError) as exc:
            raise conflict("task already has an active stage") from exc
        except ValueError:
            latest_task = repo_.get_task(task.id)
            _stage, awaiting_contract = _agent_validation_stage_decision(
                repo_,
                latest_task,
                requested_stage=stage,
            )
            if awaiting_contract is None:
                raise
            add_agent_input_confirmation_prompt_impl(
                repo_,
                task_id=task.id,
                model_profile=model_profile,
                contract_payload=awaiting_contract,
            )
            return {
                "task_id": task.id,
                "status": "awaiting_confirmation",
                "stage": "input_confirmation",
                "acceptance_mode": normalized_acceptance_mode,
                "validation_input_contract": awaiting_contract,
                "message": "validation input contract requires confirmation",
                "messages": repo_.list_agent_messages(task.id),
            }
    else:
        job_id = start_task_job(repo_, task.id, "agent")
    latest_task = repo_.get_task(task.id)
    stage, awaiting_contract = _agent_validation_stage_decision(
        repo_,
        latest_task,
        requested_stage=stage,
    )
    if awaiting_contract is not None:
        exc = ValueError("validation input contract requires confirmation")
        fail_queued_job(repo_, job_id, exc)
        add_agent_input_confirmation_prompt_impl(
            repo_,
            task_id=task.id,
            model_profile=model_profile,
            contract_payload=awaiting_contract,
        )
        return {
            "task_id": task.id,
            "status": "awaiting_confirmation",
            "stage": "input_confirmation",
            "acceptance_mode": normalized_acceptance_mode,
            "validation_input_contract": awaiting_contract,
            "message": str(exc),
            "messages": repo_.list_agent_messages(task.id),
        }
    register_agent_cancellation(task.id, job_id)
    try:
        should_create_opening_message = not (auto_accept and stage and stage != "scan")
        opening_message = (
            add_streaming_agent_message(
                repo_,
                task.id,
                stage="chat",
                model_profile=model_profile,
            )
            if should_create_opening_message
            else None
        )
        if opening_message and stage and stage != "scan":
            finalize_agent_opening_message(
                repo_,
                task_id=task.id,
                message_id=opening_message["id"],
                model_profile=model_profile,
                content=agent_stage_opening_text(
                    stage,
                    validation_workflow_version=task.validation_workflow_version,
                ),
            )
        stage_message = None
        if stage == "word_conclusion_draft":
            stage_message = add_streaming_agent_message(
                repo_,
                task.id,
                stage="word_conclusion_draft",
                model_profile=model_profile,
            )
        messages = repo_.list_agent_messages(task.id)
        background_tasks.add_task(
            run_agent_validation_job,
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
    except Exception as exc:
        fail_queued_job(repo_, job_id, exc)
        clear_agent_cancellation(task.id, job_id=job_id)
        raise
    return {
        "task_id": task.id,
        "status": "accepted",
        "stage": stage,
        "acceptance_mode": normalized_acceptance_mode,
        "message": "agent validation dispatched; poll task and agent messages",
        "messages": messages,
    }


def confirm_agent_report_conclusions(
    *,
    repo_: TaskRepository,
    task: TaskRecord,
    task_id: str,
    settings,
    text_values: dict[str, str],
    expected_revision: int | None,
    background_tasks: BackgroundTasks,
    model_profile: dict | None = None,
    model_metadata: Callable[[dict], dict] | None = None,
    hook_dispatcher=None,
) -> dict:
    initial_task = get_task_or_404(repo_, task_id)
    if expected_revision is None:
        _, expected_revision = repo_.get_report_values(task_id)
    if (
        initial_task.task_type == TASK_TYPE_VALIDATION
        and initial_task.validation_workflow_version == 2
    ):
        try:
            job_id = ValidationContractRepository(repo_.db_path).start_ready_job(
                task_id,
                "report",
            )
        except (ValidationContractActiveJobConflict, ConflictError) as exc:
            raise conflict("task already has an active stage") from exc
        except ValueError as exc:
            raise unprocessable(str(exc)) from exc
    else:
        job_id = start_task_job(repo_, task_id, "report")
    latest_task = repo_.get_task(task_id)
    if latest_task.status not in {
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.REVIEW_REQUIRED,
    }:
        exc = ValueError(
            f"cannot generate report in status {latest_task.status.value}"
        )
        fail_queued_job(repo_, job_id, exc)
        raise conflict(str(exc))
    try:
        revision = repo_.update_agent_report_conclusions_with_audit(
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
    except ConflictError as exc:
        fail_queued_job(repo_, job_id, exc)
        raise conflict(str(exc)) from exc
    except ValueError as exc:
        fail_queued_job(repo_, job_id, exc)
        raise unprocessable(str(exc)) from exc
    metadata = {
        "revision": revision,
        "confirmed_keys": sorted(text_values),
    }
    if model_profile and model_metadata is not None:
        metadata.update(model_metadata(model_profile))
    repo_.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已确认，将开始生成最终 Word 报告。",
        metadata=metadata,
    )
    background_tasks.add_task(
        run_stage_job,
        job_id,
        settings.db_path,
        run_report_stage,
        {
            "task_id": task_id,
            "settings": agent_pipeline_settings(settings, latest_task),
            "cancellation_job_id": job_id,
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
        "messages": repo_.list_agent_messages(task_id),
    }


def handle_agent_stop_message(repo_: TaskRepository, task: TaskRecord) -> dict:
    return handle_agent_stop_message_with_callbacks(
        repo_,
        task,
        request_agent_cancellation_fn=request_agent_cancellation,
        request_notebook_cancellation_fn=request_notebook_cancellation,
    )


def _normalize_agent_report_command(content: str) -> str:
    return "".join(str(content or "").lower().split()).strip("。.!！?？")


def is_agent_report_confirm_intent(content: str) -> bool:
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


def is_agent_report_regenerate_intent(content: str) -> bool:
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


def latest_pending_agent_report_draft(messages: list[dict]) -> dict:
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


def agent_evidence(request: Request, task_id: str) -> dict:
    return agent_evidence_from_settings_impl(request.app.state.settings, task_id)


def agent_chat_evidence(
    request: Request,
    repo_: TaskRepository,
    task: TaskRecord,
    conversation: list[dict],
) -> dict:
    evidence = agent_evidence(request, task.id)
    values, revision = repo_.get_report_values(task.id)
    report_payload = build_report_field_payload(request, task, values, revision)
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


def agent_memory_context(
    request: Request,
    task: TaskRecord,
    *,
    stage: str,
    user_message: str = "",
    evidence: dict | None = None,
) -> dict | None:
    return agent_memory_context_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        task,
        stage=stage,
        user_message=user_message,
        evidence=evidence,
    )


def capture_user_preference_memory(
    request: Request,
    task_id: str,
    message: dict,
) -> None:
    capture_user_preference_memory_with_extractor(
        request.app.state.settings,
        task_id,
        message,
        extractor=extract_user_preference,
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
    )


def audit_agent_memory_use(request: Request, message: dict, *, task_id: str) -> None:
    audit_agent_memory_use_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        message,
        task_id=task_id,
    )


# ---------------------------------------------------------------------------
# Validation-job background-task composition (moved from marvis.api, ARCH-1).
# This is the callback cluster `dispatch_agent_validation_job` schedules via
# BackgroundTasks: it wires the per-stage runner functions into a single
# ValidationJobCallbacks bundle passed to validation_runner.run_agent_validation_job.
# ---------------------------------------------------------------------------

def validation_stage_dependencies() -> ValidationStageDependencies:
    return ValidationStageDependencies(
        perform_scan_task=perform_scan_task,
        run_notebook_stage=run_notebook_stage,
        run_pmml_scoring_stage=run_pmml_scoring_stage,
        run_metrics_stage=run_metrics_stage,
        run_report_stage=run_report_stage,
        agent_pipeline_settings=agent_pipeline_settings,
        agent_evidence_from_settings=agent_evidence_from_settings_impl,
        add_agent_report_ready_message=add_agent_report_ready_message,
        is_metrics_failure=is_metrics_failure,
        compose_agent_start_message=compose_agent_start_message,
        summarize_stage=summarize_stage,
        generate_word_conclusions=generate_word_conclusions,
        failure_summary=failure_summary,
    )


def add_agent_job_exception_summary(
    repo_: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    model_profile: dict,
    error: Exception,
) -> None:
    error_detail = f"{error.__class__.__name__}: {error}"
    add_and_stream_agent_message(
        repo_,
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


def run_agent_validation_job(
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
    run_agent_validation_job_impl(
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
            agent_auto_accept=agent_auto_accept,
            agent_next_stage=agent_next_stage_for_job,
            raise_if_agent_cancelled=raise_if_agent_cancelled,
            open_agent_stage=open_agent_stage,
            run_scan_stage=run_agent_scan_stage,
            run_reproducibility_stage=run_agent_reproducibility_stage,
            run_metrics_stage=run_agent_metrics_stage,
            run_word_conclusion_stage=run_agent_word_conclusion_stage,
            finalize_agent_opening_message=finalize_agent_opening_message,
            mark_agent_cancelled=mark_agent_cancelled,
            agent_has_stop_ack_message=agent_has_stop_ack_message,
            add_exception_summary=add_agent_job_exception_summary,
            clear_agent_cancellation=clear_agent_cancellation,
            stop_ack_content=AGENT_STOP_ACK_CONTENT,
        ),
    )


def agent_next_stage_for_job(repo_: TaskRepository, task: TaskRecord) -> str | None:
    stage, _awaiting_contract = _agent_validation_stage_decision(repo_, task)
    return stage


def _agent_validation_stage_decision(
    repo_: TaskRepository,
    task: TaskRecord,
    *,
    requested_stage: str | None = None,
) -> tuple[str | None, dict | None]:
    stage = requested_stage or agent_next_stage(
        repo_,
        task,
        scan_failure_prefix=SCAN_FAILURE_PREFIX,
    )
    if (
        task.task_type != TASK_TYPE_VALIDATION
        or task.validation_workflow_version != 2
        or stage is None
        or stage == "scan"
    ):
        return stage, None
    contract_repo = ValidationContractRepository(repo_.db_path)
    try:
        require_confirmed_validation_input_contract(contract_repo, task.id)
    except ValueError:
        record = contract_repo.get(task.id)
        if record is None:
            return None, {
                "task_id": task.id,
                "revision": None,
                "status": "missing",
                "needs_confirmation": True,
                "read_only": True,
                "contract": None,
            }
        return None, record.to_api_payload()
    return stage, None


def open_agent_stage(
    repo_: TaskRepository,
    *,
    task: TaskRecord,
    task_id: str,
    stage: str,
    model_profile: dict,
    opening_message_id: str | None,
    auto_accept: bool = False,
) -> None:
    return open_agent_stage_impl(
        repo_,
        task=task,
        task_id=task_id,
        stage=stage,
        model_profile=model_profile,
        opening_message_id=opening_message_id,
        auto_accept=auto_accept,
        deps=validation_stage_dependencies(),
    )


def add_agent_auto_stage_start_message(
    repo_: TaskRepository,
    *,
    task_id: str,
    stage: str,
    model_profile: dict,
) -> None:
    return add_agent_auto_stage_start_message_impl(
        repo_,
        task_id=task_id,
        stage=stage,
        model_profile=model_profile,
    )


def finalize_agent_opening_message(
    repo_: TaskRepository,
    *,
    task_id: str,
    message_id: str | None,
    model_profile: dict,
    content: str,
) -> None:
    return finalize_agent_opening_message_impl(
        repo_,
        task_id=task_id,
        message_id=message_id,
        model_profile=model_profile,
        content=content,
    )


def run_agent_scan_stage(
    repo_: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return run_agent_scan_stage_impl(
        repo_,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=validation_stage_dependencies(),
    )


def run_agent_reproducibility_stage(
    repo_: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return run_agent_reproducibility_stage_impl(
        repo_,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=validation_stage_dependencies(),
    )


def run_agent_metrics_stage(
    repo_: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    return run_agent_metrics_stage_impl(
        repo_,
        settings,
        task_id,
        model_profile=model_profile,
        auto_accept=auto_accept,
        deps=validation_stage_dependencies(),
    )


def run_agent_word_conclusion_stage(
    repo_: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    draft_message_id: str | None = None,
    *,
    auto_accept: bool = False,
    rewrite_instruction: str | None = None,
) -> bool:
    return run_agent_word_conclusion_stage_impl(
        repo_,
        settings,
        task_id,
        model_profile,
        draft_message_id=draft_message_id,
        auto_accept=auto_accept,
        rewrite_instruction=rewrite_instruction,
        deps=validation_stage_dependencies(),
    )


def auto_confirm_agent_report_conclusions(
    *,
    repo_: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    values: object,
    expected_revision: object,
) -> bool:
    return auto_confirm_agent_report_conclusions_impl(
        repo=repo_,
        settings=settings,
        task_id=task_id,
        model_profile=model_profile,
        values=values,
        expected_revision=expected_revision,
        deps=validation_stage_dependencies(),
    )


def add_agent_failure_summary(
    repo_: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    stage_label: str,
    error: str,
    model_profile: dict,
    evidence: dict | None = None,
) -> None:
    return add_agent_failure_summary_impl(
        repo_,
        task_id=task_id,
        task=task,
        stage_label=stage_label,
        error=error,
        model_profile=model_profile,
        evidence=evidence,
        deps=validation_stage_dependencies(),
    )


def add_agent_continue_prompt(
    repo_: TaskRepository,
    task_id: str,
    model_profile: dict,
    *,
    next_stage: str,
) -> None:
    return add_agent_continue_prompt_impl(
        repo_,
        task_id,
        model_profile,
        next_stage=next_stage,
    )


def model_metadata(model_profile: dict) -> dict:
    return model_metadata_impl(model_profile)


def add_streaming_agent_message(
    repo_: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
) -> dict:
    return add_streaming_agent_message_impl(
        repo_,
        task_id,
        stage=stage,
        model_profile=model_profile,
    )


def add_and_stream_agent_message(
    repo_: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    return add_and_stream_agent_message_impl(
        repo_,
        task_id,
        stage=stage,
        model_profile=model_profile,
        producer=producer,
        raise_if_cancelled=raise_if_agent_cancelled,
    )


def stream_agent_message(
    repo_: TaskRepository,
    message_id: str,
    *,
    task_id: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    return stream_agent_message_impl(
        repo_,
        message_id,
        task_id=task_id,
        model_profile=model_profile,
        producer=producer,
        raise_if_cancelled=raise_if_agent_cancelled,
    )


def agent_stage_opening_text(
    stage: str,
    *,
    validation_workflow_version: int | None = None,
) -> str:
    return agent_stage_opening_text_impl(
        stage,
        validation_workflow_version=validation_workflow_version,
    )
