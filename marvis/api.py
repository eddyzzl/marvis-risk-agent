"""Composition root for the legacy /api routes.

Most validation-agent request handling has moved to
``marvis.routers.validation_agent`` (HTTP layer) and
``marvis.agent.validation_app_service`` (service layer) — see ARCH-1. The
names below are kept as compatibility re-exports for existing tests and
extension points that still import ``marvis.api._xxx``; new code should
import directly from ``marvis.agent.validation_app_service`` or the other
leaf modules referenced here.
"""

import logging

from fastapi import APIRouter

from marvis.agent.orchestrator import (
    AgentValidationCancelled,  # noqa: F401 - compatibility export for tests/imports.
    request_agent_cancellation,
)
from marvis.agent.service import (
    agent_rerun_stage,  # noqa: F401 - compatibility export for validation-agent routes/tests.
    answer_chat_message,  # noqa: F401 - compatibility export for validation-agent routes/tests.
    is_agent_advance_intent,  # noqa: F401 - compatibility export for validation-agent routes.
    is_stop_validation_intent,  # noqa: F401 - compatibility export for validation-agent routes.
    summarize_stage,  # noqa: F401 - compatibility export for validation-agent routes/tests.
    generate_word_conclusions,  # noqa: F401 - compatibility export for validation-agent routes/tests.
)
from marvis.agent import validation_service as _validation_service
from marvis.agent import validation_app_service as _vas
from marvis.agent.turn_handlers import DRIVER_AGENT_TASK_TYPES
from marvis.agent.validation_evidence import (
    agent_evidence_from_settings as _agent_evidence_from_settings,  # noqa: F401
)
from marvis.agent.validation_messages import (
    format_conclusion_values as _format_conclusion_values,  # noqa: F401 - compatibility export for tests/imports.
)
from marvis.agent.validation_runner import (
    run_agent_validation_job as _run_agent_validation_job_impl,  # noqa: F401
)
from marvis.agent.validation_stages import (
    run_agent_scan_stage as _run_agent_scan_stage_impl,  # noqa: F401
    run_agent_metrics_stage as _run_agent_metrics_stage_impl,  # noqa: F401
)
from marvis.api_task_helpers import (
    reject_if_task_has_active_job as _reject_if_task_has_active_job,  # noqa: F401
)
from marvis.api_stage_helpers import (
    run_stage_job as _run_stage_job,  # noqa: F401
)
from marvis.api_settings import router as settings_router
from marvis.api_task_payloads import (
    task_payload as _task_payload,  # noqa: F401 - compatibility alias for structure tests/imports.
)
from marvis.notebook_cancellation import request_notebook_cancellation


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


def _handle_agent_stop_message(repo, task):
    return _validation_service.handle_agent_stop_message_with_callbacks(
        repo,
        task,
        request_agent_cancellation_fn=request_agent_cancellation,
        request_notebook_cancellation_fn=request_notebook_cancellation,
    )


AGENT_ACCEPTANCE_NORMAL = _vas.AGENT_ACCEPTANCE_NORMAL
AGENT_ACCEPTANCE_AUTO = _vas.AGENT_ACCEPTANCE_AUTO
AGENT_ACCEPTANCE_MODES = _vas.AGENT_ACCEPTANCE_MODES

# ---------------------------------------------------------------------------
# ARCH-1: the validation-agent HTTP composition logic below moved to
# marvis.agent.validation_app_service. routers/validation_agent.py imports
# the public names from there directly instead of reaching into this
# module's private symbol table via a `legacy_api` service locator. The
# aliases below remain only for backward compatibility with existing
# tests/extension imports.
# ---------------------------------------------------------------------------
_is_metrics_failure = _vas.is_metrics_failure
_validation_stage_dependencies = _vas.validation_stage_dependencies
_repo = _vas.repo
_task_tier = _vas.task_tier
_confirm_agent_report_conclusions = _vas.confirm_agent_report_conclusions
_add_agent_job_exception_summary = _vas.add_agent_job_exception_summary
_run_agent_validation_job = _vas.run_agent_validation_job
_agent_next_stage = _vas.agent_next_stage_for_job
_open_agent_stage = _vas.open_agent_stage
_add_agent_auto_stage_start_message = _vas.add_agent_auto_stage_start_message
_finalize_agent_opening_message = _vas.finalize_agent_opening_message
_run_agent_scan_stage = _vas.run_agent_scan_stage
_run_agent_reproducibility_stage = _vas.run_agent_reproducibility_stage
_run_agent_metrics_stage = _vas.run_agent_metrics_stage
_run_agent_word_conclusion_stage = _vas.run_agent_word_conclusion_stage
_auto_confirm_agent_report_conclusions = _vas.auto_confirm_agent_report_conclusions
_add_agent_failure_summary = _vas.add_agent_failure_summary
_add_agent_continue_prompt = _vas.add_agent_continue_prompt
_is_agent_report_confirm_intent = _vas.is_agent_report_confirm_intent
_is_agent_report_regenerate_intent = _vas.is_agent_report_regenerate_intent
_latest_pending_agent_report_draft = _vas.latest_pending_agent_report_draft
_resolve_driver_agent_client = _vas.resolve_driver_agent_client
_driver_llm_client = _vas.driver_llm_client
_dispatch_driver_turn = _vas.dispatch_driver_turn


def _dispatch_agent_validation_job(
    *,
    repo,
    task,
    settings,
    model_profile: dict,
    acceptance_mode=None,
    background_tasks,
    forced_stage=None,
    stage_instruction=None,
):
    # Compatibility shim: the pre-ARCH-1 signature used `repo=`; the moved
    # implementation uses `repo_=` (validation_app_service.py already has a
    # module-level `repo` function, so the parameter was renamed there).
    return _vas.dispatch_agent_validation_job(
        repo_=repo,
        task=task,
        settings=settings,
        model_profile=model_profile,
        acceptance_mode=acceptance_mode,
        background_tasks=background_tasks,
        forced_stage=forced_stage,
        stage_instruction=stage_instruction,
    )


_agent_evidence = _vas.agent_evidence
_agent_chat_evidence = _vas.agent_chat_evidence
_agent_memory_context = _vas.agent_memory_context
_capture_user_preference_memory = _vas.capture_user_preference_memory
_audit_agent_memory_use = _vas.audit_agent_memory_use

# Driver-based task types run their deterministic plan flow through the agent
# endpoints in BOTH manual (user operates the controls, no LLM) and agent (an LLM
# operates them) mode. Validation, in contrast, only uses these endpoints in agent
# mode — its manual mode is the separate scan/notebook flow.
_DRIVER_AGENT_TASK_TYPES = DRIVER_AGENT_TASK_TYPES


def _require_agent_task(task) -> None:
    _vas.require_agent_task(task, _DRIVER_AGENT_TASK_TYPES)


_WIRED_AGENT_TASK_TYPES = _vas.WIRED_AGENT_TASK_TYPES


def _require_wired_agent_task_type(task) -> None:
    _vas.require_wired_agent_task_type(task, _WIRED_AGENT_TASK_TYPES)


_VALID_EFFORTS = _vas._VALID_EFFORTS
_normalize_effort = _vas.normalize_effort
_normalize_agent_acceptance_mode = _vas.normalize_agent_acceptance_mode
_agent_auto_accept = _vas.agent_auto_accept
_resolve_agent_model = _vas.resolve_agent_model
_model_metadata = _vas.model_metadata
_add_streaming_agent_message = _vas.add_streaming_agent_message
_add_and_stream_agent_message = _vas.add_and_stream_agent_message
_stream_agent_message = _vas.stream_agent_message
_agent_stage_opening_text = _vas.agent_stage_opening_text
