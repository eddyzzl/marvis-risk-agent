"""Composition root for the legacy /api routes.

Most validation-agent request handling has moved to
``marvis.routers.validation_agent`` (HTTP layer) and
``marvis.agent.validation_app_service`` (service layer) — see ARCH-1. The
private names below are compatibility re-exports that existing tests still
import as ``marvis.api._xxx``. Each one is registered (owner, users, reason,
removal condition) in ``docs/compat-symbols.md``; new code must import from
the leaf modules directly. Aliases without any remaining user were removed
(F20, 2026-11-02).
"""

from fastapi import APIRouter

from marvis.agent.orchestrator import (
    AgentValidationCancelled,  # noqa: F401 - compat export, see docs/compat-symbols.md.
    request_agent_cancellation,
)
from marvis.agent import validation_service as _validation_service
from marvis.agent import validation_app_service as _vas
from marvis.agent.validation_evidence import (
    agent_evidence_from_settings as _agent_evidence_from_settings,  # noqa: F401
)
from marvis.agent.validation_messages import (
    format_conclusion_values as _format_conclusion_values,  # noqa: F401
)
from marvis.agent.validation_runner import (
    run_agent_validation_job as _run_agent_validation_job_impl,  # noqa: F401
)
from marvis.agent.validation_stages import (
    run_agent_scan_stage as _run_agent_scan_stage_impl,  # noqa: F401
    run_agent_metrics_stage as _run_agent_metrics_stage_impl,  # noqa: F401
)
from marvis.api_stage_helpers import (
    run_stage_job as _run_stage_job,  # noqa: F401
)
from marvis.api_settings import router as settings_router
from marvis.api_task_payloads import (
    task_payload as _task_payload,  # noqa: F401
)


router = APIRouter(prefix="/api")
router.include_router(settings_router)

_agent_has_stop_ack_message = _validation_service.agent_has_stop_ack_message
_clear_agent_cancellation = _validation_service.clear_agent_and_notebook_cancellation


def _request_agent_cancellation(task_id: str) -> None:
    request_agent_cancellation(task_id)


_run_agent_validation_job = _vas.run_agent_validation_job
_run_agent_word_conclusion_stage = _vas.run_agent_word_conclusion_stage


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


_add_streaming_agent_message = _vas.add_streaming_agent_message
_stream_agent_message = _vas.stream_agent_message
