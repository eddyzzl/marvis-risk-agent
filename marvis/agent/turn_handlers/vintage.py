"""vintage driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.agent.semantic_intent import INTENT_RISK_PROFITABILITY, INTENT_RISK_STANDARD_VINTAGE, INTENT_RISK_VTG_TERMINAL
from marvis.agent.risk_analysis_setup import advance_risk_analysis_setup
from marvis.agent.vintage_setup import VintageSetupError
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskRecord

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _TurnHandlerSpec
    from . import _identity_display_text
    from . import _ingest_notice_text
    from . import _modeling_data_runtime
    from . import _run_driver_turn
    from . import join_turn_response

def run_vintage_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    expected_plan_id: str | None = None,
    expected_plan_status: str | None = None,
    expected_plan_revision: int | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_step_fingerprint: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    ui_action: str | None = None,
) -> dict:
    return _run_driver_turn(
        _VINTAGE_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        expected_plan_id=expected_plan_id,
        expected_plan_status=expected_plan_status,
        expected_plan_revision=expected_plan_revision,
        expected_plan_fingerprint=expected_plan_fingerprint,
        expected_step_fingerprint=expected_step_fingerprint,
        confirmation_source=confirmation_source,
        ui_action=ui_action,
    )

def _run_vintage_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    semantic_analysis_kind = {
        INTENT_RISK_PROFITABILITY: "profitability",
        INTENT_RISK_VTG_TERMINAL: "vtg_terminal",
        INTENT_RISK_STANDARD_VINTAGE: "standard_vintage",
    }.get(runtime.semantic_intent)
    decision = advance_risk_analysis_setup(
        registry,
        backend,
        task.id,
        task.source_dir,
        user_text=user_text,
        conversation=repo.list_agent_messages(task.id),
        analysis_kind_override=semantic_analysis_kind,
        target_col=getattr(task, "target_col", "") or None,
        time_col=getattr(task, "time_col", "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    metadata = dict(decision.metadata)
    metadata["ingest_notices"] = notices
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=f"{decision.content}{_ingest_notice_text(notices)}",
        metadata=metadata,
    )
    if decision.template_id is None:
        return join_turn_response(repo, task.id)
    return (decision.template_id, dict(decision.slots or {}), {})

_VINTAGE_SPEC = _TurnHandlerSpec(
    intent="vintage",
    setup_error_types=(VintageSetupError,),
    error_label="Vintage 风险分析出错",
    run_setup=_run_vintage_setup,
    format_user_display=_identity_display_text,
)

