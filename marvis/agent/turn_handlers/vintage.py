"""Vintage for governed Agent turns."""

from __future__ import annotations

import re

from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.agent.risk_analysis_setup import STANDARD_VINTAGE
from marvis.agent.risk_analysis_setup import _parse_label_semantics
from marvis.agent.risk_analysis_setup import advance_risk_analysis_setup
from marvis.agent.risk_analysis_setup import latest_risk_analysis_intake
from marvis.agent.semantic_intent import INTENT_RISK_PROFITABILITY
from marvis.agent.semantic_intent import INTENT_RISK_STANDARD_VINTAGE
from marvis.agent.semantic_intent import INTENT_RISK_VTG_TERMINAL
from marvis.agent.vintage_setup import VintageSetupError
from marvis.domain import TASK_TYPE_VINTAGE
from marvis.domain import TaskRecord
from marvis.repositories.tasks import TaskRepository
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import responses as responses_lane
from . import shared as shared_lane
from . import turn_runner as turn_runner_lane
from . import typed_ui as typed_ui_lane


def run_vintage_driver_turn(
    runtime: contracts_lane.DriverTurnRuntime,
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
    return turn_runner_lane._run_driver_turn(
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


_LABEL_ANSWER_MAX_CHARS = 40
_QUESTION_MARKERS = re.compile(r"[?？]|吗|呢|什么|怎么|为什么|是否|还是")


def is_pending_label_answer(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    *,
    text: str,
    conversation: list[dict],
) -> bool:
    """Whether ``text`` answers the intake's own bad-label question.

    Eval round 1: the 「当期新增」 chip the intake offers was sent to the
    semantic router, which refused it at medium confidence. A short reply
    that names exactly one label reading (and is not a question) only fills
    that slot; the plan it leads to still waits at its authorization gate.
    """

    if task.task_type != TASK_TYPE_VINTAGE:
        return False
    if (
        shared_lane._active_plan(runtime.plan_repo, task.id) is not None
        or shared_lane.latest_open_gate(list(conversation)) is not None
    ):
        return False
    state = latest_risk_analysis_intake(conversation) or {}
    if (
        state.get("phase") != "request_materials"
        or state.get("awaiting") != "label_semantics"
        or state.get("analysis_kind") != STANDARD_VINTAGE
    ):
        return False
    raw = str(text or "").strip()
    if not raw or len(raw) > _LABEL_ANSWER_MAX_CHARS or _QUESTION_MARKERS.search(raw):
        return False
    return _parse_label_semantics(raw) is not None


def _run_vintage_setup(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict | tuple:
    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
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
        content=f"{decision.content}{shared_lane._ingest_notice_text(notices)}",
        metadata=metadata,
    )
    if decision.template_id is None:
        return responses_lane.join_turn_response(repo, task.id)
    return (decision.template_id, dict(decision.slots or {}), {})


_VINTAGE_SPEC = contracts_lane._TurnHandlerSpec(
    intent="vintage",
    setup_error_types=(VintageSetupError,),
    error_label="Vintage 风险分析出错",
    run_setup=_run_vintage_setup,
    format_user_display=typed_ui_lane._identity_display_text,
)
