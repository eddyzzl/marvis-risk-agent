"""shared driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import hmac
import re
from typing import Callable
from marvis.agent.auto_drive import decide_gate
from marvis.agent.join_setup import C1TargetValidationError
from marvis.agent.memory_bridge import build_memory_anchor, build_workflow_memory_context, capture_agent_memory_for_driver_done
from marvis.agent.workflow_insights import build_workflow_insight_context, render_workflow_insight
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO, CONFIRMATION_SOURCE_HUMAN, DriverError, PlanDriver
from marvis.agent.semantic_intent import INTENT_ADHOC_CONFIRM, INTENT_ADHOC_QUERY, INTENT_ADHOC_REJECT, INTENT_ADHOC_REVISE, INTENT_CURRENT_WORKFLOW, INTENT_DATASET_ANALYSIS, INTENT_DATASET_EXPORT, INTENT_DATASET_JOIN, INTENT_DATASET_TRANSFORM, INTENT_RISK_PROFITABILITY, INTENT_RISK_STANDARD_VINTAGE, INTENT_RISK_VTG_TERMINAL, INTENT_STRATEGY_SAMPLE_BINDING, INTENT_STRATEGY_WORKFLOW, route_semantic_intent
from marvis.agent.strategy_request_compiler import utterance_targets_candidate_monthly_stability, utterance_targets_interactive_tree_frontier_group_materialization, utterance_targets_interactive_tree_frontier_materialization, utterance_targets_model_score_comparison_v2, utterance_targets_scorecard_band_build, utterance_targets_scorecard_cutoff_selection, utterance_targets_strategy_dsl_delivery, utterance_targets_strategy_impact_cube, utterance_targets_strategy_pool_stability, utterance_targets_strategy_project_context, utterance_targets_strategy_report_bundle_v2, utterance_targets_strategy_sample_design
from marvis.agent.workflow_error_diagnostics import build_workflow_error_diagnostic, failure_envelope_for_diagnostic, workflow_error_content
from marvis.agent.workflow_recovery import WorkflowFailureContext, deterministic_workflow_recovery_reply, is_explicit_cancelled_workflow_resume, is_explicit_workflow_retry, is_workflow_repair_request, latest_unresolved_workflow_failure, parse_champion_refit_revision_intent, parse_tuning_budget_revision_intent, parse_workflow_rollback_intent
from marvis.agent_memory.api_support import audit_agent_memory_use_from_store
from marvis.agent_memory.store import AgentMemoryStore
from marvis.data.registry import AuthenticatedDatasetBinding, DatasetRegistry
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_DATA_JOIN, TASK_TYPE_FEATURE_ANALYSIS, TASK_TYPE_MODELING, TASK_TYPE_PORTFOLIO, TASK_TYPE_STRATEGY, TASK_TYPE_VINTAGE, TaskRecord
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.memory_policy import load_memory_policy
from marvis.orchestrator.capability import auto_gate_budget, resolve_tier
from marvis.orchestrator.contracts import PlanStatus, StepStatus, plan_fingerprint
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.validator import PlanValidator
from marvis.repositories.plans import PlanRepository
from marvis.settings import Settings

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DRIVER_TURN_FUNCS
    from . import _STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE
    from . import _STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS
    from . import _StrategySampleDesignRequiredError
    from . import _append_spec_messages
    from . import _append_successful_ui_action_messages
    from . import _auto_decision_content
    from . import _handle_strategy_sample_binding_intent
    from . import _handle_structured_labeling_request_turn
    from . import _handle_structured_portfolio_request_turn
    from . import _handle_structured_strategy_request_turn
    from . import _has_c1_semantic_authorization
    from . import _is_strategy_request_intent
    from . import _latest_adhoc_pending
    from . import _latest_feature_target_state
    from . import _latest_pending_transform_protected_drop
    from . import _latest_strategy_nan_label_confirmation
    from . import _latest_strategy_request_pending
    from . import _maybe_handle_adhoc_turn
    from . import _maybe_handle_dataset_analysis_turn
    from . import _maybe_handle_dataset_export_turn
    from . import _maybe_handle_dataset_transform_turn
    from . import _maybe_handle_labeling_preplan_confirmation_turn
    from . import _maybe_handle_project_context_missing_answer
    from . import _maybe_handle_strategy_request_turn
    from . import _natural_language_c1_target
    from . import _persist_start_turn_atomically
    from . import _semantic_intent_clarification_response
    from . import _semantic_intent_route_contract
    from . import _semantic_intent_state_snapshot
    from . import _specialized_workflow_intake_is_pending
    from . import _specialized_workflow_intake_route
    from . import _strategy_request_clarification_response
    from . import _terminate_stale_strategy_sample_plan
    from . import _validate_typed_ui_action_target
    from . import join_turn_response

DRIVER_AGENT_TASK_TYPES = frozenset(
    {
        TASK_TYPE_DATA_JOIN,
        TASK_TYPE_FEATURE_ANALYSIS,
        TASK_TYPE_MODELING,
        TASK_TYPE_STRATEGY,
        TASK_TYPE_VINTAGE,
        TASK_TYPE_PORTFOLIO,
    }
)

AGENT_MAX_GATES = 8

_TERMINAL_PLAN_STATUS_VALUES = frozenset({"done", "failed", "cancelled"})

@dataclass(frozen=True)
class DriverTurnRuntime:
    settings: Settings
    plan_repo: PlanRepository
    plan_executor: PlanExecutor
    planner: Planner
    plan_validator: PlanValidator
    llm_client: OpenAICompatibleLLMClient | None
    tier: str
    allow_manual_gate_adapters: bool = True
    require_semantic_text_authorization: bool = False
    ui_action: str | None = None
    semantic_intent: str | None = None
    workflow_intake_route: Mapping[str, object] | None = None
    governance_service: object | None = None
    local_principal: object | None = None
    recovery_responder: Callable[..., tuple[str, dict]] | None = None
    hook_dispatcher: object | None = None
    cancellation_check: Callable[[], None] | None = None

@dataclass(frozen=True)
class _TurnHandlerSpec:
    # Metadata `intent` tag stamped on the logged user-turn message.
    intent: str
    # Exception type(s) from this type's *_setup module that map to a plain
    # chat error message (as opposed to DriverError, which always re-raises).
    setup_error_types: tuple[type[Exception], ...]
    # Human label used in the generic `except Exception` fallback message,
    # e.g. "数据拼接出错：{exc}".
    error_label: str
    # Setup callback run only when there is no active plan for the task. It
    # performs this type's proposal-building (and, for join/modeling, the C1
    # file-role gate sub-flow) and returns either:
    #   - a dict: an early-exit turn response (a gate pause, a skip
    #     confirmation, or a setup error) that should be returned as-is; or
    #   - a tuple (template_id, slots, start_kwargs): the driver.start(...)
    #     call to make once the pre-start assistant message has already been
    #     appended by the callback itself.
    run_setup: Callable[
        [DriverTurnRuntime, TaskRepository, TaskRecord, str | None, str],
        dict | tuple,
    ]
    # join/modeling display "已确认文件角色与目标列。" instead of the raw
    # [C1]-prefixed payload text when logging the user turn; the other three
    # types always log user_text verbatim.
    format_user_display: Callable[[str], str]
    # Optional per-type success_criteria builder threaded into start_kwargs
    # (mirrors _modeling_success_criteria); None means this type never injects
    # a deterministic criterion.
    success_criteria: Callable[[TaskRecord], list[dict] | None] | None = None

def _run_driver_turn(
    spec: _TurnHandlerSpec,
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
    confirmation_source: str = "human",
    ui_action: str | None = None,
) -> dict:
    semantic_assignment: dict | None = None
    c1_target_binding: tuple[
        DatasetRegistry,
        AuthenticatedDatasetBinding,
        str | None,
    ] | None = None
    feature_target_col: str | None = None
    feature_semantic_plan_start = False
    if user_text is not None and not ui_action:
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=spec.format_user_display(user_text),
            metadata={"intent": spec.intent},
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
        _validate_typed_ui_action_target(
            active,
            ui_action=ui_action,
            user_text=user_text,
            expected_plan_id=expected_plan_id,
            expected_step_id=expected_step_id,
            expected_plan_status=expected_plan_status,
            expected_plan_revision=expected_plan_revision,
            expected_plan_fingerprint=expected_plan_fingerprint,
            expected_step_fingerprint=expected_step_fingerprint,
        )
        if active is not None:
            stale_response = _terminate_stale_strategy_sample_plan(
                spec,
                runtime,
                repo,
                task,
                active,
            )
            if stale_response is not None:
                return stale_response
            driver = _driver(runtime)
            # A rendered UI control is already a typed command whose plan/step
            # target was validated above.  Feed the driver its canonical token
            # rather than context-specific display copy such as “确认采纳” or
            # “开始模型验证”; free text with those words must remain on the LLM
            # semantic route and cannot confirm an unrelated live gate.
            driver_user_text = "确认" if ui_action is not None else (user_text or "")
            resume_kwargs = {
                "plan_id": active.id,
                "user_text": driver_user_text,
                "selection": selection,
                "dedup_strategies": dedup_strategies,
                "adjust_params": adjust_params,
                "expected_step_id": expected_step_id,
                "confirmation_source": confirmation_source,
            }
            if ui_action is not None:
                resume_kwargs.update(
                    {
                        "expected_plan_status": expected_plan_status,
                        "expected_plan_revision": expected_plan_revision,
                        "expected_plan_fingerprint": expected_plan_fingerprint,
                        "expected_step_fingerprint": expected_step_fingerprint,
                        "_trusted_ui_action": True,
                    }
                )
            turn = driver.resume(
                **resume_kwargs,
            )
            _append_successful_ui_action_messages(
                spec,
                repo,
                task,
                user_text=user_text,
                ui_action=ui_action,
                expected_plan_id=expected_plan_id,
                expected_step_id=expected_step_id,
            )
            _append_spec_messages(repo, task, turn, runtime)
            return join_turn_response(repo, task.id)
        setup_result = spec.run_setup(
            runtime,
            repo,
            task,
            user_text,
            confirmation_source,
        )
        if isinstance(setup_result, dict):
            return setup_result
        template_id, slots, start_kwargs = setup_result
        if spec.success_criteria is not None and "success_criteria" not in start_kwargs:
            criteria = spec.success_criteria(task)
            if criteria is not None:
                start_kwargs = {**start_kwargs, "success_criteria": criteria}
        semantic_assignment = start_kwargs.pop(
            "_post_start_c1_assignment",
            None,
        )
        c1_target_binding = start_kwargs.pop(
            "_post_start_c1_target_binding",
            None,
        )
        feature_target_col = start_kwargs.pop(
            "_post_start_feature_target_col",
            None,
        )
        post_start_messages = start_kwargs.pop(
            "_post_start_messages",
            [],
        )
        feature_semantic_plan_start = (
            spec.intent == TASK_TYPE_FEATURE_ANALYSIS
            and feature_target_col is not None
            and _has_c1_semantic_authorization(semantic_assignment)
        )
        driver = _driver(runtime)
        driver.start(
            task_id=task.id,
            template_id=template_id,
            slots=slots,
            tier=runtime.tier,
            _persist_start_turn=lambda conn, start_turn: (
                _persist_start_turn_atomically(
                    conn,
                    spec,
                    repo,
                    task,
                    start_turn,
                    semantic_assignment=semantic_assignment,
                    c1_target_binding=c1_target_binding,
                    feature_target_col=feature_target_col,
                    post_start_messages=post_start_messages,
                    user_text=user_text,
                    ui_action=ui_action,
                    expected_plan_id=expected_plan_id,
                    expected_step_id=expected_step_id,
                )
            ),
            **start_kwargs,
        )
        # Plan, semantic authorization receipt, and overview were committed in
        # one SQLite transaction by ``_persist_start_turn_atomically`` above.
        return join_turn_response(repo, task.id)
    except _StrategySampleDesignRequiredError as exc:
        if spec.intent != "strategy":
            raise
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
            fields=_STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
        )
    except C1TargetValidationError:
        raise
    except spec.setup_error_types as exc:
        return append_workflow_error(repo, task, spec, exc, setup_error=True)
    except DriverError:
        raise
    except Exception as exc:
        diagnostic_overrides = None
        if feature_semantic_plan_start:
            diagnostic_overrides = {
                "code": "feature_target_plan_start_rolled_back",
                "retry_instruction_sha256": hashlib.sha256(
                    str(user_text or "").strip().encode("utf-8")
                ).hexdigest(),
            }
        return append_workflow_error(
            repo,
            task,
            spec,
            exc,
            diagnostic_overrides=diagnostic_overrides,
        )

def _safe_semantic_intent_state_snapshot(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
) -> str | None:
    """Return ``None`` when the complete route target cannot be authenticated."""

    try:
        return _semantic_intent_state_snapshot(runtime, repo, task)
    except Exception:  # noqa: BLE001 - an incomplete snapshot must fail closed
        return None

def _instruction_explicitly_names_identifier(
    instruction: str,
    identifier: str,
) -> bool:
    """Ground an LLM-selected binding in exact user-supplied operands.

    This does not classify intent or look for action keywords.  It only proves
    that the already-selected bounded operation names the platform candidate it
    would mutate, preventing an erroneous route from silently binding a
    different file or column.
    """

    text = str(instruction or "")
    value = str(identifier or "").strip()
    if not text or not value:
        return False
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
                text,
                re.IGNORECASE,
            )
        )
    return value in text

_SPECIALIZED_WORKFLOW_INTAKE_TYPES = frozenset(
    {TASK_TYPE_MODELING, TASK_TYPE_FEATURE_ANALYSIS}
)

_MODELING_INTAKE_PARAM_NAMES = frozenset(
    {
        "target_type",
        "recipes",
        "split_config",
        "n_trials",
        "sample_weight_col",
    }
)

def dispatch_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    agent_client,
    auto_accept_enabled: bool = False,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    expected_plan_id: str | None = None,
    expected_plan_status: str | None = None,
    expected_plan_revision: int | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_step_fingerprint: str | None = None,
    strategy_request: Mapping[str, object] | None = None,
    portfolio_request: Mapping[str, object] | None = None,
    labeling_request: Mapping[str, object] | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    ui_action: str | None = None,
    recovery_bypass: bool = False,
) -> dict:
    if labeling_request is not None:
        return _handle_structured_labeling_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            labeling_request=labeling_request,
        )
    if portfolio_request is not None:
        return _handle_structured_portfolio_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            portfolio_request=portfolio_request,
        )
    # Candidate Lab controls are already a canonical user request. They get
    # first refusal inside the same task driver-job lock and never pass through
    # recovery, text intent routing, or an LLM.
    if strategy_request is not None:
        return _handle_structured_strategy_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            strategy_request=strategy_request,
        )
    if ui_action is None:
        labeling_confirmation = _maybe_handle_labeling_preplan_confirmation_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            confirmation_source=confirmation_source,
        )
        if labeling_confirmation is not None:
            return labeling_confirmation
    # UI controls are already typed, governed commands. Route them directly to
    # the task driver so generic recovery/analysis/strategy text classifiers
    # cannot reinterpret their display copy before optimistic-lock validation.
    if ui_action is not None:
        result = DRIVER_TURN_FUNCS[task.task_type](
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
        if result.get("status") == "clarification_required":
            return result
        if agent_client is not None and auto_accept_enabled:
            agent_autodrive_turn(runtime, repo, task, client=agent_client)
            return join_turn_response(repo, task.id)
        return result
    # An unresolved structured failure owns ordinary conversation first.  In
    # particular, “为什么策略分析失败” is a question about existing evidence,
    # not authorization to compile and run a new strategy request.
    recovery = _maybe_handle_workflow_recovery_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        recovery_bypass=recovery_bypass,
    )
    if recovery is not None:
        return recovery
    text = str(user_text or "")
    if (
        _active_plan(runtime.plan_repo, task.id) is None
        and is_explicit_workflow_retry(text)
        and latest_unresolved_workflow_failure(
            repo.list_agent_messages(task.id),
            workflow=task.task_type,
        )
        is not None
    ):
        # Setup/material failures can occur before a Plan exists.  The recovery
        # helper intentionally returns None for that case so the task-owned
        # setup can run again.  Keep this state-bound retry on the deterministic
        # recovery path; sending it through top-level semantic routing would
        # turn a proven retry command into an unrelated clarification whenever
        # the router model is unavailable.
        result = DRIVER_TURN_FUNCS[task.task_type](
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
        if result.get("status") == "clarification_required":
            return result
        if agent_client is not None and auto_accept_enabled:
            agent_autodrive_turn(runtime, repo, task, client=agent_client)
            return join_turn_response(repo, task.id)
        return result
    # A reply that names exactly one candidate from the live Feature target
    # card is setup input, not an ad-hoc analysis request.  Keep it ahead of
    # the generic intent helpers both on the first attempt and after a
    # transaction-level setup failure.  The setup handler still performs the
    # full two-pass semantic authorization in Agent mode, so this routing
    # priority cannot itself authorize or persist the choice.
    feature_target_state = (
        _latest_feature_target_state(repo.list_agent_messages(task.id))
        if task.task_type == TASK_TYPE_FEATURE_ANALYSIS
        else None
    )
    if (
        feature_target_state is not None
        and _natural_language_c1_target(text, feature_target_state) is not None
    ):
        result = DRIVER_TURN_FUNCS[task.task_type](
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
        if result.get("status") == "clarification_required":
            return result
        if agent_client is not None and auto_accept_enabled:
            agent_autodrive_turn(runtime, repo, task, client=agent_client)
            return join_turn_response(repo, task.id)
        return result
    conversation = repo.list_agent_messages(task.id)
    if task.task_type == TASK_TYPE_STRATEGY and not (
        utterance_targets_strategy_project_context(text)
        or _is_strategy_request_intent(text)
    ):
        project_context_answer = _maybe_handle_project_context_missing_answer(
            runtime,
            repo,
            task,
            text=text,
            conversation=conversation,
        )
        if project_context_answer is not None:
            return project_context_answer

    # These legacy pending contracts already have their own deterministic
    # state-bound confirmation parser. Keep them out of top-level routing so
    # the new classifier cannot steal an existing confirmation turn. Ad-hoc
    # pending is intentionally excluded: it is the contract upgraded below.
    has_existing_pending_contract = (
        _latest_pending_transform_protected_drop(conversation) is not None
        or (
            task.task_type == TASK_TYPE_STRATEGY
            and (
                _latest_strategy_request_pending(conversation) is not None
                or _latest_strategy_nan_label_confirmation(conversation) is not None
            )
        )
    )
    if (
        runtime.require_semantic_text_authorization
        and text.strip()
        and not has_existing_pending_contract
        and _active_plan(runtime.plan_repo, task.id) is None
        and latest_open_gate(conversation) is None
        and _specialized_workflow_intake_is_pending(runtime, task, conversation)
    ):
        before_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if before_snapshot is None:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或数据状态无法完成一致性校验，未创建计划。",
            )
        intake_route, intake_reason = _specialized_workflow_intake_route(
            runtime,
            task,
            instruction=text,
        )
        if intake_route is None:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason=intake_reason,
            )
        after_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if after_snapshot is None or before_snapshot != after_snapshot:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="专属工作流语义理解期间任务状态已变化，旧判断已作废。",
            )
        runtime = replace(
            runtime,
            semantic_intent=INTENT_CURRENT_WORKFLOW,
            workflow_intake_route=intake_route,
        )
        result = DRIVER_TURN_FUNCS[task.task_type](
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
        if result.get("status") == "clarification_required":
            return result
        if agent_client is not None and auto_accept_enabled:
            agent_autodrive_turn(runtime, repo, task, client=agent_client)
            return join_turn_response(repo, task.id)
        return result
    semantic_decision = None
    pending_adhoc = None
    if (
        runtime.require_semantic_text_authorization
        and text.strip()
        and not has_existing_pending_contract
        and _active_plan(runtime.plan_repo, task.id) is None
        and latest_open_gate(conversation) is None
    ):
        before_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if before_snapshot is None:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或数据状态无法完成一致性校验，未执行任何动作。",
                pending_adhoc=_latest_adhoc_pending(
                    repo.list_agent_messages(task.id)
                ),
            )
        try:
            semantic_context, allowed_intents, pending_adhoc = (
                _semantic_intent_route_contract(runtime, repo, task)
            )
        except Exception:  # noqa: BLE001 - incomplete material context must fail closed
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或可用材料无法完成语义路由校验，未执行任何动作。",
                pending_adhoc=_latest_adhoc_pending(
                    repo.list_agent_messages(task.id)
                ),
            )
        semantic_decision = route_semantic_intent(
            runtime.llm_client,
            task_type=task.task_type,
            instruction=text,
            context=semantic_context,
            allowed_intents=allowed_intents,
        )
        if not semantic_decision.accepted:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason=semantic_decision.reason,
                pending_adhoc=pending_adhoc,
            )
        after_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if after_snapshot is None or before_snapshot != after_snapshot:
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="语义复核期间任务状态已变化，旧判断已作废。",
                pending_adhoc=_latest_adhoc_pending(
                    repo.list_agent_messages(task.id)
                ),
            )
        runtime = replace(runtime, semantic_intent=semantic_decision.intent)

        semantic_handler = {
            INTENT_ADHOC_QUERY: lambda: _maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
            INTENT_ADHOC_CONFIRM: lambda: _maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_CONFIRM,
            ),
            INTENT_ADHOC_REJECT: lambda: _maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_REJECT,
            ),
            INTENT_ADHOC_REVISE: lambda: _maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_REVISE,
            ),
            INTENT_DATASET_TRANSFORM: lambda: _maybe_handle_dataset_transform_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
            INTENT_DATASET_EXPORT: lambda: _maybe_handle_dataset_export_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
            INTENT_DATASET_ANALYSIS: lambda: _maybe_handle_dataset_analysis_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
            INTENT_STRATEGY_SAMPLE_BINDING: lambda: (
                _handle_strategy_sample_binding_intent(
                    runtime,
                    repo,
                    task,
                    user_text=text,
                )
            ),
            INTENT_STRATEGY_WORKFLOW: lambda: _maybe_handle_strategy_request_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
        }.get(semantic_decision.intent)
        if semantic_handler is not None:
            semantic_result = semantic_handler()
            if semantic_result is not None:
                return semantic_result
            return _semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="语义路由已识别，但确定性入口当前不满足执行条件。",
                pending_adhoc=pending_adhoc,
            )
        # A current-workflow or risk-kind decision deliberately bypasses every
        # generic lexical detector below. The task state machine and its typed
        # compiler/validator remain the only owners of the next action.
        if semantic_decision.intent in {
            INTENT_CURRENT_WORKFLOW,
            INTENT_DATASET_JOIN,
            INTENT_RISK_PROFITABILITY,
            INTENT_RISK_VTG_TERMINAL,
            INTENT_RISK_STANDARD_VINTAGE,
        }:
            result = DRIVER_TURN_FUNCS[task.task_type](
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
            if result.get("status") == "clarification_required":
                return result
            if agent_client is not None and auto_accept_enabled:
                agent_autodrive_turn(runtime, repo, task, client=agent_client)
                return join_turn_response(repo, task.id)
            return result
    # A positive command to create StrategySampleDesign owns phrases such as
    # "不设筛选" and "不丢弃缺失标签": those are sample-design contract fields,
    # not authorization to mutate the dataset.  Give only this narrowly
    # recognized Strategy workflow first refusal before the generic transform
    # detector.  References to an already-fixed SampleDesign do not match
    # utterance_targets_strategy_sample_design(), so genuine data-processing
    # requests retain the transform route and its existing safety checks.
    if (
        task.task_type == TASK_TYPE_STRATEGY
        and utterance_targets_strategy_sample_design(text)
    ):
        strategy_sample_design_request = _maybe_handle_strategy_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
        )
        if strategy_sample_design_request is not None:
            return strategy_sample_design_request
    # Dataset changes get first refusal over descriptive analysis.  Phrases
    # such as "填充缺失值" describe a governed mutation, not a request for a
    # missing-value report; the transform always creates an immutable child.
    dataset_transform = _maybe_handle_dataset_transform_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if dataset_transform is not None:
        return dataset_transform
    dataset_export = _maybe_handle_dataset_export_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if dataset_export is not None:
        return dataset_export
    if task.task_type == TASK_TYPE_STRATEGY and (
        utterance_targets_candidate_monthly_stability(text)
        or utterance_targets_model_score_comparison_v2(text)
        or utterance_targets_interactive_tree_frontier_group_materialization(
            text
        )
        or utterance_targets_interactive_tree_frontier_materialization(text)
        or utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
        or utterance_targets_strategy_sample_design(text)
        or utterance_targets_strategy_dsl_delivery(text)
        or utterance_targets_strategy_report_bundle_v2(text)
        or utterance_targets_strategy_impact_cube(text)
        or utterance_targets_strategy_pool_stability(text)
        or _STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE.search(text) is not None
    ):
        strategy_evidence_request = _maybe_handle_strategy_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
        )
        if strategy_evidence_request is not None:
            return strategy_evidence_request
    # Explicit dataset diagnostics are narrower than the strategy compiler's
    # generic "分析" operation. Give this branch first refusal so phrases such
    # as "分析当前样本" cannot be mistaken for a request to design a strategy.
    dataset_analysis = _maybe_handle_dataset_analysis_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if dataset_analysis is not None:
        return dataset_analysis
    # A strategy-specific request gets first refusal only when it names both a
    # strategy subject and an operation. Raw data questions then retain the S6
    # ad-hoc path; anything else falls through to the normal task handler.
    strategy_request = _maybe_handle_strategy_request_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if strategy_request is not None:
        return strategy_request
    adhoc = _maybe_handle_adhoc_turn(runtime, repo, task, user_text=user_text)
    if adhoc is not None:
        return adhoc
    result = DRIVER_TURN_FUNCS[task.task_type](
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
    if result.get("status") == "clarification_required":
        return result
    if agent_client is not None and auto_accept_enabled:
        agent_autodrive_turn(runtime, repo, task, client=agent_client)
        return join_turn_response(repo, task.id)
    return result

def _maybe_handle_workflow_recovery_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None,
    dedup_strategies: dict | None,
    adjust_params: dict | None,
    expected_step_id: str | None,
    recovery_bypass: bool,
) -> dict | None:
    """Keep failed Agent tasks conversational until retry is explicit."""

    text = str(user_text or "").strip()
    if recovery_bypass or task.run_mode != "agent" or not text:
        return None
    # Structured UI actions are already explicit execution input. They must keep
    # their existing gate/setup path instead of being reclassified as chat.
    if any(
        value is not None
        for value in (selection, dedup_strategies, adjust_params, expected_step_id)
    ):
        return None
    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    cancelled_recovery = _maybe_resume_cancelled_plan(
        runtime,
        repo,
        task,
        text=text,
        conversation=conversation,
    )
    if cancelled_recovery is not None:
        return cancelled_recovery
    rollback_intent = parse_workflow_rollback_intent(text)
    tuning_budget_intent = parse_tuning_budget_revision_intent(text)
    champion_refit_intent = parse_champion_refit_revision_intent(text)
    explicit_retry = is_explicit_workflow_retry(text)
    failure = latest_unresolved_workflow_failure(
        conversation,
        workflow=task.task_type,
    )
    legacy_restart_failure = _legacy_restart_failure_context(
        conversation,
        runtime.plan_repo,
        task_id=task.id,
        workflow=task.task_type,
    )
    # A failed plan is authoritative over a stale setup gate accidentally
    # appended after an old restart notice.  For all other utterances, retain
    # the normal gate routing (notably a bare “继续” must not execute a retry).
    if latest_open_gate(conversation) is not None and legacy_restart_failure is None:
        return None
    if failure is None:
        failure = legacy_restart_failure
    if failure is None:
        return None
    # A setup transaction can fail after a target choice was semantically
    # reviewed but before plan/target/receipt commit.  The prior target card is
    # still the authoritative open gate in that case.  Repeating an exact
    # candidate-bearing instruction must re-enter the C1 two-pass review,
    # rather than being swallowed as generic failure chat (which would make a
    # second authorization impossible without a magic recovery phrase).
    retryable = bool(failure.diagnostic.get("retryable", True))
    if failure.failure_envelope is not None:
        retryable = bool(failure.failure_envelope.get("retryable", retryable))
    expected_retry_hash = str(
        failure.diagnostic.get("retry_instruction_sha256") or ""
    )
    current_retry_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    if (
        retryable
        and task.task_type == TASK_TYPE_FEATURE_ANALYSIS
        and failure.diagnostic.get("code")
        == "feature_target_plan_start_rolled_back"
        and bool(expected_retry_hash)
        and hmac.compare_digest(expected_retry_hash, current_retry_hash)
    ):
        feature_target_state = _latest_feature_target_state(conversation)
        if (
            feature_target_state is not None
            and _natural_language_c1_target(text, feature_target_state)
        ):
            return None
    repair_authorized = bool(
        failure.diagnostic.get("auto_recoverable")
    ) and is_workflow_repair_request(text)
    if retryable and champion_refit_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        failed_step = next(
            (
                step
                for step in (failed_plan.steps if failed_plan is not None else [])
                if step.id == failed_step_id
            ),
            None,
        )
        if (
            failed_plan is None
            or failed_step is None
            or failed_step.tool_ref.tool != "select_experiment"
        ):
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=(
                    "当前失败位置不是“选择实验”，为避免把重训设置改到错误步骤，"
                    "本次未修改计划。"
                ),
                metadata={
                    "intent": "workflow_recovery_champion_refit_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_select_step_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        input_updates: dict[str, object] = {"refit_on_train_plus_test": False}
        if champion_refit_intent.selected_experiment_id:
            input_updates["selected_experiment_id"] = (
                champion_refit_intent.selected_experiment_id
            )
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_champion_refit_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                **input_updates,
            },
        )
        turn = _driver(runtime).retry_failed_step(
            failed_plan.id,
            failed_step_id,
            run_seq=int(envelope.get("run_seq") or 0) + 1,
            inputs=input_updates,
            # Changing refit semantics requires a fresh governed confirmation;
            # never carry the previous decision over to the revised input hash.
            preserve_target_confirmation=False,
        )
        append_driver_messages(repo, task, turn, runtime=runtime)
        return join_turn_response(repo, task.id)
    if retryable and tuning_budget_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        requested_budgets = dict(tuning_budget_intent.n_trials_by_recipe)
        if failed_plan is None or not failed_step_id:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="没有找到与当前故障证据一致的失败计划；为避免修改错误计划，本次未调整调参预算。",
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_plan_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        try:
            turn = _driver(runtime).rollback_failed_plan_to_tuning_config(
                failed_plan.id,
                failed_step_id,
                default_n_trials=tuning_budget_intent.default_n_trials,
                n_trials_by_recipe=requested_budgets,
                run_seq=int(envelope.get("run_seq") or 0) + 1,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "step_id": failed_step_id,
                    "default_n_trials": tuning_budget_intent.default_n_trials,
                    "n_trials_by_recipe": requested_budgets,
                },
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"未修改当前计划：{exc}",
                metadata={
                    "intent": "workflow_recovery_tuning_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_tuning_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                "root_step": "configure_tuning",
                "default_n_trials": tuning_budget_intent.default_n_trials,
                "n_trials_by_recipe": requested_budgets,
            },
        )
        append_driver_messages(repo, task, turn, runtime=runtime)
        return join_turn_response(repo, task.id)
    if retryable and rollback_intent is not None:
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        if failed_plan is None or not failed_step_id:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="没有找到与当前故障证据一致的失败计划；为避免修改错误计划，本次未执行回退。",
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "reason": "failed_plan_not_found",
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        try:
            turn = _driver(runtime).rollback_failed_plan_to_feature_screen(
                failed_plan.id,
                failed_step_id,
                excluded_features=list(rollback_intent.excluded_features),
                run_seq=int(envelope.get("run_seq") or 0) + 1,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "step_id": failed_step_id,
                    "root_step": rollback_intent.root_step,
                    "excluded_features": list(rollback_intent.excluded_features),
                },
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"未修改当前计划：{exc}",
                metadata={
                    "intent": "workflow_recovery_revision_rejected",
                    "recovery_of_message_id": failure.message_id,
                    "plan_id": failed_plan.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_revision",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
                "root_step": rollback_intent.root_step,
                "excluded_features": list(rollback_intent.excluded_features),
            },
        )
        append_driver_messages(repo, task, turn, runtime=runtime)
        return join_turn_response(repo, task.id)
    if retryable and (explicit_retry or repair_authorized):
        envelope = failure.failure_envelope or {}
        failed_step_id = str(envelope.get("failed_step_id") or "")
        failed_plan = next(
            (
                plan
                for plan in reversed(runtime.plan_repo.list_plans_for_task(task.id))
                if getattr(plan.status, "value", plan.status) == PlanStatus.FAILED.value
                and any(step.id == failed_step_id for step in plan.steps)
            ),
            None,
        )
        if failed_plan is None or not failed_step_id:
            return None
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_recovery_retry",
                "recovery_of_message_id": failure.message_id,
                "plan_id": failed_plan.id,
                "step_id": failed_step_id,
            },
        )
        turn = _driver(runtime).retry_failed_step(
            failed_plan.id,
            failed_step_id,
            run_seq=int(envelope.get("run_seq") or 0) + 1,
            # This branch is reached only from an explicit human Agent-mode
            # recovery command.  Reuse the confirmation only if this exact
            # failed step had already been confirmed; the repository keeps an
            # unconfirmed governed gate unconfirmed and clears all downstream
            # confirmations.
            preserve_target_confirmation=True,
        )
        append_driver_messages(repo, task, turn, runtime=runtime)
        return join_turn_response(repo, task.id)

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "workflow_recovery_chat",
            "recovery_of_message_id": failure.message_id,
        },
    )
    if runtime.recovery_responder is None:
        content = deterministic_workflow_recovery_reply(failure.diagnostic)
        response_metadata = {
            "fallback": True,
            "fallback_reason": "recovery_responder_unavailable",
        }
    else:
        content, response_metadata = runtime.recovery_responder(
            task=task,
            user_message=text,
            diagnostic=failure.diagnostic,
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=content,
        metadata={
            "intent": "workflow_recovery_chat",
            "recovery_of_message_id": failure.message_id,
            "recovery_code": failure.diagnostic.get("code"),
            **dict(response_metadata or {}),
        },
    )
    return {
        "task_id": task.id,
        "status": "message_saved",
        "messages": repo.list_agent_messages(task.id),
    }

def _maybe_resume_cancelled_plan(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    text: str,
    conversation: list[dict],
) -> dict | None:
    """Keep a stopped plan recoverable without rebuilding its setup.

    Cancellation is terminal for the executor invocation, but not destructive:
    its current step is persisted as failed and prior outputs/runs remain.  A
    fresh Agent turn can therefore reopen that exact step with a fresh
    cancellation token.  Non-command chat never starts a new plan implicitly.
    """

    plans = runtime.plan_repo.list_plans_for_task(task.id)
    if not plans:
        return None
    plan = plans[-1]
    if getattr(plan.status, "value", plan.status) != PlanStatus.CANCELLED.value:
        return None
    cancelled_message = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
            and (message.get("metadata") or {}).get("intent")
            == "execution_cancelled"
            and str((message.get("metadata") or {}).get("plan_id") or plan.id)
            == plan.id
        ),
        None,
    )
    metadata = (cancelled_message or {}).get("metadata") or {}
    step_id = str(metadata.get("step_id") or "")
    interrupted = next(
        (
            step
            for step in plan.steps
            if step.status == StepStatus.FAILED
            and (not step_id or step.id == step_id)
        ),
        None,
    )
    if interrupted is None:
        return None

    recovery_of_message_id = (cancelled_message or {}).get("id")
    if is_explicit_cancelled_workflow_resume(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={
                "intent": "workflow_cancelled_resume",
                "recovery_of_message_id": recovery_of_message_id,
                "plan_id": plan.id,
                "step_id": interrupted.id,
            },
        )
        try:
            turn = _driver(runtime).retry_failed_step(
                plan.id,
                interrupted.id,
                run_seq=int(metadata.get("run_seq") or 0) + 1,
                preserve_target_confirmation=True,
            )
        except DriverError as exc:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"当前停止点未能恢复：{exc}",
                metadata={
                    "intent": "workflow_cancelled_resume_rejected",
                    "recovery_of_message_id": recovery_of_message_id,
                    "plan_id": plan.id,
                    "step_id": interrupted.id,
                    "reason": str(exc),
                },
            )
            return {
                "task_id": task.id,
                "status": "message_saved",
                "messages": repo.list_agent_messages(task.id),
            }
        append_driver_messages(repo, task, turn, runtime=runtime)
        return join_turn_response(repo, task.id)

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "workflow_cancelled_chat",
            "recovery_of_message_id": recovery_of_message_id,
            "plan_id": plan.id,
            "step_id": interrupted.id,
        },
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"当前计划仍停在“{interrupted.title}”，本轮没有执行新步骤。"
            "已完成结果和检查点都在；要从这里恢复，请明确输入“继续当前步骤”。"
        ),
        metadata={
            "intent": "workflow_cancelled_chat",
            "recovery_of_message_id": recovery_of_message_id,
            "plan_id": plan.id,
            "step_id": interrupted.id,
        },
    )
    return {
        "task_id": task.id,
        "status": "message_saved",
        "messages": repo.list_agent_messages(task.id),
    }

def _legacy_restart_failure_context(
    messages: list[dict],
    plan_repo,
    *,
    task_id: str,
    workflow: str,
) -> WorkflowFailureContext | None:
    """Recover the target encoded by pre-envelope restart notifications.

    Older startup recovery notices only persisted ``plan_id``.  Because a
    plan-scoped assistant message is normally a progress boundary, those
    notices intentionally cannot be treated as a generic unresolved failure.
    It accepts only the latest task plan with exactly one failed step.  The
    caller still requires an explicit retry command before executing; ordinary
    conversation remains in recovery chat instead of falling back to setup.
    """

    plans = plan_repo.list_plans_for_task(task_id)
    if not plans:
        return None
    latest_plan = plans[-1]
    if getattr(latest_plan.status, "value", latest_plan.status) != PlanStatus.FAILED.value:
        return None
    failed_steps = [
        step
        for step in latest_plan.steps
        if getattr(step.status, "value", step.status) == "failed"
    ]
    if len(failed_steps) != 1:
        return None
    failed_step = failed_steps[0]

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata") or {}
        if not metadata.get("plan_interrupted_by_restart"):
            continue
        if metadata.get("plan_resumed_at_confirmation"):
            continue
        if str(metadata.get("plan_id") or "") != latest_plan.id:
            continue
        summary = (
            f"服务重启中断了“{failed_step.title}”步骤，"
            "已完成步骤和中间产物均已保留。"
        )
        return WorkflowFailureContext(
            message_id=str(message.get("id") or "") or None,
            diagnostic={
                "schema_version": "workflow_error.v1",
                "workflow": workflow,
                "code": "server_restart_interrupted",
                "phase": "execution",
                "title": "工作流执行中断",
                "summary": summary,
                "cause": "MARVIS 服务在该步骤执行期间重启；源材料没有被修改。",
                "location": failed_step.title,
                "evidence": [
                    {"label": "计划", "value": latest_plan.id},
                    {"label": "失败步骤", "value": failed_step.id},
                ],
                "actions": ["回复“重试当前步骤”，从该失败步骤继续。"],
                "retryable": True,
                "auto_recoverable": True,
                "impact": "失败步骤之后的依赖步骤尚未执行。",
                "exception_type": "ServerRestart",
                "technical_detail": str(failed_step.error or "ServerRestart"),
            },
            failure_envelope={
                "schema_version": "failure.v1",
                "failed_step_id": failed_step.id,
                "error_kind": "ServerRestart",
                "message": summary,
                "retryable": True,
                "editable_input_schema": {},
                "suggested_actions": ["retry"],
                "downstream_reset": "dependent_steps",
                "downstream_reset_steps": [],
            },
        )
    return None

def agent_autodrive_turn(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, *, client
) -> None:
    turn_fn = DRIVER_TURN_FUNCS[task.task_type]
    max_gates = _auto_gate_budget(runtime, task.id)
    processed_gates = 0
    while True:
        # A turn can start at the pre-plan C1 gate and create its real plan only
        # after the first AUTO decision.  Re-read the bounded tier budget before
        # each subsequent gate so that the newly known plan depth is honored;
        # otherwise the one-time eight-gate fallback can stop immediately before
        # the mandatory human gate it was meant to reach.
        max_gates = max(max_gates, _auto_gate_budget(runtime, task.id))
        if processed_gates >= max_gates:
            break
        gate = latest_open_gate(repo.list_agent_messages(task.id))
        if gate is None:
            return
        processed_gates += 1
        # MEM-1 read side: attach a read-only 【历史同类实验】 anchor to the gate
        # metadata (rendered by auto_drive._format_gate) before the LLM sees it.
        # build_memory_anchor is a strict no-op (returns None) unless this is a
        # modeling select-experiment/tuning gate with comparable history and the
        # reference_cross_task policy is on, so every other gate/task type is
        # completely unaffected.
        memory_anchor = None
        driver_settings = getattr(runtime, "settings", None)
        if driver_settings is not None:
            memory_anchor = build_memory_anchor(
                driver_settings,
                task,
                gate_metadata=gate.get("metadata")
                if isinstance(gate.get("metadata"), dict)
                else {},
            )
        if memory_anchor is not None:
            gate = dict(gate)
            gate_metadata = dict(gate.get("metadata") or {})
            gate_metadata["memory_anchor"] = memory_anchor["lines"]
            gate["metadata"] = gate_metadata
        try:
            decision = decide_gate(client, gate=gate)
        except LLMClientError as exc:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content=f"⚠️ 自动决策失败（{exc}），请手动确认或重试。",
                metadata={"intent": "agent_error"},
            )
            return
        action = decision["action"]
        decision_meta = {"intent": "agent_decision", "action": action}
        for key in (
            "params",
            "selection",
            "dedup_strategies",
            "replan_goal",
            "clarifying_question",
            "confidence",
            "safety_rationale",
        ):
            if key in decision:
                decision_meta[key] = decision[key]
        if memory_anchor is not None:
            decision_meta["memory_references"] = memory_anchor["references"]
        decision_message = repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=_auto_decision_content(decision),
            metadata=decision_meta,
        )
        if memory_anchor is not None and driver_settings is not None:
            try:
                audit_agent_memory_use_from_store(
                    AgentMemoryStore(driver_settings.db_path),
                    decision_message,
                    task_id=task.id,
                )
            except Exception as exc:
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        "⚠️ 自动决策引用的历史记忆未能写入审计记录；"
                        "为避免无审计执行，本次已停止。请重试或手动确认当前步骤。"
                    ),
                    metadata={
                        "intent": "agent_error",
                        "kind": "governance_block",
                        "code": "memory_use_audit_failed",
                        "decision_message_id": (
                            decision_message.get("id")
                            if isinstance(decision_message, dict)
                            else None
                        ),
                        "error_type": type(exc).__name__,
                    },
                )
                return
        gate_meta = (
            gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {}
        )
        gate_step_id = gate_meta.get("step_id")
        if action == "confirm":
            turn_fn(
                runtime,
                repo,
                task,
                user_text="确认",
                expected_step_id=gate_step_id,
                confirmation_source=CONFIRMATION_SOURCE_AUTO,
            )
            continue
        if action == "adjust":
            params = (
                decision.get("params")
                if isinstance(decision.get("params"), dict)
                else None
            )
            selection = (
                decision.get("selection")
                if isinstance(decision.get("selection"), list)
                else None
            )
            dedup = (
                decision.get("dedup_strategies")
                if isinstance(decision.get("dedup_strategies"), dict)
                else None
            )
            if not (params or selection or dedup):
                return
            turn_fn(
                runtime,
                repo,
                task,
                user_text=decision["reason"],
                selection=selection,
                dedup_strategies=dedup,
                adjust_params=params,
                expected_step_id=gate_meta.get("step_id"),
                confirmation_source=CONFIRMATION_SOURCE_AUTO,
            )
            continue
        if action == "replan":
            # AGT-8: go straight to the driver's structured replan path instead of
            # feeding replan_goal back as free-text user_text. Text loopback risked
            # (a) is_confirm misreading a phrase like "……并继续调参" as a plain
            # confirm and confirming the very gate that was supposed to be
            # restructured (same root cause as AGT-1), and (b) an extra LLM
            # round-trip re-classifying a decision that was already structured,
            # which could misjudge it as clarify and silently drop the replan.
            goal = decision.get("replan_goal") or decision["reason"]
            plan_id = gate_meta.get("plan_id")
            if not plan_id:
                return
            driver = _driver(runtime)
            try:
                turn = driver.replan_structured(
                    plan_id=plan_id,
                    goal=goal,
                    expected_step_id=gate_step_id,
                    confirmation_source=CONFIRMATION_SOURCE_AUTO,
                )
            except DriverError:
                return
            append_driver_messages(repo, task, turn, runtime=runtime)
            continue
        return
    # AGT-7: the budget ran out with a gate STILL open (every iteration matched a
    # real gate and looped back via confirm/adjust/replan) — tell the user
    # explicitly instead of silently going quiet, which previously looked like
    # the agent had inexplicably stopped responding.
    if latest_open_gate(repo.list_agent_messages(task.id)) is not None:
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"🤖 AUTO 已连续自动处理 {max_gates} 个节点，为安全起见转人工确认；"
                "请查看当前节点并回复「确认」或给出调整指令以继续。"
            ),
            metadata={"intent": "agent_budget_exhausted", "max_gates": max_gates},
        )

def append_driver_messages(
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    *,
    runtime: DriverTurnRuntime,
) -> None:
    """Persist driver output with the complete governed runtime context."""

    _append_driver_messages_core(
        repo,
        task.id,
        turn,
        settings=runtime.settings,
        task=task,
        llm_client=getattr(runtime, "llm_client", None),
        hook_dispatcher=getattr(runtime, "hook_dispatcher", None),
    )

def _append_context_free_driver_messages(
    repo: TaskRepository,
    task_id: str,
    turn,
) -> None:
    """Persist intentionally context-free output such as an ad-hoc slice."""

    _append_driver_messages_core(repo, task_id, turn)

def _append_driver_messages_core(
    repo: TaskRepository,
    task_id: str,
    turn,
    *,
    settings=None,
    task: TaskRecord | None = None,
    llm_client=None,
    hook_dispatcher=None,
) -> None:
    for message in turn.messages:
        metadata = dict(message.metadata)
        content = message.content
        memory_context = None
        if settings is not None and task is not None and message.stage in {"gate", "done"}:
            memory_context = build_workflow_memory_context(settings, task)
            if memory_context is not None:
                metadata["memory_references"] = memory_context["references"]
                metadata["memory_context"] = {
                    "category": memory_context["category"],
                    "count": len(memory_context["memories"]),
                    "lines": memory_context["lines"],
                }
                content += "\n\n**本次参考的历史记忆**\n" + "\n".join(
                    f"- {line}" for line in memory_context["lines"]
                )
        insight_context = None
        if task is not None and message.stage in {"gate", "done"}:
            insight_context = build_workflow_insight_context(
                task.task_type,
                stage=message.stage,
                metadata=metadata,
                content=content,
            )
        if insight_context is not None:
            insight = render_workflow_insight(
                insight_context,
                client=llm_client,
                memory_context=(memory_context or {}).get("memories") or [],
            )
            content += f"\n\n{insight['content']}"
            metadata["agent_insight"] = {
                key: value for key, value in insight.items() if key != "content"
            }
        receipts: list[dict] = []
        if settings is not None and task is not None and message.stage == "done":
            receipts = capture_agent_memory_for_driver_done(
                settings,
                task,
                done_message_content=content,
                done_message_metadata=metadata,
                hook_dispatcher=hook_dispatcher,
            )
            metadata["memory_capture"] = {
                "enabled": load_memory_policy(settings.workspace).auto_distill,
                "saved": receipts,
            }
            if receipts:
                content += "\n\n**本次记忆沉淀**\n" + "\n".join(
                    f"- 已保存 `{item['memory_type']}`：{item['summary']}"
                    for item in receipts
                )
                content += "\n- 已触发记忆归并，后续同类任务可引用。"
            elif not load_memory_policy(settings.workspace).auto_distill:
                content += "\n\n**本次记忆沉淀**\n- 自动沉淀已在设置中关闭，本次未保存跨任务记忆。"
        stored_message = repo.add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=content,
            metadata=metadata,
        )
        if memory_context is not None and settings is not None and task is not None:
            try:
                audit_agent_memory_use_from_store(
                    AgentMemoryStore(settings.db_path),
                    stored_message,
                    task_id=task.id,
                )
            except Exception:
                pass

def append_workflow_error(
    repo: TaskRepository,
    task: TaskRecord,
    spec: _TurnHandlerSpec,
    exc: Exception,
    *,
    setup_error: bool = False,
    diagnostic_overrides: Mapping[str, object] | None = None,
) -> dict:
    diagnostic = build_workflow_error_diagnostic(
        workflow=spec.intent,
        exc=exc,
        task=task,
        setup_error=setup_error,
    )
    if diagnostic_overrides:
        diagnostic.update(dict(diagnostic_overrides))
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=workflow_error_content(diagnostic),
        metadata={
            "error": True,
            "intent": spec.intent,
            "error_diagnostic": diagnostic,
            "failure_envelope": failure_envelope_for_diagnostic(diagnostic),
        },
    )
    return {
        "task_id": task.id,
        "status": "error",
        "messages": repo.list_agent_messages(task.id),
    }

def latest_open_gate(messages: list[dict]) -> dict | None:
    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    if last_assistant is None:
        return None
    meta = last_assistant.get("metadata") or {}
    if meta.get("error") or meta.get("join_skip"):
        return None
    if meta.get("kind") in ("gate", "plan_overview") or "join_c1" in meta:
        return last_assistant
    return None

def _driver(runtime: DriverTurnRuntime) -> PlanDriver:
    return PlanDriver(
        runtime.plan_repo,
        runtime.plan_executor,
        planner=runtime.planner,
        validator=runtime.plan_validator,
        llm_client=runtime.llm_client,
        allow_manual_gate_adapters=runtime.allow_manual_gate_adapters,
        require_semantic_text_authorization=(
            runtime.require_semantic_text_authorization
        ),
        governance_service=runtime.governance_service,
        local_principal=runtime.local_principal,
        cancellation_check=runtime.cancellation_check,
    )

def _resume_new_routed_plan(
    runtime: DriverTurnRuntime,
    driver: PlanDriver,
    *,
    plan_id: str,
    semantic_reason: str,
):
    """Confirm an auto-runnable plan against the post-start immutable snapshot."""

    kwargs: dict[str, object] = {
        "plan_id": plan_id,
        "user_text": "确认",
    }
    if runtime.require_semantic_text_authorization and runtime.semantic_intent:
        plan = runtime.plan_repo.load_plan(plan_id)
        kwargs.update(
            {
                "_confirmation_reason": semantic_reason,
                "_expected_plan_revision": int(plan.replan_count),
                "_expected_plan_status": plan.status.value,
                "_expected_plan_fingerprint": plan_fingerprint(plan),
            }
        )
    return driver.resume(**kwargs)

def _active_plan(plan_repo, task_id: str):
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        status = getattr(plan.status, "value", plan.status)
        if status not in _TERMINAL_PLAN_STATUS_VALUES:
            return plan
    return None

def _auto_gate_budget(runtime: DriverTurnRuntime, task_id: str) -> int:
    """AGT-7: size the AUTO auto-drive loop's per-turn gate budget off the active
    plan's own gate count (needs_confirmation steps + the plan-overview gate),
    capped by the task's capability tier — instead of the fixed AGENT_MAX_GATES=8
    that silently exhausted on any plan with >=9 gates (the modeling_with_join
    template alone has 7 needs_confirmation steps plus the overview + C1 gates).
    Falls back to AGENT_MAX_GATES when no plan has been built yet (e.g. before the
    first C1 file-role gate) or the plan repo is unavailable, while still honoring
    the selected tier's hard ceiling."""
    tier = resolve_tier(getattr(runtime, "tier", None))
    plan_repo = getattr(runtime, "plan_repo", None)
    plan = _active_plan(plan_repo, task_id) if plan_repo is not None else None
    if plan is None:
        return min(AGENT_MAX_GATES, tier.max_auto_gates)
    gate_count = sum(1 for step in plan.steps if step.needs_confirmation)
    # +1 for the plan-overview gate every driver plan pauses at before running.
    return auto_gate_budget(tier, gate_count + 1)

def _ingest_notice_text(notices: list[dict]) -> str:
    messages = [str(item.get("message") or "").strip() for item in notices]
    messages = [message for message in messages if message]
    if not messages:
        return ""
    return "\n\n已自动处理：\n" + "\n".join(f"- {message}" for message in messages)

def _merge_ingest_notices(*groups) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for notice in group or []:
            key = (str(notice.get("code") or ""), str(notice.get("file") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(notice))
    return merged

