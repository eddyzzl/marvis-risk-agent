"""Dispatch for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from marvis.agent.auto_drive import decide_gate
from marvis.agent.memory_bridge import build_memory_anchor
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_AUTO
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.agent.plan_driver import DriverError
from marvis.agent.semantic_intent import INTENT_ADHOC_CONFIRM
from marvis.agent.semantic_intent import INTENT_ADHOC_QUERY
from marvis.agent.semantic_intent import INTENT_ADHOC_REJECT
from marvis.agent.semantic_intent import INTENT_ADHOC_REVISE
from marvis.agent.semantic_intent import INTENT_CURRENT_WORKFLOW
from marvis.agent.semantic_intent import INTENT_DATASET_ANALYSIS
from marvis.agent.semantic_intent import INTENT_DATASET_EXPORT
from marvis.agent.semantic_intent import INTENT_DATASET_JOIN
from marvis.agent.semantic_intent import INTENT_DATASET_TRANSFORM
from marvis.agent.semantic_intent import INTENT_RISK_PROFITABILITY
from marvis.agent.semantic_intent import INTENT_RISK_STANDARD_VINTAGE
from marvis.agent.semantic_intent import INTENT_RISK_VTG_TERMINAL
from marvis.agent.semantic_intent import INTENT_STRATEGY_SAMPLE_BINDING
from marvis.agent.semantic_intent import INTENT_STRATEGY_WORKFLOW
from marvis.agent.semantic_intent import route_semantic_intent
from marvis.agent.strategy_request_compiler import (
    utterance_targets_candidate_monthly_stability,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_interactive_tree_frontier_group_materialization,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_interactive_tree_frontier_materialization,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_model_score_comparison_v2,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_scorecard_band_build,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_scorecard_cutoff_selection,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_dsl_delivery,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_impact_cube,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_pool_stability,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_project_context,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_report_bundle_v2,
)
from marvis.agent.strategy_request_compiler import (
    utterance_targets_strategy_sample_design,
)
from marvis.agent.workflow_recovery import is_explicit_workflow_retry
from marvis.agent.workflow_recovery import latest_unresolved_workflow_failure
from marvis.agent_memory.api_support import audit_agent_memory_use_from_store
from marvis.agent_memory.store import AgentMemoryStore
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TASK_TYPE_MODELING
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.domain import TaskRecord
from marvis.llm_client import LLMClientError
from marvis.repositories.tasks import TaskRepository
from . import _registry as _registry_lane
from . import adhoc as adhoc_lane
from . import c1_state as c1_state_lane
from . import contracts as contracts_lane
from . import dataset_turns as dataset_turns_lane
from . import feature as feature_lane
from . import labeling as labeling_lane
from . import portfolio as portfolio_lane
from . import recovery as recovery_lane
from . import responses as responses_lane
from . import semantic as semantic_lane
from . import shared as shared_lane
from . import strategy_contracts as strategy_contracts_lane
from . import strategy_request as strategy_request_lane
from . import strategy_sample as strategy_sample_lane
from . import typed_ui as typed_ui_lane


def _safe_semantic_intent_state_snapshot(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
) -> str | None:
    """Return ``None`` when the complete route target cannot be authenticated."""

    try:
        return semantic_lane._semantic_intent_state_snapshot(runtime, repo, task)
    except Exception:  # noqa: BLE001 - an incomplete snapshot must fail closed
        return None


def dispatch_driver_turn(
    runtime: contracts_lane.DriverTurnRuntime,
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
        return labeling_lane._handle_structured_labeling_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            labeling_request=labeling_request,
        )
    if portfolio_request is not None:
        return portfolio_lane._handle_structured_portfolio_request_turn(
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
        return strategy_request_lane._handle_structured_strategy_request_turn(
            runtime,
            repo,
            task,
            user_text=user_text,
            strategy_request=strategy_request,
        )
    if ui_action is None:
        labeling_confirmation = (
            labeling_lane._maybe_handle_labeling_preplan_confirmation_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                confirmation_source=confirmation_source,
            )
        )
        if labeling_confirmation is not None:
            return labeling_confirmation
    # UI controls are already typed, governed commands. Route them directly to
    # the task driver so generic recovery/analysis/strategy text classifiers
    # cannot reinterpret their display copy before optimistic-lock validation.
    if ui_action is not None:
        result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
            return responses_lane.join_turn_response(repo, task.id)
        return result
    # An unresolved structured failure owns ordinary conversation first.  In
    # particular, “为什么策略分析失败” is a question about existing evidence,
    # not authorization to compile and run a new strategy request.
    recovery = recovery_lane._maybe_handle_workflow_recovery_turn(
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
        shared_lane._active_plan(runtime.plan_repo, task.id) is None
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
        result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
            return responses_lane.join_turn_response(repo, task.id)
        return result
    # A reply that names exactly one candidate from the live Feature target
    # card is setup input, not an ad-hoc analysis request.  Keep it ahead of
    # the generic intent helpers both on the first attempt and after a
    # transaction-level setup failure.  The setup handler still performs the
    # full two-pass semantic authorization in Agent mode, so this routing
    # priority cannot itself authorize or persist the choice.
    feature_target_state = (
        feature_lane._latest_feature_target_state(repo.list_agent_messages(task.id))
        if task.task_type == TASK_TYPE_FEATURE_ANALYSIS
        else None
    )
    if (
        feature_target_state is not None
        and c1_state_lane._natural_language_c1_target(text, feature_target_state)
        is not None
    ):
        result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
            return responses_lane.join_turn_response(repo, task.id)
        return result
    conversation = repo.list_agent_messages(task.id)
    if task.task_type == TASK_TYPE_STRATEGY and not (
        utterance_targets_strategy_project_context(text)
        or strategy_request_lane._is_strategy_request_intent(text)
    ):
        project_context_answer = (
            strategy_request_lane._maybe_handle_project_context_missing_answer(
                runtime,
                repo,
                task,
                text=text,
                conversation=conversation,
            )
        )
        if project_context_answer is not None:
            return project_context_answer

    # These legacy pending contracts already have their own deterministic
    # state-bound confirmation parser. Keep them out of top-level routing so
    # the new classifier cannot steal an existing confirmation turn. Ad-hoc
    # pending is intentionally excluded: it is the contract upgraded below.
    has_existing_pending_contract = (
        dataset_turns_lane._latest_pending_transform_protected_drop(conversation)
        is not None
        or (
            task.task_type == TASK_TYPE_STRATEGY
            and (
                strategy_request_lane._latest_strategy_request_pending(conversation)
                is not None
                or strategy_sample_lane._latest_strategy_nan_label_confirmation(
                    conversation
                )
                is not None
            )
        )
    )
    if (
        runtime.require_semantic_text_authorization
        and text.strip()
        and not has_existing_pending_contract
        and shared_lane._active_plan(runtime.plan_repo, task.id) is None
        and shared_lane.latest_open_gate(conversation) is None
        and strategy_request_lane._specialized_workflow_intake_is_pending(
            runtime, task, conversation
        )
    ):
        before_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if before_snapshot is None:
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或数据状态无法完成一致性校验，未创建计划。",
            )
        intake_route, intake_reason = (
            strategy_request_lane._specialized_workflow_intake_route(
                runtime,
                task,
                instruction=text,
            )
        )
        if intake_route is None:
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason=intake_reason,
            )
        after_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if after_snapshot is None or before_snapshot != after_snapshot:
            return semantic_lane._semantic_intent_clarification_response(
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
        result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
            return responses_lane.join_turn_response(repo, task.id)
        return result
    semantic_decision = None
    pending_adhoc = None
    if (
        runtime.require_semantic_text_authorization
        and text.strip()
        and not has_existing_pending_contract
        and shared_lane._active_plan(runtime.plan_repo, task.id) is None
        and shared_lane.latest_open_gate(conversation) is None
    ):
        before_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if before_snapshot is None:
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或数据状态无法完成一致性校验，未执行任何动作。",
                pending_adhoc=adhoc_lane._latest_adhoc_pending(
                    repo.list_agent_messages(task.id)
                ),
            )
        try:
            semantic_context, allowed_intents, pending_adhoc = (
                semantic_lane._semantic_intent_route_contract(runtime, repo, task)
            )
        except Exception:  # noqa: BLE001 - incomplete material context must fail closed
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="当前任务或可用材料无法完成语义路由校验，未执行任何动作。",
                pending_adhoc=adhoc_lane._latest_adhoc_pending(
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
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason=semantic_decision.reason,
                pending_adhoc=pending_adhoc,
            )
        after_snapshot = _safe_semantic_intent_state_snapshot(runtime, repo, task)
        if after_snapshot is None or before_snapshot != after_snapshot:
            return semantic_lane._semantic_intent_clarification_response(
                repo,
                task,
                user_text=text,
                reason="语义复核期间任务状态已变化，旧判断已作废。",
                pending_adhoc=adhoc_lane._latest_adhoc_pending(
                    repo.list_agent_messages(task.id)
                ),
            )
        runtime = replace(runtime, semantic_intent=semantic_decision.intent)

        semantic_handler = {
            INTENT_ADHOC_QUERY: lambda: adhoc_lane._maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                force_intent=True,
            ),
            INTENT_ADHOC_CONFIRM: lambda: adhoc_lane._maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_CONFIRM,
            ),
            INTENT_ADHOC_REJECT: lambda: adhoc_lane._maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_REJECT,
            ),
            INTENT_ADHOC_REVISE: lambda: adhoc_lane._maybe_handle_adhoc_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
                pending_decision=INTENT_ADHOC_REVISE,
            ),
            INTENT_DATASET_TRANSFORM: lambda: (
                dataset_turns_lane._maybe_handle_dataset_transform_turn(
                    runtime,
                    repo,
                    task,
                    user_text=user_text,
                    force_intent=True,
                )
            ),
            INTENT_DATASET_EXPORT: lambda: (
                dataset_turns_lane._maybe_handle_dataset_export_turn(
                    runtime,
                    repo,
                    task,
                    user_text=user_text,
                    force_intent=True,
                )
            ),
            INTENT_DATASET_ANALYSIS: lambda: (
                dataset_turns_lane._maybe_handle_dataset_analysis_turn(
                    runtime,
                    repo,
                    task,
                    user_text=user_text,
                    force_intent=True,
                )
            ),
            INTENT_STRATEGY_SAMPLE_BINDING: lambda: (
                strategy_request_lane._handle_strategy_sample_binding_intent(
                    runtime,
                    repo,
                    task,
                    user_text=text,
                )
            ),
            INTENT_STRATEGY_WORKFLOW: lambda: (
                strategy_request_lane._maybe_handle_strategy_request_turn(
                    runtime,
                    repo,
                    task,
                    user_text=user_text,
                    force_intent=True,
                )
            ),
        }.get(semantic_decision.intent)
        if semantic_handler is not None:
            semantic_result = semantic_handler()
            if semantic_result is not None:
                return semantic_result
            return semantic_lane._semantic_intent_clarification_response(
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
            result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
                return responses_lane.join_turn_response(repo, task.id)
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
        strategy_sample_design_request = (
            strategy_request_lane._maybe_handle_strategy_request_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
            )
        )
        if strategy_sample_design_request is not None:
            return strategy_sample_design_request
    # Dataset changes get first refusal over descriptive analysis.  Phrases
    # such as "填充缺失值" describe a governed mutation, not a request for a
    # missing-value report; the transform always creates an immutable child.
    dataset_transform = dataset_turns_lane._maybe_handle_dataset_transform_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if dataset_transform is not None:
        return dataset_transform
    dataset_export = dataset_turns_lane._maybe_handle_dataset_export_turn(
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
        or utterance_targets_interactive_tree_frontier_group_materialization(text)
        or utterance_targets_interactive_tree_frontier_materialization(text)
        or utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
        or utterance_targets_strategy_sample_design(text)
        or utterance_targets_strategy_dsl_delivery(text)
        or utterance_targets_strategy_report_bundle_v2(text)
        or utterance_targets_strategy_impact_cube(text)
        or utterance_targets_strategy_pool_stability(text)
        or strategy_contracts_lane._STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE.search(text)
        is not None
    ):
        strategy_evidence_request = (
            strategy_request_lane._maybe_handle_strategy_request_turn(
                runtime,
                repo,
                task,
                user_text=user_text,
            )
        )
        if strategy_evidence_request is not None:
            return strategy_evidence_request
    # Explicit dataset diagnostics are narrower than the strategy compiler's
    # generic "分析" operation. Give this branch first refusal so phrases such
    # as "分析当前样本" cannot be mistaken for a request to design a strategy.
    dataset_analysis = dataset_turns_lane._maybe_handle_dataset_analysis_turn(
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
    strategy_request = strategy_request_lane._maybe_handle_strategy_request_turn(
        runtime,
        repo,
        task,
        user_text=user_text,
    )
    if strategy_request is not None:
        return strategy_request
    adhoc = adhoc_lane._maybe_handle_adhoc_turn(
        runtime, repo, task, user_text=user_text
    )
    if adhoc is not None:
        return adhoc
    result = _registry_lane.DRIVER_TURN_FUNCS[task.task_type](
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
        return responses_lane.join_turn_response(repo, task.id)
    return result


def agent_autodrive_turn(
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    client,
) -> None:
    turn_fn = _registry_lane.DRIVER_TURN_FUNCS[task.task_type]
    max_gates = shared_lane._auto_gate_budget(runtime, task.id)
    processed_gates = 0
    while True:
        # A turn can start at the pre-plan C1 gate and create its real plan only
        # after the first AUTO decision.  Re-read the bounded tier budget before
        # each subsequent gate so that the newly known plan depth is honored;
        # otherwise the one-time eight-gate fallback can stop immediately before
        # the mandatory human gate it was meant to reach.
        max_gates = max(max_gates, shared_lane._auto_gate_budget(runtime, task.id))
        if processed_gates >= max_gates:
            break
        gate = shared_lane.latest_open_gate(repo.list_agent_messages(task.id))
        if gate is None:
            return
        c1 = (gate.get("metadata") or {}).get("join_c1")
        if (
            task.task_type in {TASK_TYPE_MODELING, TASK_TYPE_FEATURE_ANALYSIS}
            and isinstance(c1, dict)
            and not c1.get("target_col")
        ):
            anchor = next(
                (item for item in c1.get("files", [])
                 if item.get("dataset_id") == c1.get("anchor_id")),
                {},
            )
            if len(anchor.get("target_candidates") or []) > 1:
                # A schema ambiguity is a missing business choice, not a low-risk
                # approval. Preserve the actionable C1 prompt; a bare AUTO
                # confirm cannot select a target and only appends a false error.
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
            content=typed_ui_lane._auto_decision_content(decision),
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
            driver = shared_lane._driver(runtime)
            try:
                turn = driver.replan_structured(
                    plan_id=plan_id,
                    goal=goal,
                    expected_step_id=gate_step_id,
                    confirmation_source=CONFIRMATION_SOURCE_AUTO,
                )
            except DriverError:
                return
            shared_lane.append_driver_messages(repo, task, turn, runtime=runtime)
            continue
        return
    # AGT-7: the budget ran out with a gate STILL open (every iteration matched a
    # real gate and looped back via confirm/adjust/replan) — tell the user
    # explicitly instead of silently going quiet, which previously looked like
    # the agent had inexplicably stopped responding.
    if shared_lane.latest_open_gate(repo.list_agent_messages(task.id)) is not None:
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
