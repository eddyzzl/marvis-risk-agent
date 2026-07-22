from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable

from marvis.agent.adhoc_analysis import (
    build_slice_spec_from_utterance,
    detect_question_intent,
)
from marvis.agent.auto_drive import decide_gate
from marvis.agent.dataset_analysis import (
    build_dataset_analysis_request,
    detect_dataset_analysis_intent,
)
from marvis.agent.dataset_export import (
    build_dataset_export_request,
    detect_dataset_export_intent,
)
from marvis.agent.dataset_transform import (
    build_dataset_transform_request,
    detect_dataset_transform_intent,
)
from marvis.agent.feature_setup import FeatureSetupError, build_feature_proposal
from marvis.agent.join_setup import JoinSetupError, build_join_proposal
from marvis.agent.memory_bridge import (
    build_memory_anchor,
    capture_agent_memory_for_driver_done,
    fetch_field_convention_hints,
)
from marvis.agent.modeling_setup import ModelingSetupError, build_modeling_proposal
from marvis.agent.plan_driver import (
    CONFIRMATION_SOURCE_AUTO,
    CONFIRMATION_SOURCE_HUMAN,
    DriverError,
    PlanDriver,
    is_confirm,
)
from marvis.agent.portfolio_setup import (
    PortfolioProposal,
    PortfolioSetupError,
    build_portfolio_proposal,
    build_states_gate_state,
    parse_states_reply,
)
from marvis.agent.strategy_setup import (
    STRATEGY_INTENT_FULL_DEVELOPMENT,
    STRATEGY_INTENT_LIMIT_PRICING,
    STRATEGY_INTENT_MONITORING,
    STRATEGY_INTENT_PORTFOLIO_ANALYSIS,
    STRATEGY_INTENT_QUICK_ANALYSIS,
    STRATEGY_INTENT_RULE_MINING,
    STRATEGY_INTENT_STANDARD_ANALYSIS,
    StrategySetupError,
    build_monitoring_setup_proposal,
    build_rule_strategy_proposal,
    build_strategy_dataset_context,
    build_strategy_development_proposal,
    build_strategy_proposal,
    preview_strategy_dataset_context,
    resolve_strategy_intent,
    strategy_development_clarification,
)
from marvis.agent.strategy_request_compiler import (
    CompiledStrategyRequestDraft,
    StandardWorkflowRequestDraft,
    StrategyRequestDraft,
    compile_strategy_request,
    utterance_targets_strategy_sample_design,
    validate_strategy_request,
)
from marvis.agent.vintage_setup import VintageSetupError, build_vintage_proposal
from marvis.agent.workflow_error_diagnostics import (
    build_workflow_error_diagnostic,
    failure_envelope_for_diagnostic,
    workflow_error_content,
)
from marvis.agent.workflow_recovery import (
    deterministic_workflow_recovery_reply,
    is_explicit_workflow_retry,
    latest_unresolved_workflow_failure,
)
from marvis.agent_memory.api_support import audit_agent_memory_use_from_store
from marvis.agent_memory.store import AgentMemoryStore
from marvis.data.backend import DataBackend
from marvis.data.labels import nan_label_mask
from marvis.data.registry import DatasetRegistry
from marvis.data.transform_semantics import effective_transform_semantic_mapping
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, StrategyRepository, TaskRepository
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_PORTFOLIO,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    StrategyProfitInput,
    StrategyTaskInput,
    TaskRecord,
)
from marvis.strategy_lifecycle import ASSET_STATUS_ADOPTED_LOCAL
from marvis.files import sha256_file
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.orchestrator.capability import auto_gate_budget, resolve_tier
from marvis.orchestrator.contracts import Plan, PlanStatus, StepStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.validator import PlanValidator
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
)
from marvis.packs.strategy.automatic_tree_leaf_tools import (
    load_verified_automatic_tree_leaf_selection_artifact_on_connection,
    load_verified_automatic_tree_source_artifact_on_connection,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ORIGIN_TOOL,
)
from marvis.packs.strategy.voting_candidate_tools import (
    load_verified_voting_candidate_artifact_on_connection,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    ASSET_ARTIFACT_KIND as CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
    ASSET_ARTIFACT_SCHEMA_VERSION as CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION,
    ORIGIN_TOOL as CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
)
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION,
    CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL,
)
from marvis.packs.strategy.cross_matrix_cell_selection_tools import (
    load_verified_cross_matrix_cell_selection_artifact_on_connection,
    load_verified_cross_matrix_source_artifact_on_connection,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignRef,
    load_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestConflictError,
    PendingStrategyRequestNotFoundError,
    PendingStrategyRequestRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    StrategyCandidatePoolRepository,
    strategy_pool_snapshot_hash,
)
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    validate_candidate_asset,
)
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
    DataWorkspaceRevisionConflict,
)
from marvis.settings import Settings


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

# AGT-7: retained as the floor/fallback when a plan's gate count can't be
# determined yet (e.g. before the first C1 file-role gate builds the real
# plan). The effective per-turn budget is dynamic — see
# marvis.orchestrator.capability.auto_gate_budget.
AGENT_MAX_GATES = 8
_TERMINAL_PLAN_STATUS_VALUES = frozenset({"done", "failed", "cancelled"})
_STRATEGY_SAMPLE_BOUND_TOOLS = frozenset(
    {
        "analyze_univariate_candidates",
        "backtest_strategy",
        "build_automatic_tree_candidate",
        "compare_strategies",
        "design_cutoff_bands",
        "design_strategy_candidate",
        "evaluate_rule_set",
        "limit_pricing_matrix",
        "measure_pool_impact",
        "mine_rules",
        "tradeoff_view",
    }
)


@dataclass(frozen=True)
class DriverTurnRuntime:
    settings: Settings
    plan_repo: PlanRepository
    plan_executor: PlanExecutor
    planner: Planner
    plan_validator: PlanValidator
    llm_client: OpenAICompatibleLLMClient | None
    tier: str
    governance_service: object | None = None
    local_principal: object | None = None
    recovery_responder: Callable[..., tuple[str, dict]] | None = None


# ARCH-4: the five run_*_driver_turn entry points below share one skeleton
# (log the user turn -> resume an active plan OR run type-specific setup and
# driver.start -> map setup errors to a chat message). _TurnHandlerSpec pins
# down every axis the five types actually differ on so that skeleton can live
# once in _run_driver_turn while each per-type "shell" stays a one-line call.
# Each axis below is copied verbatim from the pre-refactor function bodies —
# see the commit message for a couple of cross-type inconsistencies spotted
# along the way but deliberately left unchanged.
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
        [DriverTurnRuntime, TaskRepository, TaskRecord, str | None], dict | tuple
    ]
    # join/modeling display "已确认文件角色与目标列。" instead of the raw
    # [C1]-prefixed payload text when logging the user turn; the other three
    # types always log user_text verbatim.
    format_user_display: Callable[[str], str]
    # join/modeling pass settings=/task= into append_driver_messages (so a
    # terminal "done" message can trigger MEM-1 memory capture). S2: strategy
    # now also passes them (strategy_experience capture on adoption); feature/
    # vintage still don't have an extractor wired, but ARCH-4 found they were
    # never passed kwargs at all -- fixed alongside strategy so all five types
    # are parameterized the same way instead of silently diverging.
    pass_memory_kwargs: bool
    # Optional per-type success_criteria builder threaded into start_kwargs
    # (mirrors _modeling_success_criteria); None means this type never injects
    # a deterministic criterion.
    success_criteria: Callable[[TaskRecord], list[dict] | None] | None = None


def run_join_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict:
    return _run_driver_turn(
        _JOIN_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        confirmation_source=confirmation_source,
    )


def run_feature_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict:
    return _run_driver_turn(
        _FEATURE_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        confirmation_source=confirmation_source,
    )


def run_strategy_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict:
    return _run_driver_turn(
        _STRATEGY_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        confirmation_source=confirmation_source,
    )


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
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
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
        confirmation_source=confirmation_source,
    )


def run_portfolio_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict:
    return _run_driver_turn(
        _PORTFOLIO_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        confirmation_source=confirmation_source,
    )


def run_modeling_driver_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None = None,
    dedup_strategies: dict | None = None,
    adjust_params: dict | None = None,
    expected_step_id: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
) -> dict:
    return _run_driver_turn(
        _MODELING_SPEC,
        runtime,
        repo,
        task,
        user_text=user_text,
        selection=selection,
        dedup_strategies=dedup_strategies,
        adjust_params=adjust_params,
        expected_step_id=expected_step_id,
        confirmation_source=confirmation_source,
    )


def _run_driver_turn(
    spec: _TurnHandlerSpec,
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    selection: list | None,
    dedup_strategies: dict | None,
    adjust_params: dict | None,
    expected_step_id: str | None,
    confirmation_source: str,
) -> dict:
    if user_text is not None:
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=spec.format_user_display(user_text),
            metadata={"intent": spec.intent},
        )
    try:
        active = _active_plan(runtime.plan_repo, task.id)
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
            turn = driver.resume(
                plan_id=active.id,
                user_text=user_text or "",
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
                expected_step_id=expected_step_id,
                confirmation_source=confirmation_source,
            )
            _append_spec_messages(spec, repo, task, turn, runtime)
            return join_turn_response(repo, task.id)
        setup_result = spec.run_setup(runtime, repo, task, user_text)
        if isinstance(setup_result, dict):
            return setup_result
        template_id, slots, start_kwargs = setup_result
        if spec.success_criteria is not None and "success_criteria" not in start_kwargs:
            criteria = spec.success_criteria(task)
            if criteria is not None:
                start_kwargs = {**start_kwargs, "success_criteria": criteria}
        driver = _driver(runtime)
        turn = driver.start(
            task_id=task.id,
            template_id=template_id,
            slots=slots,
            tier=runtime.tier,
            **start_kwargs,
        )
        _append_spec_messages(spec, repo, task, turn, runtime)
        return join_turn_response(repo, task.id)
    except spec.setup_error_types as exc:
        return append_workflow_error(repo, task, spec, exc, setup_error=True)
    except DriverError:
        raise
    except Exception as exc:
        return append_workflow_error(repo, task, spec, exc)


def _terminate_stale_strategy_sample_plan(
    spec: _TurnHandlerSpec,
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    plan: Plan,
) -> dict | None:
    """Fail closed before resuming pre-sample-binding strategy plans.

    V2 plans are serialized, so a plan created before sample-design binding can
    outlive a deployment and otherwise resume directly into a data-reading Tool.
    Only unfinished sample-bound steps are migration-sensitive: completed
    historical evidence remains readable, while newly compiled plans carry an
    exact authenticated development-partition reference and resume unchanged.
    """

    if spec.intent != "strategy":
        return None
    stale_steps = _stale_strategy_sample_steps(plan)
    if not stale_steps:
        return None

    current_status = PlanStatus(getattr(plan.status, "value", plan.status))
    if current_status in {
        PlanStatus.DRAFT,
        PlanStatus.VALIDATED,
        PlanStatus.RUNNING,
        PlanStatus.REVIEW,
    }:
        terminal_status = PlanStatus.FAILED
    elif current_status in {
        PlanStatus.CONFIRMED,
        PlanStatus.AWAITING_CONFIRM,
    }:
        # These states cannot legally transition directly to FAILED. CANCELLED
        # is their governed terminal path and prevents any Tool invocation.
        terminal_status = PlanStatus.CANCELLED
    else:
        return None

    runtime.plan_repo.set_plan_status(plan.id, terminal_status)
    stale_step_payload = [
        {
            "step_id": step.id,
            "tool": step.tool_ref.tool,
            "step_status": getattr(step.status, "value", step.status),
        }
        for step in stale_steps
    ]
    runtime.plan_repo.write_audit(
        kind="strategy.plan.sample_design_stale",
        target_ref=plan.id,
        outcome="blocked",
        detail={
            "task_id": task.id,
            "template_id": plan.template_id,
            "from_status": current_status.value,
            "to_status": terminal_status.value,
            "clarification_code": "strategy_plan_sample_design_stale",
            "stale_steps": stale_step_payload,
        },
    )
    message = (
        "该策略计划由旧版本创建，未完成的数据分析或回测步骤没有绑定当前成熟样本设计的"
        "精确 sample_design_ref。平台已安全终止旧计划，且没有调用任何分析工具。"
        "请先确认当前成熟样本设计，再基于该设计重新发起策略请求；平台会重建计划。"
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "strategy",
            "kind": "clarification",
            "code": "strategy_plan_sample_design_stale",
            "plan_id": plan.id,
            "template_id": plan.template_id,
            "plan_status": terminal_status.value,
            "stale_steps": stale_step_payload,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": "strategy_plan_sample_design_stale",
        "plan_id": plan.id,
        "plan_status": terminal_status.value,
        "messages": repo.list_agent_messages(task.id),
    }


def _stale_strategy_sample_steps(plan: Plan) -> list:
    status = getattr(plan.status, "value", plan.status)
    if status in _TERMINAL_PLAN_STATUS_VALUES:
        return []
    stale_steps = []
    for step in plan.steps:
        step_status = getattr(step.status, "value", step.status)
        if step_status in {StepStatus.DONE.value, StepStatus.SKIPPED.value}:
            continue
        if (
            step.tool_ref.plugin != "strategy"
            or step.tool_ref.tool not in _STRATEGY_SAMPLE_BOUND_TOOLS
        ):
            continue
        try:
            StrategySampleDesignRef.from_value(
                step.inputs.get("sample_design_ref")
            )
        except StrategyError:
            stale_steps.append(step)
    return stale_steps


def _append_spec_messages(
    spec: _TurnHandlerSpec,
    repo: TaskRepository,
    task: TaskRecord,
    turn,
    runtime: DriverTurnRuntime,
) -> None:
    if spec.pass_memory_kwargs:
        append_driver_messages(
            repo, task.id, turn, settings=runtime.settings, task=task
        )
    else:
        append_driver_messages(repo, task.id, turn)


def _c1_display_text(user_text: str) -> str:
    return "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text


def _identity_display_text(user_text: str) -> str:
    return user_text


def _run_join_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
) -> dict | tuple:
    conversation = repo.list_agent_messages(task.id)
    c1_state = _latest_c1_state(conversation)
    _, registry = _modeling_data_runtime(runtime.settings)
    if c1_state is None:
        proposal = build_join_proposal(registry, task.id, task.source_dir)
        _append_c1_message(repo, task.id, proposal)
        return join_turn_response(repo, task.id)
    assignment = _parse_c1_reply(user_text, c1_state)
    if assignment is None:
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="请确认文件角色与目标列:无误就回复「确认」，或用下方控件调整后点「确认角色」。",
            metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
        )
        return join_turn_response(repo, task.id)
    if not assignment["anchor_id"]:
        return append_join_error(
            repo, task.id, "请先指定样本锚表（通常是含目标列的那张），再确认。"
        )
    if not assignment["feature_ids"]:
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="已确认样本表与目标列。只有一张表，无需拼接（数据拼接阶段已跳过）。",
            metadata={"join_skip": True},
        )
        return join_turn_response(repo, task.id)
    return (
        "data_join",
        {
            "anchor_id": assignment["anchor_id"],
            "feature_ids": assignment["feature_ids"],
        },
        {},
    )


def _run_feature_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    proposal = build_feature_proposal(
        registry, backend, task.id, task.source_dir, metrics=_feature_metrics(task)
    )
    notices = list(proposal.ingest_notices or [])
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"分析数据集 `{proposal.dataset_name}`（目标列 `{proposal.target_col}`，"
            f"{len(proposal.features)} 个候选特征）:"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={"intent": "feature_analysis", "ingest_notices": notices},
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _strategy_success_criteria(task: TaskRecord) -> list[dict] | None:
    """Turn the governed strategy contract into deterministic final-review limits."""
    strategy_input = getattr(task, "strategy_input", None)
    if isinstance(strategy_input, dict):
        bad_rate_max = strategy_input.get("max_bad_rate")
        approval_min = strategy_input.get("min_approval_rate")
    else:
        bad_rate_max = getattr(strategy_input, "max_bad_rate", None)
        approval_min = getattr(strategy_input, "min_approval_rate", None)
    # Compatibility for tasks/tests created before StrategyTaskInput existed.
    if bad_rate_max is None:
        bad_rate_max = getattr(task, "strategy_bad_rate_max", None)
    if approval_min is None:
        approval_min = getattr(task, "strategy_approval_min", None)
    criteria: list[dict] = []
    if bad_rate_max is not None:
        criteria.append({"metric": "approved_bad_rate", "max": float(bad_rate_max)})
    if approval_min is not None:
        criteria.append({"metric": "approval_rate", "min": float(approval_min)})
    return criteria or None


def _run_strategy_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    *,
    forced_intent: str | None = None,
) -> dict | tuple:
    strategy_input = getattr(task, "strategy_input", None)
    intent = forced_intent or resolve_strategy_intent(
        strategy_input, user_text, getattr(task, "model_name", None)
    )
    if intent in {
        STRATEGY_INTENT_LIMIT_PRICING,
        STRATEGY_INTENT_PORTFOLIO_ANALYSIS,
        STRATEGY_INTENT_STANDARD_ANALYSIS,
    }:
        return _strategy_intent_redirect_response(repo, task, intent)

    backend, registry = _modeling_data_runtime(runtime.settings)
    if intent == STRATEGY_INTENT_MONITORING:
        return _run_strategy_monitoring_setup(runtime, repo, task, backend, registry)
    raw_strategy_type = (
        getattr(strategy_input, "strategy_type", None)
        if not isinstance(strategy_input, dict)
        else strategy_input.get("strategy_type")
    )
    strategy_type = str(raw_strategy_type or "approval").strip().lower()
    if strategy_type not in {"approval", "reject"}:
        return _strategy_clarification_response(
            repo,
            task,
            {
                "code": "strategy_typed_spec_required",
                "entry_mode": (
                    getattr(strategy_input, "entry_mode", None)
                    if not isinstance(strategy_input, dict)
                    else strategy_input.get("entry_mode")
                )
                or "strategy_development",
                "strategy_type": strategy_type,
                "missing_fields": ["strategy_spec"],
                "message": (
                    f"{strategy_type} 策略不能套用准入 cutoff 工作流；"
                    "需要先由自然语言请求编译并确认类型化 Strategy DSL。"
                ),
            },
        )
    if intent == STRATEGY_INTENT_RULE_MINING:
        return _run_rule_strategy_setup(runtime, repo, task, backend, registry)
    if intent == STRATEGY_INTENT_FULL_DEVELOPMENT:
        clarification = strategy_development_clarification(strategy_input)
        if clarification is not None:
            return _strategy_clarification_response(repo, task, clarification)
        proposal = build_strategy_development_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            strategy_input=strategy_input,
            target_col=getattr(task, "target_col", "") or None,
            score_col=getattr(task, "score_col", "") or None,
        )
        notices = registry.consume_ingest_notices(task.id)
        note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
        bad = (
            f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
        )
        constraints = []
        if proposal.max_bad_rate is not None:
            constraints.append(f"通过客群坏率 ≤ {proposal.max_bad_rate:.2%}")
        if proposal.min_approval_rate is not None:
            constraints.append(f"通过率 ≥ {proposal.min_approval_rate:.2%}")
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始完整策略开发:样本 `{proposal.dataset_name}`，目标列 "
                f"`{proposal.target_col}`{bad}，评分列 `{proposal.score_col}`。"
                f"经营目标 `{proposal.objective}`，约束 {'；'.join(constraints)}。"
                "尚未生成 cutoff 或默认规则；计划启动后会自动扫描、构造和回测可行方案，"
                "仅在采纳时交由人工决策。"
                f"{note_text}{_ingest_notice_text(notices)}"
            ),
            metadata={
                "intent": STRATEGY_INTENT_FULL_DEVELOPMENT,
                "ingest_notices": notices,
            },
        )
        slots = proposal.template_slots()
        context = _strategy_dataset_context(runtime, task, require_target=True)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=False,
        )
        return (proposal.template_id, slots, {})

    if intent != STRATEGY_INTENT_QUICK_ANALYSIS:
        raise StrategySetupError(f"unsupported strategy intent: {intent}")
    proposal = build_strategy_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        score_col=getattr(task, "score_col", "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略分析:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"评分列 `{proposal.score_col}`。已生成默认审批策略候选，将自动完成回测和分析。"
            f"{note_text}{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_QUICK_ANALYSIS,
            "ingest_notices": notices,
        },
    )
    slots = proposal.template_slots()
    context = _strategy_dataset_context(runtime, task, require_target=True)
    slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
        runtime,
        task,
        context=context,
        drop_nan_labels=False,
    )
    return (proposal.template_id, slots, {})


def _strategy_intent_redirect_response(
    repo: TaskRepository, task: TaskRecord, intent: str
) -> dict:
    if intent == STRATEGY_INTENT_LIMIT_PRICING:
        detail = {
            "intent": intent,
            "code": "strategy_standard_workflow_inputs_required",
            "available_workflow": "limit_pricing_matrix",
            "message": (
                "已识别为额度定价矩阵。该标准 Workflow 已可执行；请用自然语言补充评分列、"
                "PD/目标列、分箱、额度与利率网格及经济参数，Agent 会编译并回显后再运行。"
            ),
        }
    elif intent == STRATEGY_INTENT_STANDARD_ANALYSIS:
        detail = {
            "intent": intent,
            "code": "strategy_standard_workflow_inputs_required",
            "available_workflows": ["profit_calc", "roll_rate_matrix"],
            "message": (
                "已识别为独立策略分析，不会降级成审批策略开发。请用自然语言补充分析列、"
                "状态顺序或利润经济口径；Agent 会选择标准 Workflow、回显口径并请求确认。"
            ),
        }
    elif intent == STRATEGY_INTENT_PORTFOLIO_ANALYSIS:
        detail = {
            "intent": intent,
            "code": "strategy_portfolio_task_redirect",
            "suggested_task_type": TASK_TYPE_PORTFOLIO,
            "message": (
                "已识别为组合分析意图。组合分析属于 V2 的独立 portfolio 任务线；"
                "请创建或切换到组合分析任务，当前策略任务不会误建 approval plan。"
            ),
        }
    else:
        raise StrategySetupError(f"unsupported strategy redirect intent: {intent}")

    redirect_fields = {
        key: value
        for key, value in detail.items()
        if key
        in {
            "available_workflow",
            "available_workflows",
            "suggested_task_type",
        }
    }
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=detail["message"],
        metadata={
            "intent": intent,
            "kind": "clarification",
            "code": detail["code"],
            **redirect_fields,
            "clarification": dict(detail),
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "intent": intent,
        "code": detail["code"],
        **redirect_fields,
        "clarification": dict(detail),
        "messages": repo.list_agent_messages(task.id),
    }


def _strategy_clarification_response(
    repo: TaskRepository, task: TaskRecord, clarification: dict
) -> dict:
    current_input = _strategy_input_snapshot(getattr(task, "strategy_input", None))
    clarification_payload = {
        **dict(clarification),
        "current_input": current_input,
    }
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"{clarification['message']} 缺少："
            + "、".join(f"`{field}`" for field in clarification["missing_fields"])
            + "。请补充后再开始策略开发；如只需技术预览，请明确选择“快速策略分析”。"
        ),
        metadata={
            "intent": "strategy_clarification",
            "kind": "clarification",
            "current_input": current_input,
            "clarification": clarification_payload,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "current_input": current_input,
        "clarification": clarification_payload,
        "messages": repo.list_agent_messages(task.id),
    }


def _strategy_input_snapshot(strategy_input) -> dict | None:
    """Return only the governed strategy contract for clarification prefill."""

    if strategy_input is None:
        return None
    if not isinstance(strategy_input, dict):
        return asdict(strategy_input)

    allowed = (
        "entry_mode",
        "strategy_type",
        "objective",
        "max_bad_rate",
        "min_approval_rate",
        "baseline_strategy_id",
        "profit",
    )
    payload = {key: strategy_input[key] for key in allowed if key in strategy_input}
    profit = payload.get("profit")
    if profit is not None and not isinstance(profit, dict):
        payload["profit"] = asdict(profit)
    return payload


def _run_rule_strategy_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    backend,
    registry,
) -> dict | tuple:
    proposal = build_rule_strategy_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        score_col=getattr(task, "score_col", "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始规则策略挖掘:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}。"
            f"将自动挖掘、选择、评估并回测候选拒绝规则，仅在采纳时交由人工决策。{note_text}"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_RULE_MINING,
            "ingest_notices": notices,
        },
    )
    slots = proposal.template_slots()
    context = _strategy_dataset_context(runtime, task, require_target=True)
    slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
        runtime,
        task,
        context=context,
        drop_nan_labels=False,
    )
    return (proposal.template_id, slots, {})


def _run_strategy_monitoring_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    backend,
    registry,
) -> dict | tuple:
    proposal = build_monitoring_setup_proposal(
        registry,
        backend,
        runtime.settings.db_path,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        score_col=getattr(task, "score_col", "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略监控:对本地已采纳策略 `{proposal.strategy_id}` 跑一次监控,样本 "
            f"`{proposal.dataset_name}`。监控自动完成后会在告警处置门交由人工决策。{note_text}"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_MONITORING,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _run_vintage_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    proposal = build_vintage_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        time_col=getattr(task, "time_col", "") or None,
    )
    notices = registry.consume_ingest_notices(task.id)
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始 Vintage 风险分析:样本 `{proposal.dataset_name}`，"
            f"cohort `{proposal.cohort_col}`，MOB `{proposal.mob_col}`，坏账列 `{proposal.bad_col}`。"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={"intent": "vintage", "ingest_notices": notices},
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _portfolio_success_criteria(task: TaskRecord) -> list[dict] | None:
    """S3: optional deterministic criterion mirroring _strategy_success_criteria.
    task's optional portfolio_el_max (getattr-based -- no schema migration backs
    it) becomes a total_el ceiling final_review can evaluate; absent -> no
    criterion injected (same graceful default as strategy/modeling)."""
    el_max = getattr(task, "portfolio_el_max", None)
    if el_max is None:
        return None
    return [{"metric": "total_el", "max": float(el_max)}]


def _latest_portfolio_states(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        if message.get("role") != "assistant":
            continue
        meta = message.get("metadata") or {}
        if "portfolio_states" in meta:
            return meta["portfolio_states"]
    return None


def _run_portfolio_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    conversation = repo.list_agent_messages(task.id)
    gate_state = _latest_portfolio_states(conversation)
    if gate_state is None:
        proposal = build_portfolio_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            segment_col=getattr(task, "segment_col", "") or None,
            score_col=getattr(task, "score_col", "") or None,
            experiment_id=getattr(task, "experiment_id", "") or None,
        )
        notices = registry.consume_ingest_notices(task.id)
        states_text = " → ".join(f"`{state}`" for state in proposal.proposed_states)
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始组合分析:表现期表 `{proposal.dataset_name}`，贷款id `{proposal.id_col}`，"
                f"快照月 `{proposal.snapshot_col}`，逾期桶 `{proposal.bucket_col}`。\n"
                f"我按恶化程度排的桶顺序（由好到坏）：{states_text}。\n"
                "**桶的语义顺序机器不可猜，必须你确认**：无误回复「确认」；要改就按由好到坏顺序"
                "重列所有桶（逗号分隔）。"
                f"{_ingest_notice_text(notices)}"
            ),
            metadata={
                "portfolio_states": build_states_gate_state(proposal),
                "kind": "gate",
                "ingest_notices": notices,
            },
        )
        return join_turn_response(repo, task.id)

    states = parse_states_reply(user_text, gate_state)
    if states is None:
        proposed = gate_state.get("proposed_states") or []
        states_text = " → ".join(f"`{state}`" for state in proposed)
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                "还没确认桶顺序。默认（由好到坏）："
                f"{states_text}。无误回复「确认」，或按由好到坏重列所有桶（逗号分隔）。"
            ),
            metadata={"portfolio_states": gate_state, "kind": "gate"},
        )
        return join_turn_response(repo, task.id)

    proposal = PortfolioProposal(
        dataset_id=gate_state["dataset_id"],
        dataset_name="",
        id_col=gate_state["id_col"],
        snapshot_col=gate_state["snapshot_col"],
        bucket_col=gate_state["bucket_col"],
        proposed_states=list(states),
        balance_col=gate_state.get("balance_col"),
        segment_col=gate_state.get("segment_col"),
        score_col=gate_state.get("score_col"),
        experiment_id=gate_state.get("experiment_id"),
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=f"已确认桶顺序：{' → '.join(states)}。开始并行分析（流量/迁徙/细分"
        + ("/趋势" if proposal.experiment_id else "")
        + "），随后汇总确认。",
        metadata={"intent": "portfolio"},
    )
    return (proposal.template_id, proposal.template_slots(states), {})


def _run_modeling_setup(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    conversation = repo.list_agent_messages(task.id)
    c1_state = _latest_c1_state(conversation)
    c1_assignment = None
    c1_proposal = build_join_proposal(registry, task.id, task.source_dir)
    c1_ingest_notices = list(c1_proposal.ingest_notices or [])
    if not c1_proposal.skip:
        if c1_state is None:
            _append_c1_message(repo, task.id, c1_proposal)
            return join_turn_response(repo, task.id)
        c1_assignment = _parse_c1_reply(user_text, c1_state)
        if c1_assignment is None:
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
                content="请先确认建模文件角色与目标列:无误就回复「确认」，或用下方控件调整后点「确认角色」。",
                metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
            )
            return join_turn_response(repo, task.id)
        if not c1_assignment["anchor_id"]:
            return append_join_error(
                repo, task.id, "请先指定建模样本主表（通常是含目标列的那张），再确认。"
            )
    proposal = build_modeling_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_type=_modeling_target_type(task),
        recipes=_modeling_recipes(task),
        sample_weight_col=getattr(task, "sample_weight_col", "") or None,
        time_col=getattr(task, "time_col", "") or None,
        anchor_id=(c1_assignment or {}).get("anchor_id"),
        join_feature_ids=(c1_assignment or {}).get("feature_ids"),
        target_col=(c1_assignment or {}).get("target_col"),
        field_hints=fetch_field_convention_hints(
            runtime.settings,
            keywords=_modeling_field_hint_keywords(task, c1_proposal),
        ),
    )
    counts = proposal.counts
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    notices = _merge_ingest_notices(c1_ingest_notices, proposal.ingest_notices)
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始建模:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"切分 `{proposal.split_col}` train/test/oot="
            f"{counts.get('train', 0)}/{counts.get('test', 0)}/{counts.get('oot', 0)}，"
            f"候选特征 {len(proposal.feature_cols)} 个。先做泄漏感知特征筛选，随后请确认特征集。"
            f"{note_text}{_ingest_notice_text(notices)}"
        ),
        metadata={"intent": "modeling", "ingest_notices": notices},
    )
    slots = proposal.template_slots()
    slots.setdefault("project_meta", _modeling_project_meta(task))
    return (
        proposal.template_id,
        slots,
        {"success_criteria": _modeling_success_criteria(task)},
    )


_JOIN_SPEC = _TurnHandlerSpec(
    intent="data_join",
    setup_error_types=(JoinSetupError,),
    error_label="数据拼接出错",
    run_setup=_run_join_setup,
    format_user_display=_c1_display_text,
    pass_memory_kwargs=True,
)

_FEATURE_SPEC = _TurnHandlerSpec(
    intent="feature_analysis",
    setup_error_types=(FeatureSetupError,),
    error_label="特征分析出错",
    run_setup=_run_feature_setup,
    format_user_display=_identity_display_text,
    pass_memory_kwargs=True,
)

_STRATEGY_SPEC = _TurnHandlerSpec(
    intent="strategy",
    setup_error_types=(StrategySetupError,),
    error_label="策略分析出错",
    run_setup=_run_strategy_setup,
    format_user_display=_identity_display_text,
    pass_memory_kwargs=True,
    success_criteria=_strategy_success_criteria,
)

_VINTAGE_SPEC = _TurnHandlerSpec(
    intent="vintage",
    setup_error_types=(VintageSetupError,),
    error_label="Vintage 风险分析出错",
    run_setup=_run_vintage_setup,
    format_user_display=_identity_display_text,
    pass_memory_kwargs=True,
)

_PORTFOLIO_SPEC = _TurnHandlerSpec(
    intent="portfolio",
    setup_error_types=(PortfolioSetupError,),
    error_label="组合分析出错",
    run_setup=_run_portfolio_setup,
    format_user_display=_identity_display_text,
    pass_memory_kwargs=True,
    success_criteria=_portfolio_success_criteria,
)

_MODELING_SPEC = _TurnHandlerSpec(
    intent="modeling",
    setup_error_types=(JoinSetupError, ModelingSetupError),
    error_label="建模出错",
    run_setup=_run_modeling_setup,
    format_user_display=_c1_display_text,
    pass_memory_kwargs=True,
)


DRIVER_TURN_FUNCS = {
    TASK_TYPE_MODELING: run_modeling_driver_turn,
    TASK_TYPE_DATA_JOIN: run_join_driver_turn,
    TASK_TYPE_FEATURE_ANALYSIS: run_feature_driver_turn,
    TASK_TYPE_STRATEGY: run_strategy_driver_turn,
    TASK_TYPE_VINTAGE: run_vintage_driver_turn,
    TASK_TYPE_PORTFOLIO: run_portfolio_driver_turn,
}


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
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    recovery_bypass: bool = False,
) -> dict:
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
        confirmation_source=confirmation_source,
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
    if latest_open_gate(conversation) is not None:
        return None
    failure = latest_unresolved_workflow_failure(
        conversation,
        workflow=task.task_type,
    )
    if failure is None:
        return None
    retryable = bool(failure.diagnostic.get("retryable", True))
    if failure.failure_envelope is not None:
        retryable = bool(failure.failure_envelope.get("retryable", retryable))
    if retryable and is_explicit_workflow_retry(text):
        return None

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


# Natural-language strategy request wiring --------------------------------------
# The compiler is an intent-to-contract boundary, not an execution runtime.
# Strictly validated new requests auto-start their reversible Workflow steps;
# only real human-responsibility gates pause. Opaque pending drafts remain here
# solely for read-compatible confirm/cancel handling of older conversations.
_STRATEGY_REQUEST_META_KEY = "strategy_request"
_STRATEGY_POOL_WORKFLOWS = frozenset(
    {
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_compile",
    }
)
_STRATEGY_POOL_MEASUREMENT_WORKFLOWS = frozenset({"strategy_pool_impact"})
_STRATEGY_REQUEST_ACTION_RE = re.compile(
    r"(?:开发|设计|制定|创建|生成|构建|训练|物化|固化|冻结|探索|做|计算|测算|分析|评估|查看|看一下|看下|回测|测试|应用|执行|打标|"
    r"对比|比较|采纳|采用|上线|报告|文档|监控|漂移|挖掘|选择|筛选|保留|合并|编辑|"
    r"添加|加入|入池|删除|移除|排序|重排|改为|编译|预览|"
    r"develop|design|create|build|train|materialize|compute|calculate|analy[sz]e|evaluate|backtest|apply|compare|"
    r"adopt|report|monitor|mine|refine|select|merge|add|remove|delete|reorder|compile|preview)",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_SUBJECT_RE = re.compile(
    r"(?:策略|策略样本|样本设计|样本边界|策略池|规则池|准入|审批|拒绝|额度|授信|定价|利率|分群|分层|规则|候选|候选箱|单变量|分箱|自动树|决策树|叶子|叶节点|投票|Voting|n[-_ ]?of[-_ ]?k|(?:二维|2\s*[dD])?\s*(?:交叉|cross)\s*(?:矩阵|matrix)|cutoff|利润|收益|"
    r"催收|滚动率|迁徙率|迁徙矩阵|定价矩阵|额度矩阵|网格|ROA|"
    r"roll(?:\s|-|_)*rate|strategy(?:\s|-|_)*pool|pool|strategy|approval|reject|limit|pricing|segment|rule|candidate|automatic(?:\s|-|_)*tree|decision(?:\s|-|_)*tree|leaf|"
    r"candidate\s+bins?|\bbins?\b|univariate|binning|sample(?:\s|-|_)*design|profit|collection)",
    re.IGNORECASE,
)
_STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE = re.compile(
    r"(?:建\s*(?:一棵)?\s*(?:自动)?(?:决策)?树(?!状|莓|屋)|"
    r"训练\s*(?:一棵)?\s*(?:自动)?(?:决策)?树(?:模型)?|"
    r"(?<![A-Za-z0-9_])(?:build|train)\s+(?:an?\s+)?"
    r"(?:(?:automatic|decision)\s+)?tree(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_CANCEL_RE = re.compile(
    r"(?:先别|不要|不用|不执行|先不|暂不|暂停|停止|取消|"
    r"do\s*not|don't|dont|stop|cancel|wait)",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_NON_EXECUTION_RE = re.compile(
    r"(?:不要执行|不要运行|先别执行|先别运行|先不执行|先不运行|"
    r"只预览|仅预览|只讨论|仅讨论|只聊|仅供讨论|"
    r"do\s+not\s+(?:execute|run)|don't\s+(?:execute|run)|"
    r"preview\s+only|discussion\s+only|discuss\s+only)",
    re.IGNORECASE,
)
_STRATEGY_POOL_COMPILE_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:编译|预览|compile|preview))",
    re.IGNORECASE,
)
_STRATEGY_POOL_IMPACT_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:影响|效果|瀑布|逐月|通过率|坏账率|风险率|测算|评估|计算|回测|"
    r"impact|effect|waterfall|monthly|approval\s+rate|bad\s+rate|risk\s+rate|"
    r"measure|assess|evaluat|calculate|backtest))",
    re.IGNORECASE,
)
_STRATEGY_NAN_LABEL_META_KEY = "strategy_nan_label_confirmation"


class _StrategySampleDesignRequiredError(StrategySetupError):
    """The current strategy request has no exact mature sample-design binding."""


_STRATEGY_DROP_NAN_CONFIRM_RE = re.compile(
    r"(?:确认|同意|允许|可以).{0,12}(?:丢弃|排除|剔除|删除).{0,12}"
    r"(?:NaN|nan|空标签|缺失标签|无效标签)|"
    r"(?:确认|同意|允许|可以).{0,12}"
    r"(?:NaN|nan|空标签|缺失标签|无效标签).{0,24}"
    r"(?:风险|坏账).{0,8}分母.{0,8}(?:排除|剔除)|"
    r"(?:confirm|allow).{0,12}(?:drop|exclude).{0,12}(?:nan|missing)\s+labels?",
    re.IGNORECASE,
)
_STRATEGY_DROP_NAN_CANCEL_RE = re.compile(
    r"(?:不丢弃|不排除|不剔除|不删除|取消|停止|"
    r"do\s+not\s+(?:drop|exclude)|don't\s+(?:drop|exclude))",
    re.IGNORECASE,
)
_TYPED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest"})
_STORED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest", "compare"})


def _is_strategy_request_intent(text: str) -> bool:
    """Recognize standard strategy requests plus narrow tree-build shorthand."""

    return bool(
        _STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE.search(text)
        or (
            _STRATEGY_REQUEST_ACTION_RE.search(text)
            and _STRATEGY_REQUEST_SUBJECT_RE.search(text)
        )
    )


def _maybe_handle_strategy_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Compile a natural-language strategy request and route it safely.

    A strict compiler result plus preflight authorizes reversible execution. The
    plan overview is auto-resumed in the same request and platform AUTO safety
    still stops at real human-responsibility gates (adoption/disposition). Legacy
    persisted request confirmations remain readable for compatibility only.
    """

    if task.task_type != TASK_TYPE_STRATEGY or runtime.llm_client is None:
        return None
    text = str(user_text or "").strip()
    if not text:
        return None

    conversation = repo.list_agent_messages(task.id)
    if latest_unresolved_workflow_failure(
        conversation,
        workflow=task.task_type,
    ) is not None and is_explicit_workflow_retry(text):
        # The recovery branch deliberately returns None for an explicit retry;
        # do not reinterpret that command as a brand-new strategy request.
        return None
    pending = _latest_strategy_request_pending(conversation)
    if pending is not None and is_confirm(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_request_confirmation"},
        )
        if (
            _active_plan(runtime.plan_repo, task.id) is not None
            or latest_open_gate(conversation) is not None
        ):
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_request_stale_confirmation",
                message="任务状态已变化，旧策略草案已失效；请完成当前计划后重新发起。",
            )
        return _run_confirmed_strategy_request(runtime, repo, task, pending)
    if pending is not None and _STRATEGY_REQUEST_CANCEL_RE.search(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_request_cancel"},
        )
        try:
            PendingStrategyRequestRepository(runtime.settings.db_path).cancel(
                task_id=task.id,
                request_id=str(pending.get("request_id") or ""),
                expected_payload_sha256=str(pending.get("payload_sha256") or ""),
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_request_stale_cancellation",
                message="上一份策略草案已被处理或失效，请重新发起需要执行的操作。",
            )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="已取消上一份策略执行草案，没有创建计划或执行工具。",
            metadata={"intent": "strategy_request_cancelled"},
        )
        return join_turn_response(repo, task.id)
    if pending is not None:
        # Any non-confirm/non-cancel reply replaces the confirmation context.
        # Make the opaque row terminal as well as hiding the old message ref so
        # abandoned drafts cannot accumulate as apparently actionable state.
        _invalidate_pending_strategy_request(runtime, task, pending)

    nan_confirmation = _latest_strategy_nan_label_confirmation(conversation)
    if nan_confirmation is not None:
        if _STRATEGY_DROP_NAN_CANCEL_RE.search(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_cancel"},
            )
            repo.add_agent_message(
                task.id,
                role="assistant",
                stage="chat",
            content="已取消本次策略执行；未应用任何空标签排除口径，也没有创建计划。",
                metadata={
                    "intent": "strategy_drop_nan_labels_cancelled",
                    "kind": "clarification",
                    "code": "strategy_drop_nan_labels_cancelled",
                },
            )
            return join_turn_response(repo, task.id)
        if _STRATEGY_DROP_NAN_CONFIRM_RE.search(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_confirm"},
            )
            return _resume_strategy_after_nan_label_confirmation(
                runtime,
                repo,
                task,
                nan_confirmation,
            )
        if is_confirm(text):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_drop_nan_labels_ambiguous"},
            )
            return _repeat_strategy_nan_label_clarification(
                repo,
                task,
                nan_confirmation,
            )

    if not _is_strategy_request_intent(text):
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    if _STRATEGY_REQUEST_NON_EXECUTION_RE.search(
        text
    ) and not _STRATEGY_POOL_COMPILE_REQUEST_RE.search(text):
        repo.add_agent_message(
            task.id,
            role="user",
            stage="chat",
            content=text,
            metadata={"intent": "strategy_preview_only"},
        )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                "已按仅预览/讨论处理：本轮不会编译执行草案、创建计划或调用策略工具。"
                "需要实际执行时，请另发一条明确的执行请求。"
            ),
            metadata={
                "intent": "strategy_preview_only",
                "kind": "clarification",
                "code": "strategy_execution_not_authorized",
            },
        )
        return {
            "task_id": task.id,
            "status": "preview_only",
            "code": "strategy_execution_not_authorized",
            "messages": repo.list_agent_messages(task.id),
        }

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={"intent": "strategy_request"},
    )
    preview = None
    preview_error = None
    is_sample_design_request = utterance_targets_strategy_sample_design(text)
    try:
        preview = (
            _strategy_pool_impact_dataset_preview(runtime, task)
            if _STRATEGY_POOL_IMPACT_REQUEST_RE.search(text)
            else (
                _strategy_sample_design_dataset_preview(runtime, task)
                if is_sample_design_request
                else _strategy_dataset_preview(runtime, task)
            )
        )
    except StrategySetupError as exc:
        preview_error = str(exc)

    if is_sample_design_request and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                "strategy_sample_design_target_invalid"
                if preview_error and "必须是数值 0/1 或真实空值" in preview_error
                else "strategy_sample_design_workspace_required"
            ),
            message=preview_error or "样本设计要求先确认活动 DataWorkspace。",
        )

    compilation = compile_strategy_request(
        text,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
        llm=runtime.llm_client,
    )
    if compilation.draft is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                compilation.clarification_code or "strategy_request_needs_clarification"
            ),
            message=compilation.clarification or "请补充策略操作、策略类型和业务口径。",
            fields=compilation.clarification_fields,
        )
    preflight = _strategy_request_preflight(runtime, task, compilation.draft)
    if preflight is not None:
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    requires_dataset = _strategy_request_requires_dataset(compilation.draft)
    if requires_dataset and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(compilation.draft) and (
        preview is None or not preview.target_col
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        compilation.draft,
        preview=preview,
        auto_start=True,
    )


def _prepare_and_run_validated_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    draft: CompiledStrategyRequestDraft,
    *,
    preview,
    auto_start: bool,
    drop_nan_labels: bool = False,
    expected_pool_binding: Mapping | None = None,
) -> dict:
    """Bind current evidence, resolve the NaN policy, then instantiate once."""

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and (
            draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
            or draft.workflow
            in {"strategy_sample_design", "limit_pricing_matrix"}
        )
        and draft.workflow_inputs.get("drop_nan_labels") is True
    ):
        # This boolean has already passed exact utterance grounding in the
        # compiler; it is the user's explicit authorization, not an LLM default.
        drop_nan_labels = True
    requires_dataset = _strategy_request_requires_dataset(draft)
    is_pool_impact = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_sample_design"
    )
    context = None
    if requires_dataset:
        try:
            context = (
                _strategy_pool_impact_dataset_context(runtime, task)
                if is_pool_impact
                else (
                    _strategy_sample_design_dataset_context(runtime, task)
                    if is_sample_design
                    else _strategy_dataset_context(
                        runtime,
                        task,
                        require_target=_strategy_request_requires_target(draft),
                    )
                )
            )
        except StrategySetupError as exc:
            return append_join_error(repo, task.id, str(exc))
        if _is_automatic_tree_build_draft(draft):
            if preview is None or not _strategy_dataset_binding_matches(
                runtime,
                task,
                preview=preview,
                context=context,
            ):
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="strategy_dataset_context_changed",
                    message=(
                        "策略样本在编译与活动工作区绑定之间发生变化；本次请求未执行，"
                        "请基于当前数据重新描述。"
                    ),
                )
            try:
                preview, context = _ensure_automatic_tree_active_workspace(
                    runtime,
                    task,
                    preview=preview,
                    context=context,
                )
            except StrategySetupError as exc:
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="automatic_tree_active_workspace_required",
                    message=str(exc),
                )
        if preview is None or not _strategy_dataset_binding_matches(
            runtime,
            task,
            preview=preview,
            context=context,
            use_confirmed_workspace_target=is_pool_impact,
            use_sample_design_workspace=is_sample_design,
        ):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_changed",
                message=(
                    "策略样本在编译与计划创建之间发生变化；本次请求未执行，"
                    "请基于当前数据重新描述。"
                ),
            )

    if _strategy_request_requires_complete_labels(draft):
        assert context is not None and context.target_col
        try:
            n_total, n_nan = _strategy_target_nan_stats(runtime, context)
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_target_labels_invalid",
                message=str(exc),
                fields=("target_col",),
            )
        if n_nan and not drop_nan_labels:
            return _strategy_nan_label_clarification_response(
                runtime,
                repo,
                task,
                draft=draft,
                context=context,
                n_total=n_total,
                n_nan=n_nan,
            )

    try:
        return _run_validated_strategy_request(
            runtime,
            repo,
            task,
            draft,
            context=context,
            auto_start=auto_start,
            drop_nan_labels=drop_nan_labels,
            expected_pool_binding=expected_pool_binding,
        )
    except _StrategySampleDesignRequiredError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
        )
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"策略请求执行出错：{exc}")


def _run_validated_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    draft: CompiledStrategyRequestDraft,
    *,
    context,
    auto_start: bool,
    drop_nan_labels: bool,
    expected_pool_binding: Mapping | None = None,
) -> dict:
    """Route one already-validated draft without another execution confirmation."""

    if (
        isinstance(draft, StrategyRequestDraft)
        and draft.strategy_spec is None
        and draft.operation == "report"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_report",
            slots={"strategy_id": draft.strategy_id},
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "automatic_tree_apply"
    ):
        if context is None:
            raise StrategySetupError(
                "自动树全量写回需要当前活动 DataWorkspace。"
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_automatic_tree_apply",
            slots=_automatic_tree_apply_slots(
                runtime,
                task_id=task.id,
                draft=draft,
                context=context,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "automatic_tree_leaf_materialization"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_automatic_tree_leaf_materialization",
            slots=_automatic_tree_leaf_materialization_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "cross_matrix_cell_selection"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_cross_matrix_cell_selection",
            slots=_cross_matrix_cell_selection_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "voting_candidate_build"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_voting_candidate_build",
            slots=_strategy_voting_candidate_plan_slots(runtime, task, draft),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_impact"
    ):
        if context is None:
            raise StrategySetupError(
                "Strategy Pool 影响测算需要活动 DataWorkspace 和确认的目标列。"
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_pool_impact",
            slots=_strategy_pool_impact_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=drop_nan_labels,
                expected_pool_binding=expected_pool_binding,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_sample_design"
    ):
        if context is None:
            raise StrategySetupError(
                "策略样本设计需要确认的活动 DataWorkspace 和二元目标列。"
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_sample_design",
            slots=_strategy_sample_design_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=drop_nan_labels,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow in _STRATEGY_POOL_WORKFLOWS
    ):
        template_id = {
            "strategy_pool_add_candidate": "strategy_pool_add_candidate",
            "strategy_pool_remove_entry": "strategy_pool_remove_entry",
            "strategy_pool_set_action": "strategy_pool_set_action",
            "strategy_pool_reorder": "strategy_pool_reorder",
            "strategy_pool_compile": "strategy_pool_compile",
        }[draft.workflow]
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=template_id,
            slots=_strategy_pool_plan_slots(runtime, task, draft),
            auto_start=auto_start,
        )

    if context is None:
        raise StrategySetupError("当前策略操作需要任务内数据上下文。")

    if isinstance(draft, StandardWorkflowRequestDraft):
        workflow_inputs = draft.to_dict()["workflow_inputs"]
        source_candidate_id = workflow_inputs.get("source_candidate_id")
        template_id = {
            "profit_calc": "strategy_profit_analysis",
            "roll_rate_matrix": "strategy_roll_rate_analysis",
            "limit_pricing_matrix": "strategy_limit_pricing_analysis",
            "univariate_candidate_analysis": ("strategy_univariate_candidate_analysis"),
            "cross_matrix_analysis": "strategy_cross_matrix_analysis",
            "automatic_tree_candidate_build": (
                "strategy_automatic_tree_candidate_build"
            ),
            "univariate_candidate_refinement": (
                "strategy_univariate_candidate_refinement_existing"
                if source_candidate_id is not None
                else "strategy_univariate_candidate_refinement"
            ),
        }[draft.workflow]
        slots = {
            "dataset_id": context.dataset_id,
            **workflow_inputs,
        }
        slots.pop("source_candidate_id", None)
        if source_candidate_id is not None:
            slots.update(
                _candidate_source_artifact_slots(
                    runtime,
                    task_id=task.id,
                    candidate_id=str(source_candidate_id),
                )
            )
        elif draft.workflow in {
            "univariate_candidate_analysis",
            "univariate_candidate_refinement",
            "automatic_tree_candidate_build",
            "cross_matrix_analysis",
        }:
            binding = {
                "expected_content_hash": getattr(context, "dataset_content_hash", None),
                "workspace_revision": getattr(context, "workspace_revision", None),
                "analysis_generation": getattr(context, "analysis_generation", None),
                "semantic_mapping_hash": getattr(
                    context, "semantic_mapping_hash", None
                ),
            }
            if (
                not isinstance(binding["expected_content_hash"], str)
                or not isinstance(binding["semantic_mapping_hash"], str)
                or isinstance(binding["workspace_revision"], bool)
                or not isinstance(binding["workspace_revision"], int)
                or isinstance(binding["analysis_generation"], bool)
                or not isinstance(binding["analysis_generation"], int)
            ):
                raise StrategySetupError(
                    "策略候选分析无法绑定当前数据工作区，请重新选择活动数据集。"
                )
            slots.update(binding)
            slots["target_col"] = context.target_col
            slots["sample_design_ref"] = (
                _latest_matching_strategy_sample_design_ref(
                    runtime,
                    task,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                    weight_col=(
                        workflow_inputs.get("sample_weight_col")
                        if draft.workflow == "automatic_tree_candidate_build"
                        else None
                    ),
                    loan_amount_col=workflow_inputs.get("loan_amount_col"),
                    overdue_amount_col=workflow_inputs.get("overdue_amount_col"),
                )
            )
        elif draft.workflow == "limit_pricing_matrix":
            slots["sample_design_ref"] = (
                _latest_matching_strategy_sample_design_ref(
                    runtime,
                    task,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                )
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=template_id,
            slots=_strategy_slots_with_drop_nan(slots, drop_nan_labels),
            auto_start=auto_start,
        )

    if _is_auto_candidate_draft(draft):
        slots = _candidate_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="deterministic_strategy_candidate_development",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            auto_start=auto_start,
        )

    if draft.strategy_spec is not None:
        slots = (
            {
                "dataset_id": context.dataset_id,
                "strategy_spec": draft.to_dict()["strategy_spec"],
            }
            if draft.operation == "apply"
            else _typed_strategy_slots(context, draft)
        )
        template_id = {
            "develop": "typed_strategy_build",
            "apply": "typed_strategy_apply",
            "analyze": "typed_strategy_evaluation",
            "backtest": "typed_strategy_evaluation",
        }[draft.operation]
        if draft.operation in {"analyze", "backtest"}:
            slots["sample_design_ref"] = (
                _latest_matching_strategy_sample_design_ref(
                    runtime,
                    task,
                    context=context,
                    drop_nan_labels=bool(drop_nan_labels),
                )
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=template_id,
            slots=_strategy_slots_with_drop_nan(slots, drop_nan_labels),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )

    if draft.operation in _STORED_EVALUATION_OPERATIONS:
        slots = _stored_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_evaluation",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )
    if draft.operation == "apply":
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_apply",
            slots={
                "dataset_id": context.dataset_id,
                "strategy_id": draft.strategy_id,
            },
            auto_start=auto_start,
        )
    if draft.operation == "adopt":
        slots = _stored_strategy_slots(context, draft)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
        )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="stored_strategy_adoption",
            slots=_strategy_slots_with_drop_nan(
                slots,
                drop_nan_labels,
            ),
            success_criteria=_strategy_request_success_criteria(draft),
            auto_start=auto_start,
        )

    if draft.operation == "develop":
        task = repo.update_strategy_input(
            task.id,
            _strategy_contract_from_draft(draft),
        )
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_FULL_DEVELOPMENT,
        )
    elif draft.operation == "mine_rules":
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_RULE_MINING,
        )
    elif draft.operation == "monitor":
        setup = _run_strategy_setup(
            runtime,
            repo,
            task,
            None,
            forced_intent=STRATEGY_INTENT_MONITORING,
        )
    else:  # guarded by _strategy_request_preflight
        raise StrategySetupError(f"strategy operation is not wired: {draft.operation}")
    if isinstance(setup, dict):
        return setup
    template_id, slots, start_kwargs = setup
    if template_id in {"strategy_development", "rule_strategy", "strategy_analysis"}:
        slots = dict(slots)
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=bool(drop_nan_labels),
        )
    if "success_criteria" not in start_kwargs:
        criteria = _strategy_request_success_criteria(draft)
        if criteria:
            start_kwargs = {**start_kwargs, "success_criteria": criteria}
    return _start_confirmed_strategy_plan(
        runtime,
        repo,
        task,
        template_id=template_id,
        slots=_strategy_slots_with_drop_nan(slots, drop_nan_labels),
        auto_start=auto_start,
        **start_kwargs,
    )


def _run_confirmed_strategy_request(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    pending: dict,
) -> dict:
    pending_repository = PendingStrategyRequestRepository(runtime.settings.db_path)
    request_id = str(pending.get("request_id") or "")
    payload_sha256 = str(pending.get("payload_sha256") or "")
    pending_record = pending_repository.get(task.id, request_id)
    if (
        pending_record is None
        or pending_record.status != "pending"
        or pending_record.payload_sha256 != payload_sha256
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_stale_confirmation",
            message="这份策略草案已被处理、篡改或失效，请重新描述并确认。",
        )
    preview = None
    preview_error = None
    try:
        preview = _strategy_dataset_preview(runtime, task)
    except StrategySetupError as exc:
        preview_error = str(exc)
    expected_identity = pending_record.dataset_identity
    if expected_identity is not None and (
        preview is None
        or expected_identity != preview.identity
        or pending_record.target_col != preview.target_col
    ):
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_changed",
            message="策略样本或目标列已变化，请重新描述并确认策略请求。",
        )

    compilation = validate_strategy_request(
        pending_record.validated_draft,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
    )
    if compilation.draft is None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_invalidated",
            message=compilation.clarification
            or "已确认的策略草案未通过重新校验，请重新描述。",
        )
    draft = compilation.draft
    preflight = _strategy_request_preflight(runtime, task, draft)
    if preflight is not None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    if _strategy_request_requires_dataset(draft) and preview is None:
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(draft) and (
        preview is None or not preview.target_col
    ):
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    context = None
    if _strategy_request_requires_dataset(draft):
        try:
            context = _strategy_dataset_context(
                runtime,
                task,
                require_target=_strategy_request_requires_target(draft),
            )
        except StrategySetupError as exc:
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_required",
                message=str(exc),
            )
        if not _strategy_dataset_binding_matches(
            runtime,
            task,
            preview=preview,
            context=context,
        ):
            _invalidate_pending_strategy_request(runtime, task, pending)
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_changed",
                message=(
                    "策略样本在确认与计划创建之间发生变化；历史草案未消费，"
                    "请基于当前数据重新描述。"
                ),
            )
        if _strategy_request_requires_complete_labels(draft):
            try:
                n_total, n_nan = _strategy_target_nan_stats(runtime, context)
            except StrategySetupError as exc:
                _invalidate_pending_strategy_request(runtime, task, pending)
                return _strategy_request_clarification_response(
                    repo,
                    task,
                    code="strategy_target_labels_invalid",
                    message=str(exc),
                    fields=("target_col",),
                )
            if n_nan:
                _invalidate_pending_strategy_request(runtime, task, pending)
                return _strategy_nan_label_clarification_response(
                    runtime,
                    repo,
                    task,
                    draft=draft,
                    context=context,
                    n_total=n_total,
                    n_nan=n_nan,
                )

    existing_plan_ids = frozenset(
        plan.id for plan in runtime.plan_repo.list_plans_for_task(task.id)
    )
    try:
        pending_repository.consume(
            task_id=task.id,
            request_id=request_id,
            expected_payload_sha256=payload_sha256,
        )
    except (
        PendingStrategyRequestConflictError,
        PendingStrategyRequestNotFoundError,
        ValueError,
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_stale_confirmation",
            message="这份策略草案已被其他请求处理或失效，请重新发起。",
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="已接收确认，正在按已校验口径创建受治理的策略计划。",
        metadata={
            "intent": "strategy_request_claimed",
            "request_id": request_id,
            "payload_sha256": payload_sha256,
            _STRATEGY_REQUEST_META_KEY: {
                "request_id": request_id,
                "payload_sha256": payload_sha256,
            },
        },
    )

    try:
        return _run_validated_strategy_request(
            runtime,
            repo,
            task,
            draft,
            context=context,
            auto_start=False,
            drop_nan_labels=False,
        )
    except _StrategySampleDesignRequiredError as exc:
        try:
            pending_repository.release_after_failed_start(
                task_id=task.id,
                request_id=request_id,
                expected_payload_sha256=payload_sha256,
                existing_plan_ids=existing_plan_ids,
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            pass
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
        )
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        try:
            pending_repository.release_after_failed_start(
                task_id=task.id,
                request_id=request_id,
                expected_payload_sha256=payload_sha256,
                existing_plan_ids=existing_plan_ids,
            )
        except (
            PendingStrategyRequestConflictError,
            PendingStrategyRequestNotFoundError,
            ValueError,
        ):
            new_plans = [
                plan
                for plan in runtime.plan_repo.list_plans_for_task(task.id)
                if plan.id not in existing_plan_ids
            ]
            if new_plans:
                recovery_plan = new_plans[-1]
                repo.add_agent_message(
                    task.id,
                    role="assistant",
                    stage="chat",
                    content=(
                        "策略计划已经创建，但启动响应未完整返回；已保留现有计划，"
                        "请从该计划继续，旧确认不会重新创建计划。"
                    ),
                    metadata={
                        "intent": "strategy_request_plan_recovery",
                        "plan_id": recovery_plan.id,
                        "plan_status": recovery_plan.status.value,
                    },
                )
                return join_turn_response(repo, task.id)
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"策略请求执行出错：{exc}")


def _start_confirmed_strategy_plan(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    template_id: str,
    slots: dict,
    success_criteria: list[dict] | None = None,
    auto_start: bool = False,
) -> dict:
    """Start a compiled Workflow, optionally auto-accepting its overview."""

    start_kwargs = {}
    if success_criteria:
        start_kwargs["success_criteria"] = success_criteria
    driver = _driver(runtime)
    start = driver.start(
        task_id=task.id,
        template_id=template_id,
        slots=slots,
        tier=runtime.tier,
        **start_kwargs,
    )
    append_driver_messages(
        repo,
        task.id,
        start,
        settings=runtime.settings,
        task=task,
    )
    if auto_start:
        resumed = driver.resume(
            plan_id=start.plan_id,
            user_text="开始",
            confirmation_source=CONFIRMATION_SOURCE_AUTO,
        )
        append_driver_messages(
            repo,
            task.id,
            resumed,
            settings=runtime.settings,
            task=task,
        )
    return join_turn_response(repo, task.id)


def _strategy_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: CompiledStrategyRequestDraft,
) -> tuple[str, str] | None:
    """Reject any compiled request whose trusted Workflow is not wired yet."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        return _standard_workflow_request_preflight(runtime, task, draft)

    if draft.candidate_design is not None:
        if draft.operation != "develop" or draft.strategy_type not in {
            "limit",
            "pricing",
            "segmentation",
        }:
            return (
                "candidate_strategy_request_invalid",
                "确定性候选输入只允许用于额度、定价或分群策略开发。",
            )
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.strategy_id,
                draft.adoption_reason,
                draft.profit,
            )
        ):
            return (
                "candidate_strategy_unused_fields",
                "候选开发只使用候选搜索空间、类型专属经济口径和可选同类型基线；"
                "目标、审批约束、策略 ID、预先采纳理由或审批利润口径不会被静默忽略。",
            )
        if draft.baseline_strategy_id:
            baseline = StrategyRepository(runtime.settings.db_path).get_strategy_meta(
                draft.baseline_strategy_id
            )
            if baseline is None or baseline.get("task_id") != task.id:
                return (
                    "strategy_baseline_not_owned_by_task",
                    "没有在当前任务中找到基线策略，不能跨任务对比。",
                )
            if baseline.get("strategy_type") != draft.strategy_type:
                return (
                    "strategy_baseline_type_mismatch",
                    "候选策略与基线策略类型不一致，不能生成同口径对比。",
                )
        return None

    if draft.strategy_spec is not None:
        if draft.strategy_id is not None:
            return (
                "strategy_request_conflicting_identity",
                "请求同时给了 strategy_spec 和 strategy_id；一个表示新规则草案、"
                "一个表示已有策略，请明确选择其一。",
            )
        if draft.adoption_reason is not None:
            return (
                "strategy_request_unused_adoption_reason",
                "当前请求不是采纳操作，采纳理由不会被静默忽略；请删除后重新确认。",
            )
        if draft.operation in {"develop", "apply"}:
            if any(
                value is not None
                for value in (
                    draft.objective,
                    draft.max_bad_rate,
                    draft.min_approval_rate,
                    draft.baseline_strategy_id,
                    draft.profit,
                    draft.economics_inputs,
                )
            ):
                operation_label = "构造" if draft.operation == "develop" else "应用"
                return (
                    "strategy_typed_operation_unused_fields",
                    f"直接{operation_label}类型化规则只使用 strategy_spec；目标、约束、"
                    "基线和经济参数不会被静默忽略，请删除这些字段或改为分析/回测。",
                )
            return None
        if draft.operation in _TYPED_EVALUATION_OPERATIONS:
            if draft.objective is not None:
                return (
                    "strategy_typed_evaluation_unused_objective",
                    "已有明确规则的分析/回测不会重新优化 objective；请删除 objective，"
                    "保留要检验的明确约束和经济参数。",
                )
            if draft.strategy_type not in {"approval", "reject"} and any(
                value is not None
                for value in (
                    draft.max_bad_rate,
                    draft.min_approval_rate,
                    draft.profit,
                )
            ):
                return (
                    "strategy_typed_business_contract_not_wired",
                    f"{draft.strategy_type} 不能套用审批通过率/坏率/利润约束；"
                    "请保留类型专属规则和经济参数。",
                )
            if draft.baseline_strategy_id:
                baseline = StrategyRepository(
                    runtime.settings.db_path
                ).get_strategy_meta(draft.baseline_strategy_id)
                if baseline is None or baseline.get("task_id") != task.id:
                    return (
                        "strategy_baseline_not_owned_by_task",
                        "没有在当前任务中找到基线策略，不能跨任务对比。",
                    )
                if baseline.get("strategy_type") != draft.strategy_type:
                    return (
                        "strategy_baseline_type_mismatch",
                        "新规则草案与基线策略类型不一致，不能生成同口径对比。",
                    )
            return None
        return (
            "strategy_operation_not_wired",
            f"已识别 {draft.operation} 请求，但该操作不能用类型化评估流程代替；"
            "当前不会静默执行成回测。",
        )
    if draft.operation in {
        *_STORED_EVALUATION_OPERATIONS,
        "apply",
        "report",
        "adopt",
    }:
        return _stored_strategy_request_preflight(runtime, task, draft)
    if draft.operation == "develop":
        if draft.strategy_id is not None or draft.adoption_reason is not None:
            return (
                "strategy_request_unused_fields",
                "新策略开发不会使用已有 strategy_id 或预先写入采纳理由，请删除这些字段。",
            )
        if draft.strategy_type not in {"approval", "reject"}:
            return (
                "strategy_typed_spec_required",
                f"{draft.strategy_type} 策略开发需要明确的类型化规则草案；"
                "请补充各规则的条件、动作和值。",
            )
        if draft.objective not in {"max_approval", "max_profit"}:
            return (
                "strategy_objective_required",
                "审批/拒绝策略开发需要明确 objective=max_approval 或 max_profit。",
            )
        if draft.max_bad_rate is None and draft.min_approval_rate is None:
            return (
                "strategy_constraint_required",
                "请至少说明最大坏账率或最低通过率，平台不会代填经营约束。",
            )
        if draft.objective == "max_profit" and draft.profit is None:
            return (
                "strategy_profit_contract_required",
                "利润目标需要完整 EAD/PD 列和利率、资金成本、LGD、单笔成本、期限口径。",
            )
        return None
    if draft.operation == "mine_rules":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_rule_request_unused_fields",
                "规则挖掘入口当前不会使用目标、约束、策略 ID 或利润字段；"
                "请删除这些字段，避免口径被静默忽略。",
            )
        if draft.strategy_type != "reject":
            return (
                "strategy_rule_type_required",
                "当前规则挖掘生成拒绝规则，请把策略类型明确为 reject。",
            )
        return None
    if draft.operation == "monitor":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_monitor_request_unused_fields",
                "监控入口只接受监控对象；目标、约束、基线、采纳理由和利润字段"
                "不会被静默忽略，请删除后重试。",
            )
        adopted = [
            meta
            for meta in StrategyRepository(runtime.settings.db_path).list_meta_for_task(
                task.id
            )
            if meta.get("asset_status") == ASSET_STATUS_ADOPTED_LOCAL
        ]
        if not adopted:
            return (
                "strategy_adopted_version_required",
                "当前任务没有本地已采纳策略，请先完成回测和人工采纳再启动监控。"
                "本地已采纳，不代表生产上线。",
            )
        selected = adopted[-1]
        if draft.strategy_id and draft.strategy_id != selected.get("id"):
            return (
                "strategy_monitor_target_mismatch",
                "当前监控入口只会执行任务内最新的本地已采纳策略；"
                "请求中的策略 ID 与其不一致。",
            )
        if draft.strategy_type != selected.get("strategy_type"):
            return (
                "strategy_monitor_type_mismatch",
                "请求中的策略类型与任务内最新的本地已采纳策略不一致，"
                "请重新确认监控对象。",
            )
        return None
    return (
        "strategy_operation_not_wired",
        f"已识别 {draft.operation} 请求，但对应受信任 Workflow 尚未接线；"
        "当前不会把它降级成其他策略操作。",
    )


def _standard_workflow_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> tuple[str, str] | None:
    if draft.workflow == "strategy_sample_design":
        try:
            context = _strategy_sample_design_dataset_context(runtime, task)
            _strategy_sample_design_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except StrategySetupError as exc:
            return ("strategy_sample_design_workspace_required", str(exc))
        return None
    if draft.workflow == "strategy_pool_impact":
        try:
            context = _strategy_pool_impact_dataset_context(runtime, task)
            _strategy_pool_impact_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except StrategySetupError as exc:
            return ("strategy_pool_impact_binding_required", str(exc))
        return None
    if draft.workflow == "automatic_tree_apply":
        try:
            context = _strategy_dataset_context(runtime, task, require_target=False)
            _automatic_tree_apply_slots(
                runtime,
                task_id=task.id,
                draft=draft,
                context=context,
            )
        except StrategySetupError as exc:
            return ("automatic_tree_apply_binding_required", str(exc))
        return None
    if draft.workflow == "automatic_tree_leaf_materialization":
        try:
            _automatic_tree_leaf_materialization_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return ("automatic_tree_leaf_source_required", str(exc))
        return None
    if draft.workflow == "cross_matrix_cell_selection":
        try:
            _cross_matrix_cell_selection_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            )
        except StrategySetupError as exc:
            return ("cross_matrix_cell_source_required", str(exc))
        return None
    if draft.workflow == "voting_candidate_build":
        try:
            _strategy_voting_candidate_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_voting_pool_binding_required", str(exc))
        return None
    if draft.workflow in _STRATEGY_POOL_WORKFLOWS:
        try:
            _strategy_pool_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_pool_binding_required", str(exc))
        return None
    if draft.workflow == "univariate_candidate_refinement":
        source_candidate_id = draft.workflow_inputs.get("source_candidate_id")
        if source_candidate_id is not None:
            try:
                _candidate_source_artifact_slots(
                    runtime,
                    task_id=task.id,
                    candidate_id=str(source_candidate_id),
                )
            except StrategySetupError as exc:
                return ("strategy_candidate_source_required", str(exc))
        return None
    if draft.workflow != "limit_pricing_matrix":
        return None
    strategy_id = draft.workflow_inputs.get("strategy_id")
    if strategy_id is None:
        return None
    meta = StrategyRepository(runtime.settings.db_path).get_strategy_meta(
        str(strategy_id)
    )
    if meta is None or meta.get("task_id") != task.id:
        return (
            "strategy_not_owned_by_task",
            "没有在当前任务中找到要关联的额度/定价策略，不能跨任务挂载矩阵产物。",
        )
    if meta.get("strategy_type") not in {"limit", "pricing"}:
        return (
            "strategy_type_mismatch",
            "额度定价矩阵只能关联当前任务中的额度或定价策略。",
        )
    return None


def _candidate_source_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    candidate_id: str,
) -> dict[str, str]:
    matches = []
    for artifact in TaskArtifactRepository(runtime.settings.db_path).list_for_task(
        task_id
    ):
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == "strategy_candidate_json"
            and artifact.get("origin_tool") == "strategy.analyze_univariate_candidates"
            and isinstance(provenance, dict)
            and provenance.get("candidate_id") == candidate_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有候选证据 {candidate_id}；请先运行单变量分析，"
            "再使用结果中展示的 candidate ID 和 source bin id。"
        )
    if len(matches) > 1:
        raise StrategySetupError(
            f"候选证据 {candidate_id} 对应多个不可变 JSON artifact，"
            "当前不能安全选择来源。"
        )
    artifact = matches[0]
    provenance = artifact["provenance"]
    content_hash = artifact.get("content_hash")
    evidence_hash = provenance.get("evidence_hash")
    artifact_id = artifact.get("id")
    if (
        not isinstance(artifact_id, str)
        or not artifact_id
        or not isinstance(content_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
        or not isinstance(evidence_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", evidence_hash) is None
    ):
        raise StrategySetupError(
            f"候选证据 {candidate_id} 的 artifact 绑定不完整，请重新生成单变量分析。"
        )
    return {
        "source_artifact_id": artifact_id,
        "expected_artifact_content_hash": content_hash,
        "expected_candidate_id": candidate_id,
        "expected_evidence_hash": evidence_hash,
    }


def _automatic_tree_leaf_materialization_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind one explicit leaf request to one verified task-owned full tree."""

    inputs = draft.to_dict()["workflow_inputs"]
    asset_id = inputs.get("tree_asset_id")
    leaf_id = inputs.get("leaf_id")
    if not isinstance(asset_id, str) or not isinstance(leaf_id, str):
        raise StrategySetupError(
            "自动树叶节点物化必须提供完整 tree asset ID 和 leaf ID。"
        )

    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        artifacts = repository.list_for_task(task_id)
    except Exception as exc:
        raise StrategySetupError(
            "当前任务的自动树 artifact registry 无法读取，不能安全绑定来源。"
        ) from exc
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategySetupError("当前任务的自动树 artifact 记录结构无效。")
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND
            and artifact.get("origin_tool")
            == AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("asset_id") == asset_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有自动树资产 {asset_id}；请使用构建结果中展示的完整 "
            "candidate-asset ID。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"自动树资产 {asset_id} 对应多个 full-tree JSON artifact，"
            "当前不能安全选择来源。"
        )

    artifact = matches[0]
    provenance = artifact.get("provenance")
    assert isinstance(provenance, Mapping)
    artifact_id = artifact.get("id")
    content_hash = artifact.get("content_hash")
    asset_hash = provenance.get("asset_hash")
    tree_result_hash = provenance.get("tree_result_hash")
    if not all(
        isinstance(value, str) and value
        for value in (
            artifact_id,
            content_hash,
            asset_hash,
            tree_result_hash,
        )
    ):
        raise StrategySetupError(
            f"自动树资产 {asset_id} 的 artifact 完整性绑定不完整，请重新构建。"
        )

    try:
        with repository.transaction() as conn:
            verified = load_verified_automatic_tree_source_artifact_on_connection(
                conn,
                tasks_dir=runtime.settings.tasks_dir,
                task_id=task_id,
                artifact_id=artifact_id,
                expected_content_hash=content_hash,
                expected_asset_id=asset_id,
                expected_asset_hash=asset_hash,
                expected_tree_result_hash=tree_result_hash,
            )
    except Exception as exc:
        raise StrategySetupError(
            f"自动树资产 {asset_id} 未通过 artifact 完整性校验，请重新构建。"
        ) from exc

    fragments = verified.asset.get("fragments")
    if isinstance(fragments, Sequence) and not isinstance(
        fragments, str | bytes | bytearray
    ):
        leaf_matches = [
            fragment
            for fragment in fragments
            if isinstance(fragment, Mapping) and fragment.get("leaf_id") == leaf_id
        ]
    else:
        leaf_matches = []
    if len(leaf_matches) != 1:
        raise StrategySetupError(
            f"自动树资产 {asset_id} 中没有唯一匹配的叶节点 {leaf_id}；"
            "请从完整叶节点清单中复制准确 leaf ID。"
        )

    slots: dict[str, object] = {
        "source_artifact_id": verified.artifact_id,
        "expected_artifact_content_hash": verified.content_hash,
        "expected_asset_id": verified.asset["asset_id"],
        "expected_asset_hash": verified.asset["asset_hash"],
        "expected_tree_result_hash": verified.asset["tree_result"]["result_hash"],
        "leaf_id": leaf_id,
    }
    if "selection_reason" in inputs:
        slots["selection_reason"] = inputs["selection_reason"]
    return slots


def _automatic_tree_apply_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    draft: StandardWorkflowRequestDraft,
    context,
) -> dict[str, object]:
    """Bind full-tree writeback to its exact task-owned source workspace."""

    inputs = draft.to_dict()["workflow_inputs"]
    asset_id = inputs.get("tree_asset_id")
    if not isinstance(asset_id, str):
        raise StrategySetupError(
            "自动树全量写回必须提供完整 tree asset ID。"
        )

    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        artifacts = repository.list_for_task(task_id)
    except Exception as exc:
        raise StrategySetupError(
            "当前任务的自动树 artifact registry 无法读取，不能安全绑定来源。"
        ) from exc
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategySetupError("当前任务的自动树 artifact 记录结构无效。")
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND
            and artifact.get("origin_tool")
            == AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("asset_id") == asset_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有自动树资产 {asset_id}；请使用构建结果中展示的完整 "
            "candidate-asset ID。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"自动树资产 {asset_id} 对应多个 full-tree JSON artifact，"
            "当前不能安全选择来源。"
        )

    artifact = matches[0]
    provenance = artifact.get("provenance")
    assert isinstance(provenance, Mapping)
    artifact_id = artifact.get("id")
    content_hash = artifact.get("content_hash")
    asset_hash = provenance.get("asset_hash")
    tree_result_hash = provenance.get("tree_result_hash")
    if not all(
        isinstance(value, str) and value
        for value in (artifact_id, content_hash, asset_hash, tree_result_hash)
    ):
        raise StrategySetupError(
            f"自动树资产 {asset_id} 的 artifact 完整性绑定不完整，请重新构建。"
        )

    try:
        with repository.transaction() as conn:
            verified = load_verified_automatic_tree_source_artifact_on_connection(
                conn,
                tasks_dir=runtime.settings.tasks_dir,
                task_id=task_id,
                artifact_id=artifact_id,
                expected_content_hash=content_hash,
                expected_asset_id=asset_id,
                expected_asset_hash=asset_hash,
                expected_tree_result_hash=tree_result_hash,
            )
    except Exception as exc:
        raise StrategySetupError(
            f"自动树资产 {asset_id} 未通过 artifact 完整性校验，请重新构建。"
        ) from exc

    identity = verified.asset.get("identity")
    if not isinstance(identity, Mapping):
        raise StrategySetupError(
            f"自动树资产 {asset_id} 缺少原始样本 lineage，请重新构建。"
        )
    try:
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task_id)
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前活动 DataWorkspace 无法验证，不能执行自动树写回。"
        ) from exc

    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    expected_binding = {
        "task_id": task_id,
        "dataset_id": getattr(context, "dataset_id", None),
        "dataset_content_hash": getattr(context, "dataset_content_hash", None),
        "workspace_revision": getattr(context, "workspace_revision", None),
        "workspace_generation": getattr(context, "analysis_generation", None),
        "semantic_mapping_hash": getattr(context, "semantic_mapping_hash", None),
    }
    live_binding = {
        "task_id": task_id,
        "dataset_id": workspace.active_dataset_id,
        "dataset_content_hash": workspace.active_dataset_content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
    }
    if any(identity.get(field) != value for field, value in expected_binding.items()):
        raise StrategySetupError(
            f"自动树资产 {asset_id} 只允许写回构建时绑定的原始样本；"
            "当前策略数据上下文已发生变化，请重新构建或切回原始 workspace。"
        )
    if live_binding != expected_binding:
        raise StrategySetupError(
            "当前 DataWorkspace 在自动树写回绑定期间发生变化；本次未执行。"
        )

    output_columns = {
        "leaf_id_column": inputs.get(
            "leaf_id_column", "automatic_tree_leaf_id"
        ),
        "rule_id_column": inputs.get(
            "rule_id_column", "automatic_tree_rule_id"
        ),
    }
    folded_source_columns = {
        str(column).casefold() for column in getattr(context, "columns", ())
    }
    if any(
        not isinstance(column, str)
        or column.casefold() in folded_source_columns
        for column in output_columns.values()
    ):
        raise StrategySetupError(
            "自动树写回输出列与当前样本已有字段冲突，请指定新的叶节点列和规则列。"
        )

    slots: dict[str, object] = {
        "source_artifact_id": verified.artifact_id,
        "expected_artifact_content_hash": verified.content_hash,
        "expected_asset_id": verified.asset["asset_id"],
        "expected_asset_hash": verified.asset["asset_hash"],
        "expected_tree_result_hash": verified.asset["tree_result"]["result_hash"],
        "dataset_id": identity["dataset_id"],
        "expected_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "analysis_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
    }
    for field in ("leaf_id_column", "rule_id_column"):
        if field in inputs:
            slots[field] = inputs[field]
    return slots


def _cross_matrix_cell_selection_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind exact user cell ids to one verified task-owned full matrix."""

    inputs = draft.to_dict()["workflow_inputs"]
    asset_id = inputs.get("cross_asset_id")
    cell_ids = inputs.get("cell_ids")
    if (
        not isinstance(asset_id, str)
        or not isinstance(cell_ids, Sequence)
        or isinstance(cell_ids, str | bytes | bytearray)
        or any(not isinstance(cell_id, str) for cell_id in cell_ids)
    ):
        raise StrategySetupError(
            "Cross Matrix 单元格选择必须提供完整 cross asset ID 和 cell ID 列表。"
        )

    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        artifacts = repository.list_for_task(task_id)
    except Exception as exc:
        raise StrategySetupError(
            "当前任务的 Cross Matrix artifact registry 无法读取，不能安全绑定来源。"
        ) from exc
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategySetupError("当前任务的 Cross Matrix artifact 记录结构无效。")
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == CROSS_MATRIX_SOURCE_ARTIFACT_KIND
            and artifact.get("origin_tool")
            == CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("schema_version")
            == CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION
            and provenance.get("asset_id") == asset_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有完整 Cross Matrix 资产 {asset_id}；请使用构建结果中"
            "展示的完整 candidate-asset ID。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 对应多个 full-matrix JSON artifact，"
            "当前不能安全选择来源。"
        )

    artifact = matches[0]
    provenance = artifact.get("provenance")
    assert isinstance(provenance, Mapping)
    artifact_id = artifact.get("id")
    content_hash = artifact.get("content_hash")
    asset_hash = provenance.get("asset_hash")
    candidate_id = provenance.get("candidate_id")
    evidence_hash = provenance.get("evidence_hash")
    if not all(
        isinstance(value, str) and value
        for value in (
            artifact_id,
            content_hash,
            asset_hash,
            candidate_id,
            evidence_hash,
        )
    ):
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 的 artifact 完整性绑定不完整，请重新构建。"
        )

    try:
        with repository.transaction() as conn:
            verified = load_verified_cross_matrix_source_artifact_on_connection(
                conn,
                tasks_dir=runtime.settings.tasks_dir,
                task_id=task_id,
                artifact_id=artifact_id,
                expected_content_hash=content_hash,
                expected_asset_id=asset_id,
                expected_asset_hash=asset_hash,
                expected_candidate_id=candidate_id,
                expected_evidence_hash=evidence_hash,
            )
    except Exception as exc:
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 未通过 artifact 完整性校验，请重新构建。"
        ) from exc

    matrix = verified.asset.get("matrix")
    cells = matrix.get("cells") if isinstance(matrix, Mapping) else None
    if not isinstance(cells, Sequence) or isinstance(cells, str | bytes | bytearray):
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 缺少完整 cell 清单，请重新构建。"
        )
    source_cell_ids = [
        cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)
    ]
    if (
        len(source_cell_ids) != len(cells)
        or len(set(source_cell_ids)) != len(source_cell_ids)
        or any(source_cell_ids.count(cell_id) != 1 for cell_id in cell_ids)
    ):
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 中无法唯一匹配全部 cell ID；"
            "请从完整单元格清单中复制准确 ID。"
        )
    requested = set(cell_ids)
    ordered_cell_ids = [cell_id for cell_id in source_cell_ids if cell_id in requested]
    if len(ordered_cell_ids) != len(cell_ids):
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 中无法唯一匹配全部 cell ID；"
            "请从完整单元格清单中复制准确 ID。"
        )

    evidence = verified.asset.get("candidate_evidence")
    if not isinstance(evidence, Mapping):
        raise StrategySetupError(
            f"Cross Matrix 资产 {asset_id} 缺少候选证据绑定，请重新构建。"
        )
    slots: dict[str, object] = {
        "source_artifact_id": verified.artifact_id,
        "expected_artifact_content_hash": verified.content_hash,
        "expected_asset_id": verified.asset["asset_id"],
        "expected_asset_hash": verified.asset["asset_hash"],
        "expected_candidate_id": evidence["candidate_id"],
        "expected_evidence_hash": evidence["evidence_hash"],
        "cell_ids": ordered_cell_ids,
    }
    if "selection_reason" in inputs:
        slots["selection_reason"] = inputs["selection_reason"]
    return slots


def _strategy_voting_candidate_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind explicit user rule ids to one exact current Pool snapshot."""

    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = inputs.get("strategy_type")
    rule_ids = inputs.get("rule_ids")
    n = inputs.get("n")
    if (
        not isinstance(strategy_type, str)
        or not isinstance(rule_ids, Sequence)
        or isinstance(rule_ids, str | bytes | bytearray)
        or isinstance(n, bool)
        or not isinstance(n, int)
    ):
        raise StrategySetupError(
            "Voting 候选需要明确的策略池类型、完整 rule_id 列表和整数 n。"
        )
    try:
        current = StrategyCandidatePoolRepository(
            runtime.settings.db_path
        ).get_current(task.id, strategy_type)
    except Exception as exc:
        raise StrategySetupError(
            "当前 Strategy Pool 状态无法通过完整性校验，不能构建 Voting 候选。"
        ) from exc
    if current is None:
        raise StrategySetupError(
            f"当前任务没有 {strategy_type} Strategy Pool；请先把至少两条候选规则加入池中。"
        )
    try:
        entries = _strategy_pool_entries(current)
        revision = int(current["revision"])
        snapshot_hash = strategy_pool_snapshot_hash(current)
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前 Strategy Pool revision、entries 或 hash 绑定不完整。"
        ) from exc

    selected: list[Mapping] = []
    for rule_id in rule_ids:
        matches = [entry for entry in entries if entry.get("rule_id") == rule_id]
        if len(matches) != 1:
            raise StrategySetupError(
                f"当前 Strategy Pool 中没有唯一匹配的规则 {rule_id}；"
                "请从最新 Pool 结果复制完整 rule_id。"
            )
        entry = matches[0]
        if entry.get("enabled") is not True:
            raise StrategySetupError(f"规则 {rule_id} 当前未启用，不能参与 Voting。")
        source = entry.get("source")
        if isinstance(source, Mapping) and source.get("asset_type") == "voting_n_of_k":
            raise StrategySetupError(
                "当前版本先拒绝嵌套 Voting 候选，以避免递归 lineage 或循环依赖；"
                "请选择原始单变量或自动树叶规则。"
            )
        selected.append(entry)
    selected.sort(key=lambda item: int(item.get("position", -1)))
    if not 2 <= len(selected) <= 50 or not 1 <= n <= len(selected):
        raise StrategySetupError("Voting 候选要求 2 到 50 条规则，且 n 必须位于 1 到 K。")
    return {
        "strategy_type": strategy_type,
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
        "selected_entry_ids": [str(entry["entry_id"]) for entry in selected],
        "n": n,
    }


def _strategy_pool_impact_pool_binding(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    strategy_type: str,
) -> tuple[Mapping, dict[str, object]]:
    """Load one non-empty Pool and return its exact confirmation binding."""

    if strategy_type not in {"approval", "reject"}:
        raise StrategySetupError(
            "Strategy Pool 影响测算首个 V2 纵切只支持 approval/reject；"
            "其他策略类型需要后续类型专属口径。"
        )
    try:
        pool = StrategyCandidatePoolRepository(
            runtime.settings.db_path
        ).get_current(task.id, strategy_type)
    except Exception as exc:
        raise StrategySetupError(
            "当前 Strategy Pool 状态无法通过完整性校验，不能执行影响测算。"
        ) from exc
    if pool is None:
        raise StrategySetupError(
            f"当前任务没有 {strategy_type} Strategy Pool，无法测算影响。"
        )
    if not _strategy_pool_entries(pool):
        raise StrategySetupError(
            f"当前 {strategy_type} Strategy Pool 为空；请先加入候选规则再测算影响。"
        )
    try:
        binding = {
            "strategy_type": strategy_type,
            "expected_pool_revision": int(pool["revision"]),
            "expected_pool_snapshot_hash": strategy_pool_snapshot_hash(pool),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前 Strategy Pool revision/hash 绑定不完整，不能执行影响测算。"
        ) from exc
    return pool, binding


def _strategy_sample_design_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
    drop_nan_labels: bool,
) -> dict[str, object]:
    """Bind user-owned sample facts to the exact active workspace snapshot."""

    try:
        workspace = _require_strategy_sample_design_workspace(runtime, task)
    except StrategySetupError:
        raise
    if (
        workspace.active_dataset_id != context.dataset_id
        or workspace.active_dataset_content_hash != context.dataset_content_hash
        or workspace.revision != context.workspace_revision
        or workspace.analysis_generation != context.analysis_generation
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 在样本设计计划创建前发生变化；请基于当前版本重试。"
        )
    target_col = workspace.semantic_mapping.target_col
    if (
        not isinstance(target_col, str)
        or not target_col
        or target_col != context.target_col
        or target_col not in context.columns
    ):
        raise StrategySetupError(
            "策略样本设计只能使用 DataWorkspace 中已确认的二元目标列。"
        )
    content_hash = workspace.active_dataset_content_hash
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if not isinstance(content_hash, str) or not content_hash:
        raise StrategySetupError("活动数据集缺少内容 hash，不能固化样本设计。")
    if semantic_hash != context.semantic_mapping_hash:
        raise StrategySetupError(
            "活动 DataWorkspace 的语义映射已变化；请重新发起样本设计。"
        )

    inputs = draft.to_dict()["workflow_inputs"]
    for field in (
        "split_col",
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        column = inputs.get(field)
        if column is not None and column not in context.columns:
            raise StrategySetupError(
                f"策略样本设计显式字段 {field} 不在当前活动数据集中。"
            )
    slots: dict[str, object] = {
        "dataset_id": workspace.active_dataset_id,
        "expected_dataset_content_hash": content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "target_col": target_col,
        "drop_nan_labels": bool(drop_nan_labels),
    }
    slots.update(
        {
            key: value
            for key, value in inputs.items()
            if key != "drop_nan_labels"
        }
    )
    return slots


def _latest_matching_strategy_sample_design_ref(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    context,
    drop_nan_labels: bool,
    month_col: str | None = None,
    weight_col: str | None = None,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
) -> dict[str, str]:
    """Bind downstream execution to the newest exact governed sample design.

    The language model never supplies artifact ids or hashes.  Selection is
    deterministic over task-owned registry rows and only considers designs
    whose immutable provenance already matches the active data/workspace/label
    boundary.  The selected artifact is then fully reloaded and authenticated.
    """

    if not isinstance(context.target_col, str) or not context.target_col:
        raise StrategySetupError(
            "策略开发需要先在 DataWorkspace 中确认二元目标列。"
        )
    expected = {
        "task_id": task.id,
        "dataset_id": context.dataset_id,
        "dataset_content_hash": context.dataset_content_hash,
        "workspace_revision": context.workspace_revision,
        "workspace_generation": context.analysis_generation,
        "semantic_mapping_hash": context.semantic_mapping_hash,
        "target_col": context.target_col,
    }
    matches: list[Mapping] = []
    try:
        artifacts = TaskArtifactRepository(
            runtime.settings.db_path
        ).list_for_task(task.id)
    except Exception as exc:
        raise StrategySetupError(
            "无法读取当前任务的策略样本设计登记，不能安全继续策略开发。"
        ) from exc
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") != SAMPLE_DESIGN_ARTIFACT_KIND
            or artifact.get("origin_tool") != SAMPLE_DESIGN_ORIGIN_TOOL
            or not isinstance(provenance, Mapping)
            or any(provenance.get(field) != value for field, value in expected.items())
        ):
            continue
        request = provenance.get("request")
        if (
            not isinstance(request, Mapping)
            or request.get("drop_nan_labels") is not bool(drop_nan_labels)
        ):
            continue
        matches.append(artifact)
    if not matches:
        raise _StrategySampleDesignRequiredError(
            "当前活动数据和标签口径没有可执行的成熟策略样本设计。"
            "请先用自然语言说明坏样本值、表现窗、观察窗、成熟度及可选切分，"
            "让 MARVIS 固化样本设计。"
        )

    artifact = matches[-1]
    provenance = artifact["provenance"]
    reference = {
        "artifact_id": artifact.get("id"),
        "artifact_content_hash": artifact.get("content_hash"),
        "sample_design_id": provenance.get("sample_design_id"),
        "sample_design_content_hash": provenance.get(
            "sample_design_content_hash"
        ),
        "partition": "development",
    }
    backend = DataBackend(runtime.settings.datasets_dir)
    read_runtime = SimpleNamespace(
        settings=runtime.settings,
        registry=DatasetRegistry(
            DatasetRepository(runtime.settings.db_path),
            backend,
            runtime.settings.datasets_dir,
        ),
        task_artifacts=TaskArtifactRepository(runtime.settings.db_path),
    )
    try:
        binding = load_strategy_sample_design_execution_binding(
            read_runtime,
            task_id=task.id,
            sample_design_ref=reference,
            dataset_id=context.dataset_id,
            dataset_content_hash=context.dataset_content_hash,
            workspace_revision=context.workspace_revision,
            workspace_generation=context.analysis_generation,
            semantic_mapping_hash=context.semantic_mapping_hash,
            target_col=context.target_col,
            drop_nan_labels=bool(drop_nan_labels),
            month_col=month_col,
            weight_col=weight_col,
            loan_amount_col=loan_amount_col,
            overdue_amount_col=overdue_amount_col,
        )
    except StrategyError as exc:
        raise StrategySetupError(
            "当前最新策略样本设计未通过完整性、成熟度或字段口径校验；"
            "请重新固化样本设计后再执行。"
        ) from exc
    return binding.to_ref_dict()


def _strategy_pool_impact_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
    drop_nan_labels: bool,
    expected_pool_binding: Mapping | None = None,
) -> dict[str, object]:
    """Bind one read-only impact request to exact Pool and workspace evidence."""

    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = str(inputs.get("strategy_type") or "")
    pool, pool_binding = _strategy_pool_impact_pool_binding(
        runtime,
        task,
        strategy_type,
    )
    if expected_pool_binding is not None and dict(expected_pool_binding) != pool_binding:
        raise StrategySetupError(
            "Strategy Pool 在用户确认期间已变化；旧确认不会绑定新的 Pool revision，"
            "请基于当前 Pool 重新发起影响测算。"
        )
    entries = _strategy_pool_entries(pool)

    try:
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task.id)
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "Strategy Pool 影响测算需要有效的活动 DataWorkspace。"
        ) from exc
    if workspace.active_dataset_id is None:
        raise StrategySetupError(
            "Strategy Pool 影响测算要求先在 DataWorkspace 选择活动数据集；"
            "不会从 source_dir 或多个样本中猜测。"
        )
    if (
        workspace.active_dataset_id != context.dataset_id
        or workspace.active_dataset_content_hash != context.dataset_content_hash
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 与策略数据上下文不一致，请重新选择活动数据集后重试。"
        )
    target_col = workspace.semantic_mapping.target_col
    if (
        not isinstance(target_col, str)
        or not target_col
        or target_col not in context.columns
        or target_col != context.target_col
    ):
        raise StrategySetupError(
            "Strategy Pool 影响测算只能使用 DataWorkspace 中已确认的二元 target；"
            "不会采用 LLM、任务旧字段或列名猜测。"
        )
    content_hash = workspace.active_dataset_content_hash
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if not isinstance(content_hash, str) or not content_hash:
        raise StrategySetupError("活动数据集缺少内容 hash，不能绑定影响测算。")
    sample_identities: list[dict] = []
    for entry in entries:
        source = entry.get("source")
        evidence_identity = (
            source.get("evidence_identity")
            if isinstance(source, Mapping)
            else None
        )
        if not isinstance(evidence_identity, Mapping):
            raise StrategySetupError(
                "当前 Strategy Pool 条目缺少受治理样本身份，不能执行影响测算。"
            )
        sample_identities.append(dict(evidence_identity))
    if any(identity != sample_identities[0] for identity in sample_identities[1:]):
        raise StrategySetupError(
            "当前 Strategy Pool 条目并非来自同一受治理样本，不能执行影响测算。"
        )
    expected_sample = {
        "dataset_id": workspace.active_dataset_id,
        "dataset_content_hash": content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
    }
    if any(
        sample_identities[0].get(field) != expected
        for field, expected in expected_sample.items()
    ):
        raise StrategySetupError(
            "当前活动 DataWorkspace 与 Strategy Pool 创建时绑定的样本或语义版本不同；"
            "请切回该 Pool 的绑定数据，或基于当前数据重建候选与 Pool 后再测算。"
        )

    comparison_mode = str(inputs.get("comparison_mode") or "absolute")
    slots: dict[str, object] = {
        "strategy_type": strategy_type,
        "expected_pool_revision": pool_binding["expected_pool_revision"],
        "expected_pool_snapshot_hash": pool_binding[
            "expected_pool_snapshot_hash"
        ],
        "dataset_id": workspace.active_dataset_id,
        "expected_dataset_content_hash": content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "target_col": target_col,
        "comparison_mode": comparison_mode,
        "drop_nan_labels": bool(drop_nan_labels),
    }
    for field, role in (
        ("month_col", "month"),
        ("loan_amount_col", "loan_amount"),
        ("overdue_amount_col", "overdue_amount"),
    ):
        column = _strategy_pool_impact_column(
            inputs,
            field=field,
            role=role,
            columns=tuple(context.columns),
            field_roles=workspace.semantic_mapping.field_roles,
        )
        if column is not None:
            slots[field] = column

    slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
        runtime,
        task,
        context=context,
        drop_nan_labels=bool(drop_nan_labels),
        month_col=slots.get("month_col"),
        loan_amount_col=slots.get("loan_amount_col"),
        overdue_amount_col=slots.get("overdue_amount_col"),
    )

    baseline_strategy_id = inputs.get("baseline_strategy_id")
    if comparison_mode == "vs_baseline":
        if not isinstance(baseline_strategy_id, str) or not baseline_strategy_id:
            raise StrategySetupError(
                "相对基线测算需要用户明确提供完整 baseline_strategy_id。"
            )
        repository = StrategyRepository(runtime.settings.db_path)
        try:
            baseline_meta = repository.get_strategy_meta(baseline_strategy_id)
            baseline = repository.get_strategy(baseline_strategy_id)
            baseline_hash = repository.get_strategy_spec_hash(baseline_strategy_id)
        except Exception as exc:
            raise StrategySetupError(
                "基线策略的 canonical StrategySpec 无法通过完整性校验。"
            ) from exc
        if (
            baseline_meta is None
            or baseline is None
            or baseline.spec is None
            or not isinstance(baseline_hash, str)
            or baseline_meta.get("task_id") != task.id
        ):
            raise StrategySetupError(
                "当前任务中没有带 canonical StrategySpec 的该基线策略，"
                "不能跨任务或用不完整策略做对比。"
            )
        if (
            baseline_meta.get("strategy_type") != strategy_type
            or baseline.strategy_type != strategy_type
        ):
            raise StrategySetupError(
                "baseline_strategy_id 的策略类型与当前 Strategy Pool 不一致。"
            )
        slots["baseline_strategy_id"] = baseline_strategy_id
    elif baseline_strategy_id is not None:
        raise StrategySetupError("absolute 影响测算禁止绑定 baseline_strategy_id。")
    return slots


def _strategy_pool_impact_column(
    inputs: Mapping,
    *,
    field: str,
    role: str,
    columns: tuple[str, ...],
    field_roles: Mapping,
) -> str | None:
    """Prefer an explicit validated column, else require a unique semantic role."""

    explicit = inputs.get(field)
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in columns:
            raise StrategySetupError(
                f"影响测算显式字段 {field} 不在当前活动数据集中。"
            )
        return explicit
    matches = [
        column
        for column, assigned_role in field_roles.items()
        if assigned_role == role and column in columns
    ]
    if len(matches) > 1:
        raise StrategySetupError(
            f"DataWorkspace 有多个 `{role}` 语义字段：{'、'.join(sorted(matches))}；"
            f"请在请求中明确指定 {field}，平台不会任意选择。"
        )
    return matches[0] if matches else None


def _strategy_pool_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict:
    """Resolve all Pool integrity inputs from task-owned state, never the LLM."""

    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = str(inputs["strategy_type"])
    try:
        current = StrategyCandidatePoolRepository(
            runtime.settings.db_path
        ).get_current(task.id, strategy_type)
    except Exception as exc:
        raise StrategySetupError(
            "当前 Strategy Pool 状态无法通过完整性校验，请先检查任务数据。"
        ) from exc

    if current is None:
        expected_revision = ABSENT_POOL_REVISION
        expected_snapshot_hash = ABSENT_POOL_SNAPSHOT_HASH
    else:
        try:
            expected_revision = int(current["revision"])
            expected_snapshot_hash = strategy_pool_snapshot_hash(current)
        except (KeyError, TypeError, ValueError) as exc:
            raise StrategySetupError(
                "当前 Strategy Pool revision/hash 绑定不完整，不能继续操作。"
            ) from exc

    slots: dict = {
        "strategy_type": strategy_type,
        "expected_pool_revision": expected_revision,
        "expected_pool_snapshot_hash": expected_snapshot_hash,
    }
    if "reason" in inputs:
        slots["reason"] = inputs["reason"]

    if draft.workflow == "strategy_pool_add_candidate":
        selection_id = inputs.get("selection_id")
        candidate_asset_id = inputs.get("candidate_asset_id")
        if (selection_id is None) == (candidate_asset_id is None):
            raise StrategySetupError(
                "加入 Strategy Pool 必须且只能指定一个 candidate asset ID "
                "或受支持的精确 selection ID。"
            )
        fragment_id: str | None = None
        is_voting_candidate = False
        if selection_id is not None:
            selection_slots, fragment_id = _candidate_selection_artifact_slots(
                runtime,
                task_id=task.id,
                selection_id=str(selection_id),
            )
            slots.update(selection_slots)
        else:
            candidate_slots = _candidate_asset_artifact_slots(
                runtime,
                task_id=task.id,
                asset_id=str(candidate_asset_id),
            )
            is_voting_candidate = (
                candidate_slots.pop("_candidate_asset_type", None) == "voting_n_of_k"
            )
            slots.update(candidate_slots)
        requested_placement = inputs.get("placement_mode")
        if is_voting_candidate:
            if requested_placement not in {
                "before_selected_members",
                "replace_selected_members",
            }:
                raise StrategySetupError(
                    "Voting 候选入池前必须明确选择：保留成员作为未达 n 时的"
                    "后续规则（before_selected_members），或由 Voting 原子替代"
                    "这些成员（replace_selected_members）。"
                )
            slots["placement_mode"] = requested_placement
        else:
            if requested_placement is not None:
                raise StrategySetupError(
                    "placement_mode 仅适用于 Voting 候选；普通候选保持追加语义。"
                )
            slots["placement_mode"] = "append"
        slots["default_action"] = inputs["default_action"]
        slots["action"] = inputs["action"]
        if current is not None and current.get("default_action") != inputs["default_action"]:
            raise StrategySetupError(
                "请求中的 default_action 与当前 Strategy Pool 不一致；"
                "不能在添加条目时静默改写 Pool 默认动作。"
            )
        asset_id = slots["expected_asset_id"]
        if current is not None:
            entries = _strategy_pool_entries(current)
            if fragment_id is None and any(
                isinstance(entry.get("source"), Mapping)
                and entry["source"].get("asset_id") == asset_id
                for entry in entries
            ):
                raise StrategySetupError(
                    f"候选资产 {asset_id} 已存在于当前 Strategy Pool。"
                )
            if fragment_id is not None and any(
                isinstance(entry.get("source"), Mapping)
                and entry["source"].get("asset_id") == asset_id
                and entry["source"].get("fragment_id") == fragment_id
                for entry in entries
            ):
                raise StrategySetupError(
                    f"候选资产 {asset_id} 的片段 {fragment_id} "
                    "已存在于当前 Strategy Pool。"
                )
        # The governed Pool kernel owns exact asset/fragment/rule uniqueness.
        # In particular, one automatic tree may contribute multiple distinct
        # leaves, so an asset-id-only preflight would reject valid requests.
        return slots

    if draft.workflow == "strategy_pool_compile":
        if current is None:
            raise StrategySetupError(
                "当前任务还没有该类型的 Strategy Pool，无法编译预览。"
            )
        return slots
    if current is None:
        raise StrategySetupError("当前任务还没有该类型的 Strategy Pool，无法执行此操作。")

    if draft.workflow in {
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
    }:
        identifier = inputs.get("rule_id") or inputs.get("entry_id")
        slots["rule_id"] = _strategy_pool_rule_id(current, str(identifier))
        if draft.workflow == "strategy_pool_set_action":
            slots["action"] = inputs["action"]
        return slots

    if draft.workflow == "strategy_pool_reorder":
        slots["ordered_rule_ids"] = _strategy_pool_complete_rule_order(
            current,
            inputs["ordered_ids"],
        )
        return slots
    raise StrategySetupError(f"未接线的 Strategy Pool Workflow：{draft.workflow}")


def _candidate_asset_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    asset_id: str,
) -> dict[str, str]:
    matches = []
    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        artifacts = repository.list_for_task(task_id)
    except Exception as exc:
        raise StrategySetupError(
            "当前任务的候选资产 artifact registry 无法读取，不能安全绑定来源。"
        ) from exc
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategySetupError("当前任务的候选资产 artifact 记录结构无效。")
        provenance = artifact.get("provenance")
        artifact_triple = (artifact.get("kind"), artifact.get("origin_tool"))
        if (
            artifact_triple
            == (
                CROSS_MATRIX_SOURCE_ARTIFACT_KIND,
                CROSS_MATRIX_SOURCE_ARTIFACT_ORIGIN_TOOL,
            )
            and isinstance(provenance, Mapping)
            and provenance.get("schema_version")
            == CROSS_MATRIX_SOURCE_ARTIFACT_SCHEMA_VERSION
            and provenance.get("asset_id") == asset_id
        ):
            raise StrategySetupError(
                "完整 Cross Matrix asset 不能直接加入 Strategy Pool；请先在"
                "单独一轮精确选择 cell，再使用 cross-matrix-cell-selection ID 入池。"
            )
        supported_triples = {
            (
                "strategy_candidate_asset_json",
                "strategy.refine_univariate_candidate",
            ),
            (VOTING_CANDIDATE_ARTIFACT_KIND, VOTING_CANDIDATE_ORIGIN_TOOL),
        }
        if (
            artifact_triple in supported_triples
            and isinstance(provenance, Mapping)
            and provenance.get("asset_id") == asset_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有候选资产 {asset_id}；请使用结果中展示的完整 candidate-asset ID。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"候选资产 {asset_id} 对应多个 artifact，当前不能安全选择来源。"
        )
    artifact = matches[0]
    artifact_id = artifact.get("id")
    content_hash = artifact.get("content_hash")
    path_value = artifact.get("path")
    provenance = artifact.get("provenance")
    asset_hash = provenance.get("asset_hash") if isinstance(provenance, dict) else None
    if (
        not isinstance(artifact_id, str)
        or not artifact_id
        or not isinstance(content_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
        or not isinstance(asset_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", asset_hash) is None
        or not isinstance(path_value, str)
        or not path_value
    ):
        raise StrategySetupError(
            f"候选资产 {asset_id} 的 artifact 完整性绑定不完整，请重新生成。"
        )
    if (
        artifact.get("kind") == VOTING_CANDIDATE_ARTIFACT_KIND
        and artifact.get("origin_tool") == VOTING_CANDIDATE_ORIGIN_TOOL
    ):
        try:
            with repository.transaction() as conn:
                verified = load_verified_voting_candidate_artifact_on_connection(
                    conn,
                    tasks_dir=runtime.settings.tasks_dir,
                    task_id=task_id,
                    artifact_id=artifact_id,
                    expected_content_hash=content_hash,
                    expected_asset_id=asset_id,
                    expected_asset_hash=asset_hash,
                )
        except Exception as exc:
            raise StrategySetupError(
                f"Voting 候选资产 {asset_id} 无法通过 artifact 完整性校验，"
                "请重新生成。"
            ) from exc
        return {
            "source_artifact_id": verified.artifact_id,
            "expected_artifact_content_hash": verified.content_hash,
            "expected_asset_id": verified.asset["asset_id"],
            "expected_asset_hash": verified.asset["asset_hash"],
            "_candidate_asset_type": "voting_n_of_k",
        }
    path = Path(path_value)
    try:
        content = path.read_bytes()
        if sha256_file(path) != content_hash:
            raise StrategySetupError(
                f"候选资产 {asset_id} 的 artifact 内容已漂移，不能加入 Strategy Pool。"
            )
        payload = json.loads(content)
        normalized_asset = validate_candidate_asset(payload)
        if canonical_candidate_asset_json(normalized_asset).encode("utf-8") != content:
            raise StrategySetupError(
                f"候选资产 {asset_id} 不是 canonical JSON，不能加入 Strategy Pool。"
            )
    except StrategySetupError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            f"候选资产 {asset_id} 无法通过 artifact 校验，请重新生成。"
        ) from exc
    if (
        normalized_asset.get("asset_id") != asset_id
        or normalized_asset.get("asset_hash") != asset_hash
    ):
        raise StrategySetupError(
            f"候选资产 {asset_id} 与 artifact provenance 不一致，不能加入 Strategy Pool。"
        )
    return {
        "source_artifact_id": artifact_id,
        "expected_artifact_content_hash": content_hash,
        "expected_asset_id": asset_id,
        "expected_asset_hash": asset_hash,
        "_candidate_asset_type": "univariate_refinement",
    }


def _automatic_tree_leaf_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Bind one exact selection ID to four verified Pool Tool slots."""

    if re.fullmatch(
        r"automatic-tree-leaf-selection-[0-9a-f]{32}", selection_id
    ) is None:
        raise StrategySetupError(
            "automatic-tree leaf selection ID 格式无效；请复制完整 selection ID。"
        )
    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        with repository.transaction() as conn:
            conn.execute("BEGIN")
            rows = conn.execute(
                """
                SELECT id, content_hash, provenance_json
                  FROM task_artifacts
                 WHERE task_id = ? AND kind = ? AND origin_tool = ?
                """,
                (
                    task_id,
                    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
                    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
                ),
            ).fetchall()
            matches: list[tuple[object, Mapping]] = []
            for row in rows:
                provenance_json = row["provenance_json"]
                if not isinstance(provenance_json, str):
                    raise StrategySetupError(
                        "当前任务的 leaf selection artifact provenance 无效。"
                    )
                try:
                    provenance = json.loads(provenance_json)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise StrategySetupError(
                        "当前任务的 leaf selection artifact provenance 无效。"
                    ) from exc
                if not isinstance(provenance, Mapping):
                    raise StrategySetupError(
                        "当前任务的 leaf selection artifact provenance 无效。"
                    )
                if provenance.get("selection_id") == selection_id:
                    matches.append((row, provenance))
            if not matches:
                raise StrategySetupError(
                    f"当前任务没有 automatic-tree leaf selection {selection_id}。"
                )
            if len(matches) != 1:
                raise StrategySetupError(
                    f"automatic-tree leaf selection {selection_id} 对应多个 "
                    "selection artifact，当前不能安全绑定来源。"
                )
            row, provenance = matches[0]
            verified = (
                load_verified_automatic_tree_leaf_selection_artifact_on_connection(
                    conn,
                    tasks_dir=runtime.settings.tasks_dir,
                    task_id=task_id,
                    artifact_id=row["id"],
                    expected_content_hash=row["content_hash"],
                    expected_asset_id=provenance.get("tree_asset_id"),
                    expected_asset_hash=provenance.get("tree_asset_hash"),
                )
            )
            if verified.selection.get("selection_id") != selection_id:
                raise StrategySetupError(
                    "leaf selection artifact 的 selection ID 与请求不一致。"
                )
    except StrategySetupError:
        raise
    except Exception as exc:
        raise StrategySetupError(
            f"automatic-tree leaf selection {selection_id} 未通过 artifact "
            "完整性校验，不能加入 Strategy Pool。"
        ) from exc

    tree_asset = verified.selection.get("tree_asset")
    leaf = verified.selection.get("leaf")
    if not isinstance(tree_asset, Mapping) or not isinstance(leaf, Mapping):
        raise StrategySetupError(
            "leaf selection artifact 缺少完整 tree/fragment 绑定。"
        )
    asset_id = tree_asset.get("asset_id")
    asset_hash = tree_asset.get("asset_hash")
    fragment_id = leaf.get("fragment_id")
    if not all(
        isinstance(value, str) and value
        for value in (asset_id, asset_hash, fragment_id)
    ):
        raise StrategySetupError(
            "leaf selection artifact 缺少完整 tree/fragment 绑定。"
        )
    return (
        {
            "source_artifact_id": verified.artifact_id,
            "expected_artifact_content_hash": verified.content_hash,
            "expected_asset_id": asset_id,
            "expected_asset_hash": asset_hash,
        },
        fragment_id,
    )


def _candidate_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Dispatch only between explicitly versioned governed selection types."""

    if re.fullmatch(
        r"automatic-tree-leaf-selection-[0-9a-f]{32}", selection_id
    ) is not None:
        return _automatic_tree_leaf_selection_artifact_slots(
            runtime,
            task_id=task_id,
            selection_id=selection_id,
        )
    if re.fullmatch(
        r"cross-matrix-cell-selection-[0-9a-f]{32}", selection_id
    ) is not None:
        return _cross_matrix_cell_selection_artifact_slots(
            runtime,
            task_id=task_id,
            selection_id=selection_id,
        )
    raise StrategySetupError(
        "selection ID 格式无效；只支持完整 automatic-tree leaf selection 或 "
        "Cross Matrix cell selection ID。"
    )


def _cross_matrix_cell_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Bind one exact Cross cell selection to verified Pool Tool inputs."""

    repository = TaskArtifactRepository(runtime.settings.db_path)
    try:
        artifacts = repository.list_for_task(task_id)
    except Exception as exc:
        raise StrategySetupError(
            "当前任务的 Cross Matrix cell selection registry 无法读取。"
        ) from exc
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise StrategySetupError(
                "当前任务的 Cross Matrix cell selection artifact 记录结构无效。"
            )
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND
            and artifact.get("origin_tool")
            == CROSS_MATRIX_CELL_SELECTION_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("schema_version")
            == CROSS_MATRIX_CELL_SELECTION_ARTIFACT_SCHEMA_VERSION
            and provenance.get("selection_id") == selection_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有 Cross Matrix cell selection {selection_id}。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"Cross Matrix cell selection {selection_id} 对应多个 artifact，"
            "当前不能安全绑定来源。"
        )

    artifact = matches[0]
    provenance = artifact.get("provenance")
    assert isinstance(provenance, Mapping)
    artifact_id = artifact.get("id")
    content_hash = artifact.get("content_hash")
    asset_id = provenance.get("source_asset_id")
    asset_hash = provenance.get("source_asset_hash")
    if not all(
        isinstance(value, str) and value
        for value in (artifact_id, content_hash, asset_id, asset_hash)
    ):
        raise StrategySetupError(
            f"Cross Matrix cell selection {selection_id} 的完整性绑定不完整。"
        )
    try:
        with repository.transaction() as conn:
            verified = (
                load_verified_cross_matrix_cell_selection_artifact_on_connection(
                    conn,
                    tasks_dir=runtime.settings.tasks_dir,
                    task_id=task_id,
                    artifact_id=artifact_id,
                    expected_content_hash=content_hash,
                    expected_asset_id=asset_id,
                    expected_asset_hash=asset_hash,
                )
            )
    except Exception as exc:
        raise StrategySetupError(
            f"Cross Matrix cell selection {selection_id} 未通过 artifact 完整性"
            "校验，不能加入 Strategy Pool。"
        ) from exc

    selection = verified.selection
    if selection.get("selection_id") != selection_id:
        raise StrategySetupError(
            "Cross Matrix cell selection artifact 的 selection ID 与请求不一致。"
        )
    source_asset = selection.get("source_asset")
    group_id = selection.get("group_id")
    if not isinstance(source_asset, Mapping) or not all(
        isinstance(value, str) and value
        for value in (
            source_asset.get("asset_id"),
            source_asset.get("asset_hash"),
            group_id,
        )
    ):
        raise StrategySetupError(
            "Cross Matrix cell selection artifact 缺少完整 asset/group 绑定。"
        )
    return (
        {
            "source_artifact_id": verified.artifact_id,
            "expected_artifact_content_hash": verified.content_hash,
            "expected_asset_id": str(source_asset["asset_id"]),
            "expected_asset_hash": str(source_asset["asset_hash"]),
        },
        str(group_id),
    )


def _strategy_pool_entries(pool: Mapping) -> list[Mapping]:
    entries = pool.get("entries")
    if not isinstance(entries, Sequence) or isinstance(
        entries, str | bytes | bytearray
    ):
        raise StrategySetupError("当前 Strategy Pool entries 无效。")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise StrategySetupError("当前 Strategy Pool entry 结构无效。")
    return list(entries)


def _strategy_pool_rule_id(pool: Mapping, identifier: str) -> str:
    matches = [
        entry
        for entry in _strategy_pool_entries(pool)
        if identifier in {str(entry.get("entry_id")), str(entry.get("rule_id"))}
    ]
    if len(matches) != 1:
        raise StrategySetupError(
            f"当前 Strategy Pool 中没有唯一匹配的 rule_id/entry_id：{identifier}。"
        )
    rule_id = matches[0].get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise StrategySetupError("当前 Strategy Pool entry 缺少完整 rule_id。")
    return rule_id


def _strategy_pool_complete_rule_order(
    pool: Mapping,
    ordered_ids: object,
) -> list[str]:
    entries = _strategy_pool_entries(pool)
    if not isinstance(ordered_ids, Sequence) or isinstance(
        ordered_ids, str | bytes | bytearray
    ):
        raise StrategySetupError("Strategy Pool reorder 必须提供完整 ID 列表。")
    resolved = [_strategy_pool_rule_id(pool, str(item)) for item in ordered_ids]
    current_rule_ids = [str(entry.get("rule_id") or "") for entry in entries]
    if (
        len(resolved) != len(current_rule_ids)
        or len(set(resolved)) != len(resolved)
        or set(resolved) != set(current_rule_ids)
    ):
        raise StrategySetupError(
            "Strategy Pool reorder 必须提供当前全部 rule_id/entry_id 的完整、无重复排列；"
            "遗漏 ID 不会被解释为删除。"
        )
    return resolved


def _stored_strategy_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StrategyRequestDraft,
) -> tuple[str, str] | None:
    if not draft.strategy_id:
        return (
            "strategy_id_required",
            f"{draft.operation} 已有策略时必须说明 strategy_id。",
        )
    repository = StrategyRepository(runtime.settings.db_path)
    meta = repository.get_strategy_meta(draft.strategy_id)
    if meta is None or meta.get("task_id") != task.id:
        return (
            "strategy_not_owned_by_task",
            "没有在当前任务中找到该策略，不能跨任务读取或执行策略。",
        )
    if meta.get("strategy_type") != draft.strategy_type:
        return (
            "strategy_type_mismatch",
            "请求中的策略类型与已保存策略不一致，请按平台记录重新确认。",
        )

    if draft.baseline_strategy_id:
        baseline = repository.get_strategy_meta(draft.baseline_strategy_id)
        if baseline is None or baseline.get("task_id") != task.id:
            return (
                "strategy_baseline_not_owned_by_task",
                "没有在当前任务中找到基线策略，不能跨任务对比。",
            )
        if baseline.get("strategy_type") != draft.strategy_type:
            return (
                "strategy_baseline_type_mismatch",
                "候选策略与基线策略类型不一致，不能生成同口径对比。",
            )
        if draft.baseline_strategy_id == draft.strategy_id:
            return (
                "strategy_baseline_same_as_candidate",
                "候选策略和基线策略不能是同一个版本。",
            )
    if draft.operation == "compare" and not draft.baseline_strategy_id:
        return (
            "strategy_baseline_required",
            "策略对比必须说明 baseline_strategy_id。",
        )

    if draft.operation == "report":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_report_request_unused_fields",
                "已有策略报告只使用 strategy_id；其他业务字段不会被静默忽略，"
                "请删除后重试。",
            )
        return None

    if draft.operation == "apply":
        if any(
            value is not None
            for value in (
                draft.objective,
                draft.max_bad_rate,
                draft.min_approval_rate,
                draft.baseline_strategy_id,
                draft.adoption_reason,
                draft.profit,
                draft.economics_inputs,
            )
        ):
            return (
                "strategy_apply_request_unused_fields",
                "应用已有策略只使用 strategy_id 和任务内样本；目标、约束、基线、"
                "采纳理由及利润字段不会被静默忽略，请删除后重试。",
            )
        return None

    if draft.operation == "adopt":
        if not draft.adoption_reason:
            return (
                "strategy_adoption_reason_required",
                "采纳策略必须给出明确的人工采纳理由。",
            )
        if meta.get("status") != "draft":
            return (
                "strategy_draft_required",
                "只有 draft 或 validated 资产状态的策略可以进入本地采纳流程。",
            )
        if draft.baseline_strategy_id is not None:
            return (
                "strategy_adoption_unused_baseline",
                "采纳入口不会使用 baseline_strategy_id，请删除后重试。",
            )
        if (
            draft.strategy_type in {"limit", "pricing"}
            and draft.economics_inputs is None
        ):
            return (
                "strategy_adoption_economics_required",
                f"采纳 {draft.strategy_type} 策略需要完整 economics_inputs，"
                "以生成可审计的经济证据和类型化监控基线。",
            )
    elif draft.adoption_reason is not None:
        return (
            "strategy_request_unused_adoption_reason",
            "当前请求不是采纳操作，采纳理由不会被静默忽略。",
        )

    if draft.strategy_type not in {"approval", "reject"} and any(
        value is not None
        for value in (
            draft.objective,
            draft.max_bad_rate,
            draft.min_approval_rate,
            draft.profit,
        )
    ):
        return (
            "strategy_typed_business_contract_not_wired",
            f"{draft.strategy_type} 的目标、约束和经济参数需要类型专属 contract；"
            "当前不会忽略或套用审批口径。",
        )
    return None


def _strategy_dataset_context(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    require_target: bool = True,
):
    backend, registry = _modeling_data_runtime(runtime.settings)
    return build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        require_target=require_target,
    )


def _strategy_pool_impact_dataset_context(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    """Resolve target only from confirmed DataWorkspace semantics for impact."""

    _require_strategy_pool_impact_workspace(runtime, task)
    backend, registry = _modeling_data_runtime(runtime.settings)
    return build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
        require_target=True,
    )


def _strategy_dataset_preview(runtime: DriverTurnRuntime, task: TaskRecord):
    backend, registry = _modeling_data_runtime(runtime.settings)
    return preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
    )


def _strategy_pool_impact_dataset_preview(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    """Preview the active sample using only its confirmed workspace target."""

    _require_strategy_pool_impact_workspace(runtime, task)
    backend, registry = _modeling_data_runtime(runtime.settings)
    return preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
    )


def _strategy_sample_design_dataset_context(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    """Resolve the sample-design source only from confirmed workspace state."""

    _require_strategy_sample_design_workspace(runtime, task)
    backend, registry = _modeling_data_runtime(runtime.settings)
    context = build_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
        require_target=True,
    )
    _validate_strategy_sample_design_target(
        registry,
        backend,
        dataset_id=context.dataset_id,
        target_col=context.target_col,
    )
    return context


def _strategy_sample_design_dataset_preview(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    """Preview the exact active sample and its confirmed workspace target."""

    _require_strategy_sample_design_workspace(runtime, task)
    backend, registry = _modeling_data_runtime(runtime.settings)
    preview = preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=None,
    )
    _validate_strategy_sample_design_target(
        registry,
        backend,
        dataset_id=preview.dataset_id,
        target_col=preview.target_col,
    )
    return preview


def _validate_strategy_sample_design_target(
    registry,
    backend,
    *,
    dataset_id: object,
    target_col: object,
) -> tuple[int, int]:
    """Accept only native numeric 0/1 plus genuine null labels.

    Numeric strings are intentionally rejected even when pandas could coerce
    them. Infinite values and other finite numbers are hard errors, never NaN
    confirmation candidates.
    """

    if not isinstance(dataset_id, str) or not dataset_id:
        raise StrategySetupError(
            "策略样本设计必须绑定已注册的活动数据集后才能校验目标列。"
        )
    if not isinstance(target_col, str) or not target_col:
        raise StrategySetupError("策略样本设计要求已确认的二元目标列。")
    try:
        path = registry.resolve_path(dataset_id)
        frame = backend.read_frame(path, columns=[target_col])
        target = frame[target_col]
    except Exception as exc:
        raise StrategySetupError(
            f"目标列 `{target_col}` 无法从当前活动数据集读取。"
        ) from exc
    dtype_kind = getattr(target.dtype, "kind", None)
    if dtype_kind not in {"i", "u", "f"}:
        raise StrategySetupError(
            f"目标列 `{target_col}` 必须是数值 0/1 或真实空值；"
            "字符串 '0'/'1'、布尔值和其他编码不接受。"
        )
    null_mask = target.isna()
    for value in target.loc[~null_mask].tolist():
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StrategySetupError(
                f"目标列 `{target_col}` 必须是数值 0/1 或真实空值。"
            ) from exc
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise StrategySetupError(
                f"目标列 `{target_col}` 必须是数值 0/1 或真实空值；"
                "inf、-inf 和 0/1 之外的值不能进入样本设计。"
            )
    return int(len(target)), int(null_mask.sum())


def _require_strategy_sample_design_workspace(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "策略样本设计需要有效且已确认的活动 DataWorkspace。"
        ) from exc
    if snapshot.active_dataset_id is None:
        raise StrategySetupError(
            "策略样本设计要求先在 DataWorkspace 选择并保存活动数据集。"
        )
    if not snapshot.semantic_mapping.target_col:
        raise StrategySetupError(
            "策略样本设计要求先在 DataWorkspace 确认二元目标列。"
        )
    return snapshot


def _require_strategy_pool_impact_workspace(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "Strategy Pool 影响测算需要有效的活动 DataWorkspace。"
        ) from exc
    if snapshot.active_dataset_id is None:
        raise StrategySetupError(
            "Strategy Pool 影响测算要求先在 DataWorkspace 选择活动数据集。"
        )
    if not snapshot.semantic_mapping.target_col:
        raise StrategySetupError(
            "Strategy Pool 影响测算要求先在 DataWorkspace 确认二元目标列。"
        )
    return snapshot


def _strategy_dataset_binding_matches(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    preview,
    context,
    use_confirmed_workspace_target: bool = False,
    use_sample_design_workspace: bool = False,
) -> bool:
    """Verify the registered snapshot still represents the compiled preview."""

    if (
        tuple(context.columns) != tuple(preview.columns)
        or context.target_col != preview.target_col
    ):
        return False
    try:
        if use_sample_design_workspace:
            refreshed = _strategy_sample_design_dataset_preview(runtime, task)
        elif use_confirmed_workspace_target:
            refreshed = _strategy_pool_impact_dataset_preview(runtime, task)
        else:
            refreshed = _strategy_dataset_preview(runtime, task)
    except StrategySetupError:
        return False
    if (
        refreshed.dataset_id != context.dataset_id
        or tuple(refreshed.columns) != tuple(context.columns)
        or refreshed.target_col != context.target_col
    ):
        return False

    identity = preview.identity if isinstance(preview.identity, dict) else {}
    refreshed_identity = (
        refreshed.identity if isinstance(refreshed.identity, dict) else {}
    )
    context_fields = {
        "workspace_revision": getattr(context, "workspace_revision", None),
        "analysis_generation": getattr(context, "analysis_generation", None),
        "semantic_mapping_hash": getattr(context, "semantic_mapping_hash", None),
    }
    for field, context_value in context_fields.items():
        if field in identity and (
            identity[field] != refreshed_identity.get(field)
            or identity[field] != context_value
        ):
            return False
    if identity.get("kind") == "registered":
        return (
            identity.get("dataset_id") == refreshed_identity.get("dataset_id")
            and identity.get("content_hash") == refreshed_identity.get("content_hash")
            and identity.get("content_hash")
            == getattr(context, "dataset_content_hash", None)
        )
    if identity.get("kind") != "source":
        return False
    source_path = identity.get("source_path")
    expected_hash = identity.get("sha256")
    if not source_path or not expected_hash:
        return False
    try:
        # CSV/XLSX source registration may normalize bytes into Parquet.  The
        # confirmation binds the original source here; the registered Parquet
        # hash is bound separately in the plan/tool inputs.
        return sha256_file(Path(str(source_path))) == str(expected_hash)
    except OSError:
        return False


def _strategy_target_nan_stats(runtime: DriverTurnRuntime, context) -> tuple[int, int]:
    backend, registry = _modeling_data_runtime(runtime.settings)
    target_col = str(context.target_col or "").strip()
    if not target_col:
        raise StrategySetupError("当前策略操作需要明确的二元目标列。")
    path = registry.resolve_path(context.dataset_id)
    try:
        frame = backend.read_frame(path, columns=[target_col])
        mask = nan_label_mask(frame, target_col)
    except Exception as exc:
        raise StrategySetupError(
            f"目标列 `{target_col}` 必须只包含 0/1 和可显式处理的空标签。"
        ) from exc
    return int(len(frame)), int(mask.sum())


def _strategy_nan_label_clarification_response(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    draft: CompiledStrategyRequestDraft,
    context,
    n_total: int,
    n_nan: int,
) -> dict:
    is_pool_impact = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_sample_design"
    )
    refreshed = (
        _strategy_pool_impact_dataset_preview(runtime, task)
        if is_pool_impact
        else (
            _strategy_sample_design_dataset_preview(runtime, task)
            if is_sample_design
            else _strategy_dataset_preview(runtime, task)
        )
    )
    state = {
        "draft": draft.to_dict(),
        "dataset_id": context.dataset_id,
        "dataset_identity": dict(refreshed.identity),
        "target_col": context.target_col,
        "n_total": int(n_total),
        "n_nan": int(n_nan),
    }
    if is_pool_impact:
        try:
            _pool, pool_binding = _strategy_pool_impact_pool_binding(
                runtime,
                task,
                str(draft.workflow_inputs.get("strategy_type") or ""),
            )
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_pool_impact_binding_required",
                message=str(exc),
            )
        state["pool_binding"] = pool_binding
    return _append_strategy_nan_label_clarification(repo, task, state)


def _append_strategy_nan_label_clarification(
    repo: TaskRepository,
    task: TaskRecord,
    state: dict,
) -> dict:
    n_nan = int(state.get("n_nan") or 0)
    n_total = int(state.get("n_total") or 0)
    target_col = str(state.get("target_col") or "")
    payload = state.get("draft")
    is_pool_impact = (
        isinstance(payload, Mapping)
        and payload.get("workflow") in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    )
    is_sample_design = (
        isinstance(payload, Mapping)
        and payload.get("workflow") == "strategy_sample_design"
    )
    if is_pool_impact or is_sample_design:
        missing_description = "空标签" if is_sample_design else "空或非有限标签"
        retained_statistics = (
            "总体、金额和权重统计" if is_sample_design else "总体、动作和金额统计"
        )
        message = (
            f"目标列 `{target_col}` 有 {n_nan}/{n_total} 行{missing_description}。"
            f"这些样本行仍会保留在{retained_statistics}中，只从坏账率/风险率分母中排除；"
            "本次尚未创建计划，平台不会默认采用该口径。"
            "如果确实允许，请明确回复「确认将空标签仅从风险分母排除并继续」；"
            "仅回复「确认」不会执行。"
        )
    else:
        message = (
            f"目标列 `{target_col}` 有 {n_nan}/{n_total} 行空或非有限标签。"
            "本次尚未创建计划，平台不会默认丢弃。"
            "如果确实允许，请明确回复「确认丢弃空标签并继续」；"
            "仅回复「确认」不会执行。"
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "strategy_drop_nan_labels_confirmation",
            "kind": "clarification",
            "code": "strategy_drop_nan_labels_confirmation_required",
            "fields": ["drop_nan_labels"],
            _STRATEGY_NAN_LABEL_META_KEY: state,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": "strategy_drop_nan_labels_confirmation_required",
        "fields": ["drop_nan_labels"],
        "label_quality": {
            "target_col": target_col,
            "n_total": n_total,
            "n_nan": n_nan,
        },
        "messages": repo.list_agent_messages(task.id),
    }


def _repeat_strategy_nan_label_clarification(
    repo: TaskRepository,
    task: TaskRecord,
    state: dict,
) -> dict:
    return _append_strategy_nan_label_clarification(repo, task, dict(state))


def _resume_strategy_after_nan_label_confirmation(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    state: dict,
) -> dict:
    if (
        _active_plan(runtime.plan_repo, task.id) is not None
        or latest_open_gate(repo.list_agent_messages(task.id)) is not None
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_stale_confirmation",
            message="任务状态已变化，空标签处理确认已失效；请完成当前计划后重新发起。",
        )
    payload = state.get("draft")
    if not isinstance(payload, dict):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_invalidated",
            message="空标签确认缺少已校验策略口径，请重新描述策略请求。",
        )
    is_pool_impact = payload.get("workflow") in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    is_sample_design = payload.get("workflow") == "strategy_sample_design"
    expected_pool_binding = None
    if is_pool_impact:
        expected_pool_binding = state.get("pool_binding")
        if not isinstance(expected_pool_binding, Mapping):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_pool_context_changed",
                message=(
                    "旧的空标签确认没有绑定 Strategy Pool revision/hash；"
                    "为避免误用当前 Pool，请重新发起影响测算。"
                ),
            )
        try:
            _pool, current_pool_binding = _strategy_pool_impact_pool_binding(
                runtime,
                task,
                str(expected_pool_binding.get("strategy_type") or ""),
            )
        except StrategySetupError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_pool_context_changed",
                message=str(exc),
            )
        if dict(expected_pool_binding) != current_pool_binding:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_pool_context_changed",
                message=(
                    "Strategy Pool 在等待空标签确认期间已变化；旧确认未执行，"
                    "请基于当前 Pool 重新发起影响测算。"
                ),
            )
    try:
        preview = (
            _strategy_pool_impact_dataset_preview(runtime, task)
            if is_pool_impact
            else (
                _strategy_sample_design_dataset_preview(runtime, task)
                if is_sample_design
                else _strategy_dataset_preview(runtime, task)
            )
        )
    except StrategySetupError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=str(exc),
        )
    expected_identity = state.get("dataset_identity")
    if (
        not isinstance(expected_identity, dict)
        or preview.identity != expected_identity
        or preview.dataset_id != state.get("dataset_id")
        or preview.target_col != state.get("target_col")
    ):
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_changed",
            message="策略样本或目标列已变化；空标签确认未执行，请重新描述策略请求。",
        )
    compilation = validate_strategy_request(
        payload,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=preview.target_col,
    )
    if compilation.draft is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_invalidated",
            message=compilation.clarification or "策略口径重新校验失败，请重新描述。",
        )
    preflight = _strategy_request_preflight(runtime, task, compilation.draft)
    if preflight is not None:
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        compilation.draft,
        preview=preview,
        auto_start=True,
        drop_nan_labels=True,
        expected_pool_binding=expected_pool_binding,
    )


def _strategy_request_allowed_columns(preview) -> tuple[str, ...]:
    if preview is None:
        return ()
    # The observed target is evidence, never a deployable strategy feature or
    # an input to an LLM-authored profit contract.
    return tuple(column for column in preview.columns if column != preview.target_col)


def _strategy_request_requires_dataset(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    if isinstance(draft, StandardWorkflowRequestDraft):
        if draft.workflow in {
            *_STRATEGY_POOL_WORKFLOWS,
            "automatic_tree_leaf_materialization",
            "cross_matrix_cell_selection",
            "voting_candidate_build",
        }:
            return False
        return True
    return not (draft.strategy_spec is None and draft.operation == "report")


def _is_automatic_tree_build_draft(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    return (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "automatic_tree_candidate_build"
    )


def _ensure_automatic_tree_active_workspace(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    preview,
    context,
):
    """Bind a fresh task's sole registered sample before governed tree evidence.

    Automatic-tree artifacts deliberately require an active, persisted data
    workspace.  A natural-language build on a fresh task may have just registered
    its sole source sample, so selecting that one unambiguous dataset is normal
    request preparation.  Multiple registered datasets remain a user decision.
    """

    repository = DataWorkspaceRepository(runtime.settings.db_path)
    try:
        snapshot = repository.get_or_default(task.id)
    except (DataWorkspaceDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "自动树需要有效的数据工作区，请先重新选择活动样本。"
        ) from exc
    if snapshot.active_dataset_id is not None:
        if (
            snapshot.active_dataset_id != context.dataset_id
            or snapshot.active_dataset_content_hash != context.dataset_content_hash
        ):
            raise StrategySetupError(
                "自动树样本与当前活动数据集不一致，请先在数据工作区明确选择样本。"
            )
        if snapshot.semantic_mapping.target_col != context.target_col:
            raise StrategySetupError(
                "自动树目标列必须与当前数据工作区的 target 语义一致，请先确认 target_col。"
            )
        return preview, context

    _backend, registry = _modeling_data_runtime(runtime.settings)
    owned = [
        dataset
        for dataset in registry.list_for_task(task.id)
        if str(dataset.task_id) == task.id
    ]
    if len(owned) != 1 or owned[0].id != context.dataset_id:
        raise StrategySetupError(
            "自动树需要一个明确的活动样本；当前存在多个或不确定的数据集，"
            "请先在数据工作区选择并保存本次样本。"
        )
    if not isinstance(context.dataset_content_hash, str) or not context.target_col:
        raise StrategySetupError(
            "自动树无法绑定样本哈希或二元目标列，请先确认数据与 target_col。"
        )

    try:
        repository.save_initial_binding(
            task.id,
            DataWorkspaceDraft(
                active_dataset_id=context.dataset_id,
                active_dataset_content_hash=context.dataset_content_hash,
                semantic_mapping=DataSemanticMapping(
                    target_col=context.target_col,
                    field_roles={context.target_col: "target"},
                ),
            ),
            expected_revision=snapshot.revision,
            audit={
                "actor": "agent:strategy-automatic-tree-build",
                "detail": {
                    "reason": "atomically bind sole task sample and target for automatic tree"
                },
            },
        )
    except (
        DataWorkspaceDataError,
        DataWorkspaceDatasetNotFound,
        DataWorkspaceRevisionConflict,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise StrategySetupError(
            "自动树数据工作区在计划创建前发生变化，请重新确认活动样本。"
        ) from exc

    refreshed_preview = _strategy_dataset_preview(runtime, task)
    refreshed_context = _strategy_dataset_context(runtime, task, require_target=True)
    return refreshed_preview, refreshed_context


def _strategy_request_requires_target(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    if isinstance(draft, StandardWorkflowRequestDraft):
        refinement_needs_current_target = (
            draft.workflow == "univariate_candidate_refinement"
            and "source_candidate_id" not in draft.workflow_inputs
        )
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "univariate_candidate_analysis",
                "automatic_tree_candidate_build",
                "cross_matrix_analysis",
                "strategy_pool_impact",
                "limit_pricing_matrix",
            }
            or refinement_needs_current_target
        )
    if draft.operation in {"apply", "report", "monitor"}:
        return False
    if draft.operation == "develop" and draft.strategy_spec is not None:
        return False
    return True


def _strategy_request_requires_complete_labels(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    """Whether execution would otherwise exclude missing supervision rows."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        refinement_needs_current_labels = (
            draft.workflow == "univariate_candidate_refinement"
            and "source_candidate_id" not in draft.workflow_inputs
        )
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "univariate_candidate_analysis",
                "automatic_tree_candidate_build",
                "cross_matrix_analysis",
                "strategy_pool_impact",
                "limit_pricing_matrix",
            }
            or refinement_needs_current_labels
        )
    if draft.operation in {"apply", "report", "monitor"}:
        return False
    if draft.operation == "develop" and draft.strategy_spec is not None:
        return False
    return True


def _strategy_slots_with_drop_nan(slots: dict, confirmed: bool) -> dict:
    if not confirmed:
        return slots
    return {**slots, "drop_nan_labels": True}


def _typed_strategy_slots(context, draft: StrategyRequestDraft) -> dict:
    slots = {
        "dataset_id": context.dataset_id,
        "target_col": context.target_col,
        "strategy_spec": draft.to_dict()["strategy_spec"],
    }
    if draft.baseline_strategy_id:
        slots["baseline_strategy_id"] = draft.baseline_strategy_id
    if draft.profit is not None:
        profit = dict(draft.profit)
        slots["ead_col"] = profit.pop("ead_col")
        slots["pd_col"] = profit.pop("pd_col")
        slots["profit_params"] = profit
    if draft.economics_inputs is not None:
        slots["economics_inputs"] = dict(draft.economics_inputs)
    return slots


def _is_auto_candidate_draft(draft: CompiledStrategyRequestDraft) -> bool:
    return (
        isinstance(draft, StrategyRequestDraft)
        and draft.operation == "develop"
        and draft.strategy_type in {"limit", "pricing", "segmentation"}
        and draft.candidate_design is not None
    )


def _candidate_strategy_slots(context, draft: StrategyRequestDraft) -> dict:
    slots = {
        "dataset_id": context.dataset_id,
        "target_col": context.target_col,
        "strategy_type": draft.strategy_type,
        "candidate_design": dict(draft.candidate_design or {}),
    }
    if draft.economics_inputs is not None:
        slots["economics_inputs"] = dict(draft.economics_inputs)
    if draft.baseline_strategy_id:
        slots["baseline_strategy_id"] = draft.baseline_strategy_id
    return slots


def _stored_strategy_slots(context, draft: StrategyRequestDraft) -> dict:
    slots = {
        "dataset_id": context.dataset_id,
        "target_col": context.target_col,
        "strategy_id": draft.strategy_id,
    }
    if draft.baseline_strategy_id:
        slots["baseline_strategy_id"] = draft.baseline_strategy_id
    if draft.adoption_reason:
        slots["adoption_reason"] = draft.adoption_reason
    if draft.profit is not None:
        profit = dict(draft.profit)
        slots["ead_col"] = profit.pop("ead_col")
        slots["pd_col"] = profit.pop("pd_col")
        slots["profit_params"] = profit
    if draft.economics_inputs is not None:
        slots["economics_inputs"] = dict(draft.economics_inputs)
    return slots


def _strategy_contract_from_draft(draft: StrategyRequestDraft) -> StrategyTaskInput:
    profit = None
    if draft.profit is not None:
        profit = StrategyProfitInput(**dict(draft.profit))
    return StrategyTaskInput(
        strategy_type=draft.strategy_type,
        objective=draft.objective or "",
        max_bad_rate=draft.max_bad_rate,
        min_approval_rate=draft.min_approval_rate,
        baseline_strategy_id=draft.baseline_strategy_id,
        profit=profit,
    )


def _strategy_request_success_criteria(
    draft: StrategyRequestDraft,
) -> list[dict] | None:
    if draft.strategy_type not in {"approval", "reject"}:
        return None
    criteria: list[dict] = []
    if draft.max_bad_rate is not None:
        criteria.append({"metric": "approved_bad_rate", "max": draft.max_bad_rate})
    if draft.min_approval_rate is not None:
        criteria.append({"metric": "approval_rate", "min": draft.min_approval_rate})
    return criteria or None


def _strategy_request_clarification_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    code: str,
    message: str,
    fields: tuple[str, ...] | list[str] = (),
) -> dict:
    normalized_fields = list(dict.fromkeys(str(field) for field in fields))
    metadata = {
        "intent": "strategy_request",
        "kind": "clarification",
        "code": code,
    }
    if normalized_fields:
        metadata["fields"] = normalized_fields
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata=metadata,
    )
    response = {
        "task_id": task.id,
        "status": "clarification_required",
        "code": code,
        "messages": repo.list_agent_messages(task.id),
    }
    if normalized_fields:
        response["fields"] = normalized_fields
    return response


def _latest_strategy_request_pending(conversation: list[dict]) -> dict | None:
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return None
    pending = (last_assistant.get("metadata") or {}).get(_STRATEGY_REQUEST_META_KEY)
    return pending if isinstance(pending, dict) else None


def _latest_strategy_nan_label_confirmation(
    conversation: list[dict],
) -> dict | None:
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return None
    state = (last_assistant.get("metadata") or {}).get(_STRATEGY_NAN_LABEL_META_KEY)
    return state if isinstance(state, dict) else None


def _invalidate_pending_strategy_request(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    pending: dict,
) -> None:
    """Best-effort terminal transition for an obsolete opaque request ref."""

    try:
        PendingStrategyRequestRepository(runtime.settings.db_path).invalidate(
            task_id=task.id,
            request_id=str(pending.get("request_id") or ""),
            expected_payload_sha256=str(pending.get("payload_sha256") or ""),
        )
    except (
        PendingStrategyRequestConflictError,
        PendingStrategyRequestNotFoundError,
        ValueError,
    ):
        # The clarification path must remain safe under a concurrent confirm or
        # cancel. The winning transition is already audited by the repository.
        return


# Governed dataset transformation --------------------------------------------


_DATASET_TRANSFORM_PROTECTED_DROP_META_KEY = "pending_protected_drop"


def _maybe_handle_dataset_transform_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Compile a natural-language data change into the closed transform AST."""

    text = str(user_text or "").strip()
    if not text:
        return None
    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    pending = _latest_pending_transform_protected_drop(conversation)
    confirming_pending = pending is not None and is_confirm(text)
    if not confirming_pending and not detect_dataset_transform_intent(text):
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "dataset_transform",
            **({"confirmation": "protected_drop"} if confirming_pending else {}),
        },
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要加工的样本。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed chat boundary
        return _dataset_transform_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    semantic_hash = data_semantic_mapping_hash(snapshot.semantic_mapping)
    effective_semantics = effective_transform_semantic_mapping(
        dataset,
        snapshot.semantic_mapping,
        source_columns=columns,
    )
    if confirming_pending:
        if not _pending_transform_matches_workspace(
            pending,
            dataset_id=dataset.id,
            content_hash=dataset.content_hash,
            analysis_generation=snapshot.analysis_generation,
            semantic_mapping_hash=semantic_hash,
        ):
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="protected_drop_source_changed",
                message=(
                    "待确认期间活动数据或字段语义已经变化。请重新说明要删除的字段，"
                    "我会基于当前版本重新确认。"
                ),
            )
        pending_operations = pending.get("operations")
        pending_protected = pending.get("protected_fields")
        if (
            not isinstance(pending_operations, list)
            or not pending_operations
            or not all(isinstance(item, dict) for item in pending_operations)
            or not isinstance(pending_protected, list)
            or not pending_protected
            or not all(isinstance(item, str) for item in pending_protected)
        ):
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="protected_drop_state_invalid",
                message="待确认的删列请求不完整，请重新说明要删除的字段。",
            )
        operations = list(pending_operations)
        confirm_protected_drop = True
    else:
        parsed = build_dataset_transform_request(
            text,
            columns=columns,
            business_names=effective_semantics.business_names,
            semantic_mapping=effective_semantics,
        )
        if parsed.request is None:
            extra_metadata: dict = {}
            if parsed.operations and parsed.protected_fields:
                extra_metadata[_DATASET_TRANSFORM_PROTECTED_DROP_META_KEY] = {
                    "request_text": text,
                    "operations": [dict(item) for item in parsed.operations],
                    "protected_fields": list(parsed.protected_fields),
                    "dataset_id": dataset.id,
                    "dataset_content_hash": dataset.content_hash,
                    "analysis_generation": snapshot.analysis_generation,
                    "semantic_mapping_hash": semantic_hash,
                }
            return _dataset_transform_clarification(
                repo,
                task.id,
                code="transform_request_clarification",
                message=parsed.clarification or "请说明要如何加工当前数据。",
                extra_metadata=extra_metadata,
            )
        request = parsed.request
        operations = list(request.operations)
        confirm_protected_drop = request.confirm_protected_drop

    slots = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "operations": operations,
        "confirm_protected_drop": confirm_protected_drop,
    }
    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_transform",
            slots=slots,
            tier=runtime.tier,
        )
        # A normal transform is reversible by selecting the immutable parent;
        # protected drops reached this point only after the explicit dialogue
        # acknowledgement above, so no second generic gate is needed.
        turn = driver.resume(plan_id=started.plan_id, user_text="确认")
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"数据加工出错：{exc}")
    append_driver_messages(
        repo,
        task.id,
        turn,
        settings=runtime.settings,
        task=task,
    )
    return join_turn_response(repo, task.id)


def _latest_pending_transform_protected_drop(
    conversation: list[dict],
) -> dict | None:
    last_assistant = next(
        (
            message
            for message in reversed(conversation)
            if message.get("role") == "assistant"
        ),
        None,
    )
    if last_assistant is None:
        return None
    pending = (last_assistant.get("metadata") or {}).get(
        _DATASET_TRANSFORM_PROTECTED_DROP_META_KEY
    )
    return pending if isinstance(pending, dict) else None


def _pending_transform_matches_workspace(
    pending: dict,
    *,
    dataset_id: str,
    content_hash: str,
    analysis_generation: int,
    semantic_mapping_hash: str,
) -> bool:
    request_text = pending.get("request_text")
    return (
        isinstance(request_text, str)
        and bool(request_text.strip())
        and pending.get("dataset_id") == dataset_id
        and pending.get("dataset_content_hash") == content_hash
        and pending.get("analysis_generation") == analysis_generation
        and pending.get("semantic_mapping_hash") == semantic_mapping_hash
    )


def _dataset_transform_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
    extra_metadata: dict | None = None,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_transform",
            "kind": "clarification",
            "code": code,
            **dict(extra_metadata or {}),
        },
    )
    return join_turn_response(repo, task_id)


# Safe task-owned dataset export ---------------------------------------------


def _maybe_handle_dataset_export_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Run a bound CSV/XLSX export when the dataset object is explicit."""

    if not detect_dataset_export_intent(user_text):
        return None
    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "dataset_export"},
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要导出的样本。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed chat boundary
        return _dataset_export_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    parsed = build_dataset_export_request(
        user_text or "",
        columns=columns,
        business_names=snapshot.semantic_mapping.business_names,
    )
    if parsed.request is None:
        return _dataset_export_clarification(
            repo,
            task.id,
            code="export_request_clarification",
            message=parsed.clarification or "请选择 CSV 或 Excel 导出格式。",
        )
    request = parsed.request
    slots = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "format": request.format,
        "text_columns": list(request.text_columns),
    }
    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_export",
            slots=slots,
            tier=runtime.tier,
        )
        turn = driver.resume(plan_id=started.plan_id, user_text="确认")
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"数据导出出错：{exc}")
    append_driver_messages(
        repo,
        task.id,
        turn,
        settings=runtime.settings,
        task=task,
    )
    return join_turn_response(repo, task.id)


def _dataset_export_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_export",
            "kind": "clarification",
            "code": code,
        },
    )
    return join_turn_response(repo, task_id)


# Report-ready dataset analysis -----------------------------------------------


def _maybe_handle_dataset_analysis_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Run the bound descriptive-analysis Workflow for an explicit request."""

    if not detect_dataset_analysis_intent(user_text):
        return None
    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "dataset_analysis"},
    )
    try:
        snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(
            task.id
        )
    except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound, KeyError) as exc:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="workspace_unavailable",
            message=f"数据工作区当前不可用：{exc}",
        )
    if snapshot.active_dataset_id is None:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="active_dataset_required",
            message="请先在数据工作区选择并保存本次要分析的样本，再让我开始分析。",
        )

    backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(snapshot.active_dataset_id)
        if dataset.task_id != task.id:
            raise PermissionError("active dataset does not belong to this task")
        path = registry.resolve_verified_path(dataset.id)
        columns = tuple(backend.column_names(path))
    except Exception as exc:  # noqa: BLE001 - converted to a typed user-facing boundary
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="dataset_unavailable",
            message=f"当前活动样本无法安全读取：{exc}",
        )

    parsed = build_dataset_analysis_request(
        user_text or "",
        columns=columns,
        target_col=snapshot.semantic_mapping.target_col,
        business_names=snapshot.semantic_mapping.business_names,
    )
    if parsed.request is None:
        return _dataset_analysis_clarification(
            repo,
            task.id,
            code="analysis_request_clarification",
            message=parsed.clarification or "请说明要分析哪些数据内容。",
        )

    request = parsed.request
    slots: dict = {
        "dataset_id": dataset.id,
        "expected_content_hash": snapshot.active_dataset_content_hash,
        "workspace_revision": snapshot.revision,
        "analysis_generation": snapshot.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "sections": list(request.sections),
    }
    if request.columns is not None:
        slots["columns"] = list(request.columns)
    if request.target_col is not None:
        slots["target_col"] = request.target_col

    driver = _driver(runtime)
    try:
        started = driver.start(
            task_id=task.id,
            template_id="dataset_descriptive_analysis",
            slots=slots,
            tier=runtime.tier,
        )
        turn = driver.resume(plan_id=started.plan_id, user_text="确认")
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"样本描述分析出错：{exc}")
    append_driver_messages(
        repo,
        task.id,
        turn,
        settings=runtime.settings,
        task=task,
    )
    return join_turn_response(repo, task.id)


def _dataset_analysis_clarification(
    repo: TaskRepository,
    task_id: str,
    *,
    code: str,
    message: str,
) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "dataset_analysis",
            "kind": "clarification",
            "code": code,
        },
    )
    return join_turn_response(repo, task_id)


# S6 ad-hoc "问数" wiring -------------------------------------------------------
# The 口径确认门 pending state reuses the SAME lightest-weight precedent the join
# C1 gate (_latest_c1_state) and the portfolio states gate (_latest_portfolio_states)
# already use: the confirmation-门 message stores the fully-validated tool inputs
# under its own metadata key (`adhoc_spec`), and the next turn scans the
# conversation back for it — no new state table, no schema change. The key is
# deliberately NOT `kind: "gate"`/`join_c1`, so latest_open_gate() (and therefore
# AUTO auto-drive) never mistakes it for a driver gate it does not know how to run.
_ADHOC_SPEC_META_KEY = "adhoc_spec"
_ADHOC_DATA_ROLES = frozenset({"sample", "feature", "strategy_sample", "derived"})


def _maybe_handle_adhoc_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Return a turn response when this turn is an ad-hoc 问数 interaction, else
    None so the caller falls through to the normal type dispatch."""
    conversation = repo.list_agent_messages(task.id)
    pending = _latest_adhoc_pending(conversation)
    if pending is not None:
        # Round B: a 口径确认门 is open. Only a confirm runs it; anything else
        # (deny / rephrase) drops the pending spec and returns to the normal flow.
        if is_confirm(user_text or ""):
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=user_text or "",
                metadata={"intent": "adhoc_query"},
            )
            return _run_adhoc_slice_plan(runtime, repo, task, pending)
        return None
    # Round A: no pending spec. Enter only when the guards all hold — conservative
    # by design (窄不触发优于劫持).
    if not detect_question_intent(user_text):
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None
    resolved = _resolve_adhoc_dataset(runtime.settings, task.id)
    if resolved is None:
        return None
    dataset_id, columns = resolved
    result = build_slice_spec_from_utterance(
        user_text or "", columns, runtime.llm_client
    )
    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=user_text or "",
        metadata={"intent": "adhoc_query"},
    )
    if result.needs_clarification:
        # A Chinese clarification (never a guess, INV-1). No pending state is
        # stored — the user simply rephrases and round A runs again.
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=result.clarify or "没能理解这个问题，请换一种说法。",
            metadata={"intent": "adhoc_query"},
        )
        return join_turn_response(repo, task.id)
    # A validated spec: show the 口径确认门 and stash the exact tool inputs on it.
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=result.confirmation_text or "",
        metadata={_ADHOC_SPEC_META_KEY: result.spec.tool_inputs(dataset_id)},
    )
    return join_turn_response(repo, task.id)


def _run_adhoc_slice_plan(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    tool_inputs: dict,
) -> dict:
    """Build + run the single-step slice_aggregate plan for a confirmed 口径.

    vintage's lightweight single-step entry is the precedent: one non-gated step
    that runs straight to DONE and renders its own table. Because the 口径 was just
    confirmed turn-side, the plan-overview 开始 gate is auto-confirmed here so the
    aggregate runs in the same turn instead of pausing again."""
    driver = _driver(runtime)
    try:
        start = driver.start(
            task_id=task.id,
            template_id="slice_aggregate",
            slots=dict(tool_inputs),
            tier=runtime.tier,
        )
        turn = driver.resume(plan_id=start.plan_id, user_text="确认")
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"即席问数出错：{exc}")
    append_driver_messages(repo, task.id, turn)
    return join_turn_response(repo, task.id)


def _latest_adhoc_pending(conversation: list[dict]) -> dict | None:
    """The pending ad-hoc tool inputs, only when the LAST assistant message is the
    口径确认门 (mirrors latest_open_gate's last-assistant anchoring). Once the
    aggregate result/error is appended, this stops matching, so a confirmed spec is
    never re-run."""
    last_assistant = next(
        (m for m in reversed(conversation) if m.get("role") == "assistant"), None
    )
    if last_assistant is None:
        return None
    spec = (last_assistant.get("metadata") or {}).get(_ADHOC_SPEC_META_KEY)
    return spec if isinstance(spec, dict) else None


def _resolve_adhoc_dataset(settings, task_id: str) -> tuple[str, list[str]] | None:
    """A task's ready dataset id + its column whitelist, or None when the task has
    no already-registered dataset (guard (a) — this branch never scans/ingests
    from source_dir; that is the setup flow's job). Prefers a target-carrying
    dataset, else the largest — same ranking feature/vintage setup use."""
    backend, registry = _modeling_data_runtime(settings)
    datasets = [
        d for d in registry.list_for_task(task_id) if d.role in _ADHOC_DATA_ROLES
    ]
    if not datasets:
        return None
    dataset = sorted(
        datasets,
        key=lambda d: (
            not bool(getattr(d, "has_target", False)),
            -int(getattr(d, "row_count", 0) or 0),
        ),
    )[0]
    try:
        columns = list(backend.column_names(registry.resolve_path(dataset.id)))
    except Exception:
        return None
    if not columns:
        return None
    return dataset.id, columns


def agent_autodrive_turn(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, *, client
) -> None:
    turn_fn = DRIVER_TURN_FUNCS[task.task_type]
    max_gates = _auto_gate_budget(runtime, task.id)
    for _ in range(max_gates):
        gate = latest_open_gate(repo.list_agent_messages(task.id))
        if gate is None:
            return
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
            except Exception:
                pass
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
            append_driver_messages(
                repo,
                task.id,
                turn,
                settings=getattr(runtime, "settings", None),
                task=task,
            )
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
    task_id: str,
    turn,
    *,
    settings=None,
    task: TaskRecord | None = None,
) -> None:
    for message in turn.messages:
        repo.add_agent_message(
            task_id,
            role="assistant",
            stage="chat",
            content=message.content,
            metadata=dict(message.metadata),
        )
        # MEM-1 write side: once a V2 modeling/data_join plan reaches its terminal
        # "done" message, capture the champion result into agent memory so future
        # same-kind tasks get a historical anchor. Optional settings/task keep this
        # a no-op for every other driver-turn call site (feature/strategy/vintage,
        # and the mid-plan gate messages of modeling/data_join itself).
        if settings is not None and task is not None and message.stage == "done":
            capture_agent_memory_for_driver_done(
                settings,
                task,
                done_message_content=message.content,
                done_message_metadata=dict(message.metadata),
            )


def join_turn_response(repo: TaskRepository, task_id: str) -> dict:
    return {
        "task_id": task_id,
        "status": "ok",
        "messages": repo.list_agent_messages(task_id),
    }


def append_join_error(repo: TaskRepository, task_id: str, detail: str) -> dict:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=detail,
        metadata={"error": True},
    )
    return {
        "task_id": task_id,
        "status": "error",
        "messages": repo.list_agent_messages(task_id),
    }


def append_workflow_error(
    repo: TaskRepository,
    task: TaskRecord,
    spec: _TurnHandlerSpec,
    exc: Exception,
    *,
    setup_error: bool = False,
) -> dict:
    diagnostic = build_workflow_error_diagnostic(
        workflow=spec.intent,
        exc=exc,
        task=task,
        setup_error=setup_error,
    )
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
        governance_service=runtime.governance_service,
        local_principal=runtime.local_principal,
    )


def _modeling_data_runtime(settings):
    datasets_root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(data_repo, backend, datasets_root)
    return backend, registry


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
    first C1 file-role gate) or the plan repo is unavailable, so pre-plan turns
    (join_c1) still get a sane budget."""
    tier = resolve_tier(getattr(runtime, "tier", None))
    plan_repo = getattr(runtime, "plan_repo", None)
    plan = _active_plan(plan_repo, task_id) if plan_repo is not None else None
    if plan is None:
        return AGENT_MAX_GATES
    gate_count = sum(1 for step in plan.steps if step.needs_confirmation)
    # +1 for the plan-overview gate every driver plan pauses at before running.
    return auto_gate_budget(tier, gate_count + 1)


def _latest_c1_state(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            c1 = (message.get("metadata") or {}).get("join_c1")
            if isinstance(c1, dict):
                return c1
    return None


def _c1_table(c1_state: dict) -> list[dict]:
    rows = [
        [
            f.get("name", ""),
            str(f.get("row_count", "")),
            str(f.get("n_cols", "")),
            "是" if f.get("has_target") else "否",
            f.get("candidate_target") or "—",
            "样本主表" if f.get("proposed_role") == "anchor" else "特征表",
        ]
        for f in c1_state.get("files") or []
    ]
    return [
        {
            "title": "输入文件（请确认角色与目标列）",
            "columns": ["文件", "行数", "列数", "含目标列", "候选目标列", "提议角色"],
            "rows": rows,
        }
    ]


def _append_c1_message(repo: TaskRepository, task_id: str, proposal) -> None:
    files = proposal.files
    anchor = next((f for f in files if f.proposed_role == "anchor"), None)
    feature_names = [f.name for f in files if f.proposed_role == "feature"]
    if proposal.skip:
        text = (
            f"我发现 {len(files)} 个数据文件。提议**样本主表 = `{anchor.name if anchor else '?'}`**"
            + (
                f"，目标列 = `{proposal.target_col}`"
                if proposal.target_col
                else "（未识别目标列，请指定）"
            )
            + "。只有一张表，确认后将跳过拼接。请确认，或用下方控件调整。"
        )
    else:
        text = (
            f"我发现 {len(files)} 个数据文件，先确认每张的**角色与目标列**（样本是锚，只贴列不改行，**1:1**）:\n"
            f"- 提议**样本主表** = `{anchor.name if anchor else '?'}`"
            + (
                f"（目标列 `{proposal.target_col}`）"
                if proposal.target_col
                else "（未识别目标列，请指定）"
            )
            + "\n- 提议**特征表** = "
            + (", ".join(f"`{name}`" for name in feature_names) or "（无）")
            + "\n确认无误回复「确认」；要改就用下方控件选好角色/目标列后点「确认角色」。"
        )
    notices = list(getattr(proposal, "ingest_notices", None) or [])
    text += _ingest_notice_text(notices)
    c1_state = {
        "files": [
            {
                "dataset_id": f.dataset_id,
                "name": f.name,
                "row_count": f.row_count,
                "n_cols": f.n_cols,
                "has_target": f.has_target,
                "candidate_target": f.candidate_target,
                "proposed_role": f.proposed_role,
                "columns": f.columns,
            }
            for f in files
        ],
        "anchor_id": proposal.anchor_id,
        "feature_ids": proposal.feature_ids,
        "target_col": proposal.target_col,
        "skip": proposal.skip,
    }
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=text,
        metadata={
            "join_c1": c1_state,
            "tables": _c1_table(c1_state),
            "ingest_notices": notices,
        },
    )


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


def _parse_c1_reply(user_text: str | None, c1_state: dict) -> dict | None:
    text = (user_text or "").strip()
    if text.startswith("[C1]"):
        try:
            payload = json.loads(text[len("[C1]") :])
        except (ValueError, TypeError):
            return None
        anchor_ids = [aid for aid in (payload.get("anchor_ids") or []) if aid]
        if not anchor_ids:
            single_anchor_id = payload.get("anchor_id")
            anchor_ids = [single_anchor_id] if single_anchor_id else []
        anchor_ids = list(dict.fromkeys(anchor_ids))  # de-dup, preserve order
        if len(anchor_ids) > 1:
            names = _c1_dataset_names(c1_state, anchor_ids)
            raise JoinSetupError(
                "样本主表只能有一个，请把 "
                + "、".join(names[1:])
                + " 改为「特征表」或「忽略」。"
            )
        anchor_id = anchor_ids[0] if anchor_ids else payload.get("anchor_id")
        feature_ids = [
            fid
            for fid in (payload.get("feature_ids") or [])
            if fid and fid != anchor_id
        ]
        return {
            "anchor_id": anchor_id,
            "feature_ids": feature_ids,
            "target_col": payload.get("target_col"),
        }
    if is_confirm(text):
        return {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    return None


def _c1_dataset_names(c1_state: dict, dataset_ids: list[str]) -> list[str]:
    by_id = {f.get("dataset_id"): f.get("name") for f in c1_state.get("files") or []}
    return [by_id.get(dataset_id) or dataset_id for dataset_id in dataset_ids]


def _feature_metrics(task: TaskRecord) -> list[str]:
    return [
        str(item).strip()
        for item in (getattr(task, "metrics", None) or [])
        if str(item).strip()
    ]


def _modeling_recipes(task: TaskRecord) -> list[str] | None:
    recipes = [
        str(item).strip()
        for item in (getattr(task, "recipes", None) or [])
        if str(item).strip()
    ]
    return recipes or None


def _modeling_target_type(task: TaskRecord) -> str | None:
    target_type = str(getattr(task, "target_type", "") or "").strip()
    return target_type or None


def _modeling_success_criteria(task: TaskRecord) -> list[dict] | None:
    """AGT-4: turn the task's optional oot_ks_min into a deterministic success
    criterion final_review can evaluate. None/absent oot_ks_min (the default) means
    no criterion is injected — the platform never hard-codes a threshold; only a
    value the user (or AUTO, once wired) explicitly set produces one."""
    oot_ks_min = getattr(task, "oot_ks_min", None)
    if oot_ks_min is None:
        return None
    return [
        {
            "metric": "oot_ks",
            "min": float(oot_ks_min),
            "aggregate": "max",
            "label": "OOT KS",
            "target_type": "binary",
        }
    ]


def _modeling_field_hint_keywords(task: TaskRecord, c1_proposal) -> tuple[str, ...]:
    # MEM-4: scope the field_convention lookup to this task's own dataset file
    # names (+ model name) so a hint only ever comes from prior tasks that look
    # like they touched the same data, never an unrelated model's column names.
    values = [getattr(task, "model_name", None)]
    values.extend(
        getattr(item, "name", None)
        for item in getattr(c1_proposal, "files", None) or ()
    )
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if str(value or "").strip()
        )
    )


def _modeling_project_meta(task: TaskRecord) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key, value in (
        ("模型名称", getattr(task, "model_name", "")),
        ("模型版本", getattr(task, "model_version", "")),
        ("验证人", getattr(task, "validator", "")),
    ):
        text = str(value or "").strip()
        if text:
            meta[key] = text
    return meta


def _auto_decision_content(decision: dict) -> str:
    reason = str(decision.get("reason") or "").strip() or "自动决策已生成。"
    action = decision.get("action")
    if action == "clarify" and decision.get("clarifying_question"):
        return f"🤖 {reason}\n\n需要确认:{decision['clarifying_question']}"
    if action == "replan" and decision.get("replan_goal"):
        return f"🤖 {reason}\n\n重规划目标:{decision['replan_goal']}"
    # LT-11 (B.3): when AUTO auto-confirms a low-risk gate, append the "why safe"
    # rationale (_apply_safety_policy attached it because no risk flag / wide reset
    # fired) so the auto-confirm explains itself. A halt already cites the specific
    # risk_flag code in its reason (from _gate_risk_reason), so no extra line there.
    rationale = str(decision.get("safety_rationale") or "").strip()
    if action == "confirm" and rationale:
        return f"🤖 {reason}\n\n为何可自动确认:{rationale}"
    return f"🤖 {reason}"
