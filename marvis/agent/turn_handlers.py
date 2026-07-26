from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
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
    utterance_targets_candidate_monthly_stability,
    utterance_targets_interactive_tree_frontier_group_materialization,
    utterance_targets_interactive_tree_frontier_materialization,
    utterance_targets_scorecard_band_build,
    utterance_targets_scorecard_cutoff_selection,
    utterance_targets_strategy_dsl_delivery,
    utterance_targets_strategy_impact_cube,
    utterance_targets_strategy_pool_stability,
    utterance_targets_strategy_project_context,
    utterance_targets_strategy_report_bundle_v2,
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
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.data.backend import DataBackend
from marvis.data.errors import DatasetContentDriftError
from marvis.data.labels import nan_label_mask
from marvis.data.registry import DatasetRegistry
from marvis.data.transform_semantics import effective_transform_semantic_mapping
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import (
    DatasetRepository,
    ModelingRepository,
    StrategyRepository,
    TaskRepository,
)
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
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
)
from marvis.packs.strategy.interactive_tree_frontier_group_selection import (
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
    interactive_tree_frontier_group_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.interactive_tree_frontier_group_tools import (
    load_verified_interactive_tree_frontier_group_selection_artifact_on_connection,
)
from marvis.packs.strategy.interactive_tree_frontier_tools import (
    load_verified_interactive_tree_frontier_selection_artifact_on_connection,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ORIGIN_TOOL,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_TYPE,
)
from marvis.packs.strategy.voting_candidate_tools import (
    load_verified_voting_candidate_artifact_on_connection,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
    load_historical_voting_candidate_search_artifact,
    resolve_voting_candidate_search_selection,
    resolve_voting_candidate_search_inputs,
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
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.strategy.dsl_delivery import MAX_EQUIVALENCE_ROWS
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
    RAW_SCORE_PRODUCT,
)
from marvis.packs.modeling.evidence_tools import (
    build_training_evidence_ref,
    load_modeling_training_evidence_artifacts,
)
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
    load_model_score_evidence_artifacts,
)
from marvis.packs.strategy.candidate_fragment import verified_fragment_pool_parts
from marvis.packs.strategy.scorecard_candidate import (
    SCORECARD_BAND_ASSET_ARTIFACT_KIND,
    SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_BAND_ASSET_ORIGIN_TOOL,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
    SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION,
    SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
    scorecard_cutoff_selection_to_verified_candidate_fragment,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    load_scorecard_band_asset_artifact,
    load_scorecard_cutoff_selection_artifact,
)
from marvis.packs.strategy.model_evidence_tools import (
    MODEL_EVIDENCE_V2_ARTIFACT_KIND,
    _MAX_UNIVARIATE_SOURCES,
    _load_candidate_sources,
    _validate_inputs as _validate_model_evidence_v2_inputs,
    load_strategy_model_evidence_v2_artifact,
)
from marvis.packs.strategy.impact_cube_tools import IMPACT_CUBE_ARTIFACT_KIND
from marvis.packs.strategy.pool_impact_tools import (
    POOL_IMPACT_ARTIFACT_KIND,
    load_historical_strategy_pool_impact_artifact,
)
from marvis.packs.strategy.pool_tools import (
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
)
from marvis.packs.strategy.pool_validation_tools import (
    load_strategy_pool_validation_artifacts,
    select_latest_strategy_pool_validation_refs,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    pool_requirement_bindings_provenance,
    project_pool_entry_requirements,
    resolve_pool_requirements,
)
from marvis.packs.strategy.project_context_tools import (
    load_current_strategy_project_context_artifact,
)
from marvis.packs.strategy.report_bundle_adapters import (
    build_strategy_report_bundle_source_inputs,
    validate_candidate_stability_report_compatibility,
)
from marvis.packs.strategy.report_bundle_tools import (
    load_strategy_impact_cube_artifact,
)
from marvis.packs.strategy.sample_design_binding import (
    StrategySampleDesignRef,
    load_strategy_sample_design_execution_binding,
)
from marvis.packs.strategy.sample_design_tools import (
    SAMPLE_DESIGN_ARTIFACT_KIND,
    SAMPLE_DESIGN_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_ORIGIN_TOOL,
    load_strategy_sample_design_v2_artifacts,
)
from marvis.repositories.plans import PlanRepository
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestConflictError,
    PendingStrategyRequestNotFoundError,
    PendingStrategyRequestRepository,
)
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    TaskArtifactRepository,
)
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    StrategyCandidatePoolRepository,
    strategy_pool_snapshot_hash,
)
from marvis.repositories.strategy_project_context import (
    StrategyProjectContextDataError,
    StrategyProjectContextRepository,
)
from marvis.repositories.strategy_reports import StrategyReportRepository
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    validate_candidate_asset,
)
from marvis.packs.strategy.candidate_asset_tools import (
    load_verified_candidate_refinement_source,
)
from marvis.packs.strategy.candidate_stability_tools import (
    ARTIFACT_KIND as CANDIDATE_STABILITY_ARTIFACT_KIND,
    load_candidate_stability_artifact,
    resolve_candidate_monthly_stability_inputs,
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
    strategy_request: Mapping[str, object] | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    recovery_bypass: bool = False,
) -> dict:
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
    text = str(user_text or "")
    if task.task_type == TASK_TYPE_STRATEGY and (
        utterance_targets_candidate_monthly_stability(text)
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
    r"(?:开发|设计|制定|创建|生成|构建|训练|物化|固化|冻结|探索|整理|梳理|汇总|归集|收集|刷新|更新|复盘|盘点|记录|做|计算|测算|分析|评估|查看|看一下|看下|回测|测试|验证|回放|应用|执行|写回|回写|回填|打标|"
    r"对比|比较|采纳|采用|上线|报告|文档|监控|漂移|挖掘|选择|筛选|保留|合并|编辑|"
    r"添加|加入|入池|删除|移除|排序|重排|改为|编译|预览|"
    r"develop|design|create|build|train|materialize|aggregate|collect|compute|calculate|analy[sz]e|evaluate|backtest|validate|replay|run|apply|compare|"
    r"adopt|report|monitor|mine|refine|select|merge|add|remove|delete|reorder|compile|preview)",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_SUBJECT_RE = re.compile(
    r"(?:策略|策略项目上下文|项目上下文|当前项目(?:现状|情况)|历史(?:版本)?策略|策略样本|样本设计|样本边界|策略池|规则池|准入|审批|拒绝|额度|授信|定价|利率|分群|分层|规则|候选|候选箱|单变量|分箱|自动树|决策树|叶子|叶节点|投票|Voting|n[-_ ]?of[-_ ]?k|(?:二维|2\s*[dD])?\s*(?:交叉|cross)\s*(?:矩阵|matrix)|cutoff|利润|收益|"
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
_STRATEGY_POOL_VALIDATION_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:独立样本|独立回放|回放验证|独立验证|"
    r"independent\s+(?:sample\s+)?replay|independent\s+validation|"
    r"replay\s+validation))"
    r"(?=.*(?:验证集|验证样本|验证分区|"
    r"(?<![A-Za-z0-9_])(?:validation|oot)(?![A-Za-z0-9_])))",
    re.IGNORECASE,
)
_STRATEGY_NAN_LABEL_META_KEY = "strategy_nan_label_confirmation"
_PROJECT_CONTEXT_UNAVAILABLE_ANSWER_RE = re.compile(
    r"(?:暂时没有|暂缺|暂无|没有|未提供|不可用|不知道|未知|待补充|"
    r"unavailable|not\s+available|unknown|missing)",
    re.IGNORECASE,
)
_PROJECT_CONTEXT_ALL_PENDING_RE = re.compile(
    r"(?:这些|上述|以上|全部|所有|都|all\s+of\s+them|all)",
    re.IGNORECASE,
)
_PROJECT_CONTEXT_ANSWER_PATTERNS = {
    "current.status_fields.volume": re.compile(
        r"申请量|进件量|放款量|业务量|规模|volume", re.IGNORECASE
    ),
    "current.status_fields.approval": re.compile(
        r"通过率|审批率|准入率|approval", re.IGNORECASE
    ),
    "current.status_fields.risk": re.compile(
        r"坏账率|风险率|逾期率|risk|bad\s+rate", re.IGNORECASE
    ),
    "current.status_fields.economics": re.compile(
        r"收益|利润|成本|经济|economics|profit", re.IGNORECASE
    ),
    "current.maturity_summary": re.compile(
        r"成熟度|表现窗|观察窗|maturity|performance\s+window", re.IGNORECASE
    ),
    "historical_strategy_reviews": re.compile(
        r"历史(?:版本)?策略|历史材料|旧版策略|上一版策略|history|historical",
        re.IGNORECASE,
    ),
}


class _StrategySampleDesignRequiredError(StrategySetupError):
    """The current strategy request has no exact mature sample-design binding."""


class _StrategyV2EvidenceSetupError(StrategySetupError):
    """Typed preflight failure for platform-owned V2 evidence discovery."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_STRATEGY_V2_ARTIFACT_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    sqlite3.Error,
)


_STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE = re.compile(
    r"(?:Strategy\s+Model\s*Evidence(?:\s+V2)?|"
    r"Model\s*Evidence(?:\s+V2)?|模型证据(?:\s*V2)?|"
    r"单变量(?:候选)?证据(?:包|汇总)?|认证单变量(?:候选)?(?:证据|结果))",
    re.IGNORECASE,
)
_STRATEGY_SAMPLE_V2_POLICY = {
    "minimum_partition_count": 1,
    "minimum_bad_count": 1,
    "minimum_label_coverage": 0.8,
    "minimum_historical_score_coverage": 0.8,
    "maximum_group_coverage_gap": 0.2,
    "diagnostic_severities": {
        "entity_overlap": "fail",
        "temporal_oot": "fail",
        "risk_outside_approval": "fail",
        "maturity": "fail",
        "label_coverage": "fail",
        "historical_score_coverage": "warn",
        "group_coverage_gap": "warn",
        "sufficiency": "fail",
    },
}


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
        utterance_targets_candidate_monthly_stability(text)
        or utterance_targets_interactive_tree_frontier_group_materialization(
            text
        )
        or utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
        or utterance_targets_strategy_dsl_delivery(text)
        or utterance_targets_strategy_pool_stability(text)
        or _STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE.search(text)
        or (
            _STRATEGY_REQUEST_ACTION_RE.search(text)
            and _STRATEGY_REQUEST_SUBJECT_RE.search(text)
        )
    )


_MANUAL_STRATEGY_WORKFLOWS = frozenset(
    {
        "univariate_candidate_analysis",
        "cross_matrix_analysis",
        "automatic_tree_candidate_build",
        "univariate_candidate_refinement",
        "scorecard_band_build",
        "scorecard_cutoff_selection",
        "candidate_monthly_stability",
        "voting_candidate_search",
        "voting_candidate_build_from_search",
        "interactive_tree_revision",
        "interactive_tree_frontier_group_materialization",
        "interactive_tree_frontier_materialization",
        "strategy_pool_add_candidate",
        "strategy_pool_compile",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_apply",
        "strategy_pool_validation",
        "strategy_pool_stability",
    }
)


def _handle_structured_strategy_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
    strategy_request: Mapping[str, object],
) -> dict:
    """Validate and run one LLM-free Candidate Lab request.

    The HTTP adapter constrains this envelope, while this core boundary keeps
    direct callers fail-closed. Platform-owned bindings are still selected by
    the same preparation path used by natural-language requests.
    """

    if task.task_type != TASK_TYPE_STRATEGY:
        raise DriverError("strategy_request 只能用于 strategy 类型任务。")
    workflow = strategy_request.get("workflow")
    if (
        not isinstance(workflow, str)
        or workflow not in _MANUAL_STRATEGY_WORKFLOWS
    ):
        raise DriverError("strategy_request 包含未开放的 Candidate Lab workflow。")

    conversation = repo.list_agent_messages(task.id)
    if _active_plan(runtime.plan_repo, task.id) is not None:
        raise DriverError("当前策略任务已有进行中的计划，不能启动新的 Candidate Lab 请求。")
    if latest_open_gate(conversation) is not None:
        raise DriverError("当前策略任务有待处理确认门，不能启动新的 Candidate Lab 请求。")

    pending = _latest_strategy_request_pending(conversation)
    if pending is not None:
        _invalidate_pending_strategy_request(runtime, task, pending)

    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=str(user_text or "").strip(),
        metadata={
            "intent": "strategy_request",
            "request_source": "manual_ui",
            "workflow": workflow,
        },
    )

    preview = None
    preview_error = None
    try:
        preview = _strategy_dataset_preview(runtime, task)
    except StrategySetupError as exc:
        preview_error = str(exc)

    compilation = validate_strategy_request(
        strategy_request,
        allowed_columns=_strategy_request_allowed_columns(preview),
        target_col=None if preview is None else preview.target_col,
    )
    if compilation.draft is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=(
                compilation.clarification_code
                or "strategy_request_needs_clarification"
            ),
            message=compilation.clarification or "请修正 Candidate Lab 策略请求。",
            fields=compilation.clarification_fields,
        )
    draft = compilation.draft
    if (
        not isinstance(draft, StandardWorkflowRequestDraft)
        or draft.workflow != workflow
    ):
        raise DriverError("Candidate Lab 请求未编译为预期的标准 Workflow。")

    preflight = _strategy_request_preflight(runtime, task, draft)
    if preflight is not None:
        code, message = preflight
        return _strategy_request_clarification_response(
            repo,
            task,
            code=code,
            message=message,
        )
    if _strategy_request_requires_dataset(draft) and preview is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_dataset_context_required",
            message=preview_error or "当前策略操作需要一个任务内样本。",
        )
    if _strategy_request_requires_target(draft) and (
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
        draft,
        preview=preview,
        auto_start=True,
        source_message=source_message,
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

    # An explicit project-context command is a new workflow request, even when
    # it also happens to answer one pending field.  Only bare follow-up answers
    # use the shortcut below; otherwise a refresh could be mistaken for an
    # answer and silently skip the newly supplied as-of/scope controls.
    project_context_answer = (
        None
        if (
            utterance_targets_strategy_project_context(text)
            or _is_strategy_request_intent(text)
        )
        else _maybe_handle_project_context_missing_answer(
            runtime,
            repo,
            task,
            text=text,
            conversation=conversation,
        )
    )
    if project_context_answer is not None:
        return project_context_answer

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

    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "strategy_request",
            "request_source": "agent_nl",
        },
    )
    preview = None
    preview_error = None
    is_project_context_request = utterance_targets_strategy_project_context(text)
    is_sample_design_request = utterance_targets_strategy_sample_design(text)
    is_report_bundle_v2_request = utterance_targets_strategy_report_bundle_v2(
        text
    )
    is_candidate_stability_request = (
        utterance_targets_candidate_monthly_stability(text)
    )
    is_pool_validation_request = (
        _STRATEGY_POOL_VALIDATION_REQUEST_RE.search(text) is not None
    )
    is_scorecard_request = (
        utterance_targets_scorecard_band_build(text)
        or utterance_targets_scorecard_cutoff_selection(text)
    )
    is_impact_cube_request = utterance_targets_strategy_impact_cube(text)
    is_pool_stability_request = utterance_targets_strategy_pool_stability(text)
    is_model_evidence_v2_request = (
        _STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE.search(text) is not None
    )
    try:
        preview = (
            None
            if (
                is_project_context_request
                or is_model_evidence_v2_request
                or is_report_bundle_v2_request
                or is_candidate_stability_request
                or is_pool_validation_request
                or is_pool_stability_request
                or is_scorecard_request
            )
            else (
                _strategy_impact_cube_dataset_preview(runtime, task)
                if is_impact_cube_request
                else (
                    _strategy_pool_impact_dataset_preview(runtime, task)
                    if _STRATEGY_POOL_IMPACT_REQUEST_RE.search(text)
                    else (
                        _strategy_sample_design_dataset_preview(runtime, task)
                        if is_sample_design_request
                        else _strategy_dataset_preview(runtime, task)
                    )
                )
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
        source_message=source_message,
    )


def _maybe_handle_project_context_missing_answer(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    text: str,
    conversation: Sequence[Mapping],
) -> dict | None:
    """Turn a direct answer to a pending context question into an audited refresh.

    The answer is retained as user evidence.  It is never parsed into a
    deterministic metric here; the Tool labels such values as user-provided
    and unverified, while an explicit unavailable answer remains null.
    """

    if (
        _active_plan(runtime.plan_repo, task.id) is not None
        or latest_open_gate(list(conversation)) is not None
    ):
        return None
    try:
        current = StrategyProjectContextRepository(
            runtime.settings.db_path
        ).get_current(task.id)
    except (StrategyProjectContextDataError, KeyError, TypeError, ValueError):
        return None
    if current is None:
        return None
    pending = [
        record
        for record in current["state"]["missing_information_records"]
        if record["status"] == "pending"
    ]
    if not pending:
        return None
    pending_paths = {record["field_path"] for record in pending}
    mentioned = [
        field_path
        for field_path, pattern in _PROJECT_CONTEXT_ANSWER_PATTERNS.items()
        if field_path in pending_paths and pattern.search(text)
    ]
    unavailable = _PROJECT_CONTEXT_UNAVAILABLE_ANSWER_RE.search(text) is not None
    if not mentioned and unavailable:
        if _PROJECT_CONTEXT_ALL_PENDING_RE.search(text):
            mentioned = sorted(pending_paths)
        elif len(pending_paths) == 1:
            mentioned = list(pending_paths)
        else:
            repo.add_agent_message(
                task.id,
                role="user",
                stage="chat",
                content=text,
                metadata={"intent": "strategy_project_context_answer_ambiguous"},
            )
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_project_context_answer_field_required",
                message=(
                    "请说明哪些字段暂时没有；可点名通过率、风险率、业务量、"
                    "收益成本、样本成熟度或历史策略，也可以明确说“以上全部暂时没有”。"
                ),
                fields=tuple(sorted(pending_paths)),
            )
    if not mentioned:
        return None

    source_message = repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={
            "intent": "strategy_project_context_answer",
            "field_paths": list(mentioned),
            "answer_status": "unavailable" if unavailable else "provided",
        },
    )
    workflow_inputs = {
        "as_of": current["state"]["as_of"],
        "business_context": (
            {} if unavailable else {field_path: text for field_path in mentioned}
        ),
        "explicit_unavailable": list(mentioned) if unavailable else [],
        "external_report_filenames": [],
    }
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_project_context",
        workflow_inputs=workflow_inputs,
    )
    return _prepare_and_run_validated_strategy_request(
        runtime,
        repo,
        task,
        draft,
        preview=None,
        auto_start=True,
        source_message=source_message,
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
    source_message: Mapping | None = None,
) -> dict:
    """Bind current evidence, resolve the NaN policy, then instantiate once."""

    expected_impact_cube_sample_binding = None
    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_impact_cube"
        and preview is not None
    ):
        identity = getattr(preview, "identity", None)
        if not isinstance(identity, Mapping):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_impact_cube_sample_invalid",
                message=(
                    "ImpactCube 编译预览缺少认证 SampleDesign 身份；"
                    "请重新固化样本设计后重试。"
                ),
            )
        expected_impact_cube_sample_binding = dict(identity)

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and (
            draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
            or draft.workflow
            in {
                "strategy_sample_design",
                "strategy_sample_design_v2",
                "limit_pricing_matrix",
            }
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
        and draft.workflow
        in {"strategy_sample_design", "strategy_sample_design_v2"}
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
            expected_impact_cube_sample_binding=(
                expected_impact_cube_sample_binding
            ),
            source_message=source_message,
        )
    except _StrategyV2EvidenceSetupError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=exc.code,
            message=str(exc),
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
    expected_impact_cube_sample_binding: Mapping | None = None,
    source_message: Mapping | None = None,
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
        and draft.workflow == "strategy_pool_stability"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_pool_stability",
            slots=_strategy_pool_stability_plan_slots(
                runtime,
                task,
                draft,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_dsl_delivery"
    ):
        if context is None:
            raise StrategySetupError(
                "策略代码交付需要当前任务内唯一且已认证的数据集。"
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_dsl_delivery",
            slots=_strategy_dsl_delivery_plan_slots(
                runtime,
                task,
                draft,
                context=context,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_project_context"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_project_context",
            slots=_strategy_project_context_plan_slots(
                runtime,
                task,
                draft,
                source_message=source_message,
            ),
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
        and draft.workflow == "interactive_tree_revision"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_interactive_tree_revision",
            slots=dict(draft.to_dict()["workflow_inputs"]),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        == "interactive_tree_frontier_group_materialization"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=(
                "strategy_interactive_tree_frontier_group_materialization"
            ),
            slots=dict(draft.to_dict()["workflow_inputs"]),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "interactive_tree_frontier_materialization"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_interactive_tree_frontier_materialization",
            slots=dict(draft.to_dict()["workflow_inputs"]),
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
        and draft.workflow == "voting_candidate_search"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_voting_candidate_search",
            slots=_strategy_voting_candidate_search_plan_slots(
                runtime,
                task,
                draft,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "voting_candidate_build_from_search"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_voting_candidate_build_from_search",
            slots=_strategy_voting_candidate_build_from_search_plan_slots(
                runtime,
                task,
                draft,
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
        and draft.workflow == "candidate_monthly_stability"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_candidate_monthly_stability",
            slots=_candidate_monthly_stability_plan_slots(
                runtime,
                task,
                draft,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "scorecard_band_build"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_scorecard_band_build",
            slots=_scorecard_band_build_plan_slots(runtime, task, draft),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "scorecard_cutoff_selection"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_scorecard_cutoff_selection",
            slots=_scorecard_cutoff_selection_plan_slots(
                runtime,
                task_id=task.id,
                draft=draft,
            ),
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
        and draft.workflow == "strategy_impact_cube"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_impact_cube",
            slots=_strategy_impact_cube_plan_slots(
                runtime,
                task,
                draft,
                expected_sample_binding=expected_impact_cube_sample_binding,
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
        and draft.workflow == "strategy_sample_design_v2"
    ):
        if context is None:
            raise StrategySetupError(
                "V2 策略样本设计需要确认的活动 DataWorkspace 和二元目标列。"
            )
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_sample_design_v2",
            slots=_strategy_sample_design_v2_plan_slots(
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
        and draft.workflow == "strategy_model_evidence_v2"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_model_evidence_v2",
            slots=_strategy_model_evidence_v2_plan_slots(
                runtime,
                task,
                verify_current=True,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_report_bundle_v2"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_report_bundle_v2",
            slots=_strategy_report_bundle_v2_plan_slots(
                runtime,
                task,
                draft,
                source_message=source_message,
            ),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_validation"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_pool_validation",
            slots=_strategy_pool_validation_plan_slots(runtime, task, draft),
            auto_start=auto_start,
        )

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_apply"
    ):
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_pool_apply",
            slots=_strategy_pool_apply_plan_slots(runtime, task, draft),
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

    if (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "univariate_candidate_refinement"
        and "source_candidate_id" in draft.workflow_inputs
    ):
        workflow_inputs = draft.to_dict()["workflow_inputs"]
        source_candidate_id = str(workflow_inputs.pop("source_candidate_id"))
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id="strategy_univariate_candidate_refinement_existing",
            slots={
                **workflow_inputs,
                **_candidate_source_artifact_slots(
                    runtime,
                    task_id=task.id,
                    candidate_id=source_candidate_id,
                    workflow_inputs=workflow_inputs,
                ),
            },
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
                    workflow_inputs=draft.workflow_inputs,
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
    persisted_payload = pending_record.validated_draft
    persisted_workflow = (
        persisted_payload.get("workflow")
        if isinstance(persisted_payload, Mapping)
        else None
    )
    persisted_sample_design = persisted_workflow in {
        "strategy_sample_design",
        "strategy_sample_design_v2",
    }
    preview = None
    preview_error = None
    try:
        preview = (
            _strategy_sample_design_dataset_preview(runtime, task)
            if persisted_sample_design
            else _strategy_dataset_preview(runtime, task)
        )
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
        allow_legacy_replay=True,
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
        is_sample_design = (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            in {"strategy_sample_design", "strategy_sample_design_v2"}
        )
        try:
            context = (
                _strategy_sample_design_dataset_context(runtime, task)
                if is_sample_design
                else _strategy_dataset_context(
                    runtime,
                    task,
                    require_target=_strategy_request_requires_target(draft),
                )
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
            use_sample_design_workspace=is_sample_design,
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
            confirmed_drop_nan = (
                isinstance(draft, StandardWorkflowRequestDraft)
                and draft.workflow_inputs.get("drop_nan_labels") is True
            )
            if n_nan and not confirmed_drop_nan:
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
        confirmed_drop_nan = (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow_inputs.get("drop_nan_labels") is True
        )
        return _run_validated_strategy_request(
            runtime,
            repo,
            task,
            draft,
            context=context,
            auto_start=False,
            drop_nan_labels=confirmed_drop_nan,
        )
    except _StrategyV2EvidenceSetupError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code=exc.code,
            message=str(exc),
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
    if draft.workflow == "strategy_project_context":
        # The exact persisted user message is request-local evidence.  It is
        # bound when the validated request starts, not reconstructed here from
        # whichever message happens to be latest during preflight.
        return None
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
    if draft.workflow == "strategy_sample_design_v2":
        try:
            context = _strategy_sample_design_dataset_context(runtime, task)
            _strategy_sample_design_v2_plan_slots(
                runtime,
                task,
                draft,
                context=context,
                drop_nan_labels=bool(
                    draft.workflow_inputs.get("drop_nan_labels", False)
                ),
            )
        except _StrategyV2EvidenceSetupError as exc:
            return (exc.code, str(exc))
        except StrategySetupError as exc:
            return ("strategy_sample_design_v2_workspace_required", str(exc))
        return None
    if draft.workflow == "strategy_model_evidence_v2":
        try:
            _strategy_model_evidence_v2_plan_slots(runtime, task)
        except _StrategyV2EvidenceSetupError as exc:
            return (exc.code, str(exc))
        except StrategySetupError as exc:
            return ("strategy_model_evidence_v2_binding_required", str(exc))
        return None
    if draft.workflow == "candidate_monthly_stability":
        try:
            _candidate_monthly_stability_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            message = str(exc)
            code = (
                "candidate_monthly_stability_month_required"
                if "月份字段" in message or "month field" in message
                else "candidate_monthly_stability_binding_required"
            )
            return (code, message)
        return None
    if draft.workflow == "strategy_report_bundle_v2":
        # Bind exact refs only once, immediately before plan creation. A
        # separate preflight read would open a second selection window where
        # current source/report heads could silently rebind.
        return None
    if draft.workflow == "strategy_pool_validation":
        # Pool and exact mature SampleDesign V2 refs are authenticated together
        # once immediately before plan creation. The Tool repeats the exact-ref
        # checks under its artifact publication lock.
        return None
    if draft.workflow == "strategy_pool_apply":
        # Select and authenticate the exact current nonempty Pool only once,
        # immediately before plan creation. The Tool revalidates the CAS under
        # its writer lock before deriving any dataset.
        return None
    if draft.workflow == "strategy_dsl_delivery":
        # Strategy and dataset refs are selected and authenticated together
        # exactly once immediately before plan creation.
        return None
    if draft.workflow == "strategy_impact_cube":
        # SampleDesign, Pool and optional current Strategy are selected and
        # authenticated together exactly once immediately before plan creation.
        return None
    if draft.workflow == "strategy_pool_stability":
        # The exact current Pool and latest complete SampleDesign V2 are bound
        # once during plan creation. The first step publishes the exact
        # ImpactCube and the second consumes only its direct output refs.
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
    if draft.workflow == "interactive_tree_revision":
        # The Tool resolves and authenticates the exact task-local tree or
        # revision under the same writer lock used for replay and persistence.
        # A separate preflight lookup would create a second binding window.
        return None
    if draft.workflow == "interactive_tree_frontier_group_materialization":
        # The Tool resolves the revision, canonicalizes all requested members
        # against its live frontier, and authenticates ancestry under one lock.
        return None
    if draft.workflow == "interactive_tree_frontier_materialization":
        # The Tool resolves the revision and recursively authenticates its
        # ancestry under the same writer lock used to register the pointer.
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
    if draft.workflow == "voting_candidate_search":
        try:
            _strategy_voting_candidate_search_plan_slots(runtime, task, draft)
        except StrategySetupError as exc:
            return ("strategy_voting_search_pool_binding_required", str(exc))
        return None
    if draft.workflow == "voting_candidate_build_from_search":
        try:
            _strategy_voting_candidate_build_from_search_plan_slots(
                runtime,
                task,
                draft,
            )
        except StrategySetupError as exc:
            return ("strategy_voting_search_selection_binding_required", str(exc))
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
                    workflow_inputs=draft.workflow_inputs,
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


def _candidate_monthly_stability_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Resolve a user pointer to one exact, executable Tool input branch."""

    if draft.workflow != "candidate_monthly_stability":
        raise StrategySetupError(
            "候选逐月稳定性 slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    if set(inputs) == {"asset_id"}:
        user_pointer: dict[str, object] = {
            "source_kind": "univariate_asset",
            "asset_id": inputs["asset_id"],
        }
    elif set(inputs) == {"strategy_type", "entry_id"}:
        user_pointer = {
            "source_kind": "pool_entry",
            "strategy_type": inputs["strategy_type"],
            "entry_id": inputs["entry_id"],
        }
    else:  # pragma: no cover - compiler validation owns this shape
        raise StrategySetupError(
            "候选逐月稳定性必须提供唯一候选资产，或 Pool 类型与唯一 entry。"
        )
    try:
        resolved = resolve_candidate_monthly_stability_inputs(
            _strategy_v2_read_runtime(runtime),
            task_id=task.id,
            user_pointer=user_pointer,
        )
    except (
        StrategyError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        message = str(exc)
        if "month field" in message:
            raise StrategySetupError(
                "当前受治理 StrategySampleDesign 没有唯一且非空的月份字段；"
                "请先补充并重新固化 month 口径，再测算候选逐月稳定性。"
            ) from exc
        raise StrategySetupError(
            "候选逐月稳定性来源、活动 workspace、SampleDesign 或 lineage "
            f"未通过完整认证：{message}"
        ) from exc
    return dict(resolved)


def _scorecard_registry_token(
    artifacts: Sequence[Mapping],
) -> str:
    """CAS all score/sample rows that may change a Scorecard source choice."""

    supported = {
        (
            MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
            MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        ),
        (
            MODEL_SCORE_VECTOR_ARTIFACT_KIND,
            MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL,
        ),
        (
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        ),
        (
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        ),
        (
            SCORECARD_BAND_ASSET_ARTIFACT_KIND,
            SCORECARD_BAND_ASSET_ORIGIN_TOOL,
        ),
        (
            SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND,
            SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL,
        ),
    }
    relevant = [
        {
            "id": artifact.get("id"),
            "kind": artifact.get("kind"),
            "content_hash": artifact.get("content_hash"),
            "origin_tool": artifact.get("origin_tool"),
            "provenance": artifact.get("provenance"),
            "created_at": artifact.get("created_at"),
        }
        for artifact in artifacts
        if (artifact.get("kind"), artifact.get("origin_tool")) in supported
    ]
    try:
        payload = json.dumps(
            relevant,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_registry_invalid",
            "Scorecard source artifact registry 无法规范化；本次未创建计划。",
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _scorecard_artifact_snapshot(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
) -> tuple[Mapping, ...]:
    try:
        artifacts = tuple(read_runtime.task_artifacts.list_for_task(task_id))
    except _STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_registry_unavailable",
            "无法读取当前任务的 Scorecard source artifact registry。",
        ) from exc
    if any(not isinstance(artifact, Mapping) for artifact in artifacts):
        raise _StrategyV2EvidenceSetupError(
            "scorecard_registry_invalid",
            "Scorecard source artifact registry 含无效记录。",
        )
    return artifacts


def _scorecard_ref_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_source_ref_invalid",
            f"最新 Scorecard source 的 {field} 缺少完整 64 位 hash。",
        )
    return value


def _scorecard_score_evidence_ref(record: Mapping) -> dict[str, str]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_evidence_invalid",
            "待判定模型评分证据缺少完整 provenance；平台不会静默跳过或"
            "回退旧 Scorecard 证据。",
        )
    return {
        "evidence_artifact_id": _scorecard_ref_hash(
            record.get("id"),
            field="evidence_artifact_id",
        ),
        "expected_evidence_artifact_content_hash": _scorecard_ref_hash(
            record.get("content_hash"),
            field="expected_evidence_artifact_content_hash",
        ),
        "score_vector_artifact_id": _scorecard_ref_hash(
            provenance.get("score_vector_artifact_id"),
            field="score_vector_artifact_id",
        ),
        "expected_score_vector_artifact_content_hash": _scorecard_ref_hash(
            provenance.get("score_vector_artifact_content_hash"),
            field="expected_score_vector_artifact_content_hash",
        ),
    }


def _scorecard_score_evidence_contract(score: object) -> bool:
    """Return False only for a fully authenticated, clearly non-scorecard model."""

    try:
        training = score.training
        evidence = training.evidence
        evidence_experiment = evidence["experiment"]
        evidence_model = evidence["model_artifact"]
        identities = (
            training.experiment.recipe_id,
            training.model_artifact.algorithm,
            evidence_experiment["recipe_id"],
            evidence_model["algorithm"],
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_evidence_invalid",
            "待判定模型评分证据缺少一致的 recipe/algorithm 身份。",
        ) from exc
    scorecard_flags = tuple(value == "scorecard" for value in identities)
    if not any(scorecard_flags):
        return False
    if not all(scorecard_flags):
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_evidence_invalid",
            "待判定评分证据的 Scorecard recipe/algorithm 身份不一致。",
        )
    metadata = evidence_model.get("scoring_metadata")
    envelope = getattr(score, "envelope", None)
    scoring_contract = (
        envelope.get("scoring_contract")
        if isinstance(envelope, Mapping)
        else None
    )
    if (
        not isinstance(metadata, Mapping)
        or not isinstance(envelope, Mapping)
        or not isinstance(scoring_contract, Mapping)
        or envelope.get("score_product") != RAW_SCORE_PRODUCT
        or metadata.get("score_product") != RAW_SCORE_PRODUCT
        or scoring_contract.get("score_direction") != "higher_is_riskier"
        or metadata.get("score_direction") != "higher_is_riskier"
        or metadata.get("points_direction") != "higher_is_better"
        or metadata.get("calibration_status") != "not_applied"
        or not isinstance(metadata.get("scorecard_table"), list)
        or not metadata["scorecard_table"]
    ):
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_contract_invalid",
            "最新 Scorecard 评分证据必须包含 raw uncalibrated bad probability、"
            "higher-is-riskier 分数方向、higher-is-better points 与完整"
            " scorecard_table。",
        )
    return True


def _scorecard_band_build_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind the newest exact score evidence and latest compatible sample."""

    if draft.workflow != "scorecard_band_build":
        raise StrategySetupError("Scorecard 分数带 slots 收到了错误的 Workflow。")
    inputs = draft.to_dict()["workflow_inputs"]
    read_runtime = _strategy_report_read_runtime(runtime)
    artifacts = _scorecard_artifact_snapshot(read_runtime, task_id=task.id)
    registry_token = _scorecard_registry_token(artifacts)
    try:
        sample = _latest_verified_strategy_sample_design_v2_binding(
            read_runtime,
            task_id=task.id,
            artifacts=artifacts,
        )
        sample_ref = _strategy_report_sample_ref(sample)
    except _StrategyV2EvidenceSetupError as exc:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_sample_invalid",
            "Scorecard 分数带需要最新且完整认证的 StrategySampleDesign V2；"
            "平台不会回退到旧样本。",
        ) from exc

    score_records = [
        artifact
        for artifact in artifacts
        if artifact.get("kind") == MODEL_SCORE_EVIDENCE_ARTIFACT_KIND
        and artifact.get("origin_tool")
        == MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_ORIGIN_TOOL
    ]
    if not score_records:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_evidence_required",
            "当前任务还没有可用于 Scorecard 分数带的模型评分证据；"
            "请先生成受治理的 raw bad-probability score evidence。",
        )
    score_ref: dict[str, str] | None = None
    for record in reversed(score_records):
        candidate_ref = _scorecard_score_evidence_ref(record)
        try:
            candidate = load_model_score_evidence_artifacts(
                read_runtime,
                task_id=task.id,
                **candidate_ref,
            )
        except (
            ModelingError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "scorecard_band_score_evidence_invalid",
                "最新待判定模型评分证据未通过文件、hash、registry、模型或"
                "分数向量完整认证；平台不会静默跳过或回退旧 Scorecard 证据。",
            ) from exc
        if not _scorecard_score_evidence_contract(candidate):
            # A fully authenticated non-scorecard result cannot satisfy this
            # workflow and is safe to skip while searching backward.
            continue
        try:
            training_ref = build_training_evidence_ref(candidate.training)
        except (ModelingError, KeyError, TypeError, ValueError) as exc:
            raise _StrategyV2EvidenceSetupError(
                "scorecard_band_score_evidence_invalid",
                "最新 Scorecard 评分证据的 TrainingEvidence 引用不完整。",
            ) from exc
        if training_ref.get("sample_design_ref") != sample_ref:
            raise _StrategyV2EvidenceSetupError(
                "scorecard_band_sample_incompatible",
                "最新 Scorecard 评分证据与最新 StrategySampleDesign V2 "
                "不属于同一不可变样本；请基于当前样本重新生成评分证据。",
            )
        score_ref = candidate_ref
        break
    if score_ref is None:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_score_evidence_required",
            "当前任务的模型评分证据均不是完整认证的 Scorecard raw-PD "
            "评分证据；请先完成 Scorecard 训练与评分证据物化。",
        )

    refreshed = _scorecard_artifact_snapshot(read_runtime, task_id=task.id)
    if _scorecard_registry_token(refreshed) != registry_token:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_band_source_changed",
            "Scorecard 的评分证据或 SampleDesign 在计划创建前发生变化；"
            "请基于最新证据重试。",
        )

    slots: dict[str, object] = {
        "score_evidence_ref": score_ref,
        "sample_design_ref": sample_ref,
    }
    if "bin_count" in inputs:
        slots["banding"] = {
            "method": "equal_frequency",
            "bin_count": inputs["bin_count"],
        }
    elif "raw_pd_band_edges" in inputs:
        slots["raw_pd_band_edges"] = list(inputs["raw_pd_band_edges"])
    return slots


def _scorecard_cutoff_selection_plan_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind one explicit asset/cutoff pair to one authenticated full band."""

    if draft.workflow != "scorecard_cutoff_selection":
        raise StrategySetupError(
            "Scorecard cutoff selection slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    asset_id = inputs.get("asset_id")
    cutoff_id = inputs.get("cutoff_id")
    read_runtime = _strategy_report_read_runtime(runtime)
    artifacts = _scorecard_artifact_snapshot(read_runtime, task_id=task_id)
    registry_token = _scorecard_registry_token(artifacts)
    matches = []
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") == SCORECARD_BAND_ASSET_ARTIFACT_KIND
            and artifact.get("origin_tool") == SCORECARD_BAND_ASSET_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("schema_version")
            == SCORECARD_BAND_ASSET_ARTIFACT_SCHEMA_VERSION
            and provenance.get("asset_id") == asset_id
        ):
            matches.append(artifact)
    if not matches:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_cutoff_source_required",
            f"当前任务没有完整 Scorecard 分数带 {asset_id}；"
            "请从最新结果复制完整 asset ID。",
        )
    if len(matches) != 1:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_cutoff_source_ambiguous",
            f"Scorecard 分数带 {asset_id} 对应多个 artifact，"
            "当前不能安全选择来源。",
        )
    record = matches[0]
    provenance = record["provenance"]
    assert isinstance(provenance, Mapping)
    artifact_id = _scorecard_ref_hash(
        record.get("id"),
        field="source_artifact_id",
    )
    content_hash = _scorecard_ref_hash(
        record.get("content_hash"),
        field="expected_source_artifact_content_hash",
    )
    asset_hash = _scorecard_ref_hash(
        provenance.get("asset_hash"),
        field="expected_asset_hash",
    )
    try:
        binding = load_scorecard_band_asset_artifact(
            read_runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_artifact_content_hash=content_hash,
            expected_asset_id=str(asset_id),
            expected_asset_hash=asset_hash,
        )
    except (
        StrategyError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_cutoff_source_invalid",
            "用户点名的完整 Scorecard 分数带未通过文件、hash、registry、"
            "score evidence 或 SampleDesign 完整认证。",
        ) from exc
    cutoffs = binding.asset.get("cutoffs")
    if (
        binding.asset.get("asset_id") != asset_id
        or binding.asset.get("asset_hash") != asset_hash
        or not isinstance(cutoffs, Sequence)
        or isinstance(cutoffs, str | bytes | bytearray)
        or len(
            [
                cutoff
                for cutoff in cutoffs
                if isinstance(cutoff, Mapping)
                and cutoff.get("cutoff_id") == cutoff_id
            ]
        )
        != 1
    ):
        raise _StrategyV2EvidenceSetupError(
            "scorecard_cutoff_pointer_invalid",
            "用户点名的 cutoff 不属于该完整 Scorecard 分数带；"
            "平台不会替换、排名或推荐其他 cutoff。",
        )
    refreshed = _scorecard_artifact_snapshot(read_runtime, task_id=task_id)
    if _scorecard_registry_token(refreshed) != registry_token:
        raise _StrategyV2EvidenceSetupError(
            "scorecard_cutoff_source_changed",
            "Scorecard 分数带在 selection 计划创建前发生变化；请重试。",
        )
    slots: dict[str, object] = {
        "source_artifact_id": binding.artifact_id,
        "expected_source_artifact_content_hash": binding.content_hash,
        "expected_asset_id": str(asset_id),
        "expected_asset_hash": asset_hash,
        "cutoff_id": str(cutoff_id),
    }
    if "reason" in inputs:
        slots["reason"] = inputs["reason"]
    return slots


def _candidate_source_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    candidate_id: str,
    workflow_inputs: Mapping,
) -> dict[str, str]:
    artifact_repository = TaskArtifactRepository(runtime.settings.db_path)
    matches = (
        artifact_repository.find_for_task_kind_origin_by_provenance_candidate_id(
            task_id,
            "strategy_candidate_json",
            "strategy.analyze_univariate_candidates",
            candidate_id,
        )
    )
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
    loader_runtime = SimpleNamespace(
        settings=runtime.settings,
        task_artifacts=artifact_repository,
    )
    try:
        verified = load_verified_candidate_refinement_source(
            loader_runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_content_hash=content_hash,
            expected_candidate_id=candidate_id,
            expected_evidence_hash=evidence_hash,
            feature=workflow_inputs.get("feature"),
            method=workflow_inputs.get("method"),
            merge_groups=workflow_inputs.get("merge_groups", []),
            selection=workflow_inputs.get("selection"),
        )
    except (OSError, StrategyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            f"候选证据 {candidate_id} 无法通过 canonical artifact 与 refinement "
            "控制校验，请重新生成分析或核对 feature、method 和 source bin id。"
        ) from exc
    return {
        "source_artifact_id": verified.artifact_id,
        "expected_artifact_content_hash": verified.content_hash,
        "expected_candidate_id": verified.candidate_id,
        "expected_evidence_hash": verified.evidence_hash,
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


def _strategy_voting_candidate_search_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind user controls to the exact current Pool through the Tool resolver."""

    if draft.workflow != "voting_candidate_search":
        raise StrategySetupError("Voting 组合搜索 slots 收到了错误的 Workflow。")
    try:
        read_runtime = _strategy_report_read_runtime(runtime)
        return resolve_voting_candidate_search_inputs(
            read_runtime,
            task_id=task.id,
            user_controls=draft.to_dict()["workflow_inputs"],
        )
    except StrategyError as exc:
        raise StrategySetupError(str(exc)) from exc


def _strategy_voting_candidate_build_from_search_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Preflight exact pointers without copying recovered state into the plan."""

    if draft.workflow != "voting_candidate_build_from_search":
        raise StrategySetupError(
            "Voting 搜索结果构建 slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    try:
        read_runtime = _strategy_report_read_runtime(runtime)
        resolve_voting_candidate_search_selection(
            read_runtime,
            task_id=task.id,
            search_id=inputs["search_id"],
            combo_id=inputs["combo_id"],
            strategy_type=inputs.get("strategy_type"),
        )
    except StrategyError as exc:
        raise StrategySetupError(str(exc)) from exc
    return dict(inputs)


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


def _strategy_pool_apply_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind one specified current nonempty Pool without exposing its identity."""

    if draft.workflow != "strategy_pool_apply":
        raise StrategySetupError(
            "Strategy Pool 应用 slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = str(inputs["strategy_type"])
    try:
        pool = StrategyCandidatePoolRepository(
            runtime.settings.db_path
        ).get_current(task.id, strategy_type)
    except Exception as exc:
        raise StrategySetupError(
            "当前 Strategy Pool 状态无法通过完整性校验，不能执行应用写回。"
        ) from exc
    if pool is None:
        raise StrategySetupError(
            f"当前任务没有 {strategy_type} Strategy Pool，无法应用到当前样本。"
        )
    if not _strategy_pool_entries(pool):
        raise StrategySetupError(
            f"当前 {strategy_type} Strategy Pool 为空；请先加入候选规则再执行应用。"
        )
    try:
        revision = pool["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("invalid Pool revision")
        snapshot_hash = strategy_pool_snapshot_hash(pool)
        if (
            not isinstance(snapshot_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", snapshot_hash) is None
        ):
            raise ValueError("invalid Pool snapshot hash")
    except (KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前 Strategy Pool revision/hash 绑定不完整，不能执行应用写回。"
        ) from exc

    slots: dict[str, object] = {
        "strategy_type": strategy_type,
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
    }
    if "output_prefix" in inputs:
        slots["output_prefix"] = inputs["output_prefix"]
    return slots


def _strategy_pool_validation_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Bind one current Pool to one exact mature independent V2 partition."""

    if draft.workflow != "strategy_pool_validation":
        raise StrategySetupError(
            "Strategy Pool 独立样本回放验证 slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = str(inputs.get("strategy_type") or "")
    partition = str(inputs.get("partition") or "")
    if strategy_type not in {"approval", "reject"}:
        raise StrategySetupError(
            "独立样本回放验证只支持 approval 或 reject Strategy Pool。"
        )
    if partition not in {"validation", "oot"}:
        raise StrategySetupError(
            "独立样本回放验证 partition 只能是 validation 或 oot。"
        )

    read_runtime = _strategy_v2_read_runtime(runtime)
    try:
        pool = _strategy_report_current_pool_binding(
            read_runtime,
            task_id=task.id,
            requested_type=strategy_type,
        )
    except _StrategyV2EvidenceSetupError as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_validation_pool_invalid",
            f"当前 {strategy_type} Strategy Pool 未通过非空 head、artifact、"
            "revision/hash 或完整 candidate lineage 认证。",
        ) from exc
    try:
        sample = _strategy_report_latest_sample_binding(
            read_runtime,
            task_id=task.id,
        )
    except _StrategyV2EvidenceSetupError as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_validation_sample_invalid",
            "独立样本回放验证需要最新且完整认证的 StrategySampleDesign V2 "
            "membership/bundle；平台不会回退到旧样本。",
        ) from exc

    try:
        design = sample.bundle["sample_design"]
        target = design["target_selector"]
        scope = design["sample_semantics"]["scope"]
        risk_populations = [
            item
            for item in sample.bundle["populations"]
            if item.get("role") == "risk"
        ]
        maturity = risk_populations[0]["maturity_evidence"]["status"]
        partition_count = sample.membership["header"]["counts"]["risk"][
            partition
        ]
        source = sample.source_binding
        dataset_id = source.dataset_id
        dataset_hash = source.dataset_content_hash
        workspace_revision = source.workspace_revision
        workspace_generation = source.workspace_generation
        semantic_hash = source.semantic_mapping_hash
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_validation_sample_invalid",
            "StrategySampleDesign V2 缺少 risk 总体、成熟度、独立分区、"
            "dataset/workspace/target 或语义绑定。",
        ) from exc
    if len(risk_populations) != 1 or maturity != "confirmed_matured":
        raise StrategySetupError(
            "独立样本回放验证要求 risk 总体具有已确认成熟的表现结果。"
        )
    if (
        target.get("status") != "resolved"
        or not isinstance(target.get("column"), str)
        or not target["column"]
        or scope != "strategy_development"
    ):
        raise StrategySetupError(
            "独立样本回放验证需要已解析的目标列和受治理的策略样本语义。"
        )
    if (
        isinstance(partition_count, bool)
        or not isinstance(partition_count, int)
        or partition_count <= 0
    ):
        raise StrategySetupError(
            f"StrategySampleDesign V2 的 risk/{partition} 独立分区为空，"
            "不能执行回放验证。"
        )
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(dataset_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset_hash) is None
        or isinstance(workspace_revision, bool)
        or not isinstance(workspace_revision, int)
        or workspace_revision < 0
        or isinstance(workspace_generation, bool)
        or not isinstance(workspace_generation, int)
        or workspace_generation < 0
        or not isinstance(semantic_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", semantic_hash) is None
    ):
        raise StrategySetupError(
            "独立样本回放验证的 dataset/workspace/semantic identity 不完整。"
        )

    try:
        resolve_pool_requirements(
            read_runtime,
            task_id=task.id,
            compiled_design=pool.compiled_design,
            sample_design=sample,
        )
    except StrategyError as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_validation_requirement_invalid",
            "当前 Strategy Pool 的模型评分要求无法绑定到精确 "
            "StrategySampleDesign V2；请重新生成评分证据或候选。",
        ) from exc

    snapshot = pool.pool
    try:
        pool_ref = {
            "artifact_id": pool.artifact_id,
            "expected_artifact_content_hash": pool.artifact_content_hash,
            "expected_pool_id": snapshot["pool_id"],
            "expected_revision": snapshot["revision"],
            "expected_revision_id": snapshot["revision_id"],
            "expected_snapshot_hash": snapshot["snapshot_hash"],
        }
        sample_ref = _strategy_report_sample_ref(sample)
    except (AttributeError, KeyError, TypeError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_validation_binding_invalid",
            "Strategy Pool 或 StrategySampleDesign V2 精确引用不完整。",
        ) from exc
    return {
        "strategy_type": strategy_type,
        "partition": partition,
        "pool_ref": pool_ref,
        "sample_design_ref": sample_ref,
        "population": "risk",
        "comparison_mode": "absolute",
    }


def _strategy_pool_stability_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
) -> dict[str, object]:
    """Freeze one current Pool and all usable comparison partitions once."""

    if draft.workflow != "strategy_pool_stability":
        raise StrategySetupError(
            "Strategy Pool stability slots 收到了错误的 Workflow。"
        )
    strategy_type = draft.workflow_inputs.get("strategy_type")
    impact_draft = StandardWorkflowRequestDraft(
        workflow="strategy_impact_cube",
        workflow_inputs={"strategy_type": strategy_type},
    )
    impact_slots = _strategy_impact_cube_plan_slots(
        runtime,
        task,
        impact_draft,
        fixed_dimension_bindings={
            "month_col": None,
            "group_col": None,
            "segment_col": None,
        },
        include_optional_context=False,
    )
    partitions = impact_slots["partitions"]
    if (
        not isinstance(partitions, list)
        or "development" not in partitions
        or not any(
            partition in partitions for partition in ("validation", "oot")
        )
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_pool_stability_comparison_required",
            "Pool 跨分区稳定性需要非空 development 基线，并至少具备一个"
            "非空 validation 或 OOT 比较分区；请先完善样本设计。",
        )
    return {
        "strategy_type": impact_slots["strategy_type"],
        "pool_ref": impact_slots["pool_ref"],
        "sample_design_ref": impact_slots["sample_design_ref"],
        "partitions": partitions,
    }


def _strategy_impact_cube_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    expected_sample_binding: Mapping | None = None,
    fixed_dimension_bindings: Mapping[str, str | None] | None = None,
    include_optional_context: bool = True,
) -> dict[str, object]:
    """Bind one ImpactCube plan to a single authenticated evidence snapshot."""

    if draft.workflow != "strategy_impact_cube":
        raise StrategySetupError(
            "Strategy ImpactCube slots 收到了错误的 Workflow。"
        )
    inputs = draft.to_dict()["workflow_inputs"]
    strategy_type = inputs.get("strategy_type")
    if strategy_type not in {
        "approval",
        "reject",
        "limit",
        "pricing",
        "segmentation",
    }:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_type_invalid",
            "统一影响测算需要明确 approval、reject、limit、pricing 或 "
            "segmentation Strategy Pool。",
        )

    read_runtime = _strategy_v2_read_runtime(runtime)
    try:
        artifacts = tuple(read_runtime.task_artifacts.list_for_task(task.id))
    except _STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_registry_unavailable",
            "无法读取当前任务的 ImpactCube source artifact registry。",
        ) from exc
    try:
        sample = _latest_verified_strategy_sample_design_v2_binding(
            read_runtime,
            task_id=task.id,
            artifacts=artifacts,
        )
    except _StrategyV2EvidenceSetupError as exc:
        code = (
            "strategy_impact_cube_sample_required"
            if exc.code.endswith("_sample_required")
            else "strategy_impact_cube_sample_invalid"
        )
        raise _StrategyV2EvidenceSetupError(
            code,
            "ImpactCube 需要最新且完整认证的 StrategySampleDesign V2 "
            "双总体样本证据；平台不会回退到旧样本。",
        ) from exc

    actual_sample_binding = {
        "kind": "strategy_sample_design_v2",
        "sample_design_ref": _strategy_report_sample_ref(sample),
        "dataset_id": sample.source_binding.dataset_id,
        "dataset_content_hash": sample.source_binding.dataset_content_hash,
    }
    if (
        expected_sample_binding is not None
        and dict(expected_sample_binding) != actual_sample_binding
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_sample_changed",
            "StrategySampleDesign V2 在请求编译与计划创建之间发生变化；"
            "本次未创建计划，请基于最新样本重新描述。",
        )

    pool_repository = StrategyCandidatePoolRepository(runtime.settings.db_path)
    try:
        current_pool = pool_repository.get_current(task.id, strategy_type)
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_pool_invalid",
            f"当前 {strategy_type} Strategy Pool head/revision 无法通过完整性复核。",
        ) from exc
    if current_pool is None or not _strategy_pool_entries(current_pool):
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_pool_required",
            f"当前任务没有非空 {strategy_type} Strategy Pool；"
            "请先用自然语言把候选加入该 Pool。",
        )
    try:
        pool_revision = int(current_pool["revision"])
        pool_snapshot_hash = strategy_pool_snapshot_hash(current_pool)
        pool = load_current_strategy_candidate_pool_artifact(
            read_runtime,
            task_id=task.id,
            strategy_type=strategy_type,
            expected_pool_revision=pool_revision,
            expected_pool_snapshot_hash=pool_snapshot_hash,
        )
    except (StrategyError, *_STRATEGY_V2_ARTIFACT_ERRORS) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_pool_invalid",
            f"当前 {strategy_type} Strategy Pool 的 artifact、来源、编译结果"
            "或 lineage 未通过完整认证。",
        ) from exc
    if pool.compiled_design.get("requirements"):
        try:
            resolve_pool_requirements(
                read_runtime,
                task_id=task.id,
                compiled_design=pool.compiled_design,
                sample_design=sample,
            )
        except StrategyError as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_pool_requirement_invalid",
                "当前 Strategy Pool 的模型评分要求无法绑定到最新 "
                "StrategySampleDesign V2；请重新生成评分证据或候选。",
            ) from exc
    if pool.artifact_id not in {
        item.get("id") for item in artifacts
    }:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_registry_changed",
            "Strategy Pool artifact 在证据选择期间发生变化；请重试本次测算。",
        )

    partitions = _strategy_impact_cube_partitions(inputs, sample=sample)
    dimensions = (
        _strategy_impact_cube_dimensions(inputs, sample=sample)
        if fixed_dimension_bindings is None
        else dict(fixed_dimension_bindings)
    )
    economics_inputs = (
        _strategy_impact_cube_economics(
            inputs.get("economics_inputs"),
            sample=sample,
        )
        if include_optional_context
        else None
    )
    current_strategy_ref = (
        _strategy_impact_cube_current_strategy_ref(
            runtime,
            task_id=task.id,
            strategy_type=strategy_type,
            requested_id=inputs.get("current_strategy_id"),
        )
        if include_optional_context
        else None
    )

    selected_artifact_ids = {
        pool.artifact_id,
        sample.membership_artifact_id,
        sample.bundle_artifact_id,
    }
    registry_token = _strategy_impact_cube_registry_token(
        artifacts,
        selected_artifact_ids=selected_artifact_ids,
    )
    try:
        refreshed_artifacts = tuple(
            read_runtime.task_artifacts.list_for_task(task.id)
        )
        refreshed_pool = pool_repository.get_current(task.id, strategy_type)
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_registry_changed",
            "ImpactCube source registry 或 Strategy Pool head 在计划创建前"
            "无法再次核对；本次未创建计划。",
        ) from exc
    if (
        _strategy_impact_cube_registry_token(
            refreshed_artifacts,
            selected_artifact_ids=selected_artifact_ids,
        )
        != registry_token
        or refreshed_pool is None
        or refreshed_pool.get("revision") != pool_revision
        or strategy_pool_snapshot_hash(refreshed_pool) != pool_snapshot_hash
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_registry_changed",
            "StrategySampleDesign V2、Strategy Pool 或 artifact registry "
            "在计划创建前发生变化；请基于最新证据重试。",
        )
    if current_strategy_ref is not None:
        refreshed_current = _strategy_impact_cube_current_strategy_ref(
            runtime,
            task_id=task.id,
            strategy_type=strategy_type,
            requested_id=current_strategy_ref["strategy_id"],
        )
        if refreshed_current != current_strategy_ref:
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_current_strategy_changed",
                "当前策略的 canonical StrategySpec 在计划创建前发生变化；"
                "本次未创建计划。",
            )

    return {
        "strategy_type": strategy_type,
        "pool_ref": {
            "artifact_id": pool.artifact_id,
            "expected_artifact_content_hash": pool.artifact_content_hash,
            "expected_pool_id": pool.pool["pool_id"],
            "expected_revision": pool.pool["revision"],
            "expected_revision_id": pool.pool["revision_id"],
            "expected_snapshot_hash": pool.pool["snapshot_hash"],
        },
        "sample_design_ref": _strategy_report_sample_ref(sample),
        "partitions": partitions,
        # ImpactCube owns both approval and risk denominators internally; this
        # retained v1 selector states which observed-outcome population supplies
        # risk metrics and cannot be supplied by the language model.
        "population": "risk",
        "dimension_bindings": dimensions,
        "current_strategy_ref": current_strategy_ref,
        "economics_inputs": economics_inputs,
    }


def _strategy_dsl_delivery_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
) -> dict[str, object]:
    """Bind one offline delivery to exact strategy and dataset snapshots."""

    if draft.workflow != "strategy_dsl_delivery":
        raise StrategySetupError(
            "Strategy DSL delivery slots 收到了错误的 Workflow。"
        )
    requested_id = draft.to_dict()["workflow_inputs"].get("strategy_id")
    repository = StrategyRepository(runtime.settings.db_path)

    if requested_id is None:
        try:
            candidate_ids = [
                str(meta["id"])
                for meta in repository.list_meta_for_task(task.id)
                if isinstance(meta.get("id"), str)
            ]
        except Exception as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_dsl_delivery_registry_unavailable",
                "无法读取当前任务的策略注册表，未创建交付计划。",
            ) from exc
        eligible: list[tuple[str, dict, dict[str, object]]] = []
        for strategy_id in candidate_ids:
            try:
                snapshot = repository.get_strategy_snapshot(strategy_id)
                strategy_ref = _strategy_dsl_delivery_strategy_ref(
                    snapshot,
                    task_id=task.id,
                )
            except (
                StrategyError,
                sqlite3.Error,
                TypeError,
                ValueError,
                _StrategyV2EvidenceSetupError,
            ):
                continue
            eligible.append((strategy_id, snapshot, strategy_ref))
        if not eligible:
            raise _StrategyV2EvidenceSetupError(
                "strategy_dsl_delivery_strategy_required",
                "当前任务没有可交付的 canonical Strategy；请先完成策略构建。",
            )
        if len(eligible) != 1:
            raise _StrategyV2EvidenceSetupError(
                "strategy_dsl_delivery_strategy_ambiguous",
                "当前任务有多个可交付策略，请在导出命令中明确完整 strategy_id。",
            )
        strategy_id, snapshot, strategy_ref = eligible[0]
    else:
        strategy_id = str(requested_id)
        try:
            snapshot = repository.get_strategy_snapshot(strategy_id)
        except Exception as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_dsl_delivery_strategy_invalid",
                "指定策略无法通过当前任务的完整性复核。",
            ) from exc
        strategy_ref = _strategy_dsl_delivery_strategy_ref(
            snapshot,
            task_id=task.id,
        )

    dataset_id = getattr(context, "dataset_id", None)
    dataset_hash = getattr(context, "dataset_content_hash", None)
    workspace_revision = getattr(context, "workspace_revision", None)
    workspace_generation = getattr(context, "analysis_generation", None)
    semantic_mapping_hash = getattr(context, "semantic_mapping_hash", None)
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(dataset_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", dataset_hash) is None
        or isinstance(workspace_revision, bool)
        or not isinstance(workspace_revision, int)
        or workspace_revision < 0
        or isinstance(workspace_generation, bool)
        or not isinstance(workspace_generation, int)
        or workspace_generation < 0
        or not isinstance(semantic_mapping_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", semantic_mapping_hash) is None
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_dataset_invalid",
            "当前策略样本缺少可认证 dataset/workspace identity。",
        )
    try:
        workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task.id)
    except (
        DataWorkspaceDataError,
        KeyError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_dataset_invalid",
            "当前 DataWorkspace 无法通过完整性复核。",
        ) from exc
    if (
        workspace.revision != workspace_revision
        or workspace.analysis_generation != workspace_generation
        or data_semantic_mapping_hash(workspace.semantic_mapping)
        != semantic_mapping_hash
        or (
            workspace.active_dataset_id is not None
            and (
                workspace.active_dataset_id != dataset_id
                or workspace.active_dataset_content_hash != dataset_hash
            )
        )
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_dataset_invalid",
            "当前活动 DataWorkspace 与已确认的数据上下文不一致。",
        )
    workspace_ref = {
        "revision": workspace_revision,
        "analysis_generation": workspace_generation,
        "semantic_mapping_hash": semantic_mapping_hash,
        "active_dataset_id": workspace.active_dataset_id,
        "active_dataset_content_hash": (
            workspace.active_dataset_content_hash
        ),
    }
    _backend, registry = _modeling_data_runtime(runtime.settings)
    try:
        dataset = registry.get(dataset_id)
        registry.resolve_verified_path(dataset_id)
    except (
        DatasetContentDriftError,
        KeyError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_dataset_invalid",
            "当前策略样本未通过 task ownership、registry 或文件 hash 复核。",
        ) from exc
    if (
        str(dataset.task_id) != task.id
        or dataset.content_hash != dataset_hash
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_dataset_invalid",
            "当前策略样本不属于本任务或 content hash 已变化。",
        )
    dataset_ref = {
        "dataset_id": dataset_id,
        "expected_content_hash": dataset_hash,
    }

    # Close the selection window before plan creation. The Tool repeats the
    # same exact-ref checks under its publication transaction.
    try:
        refreshed_snapshot = repository.get_strategy_snapshot(strategy_id)
        refreshed_ref = _strategy_dsl_delivery_strategy_ref(
            refreshed_snapshot,
            task_id=task.id,
        )
        if requested_id is None:
            refreshed_eligible = []
            for meta in repository.list_meta_for_task(task.id):
                candidate_id = meta.get("id")
                if not isinstance(candidate_id, str):
                    continue
                try:
                    candidate_snapshot = repository.get_strategy_snapshot(
                        candidate_id
                    )
                    candidate_ref = _strategy_dsl_delivery_strategy_ref(
                        candidate_snapshot,
                        task_id=task.id,
                    )
                except (
                    StrategyError,
                    sqlite3.Error,
                    TypeError,
                    ValueError,
                    _StrategyV2EvidenceSetupError,
                ):
                    continue
                refreshed_eligible.append((candidate_id, candidate_ref))
        refreshed_dataset = registry.get(dataset_id)
        registry.resolve_verified_path(dataset_id)
        refreshed_workspace = DataWorkspaceRepository(
            runtime.settings.db_path
        ).get_or_default(task.id)
    except (
        DataWorkspaceDataError,
        DatasetContentDriftError,
        KeyError,
        OSError,
        sqlite3.Error,
        TypeError,
        ValueError,
        _StrategyV2EvidenceSetupError,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_binding_changed",
            "策略或活动数据集在计划创建前发生变化；本次未创建交付计划。",
        ) from exc
    if (
        refreshed_ref != strategy_ref
        or (
            requested_id is None
            and refreshed_eligible != [(strategy_id, strategy_ref)]
        )
        or refreshed_workspace.revision != workspace_revision
        or refreshed_workspace.analysis_generation != workspace_generation
        or data_semantic_mapping_hash(refreshed_workspace.semantic_mapping)
        != semantic_mapping_hash
        or (
            refreshed_workspace.active_dataset_id is not None
            and (
                refreshed_workspace.active_dataset_id != dataset_id
                or refreshed_workspace.active_dataset_content_hash
                != dataset_hash
            )
        )
        or str(refreshed_dataset.task_id) != task.id
        or refreshed_dataset.content_hash != dataset_hash
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_binding_changed",
            "策略或活动数据集在计划创建前发生变化；本次未创建交付计划。",
        )

    return {
        "strategy_ref": strategy_ref,
        "dataset_ref": dataset_ref,
        "workspace_ref": workspace_ref,
        "maximum_equivalence_rows": MAX_EQUIVALENCE_ROWS,
    }


def _strategy_dsl_delivery_strategy_ref(
    snapshot: object,
    *,
    task_id: str,
) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "策略缺少可认证的原子 snapshot，或不属于当前任务。",
        )
    try:
        strategy = snapshot["strategy"]
        metadata = snapshot["metadata"]
        spec_hash = snapshot["strategy_spec_hash"]
        strategy_id = str(metadata["id"])
        strategy_type = str(metadata["strategy_type"])
        version = metadata["version"]
        canonical_spec_hash = (
            strategy_spec_hash(strategy.spec)
            if getattr(strategy, "spec", None) is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "策略 snapshot 的 identity、type、version 或 spec hash 不完整。",
        ) from exc
    if (
        metadata.get("task_id") != task_id
        or getattr(strategy, "id", None) != strategy_id
        or getattr(strategy, "strategy_type", None) != strategy_type
        or strategy_type
        not in {"approval", "reject", "limit", "pricing", "segmentation"}
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
        or not isinstance(spec_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None
        or canonical_spec_hash != spec_hash
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_dsl_delivery_strategy_invalid",
            "strategy_id 必须属于当前任务，并带有一致的五类 type、正版本和"
            " canonical Strategy DSL/spec hash；历史兼容行需先迁移。",
        )
    return {
        "strategy_id": strategy_id,
        "expected_strategy_type": strategy_type,
        "expected_version": version,
        "expected_spec_hash": spec_hash,
    }


def _strategy_impact_cube_partitions(
    inputs: Mapping,
    *,
    sample,
) -> list[str]:
    order = ("development", "validation", "oot")
    try:
        counts = sample.membership["header"]["counts"]
        approval_counts = counts["approval"]
        risk_counts = counts["risk"]
    except (KeyError, TypeError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_sample_invalid",
            "StrategySampleDesign V2 缺少 approval/risk 分区计数。",
        ) from exc
    for population_counts in (approval_counts, risk_counts):
        if not isinstance(population_counts, Mapping) or any(
            isinstance(population_counts.get(partition), bool)
            or not isinstance(population_counts.get(partition), int)
            or population_counts[partition] < 0
            for partition in order
        ):
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_sample_invalid",
                "StrategySampleDesign V2 的分区计数无效。",
            )

    requested = inputs.get("partitions")
    if requested is None:
        selected = [
            partition
            for partition in order
            if approval_counts[partition] > 0 and risk_counts[partition] > 0
        ]
    else:
        if (
            not isinstance(requested, Sequence)
            or isinstance(requested, str | bytes | bytearray)
        ):
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_partitions_invalid",
                "ImpactCube partitions 必须是明确的分区列表。",
            )
        requested_set = set(requested)
        if (
            not requested_set
            or len(requested_set) != len(requested)
            or not requested_set.issubset(order)
        ):
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_partitions_invalid",
                "ImpactCube 只接受不重复的 development、validation、oot 分区。",
            )
        selected = [
            partition for partition in order if partition in requested_set
        ]
    empty = [
        partition
        for partition in selected
        if approval_counts[partition] == 0 or risk_counts[partition] == 0
    ]
    if not selected or empty:
        detail = "、".join(empty) if empty else "全部"
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_partition_empty",
            f"所选分区 {detail} 没有同时具备 approval 与 risk 总体；"
            "请调整样本设计或选择非空分区。",
        )
    return selected


def _strategy_impact_cube_dimensions(
    inputs: Mapping,
    *,
    sample,
) -> dict[str, str | None]:
    columns = tuple(sample.source_binding.columns)
    roles = dict(sample.source_binding.semantic_field_roles)
    provenance_request = sample.provenance.get("request")
    field_bindings = (
        provenance_request.get("field_bindings")
        if isinstance(provenance_request, Mapping)
        else None
    )
    if not isinstance(field_bindings, Mapping):
        field_bindings = {}

    def unique_role(role: str) -> str | None:
        matches = sorted(
            column
            for column, assigned in roles.items()
            if assigned == role and column in columns
        )
        if len(matches) > 1:
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_ambiguous",
                f"当前样本有多个 `{role}` 语义字段：{'、'.join(matches)}；"
                "请在请求中明确列名。",
            )
        return matches[0] if matches else None

    defaults = {
        "month_col": field_bindings.get("month_field") or unique_role("month"),
        "group_col": field_bindings.get("group_field"),
        "segment_col": unique_role("segment"),
    }
    result: dict[str, str | None] = {}
    used: set[str] = set()
    for field in ("month_col", "group_col", "segment_col"):
        explicit = inputs.get(field)
        selected = explicit if explicit is not None else defaults[field]
        if selected is not None and (
            not isinstance(selected, str) or selected not in columns
        ):
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_invalid",
                f"ImpactCube 维度 {field} 不在最新样本绑定的数据列中。",
            )
        if selected is not None and roles.get(selected) in {"id", "target"}:
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_dimension_sensitive",
                f"字段 `{selected}` 的语义角色是 {roles[selected]}，"
                "不能作为 ImpactCube 聚合维度。",
            )
        if selected is not None and selected in used:
            if explicit is not None:
                raise _StrategyV2EvidenceSetupError(
                    "strategy_impact_cube_dimension_duplicate",
                    "月份、分组和分群维度必须使用不同字段。",
                )
            selected = None
        result[field] = selected
        if selected is not None:
            used.add(selected)
    return result


def _strategy_impact_cube_economics(
    value: object,
    *,
    sample,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_economics_invalid",
            "ImpactCube economics_inputs 必须是 typed column/scalar 映射。",
        )
    columns = set(sample.source_binding.columns)
    roles = dict(sample.source_binding.semantic_field_roles)
    result: dict[str, object] = {}
    for component, raw_binding in sorted(value.items()):
        if not isinstance(component, str) or not isinstance(raw_binding, Mapping):
            raise _StrategyV2EvidenceSetupError(
                "strategy_impact_cube_economics_invalid",
                "ImpactCube economics_inputs 组件或绑定结构无效。",
            )
        binding = dict(raw_binding)
        if binding.get("kind") == "column":
            column = binding.get("column")
            if (
                not isinstance(column, str)
                or column not in columns
                or roles.get(column) in {"id", "target"}
            ):
                raise _StrategyV2EvidenceSetupError(
                    "strategy_impact_cube_economics_column_invalid",
                    f"经济参数 {component} 未绑定到当前样本中的非敏感业务列。",
                )
        result[component] = binding
    return result


def _strategy_impact_cube_current_strategy_ref(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    strategy_type: str,
    requested_id: object,
) -> dict[str, str] | None:
    if requested_id is None:
        return None
    if not isinstance(requested_id, str) or not requested_id:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "当前策略比较需要完整 strategy_id。",
        )
    repository = StrategyRepository(runtime.settings.db_path)
    try:
        snapshot = repository.get_strategy_snapshot(requested_id)
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "当前策略的 canonical StrategySpec 无法通过完整性校验。",
        ) from exc
    if snapshot is None:
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "current_strategy_id 必须属于当前任务、类型一致并带有完整 canonical "
            "StrategySpec；平台不会跨任务或跨类型比较。",
        )
    meta = snapshot["metadata"]
    strategy = snapshot["strategy"]
    spec_hash = snapshot["strategy_spec_hash"]
    if (
        strategy.spec is None
        or meta.get("task_id") != task_id
        or meta.get("strategy_type") != strategy_type
        or strategy.strategy_type != strategy_type
        or not isinstance(spec_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_impact_cube_current_strategy_invalid",
            "current_strategy_id 必须属于当前任务、类型一致并带有完整 canonical "
            "StrategySpec；平台不会跨任务或跨类型比较。",
        )
    return {
        "strategy_id": requested_id,
        "expected_strategy_spec_hash": spec_hash,
    }


def _strategy_impact_cube_registry_token(
    artifacts: Sequence[Mapping],
    *,
    selected_artifact_ids: set[str],
) -> str:
    relevant = [
        {
            "id": item.get("id"),
            "kind": item.get("kind"),
            "content_hash": item.get("content_hash"),
            "origin_tool": item.get("origin_tool"),
            "provenance": item.get("provenance"),
        }
        for item in artifacts
        if item.get("id") in selected_artifact_ids
        or item.get("kind")
        in {
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        }
    ]
    return hashlib.sha256(
        json.dumps(
            relevant,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _strategy_project_context_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    source_message: Mapping | None,
) -> dict:
    if draft.workflow != "strategy_project_context":
        raise StrategySetupError("项目上下文 slots 收到了错误的 Workflow。")
    try:
        current = StrategyProjectContextRepository(
            runtime.settings.db_path
        ).get_current(task.id)
    except (StrategyProjectContextDataError, KeyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "当前策略项目上下文 revision 无法通过完整性校验，请先修复持久化证据。"
        ) from exc

    if (
        not isinstance(source_message, Mapping)
        or source_message.get("task_id") != task.id
        or source_message.get("role") != "user"
        or (source_message.get("metadata") or {}).get("intent")
        not in {"strategy_request", "strategy_project_context_answer"}
    ):
        raise StrategySetupError(
            "项目上下文刷新必须绑定本轮已持久化的用户消息；请重新发送整理请求。"
        )
    message_id = source_message.get("id")
    content = source_message.get("content")
    if not isinstance(message_id, str) or not message_id or not isinstance(content, str):
        raise StrategySetupError("项目上下文无法绑定有效的用户消息证据。")

    inputs = draft.to_dict()["workflow_inputs"]
    new_business = dict(inputs.get("business_context") or {})
    explicit_unavailable = set(inputs.get("explicit_unavailable") or [])
    explicit_unavailable.update(
        field_path for field_path, value in new_business.items() if value is None
    )
    explicit_unavailable.difference_update(
        field_path for field_path, value in new_business.items() if value is not None
    )
    return {
        "expected_revision": 0 if current is None else current["revision"],
        "expected_revision_id": (
            None if current is None else current["revision_id"]
        ),
        "expected_state_hash": None if current is None else current["state_hash"],
        "user_message_ref": {
            "message_id": message_id,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        },
        "as_of": inputs["as_of"],
        "scope": inputs.get("scope"),
        "business_context": new_business,
        "explicit_unavailable": sorted(explicit_unavailable),
        "external_report_filenames": list(
            inputs.get("external_report_filenames") or []
        ),
    }


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


def _strategy_sample_design_v2_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    context,
    drop_nan_labels: bool,
) -> dict[str, object]:
    """Project a strict V2 request into a lossless V1 anchor plus V2 controls."""

    workspace = _require_strategy_sample_design_workspace(runtime, task)
    if (
        workspace.active_dataset_id != context.dataset_id
        or workspace.active_dataset_content_hash != context.dataset_content_hash
        or workspace.revision != context.workspace_revision
        or workspace.analysis_generation != context.analysis_generation
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 在 V2 样本设计计划创建前发生变化；"
            "请基于当前版本重试。"
        )
    target_col = workspace.semantic_mapping.target_col
    semantic_hash = data_semantic_mapping_hash(workspace.semantic_mapping)
    if (
        not isinstance(target_col, str)
        or not target_col
        or target_col != context.target_col
        or target_col not in context.columns
    ):
        raise StrategySetupError(
            "V2 策略样本设计只能使用 DataWorkspace 中已确认的二元目标列。"
        )
    if (
        not isinstance(workspace.active_dataset_content_hash, str)
        or not workspace.active_dataset_content_hash
        or semantic_hash != context.semantic_mapping_hash
    ):
        raise StrategySetupError(
            "活动 DataWorkspace 的数据 hash 或语义映射已变化；"
            "请重新发起 V2 样本设计。"
        )

    inputs = draft.to_dict()["workflow_inputs"]
    if inputs["approval_population"] != {
        "inclusion": None,
        "exclusion": None,
    } or inputs["risk_population"] != {
        "inclusion": None,
        "exclusion": None,
    }:
        raise _StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "当前 V1 compatibility anchor 只支持 approval/risk 同 cohort 且"
            "两个总体均无纳排；本次未创建计划。",
        )
    split_col, split_values = _strategy_sample_v2_simple_split_projection(
        inputs["partitioning"]
    )
    fields = inputs["field_bindings"]
    compatibility_columns = [
        fields.get("month_field"),
        fields.get("weight_field"),
        fields.get("loan_amount_field"),
        fields.get("overdue_amount_field"),
    ]
    present_columns = [
        str(column) for column in compatibility_columns if column is not None
    ]
    if (
        split_col == target_col
        or split_col not in context.columns
        or any(column not in context.columns for column in present_columns)
        or target_col in present_columns
        or split_col in present_columns
        or len(present_columns) != len(set(present_columns))
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "当前 V2 字段绑定不能无损投影为 V1 compatibility anchor；"
            "请调整重复/冲突字段，或先完成原生 V2 bootstrap。",
        )

    maturity = inputs["maturity"]
    performance = inputs["performance_window"]
    observation = inputs["observation_window"]
    scope = (
        "strategy_development"
        if maturity["status"] == "confirmed_matured"
        and performance["status"] == "provided"
        and observation["status"] == "provided"
        else "exploration_only"
    )
    compatibility_maturity = (
        "unknown" if maturity["status"] == "unavailable" else maturity["status"]
    )
    policy = {
        **_STRATEGY_SAMPLE_V2_POLICY,
        "diagnostic_severities": dict(
            _STRATEGY_SAMPLE_V2_POLICY["diagnostic_severities"]
        ),
    }
    return {
        "dataset_id": workspace.active_dataset_id,
        "expected_dataset_content_hash": workspace.active_dataset_content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": semantic_hash,
        "target_col": target_col,
        "relationship": "nested_same_cohort",
        "scope": scope,
        "policy": policy,
        "compatibility_performance_window_status": performance["status"],
        "compatibility_performance_window_days": performance["days"],
        "compatibility_observation_window_status": observation["status"],
        "compatibility_observation_start": observation["start"],
        "compatibility_observation_end": observation["end"],
        "compatibility_maturity_status": compatibility_maturity,
        "compatibility_split_col": split_col,
        "compatibility_development_values": [split_values["development"]],
        "compatibility_validation_values": [split_values["validation"]],
        "compatibility_oot_values": [split_values["oot"]],
        "compatibility_month_col": fields.get("month_field"),
        "compatibility_weight_col": fields.get("weight_field"),
        "compatibility_loan_amount_col": fields.get("loan_amount_field"),
        "compatibility_overdue_amount_col": fields.get("overdue_amount_field"),
        "target_bad_value": inputs["target_bad_value"],
        "drop_nan_labels": bool(drop_nan_labels),
        "approval_population": inputs["approval_population"],
        "risk_population": inputs["risk_population"],
        "partitioning": inputs["partitioning"],
        "maturity": maturity,
        "performance_window": performance,
        "observation_window": observation,
        "field_bindings": fields,
        "historical_score": inputs["historical_score"],
    }


def _strategy_sample_v2_simple_split_projection(
    partitioning: object,
) -> tuple[str, dict[str, object]]:
    if (
        not isinstance(partitioning, Mapping)
        or set(partitioning) != {"method", "selectors"}
        or partitioning.get("method") != "predicate_ast"
        or not isinstance(partitioning.get("selectors"), Mapping)
    ):
        raise _StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "当前 compatibility anchor 只支持同一列上的三组简单等值切分；"
            "本次未创建计划。",
        )
    selectors = partitioning["selectors"]
    if set(selectors) != {"development", "validation", "oot"}:
        raise _StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "V2 partitioning 必须完整包含 development、validation 和 OOT。",
        )
    columns: list[str] = []
    values: dict[str, object] = {}
    for partition in ("development", "validation", "oot"):
        predicate = selectors[partition]
        if (
            not isinstance(predicate, Mapping)
            or set(predicate) != {"op", "left", "right"}
            or predicate.get("op") != "eq"
            or not isinstance(predicate.get("left"), Mapping)
            or set(predicate["left"]) != {"column"}
            or not isinstance(predicate.get("right"), Mapping)
            or set(predicate["right"]) != {"literal"}
        ):
            raise _StrategyV2EvidenceSetupError(
                "strategy_sample_design_v2_native_bootstrap_required",
                "当前 compatibility anchor 只支持 column == literal 的简单切分；"
                "本次未创建计划。",
            )
        column = predicate["left"]["column"]
        literal = predicate["right"]["literal"]
        if not isinstance(column, str) or not column or literal is None:
            raise _StrategyV2EvidenceSetupError(
                "strategy_sample_design_v2_native_bootstrap_required",
                "V2 compatibility 切分列与三个切分值必须完整。",
            )
        columns.append(column)
        values[partition] = literal
    if len(set(columns)) != 1 or len(
        {json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values.values()}
    ) != 3:
        raise _StrategyV2EvidenceSetupError(
            "strategy_sample_design_v2_native_bootstrap_required",
            "V2 compatibility 切分必须使用同一列上的三个互异标量值。",
        )
    return columns[0], values


def _strategy_model_evidence_v2_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    *,
    verify_current: bool = False,
) -> dict[str, object]:
    """Discover and live-authenticate task-owned V2 sample/candidate evidence."""

    read_runtime = _strategy_v2_read_runtime(runtime)
    artifacts = _strategy_v2_artifact_snapshot(
        read_runtime,
        task_id=task.id,
    )
    registry_token = _strategy_v2_registry_token(artifacts)
    sample_binding = _latest_verified_strategy_sample_design_v2_binding(
        read_runtime,
        task_id=task.id,
        artifacts=artifacts,
    )
    design = sample_binding.bundle["sample_design"]
    identity = design["identity"]
    dataset_ref = identity["dataset_ref"]
    workspace_ref = identity["workspace_ref"]
    legacy_ref = design["compatibility"]["legacy_development_ref"]
    candidate_requests: list[dict[str, str]] = []
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind") != "strategy_candidate_json"
            or artifact.get("origin_tool")
            != "strategy.analyze_univariate_candidates"
            or not isinstance(provenance, Mapping)
        ):
            continue
        generation = provenance.get("generation_parameters")
        if (
            provenance.get("dataset_id") != dataset_ref["dataset_id"]
            or provenance.get("dataset_content_hash") != dataset_ref["content_hash"]
            or provenance.get("workspace_revision") != workspace_ref["revision"]
            or provenance.get("workspace_generation") != workspace_ref["generation"]
            or provenance.get("semantic_mapping_hash")
            != workspace_ref["semantic_mapping_hash"]
        ):
            continue
        if not isinstance(generation, Mapping):
            raise _StrategyV2EvidenceSetupError(
                "strategy_model_evidence_v2_candidate_invalid",
                "一份属于最新 StrategySampleDesign V2 快照的单变量候选"
                "缺少完整 generation provenance；本次未创建计划。",
            )
        try:
            source_legacy_ref = StrategySampleDesignRef.from_value(
                generation.get("sample_design_ref")
            ).to_ref_dict()
        except StrategyError as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_model_evidence_v2_candidate_invalid",
                "一份属于最新 StrategySampleDesign V2 快照的单变量候选"
                "包含损坏或不完整的 sample_design_ref；本次未创建计划。",
            ) from exc
        if source_legacy_ref != legacy_ref:
            continue
        request = {
            "artifact_id": artifact.get("id"),
            "expected_artifact_content_hash": artifact.get("content_hash"),
            "expected_candidate_id": provenance.get("candidate_id"),
            "expected_evidence_hash": provenance.get("evidence_hash"),
        }
        if not all(isinstance(value, str) and value for value in request.values()):
            raise _StrategyV2EvidenceSetupError(
                "strategy_model_evidence_v2_candidate_invalid",
                "一份属于最新 StrategySampleDesign V2 快照的单变量候选"
                "缺少 artifact、candidate 或 evidence 身份；本次未创建计划。",
            )
        candidate_requests.append(request)

    if not candidate_requests:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_candidate_required",
            "当前任务没有与最新 StrategySampleDesign V2 严格兼容且通过"
            " live loader 认证的单变量候选证据；请先运行单变量分析。",
        )
    candidate_requests.sort(
        key=lambda item: (item["expected_candidate_id"], item["artifact_id"])
    )
    candidate_ids = [item["expected_candidate_id"] for item in candidate_requests]
    artifact_ids = [item["artifact_id"] for item in candidate_requests]
    if len(set(candidate_ids)) != len(candidate_ids) or len(
        set(artifact_ids)
    ) != len(artifact_ids):
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_duplicate_sources",
            "兼容的单变量证据存在重复 candidate 或 artifact 身份；"
            "平台不会猜测或重复归集。",
        )
    if len(candidate_requests) > _MAX_UNIVARIATE_SOURCES:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_candidate_budget_exceeded",
            "与最新 StrategySampleDesign V2 兼容的单变量候选数量超过"
            f"单次全局来源上限（{_MAX_UNIVARIATE_SOURCES}）；"
            "请先缩小候选范围。",
        )
    sample_design_ref = {
        "membership_artifact_id": sample_binding.membership_artifact_id,
        "expected_membership_artifact_content_hash": (
            sample_binding.membership_artifact_content_hash
        ),
        "bundle_artifact_id": sample_binding.bundle_artifact_id,
        "expected_bundle_artifact_content_hash": (
            sample_binding.bundle_artifact_content_hash
        ),
        "expected_bundle_id": sample_binding.bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }
    try:
        _validate_model_evidence_v2_inputs(
            {
                "sample_design_ref": sample_design_ref,
                "univariate_sources": candidate_requests,
            }
        )
        _load_candidate_sources(
            read_runtime,
            task_id=task.id,
            requests=candidate_requests,
            sample_binding=sample_binding,
        )
    except (StrategyError, *_STRATEGY_V2_ARTIFACT_ERRORS) as exc:
        # Validate and authenticate the entire source set in one batch.  This
        # preserves global count/duplicate/JSON and cumulative file-byte
        # budgets instead of resetting a budget for every candidate.
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_candidate_invalid",
            "一份或多份声称与最新 StrategySampleDesign V2 兼容的单变量候选"
            "未通过全局来源预算、文件、hash、路径、provenance 或 task "
            "所有权复核；本次未创建计划。",
        ) from exc
    if verify_current:
        current_artifacts = _strategy_v2_artifact_snapshot(
            read_runtime,
            task_id=task.id,
        )
        if _strategy_v2_registry_token(current_artifacts) != registry_token:
            raise _StrategyV2EvidenceSetupError(
                "strategy_model_evidence_v2_registry_changed",
                "StrategySampleDesign V2 或单变量候选 registry 在计划创建前"
                "发生变化；平台已冻结本次计划创建，请基于最新证据重试。",
            )
    return {
        "sample_design_ref": sample_design_ref,
        "univariate_sources": candidate_requests,
    }


def _strategy_v2_read_runtime(runtime: DriverTurnRuntime) -> SimpleNamespace:
    backend, registry = _modeling_data_runtime(runtime.settings)
    return SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=registry,
        task_artifacts=TaskArtifactRepository(runtime.settings.db_path),
    )


def _latest_verified_strategy_sample_design_v2_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    artifacts: Sequence[Mapping],
):
    # TaskArtifactRepository.list_for_task is deterministic
    # ORDER BY created_at, id; the final row is therefore the latest published
    # V2 bundle, and a drifted latest bundle is never bypassed for an older one.
    bundles = [
        artifact
        for artifact in artifacts
        if artifact.get("kind") == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
        and artifact.get("origin_tool") == SAMPLE_DESIGN_V2_ORIGIN_TOOL
    ]
    if not bundles:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_required",
            "当前任务还没有 StrategySampleDesign V2 双总体样本证据；"
            "请先用自然语言固化 V2 样本设计。",
        )
    newest = bundles[-1]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_invalid",
            "最新 StrategySampleDesign V2 bundle 缺少完整 provenance，"
            "本次未创建计划。",
        )
    try:
        return load_strategy_sample_design_v2_artifacts(
            read_runtime,
            task_id=task_id,
            membership_artifact_id=provenance.get("membership_artifact_id"),
            expected_membership_artifact_content_hash=provenance.get(
                "membership_artifact_content_hash"
            ),
            bundle_artifact_id=newest.get("id"),
            expected_bundle_artifact_content_hash=newest.get("content_hash"),
            expected_bundle_id=provenance.get("bundle_id"),
            expected_sample_design_id=provenance.get("sample_design_id"),
            expected_sample_design_content_hash=provenance.get(
                "sample_design_content_hash"
            ),
        )
    except (
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_sample_invalid",
            "最新 StrategySampleDesign V2 membership/bundle pair 未通过"
            "文件、registry、provenance 或数据漂移复核；请重新固化样本设计。",
        ) from exc


def _strategy_v2_artifact_snapshot(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
) -> tuple[dict, ...]:
    """Read one deterministic registry snapshot or expose a governed error."""

    try:
        return tuple(read_runtime.task_artifacts.list_for_task(task_id))
    except _STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_registry_unavailable",
            "无法读取当前任务的 StrategySampleDesign V2 artifact registry。",
        ) from exc


def _strategy_v2_registry_token(artifacts: Sequence[Mapping]) -> str:
    """CAS token for evidence rows relevant to one ModelEvidence V2 plan."""

    relevant = [
        {
            "id": artifact.get("id"),
            "kind": artifact.get("kind"),
            "content_hash": artifact.get("content_hash"),
            "origin_tool": artifact.get("origin_tool"),
            "provenance": artifact.get("provenance"),
            "created_at": artifact.get("created_at"),
        }
        for artifact in artifacts
        if (
            artifact.get("kind") == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
            and artifact.get("origin_tool") == SAMPLE_DESIGN_V2_ORIGIN_TOOL
        )
        or (
            artifact.get("kind") == "strategy_candidate_json"
            and artifact.get("origin_tool")
            == "strategy.analyze_univariate_candidates"
        )
    ]
    try:
        payload = json.dumps(
            relevant,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_registry_unavailable",
            "Strategy ModelEvidence V2 registry snapshot 无法规范化。",
        ) from exc
    return hashlib.sha256(payload).hexdigest()


_STRATEGY_REPORT_POOL_TYPE_PATTERNS = {
    "approval": re.compile(
        r"(?:审批|准入)|(?<![A-Za-z0-9_])approval(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"拒绝|(?<![A-Za-z0-9_])reject(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"额度|(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"定价|利率|(?<![A-Za-z0-9_])pricing(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "segmentation": re.compile(
        r"分群|分层|客群|"
        r"(?<![A-Za-z0-9_])segmentation(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}
_STRATEGY_REPORT_POOL_COMMAND_RE = re.compile(
    r"(?:生成|创建|制作|编制|形成|出一份|出个|给我|导出|构建)"
    r"[^；;。.!?？\n]{0,80}(?:报告|Report)|"
    r"(?<![A-Za-z0-9_])(?:generate|create|build|produce|prepare|render|export)"
    r"[^;.!?\n]{0,80}\breport(?:\s+bundle)?\b",
    re.IGNORECASE,
)
_STRATEGY_REPORT_POOL_TITLE_RE = re.compile(
    r"(?:报告标题|标题|report\s+title|title)\s*"
    r"(?:为|是|叫|is|=|:|：)\s*"
    r"(?:[“\"'《][^”\"'》\n]{1,200}[”\"'》]|"
    r"[^，,；;。.!?？\n]{1,200})",
    re.IGNORECASE,
)
_STRATEGY_REPORT_POOL_SELECTOR_RE = re.compile(
    r"(?:选择|选用|使用|采用|针对|指定|按|基于|改用|就用|要用|而是|"
    r"(?:Pool|策略)\s*类型\s*(?:为|是|=|:|：)|"
    r"(?<![A-Za-z0-9_])(?:select|choose|use|using|for|on|but|instead)"
    r"(?![A-Za-z0-9_]))\s*$",
    re.IGNORECASE,
)
_STRATEGY_REPORT_POOL_TYPE_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|禁止|排除|剔除|而非|不是|并非|"
    r"不使用|不选|不选择)\s*(?:(?:选择|选用|使用|采用|针对|指定)\s*)?"
    r"[^，,；;。.!?？\n]{0,16}$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without|exclude)"
    r"[^,;.!?\n]{0,20}$",
    re.IGNORECASE,
)
_STRATEGY_REPORT_POOL_HISTORY_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史|已归档|已生成)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historical|"
    r"last\s+time|archived|already\s+generated)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT = 64
_STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT = (
    _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
)


def _strategy_report_bundle_v2_plan_slots(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StandardWorkflowRequestDraft,
    *,
    source_message: Mapping | None,
) -> dict[str, object]:
    """Bind one report plan to immutable, fully authenticated source refs."""

    inputs = draft.to_dict()["workflow_inputs"]
    read_runtime = _strategy_report_read_runtime(runtime)

    try:
        project_context = load_current_strategy_project_context_artifact(
            read_runtime,
            task_id=task.id,
        )
    except (
        StrategyError,
        StrategyProjectContextDataError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_project_context_invalid",
            "当前 Strategy ProjectContext 未通过 head、artifact、来源或文件完整性复核；"
            "请先重新整理项目上下文。",
        ) from exc
    if project_context is None:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_project_context_required",
            "生成策略报告前必须先固化当前 Strategy ProjectContext。",
        )

    sample = _strategy_report_latest_sample_binding(
        read_runtime,
        task_id=task.id,
    )
    sample_ref = _strategy_report_sample_ref(sample)
    requested_pool_type = _strategy_report_requested_pool_type(source_message)
    pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=task.id,
        requested_type=requested_pool_type,
    )
    candidate_pool_ref = {
        "strategy_type": pool.strategy_type,
        "expected_pool_revision": pool.pool["revision"],
        "expected_pool_snapshot_hash": pool.pool["snapshot_hash"],
        "expected_artifact_id": pool.artifact_id,
        "expected_artifact_content_hash": pool.artifact_content_hash,
    }
    candidate_stability = (
        _strategy_report_latest_candidate_stability_binding(
            read_runtime,
            task_id=task.id,
            sample=sample,
            pool=pool,
        )
    )
    candidate_stability_ref = (
        None
        if candidate_stability is None
        else {
            "artifact_id": candidate_stability.artifact_id,
            "expected_artifact_content_hash": (
                candidate_stability.artifact_content_hash
            ),
            "expected_stability_id": candidate_stability.stability[
                "stability_id"
            ],
            "expected_stability_content_hash": (
                candidate_stability.stability["content_hash"]
            ),
        }
    )
    voting_candidate_search = _strategy_report_latest_voting_search_binding(
        read_runtime,
        task_id=task.id,
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )
    voting_candidate_search_ref = (
        None
        if voting_candidate_search is None
        else {
            "artifact_id": voting_candidate_search.artifact_id,
            "expected_artifact_content_hash": (
                voting_candidate_search.artifact_content_hash
            ),
            "expected_search_id": voting_candidate_search.result["search_id"],
            "expected_search_content_hash": (
                voting_candidate_search.result["content_hash"]
            ),
        }
    )
    impact_cube = _strategy_report_latest_impact_cube_binding(
        read_runtime,
        task_id=task.id,
        pool=pool,
        sample_ref=sample_ref,
    )
    impact_cube_ref = (
        None
        if impact_cube is None
        else {
            "artifact_id": impact_cube.artifact_id,
            "expected_artifact_content_hash": (
                impact_cube.artifact_content_hash
            ),
            "expected_cube_id": impact_cube.cube["cube_id"],
            "expected_cube_content_hash": impact_cube.cube["content_hash"],
        }
    )
    impact = None
    pool_impact_ref = None
    if impact_cube is None:
        if pool.strategy_type not in {"approval", "reject"}:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_impact_cube_required",
                f"当前 {pool.strategy_type} Strategy Pool 没有同 revision、"
                "snapshot 和 SampleDesign 的 ImpactCube；该策略类型不允许"
                "回退到旧 PoolImpact，请先单独完成 ImpactCube 测算。",
            )
        impact = _strategy_report_latest_pool_impact_binding(
            read_runtime,
            task_id=task.id,
            pool=pool,
        )
        pool_impact_ref = {
            "artifact_id": impact.artifact_id,
            "expected_artifact_content_hash": impact.artifact_content_hash,
            "expected_assessment_id": impact.assessment["assessment_id"],
            "expected_assessment_content_hash": impact.assessment[
                "content_hash"
            ],
        }

    try:
        pool_validation_refs = (
            select_latest_strategy_pool_validation_refs(
                read_runtime,
                task_id=task.id,
                candidate_pool=pool,
                sample_design=sample,
            )
        )
        pool_validations = load_strategy_pool_validation_artifacts(
            read_runtime,
            task_id=task.id,
            refs=pool_validation_refs,
            candidate_pool=pool,
            sample_design=sample,
        )
    except (StrategyError, *_STRATEGY_V2_ARTIFACT_ERRORS) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_validation_invalid",
            "当前 Strategy Pool 的独立样本回放证据未通过 exact ref、"
            "来源或文件完整性复核；本次未创建计划。",
        ) from exc

    model_evidence, model_evidence_ref = (
        _strategy_report_optional_model_evidence(
            read_runtime,
            task_id=task.id,
            sample_ref=sample_ref,
        )
    )
    training_evidence, training_evidence_ref = (
        _strategy_report_optional_training_evidence(
            read_runtime,
            task_id=task.id,
            sample_ref=sample_ref,
        )
    )
    score_evidence, score_evidence_ref = _strategy_report_optional_score_evidence(
        read_runtime,
        task_id=task.id,
        sample_ref=sample_ref,
        training_ref=training_evidence_ref,
    )

    strategy_identity = _strategy_report_identity(
        runtime,
        task_id=task.id,
        strategy_type=pool.strategy_type,
    )
    strategy_id = (
        None if strategy_identity is None else strategy_identity["strategy_id"]
    )
    try:
        head = StrategyReportRepository(runtime.settings.db_path).get_head(
            task_id=task.id,
            strategy_id=strategy_id,
        )
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_head_invalid",
            "当前策略报告 head/CAS 无法通过完整性复核；本次未创建计划。",
        ) from exc

    # Reuse the report adapter as the final cross-source preflight.  It proves
    # that the selected PoolImpact belongs to this exact Pool and V2
    # risk/development sample, and that every optional model chain matches the
    # same SampleDesign.
    try:
        build_strategy_report_bundle_source_inputs(
            project_context=project_context,
            sample_design=sample,
            candidate_pool=pool,
            pool_validations=pool_validations,
            candidate_stability=candidate_stability,
            voting_candidate_search=voting_candidate_search,
            pool_impact=impact,
            impact_cube=impact_cube,
            model_evidence=model_evidence,
            training_evidence=training_evidence,
            score_evidence=score_evidence,
        )
    except (ModelingError, StrategyError, *_STRATEGY_V2_ARTIFACT_ERRORS) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_source_incompatible",
            "当前 ProjectContext、SampleDesign V2、Strategy Pool、"
            "ImpactCube/兼容 PoolImpact 或可选模型证据不属于同一条"
            "可认证证据链；本次未创建计划。",
        ) from exc

    return {
        "title": inputs["title"],
        "status": inputs["status"],
        "project_context_ref": {
            "artifact_id": project_context.artifact_id,
            "expected_artifact_content_hash": (
                project_context.artifact_content_hash
            ),
            "expected_revision": project_context.revision["revision"],
            "expected_revision_id": project_context.revision["revision_id"],
            "expected_state_hash": project_context.revision["state_hash"],
        },
        "sample_design_ref": sample_ref,
        "candidate_pool_ref": candidate_pool_ref,
        "pool_validation_refs": list(pool_validation_refs),
        "candidate_stability_ref": candidate_stability_ref,
        "voting_candidate_search_ref": voting_candidate_search_ref,
        "pool_impact_ref": pool_impact_ref,
        "impact_cube_ref": impact_cube_ref,
        "report_revision": int(head["current_revision"]) + 1,
        "previous_report_id": head["current_report_id"],
        "previous_report_content_hash": head["current_content_hash"],
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_identity": strategy_identity,
        "model_evidence_ref": model_evidence_ref,
        "training_evidence_ref": training_evidence_ref,
        "score_evidence_ref": score_evidence_ref,
    }


def _strategy_report_read_runtime(
    runtime: DriverTurnRuntime,
) -> SimpleNamespace:
    backend, registry = _modeling_data_runtime(runtime.settings)
    task_artifacts = TaskArtifactRepository(runtime.settings.db_path)
    return SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=registry,
        task_artifacts=task_artifacts,
        strategies=StrategyRepository(runtime.settings.db_path),
        experiments=ExperimentStore(runtime.settings.db_path),
        modeling_repo=ModelingRepository(runtime.settings.db_path),
    )


def _strategy_report_artifact_window(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    kind: str,
    limit: int,
    unavailable_code: str,
    invalid_code: str,
    label: str,
) -> tuple[tuple[Mapping, ...], int]:
    """Read one exact newest-first artifact window without full-task allocation."""

    try:
        records, total = (
            read_runtime.task_artifacts.list_recent_for_task_kind_with_count(
                task_id,
                kind,
                limit=limit,
            )
        )
    except _STRATEGY_V2_ARTIFACT_ERRORS as exc:
        raise _StrategyV2EvidenceSetupError(
            unavailable_code,
            f"无法读取当前任务最新的 {label} artifact 窗口。",
        ) from exc
    try:
        if (
            not isinstance(records, Sequence)
            or isinstance(records, str | bytes | bytearray)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 0
        ):
            raise ValueError(f"{label} artifact window is invalid")
        window = tuple(records)
        if len(window) != min(total, limit) or any(
            not isinstance(item, Mapping) or item.get("kind") != kind
            for item in window
        ):
            raise ValueError(f"{label} artifact window is inconsistent")
        return window, total
    except (TypeError, ValueError) as exc:
        raise _StrategyV2EvidenceSetupError(
            invalid_code,
            f"{label} artifact 窗口与精确总数或 kind 不一致；"
            "其 newest-first 选择边界无法确认。",
        ) from exc


def _strategy_report_requested_pool_type(
    source_message: Mapping | None,
) -> str | None:
    text = (
        str(source_message.get("content") or "")
        if isinstance(source_message, Mapping)
        else ""
    )
    masked = list(text)
    for match in _STRATEGY_REPORT_POOL_TITLE_RE.finditer(text):
        masked[match.start() : match.end()] = " " * (
            match.end() - match.start()
        )
    command_text = "".join(masked)
    actions = tuple(_STRATEGY_REPORT_POOL_COMMAND_RE.finditer(command_text))
    selected: list[str] = []
    negated: list[str] = []
    for strategy_type, pattern in _STRATEGY_REPORT_POOL_TYPE_PATTERNS.items():
        for match in pattern.finditer(command_text):
            prefix = command_text[max(0, match.start() - 40) : match.start()]
            if _STRATEGY_REPORT_POOL_TYPE_NEGATION_RE.search(prefix):
                if strategy_type not in negated:
                    negated.append(strategy_type)
                continue
            clause_start = max(
                command_text.rfind(separator, 0, match.start())
                for separator in ("，", ",", "；", ";", "。", ".", "！", "!", "？", "?", "\n")
            )
            clause_end_candidates = [
                position
                for separator in (
                    "，",
                    ",",
                    "；",
                    ";",
                    "。",
                    ".",
                    "！",
                    "!",
                    "？",
                    "?",
                    "\n",
                )
                if (position := command_text.find(separator, match.end())) >= 0
            ]
            clause_end = (
                min(clause_end_candidates)
                if clause_end_candidates
                else len(command_text)
            )
            local_clause = command_text[clause_start + 1 : clause_end]
            if _STRATEGY_REPORT_POOL_HISTORY_RE.search(local_clause):
                continue

            inside_creation = any(
                action.start() <= match.start()
                and match.end() <= action.end()
                for action in actions
            )
            explicitly_selected = (
                _STRATEGY_REPORT_POOL_SELECTOR_RE.search(prefix) is not None
                and _strategy_report_pool_selector_shares_command(
                    command_text,
                    mention_start=match.start(),
                    mention_end=match.end(),
                    actions=actions,
                )
            )
            if inside_creation or explicitly_selected:
                if strategy_type not in selected:
                    selected.append(strategy_type)
                break
    if len(selected) > 1:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_ambiguous",
            "同一报告请求同时点名多个 Strategy Pool 类型；"
            "请只选择一种策略类型。",
        )
    if not selected and negated:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_required",
            "报告请求只排除了 Pool 类型，没有明确肯定选择要使用的"
            "审批/准入、拒绝、额度、定价或分群 Pool；平台不会绑定"
            "被否定的 Pool。",
        )
    return selected[0] if selected else None


def _strategy_report_pool_selector_shares_command(
    utterance: str,
    *,
    mention_start: int,
    mention_end: int,
    actions: Sequence[re.Match[str]],
) -> bool:
    sentence_start = max(
        utterance.rfind(separator, 0, mention_start)
        for separator in ("。", ".", "！", "!", "？", "?", "；", ";", "\n")
    )
    sentence_end_candidates = [
        position
        for separator in ("。", ".", "！", "!", "？", "?", "；", ";", "\n")
        if (position := utterance.find(separator, mention_end)) >= 0
    ]
    sentence_end = (
        min(sentence_end_candidates)
        if sentence_end_candidates
        else len(utterance)
    )
    return any(
        sentence_start < action.start()
        and action.end() <= sentence_end
        for action in actions
    )


def _strategy_report_current_pool_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    requested_type: str | None,
):
    repository = StrategyCandidatePoolRepository(
        read_runtime.settings.db_path
    )
    current: dict[str, Mapping] = {}
    strategy_types = (
        (requested_type,)
        if requested_type is not None
        else (
            "approval",
            "reject",
            "limit",
            "pricing",
            "segmentation",
        )
    )
    try:
        for strategy_type in strategy_types:
            pool = repository.get_current(task_id, strategy_type)
            if pool is not None and pool.get("entries"):
                current[strategy_type] = pool
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_invalid",
            "当前 Strategy Pool head/revision 无法通过完整性复核。",
        ) from exc

    if requested_type is not None:
        selected_type = requested_type
        if selected_type not in current:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_required",
                f"当前任务没有非空 {selected_type} Strategy Pool。",
            )
    elif len(current) == 1:
        selected_type = next(iter(current))
    elif not current:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_required",
            "当前任务没有可用于报告的非空 Strategy Pool。",
        )
    else:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_type_required",
            "当前同时存在多个非空 Strategy Pool；请在报告请求中明确"
            "选择审批/准入、拒绝、额度、定价或分群 Pool，平台不会猜测。",
        )

    selected = current[selected_type]
    try:
        return load_current_strategy_candidate_pool_artifact(
            read_runtime,
            task_id=task_id,
            strategy_type=selected_type,
            expected_pool_revision=selected["revision"],
            expected_pool_snapshot_hash=strategy_pool_snapshot_hash(selected),
        )
    except (StrategyError, *_STRATEGY_V2_ARTIFACT_ERRORS) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_invalid",
            f"当前 {selected_type} Strategy Pool 的 artifact、来源或数据绑定"
            "未通过完整性复核。",
        ) from exc


def _strategy_report_latest_sample_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
):
    bundles, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
        limit=1,
        unavailable_code="strategy_report_bundle_v2_sample_registry_unavailable",
        invalid_code="strategy_report_bundle_v2_sample_invalid",
        label="StrategySampleDesign V2 bundle",
    )
    if not bundles:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_sample_required",
            "当前任务没有 StrategySampleDesign V2 membership/bundle 证据。",
        )
    newest = bundles[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_sample_invalid",
            "最新 StrategySampleDesign V2 bundle provenance 已损坏；"
            "平台不会回退到旧样本设计。",
        )
    try:
        return load_strategy_sample_design_v2_artifacts(
            read_runtime,
            task_id=task_id,
            membership_artifact_id=provenance.get("membership_artifact_id"),
            expected_membership_artifact_content_hash=provenance.get(
                "membership_artifact_content_hash"
            ),
            bundle_artifact_id=newest.get("id"),
            expected_bundle_artifact_content_hash=newest.get("content_hash"),
            expected_bundle_id=provenance.get("bundle_id"),
            expected_sample_design_id=provenance.get("sample_design_id"),
            expected_sample_design_content_hash=provenance.get(
                "sample_design_content_hash"
            ),
        )
    except (
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_sample_invalid",
            "最新 StrategySampleDesign V2 membership/bundle 未通过文件、"
            "registry、provenance 或数据漂移复核；平台不会回退到旧版本。",
        ) from exc


def _strategy_report_sample_ref(sample) -> dict[str, object]:
    design = sample.bundle["sample_design"]
    return {
        "membership_artifact_id": sample.membership_artifact_id,
        "expected_membership_artifact_content_hash": (
            sample.membership_artifact_content_hash
        ),
        "bundle_artifact_id": sample.bundle_artifact_id,
        "expected_bundle_artifact_content_hash": (
            sample.bundle_artifact_content_hash
        ),
        "expected_bundle_id": sample.bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }


def _strategy_report_latest_candidate_stability_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
    pool,
):
    """Select the newest authenticated stability evidence for current sources.

    Every stability artifact is authenticated before its source identity is
    inspected.  A valid artifact for another Pool/SampleDesign is skipped; a
    corrupt candidate fails closed because its actual source cannot be trusted
    and the selector must not silently fall back to older evidence.
    """

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CANDIDATE_STABILITY_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_candidate_stability_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_candidate_stability_invalid",
        label="candidate stability",
    )
    for item in records:
        provenance = item.get("provenance")
        try:
            binding = load_candidate_stability_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_stability_id=(
                    provenance.get("stability_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_stability_content_hash=(
                    provenance.get("stability_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
        except (
            StrategyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_candidate_stability_invalid",
                "最新待判定的候选逐月稳定性 artifact 未通过文件、registry、"
                "provenance 或内容完整性复核；其真实 Pool/SampleDesign "
                "身份无法确认，平台不会回退到旧稳定性证据。",
            ) from exc
        try:
            validate_candidate_stability_report_compatibility(
                candidate_stability=binding,
                sample_design=sample,
                candidate_pool=pool,
            )
        except StrategyError:
            continue
        return binding
    if total > _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_candidate_stability_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个候选逐月稳定性 "
            "artifact，但 registry 仍有更早记录；平台无法证明窗口外"
            "不存在与当前 Pool/SampleDesign 完全一致的稳定性证据，"
            "本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_voting_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
    sample_ref: Mapping[str, object],
    pool,
):
    """Select newest-to-oldest fully authenticated exact Voting search evidence."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_voting_candidate_search_"
            "registry_unavailable"
        ),
        invalid_code=(
            "strategy_report_bundle_v2_voting_candidate_search_invalid"
        ),
        label="Voting candidate search",
    )
    if not records:
        return None
    try:
        current_development = bind_strategy_pool_development_execution(
            read_runtime,
            pool,
        )
        entries = [
            dict(entry)
            for entry in pool.pool["entries"]
            if entry["enabled"] is True
            and entry["source"]["asset_type"]
            != VOTING_CANDIDATE_ASSET_TYPE
        ]
        candidate_ids = sorted(str(entry["rule_id"]) for entry in entries)
        requirements = project_pool_entry_requirements(entries)
        if requirements:
            resolved = resolve_pool_requirements(
                read_runtime,
                task_id=task_id,
                compiled_design={"requirements": list(requirements)},
                sample_design=sample,
            )
            requirement_bindings = pool_requirement_bindings_provenance(
                resolved
            )
        else:
            requirement_bindings = None
        if (
            current_development.sample_design_v2 is None
            or _strategy_report_sample_ref(
                current_development.sample_design_v2
            )
            != dict(sample_ref)
        ):
            return None
    except (
        KeyError,
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_voting_candidate_search_invalid",
            "当前 Strategy Pool 的 Voting 搜索匹配身份无法通过完整认证；"
            "本次未创建报告计划。",
        ) from exc

    for item in records:
        try:
            binding = load_historical_voting_candidate_search_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
            )
        except (
            KeyError,
            ModelingError,
            StrategyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_voting_candidate_search_invalid",
                "最新待判定的 Voting 候选搜索 artifact 未通过历史安全的"
                "文件、registry、provenance、Pool 或数据绑定复核；其真实"
                "身份无法确认，平台不会回退到旧搜索证据。",
            ) from exc
        if _strategy_report_voting_search_matches(
            binding,
            task_id=task_id,
            sample_ref=sample_ref,
            pool=pool,
            current_development=current_development,
            candidate_ids=candidate_ids,
            requirement_bindings=requirement_bindings,
        ):
            return binding
    if total > _STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_voting_candidate_search_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT} 个 Voting 候选"
            "搜索 artifact，但 registry 仍有更早的搜索记录；平台无法证明"
            "窗口外不存在与当前 Pool/SampleDesign 完全一致的搜索证据，"
            "本次未创建报告计划。",
        )
    return None


def _strategy_report_voting_search_matches(
    binding,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
    pool,
    current_development,
    candidate_ids: Sequence[str],
    requirement_bindings: Mapping[str, object] | None,
) -> bool:
    provenance = binding.artifact_provenance
    historical_development = binding.pool_development
    historical_pool = historical_development.pool
    if (
        binding.task_id != task_id
        or historical_pool.artifact_id != pool.artifact_id
        or historical_pool.artifact_content_hash
        != pool.artifact_content_hash
        or historical_pool.pool != pool.pool
        or provenance["task_id"] != task_id
        or provenance["pool_ref"]
        != {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "pool_id": pool.pool["pool_id"],
            "strategy_type": pool.pool["strategy_type"],
            "revision": pool.pool["revision"],
            "revision_id": pool.pool["revision_id"],
            "snapshot_hash": pool.pool["snapshot_hash"],
        }
    ):
        return False
    historical_sample_v2 = historical_development.sample_design_v2
    if (
        historical_sample_v2 is None
        or _strategy_report_sample_ref(historical_sample_v2)
        != dict(sample_ref)
    ):
        return False

    dataset = current_development.dataset
    execution_sample = current_development.sample_design
    expected_dataset = {
        "task_id": dataset.task_id,
        "dataset_id": dataset.dataset_id,
        "dataset_source_path": dataset.source_path,
        "dataset_content_hash": dataset.content_hash,
        "dataset_registry_metadata_hash": dataset.registry_metadata_hash,
        "workspace_revision": execution_sample.workspace_revision,
        "workspace_generation": execution_sample.workspace_generation,
        "semantic_mapping_hash": execution_sample.semantic_mapping_hash,
    }
    target = provenance["target_binding"]
    expected_target_identity = {
        "column": execution_sample.target_col,
        "raw_bad_value": execution_sample.target_bad_value,
        "normalized_bad_value": 1,
        "drop_nan_labels": execution_sample.drop_nan_labels,
        "sample_partition": execution_sample.reference.partition,
    }
    if (
        provenance["dataset_binding"] != expected_dataset
        or provenance["sample_design_ref"]
        != execution_sample.to_ref_dict()
        or provenance["sample_context_hash"]
        != current_development.evidence_identity["sample_context_hash"]
        or any(
            target.get(field) != expected
            for field, expected in expected_target_identity.items()
        )
        or target["labeled_count"] + target["nan_labels_dropped"]
        != execution_sample.development_population_count
        or (
            target["nan_labels_dropped"] > 0
            and not execution_sample.drop_nan_labels
        )
        or provenance["observation_bindings"]
        != {
            "weight_col": execution_sample.weight_col,
            "amount_col": execution_sample.loan_amount_col,
        }
        or provenance["requirement_bindings"]
        != (
            None
            if requirement_bindings is None
            else dict(requirement_bindings)
        )
        or binding.result["configuration"]["candidate_ids"]
        != list(candidate_ids)
    ):
        return False
    return True


def _strategy_report_latest_impact_cube_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    pool,
    sample_ref: Mapping[str, object],
):
    """Prefer the newest exact Pool + SampleDesign ImpactCube.

    The repository window is newest-first. Authenticate each candidate before
    inspecting its embedded Pool/SampleDesign identity. A valid unrelated cube
    can be skipped, while an unauthenticatable candidate fails closed because
    its raw provenance cannot safely prove that it was unrelated.
    """

    expected_pool_ref = {
        "artifact_id": pool.artifact_id,
        "expected_artifact_content_hash": pool.artifact_content_hash,
        "expected_pool_id": pool.pool["pool_id"],
        "expected_revision": pool.pool["revision"],
        "expected_revision_id": pool.pool["revision_id"],
        "expected_snapshot_hash": pool.pool["snapshot_hash"],
    }
    same_kind, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=IMPACT_CUBE_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_impact_cube_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_impact_cube_invalid",
        label="ImpactCube",
    )
    for item in same_kind:
        provenance = item.get("provenance")
        try:
            binding = load_strategy_impact_cube_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_cube_id=(
                    provenance.get("cube_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_cube_content_hash=(
                    provenance.get("cube_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
            cube = binding.cube
            identity = cube["identity"]
            sources = cube["source_bindings"]
            pool_artifact = sources["pool_artifact"]
            sample = sources["sample_design_v2"]
            authenticated_pool_ref = {
                "artifact_id": pool_artifact["artifact_id"],
                "expected_artifact_content_hash": pool_artifact[
                    "artifact_content_hash"
                ],
                "expected_pool_id": identity["pool_id"],
                "expected_revision": identity["revision"],
                "expected_revision_id": identity["revision_id"],
                "expected_snapshot_hash": identity["snapshot_hash"],
            }
            authenticated_sample_ref = {
                "membership_artifact_id": sample[
                    "membership_artifact_id"
                ],
                "expected_membership_artifact_content_hash": sample[
                    "membership_artifact_content_hash"
                ],
                "bundle_artifact_id": sample["bundle_artifact_id"],
                "expected_bundle_artifact_content_hash": sample[
                    "bundle_artifact_content_hash"
                ],
                "expected_bundle_id": sample["bundle_id"],
                "expected_sample_design_id": sample["sample_design_id"],
                "expected_sample_design_content_hash": sample[
                    "sample_design_content_hash"
                ],
            }
        except (
            KeyError,
            StrategyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_impact_cube_invalid",
                "最新待判定的 ImpactCube 候选未通过文件、registry、"
                "provenance、producer-run 或 audit 复核；其真实 Pool/"
                "SampleDesign 身份无法确认，平台不会回退到旧 ImpactCube "
                "或 PoolImpact。",
            ) from exc
        if (
            authenticated_pool_ref == expected_pool_ref
            and authenticated_sample_ref == dict(sample_ref)
        ):
            return binding
    if total > _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_impact_cube_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个 ImpactCube artifact，"
            "但 registry 仍有更早记录；平台无法证明窗口外不存在与当前 "
            "Pool/SampleDesign 完全一致的 ImpactCube，本次未创建报告计划。",
        )
    return None


def _strategy_report_latest_pool_impact_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    pool,
):
    same_kind, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=POOL_IMPACT_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_pool_impact_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_pool_impact_invalid",
        label="PoolImpact",
    )
    for item in same_kind:
        provenance = item.get("provenance")
        try:
            binding = load_historical_strategy_pool_impact_artifact(
                read_runtime,
                task_id=task_id,
                artifact_id=item.get("id"),
                expected_artifact_content_hash=item.get("content_hash"),
                expected_assessment_id=(
                    provenance.get("assessment_id")
                    if isinstance(provenance, Mapping)
                    else None
                ),
                expected_assessment_content_hash=(
                    provenance.get("assessment_content_hash")
                    if isinstance(provenance, Mapping)
                    else None
                ),
            )
        except (
            StrategyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_impact_invalid",
                "最新待判定的 PoolImpact 未通过历史安全的文件、registry、"
                "provenance、Pool 或样本绑定复核；其真实身份无法确认，"
                "平台不会回退到旧影响证据。",
            ) from exc
        if (
            binding.stage != "development_backtest"
            or binding.pool.artifact_id != pool.artifact_id
            or binding.pool.artifact_content_hash
            != pool.artifact_content_hash
            or binding.pool.pool != pool.pool
        ):
            continue
        return binding
    if total > _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_impact_"
            "selection_window_exhausted",
            "已检查最新 "
            f"{_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT} 个 PoolImpact artifact，"
            "但 registry 仍有更早记录；平台无法证明窗口外不存在当前 "
            "Pool revision/snapshot 的精确 development 证据，"
            "本次未创建报告计划。",
        )
    raise _StrategyV2EvidenceSetupError(
        "strategy_report_bundle_v2_pool_impact_required",
        "当前非空 Strategy Pool 没有同 revision/snapshot 的 development "
        "PoolImpact；请先单独完成影响测算。",
    )


def _strategy_report_optional_model_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODEL_EVIDENCE_V2_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="ModelEvidence",
    )
    if not records:
        return None, None
    newest = records[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        _raise_corrupt_report_optional("ModelEvidence")
    try:
        binding = load_strategy_model_evidence_v2_artifact(
            read_runtime,
            task_id=task_id,
            artifact_id=newest.get("id"),
            expected_artifact_content_hash=newest.get("content_hash"),
            expected_bundle_id=provenance.get("bundle_id"),
            expected_bundle_content_hash=provenance.get("bundle_content_hash"),
            sample_design_ref=provenance.get("sample_design_ref"),
        )
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        _raise_corrupt_report_optional("ModelEvidence", cause=exc)
    reference = {
        "artifact_id": binding.artifact_id,
        "expected_artifact_content_hash": binding.artifact_content_hash,
        "expected_bundle_id": binding.bundle["bundle_id"],
        "expected_bundle_content_hash": binding.bundle["content_hash"],
    }
    if _strategy_report_sample_ref(binding.sample_design_binding) != dict(
        sample_ref
    ):
        return None, None
    return binding, reference


def _strategy_report_optional_training_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="training evidence",
    )
    if not records:
        return None, None
    newest = records[0]
    try:
        reference = _strategy_report_training_ref(
            read_runtime,
            task_id=task_id,
            record=newest,
        )
        binding = load_modeling_training_evidence_artifacts(
            read_runtime,
            task_id=task_id,
            **reference,
        )
        reference = build_training_evidence_ref(binding)
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        _raise_corrupt_report_optional("training evidence", cause=exc)
    if reference["sample_design_ref"] != dict(sample_ref):
        return None, None
    return binding, reference


def _strategy_report_training_ref(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    record: Mapping,
) -> dict[str, object]:
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training evidence provenance is invalid")
    sample_ref = _strategy_report_sample_ref_from_registry(
        read_runtime,
        task_id=task_id,
        membership_artifact_id=provenance.get(
            "sample_membership_artifact_id"
        ),
        bundle_artifact_id=provenance.get("sample_bundle_artifact_id"),
    )
    return {
        "sample_design_ref": sample_ref,
        "model_binary_artifact_id": provenance.get(
            "model_binary_artifact_id"
        ),
        "expected_model_binary_artifact_content_hash": provenance.get(
            "model_binary_artifact_content_hash"
        ),
        "evidence_artifact_id": record.get("id"),
        "expected_evidence_artifact_content_hash": record.get("content_hash"),
        "expected_experiment_id": provenance.get("experiment_id"),
        "expected_model_artifact_id": provenance.get("model_artifact_id"),
        "expected_evidence_id": provenance.get("evidence_id"),
        "expected_evidence_content_hash": provenance.get(
            "evidence_content_hash"
        ),
    }


def _strategy_report_sample_ref_from_registry(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    membership_artifact_id: object,
    bundle_artifact_id: object,
) -> dict[str, object]:
    membership = read_runtime.task_artifacts.get_for_task(
        task_id,
        membership_artifact_id,
    )
    bundle = read_runtime.task_artifacts.get_for_task(
        task_id,
        bundle_artifact_id,
    )
    if (
        not isinstance(membership, Mapping)
        or membership.get("kind")
        != SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
        or not isinstance(bundle, Mapping)
        or bundle.get("kind") != SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    ):
        raise ValueError("training evidence sample artifact pair is missing")
    provenance = bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("training evidence sample bundle provenance is invalid")
    return {
        "membership_artifact_id": membership.get("id"),
        "expected_membership_artifact_content_hash": membership.get(
            "content_hash"
        ),
        "bundle_artifact_id": bundle.get("id"),
        "expected_bundle_artifact_content_hash": bundle.get("content_hash"),
        "expected_bundle_id": provenance.get("bundle_id"),
        "expected_sample_design_id": provenance.get("sample_design_id"),
        "expected_sample_design_content_hash": provenance.get(
            "sample_design_content_hash"
        ),
    }


def _strategy_report_optional_score_evidence(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample_ref: Mapping[str, object],
    training_ref: Mapping[str, object] | None,
) -> tuple[object | None, dict[str, object] | None]:
    records, _total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        limit=1,
        unavailable_code=(
            "strategy_report_bundle_v2_optional_evidence_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_optional_evidence_invalid",
        label="score evidence",
    )
    if not records:
        return None, None
    newest = records[0]
    provenance = newest.get("provenance")
    if not isinstance(provenance, Mapping):
        _raise_corrupt_report_optional("score evidence")
    reference = {
        "evidence_artifact_id": newest.get("id"),
        "expected_evidence_artifact_content_hash": newest.get("content_hash"),
        "score_vector_artifact_id": provenance.get(
            "score_vector_artifact_id"
        ),
        "expected_score_vector_artifact_content_hash": provenance.get(
            "score_vector_artifact_content_hash"
        ),
    }
    try:
        binding = load_model_score_evidence_artifacts(
            read_runtime,
            task_id=task_id,
            **reference,
        )
    except (
        ModelingError,
        StrategyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        _raise_corrupt_report_optional("score evidence", cause=exc)
    bound_training_ref = build_training_evidence_ref(binding.training)
    if (
        bound_training_ref["sample_design_ref"] != dict(sample_ref)
        or (
            training_ref is not None
            and bound_training_ref != dict(training_ref)
        )
    ):
        return None, None
    return binding, {
        "evidence_artifact_id": binding.evidence_record["id"],
        "expected_evidence_artifact_content_hash": binding.evidence_record[
            "content_hash"
        ],
        "score_vector_artifact_id": binding.vector_record["id"],
        "expected_score_vector_artifact_content_hash": binding.vector_record[
            "content_hash"
        ],
    }


def _raise_corrupt_report_optional(
    label: str,
    *,
    cause: Exception | None = None,
) -> None:
    error = _StrategyV2EvidenceSetupError(
        "strategy_report_bundle_v2_optional_evidence_invalid",
        f"最新 {label} artifact 未通过完整认证；平台不会回退到旧证据。",
    )
    if cause is None:
        raise error
    raise error from cause


def _strategy_report_identity(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    strategy_type: str,
) -> dict[str, str] | None:
    repository = StrategyRepository(runtime.settings.db_path)
    try:
        matches = [
            item
            for item in repository.list_meta_for_task(task_id)
            if item.get("task_id") == task_id
            and item.get("strategy_type") == strategy_type
        ]
    except Exception:
        return None
    if len(matches) != 1:
        return None
    metadata = matches[0]
    strategy_id = metadata.get("id")
    version = metadata.get("version")
    if (
        not isinstance(strategy_id, str)
        or not strategy_id
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version <= 0
    ):
        return None
    try:
        strategy = repository.get_strategy(strategy_id)
        spec_hash = repository.get_strategy_spec_hash(strategy_id)
    except Exception:
        return None
    if (
        strategy is None
        or strategy.spec is None
        or strategy.strategy_type != strategy_type
        or not isinstance(spec_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", spec_hash) is None
    ):
        return None
    return {
        "strategy_id": strategy_id,
        "strategy_version": str(version),
        "strategy_type": strategy_type,
    }


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
        r"interactive-tree-frontier-group-selection-[0-9a-f]{32}",
        selection_id,
    ) is not None:
        return _interactive_tree_frontier_group_selection_artifact_slots(
            runtime,
            task_id=task_id,
            selection_id=selection_id,
        )
    if re.fullmatch(
        r"interactive-tree-frontier-selection-[0-9a-f]{32}",
        selection_id,
    ) is not None:
        return _interactive_tree_frontier_selection_artifact_slots(
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
    if re.fullmatch(
        r"scorecard-cutoff-selection-[0-9a-f]{32}", selection_id
    ) is not None:
        return _scorecard_cutoff_selection_artifact_slots(
            runtime,
            task_id=task_id,
            selection_id=selection_id,
        )
    raise StrategySetupError(
        "selection ID 格式无效；只支持完整 automatic-tree leaf selection、"
        "interactive-tree frontier group/singleton selection、"
        "Cross Matrix cell selection "
        "或 Scorecard cutoff selection ID。"
    )


def _interactive_tree_frontier_group_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Resolve one task-local frontier OR group to authenticated Pool slots."""

    if re.fullmatch(
        r"interactive-tree-frontier-group-selection-[0-9a-f]{32}",
        selection_id,
    ) is None:
        raise StrategySetupError(
            "interactive-tree frontier group selection ID 格式无效；"
            "请复制完整 selection ID。"
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
                    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
                    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
                ),
            ).fetchall()
            matches: list[tuple[object, Mapping]] = []
            for row in rows:
                provenance_json = row["provenance_json"]
                if not isinstance(provenance_json, str):
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier group "
                        "selection artifact provenance 无效。"
                    )
                try:
                    provenance = json.loads(provenance_json)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier group "
                        "selection artifact provenance 无效。"
                    ) from exc
                if not isinstance(provenance, Mapping):
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier group "
                        "selection artifact provenance 无效。"
                    )
                if provenance.get("selection_id") == selection_id:
                    matches.append((row, provenance))
            if not matches:
                raise StrategySetupError(
                    "当前任务没有 interactive-tree frontier group selection "
                    f"{selection_id}。"
                )
            if len(matches) != 1:
                raise StrategySetupError(
                    f"interactive-tree frontier group selection {selection_id} "
                    "对应多个 group selection artifact，当前不能安全绑定来源。"
                )
            row, provenance = matches[0]
            if (
                provenance.get("schema_version")
                != (
                    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION
                )
            ):
                raise StrategySetupError(
                    "interactive-tree frontier group selection artifact "
                    "schema 无效。"
                )
            semantic_tree_id = provenance.get("semantic_tree_id")
            tree_hash = provenance.get("tree_hash")
            artifact_id = row["id"]
            content_hash = row["content_hash"]
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or not isinstance(content_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                or not isinstance(semantic_tree_id, str)
                or not semantic_tree_id
                or not isinstance(tree_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", tree_hash) is None
            ):
                raise StrategySetupError(
                    "interactive-tree frontier group selection artifact "
                    "完整性绑定不完整。"
                )
            verified = (
                load_verified_interactive_tree_frontier_group_selection_artifact_on_connection(
                    conn,
                    runtime=SimpleNamespace(
                        settings=runtime.settings,
                        task_artifacts=repository,
                    ),
                    task_id=task_id,
                    artifact_id=artifact_id,
                    expected_content_hash=content_hash,
                    expected_asset_id=semantic_tree_id,
                    expected_asset_hash=tree_hash,
                )
            )
    except StrategySetupError:
        raise
    except Exception as exc:
        raise StrategySetupError(
            f"interactive-tree frontier group selection {selection_id} 未通过 "
            "selection、revision 父链与 artifact 完整性校验，不能加入 "
            "Strategy Pool。"
        ) from exc

    if verified.selection.get("selection_id") != selection_id:
        raise StrategySetupError(
            "interactive-tree frontier group selection ID 与认证 artifact "
            "不一致。"
        )
    revision = verified.revision
    ancestry = revision.ancestor_revisions
    try:
        fragment = (
            interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
                verified.selection,
                revision.revision,
                revision.automatic_source.asset,
                selection_artifact_binding=verified.artifact_binding(),
                revision_artifact_binding=revision.builder_binding(),
                parent_revision=ancestry[0] if ancestry else None,
                ancestor_revisions=ancestry[1:],
            )
        )
        pool_source, _rule_id, _execution = verified_fragment_pool_parts(
            fragment
        )
    except (StrategyError, TypeError, ValueError) as exc:
        raise StrategySetupError(
            "interactive-tree frontier group selection 未能从 live revision "
            "重放为受认证 OR fragment，不能加入 Strategy Pool。"
        ) from exc
    asset_id = pool_source.get("asset_id")
    asset_hash = pool_source.get("asset_hash")
    fragment_id = pool_source.get("fragment_id")
    if (
        asset_id != semantic_tree_id
        or asset_hash != tree_hash
        or not isinstance(fragment_id, str)
        or not fragment_id
    ):
        raise StrategySetupError(
            "interactive-tree frontier group selection 的 revision/fragment "
            "绑定与认证 artifact 不一致。"
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


def _interactive_tree_frontier_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Resolve one task-local frontier pointer to authenticated Pool slots."""

    if re.fullmatch(
        r"interactive-tree-frontier-selection-[0-9a-f]{32}",
        selection_id,
    ) is None:
        raise StrategySetupError(
            "interactive-tree frontier selection ID 格式无效；"
            "请复制完整 selection ID。"
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
                    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
                    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
                ),
            ).fetchall()
            matches: list[tuple[object, Mapping]] = []
            for row in rows:
                provenance_json = row["provenance_json"]
                if not isinstance(provenance_json, str):
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier selection "
                        "artifact provenance 无效。"
                    )
                try:
                    provenance = json.loads(provenance_json)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier selection "
                        "artifact provenance 无效。"
                    ) from exc
                if not isinstance(provenance, Mapping):
                    raise StrategySetupError(
                        "当前任务的 interactive-tree frontier selection "
                        "artifact provenance 无效。"
                    )
                if provenance.get("selection_id") == selection_id:
                    matches.append((row, provenance))
            if not matches:
                raise StrategySetupError(
                    "当前任务没有 interactive-tree frontier selection "
                    f"{selection_id}。"
                )
            if len(matches) != 1:
                raise StrategySetupError(
                    f"interactive-tree frontier selection {selection_id} "
                    "对应多个 frontier selection artifact，当前不能安全绑定来源。"
                )
            row, provenance = matches[0]
            if (
                provenance.get("schema_version")
                != INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
            ):
                raise StrategySetupError(
                    "interactive-tree frontier selection artifact schema 无效。"
                )
            semantic_tree_id = provenance.get("semantic_tree_id")
            tree_hash = provenance.get("tree_hash")
            artifact_id = row["id"]
            content_hash = row["content_hash"]
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or not isinstance(content_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", content_hash) is None
                or not isinstance(semantic_tree_id, str)
                or not semantic_tree_id
                or not isinstance(tree_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", tree_hash) is None
            ):
                raise StrategySetupError(
                    "interactive-tree frontier selection artifact "
                    "完整性绑定不完整。"
                )
            verified = (
                load_verified_interactive_tree_frontier_selection_artifact_on_connection(
                    conn,
                    runtime=SimpleNamespace(
                        settings=runtime.settings,
                        task_artifacts=repository,
                    ),
                    task_id=task_id,
                    artifact_id=artifact_id,
                    expected_content_hash=content_hash,
                    expected_asset_id=semantic_tree_id,
                    expected_asset_hash=tree_hash,
                )
            )
    except StrategySetupError:
        raise
    except Exception as exc:
        raise StrategySetupError(
            f"interactive-tree frontier selection {selection_id} 未通过 "
            "selection、revision 父链与 artifact 完整性校验，不能加入 "
            "Strategy Pool。"
        ) from exc

    selection = verified.selection
    revision = selection.get("revision")
    frontier = selection.get("frontier")
    if not isinstance(revision, Mapping) or not isinstance(frontier, Mapping):
        raise StrategySetupError(
            "interactive-tree frontier selection 缺少完整 revision/fragment 绑定。"
        )
    asset_id = revision.get("semantic_tree_id")
    asset_hash = revision.get("tree_hash")
    fragment_id = frontier.get("fragment_id")
    if (
        selection.get("selection_id") != selection_id
        or asset_id != semantic_tree_id
        or asset_hash != tree_hash
        or not isinstance(fragment_id, str)
        or not fragment_id
    ):
        raise StrategySetupError(
            "interactive-tree frontier selection 的 revision/fragment "
            "绑定与认证 artifact 不一致。"
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


def _scorecard_cutoff_selection_artifact_slots(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    selection_id: str,
) -> tuple[dict[str, str], str]:
    """Replay one authenticated Scorecard pointer back to its full band."""

    read_runtime = _strategy_report_read_runtime(runtime)
    artifacts = _scorecard_artifact_snapshot(read_runtime, task_id=task_id)
    registry_token = _scorecard_registry_token(artifacts)
    matches = []
    for artifact in artifacts:
        provenance = artifact.get("provenance")
        if (
            artifact.get("kind")
            == SCORECARD_CUTOFF_SELECTION_ARTIFACT_KIND
            and artifact.get("origin_tool")
            == SCORECARD_CUTOFF_SELECTION_ORIGIN_TOOL
            and isinstance(provenance, Mapping)
            and provenance.get("schema_version")
            == SCORECARD_CUTOFF_SELECTION_ARTIFACT_SCHEMA_VERSION
            and provenance.get("selection_id") == selection_id
        ):
            matches.append(artifact)
    if not matches:
        raise StrategySetupError(
            f"当前任务没有 Scorecard cutoff selection {selection_id}。"
        )
    if len(matches) != 1:
        raise StrategySetupError(
            f"Scorecard cutoff selection {selection_id} 对应多个 artifact，"
            "当前不能安全绑定来源。"
        )
    record = matches[0]
    provenance = record["provenance"]
    assert isinstance(provenance, Mapping)
    artifact_id = _scorecard_ref_hash(
        record.get("id"),
        field="selection_artifact_id",
    )
    content_hash = _scorecard_ref_hash(
        record.get("content_hash"),
        field="selection_artifact_content_hash",
    )
    selection_hash = _scorecard_ref_hash(
        provenance.get("selection_hash"),
        field="selection_hash",
    )
    try:
        verified = load_scorecard_cutoff_selection_artifact(
            read_runtime,
            task_id=task_id,
            artifact_id=artifact_id,
            expected_artifact_content_hash=content_hash,
            expected_selection_id=selection_id,
            expected_selection_hash=selection_hash,
        )
        source = verified.source_asset_binding
        fragment = scorecard_cutoff_selection_to_verified_candidate_fragment(
            verified.selection,
            source.asset,
            selection_artifact_binding=verified.to_domain_binding(),
            source_artifact_binding=source.to_domain_binding(),
        )
        pool_source, _rule_id, _execution = verified_fragment_pool_parts(
            fragment
        )
    except (
        StrategyError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise StrategySetupError(
            f"Scorecard cutoff selection {selection_id} 未通过 selection、"
            "完整 band、score evidence、SampleDesign 或 fragment 回放，"
            "不能加入 Strategy Pool。"
        ) from exc
    asset_id = source.asset.get("asset_id")
    asset_hash = source.asset.get("asset_hash")
    fragment_id = pool_source.get("fragment_id")
    if (
        verified.selection.get("selection_id") != selection_id
        or not isinstance(asset_id, str)
        or re.fullmatch(r"scorecard-band-asset-[0-9a-f]{32}", asset_id)
        is None
        or not isinstance(asset_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", asset_hash) is None
        or not isinstance(fragment_id, str)
        or not fragment_id
        or pool_source.get("asset_id") != asset_id
        or pool_source.get("asset_hash") != asset_hash
    ):
        raise StrategySetupError(
            "Scorecard cutoff selection 缺少一致的完整 band/fragment 绑定。"
        )
    refreshed = _scorecard_artifact_snapshot(read_runtime, task_id=task_id)
    if _scorecard_registry_token(refreshed) != registry_token:
        raise StrategySetupError(
            "Scorecard cutoff selection 或完整 band 在入池计划创建前"
            "发生变化；请基于最新 evidence 重试。"
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


def _strategy_impact_cube_dataset_preview(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
):
    """Expose compiler columns from the exact latest authenticated V2 sample."""

    read_runtime = _strategy_v2_read_runtime(runtime)
    try:
        artifacts = tuple(read_runtime.task_artifacts.list_for_task(task.id))
        sample = _latest_verified_strategy_sample_design_v2_binding(
            read_runtime,
            task_id=task.id,
            artifacts=artifacts,
        )
        target = sample.bundle["sample_design"]["target_selector"]["column"]
    except (
        KeyError,
        TypeError,
        _StrategyV2EvidenceSetupError,
        *_STRATEGY_V2_ARTIFACT_ERRORS,
    ) as exc:
        raise StrategySetupError(
            "ImpactCube 无法从最新 StrategySampleDesign V2 认证编译字段；"
            "请先重新固化样本设计。"
        ) from exc
    if (
        not isinstance(target, str)
        or not target
        or target not in sample.source_binding.columns
    ):
        raise StrategySetupError(
            "最新 StrategySampleDesign V2 的目标列绑定无效。"
        )
    return SimpleNamespace(
        dataset_id=sample.source_binding.dataset_id,
        columns=sample.source_binding.columns,
        target_col=target,
        identity={
            "kind": "strategy_sample_design_v2",
            "sample_design_ref": _strategy_report_sample_ref(sample),
            "dataset_id": sample.source_binding.dataset_id,
            "dataset_content_hash": sample.source_binding.dataset_content_hash,
        },
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
        and draft.workflow
        in {"strategy_sample_design", "strategy_sample_design_v2"}
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
        and payload.get("workflow")
        in {"strategy_sample_design", "strategy_sample_design_v2"}
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
    is_sample_design = payload.get("workflow") in {
        "strategy_sample_design",
        "strategy_sample_design_v2",
    }
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
        allow_legacy_replay=True,
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
        if draft.workflow == "univariate_candidate_refinement":
            return "source_candidate_id" not in draft.workflow_inputs
        if draft.workflow in {
            *_STRATEGY_POOL_WORKFLOWS,
            "strategy_project_context",
            "strategy_model_evidence_v2",
            "strategy_report_bundle_v2",
            "strategy_impact_cube",
            "strategy_pool_stability",
            "strategy_pool_apply",
            "strategy_pool_validation",
            "candidate_monthly_stability",
            "scorecard_band_build",
            "scorecard_cutoff_selection",
            "automatic_tree_leaf_materialization",
            "interactive_tree_revision",
            "interactive_tree_frontier_group_materialization",
            "interactive_tree_frontier_materialization",
            "cross_matrix_cell_selection",
            "voting_candidate_search",
            "voting_candidate_build_from_search",
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
        if draft.workflow == "strategy_project_context":
            return False
        refinement_needs_current_target = (
            draft.workflow == "univariate_candidate_refinement"
            and "source_candidate_id" not in draft.workflow_inputs
        )
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "strategy_sample_design_v2",
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
        if draft.workflow == "strategy_project_context":
            return False
        refinement_needs_current_labels = (
            draft.workflow == "univariate_candidate_refinement"
            and "source_candidate_id" not in draft.workflow_inputs
        )
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "strategy_sample_design_v2",
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
