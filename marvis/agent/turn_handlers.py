from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import json
import re

from marvis.agent.adhoc_analysis import (
    build_slice_spec_from_utterance,
    detect_question_intent,
)
from marvis.agent.auto_drive import decide_gate
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
    StrategyRequestDraft,
    compile_strategy_request,
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
from marvis.data.registry import DatasetRegistry
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
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.orchestrator.capability import auto_gate_budget, resolve_tier
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.validator import PlanValidator
from marvis.repositories.plans import PlanRepository
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestConflictError,
    PendingStrategyRequestNotFoundError,
    PendingStrategyRequestRepository,
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
    run_setup: Callable[[DriverTurnRuntime, TaskRepository, TaskRecord, str | None], dict | tuple]
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
    driver = _driver(runtime)
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


def _append_spec_messages(
    spec: _TurnHandlerSpec, repo: TaskRepository, task: TaskRecord, turn, runtime: DriverTurnRuntime
) -> None:
    if spec.pass_memory_kwargs:
        append_driver_messages(repo, task.id, turn, settings=runtime.settings, task=task)
    else:
        append_driver_messages(repo, task.id, turn)


def _c1_display_text(user_text: str) -> str:
    return "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text


def _identity_display_text(user_text: str) -> str:
    return user_text


def _run_join_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
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
        return append_join_error(repo, task.id, "请先指定样本锚表（通常是含目标列的那张），再确认。")
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
        {"anchor_id": assignment["anchor_id"], "feature_ids": assignment["feature_ids"]},
        {},
    )


def _run_feature_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
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
        bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
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
                "尚未生成 cutoff 或默认规则，确认计划后才会按这些口径扫描可行方案。"
                f"{note_text}{_ingest_notice_text(notices)}"
            ),
            metadata={
                "intent": STRATEGY_INTENT_FULL_DEVELOPMENT,
                "ingest_notices": notices,
            },
        )
        return (proposal.template_id, proposal.template_slots(), {})

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
            f"评分列 `{proposal.score_col}`。已生成默认审批策略候选，回测前会停下确认。"
            f"{note_text}{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_QUICK_ANALYSIS,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _strategy_intent_redirect_response(
    repo: TaskRepository, task: TaskRecord, intent: str
) -> dict:
    if intent == STRATEGY_INTENT_LIMIT_PRICING:
        detail = {
            "intent": intent,
            "code": "strategy_limit_pricing_workflow_planned",
            "planned_v2_workflow": "limit_pricing_matrix",
            "message": (
                "已识别为额度定价/定价矩阵意图。额度定价执行 Workflow 已纳入 V2 后续"
                "交付批次，尚未接入本阶段标准入口；当前不会把它误建为额度准入 approval plan。"
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
        if key in {"planned_v2_workflow", "suggested_task_type"}
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
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, backend, registry
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
            f"将挖掘候选拒绝规则，选定规则集后回测并采纳。{note_text}"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_RULE_MINING,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _run_strategy_monitoring_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, backend, registry
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
            f"开始策略监控:对已采纳策略 `{proposal.strategy_id}` 跑一次监控,样本 "
            f"`{proposal.dataset_name}`。监控完成后会在告警门停下确认。{note_text}"
            f"{_ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_MONITORING,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _run_vintage_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
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
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
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
        content=f"已确认桶顺序：{' → '.join(states)}。开始并行分析（流量/迁徙/细分" + ("/趋势" if proposal.experiment_id else "") + "），随后汇总确认。",
        metadata={"intent": "portfolio"},
    )
    return (proposal.template_id, proposal.template_slots(states), {})


def _run_modeling_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
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
            return append_join_error(repo, task.id, "请先指定建模样本主表（通常是含目标列的那张），再确认。")
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
# The compiler is an intent-to-contract boundary, not an execution runtime.  A
# validated draft is stored on the confirmation message; only the next explicit
# confirmation may instantiate a trusted Workflow.  The last-assistant anchor
# makes the draft single-use and prevents an older request from being replayed.
_STRATEGY_REQUEST_META_KEY = "strategy_request"
_STRATEGY_REQUEST_ACTION_RE = re.compile(
    r"(?:开发|设计|制定|创建|生成|分析|评估|查看|看一下|看下|回测|测试|应用|执行|打标|"
    r"对比|比较|采纳|采用|上线|报告|文档|监控|漂移|挖掘|"
    r"develop|design|create|analy[sz]e|evaluate|backtest|apply|compare|"
    r"adopt|report|monitor|mine)",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_SUBJECT_RE = re.compile(
    r"(?:策略|准入|审批|拒绝|额度|授信|定价|利率|分群|分层|规则|cutoff|"
    r"strategy|approval|reject|limit|pricing|segment|rule)",
    re.IGNORECASE,
)
_STRATEGY_REQUEST_CANCEL_RE = re.compile(
    r"(?:先别|不要|不用|不执行|先不|暂不|暂停|停止|取消|"
    r"do\s*not|don't|dont|stop|cancel|wait)",
    re.IGNORECASE,
)
_TYPED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest"})
_STORED_EVALUATION_OPERATIONS = frozenset({"analyze", "backtest", "compare"})


def _maybe_handle_strategy_request_turn(
    runtime: DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    *,
    user_text: str | None,
) -> dict | None:
    """Compile and confirm a natural-language strategy request when safe.

    Structured/manual setup remains intact.  This branch is only available on a
    strategy task with an LLM, no active plan/open driver gate, and either a
    pending compiled draft or a clear strategy action utterance.
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
                expected_payload_sha256=str(
                    pending.get("payload_sha256") or ""
                ),
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

    if not (
        _STRATEGY_REQUEST_ACTION_RE.search(text)
        and _STRATEGY_REQUEST_SUBJECT_RE.search(text)
    ):
        return None
    if _active_plan(runtime.plan_repo, task.id) is not None:
        return None
    if latest_open_gate(conversation) is not None:
        return None

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content=text,
        metadata={"intent": "strategy_request"},
    )
    preview = None
    preview_error = None
    try:
        preview = _strategy_dataset_preview(runtime, task)
    except StrategySetupError as exc:
        preview_error = str(exc)

    compilation = compile_strategy_request(
        text,
        allowed_columns=_strategy_request_allowed_columns(preview),
        llm=runtime.llm_client,
    )
    if compilation.draft is None:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_request_needs_clarification",
            message=compilation.clarification
            or "请补充策略操作、策略类型和业务口径。",
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
    if _strategy_request_requires_target(compilation.draft) and not preview.target_col:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
        )

    bound_preview = preview if requires_dataset else None
    pending_record = PendingStrategyRequestRepository(
        runtime.settings.db_path
    ).create(
        task_id=task.id,
        validated_draft=compilation.draft.to_dict(),
        dataset_identity=(
            None if bound_preview is None else bound_preview.identity
        ),
        target_col=(None if bound_preview is None else bound_preview.target_col),
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=compilation.confirmation or "请确认策略执行草案。",
        metadata={
            _STRATEGY_REQUEST_META_KEY: pending_record.to_metadata_reference(),
            "intent": "strategy_request_confirmation",
            "kind": "confirmation",
        },
    )
    return join_turn_response(repo, task.id)


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
    if _strategy_request_requires_target(draft) and not preview.target_col:
        _invalidate_pending_strategy_request(runtime, task, pending)
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_target_context_required",
            message="当前策略操作需要明确的二元目标列，请先在任务中指定 target_col。",
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
            "intent": "strategy_request_consumed",
            "request_id": request_id,
            "payload_sha256": payload_sha256,
        },
    )

    try:
        if draft.strategy_spec is None and draft.operation == "report":
            return _start_confirmed_strategy_plan(
                runtime,
                repo,
                task,
                template_id="stored_strategy_report",
                slots={"strategy_id": draft.strategy_id},
            )
        context = _strategy_dataset_context(
            runtime,
            task,
            require_target=_strategy_request_requires_target(draft),
        )
        if (
            tuple(context.columns) != tuple(preview.columns)
            or context.target_col != preview.target_col
        ):
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_dataset_context_changed",
                message="数据注册后的列或目标口径与确认前预览不一致，请重新确认。",
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
            return _start_confirmed_strategy_plan(
                runtime,
                repo,
                task,
                template_id=template_id,
                slots=slots,
                success_criteria=_strategy_request_success_criteria(draft),
            )

        if draft.operation in _STORED_EVALUATION_OPERATIONS:
            return _start_confirmed_strategy_plan(
                runtime,
                repo,
                task,
                template_id="stored_strategy_evaluation",
                slots=_stored_strategy_slots(context, draft),
                success_criteria=_strategy_request_success_criteria(draft),
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
            )
        if draft.operation == "adopt":
            return _start_confirmed_strategy_plan(
                runtime,
                repo,
                task,
                template_id="stored_strategy_adoption",
                slots=_stored_strategy_slots(context, draft),
                success_criteria=_strategy_request_success_criteria(draft),
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
            raise StrategySetupError(
                f"strategy operation is not wired: {draft.operation}"
            )
        if isinstance(setup, dict):
            return setup
        template_id, slots, start_kwargs = setup
        if "success_criteria" not in start_kwargs:
            criteria = _strategy_request_success_criteria(draft)
            if criteria:
                start_kwargs = {**start_kwargs, "success_criteria": criteria}
        return _start_confirmed_strategy_plan(
            runtime,
            repo,
            task,
            template_id=template_id,
            slots=slots,
            **start_kwargs,
        )
    except StrategySetupError as exc:
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
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
) -> dict:
    """Start a compiled Workflow and show its exact overview before execution."""

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
    return join_turn_response(repo, task.id)


def _strategy_request_preflight(
    runtime: DriverTurnRuntime,
    task: TaskRecord,
    draft: StrategyRequestDraft,
) -> tuple[str, str] | None:
    """Reject any compiled request whose trusted Workflow is not wired yet."""

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
            if meta.get("status") == "adopted"
        ]
        if not adopted:
            return (
                "strategy_adopted_version_required",
                "当前任务没有已采纳策略，请先完成回测和人工采纳再启动监控。",
            )
        selected = adopted[-1]
        if draft.strategy_id and draft.strategy_id != selected.get("id"):
            return (
                "strategy_monitor_target_mismatch",
                "当前监控入口只会执行任务内最新采纳策略；请求中的策略 ID 与其不一致。",
            )
        if draft.strategy_type != selected.get("strategy_type"):
            return (
                "strategy_monitor_type_mismatch",
                "请求中的策略类型与任务内最新采纳策略不一致，请重新确认监控对象。",
            )
        return None
    return (
        "strategy_operation_not_wired",
        f"已识别 {draft.operation} 请求，但对应受信任 Workflow 尚未接线；"
        "当前不会把它降级成其他策略操作。",
    )


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
                "只有 draft 状态的策略可以进入采纳流程。",
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


def _strategy_dataset_preview(runtime: DriverTurnRuntime, task: TaskRecord):
    backend, registry = _modeling_data_runtime(runtime.settings)
    return preview_strategy_dataset_context(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
    )


def _strategy_request_allowed_columns(preview) -> tuple[str, ...]:
    if preview is None:
        return ()
    # The observed target is evidence, never a deployable strategy feature or
    # an input to an LLM-authored profit contract.
    return tuple(
        column for column in preview.columns if column != preview.target_col
    )


def _strategy_request_requires_dataset(draft: StrategyRequestDraft) -> bool:
    return not (draft.strategy_spec is None and draft.operation == "report")


def _strategy_request_requires_target(draft: StrategyRequestDraft) -> bool:
    return draft.operation not in {"apply", "report"}


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
        criteria.append(
            {"metric": "approved_bad_rate", "max": draft.max_bad_rate}
        )
    if draft.min_approval_rate is not None:
        criteria.append(
            {"metric": "approval_rate", "min": draft.min_approval_rate}
        )
    return criteria or None


def _strategy_request_clarification_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    code: str,
    message: str,
) -> dict:
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata={
            "intent": "strategy_request",
            "kind": "clarification",
            "code": code,
        },
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": code,
        "messages": repo.list_agent_messages(task.id),
    }


def _latest_strategy_request_pending(conversation: list[dict]) -> dict | None:
    last_assistant = next(
        (message for message in reversed(conversation) if message.get("role") == "assistant"),
        None,
    )
    if last_assistant is None:
        return None
    pending = (last_assistant.get("metadata") or {}).get(
        _STRATEGY_REQUEST_META_KEY
    )
    return pending if isinstance(pending, dict) else None


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
                task.id, role="user", stage="chat",
                content=user_text or "", metadata={"intent": "adhoc_query"},
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
    result = build_slice_spec_from_utterance(user_text or "", columns, runtime.llm_client)
    repo.add_agent_message(
        task.id, role="user", stage="chat",
        content=user_text or "", metadata={"intent": "adhoc_query"},
    )
    if result.needs_clarification:
        # A Chinese clarification (never a guess, INV-1). No pending state is
        # stored — the user simply rephrases and round A runs again.
        repo.add_agent_message(
            task.id, role="assistant", stage="chat",
            content=result.clarify or "没能理解这个问题，请换一种说法。",
            metadata={"intent": "adhoc_query"},
        )
        return join_turn_response(repo, task.id)
    # A validated spec: show the 口径确认门 and stash the exact tool inputs on it.
    repo.add_agent_message(
        task.id, role="assistant", stage="chat",
        content=result.confirmation_text or "",
        metadata={_ADHOC_SPEC_META_KEY: result.spec.tool_inputs(dataset_id)},
    )
    return join_turn_response(repo, task.id)


def _run_adhoc_slice_plan(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, tool_inputs: dict
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
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _ADHOC_DATA_ROLES]
    if not datasets:
        return None
    dataset = sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
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
                gate_metadata=gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {},
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
        gate_meta = gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {}
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
            params = decision.get("params") if isinstance(decision.get("params"), dict) else None
            selection = decision.get("selection") if isinstance(decision.get("selection"), list) else None
            dedup = decision.get("dedup_strategies") if isinstance(decision.get("dedup_strategies"), dict) else None
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
            append_driver_messages(repo, task.id, turn, settings=getattr(runtime, "settings", None), task=task)
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
    return {"task_id": task_id, "status": "ok", "messages": repo.list_agent_messages(task_id)}


def append_join_error(repo: TaskRepository, task_id: str, detail: str) -> dict:
    repo.add_agent_message(task_id, role="assistant", stage="chat", content=detail, metadata={"error": True})
    return {"task_id": task_id, "status": "error", "messages": repo.list_agent_messages(task_id)}


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
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
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
            + (f"，目标列 = `{proposal.target_col}`" if proposal.target_col else "（未识别目标列，请指定）")
            + "。只有一张表，确认后将跳过拼接。请确认，或用下方控件调整。"
        )
    else:
        text = (
            f"我发现 {len(files)} 个数据文件，先确认每张的**角色与目标列**（样本是锚，只贴列不改行，**1:1**）:\n"
            f"- 提议**样本主表** = `{anchor.name if anchor else '?'}`"
            + (f"（目标列 `{proposal.target_col}`）" if proposal.target_col else "（未识别目标列，请指定）")
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
            payload = json.loads(text[len("[C1]"):])
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
            fid for fid in (payload.get("feature_ids") or []) if fid and fid != anchor_id
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
    return [str(item).strip() for item in (getattr(task, "metrics", None) or []) if str(item).strip()]


def _modeling_recipes(task: TaskRecord) -> list[str] | None:
    recipes = [str(item).strip() for item in (getattr(task, "recipes", None) or []) if str(item).strip()]
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
    values.extend(getattr(item, "name", None) for item in getattr(c1_proposal, "files", None) or ())
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value or "").strip())
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
