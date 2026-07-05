from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import json

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
from marvis.agent.plan_driver import DriverError, PlanDriver, is_confirm
from marvis.agent.portfolio_setup import (
    PortfolioProposal,
    PortfolioSetupError,
    build_portfolio_proposal,
    build_states_gate_state,
    parse_states_reply,
)
from marvis.agent.strategy_setup import (
    StrategySetupError,
    build_monitoring_setup_proposal,
    build_rule_strategy_proposal,
    build_strategy_proposal,
    is_rule_strategy_goal,
    is_strategy_monitoring_goal,
)
from marvis.agent.vintage_setup import VintageSetupError, build_vintage_proposal
from marvis.agent_memory.api_support import audit_agent_memory_use_from_store
from marvis.agent_memory.store import AgentMemoryStore
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_PORTFOLIO,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TaskRecord,
)
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.orchestrator.capability import auto_gate_budget, resolve_tier
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.validator import PlanValidator
from marvis.repositories.plans import PlanRepository
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
        return append_join_error(repo, task.id, str(exc))
    except DriverError:
        raise
    except Exception as exc:
        return append_join_error(repo, task.id, f"{spec.error_label}：{exc}")


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
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"分析数据集 `{proposal.dataset_name}`（目标列 `{proposal.target_col}`，"
            f"{len(proposal.features)} 个候选特征）:"
        ),
        metadata={"intent": "feature_analysis"},
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _strategy_success_criteria(task: TaskRecord) -> list[dict] | None:
    """S2: mirrors _modeling_success_criteria. task's optional
    strategy_bad_rate_max/strategy_approval_min (getattr-based -- no schema
    migration backs these fields today, so tasks without them simply inject no
    criterion, same graceful default as oot_ks_min) become deterministic
    approved_bad_rate/approval_rate thresholds final_review can evaluate."""
    bad_rate_max = getattr(task, "strategy_bad_rate_max", None)
    approval_min = getattr(task, "strategy_approval_min", None)
    criteria: list[dict] = []
    if bad_rate_max is not None:
        criteria.append({"metric": "approved_bad_rate", "max": float(bad_rate_max)})
    if approval_min is not None:
        criteria.append({"metric": "approval_rate", "min": float(approval_min)})
    return criteria or None


def _run_strategy_setup(
    runtime: DriverTurnRuntime, repo: TaskRepository, task: TaskRecord, user_text: str | None
) -> dict | tuple:
    backend, registry = _modeling_data_runtime(runtime.settings)
    # S5 intent branch (checked first, more specific than development/mining): a
    # monitoring goal -- in the first user message or the task's own name -- routes
    # to the strategy_monitoring template (run one monitoring pass for the task's
    # adopted strategy). Same S4 multi-recognize precedent as rule_strategy below.
    if is_strategy_monitoring_goal(user_text, getattr(task, "model_name", None)):
        return _run_strategy_monitoring_setup(runtime, repo, task, backend, registry)
    # S4 intent branch (parallel to strategy_development's goal_patterns): a rule
    # mining goal -- in the first user message or the task's own name -- routes to
    # the rule_strategy template instead of the default strategy_analysis.
    if is_rule_strategy_goal(user_text, getattr(task, "model_name", None)):
        return _run_rule_strategy_setup(runtime, repo, task, backend, registry)
    proposal = build_strategy_proposal(
        registry,
        backend,
        task.id,
        task.source_dir,
        target_col=getattr(task, "target_col", "") or None,
        score_col=getattr(task, "score_col", "") or None,
    )
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略分析:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"评分列 `{proposal.score_col}`。已生成默认审批策略候选，回测前会停下确认。"
            f"{note_text}"
        ),
        metadata={"intent": "strategy"},
    )
    return (proposal.template_id, proposal.template_slots(), {})


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
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始规则策略挖掘:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}。"
            f"将挖掘候选拒绝规则，选定规则集后回测并采纳。{note_text}"
        ),
        metadata={"intent": "strategy"},
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
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略监控:对已采纳策略 `{proposal.strategy_id}` 跑一次监控,样本 "
            f"`{proposal.dataset_name}`。监控完成后会在告警门停下确认。{note_text}"
        ),
        metadata={"intent": "strategy"},
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
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始 Vintage 风险分析:样本 `{proposal.dataset_name}`，"
            f"cohort `{proposal.cohort_col}`，MOB `{proposal.mob_col}`，坏账列 `{proposal.bad_col}`。"
        ),
        metadata={"intent": "vintage"},
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
            ),
            metadata={"portfolio_states": build_states_gate_state(proposal), "kind": "gate"},
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
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始建模:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"切分 `{proposal.split_col}` train/test/oot="
            f"{counts.get('train', 0)}/{counts.get('test', 0)}/{counts.get('oot', 0)}，"
            f"候选特征 {len(proposal.feature_cols)} 个。先做泄漏感知特征筛选，随后请确认特征集。"
            f"{note_text}"
        ),
        metadata={"intent": "modeling"},
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
) -> dict:
    # S6 ad-hoc 问数 branch (checked BEFORE the normal type dispatch). It is
    # deliberately defensive: it only ever handles a turn that is either (round B)
    # confirming/answering an already-pending 口径确认门, or (round A) a clear data
    # question on a task that has a ready dataset AND no active plan/open gate.
    # Anything else — including a non-confirm reply to a pending spec — returns
    # None and falls straight through to the task type's own handler, so this can
    # never hijack a normal instruction.
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
    )
    if agent_client is not None and auto_accept_enabled:
        agent_autodrive_turn(runtime, repo, task, client=agent_client)
        return join_turn_response(repo, task.id)
    return result


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
            turn_fn(runtime, repo, task, user_text="确认", expected_step_id=gate_step_id)
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
                    plan_id=plan_id, goal=goal, expected_step_id=gate_step_id
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
        metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
    )


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
