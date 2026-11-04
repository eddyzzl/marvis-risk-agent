"""Strategy turns for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pandas as pd

from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.agent.quick_replies import quick_reply, with_quick_replies
from marvis.agent.strategy_request_compiler.development import (
    resolve_development_score_direction,
    score_direction_conflict_message,
    score_direction_quick_commands,
    score_direction_summary,
    stated_development_score_direction,
    workspace_score_direction,
)
from marvis.agent.strategy_setup import STRATEGY_INTENT_FULL_DEVELOPMENT
from marvis.agent.strategy_setup import STRATEGY_INTENT_LIMIT_PRICING
from marvis.agent.strategy_setup import STRATEGY_INTENT_MONITORING
from marvis.agent.strategy_setup import STRATEGY_INTENT_PORTFOLIO_ANALYSIS
from marvis.agent.strategy_setup import STRATEGY_INTENT_QUICK_ANALYSIS
from marvis.agent.strategy_setup import STRATEGY_INTENT_RULE_MINING
from marvis.agent.strategy_setup import STRATEGY_INTENT_STANDARD_ANALYSIS
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.strategy_setup import build_monitoring_setup_proposal
from marvis.agent.strategy_setup import build_rule_strategy_proposal
from marvis.agent.strategy_setup import build_strategy_development_proposal
from marvis.agent.strategy_setup import build_strategy_proposal
from marvis.agent.strategy_setup import resolve_strategy_intent
from marvis.agent.strategy_setup import strategy_development_clarification
from marvis.data.direction import check_score_direction
from marvis.data.field_directions import confirmed_field_directions
from marvis.domain import TaskRecord
from marvis.packs.strategy.sample_design_v2_native_tools import (
    authenticate_native_strategy_sample_design_v2_bundle_record,
)
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository
from . import contracts as contracts_lane
from . import data_context as data_context_lane
from . import responses as responses_lane
from . import shared as shared_lane
from . import strategy_contracts as strategy_contracts_lane
from . import strategy_evidence as strategy_evidence_lane
from . import strategy_sample as strategy_sample_lane
from . import turn_runner as turn_runner_lane
from . import typed_ui as typed_ui_lane


def run_strategy_driver_turn(
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
        _STRATEGY_SPEC,
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
    runtime: contracts_lane.DriverTurnRuntime,
    repo: TaskRepository,
    task: TaskRecord,
    user_text: str | None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
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
        return responses_lane._strategy_intent_redirect_response(repo, task, intent)

    backend, registry = data_context_lane._modeling_data_runtime(runtime.settings)
    if intent == STRATEGY_INTENT_MONITORING:
        return _run_strategy_monitoring_setup(runtime, repo, task, backend, registry)
    raw_strategy_type = (
        getattr(strategy_input, "strategy_type", None)
        if not isinstance(strategy_input, dict)
        else strategy_input.get("strategy_type")
    )
    strategy_type = str(raw_strategy_type or "approval").strip().lower()
    if strategy_type not in {"approval", "reject"}:
        return responses_lane._strategy_clarification_response(
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
            return responses_lane._strategy_clarification_response(
                repo, task, clarification
            )
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
        slots = proposal.template_slots()
        context = strategy_evidence_lane._strategy_dataset_context(
            runtime, task, require_target=True
        )
        try:
            slots["sample_design_ref"] = (
                strategy_sample_lane._latest_matching_strategy_sample_design_ref(
                    runtime,
                    task,
                    context=context,
                    drop_nan_labels=False,
                    allow_native_risk_development=True,
                )
            )
        except strategy_contracts_lane._StrategySampleDesignRequiredError as exc:
            return responses_lane._strategy_request_clarification_response(
                repo,
                task,
                code="strategy_sample_design_required",
                message=str(exc),
                fields=strategy_contracts_lane._STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
                ingest_notices=notices,
            )
        direction = _development_score_direction(
            runtime,
            task,
            registry,
            backend,
            proposal=proposal,
            sample_design_ref=slots["sample_design_ref"],
            request_text=user_text,
        )
        if direction.get("status") != "resolved":
            return _score_direction_stop_response(
                repo, task, score_col=proposal.score_col, message=str(direction["message"])
            )
        slots["score_direction"] = direction["direction"]
        direction_text = score_direction_summary(
            proposal.score_col, str(direction["direction"]), str(direction["source"])
        )
        conflict = direction.get("conflict")
        if isinstance(conflict, Mapping):
            # Data disagrees with the authoritative direction: the plan waits at
            # its start gate with both directions shown. ``False`` marks the
            # conflict as unconfirmed; only a person's start sets it to True.
            slots["confirm_direction_conflict"] = False
            direction_text += score_direction_conflict_message(
                proposal.score_col,
                str(direction["direction"]),
                str(conflict["implied"]),
                corr=float(conflict["corr"]),
                labeled=int(conflict["n"]),
            )
        repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=(
                f"开始完整策略开发:样本 `{proposal.dataset_name}`，目标列 "
                f"`{proposal.target_col}`{bad}，评分列 `{proposal.score_col}`。"
                f"经营目标 `{proposal.objective}`，约束 {'；'.join(constraints)}。"
                f"{direction_text}"
                "尚未生成 cutoff 或默认规则；计划启动后会自动扫描、构造和回测可行方案，"
                "仅在采纳时交由人工决策。"
                f"{note_text}{shared_lane._ingest_notice_text(notices)}"
            ),
            metadata={
                "intent": STRATEGY_INTENT_FULL_DEVELOPMENT,
                "ingest_notices": notices,
            },
        )
        return (proposal.template_id, slots, {})

    if intent != STRATEGY_INTENT_QUICK_ANALYSIS:
        raise StrategySetupError(f"unsupported strategy intent: {intent}")
    proposal_kwargs = {
        "target_col": getattr(task, "target_col", "") or None,
        "score_col": getattr(task, "score_col", "") or None,
    }
    # This first build only resolves the columns; its data-implied direction is
    # never used as the declared direction (D-3).
    proposal = build_strategy_proposal(
        registry, backend, task.id, task.source_dir, **proposal_kwargs
    )
    notices = registry.consume_ingest_notices(task.id)
    context = strategy_evidence_lane._strategy_dataset_context(
        runtime, task, require_target=True
    )
    try:
        sample_design_ref = (
            strategy_sample_lane._latest_matching_strategy_sample_design_ref(
                runtime,
                task,
                context=context,
                drop_nan_labels=False,
                allow_native_risk_development=True,
            )
        )
    except strategy_contracts_lane._StrategySampleDesignRequiredError as exc:
        return responses_lane._strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
            fields=strategy_contracts_lane._STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
            ingest_notices=notices,
        )
    direction = _development_score_direction(
        runtime,
        task,
        registry,
        backend,
        proposal=proposal,
        sample_design_ref=sample_design_ref,
        request_text=user_text,
    )
    if direction.get("status") != "resolved":
        return _score_direction_stop_response(
            repo,
            task,
            score_col=proposal.score_col,
            message=str(direction["message"]),
            commands=_quick_analysis_direction_commands(proposal.score_col),
        )
    if proposal.template_slots()["score_direction"] != direction["direction"]:
        # The candidate rule must be cut with the authoritative direction.
        proposal = build_strategy_proposal(
            registry,
            backend,
            task.id,
            task.source_dir,
            score_direction=str(direction["direction"]),
            **proposal_kwargs,
        )
    note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
    bad = f"（坏率 {proposal.bad_rate:.2%}）" if proposal.bad_rate is not None else ""
    slots = proposal.template_slots()
    slots["sample_design_ref"] = sample_design_ref
    direction_text = score_direction_summary(
        proposal.score_col, str(direction["direction"]), str(direction["source"])
    )
    conflict = direction.get("conflict")
    if isinstance(conflict, Mapping):
        # Unconfirmed until a person starts the plan's overview gate.
        slots["confirm_direction_conflict"] = False
        direction_text += score_direction_conflict_message(
            proposal.score_col,
            str(direction["direction"]),
            str(conflict["implied"]),
            corr=float(conflict["corr"]),
            labeled=int(conflict["n"]),
        )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略分析:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"评分列 `{proposal.score_col}`。已生成默认审批策略候选，将自动完成回测和分析。"
            f"{direction_text}"
            f"{note_text}{shared_lane._ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_QUICK_ANALYSIS,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, slots, {})


def _quick_analysis_direction_commands(score_col: str) -> tuple[str, str]:
    return (
        f"快速策略分析，{score_col} 越高风险越高",
        f"快速策略分析，{score_col} 越低风险越高",
    )


def _run_rule_strategy_setup(
    runtime: contracts_lane.DriverTurnRuntime,
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
    slots = proposal.template_slots()
    context = strategy_evidence_lane._strategy_dataset_context(
        runtime, task, require_target=True
    )
    slots["sample_design_ref"] = (
        strategy_sample_lane._latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=False,
            allow_native_risk_development=True,
        )
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始规则策略挖掘:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}。"
            f"将自动挖掘、选择、评估并回测候选拒绝规则，仅在采纳时交由人工决策。{note_text}"
            f"{shared_lane._ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_RULE_MINING,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, slots, {})


def _run_strategy_monitoring_setup(
    runtime: contracts_lane.DriverTurnRuntime,
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
            f"{shared_lane._ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_MONITORING,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, proposal.template_slots(), {})


def _development_score_direction(
    runtime,
    task: TaskRecord,
    registry,
    backend,
    *,
    proposal,
    sample_design_ref: Mapping,
    request_text: str | None,
) -> dict:
    """D-3: request -> frozen sample contract -> workspace field direction.

    The sample's own data is read only to check the resolved direction.
    """

    score_col = proposal.score_col
    workspace_direction, non_monotonic = _workspace_field_direction(
        runtime, task, registry, score_col
    )
    decision = resolve_development_score_direction(
        score_col=score_col,
        request_direction=stated_development_score_direction(request_text, score_col),
        sample_contract_direction=_sample_contract_score_direction(
            runtime, task, sample_design_ref, score_col
        ),
        workspace_direction=workspace_direction,
        workspace_non_monotonic=non_monotonic,
    )
    if decision["status"] != "resolved":
        return decision
    try:
        frame = backend.read_frame(
            registry.resolve_path(proposal.dataset_id),
            columns=[proposal.target_col, score_col],
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise StrategySetupError("无法读取样本来核对分数方向，请确认数据工作区里的样本后重试。") from exc
    check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[proposal.target_col], errors="coerce").to_numpy(dtype=float),
        declared_direction=str(decision["direction"]),
    )
    if check.status == "conflict":
        decision = {
            **decision,
            "conflict": {"implied": check.implied_direction, "corr": check.corr, "n": check.n},
        }
    return decision


def _workspace_field_direction(runtime, task: TaskRecord, registry, score_col: str):
    """A user-confirmed V2.7 field direction for the exact active score column."""

    snapshot = DataWorkspaceRepository(runtime.settings.db_path).get_or_default(task.id)
    if not snapshot.active_dataset_id:
        return None, False
    try:
        dataset = registry.get(snapshot.active_dataset_id)
    except KeyError:
        return None, False
    for row in confirmed_field_directions(snapshot, dataset):
        if row["column"] == score_col and row["status"] == "confirmed":
            return workspace_score_direction(row["direction"]), row["direction"] == "non_monotonic"
    return None, False


def _sample_contract_score_direction(
    runtime, task: TaskRecord, sample_design_ref: Mapping, score_col: str
) -> str | None:
    """The historical-score direction frozen in a native sample contract."""

    if sample_design_ref.get("partition") != "risk/development":
        return None
    repository = TaskArtifactRepository(runtime.settings.db_path)
    record = next(
        (
            artifact
            for artifact in repository.list_for_task(task.id)
            if artifact.get("id") == sample_design_ref.get("artifact_id")
        ),
        None,
    )
    if record is None:
        return None
    authenticated = authenticate_native_strategy_sample_design_v2_bundle_record(
        SimpleNamespace(settings=runtime.settings, task_artifacts=repository),
        task_id=task.id,
        record=record,
    )
    historical = authenticated.bundle.get("historical_score") or {}
    if historical.get("status") != "available" or historical.get("column") != score_col:
        return None
    return {"higher_is_riskier": "higher_is_riskier", "lower_is_riskier": "higher_is_better"}.get(
        str(historical.get("direction"))
    )


def _score_direction_stop_response(
    repo: TaskRepository,
    task: TaskRecord,
    *,
    score_col: str,
    message: str,
    commands: tuple[str, ...] | None = None,
) -> dict:
    code = "strategy_score_direction_required"
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=message,
        metadata=with_quick_replies(
            {
                "intent": "strategy_request",
                "kind": "clarification",
                "code": code,
                "fields": ["score_direction"],
            },
            *(
                quick_reply(command)
                for command in (commands or score_direction_quick_commands(score_col))
            ),
        ),
    )
    return {
        "task_id": task.id,
        "status": "clarification_required",
        "code": code,
        "fields": ["score_direction"],
        "messages": repo.list_agent_messages(task.id),
    }


_STRATEGY_SPEC = contracts_lane._TurnHandlerSpec(
    intent="strategy",
    setup_error_types=(StrategySetupError,),
    error_label="策略分析出错",
    run_setup=_run_strategy_setup,
    format_user_display=typed_ui_lane._identity_display_text,
    success_criteria=_strategy_success_criteria,
)
