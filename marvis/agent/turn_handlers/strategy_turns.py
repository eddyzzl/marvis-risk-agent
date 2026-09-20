"""Strategy turns for governed Agent turns."""

from __future__ import annotations

from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
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
from marvis.domain import TaskRecord
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
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=(
            f"开始策略分析:样本 `{proposal.dataset_name}`，目标列 `{proposal.target_col}`{bad}，"
            f"评分列 `{proposal.score_col}`。已生成默认审批策略候选，将自动完成回测和分析。"
            f"{note_text}{shared_lane._ingest_notice_text(notices)}"
        ),
        metadata={
            "intent": STRATEGY_INTENT_QUICK_ANALYSIS,
            "ingest_notices": notices,
        },
    )
    return (proposal.template_id, slots, {})


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


_STRATEGY_SPEC = contracts_lane._TurnHandlerSpec(
    intent="strategy",
    setup_error_types=(StrategySetupError,),
    error_label="策略分析出错",
    run_setup=_run_strategy_setup,
    format_user_display=typed_ui_lane._identity_display_text,
    success_criteria=_strategy_success_criteria,
)
