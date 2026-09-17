"""strategy_turns driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
from marvis.agent.plan_driver import CONFIRMATION_SOURCE_HUMAN
from marvis.agent.strategy_setup import STRATEGY_INTENT_FULL_DEVELOPMENT, STRATEGY_INTENT_LIMIT_PRICING, STRATEGY_INTENT_MONITORING, STRATEGY_INTENT_PORTFOLIO_ANALYSIS, STRATEGY_INTENT_QUICK_ANALYSIS, STRATEGY_INTENT_RULE_MINING, STRATEGY_INTENT_STANDARD_ANALYSIS, StrategySetupError, build_monitoring_setup_proposal, build_rule_strategy_proposal, build_strategy_dataset_context, build_strategy_development_proposal, build_strategy_proposal, preview_strategy_dataset_context, resolve_strategy_intent, strategy_development_clarification
from marvis.agent.strategy_request_compiler import CompiledStrategyRequestDraft, StandardWorkflowRequestDraft, StrategyRequestDraft
from marvis.agent.strategy_workflows import migrated_workflow_requirements
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.data.labels import nan_label_mask
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TASK_TYPE_PORTFOLIO, StrategyProfitInput, StrategyTaskInput, TaskRecord
from marvis.strategy_lifecycle import ASSET_STATUS_ADOPTED_LOCAL
from marvis.files import sha256_file
from marvis.packs.strategy.voting_candidate import VOTING_CANDIDATE_ASSET_TYPE
from marvis.packs.strategy.voting_candidate_search_tools import VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND, load_historical_voting_candidate_search_artifact
from marvis.packs.strategy.cross_candidate_search_tools import CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND, load_cross_candidate_search_artifact
from marvis.packs.strategy.cross_rule_search_tools import CROSS_RULE_SEARCH_ARTIFACT_KIND, load_cross_rule_search_artifact
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.dsl import strategy_spec_hash
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND
from marvis.packs.modeling.evidence_tools import build_training_evidence_ref, load_modeling_training_evidence_artifacts
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.score_evidence import MODEL_SCORE_EVIDENCE_ARTIFACT_KIND
from marvis.packs.modeling.score_evidence_tools import load_model_score_evidence_artifacts
from marvis.packs.strategy.model_evidence_tools import MODEL_EVIDENCE_V2_ARTIFACT_KIND, derive_strategy_model_evidence_candidate_execution_ref, load_strategy_model_evidence_v2_artifact, strategy_model_evidence_registry_snapshot_token
from marvis.packs.strategy.impact_cube_tools import IMPACT_CUBE_ARTIFACT_KIND
from marvis.packs.strategy.pool_impact_tools import POOL_IMPACT_ARTIFACT_KIND, load_historical_strategy_pool_impact_artifact
from marvis.packs.strategy.pool_stability_tools import POOL_STABILITY_ARTIFACT_KIND, load_strategy_pool_stability_artifact
from marvis.packs.strategy.pool_tools import bind_strategy_pool_development_execution, load_current_strategy_candidate_pool_artifact
from marvis.packs.strategy.pool_requirement_resolver import pool_requirement_bindings_provenance, project_pool_entry_requirements, resolve_pool_requirements
from marvis.packs.strategy.report_bundle_adapters import validate_candidate_stability_report_compatibility, validate_cross_candidate_search_report_compatibility, validate_cross_rule_search_report_compatibility
from marvis.packs.strategy.report_bundle_tools import authenticate_strategy_report_identity_for_pool_on_connection, load_strategy_impact_cube_artifact
from marvis.packs.strategy.sample_design_v2_tools import SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND, SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND, load_any_strategy_sample_design_v2_artifacts
from marvis.packs.strategy.sample_design_v2_native_tools import SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
from marvis.repositories.task_artifacts import TaskArtifactConflictError, TaskArtifactDataError, TaskArtifactNotFoundError, TaskArtifactRepository
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository, strategy_pool_snapshot_hash
from marvis.packs.strategy.candidate_stability_tools import ARTIFACT_KIND as CANDIDATE_STABILITY_ARTIFACT_KIND, load_candidate_stability_artifact

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import DriverTurnRuntime
    from . import _STORED_EVALUATION_OPERATIONS
    from . import _STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT
    from . import _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
    from . import _STRATEGY_REPORT_POOL_COMMAND_RE
    from . import _STRATEGY_REPORT_POOL_HISTORY_RE
    from . import _STRATEGY_REPORT_POOL_SELECTOR_RE
    from . import _STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT
    from . import _STRATEGY_REPORT_POOL_TITLE_RE
    from . import _STRATEGY_REPORT_POOL_TYPE_NEGATION_RE
    from . import _STRATEGY_REPORT_POOL_TYPE_PATTERNS
    from . import _STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT
    from . import _STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS
    from . import _TYPED_EVALUATION_OPERATIONS
    from . import _TurnHandlerSpec
    from . import _append_strategy_nan_label_clarification
    from . import _identity_display_text
    from . import _ingest_notice_text
    from . import _latest_matching_strategy_sample_design_ref
    from . import _latest_verified_strategy_sample_design_v2_binding
    from . import _modeling_data_runtime
    from . import _raise_corrupt_report_optional
    from . import _require_strategy_pool_impact_workspace
    from . import _run_driver_turn
    from . import _standard_workflow_request_preflight
    from . import _stored_strategy_request_preflight
    from . import _strategy_sample_design_dataset_preview

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
    expected_plan_id: str | None = None,
    expected_plan_status: str | None = None,
    expected_plan_revision: int | None = None,
    expected_plan_fingerprint: str | None = None,
    expected_step_fingerprint: str | None = None,
    confirmation_source: str = CONFIRMATION_SOURCE_HUMAN,
    ui_action: str | None = None,
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
    runtime: DriverTurnRuntime,
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
        slots = proposal.template_slots()
        context = _strategy_dataset_context(runtime, task, require_target=True)
        try:
            slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
                runtime,
                task,
                context=context,
                drop_nan_labels=False,
                allow_native_risk_development=True,
            )
        except _StrategySampleDesignRequiredError as exc:
            return _strategy_request_clarification_response(
                repo,
                task,
                code="strategy_sample_design_required",
                message=str(exc),
                fields=_STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
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
                f"{note_text}{_ingest_notice_text(notices)}"
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
    context = _strategy_dataset_context(runtime, task, require_target=True)
    try:
        slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
            runtime,
            task,
            context=context,
            drop_nan_labels=False,
            allow_native_risk_development=True,
        )
    except _StrategySampleDesignRequiredError as exc:
        return _strategy_request_clarification_response(
            repo,
            task,
            code="strategy_sample_design_required",
            message=str(exc),
            fields=_STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS,
            ingest_notices=notices,
        )
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
            "code": "strategy_portfolio_entry_required",
            "capability_status": "available",
            "suggested_task_type": TASK_TYPE_PORTFOLIO,
            "message": (
                "已识别为组合分析意图。组合分析已有独立正式入口；当前策略任务不会误建 "
                "approval plan。请新建或切换到「组合分析」任务，绑定组合样本后继续。"
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
            "capability_status",
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
    slots = proposal.template_slots()
    context = _strategy_dataset_context(runtime, task, require_target=True)
    slots["sample_design_ref"] = _latest_matching_strategy_sample_design_ref(
        runtime,
        task,
        context=context,
        drop_nan_labels=False,
        allow_native_risk_development=True,
    )
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

_STRATEGY_SPEC = _TurnHandlerSpec(
    intent="strategy",
    setup_error_types=(StrategySetupError,),
    error_label="策略分析出错",
    run_setup=_run_strategy_setup,
    format_user_display=_identity_display_text,
    success_criteria=_strategy_success_criteria,
)

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
    r"对比|比较|搜索|查找|检索|枚举|采纳|采用|上线|报告|文档|监控|漂移|挖掘|选择|筛选|保留|合并|编辑|"
    r"添加|加入|入池|删除|移除|排序|重排|改为|编译|预览|"
    r"develop|design|create|build|train|materialize|aggregate|collect|compute|calculate|analy[sz]e|evaluate|backtest|validate|replay|run|apply|compare|"
    r"search|find|enumerate|screen|adopt|report|monitor|mine|refine|select|merge|add|remove|delete|reorder|compile|preview)",
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

class _StrategySampleDesignPolicyMismatchError(StrategySetupError):
    """A valid newest sample exists, but the probed null-label policy differs."""

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

def _strategy_v2_read_runtime(runtime: DriverTurnRuntime) -> SimpleNamespace:
    backend, registry = _modeling_data_runtime(runtime.settings)
    return SimpleNamespace(
        settings=runtime.settings,
        backend=backend,
        registry=registry,
        task_artifacts=TaskArtifactRepository(runtime.settings.db_path),
    )

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

    try:
        return strategy_model_evidence_registry_snapshot_token(artifacts)
    except StrategyError as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_model_evidence_v2_registry_unavailable",
            "Strategy ModelEvidence V2 registry snapshot 无法规范化。",
        ) from exc

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
        return load_any_strategy_sample_design_v2_artifacts(
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
        execution_sample_ref = (
            derive_strategy_model_evidence_candidate_execution_ref(sample)
        )
        if (
            _strategy_report_sample_ref(sample) != dict(sample_ref)
            or current_development.sample_design.to_ref_dict()
            != execution_sample_ref
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
            pool=pool,
            current_development=current_development,
            execution_sample_ref=execution_sample_ref,
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
    pool,
    current_development,
    execution_sample_ref: Mapping[str, object],
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
    if historical_development.sample_design.to_ref_dict() != dict(
        execution_sample_ref
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

def _strategy_report_latest_cross_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
):
    """Select the newest fully authenticated Cross search for this V2 sample."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_cross_candidate_search_"
            "registry_unavailable"
        ),
        invalid_code=(
            "strategy_report_bundle_v2_cross_candidate_search_invalid"
        ),
        label="Cross candidate search",
    )
    for item in records:
        try:
            binding = load_cross_candidate_search_artifact(
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
                "strategy_report_bundle_v2_cross_candidate_search_invalid",
                "最新待判定的 Cross 候选搜索 artifact 未通过文件、registry、"
                "provenance 或样本绑定复核；其真实身份无法确认，平台不会"
                "回退到旧搜索证据。",
            ) from exc
        if _strategy_report_cross_search_matches(
            binding,
            sample=sample,
        ):
            return binding
    if total > _STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_cross_candidate_search_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT} 个 Cross 候选搜索 "
            "artifact，但 registry 仍有更早记录；平台无法证明窗口外"
            "不存在与当前 SampleDesign 完全一致的搜索证据，本次未创建"
            "报告计划。",
        )
    return None

def _strategy_report_cross_search_matches(
    binding,
    *,
    sample,
) -> bool:
    try:
        validate_cross_candidate_search_report_compatibility(
            cross_candidate_search=binding,
            sample_design=sample,
        )
    except StrategyError:
        return False
    return True

def _strategy_report_latest_cross_rule_search_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    sample,
):
    """Select the newest authenticated Cross rule search for this V2 sample."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=CROSS_RULE_SEARCH_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_cross_rule_search_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_cross_rule_search_invalid",
        label="Cross rule search",
    )
    for item in records:
        try:
            binding = load_cross_rule_search_artifact(
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
                "strategy_report_bundle_v2_cross_rule_search_invalid",
                "最新待判定的 Cross 阈值规则搜索 artifact 未通过文件、"
                "registry、provenance 或样本绑定复核；平台不会回退到旧证据。",
            ) from exc
        try:
            validate_cross_rule_search_report_compatibility(
                cross_rule_search=binding,
                sample_design=sample,
            )
        except StrategyError:
            continue
        return binding
    if total > _STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_cross_rule_search_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT} 个 Cross 阈值"
            "规则搜索 artifact，但 registry 仍有更早记录；平台无法证明"
            "窗口外不存在兼容证据，本次未创建报告计划。",
        )
    return None

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

def _strategy_report_latest_pool_stability_binding(
    read_runtime: SimpleNamespace,
    *,
    task_id: str,
    impact_cube_ref: Mapping[str, object],
):
    """Select the newest authenticated stability for the exact report cube."""

    records, total = _strategy_report_artifact_window(
        read_runtime,
        task_id=task_id,
        kind=POOL_STABILITY_ARTIFACT_KIND,
        limit=_STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT,
        unavailable_code=(
            "strategy_report_bundle_v2_pool_stability_registry_unavailable"
        ),
        invalid_code="strategy_report_bundle_v2_pool_stability_invalid",
        label="PoolStability",
    )
    for item in records:
        provenance = item.get("provenance")
        try:
            binding = load_strategy_pool_stability_artifact(
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
            source_ref = binding.stability["source_bindings"]["impact_cube"]
        except (
            KeyError,
            StrategyError,
            TypeError,
            ValueError,
            *_STRATEGY_V2_ARTIFACT_ERRORS,
        ) as exc:
            raise _StrategyV2EvidenceSetupError(
                "strategy_report_bundle_v2_pool_stability_invalid",
                "最新待判定的 PoolStability artifact 未通过文件、registry、"
                "provenance、producer-run、唯一 audit 或 embedded "
                "ImpactCube 复核；其真实来源无法确认，平台不会回退到旧"
                "稳定性证据。",
            ) from exc
        if source_ref == dict(impact_cube_ref):
            return binding
    if total > _STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_pool_stability_"
            "selection_window_exhausted",
            "已完整认证最新 "
            f"{_STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT} 个 "
            "PoolStability artifact，但 registry 仍有更早记录；平台无法"
            "证明窗口外不存在与当前 exact ImpactCube 一致的稳定性证据，"
            "本次未创建报告计划。",
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
        not in {
            SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
            SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
        }
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

def _strategy_report_identity(
    runtime: DriverTurnRuntime,
    *,
    task_id: str,
    candidate_pool,
) -> dict[str, str] | None:
    repository = StrategyRepository(runtime.settings.db_path)
    try:
        with repository.transaction() as conn:
            authenticated = (
                authenticate_strategy_report_identity_for_pool_on_connection(
                    repository,
                    conn,
                    task_id=task_id,
                    candidate_pool=candidate_pool,
                )
            )
    except Exception as exc:
        raise _StrategyV2EvidenceSetupError(
            "strategy_report_bundle_v2_strategy_identity_invalid",
            "当前 Strategy Pool 的物化策略身份、不可变账本或生命周期"
            "未通过完整性复核；本次未创建计划。",
        ) from exc
    return (
        None
        if authenticated is None
        else dict(authenticated["identity"])
    )

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
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[0]
        if draft.workflow in {
            *_STRATEGY_POOL_WORKFLOWS,
            "strategy_project_context",
            "strategy_model_evidence_v2",
            "strategy_report_bundle_v2",
            "strategy_impact_cube",
            "strategy_pool_stability",
            "strategy_pool_apply",
            "strategy_pool_materialize",
            "strategy_pool_validation",
            "automatic_tree_leaf_materialization",
            "interactive_tree_split_search",
            "interactive_tree_auto_continuation",
            "interactive_tree_revision",
            "cross_matrix_cell_selection",
            "voting_candidate_search",
            "voting_candidate_build_from_search",
            "voting_candidate_build",
            "cross_matrix_candidate_search",
            "cross_matrix_candidate_build_from_search",
            "cross_rule_search",
            "cross_rule_candidate_build_from_search",
        }:
            return False
        return True
    return not (draft.strategy_spec is None and draft.operation == "report")

def _strategy_request_requires_target(
    draft: CompiledStrategyRequestDraft,
) -> bool:
    if isinstance(draft, StandardWorkflowRequestDraft):
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[1]
        if draft.workflow == "strategy_project_context":
            return False
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "strategy_sample_design_v2",
                "automatic_tree_candidate_build",
                "cross_matrix_analysis",
                "strategy_pool_impact",
            }
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
        migrated = migrated_workflow_requirements(
            draft.workflow,
            draft.workflow_inputs,
        )
        if migrated is not None:
            return migrated[2]
        if draft.workflow == "strategy_project_context":
            return False
        return (
            draft.workflow
            in {
                "strategy_sample_design",
                "strategy_sample_design_v2",
                "automatic_tree_candidate_build",
                "cross_matrix_analysis",
                "strategy_pool_impact",
            }
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
    ingest_notices: list[dict] | None = None,
) -> dict:
    normalized_fields = list(dict.fromkeys(str(field) for field in fields))
    normalized_notices = [
        dict(notice)
        for notice in (ingest_notices or [])
        if isinstance(notice, dict)
    ]
    metadata = {
        "intent": "strategy_request",
        "kind": "clarification",
        "code": code,
    }
    if normalized_fields:
        metadata["fields"] = normalized_fields
    if normalized_notices:
        metadata["ingest_notices"] = normalized_notices
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content=f"{message}{_ingest_notice_text(normalized_notices)}",
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
    if normalized_notices:
        response["ingest_notices"] = normalized_notices
    return response
