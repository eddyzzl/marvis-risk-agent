"""Setup (slot-filling) for strategy tasks.

The standard product entry is governed strategy development: it resolves the
sample and business contract but never invents an operating cutoff.  The old
20%-quantile candidate remains available only through the explicit quick-analysis
entry.  More-specific strategy intents are classified before either entry mode so
pricing or portfolio requests can never fall through to an approval plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from marvis.agent.sample_setup import detect_setup
from marvis.data.labels import nan_label_mask
from marvis.db import StrategyRepository
from marvis.domain import STRATEGY_OBJECTIVES, FileRole
from marvis.files import scan_source_dir, sha256_file

_DATA_ROLES = frozenset({FileRole.SAMPLE.value, "sample", "strategy_sample"})
_SCORE_HINTS = (
    "score",
    "pred",
    "prediction",
    "prob",
    "probability",
    "pd",
    "risk_score",
    "model_score",
    "credit_score",
)


# S4: goal phrases that route a strategy task to the rule_strategy template
# (rule mining) instead of the default strategy_analysis. Kept in sync with
# RULE_STRATEGY.goal_patterns; the strategy_setup intent branch multi-recognizes
# these -- parallel to how strategy_development got its own goal_patterns (S2).
_RULE_STRATEGY_GOAL_PATTERNS = ("规则挖掘", "拒绝规则", "规则策略", "rule mining", "rule strategy")

# S5: goal phrases that route a strategy task to the strategy_monitoring template
# (run one monitoring pass for an adopted strategy) instead of a development flow.
# Kept in sync with STRATEGY_MONITORING.goal_patterns; the strategy_setup intent
# branch multi-recognizes these -- the same S4 precedent as rule_strategy above.
_STRATEGY_MONITORING_GOAL_PATTERNS = (
    "策略监控",
    "跑监控",
    "监控运行策略",
    "monitoring run 策略",
    "strategy monitoring",
    "监控策略",
)

STRATEGY_ENTRY_DEVELOPMENT = "strategy_development"
STRATEGY_ENTRY_ANALYSIS = "strategy_analysis"
_QUICK_STRATEGY_ANALYSIS_GOAL_PATTERNS = (
    "快速策略分析",
    "快速策略回测",
    "quick strategy analysis",
    "quick strategy backtest",
)
_LIMIT_PRICING_GOAL_PATTERNS = (
    "额度定价",
    "定价矩阵",
    "limit pricing",
    "limit-pricing",
)
_PORTFOLIO_ANALYSIS_GOAL_PATTERNS = (
    "组合分析",
    "组合风险分析",
    "portfolio analysis",
)

STRATEGY_INTENT_FULL_DEVELOPMENT = "full_development"
STRATEGY_INTENT_QUICK_ANALYSIS = "quick_analysis"
STRATEGY_INTENT_RULE_MINING = "rule_mining"
STRATEGY_INTENT_MONITORING = "monitoring"
STRATEGY_INTENT_LIMIT_PRICING = "limit_pricing"
STRATEGY_INTENT_PORTFOLIO_ANALYSIS = "portfolio_analysis"
_PROFIT_PARAM_FIELDS = (
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
)


def is_rule_strategy_goal(*texts: str | None) -> bool:
    haystack = " ".join(text.lower() for text in texts if text)
    return any(pattern.lower() in haystack for pattern in _RULE_STRATEGY_GOAL_PATTERNS)


def is_strategy_monitoring_goal(*texts: str | None) -> bool:
    haystack = " ".join(text.lower() for text in texts if text)
    return any(pattern.lower() in haystack for pattern in _STRATEGY_MONITORING_GOAL_PATTERNS)


def is_quick_strategy_analysis_goal(*texts: str | None) -> bool:
    """True only for an explicit quick/lightweight request.

    Generic task names such as ``额度准入策略回测`` are deliberately not treated as
    a quick-mode choice: the product card itself uses that wording, so doing so would
    silently turn the standard development entry back into the lightweight workflow.
    """

    haystack = " ".join(text.lower() for text in texts if text)
    return any(
        pattern.lower() in haystack for pattern in _QUICK_STRATEGY_ANALYSIS_GOAL_PATTERNS
    )


def is_limit_pricing_goal(*texts: str | None) -> bool:
    """Recognize an explicit limit/pricing request without matching generic 额度准入."""

    haystack = " ".join(text.lower() for text in texts if text)
    return any(pattern.lower() in haystack for pattern in _LIMIT_PRICING_GOAL_PATTERNS)


def is_portfolio_analysis_goal(*texts: str | None) -> bool:
    haystack = " ".join(text.lower() for text in texts if text)
    return any(
        pattern.lower() in haystack for pattern in _PORTFOLIO_ANALYSIS_GOAL_PATTERNS
    )


def resolve_strategy_intent(strategy_input, *texts: str | None) -> str:
    """Return the canonical strategy intent using one explicit priority order.

    Monitoring and rule operations retain their existing precedence.  The two
    recognized-but-not-executed intents follow, so they can return a governed
    redirect instead of silently becoming an approval strategy.  Quick analysis
    must remain explicit; every other strategy request is full development.
    """

    if is_strategy_monitoring_goal(*texts):
        return STRATEGY_INTENT_MONITORING
    if is_rule_strategy_goal(*texts):
        return STRATEGY_INTENT_RULE_MINING
    if is_limit_pricing_goal(*texts):
        return STRATEGY_INTENT_LIMIT_PRICING
    if is_portfolio_analysis_goal(*texts):
        return STRATEGY_INTENT_PORTFOLIO_ANALYSIS
    if is_quick_strategy_analysis_goal(*texts):
        return STRATEGY_INTENT_QUICK_ANALYSIS
    entry_mode = str(_input_value(strategy_input, "entry_mode") or "").strip().lower()
    if entry_mode == STRATEGY_ENTRY_ANALYSIS:
        return STRATEGY_INTENT_QUICK_ANALYSIS
    return STRATEGY_INTENT_FULL_DEVELOPMENT


def strategy_development_clarification(strategy_input) -> dict | None:
    """Return a structured missing-input envelope, or ``None`` when start is safe."""

    strategy_type = str(
        _input_value(strategy_input, "strategy_type") or "approval"
    ).strip().lower()
    if strategy_type not in {"approval", "reject"}:
        return {
            "code": "strategy_typed_spec_required",
            "entry_mode": STRATEGY_ENTRY_DEVELOPMENT,
            "strategy_type": strategy_type,
            "missing_fields": ["strategy_spec"],
            "message": (
                f"{strategy_type} 策略不能套用准入 cutoff 工作流；"
                "需要先由自然语言请求编译并确认类型化 Strategy DSL。"
            ),
        }

    missing: list[str] = []
    objective = str(_input_value(strategy_input, "objective") or "").strip()
    if not objective:
        missing.append("objective")
    if (
        _input_value(strategy_input, "max_bad_rate") is None
        and _input_value(strategy_input, "min_approval_rate") is None
    ):
        missing.append("max_bad_rate_or_min_approval_rate")

    if objective == "max_profit":
        profit = _input_value(strategy_input, "profit")
        if not str(_input_value(profit, "ead_col") or "").strip():
            missing.append("profit.ead_col")
        if not str(_input_value(profit, "pd_col") or "").strip():
            missing.append("profit.pd_col")
        for field in _PROFIT_PARAM_FIELDS:
            if _input_value(profit, field) is None:
                missing.append(f"profit.{field}")

    if not missing:
        return None
    return {
        "code": "strategy_business_inputs_required",
        "entry_mode": STRATEGY_ENTRY_DEVELOPMENT,
        "missing_fields": missing,
        "message": "完整策略开发需要先确认经营目标和约束，平台不会用技术默认值代替经营决策。",
    }


def strategy_development_slot_clarification(slots) -> dict | None:
    """Validate the generic-plan slot shape before a full plan is instantiated.

    Generic plans expose EAD/PD as top-level slots and the financial assumptions
    under ``profit_params``.  This is intentionally a business one-of/conditional
    validator rather than a set of globally-required ``SlotSpec`` declarations.
    """

    missing: list[str] = []
    invalid: list[str] = []
    objective_value = _input_value(slots, "objective")
    objective = str(objective_value or "").strip()
    if _contract_value_missing(objective_value):
        missing.append("objective")
    elif not isinstance(objective_value, str) or objective not in STRATEGY_OBJECTIVES - {""}:
        invalid.append("objective")

    constraint_values = {
        "max_bad_rate": _input_value(slots, "max_bad_rate"),
        "min_approval_rate": _input_value(slots, "min_approval_rate"),
    }
    supplied_constraints = [
        field
        for field, value in constraint_values.items()
        if not _contract_value_missing(value)
    ]
    if not supplied_constraints:
        missing.append("max_bad_rate_or_min_approval_rate")
    else:
        for field in supplied_constraints:
            if not _valid_contract_number(
                constraint_values[field], minimum=0.0, maximum=1.0
            ):
                invalid.append(field)

    if objective == "max_profit":
        for field in ("ead_col", "pd_col"):
            value = _input_value(slots, field)
            if _contract_value_missing(value):
                missing.append(field)
            elif not isinstance(value, str):
                invalid.append(field)
        profit_params = _input_value(slots, "profit_params")
        if profit_params is not None and not isinstance(profit_params, dict):
            invalid.append("profit_params")
        else:
            for field in _PROFIT_PARAM_FIELDS:
                path = f"profit_params.{field}"
                value = _input_value(profit_params, field)
                if _contract_value_missing(value):
                    missing.append(path)
                    continue
                if field in {"annual_rate", "funding_rate", "lgd"}:
                    valid = _valid_contract_number(value, minimum=0.0, maximum=1.0)
                elif field == "operating_cost_per_loan":
                    valid = _valid_contract_number(value, minimum=0.0)
                else:
                    valid = (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value >= 1
                    )
                if not valid:
                    invalid.append(path)

    if not missing and not invalid:
        return None
    return {
        "code": (
            "strategy_business_inputs_invalid"
            if invalid
            else "strategy_business_inputs_required"
        ),
        "template_id": STRATEGY_ENTRY_DEVELOPMENT,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "message": (
            "完整策略开发参数格式或范围无效，请按经营 contract 修正。"
            if invalid
            else "完整策略开发需要先确认经营目标和约束，平台不会用技术默认值代替经营决策。"
        ),
    }


def _contract_value_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _valid_contract_number(
    value,
    *,
    minimum: float,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        return False
    return maximum is None or number <= maximum


class StrategySetupError(ValueError):
    """Raised when a strategy task cannot infer a scored binary sample."""


@dataclass(frozen=True)
class StrategyDatasetContext:
    """Task-owned dataset facts exposed to request compilation and workflows.

    This is deliberately narrower than a strategy proposal: resolving context
    must not invent a cutoff, rule, action or business objective.
    """

    dataset_id: str
    dataset_name: str
    target_col: str | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class StrategyDatasetPreview:
    """Read-only dataset facts used before the user confirms a request."""

    dataset_id: str | None
    dataset_name: str
    target_col: str | None
    columns: tuple[str, ...]
    identity: dict


def preview_strategy_dataset_context(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
) -> StrategyDatasetPreview:
    """Inspect columns without registering, converting or persisting a dataset."""

    datasets = [
        dataset
        for dataset in registry.list_for_task(task_id)
        if dataset.role in _DATA_ROLES
    ]
    if datasets:
        dataset = _select_dataset(datasets)
        path = registry.resolve_path(dataset.id)
        identity = {
            "kind": "registered",
            "dataset_id": dataset.id,
            # The catalog hash describes the bytes at registration time.  The
            # confirmation boundary must bind the bytes that are actually on
            # disk now, otherwise an out-of-band same-schema rewrite could
            # reuse an already-confirmed request against different rows.
            "content_hash": sha256_file(path),
        }
        dataset_id = dataset.id
        dataset_name = _dataset_name(dataset)
    else:
        if source_dir is None:
            raise StrategySetupError("策略分析未找到数据文件。")
        artifacts = [
            artifact
            for artifact in scan_source_dir(Path(source_dir))
            if artifact.role == FileRole.SAMPLE
        ]
        if not artifacts:
            raise StrategySetupError(f"策略分析未找到数据文件:{source_dir}")
        if len(artifacts) != 1:
            raise StrategySetupError(
                "策略目录包含多个样本；请先明确选择一个策略样本，平台不会在确认前猜测。"
            )
        artifact = artifacts[0]
        path = Path(artifact.path)
        stat = path.stat()
        identity = {
            "kind": "source",
            "source_path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": artifact.sha256,
        }
        dataset_id = None
        dataset_name = path.name

    columns = list(backend.column_names(path))
    try:
        resolved_target = _resolve_target_col(
            backend,
            path,
            columns,
            target_col,
        )
    except StrategySetupError:
        resolved_target = None
    return StrategyDatasetPreview(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        target_col=resolved_target,
        columns=tuple(columns),
        identity=identity,
    )


def build_strategy_dataset_context(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    require_target: bool = True,
) -> StrategyDatasetContext:
    """Register the sample and resolve only the evidence required by an operation.

    Evaluation needs a binary target; deterministic application to a production
    sample does not. Keeping that distinction here prevents ``apply`` from
    inventing or requiring a label that is unavailable at decision time.
    """

    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = list(backend.column_names(path))
    if require_target:
        resolved_target = _resolve_target_col(backend, path, columns, target_col)
    else:
        try:
            resolved_target = _resolve_target_col(
                backend,
                path,
                columns,
                target_col,
            )
        except StrategySetupError:
            resolved_target = None
    return StrategyDatasetContext(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        columns=tuple(columns),
    )


@dataclass
class StrategyProposal:
    dataset_id: str
    dataset_name: str
    target_col: str
    score_col: str
    strategy_type: str
    rules: list[dict]
    default_decision: str
    cutoff: float
    direction: str
    bad_rate: float | None
    notes: list[str]
    template_id: str = "strategy_analysis"

    def template_slots(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "score_col": self.score_col,
            "strategy_type": self.strategy_type,
            "rules": self.rules,
            "default_decision": self.default_decision,
        }


@dataclass
class StrategyDevelopmentProposal:
    """Slots for the full workflow, without a pre-built cutoff or rule candidate."""

    dataset_id: str
    dataset_name: str
    target_col: str
    score_col: str
    strategy_type: str
    objective: str
    max_bad_rate: float | None
    min_approval_rate: float | None
    baseline_strategy_id: str | None
    ead_col: str | None
    pd_col: str | None
    profit_params: dict | None
    bad_rate: float | None
    notes: list[str]
    template_id: str = STRATEGY_ENTRY_DEVELOPMENT

    def template_slots(self) -> dict:
        slots = {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
            "score_col": self.score_col,
            "strategy_type": self.strategy_type,
            "objective": self.objective,
            "max_bad_rate": self.max_bad_rate,
            "min_approval_rate": self.min_approval_rate,
            "baseline_strategy_id": self.baseline_strategy_id,
            "ead_col": self.ead_col,
            "pd_col": self.pd_col,
            "profit_params": self.profit_params,
        }
        return {key: value for key, value in slots.items() if value is not None}


@dataclass
class RuleStrategyProposal:
    """S4: setup proposal that routes to the rule_strategy template. Unlike the
    lightweight strategy_analysis proposal it carries no pre-built rules -- the
    template's mine_rules step discovers them -- only the dataset/target anchor
    and the rule-mining slots. The adoption reason is collected only at the final
    evidence-bound gate, never during task setup."""

    dataset_id: str
    dataset_name: str
    target_col: str
    score_col: str | None
    bad_rate: float | None
    notes: list[str]
    template_id: str = "rule_strategy"

    def template_slots(self) -> dict:
        slots: dict = {
            "dataset_id": self.dataset_id,
            "target_col": self.target_col,
        }
        if self.score_col:
            slots["score_col"] = self.score_col
        return slots


@dataclass
class MonitoringSetupProposal:
    """S5: setup proposal that routes to the strategy_monitoring template. Resolves
    the task's single adopted strategy plus the fresh monitoring sample; the
    template's run_strategy_monitoring step reads the plan off the adopted strategy
    and grades drift against the adoption baseline."""

    strategy_id: str
    dataset_id: str
    dataset_name: str
    score_col: str | None
    target_col: str | None
    notes: list[str]
    template_id: str = "strategy_monitoring"

    def template_slots(self) -> dict:
        slots: dict = {"strategy_id": self.strategy_id, "dataset_id": self.dataset_id}
        if self.score_col:
            slots["score_col"] = self.score_col
        if self.target_col:
            slots["target_col"] = self.target_col
        return slots


def build_monitoring_setup_proposal(
    registry,
    backend,
    db_path,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    score_col: str | None = None,
) -> MonitoringSetupProposal:
    """Resolve the adopted strategy + fresh monitoring sample for a monitoring task.

    A monitoring task must have exactly one adopted strategy to monitor; if none is
    adopted yet, that is a setup error (nothing to monitor). The dataset is the new
    performance/application sample (resolved the same way as a development task's
    sample). target_col/score_col are optional passthroughs (labels may not have
    matured; score_col only matters for a model-backed strategy)."""
    adopted = [
        meta for meta in StrategyRepository(db_path).list_meta_for_task(task_id)
        if meta.get("status") == "adopted"
    ]
    if not adopted:
        raise StrategySetupError("当前任务没有已采纳策略,无法执行监控;请先采纳一个策略。")
    strategy_id = str(adopted[-1]["id"])
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = target_col if (target_col and target_col in columns) else None
    resolved_score = _optional_score_col(columns, score_col)
    notes = [f"将对已采纳策略 {strategy_id} 跑一次监控,并与采纳基线对比漂移。"]
    return MonitoringSetupProposal(
        strategy_id=strategy_id,
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        score_col=resolved_score,
        target_col=resolved_target,
        notes=notes,
    )


def build_strategy_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    score_col: str | None = None,
) -> StrategyProposal:
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = _resolve_target_col(backend, path, columns, target_col)
    resolved_score = _resolve_score_col(columns, score_col)
    if not resolved_score.isidentifier():
        raise StrategySetupError(
            f"策略条件暂只支持 Python 标识符列名；评分列 `{resolved_score}` 需先重命名后再回测。"
        )
    frame = backend.read_frame(path, columns=[resolved_target, resolved_score])
    profile = _score_profile(frame, target_col=resolved_target, score_col=resolved_score)
    rule = {
        "condition": profile["condition"],
        "decision": "reject",
    }
    return StrategyProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        score_col=resolved_score,
        strategy_type="approval",
        rules=[rule],
        default_decision="approve",
        cutoff=profile["cutoff"],
        direction=profile["direction"],
        bad_rate=profile["bad_rate"],
        notes=profile["notes"],
    )


def build_strategy_development_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    strategy_input,
    target_col: str | None = None,
    score_col: str | None = None,
) -> StrategyDevelopmentProposal:
    """Build the full-workflow slots without choosing a cutoff or creating rules."""

    clarification = strategy_development_clarification(strategy_input)
    if clarification is not None:
        raise StrategySetupError(clarification["message"])

    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = _resolve_target_col(backend, path, columns, target_col)
    resolved_score = _resolve_score_col(columns, score_col)
    if not resolved_score.isidentifier():
        raise StrategySetupError(
            f"策略条件暂只支持 Python 标识符列名；评分列 `{resolved_score}` 需先重命名后再回测。"
        )

    profit = _input_value(strategy_input, "profit")
    ead_col = _optional_text(_input_value(profit, "ead_col"))
    pd_col = _optional_text(_input_value(profit, "pd_col"))
    objective = str(_input_value(strategy_input, "objective") or "").strip()
    if objective == "max_profit":
        missing_columns = [column for column in (ead_col, pd_col) if column not in columns]
        if missing_columns:
            raise StrategySetupError(
                "利润目标引用的数据列不存在：" + "、".join(f"`{column}`" for column in missing_columns)
            )

    bad_rate, n_nan_labels = _target_bad_rate(backend, path, resolved_target)
    notes = ["尚未生成 cutoff 或规则；将按已确认的经营目标和约束扫描可行方案。"]
    if n_nan_labels:
        notes.append(
            f"目标列 `{resolved_target}` 有 {n_nan_labels} 行标签为空/非法；"
            "预览坏率已排除这些行，执行时仍需通过标签处理确认门。"
        )
    return StrategyDevelopmentProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        score_col=resolved_score,
        strategy_type=str(
            _input_value(strategy_input, "strategy_type") or "approval"
        ),
        objective=objective,
        max_bad_rate=_optional_float_value(_input_value(strategy_input, "max_bad_rate")),
        min_approval_rate=_optional_float_value(
            _input_value(strategy_input, "min_approval_rate")
        ),
        baseline_strategy_id=_optional_text(
            _input_value(strategy_input, "baseline_strategy_id")
        ),
        ead_col=ead_col,
        pd_col=pd_col,
        profit_params=_profit_params_payload(profit),
        bad_rate=bad_rate,
        notes=notes,
    )


def build_rule_strategy_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    target_col: str | None = None,
    score_col: str | None = None,
) -> RuleStrategyProposal:
    """S4: resolve the dataset/target anchor for a rule-mining task. Score is
    optional here (rule mining works on arbitrary numeric features, not just a
    single score); when present it is passed through so build_strategy's rule
    direction self-check can fire on any score-band rules."""
    dataset = _resolve_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)
    resolved_target = _resolve_target_col(backend, path, columns, target_col)
    resolved_score = _optional_score_col(columns, score_col)
    bad_rate, n_nan_labels = _target_bad_rate(backend, path, resolved_target)
    notes = ["将在数据上挖掘候选拒绝规则，选定规则集后回测并采纳。"]
    if n_nan_labels:
        notes.append(
            f"注意：目标列 `{resolved_target}` 有 {n_nan_labels} 行标签为空/非法，"
            "预览坏率已排除这些行；回测时将再次确认如何处理。"
        )
    return RuleStrategyProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        target_col=resolved_target,
        score_col=resolved_score,
        bad_rate=bad_rate,
        notes=notes,
    )


def _optional_score_col(columns: list[str], requested: str | None) -> str | None:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested if requested.isidentifier() else None
    return None


def _input_value(payload, name: str):
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(name)
    return getattr(payload, name, None)


def _optional_text(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float_value(value) -> float | None:
    return None if value is None else float(value)


def _profit_params_payload(profit) -> dict | None:
    if profit is None or any(_input_value(profit, field) is None for field in _PROFIT_PARAM_FIELDS):
        return None
    return {field: _input_value(profit, field) for field in _PROFIT_PARAM_FIELDS}


def _target_bad_rate(backend, path, target_col: str) -> tuple[float | None, int]:
    """Preview bad-rate over finite labels, plus the NaN-label count excluded.

    T2-1: this is a pre-gate PREVIEW surface, so it cannot hard-stop the way the
    canonical ``resolve_labeled_frame`` gate does at backtest time. But it must not
    SILENTLY diverge either: it reads finite labels via the same ``nan_label_mask``
    the gate uses (so the bad-rate matches the gated value after a confirmed drop)
    and returns ``n_nan`` so the caller can surface exactly what the gate would.
    Returns ``(None, 0)`` when the column is unreadable or non-numeric (the preview
    stays best-effort; the gate still fires with a hard error downstream).
    """
    try:
        frame = backend.read_frame(path, columns=[target_col])
        mask = nan_label_mask(frame, target_col)
    except Exception:
        return None, 0
    finite = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)[~mask]
    if finite.size == 0:
        return None, int(mask.sum())
    return float((finite == 1).mean()), int(mask.sum())


def _resolve_dataset(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _DATA_ROLES]
    if not datasets:
        raise StrategySetupError(f"策略分析未找到数据文件:{source_dir}")
    return _select_dataset(datasets)


def _select_dataset(datasets):
    return sorted(
        datasets,
        key=lambda d: (not bool(getattr(d, "has_target", False)), -int(getattr(d, "row_count", 0) or 0)),
    )[0]


def _resolve_target_col(backend, path: Path, columns: list[str], requested: str | None) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    setup = detect_setup(backend, path)
    if setup.target_col:
        return setup.target_col
    raise StrategySetupError("未能识别 0/1 目标列；请在创建任务时指定 target_col。")


def _resolve_score_col(columns: list[str], requested: str | None) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    lowered = {column.lower(): column for column in columns}
    for hint in _SCORE_HINTS:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        low = column.lower()
        if "score" in low or low in {"pred", "pd"} or "prob" in low:
            return column
    raise StrategySetupError("未能识别评分列；请在创建任务时指定 score_col。")


def _score_profile(frame: pd.DataFrame, *, target_col: str, score_col: str) -> dict:
    clean = frame[[target_col, score_col]].copy()
    # T2-1: read finite labels via the canonical nan_label_mask so the preview
    # bad-rate uses the SAME label semantics as the resolve_labeled_frame gate that
    # backtesting applies later (no silent divergence). A NaN *score* drop is a
    # legitimately different concern (score is not a label), so it stays a plain
    # dropna; only the NaN-LABEL count is surfaced back to the user.
    try:
        label_nan_mask = nan_label_mask(clean, target_col)
    except Exception as exc:  # non-numeric labels: mirror the gate's hard error
        raise StrategySetupError(
            f"目标列 `{target_col}` 含非数值标签，无法作为 0/1 目标预览；请修正后重试。"
        ) from exc
    n_nan_labels = int(label_nan_mask.sum())
    clean[target_col] = pd.to_numeric(clean[target_col], errors="coerce")
    clean[score_col] = pd.to_numeric(clean[score_col], errors="coerce")
    clean = clean.loc[~label_nan_mask].dropna()
    if clean.empty:
        raise StrategySetupError("目标列/评分列没有可用于策略回测的有效数值。")
    target = clean[target_col].astype(int)
    scores = clean[score_col].astype(float)
    bad_rate = float((target == 1).mean())
    corr = scores.rank(method="average").corr(target)
    higher_score_riskier = bool(corr is not None and pd.notna(corr) and corr > 0)
    quantile = 0.80 if higher_score_riskier else 0.20
    cutoff = float(scores.quantile(quantile))
    cutoff_literal = _number_literal(cutoff)
    if higher_score_riskier:
        condition = f"{score_col} >= {cutoff_literal}"
        direction = "higher_score_riskier"
        notes = [f"评分越高坏样本率越高，默认拒绝评分最高约 20%（cutoff={cutoff_literal}）。"]
    else:
        condition = f"{score_col} < {cutoff_literal}"
        direction = "lower_score_riskier"
        notes = [f"评分越低坏样本率越高，默认拒绝评分最低约 20%（cutoff={cutoff_literal}）。"]
    if n_nan_labels:
        notes.append(
            f"注意：目标列 `{target_col}` 有 {n_nan_labels} 行标签为空/非法，"
            "预览坏率与 cutoff 已排除这些行；回测时将再次确认如何处理。"
        )
    return {
        "condition": condition,
        "cutoff": cutoff,
        "direction": direction,
        "bad_rate": bad_rate,
        "notes": notes,
    }


def _number_literal(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = [
    "MonitoringSetupProposal",
    "RuleStrategyProposal",
    "STRATEGY_ENTRY_ANALYSIS",
    "STRATEGY_ENTRY_DEVELOPMENT",
    "STRATEGY_INTENT_FULL_DEVELOPMENT",
    "STRATEGY_INTENT_LIMIT_PRICING",
    "STRATEGY_INTENT_MONITORING",
    "STRATEGY_INTENT_PORTFOLIO_ANALYSIS",
    "STRATEGY_INTENT_QUICK_ANALYSIS",
    "STRATEGY_INTENT_RULE_MINING",
    "StrategyDevelopmentProposal",
    "StrategyDatasetContext",
    "StrategyDatasetPreview",
    "StrategyProposal",
    "StrategySetupError",
    "build_monitoring_setup_proposal",
    "build_rule_strategy_proposal",
    "build_strategy_dataset_context",
    "build_strategy_development_proposal",
    "build_strategy_proposal",
    "preview_strategy_dataset_context",
    "is_limit_pricing_goal",
    "is_portfolio_analysis_goal",
    "is_quick_strategy_analysis_goal",
    "is_rule_strategy_goal",
    "is_strategy_monitoring_goal",
    "resolve_strategy_intent",
    "strategy_development_clarification",
    "strategy_development_slot_clarification",
]
