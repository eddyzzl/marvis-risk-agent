"""Versioned business acceptance, independent of execution and narrative.

Persist objectives in the existing task contract / Plan.success_criteria JSON.
The evaluator consumes platform-authenticated evidence for the adopted object;
it never searches candidate metrics or accepts an LLM's pass/fail opinion.
"""

from dataclasses import asdict, dataclass, field
from datetime import date
import math
import re
from typing import Any

from marvis.orchestrator.evidence import payload_hash
from marvis.packs.strategy.project_context import EFFECT_STAGES


OBJECTIVE_VERSION = "business-objective.v1"
ACCEPTANCE_VERSION = "business-acceptance.v1"
BUSINESS_STATUSES = frozenset(
    {"passed", "failed", "insufficient_evidence", "not_configured", "not_applicable"}
)
_STAGE_ORDER = ("estimated", "backtested", "oot_validated", "post_launch_observed")
assert frozenset(_STAGE_ORDER) == EFFECT_STAGES


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class BusinessCriterion:
    metric: str
    unit: str
    denominator: str
    minimum: float | None = None
    maximum: float | None = None
    comparison: str = "absolute"
    baseline_ref: str | None = None

    def __post_init__(self):
        for name in ("metric", "unit", "denominator"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.minimum is None and self.maximum is None:
            raise ValueError("a business criterion requires a threshold")
        for name in ("minimum", "maximum"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum exceeds maximum")
        if not isinstance(self.comparison, str) or self.comparison not in {
            "absolute",
            "delta",
        }:
            raise ValueError("comparison must be absolute or delta")
        if self.comparison == "delta":
            _text(self.baseline_ref, "baseline_ref")
        elif self.baseline_ref is not None:
            raise ValueError("baseline_ref requires delta comparison")


@dataclass(frozen=True)
class BusinessObjective:
    business_line: str
    decision_node: str
    population: str
    period_start: str
    period_end: str
    target_kind: str
    responsibility_source: str
    criteria: tuple[BusinessCriterion, ...]
    schema_version: str = OBJECTIVE_VERSION
    target_id: str | None = None
    target_version: str | None = None
    currency: str | None = None
    minimum_effect_stage: str = "oot_validated"
    require_mature_labels: bool = True
    claim: str = "observational"
    applicable: bool = True
    allow_not_applicable: bool = False
    not_applicable_reason: str | None = None

    def __post_init__(self):
        if self.schema_version != OBJECTIVE_VERSION:
            raise ValueError("unsupported business objective version")
        for name in (
            "business_line",
            "decision_node",
            "population",
            "responsibility_source",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("period_start", "period_end"):
            value = _text(getattr(self, name), name)
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError(f"{name} must use YYYY-MM-DD")
        if self.period_start > self.period_end:
            raise ValueError("business period is reversed")
        if not isinstance(self.target_kind, str) or self.target_kind not in {
            "model",
            "strategy",
            "workflow",
        }:
            raise ValueError("unsupported business target kind")
        if (
            not isinstance(self.minimum_effect_stage, str)
            or self.minimum_effect_stage not in EFFECT_STAGES
        ):
            raise ValueError("unknown effect stage")
        if not isinstance(self.claim, str) or self.claim not in {
            "observational",
            "causal",
        }:
            raise ValueError("unknown business claim")
        for name in ("require_mature_labels", "applicable", "allow_not_applicable"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not self.applicable and (
            not self.allow_not_applicable or not self.not_applicable_reason
        ):
            raise ValueError("not_applicable requires contract permission and a reason")
        if self.applicable and not self.criteria:
            raise ValueError("an applicable business objective requires criteria")
        if not isinstance(self.criteria, tuple) or any(
            not isinstance(item, BusinessCriterion) for item in self.criteria
        ):
            raise ValueError("criteria must contain BusinessCriterion values")
        names = [item.metric for item in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError("duplicate business metric")
        if any(
            item.unit == "currency" or item.unit.startswith("currency_")
            for item in self.criteria
        ):
            _text(self.currency, "currency")
        for name in (
            "target_id",
            "target_version",
            "currency",
            "not_applicable_reason",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _text(getattr(self, name), name))

    def to_dict(self):
        value = asdict(self)
        value["criteria"] = [asdict(item) for item in self.criteria]
        return value

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("business objective must be an object")
        fields = dict(value)
        criteria = fields.get("criteria", [])
        if not isinstance(criteria, list):
            raise ValueError("business criteria must be a list")
        try:
            fields["criteria"] = tuple(BusinessCriterion(**item) for item in criteria)
            return cls(**fields)
        except TypeError as exc:
            raise ValueError(f"invalid business objective fields: {exc}") from exc


@dataclass(frozen=True)
class BusinessEvidence:
    """Construct only after authenticating a platform producer receipt."""

    target_kind: str
    target_id: str
    target_version: str
    source_ref: str
    source_hash: str
    metrics: dict[str, Any]
    business_line: str | None = None
    decision_node: str | None = None
    metric_units: dict[str, str] = field(default_factory=dict)
    denominators: dict[str, str] = field(default_factory=dict)
    population: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    currency: str | None = None
    effect_stage: str | None = None
    labels_mature: bool | None = None
    label_origin: str | None = None
    baseline_ref: str | None = None
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_binding_hash: str | None = None
    baseline_context: dict[str, Any] = field(default_factory=dict)
    causal_identification_ref: str | None = None


def evaluate_business_acceptance(
    objective: BusinessObjective | None, evidence: BusinessEvidence | None
) -> dict:
    result = {
        "schema_version": ACCEPTANCE_VERSION,
        "status": "not_configured",
        "objective": objective.to_dict() if objective else None,
        "objective_hash": payload_hash(objective.to_dict()) if objective else None,
        "target": None,
        "evidence": None,
        "criteria": [],
        "reasons": [],
    }
    if objective is None:
        result["reasons"] = ["未配置业务验收标准；流程完成不代表业务达标。"]
        return result
    if not objective.applicable:
        result.update(
            status="not_applicable", reasons=[objective.not_applicable_reason]
        )
        return result
    result["status"] = "insufficient_evidence"
    if evidence is None:
        result["reasons"] = ["缺少实际采用对象的认证证据。"]
        return result
    result["target"] = {
        "kind": evidence.target_kind,
        "id": evidence.target_id,
        "version": evidence.target_version,
    }
    result["evidence"] = {
        "source_ref": evidence.source_ref,
        "source_hash": evidence.source_hash,
        "business_line": evidence.business_line,
        "decision_node": evidence.decision_node,
        "effect_stage": evidence.effect_stage,
        "period_start": evidence.period_start,
        "period_end": evidence.period_end,
        "labels_mature": evidence.labels_mature,
        "label_origin": evidence.label_origin,
        "claim": objective.claim,
        "baseline_ref": evidence.baseline_ref,
        "baseline_binding_hash": evidence.baseline_binding_hash,
        "baseline_context": evidence.baseline_context,
    }
    missing = []
    if (
        evidence.target_kind != objective.target_kind
        or not evidence.target_id
        or not evidence.target_version
    ):
        missing.append("采用对象或版本未绑定。")
    if objective.target_id is not None and objective.target_id != evidence.target_id:
        missing.append("证据不属于指定采用对象。")
    if (
        objective.target_version is not None
        and objective.target_version != evidence.target_version
    ):
        missing.append("采用对象版本不匹配。")
    if not evidence.source_ref or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", evidence.source_hash or ""
    ):
        missing.append("缺少认证证据引用或内容哈希。")
    for name, label in (
        ("business_line", "业务线"),
        ("decision_node", "决策节点"),
        ("population", "目标客群"),
        ("period_start", "时期起点"),
        ("period_end", "时期终点"),
        ("currency", "币种"),
    ):
        if getattr(objective, name) != getattr(evidence, name):
            missing.append(f"{label}未绑定或不匹配。")
    if evidence.effect_stage not in EFFECT_STAGES or _STAGE_ORDER.index(
        evidence.effect_stage
    ) < _STAGE_ORDER.index(objective.minimum_effect_stage):
        missing.append("证据效果阶段不足。")
    if objective.require_mature_labels and (
        evidence.labels_mature is not True or evidence.label_origin != "observed"
    ):
        missing.append("缺少成熟的真实观察标签；假设或拒绝推断不能代替。")
    if objective.claim == "causal" and (
        evidence.effect_stage != "post_launch_observed"
        or not evidence.causal_identification_ref
    ):
        missing.append("缺少因果识别条件；观察差异不能认定为增量收益。")
    failures = []
    for criterion in objective.criteria:
        item = {
            **asdict(criterion),
            "value": None,
            "baseline_value": None,
            "status": "insufficient_evidence",
            "reason": "",
        }
        try:
            value = _finite(evidence.metrics.get(criterion.metric), criterion.metric)
            if evidence.metric_units.get(criterion.metric) != criterion.unit:
                raise ValueError("指标单位未绑定或不匹配")
            if evidence.denominators.get(criterion.metric) != criterion.denominator:
                raise ValueError("指标分母未绑定或不匹配")
            if criterion.comparison == "delta":
                if evidence.baseline_ref != criterion.baseline_ref or not re.fullmatch(
                    r"sha256:[0-9a-f]{64}", evidence.baseline_binding_hash or ""
                ):
                    raise ValueError("基线未认证或不匹配")
                expected_context = {
                    "business_line": evidence.business_line,
                    "decision_node": evidence.decision_node,
                    "population": evidence.population,
                    "period_start": evidence.period_start,
                    "period_end": evidence.period_end,
                    "currency": evidence.currency,
                }
                context = evidence.baseline_context
                if any(
                    context.get(key) != value for key, value in expected_context.items()
                ):
                    raise ValueError("基线客群、时期、单位或分母不匹配")
                if (
                    context.get("metric_units", {}).get(criterion.metric)
                    != criterion.unit
                    or context.get("denominators", {}).get(criterion.metric)
                    != criterion.denominator
                ):
                    raise ValueError("基线指标单位或分母不匹配")
                baseline = _finite(
                    evidence.baseline_metrics.get(criterion.metric), "baseline"
                )
                item["baseline_value"] = baseline
                value = _finite(value - baseline, "delta")
            item["value"] = value
            passed = (criterion.minimum is None or value >= criterion.minimum) and (
                criterion.maximum is None or value <= criterion.maximum
            )
            item["status"] = "passed" if passed else "failed"
            if not passed:
                item["reason"] = f"{criterion.metric} 未达到约定阈值。"
                failures.append(item["reason"])
        except ValueError as exc:
            item["reason"] = str(exc)
            missing.append(f"{criterion.metric}: {exc}")
        result["criteria"].append(item)
    result["reasons"] = missing + failures
    result["status"] = (
        "insufficient_evidence" if missing else "failed" if failures else "passed"
    )
    return result


def acceptance_display_rows(result: dict) -> list[dict]:
    """One deterministic tabular projection for API, Agent, XLSX and DOCX."""
    if (
        result.get("schema_version") != ACCEPTANCE_VERSION
        or result.get("status") not in BUSINESS_STATUSES
    ):
        raise ValueError("invalid business acceptance result")
    return [
        {
            "metric": item["metric"],
            "value": item.get("value"),
            "unit": item["unit"],
            "denominator": item["denominator"],
            "status": item["status"],
            "reason": item.get("reason", ""),
        }
        for item in result["criteria"]
    ]
