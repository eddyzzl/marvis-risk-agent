"""Pure natural-language compiler for governed strategy requests.

The LLM only translates an utterance into a draft.  This module then validates
that draft against fixed operation/type vocabularies, the Strategy DSL and the
dataset column whitelist.  It never executes a tool and never accepts calculated
metrics from the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Any

from marvis.agent.json_reply import load_json_object
from marvis.llm_prompts import STRATEGY_REQUEST_COMPILER_SYS
from marvis.packs.strategy.candidate_design import (
    CANDIDATE_DESIGN_SCHEMA_VERSION,
    CandidateDesignError,
    normalize_candidate_design,
    normalize_candidate_economics_inputs,
)
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.errors import StrategyError
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason


_SYSTEM = STRATEGY_REQUEST_COMPILER_SYS.text


STRATEGY_OPERATIONS = (
    "develop",
    "analyze",
    "backtest",
    "apply",
    "compare",
    "adopt",
    "report",
    "monitor",
    "mine_rules",
)
STRATEGY_TYPES = (
    "approval",
    "reject",
    "limit",
    "pricing",
    "segmentation",
)
STRATEGY_REQUEST_KINDS = (
    "strategy_lifecycle",
    "standard_workflow",
)
STANDARD_STRATEGY_WORKFLOWS = (
    "profit_calc",
    "roll_rate_matrix",
    "limit_pricing_matrix",
)

_OPTIONAL_DRAFT_FIELDS = {
    "objective",
    "max_bad_rate",
    "min_approval_rate",
    "baseline_strategy_id",
    "strategy_id",
    "adoption_reason",
    "profit",
    "economics_inputs",
    "candidate_design",
    "strategy_spec",
}
_DRAFT_FIELDS = {"operation", "strategy_type"} | _OPTIONAL_DRAFT_FIELDS
_LIFECYCLE_DRAFT_FIELDS = _DRAFT_FIELDS | {"request_kind"}
_STANDARD_WORKFLOW_DRAFT_FIELDS = {
    "request_kind",
    "workflow",
    "workflow_inputs",
}
_PROFIT_FIELDS = {
    "ead_col",
    "pd_col",
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
}
_PROFIT_PARAMETER_FIELDS = _PROFIT_FIELDS - {"ead_col", "pd_col"}
_LIMIT_ECONOMICS_NAMES = ("pd", "lgd", "utilization")
_PRICING_ECONOMICS_NAMES = (
    "ead",
    "pd",
    "lgd",
    "funding_rate",
    "term_months",
    "operating_cost_per_loan",
)
_ECONOMICS_VALUE_MAXIMUMS = {
    "pd": 1.0,
    "lgd": 1.0,
    "utilization": 1.0,
    "funding_rate": 1.0,
}
_ECONOMICS_LABELS = {
    "ead": "EAD",
    "pd": "PD",
    "lgd": "LGD",
    "utilization": "额度使用率",
    "funding_rate": "资金成本率",
    "term_months": "期限",
    "operating_cost_per_loan": "单笔运营成本",
}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_COLLECTION_STRATEGY_RE = re.compile(
    r"(?:催收|\bcollection(?:s)?(?:\s+|[-_])"
    r"(?:strategy|actions?|allocation|policy|workflow|campaign|frequency)\b)",
    re.IGNORECASE,
)
_NON_REPAIRABLE_CLARIFICATION_CODES = frozenset(
    {
        "candidate_economics_ambiguous",
        "candidate_economics_incomplete",
        "candidate_requires_observed_economics",
    }
)


_CANDIDATE_DESIGN_JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "score_band_limit"},
                "score_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "limit_grid": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "uniqueItems": True,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
                "max_expected_loss_per_account": {
                    "type": "number",
                    "minimum": 0,
                },
                "missing_policy": {"const": "zero_limit"},
            },
            "required": [
                "method",
                "score_col",
                "limit_grid",
                "max_expected_loss_per_account",
            ],
            "additionalProperties": False,
        },
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "score_band_pricing"},
                "score_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "rate_grid": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "uniqueItems": True,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "min_roa": {"type": "number", "minimum": 0, "maximum": 1},
                "missing_policy": {"const": "highest_risk_rate"},
            },
            "required": ["method", "score_col", "rate_grid"],
            "additionalProperties": False,
        },
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "single_variable_segmentation"},
                "feature_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "missing_policy": {"const": "separate_segment"},
            },
            "required": ["method", "feature_col"],
            "additionalProperties": False,
        },
    ],
}


STRATEGY_REQUEST_JSON_SCHEMA = {
    "name": "strategy_request_draft",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "request_kind": {
                "type": "string",
                "enum": list(STRATEGY_REQUEST_KINDS),
            },
            "operation": {"type": "string", "enum": list(STRATEGY_OPERATIONS)},
            "strategy_type": {"type": "string", "enum": list(STRATEGY_TYPES)},
            "objective": {"type": "string", "minLength": 1},
            "max_bad_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "min_approval_rate": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "baseline_strategy_id": {"type": "string", "minLength": 1},
            "strategy_id": {"type": "string", "minLength": 1},
            "adoption_reason": {"type": "string", "minLength": 1},
            "profit": {
                "type": "object",
                "properties": {
                    "ead_col": {"type": "string", "minLength": 1},
                    "pd_col": {"type": "string", "minLength": 1},
                    "annual_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "funding_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "lgd": {"type": "number", "minimum": 0, "maximum": 1},
                    "operating_cost_per_loan": {
                        "type": "number",
                        "minimum": 0,
                    },
                    "term_months": {"type": "integer", "minimum": 1},
                },
                "required": sorted(_PROFIT_FIELDS),
                "additionalProperties": False,
            },
            "economics_inputs": {
                "type": "object",
                "properties": {
                    "ead_col": {"type": "string", "minLength": 1},
                    "ead_value": {"type": "number", "minimum": 0},
                    "pd_col": {"type": "string", "minLength": 1},
                    "pd_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "lgd_col": {"type": "string", "minLength": 1},
                    "lgd_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "utilization_col": {"type": "string", "minLength": 1},
                    "utilization_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "funding_rate_col": {"type": "string", "minLength": 1},
                    "funding_rate_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "term_months_col": {"type": "string", "minLength": 1},
                    "term_months_value": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                    "operating_cost_per_loan_col": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "operating_cost_per_loan_value": {
                        "type": "number",
                        "minimum": 0,
                    },
                },
                "oneOf": [
                    {
                        "allOf": [
                            {
                                "oneOf": [
                                    {"required": ["pd_col"]},
                                    {"required": ["pd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["lgd_col"]},
                                    {"required": ["lgd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["utilization_col"]},
                                    {"required": ["utilization_value"]},
                                ]
                            },
                            {
                                "not": {
                                    "anyOf": [
                                        {"required": ["ead_col"]},
                                        {"required": ["ead_value"]},
                                        {"required": ["funding_rate_col"]},
                                        {"required": ["funding_rate_value"]},
                                        {"required": ["term_months_col"]},
                                        {"required": ["term_months_value"]},
                                        {
                                            "required": [
                                                "operating_cost_per_loan_col"
                                            ]
                                        },
                                        {
                                            "required": [
                                                "operating_cost_per_loan_value"
                                            ]
                                        },
                                    ]
                                }
                            },
                        ]
                    },
                    {
                        "allOf": [
                            {
                                "oneOf": [
                                    {"required": ["ead_col"]},
                                    {"required": ["ead_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["pd_col"]},
                                    {"required": ["pd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["lgd_col"]},
                                    {"required": ["lgd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["funding_rate_col"]},
                                    {"required": ["funding_rate_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["term_months_col"]},
                                    {"required": ["term_months_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["operating_cost_per_loan_col"]},
                                    {"required": ["operating_cost_per_loan_value"]},
                                ]
                            },
                            {
                                "not": {
                                    "anyOf": [
                                        {"required": ["utilization_col"]},
                                        {"required": ["utilization_value"]},
                                    ]
                                }
                            },
                        ]
                    },
                ],
                "additionalProperties": False,
            },
            "candidate_design": _CANDIDATE_DESIGN_JSON_SCHEMA,
            "strategy_spec": {"type": "object"},
            "workflow": {
                "type": "string",
                "enum": list(STANDARD_STRATEGY_WORKFLOWS),
            },
            "workflow_inputs": {"type": "object"},
            "clarification": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
        "oneOf": [
            {"required": ["operation", "strategy_type"]},
            {
                "properties": {
                    "request_kind": {"const": "standard_workflow"},
                },
                "required": ["request_kind", "workflow", "workflow_inputs"],
            },
            {"required": ["clarification"]},
        ],
    },
}


@dataclass(frozen=True)
class StrategyRequestDraft(Mapping[str, Any]):
    """Canonical, platform-validated strategy request draft."""

    operation: str
    strategy_type: str
    objective: str | None = None
    max_bad_rate: float | None = None
    min_approval_rate: float | None = None
    baseline_strategy_id: str | None = None
    strategy_id: str | None = None
    adoption_reason: str | None = None
    profit: Mapping[str, Any] | None = None
    economics_inputs: Mapping[str, Any] | None = None
    candidate_design: Mapping[str, Any] | None = None
    strategy_spec: Mapping[str, Any] | None = None

    @property
    def request_kind(self) -> str:
        return "strategy_lifecycle"

    def __post_init__(self) -> None:
        if self.profit is not None:
            object.__setattr__(self, "profit", _deep_freeze(self.profit))
        if self.economics_inputs is not None:
            object.__setattr__(
                self,
                "economics_inputs",
                _deep_freeze(self.economics_inputs),
            )
        if self.candidate_design is not None:
            object.__setattr__(
                self,
                "candidate_design",
                _deep_freeze(self.candidate_design),
            )
        if self.strategy_spec is not None:
            object.__setattr__(
                self,
                "strategy_spec",
                _deep_freeze(self.strategy_spec),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "strategy_type": self.strategy_type,
        }
        for field_name in (
            "objective",
            "max_bad_rate",
            "min_approval_rate",
            "baseline_strategy_id",
            "strategy_id",
            "adoption_reason",
            "profit",
            "economics_inputs",
            "candidate_design",
            "strategy_spec",
        ):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = _deep_thaw(value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class StandardWorkflowRequestDraft(Mapping[str, Any]):
    """Canonical request for a built-in, deterministic strategy analysis."""

    workflow: str
    workflow_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workflow_inputs",
            _deep_freeze(self.workflow_inputs),
        )

    @property
    def request_kind(self) -> str:
        return "standard_workflow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_kind": self.request_kind,
            "workflow": self.workflow,
            "workflow_inputs": _deep_thaw(self.workflow_inputs),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


CompiledStrategyRequestDraft = StrategyRequestDraft | StandardWorkflowRequestDraft


@dataclass(frozen=True)
class StrategyRequestCompilation:
    """A validated draft awaiting confirmation, or a Chinese clarification."""

    draft: CompiledStrategyRequestDraft | None
    clarification: str | None
    confirmation: str | None
    clarification_code: str | None = None
    clarification_fields: tuple[str, ...] = ()

    @property
    def validated_draft(self) -> CompiledStrategyRequestDraft | None:
        return self.draft

    @property
    def clarify(self) -> str | None:
        return self.clarification

    @property
    def confirmation_text(self) -> str | None:
        return self.confirmation

    @property
    def needs_clarification(self) -> bool:
        return self.draft is None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "draft": None if self.draft is None else self.draft.to_dict(),
            "clarification": self.clarification,
            "confirmation": self.confirmation,
        }
        if self.clarification_code is not None:
            payload["clarification_code"] = self.clarification_code
        if self.clarification_fields:
            payload["clarification_fields"] = list(self.clarification_fields)
        return payload


@dataclass(frozen=True)
class _ValidationOutcome:
    result: StrategyRequestCompilation
    accepted: bool
    error: str | None = None


def compile_strategy_request(
    utterance: str,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
    llm,
    caller: str = "strategy_request_compiler",
) -> StrategyRequestCompilation:
    """Compile one utterance with at most one LLM-format repair attempt."""

    if not isinstance(utterance, str) or not utterance.strip():
        return _clarification("请说明希望执行的策略操作和策略类型。")
    normalized_utterance = utterance.strip()
    if _COLLECTION_STRATEGY_RE.search(normalized_utterance):
        return _clarification(
            "催收动作策略尚无已评审的动作、成本、产能和回收口径，"
            "当前不能映射为审批、拒绝或分群策略。请说明是否只需要风险分层分析。",
            code="collection_strategy_unsupported",
            fields=("strategy_type", "collection_action_contract"),
        )
    whitelist = _column_whitelist(allowed_columns)
    observed_target = _normalized_target_col(target_col)
    prompt = _user_prompt(normalized_utterance, whitelist, target_col=observed_target)
    try:
        raw = _complete(llm, prompt=prompt, caller=caller)
    except Exception:
        return _clarification(
            "当前暂时无法解析策略请求，请稍后重试或直接说明操作、策略类型和策略对象。"
        )
    outcome = _validate_reply(raw, whitelist, target_col=observed_target)
    if outcome.accepted:
        return outcome.result
    if outcome.result.clarification_code in _NON_REPAIRABLE_CLARIFICATION_CODES:
        # These are platform-derived business-contract gaps, not JSON-format
        # mistakes. A second LLM pass cannot supply missing economics safely and
        # must not downgrade typed code/fields into a generic clarification.
        return outcome.result

    repair_prompt = _repair_prompt(
        prompt,
        raw=raw,
        error=outcome.error or "输出格式无效",
    )
    try:
        repaired = _complete(llm, prompt=repair_prompt, caller=caller)
    except Exception:
        return outcome.result
    return _validate_reply(
        repaired,
        whitelist,
        target_col=observed_target,
    ).result


def validate_strategy_request(
    payload: object,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
) -> StrategyRequestCompilation:
    """Validate an already parsed LLM payload without invoking an LLM."""

    return _validate_payload(
        payload,
        _column_whitelist(allowed_columns),
        target_col=_normalized_target_col(target_col),
    ).result


def strategy_request_confirmation_text(
    draft: CompiledStrategyRequestDraft,
) -> str:
    """Render a plain-Chinese echo of the request before any workflow runs."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        return _standard_workflow_confirmation_text(draft)

    operation = _OPERATION_LABELS[draft.operation]
    strategy_type = _TYPE_LABELS[draft.strategy_type]
    details = [f"已识别为〔{strategy_type}〕的〔{operation}〕请求"]
    if draft.strategy_id:
        details.append(f"策略 ID：{draft.strategy_id}")
    if draft.baseline_strategy_id:
        details.append(f"基线策略 ID：{draft.baseline_strategy_id}")
    if draft.objective:
        details.append(f"业务目标：{draft.objective}")
    constraints: list[str] = []
    if draft.max_bad_rate is not None:
        constraints.append(f"最大坏账率 {draft.max_bad_rate:.2%}")
    if draft.min_approval_rate is not None:
        constraints.append(f"最低通过率 {draft.min_approval_rate:.2%}")
    if constraints:
        details.append("业务约束：" + "、".join(constraints))
    if draft.profit is not None:
        details.append(
            "利润口径："
            f"EAD 列 {draft.profit['ead_col']}，PD 列 {draft.profit['pd_col']}，"
            f"年利率 {draft.profit['annual_rate']:.2%}，"
            f"资金成本率 {draft.profit['funding_rate']:.2%}，"
            f"LGD {draft.profit['lgd']:.2%}，"
            f"单笔运营成本 {draft.profit['operating_cost_per_loan']:g}，"
            f"期限 {draft.profit['term_months']} 个月"
        )
    if draft.economics_inputs is not None:
        details.append(_economics_confirmation(draft))
    if draft.candidate_design is not None:
        details.append(_candidate_design_confirmation(draft))
    if draft.strategy_spec is not None:
        details.append(_strategy_spec_confirmation(draft.strategy_spec))
    if draft.adoption_reason:
        details.append(f"采纳理由：{draft.adoption_reason}")
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；"
        "所有指标由平台确定性计算，采纳等治理动作仍需相应人工确认。"
    )
    return "；".join(details)


def _strategy_spec_confirmation(strategy_spec: Mapping[str, Any]) -> str:
    """Echo every executable rule/action so confirmation is not blind.

    The request row remains the authoritative canonical payload.  This text is
    a deterministic, human-reviewable projection: it contains no calculated
    metrics and does not ask the LLM to explain its own draft.
    """

    rules = list(strategy_spec.get("rules") or [])
    default_action = _compact_json(strategy_spec.get("default_action") or {})
    rendered = [
        f"规则草案：{len(rules)} 条规则，匹配方式 first_match，默认动作 {default_action}"
    ]
    for index, rule in enumerate(rules, start=1):
        rule_id = str(rule.get("rule_id") or f"rule-{index}")
        priority = rule.get("priority")
        condition = _compact_json(rule.get("condition") or {})
        action = _compact_json(rule.get("action") or {})
        rendered.append(
            f"规则 {index} [{rule_id}]（优先级 {priority}）：IF {condition} THEN {action}"
        )
    return "；".join(rendered)


def _compact_json(value: object) -> str:
    return json.dumps(
        _deep_thaw(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _complete(llm, *, prompt: str, caller: str):
    return llm.complete(
        system_prompt=_SYSTEM,
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=STRATEGY_REQUEST_JSON_SCHEMA,
        stream=False,
        caller=caller,
        prompt_name=STRATEGY_REQUEST_COMPILER_SYS.name,
        prompt_version=STRATEGY_REQUEST_COMPILER_SYS.version,
    )


def _validate_reply(
    raw: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> _ValidationOutcome:
    payload, error = load_json_object(raw)
    if payload is None:
        message = "模型返回的策略草案不是有效 JSON 对象，请重新说明策略请求。"
        return _ValidationOutcome(_clarification(message), False, error or message)
    return _validate_payload(payload, whitelist, target_col=target_col)


def _validate_payload(
    payload: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> _ValidationOutcome:
    if not isinstance(payload, Mapping):
        message = "策略请求必须是 JSON 对象，请重新说明。"
        return _invalid(message)
    if any(not isinstance(key, str) for key in payload):
        return _invalid("策略请求的字段名必须是文本，请重新说明。")

    if "clarification" in payload:
        if set(payload) != {"clarification"}:
            return _invalid("澄清问题不能和策略草案字段同时出现，请重新选择一种输出。")
        value = payload["clarification"]
        if not isinstance(value, str) or not value.strip():
            return _invalid("澄清问题必须是非空文本。")
        return _ValidationOutcome(
            _clarification(_chinese_clarification(value)),
            True,
        )

    request_kind = payload.get("request_kind", "strategy_lifecycle")
    if not isinstance(request_kind, str) or request_kind not in STRATEGY_REQUEST_KINDS:
        return _invalid(
            "不支持的 request_kind；只能是 strategy_lifecycle 或 standard_workflow。"
        )
    if request_kind == "standard_workflow":
        unexpected = sorted(set(payload) - _STANDARD_WORKFLOW_DRAFT_FIELDS)
        if unexpected:
            rendered = "、".join(f"「{field}」" for field in unexpected)
            return _invalid(f"标准 Workflow 请求包含不支持的字段 {rendered}，请删除后重新说明。")
        return _validate_standard_workflow_payload(
            payload,
            whitelist,
            target_col=target_col,
        )

    unexpected = sorted(set(payload) - _LIFECYCLE_DRAFT_FIELDS)
    if unexpected:
        rendered = "、".join(f"「{field}」" for field in unexpected)
        return _invalid(f"策略请求包含不支持的字段 {rendered}，请删除后重新说明。")
    if payload.get("request_kind") not in (None, "strategy_lifecycle"):
        return _invalid("策略生命周期请求的 request_kind 必须是 strategy_lifecycle。")
    missing = [field for field in ("operation", "strategy_type") if field not in payload]
    if missing:
        rendered = "、".join(missing)
        return _invalid(f"没有识别到必需字段 {rendered}，请补充策略操作和策略类型。")

    operation = payload["operation"]
    if not isinstance(operation, str) or operation not in STRATEGY_OPERATIONS:
        return _invalid(
            "不支持的策略操作；可选操作为：" + "、".join(STRATEGY_OPERATIONS) + "。"
        )
    strategy_type = payload["strategy_type"]
    if not isinstance(strategy_type, str) or strategy_type not in STRATEGY_TYPES:
        return _invalid(
            "不支持的策略类型；可选类型为：" + "、".join(STRATEGY_TYPES) + "。"
        )

    try:
        _validate_economics_field_ownership(payload, strategy_type=strategy_type)
        _validate_candidate_field_ownership(
            payload,
            operation=operation,
            strategy_type=strategy_type,
        )
        objective = _optional_text(payload, "objective")
        max_bad_rate = _optional_ratio(payload, "max_bad_rate")
        min_approval_rate = _optional_ratio(payload, "min_approval_rate")
        baseline_strategy_id = _optional_text(payload, "baseline_strategy_id")
        strategy_id = _optional_text(payload, "strategy_id")
        adoption_reason = _optional_adoption_reason(payload)
        profit = _optional_profit(payload, whitelist)
        economics_inputs = _optional_economics_inputs(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
        candidate_design = _optional_candidate_design(
            payload,
            operation=operation,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
        if candidate_design is not None:
            try:
                economics_inputs = normalize_candidate_economics_inputs(
                    strategy_type,
                    economics_inputs,
                    allowed_columns=whitelist,
                )
            except CandidateDesignError as exc:
                raise _DraftValidationError(
                    str(exc),
                    code=exc.code,
                    fields=exc.fields,
                ) from exc
        strategy_spec = _optional_strategy_spec(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
    except _DraftValidationError as exc:
        return _invalid(str(exc), code=exc.code, fields=exc.fields)

    draft = StrategyRequestDraft(
        operation=operation,
        strategy_type=strategy_type,
        objective=objective,
        max_bad_rate=max_bad_rate,
        min_approval_rate=min_approval_rate,
        baseline_strategy_id=baseline_strategy_id,
        strategy_id=strategy_id,
        adoption_reason=adoption_reason,
        profit=profit,
        economics_inputs=economics_inputs,
        candidate_design=candidate_design,
        strategy_spec=strategy_spec,
    )
    result = StrategyRequestCompilation(
        draft=draft,
        clarification=None,
        confirmation=strategy_request_confirmation_text(draft),
    )
    return _ValidationOutcome(result, True)


def _validate_standard_workflow_payload(
    payload: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> _ValidationOutcome:
    missing = [field for field in ("workflow", "workflow_inputs") if field not in payload]
    if missing:
        return _invalid("标准 Workflow 请求缺少字段：" + "、".join(missing) + "。")
    workflow = payload["workflow"]
    if not isinstance(workflow, str) or workflow not in STANDARD_STRATEGY_WORKFLOWS:
        return _invalid(
            "不支持的标准 Workflow；可选值为："
            + "、".join(STANDARD_STRATEGY_WORKFLOWS)
            + "。"
        )
    raw_inputs = payload["workflow_inputs"]
    if not isinstance(raw_inputs, Mapping):
        return _invalid("workflow_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        return _invalid("workflow_inputs 的字段名必须是文本。")
    try:
        if workflow == "profit_calc":
            normalized = _validate_profit_workflow_inputs(raw_inputs, whitelist)
        elif workflow == "roll_rate_matrix":
            normalized = _validate_roll_rate_workflow_inputs(raw_inputs, whitelist)
        else:
            normalized = _validate_pricing_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
    except _DraftValidationError as exc:
        return _invalid(str(exc))

    draft = StandardWorkflowRequestDraft(
        workflow=workflow,
        workflow_inputs=normalized,
    )
    return _ValidationOutcome(
        StrategyRequestCompilation(
            draft=draft,
            clarification=None,
            confirmation=strategy_request_confirmation_text(draft),
        ),
        True,
    )


def _validate_profit_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    allowed = {"segment_col", "ead_col", "pd_col", "profit_params"}
    _reject_workflow_fields(inputs, allowed, workflow="profit_calc")
    missing = sorted({"ead_col", "pd_col", "profit_params"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            "profit_calc 缺少字段：" + "、".join(missing) + "。"
        )
    params = inputs["profit_params"]
    if not isinstance(params, Mapping):
        raise _DraftValidationError("profit_calc 的 profit_params 必须是对象。")
    if any(not isinstance(key, str) for key in params):
        raise _DraftValidationError("profit_calc 的 profit_params 字段名必须是文本。")
    missing_params = sorted(_PROFIT_PARAMETER_FIELDS - set(params))
    unexpected_params = sorted(set(params) - _PROFIT_PARAMETER_FIELDS)
    if missing_params:
        raise _DraftValidationError(
            "profit_calc 的 profit_params 缺少字段："
            + "、".join(missing_params)
            + "。"
        )
    if unexpected_params:
        raise _DraftValidationError(
            "profit_calc 的 profit_params 包含不支持的字段："
            + "、".join(unexpected_params)
            + "。"
        )
    profit = _optional_profit(
        {
            "profit": {
                "ead_col": inputs["ead_col"],
                "pd_col": inputs["pd_col"],
                **dict(params),
            }
        },
        whitelist,
    )
    assert profit is not None
    normalized: dict[str, Any] = {
        "ead_col": profit.pop("ead_col"),
        "pd_col": profit.pop("pd_col"),
        "profit_params": profit,
    }
    if "segment_col" in inputs:
        normalized["segment_col"] = _workflow_column(
            inputs["segment_col"],
            name="profit_calc segment_col",
            whitelist=whitelist,
        )
    return normalized


def _validate_roll_rate_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    allowed = {
        "id_col",
        "time_col",
        "status_col",
        "states",
        "balance_col",
        "observation_semantics",
    }
    _reject_workflow_fields(inputs, allowed, workflow="roll_rate_matrix")
    required = {"id_col", "time_col", "status_col", "states"}
    missing = sorted(required - set(inputs))
    if missing:
        raise _DraftValidationError(
            "roll_rate_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    normalized = {
        key: _workflow_column(inputs[key], name=f"roll_rate_matrix {key}", whitelist=whitelist)
        for key in ("id_col", "time_col", "status_col")
    }
    if len(set(normalized.values())) != len(normalized):
        raise _DraftValidationError("roll_rate_matrix 的 id_col、time_col、status_col 必须互不相同。")
    states = inputs["states"]
    if (
        not isinstance(states, Sequence)
        or isinstance(states, str | bytes | bytearray)
        or not 2 <= len(states) <= 50
    ):
        raise _DraftValidationError("roll_rate_matrix states 必须是包含 2 到 50 个状态的有序数组。")
    normalized_states = [
        _required_text(state, name="roll_rate_matrix states 状态") for state in states
    ]
    if len(set(normalized_states)) != len(normalized_states):
        raise _DraftValidationError("roll_rate_matrix states 不能包含重复状态。")
    normalized["states"] = normalized_states
    semantics = inputs.get("observation_semantics", "adjacent_observation")
    if semantics != "adjacent_observation":
        raise _DraftValidationError(
            "roll_rate_matrix observation_semantics 只能是 adjacent_observation；"
            "固定月末快照迁徙应使用 portfolio Workflow。"
        )
    normalized["observation_semantics"] = semantics
    if "balance_col" in inputs:
        balance_col = _workflow_column(
            inputs["balance_col"],
            name="roll_rate_matrix balance_col",
            whitelist=whitelist,
        )
        if balance_col in {
            normalized["id_col"],
            normalized["time_col"],
            normalized["status_col"],
        }:
            raise _DraftValidationError("roll_rate_matrix balance_col 不能复用 ID、时间或状态列。")
        normalized["balance_col"] = balance_col
    return normalized


def _validate_pricing_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    allowed = {
        "score_col",
        "pd_col",
        "target_col",
        "band_edges",
        "n_bands",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
        "strategy_id",
        "drop_nan_labels",
    }
    _reject_workflow_fields(inputs, allowed, workflow="limit_pricing_matrix")
    required = {
        "score_col",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
    }
    missing = sorted(required - set(inputs))
    if missing:
        raise _DraftValidationError(
            "limit_pricing_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    has_pd = "pd_col" in inputs
    has_target = "target_col" in inputs
    if has_pd == has_target:
        raise _DraftValidationError(
            "limit_pricing_matrix 的 pd_col 与 target_col 必须且只能二选一。"
        )
    has_edges = "band_edges" in inputs
    has_band_count = "n_bands" in inputs
    if has_edges == has_band_count:
        raise _DraftValidationError(
            "limit_pricing_matrix 的 band_edges 与 n_bands 必须且只能二选一。"
        )

    normalized: dict[str, Any] = {
        "score_col": _workflow_column(
            inputs["score_col"],
            name="limit_pricing_matrix score_col",
            whitelist=whitelist,
        ),
        "limit_grid": _number_sequence(
            inputs["limit_grid"],
            name="limit_pricing_matrix limit_grid",
            minimum=0,
            exclusive_minimum=True,
            maximum_items=50,
        ),
        "rate_grid": _number_sequence(
            inputs["rate_grid"],
            name="limit_pricing_matrix rate_grid",
            minimum=0,
            maximum=1,
            maximum_items=50,
        ),
        "lgd": _bounded_number(inputs["lgd"], name="limit_pricing_matrix lgd", maximum=1),
        "funding_rate": _bounded_number(
            inputs["funding_rate"],
            name="limit_pricing_matrix funding_rate",
            maximum=1,
        ),
        "cost_per_loan": _bounded_number(
            inputs["cost_per_loan"],
            name="limit_pricing_matrix cost_per_loan",
        ),
        "el_ead_max": _bounded_number(
            inputs["el_ead_max"],
            name="limit_pricing_matrix el_ead_max",
            maximum=1,
        ),
    }
    term_months = inputs["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or not 1 <= term_months <= 600
    ):
        raise _DraftValidationError("limit_pricing_matrix term_months 必须是 1 到 600 的整数。")
    normalized["term_months"] = term_months

    if has_pd:
        normalized["pd_col"] = _workflow_column(
            inputs["pd_col"],
            name="limit_pricing_matrix pd_col",
            whitelist=whitelist,
        )
    else:
        requested_target = _required_text(
            inputs["target_col"],
            name="limit_pricing_matrix target_col",
        )
        if target_col is None or requested_target != target_col:
            raise _DraftValidationError(
                "limit_pricing_matrix target_col 必须与任务当前确认的目标列一致。"
            )
        normalized["target_col"] = requested_target

    if has_edges:
        edges = _number_sequence(
            inputs["band_edges"],
            name="limit_pricing_matrix band_edges",
            minimum=None,
            maximum_items=51,
            minimum_items=2,
        )
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise _DraftValidationError("limit_pricing_matrix band_edges 必须严格递增。")
        normalized["band_edges"] = edges
        band_count = len(edges) - 1
    else:
        n_bands = inputs["n_bands"]
        if isinstance(n_bands, bool) or not isinstance(n_bands, int) or not 1 <= n_bands <= 20:
            raise _DraftValidationError("limit_pricing_matrix n_bands 必须是 1 到 20 的整数。")
        normalized["n_bands"] = n_bands
        band_count = n_bands
    if band_count * len(normalized["limit_grid"]) * len(normalized["rate_grid"]) > 2000:
        raise _DraftValidationError("limit_pricing_matrix 网格最多允许 2000 个组合。")
    if "strategy_id" in inputs:
        normalized["strategy_id"] = _required_text(
            inputs["strategy_id"],
            name="limit_pricing_matrix strategy_id",
        )
    if "drop_nan_labels" in inputs:
        if not isinstance(inputs["drop_nan_labels"], bool):
            raise _DraftValidationError("limit_pricing_matrix drop_nan_labels 必须是布尔值。")
        if has_pd:
            raise _DraftValidationError(
                "limit_pricing_matrix 使用 pd_col 时不会读取标签，"
                "请删除未使用的 drop_nan_labels。"
            )
        normalized["drop_nan_labels"] = inputs["drop_nan_labels"]
    return normalized


def _reject_workflow_fields(
    inputs: Mapping[str, Any],
    allowed: set[str],
    *,
    workflow: str,
) -> None:
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise _DraftValidationError(
            f"{workflow} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )


def _workflow_column(
    value: object,
    *,
    name: str,
    whitelist: tuple[str, ...],
) -> str:
    column = _required_text(value, name=name)
    if column not in whitelist:
        raise _DraftValidationError(f"{name} 使用了数据集中不存在的列「{column}」。")
    return column


def _number_sequence(
    value: object,
    *,
    name: str,
    minimum: float | None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    minimum_items: int = 1,
    maximum_items: int,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not minimum_items <= len(value) <= maximum_items
    ):
        raise _DraftValidationError(
            f"{name} 必须是包含 {minimum_items} 到 {maximum_items} 个有限数字的数组。"
        )
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise _DraftValidationError(f"{name} 只能包含有限数字。")
        number = float(item)
        if not math.isfinite(number):
            raise _DraftValidationError(f"{name} 只能包含有限数字。")
        if minimum is not None and (
            number < minimum or (exclusive_minimum and number == minimum)
        ):
            relation = "大于" if exclusive_minimum else "大于等于"
            raise _DraftValidationError(f"{name} 中每个值都必须{relation} {minimum:g}。")
        if maximum is not None and number > maximum:
            raise _DraftValidationError(f"{name} 中每个值都必须小于等于 {maximum:g}。")
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise _DraftValidationError(f"{name} 不能包含重复值。")
    return numbers


def _standard_workflow_confirmation_text(
    draft: StandardWorkflowRequestDraft,
) -> str:
    inputs = draft.workflow_inputs
    if draft.workflow == "profit_calc":
        params = inputs["profit_params"]
        details = [
            "已识别为〔标准利润分析 Workflow〕",
            f"EAD 列 {inputs['ead_col']}，PD 列 {inputs['pd_col']}",
            "分析范围："
            + (f"按 {inputs['segment_col']} 分组" if "segment_col" in inputs else "全样本"),
            (
                f"年利率 {params['annual_rate']:.2%}，资金成本率 {params['funding_rate']:.2%}，"
                f"LGD {params['lgd']:.2%}，单笔成本 {params['operating_cost_per_loan']:g}，"
                f"期限 {params['term_months']} 个月"
            ),
        ]
    elif draft.workflow == "roll_rate_matrix":
        details = [
            "已识别为〔标准滚动率矩阵 Workflow〕",
            (
                f"客户 ID 列 {inputs['id_col']}，时间列 {inputs['time_col']}，"
                f"状态列 {inputs['status_col']}"
            ),
            "状态顺序：" + " → ".join(inputs["states"]),
            "观测口径：相邻观测记录，不等同于固定月末快照迁徙",
        ]
        if "balance_col" in inputs:
            details.append(f"余额加权列：{inputs['balance_col']}")
    else:
        risk_source = (
            f"PD 列 {inputs['pd_col']}"
            if "pd_col" in inputs
            else f"目标列 {inputs['target_col']}"
        )
        banding = (
            "分箱边界 " + "、".join(f"{value:g}" for value in inputs["band_edges"])
            if "band_edges" in inputs
            else f"等频分为 {inputs['n_bands']} 档"
        )
        details = [
            "已识别为〔标准额度定价矩阵 Workflow〕",
            f"评分列 {inputs['score_col']}，风险来源 {risk_source}，{banding}",
            "额度网格：" + "、".join(f"{value:,.12g}" for value in inputs["limit_grid"]),
            "利率网格：" + "、".join(f"{value:.2%}" for value in inputs["rate_grid"]),
            (
                f"LGD {inputs['lgd']:.2%}，资金成本率 {inputs['funding_rate']:.2%}，"
                f"期限 {inputs['term_months']} 个月，单笔成本 {inputs['cost_per_loan']:g}，"
                f"EL/EAD 上限 {inputs['el_ead_max']:.2%}"
            ),
        ]
        if "target_col" in inputs:
            details.append(
                "标签缺失处理："
                + (
                    "按明确授权丢弃 NaN 标签行"
                    if inputs.get("drop_nan_labels")
                    else "不自动丢弃 NaN 标签行"
                )
            )
        if "strategy_id" in inputs:
            details.append(f"关联策略 ID：{inputs['strategy_id']}")
        details.append("平台先计算完整矩阵；接受或导出矩阵仍需第二次明确确认")
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
    )
    return "；".join(details)


class _DraftValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_strategy_request",
        fields: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(dict.fromkeys(str(field) for field in fields))


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise _DraftValidationError(f"{key} 必须是非空文本，请重新说明。")
    return value.strip()


def _optional_ratio(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise _DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    return number


def _optional_adoption_reason(payload: Mapping[str, Any]) -> str | None:
    if "adoption_reason" not in payload:
        return None
    try:
        return normalize_adoption_reason(payload["adoption_reason"])
    except AdoptionReasonError as exc:
        raise _DraftValidationError(str(exc)) from exc


def _optional_profit(
    payload: Mapping[str, Any], whitelist: tuple[str, ...]
) -> dict[str, Any] | None:
    if "profit" not in payload:
        return None
    profit = payload["profit"]
    if not isinstance(profit, Mapping):
        raise _DraftValidationError("利润参数 profit 必须是一个对象。")
    if any(not isinstance(key, str) for key in profit):
        raise _DraftValidationError("利润参数字段名必须是文本。")
    missing = sorted(_PROFIT_FIELDS - set(profit))
    unexpected = sorted(set(profit) - _PROFIT_FIELDS)
    if missing:
        raise _DraftValidationError(
            "利润参数缺少字段：" + "、".join(missing) + "。"
        )
    if unexpected:
        raise _DraftValidationError(
            "利润参数包含不支持的字段：" + "、".join(unexpected) + "。"
        )
    ead_col = _required_text(profit["ead_col"], name="利润 EAD 列 ead_col")
    pd_col = _required_text(profit["pd_col"], name="利润 PD 列 pd_col")
    for name, column in (("ead_col", ead_col), ("pd_col", pd_col)):
        if column not in whitelist:
            raise _DraftValidationError(
                f"利润参数 {name} 使用了数据集中不存在的列「{column}」，请从列白名单选择。"
            )
    annual_rate = _bounded_number(profit["annual_rate"], name="利润 annual_rate", maximum=1)
    funding_rate = _bounded_number(
        profit["funding_rate"], name="利润 funding_rate", maximum=1
    )
    lgd = _bounded_number(profit["lgd"], name="利润 lgd", maximum=1)
    operating_cost = _bounded_number(
        profit["operating_cost_per_loan"],
        name="利润 operating_cost_per_loan",
    )
    term_months = profit["term_months"]
    if isinstance(term_months, bool) or not isinstance(term_months, int) or term_months < 1:
        raise _DraftValidationError("利润 term_months 必须是大于等于 1 的整数。")
    return {
        "ead_col": ead_col,
        "pd_col": pd_col,
        "annual_rate": annual_rate,
        "funding_rate": funding_rate,
        "lgd": lgd,
        "operating_cost_per_loan": operating_cost,
        "term_months": term_months,
    }


def _validate_economics_field_ownership(
    payload: Mapping[str, Any], *, strategy_type: str
) -> None:
    if "profit" in payload and strategy_type not in {"approval", "reject"}:
        raise _DraftValidationError(
            "profit 只适用于审批或拒绝策略；额度和定价策略请使用 economics_inputs，"
            "分群策略不接受经济参数。"
        )
    if "economics_inputs" in payload and strategy_type not in {"limit", "pricing"}:
        raise _DraftValidationError(
            "economics_inputs 只适用于额度或定价策略；审批和拒绝策略请使用 profit，"
            "分群策略不接受经济参数。"
        )


def _validate_candidate_field_ownership(
    payload: Mapping[str, Any],
    *,
    operation: str,
    strategy_type: str,
) -> None:
    has_candidate = "candidate_design" in payload
    has_spec = "strategy_spec" in payload
    if has_candidate and has_spec:
        raise _DraftValidationError(
            "candidate_design 与 strategy_spec 必须二选一；LLM 不得同时提交候选输入和规则结果。",
            code="candidate_spec_mutually_exclusive",
            fields=("candidate_design", "strategy_spec"),
        )
    if has_candidate and (
        operation != "develop"
        or strategy_type not in {"limit", "pricing", "segmentation"}
    ):
        raise _DraftValidationError(
            "candidate_design 只适用于 limit、pricing、segmentation 的 develop 请求。",
            code="candidate_design_not_allowed",
            fields=("candidate_design",),
        )
    if has_spec and strategy_type in {"limit", "pricing", "segmentation"}:
        raise _DraftValidationError(
            "非审批策略的 Strategy DSL 必须由平台候选设计工具确定性生成；"
            "LLM 不得提交 strategy_spec、动作值或推荐结果。",
            code="llm_strategy_spec_forbidden",
            fields=("strategy_spec",),
        )
    if (
        operation == "develop"
        and strategy_type in {"limit", "pricing", "segmentation"}
        and not has_candidate
    ):
        raise _DraftValidationError(
            f"开发{_TYPE_LABELS[strategy_type]}需要 candidate_design；"
            "请补充候选列、候选网格和必要业务约束，平台再确定性生成规则。",
            code="candidate_design_required",
            fields=("candidate_design",),
        )


def _optional_economics_inputs(
    payload: Mapping[str, Any],
    *,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "economics_inputs" not in payload:
        return None
    raw_inputs = payload["economics_inputs"]
    if not isinstance(raw_inputs, Mapping):
        raise _DraftValidationError("经济参数 economics_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        raise _DraftValidationError("经济参数 economics_inputs 的字段名必须是文本。")

    names = (
        _LIMIT_ECONOMICS_NAMES
        if strategy_type == "limit"
        else _PRICING_ECONOMICS_NAMES
    )
    allowed_fields = {
        key for name in names for key in (f"{name}_col", f"{name}_value")
    }
    unexpected = sorted(set(raw_inputs) - allowed_fields)
    if unexpected:
        raise _DraftValidationError(
            f"{_TYPE_LABELS[strategy_type]}经济参数包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )

    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        has_column = column_key in raw_inputs
        has_value = value_key in raw_inputs
        if has_column and has_value:
            raise _DraftValidationError(
                f"经济参数 {name} 必须在 {column_key} 和 {value_key} 中二选一，不能同时提供。",
                code="candidate_economics_ambiguous",
                fields=(column_key, value_key),
            )
        if not has_column and not has_value:
            missing.append(f"{column_key}/{value_key}")
            continue
        if has_column:
            column = _required_text(
                raw_inputs[column_key],
                name=f"经济参数 {column_key}",
            )
            if column not in whitelist:
                raise _DraftValidationError(
                    f"经济参数 {column_key} 使用了数据集中不存在或不可用于策略的列"
                    f"「{column}」，请从列白名单选择。"
                )
            normalized[column_key] = column
            continue
        normalized[value_key] = _economics_value(name, raw_inputs[value_key])

    if missing:
        raise _DraftValidationError(
            f"{_TYPE_LABELS[strategy_type]}经济参数不完整，缺少："
            + "、".join(missing)
            + "。",
            code="candidate_economics_incomplete",
            fields=missing,
        )
    return normalized


def _economics_value(name: str, value: object) -> float:
    label = _ECONOMICS_LABELS[name]
    if name == "term_months":
        number = _bounded_number(value, name=f"经济参数 {label}")
        if number <= 0:
            raise _DraftValidationError(
                f"经济参数 {label} 必须是大于 0 的有限数字。"
            )
        return number
    return _bounded_number(
        value,
        name=f"经济参数 {label}",
        maximum=_ECONOMICS_VALUE_MAXIMUMS.get(name),
    )


def _economics_confirmation(draft: StrategyRequestDraft) -> str:
    assert draft.economics_inputs is not None
    names = (
        _LIMIT_ECONOMICS_NAMES
        if draft.strategy_type == "limit"
        else _PRICING_ECONOMICS_NAMES
    )
    items: list[str] = []
    for name in names:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        label = _ECONOMICS_LABELS[name]
        if column_key in draft.economics_inputs:
            items.append(f"{label} 取数据列 {draft.economics_inputs[column_key]}")
        else:
            value = draft.economics_inputs[value_key]
            if name in _ECONOMICS_VALUE_MAXIMUMS:
                items.append(f"{label} 取固定值 {value:.2%}")
            elif name == "term_months":
                items.append(f"{label} 取固定值 {value:g} 个月")
            else:
                items.append(f"{label} 取固定值 {value:g}")
    return f"{_TYPE_LABELS[draft.strategy_type]}经济参数：" + "，".join(items)


def _candidate_design_confirmation(draft: StrategyRequestDraft) -> str:
    assert draft.candidate_design is not None
    design = draft.candidate_design
    if draft.strategy_type == "limit":
        details = (
            f"评分列 {design['score_col']}，固定等频 {design['n_bands']} 箱，"
            "候选额度 "
            + "、".join(f"{value:g}" for value in design["limit_grid"])
            + f"，单户预期损失预算 {design['max_expected_loss_per_account']:g}"
        )
    elif draft.strategy_type == "pricing":
        details = (
            f"评分列 {design['score_col']}，固定等频 {design['n_bands']} 箱，"
            "候选年利率 "
            + "、".join(f"{value:.2%}" for value in design["rate_grid"])
            + f"，最小 ROA {design['min_roa']:.2%}"
        )
    else:
        details = (
            f"单变量列 {design['feature_col']}，固定等频 {design['n_bands']} 箱，"
            "风险标签由平台按样本坏率稳定生成"
        )
    return (
        f"候选设计输入：{details}；缺失策略 {design['missing_policy']}。"
        "此处只确认搜索空间和业务口径，推荐动作、规则与指标尚未生成，"
        "将由平台确定性计算。"
    )


def _optional_candidate_design(
    payload: Mapping[str, Any],
    *,
    operation: str,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "candidate_design" not in payload:
        return None
    if operation != "develop":
        raise _DraftValidationError(
            "candidate_design 只适用于 develop 请求。",
            code="candidate_design_not_allowed",
            fields=("candidate_design",),
        )
    try:
        return normalize_candidate_design(
            strategy_type,
            payload["candidate_design"],
            allowed_columns=whitelist,
        )
    except CandidateDesignError as exc:
        raise _DraftValidationError(
            str(exc),
            code=exc.code,
            fields=exc.fields,
        ) from exc


def _optional_strategy_spec(
    payload: Mapping[str, Any],
    *,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "strategy_spec" not in payload:
        return None
    if strategy_type not in {"approval", "reject"}:
        raise _DraftValidationError(
            "非审批策略的 strategy_spec 必须由平台确定性生成，LLM 不得提交。",
            code="llm_strategy_spec_forbidden",
            fields=("strategy_spec",),
        )
    raw_spec = payload["strategy_spec"]
    if not isinstance(raw_spec, Mapping):
        raise _DraftValidationError("策略规则草案 strategy_spec 必须是一个对象。")
    raw_metadata = raw_spec.get("metadata", {})
    if raw_metadata not in ({}, {"lineage": {}}):
        raise _DraftValidationError(
            "策略规则草案 metadata 由平台生成，LLM 不得写入指标结果或其他元数据。"
        )
    try:
        parsed = parse_strategy_spec(raw_spec)
    except (StrategyError, TypeError, ValueError) as exc:
        raise _DraftValidationError(
            "策略规则草案格式或取值无效，请检查规则条件、优先级和动作。"
        ) from exc
    if parsed.strategy_type != strategy_type:
        raise _DraftValidationError(
            "strategy_spec 的 strategy_type 必须与请求中的策略类型一致。"
        )
    unknown_columns = sorted(
        {
            field
            for rule in parsed.rules
            for field in _condition_fields(rule.condition)
            if field not in whitelist
        }
    )
    if unknown_columns:
        rendered = "、".join(f"「{column}」" for column in unknown_columns)
        raise _DraftValidationError(
            f"策略条件使用了数据集中不存在的列 {rendered}，请从列白名单选择。"
        )
    return parsed.to_dict()


def _condition_fields(condition: Mapping[str, Any]) -> tuple[str, ...]:
    op = condition["op"]
    if op in {"compare", "between", "is_null", "is_not_null"}:
        return (condition["field"],)
    if op in {"and", "or", "n_of_k"}:
        return tuple(
            field
            for argument in condition["args"]
            for field in _condition_fields(argument)
        )
    if op == "not":
        return _condition_fields(condition["arg"])
    return ()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _DraftValidationError(f"{name} 必须是非空文本。")
    return value.strip()


def _bounded_number(
    value: object,
    *,
    name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _DraftValidationError(f"{name} 必须是有限数字。")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (
        maximum is not None and number > maximum
    ):
        if maximum is None:
            raise _DraftValidationError(f"{name} 必须是大于等于 0 的有限数字。")
        raise _DraftValidationError(f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。")
    return number


def _column_whitelist(
    allowed_columns: Iterable[str] | None,
) -> tuple[str, ...]:
    if allowed_columns is None:
        return ()
    if isinstance(allowed_columns, str):
        values = (allowed_columns,)
    else:
        try:
            values = tuple(allowed_columns)
        except TypeError:
            return ()
    return tuple(sorted({column for column in values if isinstance(column, str)}))


def _normalized_target_col(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _user_prompt(
    utterance: str,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> str:
    return (
        "【数据集列白名单】\n"
        f"{json.dumps(list(whitelist), ensure_ascii=False)}\n"
        "【任务当前目标列（仅可作为 limit_pricing_matrix 的 target_col 风险来源，"
        "禁止用于策略规则）】\n"
        f"{json.dumps(target_col, ensure_ascii=False)}\n"
        "【用户策略请求】\n"
        f"{utterance}\n"
        "只输出结构化策略草案或一个中文 clarification，不要输出任何指标结果。"
        "对于 limit/pricing/segmentation 的 develop 请求，只能抽取 candidate_design "
        "搜索空间与用户明确给出的 economics_inputs；禁止输出 strategy_spec、规则、"
        "动作、默认动作、推荐值或计算指标。缺少必要经济口径时只返回 clarification。"
    )


def _repair_prompt(prompt: str, *, raw: object, error: str) -> str:
    if isinstance(raw, Mapping):
        raw_text = json.dumps(raw, ensure_ascii=False, default=str)
    else:
        raw_text = str(raw)
    raw_text = raw_text[:4000]
    return (
        f"{prompt}\n\n"
        "【上一次输出未通过平台校验】\n"
        f"错误：{error}\n"
        f"上一次输出：{raw_text}\n"
        "这是唯一一次修复机会。请删除未知字段、修正类型/范围/列名；"
        "不能确定时只返回中文 clarification。仍然禁止输出任何指标结果。"
    )


def _invalid(
    message: str,
    *,
    code: str = "invalid_strategy_request",
    fields: Iterable[str] = (),
) -> _ValidationOutcome:
    return _ValidationOutcome(
        _clarification(message, code=code, fields=fields),
        False,
        message,
    )


def _clarification(
    message: str,
    *,
    code: str = "clarification_required",
    fields: Iterable[str] = (),
) -> StrategyRequestCompilation:
    return StrategyRequestCompilation(
        draft=None,
        clarification=message,
        confirmation=None,
        clarification_code=code,
        clarification_fields=tuple(dict.fromkeys(str(field) for field in fields)),
    )


def _chinese_clarification(message: str) -> str:
    normalized = message.strip()
    if _CJK_RE.search(normalized):
        return normalized
    return "请补充更明确的策略操作、策略类型和相关策略对象。"


_OPERATION_LABELS = {
    "develop": "开发",
    "analyze": "分析",
    "backtest": "回测",
    "apply": "应用",
    "compare": "对比",
    "adopt": "采纳",
    "report": "生成报告",
    "monitor": "监控",
    "mine_rules": "规则挖掘",
}
_TYPE_LABELS = {
    "approval": "审批策略",
    "reject": "拒绝策略",
    "limit": "额度策略",
    "pricing": "定价策略",
    "segmentation": "分群策略",
}


__all__ = [
    "CompiledStrategyRequestDraft",
    "STANDARD_STRATEGY_WORKFLOWS",
    "STRATEGY_REQUEST_KINDS",
    "STRATEGY_OPERATIONS",
    "STRATEGY_REQUEST_JSON_SCHEMA",
    "STRATEGY_TYPES",
    "StrategyRequestCompilation",
    "StrategyRequestDraft",
    "StandardWorkflowRequestDraft",
    "compile_strategy_request",
    "strategy_request_confirmation_text",
    "validate_strategy_request",
]
