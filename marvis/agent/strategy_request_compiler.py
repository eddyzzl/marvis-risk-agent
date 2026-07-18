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

_OPTIONAL_DRAFT_FIELDS = {
    "objective",
    "max_bad_rate",
    "min_approval_rate",
    "baseline_strategy_id",
    "strategy_id",
    "adoption_reason",
    "profit",
    "economics_inputs",
    "strategy_spec",
}
_DRAFT_FIELDS = {"operation", "strategy_type"} | _OPTIONAL_DRAFT_FIELDS
_PROFIT_FIELDS = {
    "ead_col",
    "pd_col",
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
}
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


STRATEGY_REQUEST_JSON_SCHEMA = {
    "name": "strategy_request_draft",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
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
            "strategy_spec": {"type": "object"},
            "clarification": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
        "oneOf": [
            {"required": ["operation", "strategy_type"]},
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
    strategy_spec: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.profit is not None:
            object.__setattr__(self, "profit", _deep_freeze(self.profit))
        if self.economics_inputs is not None:
            object.__setattr__(
                self,
                "economics_inputs",
                _deep_freeze(self.economics_inputs),
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
class StrategyRequestCompilation:
    """A validated draft awaiting confirmation, or a Chinese clarification."""

    draft: StrategyRequestDraft | None
    clarification: str | None
    confirmation: str | None

    @property
    def validated_draft(self) -> StrategyRequestDraft | None:
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
        return {
            "draft": None if self.draft is None else self.draft.to_dict(),
            "clarification": self.clarification,
            "confirmation": self.confirmation,
        }


@dataclass(frozen=True)
class _ValidationOutcome:
    result: StrategyRequestCompilation
    accepted: bool
    error: str | None = None


def compile_strategy_request(
    utterance: str,
    *,
    allowed_columns: Iterable[str] | None,
    llm,
    caller: str = "strategy_request_compiler",
) -> StrategyRequestCompilation:
    """Compile one utterance with at most one LLM-format repair attempt."""

    if not isinstance(utterance, str) or not utterance.strip():
        return _clarification("请说明希望执行的策略操作和策略类型。")
    whitelist = _column_whitelist(allowed_columns)
    prompt = _user_prompt(utterance.strip(), whitelist)
    try:
        raw = _complete(llm, prompt=prompt, caller=caller)
    except Exception:
        return _clarification(
            "当前暂时无法解析策略请求，请稍后重试或直接说明操作、策略类型和策略对象。"
        )
    outcome = _validate_reply(raw, whitelist)
    if outcome.accepted:
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
    return _validate_reply(repaired, whitelist).result


def validate_strategy_request(
    payload: object,
    *,
    allowed_columns: Iterable[str] | None,
) -> StrategyRequestCompilation:
    """Validate an already parsed LLM payload without invoking an LLM."""

    return _validate_payload(payload, _column_whitelist(allowed_columns)).result


def strategy_request_confirmation_text(draft: StrategyRequestDraft) -> str:
    """Render a plain-Chinese echo of the request before any workflow runs."""

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


def _validate_reply(raw: object, whitelist: tuple[str, ...]) -> _ValidationOutcome:
    payload, error = load_json_object(raw)
    if payload is None:
        message = "模型返回的策略草案不是有效 JSON 对象，请重新说明策略请求。"
        return _ValidationOutcome(_clarification(message), False, error or message)
    return _validate_payload(payload, whitelist)


def _validate_payload(
    payload: object,
    whitelist: tuple[str, ...],
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

    unexpected = sorted(set(payload) - _DRAFT_FIELDS)
    if unexpected:
        rendered = "、".join(f"「{field}」" for field in unexpected)
        return _invalid(f"策略请求包含不支持的字段 {rendered}，请删除后重新说明。")
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
        strategy_spec = _optional_strategy_spec(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
    except _DraftValidationError as exc:
        return _invalid(str(exc))

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
        strategy_spec=strategy_spec,
    )
    result = StrategyRequestCompilation(
        draft=draft,
        clarification=None,
        confirmation=strategy_request_confirmation_text(draft),
    )
    return _ValidationOutcome(result, True)


class _DraftValidationError(ValueError):
    pass


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
                f"经济参数 {name} 必须在 {column_key} 和 {value_key} 中二选一，不能同时提供。"
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
            + "。"
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


def _optional_strategy_spec(
    payload: Mapping[str, Any],
    *,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "strategy_spec" not in payload:
        return None
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


def _user_prompt(utterance: str, whitelist: tuple[str, ...]) -> str:
    return (
        "【数据集列白名单】\n"
        f"{json.dumps(list(whitelist), ensure_ascii=False)}\n"
        "【用户策略请求】\n"
        f"{utterance}\n"
        "只输出结构化策略草案或一个中文 clarification，不要输出任何指标结果。"
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


def _invalid(message: str) -> _ValidationOutcome:
    return _ValidationOutcome(_clarification(message), False, message)


def _clarification(message: str) -> StrategyRequestCompilation:
    return StrategyRequestCompilation(
        draft=None,
        clarification=message,
        confirmation=None,
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
    "STRATEGY_OPERATIONS",
    "STRATEGY_REQUEST_JSON_SCHEMA",
    "STRATEGY_TYPES",
    "StrategyRequestCompilation",
    "StrategyRequestDraft",
    "compile_strategy_request",
    "strategy_request_confirmation_text",
    "validate_strategy_request",
]
