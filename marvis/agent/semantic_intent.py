"""Strict, independent two-pass routing for Agent-mode top-level intent.

The LLM only chooses a bounded route.  Existing deterministic compilers,
validators and typed controls continue to own every executable request.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from marvis.llm_prompts import (
    TOP_LEVEL_INTENT_REPAIR_SYS,
    TOP_LEVEL_INTENT_REVIEW_SYS,
    TOP_LEVEL_INTENT_ROUTER_SYS,
)


INTENT_NONE = "none"
INTENT_CURRENT_WORKFLOW = "current_workflow"
INTENT_RISK_PROFITABILITY = "risk_profitability"
INTENT_RISK_VTG_TERMINAL = "risk_vtg_terminal"
INTENT_RISK_STANDARD_VINTAGE = "risk_standard_vintage"
INTENT_ADHOC_QUERY = "adhoc_query"
INTENT_DATASET_TRANSFORM = "dataset_transform"
INTENT_DATASET_EXPORT = "dataset_export"
INTENT_DATASET_ANALYSIS = "dataset_analysis"
INTENT_DATASET_JOIN = "dataset_join"
INTENT_STRATEGY_SAMPLE_BINDING = "strategy_sample_binding"
INTENT_STRATEGY_WORKFLOW = "strategy_workflow"
INTENT_ADHOC_CONFIRM = "adhoc_confirm"
INTENT_ADHOC_REJECT = "adhoc_reject"
INTENT_ADHOC_REVISE = "adhoc_revise"

ALL_SEMANTIC_INTENTS = frozenset(
    {
        INTENT_NONE,
        INTENT_CURRENT_WORKFLOW,
        INTENT_RISK_PROFITABILITY,
        INTENT_RISK_VTG_TERMINAL,
        INTENT_RISK_STANDARD_VINTAGE,
        INTENT_ADHOC_QUERY,
        INTENT_DATASET_TRANSFORM,
        INTENT_DATASET_EXPORT,
        INTENT_DATASET_ANALYSIS,
        INTENT_DATASET_JOIN,
        INTENT_STRATEGY_SAMPLE_BINDING,
        INTENT_STRATEGY_WORKFLOW,
        INTENT_ADHOC_CONFIRM,
        INTENT_ADHOC_REJECT,
        INTENT_ADHOC_REVISE,
    }
)

_INTENT_DEFINITIONS = {
    INTENT_NONE: "语义不清、相互冲突、超出允许范围，或没有明确请求。",
    INTENT_CURRENT_WORKFLOW: (
        "回答或继续当前任务已经展示的设置、材料、字段、角色、参数或步骤；"
        "不是另起一个数据、策略、问数或风险分析请求。"
    ),
    INTENT_RISK_PROFITABILITY: "选择风险收益测算，关注收入、成本、利润或经济性拆解。",
    INTENT_RISK_VTG_TERMINAL: "选择 VTG 终值与年化不良风险分析。",
    INTENT_RISK_STANDARD_VINTAGE: "选择标准 Vintage 账龄、迁徙或不良表现分析。",
    INTENT_ADHOC_QUERY: (
        "针对当前数据提出临时分组、筛选、计数、均值、比例、坏率等问数请求。"
    ),
    INTENT_DATASET_TRANSFORM: (
        "要求改变当前数据，如删列、改名、填补、类型转换、筛行、派生或去重。"
    ),
    INTENT_DATASET_EXPORT: "要求下载或导出当前原始数据集；不是导出报告或分析结果。",
    INTENT_DATASET_ANALYSIS: (
        "要求生成当前数据的概况、目标分布、缺失、字段分布或相关性诊断。"
    ),
    INTENT_DATASET_JOIN: (
        "仅在 data_join 任务且当前至少有两张可用数据表时，开始或继续当前数据拼接探索："
        "读取可用表的结构、提出申请主表/特征表角色与目标列建议，并停在人工确认门；"
        "它不授权执行最终拼接，也不属于其他任务的数据处理请求。"
    ),
    INTENT_STRATEGY_SAMPLE_BINDING: (
        "仅在 strategy 任务且上下文给出唯一可绑定样本与唯一二元目标候选时，"
        "用户明确要求把该现有材料绑定为当前策略样本，并确认该目标/坏样本字段；"
        "只更新受认证的 DataWorkspace 绑定，不创建策略计划，不推断坏样本取值、"
        "缺失标签政策、人群、分区、时间窗或成熟度。"
    ),
    INTENT_STRATEGY_WORKFLOW: (
        "在策略任务中发起策略设计、挖掘、回测、比较、监控、证据或交付工作流。"
    ),
    INTENT_ADHOC_CONFIRM: "明确接受当前待确认的问数口径并授权执行。",
    INTENT_ADHOC_REJECT: "明确取消、拒绝或停止当前待确认的问数口径。",
    INTENT_ADHOC_REVISE: "明确修改当前待确认的问数分组、指标、筛选、月份或排序口径。",
}

_FIELDS = (
    "intent",
    "evidence_quote",
    "reason",
    "confidence",
    "is_question",
    "is_conditional",
    "requests_change",
    "withholds_action",
)
_QUESTION_SAFE_INTENTS = frozenset({INTENT_ADHOC_QUERY, INTENT_DATASET_ANALYSIS})
# DeepSeek V4 can consume completion budget as hidden reasoning even when the
# saved profile disables visible thinking.  Production evidence includes a
# length-finished empty reviewer response after all 1024 tokens were consumed
# by hidden reasoning, before any JSON was emitted.
_MAX_COMPLETION_TOKENS = 2048


@dataclass(frozen=True)
class SemanticIntentDecision:
    accepted: bool
    intent: str
    reason: str
    confidence: str
    route_evidence_quote: str
    review_evidence_quote: str
    is_question: bool
    is_conditional: bool
    requests_change: bool
    withholds_action: bool
    failure_code: str | None = None

    def as_metadata(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "reason": self.reason,
            "confidence": self.confidence,
            "route_evidence_quote": self.route_evidence_quote,
            "review_evidence_quote": self.review_evidence_quote,
            **(
                {"failure_code": self.failure_code}
                if self.failure_code is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class _PassResult:
    intent: str
    evidence_quote: str
    reason: str
    confidence: str
    is_question: bool
    is_conditional: bool
    requests_change: bool
    withholds_action: bool


@dataclass(frozen=True)
class _ParseOutcome:
    result: _PassResult | None
    failure_code: str | None
    parsed_object: dict[str, object] | None


@dataclass(frozen=True)
class _PassAttempt:
    result: _PassResult | None
    failure_code: str | None


def route_semantic_intent(
    client,
    *,
    task_type: str,
    instruction: str,
    context: Mapping[str, object],
    allowed_intents: Sequence[str],
) -> SemanticIntentDecision:
    """Classify twice independently and accept only a safe, exact agreement."""

    allowed = tuple(dict.fromkeys(str(item) for item in allowed_intents))
    if (
        client is None
        or not isinstance(task_type, str)
        or not isinstance(instruction, str)
        or not instruction.strip()
        or not allowed
        or any(item not in ALL_SEMANTIC_INTENTS for item in allowed)
    ):
        return _failed_decision("语义意图分类不可用，已保持当前状态。")
    try:
        prompt = json.dumps(
            {
                "task_type": task_type,
                "instruction": instruction,
                "current_context": dict(context),
                "allowed_intents": list(allowed),
                "intent_definitions": {
                    intent: _INTENT_DEFINITIONS[intent] for intent in allowed
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except Exception:
        return _failed_decision("语义意图上下文无效，已保持当前状态。")

    schema = _schema(allowed)
    first_attempt = _invoke_and_parse(
        client,
        system_prompt=TOP_LEVEL_INTENT_ROUTER_SYS,
        user_prompt=prompt,
        schema=schema,
        caller="semantic_intent_router",
        instruction=instruction,
        allowed=allowed,
    )
    first = first_attempt.result
    if first is None:
        code = f"semantic_intent_router_{first_attempt.failure_code or 'failed'}"
        return _failed_decision(
            f"第一遍语义意图分类失败（{code}），已保持当前状态。",
            failure_code=code,
        )
    second_attempt = _invoke_and_parse(
        client,
        system_prompt=TOP_LEVEL_INTENT_REVIEW_SYS,
        user_prompt=prompt,
        schema=schema,
        caller="semantic_intent_reviewer",
        instruction=instruction,
        allowed=allowed,
    )
    second = second_attempt.result
    if second is None:
        code = f"semantic_intent_reviewer_{second_attempt.failure_code or 'failed'}"
        return _failed_decision(
            f"独立语义意图复核失败（{code}），已保持当前状态。",
            failure_code=code,
        )
    if first.intent != second.intent:
        return _failed_decision(
            "两遍语义意图判断不一致，已保持当前状态。",
            failure_code="semantic_intent_pass_disagreement",
        )
    if not (_pass_is_safe(first) and _pass_is_safe(second)):
        return _failed_decision(
            "当前表达仍有疑问、条件、修改或保留，未执行任何动作。",
            failure_code="semantic_intent_unsafe_decision",
        )
    return SemanticIntentDecision(
        accepted=True,
        intent=first.intent,
        reason=second.reason or first.reason,
        confidence="high",
        route_evidence_quote=first.evidence_quote,
        review_evidence_quote=second.evidence_quote,
        is_question=first.is_question or second.is_question,
        is_conditional=first.is_conditional or second.is_conditional,
        requests_change=first.requests_change or second.requests_change,
        withholds_action=first.withholds_action or second.withholds_action,
    )


def _schema(allowed: Sequence[str]) -> dict:
    return {
        "name": "top_level_semantic_intent",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": list(allowed)},
                "evidence_quote": {"type": "string"},
                "reason": {"type": "string"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "is_question": {"type": "boolean"},
                "is_conditional": {"type": "boolean"},
                "requests_change": {"type": "boolean"},
                "withholds_action": {"type": "boolean"},
            },
            "required": list(_FIELDS),
            "additionalProperties": False,
        },
    }


def _invoke_and_parse(
    client,
    *,
    system_prompt,
    user_prompt: str,
    schema: dict,
    caller: str,
    instruction: str,
    allowed: Sequence[str],
) -> _PassAttempt:
    try:
        raw = client.complete(
            system_prompt=system_prompt.text,
            user_prompt=user_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
            json_schema=schema,
            max_tokens=_MAX_COMPLETION_TOKENS,
            stream=False,
            caller=caller,
            prompt_name=system_prompt.name,
            prompt_version=system_prompt.version,
        )
    except Exception:
        return _PassAttempt(result=None, failure_code="request_failed")

    outcome = _parse(raw, instruction=instruction, allowed=allowed)
    if outcome.result is not None:
        return _PassAttempt(result=outcome.result, failure_code=None)
    if outcome.parsed_object is None or not _repairable_object(
        outcome.parsed_object,
        allowed=allowed,
    ):
        return _PassAttempt(result=None, failure_code=outcome.failure_code)

    try:
        repair_prompt = _repair_prompt(
            user_prompt=user_prompt,
            first_pass=outcome.parsed_object,
            failure_code=outcome.failure_code or "invalid_object",
            instruction=instruction,
        )
    except (TypeError, ValueError):
        return _PassAttempt(result=None, failure_code="repair_context_invalid")
    try:
        repaired_raw = client.complete(
            system_prompt=TOP_LEVEL_INTENT_REPAIR_SYS.text,
            user_prompt=repair_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
            json_schema=schema,
            max_tokens=_MAX_COMPLETION_TOKENS,
            stream=False,
            caller=f"{caller}_repair",
            prompt_name=TOP_LEVEL_INTENT_REPAIR_SYS.name,
            prompt_version=TOP_LEVEL_INTENT_REPAIR_SYS.version,
        )
    except Exception:
        return _PassAttempt(result=None, failure_code="repair_request_failed")
    repaired = _parse(
        repaired_raw,
        instruction=instruction,
        allowed=allowed,
    )
    if repaired.result is None:
        return _PassAttempt(
            result=None,
            failure_code=f"repair_{repaired.failure_code or 'invalid'}",
        )
    if not _repair_preserves_bounded_semantics(
        outcome.parsed_object,
        repaired.result,
        instruction=instruction,
    ):
        return _PassAttempt(result=None, failure_code="repair_semantic_drift")
    return _PassAttempt(result=repaired.result, failure_code=None)


class _DuplicateKeyError(ValueError):
    pass


def _reject_nonstandard_json_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse(raw, *, instruction: str, allowed: Sequence[str]) -> _ParseOutcome:
    if not isinstance(raw, str):
        return _parse_failure("non_text")
    if not raw.strip():
        return _parse_failure("empty_response")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise _DuplicateKeyError
            value[key] = item
        return value

    try:
        data = json.loads(
            raw.strip(),
            object_pairs_hook=unique_object,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except _DuplicateKeyError:
        return _parse_failure("duplicate_key")
    except (TypeError, ValueError):
        return _parse_failure("non_json")
    if not isinstance(data, dict):
        return _parse_failure("non_object")
    missing = set(_FIELDS) - set(data)
    if missing:
        return _parse_failure("missing_fields", parsed_object=data)
    extra = set(data) - set(_FIELDS)
    if extra:
        return _parse_failure("extra_fields", parsed_object=data)
    flags = (
        data["is_question"],
        data["is_conditional"],
        data["requests_change"],
        data["withholds_action"],
    )
    if not isinstance(data["intent"], str):
        return _parse_failure("invalid_intent_type", parsed_object=data)
    if data["intent"] not in allowed:
        return _parse_failure("invalid_intent", parsed_object=data)
    if not isinstance(data["evidence_quote"], str):
        return _parse_failure("invalid_evidence_quote_type", parsed_object=data)
    if not data["evidence_quote"].strip():
        return _parse_failure("empty_evidence_quote", parsed_object=data)
    if data["evidence_quote"] not in instruction:
        return _parse_failure(
            "evidence_quote_not_in_instruction",
            parsed_object=data,
        )
    if not isinstance(data["reason"], str):
        return _parse_failure("invalid_reason_type", parsed_object=data)
    if not isinstance(data["confidence"], str):
        return _parse_failure("invalid_confidence_type", parsed_object=data)
    if data["confidence"] not in {"high", "medium", "low"}:
        return _parse_failure("invalid_confidence", parsed_object=data)
    if any(type(flag) is not bool for flag in flags):
        return _parse_failure("invalid_flag_type", parsed_object=data)
    return _ParseOutcome(
        result=_PassResult(
            intent=data["intent"],
            evidence_quote=data["evidence_quote"],
            reason=data["reason"],
            confidence=data["confidence"],
            is_question=flags[0],
            is_conditional=flags[1],
            requests_change=flags[2],
            withholds_action=flags[3],
        ),
        failure_code=None,
        parsed_object=data,
    )


def _parse_failure(
    failure_code: str,
    *,
    parsed_object: dict[str, object] | None = None,
) -> _ParseOutcome:
    return _ParseOutcome(
        result=None,
        failure_code=failure_code,
        parsed_object=parsed_object,
    )


def _repairable_object(
    first_pass: Mapping[str, object],
    *,
    allowed: Sequence[str],
) -> bool:
    """Never ask a repair call to invent the operative intent."""

    intent = first_pass.get("intent")
    if not isinstance(intent, str) or intent not in allowed:
        return False
    if _normalizable_confidence(first_pass.get("confidence")) is None:
        return False
    return all(
        _normalizable_bool(first_pass.get(field)) is not None
        for field in (
            "is_question",
            "is_conditional",
            "requests_change",
            "withholds_action",
        )
    )


def _repair_prompt(
    *,
    user_prompt: str,
    first_pass: Mapping[str, object],
    failure_code: str,
    instruction: str,
) -> str:
    """Build a data-only normalization request without forwarding extra fields."""

    original_request = json.loads(
        user_prompt,
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(original_request, dict):
        raise ValueError("semantic intent request must be an object")
    intent = first_pass.get("intent")
    confidence = _normalizable_confidence(first_pass.get("confidence"))
    normalized_flags = {
        field: _normalizable_bool(first_pass.get(field))
        for field in (
            "is_question",
            "is_conditional",
            "requests_change",
            "withholds_action",
        )
    }
    if not isinstance(intent, str) or confidence is None or any(
        value is None for value in normalized_flags.values()
    ):
        raise ValueError("semantic intent response is not safely normalizable")
    first_pass_fields: dict[str, object] = {
        "intent": intent,
        "confidence": confidence,
        **normalized_flags,
    }
    quote = first_pass.get("evidence_quote")
    if isinstance(quote, str) and quote.strip() and quote in instruction:
        first_pass_fields["evidence_quote"] = quote
    return json.dumps(
        {
            "operation": "normalize_existing_semantic_intent_decision",
            "original_request": original_request,
            "first_pass_fields": first_pass_fields,
            "first_pass_failure_code": failure_code,
            "had_unexpected_fields": bool(set(first_pass) - set(_FIELDS)),
            "required_fields": list(_FIELDS),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _repair_preserves_bounded_semantics(
    first_pass: Mapping[str, object],
    repaired: _PassResult,
    *,
    instruction: str,
) -> bool:
    if repaired.intent != first_pass.get("intent"):
        return False
    original_quote = first_pass.get("evidence_quote")
    if (
        isinstance(original_quote, str)
        and original_quote.strip()
        and original_quote in instruction
        and repaired.evidence_quote != original_quote
    ):
        return False
    original_confidence = _normalizable_confidence(
        first_pass.get("confidence")
    )
    if original_confidence is None or repaired.confidence != original_confidence:
        return False
    for field in (
        "is_question",
        "is_conditional",
        "requests_change",
        "withholds_action",
    ):
        original = _normalizable_bool(first_pass.get(field))
        if original is None or getattr(repaired, field) is not original:
            return False
    return True


def _normalizable_confidence(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else None


def _normalizable_bool(value: object) -> bool | None:
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _pass_is_safe(result: _PassResult) -> bool:
    if result.confidence != "high" or result.intent == INTENT_NONE:
        return False
    if result.intent == INTENT_ADHOC_REJECT:
        return (
            result.withholds_action
            and not result.is_question
            and not result.is_conditional
            and not result.requests_change
        )
    if result.intent == INTENT_ADHOC_REVISE:
        return (
            result.requests_change
            and not result.is_question
            and not result.is_conditional
            and not result.withholds_action
        )
    return (
        not result.is_conditional
        and not result.requests_change
        and not result.withholds_action
        and (not result.is_question or result.intent in _QUESTION_SAFE_INTENTS)
    )


def _failed_decision(
    reason: str,
    *,
    failure_code: str | None = None,
) -> SemanticIntentDecision:
    return SemanticIntentDecision(
        accepted=False,
        intent=INTENT_NONE,
        reason=reason,
        confidence="low",
        route_evidence_quote="",
        review_evidence_quote="",
        is_question=False,
        is_conditional=False,
        requests_change=False,
        withholds_action=True,
        failure_code=failure_code,
    )


__all__ = [
    "ALL_SEMANTIC_INTENTS",
    "INTENT_ADHOC_CONFIRM",
    "INTENT_ADHOC_QUERY",
    "INTENT_ADHOC_REJECT",
    "INTENT_ADHOC_REVISE",
    "INTENT_CURRENT_WORKFLOW",
    "INTENT_DATASET_ANALYSIS",
    "INTENT_DATASET_EXPORT",
    "INTENT_DATASET_JOIN",
    "INTENT_DATASET_TRANSFORM",
    "INTENT_NONE",
    "INTENT_RISK_PROFITABILITY",
    "INTENT_RISK_STANDARD_VINTAGE",
    "INTENT_RISK_VTG_TERMINAL",
    "INTENT_STRATEGY_SAMPLE_BINDING",
    "INTENT_STRATEGY_WORKFLOW",
    "SemanticIntentDecision",
    "route_semantic_intent",
]
