"""Core request grammar and grounding rules."""
from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from marvis.agent.json_reply import load_json_object
from marvis.agent.strategy_workflows import (
    FRESH_STANDARD_STRATEGY_WORKFLOWS,
    REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowResolutionMode,
    StrategyWorkflowValidationError,
    migrated_workflow_confirmation,
)
from marvis.agent.strategy_workflows import (
    resolve_strategy_request as resolve_standard_strategy_workflow,
)
from marvis.llm_prompts import (
    SAMPLE_DESIGN_V2_CORRECTION_SYS,
    STRATEGY_REQUEST_COMPILER_SYS,
)
from marvis.packs.strategy.candidate_design import (
    CANDIDATE_DESIGN_SCHEMA_VERSION,
    CandidateDesignError,
    normalize_candidate_design,
    normalize_candidate_economics_inputs,
)
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.errors import StrategyError
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason

from . import contracts as _contracts
from . import refinement_report as _refinement_report
from . import sample_design as _sample_design

_SYSTEM = STRATEGY_REQUEST_COMPILER_SYS.text

logger = logging.getLogger("marvis.agent.strategy_request_compiler")

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

STANDARD_STRATEGY_WORKFLOWS = FRESH_STANDARD_STRATEGY_WORKFLOWS

_STRATEGY_REPLY_MAX_CHARS = 100_000

_STRATEGY_REPLY_MAX_DEPTH = 64

_STRATEGY_REPLY_MAX_NODES = 10_000

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
        "strategy_report_bundle_v2_platform_binding_forbidden",
        "strategy_dsl_delivery_platform_binding_forbidden",
        "strategy_request_too_complex",
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
                                        {"required": ["operating_cost_per_loan_col"]},
                                        {"required": ["operating_cost_per_loan_value"]},
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

def compile_strategy_request(
    utterance: str,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
    llm,
    caller: str = "strategy_request_compiler",
) -> _contracts.StrategyRequestCompilation:
    """Compile one utterance with bounded semantic and format correction."""

    if not isinstance(utterance, str) or not utterance.strip():
        return _contracts._clarification("请说明希望执行的策略操作和策略类型。")
    normalized_utterance = utterance.strip()
    if _COLLECTION_STRATEGY_RE.search(normalized_utterance):
        return _contracts._clarification(
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
        return _contracts._clarification(
            "当前暂时无法解析策略请求，请稍后重试或直接说明操作、策略类型和策略对象。"
        )
    outcome = _validate_reply(
        raw,
        whitelist,
        target_col=observed_target,
        caller=caller,
        attempt_kind="initial",
    )
    if outcome.accepted:
        grounded = _refinement_report._ground_refinement_request(
            normalized_utterance,
            outcome.result,
            whitelist=whitelist,
            target_col=observed_target,
        )
        if grounded.clarification_code == "roll_rate_column_binding_not_grounded":
            repair_prompt = _repair_prompt(
                prompt,
                raw=raw,
                error=(
                    f"{grounded.clarification_code}：{grounded.clarification}\n"
                    "保持 workflow=roll_rate_matrix，只修正用户明确绑定的 "
                    "id_col/time_col/status_col/balance_col；不得替换为其他合法列。"
                ),
            )
            try:
                repaired = _complete(llm, prompt=repair_prompt, caller=caller)
            except Exception:
                return grounded
            repaired_outcome = _validate_reply(
                repaired,
                whitelist,
                target_col=observed_target,
                caller=caller,
                attempt_kind="roll_rate_column_correction",
            )
            if not (
                repaired_outcome.accepted
                and isinstance(
                    repaired_outcome.result.draft,
                    _contracts.StandardWorkflowRequestDraft,
                )
                and repaired_outcome.result.draft.workflow == "roll_rate_matrix"
            ):
                return grounded
            return _refinement_report._ground_refinement_request(
                normalized_utterance,
                repaired_outcome.result,
                whitelist=whitelist,
                target_col=observed_target,
            )
        if (
            grounded.clarification_code
            == "strategy_sample_design_v2_workflow_required"
        ):
            return _correct_sample_design_v2_request(
                llm,
                utterance=normalized_utterance,
                whitelist=whitelist,
                target_col=observed_target,
                error=(
                    f"{grounded.clarification_code}：{grounded.clarification}\n"
                ),
                caller=caller,
            )
        return grounded
    if _sample_design.utterance_targets_strategy_sample_design(normalized_utterance):
        return _correct_sample_design_v2_request(
            llm,
            utterance=normalized_utterance,
            whitelist=whitelist,
            target_col=observed_target,
            error=outcome.error or "SampleDesign V2 草案未通过平台校验。",
            caller=caller,
        )
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
    repaired_outcome = _validate_reply(
        repaired,
        whitelist,
        target_col=observed_target,
        caller=caller,
        attempt_kind="generic_repair",
    )
    if repaired_outcome.accepted:
        return _refinement_report._ground_refinement_request(
            normalized_utterance,
            repaired_outcome.result,
            whitelist=whitelist,
            target_col=observed_target,
        )
    return repaired_outcome.result

def validate_strategy_request(
    payload: object,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
    allow_legacy_replay: bool = False,
) -> _contracts.StrategyRequestCompilation:
    """Validate an already parsed request without invoking an LLM.

    Fresh callers intentionally cannot emit the retired V1 sample workflow.
    Only a persistence/recovery boundary may opt into ``allow_legacy_replay``.
    """

    return _validate_payload(
        payload,
        _column_whitelist(allowed_columns),
        target_col=_normalized_target_col(target_col),
        allow_legacy_replay=allow_legacy_replay,
    ).result

def strategy_request_confirmation_text(
    draft: _contracts.CompiledStrategyRequestDraft,
) -> str:
    """Render a plain-Chinese echo of the request before any workflow runs."""

    if isinstance(draft, _contracts.StandardWorkflowRequestDraft):
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
        _contracts._deep_thaw(value),
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
        # DeepSeek V4 counts reasoning tokens inside max_tokens.  This compiler
        # prompt is intentionally broad and needs enough bounded headroom for
        # thinking plus the final compact JSON decision.
        max_tokens=8192,
        stream=False,
        caller=caller,
        prompt_name=STRATEGY_REQUEST_COMPILER_SYS.name,
        prompt_version=STRATEGY_REQUEST_COMPILER_SYS.version,
    )

def _sample_design_v2_correction_prompt(
    utterance: str,
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
    error: str,
    format_repair: bool,
) -> str:
    opportunity = (
        "这是唯一一次格式修复机会。"
        if format_repair
        else "这是唯一一次语义纠正机会。"
    )
    return (
        "【原始用户意图】\n"
        f"{utterance}\n"
        "【列白名单】\n"
        f"{json.dumps(list(whitelist), ensure_ascii=False)}\n"
        "【当前目标列角色】\n"
        f"{json.dumps(target_col, ensure_ascii=False)}\n"
        "【上一次输出未通过平台校验】\n"
        f"{error}\n"
        f"{opportunity}"
        "固定输出 strategy_sample_design_v2 的完整用户自有 DTO；"
        "所有控制项必须来自原始意图并使用列白名单，不能补默认值或猜测。"
        "workflow_inputs 必须精确包含 target_bad_value、drop_nan_labels、"
        "relationship、approval_population、risk_population、partitioning、"
        "maturity、performance_window、observation_window、field_bindings、"
        "historical_score。relationship 只能是 nested_same_cohort 或 "
        "parallel_time_cohorts。"
        "time_ranges 分区必须使用精确形状："
        '{"method":"time_ranges","column":"列名","ranges":'
        '{"development":{"start":"YYYY-MM-DD或null","end":"YYYY-MM-DD或null"},'
        '"validation":{"start":"YYYY-MM-DD或null","end":"YYYY-MM-DD或null"},'
        '"oot":{"start":"YYYY-MM-DD或null","end":"YYYY-MM-DD或null"}}}。'
        "predicate_ast 分区必须使用 method=predicate_ast 和完整 selectors。"
        "approval_population 与 risk_population 都必须精确包含 inclusion、"
        "exclusion；无筛选时两者都为 null。"
        "maturity 精确包含 status、performance_window_days、cutoff_date、reason，"
        "status 只能是 confirmed_matured、not_matured、unknown、unavailable；"
        "confirmed_matured 时 performance_window_days 和 cutoff_date 必填且 reason "
        "必须为 null；not_matured 时 performance_window_days、cutoff_date、原话中的"
        "非空 reason 都必填；unknown/unavailable 时 performance_window_days 和 "
        "cutoff_date 必须为 null，reason 必须是原话中的非空说明。"
        "performance_window 精确包含 status、days，status 只能是 provided 或 "
        "unavailable，provided 时 days 必填，unavailable 时 days 必须为 null；"
        "observation_window 精确包含 status、start、end，status 只能是 provided 或 "
        "unavailable，provided 时 start/end 必填，unavailable 时 start 和 end 必须为"
        " null。field_bindings 精确包含 entity_field、time_field、"
        "group_field、month_field、weight_field、loan_amount_field、"
        "overdue_amount_field；原话明确暂不可提供的字段必须输出 null。"
        "historical_score 精确包含 status、column、"
        "direction、reason，status 只能是 available、unavailable、not_applicable，"
        "available 时 column 必填、direction 只能是 higher_is_riskier 或 "
        "lower_is_riskier 且 reason 必须为 null；unavailable/not_applicable 时 "
        "column 和 direction "
        "必须为 null，reason 必须是原话中的非空说明。"
        "只输出一个 JSON 对象；信息不足时只返回中文 clarification。"
    )

def _complete_sample_design_v2_correction(
    llm,
    *,
    prompt: str,
    caller: str,
):
    return llm.complete(
        system_prompt=SAMPLE_DESIGN_V2_CORRECTION_SYS.text,
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_sample_design.SAMPLE_DESIGN_V2_CORRECTION_JSON_SCHEMA,
        max_tokens=8192,
        stream=False,
        caller=caller,
        prompt_name=SAMPLE_DESIGN_V2_CORRECTION_SYS.name,
        prompt_version=SAMPLE_DESIGN_V2_CORRECTION_SYS.version,
    )

def _correct_sample_design_v2_request(
    llm,
    *,
    utterance: str,
    whitelist: tuple[str, ...],
    target_col: str | None,
    error: str,
    caller: str,
) -> _contracts.StrategyRequestCompilation:
    repair_prompt = _sample_design_v2_correction_prompt(
        utterance,
        whitelist=whitelist,
        target_col=target_col,
        error=error,
        format_repair=False,
    )
    try:
        repaired = _complete_sample_design_v2_correction(
            llm,
            prompt=repair_prompt,
            caller=caller,
        )
    except Exception:
        return _contracts._clarification(
            "当前暂时无法纠正 SampleDesign V2 请求，请稍后重试或补充完整样本口径。"
        )
    repaired_outcome = _validate_reply(
        repaired,
        whitelist,
        target_col=target_col,
        caller=caller,
        attempt_kind="sample_design_semantic_correction",
    )
    if repaired_outcome.accepted:
        if repaired_outcome.result.draft is None:
            return repaired_outcome.result
        return _refinement_report._ground_refinement_request(
            utterance,
            repaired_outcome.result,
            whitelist=whitelist,
            target_col=target_col,
        )
    if (
        repaired_outcome.result.clarification
        != "模型返回的策略草案不是有效 JSON 对象，请重新说明策略请求。"
    ):
        return repaired_outcome.result

    format_repair_prompt = _sample_design_v2_correction_prompt(
        utterance,
        whitelist=whitelist,
        target_col=target_col,
        error=(
            repaired_outcome.result.clarification
            or repaired_outcome.error
            or "输出格式无效"
        ),
        format_repair=True,
    )
    try:
        format_repaired = _complete_sample_design_v2_correction(
            llm,
            prompt=format_repair_prompt,
            caller=caller,
        )
    except Exception:
        return repaired_outcome.result
    format_repaired_outcome = _validate_reply(
        format_repaired,
        whitelist,
        target_col=target_col,
        caller=caller,
        attempt_kind="sample_design_format_repair",
    )
    if format_repaired_outcome.accepted:
        if format_repaired_outcome.result.draft is None:
            return format_repaired_outcome.result
        return _refinement_report._ground_refinement_request(
            utterance,
            format_repaired_outcome.result,
            whitelist=whitelist,
            target_col=target_col,
        )
    return format_repaired_outcome.result

def _strategy_payload_within_limits(payload: object) -> bool:
    stack: list[tuple[object, int]] = [(payload, 0)]
    seen_containers: set[int] = set()
    node_count = 0
    while stack:
        value, depth = stack.pop()
        node_count += 1
        if node_count > _STRATEGY_REPLY_MAX_NODES or depth > _STRATEGY_REPLY_MAX_DEPTH:
            return False
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            identity = id(value)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value)
    return True

def _reply_edge_char_kind(text: str, *, first: bool) -> str:
    stripped = text.strip()
    if not stripped:
        return "none"
    char = stripped[0] if first else stripped[-1]
    if char == "{":
        return "brace_open"
    if char == "}":
        return "brace_close"
    if char == "[":
        return "bracket_open"
    if char == "]":
        return "bracket_close"
    if char in {'"', "'"}:
        return "quote"
    if char.isdigit():
        return "digit"
    if char.isalpha():
        return "letter"
    return "other"

def _reply_has_balanced_object(text: str) -> bool:
    """Detect a balanced object boundary without retaining or parsing content."""

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for char in text[start:]:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
                if depth < 0:
                    break
        start = text.find("{", start + 1)
    return False

def _log_json_parse_failure(
    raw: object,
    *,
    caller: str,
    attempt_kind: str,
) -> None:
    """Log shape-only diagnostics; never persist completion or prompt content."""

    text = raw if isinstance(raw, str) else ""
    lowered = text.lower()
    logger.warning(
        "strategy JSON parse failed caller=%s attempt_kind=%s "
        "chars=%d has_think=%s has_fence=%s "
        "first_char_kind=%s last_char_kind=%s "
        "brace_open=%d brace_close=%d bracket_open=%d bracket_close=%d "
        "balanced_object=%s",
        caller,
        attempt_kind,
        len(text),
        "<think" in lowered,
        "```" in text,
        _reply_edge_char_kind(text, first=True),
        _reply_edge_char_kind(text, first=False),
        text.count("{"),
        text.count("}"),
        text.count("["),
        text.count("]"),
        _reply_has_balanced_object(text),
    )

def _validate_reply(
    raw: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    caller: str,
    attempt_kind: str,
) -> _contracts._ValidationOutcome:
    if isinstance(raw, str) and len(raw) > _STRATEGY_REPLY_MAX_CHARS:
        return _contracts._invalid(
            "模型返回的策略草案过长，请缩小为一个明确的策略操作。",
            code="strategy_request_too_complex",
            fields=("reply",),
        )
    payload, error = load_json_object(raw)
    if payload is None:
        _log_json_parse_failure(
            raw,
            caller=caller,
            attempt_kind=attempt_kind,
        )
        message = "模型返回的策略草案不是有效 JSON 对象，请重新说明策略请求。"
        return _contracts._ValidationOutcome(_contracts._clarification(message), False, error or message)
    return _validate_payload(
        payload,
        whitelist,
        target_col=target_col,
        allow_legacy_replay=False,
    )

def _validate_payload(
    payload: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    allow_legacy_replay: bool,
) -> _contracts._ValidationOutcome:
    if not isinstance(payload, Mapping):
        message = "策略请求必须是 JSON 对象，请重新说明。"
        return _contracts._invalid(message)
    if not _strategy_payload_within_limits(payload):
        return _contracts._invalid(
            "策略草案嵌套过深或字段过多，请缩小为一个明确的策略操作。",
            code="strategy_request_too_complex",
            fields=("payload",),
        )
    if any(not isinstance(key, str) for key in payload):
        return _contracts._invalid("策略请求的字段名必须是文本，请重新说明。")

    if "clarification" in payload:
        if set(payload) != {"clarification"}:
            return _contracts._invalid("澄清问题不能和策略草案字段同时出现，请重新选择一种输出。")
        value = payload["clarification"]
        if not isinstance(value, str) or not value.strip():
            return _contracts._invalid("澄清问题必须是非空文本。")
        return _contracts._ValidationOutcome(
            _contracts._clarification(_chinese_clarification(value)),
            True,
        )

    request_kind = payload.get("request_kind", "strategy_lifecycle")
    if not isinstance(request_kind, str) or request_kind not in STRATEGY_REQUEST_KINDS:
        return _contracts._invalid(
            "不支持的 request_kind；只能是 strategy_lifecycle 或 standard_workflow。"
        )
    if request_kind == "standard_workflow":
        unexpected = sorted(set(payload) - _STANDARD_WORKFLOW_DRAFT_FIELDS)
        if unexpected:
            rendered = "、".join(f"「{field}」" for field in unexpected)
            return _contracts._invalid(
                f"标准 Workflow 请求包含不支持的字段 {rendered}，请删除后重新说明。"
            )
        return _validate_standard_workflow_payload(
            payload,
            whitelist,
            target_col=target_col,
            allow_legacy_replay=allow_legacy_replay,
        )

    unexpected = sorted(set(payload) - _LIFECYCLE_DRAFT_FIELDS)
    if unexpected:
        rendered = "、".join(f"「{field}」" for field in unexpected)
        return _contracts._invalid(f"策略请求包含不支持的字段 {rendered}，请删除后重新说明。")
    if payload.get("request_kind") not in (None, "strategy_lifecycle"):
        return _contracts._invalid("策略生命周期请求的 request_kind 必须是 strategy_lifecycle。")
    missing = [
        field for field in ("operation", "strategy_type") if field not in payload
    ]
    if missing:
        rendered = "、".join(missing)
        return _contracts._invalid(f"没有识别到必需字段 {rendered}，请补充策略操作和策略类型。")

    operation = payload["operation"]
    if not isinstance(operation, str) or operation not in STRATEGY_OPERATIONS:
        return _contracts._invalid(
            "不支持的策略操作；可选操作为：" + "、".join(STRATEGY_OPERATIONS) + "。"
        )
    strategy_type = payload["strategy_type"]
    if not isinstance(strategy_type, str) or strategy_type not in STRATEGY_TYPES:
        return _contracts._invalid(
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
                raise _contracts._DraftValidationError(
                    str(exc),
                    code=exc.code,
                    fields=exc.fields,
                ) from exc
        strategy_spec = _optional_strategy_spec(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
    except _contracts._DraftValidationError as exc:
        return _contracts._invalid(str(exc), code=exc.code, fields=exc.fields)

    draft = _contracts.StrategyRequestDraft(
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
    result = _contracts.StrategyRequestCompilation(
        draft=draft,
        clarification=None,
        confirmation=strategy_request_confirmation_text(draft),
    )
    return _contracts._ValidationOutcome(result, True)

def _validate_standard_workflow_payload(
    payload: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    allow_legacy_replay: bool,
) -> _contracts._ValidationOutcome:
    missing = [
        field for field in ("workflow", "workflow_inputs") if field not in payload
    ]
    if missing:
        return _contracts._invalid("标准 Workflow 请求缺少字段：" + "、".join(missing) + "。")
    workflow = payload["workflow"]
    allowed_workflows = (
        REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS
        if allow_legacy_replay
        else FRESH_STANDARD_STRATEGY_WORKFLOWS
    )
    if not isinstance(workflow, str) or workflow not in allowed_workflows:
        return _contracts._invalid(
            "不支持的标准 Workflow；可选值为："
            + "、".join(allowed_workflows)
            + "。"
        )
    raw_inputs = payload["workflow_inputs"]
    if not isinstance(raw_inputs, Mapping):
        return _contracts._invalid("workflow_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        return _contracts._invalid("workflow_inputs 的字段名必须是文本。")
    try:
        resolved = resolve_standard_strategy_workflow(
            workflow,
            raw_inputs,
            context=StrategyWorkflowResolutionContext(
                allowed_columns=whitelist,
                target_col=target_col,
            ),
            mode=(
                StrategyWorkflowResolutionMode.REPLAY
                if allow_legacy_replay
                else StrategyWorkflowResolutionMode.FRESH
            ),
        )
    except StrategyWorkflowValidationError as exc:
        return _contracts._invalid(str(exc), code=exc.code, fields=exc.fields)

    draft = _contracts.StandardWorkflowRequestDraft(
        workflow=resolved.workflow_id,
        workflow_inputs=resolved.workflow_inputs,
    )
    return _contracts._ValidationOutcome(
        _contracts.StrategyRequestCompilation(
            draft=draft,
            clarification=None,
            confirmation=resolved.confirmation,
        ),
        True,
    )

def _standard_workflow_confirmation_text(
    draft: _contracts.StandardWorkflowRequestDraft,
) -> str:
    confirmation = migrated_workflow_confirmation(
        draft.workflow,
        draft.workflow_inputs,
    )
    if confirmation is None:  # pragma: no cover - catalog invariant
        raise ValueError(f"unsupported standard workflow {draft.workflow}")
    return confirmation

def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise _contracts._DraftValidationError(f"{key} 必须是非空文本，请重新说明。")
    return value.strip()

def _optional_ratio(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _contracts._DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise _contracts._DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    return number

def _optional_adoption_reason(payload: Mapping[str, Any]) -> str | None:
    if "adoption_reason" not in payload:
        return None
    try:
        return normalize_adoption_reason(payload["adoption_reason"])
    except AdoptionReasonError as exc:
        raise _contracts._DraftValidationError(str(exc)) from exc

def _optional_profit(
    payload: Mapping[str, Any], whitelist: tuple[str, ...]
) -> dict[str, Any] | None:
    if "profit" not in payload:
        return None
    profit = payload["profit"]
    if not isinstance(profit, Mapping):
        raise _contracts._DraftValidationError("利润参数 profit 必须是一个对象。")
    if any(not isinstance(key, str) for key in profit):
        raise _contracts._DraftValidationError("利润参数字段名必须是文本。")
    missing = sorted(_PROFIT_FIELDS - set(profit))
    unexpected = sorted(set(profit) - _PROFIT_FIELDS)
    if missing:
        raise _contracts._DraftValidationError("利润参数缺少字段：" + "、".join(missing) + "。")
    if unexpected:
        raise _contracts._DraftValidationError(
            "利润参数包含不支持的字段：" + "、".join(unexpected) + "。"
        )
    ead_col = _required_text(profit["ead_col"], name="利润 EAD 列 ead_col")
    pd_col = _required_text(profit["pd_col"], name="利润 PD 列 pd_col")
    for name, column in (("ead_col", ead_col), ("pd_col", pd_col)):
        if column not in whitelist:
            raise _contracts._DraftValidationError(
                f"利润参数 {name} 使用了数据集中不存在的列「{column}」，请从列白名单选择。"
            )
    annual_rate = _bounded_number(
        profit["annual_rate"], name="利润 annual_rate", maximum=1
    )
    funding_rate = _bounded_number(
        profit["funding_rate"], name="利润 funding_rate", maximum=1
    )
    lgd = _bounded_number(profit["lgd"], name="利润 lgd", maximum=1)
    operating_cost = _bounded_number(
        profit["operating_cost_per_loan"],
        name="利润 operating_cost_per_loan",
    )
    term_months = profit["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or term_months < 1
    ):
        raise _contracts._DraftValidationError("利润 term_months 必须是大于等于 1 的整数。")
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
        raise _contracts._DraftValidationError(
            "profit 只适用于审批或拒绝策略；额度和定价策略请使用 economics_inputs，"
            "分群策略不接受经济参数。"
        )
    if "economics_inputs" in payload and strategy_type not in {"limit", "pricing"}:
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError(
            "candidate_design 与 strategy_spec 必须二选一；LLM 不得同时提交候选输入和规则结果。",
            code="candidate_spec_mutually_exclusive",
            fields=("candidate_design", "strategy_spec"),
        )
    if has_candidate and (
        operation != "develop"
        or strategy_type not in {"limit", "pricing", "segmentation"}
    ):
        raise _contracts._DraftValidationError(
            "candidate_design 只适用于 limit、pricing、segmentation 的 develop 请求。",
            code="candidate_design_not_allowed",
            fields=("candidate_design",),
        )
    if has_spec and strategy_type in {"limit", "pricing", "segmentation"}:
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError("经济参数 economics_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        raise _contracts._DraftValidationError("经济参数 economics_inputs 的字段名必须是文本。")

    names = (
        _LIMIT_ECONOMICS_NAMES if strategy_type == "limit" else _PRICING_ECONOMICS_NAMES
    )
    allowed_fields = {key for name in names for key in (f"{name}_col", f"{name}_value")}
    unexpected = sorted(set(raw_inputs) - allowed_fields)
    if unexpected:
        raise _contracts._DraftValidationError(
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
            raise _contracts._DraftValidationError(
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
                raise _contracts._DraftValidationError(
                    f"经济参数 {column_key} 使用了数据集中不存在或不可用于策略的列"
                    f"「{column}」，请从列白名单选择。"
                )
            normalized[column_key] = column
            continue
        normalized[value_key] = _economics_value(name, raw_inputs[value_key])

    if missing:
        raise _contracts._DraftValidationError(
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
            raise _contracts._DraftValidationError(f"经济参数 {label} 必须是大于 0 的有限数字。")
        return number
    return _bounded_number(
        value,
        name=f"经济参数 {label}",
        maximum=_ECONOMICS_VALUE_MAXIMUMS.get(name),
    )

def _economics_confirmation(draft: _contracts.StrategyRequestDraft) -> str:
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

def _candidate_design_confirmation(draft: _contracts.StrategyRequestDraft) -> str:
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
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError(
            "非审批策略的 strategy_spec 必须由平台确定性生成，LLM 不得提交。",
            code="llm_strategy_spec_forbidden",
            fields=("strategy_spec",),
        )
    raw_spec = payload["strategy_spec"]
    if not isinstance(raw_spec, Mapping):
        raise _contracts._DraftValidationError("策略规则草案 strategy_spec 必须是一个对象。")
    raw_metadata = raw_spec.get("metadata", {})
    if raw_metadata not in ({}, {"lineage": {}}):
        raise _contracts._DraftValidationError(
            "策略规则草案 metadata 由平台生成，LLM 不得写入指标结果或其他元数据。"
        )
    try:
        parsed = parse_strategy_spec(raw_spec)
    except (StrategyError, TypeError, ValueError) as exc:
        raise _contracts._DraftValidationError(
            "策略规则草案格式或取值无效，请检查规则条件、优先级和动作。"
        ) from exc
    if parsed.strategy_type != strategy_type:
        raise _contracts._DraftValidationError(
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
        raise _contracts._DraftValidationError(
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

def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _contracts._DraftValidationError(f"{name} 必须是非空文本。")
    return value.strip()

def _bounded_number(
    value: object,
    *,
    name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _contracts._DraftValidationError(f"{name} 必须是有限数字。")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < 0
        or (maximum is not None and number > maximum)
    ):
        if maximum is None:
            raise _contracts._DraftValidationError(f"{name} 必须是大于等于 0 的有限数字。")
        raise _contracts._DraftValidationError(f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。")
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
        "对于 strategy_project_context，只能逐字抄录用户明确提供的截止日 as_of、"
        "可选 scope、business_context 文本/null、明确暂缺字段路径和任务 source_dir 下"
        "用户点名的外部报告相对文件名。禁止输出 revision/CAS、message id/hash、"
        "dataset/Pool/backtest/monitoring 引用、artifact id/hash、来源引用、可用性结论或指标。"
        "一次只刷新项目上下文，不得串联样本、候选、影响、报告、采纳或部署；没有 as_of"
        "必须 clarification，不能默认今天。"
        "对于 strategy_sample_design_v2，workflow_inputs 必须精确包含用户明确提供的"
        " target_bad_value、drop_nan_labels、relationship、approval_population、"
        "risk_population、partitioning、maturity、performance_window、observation_window、"
        "field_bindings、historical_score。relationship 必须由用户明确说明为"
        " nested_same_cohort 或 parallel_time_cohorts。所有嵌套字段、null、状态、列、值、"
        "方向与 reason 都必须能逐字回到原话。fresh population 的非 null inclusion/exclusion"
        " 只能输出 match=all/any 和 1 到 8 个简单 conditions；condition 只含"
        " column/operator/value，is_null/is_not_null 不含 value。禁止直接输出 population"
        " predicate AST。approval/risk、inclusion/exclusion 与 operator/column/value"
        " 必须在各自局部语境中逐项对应；表现窗、观察窗和 maturity cutoff 日期不得互相借用。"
        "普通表现窗不能只借用成熟表现窗；maturity cutoff 必须在同一子句带成熟度限定；"
        "historical_score 的字段和方向必须在同一局部子句绑定，不能借用其他字段的方向。"
        "平台会把 nested_same_cohort、双总体均无纳排且同列三个互异简单等值 selector"
        " 路由到 compatibility 链；parallel_time_cohorts、time_ranges、任一总体纳排或"
        "复杂 selector 路由到原生 V2，禁止删减或降级。"
        "scope/policy、legacy ref、dataset/workspace/target、membership/bundle/"
        "artifact id/hash 全部禁止填写，也不得串联建模、模型比较、报告、Strategy Pool、"
        "采纳或部署。"
        "对于 strategy_model_evidence_v2，workflow_inputs 必须是空对象；只汇总当前 task"
        " 已有认证单变量候选。所有 SampleDesign/candidate/artifact 引用由平台绑定；训练、"
        "模型对比、月度/OOT/验证模型、报告、采纳或部署的正向请求必须 clarification；"
        "明确说不做这些后续动作可以接受；“此前/已有/已认证”只有位于当前归集动作之后时"
        "才视为来源状态；“此前未汇总”“没有汇总”“有没有汇总”必须 clarification。"
        "对于 strategy_model_score_comparison_v2，workflow_inputs 必须且只能包含用户"
        "逐字明确的 population 与 partition。population 仅允许 approval/risk；partition "
        "仅允许 overall/development/validation/oot。SampleDesign、模型评分证据、分数向量、"
        "artifact id/hash、dataset/ref 与 registry CAS 全部由平台在当前 task 自动发现、"
        "选择最新兼容版本并完整认证，禁止输出。至少需要两个互异模型；本步骤只物化比较"
        "证据，固定 no_selection，不选择冠军、不采纳、不部署。正向串联选择、推荐、采纳或"
        "部署时必须 clarification；明确说不做这些后续动作可以接受。"
        "对于 strategy_report_bundle_v2，workflow_inputs 只能包含用户明确提供的 title "
        "和 status；status 仅允许 draft/partial/final。用户未提供时必须使用固定默认值"
        " title=策略迭代评审报告、status=partial。ProjectContext、SampleDesign、Pool、"
        "ImpactCube/兼容 PoolImpact、ModelEvidence/training/score、strategy identity、"
        "report revision/previous head CAS、generated_at、artifact id/hash、来源引用和"
        "所有指标必须省略，由平台在计划创建时精确绑定。报告可在原话中点名 approval/"
        "reject/limit/pricing/segmentation，但 strategy_type 也必须省略。报告请求必须是"
        "当前、肯定、单步骤命令；问句、否定、"
        "假设、演示、仅历史描述，或同轮串联训练、评分、候选、影响测算、采纳、部署、"
        "上线时必须 clarification。"
        "对于 strategy_dsl_delivery，workflow_inputs 只能包含用户原话中唯一完整的"
        "可选 strategy_id；没有 ID 时必须省略，由平台仅在当前任务只有一个可交付策略时"
        "唯一绑定。strategy_ref、策略类型/version/spec hash、dataset_ref、数据 hash、"
        "workspace_ref/revision/generation/semantic hash、"
        "maximum_equivalence_rows、artifact id/hash、等价结果和所有指标均由平台绑定，"
        "禁止输出。请求必须是当前、肯定、单步骤导出命令；问句、否定、假设、演示、"
        "仅历史描述，或同轮串联应用、写回、报告、影响测算、训练、评分、采纳、晋级、"
        "部署时必须 clarification。"
        "对于 automatic_tree_candidate_build，只能抄录用户明确提供的 features、"
        "权重/金额字段、方向和树参数；不得填写平台拥有的数据绑定、目标列、标签策略、"
        "预算、结果、叶子、动作、排名或推荐，也不得串联选叶或 Strategy Pool。"
        "对于 automatic_tree_apply，只能逐字抄录用户原话中唯一完整的 tree_asset_id；"
        "leaf_id_column 和 rule_id_column 只有在用户分别明确标注叶节点列/规则列时才能"
        "抄录，未提供时必须省略并由受控 Tool 使用默认值。不得填写 source artifact、"
        "artifact hash、asset hash、tree result hash、dataset/hash、workspace lineage、"
        "activate_result、结果或指标。它只创建 development / unvalidated 的不可变派生"
        "数据集，不激活当前 workspace；不得串联选叶、Strategy Pool、业务动作、报告、"
        "采纳或部署。"
        "对于 automatic_tree_leaf_materialization，只能逐字抄录用户原话中唯一的"
        "完整 tree_asset_id、唯一的完整 leaf_id，以及用户用“选择理由/理由/原因/说明”"
        "显式标注时的逐字 selection_reason；未显式标注时必须省略。它只创建"
        "pointer，不得复制规则、条件、指标、动作或平台 artifact/hash，也不得串联"
        "Strategy Pool、业务动作、采纳、部署或 leaf ID 写回。selection_reason 中也"
        "不得藏入理由替换、后续动作、生命周期操作或极值/排名选叶语义；它只接受"
        "人工/业务/风险/合规/样本评审依据类短说明。"
        "对于 interactive_tree_split_search，只能逐字抄录用户当前肯定命令中"
        "唯一完整的 source_tree_id、唯一完整的 node_id、明确的 "
        "all_features/selected_features 范围、每特征最大阈值数和总行评估"
        "预算；selected_features 还要逐字抄录唯一特征列表，all_features 必须"
        "省略 features。不得补默认预算，不得输出平台 artifact/hash、父链、"
        "condition、metrics、dataset/workspace/SampleDesign 或明细，不得同轮"
        "选胜者、改树、自动续建、入池、应用、报告、采纳或部署。"
        "对于 interactive_tree_auto_continuation，只能逐字抄录用户当前"
        "肯定命令中唯一完整的 search_id、唯一完整且由用户明确选择的 "
        "candidate_id、max_additional_depth、min_gini_gain、"
        "max_generated_nodes、max_thresholds_per_feature、"
        "max_row_evaluations，以及固定 objective=max_gini_gain 和 "
        "tie_break=eligible_gain_feature_threshold_candidate_id。所有控制"
        "都必须由用户明示，不得补默认值，不得按排名、最佳或第一名代选"
        "种子候选。不得输出或覆盖 source tree、node、artifact/hash、父链、"
        "condition、metrics、dataset/workspace/SampleDesign；这些由 search "
        "证据恢复。不得同轮串联改阈值、换变量、剪枝、入池、应用、报告、"
        "采纳或部署。"
        "对于 interactive_tree_revision，只能逐字抄录用户当前肯定命令中唯一完整的"
        " source_tree_id（candidate-asset- 或 interactive-tree-revision- 后接 32 位"
        "小写十六进制）、唯一完整的 split node_id（node- 后接 20 位小写十六进制）、"
        "唯一 operation=prune_subtree、adjust_split_threshold 或 "
        "replace_split_feature，以及用户显式标注时逐字一致的 reason。"
        "adjust_split_threshold 必须逐字抄录唯一有限 threshold；"
        "replace_split_feature 必须逐字抄录唯一 feature 和唯一有限 threshold；"
        "prune_subtree 必须省略 feature/threshold。“调好一点”“最佳阈值”"
        "“最佳特征”“自动优化”“全部节点”等模糊、推荐或批量修改请求必须 "
        "clarification。不得输出"
        "artifact/hash、父链、tree/frontier/condition/metrics、dataset/workspace/"
        "SampleDesign 或重放结果，这些均由平台恢复。不得按最好、风险最高、不稳定或"
        "代词替用户选节点，也不得同轮串联另一种树编辑、前沿物化、入池、业务动作、"
        "自动继续、整树应用、报告、采纳、部署或写回。"
        "对于 interactive_tree_frontier_group_materialization，只能逐字抄录用户"
        "当前肯定命令中唯一完整的 revision_id（interactive-tree-revision- 后接 "
        "32 位小写十六进制）、2 到 50 个互不重复的完整 source_node_ids（node- "
        "或 leaf- 后接 20 位小写十六进制），以及用户显式标注时逐字一致的 "
        "selection_reason。用户必须明确 OR/逻辑或/任一成员命中语义；成员输入顺序"
        "不具有语义，平台按 revision frontier 顺序规范化。不得输出 selection/group/"
        "revision artifact、hash、父链、semantic tree、fragment/rule/effect、condition/"
        "metrics、dataset/workspace/SampleDesign 或动作。不得用全部、最好、最差、"
        "风险最高、自动排名或代词替用户选节点，也不得同轮串联 Strategy Pool、"
        "业务动作、应用、采纳、部署或写回。"
        "对于 interactive_tree_frontier_materialization，只能逐字抄录用户当前肯定"
        "命令中唯一完整的 revision_id（interactive-tree-revision- 后接 32 位小写"
        "十六进制）、唯一完整的 source_node_id（node- 或 leaf- 后接 20 位小写"
        "十六进制），以及用户显式标注时逐字一致的 selection_reason。不得输出"
        "selection/revision artifact、hash、父链、semantic tree、fragment/rule/effect、"
        "condition/metrics、dataset/workspace/SampleDesign 或动作，这些均由平台恢复。"
        "不得按最好、最差、风险最高、自动排名或代词替用户选节点，也不得同轮串联"
        "Strategy Pool、业务动作、采纳、部署或写回。"
        "对于 voting_candidate_search，必须逐字抄录用户明确提供的 strategy_type、"
        "member_count/K、n 和 objective metric+direction；中文指标只能采用 system "
        "prompt 明列的别名并输出对应 canonical metric，禁止自行扩展近义词；"
        "constraints、include_rule_ids、"
        "exclude_rule_ids 未提供时固定为空数组，max_combinations 未提供时固定为 10000。"
        "include/exclude 只能采用当前句在对应标签后完整给出的 candidate-rule ID。"
        "最小化 bad_rate/weighted_bad_rate/bad_amount_rate 时，必须分别提供正数 "
        "hit_share/weighted_hit_share/hit_amount_share 的 gte 约束，绝对命中量不能"
        "替代。不得输出 Pool ref/revision/hash、dataset/target、逐行矩阵/target/weights/"
        "amounts、artifact、result 或已计算排名。该步骤只搜索，不构建、不选择、不修改"
        "或加入 Pool、不应用、不采纳、不部署；搜索/查找/优化 Voting 组合的原话必须优先"
        "路由到本 Workflow 或 clarification。"
        "对于 voting_candidate_build_from_search，只能逐字抄录用户当前请求中唯一完整"
        "的 search_id、唯一完整的 combo_id 和可选的唯一 strategy_type；strategy_type"
        " 未明确时必须省略。不得输出 artifact/hash、rule/entry/member IDs、n、rank、"
        "winner/champion、指标、结果或 Pool 身份，也不得按第一名、最好、Top N、"
        "刚才那个等表述选择组合。该步骤只构建候选；同轮串联入池、修改 Pool、设置"
        "动作、应用、采纳、部署或写回时必须 clarification。它必须优先于重新搜索和"
        "自由 rule ID Voting 构建路由。"
        "对于 voting_candidate_build，只能逐字抄录用户明确标注的 strategy_type、"
        "2 到 50 个完整 candidate-rule ID 和整数 n；不得输出 entry_id、Pool revision/hash、"
        "condition、指标、动作、推荐或平台数据绑定。规则集合必须全部来自同一条正向"
        "Voting/n-of-k 构建命令；‘最好规则’‘刚才那些’等启发式引用，或同一句串联入池、"
        "动作、采纳、部署、写回时必须澄清。问句、假设/未来/历史描述、演示文本、句尾"
        "撤销以及多个 strategy_type/n 候选也必须澄清；显式 k 必须与 rule_ids 数量一致。"
        "对于 candidate_monthly_stability，workflow_inputs 必须严格二选一：只抄录"
        "用户原话中唯一完整的 asset_id，或只抄录用户明确的 strategy_type 与唯一完整"
        "entry_id。禁止输出 source_kind、source artifact/hash、asset hash、Pool "
        "revision/hash、dataset/workspace/semantic、SampleDesign、target、month_col、"
        "基准、指标或结果；这些值由平台在 preflight 恢复。代词、多个 pointer、"
        "问句、否定、历史/未来/假设描述，或同轮串联入池、删改、重排、编译、写回、"
        "报告、采纳、部署时必须 clarification。"
        "对于 cross_matrix_candidate_search，只能逐字抄录用户唯一 "
        "features=[...] 列表中的 2 到 20 个白名单字段和明确的 1..190 "
        "max_pairs；两者都必须显式提供。不得填写轴方法、source artifact/"
        "candidate/evidence hash、dataset/target/sample、candidate asset、"
        "pair/rank/winner/champion、指标或结果。平台绑定精确父单变量证据和"
        " risk/development 样本，并由 Tool 为每个字段选择父证据中最高排名"
        "的可用方法。本步骤只搜索，不构建、不选择、不入池、不应用、不采纳、"
        "不部署；同轮后续动作必须 clarification。"
        "对于 cross_matrix_candidate_build_from_search，只能逐字抄录当前"
        "请求中唯一完整的 cross-search ID 与唯一完整的 cross-pair ID。不得"
        "输出 artifact/hash、轴字段/方法、asset fingerprint、rank/winner/"
        "champion、指标或结果，也不得按第一名、最好、Top N、刚才那个或代词"
        "选择。它必须是后续独立的单步构建请求；同轮重新搜索、入池、设置动作、"
        "应用、采纳、部署或写回必须 clarification。"
        "对于 cross_rule_search，只能逐字抄录用户唯一 features=[...] 列表中"
        "的 2 到 12 个白名单字段、dimension=2/3、完整 constraints"
        "（min_lift、min_bad_count、max_hit_share、min_amount_lift）和 1..5000 "
        "max_trials；所有控制都必须在当前请求显式提供。不得填写 source "
        "artifact/hash、dataset/target/sample、阈值、方向、rule/rank/winner/"
        "champion、指标或结果。平台从最新认证单变量证据恢复阈值和风险方向，"
        "并在 risk/development 样本做有预算的 2D/3D 聚合搜索。本步骤只搜索和"
        "排序全部已评估证据，不自动选择、不构建、不入池、不应用、不采纳、不部署。"
        "对于 cross_rule_candidate_build_from_search，只能逐字抄录当前请求中"
        "唯一完整的 cross-rule-search ID、唯一完整的 cross-rule ID，以及用户"
        "显式标注时的 selection_reason；未显式标注必须省略。不得输出 artifact/"
        "hash、条件、阈值、方向、rank/winner/champion、指标或结果，也不得按"
        "第一名、最好、Top N 或代词选择。它只精确构建一个未验证候选；入池、"
        "动作、应用、采纳或部署必须另发请求。"
        "对于 cross_matrix_analysis，只能抄录两个明确轴字段、各自方法及用户明确给出的"
        "单变量分析参数；不得输出平台数据绑定、目标列、预算、边界、cell、condition、"
        "指标、artifact/asset/effect/rule id、动作或推荐。它只构建二维矩阵证据，不能"
        "串联选格、入池、代码、写回、采纳或部署；明确的二维 Cross Matrix 请求不能"
        "改路由到其他 Workflow。"
        "对于 cross_matrix_cell_selection，只能逐字抄录用户原话中唯一完整的"
        "cross_asset_id、1 到 400 个互不重复的完整 cross-cell ID，以及显式标注时"
        "逐字一致的 selection_reason。禁止代词、排名、Top N、风险/指标极值或阈值"
        "选格。多个 cell 是集合语义并由平台按源矩阵顺序归一化为确定性 OR；不得输出"
        "condition、rule、effect、metrics、action 或任何 artifact/hash 平台绑定。它只"
        "创建 pointer，不得串联 Strategy Pool、业务动作、采纳、部署、投产或写回。"
        "对于 strategy_pool_add_candidate，candidate_asset_id 与 selection_id "
        "严格二选一且必须逐字抄录唯一完整 ID；必须分别抄录显式的策略池类型、"
        "Pool 默认动作和命中动作标签，不能对调或从动作反推 Pool 类型。reason 仅在"
        "显式标注时逐字抄录，未标注时省略；默认/命中 reason_code、output_value 和"
        "value 也必须逐字归属各自标签，不得省略或对调。可选 placement_mode 只能"
        "逐字抄录 before_selected_members/replace_selected_members，或从“保留成员"
        "作为回退并放在成员前/由 Voting 替代成员”二选一映射；用户未提供时省略。"
        "否定入池或串联采纳/部署时"
        "必须澄清。selection_id 只允许 automatic-tree-leaf-selection-、"
        "interactive-tree-frontier-group-selection-、"
        "interactive-tree-frontier-selection-、cross-matrix-cell-selection- 后接 "
        "32 位小写十六进制；完整 Cross Matrix asset"
        "不能直接入池。source ID 必须与唯一正向入池命令位于同一子句，不能从否定子句、"
        "reason、引用或代词上下文借用；未来/条件指令、问句、how-to、演示和测试也"
        "必须澄清。"
        "对于 strategy_pool_stability，只能逐字抄录用户当前肯定命令中唯一明确"
        "的五类 strategy_type。partitions、exact ImpactCube/Pool/SampleDesign "
        "artifact、revision/hash、dataset/workspace/target、阈值、PSI、分布、"
        "指标与结果全部禁止填写，由平台在计划创建和两步 Workflow 执行时冻结或"
        "确定性计算。否定、问句、历史/未来/假设、仅报告，或同轮修改 Pool、应用、"
        "创建、采纳、晋级、部署时必须 clarification。结果只是跨分区分布稳定性，"
        "不是独立效果验证，也不会修改 Pool 或进入策略生命周期。"
        "对于 strategy_impact_cube，只能抄录用户明确的五类 strategy_type、"
        "可选 development/validation/oot partitions、精确 month_col/group_col/"
        "segment_col、完整 current_strategy_id，以及 typed economics_inputs"
        "（每项仅 column 或有限 scalar）。不得输出 Pool/SampleDesign artifact、"
        "revision/hash、population、target、metrics、condition 或 strategy_spec。"
        "用户未指定分区时省略，由平台选择最新样本设计中全部非空可用分区；"
        "用户未指定维度列时省略，由平台仅绑定唯一确认语义角色。任何原话控制被遗漏、"
        "替换、否定，或同轮串联写回、报告、采纳、晋级、部署时必须澄清。"
        "对于 strategy_dsl_delivery，只能逐字抄录用户原话中唯一完整的可选"
        " strategy_id；未点名时必须省略。不得输出 strategy_ref、策略类型/version/"
        "spec hash、dataset_ref/hash、workspace_ref/revision/generation/"
        "semantic hash、等价样本预算、artifact id/hash、代码内容、"
        "等价结果或指标。它只导出离线 Python/SQL/JSON 与明确标注范围的等价证据，"
        "不得串联应用、写回、报告、影响测算、训练、评分、采纳、晋级或部署。"
        "对于 strategy_pool_impact，只能抄录用户明确的 approval/reject Pool 类型，"
        "可选 absolute/vs_baseline 比较模式、完整 baseline_strategy_id、精确 month_col/"
        "loan_amount_col/overdue_amount_col 和明确的 drop_nan_labels 布尔授权。普通肯定式"
        "请求可默认 absolute；vs_baseline 必须同时有原话中的比较表达和完整基线 ID。"
        "禁止输出 dataset/target、Pool revision/hash、workspace、sample binding、semantic hash、"
        "metrics、conditions 或 strategy_spec。用户未指定月份/金额列时必须省略，平台只会"
        "使用唯一确认的语义角色；没有角色则 unavailable，多个角色则澄清，Agent 不得猜列。"
        "limit/pricing/segmentation、否定/问句/历史/仅报告或同轮修改/采纳/部署必须澄清。"
        "对于 strategy_pool_validation，只能逐字抄录用户当前肯定命令中唯一"
        "明确的五类 strategy_type 和 validation/oot partition。"
        "Pool ref/revision/hash/artifact、SampleDesign membership/bundle/ref、"
        "dataset/workspace/target、requirements、population、comparison_mode、"
        "指标、月份、状态与结果全部禁止填写，由平台在计划创建和 Tool 执行时恢复。"
        "development、缺少或多个类型/分区、问句、"
        "否定、历史/未来/假设，或同轮修改 Pool、应用、报告、晋级、采纳、部署"
        "必须 clarification。它只发布 native typed independent replay evidence，"
        "不得声称 "
        "PSI、stability 或 drift，也不会修改 Pool、创建、晋级、采纳或部署策略。"
        "对于 strategy_pool_apply，只能抄录用户当前肯定命令中唯一明确的五类 "
        "strategy_type，以及用户以“输出前缀/output_prefix/output prefix/prefix”"
        "显式标注的可选 ASCII identifier output_prefix；未提供时必须省略并由 Tool"
        " 使用默认值。expected Pool revision/snapshot hash、Pool/artifact、dataset、"
        "SampleDesign、requirements、StrategySpec、指标和生命周期状态全部禁止填写，"
        "由平台在计划创建与执行时恢复。请求必须明确把当前 Pool 应用或写回当前样本，"
        "且必须是当前、肯定、单步骤命令；否定、问句、历史/未来/假设、模糊或多 Pool，"
        "或同轮串联 Pool 修改、采纳、激活、部署、上线、导出或报告必须 clarification。"
        "结果只创建不可变派生数据集，不激活当前 workspace，不采纳、不部署。"
        "对于 strategy_pool_materialize，只能抄录用户当前肯定命令中唯一明确的"
        "五类 strategy_type。Pool revision/snapshot hash、Pool artifact id/content "
        "hash、design hash、StrategySpec、requirements、指标和 lifecycle 全部禁止"
        "填写，由平台在计划创建和 Tool 执行时恢复。请求必须明确把当前 Pool 物化/"
        "固化/创建为 draft Strategy，且必须是当前、肯定、单步骤命令；否定、问句、"
        "历史/未来/假设、模糊或多 Pool，或同轮串联采纳、部署、回测、应用、报告、"
        "监控或 DSL 导出必须 clarification。本步骤只创建 draft Strategy，不采纳、"
        "不部署，也不声称后续 readiness。"
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
