"""Source-bounded semantic routing and normalization request adapters."""

from __future__ import annotations

import json

from marvis.llm_outbound import (
    OutboundContextError, StructuralPrompt, _DROP, _finite, _identifier, _mint_prompt, structural_summary,
)
from marvis.llm_outbound_instructions import user_instruction


_RISK_MATERIAL_QUESTION = (
    "平台正在等待用户确认材料已上传，并补充样本范围和字段口径；"
    "回答用于检查材料并准备已选分析。后续执行仍受原流程校验与确认门约束。"
)
_CONTEXT_ENUMS = {
    "risk_setup_phase": {"", "ask_goal", "request_materials", "await_materials", "ready"},
    "analysis_kind": {"profitability", "vtg_terminal", "standard_vintage"},
    "label_semantics": {"incremental", "snapshot"},
    "current_workflow": {"dataset_join", "strategy"},
    "join_exploration_phase": {"role_discovery", "role_target_confirmation"},
    "strategy_sample_binding_scope": {"authenticated_data_workspace_only_no_strategy_plan"},
    "pending": {"adhoc_query"},
    "task_status": {"draft", "running", "failed", "completed", "cancelled", "awaiting_confirm"},
}
_FLAGS = ("is_question", "is_conditional", "requests_change", "withholds_action")


def semantic_intent_context(value):
    if not isinstance(value, dict):
        raise OutboundContextError("语义意图上下文必须是结构对象，未发送请求。")
    clean = structural_summary(value)
    for key, allowed in _CONTEXT_ENUMS.items():
        if key in value:
            if not isinstance(value[key], str) or value[key] not in allowed:
                raise OutboundContextError("语义意图状态不属于当前应用契约，未发送请求。")
            clean[key] = value[key]
    for key in ("has_ready_dataset", "strategy_sample_binding_available"):
        if key in value:
            if type(value[key]) is not bool:
                raise OutboundContextError("语义意图状态类型无效，未发送请求。")
            clean[key] = value[key]
    for key in ("required_join_table_count", "available_join_table_count", "available_strategy_sample_count"):
        if key in value:
            if type(value[key]) is not int or not _finite(value[key]) or value[key] < 0:
                raise OutboundContextError("语义意图材料计数无效，未发送请求。")
            clean[key] = value[key]
    target = value.get("strategy_sample_target_candidate")
    if "strategy_sample_target_candidate" in value:
        if target is not None and _identifier(target) is _DROP:
            raise OutboundContextError("语义意图字段绑定无效，未发送请求。")
        clean["strategy_sample_target_candidate"] = target
    if (value.get("pending_question") == _RISK_MATERIAL_QUESTION
            and clean.get("risk_setup_phase") in {"request_materials", "await_materials"}
            and clean.get("analysis_kind") in _CONTEXT_ENUMS["analysis_kind"]):
        clean["pending_question"] = _RISK_MATERIAL_QUESTION
    return clean


def semantic_intent_prompt(*, task_type, instruction, context, allowed_intents):
    from marvis.agent.semantic_intent import ALL_SEMANTIC_INTENTS, _INTENT_DEFINITIONS
    from marvis.domain import VALID_TASK_TYPES

    allowed = list(allowed_intents)
    if task_type not in VALID_TASK_TYPES or not allowed or any(item not in ALL_SEMANTIC_INTENTS for item in allowed):
        raise OutboundContextError("语义意图任务或枚举未获批准，未发送请求。")
    approved = user_instruction(instruction)
    return _mint_prompt({
        "task_type": task_type, "instruction": str(approved),
        "current_context": semantic_intent_context(dict(context)), "allowed_intents": allowed,
        "intent_definitions": {intent: _INTENT_DEFINITIONS[intent] for intent in allowed},
    })


def semantic_intent_repair_prompt(prompt, *, fields, failure_code, had_unexpected_fields):
    from marvis.agent.semantic_diagnostics import _FAILURES
    from marvis.agent.semantic_intent import _FIELDS

    if not isinstance(prompt, StructuralPrompt):
        raise OutboundContextError("语义修复需要已批准的原始请求，未发送请求。")
    original = json.loads(prompt)
    allowed = original.get("allowed_intents", [])
    instruction = original.get("instruction")
    if not isinstance(instruction, str) or fields.get("intent") not in allowed or failure_code not in _FAILURES:
        raise OutboundContextError("语义修复缺少有效原始绑定，未发送请求。")
    if fields.get("confidence") not in {"high", "medium", "low"} or any(type(fields.get(key)) is not bool for key in _FLAGS):
        raise OutboundContextError("语义修复决策字段无效，未发送请求。")
    bounded = {"intent": fields["intent"], "confidence": fields["confidence"],
               **{key: fields[key] for key in _FLAGS}}
    quote = fields.get("evidence_quote")
    if isinstance(quote, str) and quote.strip() and quote in instruction:
        bounded["evidence_quote"] = quote
    if type(had_unexpected_fields) is not bool:
        raise OutboundContextError("语义修复响应分类无效，未发送请求。")
    return _mint_prompt({
        "operation": "normalize_existing_semantic_intent_decision", "original_request": original,
        "first_pass_fields": bounded, "first_pass_failure_code": failure_code,
        "had_unexpected_fields": had_unexpected_fields, "required_fields": list(_FIELDS),
    })


def workflow_intent_prompt(goal, task_context, candidates):
    from marvis.orchestrator.templates import list_templates

    installed = {template.id for template in list_templates()} | {"novel"}
    if not candidates or len(candidates) > 10000 or any(item not in installed for item in candidates):
        raise OutboundContextError("工作流候选目录无效或超限，未发送请求。")
    return _mint_prompt({
        "goal": str(user_instruction(goal)), "task_context": structural_summary(task_context),
        "candidates": list(candidates), "instruction": 'Return JSON like {"choice":"candidate_id"}.',
    })
