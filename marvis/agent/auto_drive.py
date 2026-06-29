"""Agent-mode gate operator.

Manual mode = the user operates the controls (reads each result, clicks 「确认」 at
every gate). Agent mode = an LLM operates those same controls: at each confirmation
gate it reviews the just-computed result and decides whether to proceed or halt for
human review. That is the *whole* difference between the two modes — the underlying
deterministic plan flow is identical, agent mode just hands the gate decisions to an
LLM (which is why agent mode requires a configured LLM and errors without one).

This module is pure + offline-testable: the LLM client is injected, so a FakeLLM
drives ``decide_gate`` in tests (the platform may have no LLM configured yet, in
which case the user asked us to simulate the LLM ourselves).
"""

from __future__ import annotations

import re
from typing import Any

from marvis.agent.gates import DEFAULT_GATE_ACTIONS, extract_gate_envelope
from marvis.agent.json_reply import load_json_object

_SYSTEM_TEMPLATE = (
    "你是信贷风控建模 Agent,正在自动执行一个分步计划。每到一个需要确认的节点,"
    "你会看到刚刚算出的结果(可能含表格)。请只在当前节点声明允许的动作内决策。\n"
    "允许动作:{allowed_actions}\n"
    "- confirm: 结果正常,继续下一步;\n"
    "- adjust: 仅在当前节点允许且参数/选择可安全调整时使用,必须返回 params/selection/dedup_strategies;\n"
    "- replan: 当前计划结构需要改变时使用,必须返回 replan_goal;\n"
    "- clarify: 需要用户补充一个明确问题时使用,必须返回 clarifying_question;\n"
    "- halt: 结果异常或动作超出权限,停下来请人工核对。\n"
    "严格只返回 JSON 对象。字段: action, reason, params, selection, dedup_strategies,"
    " replan_goal, clarifying_question, confidence。"
)


def decide_gate(client, *, gate: dict) -> dict:
    """Ask the injected LLM client to operate one confirmation gate.

    ``gate`` is the latest assistant gate message ``{content, metadata}``. Returns a
    normalized decision. Legacy gates allow only ``confirm`` / ``halt``; gates that
    carry a ``gate_envelope`` may also allow bounded structured actions.
    """
    prompt = _format_gate(gate)
    envelope = extract_gate_envelope(gate)
    allowed_actions = envelope.allowed_actions
    raw = client.complete(
        system_prompt=_system_prompt(allowed_actions),
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
    )
    decision, ok = _parse_decision(raw, allowed_actions=allowed_actions)
    decision = _apply_safety_policy(decision, envelope)
    if ok:
        return decision
    retry_prompt = (
        f"{prompt}\n\n"
        f"【上一次返回无法解析】\n{raw}\n\n"
        f"请严格只返回 JSON 对象,action 必须是以下之一:{', '.join(allowed_actions)}。"
    )
    raw = client.complete(
        system_prompt=_system_prompt(allowed_actions),
        user_prompt=retry_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
    )
    return _apply_safety_policy(parse_decision(raw, allowed_actions=allowed_actions), envelope)


def _system_prompt(allowed_actions: tuple[str, ...]) -> str:
    return _SYSTEM_TEMPLATE.format(allowed_actions=", ".join(allowed_actions))


def _format_gate(gate: dict) -> str:
    lines = [str(gate.get("content") or "")]
    meta = gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {}
    red_flags = _extract_red_flags(gate)
    if red_flags:
        lines.append("")
        lines.append("【平台红旗 checklist】")
        lines.append("以下异常由平台工具确定性产出,请重点复核;若不能解释清楚,优先 halt:")
        for flag in red_flags[:12]:
            lines.append(f"- {flag}")
    modeling_setup = meta.get("modeling_setup") if isinstance(meta.get("modeling_setup"), dict) else {}
    if modeling_setup:
        lines.append("")
        lines.append("【建模规格控件】")
        lines.append(f"- target_type: {modeling_setup.get('target_type') or 'binary'}")
        recipes = modeling_setup.get("recipes") or []
        if recipes:
            lines.append(f"- recipes: {', '.join(str(item) for item in recipes)}")
        selected_weight = str(modeling_setup.get("sample_weight_col") or "")
        candidates = [str(item) for item in (modeling_setup.get("sample_weight_candidates") or []) if str(item)]
        lines.append(f"- sample_weight_col: {selected_weight or '不使用'}")
        if candidates:
            lines.append(f"- sample_weight_candidates: {', '.join(candidates)}")
    for table in meta.get("tables") or []:
        lines.append("")
        lines.append(f"表:{table.get('title', '')}")
        columns = table.get("columns") or []
        if columns:
            lines.append(" | ".join(str(c) for c in columns))
        for row in (table.get("rows") or [])[:20]:
            lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def _extract_red_flags(gate: dict) -> list[str]:
    flags: list[str] = []
    meta = gate.get("metadata") if isinstance(gate.get("metadata"), dict) else {}
    content = str(gate.get("content") or "")
    if "行数发生变化" in content or ("膨胀" in content and "⚠" in content):
        flags.append("拼接执行后行数发生变化或存在膨胀提示")

    screen = meta.get("screen") if isinstance(meta.get("screen"), dict) else {}
    leakage = screen.get("leakage") or []
    suspected = screen.get("suspected") or []
    unusable = screen.get("unusable") or []
    if leakage:
        flags.append(f"筛选发现 {len(leakage)} 个疑似硬泄漏特征")
    if suspected:
        flags.append(f"筛选发现 {len(suspected)} 个可疑模型输出/泄漏特征")
    if unusable:
        flags.append(f"筛选发现 {len(unusable)} 个不可用特征")

    dedup = meta.get("dedup") if isinstance(meta.get("dedup"), dict) else {}
    needs_dedup = dedup.get("needs_dedup") or []
    if needs_dedup:
        flags.append(f"拼接存在 {len(needs_dedup)} 张特征表同键冲突,需确认去重策略")

    for table in meta.get("tables") or []:
        flags.extend(_table_red_flags(table))
    return _dedupe(flags)


def _table_red_flags(table: dict) -> list[str]:
    flags: list[str] = []
    title = str(table.get("title") or "")
    columns = [str(column) for column in (table.get("columns") or [])]
    match_idx = _first_column_index(columns, "命中率")
    fanout_idx = _first_column_index(columns, "膨胀")
    dedup_idx = _first_column_index(columns, "去重")
    feature_idx = 0 if columns else None
    for row in (table.get("rows") or [])[:20]:
        cells = list(row) if isinstance(row, (list, tuple)) else []
        label = _cell(cells, feature_idx) or title or "当前表"
        if match_idx is not None:
            match_rate = _parse_rate(_cell(cells, match_idx))
            if match_rate is not None and match_rate < 0.2:
                flags.append(f"{label} 命中率偏低({match_rate:.2%})")
        if fanout_idx is not None:
            fanout = _cell(cells, fanout_idx)
            if "⚠" in fanout or fanout.strip() in {"是", "true", "True", "1"}:
                flags.append(f"{label} 存在拼接膨胀风险")
        if dedup_idx is not None:
            dedup = _cell(cells, dedup_idx)
            if "冲突" in dedup and "⚠" in dedup:
                flags.append(f"{label} 存在同键冲突去重风险")
    return flags


def _first_column_index(columns: list[str], token: str) -> int | None:
    for index, column in enumerate(columns):
        if token in column:
            return index
    return None


def _cell(cells: list, index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return str(cells[index])


def _parse_rate(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text.lower() == "n/a":
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    number = float(match.group(0))
    return number / 100.0 if "%" in text else number


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_decision(raw: str, *, allowed_actions: tuple[str, ...] = DEFAULT_GATE_ACTIONS) -> dict:
    """Normalize the LLM's JSON reply; default to a *safe* halt on any junk."""
    decision, _ok = _parse_decision(raw, allowed_actions=allowed_actions)
    return decision


def _parse_decision(raw, *, allowed_actions: tuple[str, ...] = DEFAULT_GATE_ACTIONS) -> tuple[dict, bool]:
    data, error = load_json_object(raw)
    if data is None:
        return {"action": "halt", "reason": "无法解析模型决策,转人工确认。"}, False
    action = str(data.get("action") or "").strip().lower()
    allowed = _normalize_allowed_actions(allowed_actions)
    if action not in allowed:
        reason = str(data.get("reason") or "").strip()
        suffix = f" 原因:{reason}" if reason else ""
        return {
            "action": "halt",
            "reason": f"模型返回了当前节点不允许的动作 `{action or '空'}`,转人工确认。{suffix}",
        }, False
    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = "结果正常,继续。" if action == "confirm" else "请人工确认。"
    decision: dict[str, Any] = {"action": action, "reason": reason}
    if action == "adjust":
        decision["params"] = _object_or_empty(data.get("params"))
        selection = data.get("selection")
        if isinstance(selection, list):
            decision["selection"] = [item for item in selection if isinstance(item, (str, int, float))]
        dedup_strategies = _object_or_empty(data.get("dedup_strategies"))
        if dedup_strategies:
            decision["dedup_strategies"] = {str(k): str(v) for k, v in dedup_strategies.items()}
    elif action == "replan":
        decision["replan_goal"] = str(data.get("replan_goal") or "").strip()
    elif action == "clarify":
        decision["clarifying_question"] = str(data.get("clarifying_question") or "").strip()
    confidence = _confidence(data.get("confidence"))
    if confidence is not None:
        decision["confidence"] = confidence
    return decision, error is None


def _apply_safety_policy(decision: dict, envelope) -> dict:
    """Final deterministic guard before an AUTO decision reaches PlanDriver.

    The LLM may only operate controls explicitly declared by the current
    GateEnvelope. This keeps AUTO bounded to low-risk, typed controls and turns
    unlisted actions such as expensive tuning, algorithm swaps, export/handoff, or
    arbitrary downstream resets into a halt for human review.
    """
    action = decision.get("action")
    if action == "adjust":
        allowed_controls = {str(control.id) for control in getattr(envelope, "controls", ())}
        params = _object_or_empty(decision.get("params"))
        unknown_params = sorted(str(key) for key in params if str(key) not in allowed_controls)
        if unknown_params:
            return _policy_halt(
                f"AUTO 返回了当前节点未声明的调整参数:{', '.join(unknown_params)}。"
            )
        if decision.get("selection") and "selection" not in allowed_controls:
            return _policy_halt("AUTO 试图调整特征选择,但当前节点没有声明 selection 控件。")
        if decision.get("dedup_strategies") and "dedup_strategies" not in allowed_controls:
            return _policy_halt("AUTO 试图设置去重策略,但当前节点没有声明 dedup_strategies 控件。")
    if action == "replan" and not str(decision.get("replan_goal") or "").strip():
        return _policy_halt("AUTO 请求重规划但没有提供明确 replan_goal。")
    return decision


def _policy_halt(reason: str) -> dict:
    return {"action": "halt", "reason": f"{reason} 已转人工确认。"}


def _normalize_allowed_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    allowed: list[str] = []
    for action in actions or DEFAULT_GATE_ACTIONS:
        item = str(action or "").strip().lower()
        if item and item not in allowed:
            allowed.append(item)
    return tuple(allowed) or DEFAULT_GATE_ACTIONS


def _object_or_empty(value) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _confidence(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


__all__ = ["decide_gate", "parse_decision"]
