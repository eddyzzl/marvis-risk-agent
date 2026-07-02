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
from marvis.llm_client import DEFAULT_CONTEXT_WINDOW, estimate_tokens
from marvis.llm_prompts import GATE_SYSTEM_TEMPLATE as _GATE_SYSTEM_TEMPLATE_SPEC
from marvis.orchestrator.context.budget import truncate_text_to_token_budget

AUTO_SAFE_ADJUST_CONTROLS = frozenset({
    "dedup_strategies",
    "leakage_ks",
    "max_missing_rate",
    "sample_weight_col",
    "selection",
})
AUTO_HIGH_RISK_FLAG_TOKENS = frozenset({
    "destructive",
    "delivery",
    "export",
    "external",
    "handoff",
    "high-risk",
    "high_risk",
    "irreversible",
    "approval",
    "champion",
    "delete",
    "deploy",
    "drop",
    "manual-review",
    "manual_review",
    "overwrite",
    "production",
    "promote",
    "publish",
    "release",
    "requires-human",
    "requires_human",
    "side-effect",
    "side_effect",
    "wide-reset",
    "wide_reset",
})
AUTO_HIGH_RISK_RESET_SCOPES = frozenset({
    "all",
    "all_steps",
    "broad",
    "entire_plan",
    "full_plan",
    "wide",
})
AUTO_MAX_AUTO_RESET_STEPS = 2
# AGT-7: decide_gate already parses a confidence score out of the LLM's JSON
# reply, but nothing consumed it — a confirm at confidence=0.3 and one at 0.95
# were treated identically. Below this threshold a confirm/adjust/replan is
# downgraded to halt so a low-confidence AUTO decision always reaches a human.
AUTO_MIN_CONFIDENCE = 0.6

# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of _SYSTEM_TEMPLATE from here keep working unchanged.
_SYSTEM_TEMPLATE = _GATE_SYSTEM_TEMPLATE_SPEC.text


def decide_gate(client, *, gate: dict) -> dict:
    """Ask the injected LLM client to operate one confirmation gate.

    ``gate`` is the latest assistant gate message ``{content, metadata}``. Returns a
    normalized decision. Legacy gates allow only ``confirm`` / ``halt``; gates that
    carry a ``gate_envelope`` may also allow bounded structured actions.
    """
    prompt = _format_gate(gate)
    envelope = extract_gate_envelope(gate)
    allowed_actions = envelope.allowed_actions
    schema = _gate_decision_schema(allowed_actions)
    system_prompt = _system_prompt(allowed_actions)
    # LLM-5: this is one of the three named highest-volume touch points (gate
    # content can carry many wide tables) — truncate proactively instead of
    # relying solely on complete()'s pre-flight rejection, so a busy gate still
    # gets a decision instead of an outright context-window error.
    prompt, truncated = _truncate_gate_prompt(prompt, client, system_prompt)
    raw = client.complete(
        system_prompt=system_prompt,
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=schema,
        stream=False,
        caller="gate",
        prompt_name=_GATE_SYSTEM_TEMPLATE_SPEC.name,
        prompt_version=_GATE_SYSTEM_TEMPLATE_SPEC.version,
        truncated=truncated,
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
        system_prompt=system_prompt,
        user_prompt=retry_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=schema,
        stream=False,
        caller="gate",
        prompt_name=_GATE_SYSTEM_TEMPLATE_SPEC.name,
        prompt_version=_GATE_SYSTEM_TEMPLATE_SPEC.version,
        truncated=truncated,
    )
    return _apply_safety_policy(parse_decision(raw, allowed_actions=allowed_actions), envelope)


# LLM-5: reserve roughly a quarter of the model's context window for the gate
# content — the rest covers the system prompt, JSON schema, retry round-trip and
# completion budget. This is deliberately conservative (a fixed fraction, not a
# tight computation) since the exact system-prompt/schema overhead varies by
# gate type.
_GATE_PROMPT_BUDGET_FRACTION = 0.25


def _truncate_gate_prompt(prompt: str, client, system_prompt: str) -> tuple[str, bool]:
    profile = getattr(client, "profile", None) or {}
    context_window = int(profile.get("context_window") or DEFAULT_CONTEXT_WINDOW)
    budget = max(int(context_window * _GATE_PROMPT_BUDGET_FRACTION), 256)
    if estimate_tokens(prompt) <= budget:
        return prompt, False
    return truncate_text_to_token_budget(prompt, max_tokens=budget)


def _system_prompt(allowed_actions: tuple[str, ...]) -> str:
    return _SYSTEM_TEMPLATE.format(allowed_actions=", ".join(allowed_actions))


def _gate_decision_schema(allowed_actions: tuple[str, ...]) -> dict:
    """Skeleton json_schema for a gate decision; the validator still owns detail."""
    return {
        "name": "gate_decision",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(allowed_actions)},
                "reason": {"type": "string"},
                "params": {"type": "object"},
                "selection": {"type": "array"},
                "dedup_strategies": {"type": "object"},
                "replan_goal": {"type": "string"},
                "clarifying_question": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["action", "reason"],
            "additionalProperties": True,
        },
    }


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
    memory_anchor = meta.get("memory_anchor")
    anchor_lines = [str(item) for item in memory_anchor] if isinstance(memory_anchor, list) else []
    if anchor_lines:
        lines.append("")
        lines.append("【历史同类实验(只读参照)】")
        lines.append("以下来自 agent memory 的历史同类实验,仅供参考,不得替代平台实测指标:")
        for line in anchor_lines:
            lines.append(f"- {line}")
    modeling_setup = meta.get("modeling_setup") if isinstance(meta.get("modeling_setup"), dict) else {}
    envelope = extract_gate_envelope(gate)
    if envelope.risk_flags:
        lines.append("")
        lines.append("【Gate 风险标记】")
        for flag in envelope.risk_flags[:8]:
            lines.append(f"- {flag}")
    reset_policy = envelope.downstream_reset_policy or {}
    if reset_policy:
        lines.append("")
        lines.append("【下游重置策略】")
        for key in ("scope", "count", "step_count", "step_ids"):
            if key in reset_policy:
                lines.append(f"- {key}: {reset_policy[key]}")
    if modeling_setup:
        lines.append("")
        lines.append("【建模规格控件】")
        lines.append(f"- target_type: {modeling_setup.get('target_type') or 'binary'}")
        recipes = modeling_setup.get("recipes") or []
        if recipes:
            lines.append(f"- recipes: {', '.join(str(item) for item in recipes)}")
        if modeling_setup.get("feature_count") is not None:
            lines.append(f"- feature_count: {modeling_setup.get('feature_count')}")
        if modeling_setup.get("n_trials") is not None:
            lines.append(f"- n_trials: {modeling_setup.get('n_trials')}")
        if modeling_setup.get("metric_policy"):
            lines.append(f"- metric_policy: {modeling_setup.get('metric_policy')}")
        split_summary = (
            modeling_setup.get("split_summary")
            if isinstance(modeling_setup.get("split_summary"), dict)
            else {}
        )
        if split_summary:
            lines.append(f"- split_col: {split_summary.get('split_col') or 'split'}")
            counts = split_summary.get("split_counts") if isinstance(split_summary.get("split_counts"), dict) else {}
            if counts:
                lines.append(
                    "- split_counts: "
                    + ", ".join(f"{key}={value}" for key, value in counts.items())
                )
            for warning in split_summary.get("warnings") or []:
                lines.append(f"- split_warning: {warning}")
        selected_weight = str(modeling_setup.get("sample_weight_col") or "")
        candidates = [str(item) for item in (modeling_setup.get("sample_weight_candidates") or []) if str(item)]
        lines.append(f"- sample_weight_col: {selected_weight or '不使用'}")
        if candidates:
            lines.append(f"- sample_weight_candidates: {', '.join(candidates)}")
        diagnostics = [
            item for item in (modeling_setup.get("sample_weight_diagnostics") or [])
            if isinstance(item, dict)
        ]
        for item in diagnostics[:5]:
            column = str(item.get("column") or "")
            if not column:
                continue
            status = "valid" if item.get("valid") else "invalid"
            lines.append(
                "- sample_weight_diagnostic: "
                f"{column} {status}, missing_rate={item.get('missing_rate')}, "
                f"min={item.get('min')}, max={item.get('max')}, reason={item.get('reason') or ''}"
            )
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

    # AGT-9: prefer the composer-computed deterministic red flags (modeling
    # tuning-config / select-experiment gates, and any future gate that starts
    # populating meta['red_flags']) over parsing table strings — this is a
    # structured, INV-1-safe source, not a legacy fallback.
    structured = [str(item) for item in meta.get("red_flags") or [] if str(item).strip()]
    flags.extend(structured)

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
    arbitrary downstream resets into a halt for human review. Gates that explicitly
    declare human-review risk cannot be auto-confirmed either.
    """
    action = decision.get("action")
    if action in {"confirm", "adjust", "replan"}:
        gate_risk = _gate_risk_reason(envelope)
        if gate_risk:
            return _policy_halt(gate_risk)
        confidence = decision.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < AUTO_MIN_CONFIDENCE:
            return _policy_halt(
                f"AUTO 决策置信度 {confidence:.2f} 低于阈值 {AUTO_MIN_CONFIDENCE},不足以自动{action}。"
            )
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
        requested_controls = set(str(key) for key in params)
        if decision.get("selection"):
            requested_controls.add("selection")
        if decision.get("dedup_strategies"):
            requested_controls.add("dedup_strategies")
        unsafe_controls = sorted(control for control in requested_controls if control not in AUTO_SAFE_ADJUST_CONTROLS)
        if unsafe_controls:
            return _policy_halt(
                f"AUTO 请求了需人工确认的高风险控件:{', '.join(unsafe_controls)}。"
            )
    if action == "replan" and not str(decision.get("replan_goal") or "").strip():
        return _policy_halt("AUTO 请求重规划但没有提供明确 replan_goal。")
    return decision


def _gate_risk_reason(envelope) -> str:
    for flag in getattr(envelope, "risk_flags", ()) or ():
        text = str(flag or "").strip()
        normalized = text.lower().replace(" ", "_")
        if any(token in normalized for token in AUTO_HIGH_RISK_FLAG_TOKENS):
            return f"当前节点带有需人工确认的风险标记:{text}。"
    reset_policy = getattr(envelope, "downstream_reset_policy", {}) or {}
    scope = str(
        reset_policy.get("scope")
        or reset_policy.get("mode")
        or reset_policy.get("downstream_reset")
        or getattr(getattr(envelope, "retry_policy", None), "downstream_reset", "")
        or ""
    ).strip().lower()
    if scope in AUTO_HIGH_RISK_RESET_SCOPES:
        return f"当前节点声明了大范围下游重置策略:{scope}。"
    reset_count = _reset_step_count(reset_policy)
    if reset_count is not None and reset_count > AUTO_MAX_AUTO_RESET_STEPS:
        return f"当前节点会重置 {reset_count} 个下游步骤,超出 AUTO 自动调整上限。"
    return ""


def _reset_step_count(reset_policy: dict[str, Any]) -> int | None:
    for key in ("count", "step_count", "affected_step_count"):
        try:
            return int(reset_policy[key])
        except (KeyError, TypeError, ValueError):
            continue
    step_ids = reset_policy.get("step_ids") or reset_policy.get("steps") or reset_policy.get("reset_step_ids")
    if isinstance(step_ids, list):
        return len(step_ids)
    return None


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
