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

import json

_SYSTEM = (
    "你是信贷风控建模 Agent,正在自动执行一个分步计划。每到一个需要确认的节点,"
    "你会看到刚刚算出的结果(可能含表格)。请判断结果是否合理、是否可以继续:\n"
    "- 结果正常 → action=confirm,继续下一步;\n"
    "- 结果明显异常(如命中率极低、拼接后行数膨胀、目标列缺失、指标明显异常)"
    "→ action=halt,停下来请人工核对。\n"
    '严格只返回 JSON:{"action":"confirm"|"halt","reason":"一句话中文理由"}。'
)


def decide_gate(client, *, gate: dict) -> dict:
    """Ask the injected LLM client to operate one confirmation gate.

    ``gate`` is the latest assistant gate message ``{content, metadata}``. Returns a
    normalized ``{"action": "confirm"|"halt", "reason": str}`` decision.
    """
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=_format_gate(gate),
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
    )
    return parse_decision(raw)


def _format_gate(gate: dict) -> str:
    lines = [str(gate.get("content") or "")]
    meta = gate.get("metadata") or {}
    for table in meta.get("tables") or []:
        lines.append("")
        lines.append(f"表:{table.get('title', '')}")
        columns = table.get("columns") or []
        if columns:
            lines.append(" | ".join(str(c) for c in columns))
        for row in (table.get("rows") or [])[:20]:
            lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def parse_decision(raw: str) -> dict:
    """Normalize the LLM's JSON reply; default to a *safe* halt on any junk."""
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("decision is not a JSON object")
    except (ValueError, TypeError):
        return {"action": "halt", "reason": "无法解析模型决策,转人工确认。"}
    action = str(data.get("action") or "").strip().lower()
    if action not in ("confirm", "halt"):
        action = "halt"
    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = "结果正常,继续。" if action == "confirm" else "请人工确认。"
    return {"action": action, "reason": reason}


__all__ = ["decide_gate", "parse_decision"]
