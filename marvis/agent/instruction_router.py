"""Agent-mode gate-instruction router.

At a confirmation gate the user (agent mode) may reply with free text instead of
「确认」 — e.g. "阈值放宽到 0.1", "n_trials 调到 20", "改用 xgb 重新建模". This routes
that instruction into a structured action the driver can execute:

  - confirm: the text actually means "proceed" → confirm the gate.
  - adjust:  tweak the parameters of the just-computed step and re-run it.
  - replan:  a structural change (add/remove steps, switch algorithm) → regenerate
             the remaining plan with the instruction as a constraint.
  - clarify: the instruction is unclear / unactionable → ask the user.

Pure + offline-testable: the LLM client is injected, so a FakeLLM drives it in
tests (the platform may have no LLM configured yet).
"""

from __future__ import annotations

from marvis.agent.json_reply import load_json_object
from marvis.llm_prompts import GATE_INSTRUCTION_ROUTER_SYS as _GATE_INSTRUCTION_ROUTER_SYS_SPEC

_ACTIONS = ("confirm", "adjust", "replan", "clarify")

# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of _SYSTEM from here keep working unchanged.
_SYSTEM = _GATE_INSTRUCTION_ROUTER_SYS_SPEC.text

_ROUTE_SCHEMA = {
    "name": "gate_instruction_route",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "params": {"type": "object"},
            "constraint": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["action", "params", "constraint", "reason"],
        "additionalProperties": True,
    },
}


def route_instruction(client, *, gate_context, instruction, tables=None, param_schema=None):
    """Ask the injected LLM to classify one free-text gate instruction.

    ``param_schema`` (optional, AGT-5): the current gate's adjustable-parameter
    summary — a list of ``{"name", "type", "current", "bounds"}`` dicts assembled
    from the gate's dependency step inputs (see
    ``marvis.agent.gate_param_schema.gate_param_schema``). Injected into the
    prompt so the routing LLM extracts ``adjust`` params against real parameter
    names/bounds instead of guessing key names from the instruction text alone."""
    prompt = _format(gate_context, instruction, tables or [], param_schema or [])
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_ROUTE_SCHEMA,
        stream=False,
        caller="router",
    )
    route, ok = _parse_route(raw)
    if ok:
        return route
    retry_prompt = (
        f"{prompt}\n\n"
        f"【上一次返回无法解析】\n{raw}\n\n"
        '请严格只返回 JSON 对象:{"action":"confirm|adjust|replan|clarify","params":{},'
        '"constraint":"","reason":"一句话中文"}。'
    )
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=retry_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_ROUTE_SCHEMA,
        stream=False,
        caller="router",
    )
    return parse_route(raw)


_MAX_PARAM_SCHEMA_ITEMS = 12
_MAX_PARAM_VALUE_CHARS = 80


def _format(gate_context, instruction, tables, param_schema):
    lines = ["【当前节点】", str(gate_context or "")]
    for table in tables:
        lines.append(f"表:{table.get('title', '')} 列={table.get('columns')}")
    schema_lines = _format_param_schema(param_schema)
    if schema_lines:
        lines.append("【可调参数】(adjust 的 params 键只能取自这里)")
        lines.extend(schema_lines)
    lines.append("【用户指令】")
    lines.append(str(instruction or ""))
    return "\n".join(lines)


def _format_param_schema(param_schema) -> list[str]:
    """Render a length-bounded 参数名/类型/当前值(/取值范围) summary line per
    adjustable parameter (AGT-5). Silently drops malformed entries rather than
    erroring — this is prompt context, not a validated control payload."""
    lines: list[str] = []
    for item in list(param_schema or [])[:_MAX_PARAM_SCHEMA_ITEMS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        kind = str(item.get("type") or "").strip() or "unknown"
        current = _truncate(item.get("current"))
        bounds = item.get("bounds") if isinstance(item.get("bounds"), dict) else None
        line = f"- {name} (类型={kind}, 当前值={current})"
        if bounds:
            bounds_text = ", ".join(f"{k}={v}" for k, v in bounds.items())
            line += f" 取值范围: {bounds_text}"
        lines.append(line)
    return lines


def _truncate(value) -> str:
    text = str(value if value is not None else "-")
    if len(text) > _MAX_PARAM_VALUE_CHARS:
        return text[:_MAX_PARAM_VALUE_CHARS] + "…"
    return text


def parse_route(raw):
    """Normalize the LLM reply; default to a safe clarify on junk or empty adjust."""
    route, _ok = _parse_route(raw)
    return route


def _parse_route(raw) -> tuple[dict, bool]:
    data, error = load_json_object(raw)
    if data is None:
        return {"action": "clarify", "params": {}, "constraint": "", "reason": "无法解析指令,请换种说法。"}, False
    action = str(data.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        action = "clarify"
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    constraint = str(data.get("constraint") or "").strip()
    reason = str(data.get("reason") or "").strip()
    # An "adjust" with no extractable parameters is not actionable → clarify.
    if action == "adjust" and not params:
        action = "clarify"
        reason = reason or "没识别到要调整的参数,请写明参数名和取值。"
    return {"action": action, "params": params, "constraint": constraint, "reason": reason}, error is None


__all__ = ["route_instruction", "parse_route"]
