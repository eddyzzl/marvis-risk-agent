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

_ACTIONS = ("confirm", "adjust", "replan", "clarify")

_SYSTEM = (
    "你是信贷风控建模 Agent。用户在一个需要确认的节点没有直接确认,而是提了一条指令。"
    "判断该指令属于哪类并抽取要素:\n"
    "- confirm:其实是同意继续(如\"可以\"\"没问题\")。\n"
    "- adjust:调整刚算出这一步的参数后重算(如\"n_trials 调到 20\"\"阈值放宽到 0.1\")。"
    "把参数抽成 params 字典(键=参数名,值=新值,数字请用数字)。\n"
    "- replan:结构性改动(加/删步骤、换算法、换流程),把诉求写进 constraint。\n"
    "- clarify:看不懂或信息不足。\n"
    '严格只返回 JSON:'
    '{"action":"confirm|adjust|replan|clarify","params":{},"constraint":"","reason":"一句话中文"}。'
)

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


def route_instruction(client, *, gate_context, instruction, tables=None):
    """Ask the injected LLM to classify one free-text gate instruction."""
    prompt = _format(gate_context, instruction, tables or [])
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


def _format(gate_context, instruction, tables):
    lines = ["【当前节点】", str(gate_context or "")]
    for table in tables:
        lines.append(f"表:{table.get('title', '')} 列={table.get('columns')}")
    lines.append("【用户指令】")
    lines.append(str(instruction or ""))
    return "\n".join(lines)


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
