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

import json

from marvis.agent.adjust_specs import normalize_adjust_params
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
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "explicit_authorization": {"type": "boolean"},
        },
        "required": [
            "action",
            "params",
            "constraint",
            "reason",
            "confidence",
            "explicit_authorization",
        ],
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
        prompt_name=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.name,
        prompt_version=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.version,
    )
    route, ok = _parse_route(raw)
    if ok:
        route = _recover_declared_parameter_adjustment(
            client,
            route=route,
            prompt=prompt,
            instruction=instruction,
            param_schema=param_schema or [],
        )
        return _recover_declared_selection_decision(
            client,
            route=route,
            prompt=prompt,
            param_schema=param_schema or [],
        )
    retry_prompt = (
        f"{prompt}\n\n"
        f"【上一次返回无法解析】\n{raw}\n\n"
        '请严格只返回 JSON 对象:{"action":"confirm|adjust|replan|clarify","params":{},'
        '"constraint":"","reason":"一句话中文","confidence":"high|medium|low",'
        '"explicit_authorization":false}。'
    )
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=retry_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_ROUTE_SCHEMA,
        stream=False,
        caller="router",
        prompt_name=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.name,
        prompt_version=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.version,
    )
    route = parse_route(raw)
    route = _recover_declared_parameter_adjustment(
        client,
        route=route,
        prompt=prompt,
        instruction=instruction,
        param_schema=param_schema or [],
    )
    return _recover_declared_selection_decision(
        client,
        route=route,
        prompt=prompt,
        param_schema=param_schema or [],
    )


_MODELING_ADJUST_PARAM_NAMES = frozenset(
    {"split_config", "target_type", "recipes", "n_trials", "sample_weight_col"}
)
_MODELING_ADJUST_CUES = (
    "oot",
    "切分",
    "划分",
    "训练集",
    "测试集",
    "时间外推",
    "随机留出",
    "算法",
    "模型",
    "调参",
    "轮",
    "lgb",
    "lightgbm",
    "xgb",
    "xgboost",
    "catboost",
    "scorecard",
    "评分卡",
    "逻辑回归",
    "mlp",
    "样本权重",
)
_STRUCTURAL_REPLAN_CUES = (
    "调整步骤顺序",
    "重排步骤",
    "改流程",
    "换流程",
    "切换流程",
)


def _recover_declared_parameter_adjustment(
    client,
    *,
    route: dict,
    prompt: str,
    instruction,
    param_schema,
) -> dict:
    """Give a modeling control request one bounded completeness review.

    Modeling setup now exposes algorithms, tuning budget, target family, sample
    weight, and split configuration as typed controls on the current gate.  The
    historical router prompt treated any algorithm change as a structural
    replan, which sent an otherwise local patch through a much broader
    best-effort planner.  We only reconsider when the requested concepts are
    declared by this gate and the text does not explicitly add/remove/reorder
    steps.  The returned params still pass the driver's normal gate-scoped
    validation before any step is reset.
    """

    if route.get("action") not in {"adjust", "replan"}:
        return route
    declared = {
        str(item.get("name") or "").strip()
        for item in list(param_schema or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    fallback = dict(route)
    if fallback.get("action") == "adjust":
        fallback["params"] = normalize_adjust_params(fallback.get("params"))
        fallback = _canonicalize_sample_weight_selection(
            fallback,
            declared=declared,
        )
    if not (declared & _MODELING_ADJUST_PARAM_NAMES):
        return fallback
    text = str(instruction or "").strip()
    lowered = text.lower()
    changes_steps = "步骤" in text and any(
        verb in text
        for verb in ("新增", "增加", "添加", "删除", "移除", "去掉", "跳过", "重排")
    )
    if changes_steps or any(cue in text for cue in _STRUCTURAL_REPLAN_CUES):
        return fallback
    if not any(cue in lowered for cue in _MODELING_ADJUST_CUES):
        return fallback

    declared_text = "、".join(sorted(declared & _MODELING_ADJUST_PARAM_NAMES))
    first_pass = json.dumps(
        route,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    review_prompt = (
        f"{prompt}\n\n"
        "【当前节点参数调整复核】\n"
        f"第一遍结构化结果：{first_pass}\n"
        f"当前节点已经声明这些建模参数可安全重算：{declared_text}。"
        "算法组合、调参轮数、目标类型、样本权重、时间/随机/无 OOT 切分，"
        "只要能完整映射到上述参数，就属于 adjust，不是 replan；"
        "只有新增、删除、重排步骤或切换 Workflow 才属于 replan。"
        "请逐项对照用户原句与第一遍结果，把用户明确点名的每个控件都抽入 params，"
        "不得遗漏；明确说“不可用、不要、不选择、保持不可用”"
        "的算法不得放进 recipes。时间 OOT 用 split_config.oot_by_time，"
        "随机 OOT 用 split_config.random_oot=true，无 OOT 不设置这两项。"
        "sample_weight_candidates 是只读诊断，绝不能作为调整参数："
        "明确启用某一列时必须返回 sample_weight_col=\"列名\"，"
        "明确不使用权重时返回 sample_weight_col=\"\"；"
        "有多个候选但没有明确选择时必须返回 clarify。"
        "信息不足时返回 clarify，不要猜列名或比例。\n"
        '请严格只返回 JSON 对象:{"action":"adjust|replan|clarify","params":{},'
        '"constraint":"","reason":"一句话中文","confidence":"high|medium|low",'
        '"explicit_authorization":false}。'
    )
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=review_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_ROUTE_SCHEMA,
        stream=False,
        caller="router",
        prompt_name=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.name,
        prompt_version=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.version,
    )
    candidate, ok = _parse_route(raw)
    if ok and candidate.get("action") == "adjust":
        candidate["params"] = normalize_adjust_params(candidate.get("params"))
        candidate = _canonicalize_sample_weight_selection(
            candidate,
            declared=declared,
        )
    allowed = declared & _MODELING_ADJUST_PARAM_NAMES
    if (
        ok
        and candidate.get("action") == "adjust"
        and candidate.get("params")
        and set(candidate["params"]) <= allowed
    ):
        return candidate
    if ok and candidate.get("action") in {"clarify", "replan"}:
        return candidate
    return fallback


def _canonicalize_sample_weight_selection(
    route: dict,
    *,
    declared: set[str],
) -> dict:
    """Translate one LLM-decided weight selection into the writable control.

    ``sample_weight_candidates`` is evidence produced by ``choose_modeling_spec``;
    it is not a user-editable selection. Some model replies nevertheless put a
    clearly selected single column there. The semantic ``adjust`` decision still
    comes from the LLM; this bounded canonicalizer only maps that one unambiguous
    value to ``sample_weight_col``. It never chooses among multiple candidates.
    """

    params = dict(route.get("params") or {})
    if "sample_weight_candidates" not in params:
        return route

    raw_candidates = params.pop("sample_weight_candidates")
    if "sample_weight_col" in params:
        return {**route, "params": params}

    candidates = (
        [
            value.strip()
            for value in raw_candidates
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(raw_candidates, list)
        else []
    )
    candidates = list(dict.fromkeys(candidates))
    if "sample_weight_col" in declared and len(candidates) == 1:
        params["sample_weight_col"] = candidates[0]
        return {**route, "params": params}

    return {
        "action": "clarify",
        "params": {},
        "constraint": "",
        "reason": (
            "识别到多个样本权重候选，请明确选择一列或说明不使用权重。"
            if len(candidates) > 1
            else "未识别到明确的样本权重列，请指定一列或说明不使用权重。"
        ),
        "confidence": "low",
        "explicit_authorization": False,
    }


def _recover_declared_selection_decision(
    client,
    *,
    route: dict,
    prompt: str,
    param_schema,
) -> dict:
    """Give a candidate-selection gate one bounded semantic consistency pass.

    A live modeling turn exposed a subtle failure mode: the user named the
    platform-starred experiment and explicitly asked to adopt it, but the first
    router pass called that a structural replan.  The trigger here is the
    *declared gate contract* (``selected_experiment_id``), not words in the
    utterance.  The LLM still decides what the complete sentence means; the
    driver later validates the returned id against the persisted candidate set
    before any governed confirmation is recorded.
    """

    if route.get("action") not in {"confirm", "adjust", "replan", "clarify"}:
        return route
    selection_spec = next(
        (
            item
            for item in list(param_schema or [])
            if isinstance(item, dict)
            and str(item.get("name") or "").strip() == "selected_experiment_id"
        ),
        None,
    )
    if selection_spec is None:
        return route

    allowed_values = _selection_enum_values(selection_spec)
    if not allowed_values:
        return {
            "action": "clarify",
            "params": {},
            "constraint": "",
            "reason": "当前没有可供选择的候选实验。",
            "confidence": "low",
            "explicit_authorization": False,
        }
    if route.get("action") == "confirm":
        selected_id = str(
            (route.get("params") or {}).get("selected_experiment_id") or ""
        ).strip()
        if (
            selected_id in allowed_values
            and route.get("confidence") == "high"
            and route.get("explicit_authorization") is True
        ):
            canonical = dict(route)
            canonical["params"] = {"selected_experiment_id": selected_id}
            return canonical
        return {
            "action": "clarify",
            "params": {},
            "constraint": "",
            "reason": "候选实验或采用授权不明确，请从当前候选中明确选择。",
            "confidence": "low",
            "explicit_authorization": False,
        }

    first_pass = json.dumps(
        route,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    candidate_context = selection_spec.get("candidates")
    if not isinstance(candidate_context, list):
        candidate_context = [
            {
                "experiment_id": experiment_id,
                "recipe": "",
                "display_name": experiment_id,
                "recommended": experiment_id == selection_spec.get("current"),
            }
            for experiment_id in allowed_values
        ]
    candidate_json = json.dumps(
        candidate_context,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    review_prompt = (
        f"{prompt}\n\n"
        "【候选选择节点语义复核】\n"
        f"第一遍结构化结果：{first_pass}\n"
        f"当前候选 JSON：{candidate_json}\n"
        "当前节点已经完成候选实验比较，selected_experiment_id 的 enum 是平台"
        "实际展示且允许选择的候选，current 是平台当前推荐项；candidates 同时给出"
        "每个候选的 experiment_id、recipe、display_name 和 recommended。"
        "请重新理解用户整句话：用户可以用实验 id、recipe、算法显示名或推荐关系"
        "表达选择；仅在名称或 recipe 唯一匹配到一个候选，且用户明确授权采用或进入"
        "后续步骤时返回 confirm，"
        "params 只保留 selected_experiment_id，并设置 confidence=high、"
        "explicit_authorization=true；这只是当前决策节点的选择，不是修改 DAG。"
        "若同一名称或 recipe 对应多个候选、没有唯一匹配，或用户只是在询问、评价"
        "或讨论候选而未授权采用，返回 clarify。"
        "只有确实新增、删除、重排步骤或切换 Workflow 时才保留 replan。"
        "不得猜测 enum 之外的 id。\n"
        '请严格只返回 JSON 对象:{"action":"confirm|replan|clarify","params":{},'
        '"constraint":"","reason":"一句话中文","confidence":"high|medium|low",'
        '"explicit_authorization":false}。'
    )
    raw = client.complete(
        system_prompt=_SYSTEM,
        user_prompt=review_prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=_selection_route_schema(allowed_values),
        stream=False,
        caller="router",
        prompt_name=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.name,
        prompt_version=_GATE_INSTRUCTION_ROUTER_SYS_SPEC.version,
    )
    candidate, ok = _parse_route(raw)
    if not ok:
        return route
    if candidate.get("action") == "confirm":
        selected_id = str(
            (candidate.get("params") or {}).get("selected_experiment_id") or ""
        ).strip()
        allowed = _selection_enum(selection_spec)
        if (
            selected_id
            and selected_id in allowed
            and candidate.get("confidence") == "high"
            and candidate.get("explicit_authorization") is True
            and set(candidate.get("params") or {}) == {"selected_experiment_id"}
        ):
            return candidate
        return {
            "action": "clarify",
            "params": {},
            "constraint": "",
            "reason": "候选实验或采用授权不明确，请从当前候选中明确选择。",
            "confidence": "low",
            "explicit_authorization": False,
        }
    if candidate.get("action") in {"clarify", "replan"}:
        return candidate
    return route


def _selection_enum(spec: dict) -> set[str]:
    return set(_selection_enum_values(spec))


def _selection_enum_values(spec: dict) -> list[str]:
    raw = spec.get("enum")
    if not isinstance(raw, list):
        bounds = spec.get("bounds")
        raw = bounds.get("enum") if isinstance(bounds, dict) else None
    values: list[str] = []
    for item in list(raw or []):
        value = str(item).strip()
        if value and value not in values:
            values.append(value)
    return values


def _selection_route_schema(allowed_values: list[str]) -> dict:
    """Constrain the one-shot candidate review to persisted experiment ids.

    Recipe ids and display names remain semantic evidence for the LLM, but the
    only executable value it can emit is one of the current gate's experiment
    ids. The driver performs the same persisted-candidate validation again
    before recording authorization.
    """

    return {
        "name": "candidate_selection_route",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["confirm", "replan", "clarify"],
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "selected_experiment_id": {
                            "type": "string",
                            "enum": list(allowed_values),
                        }
                    },
                    "additionalProperties": False,
                },
                "constraint": {"type": "string"},
                "reason": {"type": "string"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "explicit_authorization": {"type": "boolean"},
            },
            "required": [
                "action",
                "params",
                "constraint",
                "reason",
                "confidence",
                "explicit_authorization",
            ],
            "additionalProperties": False,
        },
    }


_MAX_PARAM_SCHEMA_ITEMS = 12
_MAX_PARAM_VALUE_CHARS = 80
_MAX_PARAM_CANDIDATES = 12
_MAX_PARAM_CANDIDATE_CHARS = 2400


def _format(gate_context, instruction, tables, param_schema):
    lines = ["【当前节点】", str(gate_context or "")]
    for table in tables:
        lines.append(f"表:{table.get('title', '')} 列={table.get('columns')}")
    schema_lines = _format_param_schema(param_schema)
    if schema_lines:
        lines.append("【可调参数】(返回的 params 键只能取自这里)")
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
        candidates = item.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate_text = json.dumps(
                candidates[:_MAX_PARAM_CANDIDATES],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(candidate_text) > _MAX_PARAM_CANDIDATE_CHARS:
                candidate_text = (
                    candidate_text[:_MAX_PARAM_CANDIDATE_CHARS] + "…"
                )
            line += f" 候选上下文: {candidate_text}"
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
        return {
            "action": "clarify",
            "params": {},
            "constraint": "",
            "reason": "无法解析指令，请换种说法。",
            "confidence": "low",
            "explicit_authorization": False,
        }, False
    action = str(data.get("action") or "").strip().lower()
    if action not in _ACTIONS:
        action = "clarify"
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    constraint = str(data.get("constraint") or "").strip()
    reason = str(data.get("reason") or "").strip()
    confidence = str(data.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    explicit_authorization = data.get("explicit_authorization") is True
    # An "adjust" with no extractable parameters is not actionable → clarify.
    if action == "adjust" and not params:
        action = "clarify"
        reason = reason or "没识别到要调整的参数，请写明参数名和取值。"
    return {
        "action": action,
        "params": params,
        "constraint": constraint,
        "reason": reason,
        "confidence": confidence,
        "explicit_authorization": explicit_authorization,
    }, error is None


__all__ = ["route_instruction", "parse_route"]
