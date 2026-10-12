"""Installed response contracts for analysis and pure-computation Draft tools."""
from marvis.llm_outbound import OutboundContextError


def _fields(bindings, key):
    from marvis.llm_outbound_analysis import field_names

    return field_names(bindings.get(key))


def installed_noncore_schema(contract, bindings):
    if contract == "adhoc_slice_spec":
        from marvis.agent.adhoc_analysis import _ALLOWED_OPS, _ALLOWED_FILTER_OPS, _MAX_FILTERS, _MAX_GROUP_BY

        if set(bindings) != {"columns"}:
            raise OutboundContextError("问数输出契约绑定无效，未发送请求。")
        columns = _fields(bindings, "columns")
        schema = {"type": "object", "properties": {
            "clarify": {"type": "string"},
            "group_by": {"type": "array", "items": {"type": "string", "enum": columns}, "maxItems": _MAX_GROUP_BY},
            "metrics": {"type": "array", "items": {"type": "object", "properties": {
                "op": {"type": "string", "enum": sorted(_ALLOWED_OPS)},
                "col": {"type": "string", "enum": columns},
            }, "required": ["op"], "additionalProperties": False}, "maxItems": 64},
            "filters": {"type": "array", "items": {"type": "object", "properties": {
                "col": {"type": "string", "enum": columns},
                "op": {"type": "string", "enum": sorted(_ALLOWED_FILTER_OPS)},
                "value": {"type": ["string", "number", "integer", "boolean", "null", "array"]},
            }, "required": ["col", "op", "value"], "additionalProperties": False}, "maxItems": _MAX_FILTERS},
            "month_col": {"type": "string", "enum": columns},
            "months": {"type": "array", "items": {"type": "string", "pattern": r"^\d{4}-?\d{2}$"}, "maxItems": 240},
            "sort_by": {"type": "string"},
        }, "additionalProperties": False, "anyOf": [{"required": ["clarify"]}, {"required": ["metrics"]}]}
    elif contract == "feature_cross_recommendations":
        from marvis.feature.derive import ALLOWED_CROSS_OPS, CONFIDENCE_LEVELS

        if set(bindings) != {"features", "max_candidates"}:
            raise OutboundContextError("特征交叉契约绑定无效，未发送请求。")
        fields = _fields(bindings, "features")
        limit = bindings["max_candidates"]
        if type(limit) is not int or not 1 <= limit <= 100:
            raise OutboundContextError("特征交叉候选数量超限，未发送请求。")
        schema = {"type": "object", "properties": {"recommendations": {"type": "array", "maxItems": limit,
            "items": {"type": "object", "properties": {
                "col_a": {"type": "string", "enum": fields}, "col_b": {"type": "string", "enum": fields},
                "ops": {"type": "array", "items": {"type": "string", "enum": sorted(ALLOWED_CROSS_OPS)}, "minItems": 1, "maxItems": 5},
                "rationale": {"type": "string", "maxLength": 500},
                "confidence": {"type": "string", "enum": sorted(CONFIDENCE_LEVELS)},
            }, "required": ["col_a", "col_b", "ops", "rationale", "confidence"], "additionalProperties": False}}},
            "required": ["recommendations"], "additionalProperties": False}
    elif contract == "draft_authoring":
        if bindings:
            raise OutboundContextError("Draft 输出契约不允许调用者覆盖，未发送请求。")
        schema = {"type": "object", "properties": {
            "name": {"type": "string", "pattern": r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"},
            "summary": {"type": "string", "maxLength": 2000}, "code": {"type": "string", "maxLength": 30000},
            "input_schema": {"type": "object"}, "output_schema": {"type": "object"},
            "determinism": {"type": "string", "enum": ["deterministic", "stochastic"]},
        }, "required": ["name", "summary", "code", "input_schema", "output_schema", "determinism"], "additionalProperties": False}
    elif contract in {"memory_distillation", "learning_note"}:
        if bindings:
            raise OutboundContextError("摘要输出契约不允许调用者覆盖，未发送请求。")
        key, limit = ("summary", 400) if contract == "memory_distillation" else ("note", 4000)
        schema = {"type": "object", "properties": {key: {"type": "string", "minLength": 1, "maxLength": limit}},
            "required": [key], "additionalProperties": False}
    else:
        raise OutboundContextError("未安装此业务输出契约，未发送请求。")
    return {"name": contract, "strict": False, "schema": schema}
