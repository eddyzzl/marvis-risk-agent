"""Exact planning controls, separate from private task and memory prose."""

from datetime import date, datetime
import json
import re

from marvis.llm_outbound import (
    OutboundContextError, SafeInstruction, _DROP, _HASH, _ID_KEYS, _IDENTIFIER_KEYS,
    _IDENTIFIER_LISTS, _REFERENCE, _SAFE_ENUMS, _check_schema_limits, _identifier,
    _schema_number, safe_schema, structural_summary,
)
from marvis.llm_outbound_catalog import catalog_schema_contracts
from marvis.llm_outbound_instructions import column_identifiers


_CONTROL_TOKEN = object()
_TASK_IDS = _ID_KEYS | {"active_dataset_id", "current_dataset_id", "active_plan_id", "active_model_id"}
_REVISIONS = {"active_dataset_revision", "dataset_revision", "current_dataset_revision", "workspace_revision", "analysis_generation"}
_LABELS = {
    "entrypoint": {"validation", "model_validation", "modeling", "model_development", "vintage", "risk_analysis", "data_join", "data_preparation", "feature_analysis", "strategy", "monitoring", "portfolio"},
    "workflow_family": {"fixed", "adaptive", "novel", "guardrail"},
    "analysis_kind": {"profitability", "vtg_terminal", "standard_vintage"},
    "label_semantics": {"incremental", "snapshot"},
    "role": {"anchor", "feature", "target", "score", "weight", "time", "group"},
}


class _ApprovedInputs(dict):
    def __init__(self, values, *, _token=None):
        if _token is not _CONTROL_TOKEN:
            raise OutboundContextError("规划输入缺少结构来源，未发送请求。")
        super().__init__(values)


def _calendar(value):
    if not isinstance(value, str):
        return False
    try:
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return date.fromisoformat(value + "-01").strftime("%Y-%m") == value
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat() == value
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?", value):
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
    except ValueError:
        pass
    return False


def planner_task_context(value):
    if not isinstance(value, dict):
        raise OutboundContextError("任务控制上下文无效，未发送规划请求。")
    clean = structural_summary(value)
    if "historical_context" in value:
        clean["historical_context"] = {"__omitted_context__": "outbound_policy", "original_type": "str"}
    for key in _TASK_IDS & value.keys():
        if _identifier(value[key], opaque=True) is _DROP:
            raise OutboundContextError("任务对象绑定需要不含路径的结构编号，未发送规划请求。")
        clean[key] = value[key]
    for key in _REVISIONS & value.keys():
        if type(value[key]) is not int or not 0 <= value[key] <= 10**15:
            raise OutboundContextError("任务版本绑定无效，未发送规划请求。")
        clean[key] = value[key]
    for key in _LABELS.keys() & value.keys():
        if not isinstance(value[key], str) or value[key] not in _LABELS[key]:
            raise OutboundContextError("任务控制标签不属于已安装契约，未发送规划请求。")
        clean[key] = value[key]
    for key in {"target_col", "score_col", "time_col", "split_col", "date_col", "group_col", "sample_weight_col"} & value.keys():
        clean[key] = None if value[key] is None else column_identifiers([value[key]])[0]
    for key in {"features", "feature_columns", "feature_names", "target_candidates"} & value.keys():
        clean[key] = column_identifiers(value[key])
    if "user_constraint" in value:
        from marvis.llm_outbound_instructions import user_instruction
        clean["user_constraint"] = str(user_instruction(value["user_constraint"]))
    if "column_map" in value:
        column_map = value["column_map"]
        if not isinstance(column_map, dict) or len(column_map) > 10000 or any(_identifier(key) is _DROP for key in column_map):
            raise OutboundContextError("字段角色绑定无效，未发送规划请求。")
        clean["column_map"] = dict(zip(column_map, column_identifiers(list(column_map.values())), strict=True))
    if "split" in value:
        split = value["split"]
        if not isinstance(split, dict) or set(split) - {"train", "test", "oot"}:
            raise OutboundContextError("样本切分控制无效，未发送规划请求。")
        for item in split.values():
            values = item if isinstance(item, list) else [item]
            if len(values) > 10000 or any(not (_schema_number(part) or _calendar(part)) for part in values):
                raise OutboundContextError("样本切分只接受数值或时间控制，未发送规划请求。")
        clean["split"] = split
    if "split_config" in value:
        from marvis.llm_outbound_gates import split_configuration
        clean["split_config"] = split_configuration(value["split_config"])
    return clean


def _schema_node(schema, root):
    if isinstance(schema, bool):
        return schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise OutboundContextError("规划输入只支持本地指针契约，未发送请求。")
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return _schema_node(target, root)
    return schema


def _input_value(value, schema, root, *, field, approved_goal, depth=0):
    if depth > 64:
        raise OutboundContextError("规划输入结构过深，未截断或发送请求。")
    schema = _schema_node(schema, root)
    if schema is False:
        raise OutboundContextError("规划输入不符合工具契约，未发送请求。")
    schema = schema if isinstance(schema, dict) else {}
    if isinstance(value, str) and _REFERENCE.fullmatch(value):
        return value
    # Literal subsets need not satisfy the root's required keys. Each supplied
    # field and complete nested value still obeys the installed constraint.
    if depth and not _contains_reference(value):
        from jsonschema import Draft202012Validator
        if not Draft202012Validator(root).evolve(schema=schema).is_valid(value):
            raise OutboundContextError("规划输入值不符合已安装工具约束，未发送请求。")
    if isinstance(value, str):
        constants = schema.get("enum", []) + ([schema["const"]] if "const" in schema else [])
        if value in constants or value in _SAFE_ENUMS:
            return value
        if isinstance(value, SafeInstruction) or (value and value in approved_goal):
            return str(value)
        if (field in _ID_KEYS or field.endswith("_id")) and _identifier(value, opaque=True) is not _DROP:
            return value
        if field.endswith(("_hash", "_sha256")) and _HASH.fullmatch(value):
            return value
        if field in (_IDENTIFIER_KEYS | _IDENTIFIER_LISTS | {"column_map", "metrics", "feature_columns", "feature_names"}):
            return column_identifiers([value])[0]
        if _calendar(value):
            return value
        raise OutboundContextError("工具输入含未获准的自由文本，未丢弃执行约束或发送请求。")
    if value is None or type(value) is bool or _schema_number(value):
        return value
    if isinstance(value, list):
        items = schema.get("items", {})
        return [_input_value(item, items, root, field=field, approved_goal=approved_goal, depth=depth + 1)
                for item in value]
    if isinstance(value, dict):
        properties = dict(schema.get("properties", {}))
        for keyword in ("allOf", "anyOf", "oneOf"):
            for branch in schema.get(keyword, []):
                node = _schema_node(branch, root)
                if isinstance(node, dict):
                    properties.update(node.get("properties", {}))
        additional = schema.get("additionalProperties")
        clean = {}
        for key, child in value.items():
            if key not in properties and not isinstance(additional, dict):
                raise OutboundContextError("工具输入字段不属于已安装契约，未丢弃或发送请求。")
            if key not in properties and _identifier(key) is _DROP:
                raise OutboundContextError("工具输入映射字段无效，未发送请求。")
            clean[key] = _input_value(child, properties.get(key, additional), root,
                                      field=field if field == "column_map" else key,
                                      approved_goal=approved_goal, depth=depth + 1)
        return clean
    raise OutboundContextError("工具输入包含非结构值，未发送请求。")


def _contains_reference(value):
    if isinstance(value, str):
        return bool(_REFERENCE.fullmatch(value))
    if isinstance(value, dict):
        return any(_contains_reference(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_reference(child) for child in value)
    return False


def attach_planning_controls(clean, payload):
    if "task_context" in payload:
        clean["task_context"] = planner_task_context(payload["task_context"])
    contracts = {}
    for item in payload.get("available_tools", []):
        capability = catalog_schema_contracts(item).get("input_schema")
        if capability is not None:
            contracts[(item.get("plugin"), item.get("tool"))] = safe_schema(capability)
    paths = []
    if isinstance(payload.get("planner_constraints"), dict):
        paths.extend(("planner_constraints", "required_literal_inputs", index)
                     for index, _ in enumerate(payload["planner_constraints"].get("required_literal_inputs", [])))
    paths.extend(("remaining_steps", index) for index, _ in enumerate(payload.get("remaining_steps", [])))
    controls = []
    available_refs = {
        (item.get("plugin"), item.get("tool"))
        for item in clean.get("available_tools", []) if isinstance(item, dict)
    }
    for path in paths:
        source = _at(payload, path)
        target = _at(clean, path)
        ref = source.get("tool_ref", source.get("tool", {}))
        if (ref.get("plugin"), ref.get("tool")) not in available_refs:
            raise OutboundContextError("规划执行约束绑定的工具未完整保留，未发送请求。")
        inputs = source.get("inputs")
        if not isinstance(inputs, dict):
            raise OutboundContextError("规划输入契约无效，未发送请求。")
        if isinstance(inputs, _ApprovedInputs):
            approved = dict(inputs)
        else:
            schema = contracts.get((ref.get("plugin"), ref.get("tool")))
            if schema is None:
                projected = structural_summary(inputs)
                if projected != inputs:
                    raise OutboundContextError("规划输入缺少已安装结构契约，未丢弃或发送请求。")
                approved = projected
            else:
                _check_schema_limits(inputs)
                approved = _input_value(inputs, schema, schema, field="inputs", approved_goal=clean.get("goal", ""))
        target["inputs"] = approved
        controls.append(((*path, "inputs"), json.dumps(approved, ensure_ascii=False, sort_keys=True, allow_nan=False)))
    return tuple(controls)


def _at(payload, path):
    current = payload
    for key in path:
        current = current[key]
    return current


def restore_planning_controls(payload, prompt):
    for path, encoded in getattr(prompt, "_planning_controls", ()):
        value = _at(payload, path)
        if json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) != encoded:
            raise OutboundContextError("规划执行约束在预算调整时发生变化，未发送请求。")
        _at(payload, path[:-1])[path[-1]] = _ApprovedInputs(value, _token=_CONTROL_TOKEN)
    return payload
