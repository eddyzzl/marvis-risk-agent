from __future__ import annotations

import json
from typing import Any


def summarize_output(output: dict, tool_spec, *, max_chars: int = 600) -> dict:
    properties = _schema_properties(getattr(tool_spec, "output_schema", {}))
    summary: dict[str, Any] = {}
    for field, spec in properties.items():
        if field not in output:
            continue
        value = output[field]
        if _is_scalar_schema(spec) or _is_scalar(value):
            summary[field] = _truncate_value(value, 120)
        elif spec.get("type") == "array" or isinstance(value, list):
            items = value if isinstance(value, list) else []
            summary[field] = {
                "len": len(items),
                "head": [_truncate_value(item, 120) for item in items[:2]],
            }
        elif isinstance(value, dict):
            summary[field] = {"type": "object", "keys": sorted(value)[:10]}
    if not summary:
        summary = _shape_summary(output)
    return _fit_summary(summary, max_chars)


def summarize_failure(error: str, error_kind: str, *, max_chars: int = 300) -> dict:
    return {
        "error_kind": str(error_kind or "execution"),
        "error": str(error or "")[:max_chars],
    }


def _schema_properties(schema: dict) -> dict:
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    return properties if isinstance(properties, dict) else {}


def _is_scalar_schema(schema: dict) -> bool:
    return schema.get("type") in {"string", "number", "integer", "boolean", "null"}


def _is_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _truncate_value(value, max_chars: int):
    if isinstance(value, str) and len(value) > max_chars:
        return value[: max_chars - 14] + "...[truncated]"
    if isinstance(value, dict):
        return _fit_summary(value, max_chars)
    return value


def _shape_summary(output: dict) -> dict:
    summary = {}
    for key, value in output.items():
        if _is_scalar(value):
            summary[key] = _truncate_value(value, 120)
        elif isinstance(value, list):
            summary[key] = {"len": len(value), "head": value[:2]}
        elif isinstance(value, dict):
            summary[key] = {"type": "object", "keys": sorted(value)[:10]}
    return summary


def _fit_summary(summary: dict, max_chars: int) -> dict:
    fitted = {}
    for key, value in summary.items():
        candidate = {**fitted, key: value}
        if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True)) <= max_chars:
            fitted[key] = value
    if fitted:
        return fitted
    return {"truncated": True}
