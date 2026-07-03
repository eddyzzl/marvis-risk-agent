from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jsonschema import Draft202012Validator

from marvis.plugins.errors import SchemaValidationError


def validate_against_schema(value: Any, schema: dict[str, Any], *, label: str) -> None:
    """Validate a tool input/output value against a Draft 2020-12 JSON Schema."""

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    error = next(validator.iter_errors(value), None)
    if error is None:
        return
    raise SchemaValidationError(label, f"{_json_path(error.absolute_path)}: {error.message}")


def _json_path(parts: Iterable[Any]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path
