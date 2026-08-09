"""Fail-closed JSON readers for persisted governance artifacts.

This parser is deliberately stricter than the tolerant LLM response parser:
it accepts one standards-compliant JSON object and rejects duplicate keys,
non-finite numbers, prose/fences, truncated payloads, and invalid UTF-8.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any


class GovernedJsonError(ValueError):
    """Raised when an authenticated artifact is not strict JSON."""


ErrorFactory = Callable[[str], Exception]


def strict_json_object_from_text(
    value: str,
    name: str,
    *,
    error_factory: ErrorFactory = GovernedJsonError,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in pairs:
            if key in result:
                raise error_factory(
                    f"{name} contains a duplicate JSON key: {key}"
                )
            result[key] = child
        return result

    def reject_constant(constant: str) -> None:
        raise error_factory(f"{name} contains non-finite JSON: {constant}")

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (GovernedJsonError,):
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise error_factory(f"{name} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise error_factory(f"{name} must be a JSON object")
    return parsed


def strict_json_object_from_bytes(
    value: bytes,
    name: str,
    *,
    error_factory: ErrorFactory = GovernedJsonError,
) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise error_factory(f"{name} must be strict UTF-8 JSON") from exc
    return strict_json_object_from_text(
        text,
        name,
        error_factory=error_factory,
    )


__all__ = [
    "GovernedJsonError",
    "strict_json_object_from_bytes",
    "strict_json_object_from_text",
]
