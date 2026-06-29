"""Utilities for normalizing LLM JSON replies."""

from __future__ import annotations

import json
from typing import Any


def load_json_object(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON object from a strict or lightly wrapped LLM reply."""
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, "reply is not text"
    text = raw.strip()
    candidates = [text]
    extracted = _extract_first_object(text)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError) as exc:
            last_error = str(exc)
            continue
        if isinstance(data, dict):
            return data, None
        last_error = "JSON value is not an object"
    return None, last_error


def _extract_first_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        start = text.find("{", start + 1)
    return None


__all__ = ["load_json_object"]
