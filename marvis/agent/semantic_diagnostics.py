"""Content-free semantic diagnostics; these facts never authorize an action.

Only finite enums, strict booleans and bounded integer positions may leave this
module. Raw replies, quotes, reasons, parameters and exception messages are not
part of the persistence contract. JSON positions refer to the stripped document
passed to the existing parser, not the original reply.
"""

from __future__ import annotations

import json

SCHEMA = "semantic-diagnostics.v1"
_STAGES = frozenset(
    {
        "intent",
        "intent_input",
        "intent_context",
        "semantic_intent_router",
        "semantic_intent_router_repair",
        "semantic_intent_reviewer",
        "semantic_intent_reviewer_repair",
        "authorization",
        "authorization_snapshot",
    }
)
_FAILURES = frozenset(
    {
        "input_invalid",
        "context_invalid",
        "request_failed",
        "non_text",
        "empty_response",
        "duplicate_key",
        "non_json",
        "non_object",
        "missing_fields",
        "extra_fields",
        "invalid_intent_type",
        "invalid_intent",
        "invalid_evidence_quote_type",
        "empty_evidence_quote",
        "evidence_quote_not_in_instruction",
        "invalid_reason_type",
        "invalid_confidence_type",
        "invalid_confidence",
        "invalid_flag_type",
        "invalid_contract",
        "invalid_field_type",
        "repair_context_invalid",
        "repair_request_failed",
        "repair_semantic_drift",
        "unexpected_parser_failure",
        "unauthorized",
        "snapshot_changed",
        "semantic_intent_pass_disagreement",
        "semantic_intent_unsafe_decision",
        "failed",
    }
)
_FAILURES = _FAILURES | {"repair_" + code for code in _FAILURES}
_FAILURES = _FAILURES | {
    prefix + code
    for prefix in ("semantic_intent_router_", "semantic_intent_reviewer_")
    for code in _FAILURES
}
_FLAGS = frozenset(
    {
        "is_question",
        "is_conditional",
        "requests_change",
        "withholds_action",
        "withholds_authorization",
        "evidence_nonempty",
        "evidence_in_instruction",
    }
)
_TOKENS = frozenset(
    {
        "non_text",
        "empty",
        "object",
        "array",
        "quote",
        "code_fence",
        "number",
        "literal",
        "text",
        "other",
    }
)
_ERRORS = frozenset(
    {
        "expecting_value",
        "expecting_property",
        "expecting_colon",
        "expecting_delimiter",
        "extra_data",
        "unterminated_string",
        "invalid_control",
        "invalid_escape",
        "unexpected_bom",
        "other_json_error",
    }
)


def response_shape(raw):
    if not isinstance(raw, str):
        return {"first_token": "non_text", "code_fence": False}
    value = raw.strip()
    fence = value.startswith("```")
    first = value[:1]
    token = (
        "empty"
        if not value
        else "code_fence"
        if fence
        else {"{": "object", "[": "array", '"': "quote"}.get(first)
        or (
            "number"
            if first in "-0123456789"
            else "literal"
            if value.startswith(("true", "false", "null"))
            else "text"
            if first.isalpha()
            else "other"
        )
    )
    return {"first_token": token, "code_fence": fence}


def json_error_shape(error):
    if not isinstance(error, json.JSONDecodeError):
        return {}
    # Compare parser-owned messages, but never persist their text or document.
    prefixes = {
        "Expecting value": "expecting_value",
        "Expecting property": "expecting_property",
        "Expecting ':'": "expecting_colon",
        "Expecting ','": "expecting_delimiter",
        "Extra data": "extra_data",
        "Unterminated string": "unterminated_string",
        "Invalid control": "invalid_control",
        "Invalid": "invalid_escape",
        "Unexpected UTF-8 BOM": "unexpected_bom",
    }
    category = next(
        (value for key, value in prefixes.items() if error.msg.startswith(key)),
        "other_json_error",
    )
    return {
        "json_error_category": category,
        "json_error_position": error.pos,
        "json_error_line": error.lineno,
        "json_error_column": error.colno,
    }


def decision_fields(data, instruction):
    if not isinstance(data, dict):
        return {}
    result = {key: data[key] for key in _FLAGS if type(data.get(key)) is bool}
    # These strings are passed through the final enum filter below.
    for key in ("intent", "verdict", "confidence"):
        if isinstance(data.get(key), str):
            result[key] = data[key]
    quote = data.get("evidence_quote")
    if isinstance(quote, str) and isinstance(instruction, str):
        result.update(
            evidence_nonempty=bool(quote.strip()),
            evidence_in_instruction=quote in instruction,
        )
    return _decision(result)


def _decision(value):
    from marvis.agent.semantic_intent import ALL_SEMANTIC_INTENTS

    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in _FLAGS if type(value.get(key)) is bool}
    for key, choices in (
        ("intent", ALL_SEMANTIC_INTENTS),
        ("verdict", {"authorize", "reject", "ambiguous"}),
        ("confidence", {"high", "medium", "low"}),
    ):
        if isinstance(value.get(key), str) and value[key] in choices:
            result[key] = value[key]
    return result


def _record(value, *, nested=True):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, choices in (
        ("stage", _STAGES),
        ("failure_code", _FAILURES),
        ("repair_status", {"not_attempted", "succeeded", "failed"}),
        ("first_token", _TOKENS),
        ("json_error_category", _ERRORS),
    ):
        if isinstance(value.get(key), str) and value[key] in choices:
            result[key] = value[key]
    if value.get("failure_code", False) is None:
        result["failure_code"] = None
    for key in ("accepted", "parse_valid", "code_fence"):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    for key in ("json_error_position", "json_error_line", "json_error_column"):
        if type(value.get(key)) is int and 0 <= value[key] <= 1_000_000_000:
            result[key] = value[key]
    decision = _decision(value.get("decision"))
    if decision:
        result["decision"] = decision
    if nested and isinstance(value.get("attempts"), (list, tuple)):
        result["attempts"] = [
            _record(item, nested=False) for item in value["attempts"][:2]
        ]
    return result


def sanitize_diagnostics(value):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        return None
    result = {"schema_version": SCHEMA, **_record(value)}
    if isinstance(value.get("passes"), (list, tuple)):
        result["passes"] = [_record(item) for item in value["passes"][:2]]
    return result


def diagnostic(stage, *, failure_code=None, accepted=False, passes=(), **fields):
    return sanitize_diagnostics(
        {
            "schema_version": SCHEMA,
            "stage": stage,
            "failure_code": failure_code,
            "accepted": accepted,
            "passes": list(passes),
            **fields,
        }
    )
