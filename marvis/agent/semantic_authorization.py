"""Independent, fail-closed semantic authorization review for confirmation gates."""

from __future__ import annotations

from dataclasses import dataclass
import json

from marvis.llm_client import LLMClientError
from marvis.llm_prompts import (
    GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS as _REVIEW_PROMPT,
)

_VERDICTS = ("authorize", "reject", "ambiguous")
_CONFIDENCE_LEVELS = ("high", "medium", "low")
_RESPONSE_FIELDS = (
    "verdict",
    "evidence_quote",
    "reason",
    "confidence",
    "is_question",
    "is_conditional",
    "requests_change",
    "withholds_authorization",
)
# DeepSeek V4 may charge hidden reasoning against max_tokens even when visible
# thinking is disabled.  Keep enough bounded headroom for the final strict JSON;
# an empty/length-finished response still fails closed below.
_MAX_COMPLETION_TOKENS = 1024

_REVIEW_SCHEMA = {
    "name": "gate_semantic_authorization_review",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": list(_VERDICTS),
            },
            "evidence_quote": {"type": "string"},
            "reason": {"type": "string"},
            "confidence": {
                "type": "string",
                "enum": list(_CONFIDENCE_LEVELS),
            },
            "is_question": {"type": "boolean"},
            "is_conditional": {"type": "boolean"},
            "requests_change": {"type": "boolean"},
            "withholds_authorization": {"type": "boolean"},
        },
        "required": list(_RESPONSE_FIELDS),
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class SemanticAuthorizationReview:
    """The model's strict review plus the platform-owned authorization verdict."""

    authorized: bool
    verdict: str
    evidence_quote: str
    reason: str
    confidence: str
    is_question: bool
    is_conditional: bool
    requests_change: bool
    withholds_authorization: bool


def review_semantic_authorization(
    client,
    *,
    gate_context,
    instruction,
    proposed_params,
) -> SemanticAuthorizationReview:
    """Run exactly one independent review and fail closed on every invalid result."""

    try:
        if not isinstance(instruction, str):
            return _failed_review()
        user_prompt = json.dumps(
            {
                "gate_context": gate_context,
                "instruction": instruction,
                "proposed_params": proposed_params,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        raw = client.complete(
            system_prompt=_REVIEW_PROMPT.text,
            user_prompt=user_prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
            json_schema=_REVIEW_SCHEMA,
            max_tokens=_MAX_COMPLETION_TOKENS,
            stream=False,
            caller="semantic_authorization_reviewer",
            prompt_name=_REVIEW_PROMPT.name,
            prompt_version=_REVIEW_PROMPT.version,
        )
    except LLMClientError:
        return _failed_review()
    except Exception:
        return _failed_review()

    try:
        return _parse_review(raw, instruction=instruction)
    except Exception:
        return _failed_review()


def _parse_review(raw, *, instruction: str) -> SemanticAuthorizationReview:
    if not isinstance(raw, str):
        return _failed_review()

    def unique_object(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"duplicate key: {key}")
            data[key] = value
        return data

    try:
        data = json.loads(raw.strip(), object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return _failed_review()
    if not isinstance(data, dict) or set(data) != set(_RESPONSE_FIELDS):
        return _failed_review()

    verdict = data["verdict"]
    evidence_quote = data["evidence_quote"]
    reason = data["reason"]
    confidence = data["confidence"]
    flags = (
        data["is_question"],
        data["is_conditional"],
        data["requests_change"],
        data["withholds_authorization"],
    )
    if (
        not isinstance(verdict, str)
        or verdict not in _VERDICTS
        or not isinstance(evidence_quote, str)
        or not isinstance(reason, str)
        or not isinstance(confidence, str)
        or confidence not in _CONFIDENCE_LEVELS
        or any(type(flag) is not bool for flag in flags)
    ):
        return _failed_review()

    authorized = (
        verdict == "authorize"
        and confidence == "high"
        and bool(evidence_quote.strip())
        and evidence_quote in instruction
        and not any(flags)
    )
    return SemanticAuthorizationReview(
        authorized=authorized,
        verdict=verdict,
        evidence_quote=evidence_quote,
        reason=reason,
        confidence=confidence,
        is_question=flags[0],
        is_conditional=flags[1],
        requests_change=flags[2],
        withholds_authorization=flags[3],
    )


def _failed_review() -> SemanticAuthorizationReview:
    return SemanticAuthorizationReview(
        authorized=False,
        verdict="ambiguous",
        evidence_quote="",
        reason="语义授权复核失败，已按未授权处理。",
        confidence="low",
        is_question=False,
        is_conditional=False,
        requests_change=False,
        withholds_authorization=True,
    )


__all__ = ["SemanticAuthorizationReview", "review_semantic_authorization"]
