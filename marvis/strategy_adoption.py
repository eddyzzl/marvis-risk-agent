"""Shared validation for evidence-bound strategy adoption reasons.

The plan may carry an empty string while it is waiting at the adoption gate, but
that pending value is never a valid business decision.  Both the gate path and
the persistence/tool boundaries use this module so a direct invocation cannot
turn a blank or placeholder into an adopted strategy.
"""

from __future__ import annotations

import re


class AdoptionReasonError(ValueError):
    """Raised when a strategy adoption reason is not a real business reason."""


ADOPTION_REASON_MIN_LENGTH = 2
_PLACEHOLDER_KEY_RE = re.compile(r"[\s（）()\[\]【】<>《》:：,，。.!！?？_\-/\\]+")
_PLACEHOLDER_KEYS = frozenset({
    "todo",
    "todo待确认",
    "tbd",
    "pending",
    "pendingapproval",
    "pendingconfirmation",
    "placeholder",
    "na",
    "none",
    "null",
    "占位",
    "占位符",
    "待确认",
    "待采纳时确认",
    "待填写",
    "待补充",
})
_PLACEHOLDER_PREFIXES = (
    "todo",
    "tbd",
    "pending",
    "placeholder",
    "占位",
    "待确认",
    "待采纳",
    "待填写",
    "待补充",
)


def normalize_adoption_reason(value: object) -> str:
    """Return a trimmed adoption reason or fail closed on pending placeholders."""

    if not isinstance(value, str):
        raise AdoptionReasonError("采纳理由必须是非空文本，不能使用待确认占位。")
    reason = value.strip()
    if not reason:
        raise AdoptionReasonError("采纳理由不能为空，不能使用待确认占位。")
    if len(reason) < ADOPTION_REASON_MIN_LENGTH:
        raise AdoptionReasonError(
            f"采纳理由至少需要 {ADOPTION_REASON_MIN_LENGTH} 个字符。"
        )
    placeholder_key = _PLACEHOLDER_KEY_RE.sub("", reason).casefold()
    if (
        not placeholder_key
        or placeholder_key in _PLACEHOLDER_KEYS
        or any(placeholder_key.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)
    ):
        raise AdoptionReasonError(
            "采纳理由不能使用待确认、TODO、pending 或其他占位文本。"
        )
    return reason


def is_valid_adoption_reason(value: object) -> bool:
    """Cheap predicate for gate rendering/guards without duplicating policy."""

    try:
        normalize_adoption_reason(value)
    except AdoptionReasonError:
        return False
    return True


__all__ = [
    "ADOPTION_REASON_MIN_LENGTH",
    "AdoptionReasonError",
    "is_valid_adoption_reason",
    "normalize_adoption_reason",
]
