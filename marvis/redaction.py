from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<![0-9A-Za-z])1[3-9]\d{9}(?![0-9A-Za-z])")
_IDCARD_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
_BANK_CARD_RE = re.compile(r"(?<![0-9A-Za-z])\d{13,19}(?![0-9A-Za-z])")
_SECRET_RE = re.compile(
    r"\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*[A-Za-z0-9_\-./+=]{6,}",
    re.IGNORECASE,
)
_SECRET_JSON_RE = re.compile(
    r"(?P<prefix>[\"']?(?:api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"']?)"
    r"(?P<secret>[A-Za-z0-9_\-./+=]{6,})"
    r"(?P<suffix>[\"']?)",
    re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9_\-./+=]{8,}\b", re.IGNORECASE)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b")
_DB_URL_RE = re.compile(r"\b(?:postgresql|mysql|oracle|sqlite|mongodb)://[^\s\"']+", re.IGNORECASE)
_SENSITIVE_KEY_RE = re.compile(
    "|".join([
        r"api[_-]?key",
        r"secret",
        r"token",
        r"password",
        r"phone",
        r"mobile",
        r"email",
        r"idcard",
        r"cert(?:ificate)?[_-]?no",
        r"card[_-]?no",
        r"bank[_-]?card",
        r"account[_-]?no",
        r"customer[_-]?name",
        r"手机号",
        r"身份证",
        r"银行卡",
        r"姓名",
    ]),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RedactionResult:
    value: Any
    redacted_count: int


def redact_text(value: Any) -> str:
    return str(redact_value(str(value)).value)


def redact_value(value: Any) -> RedactionResult:
    redacted, count = _redact_value(value)
    return RedactionResult(redacted, count)


def _redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, dict):
        output = {}
        count = 0
        for key, item in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY_RE.search(key_text):
                output[key] = "[REDACTED]"
                count += 1
                continue
            redacted, item_count = _redact_value(item)
            output[key] = redacted
            count += item_count
        return output, count
    if isinstance(value, tuple):
        items = [_redact_value(item) for item in value]
        return tuple(item for item, _count in items), sum(count for _item, count in items)
    if isinstance(value, list):
        items = [_redact_value(item) for item in value]
        return [item for item, _count in items], sum(count for _item, count in items)
    if isinstance(value, str):
        return _redact_string(value)
    return value, 0


def _redact_string(value: str) -> tuple[str, int]:
    text = str(value)
    count = 0
    for pattern, replacement in (
        (_SECRET_JSON_RE, _secret_json_repl),
        (_SECRET_RE, "[REDACTED_SECRET]"),
        (_BEARER_TOKEN_RE, "Bearer [REDACTED_SECRET]"),
        (_OPENAI_KEY_RE, "[REDACTED_SECRET]"),
        (_DB_URL_RE, "[REDACTED_DB_URL]"),
        (_EMAIL_RE, "[REDACTED_EMAIL]"),
        (_IDCARD_RE, _mask_match(keep_start=4, keep_end=2)),
        (_PHONE_RE, _mask_match(keep_start=3, keep_end=2)),
        (_BANK_CARD_RE, _mask_match(keep_start=4, keep_end=4)),
    ):
        text, changed = pattern.subn(replacement, text)
        count += changed
    return text, count


def _secret_json_repl(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}[REDACTED_SECRET]{match.group('suffix')}"


def _mask_match(*, keep_start: int, keep_end: int):
    def repl(match: re.Match[str]) -> str:
        text = match.group(0)
        if len(text) <= keep_start + keep_end:
            return "*" * len(text)
        return f"{text[:keep_start]}{'*' * (len(text) - keep_start - keep_end)}{text[-keep_end:]}"

    return repl


__all__ = ["RedactionResult", "redact_text", "redact_value"]
