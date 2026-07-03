from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
# Bare 11-digit mobile numbers, and the same numbers written with a leading
# +86/86 country code and/or space/hyphen separators between the 3-4-4 groups
# (e.g. "+86 138-1234-5678", "86 13812345678", "138 1234 5678").
_PHONE_RE = re.compile(r"(?<![0-9A-Za-z])1[3-9]\d{9}(?![0-9A-Za-z])")
_PHONE_SEPARATED_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:\+?86[\s-]?)?1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}(?![0-9A-Za-z])"
)
# 18-digit (GB 11643-1999) and legacy 15-digit (pre-1999) resident ID numbers.
# The 18-digit form ends with a numeric or 'X' check character.
_IDCARD_RE = re.compile(r"(?<![0-9A-Za-z])\d{17}[0-9Xx](?![0-9A-Za-z])")
_IDCARD_LEGACY_RE = re.compile(r"(?<![0-9A-Za-z])\d{15}(?![0-9A-Za-z])")
# Bank card numbers are 13-19 bare digits, but so are millisecond timestamps,
# row counts and other ordinary large integers -- blanket-masking every such
# run over-redacts audit evidence. Require a Luhn checksum pass (which real
# card numbers satisfy and arbitrary integers only satisfy ~1-in-10 of the
# time) before treating a bare digit run as a bank card.
_BANK_CARD_CANDIDATE_RE = re.compile(r"(?<![0-9A-Za-z])\d{13,19}(?![0-9A-Za-z])")
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
        (_IDCARD_LEGACY_RE, _mask_match(keep_start=4, keep_end=2)),
        # Phone patterns run before the bank-card candidate scan: a
        # country-code-prefixed phone number with no separators (e.g.
        # "8613812345678") is 13 bare digits and would otherwise also match
        # the bank-card candidate pattern below.
        (_PHONE_SEPARATED_RE, _mask_match(keep_start=3, keep_end=2)),
        (_PHONE_RE, _mask_match(keep_start=3, keep_end=2)),
        (_BANK_CARD_CANDIDATE_RE, _mask_bank_card),
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


def _luhn_checksum_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# Common card-network BIN prefixes paired with that network's real card
# lengths (Visa/Mastercard/UnionPay/Amex/Discover). A digit run is treated as
# a card candidate even when its checksum happens to be wrong (mistyped
# numbers, or synthetic sample/test data never built to satisfy Luhn) only
# when BOTH its prefix and its length match a real network -- constraining on
# length too (not just the leading digit) keeps ordinary large integers such
# as timestamps, row counts and order numbers from colliding with the
# single-leading-digit Visa/UnionPay prefixes.
_BANK_CARD_BIN_RULES: tuple[tuple[str, frozenset[int]], ...] = (
    ("4", frozenset({13, 16, 19})),  # Visa
    ("51", frozenset({16})), ("52", frozenset({16})),
    ("53", frozenset({16})), ("54", frozenset({16})), ("55", frozenset({16})),  # Mastercard
    ("2221", frozenset({16})), ("2222", frozenset({16})), ("2223", frozenset({16})),
    ("2224", frozenset({16})), ("2225", frozenset({16})), ("2226", frozenset({16})),
    ("2227", frozenset({16})), ("2228", frozenset({16})), ("2229", frozenset({16})),
    ("223", frozenset({16})), ("224", frozenset({16})), ("225", frozenset({16})),
    ("226", frozenset({16})), ("227", frozenset({16})), ("228", frozenset({16})),
    ("229", frozenset({16})), ("23", frozenset({16})), ("24", frozenset({16})),
    ("25", frozenset({16})), ("26", frozenset({16})),
    ("270", frozenset({16})), ("271", frozenset({16})), ("2720", frozenset({16})),  # Mastercard 2-series
    ("34", frozenset({15})), ("37", frozenset({15})),  # American Express
    ("6011", frozenset({16})), ("65", frozenset({16})),  # Discover
    ("62", frozenset({16, 17, 18, 19})),  # UnionPay
)


def _has_bank_card_bin_prefix(digits: str) -> bool:
    length = len(digits)
    return any(
        digits.startswith(prefix) and length in lengths
        for prefix, lengths in _BANK_CARD_BIN_RULES
    )


_BANK_CARD_MASK = _mask_match(keep_start=4, keep_end=4)


def _mask_bank_card(match: re.Match[str]) -> str:
    # A bare 13-19 digit run is masked as a bank card only when it either (a)
    # passes a Luhn checksum, or (b) starts with a recognized card BIN
    # prefix. Requiring at least one of these (rather than blanket-masking
    # every 13-19 digit run) keeps ordinary large integers -- millisecond
    # timestamps, row counts, order numbers -- readable in audit evidence,
    # while still catching genuine and mistyped/sample card numbers.
    digits = match.group(0)
    if not (_luhn_checksum_valid(digits) or _has_bank_card_bin_prefix(digits)):
        return digits
    return _BANK_CARD_MASK(match)


__all__ = ["RedactionResult", "redact_text", "redact_value"]
