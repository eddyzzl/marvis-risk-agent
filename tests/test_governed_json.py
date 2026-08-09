from __future__ import annotations

import pytest

from marvis.governed_json import (
    GovernedJsonError,
    strict_json_object_from_bytes,
    strict_json_object_from_text,
)


def test_strict_governed_json_accepts_one_nested_object() -> None:
    assert strict_json_object_from_text(
        '{"schema":"v1","nested":{"value":1}}',
        "evidence",
    ) == {"schema": "v1", "nested": {"value": 1}}


@pytest.mark.parametrize(
    "payload",
    [
        '{"id":"a","id":"b"}',
        '{"nested":{"id":"a","id":"b"}}',
    ],
)
def test_strict_governed_json_rejects_duplicate_keys(payload: str) -> None:
    with pytest.raises(GovernedJsonError, match="duplicate JSON key: id"):
        strict_json_object_from_text(payload, "evidence")


@pytest.mark.parametrize(
    "payload",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        "```json\n{}\n```",
        "prefix {}",
        "<think>x</think>{}",
        '{"truncated":',
        "[]",
    ],
)
def test_strict_governed_json_rejects_noncanonical_artifacts(payload: str) -> None:
    with pytest.raises(GovernedJsonError):
        strict_json_object_from_text(payload, "evidence")


def test_strict_governed_json_rejects_invalid_utf8() -> None:
    with pytest.raises(GovernedJsonError, match="strict UTF-8 JSON"):
        strict_json_object_from_bytes(b'{"value":"\xff"}', "evidence")


def test_strict_governed_json_translates_to_domain_exception() -> None:
    class DomainError(ValueError):
        pass

    with pytest.raises(DomainError, match="duplicate JSON key"):
        strict_json_object_from_text(
            '{"id":1,"id":2}',
            "asset",
            error_factory=DomainError,
        )
