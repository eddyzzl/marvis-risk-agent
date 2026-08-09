"""Canonical validation for platform-owned Strategy Pool evidence bindings."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
POOL_ID_RE = re.compile(r"^strategy-pool-[0-9a-f]{32}$")
POOL_REVISION_ID_RE = re.compile(
    r"^strategy-pool-revision-[0-9a-f]{32}$"
)
POOL_REF_FIELDS = frozenset(
    {
        "artifact_id",
        "expected_artifact_content_hash",
        "expected_pool_id",
        "expected_revision",
        "expected_revision_id",
        "expected_snapshot_hash",
    }
)


def validate_pool_ref(value: object) -> dict[str, Any]:
    """Authenticate the shape and immutable identity fields of a Pool ref."""

    obj = _canonical_json_object(value, "pool_ref")
    _exact_fields(obj, POOL_REF_FIELDS, "pool_ref")
    pool_id = _text(obj["expected_pool_id"], "pool_ref.expected_pool_id")
    if POOL_ID_RE.fullmatch(pool_id) is None:
        raise StrategyError("pool_ref.expected_pool_id is invalid")
    revision_id = _text(
        obj["expected_revision_id"],
        "pool_ref.expected_revision_id",
    )
    if POOL_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise StrategyError("pool_ref.expected_revision_id is invalid")
    return {
        "artifact_id": _hash(obj["artifact_id"], "pool_ref.artifact_id"),
        "expected_artifact_content_hash": _hash(
            obj["expected_artifact_content_hash"],
            "pool_ref.expected_artifact_content_hash",
        ),
        "expected_pool_id": pool_id,
        "expected_revision": _positive_int(
            obj["expected_revision"],
            "pool_ref.expected_revision",
        ),
        "expected_revision_id": revision_id,
        "expected_snapshot_hash": _hash(
            obj["expected_snapshot_hash"],
            "pool_ref.expected_snapshot_hash",
        ),
    }


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyError(f"{name} must be canonical JSON") from exc
    if not isinstance(normalized, dict):
        raise StrategyError(f"{name} must be an object")
    if normalized != value:
        raise StrategyError(f"{name} contains non-canonical JSON values")
    return normalized


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) == expected:
        return
    unexpected = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    details: list[str] = []
    if unexpected:
        details.append("unsupported fields: " + ", ".join(unexpected))
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    raise StrategyError(f"{name} has invalid fields ({'; '.join(details)})")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise StrategyError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StrategyError(f"{name} must be a positive integer")
    return value


__all__ = ["validate_pool_ref"]
