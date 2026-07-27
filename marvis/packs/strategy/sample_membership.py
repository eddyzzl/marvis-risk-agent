"""Canonical row-membership evidence for StrategySampleDesign V2.

The codec is intentionally small and pure.  Callers resolve row membership;
this module only freezes six masks against one exact dataset row ordinal and
verifies their structural integrity.  It does not load datasets, query a
database, or decide which rows belong to a population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
import re
import struct
from typing import Any

import numpy as np

from marvis.packs.strategy.errors import StrategyError


SAMPLE_MEMBERSHIP_SCHEMA_VERSION = "strategy.sample-membership.v2"
SAMPLE_MEMBERSHIP_CODEC_VERSION = "marvis.strategy.sample-membership/2"
MEMBERSHIP_MASK_ORDER = (
    "approval/development",
    "approval/validation",
    "approval/oot",
    "risk/development",
    "risk/validation",
    "risk/oot",
)

_MAGIC = b"MRVSMB2\x00"
_HEADER_LENGTH = struct.Struct("<Q")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MEMBERSHIP_ID_RE = re.compile(r"^strategy-sample-membership-[0-9a-f]{24}$")
_HEADER_FIELDS = frozenset(
    {
        "schema_version",
        "codec_version",
        "membership_id",
        "task_id",
        "dataset_ref",
        "row_count",
        "row_ordinal",
        "mask_order",
        "bitorder",
        "bytes_per_mask",
        "payload_bytes",
        "counts",
        "payload_hash",
        "content_hash",
    }
)
_DATASET_REF_FIELDS = frozenset({"dataset_id", "content_hash"})
_ROW_ORDINAL_FIELDS = frozenset({"start", "stop", "step"})
_COUNTS_FIELDS = frozenset(
    {"analysis_universe", "approval", "risk", "relationship"}
)
_POPULATION_COUNTS_FIELDS = frozenset(
    {"development", "validation", "oot", "total"}
)
_RELATIONSHIP_COUNTS_FIELDS = frozenset(
    {"risk_within_approval", "risk_outside_approval"}
)

MAX_MEMBERSHIP_HEADER_BYTES = 1024 * 1024
MAX_MEMBERSHIP_PAYLOAD_BYTES = 256 * 1024 * 1024


class StrategySampleMembershipError(StrategyError):
    """The sample-membership bytes do not satisfy the exact V2 contract."""


def encode_sample_membership(
    *,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    masks: Mapping[str, object],
    codec_version: str = SAMPLE_MEMBERSHIP_CODEC_VERSION,
) -> bytes:
    """Encode the fixed six masks into one deterministic binary artifact.

    All masks describe the same positive, zero-based row ordinal.  The three
    partitions within each population must be mutually exclusive.  Approval
    and risk masks may overlap; their semantic relationship is evaluated by
    the V2 diagnostics layer.
    """

    task = _text(task_id, "task_id")
    dataset = _text(dataset_id, "dataset_id")
    dataset_hash = _hash(dataset_content_hash, "dataset_content_hash")
    codec = _text(codec_version, "codec_version")
    if codec != SAMPLE_MEMBERSHIP_CODEC_VERSION:
        raise StrategySampleMembershipError("codec_version is unsupported")
    normalized_masks = _normalize_masks(masks)
    row_count = len(normalized_masks[MEMBERSHIP_MASK_ORDER[0]])
    bytes_per_mask = math.ceil(row_count / 8)
    payload_bytes = bytes_per_mask * len(MEMBERSHIP_MASK_ORDER)
    if payload_bytes > MAX_MEMBERSHIP_PAYLOAD_BYTES:
        raise StrategySampleMembershipError("membership payload exceeds byte budget")
    _validate_population_exclusivity(normalized_masks)

    packed = [
        np.packbits(normalized_masks[name], bitorder="little").tobytes()
        for name in MEMBERSHIP_MASK_ORDER
    ]
    payload = b"".join(packed)
    payload_hash = _sha256(payload)
    counts = _membership_counts(normalized_masks)
    body = {
        "schema_version": SAMPLE_MEMBERSHIP_SCHEMA_VERSION,
        "codec_version": codec,
        "task_id": task,
        "dataset_ref": {
            "dataset_id": dataset,
            "content_hash": dataset_hash,
        },
        "row_count": row_count,
        "row_ordinal": {"start": 0, "stop": row_count, "step": 1},
        "mask_order": list(MEMBERSHIP_MASK_ORDER),
        "bitorder": "little",
        "bytes_per_mask": bytes_per_mask,
        "payload_bytes": len(payload),
        "counts": counts,
        "payload_hash": payload_hash,
    }
    header = _address_header(body)
    header_bytes = _canonical_json(header).encode("utf-8")
    if len(header_bytes) > MAX_MEMBERSHIP_HEADER_BYTES:
        raise StrategySampleMembershipError("membership header exceeds byte budget")
    return _MAGIC + _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + payload


def decode_sample_membership(raw: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Decode and fully authenticate a sample-membership artifact."""

    encoded = _bytes(raw, "sample membership")
    prefix_size = len(_MAGIC) + _HEADER_LENGTH.size
    if len(encoded) < prefix_size or not encoded.startswith(_MAGIC):
        raise StrategySampleMembershipError("sample membership magic is invalid")
    header_length = _HEADER_LENGTH.unpack_from(encoded, len(_MAGIC))[0]
    if header_length == 0 or header_length > MAX_MEMBERSHIP_HEADER_BYTES:
        raise StrategySampleMembershipError("membership header length is invalid")
    payload_start = prefix_size + header_length
    if payload_start > len(encoded):
        raise StrategySampleMembershipError("sample membership is truncated")
    header_bytes = encoded[prefix_size:payload_start]
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except StrategySampleMembershipError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategySampleMembershipError(
            "membership header is not valid JSON"
        ) from exc
    if not isinstance(header, dict):
        raise StrategySampleMembershipError("membership header must be an object")
    normalized_header = validate_sample_membership_header(header)
    if header_bytes != _canonical_json(normalized_header).encode("utf-8"):
        raise StrategySampleMembershipError("membership header is not canonical JSON")

    payload = encoded[payload_start:]
    if len(payload) != normalized_header["payload_bytes"]:
        raise StrategySampleMembershipError(
            "membership payload length does not match header"
        )
    if len(payload) > MAX_MEMBERSHIP_PAYLOAD_BYTES:
        raise StrategySampleMembershipError("membership payload exceeds byte budget")
    if not hmac.compare_digest(
        normalized_header["payload_hash"], _sha256(payload)
    ):
        raise StrategySampleMembershipError(
            "membership payload_hash does not match payload"
        )

    masks = _unpack_masks(
        payload,
        row_count=normalized_header["row_count"],
        bytes_per_mask=normalized_header["bytes_per_mask"],
    )
    _validate_population_exclusivity(masks)
    if _membership_counts(masks) != normalized_header["counts"]:
        raise StrategySampleMembershipError(
            "membership counts do not match payload"
        )
    return {"header": normalized_header, "masks": masks}


def validate_sample_membership(
    raw: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Validate an encoded membership artifact and return its decoded value."""

    return decode_sample_membership(raw)


def validate_sample_membership_header(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a standalone canonical membership header."""

    obj = _object(payload, "membership header")
    _require_exact_fields(obj, _HEADER_FIELDS, "membership header")
    if obj["schema_version"] != SAMPLE_MEMBERSHIP_SCHEMA_VERSION:
        raise StrategySampleMembershipError(
            "membership header schema_version is invalid"
        )
    codec = _text(obj["codec_version"], "membership header.codec_version")
    if codec != SAMPLE_MEMBERSHIP_CODEC_VERSION:
        raise StrategySampleMembershipError(
            "membership header.codec_version is unsupported"
        )
    task_id = _text(obj["task_id"], "membership header.task_id")
    dataset_ref = _dataset_ref(obj["dataset_ref"])
    row_count = _positive_int(obj["row_count"], "membership header.row_count")
    row_ordinal = _row_ordinal(obj["row_ordinal"], row_count=row_count)
    mask_order = _mask_order(obj["mask_order"])
    if obj["bitorder"] != "little":
        raise StrategySampleMembershipError(
            "membership header.bitorder must be little"
        )
    expected_bytes_per_mask = math.ceil(row_count / 8)
    bytes_per_mask = _positive_int(
        obj["bytes_per_mask"], "membership header.bytes_per_mask"
    )
    if bytes_per_mask != expected_bytes_per_mask:
        raise StrategySampleMembershipError(
            "membership header.bytes_per_mask does not match row_count"
        )
    payload_bytes = _positive_int(
        obj["payload_bytes"], "membership header.payload_bytes"
    )
    if payload_bytes != bytes_per_mask * len(MEMBERSHIP_MASK_ORDER):
        raise StrategySampleMembershipError(
            "membership header.payload_bytes is invalid"
        )
    if payload_bytes > MAX_MEMBERSHIP_PAYLOAD_BYTES:
        raise StrategySampleMembershipError("membership payload exceeds byte budget")
    counts = _counts(obj["counts"], row_count=row_count)
    payload_hash = _hash(
        obj["payload_hash"], "membership header.payload_hash"
    )
    normalized_body = {
        "schema_version": SAMPLE_MEMBERSHIP_SCHEMA_VERSION,
        "codec_version": codec,
        "task_id": task_id,
        "dataset_ref": dataset_ref,
        "row_count": row_count,
        "row_ordinal": row_ordinal,
        "mask_order": mask_order,
        "bitorder": "little",
        "bytes_per_mask": bytes_per_mask,
        "payload_bytes": payload_bytes,
        "counts": counts,
        "payload_hash": payload_hash,
    }
    return _validate_addressed_header(obj, normalized_body)


def canonical_sample_membership_header_json(payload: Mapping[str, Any]) -> str:
    """Return the sole canonical JSON representation of a valid header."""

    return _canonical_json(validate_sample_membership_header(payload))


def sample_membership_header_from_json(
    raw: str | bytes | bytearray,
) -> dict[str, Any]:
    """Load a standalone header while rejecting duplicate keys."""

    if not isinstance(raw, (str, bytes, bytearray)):
        raise StrategySampleMembershipError("membership header JSON must be text or bytes")
    size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    if size > MAX_MEMBERSHIP_HEADER_BYTES:
        raise StrategySampleMembershipError("membership header exceeds byte budget")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except StrategySampleMembershipError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StrategySampleMembershipError(
            "membership header is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategySampleMembershipError("membership header must be an object")
    return validate_sample_membership_header(payload)


def _normalize_masks(masks: Mapping[str, object]) -> dict[str, np.ndarray]:
    obj = _object(masks, "masks")
    _require_exact_fields(obj, frozenset(MEMBERSHIP_MASK_ORDER), "masks")
    normalized: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name in MEMBERSHIP_MASK_ORDER:
        raw = obj[name]
        if isinstance(raw, np.ndarray):
            array = raw
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            array = np.asarray(raw)
        else:
            raise StrategySampleMembershipError(f"mask {name} must be a boolean array")
        if array.ndim != 1 or array.dtype.kind != "b":
            raise StrategySampleMembershipError(
                f"mask {name} must be a one-dimensional boolean array"
            )
        if row_count is None:
            row_count = len(array)
            if row_count == 0:
                raise StrategySampleMembershipError("membership masks must not be empty")
        elif len(array) != row_count:
            raise StrategySampleMembershipError(
                "membership masks must have the same row count"
            )
        normalized[name] = np.ascontiguousarray(array, dtype=np.bool_)
    return normalized


def _unpack_masks(
    payload: bytes, *, row_count: int, bytes_per_mask: int
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    valid_bits_in_last_byte = row_count % 8
    for index, name in enumerate(MEMBERSHIP_MASK_ORDER):
        start = index * bytes_per_mask
        segment = payload[start : start + bytes_per_mask]
        if valid_bits_in_last_byte:
            invalid_tail_mask = 0xFF ^ ((1 << valid_bits_in_last_byte) - 1)
            if segment[-1] & invalid_tail_mask:
                raise StrategySampleMembershipError(
                    f"mask {name} has non-zero unused tail bits"
                )
        masks[name] = np.unpackbits(
            np.frombuffer(segment, dtype=np.uint8), bitorder="little"
        )[:row_count].astype(np.bool_, copy=True)
    return masks


def _validate_population_exclusivity(
    masks: Mapping[str, np.ndarray],
) -> None:
    for population in ("approval", "risk"):
        names = (
            f"{population}/development",
            f"{population}/validation",
            f"{population}/oot",
        )
        overlaps = (
            (masks[names[0]] & masks[names[1]])
            | (masks[names[0]] & masks[names[2]])
            | (masks[names[1]] & masks[names[2]])
        )
        if bool(np.any(overlaps)):
            raise StrategySampleMembershipError(
                f"{population} development/validation/oot masks must be mutually exclusive"
            )


def _membership_counts(masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    row_count = len(masks[MEMBERSHIP_MASK_ORDER[0]])
    result: dict[str, Any] = {"analysis_universe": row_count}
    for population in ("approval", "risk"):
        counts = {
            split: int(np.count_nonzero(masks[f"{population}/{split}"]))
            for split in ("development", "validation", "oot")
        }
        result[population] = {**counts, "total": sum(counts.values())}
    within = {
        split: int(
            np.count_nonzero(
                masks[f"risk/{split}"] & masks[f"approval/{split}"]
            )
        )
        for split in ("development", "validation", "oot")
    }
    outside = {
        split: int(
            np.count_nonzero(
                masks[f"risk/{split}"] & ~masks[f"approval/{split}"]
            )
        )
        for split in ("development", "validation", "oot")
    }
    result["relationship"] = {
        "risk_within_approval": {**within, "total": sum(within.values())},
        "risk_outside_approval": {**outside, "total": sum(outside.values())},
    }
    return result


def _address_header(body: Mapping[str, Any]) -> dict[str, Any]:
    membership_id = (
        "strategy-sample-membership-" + _sha256(_canonical_json(body))[:24]
    )
    without_hash = {**body, "membership_id": membership_id}
    return {
        **without_hash,
        "content_hash": _sha256(_canonical_json(without_hash)),
    }


def _validate_addressed_header(
    original: Mapping[str, Any], normalized_body: Mapping[str, Any]
) -> dict[str, Any]:
    membership_id = original["membership_id"]
    if (
        not isinstance(membership_id, str)
        or _MEMBERSHIP_ID_RE.fullmatch(membership_id) is None
    ):
        raise StrategySampleMembershipError("membership header.membership_id is invalid")
    expected_id = (
        "strategy-sample-membership-"
        + _sha256(_canonical_json(normalized_body))[:24]
    )
    if not hmac.compare_digest(membership_id, expected_id):
        raise StrategySampleMembershipError(
            "membership header.membership_id does not match content"
        )
    content_hash = _hash(
        original["content_hash"], "membership header.content_hash"
    )
    without_hash = {**normalized_body, "membership_id": membership_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(content_hash, expected_hash):
        raise StrategySampleMembershipError(
            "membership header.content_hash does not match content"
        )
    return {**without_hash, "content_hash": content_hash}


def _dataset_ref(value: object) -> dict[str, str]:
    obj = _object(value, "membership header.dataset_ref")
    _require_exact_fields(obj, _DATASET_REF_FIELDS, "membership header.dataset_ref")
    return {
        "dataset_id": _text(
            obj["dataset_id"], "membership header.dataset_ref.dataset_id"
        ),
        "content_hash": _hash(
            obj["content_hash"], "membership header.dataset_ref.content_hash"
        ),
    }


def _row_ordinal(value: object, *, row_count: int) -> dict[str, int]:
    obj = _object(value, "membership header.row_ordinal")
    _require_exact_fields(obj, _ROW_ORDINAL_FIELDS, "membership header.row_ordinal")
    expected = {"start": 0, "stop": row_count, "step": 1}
    if dict(obj) != expected:
        raise StrategySampleMembershipError(
            "membership header.row_ordinal must cover exactly 0..row_count-1"
        )
    return expected


def _mask_order(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StrategySampleMembershipError("membership header.mask_order must be an array")
    normalized = list(value)
    if normalized != list(MEMBERSHIP_MASK_ORDER):
        raise StrategySampleMembershipError(
            "membership header.mask_order is invalid"
        )
    return normalized


def _counts(value: object, *, row_count: int) -> dict[str, Any]:
    obj = _object(value, "membership header.counts")
    _require_exact_fields(obj, _COUNTS_FIELDS, "membership header.counts")
    universe = _non_negative_int(
        obj["analysis_universe"],
        "membership header.counts.analysis_universe",
    )
    if universe != row_count:
        raise StrategySampleMembershipError(
            "membership header analysis_universe count does not match row_count"
        )
    normalized: dict[str, Any] = {"analysis_universe": universe}
    for population in ("approval", "risk"):
        pop = _object(obj[population], f"membership header.counts.{population}")
        _require_exact_fields(
            pop,
            _POPULATION_COUNTS_FIELDS,
            f"membership header.counts.{population}",
        )
        split_counts = {
            split: _non_negative_int(
                pop[split],
                f"membership header.counts.{population}.{split}",
            )
            for split in ("development", "validation", "oot")
        }
        total = _non_negative_int(
            pop["total"], f"membership header.counts.{population}.total"
        )
        if total != sum(split_counts.values()) or total > row_count:
            raise StrategySampleMembershipError(
                f"membership header {population} counts do not conserve"
            )
        normalized[population] = {**split_counts, "total": total}
    relationship = _object(
        obj["relationship"], "membership header.counts.relationship"
    )
    _require_exact_fields(
        relationship,
        _RELATIONSHIP_COUNTS_FIELDS,
        "membership header.counts.relationship",
    )
    normalized_relationship: dict[str, dict[str, int]] = {}
    for relationship_name in (
        "risk_within_approval",
        "risk_outside_approval",
    ):
        raw_counts = _object(
            relationship[relationship_name],
            f"membership header.counts.relationship.{relationship_name}",
        )
        _require_exact_fields(
            raw_counts,
            _POPULATION_COUNTS_FIELDS,
            f"membership header.counts.relationship.{relationship_name}",
        )
        split_counts = {
            split: _non_negative_int(
                raw_counts[split],
                "membership header.counts.relationship."
                f"{relationship_name}.{split}",
            )
            for split in ("development", "validation", "oot")
        }
        total = _non_negative_int(
            raw_counts["total"],
            "membership header.counts.relationship."
            f"{relationship_name}.total",
        )
        if total != sum(split_counts.values()):
            raise StrategySampleMembershipError(
                f"membership header {relationship_name} counts do not conserve"
            )
        normalized_relationship[relationship_name] = {
            **split_counts,
            "total": total,
        }
    within = normalized_relationship["risk_within_approval"]
    outside = normalized_relationship["risk_outside_approval"]
    for split in ("development", "validation", "oot"):
        if within[split] + outside[split] != normalized["risk"][split]:
            raise StrategySampleMembershipError(
                "membership header relationship counts do not conserve risk membership"
            )
        if within[split] > normalized["approval"][split]:
            raise StrategySampleMembershipError(
                "membership header risk-within-approval count exceeds approval membership"
            )
    if within["total"] + outside["total"] != normalized["risk"]["total"]:
        raise StrategySampleMembershipError(
            "membership header relationship totals do not conserve risk membership"
        )
    normalized["relationship"] = normalized_relationship
    return normalized


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategySampleMembershipError(f"{name} must be an object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise StrategySampleMembershipError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise StrategySampleMembershipError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise StrategySampleMembershipError("value is not canonical JSON") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrategySampleMembershipError(
                f"membership header JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _bytes(value: object, name: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise StrategySampleMembershipError(f"{name} must be bytes")
    return bytes(value)


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategySampleMembershipError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise StrategySampleMembershipError(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategySampleMembershipError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _non_negative_int(value, name)
    if result == 0:
        raise StrategySampleMembershipError(f"{name} must be a positive integer")
    return result


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "MAX_MEMBERSHIP_HEADER_BYTES",
    "MAX_MEMBERSHIP_PAYLOAD_BYTES",
    "MEMBERSHIP_MASK_ORDER",
    "SAMPLE_MEMBERSHIP_CODEC_VERSION",
    "SAMPLE_MEMBERSHIP_SCHEMA_VERSION",
    "StrategySampleMembershipError",
    "canonical_sample_membership_header_json",
    "decode_sample_membership",
    "encode_sample_membership",
    "sample_membership_header_from_json",
    "validate_sample_membership",
    "validate_sample_membership_header",
]
