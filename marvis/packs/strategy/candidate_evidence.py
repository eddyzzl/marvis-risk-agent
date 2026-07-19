"""Strict, deterministic evidence contract for strategy candidates.

The contract intentionally describes development evidence only.  It cannot be
used to claim validation or adoption, and it does not own persistence or any
candidate lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import re
from typing import Any

from marvis.packs.strategy.errors import StrategyError


CANDIDATE_EVIDENCE_SCHEMA_VERSION = "strategy.candidate-evidence.v1"
DEFAULT_PRODUCER_VERSION = "strategy.candidate-evidence/1"
METRIC_DIMENSIONS = ("count", "loan_amount", "overdue_amount")
METRIC_STATUSES = frozenset(
    {"observed", "unavailable", "insufficient_data", "not_applicable"}
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_type",
        "effect_stage",
        "validation_status",
        "identity",
        "generation",
        "analysis",
        "metrics",
        "source_refs",
        "red_flags",
        "producer_version",
        "evidence_hash",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
    }
)
_GENERATION_FIELDS = frozenset({"parameters", "seed", "budget", "truncated"})
_METRIC_FIELDS = frozenset({"metric_name", "dimension", "status", "value"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")


class CandidateEvidenceError(StrategyError):
    """The candidate evidence is not the exact V1 contract."""


@dataclass(frozen=True)
class MetricObservation:
    """One metric measured through one explicit business denominator."""

    metric_name: str
    dimension: str
    status: str
    value: int | float | None

    def to_dict(self) -> dict[str, Any]:
        return _normalize_metric(self.to_unchecked_dict(), index=None)

    def to_unchecked_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "dimension": self.dimension,
            "status": self.status,
            "value": self.value,
        }


def build_candidate_evidence(
    *,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    generation_parameters: Mapping[str, Any],
    seed: int,
    budget: int,
    truncated: bool,
    analysis: Mapping[str, Any],
    metrics: Sequence[MetricObservation | Mapping[str, Any]],
    source_refs: Sequence[str],
    red_flags: Sequence[str] = (),
    producer_version: str = DEFAULT_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Build a self-authenticating development-stage univariate payload."""

    base = {
        "schema_version": CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "candidate_type": "univariate",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "identity": {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "dataset_content_hash": dataset_content_hash,
            "workspace_revision": workspace_revision,
            "workspace_generation": workspace_generation,
            "semantic_mapping_hash": semantic_mapping_hash,
        },
        "generation": {
            "parameters": dict(generation_parameters),
            "seed": seed,
            "budget": budget,
            "truncated": truncated,
        },
        "analysis": dict(analysis),
        "metrics": [
            item.to_unchecked_dict()
            if isinstance(item, MetricObservation)
            else dict(item)
            for item in metrics
        ],
        "source_refs": list(source_refs),
        "red_flags": list(red_flags),
        "producer_version": producer_version,
    }
    normalized_base = _normalize_base(base)
    candidate_id = _candidate_id(normalized_base)
    without_hash = {**normalized_base, "candidate_id": candidate_id}
    evidence_hash = _sha256(_canonical_json(without_hash))
    return validate_candidate_evidence({**without_hash, "evidence_hash": evidence_hash})


def validate_candidate_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a detached canonical payload; fail closed on drift."""

    if not isinstance(payload, Mapping):
        raise CandidateEvidenceError("candidate evidence must be an object")
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, field_name="candidate evidence")
    candidate_id = _required_text(payload["candidate_id"], "candidate_id")
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise CandidateEvidenceError("candidate_id has an invalid format")
    evidence_hash = _sha256_text(payload["evidence_hash"], "evidence_hash")

    base = {
        key: payload[key]
        for key in payload
        if key not in {"candidate_id", "evidence_hash"}
    }
    normalized_base = _normalize_base(base)
    expected_id = _candidate_id(normalized_base)
    if candidate_id != expected_id:
        raise CandidateEvidenceError(
            "candidate_id does not match canonical candidate identity"
        )

    normalized_without_hash = {**normalized_base, "candidate_id": candidate_id}
    expected_hash = _sha256(_canonical_json(normalized_without_hash))
    if not hmac.compare_digest(evidence_hash, expected_hash):
        raise CandidateEvidenceError("evidence_hash does not match canonical evidence")
    return {**normalized_without_hash, "evidence_hash": evidence_hash}


def canonicalize_candidate_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Alias emphasizing the normalized return value of strict validation."""

    return validate_candidate_evidence(payload)


def canonical_candidate_evidence_json(payload: Mapping[str, Any]) -> str:
    return _canonical_json(validate_candidate_evidence(payload))


def candidate_evidence_hash(payload: Mapping[str, Any]) -> str:
    """Return the verified evidence hash, never a hash of unvalidated input."""

    return validate_candidate_evidence(payload)["evidence_hash"]


def candidate_evidence_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize using the sole canonical JSON representation."""

    return canonical_candidate_evidence_json(payload)


def candidate_evidence_from_json(raw: str | bytes | bytearray) -> dict[str, Any]:
    """Parse a JSON roundtrip through the same strict contract boundary."""

    if not isinstance(raw, (str, bytes, bytearray)):
        raise CandidateEvidenceError("candidate evidence JSON must be text or bytes")
    try:
        payload = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateEvidenceError(
            f"candidate evidence is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CandidateEvidenceError("candidate evidence JSON must contain an object")
    return validate_candidate_evidence(payload)


def _normalize_base(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = _TOP_LEVEL_FIELDS - {"candidate_id", "evidence_hash"}
    _require_exact_fields(payload, expected, field_name="candidate evidence body")
    if payload["schema_version"] != CANDIDATE_EVIDENCE_SCHEMA_VERSION:
        raise CandidateEvidenceError(
            f"schema_version must be {CANDIDATE_EVIDENCE_SCHEMA_VERSION}"
        )
    if payload["candidate_type"] != "univariate":
        raise CandidateEvidenceError("candidate_type must be univariate")
    if payload["effect_stage"] != "development":
        raise CandidateEvidenceError("effect_stage must be development")
    if payload["validation_status"] != "unvalidated":
        raise CandidateEvidenceError(
            "development candidate evidence cannot claim validation"
        )

    identity = _normalize_identity(payload["identity"])
    generation = _normalize_generation(payload["generation"])
    analysis = _json_object(payload["analysis"], "analysis")
    _reject_validation_claims(analysis, field_name="analysis")
    metrics = _normalize_metrics(payload["metrics"])
    source_refs = _normalize_text_set(
        payload["source_refs"], "source_refs", required=True
    )
    red_flags = _normalize_text_set(payload["red_flags"], "red_flags", required=False)
    producer_version = _required_text(payload["producer_version"], "producer_version")
    return {
        "schema_version": CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "candidate_type": "univariate",
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "identity": identity,
        "generation": generation,
        "analysis": analysis,
        "metrics": metrics,
        "source_refs": source_refs,
        "red_flags": red_flags,
        "producer_version": producer_version,
    }


def _normalize_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError("identity must be an object")
    _require_exact_fields(value, _IDENTITY_FIELDS, field_name="identity")
    return {
        "task_id": _required_text(value["task_id"], "identity.task_id"),
        "dataset_id": _required_text(value["dataset_id"], "identity.dataset_id"),
        "dataset_content_hash": _sha256_text(
            value["dataset_content_hash"], "identity.dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "identity.workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "identity.workspace_generation"
        ),
        "semantic_mapping_hash": _sha256_text(
            value["semantic_mapping_hash"], "identity.semantic_mapping_hash"
        ),
    }


def _normalize_generation(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError("generation must be an object")
    _require_exact_fields(value, _GENERATION_FIELDS, field_name="generation")
    truncated = value["truncated"]
    if not isinstance(truncated, bool):
        raise CandidateEvidenceError("generation.truncated must be a boolean")
    return {
        "parameters": _generation_parameters(value["parameters"]),
        "seed": _non_negative_int(value["seed"], "generation.seed"),
        "budget": _positive_int(value["budget"], "generation.budget"),
        "truncated": truncated,
    }


def _normalize_metrics(value: object) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CandidateEvidenceError("metrics must be an array")
    if not value:
        raise CandidateEvidenceError("metrics must not be empty")
    normalized = [
        _normalize_metric(item, index=index) for index, item in enumerate(value)
    ]
    identities = [(item["metric_name"], item["dimension"]) for item in normalized]
    if len(set(identities)) != len(identities):
        raise CandidateEvidenceError("metrics contains duplicate metric identities")
    dimensions_by_metric: dict[str, set[str]] = {}
    for metric_name, dimension in identities:
        dimensions_by_metric.setdefault(metric_name, set()).add(dimension)
    incomplete = sorted(
        metric_name
        for metric_name, dimensions in dimensions_by_metric.items()
        if dimensions != set(METRIC_DIMENSIONS)
    )
    if incomplete:
        raise CandidateEvidenceError(
            "every metric must explicitly cover count, loan_amount, and overdue_amount; "
            "missing dimensions require a non-observed status: " + ", ".join(incomplete)
        )
    return sorted(normalized, key=lambda item: (item["metric_name"], item["dimension"]))


def _normalize_metric(value: object, *, index: int | None) -> dict[str, Any]:
    prefix = "metric" if index is None else f"metrics[{index}]"
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError(f"{prefix} must be an object")
    _require_exact_fields(value, _METRIC_FIELDS, field_name=prefix)
    metric_name = _required_text(value["metric_name"], f"{prefix}.metric_name")
    dimension = value["dimension"]
    if dimension not in METRIC_DIMENSIONS:
        raise CandidateEvidenceError(
            f"{prefix}.dimension must be one of {', '.join(METRIC_DIMENSIONS)}"
        )
    status = value["status"]
    if status not in METRIC_STATUSES:
        raise CandidateEvidenceError(f"{prefix}.status is unsupported")
    metric_value = value["value"]
    if status == "observed":
        metric_value = _finite_number(metric_value, f"{prefix}.value")
    elif metric_value is not None:
        raise CandidateEvidenceError(
            f"{prefix}.value must be null when status is {status}"
        )
    return {
        "metric_name": metric_name,
        "dimension": dimension,
        "status": status,
        "value": metric_value,
    }


def _normalize_text_set(value: object, field_name: str, *, required: bool) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CandidateEvidenceError(f"{field_name} must be an array")
    items = [_required_text(item, f"{field_name}[]") for item in value]
    if required and not items:
        raise CandidateEvidenceError(f"{field_name} must not be empty")
    if len(set(items)) != len(items):
        raise CandidateEvidenceError(f"{field_name} must not contain duplicates")
    return sorted(items)


def _json_object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateEvidenceError(f"{field_name} must be an object")
    copied = _normalize_json_value(value, field_name=field_name)
    assert isinstance(copied, dict)
    _canonical_json(copied, field_name=field_name)
    return copied


def _generation_parameters(value: object) -> dict[str, Any]:
    parameters = _json_object(value, "generation.parameters")
    _reject_validation_claims(parameters, field_name="generation.parameters")
    return parameters


def _reject_validation_claims(value: object, *, field_name: str) -> None:
    reserved = {
        "adopted",
        "approval_status",
        "effect_stage",
        "is_adopted",
        "is_validated",
        "validated",
        "validation_status",
    }
    if isinstance(value, Mapping):
        claimed = sorted(key for key in value if key.casefold() in reserved)
        if claimed:
            raise CandidateEvidenceError(
                f"{field_name} cannot contain validation or adoption claims: "
                + ", ".join(claimed)
            )
        for key, child in value.items():
            _reject_validation_claims(child, field_name=f"{field_name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_validation_claims(child, field_name=f"{field_name}[{index}]")


def _normalize_json_value(value: object, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CandidateEvidenceError(
                f"{field_name} is not finite canonical JSON: non-finite number"
            )
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidateEvidenceError(f"{field_name} keys must be strings")
        return {
            key: _normalize_json_value(child, field_name=f"{field_name}.{key}")
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json_value(child, field_name=f"{field_name}[{index}]")
            for index, child in enumerate(value)
        ]
    raise CandidateEvidenceError(
        f"{field_name} is not finite canonical JSON: unsupported {type(value).__name__}"
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateEvidenceError(
                f"candidate evidence JSON has duplicate key: {key}"
            )
        result[key] = value
    return result


def _canonical_json(value: object, *, field_name: str = "candidate evidence") -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateEvidenceError(
            f"{field_name} is not finite canonical JSON: {exc}"
        ) from exc


def _candidate_id(normalized_base: Mapping[str, Any]) -> str:
    return "candidate-" + _sha256(_canonical_json(normalized_base))[:32]


def _sha256(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CandidateEvidenceError(f"{field_name} must be a lowercase SHA-256 hash")
    return value


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CandidateEvidenceError(f"{field_name} must be non-empty canonical text")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateEvidenceError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CandidateEvidenceError(f"{field_name} must be a positive integer")
    return value


def _finite_number(value: object, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateEvidenceError(f"{field_name} must be a finite number")
    if not math.isfinite(value):
        raise CandidateEvidenceError(f"{field_name} must be a finite number")
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], *, field_name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CandidateEvidenceError(f"{field_name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unknown:
            detail.append("unknown: " + ", ".join(unknown))
        raise CandidateEvidenceError(
            f"{field_name} fields are invalid ({'; '.join(detail)})"
        )


__all__ = [
    "CANDIDATE_EVIDENCE_SCHEMA_VERSION",
    "DEFAULT_PRODUCER_VERSION",
    "METRIC_DIMENSIONS",
    "METRIC_STATUSES",
    "CandidateEvidenceError",
    "MetricObservation",
    "build_candidate_evidence",
    "candidate_evidence_from_json",
    "candidate_evidence_hash",
    "candidate_evidence_to_json",
    "canonical_candidate_evidence_json",
    "canonicalize_candidate_evidence",
    "validate_candidate_evidence",
]
