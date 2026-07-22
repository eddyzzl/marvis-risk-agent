"""Generic, self-authenticating candidate fragments accepted by Strategy Pool.

Candidate-producing algorithms own their concrete artifacts and evidence replay.
The Pool consumes only this verified in-memory projection: one executable rule
fragment plus immutable artifact, asset, effect, and sample provenance.  The
projection is deliberately not a new persisted candidate artifact and cannot
claim validation, adoption, or deployment.

``build_verified_candidate_fragment`` and its validator are a trusted-adapter
handoff, not a generic Tool admission API.  Real Tool persistence must first
dispatch an explicit ``(artifact_kind, origin_tool, artifact_schema_version)``
allowlist and replay the concrete artifact, evidence, and dataset lineage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
import re
from typing import Any

from marvis.packs.strategy.candidate_asset import validate_candidate_asset
from marvis.packs.strategy.candidate_evidence import validate_candidate_evidence
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_binding import StrategySampleDesignRef


VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION = (
    "strategy.verified-candidate-fragment.v1"
)
UNIVARIATE_ASSET_ARTIFACT_KIND = "strategy_candidate_asset_json"
UNIVARIATE_ASSET_ARTIFACT_SCHEMA_VERSION = "strategy.candidate-asset-artifact.v1"
UNIVARIATE_ASSET_ORIGIN_TOOL = "strategy.refine_univariate_candidate"
UNIVARIATE_ASSET_SCHEMA_VERSION = "strategy.candidate-asset.v1"

_CANDIDATE_STAGE = "development"
_OBSERVATION_STAGE = "backtested"
_VALIDATION_STATUS = "unvalidated"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact",
        "asset",
        "fragment",
        "evidence",
        "candidate_stage",
        "observation_stage",
        "validation_status",
        "fragment_hash",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "artifact_kind",
        "artifact_schema_version",
        "artifact_content_hash",
        "origin_tool",
    }
)
_ASSET_FIELDS = frozenset(
    {"schema_version", "asset_id", "asset_hash", "asset_type"}
)
_FRAGMENT_FIELDS = frozenset(
    {
        "fragment_id",
        "fragment_type",
        "rule_id",
        "condition",
        "requirements",
        "effect_id",
    }
)
_EVIDENCE_FIELDS = frozenset({"evidence_id", "evidence_hash", "identity"})
_EVIDENCE_IDENTITY_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "sample_context_hash",
    }
)
_LEGACY_UNIVARIATE_SOURCE_FIELDS = frozenset(
    {
        "artifact_id",
        "kind",
        "content_hash",
        "origin_tool",
        "artifact_schema_version",
        "asset_id",
        "asset_hash",
        "candidate_kind",
        "fragment_id",
        "effect_id",
        "effect_stage",
        "validation_status",
        "parent_candidate_id",
        "parent_evidence_hash",
        "evidence_identity",
    }
)
_LEGACY_EVIDENCE_IDENTITY_FIELDS = _EVIDENCE_IDENTITY_FIELDS - {
    "sample_context_hash"
}


class CandidateFragmentError(StrategyError):
    """A verified candidate fragment or concrete adapter failed closed."""


def build_verified_candidate_fragment(
    *,
    artifact: Mapping[str, Any],
    asset: Mapping[str, Any],
    fragment_type: str,
    rule_id: str,
    condition: Mapping[str, Any],
    requirements: Sequence[Mapping[str, Any]],
    effect_id: str,
    evidence_id: str,
    evidence_hash: str,
    evidence_identity: Mapping[str, Any],
    fragment_id: str | None = None,
    candidate_stage: str = _CANDIDATE_STAGE,
    observation_stage: str = _OBSERVATION_STAGE,
    validation_status: str = _VALIDATION_STATUS,
) -> dict[str, Any]:
    """Build one canonical fragment for a trusted concrete artifact adapter."""

    normalized_artifact = _artifact(artifact)
    normalized_asset = _asset(asset)
    normalized_type = _text(fragment_type, "fragment_type")
    normalized_rule_id = _text(rule_id, "rule_id")
    normalized_condition = _condition(condition)
    normalized_requirements = _requirements(requirements)
    normalized_effect_id = _text(effect_id, "effect_id")
    normalized_evidence_id = _text(evidence_id, "evidence_id")
    normalized_evidence_hash = _hash(evidence_hash, "evidence_hash")
    normalized_identity = _evidence_identity(evidence_identity)
    if fragment_id is None:
        normalized_fragment_id = _stable_id(
            "candidate-fragment",
            {
                "asset_id": normalized_asset["asset_id"],
                "asset_hash": normalized_asset["asset_hash"],
                "fragment_type": normalized_type,
                "rule_id": normalized_rule_id,
                "condition": normalized_condition,
                "requirements": normalized_requirements,
                "effect_id": normalized_effect_id,
                "evidence_id": normalized_evidence_id,
                "evidence_hash": normalized_evidence_hash,
            },
        )
    else:
        normalized_fragment_id = _text(fragment_id, "fragment_id")
    body = {
        "schema_version": VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        "artifact": normalized_artifact,
        "asset": normalized_asset,
        "fragment": {
            "fragment_id": normalized_fragment_id,
            "fragment_type": normalized_type,
            "rule_id": normalized_rule_id,
            "condition": normalized_condition,
            "requirements": normalized_requirements,
            "effect_id": normalized_effect_id,
        },
        "evidence": {
            "evidence_id": normalized_evidence_id,
            "evidence_hash": normalized_evidence_hash,
            "identity": normalized_identity,
        },
        "candidate_stage": candidate_stage,
        "observation_stage": observation_stage,
        "validation_status": validation_status,
    }
    normalized_body = _normalize_body(body)
    return validate_verified_candidate_fragment(
        {**normalized_body, "fragment_hash": _sha256(_canonical_json(normalized_body))}
    )


def validate_verified_candidate_fragment(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate an exact self-authenticating fragment contract."""

    if not isinstance(payload, Mapping):
        raise CandidateFragmentError("verified candidate fragment must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "verified candidate fragment")
    supplied_hash = _hash(payload["fragment_hash"], "fragment_hash")
    body = _normalize_body(
        {key: payload[key] for key in payload if key != "fragment_hash"}
    )
    expected_hash = _sha256(_canonical_json(body))
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise CandidateFragmentError(
            "fragment_hash does not match canonical verified candidate fragment"
        )
    return {**body, "fragment_hash": supplied_hash}


def canonical_verified_candidate_fragment_json(payload: Mapping[str, Any]) -> str:
    return _canonical_json(validate_verified_candidate_fragment(payload))


def sample_context_hash_from_candidate_evidence(
    candidate_evidence: Mapping[str, Any],
) -> str:
    """Bind the exact labelled sample context without binding feature/bin choices.

    The projection binds the immutable StrategySampleDesign reference together
    with dataset/workspace identity, label definition, labelled row count, and
    row-selection parameters while deliberately excluding feature/bin choices.
    """

    try:
        evidence = validate_candidate_evidence(candidate_evidence)
    except StrategyError as exc:
        raise CandidateFragmentError("candidate evidence failed strict validation") from exc
    if evidence["candidate_type"] != "univariate":
        raise CandidateFragmentError("candidate evidence must be univariate")
    analysis = evidence["analysis"]
    generation = evidence["generation"]["parameters"]
    required_analysis = {"schema_version", "target", "target_definition", "row_count"}
    if not required_analysis <= set(analysis):
        raise CandidateFragmentError(
            "univariate evidence lacks a complete sample context"
        )
    try:
        sample_design_ref = StrategySampleDesignRef.from_value(
            generation.get("sample_design_ref")
        ).to_ref_dict()
    except StrategyError as exc:
        raise CandidateFragmentError(
            "univariate evidence lacks a valid sample_design_ref"
        ) from exc
    sample_parameters = {
        key: generation.get(key)
        for key in (
            "analysis_schema_version",
            "target_col",
            "drop_nan_labels",
            "nan_labels_dropped",
            "loan_amount_col",
            "overdue_amount_col",
            "registry_metadata_hash",
        )
        if key in generation
    }
    sample_parameters["sample_design_ref"] = sample_design_ref
    context = {
        "schema_version": "strategy.sample-context.v1",
        "identity": evidence["identity"],
        "analysis": {
            "schema_version": analysis["schema_version"],
            "target": analysis["target"],
            "target_definition": analysis["target_definition"],
            "row_count": analysis["row_count"],
        },
        "sample_parameters": _json_object(
            sample_parameters, "sample_context.sample_parameters"
        ),
    }
    return _sha256(_canonical_json(context))


def univariate_asset_to_verified_fragment(
    candidate_asset: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any],
    candidate_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict adapter from immutable Candidate Asset v1 to the generic seam.

    The adapter never mutates or reserializes the concrete asset.  The Tool
    boundary remains responsible for reloading the canonical artifact and
    replaying its parent evidence and dataset bytes before invoking this mapper.
    """

    try:
        asset = validate_candidate_asset(candidate_asset)
    except StrategyError as exc:
        raise CandidateFragmentError("candidate asset failed strict validation") from exc
    if not isinstance(source_binding, Mapping):
        raise CandidateFragmentError("source_binding must be an object")
    _exact_fields(
        source_binding,
        _LEGACY_UNIVARIATE_SOURCE_FIELDS,
        "univariate source_binding",
    )
    binding_identity_raw = source_binding["evidence_identity"]
    if not isinstance(binding_identity_raw, Mapping):
        raise CandidateFragmentError("source evidence_identity must be an object")
    _exact_fields(
        binding_identity_raw,
        _LEGACY_EVIDENCE_IDENTITY_FIELDS,
        "source evidence_identity",
    )
    binding_identity = {
        "dataset_id": _text(
            binding_identity_raw["dataset_id"], "evidence dataset_id"
        ),
        "dataset_content_hash": _hash(
            binding_identity_raw["dataset_content_hash"],
            "evidence dataset_content_hash",
        ),
        "workspace_revision": _integer(
            binding_identity_raw["workspace_revision"],
            "evidence workspace_revision",
        ),
        "workspace_generation": _integer(
            binding_identity_raw["workspace_generation"],
            "evidence workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            binding_identity_raw["semantic_mapping_hash"],
            "evidence semantic_mapping_hash",
        ),
    }
    if source_binding["kind"] != UNIVARIATE_ASSET_ARTIFACT_KIND:
        raise CandidateFragmentError(
            "univariate source kind must be strategy_candidate_asset_json"
        )
    if source_binding["origin_tool"] != UNIVARIATE_ASSET_ORIGIN_TOOL:
        raise CandidateFragmentError(
            "univariate source origin_tool must be "
            "strategy.refine_univariate_candidate"
        )
    if (
        source_binding["artifact_schema_version"]
        != UNIVARIATE_ASSET_ARTIFACT_SCHEMA_VERSION
    ):
        raise CandidateFragmentError(
            "univariate source artifact_schema_version must be "
            "strategy.candidate-asset-artifact.v1"
        )
    comparisons = {
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": asset["asset_type"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
    }
    for field, expected in comparisons.items():
        actual = source_binding[field]
        if field in {"asset_hash", "parent_evidence_hash"}:
            actual = _hash(actual, f"source {field}")
        else:
            actual = _text(actual, f"source {field}")
        if actual != expected:
            raise CandidateFragmentError(
                f"source {field} does not match the candidate asset"
            )
    if (
        asset["schema_version"] != UNIVARIATE_ASSET_SCHEMA_VERSION
        or asset["asset_type"] != "univariate_refinement"
        or asset["effect_stage"] != _CANDIDATE_STAGE
        or asset["validation_status"] != _VALIDATION_STATUS
    ):
        raise CandidateFragmentError(
            "univariate candidate asset must remain development and unvalidated"
        )
    if candidate_evidence is not None:
        try:
            evidence = validate_candidate_evidence(candidate_evidence)
        except StrategyError as exc:
            raise CandidateFragmentError(
                "candidate evidence failed strict validation"
            ) from exc
        expected_parent = asset["parent"]
        if (
            evidence["candidate_id"] != expected_parent["candidate_id"]
            or not hmac.compare_digest(
                evidence["evidence_hash"], expected_parent["evidence_hash"]
            )
        ):
            raise CandidateFragmentError(
                "candidate evidence does not match candidate asset parent"
            )
        evidence_identity = evidence["identity"]
        for field, expected in binding_identity.items():
            if evidence_identity[field] != expected:
                raise CandidateFragmentError(
                    f"candidate evidence identity {field} does not match source"
                )
        sample_context_hash = sample_context_hash_from_candidate_evidence(evidence)
    else:
        # Compatibility for pure-core callers that only have the historical
        # source binding.  Tool callers always provide the canonical evidence.
        sample_context_hash = _sha256(
            _canonical_json(
                {
                    "schema_version": "strategy.sample-context.legacy-univariate.v1",
                    "identity": binding_identity,
                    "evidence_id": asset["parent"]["candidate_id"],
                    "evidence_hash": asset["parent"]["evidence_hash"],
                }
            )
        )
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": _text(source_binding["artifact_id"], "source artifact_id"),
            "artifact_kind": UNIVARIATE_ASSET_ARTIFACT_KIND,
            "artifact_schema_version": _text(
                source_binding["artifact_schema_version"],
                "source artifact_schema_version",
            ),
            "artifact_content_hash": _hash(
                source_binding["content_hash"], "source content_hash"
            ),
            "origin_tool": UNIVARIATE_ASSET_ORIGIN_TOOL,
        },
        asset={
            "schema_version": asset["schema_version"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "asset_type": asset["asset_type"],
        },
        fragment_type="strategy_rule",
        rule_id=asset["rule"]["rule_id"],
        condition=asset["rule"]["condition"],
        requirements=[],
        effect_id=asset["effect"]["effect_id"],
        evidence_id=asset["parent"]["candidate_id"],
        evidence_hash=asset["parent"]["evidence_hash"],
        evidence_identity={
            **binding_identity,
            "sample_context_hash": sample_context_hash,
        },
    )


def verified_fragment_pool_parts(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Project a verified fragment to the Pool's source/rule/execution fields."""

    fragment = validate_verified_candidate_fragment(payload)
    artifact = fragment["artifact"]
    asset = fragment["asset"]
    rule_fragment = fragment["fragment"]
    evidence = fragment["evidence"]
    source = {
        **artifact,
        "asset_schema_version": asset["schema_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "asset_type": asset["asset_type"],
        "fragment_id": rule_fragment["fragment_id"],
        "fragment_hash": fragment["fragment_hash"],
        "fragment_type": rule_fragment["fragment_type"],
        "effect_id": rule_fragment["effect_id"],
        "evidence_id": evidence["evidence_id"],
        "evidence_hash": evidence["evidence_hash"],
        "candidate_stage": fragment["candidate_stage"],
        "observation_stage": fragment["observation_stage"],
        "validation_status": fragment["validation_status"],
        "evidence_identity": evidence["identity"],
    }
    execution = {
        "condition": rule_fragment["condition"],
        "requirements": rule_fragment["requirements"],
    }
    return source, rule_fragment["rule_id"], execution


def verified_fragment_from_pool_parts(
    *,
    source: Mapping[str, Any],
    rule_id: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct and verify a persisted Pool projection including its hash."""

    if not isinstance(source, Mapping):
        raise CandidateFragmentError("pool source must be an object")
    if not isinstance(execution, Mapping):
        raise CandidateFragmentError("pool execution must be an object")
    payload = {
        "schema_version": VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        "artifact": {
            "artifact_id": source.get("artifact_id"),
            "artifact_kind": source.get("artifact_kind"),
            "artifact_schema_version": source.get("artifact_schema_version"),
            "artifact_content_hash": source.get("artifact_content_hash"),
            "origin_tool": source.get("origin_tool"),
        },
        "asset": {
            "schema_version": source.get("asset_schema_version"),
            "asset_id": source.get("asset_id"),
            "asset_hash": source.get("asset_hash"),
            "asset_type": source.get("asset_type"),
        },
        "fragment": {
            "fragment_id": source.get("fragment_id"),
            "fragment_type": source.get("fragment_type"),
            "rule_id": rule_id,
            "condition": execution.get("condition"),
            "requirements": execution.get("requirements"),
            "effect_id": source.get("effect_id"),
        },
        "evidence": {
            "evidence_id": source.get("evidence_id"),
            "evidence_hash": source.get("evidence_hash"),
            "identity": source.get("evidence_identity"),
        },
        "candidate_stage": source.get("candidate_stage"),
        "observation_stage": source.get("observation_stage"),
        "validation_status": source.get("validation_status"),
        "fragment_hash": source.get("fragment_hash"),
    }
    return validate_verified_candidate_fragment(payload)


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = _TOP_LEVEL_FIELDS - {"fragment_hash"}
    _exact_fields(value, expected, "verified candidate fragment body")
    if value["schema_version"] != VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION:
        raise CandidateFragmentError(
            f"schema_version must be {VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION}"
        )
    candidate_stage = _text(value["candidate_stage"], "candidate_stage")
    if candidate_stage != _CANDIDATE_STAGE:
        raise CandidateFragmentError("candidate_stage must remain development")
    observation_stage = _text(value["observation_stage"], "observation_stage")
    if observation_stage != _OBSERVATION_STAGE:
        raise CandidateFragmentError("observation_stage must remain backtested")
    validation_status = _text(value["validation_status"], "validation_status")
    if validation_status != _VALIDATION_STATUS:
        raise CandidateFragmentError("validation_status must remain unvalidated")
    artifact = _artifact(value["artifact"])
    asset = _asset(value["asset"])
    fragment = _fragment(value["fragment"])
    evidence = _evidence(value["evidence"])
    return {
        "schema_version": VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
        "artifact": artifact,
        "asset": asset,
        "fragment": fragment,
        "evidence": evidence,
        "candidate_stage": candidate_stage,
        "observation_stage": observation_stage,
        "validation_status": validation_status,
    }


def _artifact(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("artifact must be an object")
    _exact_fields(value, _ARTIFACT_FIELDS, "artifact")
    return {
        "artifact_id": _text(value["artifact_id"], "artifact_id"),
        "artifact_kind": _text(value["artifact_kind"], "artifact_kind"),
        "artifact_schema_version": _text(
            value["artifact_schema_version"], "artifact_schema_version"
        ),
        "artifact_content_hash": _hash(
            value["artifact_content_hash"], "artifact_content_hash"
        ),
        "origin_tool": _text(value["origin_tool"], "artifact origin_tool"),
    }


def _asset(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("asset must be an object")
    _exact_fields(value, _ASSET_FIELDS, "asset")
    return {
        "schema_version": _text(value["schema_version"], "asset schema_version"),
        "asset_id": _text(value["asset_id"], "asset_id"),
        "asset_hash": _hash(value["asset_hash"], "asset_hash"),
        "asset_type": _text(value["asset_type"], "asset_type"),
    }


def _fragment(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("fragment must be an object")
    _exact_fields(value, _FRAGMENT_FIELDS, "fragment")
    return {
        "fragment_id": _text(value["fragment_id"], "fragment_id"),
        "fragment_type": _text(value["fragment_type"], "fragment_type"),
        "rule_id": _text(value["rule_id"], "rule_id"),
        "condition": _condition(value["condition"]),
        "requirements": _requirements(value["requirements"]),
        "effect_id": _text(value["effect_id"], "effect_id"),
    }


def _evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("evidence must be an object")
    _exact_fields(value, _EVIDENCE_FIELDS, "evidence")
    return {
        "evidence_id": _text(value["evidence_id"], "evidence_id"),
        "evidence_hash": _hash(value["evidence_hash"], "evidence_hash"),
        "identity": _evidence_identity(value["identity"]),
    }


def _evidence_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("evidence identity must be an object")
    _exact_fields(value, _EVIDENCE_IDENTITY_FIELDS, "evidence identity")
    return {
        "dataset_id": _text(value["dataset_id"], "evidence dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "evidence dataset_content_hash"
        ),
        "workspace_revision": _integer(
            value["workspace_revision"], "evidence workspace_revision"
        ),
        "workspace_generation": _integer(
            value["workspace_generation"], "evidence workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "evidence semantic_mapping_hash"
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"], "evidence sample_context_hash"
        ),
    }


def _condition(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateFragmentError("fragment condition must be an object")
    try:
        canonical = canonicalize_expression(value)
    except StrategyError as exc:
        raise CandidateFragmentError(f"fragment condition is invalid: {exc}") from exc
    if _canonical_json(canonical) != _canonical_json(value):
        raise CandidateFragmentError("fragment condition must be canonical Strategy DSL")
    return canonical


def _requirements(value: object) -> list[dict[str, Any]]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise CandidateFragmentError("fragment requirements must be an array")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        requirement = _json_object(item, f"fragment requirements[{index}]")
        if "type" not in requirement:
            raise CandidateFragmentError(
                f"fragment requirements[{index}] requires a typed type field"
            )
        requirement["type"] = _text(
            requirement["type"], f"fragment requirements[{index}].type"
        )
        normalized.append(requirement)
    return normalized


def _json_object(value: object, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise CandidateFragmentError(f"{name} must be an object")
    return normalized


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise CandidateFragmentError(f"{name} must contain finite JSON")
        return normalized
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CandidateFragmentError(f"{name} keys must be strings")
        return {key: _json_value(child, f"{name}.{key}") for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(child, f"{name}[{index}]") for index, child in enumerate(value)]
    raise CandidateFragmentError(f"{name} must contain canonical JSON values")


def _exact_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise CandidateFragmentError(f"{name} has " + "; ".join(details))


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateFragmentError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if not _HASH_RE.fullmatch(normalized):
        raise CandidateFragmentError(f"{name} must be a lowercase SHA-256")
    return normalized


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CandidateFragmentError(f"{name} must be a non-negative integer")
    return value


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CandidateFragmentError("candidate fragment must be canonical JSON") from exc
