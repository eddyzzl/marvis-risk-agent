"""Immutable candidate materialized from one exact Cross rule search pointer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import re
from typing import Any

from marvis.packs.strategy.cross_rule_search import (
    validate_cross_rule_search_result,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError


CROSS_RULE_CANDIDATE_SCHEMA_VERSION = (
    "strategy.cross-rule-candidate.v1"
)
CROSS_RULE_CANDIDATE_PRODUCER_VERSION = (
    "strategy.cross-rule-candidate/1"
)
CROSS_RULE_CANDIDATE_ASSET_TYPE = "cross_threshold_rule"

_FIELDS = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "effect_stage",
        "validation_status",
        "source_selection",
        "dimension",
        "feature_bindings",
        "condition",
        "metrics",
        "effect_id",
        "lifecycle",
        "producer_version",
        "selection_reason",
        "selection_audit_hash",
        "asset_hash",
    }
)
_SELECTION_FIELDS = frozenset(
    {
        "search_artifact_id",
        "search_artifact_content_hash",
        "search_id",
        "search_content_hash",
        "rule_id",
        "rule_rank",
        "eligible",
        "constraint_failures",
    }
)
_SEARCH_REF_FIELDS = frozenset(
    {"artifact_id", "artifact_content_hash"}
)
_LIFECYCLE = {
    "admitted": False,
    "applied": False,
    "adopted": False,
    "deployed": False,
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^cross-rule-asset-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^cross-rule-effect-[0-9a-f]{32}$")
_SEARCH_ID_RE = re.compile(r"^cross-rule-search-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^cross-rule-[0-9a-f]{32}$")


class CrossRuleCandidateError(StrategyError):
    """An exact Cross rule pointer could not become a candidate."""


def build_cross_rule_candidate(
    search_result: Mapping[str, Any],
    *,
    search_artifact_ref: Mapping[str, Any],
    rule_id: str,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Materialize one exact evaluated rule without admitting it to a Pool."""

    search = validate_cross_rule_search_result(search_result)
    reference = _search_ref(search_artifact_ref)
    requested_rule = _id(rule_id, "rule_id", _RULE_ID_RE)
    matches = [
        item
        for item in search["rules"]
        if hmac.compare_digest(item["rule_id"], requested_rule)
    ]
    if len(matches) != 1:
        raise CrossRuleCandidateError(
            "rule_id is not an authenticated evaluated Cross rule"
        )
    rule = matches[0]
    configured = {
        item["feature"]: item
        for item in search["configuration"]["features"]
    }
    feature_bindings = [
        configured[condition["feature"]]
        for condition in rule["conditions"]
    ]
    condition = _rule_expression(
        rule["conditions"],
        feature_bindings=feature_bindings,
    )
    selection = {
        "search_artifact_id": reference["artifact_id"],
        "search_artifact_content_hash": reference[
            "artifact_content_hash"
        ],
        "search_id": search["search_id"],
        "search_content_hash": search["content_hash"],
        "rule_id": rule["rule_id"],
        "rule_rank": rule["rank"],
        "eligible": rule["eligible"],
        "constraint_failures": rule["constraint_failures"],
    }
    effect_id = _stable_id(
        "cross-rule-effect",
        {
            "source_selection": selection,
            "condition": condition,
            "metrics": rule["metrics"],
        },
    )
    semantic_body = {
        "schema_version": CROSS_RULE_CANDIDATE_SCHEMA_VERSION,
        "asset_type": CROSS_RULE_CANDIDATE_ASSET_TYPE,
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "source_selection": selection,
        "dimension": search["configuration"]["dimension"],
        "feature_bindings": feature_bindings,
        "condition": condition,
        "metrics": rule["metrics"],
        "effect_id": effect_id,
        "lifecycle": dict(_LIFECYCLE),
        "producer_version": CROSS_RULE_CANDIDATE_PRODUCER_VERSION,
    }
    asset_id = _stable_id("cross-rule-asset", semantic_body)
    asset_hash = _sha256(
        _canonical_json({**semantic_body, "asset_id": asset_id})
    )
    reason = _optional_text(selection_reason, "selection_reason")
    audit_hash = _selection_audit_hash(
        asset_id=asset_id,
        source_selection=selection,
        selection_reason=reason,
    )
    return validate_cross_rule_candidate(
        {
            **semantic_body,
            "asset_id": asset_id,
            "selection_reason": reason,
            "selection_audit_hash": audit_hash,
            "asset_hash": asset_hash,
        }
    )


def validate_cross_rule_candidate(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the candidate's canonical semantic and audit identities."""

    obj = _object(payload, "Cross rule candidate")
    if set(obj) != _FIELDS:
        raise CrossRuleCandidateError(
            "Cross rule candidate fields are invalid"
        )
    if obj["schema_version"] != CROSS_RULE_CANDIDATE_SCHEMA_VERSION:
        raise CrossRuleCandidateError(
            "Cross rule candidate schema_version is invalid"
        )
    if obj["asset_type"] != CROSS_RULE_CANDIDATE_ASSET_TYPE:
        raise CrossRuleCandidateError(
            "Cross rule candidate asset_type is invalid"
        )
    if (
        obj["effect_stage"] != "development"
        or obj["validation_status"] != "unvalidated"
    ):
        raise CrossRuleCandidateError(
            "Cross rule candidate cannot claim validation"
        )
    asset_id = _id(obj["asset_id"], "asset_id", _ASSET_ID_RE)
    selection = _selection(obj["source_selection"])
    dimension = _integer(
        obj["dimension"],
        "dimension",
        minimum=2,
        maximum=3,
    )
    raw_bindings = _array(obj["feature_bindings"], "feature_bindings")
    if len(raw_bindings) != dimension:
        raise CrossRuleCandidateError(
            "feature_bindings must contain dimension values"
        )
    feature_bindings = [
        json.loads(_canonical_json(item)) for item in raw_bindings
    ]
    if any(not isinstance(item, dict) for item in feature_bindings):
        raise CrossRuleCandidateError(
            "feature_bindings must contain objects"
        )
    feature_names = [item.get("feature") for item in feature_bindings]
    if (
        any(not isinstance(item, str) or not item for item in feature_names)
        or len(set(feature_names)) != dimension
        or feature_names != sorted(feature_names)
    ):
        raise CrossRuleCandidateError(
            "feature_bindings are not canonical"
        )
    try:
        condition = canonicalize_expression(
            _object(obj["condition"], "condition")
        )
    except StrategyError as exc:
        raise CrossRuleCandidateError(
            "Cross rule candidate condition is invalid"
        ) from exc
    if _canonical_json(condition) != _canonical_json(obj["condition"]):
        raise CrossRuleCandidateError(
            "Cross rule candidate condition is not canonical"
        )
    metrics = _object(obj["metrics"], "metrics")
    effect_id = _id(obj["effect_id"], "effect_id", _EFFECT_ID_RE)
    lifecycle = _object(obj["lifecycle"], "lifecycle")
    if lifecycle != _LIFECYCLE:
        raise CrossRuleCandidateError(
            "Cross rule candidate lifecycle changed"
        )
    if obj["producer_version"] != CROSS_RULE_CANDIDATE_PRODUCER_VERSION:
        raise CrossRuleCandidateError(
            "Cross rule candidate producer_version is invalid"
        )
    semantic_body = {
        "schema_version": CROSS_RULE_CANDIDATE_SCHEMA_VERSION,
        "asset_type": CROSS_RULE_CANDIDATE_ASSET_TYPE,
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "source_selection": selection,
        "dimension": dimension,
        "feature_bindings": feature_bindings,
        "condition": condition,
        "metrics": metrics,
        "effect_id": effect_id,
        "lifecycle": dict(_LIFECYCLE),
        "producer_version": CROSS_RULE_CANDIDATE_PRODUCER_VERSION,
    }
    expected_id = _stable_id("cross-rule-asset", semantic_body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise CrossRuleCandidateError(
            "asset_id does not match Cross rule semantics"
        )
    asset_hash = _hash(obj["asset_hash"], "asset_hash")
    expected_hash = _sha256(
        _canonical_json({**semantic_body, "asset_id": asset_id})
    )
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise CrossRuleCandidateError(
            "asset_hash does not match Cross rule candidate"
        )
    reason = _optional_text(obj["selection_reason"], "selection_reason")
    audit_hash = _hash(
        obj["selection_audit_hash"],
        "selection_audit_hash",
    )
    expected_audit = _selection_audit_hash(
        asset_id=asset_id,
        source_selection=selection,
        selection_reason=reason,
    )
    if not hmac.compare_digest(audit_hash, expected_audit):
        raise CrossRuleCandidateError(
            "selection_audit_hash does not match selection reason"
        )
    return {
        **semantic_body,
        "asset_id": asset_id,
        "selection_reason": reason,
        "selection_audit_hash": audit_hash,
        "asset_hash": asset_hash,
    }


def canonical_cross_rule_candidate_json(
    payload: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_cross_rule_candidate(payload))


def cross_rule_candidate_to_verified_fragment(
    candidate: Mapping[str, Any],
    *,
    artifact_binding: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a replayed concrete candidate into the generic Pool seam."""

    from marvis.packs.strategy.candidate_fragment import (
        build_verified_candidate_fragment,
    )

    asset = validate_cross_rule_candidate(candidate)
    binding = _object(artifact_binding, "artifact_binding")
    evidence = _object(evidence_identity, "evidence_identity")
    return build_verified_candidate_fragment(
        artifact=binding,
        asset={
            "schema_version": asset["schema_version"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "asset_type": asset["asset_type"],
        },
        fragment_type="cross_threshold_rule",
        rule_id=asset["source_selection"]["rule_id"],
        condition=asset["condition"],
        requirements=[],
        effect_id=asset["effect_id"],
        evidence_id=asset["source_selection"]["search_id"],
        evidence_hash=asset["source_selection"]["search_content_hash"],
        evidence_identity=evidence,
    )


def _rule_expression(
    conditions: Sequence[Mapping[str, Any]],
    *,
    feature_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = {
        item["feature"]: item for item in feature_bindings
    }
    expressions = []
    for descriptor in conditions:
        feature = descriptor["feature"]
        binding = bindings[feature]
        base_parts = [
            {
                "op": "compare",
                "field": feature,
                "operator": (
                    ">=" if descriptor["operator"] == "gte" else "<"
                ),
                "value": descriptor["threshold"],
                "missing": "no_match",
            }
        ]
        excluded = binding.get("excluded_values")
        if excluded:
            base_parts.append(
                {
                    "op": "compare",
                    "field": feature,
                    "operator": "not_in",
                    "value": excluded,
                    "missing": "no_match",
                }
            )
        base = (
            base_parts[0]
            if len(base_parts) == 1
            else {"op": "and", "args": base_parts}
        )
        if descriptor["include_missing"]:
            base = {
                "op": "or",
                "args": [
                    base,
                    {"op": "is_null", "field": feature},
                ],
            }
        expressions.append(base)
    return canonicalize_expression({"op": "and", "args": expressions})


def _search_ref(value: object) -> dict[str, str]:
    obj = _object(value, "search_artifact_ref")
    if set(obj) != _SEARCH_REF_FIELDS:
        raise CrossRuleCandidateError(
            "search_artifact_ref fields are invalid"
        )
    return {
        "artifact_id": _hash(
            obj["artifact_id"],
            "search_artifact_ref.artifact_id",
        ),
        "artifact_content_hash": _hash(
            obj["artifact_content_hash"],
            "search_artifact_ref.artifact_content_hash",
        ),
    }


def _selection(value: object) -> dict[str, Any]:
    obj = _object(value, "source_selection")
    if set(obj) != _SELECTION_FIELDS:
        raise CrossRuleCandidateError(
            "source_selection fields are invalid"
        )
    failures = [
        _text(item, f"constraint_failures[{index}]")
        for index, item in enumerate(
            _array(
                obj["constraint_failures"],
                "constraint_failures",
                allow_empty=True,
            )
        )
    ]
    eligible = obj["eligible"]
    if not isinstance(eligible, bool) or eligible is not (not failures):
        raise CrossRuleCandidateError(
            "source_selection eligible is inconsistent"
        )
    return {
        "search_artifact_id": _hash(
            obj["search_artifact_id"],
            "source_selection.search_artifact_id",
        ),
        "search_artifact_content_hash": _hash(
            obj["search_artifact_content_hash"],
            "source_selection.search_artifact_content_hash",
        ),
        "search_id": _id(
            obj["search_id"],
            "source_selection.search_id",
            _SEARCH_ID_RE,
        ),
        "search_content_hash": _hash(
            obj["search_content_hash"],
            "source_selection.search_content_hash",
        ),
        "rule_id": _id(
            obj["rule_id"],
            "source_selection.rule_id",
            _RULE_ID_RE,
        ),
        "rule_rank": _integer(
            obj["rule_rank"],
            "source_selection.rule_rank",
            minimum=1,
        ),
        "eligible": eligible,
        "constraint_failures": failures,
    }


def _selection_audit_hash(
    *,
    asset_id: str,
    source_selection: Mapping[str, Any],
    selection_reason: str | None,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "asset_id": asset_id,
                "source_selection": source_selection,
                "selection_reason": selection_reason,
            }
        )
    )


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise CrossRuleCandidateError(f"{name} must be an object")
    return dict(value)


def _array(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> list[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (not allow_empty and not value)
    ):
        raise CrossRuleCandidateError(f"{name} must be an array")
    return list(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise CrossRuleCandidateError(f"{name} must be non-empty text")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CrossRuleCandidateError(f"{name} must be an integer")
    if value < minimum or (
        maximum is not None and value > maximum
    ):
        raise CrossRuleCandidateError(f"{name} is outside its range")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise CrossRuleCandidateError(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return value


def _id(value: object, name: str, pattern: re.Pattern[str]) -> str:
    text = _text(value, name)
    if pattern.fullmatch(text) is None:
        raise CrossRuleCandidateError(f"{name} has an invalid format")
    return text


def _stable_id(prefix: str, value: object) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CrossRuleCandidateError(
            "Cross rule candidate is not canonical JSON"
        ) from exc


__all__ = [
    "CROSS_RULE_CANDIDATE_ASSET_TYPE",
    "CROSS_RULE_CANDIDATE_PRODUCER_VERSION",
    "CROSS_RULE_CANDIDATE_SCHEMA_VERSION",
    "CrossRuleCandidateError",
    "build_cross_rule_candidate",
    "canonical_cross_rule_candidate_json",
    "cross_rule_candidate_to_verified_fragment",
    "validate_cross_rule_candidate",
]
