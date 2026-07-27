"""Immutable, self-authenticating assets for automatic weighted rule trees.

The weighted-tree kernel owns fitting, topology, replay, and measured metrics.
This module freezes one already-validated kernel result into a development-stage
strategy candidate asset.  It adds task/sample identity, a deterministic leaf
fragment index, derived direction red flags, and two acyclic integrity layers:

* candidate evidence id/hash authenticate every substantive field except the
  evidence and asset identities themselves;
* asset id/hash authenticate that same content plus the candidate evidence.

The full-tree asset is deliberately not a Strategy Pool fragment.  A later,
explicit adapter may persist one leaf-fragment artifact which references this
asset; it must not copy or weaken the tree topology contract here.
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

from marvis.feature.weighted_rule_tree import (
    WeightedRuleTreeError,
    validate_weighted_rule_tree,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError


AUTOMATIC_TREE_ASSET_SCHEMA_VERSION = "strategy.automatic-tree-asset.v1"
AUTOMATIC_TREE_ASSET_TYPE = "automatic_rule_tree"
AUTOMATIC_TREE_ASSET_PRODUCER_VERSION = "strategy.automatic-tree-asset/1"

_CANDIDATE_STAGE = "development"
_OBSERVATION_STAGE = "backtested"
_VALIDATION_STATUS = "unvalidated"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_FRAGMENT_ID_RE = re.compile(r"^candidate-fragment-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "asset_type",
        "lifecycle",
        "identity",
        "tree_result",
        "fragments",
        "diagnostics",
        "candidate_evidence",
        "source_refs",
        "producer_version",
        "asset_id",
        "asset_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"asset_id", "asset_hash"}
_CORE_FIELDS = _BODY_FIELDS - {"candidate_evidence"}
_LIFECYCLE_FIELDS = frozenset(
    {"candidate_stage", "observation_stage", "validation_status"}
)
_IDENTITY_FIELDS = frozenset(
    {
        "task_id",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "registry_metadata_hash",
        "sample_context_hash",
    }
)
_FRAGMENT_FIELDS = frozenset(
    {
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "condition",
        "requirements",
        "effect_id",
        "metrics",
    }
)
_CANDIDATE_EVIDENCE_FIELDS = frozenset({"candidate_id", "evidence_hash"})
_DIAGNOSTIC_FIELDS = frozenset({"direction_violations", "red_flags"})
_DIRECTION_VIOLATION_FIELDS = frozenset(
    {
        "node_id",
        "feature",
        "expected_direction",
        "basis",
        "primary_bad_rate_delta",
    }
)
_RED_FLAG_FIELDS = frozenset({"code", "node_id", "feature", "expected_direction"})


class AutomaticTreeAssetError(StrategyError):
    """An automatic-tree candidate asset failed its exact frozen contract."""


def build_automatic_tree_asset(
    tree_result: Mapping[str, Any],
    *,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    workspace_revision: int,
    workspace_generation: int,
    semantic_mapping_hash: str,
    registry_metadata_hash: str,
    sample_context_hash: str,
    source_refs: Sequence[str],
    producer_version: str = AUTOMATIC_TREE_ASSET_PRODUCER_VERSION,
) -> dict[str, Any]:
    """Freeze one strict weighted-tree result as a complete candidate asset."""

    identity = _normalize_identity(
        {
            "task_id": task_id,
            "dataset_id": dataset_id,
            "dataset_content_hash": dataset_content_hash,
            "workspace_revision": workspace_revision,
            "workspace_generation": workspace_generation,
            "semantic_mapping_hash": semantic_mapping_hash,
            "registry_metadata_hash": registry_metadata_hash,
            "sample_context_hash": sample_context_hash,
        }
    )
    canonical_tree = _validated_tree(tree_result)
    core = _normalize_core(
        {
            "schema_version": AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
            "asset_type": AUTOMATIC_TREE_ASSET_TYPE,
            "lifecycle": {
                "candidate_stage": _CANDIDATE_STAGE,
                "observation_stage": _OBSERVATION_STAGE,
                "validation_status": _VALIDATION_STATUS,
            },
            "identity": identity,
            "tree_result": canonical_tree,
            "fragments": _derive_fragments(canonical_tree, identity=identity),
            "diagnostics": _derive_diagnostics(canonical_tree),
            "source_refs": list(source_refs),
            "producer_version": producer_version,
        }
    )
    candidate_evidence = _derive_candidate_evidence(core)
    body = {**core, "candidate_evidence": candidate_evidence}
    asset_id = _stable_id("candidate-asset", body)
    without_hash = {**body, "asset_id": asset_id}
    asset_hash = _sha256(_canonical_json(without_hash))
    return validate_automatic_tree_asset({**without_hash, "asset_hash": asset_hash})


def validate_automatic_tree_asset(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact asset and return a detached canonical representation."""

    if not isinstance(payload, Mapping):
        raise AutomaticTreeAssetError("automatic tree asset must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "automatic tree asset")
    asset_id = _text(payload["asset_id"], "asset_id")
    if not _ASSET_ID_RE.fullmatch(asset_id):
        raise AutomaticTreeAssetError("asset_id has an invalid format")
    asset_hash = _hash(payload["asset_hash"], "asset_hash")
    body = _normalize_body(
        {key: payload[key] for key in payload if key not in {"asset_id", "asset_hash"}}
    )
    expected_id = _stable_id("candidate-asset", body)
    if not hmac.compare_digest(asset_id, expected_id):
        raise AutomaticTreeAssetError(
            "asset_id does not match canonical automatic tree asset"
        )
    without_hash = {**body, "asset_id": asset_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(asset_hash, expected_hash):
        raise AutomaticTreeAssetError(
            "asset_hash does not match canonical automatic tree asset"
        )
    return {**without_hash, "asset_hash": asset_hash}


def canonical_automatic_tree_asset_json(payload: Mapping[str, Any]) -> str:
    """Return the sole canonical JSON serialization of a verified asset."""

    return _canonical_json(validate_automatic_tree_asset(payload))


def _normalize_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, _BODY_FIELDS, "automatic tree asset body")
    core = _normalize_core(
        {key: payload[key] for key in payload if key != "candidate_evidence"}
    )
    candidate_evidence = _normalize_candidate_evidence(payload["candidate_evidence"])
    expected_evidence = _derive_candidate_evidence(core)
    if candidate_evidence != expected_evidence:
        raise AutomaticTreeAssetError(
            "candidate evidence id/hash does not authenticate the asset content"
        )
    return {**core, "candidate_evidence": candidate_evidence}


def _normalize_core(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, _CORE_FIELDS, "automatic tree asset content")
    if payload["schema_version"] != AUTOMATIC_TREE_ASSET_SCHEMA_VERSION:
        raise AutomaticTreeAssetError(
            f"schema_version must be {AUTOMATIC_TREE_ASSET_SCHEMA_VERSION}"
        )
    if payload["asset_type"] != AUTOMATIC_TREE_ASSET_TYPE:
        raise AutomaticTreeAssetError(f"asset_type must be {AUTOMATIC_TREE_ASSET_TYPE}")
    lifecycle = _normalize_lifecycle(payload["lifecycle"])
    identity = _normalize_identity(payload["identity"])
    tree_result = _validated_tree(payload["tree_result"])
    fragments = _normalize_fragments(
        payload["fragments"], tree_result=tree_result, identity=identity
    )
    diagnostics = _normalize_diagnostics(
        payload["diagnostics"], tree_result=tree_result
    )
    source_refs = _normalize_source_refs(payload["source_refs"])
    producer_version = _text(payload["producer_version"], "producer_version")
    if producer_version != AUTOMATIC_TREE_ASSET_PRODUCER_VERSION:
        raise AutomaticTreeAssetError(
            "producer_version must be " + AUTOMATIC_TREE_ASSET_PRODUCER_VERSION
        )
    return {
        "schema_version": AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
        "asset_type": AUTOMATIC_TREE_ASSET_TYPE,
        "lifecycle": lifecycle,
        "identity": identity,
        "tree_result": tree_result,
        "fragments": fragments,
        "diagnostics": diagnostics,
        "source_refs": source_refs,
        "producer_version": producer_version,
    }


def _normalize_lifecycle(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError("lifecycle must be an object")
    _exact_fields(value, _LIFECYCLE_FIELDS, "lifecycle")
    expected = {
        "candidate_stage": _CANDIDATE_STAGE,
        "observation_stage": _OBSERVATION_STAGE,
        "validation_status": _VALIDATION_STATUS,
    }
    if value != expected:
        raise AutomaticTreeAssetError(
            "automatic tree lifecycle must remain development/backtested/unvalidated"
        )
    return expected


def _normalize_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError("identity must be an object")
    _exact_fields(value, _IDENTITY_FIELDS, "identity")
    return {
        "task_id": _text(value["task_id"], "identity.task_id"),
        "dataset_id": _text(value["dataset_id"], "identity.dataset_id"),
        "dataset_content_hash": _hash(
            value["dataset_content_hash"], "identity.dataset_content_hash"
        ),
        "workspace_revision": _non_negative_int(
            value["workspace_revision"], "identity.workspace_revision"
        ),
        "workspace_generation": _non_negative_int(
            value["workspace_generation"], "identity.workspace_generation"
        ),
        "semantic_mapping_hash": _hash(
            value["semantic_mapping_hash"], "identity.semantic_mapping_hash"
        ),
        "registry_metadata_hash": _hash(
            value["registry_metadata_hash"], "identity.registry_metadata_hash"
        ),
        "sample_context_hash": _hash(
            value["sample_context_hash"], "identity.sample_context_hash"
        ),
    }


def _validated_tree(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError("tree_result must be an object")
    try:
        return validate_weighted_rule_tree(value)
    except WeightedRuleTreeError as exc:
        raise AutomaticTreeAssetError(
            f"tree_result failed strict weighted-tree validation: {exc}"
        ) from exc


def _derive_fragments(
    tree_result: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for rule in tree_result["rules"]:
        effect_id = _stable_id(
            "candidate-effect",
            {
                "schema_version": "strategy.automatic-tree-effect.v1",
                "identity": identity,
                "tree_result_hash": tree_result["result_hash"],
                "leaf_id": rule["leaf_id"],
                "rule_id": rule["rule_id"],
                "metrics": rule["metrics"],
            },
        )
        fragment_content = {
            "leaf_id": rule["leaf_id"],
            "rule_id": rule["rule_id"],
            "condition": rule["condition"],
            "requirements": [],
            "effect_id": effect_id,
            "metrics": rule["metrics"],
        }
        fragment_id = _stable_id(
            "candidate-fragment",
            {
                "schema_version": "strategy.automatic-tree-fragment-index.v1",
                "identity": identity,
                "tree_result_hash": tree_result["result_hash"],
                **fragment_content,
            },
        )
        without_hash = {
            "leaf_id": fragment_content["leaf_id"],
            "fragment_id": fragment_id,
            "rule_id": fragment_content["rule_id"],
            "condition": fragment_content["condition"],
            "requirements": [],
            "effect_id": effect_id,
            "metrics": fragment_content["metrics"],
        }
        fragments.append(
            {
                **without_hash,
                "fragment_hash": _sha256(_canonical_json(without_hash)),
            }
        )
    return fragments


def _normalize_fragments(
    value: object,
    *,
    tree_result: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise AutomaticTreeAssetError("fragments must be an array")
    supplied = [
        _normalize_fragment(item, index=index) for index, item in enumerate(value)
    ]
    expected = _derive_fragments(tree_result, identity=identity)
    if supplied != expected:
        raise AutomaticTreeAssetError(
            "fragments must exactly index tree rules in canonical leaf order"
        )
    return supplied


def _normalize_fragment(value: object, *, index: int) -> dict[str, Any]:
    name = f"fragments[{index}]"
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError(f"{name} must be an object")
    _exact_fields(value, _FRAGMENT_FIELDS, name)
    leaf_id = _text(value["leaf_id"], f"{name}.leaf_id")
    fragment_id = _text(value["fragment_id"], f"{name}.fragment_id")
    if not _FRAGMENT_ID_RE.fullmatch(fragment_id):
        raise AutomaticTreeAssetError(f"{name}.fragment_id has an invalid format")
    fragment_hash = _hash(value["fragment_hash"], f"{name}.fragment_hash")
    rule_id = _text(value["rule_id"], f"{name}.rule_id")
    condition_raw = _json_object(value["condition"], f"{name}.condition")
    try:
        condition = canonicalize_expression(condition_raw)
    except StrategyError as exc:
        raise AutomaticTreeAssetError(f"{name}.condition is invalid: {exc}") from exc
    if condition != condition_raw:
        raise AutomaticTreeAssetError(
            f"{name}.condition must be canonical Strategy DSL"
        )
    if value["requirements"] != []:
        raise AutomaticTreeAssetError(f"{name}.requirements must be empty")
    effect_id = _text(value["effect_id"], f"{name}.effect_id")
    if not _EFFECT_ID_RE.fullmatch(effect_id):
        raise AutomaticTreeAssetError(f"{name}.effect_id has an invalid format")
    metrics = _json_object(value["metrics"], f"{name}.metrics")
    without_hash = {
        "leaf_id": leaf_id,
        "fragment_id": fragment_id,
        "rule_id": rule_id,
        "condition": condition,
        "requirements": [],
        "effect_id": effect_id,
        "metrics": metrics,
    }
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(fragment_hash, expected_hash):
        raise AutomaticTreeAssetError(
            f"{name}.fragment_hash does not authenticate the fragment"
        )
    return {**without_hash, "fragment_hash": fragment_hash}


def _derive_diagnostics(tree_result: Mapping[str, Any]) -> dict[str, Any]:
    violations = []
    for node in tree_result["tree"]["nodes"]:
        if node["kind"] != "split":
            continue
        diagnostic = node["direction_diagnostic"]
        if diagnostic["status"] != "violation":
            continue
        violations.append(
            {
                "node_id": node["node_id"],
                "feature": node["feature"],
                "expected_direction": diagnostic["expected_direction"],
                "basis": diagnostic["basis"],
                "primary_bad_rate_delta": diagnostic["primary_bad_rate_delta"],
            }
        )
    return {
        "direction_violations": violations,
        "red_flags": [
            {
                "code": "direction_violation",
                "node_id": row["node_id"],
                "feature": row["feature"],
                "expected_direction": row["expected_direction"],
            }
            for row in violations
        ],
    }


def _normalize_diagnostics(
    value: object,
    *,
    tree_result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError("diagnostics must be an object")
    _exact_fields(value, _DIAGNOSTIC_FIELDS, "diagnostics")
    violations_raw = value["direction_violations"]
    red_flags_raw = value["red_flags"]
    if isinstance(violations_raw, str | bytes | bytearray) or not isinstance(
        violations_raw, Sequence
    ):
        raise AutomaticTreeAssetError("direction_violations must be an array")
    if isinstance(red_flags_raw, str | bytes | bytearray) or not isinstance(
        red_flags_raw, Sequence
    ):
        raise AutomaticTreeAssetError("red_flags must be an array")
    violations = [
        _normalize_direction_violation(item, index=index)
        for index, item in enumerate(violations_raw)
    ]
    red_flags = [
        _normalize_red_flag(item, index=index)
        for index, item in enumerate(red_flags_raw)
    ]
    supplied = {
        "direction_violations": violations,
        "red_flags": red_flags,
    }
    if supplied != _derive_diagnostics(tree_result):
        raise AutomaticTreeAssetError(
            "direction violations and red flags must be exact tree derivatives"
        )
    return supplied


def _normalize_direction_violation(value: object, *, index: int) -> dict[str, Any]:
    name = f"direction_violations[{index}]"
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError(f"{name} must be an object")
    _exact_fields(value, _DIRECTION_VIOLATION_FIELDS, name)
    direction = _text(value["expected_direction"], f"{name}.expected_direction")
    if direction not in {"increasing", "decreasing"}:
        raise AutomaticTreeAssetError(
            f"{name}.expected_direction must be increasing or decreasing"
        )
    basis = _text(value["basis"], f"{name}.basis")
    if basis not in {"unweighted", "weighted"}:
        raise AutomaticTreeAssetError(f"{name}.basis is invalid")
    return {
        "node_id": _text(value["node_id"], f"{name}.node_id"),
        "feature": _text(value["feature"], f"{name}.feature"),
        "expected_direction": direction,
        "basis": basis,
        "primary_bad_rate_delta": _finite_number(
            value["primary_bad_rate_delta"], f"{name}.primary_bad_rate_delta"
        ),
    }


def _normalize_red_flag(value: object, *, index: int) -> dict[str, str]:
    name = f"red_flags[{index}]"
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError(f"{name} must be an object")
    _exact_fields(value, _RED_FLAG_FIELDS, name)
    if value["code"] != "direction_violation":
        raise AutomaticTreeAssetError(f"{name}.code is invalid")
    direction = _text(value["expected_direction"], f"{name}.expected_direction")
    if direction not in {"increasing", "decreasing"}:
        raise AutomaticTreeAssetError(
            f"{name}.expected_direction must be increasing or decreasing"
        )
    return {
        "code": "direction_violation",
        "node_id": _text(value["node_id"], f"{name}.node_id"),
        "feature": _text(value["feature"], f"{name}.feature"),
        "expected_direction": direction,
    }


def _derive_candidate_evidence(core: Mapping[str, Any]) -> dict[str, str]:
    candidate_id = _stable_id("candidate", core)
    evidence_hash = _sha256(_canonical_json({**core, "candidate_id": candidate_id}))
    return {"candidate_id": candidate_id, "evidence_hash": evidence_hash}


def _normalize_candidate_evidence(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeAssetError("candidate_evidence must be an object")
    _exact_fields(value, _CANDIDATE_EVIDENCE_FIELDS, "candidate_evidence")
    candidate_id = _text(value["candidate_id"], "candidate_evidence.candidate_id")
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise AutomaticTreeAssetError("candidate_evidence.candidate_id is invalid")
    return {
        "candidate_id": candidate_id,
        "evidence_hash": _hash(
            value["evidence_hash"], "candidate_evidence.evidence_hash"
        ),
    }


def _normalize_source_refs(value: object) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise AutomaticTreeAssetError("source_refs must be an array")
    refs = [_text(item, f"source_refs[{index}]") for index, item in enumerate(value)]
    if not refs:
        raise AutomaticTreeAssetError("source_refs must not be empty")
    if len(refs) != len(set(refs)):
        raise AutomaticTreeAssetError("source_refs must not contain duplicates")
    return sorted(refs)


def _json_object(value: object, name: str) -> dict[str, Any]:
    normalized = _json_value(value, name)
    if not isinstance(normalized, dict):
        raise AutomaticTreeAssetError(f"{name} must be an object")
    return normalized


def _json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise AutomaticTreeAssetError(f"{name} must contain finite JSON")
        return normalized
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AutomaticTreeAssetError(f"{name} keys must be strings")
        return {
            key: _json_value(child, f"{name}.{key}") for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _json_value(child, f"{name}[{index}]") for index, child in enumerate(value)
        ]
    raise AutomaticTreeAssetError(f"{name} must contain canonical JSON values")


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise AutomaticTreeAssetError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise AutomaticTreeAssetError(f"{name} has " + "; ".join(details))


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AutomaticTreeAssetError(f"{name} must be non-empty canonical text")
    return value


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise AutomaticTreeAssetError(f"{name} must be a lowercase SHA-256")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutomaticTreeAssetError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AutomaticTreeAssetError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AutomaticTreeAssetError(f"{name} must be a finite number")
    return normalized


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
        raise AutomaticTreeAssetError(
            "automatic tree asset must be finite canonical JSON"
        ) from exc


__all__ = [
    "AUTOMATIC_TREE_ASSET_PRODUCER_VERSION",
    "AUTOMATIC_TREE_ASSET_SCHEMA_VERSION",
    "AUTOMATIC_TREE_ASSET_TYPE",
    "AutomaticTreeAssetError",
    "build_automatic_tree_asset",
    "canonical_automatic_tree_asset_json",
    "validate_automatic_tree_asset",
]
