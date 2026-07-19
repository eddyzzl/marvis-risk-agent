"""Explicit immutable leaf selections for full automatic-tree assets.

An automatic tree is one indivisible measured asset.  This module persists only
an explicit pointer to one of its already-indexed leaves; it never copies tree
topology, executable conditions, requirements, measured metrics, or a business
action.  The pure projection seam replays the pointer against the full tree
asset and derives every executable or evidentiary fact from that asset alone.

This module deliberately has no database or filesystem authority.  Registry
liveness, task ownership, canonical paths, provenance, and persisted bytes must
be independently verified by the calling Tool for both the selection artifact
and its referenced full-tree artifact before this pure seam is called.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import re
from typing import Any, TypedDict
import unicodedata

from marvis.packs.strategy.automatic_tree_asset import (
    AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
    AutomaticTreeAssetError,
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError


AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION = "strategy.automatic-tree-leaf-fragment.v1"
AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION = (
    "strategy.automatic-tree-leaf-fragment/1"
)

AUTOMATIC_TREE_ASSET_ARTIFACT_KIND = "strategy_automatic_tree_asset_json"
AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION = (
    "strategy.automatic-tree-asset-artifact.v1"
)
AUTOMATIC_TREE_ASSET_ORIGIN_TOOL = "strategy.build_automatic_tree_candidate"

# Source-prefixed aliases make the direction of the leaf-selection reference
# unambiguous to callers while retaining asset-oriented names for Tool code.
AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND = AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION = (
    AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION
)
AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL = AUTOMATIC_TREE_ASSET_ORIGIN_TOOL

AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND = (
    "strategy_automatic_tree_leaf_fragment_json"
)
AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION = (
    "strategy.automatic-tree-leaf-fragment-artifact.v1"
)
AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL = (
    "strategy.materialize_automatic_tree_leaf_fragment"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_FRAGMENT_ID_RE = re.compile(r"^candidate-fragment-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(r"^automatic-tree-leaf-selection-[0-9a-f]{32}$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "tree_artifact",
        "tree_asset",
        "leaf",
        "selection_reason",
        "producer_version",
        "selection_id",
        "selection_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"selection_id", "selection_hash"}
_TREE_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "artifact_schema_version",
        "content_hash",
        "origin_tool",
        "path",
        "provenance",
    }
)
_TREE_ARTIFACT_BINDING_FIELDS = _TREE_ARTIFACT_FIELDS | {"canonical_bytes"}
_TREE_ARTIFACT_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "asset_id",
        "asset_hash",
        "tree_result_hash",
        "dataset_id",
        "dataset_content_hash",
        "workspace_revision",
        "workspace_generation",
        "semantic_mapping_hash",
        "registry_metadata_hash",
        "sample_context_hash",
    }
)
_TREE_ASSET_FIELDS = frozenset(
    {"schema_version", "asset_id", "asset_hash", "tree_result_hash"}
)
_LEAF_FIELDS = frozenset(
    {"leaf_id", "fragment_id", "fragment_hash", "rule_id", "effect_id"}
)
_SELECTION_ARTIFACT_BINDING_FIELDS = frozenset(
    {
        "artifact_id",
        "task_id",
        "kind",
        "content_hash",
        "origin_tool",
        "artifact_schema_version",
        "producer_version",
        "selection_id",
        "selection_hash",
        "tree_artifact_id",
        "tree_artifact_kind",
        "tree_artifact_schema_version",
        "tree_artifact_content_hash",
        "tree_artifact_origin_tool",
        "tree_artifact_path",
        "tree_artifact_provenance",
        "tree_asset_schema_version",
        "tree_asset_id",
        "tree_asset_hash",
        "tree_result_hash",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    }
)


class AutomaticTreeLeafFragmentError(StrategyError):
    """An automatic-tree leaf selection or its replay failed closed."""


class IndependentlyVerifiedAutomaticTreeArtifactBinding(TypedDict):
    """A caller-verified snapshot of one live full-tree TaskArtifact.

    The type name expresses a required caller precondition, not a capability of
    this module: constructing this mapping does not prove that the row exists in
    the live registry.  ``canonical_bytes`` must be the bytes independently read
    from the already-verified canonical artifact path.
    """

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    path: str
    provenance: dict[str, Any]
    canonical_bytes: bytes


def build_automatic_tree_leaf_fragment(
    full_tree_asset: Mapping[str, Any],
    *,
    tree_artifact_binding: IndependentlyVerifiedAutomaticTreeArtifactBinding,
    leaf_id: str,
    selection_reason: str | None = None,
) -> dict[str, Any]:
    """Build one canonical, explicit leaf-selection audit event.

    ``selection_reason`` is audit-event content: changing it changes both the
    selection id and hash, while the tree/leaf/fragment/rule/effect references
    remain exact derivatives of the same full-tree asset.

    ``tree_artifact_binding`` must come from an independent live-registry,
    canonical-path, provenance, and byte verification performed by the calling
    Tool.  This pure builder only freezes and cross-checks that verified
    snapshot; it does not establish registry liveness itself.
    """

    asset = _full_tree_asset(full_tree_asset)
    verified_tree_binding = _verified_tree_artifact_binding(tree_artifact_binding)
    tree_artifact = _tree_artifact_pointer(verified_tree_binding)
    _require_tree_artifact_binding_matches_asset(
        verified_tree_binding,
        asset=asset,
    )
    normalized_leaf_id = _canonical_text(leaf_id, "leaf_id")
    fragment = _leaf_from_asset(asset, normalized_leaf_id)
    reason = _canonicalize_selection_reason(selection_reason)

    body = _normalize_body(
        {
            "schema_version": AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION,
            "tree_artifact": tree_artifact,
            "tree_asset": _tree_asset_reference_from_asset(asset),
            "leaf": _leaf_reference_from_fragment(fragment),
            "selection_reason": reason,
            "producer_version": AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION,
        }
    )
    selection_id = _stable_id("automatic-tree-leaf-selection", body)
    without_hash = {**body, "selection_id": selection_id}
    selection_hash = _sha256(_canonical_json(without_hash))
    return validate_automatic_tree_leaf_fragment(
        {**without_hash, "selection_hash": selection_hash}
    )


def validate_automatic_tree_leaf_fragment(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact self-authenticating leaf-selection payload."""

    if not isinstance(payload, Mapping):
        raise AutomaticTreeLeafFragmentError("leaf selection must be an object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "leaf selection")
    selection_id = _canonical_text(payload["selection_id"], "selection_id")
    if _SELECTION_ID_RE.fullmatch(selection_id) is None:
        raise AutomaticTreeLeafFragmentError("selection_id has an invalid format")
    selection_hash = _hash(payload["selection_hash"], "selection_hash")
    body = _normalize_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"selection_id", "selection_hash"}
        }
    )
    expected_id = _stable_id("automatic-tree-leaf-selection", body)
    if not hmac.compare_digest(selection_id, expected_id):
        raise AutomaticTreeLeafFragmentError(
            "selection_id does not match canonical leaf selection"
        )
    without_hash = {**body, "selection_id": selection_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(selection_hash, expected_hash):
        raise AutomaticTreeLeafFragmentError(
            "selection_hash does not match canonical leaf selection"
        )
    return {**without_hash, "selection_hash": selection_hash}


def canonical_automatic_tree_leaf_fragment_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON representation of a valid selection."""

    return _canonical_json(validate_automatic_tree_leaf_fragment(payload))


def automatic_tree_leaf_fragment_content_hash(
    payload: Mapping[str, Any],
) -> str:
    """Hash the canonical persisted bytes of a valid selection artifact."""

    return _sha256(canonical_automatic_tree_leaf_fragment_json(payload))


def automatic_tree_leaf_fragment_to_verified_candidate_fragment(
    selection_payload: Mapping[str, Any],
    full_tree_asset: Mapping[str, Any],
    *,
    selection_artifact_binding: Mapping[str, Any],
    tree_artifact_binding: IndependentlyVerifiedAutomaticTreeArtifactBinding,
) -> dict[str, Any]:
    """Purely replay one independently verified selection and tree artifact.

    This function performs no registry lookup and therefore cannot prove that
    either TaskArtifact currently exists.  Before calling it, the Tool must
    independently load and verify the selection TaskArtifact *and* referenced
    full-tree TaskArtifact from the live registry, including task ownership,
    kind/schema/origin, canonical path, provenance, content SHA-256, and exact
    canonical persisted bytes.  ``selection_artifact_binding`` and
    ``tree_artifact_binding`` are snapshots from that external verification.

    The function then fails closed unless those snapshots, the persisted
    pointer, and the supplied full-tree asset agree exactly.  It does not admit
    the result to Strategy Pool or perform any other stateful action.
    """

    selection = validate_automatic_tree_leaf_fragment(selection_payload)
    asset = _full_tree_asset(full_tree_asset)
    binding = _selection_artifact_binding(selection_artifact_binding)
    verified_tree_binding = _verified_tree_artifact_binding(tree_artifact_binding)
    _require_selection_binding_matches_payload(binding, selection=selection)
    _require_tree_binding_matches_selection_pointer(
        verified_tree_binding,
        selection=selection,
    )
    fragment = _replay_selection(selection, asset=asset)
    _require_tree_artifact_binding_matches_asset(
        verified_tree_binding,
        asset=asset,
    )

    identity = asset["identity"]
    evidence = asset["candidate_evidence"]
    lifecycle = asset["lifecycle"]
    try:
        return build_verified_candidate_fragment(
            artifact={
                "artifact_id": binding["artifact_id"],
                "artifact_kind": binding["kind"],
                "artifact_schema_version": binding["artifact_schema_version"],
                "artifact_content_hash": binding["content_hash"],
                "origin_tool": binding["origin_tool"],
            },
            asset={
                "schema_version": asset["schema_version"],
                "asset_id": asset["asset_id"],
                "asset_hash": asset["asset_hash"],
                "asset_type": asset["asset_type"],
            },
            fragment_id=fragment["fragment_id"],
            fragment_type="strategy_rule",
            rule_id=fragment["rule_id"],
            condition=fragment["condition"],
            requirements=fragment["requirements"],
            effect_id=fragment["effect_id"],
            evidence_id=evidence["candidate_id"],
            evidence_hash=evidence["evidence_hash"],
            evidence_identity={
                "dataset_id": identity["dataset_id"],
                "dataset_content_hash": identity["dataset_content_hash"],
                "workspace_revision": identity["workspace_revision"],
                "workspace_generation": identity["workspace_generation"],
                "semantic_mapping_hash": identity["semantic_mapping_hash"],
                "sample_context_hash": identity["sample_context_hash"],
            },
            candidate_stage=lifecycle["candidate_stage"],
            observation_stage=lifecycle["observation_stage"],
            validation_status=lifecycle["validation_status"],
        )
    except CandidateFragmentError as exc:
        raise AutomaticTreeLeafFragmentError(
            "automatic-tree leaf failed generic fragment projection"
        ) from exc


def _normalize_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(payload, _BODY_FIELDS, "leaf selection body")
    if payload["schema_version"] != AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION:
        raise AutomaticTreeLeafFragmentError(
            "schema_version must be " + AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION
        )
    producer = _canonical_text(payload["producer_version"], "producer_version")
    if producer != AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION:
        raise AutomaticTreeLeafFragmentError(
            "producer_version must be " + AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION
        )
    return {
        "schema_version": AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION,
        "tree_artifact": _tree_artifact(payload["tree_artifact"]),
        "tree_asset": _tree_asset_reference(payload["tree_asset"]),
        "leaf": _leaf_reference(payload["leaf"]),
        "selection_reason": _canonical_selection_reason(payload["selection_reason"]),
        "producer_version": producer,
    }


def _full_tree_asset(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError("full tree asset must be an object")
    try:
        asset = validate_automatic_tree_asset(value)
    except AutomaticTreeAssetError as exc:
        raise AutomaticTreeLeafFragmentError(
            "full tree asset failed strict validation"
        ) from exc
    if asset["schema_version"] != AUTOMATIC_TREE_ASSET_SCHEMA_VERSION:
        raise AutomaticTreeLeafFragmentError(
            "full tree asset must use the committed automatic-tree schema"
        )
    return asset


def _tree_artifact(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError("tree_artifact must be an object")
    _exact_fields(value, _TREE_ARTIFACT_FIELDS, "tree_artifact")
    kind = _canonical_text(value["kind"], "tree_artifact.kind")
    if kind != AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND:
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact.kind must be " + AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND
        )
    schema = _canonical_text(
        value["artifact_schema_version"],
        "tree_artifact.artifact_schema_version",
    )
    if schema != AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION:
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact artifact schema must be "
            + AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION
        )
    origin = _canonical_text(value["origin_tool"], "tree_artifact.origin_tool")
    if origin != AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL:
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact.origin_tool must be "
            + AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL
        )
    return {
        "artifact_id": _canonical_text(
            value["artifact_id"], "tree_artifact.artifact_id"
        ),
        "task_id": _canonical_text(value["task_id"], "tree_artifact.task_id"),
        "kind": kind,
        "artifact_schema_version": schema,
        "content_hash": _hash(value["content_hash"], "tree_artifact.content_hash"),
        "origin_tool": origin,
        "path": _canonical_text(value["path"], "tree_artifact.path"),
        "provenance": _tree_artifact_provenance(
            value["provenance"],
        ),
    }


def _verified_tree_artifact_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError("tree_artifact_binding must be an object")
    _exact_fields(
        value,
        _TREE_ARTIFACT_BINDING_FIELDS,
        "tree_artifact_binding",
    )
    pointer = _tree_artifact({field: value[field] for field in _TREE_ARTIFACT_FIELDS})
    canonical_bytes = value["canonical_bytes"]
    if not isinstance(canonical_bytes, bytes):
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact_binding.canonical_bytes must be bytes"
        )
    return {**pointer, "canonical_bytes": canonical_bytes}


def _tree_artifact_provenance(value: object) -> dict[str, Any]:
    name = "tree_artifact.provenance"
    normalized = _canonical_json_object(value, name)
    _exact_fields(normalized, _TREE_ARTIFACT_PROVENANCE_FIELDS, name)
    schema = _canonical_text(normalized["schema_version"], f"{name}.schema_version")
    if schema != AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION:
        raise AutomaticTreeLeafFragmentError(
            f"{name}.schema_version must be "
            + AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION
        )
    kind = _canonical_text(normalized["kind"], f"{name}.kind")
    if kind != AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND:
        raise AutomaticTreeLeafFragmentError(
            f"{name}.kind must be {AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND}"
        )
    artifact_format = _canonical_text(normalized["format"], f"{name}.format")
    if artifact_format != "json":
        raise AutomaticTreeLeafFragmentError(f"{name}.format must be json")
    return {
        "schema_version": schema,
        "producer_version": _canonical_text(
            normalized["producer_version"],
            f"{name}.producer_version",
        ),
        "task_id": _canonical_text(normalized["task_id"], f"{name}.task_id"),
        "kind": kind,
        "format": artifact_format,
        "asset_id": _canonical_text(normalized["asset_id"], f"{name}.asset_id"),
        "asset_hash": _hash(normalized["asset_hash"], f"{name}.asset_hash"),
        "tree_result_hash": _hash(
            normalized["tree_result_hash"],
            f"{name}.tree_result_hash",
        ),
        "dataset_id": _canonical_text(
            normalized["dataset_id"],
            f"{name}.dataset_id",
        ),
        "dataset_content_hash": _hash(
            normalized["dataset_content_hash"],
            f"{name}.dataset_content_hash",
        ),
        "workspace_revision": _non_negative_int(
            normalized["workspace_revision"],
            f"{name}.workspace_revision",
        ),
        "workspace_generation": _non_negative_int(
            normalized["workspace_generation"],
            f"{name}.workspace_generation",
        ),
        "semantic_mapping_hash": _hash(
            normalized["semantic_mapping_hash"],
            f"{name}.semantic_mapping_hash",
        ),
        "registry_metadata_hash": _hash(
            normalized["registry_metadata_hash"],
            f"{name}.registry_metadata_hash",
        ),
        "sample_context_hash": _hash(
            normalized["sample_context_hash"],
            f"{name}.sample_context_hash",
        ),
    }


def _tree_artifact_pointer(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return _tree_artifact({field: binding[field] for field in _TREE_ARTIFACT_FIELDS})


def _tree_asset_reference(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError("tree_asset must be an object")
    _exact_fields(value, _TREE_ASSET_FIELDS, "tree_asset")
    if value["schema_version"] != AUTOMATIC_TREE_ASSET_SCHEMA_VERSION:
        raise AutomaticTreeLeafFragmentError(
            "tree_asset.schema_version must be " + AUTOMATIC_TREE_ASSET_SCHEMA_VERSION
        )
    asset_id = _canonical_text(value["asset_id"], "tree_asset.asset_id")
    if _ASSET_ID_RE.fullmatch(asset_id) is None:
        raise AutomaticTreeLeafFragmentError(
            "tree_asset.asset_id has an invalid format"
        )
    return {
        "schema_version": AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
        "asset_id": asset_id,
        "asset_hash": _hash(value["asset_hash"], "tree_asset.asset_hash"),
        "tree_result_hash": _hash(
            value["tree_result_hash"], "tree_asset.tree_result_hash"
        ),
    }


def _tree_asset_reference_from_asset(
    asset: Mapping[str, Any],
) -> dict[str, str]:
    return _tree_asset_reference(
        {
            "schema_version": asset["schema_version"],
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "tree_result_hash": asset["tree_result"]["result_hash"],
        }
    )


def _leaf_reference(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError("leaf must be an object")
    _exact_fields(value, _LEAF_FIELDS, "leaf")
    fragment_id = _canonical_text(value["fragment_id"], "leaf.fragment_id")
    if _FRAGMENT_ID_RE.fullmatch(fragment_id) is None:
        raise AutomaticTreeLeafFragmentError("leaf.fragment_id has an invalid format")
    effect_id = _canonical_text(value["effect_id"], "leaf.effect_id")
    if _EFFECT_ID_RE.fullmatch(effect_id) is None:
        raise AutomaticTreeLeafFragmentError("leaf.effect_id has an invalid format")
    return {
        "leaf_id": _canonical_text(value["leaf_id"], "leaf.leaf_id"),
        "fragment_id": fragment_id,
        "fragment_hash": _hash(value["fragment_hash"], "leaf.fragment_hash"),
        "rule_id": _canonical_text(value["rule_id"], "leaf.rule_id"),
        "effect_id": effect_id,
    }


def _leaf_reference_from_fragment(
    fragment: Mapping[str, Any],
) -> dict[str, str]:
    return _leaf_reference(
        {
            "leaf_id": fragment["leaf_id"],
            "fragment_id": fragment["fragment_id"],
            "fragment_hash": fragment["fragment_hash"],
            "rule_id": fragment["rule_id"],
            "effect_id": fragment["effect_id"],
        }
    )


def _selection_artifact_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError(
            "selection_artifact_binding must be an object"
        )
    _exact_fields(
        value,
        _SELECTION_ARTIFACT_BINDING_FIELDS,
        "selection_artifact_binding",
    )
    normalized = {
        "artifact_id": _canonical_text(value["artifact_id"], "selection artifact_id"),
        "task_id": _canonical_text(value["task_id"], "selection task_id"),
        "kind": _canonical_text(value["kind"], "selection kind"),
        "content_hash": _hash(value["content_hash"], "selection content_hash"),
        "origin_tool": _canonical_text(value["origin_tool"], "selection origin_tool"),
        "artifact_schema_version": _canonical_text(
            value["artifact_schema_version"],
            "selection artifact_schema_version",
        ),
        "producer_version": _canonical_text(
            value["producer_version"], "selection producer_version"
        ),
        "selection_id": _canonical_text(
            value["selection_id"], "selection binding selection_id"
        ),
        "selection_hash": _hash(
            value["selection_hash"], "selection binding selection_hash"
        ),
        "tree_artifact_id": _canonical_text(
            value["tree_artifact_id"], "selection tree_artifact_id"
        ),
        "tree_artifact_kind": _canonical_text(
            value["tree_artifact_kind"], "selection tree_artifact_kind"
        ),
        "tree_artifact_schema_version": _canonical_text(
            value["tree_artifact_schema_version"],
            "selection tree_artifact_schema_version",
        ),
        "tree_artifact_content_hash": _hash(
            value["tree_artifact_content_hash"],
            "selection tree_artifact_content_hash",
        ),
        "tree_artifact_origin_tool": _canonical_text(
            value["tree_artifact_origin_tool"],
            "selection tree_artifact_origin_tool",
        ),
        "tree_artifact_path": _canonical_text(
            value["tree_artifact_path"],
            "selection tree_artifact_path",
        ),
        "tree_artifact_provenance": _canonical_json_object(
            value["tree_artifact_provenance"],
            "selection tree_artifact_provenance",
        ),
        "tree_asset_schema_version": _canonical_text(
            value["tree_asset_schema_version"],
            "selection tree_asset_schema_version",
        ),
        "tree_asset_id": _canonical_text(
            value["tree_asset_id"], "selection tree_asset_id"
        ),
        "tree_asset_hash": _hash(value["tree_asset_hash"], "selection tree_asset_hash"),
        "tree_result_hash": _hash(
            value["tree_result_hash"], "selection tree_result_hash"
        ),
        "leaf_id": _canonical_text(value["leaf_id"], "selection leaf_id"),
        "fragment_id": _canonical_text(value["fragment_id"], "selection fragment_id"),
        "fragment_hash": _hash(value["fragment_hash"], "selection fragment_hash"),
        "rule_id": _canonical_text(value["rule_id"], "selection rule_id"),
        "effect_id": _canonical_text(value["effect_id"], "selection effect_id"),
    }
    fixed = {
        "kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "origin_tool": AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
        "artifact_schema_version": (
            AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION,
        "tree_artifact_kind": AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        "tree_artifact_schema_version": (AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION),
        "tree_artifact_origin_tool": AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        "tree_asset_schema_version": AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
    }
    for field, expected in fixed.items():
        if normalized[field] != expected:
            raise AutomaticTreeLeafFragmentError(
                f"selection {field} must be {expected}"
            )
    if _SELECTION_ID_RE.fullmatch(normalized["selection_id"]) is None:
        raise AutomaticTreeLeafFragmentError(
            "selection binding selection_id has an invalid format"
        )
    if _ASSET_ID_RE.fullmatch(normalized["tree_asset_id"]) is None:
        raise AutomaticTreeLeafFragmentError(
            "selection tree_asset_id has an invalid format"
        )
    if _FRAGMENT_ID_RE.fullmatch(normalized["fragment_id"]) is None:
        raise AutomaticTreeLeafFragmentError(
            "selection fragment_id has an invalid format"
        )
    if _EFFECT_ID_RE.fullmatch(normalized["effect_id"]) is None:
        raise AutomaticTreeLeafFragmentError(
            "selection effect_id has an invalid format"
        )
    return normalized


def _require_tree_artifact_matches_asset(
    tree_artifact: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> None:
    if tree_artifact["task_id"] != asset["identity"]["task_id"]:
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact.task_id does not match full tree asset"
        )
    expected_content_hash = _sha256(canonical_automatic_tree_asset_json(asset))
    if not hmac.compare_digest(tree_artifact["content_hash"], expected_content_hash):
        raise AutomaticTreeLeafFragmentError(
            "tree_artifact.content_hash does not match canonical full tree asset"
        )


def _require_tree_binding_matches_selection_pointer(
    binding: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> None:
    actual_pointer = _tree_artifact_pointer(binding)
    expected_pointer = selection["tree_artifact"]
    for field in sorted(_TREE_ARTIFACT_FIELDS):
        actual = actual_pointer[field]
        expected = expected_pointer[field]
        if field.endswith("hash"):
            matches = hmac.compare_digest(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise AutomaticTreeLeafFragmentError(
                f"tree artifact binding {field} does not match selection pointer"
            )


def _require_tree_artifact_binding_matches_asset(
    binding: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> None:
    pointer = _tree_artifact_pointer(binding)
    _require_tree_artifact_matches_asset(pointer, asset=asset)
    expected_provenance = _tree_artifact_provenance_from_asset(asset)
    if pointer["provenance"] != expected_provenance:
        raise AutomaticTreeLeafFragmentError(
            "tree artifact provenance does not match canonical full tree asset"
        )
    expected_bytes = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    if not hmac.compare_digest(binding["canonical_bytes"], expected_bytes):
        raise AutomaticTreeLeafFragmentError(
            "tree artifact canonical bytes do not match canonical full tree asset"
        )
    actual_content_hash = _sha256_bytes(binding["canonical_bytes"])
    if not hmac.compare_digest(actual_content_hash, pointer["content_hash"]):
        raise AutomaticTreeLeafFragmentError(
            "tree artifact canonical bytes do not match content_hash"
        )


def _tree_artifact_provenance_from_asset(
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    identity = asset["identity"]
    return _tree_artifact_provenance(
        {
            "schema_version": AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
            "producer_version": asset["producer_version"],
            "task_id": identity["task_id"],
            "kind": AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
            "format": "json",
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "tree_result_hash": asset["tree_result"]["result_hash"],
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "registry_metadata_hash": identity["registry_metadata_hash"],
            "sample_context_hash": identity["sample_context_hash"],
        }
    )


def _require_selection_binding_matches_payload(
    binding: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
) -> None:
    tree_artifact = selection["tree_artifact"]
    tree_asset = selection["tree_asset"]
    leaf = selection["leaf"]
    expected = {
        "task_id": tree_artifact["task_id"],
        "content_hash": automatic_tree_leaf_fragment_content_hash(selection),
        "producer_version": selection["producer_version"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "tree_artifact_id": tree_artifact["artifact_id"],
        "tree_artifact_kind": tree_artifact["kind"],
        "tree_artifact_schema_version": tree_artifact["artifact_schema_version"],
        "tree_artifact_content_hash": tree_artifact["content_hash"],
        "tree_artifact_origin_tool": tree_artifact["origin_tool"],
        "tree_artifact_path": tree_artifact["path"],
        "tree_artifact_provenance": tree_artifact["provenance"],
        "tree_asset_schema_version": tree_asset["schema_version"],
        "tree_asset_id": tree_asset["asset_id"],
        "tree_asset_hash": tree_asset["asset_hash"],
        "tree_result_hash": tree_asset["tree_result_hash"],
        "leaf_id": leaf["leaf_id"],
        "fragment_id": leaf["fragment_id"],
        "fragment_hash": leaf["fragment_hash"],
        "rule_id": leaf["rule_id"],
        "effect_id": leaf["effect_id"],
    }
    for field, expected_value in expected.items():
        actual = binding[field]
        if field.endswith("hash"):
            matches = hmac.compare_digest(actual, expected_value)
        else:
            matches = actual == expected_value
        if not matches:
            raise AutomaticTreeLeafFragmentError(
                f"selection artifact binding {field} does not match payload"
            )


def _replay_selection(
    selection: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    tree_artifact = selection["tree_artifact"]
    _require_tree_artifact_matches_asset(tree_artifact, asset=asset)
    expected_asset = _tree_asset_reference_from_asset(asset)
    for field, expected in expected_asset.items():
        actual = selection["tree_asset"][field]
        if field.endswith("hash"):
            matches = hmac.compare_digest(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise AutomaticTreeLeafFragmentError(
                f"tree asset {field} does not match leaf selection"
            )

    leaf_reference = selection["leaf"]
    fragment = _leaf_from_asset(asset, leaf_reference["leaf_id"])
    expected_leaf = _leaf_reference_from_fragment(fragment)
    for field, expected in expected_leaf.items():
        actual = leaf_reference[field]
        if field.endswith("hash"):
            matches = hmac.compare_digest(actual, expected)
        else:
            matches = actual == expected
        if not matches:
            raise AutomaticTreeLeafFragmentError(
                f"leaf {field} does not match full tree asset"
            )
    return fragment


def _leaf_from_asset(asset: Mapping[str, Any], leaf_id: str) -> dict[str, Any]:
    matches = [
        fragment for fragment in asset["fragments"] if fragment["leaf_id"] == leaf_id
    ]
    if len(matches) != 1:
        raise AutomaticTreeLeafFragmentError(
            f"leaf_id is not an explicit known full-tree leaf: {leaf_id}"
        )
    return matches[0]


def _canonicalize_selection_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AutomaticTreeLeafFragmentError(
            "selection_reason must be a string or null"
        )
    if "\x00" in value:
        raise AutomaticTreeLeafFragmentError("selection_reason must not contain NUL")
    normalized = unicodedata.normalize("NFC", value)
    canonical = " ".join(normalized.split())
    if not canonical:
        raise AutomaticTreeLeafFragmentError(
            "selection_reason must be non-empty when provided"
        )
    return canonical


def _canonical_selection_reason(value: object) -> str | None:
    canonical = _canonicalize_selection_reason(value)
    if value is not None and value != canonical:
        raise AutomaticTreeLeafFragmentError(
            "selection_reason must use canonical whitespace and Unicode"
        )
    return canonical


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(str(field) for field in actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise AutomaticTreeLeafFragmentError(f"{name} has " + "; ".join(details))


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AutomaticTreeLeafFragmentError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise AutomaticTreeLeafFragmentError(f"{name} must not contain NUL")
    canonical = unicodedata.normalize("NFC", value)
    if value != canonical or value != value.strip():
        raise AutomaticTreeLeafFragmentError(f"{name} must be canonical text")
    return value


def _hash(value: object, name: str) -> str:
    normalized = _canonical_text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise AutomaticTreeLeafFragmentError(f"{name} must be a lowercase SHA-256")
    return normalized


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutomaticTreeLeafFragmentError(f"{name} must be a non-negative integer")
    return value


def _canonical_json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeLeafFragmentError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AutomaticTreeLeafFragmentError(f"{name} keys must be strings")
    try:
        canonical = _canonical_json(value)
    except AutomaticTreeLeafFragmentError as exc:
        raise AutomaticTreeLeafFragmentError(
            f"{name} must contain finite JSON"
        ) from exc
    normalized = json.loads(canonical)
    if not isinstance(normalized, dict):
        raise AutomaticTreeLeafFragmentError(f"{name} must be an object")
    return normalized


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{_sha256(_canonical_json(value))[:32]}"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise AutomaticTreeLeafFragmentError(
            "leaf selection must contain canonical JSON"
        ) from exc


__all__ = [
    "AUTOMATIC_TREE_ASSET_ARTIFACT_KIND",
    "AUTOMATIC_TREE_ASSET_ARTIFACT_SCHEMA_VERSION",
    "AUTOMATIC_TREE_ASSET_ORIGIN_TOOL",
    "AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND",
    "AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION",
    "AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL",
    "AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION",
    "AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION",
    "AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND",
    "AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL",
    "AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION",
    "AutomaticTreeLeafFragmentError",
    "IndependentlyVerifiedAutomaticTreeArtifactBinding",
    "automatic_tree_leaf_fragment_content_hash",
    "automatic_tree_leaf_fragment_to_verified_candidate_fragment",
    "build_automatic_tree_leaf_fragment",
    "canonical_automatic_tree_leaf_fragment_json",
    "validate_automatic_tree_leaf_fragment",
]
