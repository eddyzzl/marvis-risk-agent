"""Pointer-only singleton selections from an interactive-tree frontier.

An interactive-tree revision owns its semantic frontier, executable conditions,
measured effects, and sample identity.  This module persists only one explicit
frontier-node pointer plus the exact authenticated revision TaskArtifact
binding.  It never copies a condition, metric bundle, lifecycle claim, or
business action.

The module has no database or filesystem authority.  Callers must independently
verify task ownership, registry liveness, canonical paths, provenance, and
persisted bytes before constructing the artifact binding accepted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import re
from typing import Any, TypedDict
import unicodedata

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_ASSET_TYPE,
    INTERACTIVE_TREE_REVISION_PRODUCER_VERSION,
    INTERACTIVE_TREE_REVISION_SCHEMA_VERSION,
    INTERACTIVE_TREE_REVISION_V2_PRODUCER_VERSION,
    INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
    canonical_interactive_tree_revision_json,
    validate_interactive_tree_revision,
)
from marvis.packs.strategy.sample_design_execution import (
    StrategyRiskDevelopmentRef,
)


INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION = (
    "strategy.interactive-tree-frontier-selection.v1"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION = (
    "strategy.interactive-tree-frontier-selection/1"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION_V2 = (
    "strategy.interactive-tree-frontier-selection.v2"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2 = (
    "strategy.interactive-tree-frontier-selection/2"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND = (
    "strategy_interactive_tree_frontier_selection_json"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.interactive-tree-frontier-selection-artifact.v1"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2 = (
    "strategy.interactive-tree-frontier-selection-artifact.v2"
)
INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL = (
    "strategy.materialize_interactive_tree_frontier_selection"
)

_REVISION_ARTIFACT_KIND = "strategy_interactive_tree_revision_json"
_REVISION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.interactive-tree-revision-artifact.v1"
)
_REVISION_ARTIFACT_SCHEMA_VERSION_V2 = (
    "strategy.interactive-tree-revision-artifact.v2"
)
_REVISION_ARTIFACT_ORIGIN_TOOL = "strategy.revise_interactive_tree"
_MAX_SELECTION_REASON_LENGTH = 500

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_REVISION_ID_RE = re.compile(r"^interactive-tree-revision-[0-9a-f]{32}$")
_SEMANTIC_TREE_ID_RE = re.compile(r"^interactive-tree-[0-9a-f]{32}$")
_NODE_ID_RE = re.compile(r"^node-[0-9a-f]{20}$")
_SOURCE_NODE_ID_RE = re.compile(r"^(?:node|leaf)-[0-9a-f]{20}$")
_LEAF_ID_RE = re.compile(r"^interactive-leaf-[0-9a-f]{32}$")
_FRAGMENT_ID_RE = re.compile(r"^candidate-fragment-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")
_SELECTION_ID_RE = re.compile(
    r"^interactive-tree-frontier-selection-[0-9a-f]{32}$"
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "revision_artifact",
        "revision",
        "frontier",
        "selection_reason",
        "selection_id",
        "selection_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"selection_id", "selection_hash"}
_REVISION_ARTIFACT_FIELDS = frozenset(
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
_REVISION_ARTIFACT_BINDING_FIELDS = _REVISION_ARTIFACT_FIELDS | {
    "canonical_bytes"
}
_REVISION_FIELDS = frozenset(
    {
        "schema_version",
        "revision_id",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "asset_type",
    }
)
_FRONTIER_FIELDS = frozenset(
    {
        "source_node_id",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    }
)
_REVISION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "revision_id",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "base_asset_id",
        "base_asset_hash",
        "base_tree_result_hash",
        "parent_revision_id",
        "source_tree_id",
        "edit_operation",
        "edit_node_id",
        "sample_design_ref",
    }
)
_SELECTION_ARTIFACT_FIELDS = frozenset(
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
_SELECTION_ARTIFACT_BINDING_FIELDS = _SELECTION_ARTIFACT_FIELDS | {
    "canonical_bytes"
}
_SELECTION_PROVENANCE_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "task_id",
        "kind",
        "format",
        "selection_id",
        "selection_hash",
        "revision_artifact_id",
        "revision_artifact_kind",
        "revision_artifact_schema_version",
        "revision_artifact_content_hash",
        "revision_artifact_origin_tool",
        "revision_artifact_path",
        "revision_artifact_provenance",
        "revision_schema_version",
        "revision_id",
        "revision_hash",
        "semantic_tree_id",
        "tree_hash",
        "asset_type",
        "source_node_id",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
    }
)


class InteractiveTreeFrontierSelectionError(StrategyError):
    """An interactive-tree frontier selection or replay failed closed."""


class IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding(TypedDict):
    """Caller-verified live TaskArtifact facts for one exact revision."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    path: str
    provenance: dict[str, Any]
    canonical_bytes: bytes


class IndependentlyVerifiedInteractiveTreeFrontierSelectionArtifactBinding(
    TypedDict
):
    """Caller-verified live TaskArtifact facts for one exact selection."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    path: str
    provenance: dict[str, Any]
    canonical_bytes: bytes


def build_interactive_tree_frontier_selection(
    revision_payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    revision_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding
    ),
    source_node_id: str,
    selection_reason: str | None = None,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one canonical audit artifact pointing at one frontier fragment."""

    asset = validate_automatic_tree_asset(automatic_tree_asset)
    ancestors = _ancestor_sequence(ancestor_revisions)
    revision = validate_interactive_tree_revision(
        revision_payload,
        asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestors,
    )
    binding = _verified_revision_artifact_binding(
        revision_artifact_binding,
        revision=revision,
        asset=asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestors,
    )
    node_id = _identifier(
        source_node_id,
        "source_node_id",
        pattern=_SOURCE_NODE_ID_RE,
    )
    matches = [
        fragment
        for fragment in revision["fragments"]
        if fragment["source_node_id"] == node_id
    ]
    if len(matches) != 1 or node_id not in revision["tree"]["frontier_node_ids"]:
        raise InteractiveTreeFrontierSelectionError(
            "source_node_id must identify exactly one current revision frontier node"
        )
    reason = _canonicalize_selection_reason(selection_reason)
    is_v2 = (
        revision["schema_version"]
        == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
    )
    body = _normalize_body(
        {
            "schema_version": (
                INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION_V2
                if is_v2
                else INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION
            ),
            "producer_version": (
                INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2
                if is_v2
                else INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
            ),
            "revision_artifact": _revision_artifact_pointer(
                {
                    field: binding[field]
                    for field in _REVISION_ARTIFACT_FIELDS
                }
            ),
            "revision": _revision_reference(revision),
            "frontier": _frontier_reference(matches[0]),
            "selection_reason": reason,
        }
    )
    selection_id = _stable_id("interactive-tree-frontier-selection", body)
    without_hash = {**body, "selection_id": selection_id}
    return validate_interactive_tree_frontier_selection(
        {
            **without_hash,
            "selection_hash": _sha256(_canonical_json(without_hash)),
        }
    )


def validate_interactive_tree_frontier_selection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact self-authenticating singleton frontier pointer."""

    if not isinstance(payload, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "interactive-tree frontier selection must be an object"
        )
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "interactive-tree frontier selection")
    selection_id = _identifier(
        payload["selection_id"],
        "selection_id",
        pattern=_SELECTION_ID_RE,
    )
    selection_hash = _hash(payload["selection_hash"], "selection_hash")
    body = _normalize_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"selection_id", "selection_hash"}
        }
    )
    expected_id = _stable_id("interactive-tree-frontier-selection", body)
    if not hmac.compare_digest(selection_id, expected_id):
        raise InteractiveTreeFrontierSelectionError(
            "selection_id does not match canonical frontier selection"
        )
    without_hash = {**body, "selection_id": selection_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(selection_hash, expected_hash):
        raise InteractiveTreeFrontierSelectionError(
            "selection_hash does not match canonical frontier selection"
        )
    return {**without_hash, "selection_hash": selection_hash}


def canonical_interactive_tree_frontier_selection_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON representation of a valid selection."""

    return _canonical_json(validate_interactive_tree_frontier_selection(payload))


def interactive_tree_frontier_selection_to_verified_candidate_fragment(
    selection_payload: Mapping[str, Any],
    revision_payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    selection_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeFrontierSelectionArtifactBinding
    ),
    revision_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding
    ),
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Replay one verified singleton selection into the generic Pool seam.

    The caller must first establish live registry ownership, canonical paths,
    and persisted bytes for both TaskArtifacts.  This pure adapter then proves
    that those independently verified snapshots, the selection pointers, and
    the supplied revision lineage all identify one exact current frontier
    fragment.  It does not admit the fragment to Strategy Pool.
    """

    selection = validate_interactive_tree_frontier_selection(selection_payload)
    asset = validate_automatic_tree_asset(automatic_tree_asset)
    ancestors = _ancestor_sequence(ancestor_revisions)
    revision = validate_interactive_tree_revision(
        revision_payload,
        asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestors,
    )
    verified_revision_binding = _verified_revision_artifact_binding(
        revision_artifact_binding,
        revision=revision,
        asset=asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestors,
    )
    verified_selection_binding = _verified_selection_artifact_binding(
        selection_artifact_binding,
        selection=selection,
    )
    if (
        selection["revision_artifact"]
        != _revision_artifact_pointer(
            {
                field: verified_revision_binding[field]
                for field in _REVISION_ARTIFACT_FIELDS
            }
        )
    ):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact binding does not match selection pointer"
        )
    if selection["revision"] != _revision_reference(revision):
        raise InteractiveTreeFrontierSelectionError(
            "selection revision reference does not match the live revision"
        )
    fragment = _replay_frontier_selection(selection, revision=revision)

    identity = revision["identity"]
    evidence = revision["candidate_evidence"]
    lifecycle = revision["lifecycle"]
    try:
        return build_verified_candidate_fragment(
            artifact={
                "artifact_id": verified_selection_binding["artifact_id"],
                "artifact_kind": verified_selection_binding["kind"],
                "artifact_schema_version": verified_selection_binding[
                    "artifact_schema_version"
                ],
                "artifact_content_hash": verified_selection_binding[
                    "content_hash"
                ],
                "origin_tool": verified_selection_binding["origin_tool"],
            },
            asset={
                "schema_version": revision["schema_version"],
                "asset_id": revision["semantic_tree_id"],
                "asset_hash": revision["tree"]["tree_hash"],
                "asset_type": revision["asset_type"],
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
        raise InteractiveTreeFrontierSelectionError(
            "interactive-tree frontier failed generic fragment projection"
        ) from exc


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(value, _BODY_FIELDS, "interactive-tree frontier selection body")
    schema = value["schema_version"]
    if schema == INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION:
        producer = INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
    elif schema == INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION_V2:
        producer = INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2
    else:
        raise InteractiveTreeFrontierSelectionError(
            "interactive-tree frontier selection schema_version is invalid"
        )
    if value["producer_version"] != producer:
        raise InteractiveTreeFrontierSelectionError(
            "interactive-tree frontier selection producer_version is invalid"
        )
    revision_artifact = _revision_artifact_pointer(value["revision_artifact"])
    revision = _normalize_revision_reference(value["revision"])
    frontier = _normalize_frontier_reference(value["frontier"])
    reason = _selection_reason(value["selection_reason"])
    _require_pointer_cross_references(
        revision_artifact=revision_artifact,
        revision=revision,
    )
    expected_schema = (
        INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION_V2
        if revision["schema_version"] == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
        else INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION
    )
    if schema != expected_schema:
        raise InteractiveTreeFrontierSelectionError(
            "selection schema does not match revision schema"
        )
    return {
        "schema_version": schema,
        "producer_version": producer,
        "revision_artifact": revision_artifact,
        "revision": revision,
        "frontier": frontier,
        "selection_reason": reason,
    }


def _verified_revision_artifact_binding(
    value: object,
    *,
    revision: Mapping[str, Any],
    asset: Mapping[str, Any],
    parent_revision: Mapping[str, Any] | None,
    ancestor_revisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "revision_artifact_binding must be an object"
        )
    _exact_fields(
        value,
        _REVISION_ARTIFACT_BINDING_FIELDS,
        "revision_artifact_binding",
    )
    pointer = _revision_artifact_pointer(
        {field: value[field] for field in _REVISION_ARTIFACT_FIELDS}
    )
    canonical_bytes = value["canonical_bytes"]
    if not isinstance(canonical_bytes, bytes):
        raise InteractiveTreeFrontierSelectionError(
            "revision_artifact_binding canonical_bytes must be bytes"
        )
    expected_bytes = canonical_interactive_tree_revision_json(
        revision,
        asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestor_revisions,
    ).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, expected_bytes):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact canonical bytes do not match the revision"
        )
    if not hmac.compare_digest(
        pointer["content_hash"],
        hashlib.sha256(canonical_bytes).hexdigest(),
    ):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact content hash does not match canonical bytes"
        )
    expected_provenance = _expected_revision_provenance(
        revision,
        asset=asset,
    )
    if pointer["provenance"] != expected_provenance:
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact provenance does not match the revision"
        )
    if pointer["task_id"] != revision["identity"]["task_id"]:
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact task_id does not match the revision"
        )
    return {**pointer, "canonical_bytes": canonical_bytes}


def _verified_selection_artifact_binding(
    value: object,
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "selection_artifact_binding must be an object"
        )
    _exact_fields(
        value,
        _SELECTION_ARTIFACT_BINDING_FIELDS,
        "selection_artifact_binding",
    )
    pointer = _selection_artifact_pointer(value)
    canonical_bytes = value["canonical_bytes"]
    if not isinstance(canonical_bytes, bytes):
        raise InteractiveTreeFrontierSelectionError(
            "selection_artifact_binding canonical_bytes must be bytes"
        )
    expected_bytes = canonical_interactive_tree_frontier_selection_json(
        selection
    ).encode("utf-8")
    if not hmac.compare_digest(canonical_bytes, expected_bytes):
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact canonical bytes do not match the selection"
        )
    if not hmac.compare_digest(
        pointer["content_hash"],
        hashlib.sha256(canonical_bytes).hexdigest(),
    ):
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact content hash does not match canonical bytes"
        )
    expected_provenance = _expected_selection_provenance(selection)
    if (
        pointer["artifact_schema_version"]
        != expected_provenance["schema_version"]
        or pointer["provenance"] != expected_provenance
    ):
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact provenance does not match the selection"
        )
    if pointer["task_id"] != selection["revision_artifact"]["task_id"]:
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact task_id does not match the revision artifact"
        )
    return {**pointer, "canonical_bytes": canonical_bytes}


def _selection_artifact_pointer(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "selection_artifact must be an object"
        )
    actual = set(value) - {"canonical_bytes"}
    _exact_field_sets(
        actual,
        _SELECTION_ARTIFACT_FIELDS,
        "selection_artifact",
    )
    kind = _canonical_text(value["kind"], "selection_artifact.kind")
    schema = _canonical_text(
        value["artifact_schema_version"],
        "selection_artifact.artifact_schema_version",
    )
    origin = _canonical_text(
        value["origin_tool"],
        "selection_artifact.origin_tool",
    )
    if kind != INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND:
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact kind is invalid"
        )
    if schema not in {
        INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION,
        INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2,
    }:
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact schema version is invalid"
        )
    if origin != INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL:
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact origin tool is invalid"
        )
    return {
        "artifact_id": _canonical_text(
            value["artifact_id"],
            "selection_artifact.artifact_id",
        ),
        "task_id": _canonical_text(
            value["task_id"],
            "selection_artifact.task_id",
        ),
        "kind": kind,
        "artifact_schema_version": schema,
        "content_hash": _hash(
            value["content_hash"],
            "selection_artifact.content_hash",
        ),
        "origin_tool": origin,
        "path": _canonical_text(value["path"], "selection_artifact.path"),
        "provenance": _normalize_selection_provenance(value["provenance"]),
    }


def _normalize_selection_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact provenance must be an object"
        )
    _exact_fields(
        value,
        _SELECTION_PROVENANCE_FIELDS,
        "selection artifact provenance",
    )
    normalized = {
        "schema_version": _canonical_text(
            value["schema_version"],
            "selection provenance schema_version",
        ),
        "producer_version": _canonical_text(
            value["producer_version"],
            "selection provenance producer_version",
        ),
        "task_id": _canonical_text(
            value["task_id"],
            "selection provenance task_id",
        ),
        "kind": _canonical_text(value["kind"], "selection provenance kind"),
        "format": _canonical_text(value["format"], "selection provenance format"),
        "selection_id": _identifier(
            value["selection_id"],
            "selection provenance selection_id",
            pattern=_SELECTION_ID_RE,
        ),
        "selection_hash": _hash(
            value["selection_hash"],
            "selection provenance selection_hash",
        ),
        "revision_artifact_id": _canonical_text(
            value["revision_artifact_id"],
            "selection provenance revision_artifact_id",
        ),
        "revision_artifact_kind": _canonical_text(
            value["revision_artifact_kind"],
            "selection provenance revision_artifact_kind",
        ),
        "revision_artifact_schema_version": _canonical_text(
            value["revision_artifact_schema_version"],
            "selection provenance revision_artifact_schema_version",
        ),
        "revision_artifact_content_hash": _hash(
            value["revision_artifact_content_hash"],
            "selection provenance revision_artifact_content_hash",
        ),
        "revision_artifact_origin_tool": _canonical_text(
            value["revision_artifact_origin_tool"],
            "selection provenance revision_artifact_origin_tool",
        ),
        "revision_artifact_path": _canonical_text(
            value["revision_artifact_path"],
            "selection provenance revision_artifact_path",
        ),
        "revision_artifact_provenance": _normalize_revision_provenance(
            value["revision_artifact_provenance"]
        ),
        "revision_schema_version": _canonical_text(
            value["revision_schema_version"],
            "selection provenance revision_schema_version",
        ),
        "revision_id": _identifier(
            value["revision_id"],
            "selection provenance revision_id",
            pattern=_REVISION_ID_RE,
        ),
        "revision_hash": _hash(
            value["revision_hash"],
            "selection provenance revision_hash",
        ),
        "semantic_tree_id": _identifier(
            value["semantic_tree_id"],
            "selection provenance semantic_tree_id",
            pattern=_SEMANTIC_TREE_ID_RE,
        ),
        "tree_hash": _hash(
            value["tree_hash"],
            "selection provenance tree_hash",
        ),
        "asset_type": _canonical_text(
            value["asset_type"],
            "selection provenance asset_type",
        ),
        "source_node_id": _identifier(
            value["source_node_id"],
            "selection provenance source_node_id",
            pattern=_SOURCE_NODE_ID_RE,
        ),
        "leaf_id": _identifier(
            value["leaf_id"],
            "selection provenance leaf_id",
            pattern=_LEAF_ID_RE,
        ),
        "fragment_id": _identifier(
            value["fragment_id"],
            "selection provenance fragment_id",
            pattern=_FRAGMENT_ID_RE,
        ),
        "fragment_hash": _hash(
            value["fragment_hash"],
            "selection provenance fragment_hash",
        ),
        "rule_id": _identifier(
            value["rule_id"],
            "selection provenance rule_id",
            pattern=_RULE_ID_RE,
        ),
        "effect_id": _identifier(
            value["effect_id"],
            "selection provenance effect_id",
            pattern=_EFFECT_ID_RE,
        ),
    }
    revision_schema = normalized["revision_schema_version"]
    if revision_schema == INTERACTIVE_TREE_REVISION_SCHEMA_VERSION:
        selection_schema = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        )
        selection_producer = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
        )
        revision_artifact_schema = _REVISION_ARTIFACT_SCHEMA_VERSION
    elif revision_schema == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION:
        selection_schema = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2
        )
        selection_producer = (
            INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION_V2
        )
        revision_artifact_schema = _REVISION_ARTIFACT_SCHEMA_VERSION_V2
    else:
        raise InteractiveTreeFrontierSelectionError(
            "selection artifact provenance revision schema is invalid"
        )
    constants = {
        "schema_version": selection_schema,
        "producer_version": selection_producer,
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "revision_artifact_kind": _REVISION_ARTIFACT_KIND,
        "revision_artifact_schema_version": revision_artifact_schema,
        "revision_artifact_origin_tool": _REVISION_ARTIFACT_ORIGIN_TOOL,
        "asset_type": INTERACTIVE_TREE_ASSET_TYPE,
    }
    for field, expected in constants.items():
        if normalized[field] != expected:
            raise InteractiveTreeFrontierSelectionError(
                f"selection artifact provenance {field} is invalid"
            )
    return normalized


def _expected_selection_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    revision_artifact = selection["revision_artifact"]
    revision = selection["revision"]
    frontier = selection["frontier"]
    return _normalize_selection_provenance(
        {
            "schema_version": (
                INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION_V2
                if revision["schema_version"]
                == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
                else INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
            ),
            "producer_version": selection["producer_version"],
            "task_id": revision_artifact["task_id"],
            "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
            "format": "json",
            "selection_id": selection["selection_id"],
            "selection_hash": selection["selection_hash"],
            "revision_artifact_id": revision_artifact["artifact_id"],
            "revision_artifact_kind": revision_artifact["kind"],
            "revision_artifact_schema_version": revision_artifact[
                "artifact_schema_version"
            ],
            "revision_artifact_content_hash": revision_artifact["content_hash"],
            "revision_artifact_origin_tool": revision_artifact["origin_tool"],
            "revision_artifact_path": revision_artifact["path"],
            "revision_artifact_provenance": revision_artifact["provenance"],
            "revision_schema_version": revision["schema_version"],
            "revision_id": revision["revision_id"],
            "revision_hash": revision["revision_hash"],
            "semantic_tree_id": revision["semantic_tree_id"],
            "tree_hash": revision["tree_hash"],
            "asset_type": revision["asset_type"],
            "source_node_id": frontier["source_node_id"],
            "leaf_id": frontier["leaf_id"],
            "fragment_id": frontier["fragment_id"],
            "fragment_hash": frontier["fragment_hash"],
            "rule_id": frontier["rule_id"],
            "effect_id": frontier["effect_id"],
        }
    )


def _replay_frontier_selection(
    selection: Mapping[str, Any],
    *,
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    source_node_id = selection["frontier"]["source_node_id"]
    matches = [
        fragment
        for fragment in revision["fragments"]
        if fragment["source_node_id"] == source_node_id
    ]
    if (
        len(matches) != 1
        or source_node_id not in revision["tree"]["frontier_node_ids"]
    ):
        raise InteractiveTreeFrontierSelectionError(
            "selection no longer identifies one current revision frontier node"
        )
    if selection["frontier"] != _frontier_reference(matches[0]):
        raise InteractiveTreeFrontierSelectionError(
            "selection frontier pointer does not match the live revision fragment"
        )
    return matches[0]


def _revision_artifact_pointer(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "revision_artifact must be an object"
        )
    _exact_fields(value, _REVISION_ARTIFACT_FIELDS, "revision_artifact")
    kind = _canonical_text(value["kind"], "revision_artifact.kind")
    schema = _canonical_text(
        value["artifact_schema_version"],
        "revision_artifact.artifact_schema_version",
    )
    origin = _canonical_text(
        value["origin_tool"],
        "revision_artifact.origin_tool",
    )
    if kind != _REVISION_ARTIFACT_KIND:
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact kind is invalid"
        )
    if schema not in {
        _REVISION_ARTIFACT_SCHEMA_VERSION,
        _REVISION_ARTIFACT_SCHEMA_VERSION_V2,
    }:
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact schema version is invalid"
        )
    if origin != _REVISION_ARTIFACT_ORIGIN_TOOL:
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact origin tool is invalid"
        )
    return {
        "artifact_id": _canonical_text(
            value["artifact_id"],
            "revision_artifact.artifact_id",
        ),
        "task_id": _canonical_text(
            value["task_id"],
            "revision_artifact.task_id",
        ),
        "kind": kind,
        "artifact_schema_version": schema,
        "content_hash": _hash(
            value["content_hash"],
            "revision_artifact.content_hash",
        ),
        "origin_tool": origin,
        "path": _canonical_text(value["path"], "revision_artifact.path"),
        "provenance": _normalize_revision_provenance(
            value["provenance"],
        ),
    }


def _normalize_revision_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact provenance must be an object"
        )
    _exact_fields(
        value,
        _REVISION_PROVENANCE_FIELDS,
        "revision artifact provenance",
    )
    parent_revision_id = value["parent_revision_id"]
    if parent_revision_id is not None:
        parent_revision_id = _identifier(
            parent_revision_id,
            "revision provenance parent_revision_id",
            pattern=_REVISION_ID_RE,
        )
    source_tree_id = _canonical_text(
        value["source_tree_id"],
        "revision provenance source_tree_id",
    )
    if (
        _ASSET_ID_RE.fullmatch(source_tree_id) is None
        and _REVISION_ID_RE.fullmatch(source_tree_id) is None
    ):
        raise InteractiveTreeFrontierSelectionError(
            "revision provenance source_tree_id has an invalid format"
        )
    try:
        sample_ref = StrategyRiskDevelopmentRef.from_value(
            value["sample_design_ref"]
        ).to_ref_dict()
        if sample_ref["partition"] not in {
            "development",
            "risk/development",
        }:
            raise StrategyError(
                "revision provenance sample_design_ref partition is invalid"
            )
    except StrategyError as exc:
        raise InteractiveTreeFrontierSelectionError(
            "revision provenance sample_design_ref is invalid"
        ) from exc
    normalized = {
        "schema_version": _canonical_text(
            value["schema_version"],
            "revision provenance schema_version",
        ),
        "producer_version": _canonical_text(
            value["producer_version"],
            "revision provenance producer_version",
        ),
        "task_id": _canonical_text(
            value["task_id"],
            "revision provenance task_id",
        ),
        "kind": _canonical_text(value["kind"], "revision provenance kind"),
        "format": _canonical_text(value["format"], "revision provenance format"),
        "revision_id": _identifier(
            value["revision_id"],
            "revision provenance revision_id",
            pattern=_REVISION_ID_RE,
        ),
        "revision_hash": _hash(
            value["revision_hash"],
            "revision provenance revision_hash",
        ),
        "semantic_tree_id": _identifier(
            value["semantic_tree_id"],
            "revision provenance semantic_tree_id",
            pattern=_SEMANTIC_TREE_ID_RE,
        ),
        "tree_hash": _hash(
            value["tree_hash"],
            "revision provenance tree_hash",
        ),
        "base_asset_id": _identifier(
            value["base_asset_id"],
            "revision provenance base_asset_id",
            pattern=_ASSET_ID_RE,
        ),
        "base_asset_hash": _hash(
            value["base_asset_hash"],
            "revision provenance base_asset_hash",
        ),
        "base_tree_result_hash": _hash(
            value["base_tree_result_hash"],
            "revision provenance base_tree_result_hash",
        ),
        "parent_revision_id": parent_revision_id,
        "source_tree_id": source_tree_id,
        "edit_operation": _canonical_text(
            value["edit_operation"],
            "revision provenance edit_operation",
        ),
        "edit_node_id": _identifier(
            value["edit_node_id"],
            "revision provenance edit_node_id",
            pattern=_NODE_ID_RE,
        ),
        "sample_design_ref": sample_ref,
    }
    if normalized["schema_version"] == _REVISION_ARTIFACT_SCHEMA_VERSION:
        producer = INTERACTIVE_TREE_REVISION_PRODUCER_VERSION
        operations = {"prune_subtree"}
    elif normalized["schema_version"] == _REVISION_ARTIFACT_SCHEMA_VERSION_V2:
        producer = INTERACTIVE_TREE_REVISION_V2_PRODUCER_VERSION
        operations = {"prune_subtree", "adjust_split_threshold"}
    else:
        producer = ""
        operations = set()
    if (
        normalized["producer_version"] != producer
        or normalized["kind"] != _REVISION_ARTIFACT_KIND
        or normalized["format"] != "json"
        or normalized["edit_operation"] not in operations
    ):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact provenance constants are invalid"
        )
    return normalized


def _expected_revision_provenance(
    revision: Mapping[str, Any],
    *,
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    parent = revision["parent_revision"]
    parent_revision_id = None if parent is None else parent["revision_id"]
    return _normalize_revision_provenance(
        {
            "schema_version": (
                _REVISION_ARTIFACT_SCHEMA_VERSION_V2
                if revision["schema_version"]
                == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
                else _REVISION_ARTIFACT_SCHEMA_VERSION
            ),
            "producer_version": revision["producer_version"],
            "task_id": revision["identity"]["task_id"],
            "kind": _REVISION_ARTIFACT_KIND,
            "format": "json",
            "revision_id": revision["revision_id"],
            "revision_hash": revision["revision_hash"],
            "semantic_tree_id": revision["semantic_tree_id"],
            "tree_hash": revision["tree"]["tree_hash"],
            "base_asset_id": revision["base_tree"]["asset_id"],
            "base_asset_hash": revision["base_tree"]["asset_hash"],
            "base_tree_result_hash": revision["base_tree"]["tree_result_hash"],
            "parent_revision_id": parent_revision_id,
            "source_tree_id": (
                revision["base_tree"]["asset_id"]
                if parent_revision_id is None
                else parent_revision_id
            ),
            "edit_operation": revision["edit"]["operation"],
            "edit_node_id": revision["edit"]["node_id"],
            "sample_design_ref": (
                sample_design_ref_from_automatic_tree_source_refs(
                    asset["source_refs"]
                )
            ),
        }
    )


def _revision_reference(revision: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_revision_reference(
        {
            "schema_version": revision["schema_version"],
            "revision_id": revision["revision_id"],
            "revision_hash": revision["revision_hash"],
            "semantic_tree_id": revision["semantic_tree_id"],
            "tree_hash": revision["tree"]["tree_hash"],
            "asset_type": revision["asset_type"],
        }
    )


def _normalize_revision_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "revision reference must be an object"
        )
    _exact_fields(value, _REVISION_FIELDS, "revision reference")
    schema = _canonical_text(value["schema_version"], "revision.schema_version")
    asset_type = _canonical_text(value["asset_type"], "revision.asset_type")
    if schema not in {
        INTERACTIVE_TREE_REVISION_SCHEMA_VERSION,
        INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
    }:
        raise InteractiveTreeFrontierSelectionError(
            "revision schema_version is invalid"
        )
    if asset_type != INTERACTIVE_TREE_ASSET_TYPE:
        raise InteractiveTreeFrontierSelectionError(
            "revision asset_type is invalid"
        )
    return {
        "schema_version": schema,
        "revision_id": _identifier(
            value["revision_id"],
            "revision.revision_id",
            pattern=_REVISION_ID_RE,
        ),
        "revision_hash": _hash(
            value["revision_hash"],
            "revision.revision_hash",
        ),
        "semantic_tree_id": _identifier(
            value["semantic_tree_id"],
            "revision.semantic_tree_id",
            pattern=_SEMANTIC_TREE_ID_RE,
        ),
        "tree_hash": _hash(value["tree_hash"], "revision.tree_hash"),
        "asset_type": asset_type,
    }


def _frontier_reference(fragment: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_frontier_reference(
        {key: fragment[key] for key in _FRONTIER_FIELDS}
    )


def _normalize_frontier_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierSelectionError(
            "frontier reference must be an object"
        )
    _exact_fields(value, _FRONTIER_FIELDS, "frontier reference")
    return {
        "source_node_id": _identifier(
            value["source_node_id"],
            "frontier.source_node_id",
            pattern=_SOURCE_NODE_ID_RE,
        ),
        "leaf_id": _identifier(
            value["leaf_id"],
            "frontier.leaf_id",
            pattern=_LEAF_ID_RE,
        ),
        "fragment_id": _identifier(
            value["fragment_id"],
            "frontier.fragment_id",
            pattern=_FRAGMENT_ID_RE,
        ),
        "fragment_hash": _hash(
            value["fragment_hash"],
            "frontier.fragment_hash",
        ),
        "rule_id": _identifier(
            value["rule_id"],
            "frontier.rule_id",
            pattern=_RULE_ID_RE,
        ),
        "effect_id": _identifier(
            value["effect_id"],
            "frontier.effect_id",
            pattern=_EFFECT_ID_RE,
        ),
    }


def _require_pointer_cross_references(
    *,
    revision_artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> None:
    provenance = revision_artifact["provenance"]
    expected_artifact_schema = (
        _REVISION_ARTIFACT_SCHEMA_VERSION_V2
        if revision["schema_version"] == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
        else _REVISION_ARTIFACT_SCHEMA_VERSION
    )
    if (
        revision_artifact["artifact_schema_version"]
        != expected_artifact_schema
        or provenance["schema_version"] != expected_artifact_schema
    ):
        raise InteractiveTreeFrontierSelectionError(
            "revision artifact schema does not match selection revision"
        )
    comparisons = {
        "task_id": revision_artifact["task_id"],
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree_hash"],
    }
    for field, expected in comparisons.items():
        if provenance[field] != expected:
            raise InteractiveTreeFrontierSelectionError(
                f"revision artifact provenance {field} does not match selection"
            )


def _canonicalize_selection_reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise InteractiveTreeFrontierSelectionError(
            "selection_reason must be text or null"
        )
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized or len(normalized) > _MAX_SELECTION_REASON_LENGTH:
        raise InteractiveTreeFrontierSelectionError(
            "selection_reason must be 1 to 500 canonical characters"
        )
    return normalized


def _selection_reason(value: object) -> str | None:
    normalized = _canonicalize_selection_reason(value)
    if value is not None and value != normalized:
        raise InteractiveTreeFrontierSelectionError(
            "selection_reason must use canonical whitespace and Unicode"
        )
    return normalized


def _ancestor_sequence(
    value: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise InteractiveTreeFrontierSelectionError(
            "ancestor_revisions must be a sequence"
        )
    if any(not isinstance(item, Mapping) for item in value):
        raise InteractiveTreeFrontierSelectionError(
            "ancestor_revisions must contain objects"
        )
    return tuple(value)


def _identifier(value: object, name: str, *, pattern: re.Pattern[str]) -> str:
    normalized = _canonical_text(value, name)
    if pattern.fullmatch(normalized) is None:
        raise InteractiveTreeFrontierSelectionError(
            f"{name} has an invalid format"
        )
    return normalized


def _hash(value: object, name: str) -> str:
    return _identifier(value, name, pattern=_HASH_RE)


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise InteractiveTreeFrontierSelectionError(
            f"{name} must be non-empty canonical text"
        )
    normalized = unicodedata.normalize("NFC", value)
    if value != normalized or value != value.strip():
        raise InteractiveTreeFrontierSelectionError(
            f"{name} must be canonical text"
        )
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    _exact_field_sets(set(value), expected, name)


def _exact_field_sets(
    actual: set[str],
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise InteractiveTreeFrontierSelectionError(
            f"{name} has invalid fields (" + "; ".join(details) + ")"
        )


def _stable_id(prefix: str, value: object) -> str:
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
        raise InteractiveTreeFrontierSelectionError(
            "interactive-tree frontier selection is not canonical JSON"
        ) from exc


__all__ = [
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION",
    "INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION",
    "IndependentlyVerifiedInteractiveTreeFrontierSelectionArtifactBinding",
    "IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding",
    "InteractiveTreeFrontierSelectionError",
    "build_interactive_tree_frontier_selection",
    "canonical_interactive_tree_frontier_selection_json",
    "interactive_tree_frontier_selection_to_verified_candidate_fragment",
    "validate_interactive_tree_frontier_selection",
]
