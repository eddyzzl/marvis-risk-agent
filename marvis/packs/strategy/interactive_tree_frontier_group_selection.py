"""Pointer-only explicit OR groups from an interactive-tree frontier.

The persisted document identifies an exact authenticated interactive-tree
revision plus two to fifty current frontier node ids.  It intentionally owns no
copied conditions, metrics, requirements, actions, or lifecycle claims.  Replay
against the live revision is the only path to an executable generic candidate
fragment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import re
from typing import Any, TypedDict

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.candidate_fragment import (
    CandidateFragmentError,
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding,
    _canonicalize_selection_reason,
    _REVISION_ARTIFACT_SCHEMA_VERSION,
    _REVISION_ARTIFACT_SCHEMA_VERSION_V2,
    _revision_artifact_pointer,
    _revision_reference,
    _selection_reason,
    _verified_revision_artifact_binding,
)
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
    validate_interactive_tree_revision,
)


INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION = (
    "strategy.interactive-tree-frontier-group-selection.v1"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION = (
    "strategy.interactive-tree-frontier-group-selection/1"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION_V2 = (
    "strategy.interactive-tree-frontier-group-selection.v2"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION_V2 = (
    "strategy.interactive-tree-frontier-group-selection/2"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND = (
    "strategy_interactive_tree_frontier_group_selection_json"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    "strategy.interactive-tree-frontier-group-selection-artifact.v1"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION_V2 = (
    "strategy.interactive-tree-frontier-group-selection-artifact.v2"
)
INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL = (
    "strategy.materialize_interactive_tree_frontier_group_selection"
)

_MIN_GROUP_MEMBERS = 2
_MAX_GROUP_MEMBERS = 50
_SOURCE_NODE_ID_RE = re.compile(r"^(?:node|leaf)-[0-9a-f]{20}$")
_GROUP_ID_RE = re.compile(
    r"^interactive-tree-frontier-group-[0-9a-f]{32}$"
)
_SELECTION_ID_RE = re.compile(
    r"^interactive-tree-frontier-group-selection-[0-9a-f]{32}$"
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "revision_artifact",
        "revision",
        "source_node_ids",
        "group_id",
        "selection_reason",
        "selection_id",
        "selection_hash",
    }
)
_BODY_FIELDS = _TOP_LEVEL_FIELDS - {"selection_id", "selection_hash"}
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
        "group_id",
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
        "source_node_ids",
        "member_count",
    }
)


class InteractiveTreeFrontierGroupSelectionError(StrategyError):
    """An interactive-tree frontier group pointer or replay failed closed."""


class IndependentlyVerifiedInteractiveTreeFrontierGroupSelectionArtifactBinding(
    TypedDict
):
    """Caller-verified live TaskArtifact facts for one exact group selection."""

    artifact_id: str
    task_id: str
    kind: str
    artifact_schema_version: str
    content_hash: str
    origin_tool: str
    path: str
    provenance: dict[str, Any]
    canonical_bytes: bytes


def build_interactive_tree_frontier_group_selection(
    revision_payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    revision_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding
    ),
    source_node_ids: Sequence[str],
    selection_reason: str | None = None,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one canonical pointer-only explicit OR group."""

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
    canonical_node_ids = _canonical_source_node_ids(
        source_node_ids,
        revision=revision,
    )
    revision_ref = _revision_reference(revision)
    group_id = _group_id(
        revision=revision_ref,
        source_node_ids=canonical_node_ids,
    )
    is_v2 = (
        revision["schema_version"]
        == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
    )
    body = _normalize_body(
        {
            "schema_version": (
                INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION_V2
                if is_v2
                else INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION
            ),
            "producer_version": (
                INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION_V2
                if is_v2
                else INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION
            ),
            "revision_artifact": _revision_artifact_pointer(
                {
                    field: binding[field]
                    for field in binding
                    if field != "canonical_bytes"
                }
            ),
            "revision": revision_ref,
            "source_node_ids": canonical_node_ids,
            "group_id": group_id,
            "selection_reason": _canonicalize_selection_reason(
                selection_reason
            ),
        }
    )
    selection_id = _stable_id(
        "interactive-tree-frontier-group-selection",
        body,
    )
    without_hash = {**body, "selection_id": selection_id}
    return validate_interactive_tree_frontier_group_selection(
        {
            **without_hash,
            "selection_hash": _sha256(_canonical_json(without_hash)),
        }
    )


def validate_interactive_tree_frontier_group_selection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate one self-authenticating pointer-only group."""

    if not isinstance(payload, Mapping):
        raise InteractiveTreeFrontierGroupSelectionError(
            "interactive-tree frontier group selection must be an object"
        )
    _exact_fields(
        payload,
        _TOP_LEVEL_FIELDS,
        "interactive-tree frontier group selection",
    )
    selection_id = _identifier(
        payload["selection_id"],
        "selection_id",
        _SELECTION_ID_RE,
    )
    selection_hash = _hash(payload["selection_hash"], "selection_hash")
    body = _normalize_body(
        {
            key: payload[key]
            for key in payload
            if key not in {"selection_id", "selection_hash"}
        }
    )
    expected_selection_id = _stable_id(
        "interactive-tree-frontier-group-selection",
        body,
    )
    if not hmac.compare_digest(selection_id, expected_selection_id):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection_id does not match canonical frontier group selection"
        )
    without_hash = {**body, "selection_id": selection_id}
    expected_hash = _sha256(_canonical_json(without_hash))
    if not hmac.compare_digest(selection_hash, expected_hash):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection_hash does not match canonical frontier group selection"
        )
    return {**without_hash, "selection_hash": selection_hash}


def canonical_interactive_tree_frontier_group_selection_json(
    payload: Mapping[str, Any],
) -> str:
    """Return the sole canonical JSON representation of a valid group."""

    return _canonical_json(
        validate_interactive_tree_frontier_group_selection(payload)
    )


def interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
    selection_payload: Mapping[str, Any],
    revision_payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    selection_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeFrontierGroupSelectionArtifactBinding
    ),
    revision_artifact_binding: (
        IndependentlyVerifiedInteractiveTreeRevisionArtifactBinding
    ),
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Replay one verified group into one generic OR candidate fragment."""

    selection = validate_interactive_tree_frontier_group_selection(
        selection_payload
    )
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
    expected_revision_pointer = _revision_artifact_pointer(
        {
            field: verified_revision_binding[field]
            for field in verified_revision_binding
            if field != "canonical_bytes"
        }
    )
    if selection["revision_artifact"] != expected_revision_pointer:
        raise InteractiveTreeFrontierGroupSelectionError(
            "revision artifact binding does not match group selection pointer"
        )
    if selection["revision"] != _revision_reference(revision):
        raise InteractiveTreeFrontierGroupSelectionError(
            "group selection revision reference does not match live revision"
        )
    selected = _replay_group_members(selection, revision=revision)
    condition = canonicalize_expression(
        {
            "op": "or",
            "args": [fragment["condition"] for fragment in selected],
        }
    )
    requirements = _merged_requirements(selected)
    rule_id = _stable_id(
        "candidate-rule",
        {
            "group_id": selection["group_id"],
            "condition": condition,
        },
    )
    effect_id = _stable_id(
        "candidate-effect",
        {
            "group_id": selection["group_id"],
            "member_effect_ids": [
                fragment["effect_id"] for fragment in selected
            ],
        },
    )
    fragment_id = _stable_id(
        "candidate-fragment",
        {
            "group_id": selection["group_id"],
            "rule_id": rule_id,
            "condition": condition,
            "requirements": requirements,
            "effect_id": effect_id,
        },
    )
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
            fragment_id=fragment_id,
            fragment_type="strategy_rule",
            rule_id=rule_id,
            condition=condition,
            requirements=requirements,
            effect_id=effect_id,
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
        raise InteractiveTreeFrontierGroupSelectionError(
            "interactive-tree frontier group failed generic fragment projection"
        ) from exc


def expected_interactive_tree_frontier_group_selection_provenance(
    selection_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive exact immutable TaskArtifact provenance for a valid group."""

    selection = validate_interactive_tree_frontier_group_selection(
        selection_payload
    )
    revision_artifact = selection["revision_artifact"]
    revision = selection["revision"]
    provenance = {
        "schema_version": (
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION_V2
            if revision["schema_version"]
            == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
            else INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": selection["producer_version"],
        "task_id": revision_artifact["task_id"],
        "kind": INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "group_id": selection["group_id"],
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
        "source_node_ids": list(selection["source_node_ids"]),
        "member_count": len(selection["source_node_ids"]),
    }
    _exact_fields(
        provenance,
        _SELECTION_PROVENANCE_FIELDS,
        "interactive-tree frontier group selection provenance",
    )
    return provenance


def _normalize_body(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_fields(
        value,
        _BODY_FIELDS,
        "interactive-tree frontier group selection body",
    )
    schema = value["schema_version"]
    if schema == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION:
        producer = INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION
    elif schema == INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION_V2:
        producer = (
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION_V2
        )
    else:
        raise InteractiveTreeFrontierGroupSelectionError(
            "interactive-tree frontier group selection schema_version is invalid"
        )
    if value["producer_version"] != producer:
        raise InteractiveTreeFrontierGroupSelectionError(
            "interactive-tree frontier group selection producer_version is invalid"
        )
    revision_artifact = _revision_artifact_pointer(
        value["revision_artifact"]
    )
    revision = _revision_reference_from_value(value["revision"])
    source_node_ids = _source_node_id_sequence(value["source_node_ids"])
    group_id = _identifier(value["group_id"], "group_id", _GROUP_ID_RE)
    expected_group_id = _group_id(
        revision=revision,
        source_node_ids=source_node_ids,
    )
    if not hmac.compare_digest(group_id, expected_group_id):
        raise InteractiveTreeFrontierGroupSelectionError(
            "group_id does not match canonical semantic frontier group"
        )
    _require_revision_pointer_cross_references(
        revision_artifact=revision_artifact,
        revision=revision,
    )
    expected_schema = (
        INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION_V2
        if revision["schema_version"] == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
        else INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION
    )
    if schema != expected_schema:
        raise InteractiveTreeFrontierGroupSelectionError(
            "frontier group selection schema does not match revision schema"
        )
    return {
        "schema_version": schema,
        "producer_version": producer,
        "revision_artifact": revision_artifact,
        "revision": revision,
        "source_node_ids": source_node_ids,
        "group_id": group_id,
        "selection_reason": _selection_reason(value["selection_reason"]),
    }


def _verified_selection_artifact_binding(
    value: object,
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierGroupSelectionError(
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
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact canonical_bytes must be bytes"
        )
    expected_bytes = (
        canonical_interactive_tree_frontier_group_selection_json(
            selection
        ).encode("utf-8")
    )
    if not hmac.compare_digest(canonical_bytes, expected_bytes):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact canonical bytes do not match the selection"
        )
    if not hmac.compare_digest(
        pointer["content_hash"],
        hashlib.sha256(canonical_bytes).hexdigest(),
    ):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact content hash does not match canonical bytes"
        )
    expected_provenance = (
        expected_interactive_tree_frontier_group_selection_provenance(
            selection
        )
    )
    if (
        pointer["artifact_schema_version"]
        != expected_provenance["schema_version"]
        or pointer["provenance"] != expected_provenance
    ):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact provenance does not match the selection"
        )
    if pointer["task_id"] != selection["revision_artifact"]["task_id"]:
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact task_id does not match revision artifact"
        )
    return {**pointer, "canonical_bytes": canonical_bytes}


def _selection_artifact_pointer(value: Mapping[str, Any]) -> dict[str, Any]:
    kind = _text(value["kind"], "selection artifact kind")
    schema = _text(
        value["artifact_schema_version"],
        "selection artifact schema_version",
    )
    origin = _text(value["origin_tool"], "selection artifact origin_tool")
    constants = {
        "kind": (
            kind,
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
        ),
        "schema_version": (
            schema,
            {
                INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION,
                INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION_V2,
            },
        ),
        "origin_tool": (
            origin,
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
        ),
    }
    for name, (actual, expected) in constants.items():
        if (
            actual not in expected
            if isinstance(expected, set)
            else actual != expected
        ):
            raise InteractiveTreeFrontierGroupSelectionError(
                f"selection artifact {name} is invalid"
            )
    provenance = value["provenance"]
    if not isinstance(provenance, Mapping):
        raise InteractiveTreeFrontierGroupSelectionError(
            "selection artifact provenance must be an object"
        )
    _exact_fields(
        provenance,
        _SELECTION_PROVENANCE_FIELDS,
        "selection artifact provenance",
    )
    return {
        "artifact_id": _text(
            value["artifact_id"],
            "selection artifact artifact_id",
        ),
        "task_id": _text(value["task_id"], "selection artifact task_id"),
        "kind": kind,
        "artifact_schema_version": schema,
        "content_hash": _hash(
            value["content_hash"],
            "selection artifact content_hash",
        ),
        "origin_tool": origin,
        "path": _text(value["path"], "selection artifact path"),
        "provenance": json.loads(_canonical_json(provenance)),
    }


def _replay_group_members(
    selection: Mapping[str, Any],
    *,
    revision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical_node_ids = _canonical_source_node_ids(
        selection["source_node_ids"],
        revision=revision,
    )
    if canonical_node_ids != selection["source_node_ids"]:
        raise InteractiveTreeFrontierGroupSelectionError(
            "source_node_ids are not in live revision frontier order"
        )
    by_node_id = {
        fragment["source_node_id"]: fragment
        for fragment in revision["fragments"]
    }
    try:
        return [by_node_id[node_id] for node_id in canonical_node_ids]
    except KeyError as exc:
        raise InteractiveTreeFrontierGroupSelectionError(
            "group no longer identifies current revision frontier fragments"
        ) from exc


def _merged_requirements(
    fragments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for fragment in fragments:
        for requirement in fragment["requirements"]:
            key = _canonical_json(requirement)
            if key not in by_key:
                order.append(key)
                by_key[key] = json.loads(key)
    return [by_key[key] for key in order]


def _canonical_source_node_ids(
    value: object,
    *,
    revision: Mapping[str, Any],
) -> list[str]:
    requested = _source_node_id_sequence(value)
    requested_set = set(requested)
    frontier = list(revision["tree"]["frontier_node_ids"])
    unknown = requested_set - set(frontier)
    if unknown:
        raise InteractiveTreeFrontierGroupSelectionError(
            "source_node_ids must identify current revision frontier nodes"
        )
    return [node_id for node_id in frontier if node_id in requested_set]


def _source_node_id_sequence(value: object) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise InteractiveTreeFrontierGroupSelectionError(
            "source_node_ids must be an array of 2 to 50 node ids"
        )
    if not _MIN_GROUP_MEMBERS <= len(value) <= _MAX_GROUP_MEMBERS:
        raise InteractiveTreeFrontierGroupSelectionError(
            "source_node_ids must contain 2 to 50 node ids"
        )
    normalized = [
        _identifier(item, f"source_node_ids[{index}]", _SOURCE_NODE_ID_RE)
        for index, item in enumerate(value)
    ]
    if len(set(normalized)) != len(normalized):
        raise InteractiveTreeFrontierGroupSelectionError(
            "source_node_ids must not contain duplicate nodes"
        )
    return normalized


def _group_id(
    *,
    revision: Mapping[str, Any],
    source_node_ids: Sequence[str],
) -> str:
    return _stable_id(
        "interactive-tree-frontier-group",
        {
            "semantic_tree_id": revision["semantic_tree_id"],
            "tree_hash": revision["tree_hash"],
            "source_node_ids": list(source_node_ids),
        },
    )


def _revision_reference_from_value(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeFrontierGroupSelectionError(
            "revision reference must be an object"
        )
    try:
        return _revision_reference(
            {
                "schema_version": value["schema_version"],
                "revision_id": value["revision_id"],
                "revision_hash": value["revision_hash"],
                "semantic_tree_id": value["semantic_tree_id"],
                "tree": {"tree_hash": value["tree_hash"]},
                "asset_type": value["asset_type"],
            }
        )
    except (KeyError, StrategyError) as exc:
        raise InteractiveTreeFrontierGroupSelectionError(
            "revision reference is invalid"
        ) from exc


def _require_revision_pointer_cross_references(
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
        raise InteractiveTreeFrontierGroupSelectionError(
            "revision artifact schema does not match group revision"
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
            raise InteractiveTreeFrontierGroupSelectionError(
                f"revision artifact provenance {field} does not match selection"
            )


def _ancestor_sequence(
    value: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise InteractiveTreeFrontierGroupSelectionError(
            "ancestor_revisions must be an array"
        )
    return tuple(value)


def _identifier(
    value: object,
    name: str,
    pattern: re.Pattern[str],
) -> str:
    normalized = _text(value, name)
    if pattern.fullmatch(normalized) is None:
        raise InteractiveTreeFrontierGroupSelectionError(
            f"{name} has an invalid format"
        )
    return normalized


def _hash(value: object, name: str) -> str:
    normalized = _text(value, name)
    if _HASH_RE.fullmatch(normalized) is None:
        raise InteractiveTreeFrontierGroupSelectionError(
            f"{name} must be a lowercase SHA-256 hash"
        )
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractiveTreeFrontierGroupSelectionError(
            f"{name} must be non-empty text"
        )
    return value.strip()


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
        raise InteractiveTreeFrontierGroupSelectionError(
            "interactive-tree frontier group must be finite canonical JSON"
        ) from exc


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unsupported fields " + ", ".join(unexpected))
        raise InteractiveTreeFrontierGroupSelectionError(
            f"{name} has " + "; ".join(details)
        )


__all__ = [
    "INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND",
    "INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION",
    "INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL",
    "INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_PRODUCER_VERSION",
    "INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_SCHEMA_VERSION",
    "IndependentlyVerifiedInteractiveTreeFrontierGroupSelectionArtifactBinding",
    "InteractiveTreeFrontierGroupSelectionError",
    "build_interactive_tree_frontier_group_selection",
    "canonical_interactive_tree_frontier_group_selection_json",
    "expected_interactive_tree_frontier_group_selection_provenance",
    "interactive_tree_frontier_group_selection_to_verified_candidate_fragment",
    "validate_interactive_tree_frontier_group_selection",
]
