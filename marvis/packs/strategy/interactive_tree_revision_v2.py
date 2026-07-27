"""Strict v2 interactive-tree revisions with effective threshold semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import hmac
import json
import math
from numbers import Real
import re
from typing import Any

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_replay import (
    InteractiveTreeReplayResult,
)


INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION = (
    "strategy.interactive-tree-revision.v2"
)
INTERACTIVE_TREE_REVISION_V2_PRODUCER_VERSION = (
    "strategy.interactive-tree-revision/2"
)
INTERACTIVE_TREE_ASSET_TYPE = "interactive_rule_tree"

_LIFECYCLE = {
    "candidate_stage": "development",
    "observation_stage": "backtested",
    "validation_status": "unvalidated",
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "asset_type",
        "lifecycle",
        "identity",
        "base_tree",
        "parent_revision",
        "edit",
        "semantic_tree_id",
        "tree",
        "fragments",
        "checks",
        "candidate_evidence",
        "source_refs",
        "revision_id",
        "revision_hash",
    }
)
_REVISION_ID_RE = re.compile(r"^interactive-tree-revision-[0-9a-f]{32}$")
_SEMANTIC_TREE_ID_RE = re.compile(r"^interactive-tree-[0-9a-f]{32}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATED_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_WARNING_CODES = frozenset(
    {"threshold_grouping_unchanged", "split_grouping_unchanged"}
)
_MAX_VISIBLE_NODES = 511
_CONTINUATION_REPLAY_SCHEMA = (
    "strategy.interactive-tree-continuation-replay.v1"
)
_CONTINUATION_OBJECTIVE = "max_gini_gain"
_CONTINUATION_TIE_BREAK = (
    "eligible_gain_feature_threshold_candidate_id"
)


class InteractiveTreeRevisionV2Error(StrategyError):
    """A v2 effective interactive-tree revision failed closed."""


def build_adjusted_interactive_tree_revision_v2(
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    threshold: float,
    reason: str | None,
    replay: InteractiveTreeReplayResult,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one v2 threshold revision from deterministic replay output."""

    source = validate_automatic_tree_asset(automatic_tree_asset)
    parent = _validated_parent(
        parent_revision,
        source,
        ancestor_revisions=ancestor_revisions,
    )
    current = _effective_view(source, parent)
    normalized_node_id = _text(node_id, "node_id")
    node = current["node_by_id"].get(normalized_node_id)
    if node is None or node["kind"] != "split":
        raise InteractiveTreeRevisionV2Error(
            "threshold edit requires a current visible split node"
        )
    if normalized_node_id in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "threshold edit cannot target a frontier node"
        )
    normalized_threshold = _finite_number(threshold, "threshold")
    previous_threshold = _finite_number(node["threshold"], "previous_threshold")
    if normalized_threshold == previous_threshold:
        raise InteractiveTreeRevisionV2Error(
            "threshold edit must change the current effective threshold"
        )
    if (
        tuple(replay.visible_node_ids) != tuple(current["visible"])
        or tuple(replay.frontier_node_ids) != tuple(current["frontier"])
    ):
        raise InteractiveTreeRevisionV2Error(
            "threshold replay changed the current visible topology"
        )
    replay_nodes = [deepcopy(item) for item in replay.nodes]
    checks = _checks(
        replay.replay.get("warning_codes", []),
    )
    payload = _assemble(
        source,
        parent=parent,
        edit={
            "operation": "adjust_split_threshold",
            "node_id": normalized_node_id,
            "previous_threshold": previous_threshold,
            "threshold": normalized_threshold,
            "reason": _reason(reason),
        },
        nodes=replay_nodes,
        visible=tuple(replay.visible_node_ids),
        frontier=tuple(replay.frontier_node_ids),
        checks=checks,
    )
    return validate_interactive_tree_revision_v2(
        payload,
        source,
        parent_revision=parent,
        ancestor_revisions=ancestor_revisions,
    )


def build_replaced_interactive_tree_split_revision_v2(
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    feature: str,
    threshold: float,
    reason: str | None,
    replay: InteractiveTreeReplayResult,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one immutable exact feature/threshold replacement revision."""

    source = validate_automatic_tree_asset(automatic_tree_asset)
    parent = _validated_parent(
        parent_revision,
        source,
        ancestor_revisions=ancestor_revisions,
    )
    current = _effective_view(source, parent)
    normalized_node_id = _text(node_id, "node_id")
    node = current["node_by_id"].get(normalized_node_id)
    if node is None or node["kind"] != "split":
        raise InteractiveTreeRevisionV2Error(
            "split replacement requires a current visible split node"
        )
    if normalized_node_id in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "split replacement cannot target a frontier node"
        )
    normalized_feature = _text(feature, "feature")
    allowed_features = tuple(
        source["tree_result"]["training"]["feature_order"]
    )
    if normalized_feature not in allowed_features:
        raise InteractiveTreeRevisionV2Error(
            "split replacement feature is outside the authenticated universe"
        )
    previous_feature = _text(node["feature"], "previous_feature")
    if normalized_feature == previous_feature:
        raise InteractiveTreeRevisionV2Error(
            "split replacement must change the current feature"
        )
    normalized_threshold = _finite_number(threshold, "threshold")
    previous_threshold = _finite_number(
        node["threshold"],
        "previous_threshold",
    )
    if (
        tuple(replay.visible_node_ids) != tuple(current["visible"])
        or tuple(replay.frontier_node_ids) != tuple(current["frontier"])
    ):
        raise InteractiveTreeRevisionV2Error(
            "split replacement replay changed the visible topology"
        )
    payload = _assemble(
        source,
        parent=parent,
        edit={
            "operation": "replace_split_feature",
            "node_id": normalized_node_id,
            "previous_feature": previous_feature,
            "feature": normalized_feature,
            "previous_threshold": previous_threshold,
            "threshold": normalized_threshold,
            "reason": _reason(reason),
        },
        nodes=[deepcopy(item) for item in replay.nodes],
        visible=tuple(replay.visible_node_ids),
        frontier=tuple(replay.frontier_node_ids),
        checks=_checks(replay.replay.get("warning_codes", [])),
    )
    return validate_interactive_tree_revision_v2(
        payload,
        source,
        parent_revision=parent,
        ancestor_revisions=ancestor_revisions,
    )


def build_continued_interactive_tree_revision_v2(
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    search_id: str,
    search_hash: str,
    candidate_id: str,
    feature: str,
    threshold: float,
    missing_child: str,
    controls: Mapping[str, Any],
    reason: str | None,
    continuation: Any,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build one immutable, explicitly seeded bounded subtree continuation."""

    source = validate_automatic_tree_asset(automatic_tree_asset)
    parent = _validated_parent(
        parent_revision,
        source,
        ancestor_revisions=ancestor_revisions,
    )
    current = _effective_view(source, parent)
    normalized_node_id = _text(node_id, "node_id")
    node = current["node_by_id"].get(normalized_node_id)
    if node is None or normalized_node_id not in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "automatic continuation requires a current frontier node"
        )
    replay = _normalize_continuation_replay(
        continuation.replay,
        node_id=normalized_node_id,
        candidate_id=_text(candidate_id, "candidate_id"),
    )
    normalized_controls = _normalize_continuation_controls(controls)
    if replay["controls"] != normalized_controls:
        raise InteractiveTreeRevisionV2Error(
            "automatic continuation controls changed from deterministic replay"
        )
    edit = {
        "operation": "auto_continue_subtree",
        "node_id": normalized_node_id,
        "search_id": _text(search_id, "search_id"),
        "search_hash": _hash(search_hash, "search_hash"),
        "candidate_id": _text(candidate_id, "candidate_id"),
        "feature": _text(feature, "feature"),
        "threshold": _finite_number(threshold, "threshold"),
        "missing_child": _missing_child(missing_child),
        "objective": _CONTINUATION_OBJECTIVE,
        "tie_break": _CONTINUATION_TIE_BREAK,
        "controls": normalized_controls,
        "replay": replay,
        "reason": _reason(reason),
    }
    target = next(
        (
            item
            for item in continuation.nodes
            if item.get("node_id") == normalized_node_id
        ),
        None,
    )
    if (
        target is None
        or target.get("kind") != "split"
        or target.get("feature") != edit["feature"]
        or float(target.get("threshold")) != edit["threshold"]
        or target.get("missing_child") != edit["missing_child"]
    ):
        raise InteractiveTreeRevisionV2Error(
            "automatic continuation seed changed from the generated root"
        )
    payload = _assemble(
        source,
        parent=parent,
        edit=edit,
        nodes=[deepcopy(item) for item in continuation.nodes],
        visible=tuple(continuation.visible_node_ids),
        frontier=tuple(continuation.frontier_node_ids),
        checks=_checks([]),
    )
    return validate_interactive_tree_revision_v2(
        payload,
        source,
        parent_revision=parent,
        ancestor_revisions=ancestor_revisions,
    )


def build_pruned_interactive_tree_revision_v2(
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    reason: str | None,
    parent_revision: Mapping[str, Any],
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Prune a v2 chain without reverting its effective thresholds."""

    source = validate_automatic_tree_asset(automatic_tree_asset)
    parent = _validated_parent(
        parent_revision,
        source,
        ancestor_revisions=ancestor_revisions,
    )
    if parent is None or parent["schema_version"] != (
        INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 prune requires a v2 parent revision"
        )
    current = _effective_view(source, parent)
    normalized_node_id = _text(node_id, "node_id")
    node = current["node_by_id"].get(normalized_node_id)
    if node is None or node["kind"] != "split":
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree prune requires a current visible split node"
        )
    if normalized_node_id in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree prune node is already a frontier leaf"
        )
    frontier = _pruned_frontier_from_current(
        current,
        current_frontier=tuple(current["frontier"]),
        node_id=normalized_node_id,
    )
    visible = _visible_node_ids_from_current(current, frontier=frontier)
    nodes = [
        deepcopy(current["node_by_id"][item])
        for item in visible
    ]
    payload = _assemble(
        source,
        parent=parent,
        edit={
            "operation": "prune_subtree",
            "node_id": normalized_node_id,
            "previous_threshold": None,
            "threshold": None,
            "reason": _reason(reason),
        },
        nodes=nodes,
        visible=visible,
        frontier=frontier,
        checks=_checks(parent["checks"]["warning_codes"]),
    )
    return validate_interactive_tree_revision_v2(
        payload,
        source,
        parent_revision=parent,
        ancestor_revisions=ancestor_revisions,
    )


def validate_interactive_tree_revision_v2(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate canonical v2 structure, lineage and effective semantics."""

    if not isinstance(payload, Mapping):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 revision must be an object"
        )
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "interactive-tree v2 revision")
    source = validate_automatic_tree_asset(automatic_tree_asset)
    parent = _validated_parent(
        parent_revision,
        source,
        ancestor_revisions=ancestor_revisions,
    )
    if payload["schema_version"] != INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 schema_version is invalid"
        )
    if (
        payload["producer_version"]
        != INTERACTIVE_TREE_REVISION_V2_PRODUCER_VERSION
    ):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 producer_version is invalid"
        )
    if payload["asset_type"] != INTERACTIVE_TREE_ASSET_TYPE:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 asset_type is invalid"
        )
    if payload["lifecycle"] != _LIFECYCLE:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 lifecycle is invalid"
        )
    if payload["identity"] != source["identity"]:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 identity changed from the base tree"
        )
    if payload["base_tree"] != _base_tree(source):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 base tree binding changed"
        )
    expected_parent = None if parent is None else _parent_ref(parent)
    if payload["parent_revision"] != expected_parent:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 parent binding changed"
        )

    edit = _normalize_edit(payload["edit"])
    if payload["edit"] != edit:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 edit is not canonical"
        )
    source_node_ids = {
        item["node_id"]
        for item in source["tree_result"]["tree"]["nodes"]
    }
    parent_has_generated_nodes = (
        parent is not None
        and parent["schema_version"]
        == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
        and any(
            item["node_id"] not in source_node_ids
            for item in parent["tree"]["nodes"]
        )
    )
    allow_generated = (
        edit["operation"] == "auto_continue_subtree"
        or parent_has_generated_nodes
    )
    tree = _normalize_tree(
        payload["tree"],
        source=source,
        allow_generated=allow_generated,
    )
    if payload["tree"] != tree:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 tree is not canonical"
        )
    current = _effective_view(source, parent)
    _require_edit_transition(
        edit,
        tree=tree,
        current=current,
        source=source,
        parent=parent,
    )
    if allow_generated:
        _require_frontier_cover_from_tree(tree)
    else:
        _require_frontier_cover(source, tuple(tree["frontier_node_ids"]))
    _require_metric_conservation(tree)
    _require_split_diagnostics(tree, source=source)

    fragments = _derive_fragments(
        source,
        tree=tree,
    )
    if payload["fragments"] != fragments:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 fragments changed from effective nodes"
        )
    checks = _normalize_checks(payload["checks"])
    if payload["checks"] != checks:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 checks are not canonical"
        )
    semantic_body = _semantic_body(source, tree=tree, fragments=fragments)
    semantic_hash = _sha256(semantic_body)
    if payload["semantic_tree_id"] != (
        f"interactive-tree-{semantic_hash[:32]}"
    ):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 semantic_tree_id is invalid"
        )
    if tree["tree_hash"] != semantic_hash:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 tree_hash is invalid"
        )
    evidence = _candidate_evidence(semantic_body)
    if payload["candidate_evidence"] != evidence:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 candidate evidence is invalid"
        )
    refs = _source_refs(source, parent=parent)
    if payload["source_refs"] != refs:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 source refs are invalid"
        )
    body = {
        key: deepcopy(payload[key])
        for key in payload
        if key not in {"revision_id", "revision_hash"}
    }
    revision_id = f"interactive-tree-revision-{_sha256(body)[:32]}"
    if payload["revision_id"] != revision_id:
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 revision_id is invalid"
        )
    without_hash = {**body, "revision_id": revision_id}
    revision_hash = _sha256(without_hash)
    if not hmac.compare_digest(str(payload["revision_hash"]), revision_hash):
        raise InteractiveTreeRevisionV2Error(
            "interactive-tree v2 revision_hash is invalid"
        )
    return {**without_hash, "revision_hash": revision_hash}


def canonical_interactive_tree_revision_v2_json(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return canonical JSON for one validated v2 revision."""

    return json.dumps(
        validate_interactive_tree_revision_v2(
            payload,
            automatic_tree_asset,
            parent_revision=parent_revision,
            ancestor_revisions=ancestor_revisions,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def interactive_tree_revision_v2_to_candidate_fragments(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Project detached generic fragments for exact Pool replay."""

    revision = validate_interactive_tree_revision_v2(
        payload,
        automatic_tree_asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestor_revisions,
    )
    return [
        {
            "asset_type": INTERACTIVE_TREE_ASSET_TYPE,
            "asset_id": revision["semantic_tree_id"],
            "asset_hash": revision["tree"]["tree_hash"],
            "candidate_id": revision["candidate_evidence"]["candidate_id"],
            "evidence_hash": revision["candidate_evidence"]["evidence_hash"],
            "fragment_id": item["fragment_id"],
            "fragment_hash": item["fragment_hash"],
            "rule_id": item["rule_id"],
            "effect_id": item["effect_id"],
            "condition": deepcopy(item["condition"]),
            "requirements": [],
            "metrics": deepcopy(item["metrics"]),
            "source_node_id": item["source_node_id"],
            "leaf_id": item["leaf_id"],
        }
        for item in revision["fragments"]
    ]


def interactive_tree_topology_evidence_v2(
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the effective visible topology stored by a validated caller."""

    frontier = set(revision["tree"]["frontier_node_ids"])
    return {
        "root_node_id": revision["tree"]["root_node_id"],
        "visible_node_ids": list(revision["tree"]["visible_node_ids"]),
        "frontier_node_ids": list(revision["tree"]["frontier_node_ids"]),
        "nodes": [
            {
                **deepcopy(node),
                "is_visible": True,
                "is_frontier": node["node_id"] in frontier,
                "can_prune": (
                    node["kind"] == "split"
                    and node["node_id"] not in frontier
                ),
            }
            for node in revision["tree"]["nodes"]
        ],
    }


def _assemble(
    source: Mapping[str, Any],
    *,
    parent: Mapping[str, Any] | None,
    edit: dict[str, Any],
    nodes: list[dict[str, Any]],
    visible: tuple[str, ...],
    frontier: tuple[str, ...],
    checks: dict[str, Any],
) -> dict[str, Any]:
    tree_without_hash = {
        "root_node_id": source["tree_result"]["tree"]["root_node_id"],
        "visible_node_ids": list(visible),
        "frontier_node_ids": list(frontier),
        "nodes": nodes,
    }
    fragments = _derive_fragments(
        source,
        tree={**tree_without_hash, "tree_hash": "0" * 64},
    )
    semantic_body = _semantic_body(
        source,
        tree={**tree_without_hash, "tree_hash": "0" * 64},
        fragments=fragments,
    )
    tree_hash = _sha256(semantic_body)
    body = {
        "schema_version": INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_REVISION_V2_PRODUCER_VERSION,
        "asset_type": INTERACTIVE_TREE_ASSET_TYPE,
        "lifecycle": deepcopy(_LIFECYCLE),
        "identity": deepcopy(source["identity"]),
        "base_tree": _base_tree(source),
        "parent_revision": None if parent is None else _parent_ref(parent),
        "edit": edit,
        "semantic_tree_id": f"interactive-tree-{tree_hash[:32]}",
        "tree": {**tree_without_hash, "tree_hash": tree_hash},
        "fragments": fragments,
        "checks": checks,
        "candidate_evidence": _candidate_evidence(semantic_body),
        "source_refs": _source_refs(source, parent=parent),
    }
    revision_id = f"interactive-tree-revision-{_sha256(body)[:32]}"
    without_hash = {**body, "revision_id": revision_id}
    return {**without_hash, "revision_hash": _sha256(without_hash)}


def _effective_view(
    source: Mapping[str, Any],
    parent: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_nodes = source["tree_result"]["tree"]["nodes"]
    source_by_id = {item["node_id"]: item for item in source_nodes}
    if parent is None:
        visible = tuple(item["node_id"] for item in source_nodes)
        frontier = tuple(source["tree_result"]["tree"]["leaf_ids"])
        nodes = [_base_effective_node(item, source=source) for item in source_nodes]
    elif parent["schema_version"] == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION:
        visible = tuple(parent["tree"]["visible_node_ids"])
        frontier = tuple(parent["tree"]["frontier_node_ids"])
        nodes = [deepcopy(item) for item in parent["tree"]["nodes"]]
    else:
        visible = tuple(parent["tree"]["visible_node_ids"])
        frontier = tuple(parent["tree"]["frontier_node_ids"])
        conditions = _conditions_from_configs(
            source,
            {
                item["node_id"]: {
                    "feature": item["feature"],
                    "threshold": float(item["threshold"]),
                }
                for item in source_nodes
                if item["kind"] == "split"
            },
        )
        nodes = []
        for node_id in visible:
            node = _base_effective_node(source_by_id[node_id], source=source)
            node["condition"] = conditions[node_id]
            nodes.append(node)
    return {
        "visible": visible,
        "frontier": frontier,
        "nodes": nodes,
        "node_by_id": {item["node_id"]: item for item in nodes},
    }


def _base_effective_node(
    node: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    conditions = _conditions_from_configs(
        source,
        {
            item["node_id"]: {
                "feature": item["feature"],
                "threshold": float(item["threshold"]),
            }
            for item in source["tree_result"]["tree"]["nodes"]
            if item["kind"] == "split"
        },
    )
    result = {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "depth": node["depth"],
        "path": list(node["path"]),
        "condition": conditions[node["node_id"]],
        "metrics": deepcopy(node["metrics"]),
    }
    if node["kind"] == "leaf":
        result["rule_id"] = node["rule_id"]
    else:
        result.update(
            {
                "feature": node["feature"],
                "threshold": float(node["threshold"]),
                "base_threshold": float(node["threshold"]),
                "missing_child": node["missing_child"],
                "left_child_id": node["left_child_id"],
                "right_child_id": node["right_child_id"],
                "direction_diagnostic": deepcopy(
                    node["direction_diagnostic"]
                ),
            }
        )
    return result


def _normalize_tree(
    value: object,
    *,
    source: Mapping[str, Any],
    allow_generated: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error("v2 tree must be an object")
    _exact_fields(
        value,
        {
            "tree_hash",
            "root_node_id",
            "visible_node_ids",
            "frontier_node_ids",
            "nodes",
        },
        "interactive-tree v2 tree",
    )
    tree_hash = _hash(value["tree_hash"], "tree.tree_hash")
    root_id = source["tree_result"]["tree"]["root_node_id"]
    if value["root_node_id"] != root_id:
        raise InteractiveTreeRevisionV2Error("v2 tree root changed")
    visible = _text_sequence(value["visible_node_ids"], "visible_node_ids")
    frontier = _text_sequence(value["frontier_node_ids"], "frontier_node_ids")
    nodes_value = value["nodes"]
    if not isinstance(nodes_value, list):
        raise InteractiveTreeRevisionV2Error("v2 tree nodes must be a list")
    if allow_generated:
        return _normalize_generated_tree(
            tree_hash=tree_hash,
            root_id=root_id,
            visible=visible,
            frontier=frontier,
            nodes_value=nodes_value,
            source=source,
        )
    source_by_id = {
        item["node_id"]: item
        for item in source["tree_result"]["tree"]["nodes"]
    }
    medians = source["tree_result"]["preprocessing"]["medians"]
    nodes: list[dict[str, Any]] = []
    configs: dict[str, dict[str, Any]] = {}
    allowed_features = set(
        source["tree_result"]["training"]["feature_order"]
    )
    for raw in nodes_value:
        if not isinstance(raw, Mapping):
            raise InteractiveTreeRevisionV2Error("v2 tree node must be an object")
        node_id = _text(raw.get("node_id"), "node.node_id")
        base = source_by_id.get(node_id)
        if base is None:
            raise InteractiveTreeRevisionV2Error(
                "v2 tree node does not exist in the base topology"
            )
        common = {
            "node_id",
            "kind",
            "depth",
            "path",
            "condition",
            "metrics",
        }
        expected = (
            common | {"rule_id"}
            if base["kind"] == "leaf"
            else common
            | {
                "feature",
                "threshold",
                "base_threshold",
                "missing_child",
                "left_child_id",
                "right_child_id",
                "direction_diagnostic",
            }
        )
        _exact_fields(raw, expected, f"interactive-tree v2 node {node_id}")
        if (
            raw["kind"] != base["kind"]
            or raw["depth"] != base["depth"]
            or raw["path"] != base["path"]
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 tree node topology changed from the base"
            )
        node = {
            "node_id": node_id,
            "kind": base["kind"],
            "depth": base["depth"],
            "path": list(base["path"]),
            "condition": canonicalize_expression(raw["condition"]),
            "metrics": _json_object(raw["metrics"], "node.metrics"),
        }
        if base["kind"] == "leaf":
            if raw["rule_id"] != base["rule_id"]:
                raise InteractiveTreeRevisionV2Error(
                    "v2 leaf base rule_id changed"
                )
            node["rule_id"] = base["rule_id"]
        else:
            threshold = _finite_number(raw["threshold"], "node.threshold")
            base_threshold = _finite_number(
                raw["base_threshold"],
                "node.base_threshold",
            )
            if base_threshold != float(base["threshold"]):
                raise InteractiveTreeRevisionV2Error(
                    "v2 node base_threshold changed"
                )
            feature = _text(raw["feature"], "node.feature")
            if feature not in allowed_features or feature not in medians:
                raise InteractiveTreeRevisionV2Error(
                    "v2 split feature is outside the authenticated universe"
                )
            missing = (
                "left"
                if float(medians[feature]) <= threshold
                else "right"
            )
            if (
                raw["missing_child"] != missing
                or raw["left_child_id"] != base["left_child_id"]
                or raw["right_child_id"] != base["right_child_id"]
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 split structure or missing route changed"
                )
            node.update(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "base_threshold": base_threshold,
                    "missing_child": missing,
                    "left_child_id": base["left_child_id"],
                    "right_child_id": base["right_child_id"],
                    "direction_diagnostic": _json_object(
                        raw["direction_diagnostic"],
                        "node.direction_diagnostic",
                    ),
                }
            )
            configs[node_id] = {
                "feature": feature,
                "threshold": threshold,
            }
        nodes.append(node)
    if [item["node_id"] for item in nodes] != visible:
        raise InteractiveTreeRevisionV2Error(
            "v2 tree nodes do not match visible_node_ids"
        )
    expected_visible = _visible_node_ids(source, frontier=tuple(frontier))
    if tuple(visible) != expected_visible:
        raise InteractiveTreeRevisionV2Error(
            "v2 visible topology is not canonical"
        )
    effective_configs = {
        item["node_id"]: (
            configs[item["node_id"]]
            if item["node_id"] in configs
            else {
                "feature": item["feature"],
                "threshold": float(item["threshold"]),
            }
        )
        for item in source["tree_result"]["tree"]["nodes"]
        if item["kind"] == "split"
    }
    conditions = _conditions_from_configs(source, effective_configs)
    for node in nodes:
        if node["condition"] != conditions[node["node_id"]]:
            raise InteractiveTreeRevisionV2Error(
                "v2 node condition changed from effective thresholds"
            )
    return {
        "tree_hash": tree_hash,
        "root_node_id": root_id,
        "visible_node_ids": visible,
        "frontier_node_ids": frontier,
        "nodes": nodes,
    }


def _normalize_generated_tree(
    *,
    tree_hash: str,
    root_id: str,
    visible: list[str],
    frontier: list[str],
    nodes_value: list[object],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not visible
        or len(visible) > _MAX_VISIBLE_NODES
        or len(visible) != len(set(visible))
        or not frontier
        or len(frontier) != len(set(frontier))
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 generated topology exceeds its structural bounds"
        )
    source_by_id = {
        item["node_id"]: item
        for item in source["tree_result"]["tree"]["nodes"]
    }
    medians = source["tree_result"]["preprocessing"]["medians"]
    allowed_features = set(
        source["tree_result"]["training"]["feature_order"]
    )
    nodes: list[dict[str, Any]] = []
    paths: set[tuple[str, ...]] = set()
    for raw in nodes_value:
        if not isinstance(raw, Mapping):
            raise InteractiveTreeRevisionV2Error(
                "v2 generated tree node must be an object"
            )
        node_id = _text(raw.get("node_id"), "node.node_id")
        kind = raw.get("kind")
        if kind not in {"split", "leaf"}:
            raise InteractiveTreeRevisionV2Error(
                "v2 generated node kind is invalid"
            )
        common = {
            "node_id",
            "kind",
            "depth",
            "path",
            "condition",
            "metrics",
        }
        expected = (
            common | {"rule_id"}
            if kind == "leaf"
            else common
            | {
                "feature",
                "threshold",
                "base_threshold",
                "missing_child",
                "left_child_id",
                "right_child_id",
                "direction_diagnostic",
            }
        )
        _exact_fields(raw, expected, f"interactive-tree v2 node {node_id}")
        path_value = raw["path"]
        if (
            not isinstance(path_value, list)
            or any(item not in {"left", "right"} for item in path_value)
            or isinstance(raw["depth"], bool)
            or not isinstance(raw["depth"], int)
            or raw["depth"] != len(path_value)
            or tuple(path_value) in paths
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 generated node path is invalid"
            )
        paths.add(tuple(path_value))
        base = source_by_id.get(node_id)
        if base is not None and (
            base["depth"] != raw["depth"]
            or base["path"] != path_value
            or (
                base["kind"] != kind
                and not (base["kind"] == "leaf" and kind == "split")
            )
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 generated topology changed a base node identity"
            )
        node = {
            "node_id": node_id,
            "kind": kind,
            "depth": raw["depth"],
            "path": list(path_value),
            "condition": canonicalize_expression(raw["condition"]),
            "metrics": _json_object(raw["metrics"], "node.metrics"),
        }
        if kind == "leaf":
            rule_id = _text(raw["rule_id"], "node.rule_id")
            if base is not None and (
                base["kind"] != "leaf" or rule_id != base["rule_id"]
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 base leaf semantics changed"
                )
            if (
                base is None
                and _GENERATED_RULE_ID_RE.fullmatch(rule_id) is None
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 generated leaf rule identity changed"
                )
            node["rule_id"] = rule_id
        else:
            feature = _text(raw["feature"], "node.feature")
            threshold = _finite_number(raw["threshold"], "node.threshold")
            base_threshold = _finite_number(
                raw["base_threshold"],
                "node.base_threshold",
            )
            if feature not in allowed_features or feature not in medians:
                raise InteractiveTreeRevisionV2Error(
                    "v2 generated split feature is outside the universe"
                )
            if (
                base is not None
                and base["kind"] == "split"
                and base_threshold != float(base["threshold"])
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 base split threshold identity changed"
                )
            missing = (
                "left"
                if float(medians[feature]) <= threshold
                else "right"
            )
            if raw["missing_child"] != missing:
                raise InteractiveTreeRevisionV2Error(
                    "v2 generated split missing route changed"
                )
            node.update(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "base_threshold": base_threshold,
                    "missing_child": missing,
                    "left_child_id": _text(
                        raw["left_child_id"],
                        "node.left_child_id",
                    ),
                    "right_child_id": _text(
                        raw["right_child_id"],
                        "node.right_child_id",
                    ),
                    "direction_diagnostic": _json_object(
                        raw["direction_diagnostic"],
                        "node.direction_diagnostic",
                    ),
                }
            )
        nodes.append(node)
    if [item["node_id"] for item in nodes] != visible:
        raise InteractiveTreeRevisionV2Error(
            "v2 generated nodes do not match visible_node_ids"
        )
    by_id = {item["node_id"]: item for item in nodes}
    if len(by_id) != len(nodes) or root_id not in by_id:
        raise InteractiveTreeRevisionV2Error(
            "v2 generated node identity is invalid"
        )
    frontier_set = set(frontier)
    expected_visible: list[str] = []
    expected_frontier: list[str] = []

    def visit(node_id: str, expected_path: tuple[str, ...]) -> None:
        node = by_id.get(node_id)
        if node is None or tuple(node["path"]) != expected_path:
            raise InteractiveTreeRevisionV2Error(
                "v2 generated child topology is invalid"
            )
        expected_visible.append(node_id)
        if node_id in frontier_set:
            expected_frontier.append(node_id)
            return
        if node["kind"] != "split":
            raise InteractiveTreeRevisionV2Error(
                "v2 generated leaf is missing from the frontier"
            )
        visit(node["left_child_id"], (*expected_path, "left"))
        visit(node["right_child_id"], (*expected_path, "right"))

    visit(root_id, ())
    if expected_visible != visible or expected_frontier != frontier:
        raise InteractiveTreeRevisionV2Error(
            "v2 generated traversal order is not canonical"
        )
    expected_conditions = {
        root_id: deepcopy(by_id[root_id]["condition"])
    }

    def derive(node_id: str) -> None:
        node = by_id[node_id]
        if node_id in frontier_set:
            return
        left_clause, right_clause = _branch_clauses(
            node["feature"],
            node["threshold"],
            node["missing_child"],
        )
        for child_id, clause in (
            (node["left_child_id"], left_clause),
            (node["right_child_id"], right_clause),
        ):
            expression = (
                clause
                if node["path"] == []
                else {
                    "op": "and",
                    "args": (
                        [
                            *expected_conditions[node_id]["args"],
                            clause,
                        ]
                        if expected_conditions[node_id].get("op") == "and"
                        else [expected_conditions[node_id], clause]
                    ),
                }
            )
            expected_conditions[child_id] = canonicalize_expression(
                expression
            )
            derive(child_id)

    derive(root_id)
    for node in nodes:
        if node["condition"] != expected_conditions[node["node_id"]]:
            raise InteractiveTreeRevisionV2Error(
                "v2 generated node condition changed from its topology: "
                f"{node['node_id']}"
            )
    return {
        "tree_hash": tree_hash,
        "root_node_id": root_id,
        "visible_node_ids": visible,
        "frontier_node_ids": frontier,
        "nodes": nodes,
    }


def _require_edit_transition(
    edit: Mapping[str, Any],
    *,
    tree: Mapping[str, Any],
    current: Mapping[str, Any],
    source: Mapping[str, Any],
    parent: Mapping[str, Any] | None,
) -> None:
    operation = edit["operation"]
    node_id = edit["node_id"]
    current_node = current["node_by_id"].get(node_id)
    if operation == "auto_continue_subtree":
        _require_continuation_transition(
            edit,
            tree=tree,
            current=current,
        )
        return
    if current_node is None or current_node["kind"] != "split":
        raise InteractiveTreeRevisionV2Error(
            "v2 edit target is not a current visible split"
        )
    if node_id in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "v2 edit target is already a frontier node"
        )
    if operation == "prune_subtree":
        if parent is None or parent["schema_version"] != (
            INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 prune requires a v2 parent"
            )
        expected_frontier = _pruned_frontier_from_current(
            current,
            current_frontier=tuple(current["frontier"]),
            node_id=node_id,
        )
        expected_visible = _visible_node_ids_from_current(
            current,
            frontier=expected_frontier,
        )
        if (
            tuple(tree["frontier_node_ids"]) != expected_frontier
            or tuple(tree["visible_node_ids"]) != expected_visible
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 prune topology does not match its parent"
            )
        expected_nodes = [
            current["node_by_id"][item] for item in expected_visible
        ]
        if tree["nodes"] != expected_nodes:
            raise InteractiveTreeRevisionV2Error(
                "v2 prune changed effective node semantics"
            )
        return
    if (
        tuple(tree["visible_node_ids"]) != tuple(current["visible"])
        or tuple(tree["frontier_node_ids"]) != tuple(current["frontier"])
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 split edit changed visible topology"
        )
    previous = _finite_number(
        current_node["threshold"],
        "current threshold",
    )
    if edit["previous_threshold"] != previous:
        raise InteractiveTreeRevisionV2Error(
            "v2 split edit previous_threshold changed"
        )
    target = next(
        item for item in tree["nodes"] if item["node_id"] == node_id
    )
    if target["threshold"] != edit["threshold"]:
        raise InteractiveTreeRevisionV2Error(
            "v2 split edit does not match effective node"
        )
    if operation == "replace_split_feature":
        if edit["previous_feature"] != current_node["feature"]:
            raise InteractiveTreeRevisionV2Error(
                "v2 split edit previous_feature changed"
            )
        if target["feature"] != edit["feature"]:
            raise InteractiveTreeRevisionV2Error(
                "v2 split edit feature does not match effective node"
            )
    elif target["feature"] != current_node["feature"]:
        raise InteractiveTreeRevisionV2Error(
            "v2 threshold edit changed the effective feature"
        )
    target_path = tuple(current_node["path"])
    tree_by_id = {item["node_id"]: item for item in tree["nodes"]}
    for current_id in current["visible"]:
        current_item = current["node_by_id"][current_id]
        path = tuple(current_item["path"])
        descendant = (
            len(path) > len(target_path)
            and path[: len(target_path)] == target_path
        )
        if descendant or current_id == node_id:
            continue
        if tree_by_id[current_id] != current_item:
            raise InteractiveTreeRevisionV2Error(
                "v2 split edit changed a node outside the target subtree"
            )
    invariant_fields = [
        "node_id",
        "kind",
        "depth",
        "path",
        "metrics",
        "base_threshold",
        "left_child_id",
        "right_child_id",
    ]
    if operation != "replace_split_feature":
        invariant_fields.append("feature")
    for field in invariant_fields:
        if target[field] != current_node[field]:
            raise InteractiveTreeRevisionV2Error(
                f"v2 split edit changed target {field}"
            )


def _require_continuation_transition(
    edit: Mapping[str, Any],
    *,
    tree: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    node_id = edit["node_id"]
    current_node = current["node_by_id"].get(node_id)
    if current_node is None or node_id not in set(current["frontier"]):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation target is not a current frontier"
        )
    tree_by_id = {item["node_id"]: item for item in tree["nodes"]}
    target = tree_by_id.get(node_id)
    if (
        target is None
        or target["kind"] != "split"
        or target["feature"] != edit["feature"]
        or target["threshold"] != edit["threshold"]
        or target["missing_child"] != edit["missing_child"]
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation root changed from the selected candidate"
        )
    for field in ("node_id", "depth", "path", "condition", "metrics"):
        if target[field] != current_node[field]:
            raise InteractiveTreeRevisionV2Error(
                f"v2 continuation changed target {field}"
            )
    expected_base_threshold = (
        float(current_node["base_threshold"])
        if current_node["kind"] == "split"
        else edit["threshold"]
    )
    if target["base_threshold"] != expected_base_threshold:
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation root base threshold changed"
        )
    target_path = tuple(current_node["path"])
    expected_outside_frontier = [
        item
        for item in current["frontier"]
        if tuple(current["node_by_id"][item]["path"])[: len(target_path)]
        != target_path
    ]
    new_subtree_frontier = [
        item
        for item in tree["frontier_node_ids"]
        if tuple(tree_by_id[item]["path"])[: len(target_path)] == target_path
    ]
    if (
        not new_subtree_frontier
        or list(tree["frontier_node_ids"]) != sorted(
            [*expected_outside_frontier, *new_subtree_frontier],
            key=lambda item: _path_sort_key(tree_by_id[item]["path"]),
        )
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation frontier changed outside its target subtree"
        )
    for current_id in current["visible"]:
        current_item = current["node_by_id"][current_id]
        path = tuple(current_item["path"])
        inside = (
            len(path) >= len(target_path)
            and path[: len(target_path)] == target_path
        )
        if inside:
            continue
        if tree_by_id.get(current_id) != current_item:
            raise InteractiveTreeRevisionV2Error(
                "v2 continuation changed a node outside its target subtree"
            )
    replay = edit["replay"]
    generated = [
        item
        for item in tree["nodes"]
        if tuple(item["path"])[: len(target_path)] == target_path
    ]
    current_ids = set(current["node_by_id"])
    for item in generated:
        if item["node_id"] == node_id:
            continue
        if (
            item["node_id"] in current_ids
            or not item["node_id"].startswith("node-")
            or (
                item["kind"] == "split"
                and item["base_threshold"] != item["threshold"]
            )
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 continuation generated node identity changed"
            )
    if (
        replay["visible_node_count"] != len(tree["visible_node_ids"])
        or replay["frontier_count"] != len(tree["frontier_node_ids"])
        or replay["observed"]["generated_node_count"] != len(generated)
        or replay["observed"]["generated_split_count"]
        != sum(item["kind"] == "split" for item in generated)
        or replay["observed"]["generated_leaf_count"]
        != len(new_subtree_frontier)
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation replay counts changed from its topology"
        )


def _derive_fragments(
    source: Mapping[str, Any],
    *,
    tree: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {item["node_id"]: item for item in tree["nodes"]}
    fragments = []
    for node_id in tree["frontier_node_ids"]:
        node = by_id[node_id]
        condition = deepcopy(node["condition"])
        metrics = deepcopy(node["metrics"])
        semantic = {
            "schema_version": "strategy.interactive-tree-fragment.v2",
            "base_tree_result_hash": source["tree_result"]["result_hash"],
            "source_node_id": node_id,
            "condition": condition,
            "metrics": metrics,
        }
        leaf_id = f"interactive-leaf-{_sha256(semantic)[:32]}"
        rule_id = f"candidate-rule-{_sha256(condition)[:32]}"
        effect_id = (
            "candidate-effect-"
            + _sha256(
                {
                    "schema_version": "strategy.interactive-tree-effect.v2",
                    "condition": condition,
                    "metrics": metrics,
                }
            )[:32]
        )
        core = {
            "source_node_id": node_id,
            "leaf_id": leaf_id,
            "rule_id": rule_id,
            "condition": condition,
            "requirements": [],
            "effect_id": effect_id,
            "metrics": metrics,
        }
        fragment_id = f"candidate-fragment-{_sha256(core)[:32]}"
        without_hash = {**core, "fragment_id": fragment_id}
        fragments.append(
            {**without_hash, "fragment_hash": _sha256(without_hash)}
        )
    return fragments


def _semantic_body(
    source: Mapping[str, Any],
    *,
    tree: Mapping[str, Any],
    fragments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "strategy.interactive-tree-semantic.v2",
        "base_tree": _base_tree(source),
        "effective_tree": {
            key: deepcopy(tree[key])
            for key in (
                "root_node_id",
                "visible_node_ids",
                "frontier_node_ids",
                "nodes",
            )
        },
        "fragments": deepcopy(fragments),
    }


def _candidate_evidence(semantic_body: Mapping[str, Any]) -> dict[str, str]:
    evidence_hash = _sha256(
        {
            "schema_version": "strategy.interactive-tree-evidence.v2",
            "semantic_tree": semantic_body,
        }
    )
    return {
        "candidate_id": f"candidate-{evidence_hash[:32]}",
        "evidence_hash": evidence_hash,
    }


def _conditions_from_configs(
    source: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    source_by_id = {
        item["node_id"]: item
        for item in source["tree_result"]["tree"]["nodes"]
    }
    medians = source["tree_result"]["preprocessing"]["medians"]
    root_id = source["tree_result"]["tree"]["root_node_id"]
    conditions: dict[str, dict[str, Any]] = {}

    def visit(node_id: str, clauses: tuple[dict[str, Any], ...]) -> None:
        node = source_by_id[node_id]
        if clauses:
            expression: dict[str, Any] = (
                clauses[0]
                if len(clauses) == 1
                else {"op": "and", "args": list(clauses)}
            )
        else:
            config = configs[node_id]
            feature = _text(config["feature"], "split feature")
            threshold = _finite_number(
                config["threshold"],
                "split threshold",
            )
            missing = (
                "left"
                if float(medians[feature]) <= threshold
                else "right"
            )
            left, right = _branch_clauses(
                feature,
                threshold,
                missing,
            )
            expression = {"op": "or", "args": [left, right]}
        conditions[node_id] = canonicalize_expression(expression)
        if node["kind"] == "leaf":
            return
        config = configs[node_id]
        feature = _text(config["feature"], "split feature")
        threshold = _finite_number(config["threshold"], "split threshold")
        missing = (
            "left"
            if float(medians[feature]) <= threshold
            else "right"
        )
        left, right = _branch_clauses(feature, threshold, missing)
        visit(node["left_child_id"], (*clauses, left))
        visit(node["right_child_id"], (*clauses, right))

    visit(root_id, ())
    return conditions


def _branch_clauses(
    feature: str,
    threshold: float,
    missing_child: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "op": "compare",
            "field": feature,
            "operator": "<=",
            "value": threshold,
            "missing": "match" if missing_child == "left" else "no_match",
        },
        {
            "op": "compare",
            "field": feature,
            "operator": ">",
            "value": threshold,
            "missing": "match" if missing_child == "right" else "no_match",
        },
    )


def _visible_node_ids(
    source: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
) -> tuple[str, ...]:
    by_id = {
        item["node_id"]: item
        for item in source["tree_result"]["tree"]["nodes"]
    }
    frontier_set = set(frontier)
    visible: list[str] = []

    def visit(node_id: str) -> None:
        visible.append(node_id)
        if node_id in frontier_set:
            return
        node = by_id[node_id]
        if node["kind"] != "split":
            raise InteractiveTreeRevisionV2Error(
                "v2 frontier does not cover the topology"
            )
        visit(node["left_child_id"])
        visit(node["right_child_id"])

    visit(source["tree_result"]["tree"]["root_node_id"])
    return tuple(visible)


def _pruned_frontier(
    source: Mapping[str, Any],
    *,
    current_frontier: tuple[str, ...],
    node_id: str,
) -> tuple[str, ...]:
    nodes = source["tree_result"]["tree"]["nodes"]
    by_id = {item["node_id"]: item for item in nodes}
    path = tuple(by_id[node_id]["path"])
    descendants = {
        item
        for item in current_frontier
        if tuple(by_id[item]["path"])[: len(path)] == path
    }
    if not descendants:
        raise InteractiveTreeRevisionV2Error(
            "v2 prune does not cover any current frontier node"
        )
    selected = (set(current_frontier) - descendants) | {node_id}
    return tuple(item["node_id"] for item in nodes if item["node_id"] in selected)


def _pruned_frontier_from_current(
    current: Mapping[str, Any],
    *,
    current_frontier: tuple[str, ...],
    node_id: str,
) -> tuple[str, ...]:
    path = tuple(current["node_by_id"][node_id]["path"])
    descendants = {
        item
        for item in current_frontier
        if tuple(current["node_by_id"][item]["path"])[: len(path)] == path
    }
    if not descendants:
        raise InteractiveTreeRevisionV2Error(
            "v2 prune does not cover any current frontier node"
        )
    selected = (set(current_frontier) - descendants) | {node_id}
    return tuple(
        item
        for item in current["visible"]
        if item in selected
    )


def _visible_node_ids_from_current(
    current: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
) -> tuple[str, ...]:
    by_id = current["node_by_id"]
    frontier_set = set(frontier)
    visible: list[str] = []
    root_id = next(
        item["node_id"]
        for item in current["nodes"]
        if item["path"] == []
    )

    def visit(node_id: str) -> None:
        visible.append(node_id)
        if node_id in frontier_set:
            return
        node = by_id[node_id]
        if node["kind"] != "split":
            raise InteractiveTreeRevisionV2Error(
                "v2 frontier does not cover the current topology"
            )
        visit(node["left_child_id"])
        visit(node["right_child_id"])

    visit(root_id)
    return tuple(visible)


def _require_frontier_cover(
    source: Mapping[str, Any],
    frontier: tuple[str, ...],
) -> None:
    leaf_ids = source["tree_result"]["tree"]["leaf_ids"]
    by_id = {
        item["node_id"]: item
        for item in source["tree_result"]["tree"]["nodes"]
    }
    if not frontier or len(frontier) != len(set(frontier)):
        raise InteractiveTreeRevisionV2Error("v2 frontier is invalid")
    for leaf_id in leaf_ids:
        leaf_path = tuple(by_id[leaf_id]["path"])
        owners = [
            item
            for item in frontier
            if leaf_path[: len(by_id[item]["path"])]
            == tuple(by_id[item]["path"])
        ]
        if len(owners) != 1:
            raise InteractiveTreeRevisionV2Error(
                "v2 frontier does not cover each base leaf exactly once"
            )


def _require_frontier_cover_from_tree(tree: Mapping[str, Any]) -> None:
    by_id = {item["node_id"]: item for item in tree["nodes"]}
    frontier = tuple(tree["frontier_node_ids"])
    if not frontier or len(frontier) != len(set(frontier)):
        raise InteractiveTreeRevisionV2Error("v2 frontier is invalid")
    visited: list[str] = []

    def visit(node_id: str) -> None:
        node = by_id.get(node_id)
        if node is None:
            raise InteractiveTreeRevisionV2Error(
                "v2 frontier references an absent node"
            )
        if node_id in set(frontier):
            visited.append(node_id)
            return
        if node["kind"] != "split":
            raise InteractiveTreeRevisionV2Error(
                "v2 frontier does not cover the effective topology"
            )
        visit(node["left_child_id"])
        visit(node["right_child_id"])

    visit(tree["root_node_id"])
    if tuple(visited) != frontier:
        raise InteractiveTreeRevisionV2Error(
            "v2 frontier traversal is not canonical"
        )


def _path_sort_key(value: Sequence[str]) -> tuple[int, ...]:
    return tuple(0 if item == "left" else 1 for item in value)


def _require_metric_conservation(tree: Mapping[str, Any]) -> None:
    by_id = {item["node_id"]: item for item in tree["nodes"]}
    root = by_id[tree["root_node_id"]]["metrics"]
    rows = [by_id[item]["metrics"] for item in tree["frontier_node_ids"]]
    for basis in ("unweighted", "weighted"):
        if root[basis] == {"status": "not_applicable"}:
            if any(item[basis] != root[basis] for item in rows):
                raise InteractiveTreeRevisionV2Error(
                    "v2 frontier metric applicability changed"
                )
            continue
        for field in (
            "total",
            "good",
            "bad",
            "loan_amount_total",
            "loan_amount_coverage_count",
            "loan_amount_coverage",
            "overdue_amount_total",
            "overdue_amount_coverage_count",
            "overdue_amount_coverage",
            "amount_pair_coverage_count",
            "amount_pair_coverage",
            "paired_loan_amount_total",
            "paired_overdue_amount_total",
        ):
            if field not in root[basis]:
                continue
            actual = sum(float(item[basis][field]) for item in rows)
            expected = float(root[basis][field])
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
                raise InteractiveTreeRevisionV2Error(
                    f"v2 frontier does not conserve {basis}.{field}"
                )


def _require_split_diagnostics(
    tree: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
) -> None:
    by_id = {item["node_id"]: item for item in tree["nodes"]}
    directions = source["tree_result"]["directions"]
    frontier = set(tree["frontier_node_ids"])
    for node in tree["nodes"]:
        if node["kind"] != "split":
            continue
        diagnostic = node["direction_diagnostic"]
        expected_fields = {
            "expected_direction",
            "status",
            "basis",
            "primary_bad_rate_delta",
            "left",
            "right",
        }
        _exact_fields(
            diagnostic,
            expected_fields,
            f"v2 split diagnostic {node['node_id']}",
        )
        direction = directions[node["feature"]]
        if diagnostic["expected_direction"] != direction:
            raise InteractiveTreeRevisionV2Error(
                "v2 split diagnostic direction changed"
            )
        weight_available = node["metrics"]["weighted"] != {
            "status": "not_applicable"
        }
        basis = "weighted" if weight_available else "unweighted"
        if diagnostic["basis"] != basis:
            raise InteractiveTreeRevisionV2Error(
                "v2 split diagnostic basis changed"
            )
        counts = []
        bads = []
        weighted_totals = []
        weighted_bads = []
        primary_rates = []
        for side in ("left", "right"):
            value = diagnostic[side]
            _exact_fields(
                value,
                {"count", "bad_rate", "weighted"},
                f"v2 split diagnostic {side}",
            )
            count = value["count"]
            rate = _finite_number(
                value["bad_rate"],
                f"diagnostic.{side}.bad_rate",
            )
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                or not 0 <= rate <= 1
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 split diagnostic count or bad rate is invalid"
                )
            counts.append(count)
            bads.append(count * rate)
            weighted = value["weighted"]
            if weight_available:
                _exact_fields(
                    weighted,
                    {"status", "total", "bad_rate"},
                    f"v2 split diagnostic {side} weighted",
                )
                weighted_total = _finite_number(
                    weighted["total"],
                    f"diagnostic.{side}.weighted.total",
                )
                weighted_rate = _finite_number(
                    weighted["bad_rate"],
                    f"diagnostic.{side}.weighted.bad_rate",
                )
                if (
                    weighted["status"] != "available"
                    or weighted_total <= 0
                    or not 0 <= weighted_rate <= 1
                ):
                    raise InteractiveTreeRevisionV2Error(
                        "v2 weighted split diagnostic is invalid"
                    )
                weighted_totals.append(weighted_total)
                weighted_bads.append(weighted_total * weighted_rate)
                primary_rates.append(weighted_rate)
            else:
                if weighted != {"status": "not_applicable"}:
                    raise InteractiveTreeRevisionV2Error(
                        "v2 split diagnostic weighted applicability changed"
                    )
                primary_rates.append(rate)
        node_unweighted = node["metrics"]["unweighted"]
        if (
            sum(counts) != int(node_unweighted["total"])
            or not math.isclose(
                sum(bads),
                float(node_unweighted["bad"]),
                rel_tol=1e-10,
                abs_tol=1e-9,
            )
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 split diagnostic does not conserve node counts"
            )
        if weight_available:
            node_weighted = node["metrics"]["weighted"]
            if (
                not math.isclose(
                    sum(weighted_totals),
                    float(node_weighted["total"]),
                    rel_tol=1e-10,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    sum(weighted_bads),
                    float(node_weighted["bad"]),
                    rel_tol=1e-10,
                    abs_tol=1e-9,
                )
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 split diagnostic does not conserve weighted metrics"
                )
        delta = _finite_number(
            diagnostic["primary_bad_rate_delta"],
            "diagnostic.primary_bad_rate_delta",
        )
        left_rate, right_rate = primary_rates
        if not math.isclose(
            delta,
            right_rate - left_rate,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 split diagnostic delta is invalid"
            )
        expected_status = (
            "inconclusive"
            if direction == "unordered"
            else (
                "consistent"
                if (
                    right_rate >= left_rate
                    if direction == "increasing"
                    else right_rate <= left_rate
                )
                else "violation"
            )
        )
        if diagnostic["status"] != expected_status:
            raise InteractiveTreeRevisionV2Error(
                "v2 split diagnostic status is invalid"
            )
        if node["node_id"] in frontier:
            continue
        left = by_id[node["left_child_id"]]["metrics"]
        right = by_id[node["right_child_id"]]["metrics"]
        for index, metrics in enumerate((left, right)):
            if (
                counts[index] != int(metrics["unweighted"]["total"])
                or not math.isclose(
                    bads[index],
                    float(metrics["unweighted"]["bad"]),
                    rel_tol=1e-10,
                    abs_tol=1e-9,
                )
            ):
                raise InteractiveTreeRevisionV2Error(
                    "v2 split diagnostic changed from visible child metrics"
                )


def _checks(warning_codes: object) -> dict[str, Any]:
    if not isinstance(warning_codes, list):
        raise InteractiveTreeRevisionV2Error("warning_codes must be a list")
    if (
        any(not isinstance(item, str) or item not in _WARNING_CODES for item in warning_codes)
        or len(warning_codes) != len(set(warning_codes))
    ):
        raise InteractiveTreeRevisionV2Error("warning_codes are invalid")
    return {
        "frontier_prefix_free": True,
        "all_base_leaves_covered_once": True,
        "fragment_source_matches": True,
        "metric_conservation": "passed",
        "all_visible_metrics_replayed": True,
        "all_visible_split_diagnostics_replayed": True,
        "frontier_conditions_evaluator_equivalent": True,
        "minimum_leaf_constraints": "passed",
        "warning_codes": sorted(warning_codes),
    }


def _normalize_checks(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error("v2 checks must be an object")
    expected = _checks(value.get("warning_codes"))
    _exact_fields(value, set(expected), "interactive-tree v2 checks")
    for field, expected_value in expected.items():
        if field == "warning_codes":
            continue
        if value[field] != expected_value:
            raise InteractiveTreeRevisionV2Error(
                f"v2 deterministic check failed: {field}"
            )
    return expected


def _normalize_edit(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error("v2 edit must be an object")
    operation = value.get("operation")
    expected_fields = {
        "operation",
        "node_id",
        "previous_threshold",
        "threshold",
        "reason",
    }
    if operation == "replace_split_feature":
        expected_fields |= {"previous_feature", "feature"}
    elif operation == "auto_continue_subtree":
        expected_fields = {
            "operation",
            "node_id",
            "search_id",
            "search_hash",
            "candidate_id",
            "feature",
            "threshold",
            "missing_child",
            "objective",
            "tie_break",
            "controls",
            "replay",
            "reason",
        }
    _exact_fields(
        value,
        expected_fields,
        "interactive-tree v2 edit",
    )
    if operation not in {
        "adjust_split_threshold",
        "replace_split_feature",
        "prune_subtree",
        "auto_continue_subtree",
    }:
        raise InteractiveTreeRevisionV2Error("v2 edit operation is invalid")
    if operation == "auto_continue_subtree":
        candidate_id = _text(value["candidate_id"], "edit.candidate_id")
        normalized = {
            "operation": operation,
            "node_id": _text(value["node_id"], "edit.node_id"),
            "search_id": _text(value["search_id"], "edit.search_id"),
            "search_hash": _hash(value["search_hash"], "edit.search_hash"),
            "candidate_id": candidate_id,
            "feature": _text(value["feature"], "edit.feature"),
            "threshold": _finite_number(
                value["threshold"],
                "edit.threshold",
            ),
            "missing_child": _missing_child(value["missing_child"]),
            "objective": _text(value["objective"], "edit.objective"),
            "tie_break": _text(value["tie_break"], "edit.tie_break"),
            "controls": _normalize_continuation_controls(value["controls"]),
            "replay": _normalize_continuation_replay(
                value["replay"],
                node_id=_text(value["node_id"], "edit.node_id"),
                candidate_id=candidate_id,
            ),
            "reason": _reason(value["reason"]),
        }
        if (
            normalized["objective"] != _CONTINUATION_OBJECTIVE
            or normalized["tie_break"] != _CONTINUATION_TIE_BREAK
            or normalized["replay"]["objective"]
            != normalized["objective"]
            or normalized["replay"]["tie_break"]
            != normalized["tie_break"]
            or normalized["replay"]["controls"]
            != normalized["controls"]
        ):
            raise InteractiveTreeRevisionV2Error(
                "v2 continuation decision policy changed"
            )
        return normalized
    if operation in {
        "adjust_split_threshold",
        "replace_split_feature",
    }:
        previous = _finite_number(
            value["previous_threshold"],
            "edit.previous_threshold",
        )
        threshold = _finite_number(value["threshold"], "edit.threshold")
        if operation == "adjust_split_threshold" and threshold == previous:
            raise InteractiveTreeRevisionV2Error(
                "v2 threshold edit is a no-op"
            )
        if operation == "replace_split_feature":
            previous_feature = _text(
                value["previous_feature"],
                "edit.previous_feature",
            )
            feature = _text(value["feature"], "edit.feature")
            if feature == previous_feature:
                raise InteractiveTreeRevisionV2Error(
                    "v2 split feature edit is a no-op"
                )
        else:
            previous_feature = None
            feature = None
    else:
        if value["previous_threshold"] is not None or value["threshold"] is not None:
            raise InteractiveTreeRevisionV2Error(
                "v2 prune cannot carry threshold values"
            )
        previous = None
        threshold = None
        previous_feature = None
        feature = None
    normalized = {
        "operation": operation,
        "node_id": _text(value["node_id"], "edit.node_id"),
        "previous_threshold": previous,
        "threshold": threshold,
        "reason": _reason(value["reason"]),
    }
    if operation == "replace_split_feature":
        normalized["previous_feature"] = previous_feature
        normalized["feature"] = feature
    return normalized


def _normalize_continuation_controls(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation controls must be an object"
        )
    _exact_fields(
        value,
        {
            "features",
            "max_additional_depth",
            "min_gini_gain",
            "max_generated_nodes",
            "max_thresholds_per_feature",
            "max_row_evaluations",
        },
        "v2 continuation controls",
    )
    features = _text_sequence(value["features"], "controls.features")
    if (
        not features
        or len(features) > 50
        or features != sorted(features)
        or len(features) != len(set(features))
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation features are not canonical"
        )
    minimum_gain = _finite_number(
        value["min_gini_gain"],
        "controls.min_gini_gain",
    )
    if not 0 <= minimum_gain <= 0.5:
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation min_gini_gain is invalid"
        )
    nodes = _bounded_positive_int(
        value["max_generated_nodes"],
        "controls.max_generated_nodes",
        maximum=127,
    )
    if nodes < 3:
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation requires at least three generated nodes"
        )
    return {
        "features": features,
        "max_additional_depth": _bounded_positive_int(
            value["max_additional_depth"],
            "controls.max_additional_depth",
            maximum=6,
        ),
        "min_gini_gain": minimum_gain,
        "max_generated_nodes": nodes,
        "max_thresholds_per_feature": _bounded_positive_int(
            value["max_thresholds_per_feature"],
            "controls.max_thresholds_per_feature",
            maximum=20,
        ),
        "max_row_evaluations": _bounded_positive_int(
            value["max_row_evaluations"],
            "controls.max_row_evaluations",
            maximum=20_000_000,
        ),
    }


def _normalize_continuation_replay(
    value: object,
    *,
    node_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation replay must be an object"
        )
    expected_fields = {
        "schema_version",
        "producer_version",
        "source_tree_id",
        "node_id",
        "seed_candidate_id",
        "objective",
        "tie_break",
        "controls",
        "observed",
        "source_row_count",
        "visible_node_count",
        "frontier_count",
        "exactly_once",
        "current_tree_replayed",
        "minimum_leaf_constraints_passed",
        "frontier_conditions_evaluator_equivalent",
        "assignment_hash",
        "result_hash",
    }
    _exact_fields(value, expected_fields, "v2 continuation replay")
    observed = value["observed"]
    if not isinstance(observed, Mapping):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation observed evidence must be an object"
        )
    _exact_fields(
        observed,
        {
            "generated_node_count",
            "generated_split_count",
            "generated_leaf_count",
            "row_evaluations",
            "stop_reasons",
        },
        "v2 continuation observed evidence",
    )
    stop_reasons = observed["stop_reasons"]
    if (
        not isinstance(stop_reasons, Mapping)
        or any(
            not isinstance(key, str)
            or not key
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for key, count in stop_reasons.items()
        )
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation stop reasons are invalid"
        )
    controls = _normalize_continuation_controls(value["controls"])
    normalized_observed = {
        "generated_node_count": _bounded_positive_int(
            observed["generated_node_count"],
            "observed.generated_node_count",
            maximum=controls["max_generated_nodes"],
        ),
        "generated_split_count": _bounded_positive_int(
            observed["generated_split_count"],
            "observed.generated_split_count",
            maximum=controls["max_generated_nodes"],
        ),
        "generated_leaf_count": _bounded_positive_int(
            observed["generated_leaf_count"],
            "observed.generated_leaf_count",
            maximum=controls["max_generated_nodes"],
        ),
        "row_evaluations": _bounded_nonnegative_int(
            observed["row_evaluations"],
            "observed.row_evaluations",
            maximum=controls["max_row_evaluations"],
        ),
        "stop_reasons": {
            key: int(stop_reasons[key])
            for key in sorted(stop_reasons)
        },
    }
    normalized = {
        "schema_version": value["schema_version"],
        "producer_version": value["producer_version"],
        "source_tree_id": _text(
            value["source_tree_id"],
            "replay.source_tree_id",
        ),
        "node_id": _text(value["node_id"], "replay.node_id"),
        "seed_candidate_id": _text(
            value["seed_candidate_id"],
            "replay.seed_candidate_id",
        ),
        "objective": _text(value["objective"], "replay.objective"),
        "tie_break": _text(value["tie_break"], "replay.tie_break"),
        "controls": controls,
        "observed": normalized_observed,
        "source_row_count": _bounded_positive_int(
            value["source_row_count"],
            "replay.source_row_count",
            maximum=2_147_483_647,
        ),
        "visible_node_count": _bounded_positive_int(
            value["visible_node_count"],
            "replay.visible_node_count",
            maximum=_MAX_VISIBLE_NODES,
        ),
        "frontier_count": _bounded_positive_int(
            value["frontier_count"],
            "replay.frontier_count",
            maximum=_MAX_VISIBLE_NODES,
        ),
        "exactly_once": value["exactly_once"],
        "current_tree_replayed": value["current_tree_replayed"],
        "minimum_leaf_constraints_passed": value[
            "minimum_leaf_constraints_passed"
        ],
        "frontier_conditions_evaluator_equivalent": value[
            "frontier_conditions_evaluator_equivalent"
        ],
        "assignment_hash": _hash(
            value["assignment_hash"],
            "replay.assignment_hash",
        ),
    }
    if (
        normalized["schema_version"] != _CONTINUATION_REPLAY_SCHEMA
        or normalized["producer_version"]
        != "strategy.interactive-tree-continuation-replay/1"
        or normalized["node_id"] != node_id
        or normalized["seed_candidate_id"] != candidate_id
        or normalized["objective"] != _CONTINUATION_OBJECTIVE
        or normalized["tie_break"] != _CONTINUATION_TIE_BREAK
        or any(
            normalized[field] is not True
            for field in (
                "exactly_once",
                "current_tree_replayed",
                "minimum_leaf_constraints_passed",
                "frontier_conditions_evaluator_equivalent",
            )
        )
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation replay claims changed"
        )
    if (
        normalized_observed["generated_split_count"]
        + normalized_observed["generated_leaf_count"]
        != normalized_observed["generated_node_count"]
    ):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation replay node counts do not conserve"
        )
    result = {
        **normalized,
        "result_hash": _hash(value["result_hash"], "replay.result_hash"),
    }
    if result["result_hash"] != _sha256(normalized):
        raise InteractiveTreeRevisionV2Error(
            "v2 continuation replay hash changed"
        )
    return result


def _missing_child(value: object) -> str:
    if value not in {"left", "right"}:
        raise InteractiveTreeRevisionV2Error(
            "v2 split missing_child is invalid"
        )
    return str(value)


def _bounded_positive_int(
    value: object,
    name: str,
    *,
    maximum: int,
) -> int:
    result = _bounded_nonnegative_int(value, name, maximum=maximum)
    if result < 1:
        raise InteractiveTreeRevisionV2Error(f"{name} must be positive")
    return result


def _bounded_nonnegative_int(
    value: object,
    name: str,
    *,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise InteractiveTreeRevisionV2Error(f"{name} is outside its bounds")
    return value


def _validated_parent(
    parent: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    *,
    ancestor_revisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if parent is None:
        if ancestor_revisions:
            raise InteractiveTreeRevisionV2Error(
                "ancestor evidence requires a parent revision"
            )
        return None
    if not isinstance(parent, Mapping):
        raise InteractiveTreeRevisionV2Error("parent revision must be an object")
    ancestors = tuple(ancestor_revisions)
    direct_parent = ancestors[0] if ancestors else None
    remaining = ancestors[1:] if ancestors else ()
    if parent.get("schema_version") == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION:
        return validate_interactive_tree_revision_v2(
            parent,
            source,
            parent_revision=direct_parent,
            ancestor_revisions=remaining,
        )
    from marvis.packs.strategy.interactive_tree_revision import (
        validate_interactive_tree_revision,
    )

    return validate_interactive_tree_revision(
        parent,
        source,
        parent_revision=direct_parent,
        ancestor_revisions=remaining,
    )


def _base_tree(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": source["asset_id"],
        "asset_hash": source["asset_hash"],
        "tree_id": source["tree_result"]["tree"]["tree_id"],
        "tree_result_hash": source["tree_result"]["result_hash"],
    }


def _parent_ref(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": parent["schema_version"],
        "revision_id": parent["revision_id"],
        "revision_hash": parent["revision_hash"],
        "semantic_tree_id": parent["semantic_tree_id"],
        "tree_hash": parent["tree"]["tree_hash"],
    }


def _source_refs(
    source: Mapping[str, Any],
    *,
    parent: Mapping[str, Any] | None,
) -> list[str]:
    refs = [
        *source["source_refs"],
        (
            f"automatic-tree-asset:{source['asset_id']}"
            f"@sha256:{source['asset_hash']}"
        ),
    ]
    if parent is not None:
        refs.append(
            f"interactive-tree-revision:{parent['revision_id']}"
            f"@sha256:{parent['revision_hash']}"
        )
    return refs


def _text_sequence(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise InteractiveTreeRevisionV2Error(f"{name} must be a list")
    result = [_text(item, name) for item in value]
    if len(result) != len(set(result)):
        raise InteractiveTreeRevisionV2Error(f"{name} must be unique")
    return result


def _json_object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionV2Error(f"{name} must be an object")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise InteractiveTreeRevisionV2Error(
            f"{name} must be canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise InteractiveTreeRevisionV2Error(f"{name} must be an object")
    return decoded


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InteractiveTreeRevisionV2Error(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise InteractiveTreeRevisionV2Error(f"{name} must be a finite number")
    return result


def _reason(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(_text(value, "reason").split())
    if len(text) > 500:
        raise InteractiveTreeRevisionV2Error("reason exceeds 500 characters")
    return text


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InteractiveTreeRevisionV2Error(f"{name} must be non-empty text")
    return value.strip()


def _hash(value: object, name: str) -> str:
    text = _text(value, name)
    if _HASH_RE.fullmatch(text) is None:
        raise InteractiveTreeRevisionV2Error(f"{name} must be a sha256 hash")
    return text


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise InteractiveTreeRevisionV2Error(
            f"{name} fields are invalid: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
