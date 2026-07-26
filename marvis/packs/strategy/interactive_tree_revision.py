"""Immutable interactive-tree prune revisions over one automatic-tree asset.

The automatic weighted-tree asset remains the owner of fitted topology,
root-to-node routing and measured node effects.  An interactive revision stores
only a canonical frontier over that authenticated topology.  Pruning a visible
split node replaces all of its descendant frontier leaves with the node itself;
no metric is recomputed or accepted from a caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import hmac
import json
import re
from typing import Any

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError


INTERACTIVE_TREE_REVISION_SCHEMA_VERSION = (
    "strategy.interactive-tree-revision.v1"
)
INTERACTIVE_TREE_REVISION_PRODUCER_VERSION = (
    "strategy.interactive-tree-revision/1"
)
INTERACTIVE_TREE_ASSET_TYPE = "interactive_rule_tree"
MAX_INTERACTIVE_TREE_NODES = 511
MAX_EDIT_REASON_LENGTH = 500

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_ID_RE = re.compile(
    r"^interactive-tree-revision-[0-9a-f]{32}$"
)
_SEMANTIC_TREE_ID_RE = re.compile(r"^interactive-tree-[0-9a-f]{32}$")
_LEAF_ID_RE = re.compile(r"^interactive-leaf-[0-9a-f]{32}$")
_FRAGMENT_ID_RE = re.compile(r"^candidate-fragment-[0-9a-f]{32}$")
_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_EFFECT_ID_RE = re.compile(r"^candidate-effect-[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")

_LIFECYCLE = {
    "candidate_stage": "development",
    "observation_stage": "backtested",
    "validation_status": "unvalidated",
}
_CHECKS = {
    "frontier_prefix_free": True,
    "all_base_leaves_covered_once": True,
    "fragment_source_matches": True,
    "metric_conservation": "passed",
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


class InteractiveTreeRevisionError(StrategyError):
    """An immutable interactive-tree revision is invalid."""


def build_interactive_tree_revision(
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    reason: str | None = None,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create one immutable ``prune_subtree`` revision.

    ``node_id`` must identify a split node that is currently visible.  Hidden
    descendants and already-materialized frontier nodes cannot be pruned again.
    """

    source = validate_automatic_tree_asset(automatic_tree_asset)
    source_index = _source_index(source)
    normalized_node_id = _text(node_id, "node_id")
    normalized_reason = _reason(reason)
    ancestors = _ancestor_chain(ancestor_revisions)

    parent = None
    if parent_revision is not None:
        parent = validate_interactive_tree_revision(
            parent_revision,
            source,
            parent_revision=ancestors[0] if ancestors else None,
            ancestor_revisions=ancestors[1:],
        )
        _require_same_base(parent, source)
        current_frontier = tuple(parent["tree"]["frontier_node_ids"])
        parent_ref = _parent_projection(parent)
    else:
        if ancestors:
            raise InteractiveTreeRevisionError(
                "interactive-tree ancestor evidence requires a parent revision"
            )
        current_frontier = tuple(source_index["leaf_ids"])
        parent_ref = None

    current_visible = _visible_node_ids(
        source_index,
        frontier=current_frontier,
    )
    node = source_index["node_by_id"].get(normalized_node_id)
    if node is None:
        raise InteractiveTreeRevisionError(
            "interactive-tree prune node does not exist in the base tree"
        )
    if node["kind"] != "split":
        raise InteractiveTreeRevisionError(
            "interactive-tree prune requires a split node"
        )
    if normalized_node_id not in current_visible:
        raise InteractiveTreeRevisionError(
            "interactive-tree prune node is hidden by the current frontier"
        )
    if normalized_node_id in current_frontier:
        raise InteractiveTreeRevisionError(
            "interactive-tree prune node is already a frontier leaf"
        )

    frontier = _pruned_frontier(
        source_index,
        frontier=current_frontier,
        node_id=normalized_node_id,
    )
    payload = _assemble_revision(
        source,
        source_index=source_index,
        frontier=frontier,
        parent_ref=parent_ref,
        node_id=normalized_node_id,
        reason=normalized_reason,
    )
    return validate_interactive_tree_revision(
        payload,
        source,
        parent_revision=parent,
        ancestor_revisions=ancestors,
    )


def validate_interactive_tree_revision(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate one revision against its authenticated automatic-tree source."""

    if not isinstance(payload, Mapping):
        raise InteractiveTreeRevisionError(
            "interactive-tree revision must be an object"
        )
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "interactive-tree revision")
    source = validate_automatic_tree_asset(automatic_tree_asset)
    source_index = _source_index(source)
    ancestors = _ancestor_chain(ancestor_revisions)

    if payload["schema_version"] != INTERACTIVE_TREE_REVISION_SCHEMA_VERSION:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision schema_version is invalid"
        )
    if payload["producer_version"] != INTERACTIVE_TREE_REVISION_PRODUCER_VERSION:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision producer_version is invalid"
        )
    if payload["asset_type"] != INTERACTIVE_TREE_ASSET_TYPE:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision asset_type is invalid"
        )
    if payload["lifecycle"] != _LIFECYCLE:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision lifecycle is invalid"
        )
    if payload["identity"] != source["identity"]:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision identity changed from the base tree"
        )
    expected_base = _base_tree_projection(source)
    if payload["base_tree"] != expected_base:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision base tree binding changed"
        )

    edit = _normalize_edit(payload["edit"])
    if payload["edit"] != edit:
        raise InteractiveTreeRevisionError(
            "interactive-tree edit is not canonical"
        )
    parent_ref = _normalize_parent_ref(payload["parent_revision"])
    if payload["parent_revision"] != parent_ref:
        raise InteractiveTreeRevisionError(
            "interactive-tree parent_revision is not canonical"
        )
    parent = None
    if parent_revision is not None:
        parent = validate_interactive_tree_revision(
            parent_revision,
            source,
            parent_revision=ancestors[0] if ancestors else None,
            ancestor_revisions=ancestors[1:],
        )
        _require_same_base(parent, source)
        if parent_ref != _parent_projection(parent):
            raise InteractiveTreeRevisionError(
                "interactive-tree parent revision binding changed"
            )
    elif parent_ref is not None:
        raise InteractiveTreeRevisionError(
            "interactive-tree parent revision evidence is required"
        )
    elif ancestors:
        raise InteractiveTreeRevisionError(
            "interactive-tree ancestor evidence requires a parent revision"
        )

    tree = _normalize_tree(payload["tree"], source_index=source_index)
    if payload["tree"] != tree:
        raise InteractiveTreeRevisionError(
            "interactive-tree tree is not canonical"
        )
    frontier = tuple(tree["frontier_node_ids"])
    expected_visible = _visible_node_ids(source_index, frontier=frontier)
    if tree["visible_node_ids"] != list(expected_visible):
        raise InteractiveTreeRevisionError(
            "interactive-tree visible node index is not canonical"
        )
    _require_frontier_cover(source_index, frontier=frontier)
    _require_metric_conservation(source_index, frontier=frontier)

    node = source_index["node_by_id"].get(edit["node_id"])
    if node is None or node["kind"] != "split" or edit["node_id"] not in frontier:
        raise InteractiveTreeRevisionError(
            "interactive-tree edit does not identify the pruned frontier node"
        )

    if parent_ref is None:
        expected_frontier = _pruned_frontier(
            source_index,
            frontier=tuple(source_index["leaf_ids"]),
            node_id=edit["node_id"],
        )
        if frontier != expected_frontier:
            raise InteractiveTreeRevisionError(
                "interactive-tree first revision frontier is inconsistent"
            )
    elif parent is not None:
        previous_frontier = tuple(parent["tree"]["frontier_node_ids"])
        if edit["node_id"] in previous_frontier:
            raise InteractiveTreeRevisionError(
                "interactive-tree parent edit is an already-frontier no-op"
            )
        if edit["node_id"] not in _visible_node_ids(
            source_index,
            frontier=previous_frontier,
        ):
            raise InteractiveTreeRevisionError(
                "interactive-tree edit node was hidden in the parent revision"
            )
        expected_frontier = _pruned_frontier(
            source_index,
            frontier=previous_frontier,
            node_id=edit["node_id"],
        )
        if frontier != expected_frontier:
            raise InteractiveTreeRevisionError(
                "interactive-tree frontier does not match its parent edit"
            )

    fragments = _derive_fragments(
        source,
        source_index=source_index,
        frontier=frontier,
    )
    if payload["fragments"] != fragments:
        raise InteractiveTreeRevisionError(
            "interactive-tree fragments changed from the base topology"
        )
    if payload["checks"] != _CHECKS:
        raise InteractiveTreeRevisionError(
            "interactive-tree deterministic checks are invalid"
        )

    semantic_body = _semantic_body(
        source,
        frontier=frontier,
        fragments=fragments,
    )
    semantic_hash = _sha256(semantic_body)
    semantic_tree_id = f"interactive-tree-{semantic_hash[:32]}"
    if payload["semantic_tree_id"] != semantic_tree_id:
        raise InteractiveTreeRevisionError(
            "interactive-tree semantic_tree_id is invalid"
        )
    if tree["tree_hash"] != semantic_hash:
        raise InteractiveTreeRevisionError(
            "interactive-tree tree_hash is invalid"
        )

    candidate_evidence = _candidate_evidence(semantic_body)
    if payload["candidate_evidence"] != candidate_evidence:
        raise InteractiveTreeRevisionError(
            "interactive-tree candidate evidence is invalid"
        )
    source_refs = _source_refs(
        source,
        parent_ref=parent_ref,
    )
    if payload["source_refs"] != source_refs:
        raise InteractiveTreeRevisionError(
            "interactive-tree source_refs are invalid"
        )

    without_revision_identity = {
        key: deepcopy(payload[key])
        for key in payload
        if key not in {"revision_id", "revision_hash"}
    }
    revision_id = (
        "interactive-tree-revision-"
        + _sha256(without_revision_identity)[:32]
    )
    if payload["revision_id"] != revision_id:
        raise InteractiveTreeRevisionError(
            "interactive-tree revision_id is invalid"
        )
    without_hash = {
        **without_revision_identity,
        "revision_id": revision_id,
    }
    revision_hash = _sha256(without_hash)
    if not hmac.compare_digest(str(payload["revision_hash"]), revision_hash):
        raise InteractiveTreeRevisionError(
            "interactive-tree revision_hash is invalid"
        )

    canonical = {
        **without_hash,
        "revision_hash": revision_hash,
    }
    _validate_identifier_shapes(canonical)
    return canonical


def canonical_interactive_tree_revision_json(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Return strict canonical JSON for one verified revision."""

    validated = validate_interactive_tree_revision(
        payload,
        automatic_tree_asset,
        parent_revision=parent_revision,
        ancestor_revisions=ancestor_revisions,
    )
    return json.dumps(
        validated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def interactive_tree_revision_to_candidate_fragments(
    payload: Mapping[str, Any],
    automatic_tree_asset: Mapping[str, Any],
    *,
    parent_revision: Mapping[str, Any] | None = None,
    ancestor_revisions: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return detached generic candidate fragments for Pool adapters."""

    revision = validate_interactive_tree_revision(
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


def _assemble_revision(
    source: Mapping[str, Any],
    *,
    source_index: Mapping[str, Any],
    frontier: tuple[str, ...],
    parent_ref: dict[str, Any] | None,
    node_id: str,
    reason: str | None,
) -> dict[str, Any]:
    fragments = _derive_fragments(
        source,
        source_index=source_index,
        frontier=frontier,
    )
    semantic_body = _semantic_body(
        source,
        frontier=frontier,
        fragments=fragments,
    )
    tree_hash = _sha256(semantic_body)
    visible = _visible_node_ids(source_index, frontier=frontier)
    body = {
        "schema_version": INTERACTIVE_TREE_REVISION_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_REVISION_PRODUCER_VERSION,
        "asset_type": INTERACTIVE_TREE_ASSET_TYPE,
        "lifecycle": deepcopy(_LIFECYCLE),
        "identity": deepcopy(source["identity"]),
        "base_tree": _base_tree_projection(source),
        "parent_revision": deepcopy(parent_ref),
        "edit": {
            "operation": "prune_subtree",
            "node_id": node_id,
            "reason": reason,
        },
        "semantic_tree_id": f"interactive-tree-{tree_hash[:32]}",
        "tree": {
            "tree_hash": tree_hash,
            "root_node_id": source_index["root_id"],
            "visible_node_ids": list(visible),
            "frontier_node_ids": list(frontier),
        },
        "fragments": fragments,
        "checks": deepcopy(_CHECKS),
        "candidate_evidence": _candidate_evidence(semantic_body),
        "source_refs": _source_refs(source, parent_ref=parent_ref),
    }
    revision_id = f"interactive-tree-revision-{_sha256(body)[:32]}"
    without_hash = {**body, "revision_id": revision_id}
    return {
        **without_hash,
        "revision_hash": _sha256(without_hash),
    }


def _source_index(source: Mapping[str, Any]) -> dict[str, Any]:
    tree = source["tree_result"]["tree"]
    nodes = tree["nodes"]
    if len(nodes) > MAX_INTERACTIVE_TREE_NODES:
        raise InteractiveTreeRevisionError(
            "automatic tree exceeds the interactive node budget"
        )
    node_by_id = {item["node_id"]: item for item in nodes}
    children: dict[str, tuple[str, ...]] = {}
    parent: dict[str, str] = {}
    for node in nodes:
        if node["kind"] == "split":
            pair = (node["left_child_id"], node["right_child_id"])
            children[node["node_id"]] = pair
            for child in pair:
                parent[child] = node["node_id"]
        else:
            children[node["node_id"]] = ()
    return {
        "nodes": nodes,
        "node_ids": tuple(item["node_id"] for item in nodes),
        "node_by_id": node_by_id,
        "children": children,
        "parent": parent,
        "root_id": tree["root_node_id"],
        "leaf_ids": tuple(tree["leaf_ids"]),
    }


def _pruned_frontier(
    source_index: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
    node_id: str,
) -> tuple[str, ...]:
    node = source_index["node_by_id"].get(node_id)
    if node is None or node["kind"] != "split":
        raise InteractiveTreeRevisionError(
            "interactive-tree prune requires a base split node"
        )
    node_path = tuple(node["path"])
    descendants = {
        candidate
        for candidate in frontier
        if tuple(source_index["node_by_id"][candidate]["path"])[
            : len(node_path)
        ]
        == node_path
    }
    if not descendants:
        raise InteractiveTreeRevisionError(
            "interactive-tree prune node does not cover the current frontier"
        )
    replacement = (set(frontier) - descendants) | {node_id}
    return tuple(
        candidate
        for candidate in source_index["node_ids"]
        if candidate in replacement
    )


def _visible_node_ids(
    source_index: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
) -> tuple[str, ...]:
    frontier_set = set(frontier)
    visible: list[str] = []

    def visit(node_id: str) -> None:
        visible.append(node_id)
        if node_id in frontier_set:
            return
        children = source_index["children"][node_id]
        if not children:
            raise InteractiveTreeRevisionError(
                "interactive-tree frontier does not cover a base leaf"
            )
        for child in children:
            visit(child)

    visit(source_index["root_id"])
    return tuple(visible)


def _require_frontier_cover(
    source_index: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
) -> None:
    if (
        not frontier
        or len(frontier) != len(set(frontier))
        or any(item not in source_index["node_by_id"] for item in frontier)
    ):
        raise InteractiveTreeRevisionError(
            "interactive-tree frontier is invalid"
        )
    paths = {
        item: tuple(source_index["node_by_id"][item]["path"])
        for item in frontier
    }
    for left_id, left_path in paths.items():
        for right_id, right_path in paths.items():
            if left_id == right_id:
                continue
            if right_path[: len(left_path)] == left_path:
                raise InteractiveTreeRevisionError(
                    "interactive-tree frontier is not prefix-free"
                )
    for leaf_id in source_index["leaf_ids"]:
        leaf_path = tuple(source_index["node_by_id"][leaf_id]["path"])
        owners = [
            item
            for item, path in paths.items()
            if leaf_path[: len(path)] == path
        ]
        if len(owners) != 1:
            raise InteractiveTreeRevisionError(
                "interactive-tree frontier does not cover every base leaf once"
            )


def _derive_fragments(
    source: Mapping[str, Any],
    *,
    source_index: Mapping[str, Any],
    frontier: tuple[str, ...],
) -> list[dict[str, Any]]:
    path_conditions = _path_conditions(source_index)
    fragments = []
    for node_id in frontier:
        node = source_index["node_by_id"][node_id]
        condition = path_conditions[node_id]
        semantic = {
            "schema_version": "strategy.interactive-tree-fragment.v1",
            "base_tree_result_hash": source["tree_result"]["result_hash"],
            "source_node_id": node_id,
            "condition": condition,
            "metrics": node["metrics"],
        }
        leaf_id = f"interactive-leaf-{_sha256(semantic)[:32]}"
        rule_id = f"candidate-rule-{_sha256(condition)[:32]}"
        effect_id = (
            "candidate-effect-"
            + _sha256(
                {
                    "base_tree_result_hash": source["tree_result"]["result_hash"],
                    "source_node_id": node_id,
                    "metrics": node["metrics"],
                }
            )[:32]
        )
        fragment_core = {
            "source_node_id": node_id,
            "leaf_id": leaf_id,
            "rule_id": rule_id,
            "condition": condition,
            "requirements": [],
            "effect_id": effect_id,
            "metrics": deepcopy(node["metrics"]),
        }
        fragment_id = (
            f"candidate-fragment-{_sha256(fragment_core)[:32]}"
        )
        without_hash = {
            **fragment_core,
            "fragment_id": fragment_id,
        }
        fragments.append(
            {
                **without_hash,
                "fragment_hash": _sha256(without_hash),
            }
        )
    return fragments


def _path_conditions(
    source_index: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    conditions: dict[str, dict[str, Any]] = {}

    def visit(
        node_id: str,
        clauses: tuple[dict[str, Any], ...],
    ) -> None:
        node = source_index["node_by_id"][node_id]
        if clauses:
            condition = (
                clauses[0]
                if len(clauses) == 1
                else {"op": "and", "args": list(clauses)}
            )
        else:
            if node["kind"] != "split":
                raise InteractiveTreeRevisionError(
                    "automatic base tree root cannot be a leaf"
                )
            left, right = _branch_clauses(node)
            condition = {"op": "or", "args": [left, right]}
        conditions[node_id] = canonicalize_expression(condition)
        if node["kind"] != "split":
            return
        left_clause, right_clause = _branch_clauses(node)
        visit(node["left_child_id"], (*clauses, left_clause))
        visit(node["right_child_id"], (*clauses, right_clause))

    visit(source_index["root_id"], ())
    return conditions


def _branch_clauses(
    node: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        canonicalize_expression(
            {
                "op": "compare",
                "field": node["feature"],
                "operator": "<=",
                "value": node["threshold"],
                "missing": (
                    "match" if node["missing_child"] == "left" else "no_match"
                ),
            }
        ),
        canonicalize_expression(
            {
                "op": "compare",
                "field": node["feature"],
                "operator": ">",
                "value": node["threshold"],
                "missing": (
                    "match" if node["missing_child"] == "right" else "no_match"
                ),
            }
        ),
    )


def _require_metric_conservation(
    source_index: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
) -> None:
    root = source_index["node_by_id"][source_index["root_id"]]["metrics"]
    rows = [source_index["node_by_id"][item]["metrics"] for item in frontier]
    for basis in ("unweighted", "weighted"):
        root_basis = root[basis]
        row_basis = [item[basis] for item in rows]
        if root_basis == {"status": "not_applicable"}:
            if any(item != root_basis for item in row_basis):
                raise InteractiveTreeRevisionError(
                    "interactive-tree metric applicability changed"
                )
            continue
        additive = (
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
        )
        for field in additive:
            if field not in root_basis:
                continue
            total = sum(float(item[field]) for item in row_basis)
            expected = float(root_basis[field])
            if abs(total - expected) > max(1e-9, abs(expected) * 1e-10):
                raise InteractiveTreeRevisionError(
                    f"interactive-tree frontier {basis}.{field} does not conserve"
                )


def _normalize_tree(
    value: object,
    *,
    source_index: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionError(
            "interactive-tree tree must be an object"
        )
    _exact_fields(
        value,
        {
            "tree_hash",
            "root_node_id",
            "visible_node_ids",
            "frontier_node_ids",
        },
        "interactive-tree tree",
    )
    if value["root_node_id"] != source_index["root_id"]:
        raise InteractiveTreeRevisionError(
            "interactive-tree root node changed"
        )
    visible = _node_id_sequence(
        value["visible_node_ids"],
        "tree.visible_node_ids",
    )
    frontier = _node_id_sequence(
        value["frontier_node_ids"],
        "tree.frontier_node_ids",
    )
    tree_hash = _hash(value["tree_hash"], "tree.tree_hash")
    return {
        "tree_hash": tree_hash,
        "root_node_id": source_index["root_id"],
        "visible_node_ids": visible,
        "frontier_node_ids": frontier,
    }


def _semantic_body(
    source: Mapping[str, Any],
    *,
    frontier: tuple[str, ...],
    fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "strategy.interactive-tree-semantic.v1",
        "base_tree": _base_tree_projection(source),
        "frontier_node_ids": list(frontier),
        "fragments": [deepcopy(dict(item)) for item in fragments],
    }


def _candidate_evidence(
    semantic_body: Mapping[str, Any],
) -> dict[str, str]:
    evidence_hash = _sha256(
        {
            "schema_version": "strategy.interactive-tree-evidence.v1",
            "semantic_tree": semantic_body,
        }
    )
    return {
        "candidate_id": f"candidate-{evidence_hash[:32]}",
        "evidence_hash": evidence_hash,
    }


def _base_tree_projection(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": source["asset_id"],
        "asset_hash": source["asset_hash"],
        "tree_id": source["tree_result"]["tree"]["tree_id"],
        "tree_result_hash": source["tree_result"]["result_hash"],
    }


def _parent_projection(parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "revision_id": parent["revision_id"],
        "revision_hash": parent["revision_hash"],
        "semantic_tree_id": parent["semantic_tree_id"],
        "tree_hash": parent["tree"]["tree_hash"],
    }


def _source_refs(
    source: Mapping[str, Any],
    *,
    parent_ref: Mapping[str, Any] | None,
) -> list[str]:
    refs = [
        *source["source_refs"],
        (
            f"automatic-tree-asset:{source['asset_id']}"
            f"@sha256:{source['asset_hash']}"
        ),
    ]
    if parent_ref is not None:
        refs.append(
            f"interactive-tree-revision:{parent_ref['revision_id']}"
            f"@sha256:{parent_ref['revision_hash']}"
        )
    return refs


def _normalize_edit(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionError(
            "interactive-tree edit must be an object"
        )
    _exact_fields(
        value,
        {"operation", "node_id", "reason"},
        "interactive-tree edit",
    )
    if value["operation"] != "prune_subtree":
        raise InteractiveTreeRevisionError(
            "interactive-tree edit operation is invalid"
        )
    return {
        "operation": "prune_subtree",
        "node_id": _text(value["node_id"], "edit.node_id"),
        "reason": _reason(value["reason"]),
    }


def _ancestor_chain(
    value: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise InteractiveTreeRevisionError(
            "interactive-tree ancestor evidence must be a sequence"
        )
    if len(value) > MAX_INTERACTIVE_TREE_NODES:
        raise InteractiveTreeRevisionError(
            "interactive-tree ancestor evidence exceeds the chain limit"
        )
    result = tuple(value)
    if any(not isinstance(item, Mapping) for item in result):
        raise InteractiveTreeRevisionError(
            "interactive-tree ancestor evidence must contain revisions"
        )
    return result


def _normalize_parent_ref(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InteractiveTreeRevisionError(
            "interactive-tree parent_revision must be an object or null"
        )
    _exact_fields(
        value,
        {"revision_id", "revision_hash", "semantic_tree_id", "tree_hash"},
        "interactive-tree parent_revision",
    )
    result = {
        "revision_id": _text(
            value["revision_id"],
            "parent_revision.revision_id",
        ),
        "revision_hash": _hash(
            value["revision_hash"],
            "parent_revision.revision_hash",
        ),
        "semantic_tree_id": _text(
            value["semantic_tree_id"],
            "parent_revision.semantic_tree_id",
        ),
        "tree_hash": _hash(
            value["tree_hash"],
            "parent_revision.tree_hash",
        ),
    }
    if _REVISION_ID_RE.fullmatch(result["revision_id"]) is None:
        raise InteractiveTreeRevisionError(
            "interactive-tree parent revision_id is invalid"
        )
    if _SEMANTIC_TREE_ID_RE.fullmatch(result["semantic_tree_id"]) is None:
        raise InteractiveTreeRevisionError(
            "interactive-tree parent semantic_tree_id is invalid"
        )
    return result


def _require_same_base(
    revision: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    if (
        revision["identity"] != source["identity"]
        or revision["base_tree"] != _base_tree_projection(source)
    ):
        raise InteractiveTreeRevisionError(
            "interactive-tree parent uses another automatic base"
        )


def _validate_identifier_shapes(payload: Mapping[str, Any]) -> None:
    checks = (
        (
            payload["semantic_tree_id"],
            _SEMANTIC_TREE_ID_RE,
            "semantic_tree_id",
        ),
        (payload["revision_id"], _REVISION_ID_RE, "revision_id"),
        (
            payload["candidate_evidence"]["candidate_id"],
            _CANDIDATE_ID_RE,
            "candidate_id",
        ),
    )
    for value, pattern, name in checks:
        if pattern.fullmatch(str(value)) is None:
            raise InteractiveTreeRevisionError(
                f"interactive-tree {name} is invalid"
            )
    _hash(
        payload["candidate_evidence"]["evidence_hash"],
        "candidate_evidence.evidence_hash",
    )
    _hash(payload["revision_hash"], "revision_hash")
    for item in payload["fragments"]:
        fields = (
            ("leaf_id", _LEAF_ID_RE),
            ("fragment_id", _FRAGMENT_ID_RE),
            ("rule_id", _RULE_ID_RE),
            ("effect_id", _EFFECT_ID_RE),
        )
        for field, pattern in fields:
            if pattern.fullmatch(str(item[field])) is None:
                raise InteractiveTreeRevisionError(
                    f"interactive-tree fragment {field} is invalid"
                )
        _hash(item["fragment_hash"], "fragment.fragment_hash")


def _node_id_sequence(value: object, name: str) -> list[str]:
    if (
        isinstance(value, str | bytes | bytearray)
        or not isinstance(value, Sequence)
        or not 1 <= len(value) <= MAX_INTERACTIVE_TREE_NODES
    ):
        raise InteractiveTreeRevisionError(
            f"interactive-tree {name} must be a bounded non-empty array"
        )
    result = [_text(item, name) for item in value]
    if len(result) != len(set(result)):
        raise InteractiveTreeRevisionError(
            f"interactive-tree {name} contains duplicates"
        )
    return result


def _reason(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InteractiveTreeRevisionError(
            "interactive-tree edit reason must be text or null"
        )
    normalized = " ".join(value.split())
    if not normalized:
        raise InteractiveTreeRevisionError(
            "interactive-tree edit reason must not be blank"
        )
    if len(normalized) > MAX_EDIT_REASON_LENGTH or "\x00" in normalized:
        raise InteractiveTreeRevisionError(
            "interactive-tree edit reason is invalid"
        )
    return normalized


def _hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise InteractiveTreeRevisionError(
            f"interactive-tree {name} must be a lowercase SHA-256"
        )
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise InteractiveTreeRevisionError(
            f"interactive-tree {name} must be non-empty text"
        )
    return value.strip()


def _exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    name: str,
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise InteractiveTreeRevisionError(
            f"{name} keys must be strings"
        )
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise InteractiveTreeRevisionError(
            f"{name} fields are invalid ({'; '.join(details)})"
        )


def _sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InteractiveTreeRevisionError(
            "interactive-tree evidence must be finite canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "INTERACTIVE_TREE_ASSET_TYPE",
    "INTERACTIVE_TREE_REVISION_PRODUCER_VERSION",
    "INTERACTIVE_TREE_REVISION_SCHEMA_VERSION",
    "InteractiveTreeRevisionError",
    "build_interactive_tree_revision",
    "canonical_interactive_tree_revision_json",
    "interactive_tree_revision_to_candidate_fragments",
    "validate_interactive_tree_revision",
]
