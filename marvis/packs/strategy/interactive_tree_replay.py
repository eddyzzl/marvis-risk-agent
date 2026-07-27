"""Deterministic replay kernel for interactive-tree threshold revisions.

The kernel is intentionally filesystem and repository free.  Callers provide an
already authenticated automatic tree, its current revision (if any), and the
exact labeled development frame.  The kernel then rebuilds the complete current
visible topology from typed predicates and returns only aggregate evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.weighted_rule_tree import (
    _direction_diagnostic,
    _metrics_bundle,
)
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame


INTERACTIVE_TREE_REPLAY_SCHEMA_VERSION = "strategy.interactive-tree-replay.v2"
INTERACTIVE_TREE_REPLAY_PRODUCER_VERSION = "strategy.interactive-tree-replay/2"


@dataclass(frozen=True)
class InteractiveTreeReplayResult:
    """Aggregate-only output used to construct one immutable v2 revision."""

    nodes: tuple[dict[str, Any], ...]
    visible_node_ids: tuple[str, ...]
    frontier_node_ids: tuple[str, ...]
    replay: dict[str, Any]


def replay_interactive_tree_threshold(
    frame: pd.DataFrame,
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    threshold: object,
    target: np.ndarray,
    weights: np.ndarray | None,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    parent_revision: Mapping[str, Any] | None = None,
) -> InteractiveTreeReplayResult:
    """Replay one finite threshold override against the current visible tree."""

    return replay_interactive_tree_split(
        frame,
        automatic_tree_asset,
        node_id=node_id,
        feature=None,
        threshold=threshold,
        target=target,
        weights=weights,
        loan_values=loan_values,
        overdue_values=overdue_values,
        parent_revision=parent_revision,
    )


def replay_interactive_tree_split(
    frame: pd.DataFrame,
    automatic_tree_asset: Mapping[str, Any],
    *,
    node_id: str,
    feature: object | None,
    threshold: object,
    target: np.ndarray,
    weights: np.ndarray | None,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    parent_revision: Mapping[str, Any] | None = None,
) -> InteractiveTreeReplayResult:
    """Replay one exact feature/threshold split over the current topology."""

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("interactive-tree split replay frame is invalid")
    normalized_threshold = _finite_number(threshold, "threshold")
    source_nodes = automatic_tree_asset["tree_result"]["tree"]["nodes"]
    source_by_id = {item["node_id"]: item for item in source_nodes}
    root_id = automatic_tree_asset["tree_result"]["tree"]["root_node_id"]
    base_leaf_ids = tuple(
        automatic_tree_asset["tree_result"]["tree"]["leaf_ids"]
    )
    if parent_revision is None:
        visible = tuple(item["node_id"] for item in source_nodes)
        frontier = base_leaf_ids
        current_nodes = {
            item["node_id"]: _base_effective_node(item)
            for item in source_nodes
        }
    else:
        visible = tuple(parent_revision["tree"]["visible_node_ids"])
        frontier = tuple(parent_revision["tree"]["frontier_node_ids"])
        if parent_revision["schema_version"].endswith(".v2"):
            current_nodes = {
                item["node_id"]: deepcopy(dict(item))
                for item in parent_revision["tree"]["nodes"]
            }
        else:
            current_nodes = {
                node_id: _base_effective_node(source_by_id[node_id])
                for node_id in visible
            }
    topology_by_id = (
        current_nodes
        if parent_revision is not None
        and parent_revision["schema_version"].endswith(".v2")
        else source_by_id
    )

    if node_id not in set(visible):
        raise StrategyError(
            "interactive-tree threshold node is hidden by the current frontier"
        )
    if node_id in set(frontier):
        raise StrategyError(
            "interactive-tree threshold node is already a frontier leaf"
        )
    source_node = topology_by_id.get(node_id)
    if source_node is None or source_node["kind"] != "split":
        raise StrategyError(
            "interactive-tree split adjustment requires a visible split node"
        )
    current_feature = str(current_nodes[node_id]["feature"])
    if feature is None:
        normalized_feature = current_feature
        replacing_feature = False
    else:
        if not isinstance(feature, str) or not feature.strip():
            raise StrategyError(
                "interactive-tree replacement feature must be non-empty text"
            )
        normalized_feature = feature.strip()
        allowed_features = tuple(
            automatic_tree_asset["tree_result"]["training"]["feature_order"]
        )
        if normalized_feature not in allowed_features:
            raise StrategyError(
                "interactive-tree replacement feature is outside the "
                "authenticated feature universe"
            )
        if normalized_feature == current_feature:
            raise StrategyError(
                "interactive-tree replacement feature must change the current feature"
            )
        replacing_feature = True
    current_threshold = _finite_number(
        current_nodes[node_id]["threshold"],
        "current threshold",
    )
    if not replacing_feature and normalized_threshold == current_threshold:
        raise StrategyError(
            "interactive-tree threshold adjustment is a no-op"
        )

    medians = automatic_tree_asset["tree_result"]["preprocessing"]["medians"]
    directions = automatic_tree_asset["tree_result"]["directions"]
    old_configs = _effective_split_configs(
        topology_by_id,
        current_nodes=current_nodes,
        medians=medians,
    )
    new_configs = deepcopy(old_configs)
    new_configs[node_id]["feature"] = normalized_feature
    new_configs[node_id]["threshold"] = normalized_threshold
    new_configs[node_id]["missing_child"] = (
        "left"
        if float(medians[normalized_feature]) <= normalized_threshold
        else "right"
    )

    old_masks = _route_masks(
        frame,
        root_id=root_id,
        visible=visible,
        frontier=frontier,
        source_by_id=topology_by_id,
        configs=old_configs,
    )
    new_masks = _route_masks(
        frame,
        root_id=root_id,
        visible=visible,
        frontier=frontier,
        source_by_id=topology_by_id,
        configs=new_configs,
    )
    _require_adjustment_scope(
        node_id=node_id,
        visible=visible,
        source_by_id=topology_by_id,
        old_masks=old_masks,
        new_masks=new_masks,
    )
    root_mask = np.ones(len(frame), dtype=bool)
    for current_id in visible:
        current = current_nodes[current_id]
        current_metrics = _metrics_bundle(
            old_masks[current_id],
            target,
            weights=weights,
            root_mask=root_mask,
            loan_values=loan_values,
            overdue_values=overdue_values,
        )
        if _canonical_json(current_metrics) != _canonical_json(
            current["metrics"]
        ):
            raise StrategyError(
                "interactive-tree current visible metrics do not replay"
            )
        source_node = topology_by_id[current_id]
        if source_node["kind"] == "split":
            config = old_configs[current_id]
            left_mask, right_mask = _split_child_masks(
                frame,
                parent_mask=old_masks[current_id],
                feature=config["feature"],
                threshold=config["threshold"],
                missing_child=config["missing_child"],
            )
            diagnostic = _direction_diagnostic(
                direction=directions[config["feature"]],
                left_mask=left_mask,
                right_mask=right_mask,
                target=target,
                weights=weights,
            )
            if _canonical_json(diagnostic) != _canonical_json(
                current["direction_diagnostic"]
            ):
                raise StrategyError(
                    "interactive-tree current visible split diagnostic "
                    "does not replay"
                )
    replayed_nodes: list[dict[str, Any]] = []
    for current_id in visible:
        source_node = topology_by_id[current_id]
        mask = new_masks[current_id]
        node = {
            "node_id": current_id,
            "kind": source_node["kind"],
            "depth": source_node["depth"],
            "path": list(source_node["path"]),
            "condition": _path_condition(
                current_id,
                root_id=root_id,
                source_by_id=topology_by_id,
                configs=new_configs,
            ),
            "metrics": _metrics_bundle(
                mask,
                target,
                weights=weights,
                root_mask=root_mask,
                loan_values=loan_values,
                overdue_values=overdue_values,
            ),
        }
        if source_node["kind"] == "leaf":
            node["rule_id"] = source_node["rule_id"]
        else:
            config = new_configs[current_id]
            left_mask, right_mask = _split_child_masks(
                frame,
                parent_mask=mask,
                feature=config["feature"],
                threshold=config["threshold"],
                missing_child=config["missing_child"],
            )
            if not bool(left_mask.any()) or not bool(right_mask.any()):
                raise StrategyError(
                    "interactive-tree threshold creates an empty visible split child"
                )
            node.update(
                {
                    "feature": config["feature"],
                    "threshold": config["threshold"],
                    "base_threshold": float(
                        current_nodes[current_id].get(
                            "base_threshold",
                            source_node["threshold"],
                        )
                    ),
                    "missing_child": config["missing_child"],
                    "left_child_id": source_node["left_child_id"],
                    "right_child_id": source_node["right_child_id"],
                    "direction_diagnostic": _direction_diagnostic(
                        direction=directions[config["feature"]],
                        left_mask=left_mask,
                        right_mask=right_mask,
                        target=target,
                        weights=weights,
                    ),
                }
            )
        replayed_nodes.append(node)

    _require_frontier_constraints(
        frontier=frontier,
        masks=new_masks,
        training=automatic_tree_asset["tree_result"]["training"],
        weights=weights,
    )
    new_assignment = _frontier_assignment(frontier, new_masks, len(frame))
    old_assignment = _frontier_assignment(frontier, old_masks, len(frame))
    _require_frontier_evaluator_equivalence(
        frame,
        frontier=frontier,
        nodes=replayed_nodes,
        masks=new_masks,
    )
    affected = int(
        sum(
            left != right
            for left, right in zip(
                old_assignment,
                new_assignment,
                strict=True,
            )
        )
    )
    grouping_unchanged = affected == 0
    warning_codes = (
        [
            (
                "split_grouping_unchanged"
                if replacing_feature
                else "threshold_grouping_unchanged"
            )
        ]
        if grouping_unchanged
        else []
    )
    replay_body = {
        "schema_version": INTERACTIVE_TREE_REPLAY_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_REPLAY_PRODUCER_VERSION,
        "source_row_count": len(frame),
        "visible_node_count": len(visible),
        "visible_split_count": sum(
            topology_by_id[item]["kind"] == "split" for item in visible
        ),
        "frontier_count": len(frontier),
        "exactly_once": True,
        "all_visible_metrics_matched": True,
        "all_visible_split_diagnostics_matched": True,
        "frontier_conditions_evaluator_equivalent": True,
        "minimum_leaf_constraints_passed": True,
        "metric_conservation_passed": True,
        "previous_threshold": current_threshold,
        "threshold": normalized_threshold,
        "previous_missing_child": old_configs[node_id]["missing_child"],
        "missing_child": new_configs[node_id]["missing_child"],
        "affected_row_count": affected,
        "grouping_unchanged": grouping_unchanged,
        "warning_codes": warning_codes,
        "assignment_hash": _sha256(new_assignment),
    }
    if replacing_feature:
        replay_body["previous_feature"] = current_feature
        replay_body["feature"] = normalized_feature
    replay = {**replay_body, "result_hash": _sha256(replay_body)}
    return InteractiveTreeReplayResult(
        nodes=tuple(replayed_nodes),
        visible_node_ids=visible,
        frontier_node_ids=frontier,
        replay=replay,
    )


def _base_effective_node(node: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "node_id": node["node_id"],
        "kind": node["kind"],
        "depth": node["depth"],
        "path": list(node["path"]),
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


def _effective_split_configs(
    source_by_id: Mapping[str, Mapping[str, Any]],
    *,
    current_nodes: Mapping[str, Mapping[str, Any]],
    medians: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node_id, source in source_by_id.items():
        if source["kind"] != "split":
            continue
        current = current_nodes.get(node_id, source)
        feature = str(current.get("feature", source["feature"]))
        threshold = _finite_number(current["threshold"], "effective threshold")
        result[node_id] = {
            "feature": feature,
            "threshold": threshold,
            "missing_child": (
                "left"
                if float(medians[feature]) <= threshold
                else "right"
            ),
        }
    return result


def _route_masks(
    frame: pd.DataFrame,
    *,
    root_id: str,
    visible: tuple[str, ...],
    frontier: tuple[str, ...],
    source_by_id: Mapping[str, Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    visible_set = set(visible)
    frontier_set = set(frontier)
    masks: dict[str, np.ndarray] = {}

    def visit(current_id: str, mask: np.ndarray) -> None:
        if current_id not in visible_set:
            raise StrategyError(
                "interactive-tree visible topology is not prefix complete"
            )
        masks[current_id] = mask
        if current_id in frontier_set:
            return
        node = source_by_id[current_id]
        if node["kind"] != "split":
            raise StrategyError(
                "interactive-tree non-frontier visible node is not a split"
            )
        config = configs[current_id]
        left, right = _split_child_masks(
            frame,
            parent_mask=mask,
            feature=config["feature"],
            threshold=config["threshold"],
            missing_child=config["missing_child"],
        )
        visit(node["left_child_id"], left)
        visit(node["right_child_id"], right)

    visit(root_id, np.ones(len(frame), dtype=bool))
    if tuple(item for item in visible if item in masks) != visible:
        raise StrategyError(
            "interactive-tree visible topology order is not canonical"
        )
    return masks


def _split_child_masks(
    frame: pd.DataFrame,
    *,
    parent_mask: np.ndarray,
    feature: str,
    threshold: float,
    missing_child: str,
) -> tuple[np.ndarray, np.ndarray]:
    if feature not in frame.columns:
        raise StrategyError(
            f"interactive-tree replay feature is missing: {feature}"
        )
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    invalid = numeric.notna() & ~np.isfinite(numeric.to_numpy(dtype=float))
    if bool(invalid.any()):
        raise StrategyError(
            f"interactive-tree replay feature contains infinite values: {feature}"
        )
    missing = numeric.isna().to_numpy(dtype=bool)
    values = numeric.to_numpy(dtype=float)
    present = ~missing
    left_route = present & (values <= threshold)
    if missing_child == "left":
        left_route |= missing
    elif missing_child != "right":
        raise StrategyError("interactive-tree missing route is invalid")
    left = parent_mask & left_route
    right = parent_mask & ~left_route
    if np.any(left & right) or not np.array_equal(left | right, parent_mask):
        raise StrategyError("interactive-tree split routing does not conserve")
    return left, right


def _path_condition(
    node_id: str,
    *,
    root_id: str,
    source_by_id: Mapping[str, Mapping[str, Any]],
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    parent: dict[str, tuple[str, str]] = {}
    for item in source_by_id.values():
        if item["kind"] != "split":
            continue
        parent[item["left_child_id"]] = (item["node_id"], "left")
        parent[item["right_child_id"]] = (item["node_id"], "right")
    steps: list[tuple[str, str]] = []
    current = node_id
    while current != root_id:
        if current not in parent:
            raise StrategyError("interactive-tree node is disconnected")
        owner, side = parent[current]
        steps.append((owner, side))
        current = owner
    clauses = []
    for owner, side in reversed(steps):
        config = configs[owner]
        clauses.append(
            {
                "op": "compare",
                "field": config["feature"],
                "operator": "<=" if side == "left" else ">",
                "value": config["threshold"],
                "missing": (
                    "match"
                    if config["missing_child"] == side
                    else "no_match"
                ),
            }
        )
    if not clauses:
        config = configs[root_id]
        clauses = [
            {
                "op": "or",
                "args": [
                    {
                        "op": "compare",
                        "field": config["feature"],
                        "operator": "<=",
                        "value": config["threshold"],
                        "missing": (
                            "match"
                            if config["missing_child"] == "left"
                            else "no_match"
                        ),
                    },
                    {
                        "op": "compare",
                        "field": config["feature"],
                        "operator": ">",
                        "value": config["threshold"],
                        "missing": (
                            "match"
                            if config["missing_child"] == "right"
                            else "no_match"
                        ),
                    },
                ],
            }
        ]
    expression = (
        clauses[0]
        if len(clauses) == 1
        else {"op": "and", "args": clauses}
    )
    return canonicalize_expression(expression)


def _require_adjustment_scope(
    *,
    node_id: str,
    visible: tuple[str, ...],
    source_by_id: Mapping[str, Mapping[str, Any]],
    old_masks: Mapping[str, np.ndarray],
    new_masks: Mapping[str, np.ndarray],
) -> None:
    target_path = tuple(source_by_id[node_id]["path"])
    for current_id in visible:
        path = tuple(source_by_id[current_id]["path"])
        is_descendant = (
            len(path) > len(target_path)
            and path[: len(target_path)] == target_path
        )
        if not is_descendant and not np.array_equal(
            old_masks[current_id],
            new_masks[current_id],
        ):
            raise StrategyError(
                "interactive-tree split replay changed rows outside the "
                "target descendants"
            )


def _require_frontier_constraints(
    *,
    frontier: tuple[str, ...],
    masks: Mapping[str, np.ndarray],
    training: Mapping[str, Any],
    weights: np.ndarray | None,
) -> None:
    cart = training["cart"]
    min_count = int(cart["min_leaf_count"])
    min_weight_fraction = float(cart["min_weight_fraction_leaf"])
    root_weight = (
        float(len(next(iter(masks.values()))))
        if weights is None
        else float(weights.sum())
    )
    for node_id in frontier:
        mask = masks[node_id]
        if int(mask.sum()) < min_count:
            raise StrategyError(
                "interactive-tree threshold violates min_leaf_count"
            )
        if min_weight_fraction > 0:
            weight = (
                float(mask.sum())
                if weights is None
                else float(weights[mask].sum())
            )
            if weight + 1e-12 < root_weight * min_weight_fraction:
                raise StrategyError(
                    "interactive-tree threshold violates "
                    "min_weight_fraction_leaf"
                )


def _frontier_assignment(
    frontier: tuple[str, ...],
    masks: Mapping[str, np.ndarray],
    row_count: int,
) -> list[str]:
    assignment: list[str | None] = [None] * row_count
    for node_id in frontier:
        for index in np.flatnonzero(masks[node_id]):
            position = int(index)
            if assignment[position] is not None:
                raise StrategyError(
                    "interactive-tree frontier assigns a row more than once"
                )
            assignment[position] = node_id
    if any(item is None for item in assignment):
        raise StrategyError(
            "interactive-tree frontier does not assign every row"
        )
    return [str(item) for item in assignment]


def _require_frontier_evaluator_equivalence(
    frame: pd.DataFrame,
    *,
    frontier: tuple[str, ...],
    nodes: list[dict[str, Any]],
    masks: Mapping[str, np.ndarray],
) -> None:
    by_id = {item["node_id"]: item for item in nodes}
    for node_id in frontier:
        evaluated = evaluate_expression_frame(
            frame,
            by_id[node_id]["condition"],
        ).to_numpy(dtype=bool, copy=False)
        if not np.array_equal(evaluated, masks[node_id]):
            raise StrategyError(
                "interactive-tree frontier condition changed from routed rows"
            )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"interactive-tree {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StrategyError(f"interactive-tree {name} must be a finite number")
    return result


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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
