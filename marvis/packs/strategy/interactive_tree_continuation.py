"""Deterministic bounded subtree continuation for one interactive frontier."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
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
from marvis.packs.strategy.interactive_tree_revision import (
    interactive_tree_topology_evidence,
)
from marvis.packs.strategy.interactive_tree_split_search import (
    MAX_ROW_EVALUATIONS,
    MAX_SEARCH_FEATURES,
    MAX_THRESHOLDS_PER_FEATURE,
    search_interactive_tree_split_candidates,
)


INTERACTIVE_TREE_CONTINUATION_REPLAY_SCHEMA_VERSION = (
    "strategy.interactive-tree-continuation-replay.v1"
)
INTERACTIVE_TREE_CONTINUATION_REPLAY_PRODUCER_VERSION = (
    "strategy.interactive-tree-continuation-replay/1"
)
MAX_CONTINUATION_DEPTH = 6
MAX_CONTINUATION_NODES = 127
_OBJECTIVE = "max_gini_gain"
_TIE_BREAK = "eligible_gain_feature_threshold_candidate_id"


@dataclass(frozen=True)
class InteractiveTreeContinuationResult:
    """Aggregate-only effective tree and replay evidence."""

    nodes: tuple[dict[str, Any], ...]
    visible_node_ids: tuple[str, ...]
    frontier_node_ids: tuple[str, ...]
    replay: dict[str, Any]


def continue_interactive_tree_subtree(
    frame: pd.DataFrame,
    automatic_tree_asset: Mapping[str, Any],
    *,
    source_tree_id: str,
    node_id: str,
    seed_candidate: Mapping[str, Any],
    features: Sequence[str],
    target: np.ndarray,
    weights: np.ndarray | None,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    parent_revision: Mapping[str, Any] | None,
    ancestor_revisions: Sequence[Mapping[str, Any]],
    max_additional_depth: int,
    min_gini_gain: float,
    max_generated_nodes: int,
    max_thresholds_per_feature: int,
    max_row_evaluations: int,
    objective: str,
    tie_break: str,
) -> InteractiveTreeContinuationResult:
    """Replace one current frontier with a deterministic bounded subtree."""

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("interactive-tree continuation frame is invalid")
    source_id = _text(source_tree_id, "source_tree_id")
    target_node_id = _text(node_id, "node_id")
    normalized_features = _features(features)
    depth_limit = _bounded_int(
        max_additional_depth,
        "max_additional_depth",
        MAX_CONTINUATION_DEPTH,
    )
    node_limit = _bounded_int(
        max_generated_nodes,
        "max_generated_nodes",
        MAX_CONTINUATION_NODES,
    )
    if node_limit < 3:
        raise StrategyError(
            "interactive-tree continuation max_generated_nodes must be at least 3"
        )
    threshold_limit = _bounded_int(
        max_thresholds_per_feature,
        "max_thresholds_per_feature",
        MAX_THRESHOLDS_PER_FEATURE,
    )
    evaluation_limit = _bounded_int(
        max_row_evaluations,
        "max_row_evaluations",
        MAX_ROW_EVALUATIONS,
    )
    minimum_gain = _finite(min_gini_gain, "min_gini_gain")
    if not 0 <= minimum_gain <= 0.5:
        raise StrategyError(
            "interactive-tree continuation min_gini_gain must be within 0..0.5"
        )
    if objective != _OBJECTIVE or tie_break != _TIE_BREAK:
        raise StrategyError(
            "interactive-tree continuation objective or tie_break changed"
        )
    y = np.asarray(target)
    if y.ndim != 1 or len(y) != len(frame) or not np.isin(y, [0, 1]).all():
        raise StrategyError("interactive-tree continuation target is invalid")
    y = y.astype(np.int8, copy=False)
    w = _weights(weights, len(frame))
    loan = _amounts(loan_values, len(frame), "loan_values")
    overdue = _amounts(overdue_values, len(frame), "overdue_values")
    training = automatic_tree_asset["tree_result"]["training"]
    feature_universe = tuple(training["feature_order"])
    if any(feature not in feature_universe for feature in normalized_features):
        raise StrategyError(
            "interactive-tree continuation feature is outside the "
            "authenticated universe"
        )
    topology = interactive_tree_topology_evidence(
        automatic_tree_asset,
        revision_payload=parent_revision,
        parent_revision=(
            ancestor_revisions[0]
            if parent_revision is not None and ancestor_revisions
            else None
        ),
        ancestor_revisions=(
            ancestor_revisions[1:]
            if parent_revision is not None and ancestor_revisions
            else ()
        ),
    )
    current_nodes = [
        dict(item)
        for item in topology["nodes"]
        if item.get("is_visible") is True
    ]
    current_by_id = {item["node_id"]: item for item in current_nodes}
    target_node = current_by_id.get(target_node_id)
    if (
        target_node is None
        or target_node.get("is_visible") is not True
        or target_node.get("is_frontier") is not True
    ):
        raise StrategyError(
            "interactive-tree continuation requires one current frontier node"
        )
    root_mask = np.ones(len(frame), dtype=bool)
    _revalidate_current_tree(
        frame,
        nodes=current_nodes,
        target=y,
        weights=w,
        loan_values=loan,
        overdue_values=overdue,
        root_mask=root_mask,
        directions=automatic_tree_asset["tree_result"]["directions"],
    )
    target_mask = evaluate_expression_frame(
        frame,
        target_node["condition"],
    ).to_numpy(dtype=bool)
    seed = _seed_candidate(
        seed_candidate,
        source_tree_id=source_id,
        node_id=target_node_id,
        features=normalized_features,
    )
    if float(seed["gain"]) + 1e-15 < minimum_gain:
        raise StrategyError(
            "interactive-tree continuation seed is below min_gini_gain"
        )
    medians = automatic_tree_asset["tree_result"]["preprocessing"]["medians"]
    directions = automatic_tree_asset["tree_result"]["directions"]
    cart = training["cart"]
    min_leaf_count = int(cart["min_leaf_count"])
    min_weight_fraction = float(cart["min_weight_fraction_leaf"])
    root_weight = float(len(frame)) if w is None else float(w.sum())
    seed_left, seed_right = _split_masks(
        frame,
        parent_mask=target_mask,
        feature=seed["feature"],
        threshold=float(seed["threshold"]),
        missing_child=seed["missing_child"],
    )
    _require_leaf_constraints(
        seed_left,
        seed_right,
        weights=w,
        root_weight=root_weight,
        min_leaf_count=min_leaf_count,
        min_weight_fraction=min_weight_fraction,
    )
    _require_seed_populations(seed, seed_left, seed_right, y, w)

    identity = {
        "source_tree_id": source_id,
        "node_id": target_node_id,
        "seed_candidate_id": seed["candidate_id"],
        "features": normalized_features,
        "max_additional_depth": depth_limit,
        "min_gini_gain": minimum_gain,
        "max_generated_nodes": node_limit,
        "max_thresholds_per_feature": threshold_limit,
        "max_row_evaluations": evaluation_limit,
        "objective": objective,
        "tie_break": tie_break,
    }
    generated: list[dict[str, Any]] = []
    frontier: list[str] = []
    counters: Counter[str] = Counter()
    row_evaluations = 0

    def generated_id(relative_path: tuple[str, ...]) -> str:
        return "node-" + _sha256(
            {
                "schema_version": "strategy.interactive-tree-generated-node.v1",
                "identity": identity,
                "path": list(relative_path),
            }
        )[:20]

    def leaf(
        current_id: str,
        mask: np.ndarray,
        *,
        path: tuple[str, ...],
        condition: Mapping[str, Any],
        stop_reason: str,
    ) -> None:
        counters[stop_reason] += 1
        canonical_condition = canonicalize_expression(condition)
        generated.append(
            {
                "node_id": current_id,
                "kind": "leaf",
                "depth": len(path),
                "path": list(path),
                "condition": canonical_condition,
                "metrics": _metrics_bundle(
                    mask,
                    y,
                    weights=w,
                    root_mask=root_mask,
                    loan_values=loan,
                    overdue_values=overdue,
                ),
                "rule_id": (
                    "candidate-rule-" + _sha256(canonical_condition)[:32]
                ),
            }
        )
        frontier.append(current_id)

    def split_node(
        current_id: str,
        mask: np.ndarray,
        *,
        path: tuple[str, ...],
        relative_path: tuple[str, ...],
        condition: Mapping[str, Any],
        candidate: Mapping[str, Any],
        depth_from_target: int,
        base_threshold: float,
        reserved_nodes: int,
    ) -> None:
        nonlocal row_evaluations
        feature = str(candidate["feature"])
        threshold = float(candidate["threshold"])
        missing_child = str(candidate["missing_child"])
        left_mask, right_mask = _split_masks(
            frame,
            parent_mask=mask,
            feature=feature,
            threshold=threshold,
            missing_child=missing_child,
        )
        _require_leaf_constraints(
            left_mask,
            right_mask,
            weights=w,
            root_weight=root_weight,
            min_leaf_count=min_leaf_count,
            min_weight_fraction=min_weight_fraction,
        )
        left_relative = (*relative_path, "left")
        right_relative = (*relative_path, "right")
        left_id = generated_id(left_relative)
        right_id = generated_id(right_relative)
        canonical_condition = canonicalize_expression(condition)
        generated.append(
            {
                "node_id": current_id,
                "kind": "split",
                "depth": len(path),
                "path": list(path),
                "condition": canonical_condition,
                "metrics": _metrics_bundle(
                    mask,
                    y,
                    weights=w,
                    root_mask=root_mask,
                    loan_values=loan,
                    overdue_values=overdue,
                ),
                "feature": feature,
                "threshold": threshold,
                "base_threshold": base_threshold,
                "missing_child": missing_child,
                "left_child_id": left_id,
                "right_child_id": right_id,
                "direction_diagnostic": _direction_diagnostic(
                    direction=directions[feature],
                    left_mask=left_mask,
                    right_mask=right_mask,
                    target=y,
                    weights=w,
                ),
            }
        )
        children = (
            ("left", left_id, left_mask, left_relative),
            ("right", right_id, right_mask, right_relative),
        )
        for child_index, (
            side,
            child_id,
            child_mask,
            child_relative,
        ) in enumerate(children):
            child_reserve = reserved_nodes + (len(children) - child_index - 1)
            child_path = (*path, side)
            child_condition = _append_condition(
                canonical_condition,
                parent_path=path,
                feature=feature,
                threshold=threshold,
                missing_child=missing_child,
                side=side,
            )
            if depth_from_target >= depth_limit:
                leaf(
                    child_id,
                    child_mask,
                    path=child_path,
                    condition=child_condition,
                    stop_reason="max_additional_depth",
                )
                continue
            if len(generated) + 3 + child_reserve > node_limit:
                leaf(
                    child_id,
                    child_mask,
                    path=child_path,
                    condition=child_condition,
                    stop_reason="max_generated_nodes",
                )
                continue
            remaining = evaluation_limit - row_evaluations
            if remaining < 1:
                leaf(
                    child_id,
                    child_mask,
                    path=child_path,
                    condition=child_condition,
                    stop_reason="max_row_evaluations",
                )
                continue
            try:
                search = search_interactive_tree_split_candidates(
                    frame,
                    node_mask=child_mask,
                    node_id=child_id,
                    source_tree_id=source_id,
                    features=normalized_features,
                    target=y,
                    weights=w,
                    medians=medians,
                    directions=directions,
                    min_leaf_count=min_leaf_count,
                    max_thresholds_per_feature=threshold_limit,
                    max_row_evaluations=remaining,
                )
            except StrategyError as exc:
                message = str(exc)
                if "minimum leaf cannot be satisfied" in message:
                    stop_reason = "no_eligible_gain"
                elif "row evaluation budget exceeded" in message:
                    stop_reason = "max_row_evaluations"
                else:
                    raise
                leaf(
                    child_id,
                    child_mask,
                    path=child_path,
                    condition=child_condition,
                    stop_reason=stop_reason,
                )
                continue
            row_evaluations += int(search["budget"]["row_evaluations"])
            eligible = [
                item
                for item in search["candidates"]
                if item["eligible"]
                and float(item["gain"]) + 1e-15 >= minimum_gain
                and _weight_fraction_eligible(
                    item,
                    root_weight=root_weight,
                    min_weight_fraction=min_weight_fraction,
                )
            ]
            if not eligible:
                leaf(
                    child_id,
                    child_mask,
                    path=child_path,
                    condition=child_condition,
                    stop_reason="no_eligible_gain",
                )
                continue
            selected = eligible[0]
            split_node(
                child_id,
                child_mask,
                path=child_path,
                relative_path=child_relative,
                condition=child_condition,
                candidate=selected,
                depth_from_target=depth_from_target + 1,
                base_threshold=float(selected["threshold"]),
                reserved_nodes=child_reserve,
            )

    target_base_threshold = (
        float(
            target_node.get(
                "base_threshold",
                target_node["threshold"],
            )
        )
        if target_node["kind"] == "split"
        else float(seed["threshold"])
    )
    split_node(
        target_node_id,
        target_mask,
        path=tuple(target_node["path"]),
        relative_path=(),
        condition=target_node["condition"],
        candidate=seed,
        depth_from_target=1,
        base_threshold=target_base_threshold,
        reserved_nodes=0,
    )
    if len(generated) > node_limit:
        raise StrategyError(
            "interactive-tree continuation generated node budget exceeded"
        )
    target_path = tuple(target_node["path"])
    outside = [
        _semantic_node(node)
        for node in current_nodes
        if (
            tuple(node["path"])[: len(target_path)] != target_path
            or len(node["path"]) < len(target_path)
        )
    ]
    nodes = sorted(
        [*outside, *generated],
        key=lambda item: _path_sort_key(item["path"]),
    )
    visible = tuple(item["node_id"] for item in nodes)
    old_frontier = [
        item
        for item in topology["frontier_node_ids"]
        if tuple(current_by_id[item]["path"])[: len(target_path)] != target_path
    ]
    full_frontier = tuple(
        sorted(
            [*old_frontier, *frontier],
            key=lambda item: _path_sort_key(
                next(node["path"] for node in nodes if node["node_id"] == item)
            ),
        )
    )
    assignment = _frontier_assignment(
        frame,
        nodes=nodes,
        frontier=full_frontier,
    )
    replay_body = {
        "schema_version": INTERACTIVE_TREE_CONTINUATION_REPLAY_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_CONTINUATION_REPLAY_PRODUCER_VERSION,
        "source_tree_id": source_id,
        "node_id": target_node_id,
        "seed_candidate_id": seed["candidate_id"],
        "objective": objective,
        "tie_break": tie_break,
        "controls": {
            "features": normalized_features,
            "max_additional_depth": depth_limit,
            "min_gini_gain": minimum_gain,
            "max_generated_nodes": node_limit,
            "max_thresholds_per_feature": threshold_limit,
            "max_row_evaluations": evaluation_limit,
        },
        "observed": {
            "generated_node_count": len(generated),
            "generated_split_count": sum(
                item["kind"] == "split" for item in generated
            ),
            "generated_leaf_count": len(frontier),
            "row_evaluations": row_evaluations,
            "stop_reasons": dict(sorted(counters.items())),
        },
        "source_row_count": len(frame),
        "visible_node_count": len(visible),
        "frontier_count": len(full_frontier),
        "exactly_once": True,
        "current_tree_replayed": True,
        "minimum_leaf_constraints_passed": True,
        "frontier_conditions_evaluator_equivalent": True,
        "assignment_hash": _sha256(assignment),
    }
    replay = {**replay_body, "result_hash": _sha256(replay_body)}
    return InteractiveTreeContinuationResult(
        nodes=tuple(nodes),
        visible_node_ids=visible,
        frontier_node_ids=full_frontier,
        replay=replay,
    )


def _revalidate_current_tree(
    frame: pd.DataFrame,
    *,
    nodes: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    weights: np.ndarray | None,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
    root_mask: np.ndarray,
    directions: Mapping[str, str],
) -> None:
    for node in nodes:
        mask = evaluate_expression_frame(
            frame,
            node["condition"],
        ).to_numpy(dtype=bool)
        metrics = _metrics_bundle(
            mask,
            target,
            weights=weights,
            root_mask=root_mask,
            loan_values=loan_values,
            overdue_values=overdue_values,
        )
        if _canonical_json(metrics) != _canonical_json(node["metrics"]):
            raise StrategyError(
                "interactive-tree continuation current metrics do not replay"
            )
        if node["kind"] != "split" or node.get("is_frontier") is True:
            continue
        left, right = _split_masks(
            frame,
            parent_mask=mask,
            feature=node["feature"],
            threshold=float(node["threshold"]),
            missing_child=node["missing_child"],
        )
        diagnostic = _direction_diagnostic(
            direction=directions[node["feature"]],
            left_mask=left,
            right_mask=right,
            target=target,
            weights=weights,
        )
        if _canonical_json(diagnostic) != _canonical_json(
            node["direction_diagnostic"]
        ):
            raise StrategyError(
                "interactive-tree continuation current split diagnostic "
                "does not replay"
            )


def _seed_candidate(
    value: Mapping[str, Any],
    *,
    source_tree_id: str,
    node_id: str,
    features: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyError(
            "interactive-tree continuation seed candidate is invalid"
        )
    candidate = json.loads(_canonical_json(value))
    if (
        candidate.get("eligible") is not True
        or candidate.get("feature") not in features
        or candidate.get("missing_child") not in {"left", "right"}
        or not isinstance(candidate.get("candidate_id"), str)
    ):
        raise StrategyError(
            "interactive-tree continuation seed candidate is ineligible"
        )
    semantic = {
        "schema_version": "strategy.interactive-tree-split-candidate.v1",
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "feature": candidate["feature"],
        "threshold": _finite(candidate["threshold"], "seed threshold"),
        "missing_child": candidate["missing_child"],
    }
    expected_id = "interactive-tree-split-candidate-" + _sha256(semantic)[:32]
    if candidate["candidate_id"] != expected_id:
        raise StrategyError(
            "interactive-tree continuation seed identity changed"
        )
    return candidate


def _require_seed_populations(
    candidate: Mapping[str, Any],
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> None:
    for expected, mask, label in (
        (candidate["left"], left, "left"),
        (candidate["right"], right, "right"),
    ):
        actual = _population(mask, target, weights)
        if _canonical_json(actual) != _canonical_json(expected):
            raise StrategyError(
                f"interactive-tree continuation seed {label} population changed"
            )


def _population(
    mask: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    count = int(mask.sum())
    bad = int(target[mask].sum())
    result: dict[str, Any] = {
        "count": count,
        "good": count - bad,
        "bad": bad,
        "bad_rate": 0.0 if count == 0 else bad / count,
    }
    if weights is None:
        result["weighted"] = {"status": "not_applicable"}
    else:
        total = float(weights[mask].sum())
        weighted_bad = float(weights[mask & (target == 1)].sum())
        result["weighted"] = {
            "status": "available",
            "total": total,
            "bad": weighted_bad,
            "bad_rate": 0.0 if total == 0 else weighted_bad / total,
        }
    return result


def _split_masks(
    frame: pd.DataFrame,
    *,
    parent_mask: np.ndarray,
    feature: str,
    threshold: float,
    missing_child: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(
        dtype=float
    )
    if np.isinf(values).any():
        raise StrategyError(
            f"interactive-tree continuation feature contains infinity: {feature}"
        )
    missing = np.isnan(values)
    left_route = (~missing) & (values <= threshold)
    if missing_child == "left":
        left_route |= missing
    left = parent_mask & left_route
    right = parent_mask & ~left_route
    if np.any(left & right) or not np.array_equal(left | right, parent_mask):
        raise StrategyError(
            "interactive-tree continuation split does not conserve rows"
        )
    return left, right


def _require_leaf_constraints(
    left: np.ndarray,
    right: np.ndarray,
    *,
    weights: np.ndarray | None,
    root_weight: float,
    min_leaf_count: int,
    min_weight_fraction: float,
) -> None:
    if int(left.sum()) < min_leaf_count or int(right.sum()) < min_leaf_count:
        raise StrategyError(
            "interactive-tree continuation violates min_leaf_count"
        )
    if min_weight_fraction <= 0:
        return
    for mask in (left, right):
        total = float(mask.sum()) if weights is None else float(weights[mask].sum())
        if total + 1e-12 < root_weight * min_weight_fraction:
            raise StrategyError(
                "interactive-tree continuation violates "
                "min_weight_fraction_leaf"
            )


def _weight_fraction_eligible(
    candidate: Mapping[str, Any],
    *,
    root_weight: float,
    min_weight_fraction: float,
) -> bool:
    if min_weight_fraction <= 0:
        return True
    for side in ("left", "right"):
        population = candidate[side]
        weighted = population["weighted"]
        total = (
            float(weighted["total"])
            if weighted["status"] == "available"
            else float(population["count"])
        )
        if total + 1e-12 < root_weight * min_weight_fraction:
            return False
    return True


def _append_condition(
    parent: Mapping[str, Any],
    *,
    parent_path: Sequence[str],
    feature: str,
    threshold: float,
    missing_child: str,
    side: str,
) -> dict[str, Any]:
    clause = {
        "op": "compare",
        "field": feature,
        "operator": "<=" if side == "left" else ">",
        "value": threshold,
        "missing": "match" if missing_child == side else "no_match",
    }
    if not parent_path:
        return canonicalize_expression(clause)
    args = (
        [*parent["args"], clause]
        if parent.get("op") == "and"
        else [parent, clause]
    )
    return canonicalize_expression({"op": "and", "args": args})


def _frontier_assignment(
    frame: pd.DataFrame,
    *,
    nodes: Sequence[Mapping[str, Any]],
    frontier: Sequence[str],
) -> list[str]:
    by_id = {item["node_id"]: item for item in nodes}
    assignment: list[str | None] = [None] * len(frame)
    for node_id in frontier:
        mask = evaluate_expression_frame(
            frame,
            by_id[node_id]["condition"],
        ).to_numpy(dtype=bool)
        for row in np.flatnonzero(mask):
            index = int(row)
            if assignment[index] is not None:
                raise StrategyError(
                    "interactive-tree continuation frontier overlaps"
                )
            assignment[index] = node_id
    if any(value is None for value in assignment):
        raise StrategyError(
            "interactive-tree continuation frontier does not cover every row"
        )
    return [str(value) for value in assignment]


def _path_sort_key(value: Sequence[str]) -> tuple[int, ...]:
    return tuple(0 if item == "left" else 1 for item in value)


def _semantic_node(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "node_id",
        "kind",
        "depth",
        "path",
        "condition",
        "metrics",
        "rule_id",
        "feature",
        "threshold",
        "base_threshold",
        "missing_child",
        "left_child_id",
        "right_child_id",
        "direction_diagnostic",
    }
    return {
        key: deepcopy(value[key])
        for key in value
        if key in fields
    }


def _features(value: Sequence[str]) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError("interactive-tree continuation features are invalid")
    result = [_text(item, "feature") for item in value]
    if (
        not result
        or len(result) > MAX_SEARCH_FEATURES
        or len(result) != len(set(result))
        or result != sorted(result)
    ):
        raise StrategyError(
            "interactive-tree continuation features must be sorted and unique"
        )
    return result


def _weights(value: np.ndarray | None, rows: int) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or len(result) != rows
        or not np.isfinite(result).all()
        or np.any(result <= 0)
    ):
        raise StrategyError("interactive-tree continuation weights are invalid")
    return result


def _amounts(
    value: np.ndarray | None,
    rows: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or len(result) != rows
        or np.isinf(result).any()
        or np.any(result[~np.isnan(result)] < 0)
    ):
        raise StrategyError(f"interactive-tree continuation {name} is invalid")
    return result


def _bounded_int(value: object, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 1
        or int(value) > maximum
    ):
        raise StrategyError(
            f"interactive-tree continuation {name} is outside its hard budget"
        )
    return int(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"interactive-tree continuation {name} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise StrategyError(f"interactive-tree continuation {name} is invalid")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"interactive-tree continuation {name} is invalid")
    return value.strip()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        raise StrategyError(
            "interactive-tree continuation is not canonical JSON"
        ) from exc


__all__ = [
    "INTERACTIVE_TREE_CONTINUATION_REPLAY_SCHEMA_VERSION",
    "InteractiveTreeContinuationResult",
    "continue_interactive_tree_subtree",
]
