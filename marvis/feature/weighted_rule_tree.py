"""Deterministic, governed numeric CART rule-tree kernel.

The module deliberately owns only measured tree construction and replay.  It does
not select a business action, validate a strategy, or claim adoption/deployment.
Every leaf is emitted as a canonical typed comparison expression and is replayed
against the training frame before a result can leave the kernel.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier

from marvis.feature.errors import FeatureError
from marvis.packs.strategy.dsl import canonicalize_expression
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.evaluator import evaluate_expression_frame


WEIGHTED_RULE_TREE_SCHEMA_VERSION = "feature.weighted-rule-tree.v1"
DEFAULT_WEIGHTED_RULE_TREE_SEED = 20260719

_HARD_BUDGETS = {
    "max_rows": 1_000_000,
    "max_features": 50,
    "max_cells": 50_000_000,
    "max_nodes": 511,
    "max_cutpoint_evaluations": 50_000_000,
}
_DIRECTIONS = frozenset({"increasing", "decreasing", "unordered"})


class WeightedRuleTreeError(FeatureError):
    """A typed, fail-closed weighted-rule-tree error."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        self.code = str(code)
        self.details = dict(details)
        super().__init__(message)

    def to_detail(self) -> dict[str, object]:
        return {
            "kind": "weighted_rule_tree_error",
            "code": self.code,
            **self.details,
        }


@dataclass(frozen=True)
class WeightedRuleTreeBudgets:
    """Caller-controlled limits, each bounded by a non-overridable hard cap."""

    max_rows: int = 1_000_000
    max_features: int = 50
    max_cells: int = 50_000_000
    max_nodes: int = 511
    max_cutpoint_evaluations: int = 50_000_000

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WeightedRuleTreeError(
                    "invalid_config",
                    f"{name} must be a positive integer",
                    field=name,
                )
            hard_limit = _HARD_BUDGETS[name]
            if value > hard_limit:
                raise WeightedRuleTreeError(
                    "invalid_config",
                    f"{name} must be at most {hard_limit}",
                    field=name,
                    actual=value,
                    hard_limit=hard_limit,
                )


def build_weighted_rule_tree(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    sample_weight_col: str | None = None,
    directions: Mapping[str, str] | None = None,
    max_depth: int = 4,
    min_leaf_count: int = 200,
    min_weight_fraction_leaf: float = 0.0,
    seed: int = DEFAULT_WEIGHTED_RULE_TREE_SEED,
    loan_amount_col: str | None = None,
    overdue_amount_col: str | None = None,
    budgets: WeightedRuleTreeBudgets | Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Fit an exact numeric CART tree and return canonical JSON-native evidence.

    Missing selected-feature values are median-imputed for fitting.  Each frozen
    median is then translated back into the leaf DSL through an explicit
    ``missing=match|no_match`` policy, so replay never depends on an implicit
    preprocessing step.  Selected columns that are non-numeric, infinite,
    entirely missing, or constant fail rather than being silently filtered.
    """

    _require_frame(frame)
    feature_order = _normalize_features(feature_cols)
    target_name = _column_name(target_col, field="target_col")
    weight_name = _optional_column_name(sample_weight_col, field="sample_weight_col")
    loan_name = _optional_column_name(loan_amount_col, field="loan_amount_col")
    overdue_name = _optional_column_name(overdue_amount_col, field="overdue_amount_col")
    _validate_fit_config(
        max_depth=max_depth,
        min_leaf_count=min_leaf_count,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        seed=seed,
    )
    resolved_budgets = _resolve_budgets(budgets)
    _assert_distinct_roles(
        feature_order=feature_order,
        target_col=target_name,
        sample_weight_col=weight_name,
        loan_amount_col=loan_name,
        overdue_amount_col=overdue_name,
    )

    required = [*feature_order, target_name]
    required.extend(
        name for name in (weight_name, loan_name, overdue_name) if name is not None
    )
    _assert_columns(frame, required)
    row_count = int(len(frame))
    if row_count == 0:
        raise WeightedRuleTreeError(
            "invalid_input", "weighted rule tree requires at least one row"
        )
    _check_budget("rows", row_count, resolved_budgets.max_rows)
    _check_budget("features", len(feature_order), resolved_budgets.max_features)
    _check_budget("cells", row_count * len(feature_order), resolved_budgets.max_cells)
    cutpoint_upper_bound = row_count * len(feature_order) * int(max_depth)
    _check_budget(
        "cutpoint_evaluations",
        cutpoint_upper_bound,
        resolved_budgets.max_cutpoint_evaluations,
    )

    target = _strict_binary_target(frame[target_name], column=target_name)
    x, medians = _fit_feature_matrix(frame, feature_order)
    weights = (
        None
        if weight_name is None
        else _strict_positive_weights(frame[weight_name], column=weight_name)
    )
    loan_values = (
        None
        if loan_name is None
        else _strict_amounts(frame[loan_name], column=loan_name)
    )
    overdue_values = (
        None
        if overdue_name is None
        else _strict_amounts(frame[overdue_name], column=overdue_name)
    )
    normalized_directions = _normalize_directions(directions, feature_order)

    classifier = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=int(max_depth),
        min_samples_split=2,
        min_samples_leaf=int(min_leaf_count),
        min_weight_fraction_leaf=float(min_weight_fraction_leaf),
        random_state=int(seed),
    )
    try:
        classifier.fit(x, target, sample_weight=weights)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WeightedRuleTreeError(
            "fit_failed",
            f"sklearn CART rejected the validated training matrix: {exc}",
        ) from exc

    raw_tree = classifier.tree_
    actual_nodes = int(raw_tree.node_count)
    if actual_nodes == 1:
        raise WeightedRuleTreeError(
            "infeasible_tree",
            "CART produced no valid split under the requested leaf constraints",
            row_count=row_count,
            min_leaf_count=int(min_leaf_count),
            min_weight_fraction_leaf=float(min_weight_fraction_leaf),
        )
    _check_budget("nodes", actual_nodes, resolved_budgets.max_nodes)

    decision_path = classifier.decision_path(x).astype(bool)
    sklearn_leaf_ids = classifier.apply(x)
    root_mask = np.ones(row_count, dtype=bool)

    temporary: dict[int, dict[str, Any]] = {}

    def visit(raw_node_id: int, path: tuple[str, ...]) -> None:
        node_mask = _sparse_column_mask(decision_path, raw_node_id, row_count)
        left_raw = int(raw_tree.children_left[raw_node_id])
        right_raw = int(raw_tree.children_right[raw_node_id])
        base: dict[str, Any] = {
            "raw_node_id": raw_node_id,
            "path": path,
            "depth": len(path),
            "mask": node_mask,
            "metrics": _metrics_bundle(
                node_mask,
                target,
                weights=weights,
                root_mask=root_mask,
                loan_values=loan_values,
                overdue_values=overdue_values,
            ),
        }
        if left_raw == -1 and right_raw == -1:
            base["kind"] = "leaf"
            temporary[raw_node_id] = base
            return
        if left_raw < 0 or right_raw < 0:
            raise WeightedRuleTreeError(
                "routing_mismatch", "CART returned a partially defined split node"
            )
        feature_index = int(raw_tree.feature[raw_node_id])
        if feature_index < 0 or feature_index >= len(feature_order):
            raise WeightedRuleTreeError(
                "routing_mismatch", "CART split references an unknown feature index"
            )
        feature = feature_order[feature_index]
        left_mask = _sparse_column_mask(decision_path, left_raw, row_count)
        right_mask = _sparse_column_mask(decision_path, right_raw, row_count)
        _assert_child_partition(node_mask, left_mask, right_mask)
        sklearn_threshold = float(raw_tree.threshold[raw_node_id])
        threshold, threshold_adjustment = _equivalent_threshold(
            values=x[:, feature_index],
            node_mask=node_mask,
            sklearn_left_mask=left_mask,
            sklearn_threshold=sklearn_threshold,
        )
        missing_child = "left" if medians[feature] <= threshold else "right"
        base.update(
            {
                "kind": "split",
                "feature": feature,
                "threshold": threshold,
                "sklearn_threshold": sklearn_threshold,
                "threshold_adjustment": threshold_adjustment,
                "missing_child": missing_child,
                "left_raw": left_raw,
                "right_raw": right_raw,
                "direction_diagnostic": _direction_diagnostic(
                    direction=normalized_directions[feature],
                    left_mask=left_mask,
                    right_mask=right_mask,
                    target=target,
                    weights=weights,
                ),
            }
        )
        temporary[raw_node_id] = base
        visit(left_raw, (*path, "left"))
        visit(right_raw, (*path, "right"))

    visit(0, ())
    _assert_conservation(
        temporary,
        target=target,
        weights=weights,
        loan_values=loan_values,
        overdue_values=overdue_values,
    )

    structural_payload = _structure_payload(temporary, 0)
    tree_fingerprint = _digest(
        {
            "schema_version": WEIGHTED_RULE_TREE_SCHEMA_VERSION,
            "feature_order": feature_order,
            "structure": structural_payload,
        }
    )
    tree_id = f"tree-{tree_fingerprint[:20]}"
    stable_ids = {
        raw_id: _stable_node_id(tree_fingerprint, node)
        for raw_id, node in temporary.items()
    }

    nodes: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []

    def emit(
        raw_node_id: int,
        clauses: tuple[dict[str, Any], ...],
    ) -> None:
        temp = temporary[raw_node_id]
        stable_id = stable_ids[raw_node_id]
        node: dict[str, Any] = {
            "node_id": stable_id,
            "kind": temp["kind"],
            "depth": temp["depth"],
            "path": list(temp["path"]),
            "metrics": temp["metrics"],
        }
        if temp["kind"] == "leaf":
            condition = _path_condition(clauses)
            rule_id = f"rule-{_digest(condition)[:20]}"
            node["rule_id"] = rule_id
            rules.append(
                {
                    "rule_id": rule_id,
                    "leaf_id": stable_id,
                    "condition": condition,
                    "clauses": [dict(clause) for clause in clauses],
                    "metrics": temp["metrics"],
                }
            )
            nodes.append(node)
            return

        left_raw = int(temp["left_raw"])
        right_raw = int(temp["right_raw"])
        node.update(
            {
                "feature": temp["feature"],
                "threshold": temp["threshold"],
                "sklearn_threshold": temp["sklearn_threshold"],
                "threshold_adjustment": temp["threshold_adjustment"],
                "missing_child": temp["missing_child"],
                "left_child_id": stable_ids[left_raw],
                "right_child_id": stable_ids[right_raw],
                "direction_diagnostic": temp["direction_diagnostic"],
            }
        )
        nodes.append(node)
        left_clause = _compare_clause(
            field=temp["feature"],
            operator="<=",
            value=temp["threshold"],
            missing=temp["missing_child"] == "left",
        )
        right_clause = _compare_clause(
            field=temp["feature"],
            operator=">",
            value=temp["threshold"],
            missing=temp["missing_child"] == "right",
        )
        emit(left_raw, (*clauses, left_clause))
        emit(right_raw, (*clauses, right_clause))

    emit(0, ())
    leaf_ids = [rule["leaf_id"] for rule in rules]

    result: dict[str, Any] = {
        "schema_version": WEIGHTED_RULE_TREE_SCHEMA_VERSION,
        "lifecycle": {
            "stage": "development",
            "validation_status": "unvalidated",
            "evidence_status": "backtested",
        },
        "training": {
            "row_count": row_count,
            "feature_order": feature_order,
            "target_col": target_name,
            "sample_weight": (
                {"status": "not_applicable"}
                if weight_name is None
                else {"status": "available", "column": weight_name}
            ),
            "loan_amount_col": loan_name,
            "overdue_amount_col": overdue_name,
            "seed": int(seed),
            "sklearn_version": str(sklearn.__version__),
            "cart": {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": int(max_depth),
                "min_samples_split": 2,
                "min_leaf_count": int(min_leaf_count),
                "min_weight_fraction_leaf": float(min_weight_fraction_leaf),
            },
        },
        "preprocessing": {
            "missing_policy": "training_median_frozen_into_typed_dsl",
            "medians": medians,
        },
        "directions": normalized_directions,
        "budgets": asdict(resolved_budgets),
        "search": {
            "method": "greedy_cart",
            "truncated": False,
            "cutpoint_evaluations_upper_bound": cutpoint_upper_bound,
            "tie_break": {
                "feature_order": "lexicographic_by_name",
                "splitter": "sklearn_best",
                "random_state": int(seed),
            },
        },
        "tree": {
            "tree_id": tree_id,
            "root_node_id": stable_ids[0],
            "node_count": len(nodes),
            "leaf_count": len(leaf_ids),
            "leaf_ids": leaf_ids,
            "nodes": nodes,
        },
        "rules": rules,
        "checks": {
            "sklearn_route_matches_typed_dsl": True,
            "all_training_rows_assigned_once": True,
            "conservation": "passed",
        },
    }
    result["result_hash"] = _digest(result)

    replay_leaf_ids = apply_weighted_rule_tree(frame, result)
    expected_leaf_ids = np.array(
        [stable_ids[int(raw_id)] for raw_id in sklearn_leaf_ids], dtype=object
    )
    if not np.array_equal(replay_leaf_ids, expected_leaf_ids):
        raise WeightedRuleTreeError(
            "routing_mismatch",
            "typed DSL leaf routing does not reproduce sklearn training routes",
        )
    return result


def apply_weighted_rule_tree(
    frame: pd.DataFrame,
    result: Mapping[str, Any],
) -> np.ndarray:
    """Apply the persisted typed leaf rules and return one stable leaf id per row."""

    _require_frame(frame)
    result = validate_weighted_rule_tree(result)

    training = result.get("training")
    rules = result.get("rules")
    if not isinstance(training, Mapping) or not isinstance(rules, Sequence):
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree result is incomplete"
        )
    feature_order = training.get("feature_order")
    if not isinstance(feature_order, Sequence) or isinstance(
        feature_order, str | bytes | bytearray
    ):
        raise WeightedRuleTreeError(
            "invalid_result", "training feature_order must be a list"
        )
    features = _normalize_features(list(feature_order))
    _assert_columns(frame, features)
    for feature in features:
        _strict_numeric_feature(
            frame[feature],
            column=feature,
            require_nonempty=False,
            require_variable=False,
        )

    row_count = int(len(frame))
    hits = np.zeros((row_count, len(rules)), dtype=bool)
    leaf_ids: list[str] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise WeightedRuleTreeError(
                "invalid_result", "weighted rule tree rule must be an object"
            )
        leaf_id = rule.get("leaf_id")
        if not isinstance(leaf_id, str) or not leaf_id:
            raise WeightedRuleTreeError(
                "invalid_result", "weighted rule tree leaf_id is invalid"
            )
        condition = rule.get("condition")
        if not isinstance(condition, Mapping):
            raise WeightedRuleTreeError(
                "invalid_result", "weighted rule tree condition must be an object"
            )
        leaf_ids.append(leaf_id)
        try:
            canonical_condition = canonicalize_expression(condition)
            hits[:, index] = evaluate_expression_frame(
                frame, canonical_condition
            ).to_numpy(dtype=bool)
        except (StrategyError, TypeError, ValueError) as exc:
            raise WeightedRuleTreeError(
                "invalid_result",
                f"weighted rule tree condition is invalid: {exc}",
            ) from exc

    if not rules:
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree must contain at least one leaf rule"
        )
    match_counts = hits.sum(axis=1)
    if np.any(match_counts != 1):
        raise WeightedRuleTreeError(
            "routing_mismatch",
            "leaf rules must assign every row exactly once",
            unmatched_rows=int((match_counts == 0).sum()),
            multiply_matched_rows=int((match_counts > 1).sum()),
        )
    if row_count == 0:
        return np.array([], dtype=object)
    chosen = hits.argmax(axis=1)
    return np.array([leaf_ids[int(index)] for index in chosen], dtype=object)


def validate_weighted_rule_tree(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a persisted tree and return a detached canonical dict.

    The check is independent of :func:`build_weighted_rule_tree`: it validates
    exact schemas, topology, content-derived ids, leaf expressions, metric
    arithmetic/conservation, diagnostic child references, and the self hash.
    """

    canonical = _strict_json_detach(payload, path="weighted_rule_tree")
    if not isinstance(canonical, dict):
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree result must be an object"
        )
    _exact_keys(
        canonical,
        {
            "schema_version",
            "lifecycle",
            "training",
            "preprocessing",
            "directions",
            "budgets",
            "search",
            "tree",
            "rules",
            "checks",
            "result_hash",
        },
        path="weighted_rule_tree",
    )
    if canonical["schema_version"] != WEIGHTED_RULE_TREE_SCHEMA_VERSION:
        raise WeightedRuleTreeError(
            "invalid_result", "unsupported weighted rule tree schema version"
        )
    expected_hash = canonical["result_hash"]
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree result_hash is invalid"
        )
    unhashed = dict(canonical)
    unhashed.pop("result_hash")
    if _digest(unhashed) != expected_hash:
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree result hash mismatch"
        )

    lifecycle = _object_at(canonical["lifecycle"], path="lifecycle")
    _exact_keys(
        lifecycle,
        {"stage", "validation_status", "evidence_status"},
        path="lifecycle",
    )
    if lifecycle != {
        "stage": "development",
        "validation_status": "unvalidated",
        "evidence_status": "backtested",
    }:
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree lifecycle claims are invalid"
        )

    training = _object_at(canonical["training"], path="training")
    _exact_keys(
        training,
        {
            "row_count",
            "feature_order",
            "target_col",
            "sample_weight",
            "loan_amount_col",
            "overdue_amount_col",
            "seed",
            "sklearn_version",
            "cart",
        },
        path="training",
    )
    row_count = _positive_int(training["row_count"], path="training.row_count")
    feature_order = _string_list(
        training["feature_order"], path="training.feature_order"
    )
    if not feature_order or feature_order != sorted(set(feature_order)):
        raise WeightedRuleTreeError(
            "invalid_result", "training.feature_order must be sorted and unique"
        )
    target_col = _nonempty_string(training["target_col"], path="training.target_col")
    if target_col in feature_order:
        raise WeightedRuleTreeError(
            "invalid_result", "training target cannot also be a feature"
        )
    sample_weight = _object_at(training["sample_weight"], path="training.sample_weight")
    weight_available = sample_weight.get("status") == "available"
    if weight_available:
        _exact_keys(sample_weight, {"status", "column"}, path="training.sample_weight")
        weight_col = _nonempty_string(
            sample_weight["column"], path="training.sample_weight.column"
        )
        if weight_col in feature_order:
            raise WeightedRuleTreeError(
                "invalid_result", "training weight cannot also be a feature"
            )
    else:
        _exact_keys(sample_weight, {"status"}, path="training.sample_weight")
        if sample_weight.get("status") != "not_applicable":
            raise WeightedRuleTreeError(
                "invalid_result", "training sample_weight status is invalid"
            )
    loan_col = _nullable_column(
        training["loan_amount_col"], path="training.loan_amount_col"
    )
    overdue_col = _nullable_column(
        training["overdue_amount_col"], path="training.overdue_amount_col"
    )
    role_columns = [target_col]
    role_columns.extend(
        column
        for column in (
            sample_weight.get("column") if weight_available else None,
            loan_col,
            overdue_col,
        )
        if column is not None
    )
    if any(column in feature_order for column in role_columns) or len(
        role_columns
    ) != len(set(role_columns)):
        raise WeightedRuleTreeError(
            "invalid_result",
            "training target, weight, and amount roles must be distinct from features and each other",
        )
    _integer(training["seed"], path="training.seed")
    _nonempty_string(training["sklearn_version"], path="training.sklearn_version")
    cart = _object_at(training["cart"], path="training.cart")
    _exact_keys(
        cart,
        {
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_leaf_count",
            "min_weight_fraction_leaf",
        },
        path="training.cart",
    )
    if (
        cart["criterion"] != "gini"
        or cart["splitter"] != "best"
        or cart["min_samples_split"] != 2
        or isinstance(cart["min_samples_split"], bool)
    ):
        raise WeightedRuleTreeError(
            "invalid_result", "training CART fixed provenance is invalid"
        )
    depth_limit = _integer(cart["max_depth"], path="training.cart.max_depth")
    min_leaf_count = _integer(
        cart["min_leaf_count"], path="training.cart.min_leaf_count"
    )
    weight_fraction = _finite_number(
        cart["min_weight_fraction_leaf"],
        path="training.cart.min_weight_fraction_leaf",
    )
    _validate_fit_config(
        max_depth=depth_limit,
        min_leaf_count=min_leaf_count,
        min_weight_fraction_leaf=weight_fraction,
        seed=training["seed"],
    )

    preprocessing = _object_at(canonical["preprocessing"], path="preprocessing")
    _exact_keys(preprocessing, {"missing_policy", "medians"}, path="preprocessing")
    if preprocessing["missing_policy"] != "training_median_frozen_into_typed_dsl":
        raise WeightedRuleTreeError(
            "invalid_result", "preprocessing missing policy is invalid"
        )
    medians = _object_at(preprocessing["medians"], path="preprocessing.medians")
    if sorted(medians) != feature_order:
        raise WeightedRuleTreeError(
            "invalid_result", "preprocessing medians must exactly cover features"
        )
    for feature, value in medians.items():
        _finite_number(value, path=f"preprocessing.medians.{feature}")

    directions = _object_at(canonical["directions"], path="directions")
    if sorted(directions) != feature_order or any(
        value not in _DIRECTIONS for value in directions.values()
    ):
        raise WeightedRuleTreeError(
            "invalid_result", "directions must exactly cover selected features"
        )

    budget_payload = _object_at(canonical["budgets"], path="budgets")
    _exact_keys(budget_payload, set(_HARD_BUDGETS), path="budgets")
    resolved_budgets = WeightedRuleTreeBudgets(**budget_payload)
    _check_budget("rows", row_count, resolved_budgets.max_rows)
    _check_budget("features", len(feature_order), resolved_budgets.max_features)
    _check_budget("cells", row_count * len(feature_order), resolved_budgets.max_cells)

    search = _object_at(canonical["search"], path="search")
    _exact_keys(
        search,
        {"method", "truncated", "cutpoint_evaluations_upper_bound", "tie_break"},
        path="search",
    )
    expected_cutpoints = row_count * len(feature_order) * depth_limit
    if (
        search["method"] != "greedy_cart"
        or search["truncated"] is not False
        or search["cutpoint_evaluations_upper_bound"] != expected_cutpoints
    ):
        raise WeightedRuleTreeError(
            "invalid_result", "search method or budget accounting is inconsistent"
        )
    _check_budget(
        "cutpoint_evaluations",
        expected_cutpoints,
        resolved_budgets.max_cutpoint_evaluations,
    )
    tie_break = _object_at(search["tie_break"], path="search.tie_break")
    _exact_keys(
        tie_break,
        {"feature_order", "splitter", "random_state"},
        path="search.tie_break",
    )
    if tie_break != {
        "feature_order": "lexicographic_by_name",
        "splitter": "sklearn_best",
        "random_state": training["seed"],
    }:
        raise WeightedRuleTreeError(
            "invalid_result", "search tie-break evidence is inconsistent"
        )

    checks = _object_at(canonical["checks"], path="checks")
    _exact_keys(
        checks,
        {
            "sklearn_route_matches_typed_dsl",
            "all_training_rows_assigned_once",
            "conservation",
        },
        path="checks",
    )
    if checks != {
        "sklearn_route_matches_typed_dsl": True,
        "all_training_rows_assigned_once": True,
        "conservation": "passed",
    }:
        raise WeightedRuleTreeError(
            "invalid_result", "weighted rule tree checks are not successful"
        )

    tree = _object_at(canonical["tree"], path="tree")
    _exact_keys(
        tree,
        {
            "tree_id",
            "root_node_id",
            "node_count",
            "leaf_count",
            "leaf_ids",
            "nodes",
        },
        path="tree",
    )
    nodes = _list_at(tree["nodes"], path="tree.nodes")
    if not nodes:
        raise WeightedRuleTreeError("invalid_result", "tree.nodes cannot be empty")
    node_objects = [
        _object_at(node, path=f"tree.nodes[{index}]")
        for index, node in enumerate(nodes)
    ]
    node_ids = [
        _nonempty_string(node.get("node_id"), path=f"tree.nodes[{index}].node_id")
        for index, node in enumerate(node_objects)
    ]
    if len(node_ids) != len(set(node_ids)):
        raise WeightedRuleTreeError("invalid_result", "tree node ids must be unique")
    node_by_id = dict(zip(node_ids, node_objects, strict=True))
    root_node_id = _nonempty_string(tree["root_node_id"], path="tree.root_node_id")
    if root_node_id not in node_by_id:
        raise WeightedRuleTreeError("invalid_result", "tree root node does not exist")

    traversal_ids: list[str] = []
    expected_rules: list[
        tuple[str, list[dict[str, Any]], dict[str, Any], Mapping[str, Any]]
    ] = []
    visiting: set[str] = set()

    def validate_node(
        node_id: str,
        path: tuple[str, ...],
        clauses: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        if node_id in visiting:
            raise WeightedRuleTreeError("invalid_result", "tree contains a cycle")
        if node_id in traversal_ids:
            raise WeightedRuleTreeError(
                "invalid_result", "tree node has more than one parent"
            )
        visiting.add(node_id)
        traversal_ids.append(node_id)
        node = node_by_id[node_id]
        kind = node.get("kind")
        common = {"node_id", "kind", "depth", "path", "metrics"}
        if kind == "leaf":
            _exact_keys(node, common | {"rule_id"}, path=f"node.{node_id}")
        elif kind == "split":
            _exact_keys(
                node,
                common
                | {
                    "feature",
                    "threshold",
                    "sklearn_threshold",
                    "threshold_adjustment",
                    "missing_child",
                    "left_child_id",
                    "right_child_id",
                    "direction_diagnostic",
                },
                path=f"node.{node_id}",
            )
        else:
            raise WeightedRuleTreeError("invalid_result", "tree node kind is invalid")
        if node.get("depth") != len(path) or node.get("path") != list(path):
            raise WeightedRuleTreeError(
                "invalid_result", "tree node depth/path is inconsistent"
            )
        if len(path) > depth_limit:
            raise WeightedRuleTreeError(
                "invalid_result", "tree node exceeds the recorded max_depth"
            )
        metrics = _object_at(node["metrics"], path=f"node.{node_id}.metrics")
        if kind == "leaf":
            _nonempty_string(node["rule_id"], path=f"node.{node_id}.rule_id")
            condition = _path_condition(clauses)
            expected_rules.append((node_id, list(clauses), condition, metrics))
            visiting.remove(node_id)
            return {"kind": "leaf"}

        feature = node["feature"]
        if not isinstance(feature, str) or feature not in feature_order:
            raise WeightedRuleTreeError(
                "invalid_result", "split node references an unknown feature"
            )
        threshold = _finite_number(node["threshold"], path=f"node.{node_id}.threshold")
        sklearn_threshold = _finite_number(
            node["sklearn_threshold"], path=f"node.{node_id}.sklearn_threshold"
        )
        adjustment = node["threshold_adjustment"]
        if adjustment not in {"none", "training_partition_boundary"}:
            raise WeightedRuleTreeError(
                "invalid_result", "split threshold_adjustment is invalid"
            )
        if adjustment == "none" and threshold != sklearn_threshold:
            raise WeightedRuleTreeError(
                "invalid_result", "unadjusted threshold must equal sklearn threshold"
            )
        if (
            adjustment == "training_partition_boundary"
            and threshold == sklearn_threshold
        ):
            raise WeightedRuleTreeError(
                "invalid_result",
                "adjusted threshold must differ from sklearn threshold",
            )
        missing_child = node["missing_child"]
        if missing_child not in {"left", "right"}:
            raise WeightedRuleTreeError(
                "invalid_result", "split missing_child is invalid"
            )
        expected_missing_child = "left" if medians[feature] <= threshold else "right"
        if missing_child != expected_missing_child:
            raise WeightedRuleTreeError(
                "invalid_result", "split missing route disagrees with frozen median"
            )
        left_id = _nonempty_string(
            node["left_child_id"], path=f"node.{node_id}.left_child_id"
        )
        right_id = _nonempty_string(
            node["right_child_id"], path=f"node.{node_id}.right_child_id"
        )
        if (
            left_id == right_id
            or left_id not in node_by_id
            or right_id not in node_by_id
        ):
            raise WeightedRuleTreeError(
                "invalid_result", "split child references are invalid"
            )
        left_clause = _compare_clause(
            field=feature,
            operator="<=",
            value=threshold,
            missing=missing_child == "left",
        )
        right_clause = _compare_clause(
            field=feature,
            operator=">",
            value=threshold,
            missing=missing_child == "right",
        )
        left_structure = validate_node(
            left_id, (*path, "left"), (*clauses, left_clause)
        )
        right_structure = validate_node(
            right_id, (*path, "right"), (*clauses, right_clause)
        )
        visiting.remove(node_id)
        return {
            "kind": "split",
            "feature": feature,
            "threshold": threshold,
            "missing_child": missing_child,
            "left": left_structure,
            "right": right_structure,
        }

    structure = validate_node(root_node_id, (), ())
    if set(traversal_ids) != set(node_ids) or traversal_ids != node_ids:
        raise WeightedRuleTreeError(
            "invalid_result",
            "tree.nodes must be one complete deterministic pre-order traversal",
        )
    if tree["node_count"] != len(nodes):
        raise WeightedRuleTreeError("invalid_result", "tree.node_count is inconsistent")
    _check_budget("nodes", len(nodes), resolved_budgets.max_nodes)
    expected_leaf_ids = [item[0] for item in expected_rules]
    if (
        tree["leaf_count"] != len(expected_leaf_ids)
        or tree["leaf_ids"] != expected_leaf_ids
    ):
        raise WeightedRuleTreeError("invalid_result", "tree leaf index is inconsistent")
    if len(expected_leaf_ids) < 2 or len(nodes) != 2 * len(expected_leaf_ids) - 1:
        raise WeightedRuleTreeError(
            "invalid_result", "tree must be a non-root full binary tree"
        )

    fingerprint = _digest(
        {
            "schema_version": WEIGHTED_RULE_TREE_SCHEMA_VERSION,
            "feature_order": feature_order,
            "structure": structure,
        }
    )
    if tree["tree_id"] != f"tree-{fingerprint[:20]}":
        raise WeightedRuleTreeError("invalid_result", "tree_id is not content-derived")
    for node_id in traversal_ids:
        if node_id != _stable_node_id(fingerprint, node_by_id[node_id]):
            raise WeightedRuleTreeError(
                "invalid_result", "node_id is not content/path-derived"
            )

    root_metrics = _object_at(node_by_id[root_node_id]["metrics"], path="root.metrics")
    _validate_metrics_bundle(
        root_metrics,
        root_metrics=root_metrics,
        weight_available=weight_available,
        loan_available=loan_col is not None,
        overdue_available=overdue_col is not None,
        path="root.metrics",
    )
    if root_metrics["unweighted"]["total"] != row_count:
        raise WeightedRuleTreeError(
            "invalid_result", "root row total disagrees with training.row_count"
        )
    if (
        root_metrics["unweighted"]["good"] <= 0
        or root_metrics["unweighted"]["bad"] <= 0
    ):
        raise WeightedRuleTreeError(
            "invalid_result",
            "a successful fitted tree must contain both target classes",
        )
    for node_id in traversal_ids:
        if node_id == root_node_id:
            continue
        _validate_metrics_bundle(
            node_by_id[node_id]["metrics"],
            root_metrics=root_metrics,
            weight_available=weight_available,
            loan_available=loan_col is not None,
            overdue_available=overdue_col is not None,
            path=f"node.{node_id}.metrics",
        )
        if node_by_id[node_id]["kind"] == "leaf":
            leaf_metrics = node_by_id[node_id]["metrics"]
            if leaf_metrics["unweighted"]["total"] < min_leaf_count:
                raise WeightedRuleTreeError(
                    "invalid_result", "leaf violates the recorded min_leaf_count"
                )
            fraction_basis = (
                leaf_metrics["weighted"]
                if weight_available
                else leaf_metrics["unweighted"]
            )
            root_fraction_basis = (
                root_metrics["weighted"]
                if weight_available
                else root_metrics["unweighted"]
            )
            if float(fraction_basis["total"]) + 1e-12 < weight_fraction * float(
                root_fraction_basis["total"]
            ):
                raise WeightedRuleTreeError(
                    "invalid_result",
                    "leaf violates the recorded min_weight_fraction_leaf",
                )
    for node_id in traversal_ids:
        node = node_by_id[node_id]
        if node["kind"] != "split":
            continue
        left_id = node["left_child_id"]
        right_id = node["right_child_id"]
        _assert_metric_conservation(
            node["metrics"],
            node_by_id[left_id]["metrics"],
            node_by_id[right_id]["metrics"],
        )
        _validate_direction_payload(
            node["direction_diagnostic"],
            direction=directions[node["feature"]],
            left_metrics=node_by_id[left_id]["metrics"],
            right_metrics=node_by_id[right_id]["metrics"],
            weight_available=weight_available,
        )

    rules = _list_at(canonical["rules"], path="rules")
    if len(rules) != len(expected_rules):
        raise WeightedRuleTreeError(
            "invalid_result", "rules must contain exactly one rule per leaf"
        )
    for index, (raw_rule, expected) in enumerate(
        zip(rules, expected_rules, strict=True)
    ):
        rule = _object_at(raw_rule, path=f"rules[{index}]")
        _exact_keys(
            rule,
            {"rule_id", "leaf_id", "condition", "clauses", "metrics"},
            path=f"rules[{index}]",
        )
        leaf_id, clauses, condition, metrics = expected
        if rule["leaf_id"] != leaf_id or rule["metrics"] != metrics:
            raise WeightedRuleTreeError(
                "invalid_result", "rule leaf reference or metrics are inconsistent"
            )
        canonical_clauses = []
        for clause in _list_at(rule["clauses"], path=f"rules[{index}].clauses"):
            if not isinstance(clause, Mapping):
                raise WeightedRuleTreeError(
                    "invalid_result", "rule clause must be an object"
                )
            try:
                canonical_clauses.append(canonicalize_expression(clause))
            except StrategyError as exc:
                raise WeightedRuleTreeError(
                    "invalid_result", f"rule clause is invalid: {exc}"
                ) from exc
        try:
            canonical_condition = canonicalize_expression(rule["condition"])
        except (StrategyError, TypeError) as exc:
            raise WeightedRuleTreeError(
                "invalid_result", f"rule condition is invalid: {exc}"
            ) from exc
        if canonical_clauses != clauses or canonical_condition != condition:
            raise WeightedRuleTreeError(
                "invalid_result", "rule condition does not equal its exact leaf path"
            )
        expected_rule_id = f"rule-{_digest(condition)[:20]}"
        if (
            rule["rule_id"] != expected_rule_id
            or node_by_id[leaf_id]["rule_id"] != expected_rule_id
        ):
            raise WeightedRuleTreeError(
                "invalid_result", "rule_id is not condition-derived"
            )

    return canonical


def canonical_weighted_rule_tree_json(payload: Mapping[str, Any]) -> str:
    """Return the strict canonical JSON representation of a validated result."""

    return _canonical_json(validate_weighted_rule_tree(payload))


def _strict_json_detach(value: object, *, path: str) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WeightedRuleTreeError(
                "invalid_result", f"{path} contains a non-finite JSON number"
            )
        return float(value)
    if isinstance(value, Mapping):
        detached: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WeightedRuleTreeError(
                    "invalid_result", f"{path} contains a non-string object key"
                )
            detached[key] = _strict_json_detach(item, path=f"{path}.{key}")
        return detached
    if isinstance(value, list):
        return [
            _strict_json_detach(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise WeightedRuleTreeError(
        "invalid_result", f"{path} must contain only JSON-native values"
    )


def _object_at(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeightedRuleTreeError("invalid_result", f"{path} must be an object")
    return value


def _list_at(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise WeightedRuleTreeError("invalid_result", f"{path} must be a list")
    return value


def _exact_keys(payload: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise WeightedRuleTreeError(
            "invalid_result",
            f"{path} fields are invalid",
            path=path,
            missing_fields=sorted(expected - actual),
            unknown_fields=sorted(actual - expected),
        )


def _nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise WeightedRuleTreeError(
            "invalid_result", f"{path} must be a non-empty string"
        )
    return value


def _nullable_column(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path=path)


def _string_list(value: object, *, path: str) -> list[str]:
    items = _list_at(value, path=path)
    return [
        _nonempty_string(item, path=f"{path}[{index}]")
        for index, item in enumerate(items)
    ]


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WeightedRuleTreeError("invalid_result", f"{path} must be an integer")
    return int(value)


def _positive_int(value: object, *, path: str) -> int:
    result = _integer(value, path=path)
    if result <= 0:
        raise WeightedRuleTreeError(
            "invalid_result", f"{path} must be a positive integer"
        )
    return result


def _finite_number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WeightedRuleTreeError("invalid_result", f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WeightedRuleTreeError("invalid_result", f"{path} must be a finite number")
    return result


_BASE_METRIC_KEYS = {
    "total",
    "good",
    "bad",
    "bad_rate",
    "share",
    "bad_capture",
    "lift",
}
_LOAN_METRIC_KEYS = {
    "loan_amount_total",
    "loan_amount_coverage_count",
    "loan_amount_coverage",
    "loan_amount_coverage_rate",
}
_OVERDUE_METRIC_KEYS = {
    "overdue_amount_total",
    "overdue_amount_coverage_count",
    "overdue_amount_coverage",
    "overdue_amount_coverage_rate",
}
_PAIR_METRIC_KEYS = {
    "amount_pair_coverage_count",
    "amount_pair_coverage",
    "amount_pair_coverage_rate",
    "paired_loan_amount_total",
    "paired_overdue_amount_total",
    "overdue_rate",
}


def _validate_metrics_bundle(
    payload: object,
    *,
    root_metrics: Mapping[str, Any],
    weight_available: bool,
    loan_available: bool,
    overdue_available: bool,
    path: str,
) -> None:
    metrics = _object_at(payload, path=path)
    _exact_keys(metrics, {"unweighted", "weighted"}, path=path)
    expected = set(_BASE_METRIC_KEYS)
    if loan_available:
        expected |= _LOAN_METRIC_KEYS
    if overdue_available:
        expected |= _OVERDUE_METRIC_KEYS
    if loan_available and overdue_available:
        expected |= _PAIR_METRIC_KEYS

    unweighted = _object_at(metrics["unweighted"], path=f"{path}.unweighted")
    _exact_keys(unweighted, expected, path=f"{path}.unweighted")
    _validate_metric_basis(
        unweighted,
        root=_object_at(root_metrics["unweighted"], path="root.unweighted"),
        unweighted=unweighted,
        weighted=False,
        loan_available=loan_available,
        overdue_available=overdue_available,
        path=f"{path}.unweighted",
    )
    weighted_payload = _object_at(metrics["weighted"], path=f"{path}.weighted")
    if not weight_available:
        if weighted_payload != {"status": "not_applicable"}:
            raise WeightedRuleTreeError(
                "invalid_result", f"{path}.weighted must be not_applicable"
            )
        return
    _exact_keys(weighted_payload, expected | {"status"}, path=f"{path}.weighted")
    if weighted_payload["status"] != "available":
        raise WeightedRuleTreeError(
            "invalid_result", f"{path}.weighted status must be available"
        )
    root_weighted = _object_at(root_metrics["weighted"], path="root.weighted")
    _validate_metric_basis(
        weighted_payload,
        root=root_weighted,
        unweighted=unweighted,
        weighted=True,
        loan_available=loan_available,
        overdue_available=overdue_available,
        path=f"{path}.weighted",
    )


def _validate_metric_basis(
    payload: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    unweighted: Mapping[str, Any],
    weighted: bool,
    loan_available: bool,
    overdue_available: bool,
    path: str,
) -> None:
    total = _finite_number(payload["total"], path=f"{path}.total")
    good = _finite_number(payload["good"], path=f"{path}.good")
    bad = _finite_number(payload["bad"], path=f"{path}.bad")
    if min(total, good, bad) < 0 or total <= 0 or not _close(total, good + bad):
        raise WeightedRuleTreeError(
            "invalid_result", f"{path} population totals are inconsistent"
        )
    if not weighted:
        for key in ("total", "good", "bad"):
            _integer(payload[key], path=f"{path}.{key}")
    root_total = _finite_number(root["total"], path="root.total")
    root_bad = _finite_number(root["bad"], path="root.bad")
    bad_rate = _finite_number(payload["bad_rate"], path=f"{path}.bad_rate")
    share = _finite_number(payload["share"], path=f"{path}.share")
    capture = _finite_number(payload["bad_capture"], path=f"{path}.bad_capture")
    lift = _finite_number(payload["lift"], path=f"{path}.lift")
    root_bad_rate = _safe_ratio(root_bad, root_total)
    expected_values = (
        (bad_rate, _safe_ratio(bad, total)),
        (share, _safe_ratio(total, root_total)),
        (capture, _safe_ratio(bad, root_bad)),
        (lift, _safe_ratio(bad_rate, root_bad_rate)),
    )
    if any(not _close(actual, expected) for actual, expected in expected_values):
        raise WeightedRuleTreeError(
            "invalid_result", f"{path} derived rates are inconsistent"
        )
    if (
        not 0 <= bad_rate <= 1
        or not 0 <= share <= 1
        or not 0 <= capture <= 1
        or lift < 0
    ):
        raise WeightedRuleTreeError(
            "invalid_result", f"{path} derived rates are outside valid bounds"
        )

    for prefix, available in (
        ("loan_amount", loan_available),
        ("overdue_amount", overdue_available),
    ):
        if not available:
            continue
        amount_total = _finite_number(
            payload[f"{prefix}_total"], path=f"{path}.{prefix}_total"
        )
        coverage_count = _integer(
            payload[f"{prefix}_coverage_count"],
            path=f"{path}.{prefix}_coverage_count",
        )
        coverage = _finite_number(
            payload[f"{prefix}_coverage"], path=f"{path}.{prefix}_coverage"
        )
        coverage_rate = _finite_number(
            payload[f"{prefix}_coverage_rate"],
            path=f"{path}.{prefix}_coverage_rate",
        )
        raw_total = _integer(unweighted["total"], path=f"{path}.unweighted.total")
        if (
            amount_total < 0
            or not 0 <= coverage_count <= raw_total
            or not 0 <= coverage <= total
            or not _close(coverage_rate, _safe_ratio(coverage, total))
            or not 0 <= coverage_rate <= 1
            or (not weighted and not _close(coverage, coverage_count))
        ):
            raise WeightedRuleTreeError(
                "invalid_result", f"{path}.{prefix} coverage is inconsistent"
            )

    if loan_available and overdue_available:
        pair_count = _integer(
            payload["amount_pair_coverage_count"],
            path=f"{path}.amount_pair_coverage_count",
        )
        pair_coverage = _finite_number(
            payload["amount_pair_coverage"], path=f"{path}.amount_pair_coverage"
        )
        pair_rate = _finite_number(
            payload["amount_pair_coverage_rate"],
            path=f"{path}.amount_pair_coverage_rate",
        )
        paired_loan = _finite_number(
            payload["paired_loan_amount_total"],
            path=f"{path}.paired_loan_amount_total",
        )
        paired_overdue = _finite_number(
            payload["paired_overdue_amount_total"],
            path=f"{path}.paired_overdue_amount_total",
        )
        raw_total = int(unweighted["total"])
        if (
            not 0 <= pair_count <= raw_total
            or not 0 <= pair_coverage <= total
            or not _close(pair_rate, _safe_ratio(pair_coverage, total))
            or not 0 <= pair_rate <= 1
            or (not weighted and not _close(pair_coverage, pair_count))
            or pair_count > int(payload["loan_amount_coverage_count"])
            or pair_count > int(payload["overdue_amount_coverage_count"])
            or pair_coverage > float(payload["loan_amount_coverage"]) + 1e-12
            or pair_coverage > float(payload["overdue_amount_coverage"]) + 1e-12
            or paired_loan < 0
            or paired_overdue < 0
            or paired_loan > float(payload["loan_amount_total"]) + 1e-9
            or paired_overdue > float(payload["overdue_amount_total"]) + 1e-9
        ):
            raise WeightedRuleTreeError(
                "invalid_result", f"{path} paired amount coverage is inconsistent"
            )
        overdue_rate = payload["overdue_rate"]
        if paired_loan == 0:
            if overdue_rate is not None:
                raise WeightedRuleTreeError(
                    "invalid_result",
                    f"{path}.overdue_rate must be null without a denominator",
                )
        else:
            actual_rate = _finite_number(overdue_rate, path=f"{path}.overdue_rate")
            if not _close(actual_rate, paired_overdue / paired_loan):
                raise WeightedRuleTreeError(
                    "invalid_result", f"{path}.overdue_rate is inconsistent"
                )


def _validate_direction_payload(
    payload: object,
    *,
    direction: str,
    left_metrics: Mapping[str, Any],
    right_metrics: Mapping[str, Any],
    weight_available: bool,
) -> None:
    diagnostic = _object_at(payload, path="direction_diagnostic")
    _exact_keys(
        diagnostic,
        {
            "expected_direction",
            "status",
            "basis",
            "primary_bad_rate_delta",
            "left",
            "right",
        },
        path="direction_diagnostic",
    )
    if diagnostic["expected_direction"] != direction:
        raise WeightedRuleTreeError(
            "invalid_result", "direction diagnostic references the wrong direction"
        )
    basis = "weighted" if weight_available else "unweighted"
    if diagnostic["basis"] != basis:
        raise WeightedRuleTreeError(
            "invalid_result", "direction diagnostic basis is inconsistent"
        )
    primary_rates: list[float] = []
    for side, metrics in (("left", left_metrics), ("right", right_metrics)):
        item = _object_at(diagnostic[side], path=f"direction_diagnostic.{side}")
        _exact_keys(
            item, {"count", "bad_rate", "weighted"}, path=f"direction_diagnostic.{side}"
        )
        if item["count"] != metrics["unweighted"]["total"] or not _close(
            _finite_number(
                item["bad_rate"], path=f"direction_diagnostic.{side}.bad_rate"
            ),
            float(metrics["unweighted"]["bad_rate"]),
        ):
            raise WeightedRuleTreeError(
                "invalid_result", "direction child count/rate is inconsistent"
            )
        weighted_payload = _object_at(
            item["weighted"], path=f"direction_diagnostic.{side}.weighted"
        )
        if weight_available:
            _exact_keys(
                weighted_payload,
                {"status", "total", "bad_rate"},
                path=f"direction_diagnostic.{side}.weighted",
            )
            if (
                weighted_payload["status"] != "available"
                or not _close(
                    _finite_number(
                        weighted_payload["total"],
                        path=f"direction_diagnostic.{side}.weighted.total",
                    ),
                    float(metrics["weighted"]["total"]),
                )
                or not _close(
                    _finite_number(
                        weighted_payload["bad_rate"],
                        path=f"direction_diagnostic.{side}.weighted.bad_rate",
                    ),
                    float(metrics["weighted"]["bad_rate"]),
                )
            ):
                raise WeightedRuleTreeError(
                    "invalid_result",
                    "weighted direction child metrics are inconsistent",
                )
            primary_rates.append(float(weighted_payload["bad_rate"]))
        else:
            if weighted_payload != {"status": "not_applicable"}:
                raise WeightedRuleTreeError(
                    "invalid_result", "direction weighted status must be not_applicable"
                )
            primary_rates.append(float(item["bad_rate"]))
    left_rate, right_rate = primary_rates
    delta = _finite_number(
        diagnostic["primary_bad_rate_delta"],
        path="direction_diagnostic.primary_bad_rate_delta",
    )
    if not _close(delta, right_rate - left_rate):
        raise WeightedRuleTreeError(
            "invalid_result", "direction primary bad-rate delta is inconsistent"
        )
    if direction == "unordered":
        expected_status = "inconclusive"
    elif direction == "increasing":
        expected_status = "consistent" if right_rate >= left_rate else "violation"
    else:
        expected_status = "consistent" if right_rate <= left_rate else "violation"
    if diagnostic["status"] != expected_status:
        raise WeightedRuleTreeError(
            "invalid_result", "direction diagnostic status is inconsistent"
        )


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _require_frame(frame: object) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise WeightedRuleTreeError("invalid_input", "frame must be a pandas DataFrame")


def _column_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WeightedRuleTreeError(
            "invalid_input", f"{field} must be a non-empty string", field=field
        )
    return value


def _optional_column_name(value: object, *, field: str) -> str | None:
    if value is None or value == "":
        return None
    return _column_name(value, field=field)


def _normalize_features(feature_cols: object) -> list[str]:
    if not isinstance(feature_cols, Sequence) or isinstance(
        feature_cols, str | bytes | bytearray
    ):
        raise WeightedRuleTreeError(
            "invalid_feature", "feature_cols must be a list of column names"
        )
    names: list[str] = []
    for value in feature_cols:
        if not isinstance(value, str) or not value.strip():
            raise WeightedRuleTreeError(
                "invalid_feature", "every selected feature must be a non-empty string"
            )
        names.append(value)
    normalized = sorted(set(names))
    if not normalized:
        raise WeightedRuleTreeError(
            "invalid_feature", "at least one feature must be selected"
        )
    return normalized


def _assert_distinct_roles(
    *,
    feature_order: Sequence[str],
    target_col: str,
    sample_weight_col: str | None,
    loan_amount_col: str | None,
    overdue_amount_col: str | None,
) -> None:
    role_columns = {
        "target_col": target_col,
        "sample_weight_col": sample_weight_col,
        "loan_amount_col": loan_amount_col,
        "overdue_amount_col": overdue_amount_col,
    }
    feature_conflicts = sorted(
        name
        for name in role_columns.values()
        if name is not None and name in feature_order
    )
    if feature_conflicts:
        raise WeightedRuleTreeError(
            "invalid_feature",
            "target, weight, and amount role columns cannot be selected as features",
            columns=feature_conflicts,
        )
    assigned = [name for name in role_columns.values() if name is not None]
    duplicates = sorted({name for name in assigned if assigned.count(name) > 1})
    if duplicates:
        raise WeightedRuleTreeError(
            "invalid_input",
            "target, weight, loan, and overdue roles must use distinct columns",
            columns=duplicates,
        )


def _assert_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise WeightedRuleTreeError(
            "missing_columns",
            "required columns are missing",
            columns=sorted(set(missing)),
        )
    ambiguous = [name for name in columns if int((frame.columns == name).sum()) != 1]
    if ambiguous:
        raise WeightedRuleTreeError(
            "invalid_input",
            "required columns must be unique in the DataFrame",
            columns=sorted(set(ambiguous)),
        )


def _validate_fit_config(
    *,
    max_depth: object,
    min_leaf_count: object,
    min_weight_fraction_leaf: object,
    seed: object,
) -> None:
    if (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, Integral)
        or not 1 <= int(max_depth) <= 8
    ):
        raise WeightedRuleTreeError(
            "invalid_config", "max_depth must be an integer between 1 and 8"
        )
    if (
        isinstance(min_leaf_count, bool)
        or not isinstance(min_leaf_count, Integral)
        or int(min_leaf_count) < 1
    ):
        raise WeightedRuleTreeError(
            "invalid_config", "min_leaf_count must be a positive integer"
        )
    if (
        isinstance(min_weight_fraction_leaf, bool)
        or not isinstance(min_weight_fraction_leaf, Real)
        or not math.isfinite(float(min_weight_fraction_leaf))
        or not 0 <= float(min_weight_fraction_leaf) <= 0.5
    ):
        raise WeightedRuleTreeError(
            "invalid_config",
            "min_weight_fraction_leaf must be a finite number between 0 and 0.5",
        )
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise WeightedRuleTreeError("invalid_config", "seed must be an integer")


def _resolve_budgets(
    budgets: WeightedRuleTreeBudgets | Mapping[str, int] | None,
) -> WeightedRuleTreeBudgets:
    if budgets is None:
        return WeightedRuleTreeBudgets()
    if isinstance(budgets, WeightedRuleTreeBudgets):
        return budgets
    if not isinstance(budgets, Mapping):
        raise WeightedRuleTreeError("invalid_config", "budgets must be an object")
    unexpected = sorted(set(budgets) - set(_HARD_BUDGETS))
    if unexpected:
        raise WeightedRuleTreeError(
            "invalid_config", "budgets contain unsupported fields", fields=unexpected
        )
    values = asdict(WeightedRuleTreeBudgets())
    values.update(budgets)
    return WeightedRuleTreeBudgets(**values)


def _check_budget(dimension: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise WeightedRuleTreeError(
            "budget_exceeded",
            f"weighted rule tree {dimension} budget exceeded: actual={actual}, limit={limit}",
            dimension=dimension,
            actual=int(actual),
            limit=int(limit),
        )


def _strict_binary_target(series: pd.Series, *, column: str) -> np.ndarray:
    values = series.to_numpy(dtype=object)
    normalized = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        if _is_null(value) or isinstance(value, str | bytes | bytearray | bool):
            raise WeightedRuleTreeError(
                "invalid_target",
                f"target column `{column}` must contain only numeric 0/1 values",
                column=column,
                row=index,
            )
        if (
            not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) not in {0.0, 1.0}
        ):
            raise WeightedRuleTreeError(
                "invalid_target",
                f"target column `{column}` must contain only numeric 0/1 values",
                column=column,
                row=index,
            )
        normalized[index] = int(value)
    return normalized


def _fit_feature_matrix(
    frame: pd.DataFrame, feature_order: Sequence[str]
) -> tuple[np.ndarray, dict[str, float]]:
    columns: list[np.ndarray] = []
    medians: dict[str, float] = {}
    for column in feature_order:
        values, missing = _strict_numeric_feature(
            frame[column], column=column, require_nonempty=True, require_variable=True
        )
        present = values[~missing]
        median = float(np.median(present))
        if not math.isfinite(median):
            raise WeightedRuleTreeError(
                "invalid_feature",
                f"feature `{column}` has no finite median",
                column=column,
            )
        filled = values.copy()
        filled[missing] = median
        columns.append(filled)
        medians[column] = median
    return np.column_stack(columns).astype(float, copy=False), medians


def _strict_numeric_feature(
    series: pd.Series,
    *,
    column: str,
    require_nonempty: bool,
    require_variable: bool,
) -> tuple[np.ndarray, np.ndarray]:
    raw = series.to_numpy(dtype=object)
    values = np.empty(len(raw), dtype=float)
    missing = np.zeros(len(raw), dtype=bool)
    for index, value in enumerate(raw):
        if _is_null(value):
            missing[index] = True
            values[index] = np.nan
            continue
        if isinstance(value, bool | str | bytes | bytearray) or not isinstance(
            value, Real
        ):
            raise WeightedRuleTreeError(
                "invalid_feature",
                f"feature `{column}` contains a present non-numeric value",
                column=column,
                row=index,
            )
        normalized = float(value)
        if not math.isfinite(normalized):
            raise WeightedRuleTreeError(
                "invalid_feature",
                f"feature `{column}` contains an infinite value",
                column=column,
                row=index,
            )
        values[index] = normalized
    present = values[~missing]
    if require_nonempty and not len(present):
        raise WeightedRuleTreeError(
            "invalid_feature", f"feature `{column}` is entirely missing", column=column
        )
    if require_variable and len(np.unique(present)) < 2:
        raise WeightedRuleTreeError(
            "invalid_feature", f"feature `{column}` is constant", column=column
        )
    return values, missing


def _strict_positive_weights(series: pd.Series, *, column: str) -> np.ndarray:
    values = _strict_numeric_vector(series, column=column, code="invalid_weight")
    if np.any(values <= 0):
        raise WeightedRuleTreeError(
            "invalid_weight",
            f"sample weight column `{column}` must be strictly positive",
            column=column,
        )
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise WeightedRuleTreeError(
            "invalid_weight",
            f"sample weight column `{column}` must have a finite positive total",
            column=column,
        )
    return values


def _strict_amounts(series: pd.Series, *, column: str) -> np.ndarray:
    raw = series.to_numpy(dtype=object)
    values = np.full(len(raw), np.nan, dtype=float)
    for index, value in enumerate(raw):
        if _is_null(value):
            continue
        if (
            isinstance(value, bool | str | bytes | bytearray)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise WeightedRuleTreeError(
                "invalid_amount",
                f"amount column `{column}` present values must be finite and non-negative",
                column=column,
                row=index,
            )
        values[index] = float(value)
    return values


def _strict_numeric_vector(series: pd.Series, *, column: str, code: str) -> np.ndarray:
    raw = series.to_numpy(dtype=object)
    values = np.empty(len(raw), dtype=float)
    for index, value in enumerate(raw):
        if (
            _is_null(value)
            or isinstance(value, bool | str | bytes | bytearray)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise WeightedRuleTreeError(
                code,
                f"column `{column}` must contain finite numeric values",
                column=column,
                row=index,
            )
        values[index] = float(value)
    return values


def _normalize_directions(
    directions: Mapping[str, str] | None, feature_order: Sequence[str]
) -> dict[str, str]:
    if directions is None:
        return {feature: "unordered" for feature in feature_order}
    if not isinstance(directions, Mapping):
        raise WeightedRuleTreeError("invalid_config", "directions must be an object")
    unexpected = sorted(set(directions) - set(feature_order))
    if unexpected:
        raise WeightedRuleTreeError(
            "invalid_config",
            "directions reference unselected features",
            features=unexpected,
        )
    normalized: dict[str, str] = {}
    for feature in feature_order:
        value = directions.get(feature, "unordered")
        if not isinstance(value, str) or value not in _DIRECTIONS:
            raise WeightedRuleTreeError(
                "invalid_config",
                "direction must be increasing, decreasing, or unordered",
                feature=feature,
            )
        normalized[feature] = value
    return normalized


def _sparse_column_mask(matrix: Any, column: int, row_count: int) -> np.ndarray:
    return np.asarray(matrix[:, column].toarray(), dtype=bool).reshape(row_count)


def _assert_child_partition(
    parent: np.ndarray, left: np.ndarray, right: np.ndarray
) -> None:
    if np.any(left & right) or not np.array_equal(parent, left | right):
        raise WeightedRuleTreeError(
            "routing_mismatch", "CART child masks are not a disjoint parent partition"
        )


def _equivalent_threshold(
    *,
    values: np.ndarray,
    node_mask: np.ndarray,
    sklearn_left_mask: np.ndarray,
    sklearn_threshold: float,
) -> tuple[float, str]:
    expected = sklearn_left_mask[node_mask]
    node_values = values[node_mask]
    if np.array_equal(node_values <= sklearn_threshold, expected):
        return sklearn_threshold, "none"
    left_values = node_values[expected]
    right_values = node_values[~expected]
    if not len(left_values) or not len(right_values):
        raise WeightedRuleTreeError(
            "routing_mismatch", "CART split has an empty training child"
        )
    left_max = float(left_values.max())
    right_min = float(right_values.min())
    if not left_max < right_min:
        raise WeightedRuleTreeError(
            "routing_mismatch",
            "float32 CART routing cannot be represented by a numeric DSL boundary",
            left_max=left_max,
            right_min=right_min,
        )
    if not np.array_equal(node_values <= left_max, expected):
        raise WeightedRuleTreeError(
            "routing_mismatch",
            "training-partition threshold calibration failed",
        )
    return left_max, "training_partition_boundary"


def _metrics_bundle(
    mask: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None,
    root_mask: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    unweighted = _population_stats(
        mask,
        target,
        weights=None,
        root_mask=root_mask,
        loan_values=loan_values,
        overdue_values=overdue_values,
    )
    if weights is None:
        weighted: dict[str, Any] = {"status": "not_applicable"}
    else:
        weighted = {
            "status": "available",
            **_population_stats(
                mask,
                target,
                weights=weights,
                root_mask=root_mask,
                loan_values=loan_values,
                overdue_values=overdue_values,
            ),
        }
    return {"unweighted": unweighted, "weighted": weighted}


def _population_stats(
    mask: np.ndarray,
    target: np.ndarray,
    *,
    weights: np.ndarray | None,
    root_mask: np.ndarray,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> dict[str, Any]:
    if weights is None:
        total: int | float = int(mask.sum())
        bad: int | float = int(target[mask].sum())
        root_total = int(root_mask.sum())
        root_bad = int(target[root_mask].sum())
        good: int | float = int(total - bad)
    else:
        total = float(weights[mask].sum())
        bad = float(weights[mask & (target == 1)].sum())
        good = float(total - bad)
        root_total = float(weights[root_mask].sum())
        root_bad = float(weights[root_mask & (target == 1)].sum())
    bad_rate = _safe_ratio(bad, total)
    root_bad_rate = _safe_ratio(root_bad, root_total)
    result: dict[str, Any] = {
        "total": total,
        "good": good,
        "bad": bad,
        "bad_rate": bad_rate,
        "share": _safe_ratio(total, root_total),
        "bad_capture": _safe_ratio(bad, root_bad),
        "lift": _safe_ratio(bad_rate, root_bad_rate),
    }
    if loan_values is not None:
        result.update(
            _amount_coverage_stats(
                mask,
                loan_values,
                weights=weights,
                prefix="loan_amount",
                node_total=total,
            )
        )
    if overdue_values is not None:
        result.update(
            _amount_coverage_stats(
                mask,
                overdue_values,
                weights=weights,
                prefix="overdue_amount",
                node_total=total,
            )
        )
    if loan_values is not None and overdue_values is not None:
        pair_mask = mask & np.isfinite(loan_values) & np.isfinite(overdue_values)
        pair_count = int(pair_mask.sum())
        if weights is None:
            pair_coverage: int | float = pair_count
            paired_loan = float(loan_values[pair_mask].sum())
            paired_overdue = float(overdue_values[pair_mask].sum())
        else:
            pair_coverage = float(weights[pair_mask].sum())
            paired_loan = float((loan_values[pair_mask] * weights[pair_mask]).sum())
            paired_overdue = float(
                (overdue_values[pair_mask] * weights[pair_mask]).sum()
            )
        result.update(
            {
                "amount_pair_coverage_count": pair_count,
                "amount_pair_coverage": pair_coverage,
                "amount_pair_coverage_rate": _safe_ratio(pair_coverage, total),
                "paired_loan_amount_total": paired_loan,
                "paired_overdue_amount_total": paired_overdue,
                "overdue_rate": (
                    None if paired_loan == 0 else float(paired_overdue / paired_loan)
                ),
            }
        )
    return result


def _amount_coverage_stats(
    mask: np.ndarray,
    values: np.ndarray,
    *,
    weights: np.ndarray | None,
    prefix: str,
    node_total: int | float,
) -> dict[str, int | float]:
    available = mask & np.isfinite(values)
    coverage_count = int(available.sum())
    if weights is None:
        coverage: int | float = coverage_count
        amount_total = float(values[available].sum())
    else:
        coverage = float(weights[available].sum())
        amount_total = float((values[available] * weights[available]).sum())
    return {
        f"{prefix}_total": amount_total,
        f"{prefix}_coverage_count": coverage_count,
        f"{prefix}_coverage": coverage,
        f"{prefix}_coverage_rate": _safe_ratio(coverage, node_total),
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _direction_diagnostic(
    *,
    direction: str,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    left_count = int(left_mask.sum())
    right_count = int(right_mask.sum())
    left_unweighted_rate = float(target[left_mask].mean())
    right_unweighted_rate = float(target[right_mask].mean())
    if weights is None:
        basis = "unweighted"
        left_rate = left_unweighted_rate
        right_rate = right_unweighted_rate
        left_weighted: dict[str, Any] = {"status": "not_applicable"}
        right_weighted: dict[str, Any] = {"status": "not_applicable"}
    else:
        basis = "weighted"
        left_total = float(weights[left_mask].sum())
        right_total = float(weights[right_mask].sum())
        left_bad = float(weights[left_mask & (target == 1)].sum())
        right_bad = float(weights[right_mask & (target == 1)].sum())
        left_rate = left_bad / left_total
        right_rate = right_bad / right_total
        left_weighted = {
            "status": "available",
            "total": left_total,
            "bad_rate": float(left_rate),
        }
        right_weighted = {
            "status": "available",
            "total": right_total,
            "bad_rate": float(right_rate),
        }
    if direction == "unordered":
        status = "inconclusive"
    elif direction == "increasing":
        status = "consistent" if right_rate >= left_rate else "violation"
    else:
        status = "consistent" if right_rate <= left_rate else "violation"
    return {
        "expected_direction": direction,
        "status": status,
        "basis": basis,
        "primary_bad_rate_delta": float(right_rate - left_rate),
        "left": {
            "count": left_count,
            "bad_rate": left_unweighted_rate,
            "weighted": left_weighted,
        },
        "right": {
            "count": right_count,
            "bad_rate": right_unweighted_rate,
            "weighted": right_weighted,
        },
    }


def _assert_conservation(
    temporary: Mapping[int, Mapping[str, Any]],
    *,
    target: np.ndarray,
    weights: np.ndarray | None,
    loan_values: np.ndarray | None,
    overdue_values: np.ndarray | None,
) -> None:
    leaves = [node for node in temporary.values() if node["kind"] == "leaf"]
    memberships = np.zeros(len(target), dtype=int)
    for leaf in leaves:
        memberships += np.asarray(leaf["mask"], dtype=int)
    if np.any(memberships != 1):
        raise WeightedRuleTreeError(
            "routing_mismatch",
            "leaf masks do not assign every training row exactly once",
        )

    for node in temporary.values():
        if node["kind"] != "split":
            continue
        parent_mask = np.asarray(node["mask"], dtype=bool)
        left_mask = np.asarray(temporary[int(node["left_raw"])]["mask"], dtype=bool)
        right_mask = np.asarray(temporary[int(node["right_raw"])]["mask"], dtype=bool)
        _assert_child_partition(parent_mask, left_mask, right_mask)
        _assert_metric_conservation(
            node["metrics"],
            temporary[int(node["left_raw"])]["metrics"],
            temporary[int(node["right_raw"])]["metrics"],
        )

    root = temporary[0]["metrics"]
    for key in ("total", "good", "bad"):
        leaf_total = sum(float(leaf["metrics"]["unweighted"][key]) for leaf in leaves)
        if not math.isclose(
            leaf_total, float(root["unweighted"][key]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise WeightedRuleTreeError(
                "routing_mismatch", f"leaf {key} conservation failed"
            )
    if weights is not None:
        for key in ("total", "good", "bad"):
            leaf_total = sum(float(leaf["metrics"]["weighted"][key]) for leaf in leaves)
            if not math.isclose(
                leaf_total, float(root["weighted"][key]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise WeightedRuleTreeError(
                    "routing_mismatch", f"weighted leaf {key} conservation failed"
                )
    del loan_values, overdue_values  # values are covered by node metric conservation.


def _assert_metric_conservation(
    parent: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]
) -> None:
    for basis in ("unweighted", "weighted"):
        if parent[basis].get("status") == "not_applicable":
            continue
        keys = [
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
        ]
        for key in keys:
            if key not in parent[basis]:
                continue
            children = float(left[basis][key]) + float(right[basis][key])
            if not math.isclose(
                float(parent[basis][key]), children, rel_tol=1e-12, abs_tol=1e-9
            ):
                raise WeightedRuleTreeError(
                    "routing_mismatch",
                    f"node metric conservation failed for {basis}.{key}",
                )


def _structure_payload(
    temporary: Mapping[int, Mapping[str, Any]], raw_node_id: int
) -> dict[str, Any]:
    node = temporary[raw_node_id]
    if node["kind"] == "leaf":
        return {"kind": "leaf"}
    return {
        "kind": "split",
        "feature": node["feature"],
        "threshold": node["threshold"],
        "missing_child": node["missing_child"],
        "left": _structure_payload(temporary, int(node["left_raw"])),
        "right": _structure_payload(temporary, int(node["right_raw"])),
    }


def _stable_node_id(tree_fingerprint: str, node: Mapping[str, Any]) -> str:
    prefix = "leaf" if node["kind"] == "leaf" else "node"
    digest = _digest(
        {
            "tree_fingerprint": tree_fingerprint,
            "kind": node["kind"],
            "path": list(node["path"]),
        }
    )
    return f"{prefix}-{digest[:20]}"


def _compare_clause(
    *, field: str, operator: str, value: float, missing: bool
) -> dict[str, Any]:
    return canonicalize_expression(
        {
            "op": "compare",
            "field": field,
            "operator": operator,
            "value": float(value),
            "missing": "match" if missing else "no_match",
        }
    )


def _path_condition(clauses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not clauses:
        raise WeightedRuleTreeError(
            "infeasible_tree", "a root-only tree cannot produce a strategy rule"
        )
    if len(clauses) == 1:
        return canonicalize_expression(dict(clauses[0]))
    return canonicalize_expression(
        {"op": "and", "args": [dict(clause) for clause in clauses]}
    )


def _is_null(value: object) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool | np.bool_) else False


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "DEFAULT_WEIGHTED_RULE_TREE_SEED",
    "WEIGHTED_RULE_TREE_SCHEMA_VERSION",
    "WeightedRuleTreeBudgets",
    "WeightedRuleTreeError",
    "apply_weighted_rule_tree",
    "build_weighted_rule_tree",
    "canonical_weighted_rule_tree_json",
    "validate_weighted_rule_tree",
]
