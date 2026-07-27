"""Pure bounded split-candidate search for one authenticated tree node."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import hmac
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from marvis.packs.strategy.errors import StrategyError


INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION = (
    "strategy.interactive-tree-split-search.v1"
)
INTERACTIVE_TREE_SPLIT_SEARCH_PRODUCER_VERSION = (
    "strategy.interactive-tree-split-search/1"
)
MAX_SEARCH_FEATURES = 50
MAX_THRESHOLDS_PER_FEATURE = 20
MAX_SEARCH_CANDIDATES = MAX_SEARCH_FEATURES * MAX_THRESHOLDS_PER_FEATURE
MAX_ROW_EVALUATIONS = 20_000_000
_DIRECTIONS = frozenset({"increasing", "decreasing", "unordered"})


def search_interactive_tree_split_candidates(
    frame: pd.DataFrame,
    *,
    node_mask: np.ndarray,
    node_id: str,
    source_tree_id: str,
    features: Sequence[str],
    target: np.ndarray,
    weights: np.ndarray | None,
    medians: Mapping[str, object],
    directions: Mapping[str, object],
    min_leaf_count: int,
    max_thresholds_per_feature: int,
    max_row_evaluations: int = MAX_ROW_EVALUATIONS,
) -> dict[str, Any]:
    """Search a deterministic prefix of numeric split candidates."""

    if not isinstance(frame, pd.DataFrame):
        raise StrategyError("interactive-tree split search frame is invalid")
    rows = len(frame)
    mask = np.asarray(node_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != rows:
        raise StrategyError("interactive-tree split search node mask is invalid")
    normalized_features = _features(features)
    node = _text(node_id, "node_id")
    source = _text(source_tree_id, "source_tree_id")
    y = np.asarray(target)
    if y.ndim != 1 or len(y) != rows or not np.isin(y, [0, 1]).all():
        raise StrategyError("interactive-tree split search target is invalid")
    y = y.astype(np.int8, copy=False)
    w = _weights(weights, rows=rows)
    minimum = _positive_int(min_leaf_count, "minimum leaf")
    per_feature = _bounded_int(
        max_thresholds_per_feature,
        "max_thresholds_per_feature",
        maximum=MAX_THRESHOLDS_PER_FEATURE,
    )
    evaluation_limit = _bounded_int(
        max_row_evaluations,
        "max_row_evaluations",
        maximum=MAX_ROW_EVALUATIONS,
    )
    node_count = int(mask.sum())
    if node_count < minimum * 2:
        raise StrategyError(
            "interactive-tree split search minimum leaf cannot be satisfied"
        )
    for feature in normalized_features:
        if feature not in frame.columns:
            raise StrategyError(f"interactive-tree search feature is missing: {feature}")
        if feature not in medians or feature not in directions:
            raise StrategyError(
                "interactive-tree search feature lacks authenticated semantics"
            )
        if directions[feature] not in _DIRECTIONS:
            raise StrategyError("interactive-tree search risk direction is invalid")
        _finite(medians[feature], f"median.{feature}")

    vectors = {
        feature: _numeric(frame[feature], feature=feature)
        for feature in normalized_features
    }
    threshold_map = {
        feature: _thresholds(
            vectors[feature][mask & ~np.isnan(vectors[feature])],
            limit=per_feature,
        )
        for feature in normalized_features
    }
    planned_candidates = sum(len(value) for value in threshold_map.values())
    planned_row_evaluations = node_count * planned_candidates
    if planned_row_evaluations > evaluation_limit:
        raise StrategyError(
            "interactive-tree split search row evaluation budget exceeded"
        )
    parent = _population(mask, y, w)
    parent_gini = _gini(mask, y, w)
    candidates: list[dict[str, Any]] = []
    for feature in normalized_features:
        vector = vectors[feature]
        median = float(medians[feature])
        direction = str(directions[feature])
        for threshold in threshold_map[feature]:
            missing_child = "left" if median <= threshold else "right"
            present = ~np.isnan(vector)
            left_route = present & (vector <= threshold)
            if missing_child == "left":
                left_route |= ~present
            left = mask & left_route
            right = mask & ~left_route
            failures: list[str] = []
            if int(left.sum()) < minimum:
                failures.append("left_below_min_leaf_count")
            if int(right.sum()) < minimum:
                failures.append("right_below_min_leaf_count")
            left_population = _population(left, y, w)
            right_population = _population(right, y, w)
            child_gini = _weighted_child_gini(
                left,
                right,
                y,
                w,
            )
            gain = parent_gini - child_gini
            if gain <= 0:
                failures.append("non_positive_gini_gain")
            direction_evidence = _direction(
                direction,
                left_population=left_population,
                right_population=right_population,
            )
            semantic = {
                "schema_version": "strategy.interactive-tree-split-candidate.v1",
                "source_tree_id": source,
                "node_id": node,
                "feature": feature,
                "threshold": threshold,
                "missing_child": missing_child,
            }
            candidates.append(
                {
                    "candidate_id": (
                        "interactive-tree-split-candidate-"
                        + _sha256(semantic)[:32]
                    ),
                    "rank": 0,
                    "feature": feature,
                    "threshold": threshold,
                    "missing_child": missing_child,
                    "eligible": not failures,
                    "failures": failures,
                    "gain": gain,
                    "parent": parent,
                    "left": left_population,
                    "right": right_population,
                    "direction": direction_evidence,
                }
            )
    candidates.sort(
        key=lambda item: (
            not item["eligible"],
            -float(item["gain"]),
            item["feature"],
            float(item["threshold"]),
            item["candidate_id"],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    body = {
        "schema_version": INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION,
        "producer_version": INTERACTIVE_TREE_SPLIT_SEARCH_PRODUCER_VERSION,
        "lifecycle": {
            "candidate_stage": "development",
            "observation_stage": "backtested",
            "validation_status": "unvalidated",
        },
        "source": {
            "source_tree_id": source,
            "node_id": node,
        },
        "request": {
            "features": normalized_features,
            "min_leaf_count": minimum,
            "max_thresholds_per_feature": per_feature,
        },
        "population": parent,
        "budget": {
            "feature_count": len(normalized_features),
            "candidate_space": planned_candidates,
            "evaluated_candidates": len(candidates),
            "node_row_count": node_count,
            "row_evaluations": planned_row_evaluations,
            "max_row_evaluations": evaluation_limit,
            "truncated": any(
                len(np.unique(vectors[feature][mask & ~np.isnan(vectors[feature])]))
                - 1
                > len(threshold_map[feature])
                for feature in normalized_features
            ),
        },
        "candidates": candidates,
        "claims": {
            "rank_is_navigation_only": True,
            "winner_selected": False,
            "tree_modified": False,
        },
    }
    search_id = f"interactive-tree-split-search-{_sha256(body)[:32]}"
    without_hash = {**body, "search_id": search_id}
    result = {**without_hash, "search_hash": _sha256(without_hash)}
    return validate_interactive_tree_split_search(result)


def validate_interactive_tree_split_search(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact schema, ordering and content-derived identity."""

    if not isinstance(payload, Mapping):
        raise StrategyError("interactive-tree split search must be an object")
    expected = {
        "schema_version",
        "producer_version",
        "lifecycle",
        "source",
        "request",
        "population",
        "budget",
        "candidates",
        "claims",
        "search_id",
        "search_hash",
    }
    if set(payload) != expected:
        raise StrategyError("interactive-tree split search fields changed")
    if payload["schema_version"] != INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION:
        raise StrategyError("interactive-tree split search schema changed")
    if (
        payload["producer_version"]
        != INTERACTIVE_TREE_SPLIT_SEARCH_PRODUCER_VERSION
    ):
        raise StrategyError("interactive-tree split search producer changed")
    detached = json.loads(_canonical_json(payload))
    if detached["lifecycle"] != {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }:
        raise StrategyError("interactive-tree split search lifecycle changed")
    source = detached["source"]
    if not isinstance(source, dict) or set(source) != {
        "source_tree_id",
        "node_id",
    }:
        raise StrategyError("interactive-tree split search source changed")
    _text(source["source_tree_id"], "source_tree_id")
    _text(source["node_id"], "node_id")
    request = detached["request"]
    if not isinstance(request, dict) or set(request) != {
        "features",
        "min_leaf_count",
        "max_thresholds_per_feature",
    }:
        raise StrategyError("interactive-tree split search request changed")
    features = _features(request["features"])
    minimum = _positive_int(request["min_leaf_count"], "minimum leaf")
    per_feature = _bounded_int(
        request["max_thresholds_per_feature"],
        "max_thresholds_per_feature",
        maximum=MAX_THRESHOLDS_PER_FEATURE,
    )
    population = _validate_population(
        detached["population"],
        "population",
    )
    budget = detached["budget"]
    _validate_budget(
        budget,
        features=features,
        population=population,
        per_feature=per_feature,
    )
    candidates = detached["candidates"]
    if not isinstance(candidates, list) or len(candidates) > MAX_SEARCH_CANDIDATES:
        raise StrategyError("interactive-tree split search candidates are invalid")
    normalized_candidates = [
        _validate_candidate(
            candidate,
            source=source,
            features=features,
            parent=population,
            minimum=minimum,
        )
        for candidate in candidates
    ]
    if normalized_candidates != candidates:
        raise StrategyError(
            "interactive-tree split search candidates are not canonical"
        )
    identities = [
        (
            candidate["candidate_id"],
            candidate["feature"],
            candidate["threshold"],
        )
        for candidate in candidates
    ]
    if len(identities) != len(set(identities)):
        raise StrategyError(
            "interactive-tree split search candidates are duplicated"
        )
    if (
        budget["evaluated_candidates"] != len(candidates)
        or budget["candidate_space"] != len(candidates)
        or any(
            sum(
                candidate["feature"] == feature
                for candidate in candidates
            )
            > per_feature
            for feature in features
        )
    ):
        raise StrategyError(
            "interactive-tree split search candidate budget changed"
        )
    expected_order = sorted(
        candidates,
        key=lambda item: (
            not item["eligible"],
            -float(item["gain"]),
            item["feature"],
            float(item["threshold"]),
            item["candidate_id"],
        ),
    )
    if candidates != expected_order or [
        item.get("rank") for item in candidates
    ] != list(range(1, len(candidates) + 1)):
        raise StrategyError("interactive-tree split search rank is not canonical")
    body = {
        key: detached[key]
        for key in detached
        if key not in {"search_id", "search_hash"}
    }
    expected_id = f"interactive-tree-split-search-{_sha256(body)[:32]}"
    if detached["search_id"] != expected_id:
        raise StrategyError("interactive-tree split search identity changed")
    without_hash = {**body, "search_id": expected_id}
    expected_hash = _sha256(without_hash)
    if not hmac.compare_digest(str(detached["search_hash"]), expected_hash):
        raise StrategyError("interactive-tree split search hash changed")
    if detached["claims"] != {
        "rank_is_navigation_only": True,
        "winner_selected": False,
        "tree_modified": False,
    }:
        raise StrategyError("interactive-tree split search claims changed")
    return detached


def _validate_budget(
    value: object,
    *,
    features: Sequence[str],
    population: Mapping[str, Any],
    per_feature: int,
) -> None:
    expected = {
        "feature_count",
        "candidate_space",
        "evaluated_candidates",
        "node_row_count",
        "row_evaluations",
        "max_row_evaluations",
        "truncated",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise StrategyError("interactive-tree split search budget changed")
    feature_count = _positive_int(value["feature_count"], "feature_count")
    candidate_space = _nonnegative_int(
        value["candidate_space"],
        "candidate_space",
    )
    evaluated = _nonnegative_int(
        value["evaluated_candidates"],
        "evaluated_candidates",
    )
    node_count = _positive_int(value["node_row_count"], "node_row_count")
    row_evaluations = _nonnegative_int(
        value["row_evaluations"],
        "row_evaluations",
    )
    maximum = _bounded_int(
        value["max_row_evaluations"],
        "max_row_evaluations",
        maximum=MAX_ROW_EVALUATIONS,
    )
    if (
        feature_count != len(features)
        or feature_count > MAX_SEARCH_FEATURES
        or candidate_space > feature_count * per_feature
        or evaluated != candidate_space
        or node_count != population["count"]
        or row_evaluations != node_count * evaluated
        or row_evaluations > maximum
        or not isinstance(value["truncated"], bool)
    ):
        raise StrategyError(
            "interactive-tree split search budget evidence changed"
        )


def _validate_candidate(
    value: object,
    *,
    source: Mapping[str, Any],
    features: Sequence[str],
    parent: Mapping[str, Any],
    minimum: int,
) -> dict[str, Any]:
    expected = {
        "candidate_id",
        "rank",
        "feature",
        "threshold",
        "missing_child",
        "eligible",
        "failures",
        "gain",
        "parent",
        "left",
        "right",
        "direction",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise StrategyError(
            "interactive-tree split search candidate fields changed"
        )
    candidate = json.loads(_canonical_json(value))
    feature = _text(candidate["feature"], "candidate feature")
    if feature not in features:
        raise StrategyError(
            "interactive-tree split search candidate feature changed"
        )
    threshold = _finite(candidate["threshold"], "candidate threshold")
    if candidate["missing_child"] not in {"left", "right"}:
        raise StrategyError(
            "interactive-tree split search missing route changed"
        )
    left = _validate_population(candidate["left"], "candidate left")
    right = _validate_population(candidate["right"], "candidate right")
    candidate_parent = _validate_population(
        candidate["parent"],
        "candidate parent",
    )
    if (
        candidate_parent != parent
        or left["count"] + right["count"] != parent["count"]
        or left["good"] + right["good"] != parent["good"]
        or left["bad"] + right["bad"] != parent["bad"]
    ):
        raise StrategyError(
            "interactive-tree split search population conservation changed"
        )
    weighted_statuses = {
        parent["weighted"]["status"],
        left["weighted"]["status"],
        right["weighted"]["status"],
    }
    if len(weighted_statuses) != 1:
        raise StrategyError(
            "interactive-tree split search weighted basis changed"
        )
    if parent["weighted"]["status"] == "available" and (
        not math.isclose(
            left["weighted"]["total"] + right["weighted"]["total"],
            parent["weighted"]["total"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            left["weighted"]["bad"] + right["weighted"]["bad"],
            parent["weighted"]["bad"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise StrategyError(
            "interactive-tree split search weighted conservation changed"
        )
    failures: list[str] = []
    if left["count"] < minimum:
        failures.append("left_below_min_leaf_count")
    if right["count"] < minimum:
        failures.append("right_below_min_leaf_count")
    gain = _gini_from_population(parent) - _child_gini_from_populations(
        left,
        right,
    )
    if gain <= 0:
        failures.append("non_positive_gini_gain")
    if (
        candidate["failures"] != failures
        or candidate["eligible"] is not (not failures)
        or not math.isclose(
            _finite(candidate["gain"], "candidate gain"),
            gain,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise StrategyError(
            "interactive-tree split search candidate eligibility changed"
        )
    direction = candidate["direction"]
    if not isinstance(direction, dict) or set(direction) != {
        "expected",
        "basis",
        "status",
        "bad_rate_delta",
    }:
        raise StrategyError(
            "interactive-tree split search direction fields changed"
        )
    expected_direction = _direction(
        _text(direction["expected"], "candidate direction"),
        left_population=left,
        right_population=right,
    )
    if direction != expected_direction:
        raise StrategyError(
            "interactive-tree split search direction evidence changed"
        )
    semantic = {
        "schema_version": "strategy.interactive-tree-split-candidate.v1",
        "source_tree_id": source["source_tree_id"],
        "node_id": source["node_id"],
        "feature": feature,
        "threshold": threshold,
        "missing_child": candidate["missing_child"],
    }
    expected_id = "interactive-tree-split-candidate-" + _sha256(semantic)[:32]
    if candidate["candidate_id"] != expected_id:
        raise StrategyError(
            "interactive-tree split search candidate identity changed"
        )
    _positive_int(candidate["rank"], "candidate rank")
    return candidate


def _validate_population(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "count",
        "good",
        "bad",
        "bad_rate",
        "weighted",
    }:
        raise StrategyError(f"interactive-tree split search {name} changed")
    result = json.loads(_canonical_json(value))
    count = _nonnegative_int(result["count"], f"{name}.count")
    good = _nonnegative_int(result["good"], f"{name}.good")
    bad = _nonnegative_int(result["bad"], f"{name}.bad")
    bad_rate = _finite(result["bad_rate"], f"{name}.bad_rate")
    if (
        good + bad != count
        or not math.isclose(
            bad_rate,
            0.0 if count == 0 else bad / count,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise StrategyError(
            f"interactive-tree split search {name} counts changed"
        )
    weighted = result["weighted"]
    if weighted == {"status": "not_applicable"}:
        return result
    if not isinstance(weighted, dict) or set(weighted) != {
        "status",
        "total",
        "bad",
        "bad_rate",
    }:
        raise StrategyError(
            f"interactive-tree split search {name} weighted evidence changed"
        )
    total = _finite(weighted["total"], f"{name}.weighted.total")
    weighted_bad = _finite(weighted["bad"], f"{name}.weighted.bad")
    weighted_rate = _finite(
        weighted["bad_rate"],
        f"{name}.weighted.bad_rate",
    )
    if (
        weighted["status"] != "available"
        or total < 0
        or weighted_bad < 0
        or weighted_bad > total
        or not math.isclose(
            weighted_rate,
            0.0 if total == 0 else weighted_bad / total,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise StrategyError(
            f"interactive-tree split search {name} weighted values changed"
        )
    return result


def _gini_from_population(value: Mapping[str, Any]) -> float:
    weighted = value["weighted"]
    rate = (
        weighted["bad_rate"]
        if weighted["status"] == "available"
        else value["bad_rate"]
    )
    return float(2.0 * rate * (1.0 - rate))


def _child_gini_from_populations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    weighted = left["weighted"]["status"] == "available"
    left_total = (
        left["weighted"]["total"] if weighted else left["count"]
    )
    right_total = (
        right["weighted"]["total"] if weighted else right["count"]
    )
    total = left_total + right_total
    if total <= 0:
        return 0.0
    return float(
        (
            left_total * _gini_from_population(left)
            + right_total * _gini_from_population(right)
        )
        / total
    )


def canonical_interactive_tree_split_search_json(
    payload: Mapping[str, Any],
) -> str:
    return _canonical_json(validate_interactive_tree_split_search(payload))


def _features(value: Sequence[str]) -> list[str]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(
        value,
        Sequence,
    ):
        raise StrategyError("interactive-tree split search features are invalid")
    result = [_text(item, "feature") for item in value]
    if (
        not result
        or len(result) > MAX_SEARCH_FEATURES
        or len(result) != len(set(result))
        or result != sorted(result)
    ):
        raise StrategyError(
            "interactive-tree split search feature list must be sorted and unique"
        )
    return result


def _thresholds(values: np.ndarray, *, limit: int) -> list[float]:
    unique = np.unique(values)
    if len(unique) < 2:
        return []
    midpoints = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
    if len(midpoints) <= limit:
        chosen = midpoints
    else:
        indices = np.linspace(0, len(midpoints) - 1, num=limit, dtype=int)
        chosen = midpoints[np.unique(indices)]
    return [float(value) for value in chosen if math.isfinite(float(value))]


def _numeric(series: pd.Series, *, feature: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise StrategyError(
            f"interactive-tree split search feature contains infinity: {feature}"
        )
    return numeric


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


def _gini(
    mask: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    if weights is None:
        total = int(mask.sum())
        bad = int(target[mask].sum())
    else:
        total = float(weights[mask].sum())
        bad = float(weights[mask & (target == 1)].sum())
    if total <= 0:
        return 0.0
    p = bad / total
    return float(2.0 * p * (1.0 - p))


def _weighted_child_gini(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    if weights is None:
        left_total = int(left.sum())
        right_total = int(right.sum())
    else:
        left_total = float(weights[left].sum())
        right_total = float(weights[right].sum())
    total = left_total + right_total
    if total <= 0:
        return 0.0
    return (
        left_total * _gini(left, target, weights)
        + right_total * _gini(right, target, weights)
    ) / total


def _direction(
    expected: str,
    *,
    left_population: Mapping[str, Any],
    right_population: Mapping[str, Any],
) -> dict[str, Any]:
    if expected not in _DIRECTIONS:
        raise StrategyError(
            "interactive-tree split search risk direction is invalid"
        )
    weighted = left_population["weighted"]["status"] == "available"
    basis = "weighted" if weighted else "unweighted"
    left_rate = (
        left_population["weighted"]["bad_rate"]
        if weighted
        else left_population["bad_rate"]
    )
    right_rate = (
        right_population["weighted"]["bad_rate"]
        if weighted
        else right_population["bad_rate"]
    )
    status = (
        "inconclusive"
        if expected == "unordered"
        else (
            "consistent"
            if (
                right_rate >= left_rate
                if expected == "increasing"
                else right_rate <= left_rate
            )
            else "violation"
        )
    )
    return {
        "expected": expected,
        "basis": basis,
        "status": status,
        "bad_rate_delta": float(right_rate - left_rate),
    }


def _weights(value: np.ndarray | None, *, rows: int) -> np.ndarray | None:
    if value is None:
        return None
    weights = np.asarray(value, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) != rows
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise StrategyError("interactive-tree split search weights are invalid")
    return weights


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise StrategyError(f"interactive-tree split search {name} is invalid")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise StrategyError(f"interactive-tree split search {name} is invalid")
    return int(value)


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    normalized = _positive_int(value, name)
    if normalized > maximum:
        raise StrategyError(f"interactive-tree split search {name} exceeds budget")
    return normalized


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StrategyError(f"interactive-tree split search {name} is invalid")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise StrategyError(f"interactive-tree split search {name} is invalid")
    return normalized


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyError(f"interactive-tree split search {name} is invalid")
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
            "interactive-tree split search is not finite canonical JSON"
        ) from exc


__all__ = [
    "INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION",
    "canonical_interactive_tree_split_search_json",
    "search_interactive_tree_split_candidates",
    "validate_interactive_tree_split_search",
]
