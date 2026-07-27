from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import (
    WeightedRuleTreeError,
    apply_weighted_rule_tree,
    build_weighted_rule_tree,
    canonical_weighted_rule_tree_json,
    validate_weighted_rule_tree,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )


def _build(frame: pd.DataFrame | None = None, **overrides: object) -> dict:
    kwargs: dict[str, object] = {
        "feature_cols": ["x", "z"],
        "target_col": "bad",
        "max_depth": 2,
        "min_leaf_count": 1,
    }
    kwargs.update(overrides)
    return build_weighted_rule_tree(frame if frame is not None else _frame(), **kwargs)


def _root(result: dict) -> dict:
    root_id = result["tree"]["root_node_id"]
    return next(node for node in result["tree"]["nodes"] if node["node_id"] == root_id)


def _node(result: dict, node_id: str) -> dict:
    return next(node for node in result["tree"]["nodes"] if node["node_id"] == node_id)


def test_tree_is_deterministic_json_native_and_feature_order_independent() -> None:
    first = _build(feature_cols=["z", "x", "z"])
    second = _build(feature_cols=["x", "z"])

    assert first == second
    assert first["training"]["feature_order"] == ["x", "z"]
    assert first["lifecycle"] == {
        "stage": "development",
        "validation_status": "unvalidated",
        "evidence_status": "backtested",
    }
    assert first["search"]["truncated"] is False
    assert first["training"]["cart"]["min_samples_split"] == 2
    assert first["checks"]["all_training_rows_assigned_once"] is True
    json.dumps(first, allow_nan=False, sort_keys=True)


def test_all_one_weights_preserve_tree_and_weighted_metrics_match_unweighted() -> None:
    frame = _frame().assign(weight=1.0)
    plain = _build(frame)
    weighted = _build(frame, sample_weight_col="weight")

    plain_splits = [
        (node.get("feature"), node.get("threshold"), node["path"])
        for node in plain["tree"]["nodes"]
    ]
    weighted_splits = [
        (node.get("feature"), node.get("threshold"), node["path"])
        for node in weighted["tree"]["nodes"]
    ]
    assert weighted_splits == plain_splits
    for node in weighted["tree"]["nodes"]:
        metrics = node["metrics"]
        assert metrics["weighted"]["status"] == "available"
        for key in ("total", "good", "bad", "bad_rate", "share", "bad_capture", "lift"):
            assert metrics["weighted"][key] == pytest.approx(metrics["unweighted"][key])
    assert _root(plain)["metrics"]["weighted"] == {"status": "not_applicable"}


def test_weighted_cart_split_and_metrics_are_hand_checkable() -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "bad": [0, 1, 0, 0, 0, 1],
            "weight": [1.0, 10.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    result = _build(
        frame,
        feature_cols=["x"],
        sample_weight_col="weight",
        max_depth=1,
    )
    root = _root(result)
    assert root["threshold"] == 1.5
    assert root["metrics"]["unweighted"] == {
        "total": 6,
        "good": 4,
        "bad": 2,
        "bad_rate": pytest.approx(2 / 6),
        "share": 1.0,
        "bad_capture": 1.0,
        "lift": 1.0,
    }
    weighted = root["metrics"]["weighted"]
    assert weighted["total"] == 15.0
    assert weighted["good"] == 4.0
    assert weighted["bad"] == 11.0
    assert weighted["bad_rate"] == pytest.approx(11 / 15)


@pytest.mark.parametrize(
    ("target", "code"),
    [
        ([0, 1, 2, 0, 1, 0], "invalid_target"),
        ([0, 1, None, 0, 1, 0], "invalid_target"),
        (["0", "1", "0", "1", "0", "1"], "invalid_target"),
    ],
)
def test_target_is_strict_binary(target: list[object], code: str) -> None:
    frame = _frame().assign(bad=target)
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame)
    assert exc.value.code == code
    assert exc.value.to_detail()["kind"] == "weighted_rule_tree_error"


@pytest.mark.parametrize(
    "weights",
    [
        [1, 1, 0, 1, 1, 1],
        [1, 1, -1, 1, 1, 1],
        [1, 1, None, 1, 1, 1],
        [1, 1, "2", 1, 1, 1],
        [1, 1, np.inf, 1, 1, 1],
    ],
)
def test_weight_is_finite_numeric_strictly_positive(weights: list[object]) -> None:
    frame = _frame().assign(weight=weights)
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame, sample_weight_col="weight")
    assert exc.value.code == "invalid_weight"


def test_weight_column_cannot_be_a_feature() -> None:
    frame = _frame().assign(weight=1.0)
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame, feature_cols=["x", "weight"], sample_weight_col="weight")
    assert exc.value.code == "invalid_feature"


@pytest.mark.parametrize(
    "overrides",
    [
        {"loan_amount_col": "x"},
        {"loan_amount_col": "bad"},
        {"sample_weight_col": "weight", "loan_amount_col": "weight"},
        {"loan_amount_col": "loan", "overdue_amount_col": "loan"},
    ],
)
def test_target_weight_and_amount_roles_must_use_distinct_nonfeature_columns(
    overrides: dict[str, object],
) -> None:
    frame = _frame().assign(weight=1.0, loan=100.0)
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame, **overrides)
    assert exc.value.code in {"invalid_feature", "invalid_input"}


def test_numpy_numeric_scalars_are_accepted_without_string_coercion() -> None:
    frame = pd.DataFrame(
        {
            "x": np.asarray([0, 1, 2, 3], dtype=np.float32),
            "bad": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "weight": np.asarray([1, 2, 1, 2], dtype=np.float64),
        }
    )
    result = _build(
        frame,
        feature_cols=["x"],
        sample_weight_col="weight",
        max_depth=1,
    )
    assert len(apply_weighted_rule_tree(frame, result)) == 4


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, "2", 3, 4, 5],
        [0, 1, np.inf, 3, 4, 5],
        [None, None, None, None, None, None],
        [1, 1, 1, 1, 1, 1],
    ],
)
def test_selected_features_fail_closed_instead_of_being_silently_dropped(
    values: list[object],
) -> None:
    frame = _frame().assign(x=values)
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame, feature_cols=["x"])
    assert exc.value.code == "invalid_feature"


@pytest.mark.parametrize(
    "budgets",
    [
        {"max_rows": 5},
        {"max_features": 1},
        {"max_cells": 10},
        {"max_cutpoint_evaluations": 20},
        {"max_nodes": 1},
    ],
)
def test_resource_budgets_fail_with_structured_dimension(budgets: dict) -> None:
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(budgets=budgets)
    assert exc.value.code == "budget_exceeded"
    assert "dimension" in exc.value.to_detail()


def test_missing_median_route_and_threshold_equality_match_canonical_dsl() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, None, 3.0, 4.0], "bad": [0, 0, 0, 1, 1]})
    result = _build(frame, feature_cols=["x"], max_depth=1)
    root = _root(result)
    left = _node(result, root["left_child_id"])
    right = _node(result, root["right_child_id"])
    assert result["preprocessing"]["medians"]["x"] == 2.0
    assert root["missing_child"] == "left"
    left_condition = next(
        rule["condition"]
        for rule in result["rules"]
        if rule["leaf_id"] == left["node_id"]
    )
    right_condition = next(
        rule["condition"]
        for rule in result["rules"]
        if rule["leaf_id"] == right["node_id"]
    )
    assert left_condition["missing"] == "match"
    assert right_condition["missing"] == "no_match"

    scored = pd.DataFrame(
        {"x": [None, root["threshold"], np.nextafter(root["threshold"], np.inf)]}
    )
    leaf_ids = apply_weighted_rule_tree(scored, result)
    assert leaf_ids.tolist() == [left["node_id"], left["node_id"], right["node_id"]]


def test_float32_threshold_mismatch_is_calibrated_to_training_boundary() -> None:
    frame = pd.DataFrame({"x": [999_999.93, 999_999.96875], "bad": [0, 1]})
    result = _build(frame, feature_cols=["x"], max_depth=1)
    root = _root(result)
    assert root["sklearn_threshold"] == 999_999.96875
    assert root["threshold"] == 999_999.93
    assert root["threshold_adjustment"] == "training_partition_boundary"
    left = _node(result, root["left_child_id"])["node_id"]
    right = _node(result, root["right_child_id"])["node_id"]
    assert apply_weighted_rule_tree(frame, result).tolist() == [left, right]


def test_leaf_rules_form_one_complete_partition_and_ids_are_stable() -> None:
    result = _build()
    leaf_ids = apply_weighted_rule_tree(_frame(), result)
    assert len(leaf_ids) == len(_frame())
    assert set(leaf_ids) == {rule["leaf_id"] for rule in result["rules"]}
    assert all(
        node["node_id"].startswith(("node-", "leaf-"))
        for node in result["tree"]["nodes"]
    )
    assert all(rule["rule_id"].startswith("rule-") for rule in result["rules"])
    assert _build()["tree"]["nodes"] == result["tree"]["nodes"]


def test_direction_diagnostic_is_node_local_and_does_not_modify_tree() -> None:
    frame = pd.DataFrame({"x": list(range(8)), "bad": [0, 0, 0, 0, 0, 0, 1, 0]})
    result = _build(
        frame,
        feature_cols=["x"],
        max_depth=2,
        directions={"x": "increasing"},
    )
    diagnostics = [
        node["direction_diagnostic"]
        for node in result["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    assert {item["status"] for item in diagnostics} == {"consistent", "violation"}
    assert all(item["basis"] == "unweighted" for item in diagnostics)
    assert all("left" in item and "right" in item for item in diagnostics)


@pytest.mark.parametrize(
    ("direction", "status"),
    [
        ("increasing", "violation"),
        ("decreasing", "consistent"),
        ("unordered", "inconclusive"),
    ],
)
def test_direction_uses_weighted_child_bad_rates_as_primary(
    direction: str,
    status: str,
) -> None:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "bad": [0, 1, 0, 1],
            "weight": [1.0, 10.0, 10.0, 1.0],
        }
    )
    result = _build(
        frame,
        feature_cols=["x"],
        sample_weight_col="weight",
        directions={"x": direction},
        max_depth=1,
    )
    diagnostic = _root(result)["direction_diagnostic"]
    assert diagnostic["basis"] == "weighted"
    assert diagnostic["left"]["bad_rate"] == diagnostic["right"]["bad_rate"] == 0.5
    assert diagnostic["left"]["weighted"]["bad_rate"] == pytest.approx(10 / 11)
    assert diagnostic["right"]["weighted"]["bad_rate"] == pytest.approx(1 / 11)
    assert diagnostic["primary_bad_rate_delta"] == pytest.approx(-9 / 11)
    assert diagnostic["status"] == status


def test_conservation_and_optional_amount_metrics() -> None:
    frame = _frame().assign(
        weight=[1, 2, 1, 2, 1, 2],
        loan=[100, 100, 200, 200, 300, 300],
        overdue=[0, 10, 20, 0, 60, 90],
    )
    result = _build(
        frame,
        sample_weight_col="weight",
        loan_amount_col="loan",
        overdue_amount_col="overdue",
    )
    root = _root(result)["metrics"]
    assert root["unweighted"]["loan_amount_total"] == 1200.0
    assert root["unweighted"]["overdue_amount_total"] == 180.0
    assert root["unweighted"]["overdue_rate"] == pytest.approx(0.15)
    assert root["weighted"]["loan_amount_total"] == 1800.0
    assert root["weighted"]["overdue_amount_total"] == 280.0
    assert root["weighted"]["overdue_rate"] == pytest.approx(280 / 1800)
    assert result["checks"]["conservation"] == "passed"


def test_amount_nulls_are_partial_coverage_not_silent_zero_imputation() -> None:
    frame = _frame().assign(
        loan=[100, None, 200, 200, 300, 300],
        overdue=[None, 10, 20, 0, 60, 90],
    )
    result = _build(
        frame,
        loan_amount_col="loan",
        overdue_amount_col="overdue",
    )
    root = _root(result)["metrics"]["unweighted"]
    assert root["loan_amount_total"] == 1100.0
    assert root["loan_amount_coverage_count"] == 5
    assert root["loan_amount_coverage_rate"] == pytest.approx(5 / 6)
    assert root["overdue_amount_total"] == 180.0
    assert root["overdue_amount_coverage_count"] == 5
    assert root["amount_pair_coverage_count"] == 4
    assert root["paired_loan_amount_total"] == 1000.0
    assert root["paired_overdue_amount_total"] == 170.0
    assert root["overdue_rate"] == pytest.approx(0.17)


def test_all_missing_optional_amounts_report_zero_coverage_and_no_rate() -> None:
    frame = _frame().assign(loan=None, overdue=None)
    result = _build(
        frame,
        loan_amount_col="loan",
        overdue_amount_col="overdue",
    )
    root = _root(result)["metrics"]["unweighted"]
    assert root["loan_amount_total"] == 0.0
    assert root["loan_amount_coverage_count"] == 0
    assert root["overdue_amount_total"] == 0.0
    assert root["overdue_amount_coverage_count"] == 0
    assert root["amount_pair_coverage_count"] == 0
    assert root["overdue_rate"] is None


def _rehash(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload.pop("result_hash", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    payload["result_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_strict_validator_returns_detached_canonical_payload() -> None:
    result = _build()
    validated = validate_weighted_rule_tree(result)
    assert validated == result
    assert validated is not result
    assert validated["tree"] is not result["tree"]
    assert canonical_weighted_rule_tree_json(result) == json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@pytest.mark.parametrize("tamper", ["unknown", "training_leaf_ids"])
def test_strict_validator_rejects_unknown_fields_even_with_rehashed_payload(
    tamper: str,
) -> None:
    result = _build()
    result[tamper] = []
    with pytest.raises(WeightedRuleTreeError) as exc:
        validate_weighted_rule_tree(_rehash(result))
    assert exc.value.code == "invalid_result"


def test_strict_validator_rejects_rehashed_threshold_tamper() -> None:
    result = _build()
    _root(result)["threshold"] += 0.25
    with pytest.raises(WeightedRuleTreeError):
        validate_weighted_rule_tree(_rehash(result))


def test_strict_validator_rejects_rehashed_child_reference_tamper() -> None:
    result = _build()
    root = _root(result)
    root["left_child_id"], root["right_child_id"] = (
        root["right_child_id"],
        root["left_child_id"],
    )
    with pytest.raises(WeightedRuleTreeError):
        validate_weighted_rule_tree(_rehash(result))


def test_strict_validator_rejects_rehashed_rule_condition_tamper() -> None:
    result = _build()
    condition = result["rules"][0]["condition"]
    clause = condition if condition["op"] == "compare" else condition["args"][0]
    clause["value"] += 0.25
    with pytest.raises(WeightedRuleTreeError):
        validate_weighted_rule_tree(_rehash(result))


def test_strict_validator_rejects_rehashed_metric_conservation_tamper() -> None:
    result = _build()
    leaf = next(node for node in result["tree"]["nodes"] if node["kind"] == "leaf")
    leaf["metrics"]["unweighted"]["bad"] += 1
    with pytest.raises(WeightedRuleTreeError):
        validate_weighted_rule_tree(_rehash(result))


def test_strict_validator_rejects_rehashed_role_overlap() -> None:
    result = _build()
    result["training"]["loan_amount_col"] = "x"
    with pytest.raises(WeightedRuleTreeError) as exc:
        validate_weighted_rule_tree(_rehash(result))
    assert exc.value.code == "invalid_result"


def test_root_only_tree_is_typed_infeasible_not_a_fake_rule() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "bad": [0, 1, 1]})
    with pytest.raises(WeightedRuleTreeError) as exc:
        _build(frame, feature_cols=["x"], min_leaf_count=2)
    assert exc.value.code == "infeasible_tree"
