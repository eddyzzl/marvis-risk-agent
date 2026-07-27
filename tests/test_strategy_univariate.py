from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from marvis.feature.errors import FeatureError
from marvis.feature.univariate import (
    MANUAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    analyze_univariate,
)
from marvis.packs.strategy.evaluator import (
    evaluate_expression,
    evaluate_expression_frame,
)


def _method(result: dict, feature: str, method: str) -> dict:
    feature_result = next(
        item for item in result["features"] if item["feature"] == feature
    )
    return next(item for item in feature_result["methods"] if item["method"] == method)


def test_manual_breakpoints_are_exact_v2_evidence_without_changing_v1() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )

    manual = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["manual"],
        manual_breakpoints={"x": [2, 4.0]},
        bin_count=3,
        min_bin_pct=0,
    )

    assert manual["schema_version"] == MANUAL_SCHEMA_VERSION
    assert manual["parameters"]["manual_breakpoints"] == {"x": [2.0, 4.0]}
    method = _method(manual, "x", "manual")
    assert method["requested_method"] == "manual"
    assert method["actual_method"] == "manual"
    assert method["manual_breakpoints"] == [2.0, 4.0]
    regular = [row for row in method["bins"] if row["kind"] == "numeric_interval"]
    assert [(row["lower"], row["upper"]) for row in regular] == [
        (None, 2.0),
        (2.0, 4.0),
        (4.0, None),
    ]
    assert [row["count"] for row in regular] == [2, 2, 2]
    for value, expected_index in [(0, 0), (2, 1), (4, 2), (5, 2)]:
        matches = [
            row["index"]
            for row in regular
            if evaluate_expression({"x": value}, row["condition"])
        ]
        assert matches == [expected_index]

    legacy = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        min_bin_pct=0,
    )
    assert legacy["schema_version"] == SCHEMA_VERSION
    assert "manual_breakpoints" not in legacy["parameters"]
    assert "manual_breakpoints" not in _method(legacy, "x", "equal_width")


@pytest.mark.parametrize(
    ("manual_breakpoints", "message"),
    [
        (None, "manual_breakpoints must provide at least one"),
        ({}, "manual_breakpoints must provide at least one"),
        ({"x": []}, "at least one"),
        ({"x": [2.0, 1.0]}, "strictly increasing"),
        ({"x": [1.0, 1.0]}, "strictly increasing"),
        ({"x": [float("nan")]}, "finite numbers"),
        ({"x": [float("inf")]}, "finite numbers"),
        ({"x": [True]}, "finite numbers"),
        ({"x": ["1"]}, "finite numbers"),
        ({"x": [10**1000]}, "exact numeric precision"),
        ({"x": list(range(20))}, "configured bin budget"),
        ({"other": [1.0]}, "unknown or non-numeric"),
    ],
)
def test_manual_breakpoints_fail_closed_on_invalid_contract(
    manual_breakpoints: object,
    message: str,
) -> None:
    frame = pd.DataFrame({"x": [0, 1, 2, 3], "bad": [0, 0, 1, 1]})

    with pytest.raises(FeatureError, match=message):
        analyze_univariate(
            frame,
            features=["x"],
            target="bad",
            methods=["manual"],
            manual_breakpoints=manual_breakpoints,
            bin_count=3,
        )


def test_manual_breakpoints_are_scoped_to_explicit_features() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [10, 11, 12, 13],
            "segment": ["a", "a", "b", "b"],
            "bad": [0, 0, 1, 1],
        }
    )

    with pytest.raises(FeatureError, match="only allowed when manual"):
        analyze_univariate(
            frame,
            features=["x"],
            target="bad",
            methods=["equal_width"],
            manual_breakpoints={"x": [1.5]},
            bin_count=3,
        )
    with pytest.raises(FeatureError, match="no applicable requested method"):
        analyze_univariate(
            frame,
            features=["x", "y"],
            target="bad",
            methods=["manual"],
            manual_breakpoints={"x": [1.5]},
            bin_count=3,
        )
    mixed = analyze_univariate(
        frame,
        features=["x", "y"],
        target="bad",
        methods=["tree", "manual"],
        manual_breakpoints={"x": [1.5]},
        bin_count=3,
        min_bin_pct=0,
    )
    assert [row["method"] for row in mixed["features"][0]["methods"]] == [
        "tree",
        "manual",
    ]
    assert [row["method"] for row in mixed["features"][1]["methods"]] == [
        "tree"
    ]
    with pytest.raises(FeatureError, match="unknown or non-numeric"):
        analyze_univariate(
            frame,
            features=["segment"],
            target="bad",
            methods=["manual"],
            feature_types={"segment": "categorical"},
            manual_breakpoints={"segment": [1.5]},
            bin_count=3,
        )


def test_numeric_bins_have_exact_left_closed_right_open_dsl_semantics() -> None:
    frame = pd.DataFrame(
        {
            "x": [-999, 0, 1, 2, 3, 4, 5, 6, 8, np.nan],
            "bad": [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
        }
    )

    result = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=4,
        sentinel_values={"x": [-999]},
        min_bin_pct=0,
    )

    method = _method(result, "x", "equal_width")
    assert method["requested_method"] == "equal_width"
    assert method["actual_method"] == "equal_width"
    regular = [item for item in method["bins"] if item["kind"] == "numeric_interval"]
    assert [(item["lower"], item["upper"]) for item in regular] == [
        (None, 2.0),
        (2.0, 4.0),
        (4.0, 6.0),
        (6.0, None),
    ]
    # Exact edges belong to the bin on their right (>= lower, < upper).
    for value, expected_index in [(0, 0), (2, 1), (4, 2), (6, 3), (8, 3)]:
        matches = [
            item["index"]
            for item in regular
            if evaluate_expression({"x": value}, item["condition"])
        ]
        assert matches == [expected_index]
    assert not any(
        evaluate_expression({"x": -999}, item["condition"]) for item in regular
    )
    assert (
        next(item for item in method["bins"] if item["kind"] == "sentinel")["count"]
        == 1
    )
    assert (
        next(item for item in method["bins"] if item["kind"] == "missing")["count"] == 1
    )


def test_categorical_bins_preserve_scalar_types_and_null_as_explicit_bin() -> None:
    frame = pd.DataFrame(
        {
            "segment": pd.Series(["2", 2, "A", None, "A", 2], dtype=object),
            "bad": [0, 1, 0, 1, 1, 0],
        }
    )

    result = analyze_univariate(
        frame,
        features=["segment"],
        target="bad",
        methods=["categorical"],
        bin_count=3,
        min_bin_pct=0,
    )

    bins = _method(result, "segment", "categorical")["bins"]
    values = [item["value"] for item in bins if item["kind"] == "category"]
    assert values == [2, "2", "A"]
    assert next(item for item in bins if item["kind"] == "missing")["condition"] == {
        "op": "is_null",
        "field": "segment",
    }
    int_bin = next(item for item in bins if item.get("value") == 2)
    str_bin = next(item for item in bins if item.get("value") == "2")
    assert evaluate_expression({"segment": 2}, int_bin["condition"])
    assert evaluate_expression({"segment": "2"}, str_bin["condition"])
    assert not evaluate_expression({"segment": "2"}, int_bin["condition"])
    assert not evaluate_expression({"segment": 2}, str_bin["condition"])
    assert int_bin["count"] == 2
    assert str_bin["count"] == 1
    match_counts = pd.Series(0, index=frame.index, dtype="int64")
    for item in bins:
        mask = evaluate_expression_frame(frame, item["condition"])
        assert int(mask.sum()) == item["count"]
        match_counts += mask.astype("int64")
    assert match_counts.tolist() == [1] * len(frame)


def test_amount_metrics_report_coverage_and_never_substitute_zero_for_missing() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 0, 0, 1, 1, 1],
            "loan": [0.0, 0.0, np.nan, 100.0, 200.0, 300.0],
            "overdue": [0.0, 5.0, np.nan, 10.0, 20.0, 30.0],
        }
    )

    result = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        loan_amount="loan",
        overdue_amount="overdue",
        min_bin_pct=0,
    )

    amount = _method(result, "x", "equal_width")["metrics"]["amount_metrics"]
    assert amount["loan_amount"] == {
        "status": "available",
        "sum": 600.0,
        "covered_count": 5,
        "coverage_rate": pytest.approx(5 / 6),
    }
    assert amount["overdue_rate"]["status"] == "available"
    assert amount["overdue_rate"]["value"] == pytest.approx(65 / 600)
    first_bin = _method(result, "x", "equal_width")["bins"][0]
    assert first_bin["amount_metrics"]["overdue_rate"] == {
        "status": "not_applicable",
        "reason": "zero_loan_amount",
    }

    without_amounts = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
    )
    missing_amount = _method(without_amounts, "x", "equal_width")["metrics"][
        "amount_metrics"
    ]
    assert missing_amount["loan_amount"]["status"] == "unavailable"
    assert "sum" not in missing_amount["loan_amount"]
    assert missing_amount["overdue_rate"]["status"] == "unavailable"


def test_empty_numeric_bin_has_observed_zero_amount_and_not_applicable_rate() -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 0, 0, 10],
            "bad": [0, 0, 1, 1],
            "loan": [10.0, 20.0, 30.0, 40.0],
            "overdue": [0.0, 1.0, 2.0, 3.0],
        }
    )

    result = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        loan_amount="loan",
        overdue_amount="overdue",
        min_bin_pct=0,
    )

    empty = next(
        item
        for item in _method(result, "x", "equal_width")["bins"]
        if item["count"] == 0
    )
    assert empty["amount_metrics"]["loan_amount"] == {
        "status": "available",
        "sum": 0.0,
        "covered_count": 0,
        "coverage_rate": 1.0,
    }
    assert empty["amount_metrics"]["overdue_rate"] == {
        "status": "not_applicable",
        "reason": "empty_bin",
    }


def test_unavailable_methods_are_typed_and_never_fall_back() -> None:
    frame = pd.DataFrame(
        {
            "constant": [7, 7, 7, 7],
            "category": ["a", "b", "a", "b"],
            "bad": [0, 0, 1, 1],
        }
    )

    result = analyze_univariate(
        frame,
        features=["constant", "category"],
        target="bad",
        methods=["tree", "categorical"],
        bin_count=3,
    )

    constant_tree = _method(result, "constant", "tree")
    assert constant_tree == {
        "method": "tree",
        "requested_method": "tree",
        "actual_method": None,
        "status": "unavailable",
        "evidence": {
            "kind": "insufficient_supervised_variation",
            "unique_values": 1,
            "target_classes": 2,
        },
        "metrics": None,
        "bins": [],
    }
    category_tree = _method(result, "category", "tree")
    assert category_tree["evidence"] == {
        "kind": "incompatible_feature_type",
        "expected": "numeric",
        "actual": "categorical",
    }
    assert category_tree["actual_method"] is None
    constant_categorical = _method(result, "constant", "categorical")
    assert constant_categorical["evidence"]["kind"] == "incompatible_feature_type"


@pytest.mark.parametrize("feature_type", ["numeric", "categorical"])
def test_unsafe_large_integer_categories_or_edges_fail_typed(feature_type: str) -> None:
    frame = pd.DataFrame(
        {
            "x": pd.Series(
                [2**53, 2**53 + 1, 2**53, 2**53 + 1],
                dtype=object,
            ),
            "bad": [0, 1, 0, 1],
        }
    )
    method = "equal_width" if feature_type == "numeric" else "categorical"

    result = analyze_univariate(
        frame,
        features=["x"],
        target="bad",
        methods=[method],
        feature_types={"x": feature_type},
        bin_count=3,
    )

    unavailable = _method(result, "x", method)
    assert unavailable["status"] == "unavailable"
    assert unavailable["evidence"]["kind"] == "unsafe_numeric_precision"


def test_cumulative_ks_uses_the_same_risk_order_as_method_ks() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "bad": [1] * 8 + [0] * 2 + [1] + [0] * 9 + [1] * 7 + [0] * 3,
        }
    )

    result = analyze_univariate(
        frame,
        features=["segment"],
        target="bad",
        methods=["categorical"],
        bin_count=3,
        min_bin_pct=0,
    )

    method = _method(result, "segment", "categorical")
    assert max(item["cumulative_ks"] for item in method["bins"]) == pytest.approx(
        method["metrics"]["ks"]
    )


def test_degenerate_numeric_condition_is_valid_and_minimum_bin_failure_is_red_flagged() -> (
    None
):
    constant = pd.DataFrame({"x": [7, 7, 7, 7], "bad": [0, 0, 1, 1]})
    constant_result = analyze_univariate(
        constant,
        features=["x"],
        target="bad",
        methods=["equal_frequency"],
        bin_count=3,
    )
    only_bin = _method(constant_result, "x", "equal_frequency")["bins"][0]
    assert only_bin["condition"] == {"op": "is_not_null", "field": "x"}
    assert evaluate_expression({"x": 7}, only_bin["condition"])

    skewed = pd.DataFrame({"x": [0] * 9 + [10], "bad": [0] * 5 + [1] * 5})
    skewed_result = analyze_univariate(
        skewed,
        features=["x"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        min_bin_pct=0.2,
    )
    evidence = _method(skewed_result, "x", "equal_width")["evidence"]
    assert any(
        item["kind"] == "min_bin_pct_not_achieved" and item["severity"] == "red_flag"
        for item in evidence
    )


def test_result_is_finite_deterministic_ranked_and_budgeted() -> None:
    frame = pd.DataFrame(
        {
            "z": list(range(20)),
            "a": list(range(20)),
            "bad": [0] * 10 + [1] * 10,
        }
    )
    kwargs = dict(
        features=["z", "a"],
        target="bad",
        methods=["tree", "equal_frequency"],
        bin_count=4,
        min_bin_pct=0.1,
        seed=19,
    )

    first = analyze_univariate(frame, **kwargs)
    second = analyze_univariate(frame, **kwargs)

    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )
    assert first["resource_budget"] == {
        "max_rows": 1_000_000,
        "max_features": 50,
        "max_bins": 20,
        "max_categories": 100,
        "rows_used": 20,
        "features_used": 2,
        "method_runs": 4,
        "truncated": False,
    }
    # Exact metric ties resolve by feature name, then fixed method order.
    assert [item["feature"] for item in first["rankings"][:2]] == ["a", "z"]
    assert first["parameters"]["min_bin_pct"] == 0.1
    assert all(
        item["requested_method"] == item["actual_method"]
        for feature in first["features"]
        for item in feature["methods"]
        if item["status"] == "available"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"features": []}, "features must contain"),
        ({"features": [f"x{index}" for index in range(51)]}, "features must contain"),
        ({"min_bin_pct": 0.51}, "min_bin_pct"),
        ({"bin_count": 21}, "bin_count"),
    ],
)
def test_resource_constraints_fail_closed(kwargs: dict, message: str) -> None:
    frame = pd.DataFrame(
        {
            **{f"x{index}": [0, 1] for index in range(51)},
            "bad": [0, 1],
        }
    )
    params = {"features": ["x0"], "target": "bad", "bin_count": 3, **kwargs}
    with pytest.raises(FeatureError, match=message):
        analyze_univariate(frame, **params)
