from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.descriptive import (
    DATA_ANALYSIS_SCHEMA_VERSION,
    DescriptiveBudgetError,
    DescriptiveConfig,
    DescriptiveConfigError,
    DescriptiveInputError,
    DescriptiveSanitizerError,
    analyze_parquet,
)


def _write_parquet(tmp_path, frame: pd.DataFrame, name: str = "normalized.parquet"):
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def _analyze(
    tmp_path,
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    config: DescriptiveConfig | None = None,
    value_sanitizers=None,
):
    path = _write_parquet(tmp_path, frame)
    return analyze_parquet(
        path,
        temp_directory=tmp_path / "duckdb-tmp",
        target_column=target_column,
        config=config,
        value_sanitizers=value_sanitizers,
    )


def _field(report: dict, name: str) -> dict:
    return next(item for item in report["fields"] if item["name"] == name)


def _frequency_counts(field: dict) -> list[tuple[dict, int]]:
    return [
        (item["value"], item["count"])
        for item in field["frequency"]["items"]
    ]


def test_empty_parquet_has_exact_overview_and_explicit_empty_reasons(tmp_path):
    frame = pd.DataFrame(
        {
            "amount": pd.Series([], dtype="float64"),
            "segment": pd.Series([], dtype="string"),
        }
    )

    report = _analyze(tmp_path, frame)

    assert report["schema_version"] == DATA_ANALYSIS_SCHEMA_VERSION == "data-analysis.v1"
    assert report["dataset"] == {
        "source_format": "parquet",
        "row_count": 0,
        "source_column_count": 2,
        "column_count": 2,
        "numeric_column_count": 1,
        "columns_requested": False,
        "target_auto_included": False,
    }
    assert report["config"] == DescriptiveConfig().to_dict()
    amount = _field(report, "amount")
    assert (amount["row_count"], amount["null_count"], amount["distinct_count"]) == (0, 0, 0)
    assert amount["null_rate"] == 0.0
    assert amount["numeric"]["finite_count"] == 0
    assert amount["numeric"]["nonfinite_count"] == 0
    assert amount["numeric"]["min"] is None
    assert amount["numeric"]["mean"] is None
    assert amount["histogram"] == {
        "basis": "finite_only",
        "finite_count": 0,
        "reason": "empty",
        "bins": [],
    }
    assert amount["frequency"]["items"] == []
    assert amount["frequency"]["complete"] is True
    assert amount["frequency"]["other_count"] == 0
    assert report["target_distribution"] == {
        "status": "not_configured",
        "column": None,
    }
    assert report["correlations"]["values"] == [[None]]
    assert report["correlations"]["pair_counts"] == [[0]]
    assert report["correlations"]["reasons"] == [["insufficient_pairs"]]


def test_null_and_infinity_are_explicit_and_numeric_stats_use_only_finite_rows(tmp_path):
    report = _analyze(
        tmp_path,
        pd.DataFrame({"x": [None, -np.inf, 1.0, 3.0, np.inf]}),
        config=DescriptiveConfig(low_cardinality_threshold=10),
    )

    field = _field(report, "x")
    assert (field["row_count"], field["null_count"], field["distinct_count"]) == (5, 1, 4)
    assert field["null_rate"] == pytest.approx(0.2)
    assert field["numeric"] == {
        "basis": "finite_only",
        "finite_count": 2,
        "nonfinite_count": 2,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "stddev_pop": 1.0,
        "p25": 1.5,
        "p50": 2.0,
        "p75": 2.5,
    }
    counts = _frequency_counts(field)
    assert counts == [
        ({"type": "null", "value": None}, 1),
        ({"type": "float", "value": None, "nonfinite": "negative_infinity"}, 1),
        ({"type": "float", "value": 1.0}, 1),
        ({"type": "float", "value": 3.0}, 1),
        ({"type": "float", "value": None, "nonfinite": "positive_infinity"}, 1),
    ]
    null_item = field["frequency"]["items"][0]
    assert null_item["rate_all"] == pytest.approx(0.2)
    assert null_item["rate_non_missing"] is None
    for item in field["frequency"]["items"][1:]:
        assert item["rate_non_missing"] == pytest.approx(0.25)
    assert field["frequency"]["other_count"] == 0
    assert field["frequency"]["complete"] is True


def test_mixed_scalar_types_are_tagged_without_type_coercion(tmp_path):
    frame = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None], dtype="boolean"),
            "count": pd.Series([1, 2, None], dtype="Int64"),
            "ratio": [1.5, 2.5, None],
            "label": pd.Series(["1", "2", None], dtype="string"),
            "as_of": [date(2026, 1, 1), date(2026, 1, 2), None],
            "event_at": pd.to_datetime(["2026-01-01 01:02:03", "2026-01-02 04:05:06", None]),
        }
    )

    report = _analyze(
        tmp_path,
        frame,
        config=DescriptiveConfig(low_cardinality_threshold=20),
    )

    expected_types = {
        "flag": "bool",
        "count": "int",
        "ratio": "float",
        "label": "string",
        "as_of": "date",
        "event_at": "datetime",
    }
    for column, expected_type in expected_types.items():
        values = [
            item["value"]
            for item in _field(report, column)["frequency"]["items"]
            if item["value"]["type"] != "null"
        ]
        assert values
        assert {value["type"] for value in values} == {expected_type}

    assert {item["value"]["value"] for item in _field(report, "count")["frequency"]["items"]} == {
        None,
        1,
        2,
    }
    assert {item["value"]["value"] for item in _field(report, "label")["frequency"]["items"]} == {
        None,
        "1",
        "2",
    }
    assert json.dumps(report, allow_nan=False, sort_keys=True)


def test_target_distribution_reports_unavailable_and_present_states(tmp_path):
    frame = pd.DataFrame({"bad": [0, 1, 0, None], "x": [1, 2, 3, 4]})

    unavailable = _analyze(tmp_path, frame, target_column="missing")
    assert unavailable["target_distribution"] == {
        "status": "unavailable",
        "column": "missing",
        "reason": "column_not_found",
    }

    present = _analyze(tmp_path, frame, target_column="bad")
    target = present["target_distribution"]
    assert target["status"] == "available"
    assert target["column"] == "bad"
    assert target["frequency"] == _field(present, "bad")["frequency"]
    assert sum(item["count"] for item in target["frequency"]["items"]) == 4


def test_columns_subset_is_deduplicated_in_order_and_auto_includes_target(tmp_path):
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
            "bad": [0, 1, 0],
            "unused": [10, 20, 30],
        }
    )
    path = _write_parquet(tmp_path, frame)

    report = analyze_parquet(
        path,
        temp_directory=tmp_path / "duckdb-tmp",
        target_column="bad",
        columns=["b", "b", "a"],
        config=DescriptiveConfig(max_columns=3, max_numeric_columns=3, max_pairs=3),
    )

    assert [field["name"] for field in report["fields"]] == ["b", "a", "bad"]
    assert [field["selection_role"] for field in report["fields"]] == [
        "requested",
        "requested",
        "target_auto_included",
    ]
    assert report["dataset"] == {
        "source_format": "parquet",
        "row_count": 3,
        "source_column_count": 4,
        "column_count": 3,
        "numeric_column_count": 3,
        "columns_requested": True,
        "target_auto_included": True,
    }
    assert report["target_distribution"]["auto_included"] is True
    assert report["correlations"]["columns"] == ["b", "a", "bad"]


def test_columns_subset_budget_is_checked_after_selection_and_unknown_columns_fail(tmp_path):
    path = _write_parquet(
        tmp_path,
        pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}),
    )

    selected = analyze_parquet(
        path,
        temp_directory=tmp_path / "temp-selected",
        columns=["b"],
        config=DescriptiveConfig(max_columns=1, max_numeric_columns=1, max_pairs=1),
    )
    assert [field["name"] for field in selected["fields"]] == ["b"]

    with pytest.raises(DescriptiveInputError, match="unknown descriptive column"):
        analyze_parquet(
            path,
            temp_directory=tmp_path / "temp-unknown",
            columns=["missing"],
        )
    with pytest.raises(DescriptiveInputError, match="not a string"):
        analyze_parquet(
            path,
            temp_directory=tmp_path / "temp-string",
            columns="abc",
        )


def test_frequency_ties_null_and_top_k_other_are_stable_and_conservative(tmp_path):
    frame = pd.DataFrame({"segment": ["b", "a", "c", "b", "a", None, "d", "e"]})
    config = DescriptiveConfig(frequency_top_k=2, low_cardinality_threshold=2)

    first = _analyze(tmp_path, frame, config=config)
    second = _analyze(tmp_path, frame, config=config)
    frequency = _field(first, "segment")["frequency"]

    assert first == second
    assert frequency["mode"] == "top_k"
    assert _frequency_counts(_field(first, "segment")) == [
        ({"type": "string", "value": "a"}, 2),
        ({"type": "string", "value": "b"}, 2),
        ({"type": "null", "value": None}, 1),
    ]
    assert frequency["other_count"] == 3
    assert frequency["other_rate_all"] == pytest.approx(3 / 8)
    assert frequency["other_rate_non_missing"] == pytest.approx(3 / 7)
    assert frequency["complete"] is False
    assert sum(item["count"] for item in frequency["items"]) + frequency["other_count"] == 8


def test_low_cardinality_frequency_is_complete_and_count_sorted(tmp_path):
    report = _analyze(
        tmp_path,
        pd.DataFrame({"segment": ["b", "a", "c", "b", "a", None]}),
        config=DescriptiveConfig(low_cardinality_threshold=10),
    )
    frequency = _field(report, "segment")["frequency"]

    assert frequency["mode"] == "exact"
    assert _frequency_counts(_field(report, "segment")) == [
        ({"type": "string", "value": "a"}, 2),
        ({"type": "string", "value": "b"}, 2),
        ({"type": "null", "value": None}, 1),
        ({"type": "string", "value": "c"}, 1),
    ]
    assert frequency["complete"] is True
    assert frequency["other_count"] == 0


def test_histogram_equal_width_boundaries_and_constant_reason(tmp_path):
    report = _analyze(
        tmp_path,
        pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "constant": [5, 5, 5, 5, 5]}),
        config=DescriptiveConfig(histogram_bins=2),
    )

    histogram = _field(report, "x")["histogram"]
    assert histogram["basis"] == "finite_only"
    assert histogram["reason"] is None
    assert histogram["finite_count"] == 5
    assert histogram["bins"] == [
        {
            "index": 0,
            "lower": 0.0,
            "upper": 2.0,
            "lower_inclusive": True,
            "upper_inclusive": False,
            "count": 2,
            "rate_finite": 0.4,
        },
        {
            "index": 1,
            "lower": 2.0,
            "upper": 4.0,
            "lower_inclusive": True,
            "upper_inclusive": True,
            "count": 3,
            "rate_finite": 0.6,
        },
    ]
    assert sum(item["count"] for item in histogram["bins"]) == 5
    assert _field(report, "constant")["histogram"] == {
        "basis": "finite_only",
        "finite_count": 5,
        "reason": "constant",
        "bins": [],
    }


def test_unsafe_ubigint_precision_is_explicitly_unavailable_not_collapsed(tmp_path):
    report = _analyze(
        tmp_path,
        pd.DataFrame(
            {
                "large": pd.Series([2**53, 2**53 + 1], dtype="uint64"),
                "small": pd.Series([1, 2], dtype="int64"),
            }
        ),
        config=DescriptiveConfig(low_cardinality_threshold=10),
    )

    large = _field(report, "large")
    assert large["duckdb_type"] == "UBIGINT"
    assert large["distinct_count"] == 2
    assert _frequency_counts(large) == [
        ({"type": "bigint", "value": str(2**53)}, 1),
        ({"type": "bigint", "value": str(2**53 + 1)}, 1),
    ]
    assert json.dumps(report, allow_nan=False, sort_keys=True)
    assert large["numeric"] == {
        "basis": "finite_only",
        "status": "unavailable",
        "reason": "unsafe_numeric_precision",
        "finite_count": 2,
        "nonfinite_count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "stddev_pop": None,
        "p25": None,
        "p50": None,
        "p75": None,
    }
    assert large["histogram"] == {
        "basis": "finite_only",
        "finite_count": 2,
        "reason": "unsafe_numeric_precision",
        "bins": [],
    }

    correlations = report["correlations"]
    indexes = {name: index for index, name in enumerate(correlations["columns"])}
    large_index = indexes["large"]
    small_index = indexes["small"]
    assert correlations["values"][large_index][large_index] is None
    assert (
        correlations["reasons"][large_index][large_index]
        == "unsafe_numeric_precision"
    )
    assert correlations["values"][large_index][small_index] is None
    assert (
        correlations["reasons"][large_index][small_index]
        == "unsafe_numeric_precision_left"
    )
    assert (
        correlations["reasons"][small_index][large_index]
        == "unsafe_numeric_precision_right"
    )


def test_high_precision_decimal_is_unavailable_in_double_based_metrics(tmp_path):
    report = _analyze(
        tmp_path,
        pd.DataFrame(
            {
                "precise": [
                    Decimal("0.123456789012345678"),
                    Decimal("0.123456789012345679"),
                ]
            }
        ),
        config=DescriptiveConfig(low_cardinality_threshold=10),
    )

    precise = _field(report, "precise")
    assert precise["duckdb_type"].startswith("DECIMAL(")
    assert precise["distinct_count"] == 2
    assert precise["numeric"]["status"] == "unavailable"
    assert precise["numeric"]["reason"] == "unsafe_numeric_precision"
    assert precise["numeric"]["min"] is None
    assert precise["numeric"]["max"] is None
    assert precise["numeric"]["stddev_pop"] is None
    assert precise["histogram"]["reason"] == "unsafe_numeric_precision"
    assert precise["histogram"]["bins"] == []
    assert report["correlations"]["values"] == [[None]]
    assert report["correlations"]["reasons"] == [["unsafe_numeric_precision"]]


def test_decimal_near_declared_limit_fails_closed_without_cast_overflow(tmp_path):
    maximum_minus_one = Decimal("99999999999999999999999999999999999998")
    maximum = Decimal("99999999999999999999999999999999999999")
    report = _analyze(
        tmp_path,
        pd.DataFrame({"precise": [maximum_minus_one, maximum]}),
        config=DescriptiveConfig(low_cardinality_threshold=10),
    )

    precise = _field(report, "precise")
    assert precise["duckdb_type"] == "DECIMAL(38,0)"
    assert precise["distinct_count"] == 2
    assert precise["numeric"]["status"] == "unavailable"
    assert precise["numeric"]["reason"] == "unsafe_numeric_precision"
    assert _frequency_counts(precise) == [
        ({"type": "bigint", "value": str(maximum_minus_one)}, 1),
        ({"type": "bigint", "value": str(maximum)}, 1),
    ]
    assert json.dumps(report, allow_nan=False, sort_keys=True)


def test_correlation_is_pairwise_finite_with_null_reasons_and_directional_symmetry(tmp_path):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, None],
            "y": [2.0, 4.0, None, np.inf],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "constant_2": [7.0, 7.0, 7.0, 7.0],
            "lonely": [None, None, None, 1.0],
        }
    )
    report = _analyze(tmp_path, frame, config=DescriptiveConfig(correlation_batch_size=1))
    corr = report["correlations"]
    idx = {name: index for index, name in enumerate(corr["columns"])}

    x, y = idx["x"], idx["y"]
    constant, constant_2, lonely = idx["constant"], idx["constant_2"], idx["lonely"]
    assert corr["method"] == "pearson"
    assert corr["basis"] == "pairwise_finite"
    assert corr["pair_counts"][x][y] == corr["pair_counts"][y][x] == 2
    assert corr["values"][x][y] == corr["values"][y][x] == pytest.approx(1.0)
    assert corr["reasons"][x][y] == corr["reasons"][y][x] == "ok"
    assert corr["values"][x][x] == 1.0
    assert corr["reasons"][x][x] == "ok"
    assert corr["values"][constant][constant] is None
    assert corr["reasons"][constant][constant] == "zero_variance_both"
    assert corr["reasons"][constant][constant_2] == "zero_variance_both"
    assert corr["reasons"][x][constant] == "zero_variance_right"
    assert corr["reasons"][constant][x] == "zero_variance_left"
    assert corr["pair_counts"][x][lonely] == 0
    assert corr["values"][x][lonely] is None
    assert corr["reasons"][x][lonely] == "insufficient_pairs"
    assert corr["pair_counts"][lonely][lonely] == 1
    assert corr["reasons"][lonely][lonely] == "insufficient_pairs"


def test_summary_and_correlation_batch_sizes_do_not_change_results(tmp_path):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": ["x", "y", "x"]})
    path = _write_parquet(tmp_path, frame)
    base = analyze_parquet(path, temp_directory=tmp_path / "temp-a")
    tiny_batches = analyze_parquet(
        path,
        temp_directory=tmp_path / "temp-b",
        config=DescriptiveConfig(summary_batch_size=1, correlation_batch_size=1),
    )

    assert {key: value for key, value in base.items() if key != "config"} == {
        key: value for key, value in tiny_batches.items() if key != "config"
    }


@pytest.mark.parametrize(
    ("kwargs", "dimension", "actual", "limit"),
    [
        ({"max_columns": 1}, "columns", 3, 1),
        ({"max_numeric_columns": 1}, "numeric_columns", 3, 1),
        ({"max_pairs": 1}, "correlation_pairs", 3, 1),
    ],
)
def test_budget_overflow_is_typed_and_never_silently_truncated(
    tmp_path,
    kwargs,
    dimension,
    actual,
    limit,
):
    path = _write_parquet(tmp_path, pd.DataFrame({"a": [1], "b": [2], "c": [3]}))

    with pytest.raises(DescriptiveBudgetError) as raised:
        analyze_parquet(
            path,
            temp_directory=tmp_path / "duckdb-tmp",
            config=DescriptiveConfig(**kwargs),
        )

    assert raised.value.to_detail() == {
        "kind": "descriptive_budget_exceeded",
        "dimension": dimension,
        "actual": actual,
        "limit": limit,
    }


@pytest.mark.parametrize(
    "field",
    [
        "max_columns",
        "max_numeric_columns",
        "max_pairs",
        "frequency_top_k",
        "low_cardinality_threshold",
        "histogram_bins",
        "summary_batch_size",
        "correlation_batch_size",
    ],
)
def test_every_config_budget_must_be_an_explicit_positive_integer(field):
    with pytest.raises(DescriptiveConfigError, match=field):
        DescriptiveConfig(**{field: 0})
    with pytest.raises(DescriptiveConfigError, match=field):
        DescriptiveConfig(**{field: True})


@pytest.mark.parametrize(
    ("field", "maximum"),
    [
        ("max_columns", 500),
        ("max_numeric_columns", 128),
        ("max_pairs", 8128),
        ("frequency_top_k", 100),
        ("low_cardinality_threshold", 1000),
        ("histogram_bins", 200),
        ("summary_batch_size", 64),
        ("correlation_batch_size", 512),
    ],
)
def test_every_config_budget_has_a_non_bypassable_hard_maximum(field, maximum):
    configured = DescriptiveConfig(**{field: maximum})
    assert configured.to_dict()[field] == maximum

    with pytest.raises(
        DescriptiveConfigError,
        match=rf"{field} must be at most {maximum}",
    ):
        DescriptiveConfig(**{field: maximum + 1})


def test_only_parquet_is_accepted(tmp_path):
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame({"x": [1]}).to_csv(csv_path, index=False)

    with pytest.raises(DescriptiveInputError, match="normalized Parquet"):
        analyze_parquet(csv_path, temp_directory=tmp_path / "duckdb-tmp")


def test_field_value_sanitizer_is_stable_json_safe_and_preserves_exact_counts(tmp_path):
    frame = pd.DataFrame({"customer_name": ["Alice", "Bob", "Alice", None], "bad": [0, 1, 0, 1]})

    def stable_token(value: dict) -> dict:
        canonical = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        token = hashlib.sha256(("test-only\0" + canonical).encode()).hexdigest()[:16]
        return {"type": "string", "value": f"token:{token}"}

    config = DescriptiveConfig(low_cardinality_threshold=10)
    first = _analyze(
        tmp_path,
        frame,
        target_column="bad",
        config=config,
        value_sanitizers={"customer_name": stable_token, "bad": stable_token},
    )
    second = _analyze(
        tmp_path,
        frame,
        target_column="bad",
        config=config,
        value_sanitizers={"customer_name": stable_token, "bad": stable_token},
    )

    assert first == second
    serialized = json.dumps(first, allow_nan=False, sort_keys=True)
    assert "Alice" not in serialized
    assert "Bob" not in serialized
    for column in ("customer_name", "bad"):
        frequency = _field(first, column)["frequency"]
        assert frequency["values_sanitized"] is True
        assert sum(item["count"] for item in frequency["items"]) + frequency["other_count"] == 4
        assert sum(item["rate_all"] for item in frequency["items"]) + frequency["other_rate_all"] == pytest.approx(1.0)
    assert first["target_distribution"]["frequency"] == _field(first, "bad")["frequency"]


@pytest.mark.parametrize(
    "bad_value",
    [
        {"type": "null", "value": "raw-value"},
        {"type": "float", "value": np.inf},
        {"type": "bigint", "value": 9007199254740993},
        {"type": "bigint", "value": "09007199254740993"},
        {"type": "token", "value": "unsupported-tag"},
    ],
)
def test_value_sanitizer_fails_closed_on_invalid_or_non_json_scalar(tmp_path, bad_value):
    path = _write_parquet(tmp_path, pd.DataFrame({"name": ["Alice"]}))

    with pytest.raises(DescriptiveSanitizerError, match="tagged scalar"):
        analyze_parquet(
            path,
            temp_directory=tmp_path / "duckdb-tmp",
            value_sanitizers={"name": lambda value: bad_value},
        )


def test_value_sanitizer_fails_closed_when_tagged_scalar_contains_extra_keys(tmp_path):
    path = _write_parquet(tmp_path, pd.DataFrame({"name": ["Alice"]}))

    def leaking_sanitizer(value: dict) -> dict:
        return {
            "type": "string",
            "value": "token:alice",
            "raw": value["value"],
        }

    with pytest.raises(DescriptiveSanitizerError, match="tagged scalar"):
        analyze_parquet(
            path,
            temp_directory=tmp_path / "duckdb-tmp",
            value_sanitizers={"name": leaking_sanitizer},
        )


def test_core_never_calls_data_backend_read_frame_for_full_table(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, pd.DataFrame({"x": [1, 2, 3]}))

    def forbidden(*args, **kwargs):
        raise AssertionError("full-table DataBackend.read_frame is forbidden")

    monkeypatch.setattr(DataBackend, "read_frame", forbidden)
    report = analyze_parquet(path, temp_directory=tmp_path / "duckdb-tmp")

    assert report["dataset"]["row_count"] == 3
    assert json.dumps(report, allow_nan=False)
