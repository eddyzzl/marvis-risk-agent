from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from marvis.data.transforms import (
    TRANSFORM_EXECUTION_MODE,
    TRANSFORM_RESULT_SCHEMA_VERSION,
    TransformBudgetError,
    TransformConfig,
    TransformConfigError,
    TransformExecutionError,
    TransformInputError,
    transform_parquet,
)


def _write_parquet(tmp_path: Path, select_sql: str) -> Path:
    path = tmp_path / "source.parquet"
    conn = duckdb.connect()
    try:
        escaped_path = str(path).replace("'", "''")
        conn.execute(f"COPY ({select_sql}) TO '{escaped_path}' (FORMAT PARQUET)")
    finally:
        conn.close()
    return path


def _rows(path: Path, columns: str = "*") -> list[tuple]:
    conn = duckdb.connect()
    try:
        return conn.execute(
            f"SELECT {columns} FROM read_parquet(?)", [str(path)]
        ).fetchall()
    finally:
        conn.close()


def _run(tmp_path: Path, source: Path, operations: list[dict], **kwargs):
    output = tmp_path / "result.parquet"
    evidence = transform_parquet(
        source,
        output,
        temp_directory=tmp_path / "duckdb-tmp",
        operations=operations,
        **kwargs,
    )
    return output, evidence


def test_ordered_pipeline_writes_parquet_and_complete_finite_evidence(tmp_path):
    source = _write_parquet(
        tmp_path,
        """
        SELECT * FROM (VALUES
            (1, '10.0', NULL, 2.0),
            (1, '20.0', 'A', 3.0),
            (2, 'bad',  'B', 4.0),
            (3, '30.0', NULL, 5.0)
        ) t(customer_id, raw_amount, segment, divisor)
        """,
    )

    output, evidence = _run(
        tmp_path,
        source,
        [
            {
                "op": "rename_columns",
                "mapping": {"raw_amount": "amount"},
            },
            {
                "op": "cast_columns",
                "casts": [
                    {"column": "amount", "to_type": "DOUBLE", "mode": "try"}
                ],
            },
            {
                "op": "fill_missing",
                "fills": [
                    {"column": "amount", "method": "mean"},
                    {
                        "column": "segment",
                        "method": "constant",
                        "value": "UNKNOWN",
                    },
                ],
            },
            {
                "op": "derive_columns",
                "derivations": [
                    {
                        "name": "ratio",
                        "expression": {
                            "op": "divide",
                            "left": {"column": "amount"},
                            "right": {"column": "divisor"},
                        },
                        "to_type": "DOUBLE",
                    },
                    {
                        "name": "band",
                        "expression": {
                            "op": "case",
                            "cases": [
                                {
                                    "when": {
                                        "op": "gte",
                                        "left": {"column": "amount"},
                                        "right": {
                                            "literal": 20,
                                            "type": "DOUBLE",
                                        },
                                    },
                                    "then": {"literal": "high"},
                                }
                            ],
                            "else": {
                                "op": "coalesce",
                                "args": [
                                    {"column": "segment"},
                                    {"literal": "low"},
                                ],
                            },
                        },
                    },
                ],
            },
            {
                "op": "filter_rows",
                "predicate": {
                    "op": "and",
                    "args": [
                        {
                            "op": "gt",
                            "left": {"column": "amount"},
                            "right": {"literal": 5.0},
                        },
                        {
                            "op": "not",
                            "arg": {
                                "op": "eq",
                                "left": {"column": "segment"},
                                "right": {"literal": "B"},
                            },
                        },
                    ],
                },
            },
            {
                "op": "deduplicate",
                "keys": ["customer_id"],
                "order_by": [
                    {"column": "amount", "direction": "desc", "nulls": "last"}
                ],
            },
            {"op": "drop_columns", "columns": ["divisor"]},
        ],
    )

    assert output.is_file()
    assert _rows(output, "customer_id, amount, segment, ratio, band") == [
        (1, 20.0, "A", pytest.approx(20.0 / 3.0), "high"),
        (3, 30.0, "UNKNOWN", 6.0, "high"),
    ]
    assert evidence["schema_version"] == TRANSFORM_RESULT_SCHEMA_VERSION
    assert evidence["schema_version"] == "transform-result.v1"
    assert evidence["summary"]["row_count_before"] == 4
    assert evidence["summary"]["row_count_after"] == 2
    assert evidence["summary"]["column_count_before"] == 4
    assert evidence["summary"]["column_count_after"] == 5
    assert [step["op"] for step in evidence["steps"]] == [
        "rename_columns",
        "cast_columns",
        "fill_missing",
        "derive_columns",
        "filter_rows",
        "deduplicate",
        "drop_columns",
    ]
    assert evidence["steps"][1]["impact"]["invalid_to_null_count"] == 1
    assert evidence["steps"][2]["impact"]["filled_count"] == 3
    assert evidence["steps"][4]["impact"]["removed_rows"] == 1
    assert evidence["steps"][5]["impact"]["removed_rows"] == 1
    assert evidence["output"]["content_hash"]
    assert evidence["output"]["size_bytes"] == output.stat().st_size
    assert evidence["output"]["path"] == str(output)
    json.dumps(evidence, allow_nan=False)


def test_strict_cast_fails_closed_and_try_cast_records_invalid_rows(tmp_path):
    source = _write_parquet(
        tmp_path,
        "SELECT * FROM (VALUES ('1'), ('oops'), (NULL)) t(value)",
    )
    output = tmp_path / "strict.parquet"

    with pytest.raises(TransformExecutionError, match="cast_columns"):
        transform_parquet(
            source,
            output,
            temp_directory=tmp_path / "duckdb-tmp",
            operations=[
                {
                    "op": "cast_columns",
                    "casts": [
                        {"column": "value", "to_type": "INTEGER", "mode": "strict"}
                    ],
                }
            ],
        )

    assert not output.exists()
    try_output, evidence = _run(
        tmp_path,
        source,
        [
            {
                "op": "cast_columns",
                "casts": [
                    {"column": "value", "to_type": "INTEGER", "mode": "try"}
                ],
            }
        ],
    )
    assert _rows(try_output) == [(1,), (None,), (None,)]
    assert evidence["steps"][0]["impact"] == {
        "columns": ["value"],
        "mode_by_column": {"value": "try"},
        "non_null_input_count": 2,
        "invalid_to_null_count": 1,
        "by_column": {
            "value": {
                "mode": "try",
                "non_null_input_count": 2,
                "invalid_to_null_count": 1,
            }
        },
    }


def test_fill_statistics_are_finite_only_and_keep_declared_column_type(tmp_path):
    source = _write_parquet(
        tmp_path,
        "SELECT * FROM (VALUES (1.0), (3.0), (NULL), ('Infinity'::DOUBLE)) t(x)",
    )

    output, evidence = _run(
        tmp_path,
        source,
        [
            {
                "op": "fill_missing",
                "fills": [{"column": "x", "method": "mean"}],
            }
        ],
    )

    assert _rows(output) == [(1.0,), (3.0,), (2.0,), (float("inf"),)]
    assert evidence["steps"][0]["impact"]["filled_count"] == 1
    assert evidence["output"]["columns"] == [{"name": "x", "duckdb_type": "DOUBLE"}]
    json.dumps(evidence, allow_nan=False)


def test_float_mean_imputation_is_reproducible_across_runtime_thread_settings(
    tmp_path,
    monkeypatch,
):
    source = _write_parquet(
        tmp_path,
        """
        SELECT
            CASE
                WHEN i = 400000 THEN NULL
                WHEN i % 4 = 0 THEN 1e16::DOUBLE
                WHEN i % 4 = 1 THEN 1.0::DOUBLE
                WHEN i % 4 = 2 THEN -1e16::DOUBLE
                ELSE i::DOUBLE / 1000003.0
            END AS x,
            i AS ordinal
        FROM range(400001) AS values_table(i)
        """,
    )
    operations = [
        {
            "op": "fill_missing",
            "fills": [{"column": "x", "method": "mean"}],
        }
    ]

    monkeypatch.setenv("MARVIS_DUCKDB_THREADS", "2")
    output_two = tmp_path / "result-threads-2.parquet"
    evidence_two = transform_parquet(
        source,
        output_two,
        temp_directory=tmp_path / "duckdb-tmp",
        operations=operations,
    )
    monkeypatch.setenv("MARVIS_DUCKDB_THREADS", "8")
    output_eight = tmp_path / "result-threads-8.parquet"
    evidence_eight = transform_parquet(
        source,
        output_eight,
        temp_directory=tmp_path / "duckdb-tmp",
        operations=operations,
    )

    assert evidence_two["execution"] == evidence_eight["execution"] == {
        "mode": TRANSFORM_EXECUTION_MODE,
        "duckdb_threads": 1,
        "preserve_insertion_order": True,
    }
    assert evidence_two["output"]["content_hash"] == evidence_eight["output"][
        "content_hash"
    ]
    assert output_two.read_bytes() == output_eight.read_bytes()

    conn = duckdb.connect()
    try:
        imputed_two = conn.execute(
            "SELECT x FROM read_parquet(?) WHERE ordinal = 400000",
            [str(output_two)],
        ).fetchone()[0]
        imputed_eight = conn.execute(
            "SELECT x FROM read_parquet(?) WHERE ordinal = 400000",
            [str(output_eight)],
        ).fetchone()[0]
    finally:
        conn.close()
    assert imputed_two == imputed_eight


@pytest.mark.parametrize(
    "operations,match",
    [
        ([], "at least one"),
        ([{"op": "drop_columns", "columns": []}], "must not be empty"),
        (
            [{"op": "rename_columns", "mapping": {"a": "a"}}],
            "same name",
        ),
        (
            [{"op": "rename_columns", "mapping": {"a": "b"}}],
            "duplicate output column",
        ),
        (
            [
                {
                    "op": "derive_columns",
                    "derivations": [
                        {"name": "a", "expression": {"literal": 1}}
                    ],
                }
            ],
            "already exists",
        ),
    ],
)
def test_empty_and_overwriting_operations_are_rejected(tmp_path, operations, match):
    source = _write_parquet(tmp_path, "SELECT 1 AS a, 2 AS b")

    with pytest.raises(TransformInputError, match=match):
        _run(tmp_path, source, operations)

    assert not (tmp_path / "result.parquet").exists()


def test_filter_and_expression_ast_reject_sql_and_unknown_fields(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS safe")
    invalid_operations = [
        [{"op": "filter_rows", "predicate": {"sql": "TRUE; DROP TABLE x"}}],
        [
            {
                "op": "filter_rows",
                "predicate": {
                    "op": "eq",
                    "left": {"column": 'safe\" OR TRUE --'},
                    "right": {"literal": 1},
                },
            }
        ],
        [
            {
                "op": "derive_columns",
                "derivations": [
                    {
                        "name": "x",
                        "expression": {
                            "op": "shell",
                            "args": [{"literal": "rm -rf /"}],
                        },
                    }
                ],
            }
        ],
    ]

    for index, operations in enumerate(invalid_operations):
        with pytest.raises(TransformInputError):
            transform_parquet(
                source,
                tmp_path / f"bad-{index}.parquet",
                temp_directory=tmp_path / "duckdb-tmp",
                operations=operations,
            )
        assert not (tmp_path / f"bad-{index}.parquet").exists()


def test_nonfinite_literals_and_unbounded_integer_literals_are_rejected(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS x")

    for value in (float("nan"), float("inf"), 2**53):
        with pytest.raises(TransformInputError, match="literal"):
            transform_parquet(
                source,
                tmp_path / f"bad-{type(value).__name__}.parquet",
                temp_directory=tmp_path / "duckdb-tmp",
                operations=[
                    {
                        "op": "filter_rows",
                        "predicate": {
                            "op": "eq",
                            "left": {"column": "x"},
                            "right": {"literal": value},
                        },
                    }
                ],
            )


def test_deduplicate_requires_keys_and_explicit_total_order(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS id, 2 AS value")

    for operation in (
        {"op": "deduplicate", "keys": [], "order_by": [{"column": "value"}]},
        {"op": "deduplicate", "keys": ["id"], "order_by": []},
        {
            "op": "deduplicate",
            "keys": ["id"],
            "order_by": [{"column": "value", "direction": "sideways"}],
        },
    ):
        with pytest.raises(TransformInputError):
            _run(tmp_path, source, [operation])


def test_resource_caps_fail_before_writing_output(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS a, 2 AS b")

    with pytest.raises(TransformBudgetError, match="operations"):
        _run(
            tmp_path,
            source,
            [
                {"op": "rename_columns", "mapping": {"a": "x"}},
                {"op": "rename_columns", "mapping": {"x": "y"}},
            ],
            config=TransformConfig(max_operations=1),
        )
    assert not (tmp_path / "result.parquet").exists()

    with pytest.raises(TransformConfigError, match="max_operations"):
        TransformConfig(max_operations=101)


def test_expression_budgets_keep_transform_error_contract(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS value")
    comparison = {
        "op": "eq",
        "left": {"column": "value"},
        "right": {"literal": 1},
    }

    with pytest.raises(TransformBudgetError) as nodes:
        _run(
            tmp_path,
            source,
            [{"op": "filter_rows", "predicate": comparison}],
            config=TransformConfig(max_ast_nodes=2),
        )
    assert (nodes.value.dimension, nodes.value.actual, nodes.value.limit) == (
        "expression_nodes",
        3,
        2,
    )

    with pytest.raises(TransformBudgetError) as depth:
        _run(
            tmp_path,
            source,
            [
                {
                    "op": "filter_rows",
                    "predicate": {"op": "not", "arg": comparison},
                }
            ],
            config=TransformConfig(max_ast_depth=2),
        )
    assert (depth.value.dimension, depth.value.actual, depth.value.limit) == (
        "expression_depth",
        3,
        2,
    )


def test_existing_output_is_never_overwritten(tmp_path):
    source = _write_parquet(tmp_path, "SELECT 1 AS x")
    output = tmp_path / "result.parquet"
    output.write_bytes(b"user-owned")

    with pytest.raises(TransformInputError, match="already exists"):
        transform_parquet(
            source,
            output,
            temp_directory=tmp_path / "duckdb-tmp",
            operations=[{"op": "rename_columns", "mapping": {"x": "y"}}],
        )

    assert output.read_bytes() == b"user-owned"


def test_input_and_operation_shapes_are_strict(tmp_path):
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(TransformInputError, match="Parquet"):
        _run(
            tmp_path,
            csv_path,
            [{"op": "rename_columns", "mapping": {"x": "y"}}],
        )

    source = _write_parquet(tmp_path, "SELECT 1 AS x")
    with pytest.raises(TransformInputError, match="unexpected fields"):
        _run(
            tmp_path,
            source,
            [{"op": "rename_columns", "mapping": {"x": "y"}, "sql": "bad"}],
        )
