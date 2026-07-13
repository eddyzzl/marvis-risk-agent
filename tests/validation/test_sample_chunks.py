from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marvis.validation.sample_chunks import (
    iter_sample_chunks,
    read_selected_columns,
)
from marvis.validation.sample_schema import inspect_sample_schema


def test_csv_chunks_project_columns_and_assign_contiguous_row_ids(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {"x": range(5), "y": [0, 1, 0, 1, 0], "unused": range(10, 15)}
    ).to_csv(path, index=False)

    chunks = list(iter_sample_chunks(path, columns=("y", "x"), chunk_size=2))

    assert [chunk.frame.columns.tolist() for chunk in chunks] == [
        ["y", "x"],
        ["y", "x"],
        ["y", "x"],
    ]
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert all(chunk.row_ids.dtype == np.dtype("int64") for chunk in chunks)
    assert [len(chunk.frame) for chunk in chunks] == [2, 2, 1]


@pytest.mark.parametrize("chunk_size", [False, True, 1.5, 0, -1, "2", None])
def test_chunk_reader_rejects_non_positive_integer_chunk_size(tmp_path, chunk_size):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        list(
            iter_sample_chunks(
                path,
                columns=("x",),
                chunk_size=chunk_size,  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("chunk_size", [False, True, 1.5, 0, -1, "2", None])
def test_selected_reader_rejects_non_positive_integer_chunk_size(
    tmp_path, chunk_size
):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        read_selected_columns(
            path,
            columns=("x",),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )


def test_utf8_and_gb18030_csv_use_inspected_encoding_at_different_chunk_sizes(
    tmp_path,
):
    utf8_path = tmp_path / "utf8.csv"
    gb_path = tmp_path / "gb.csv"
    expected = pd.DataFrame({"客户": ["甲", "乙", "丙"], "分数": [1, 2, 3]})
    expected.to_csv(utf8_path, index=False, encoding="utf-8")
    expected.to_csv(gb_path, index=False, encoding="gb18030")
    utf8_schema = inspect_sample_schema(utf8_path)
    gb_schema = inspect_sample_schema(gb_path)

    utf8 = read_selected_columns(
        utf8_path,
        columns=("分数", "客户"),
        chunk_size=1,
        schema=utf8_schema,
    )
    gb18030 = read_selected_columns(
        gb_path,
        columns=("分数", "客户"),
        chunk_size=2,
        schema=gb_schema,
    )

    assert utf8.to_dict("list") == {"分数": [1, 2, 3], "客户": ["甲", "乙", "丙"]}
    assert gb18030.to_dict("list") == utf8.to_dict("list")
    assert gb_schema.encoding == "gb18030"


def test_parquet_multiple_row_groups_are_rechunked_with_contiguous_ids(tmp_path):
    path = tmp_path / "sample.parquet"
    table = pa.table({"a": range(7), "b": range(10, 17), "unused": range(7)})
    pq.write_table(table, path, row_group_size=2)
    assert pq.ParquetFile(path).metadata.num_row_groups == 4

    chunks = list(iter_sample_chunks(path, columns=("b", "a"), chunk_size=3))

    assert [len(chunk.frame) for chunk in chunks] == [3, 3, 1]
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1, 2], [3, 4, 5], [6]]
    assert pd.concat([chunk.frame for chunk in chunks], ignore_index=True).to_dict(
        "list"
    ) == {"b": list(range(10, 17)), "a": list(range(7))}


def test_feather_chunks_project_in_requested_order_and_keep_short_tail(tmp_path):
    path = tmp_path / "sample.feather"
    pd.DataFrame({"x": range(5), "y": range(5, 10), "z": range(10, 15)}).to_feather(
        path
    )

    chunks = list(iter_sample_chunks(path, columns=("z", "x"), chunk_size=2))

    assert [len(chunk.frame) for chunk in chunks] == [2, 2, 1]
    assert all(tuple(chunk.frame.columns) == ("z", "x") for chunk in chunks)
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]


def test_xlsx_uses_confirmed_sheet_and_emits_chunks_after_whole_sheet_read(tmp_path):
    """Excel is intentionally loaded as a whole before it is sliced into chunks."""

    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="empty", index=False, header=False)
        pd.DataFrame({"x": range(5), "y": range(5, 10)}).to_excel(
            writer, sheet_name="selected", index=False
        )
    schema = inspect_sample_schema(path)

    frame = read_selected_columns(
        path,
        columns=("y", "x"),
        chunk_size=2,
        schema=schema,
    )

    assert schema.sheet_name == "selected"
    assert frame.to_dict("list") == {"y": list(range(5, 10)), "x": list(range(5))}


def test_supplied_schema_prevents_reinspection(tmp_path, monkeypatch):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)
    schema = inspect_sample_schema(path)

    import marvis.validation.sample_chunks as module

    monkeypatch.setattr(
        module,
        "inspect_sample_schema",
        lambda _path: pytest.fail("supplied schema must prevent reinspection"),
    )

    chunks = list(
        iter_sample_chunks(path, columns=("x",), chunk_size=2, schema=schema)
    )

    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2]]


def test_missing_projection_column_is_rejected(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    schema = inspect_sample_schema(path)

    with pytest.raises(ValueError, match="missing columns: absent"):
        list(
            iter_sample_chunks(
                path,
                columns=("absent",),
                chunk_size=2,
                schema=schema,
            )
        )


@pytest.mark.parametrize("suffix", ["csv", "parquet", "feather", "xlsx"])
def test_zero_column_projection_uses_carrier_but_returns_no_columns(tmp_path, suffix):
    path = tmp_path / f"sample.{suffix}"
    source = pd.DataFrame({"carrier": range(5), "other": range(10, 15)})
    if suffix == "csv":
        source.to_csv(path, index=False)
    elif suffix == "parquet":
        source.to_parquet(path, index=False, row_group_size=2)
    elif suffix == "feather":
        source.to_feather(path)
    else:
        source.to_excel(path, index=False)
    schema = inspect_sample_schema(path)

    chunks = list(iter_sample_chunks(path, columns=(), chunk_size=2, schema=schema))
    result = read_selected_columns(path, columns=(), chunk_size=3, schema=schema)

    assert [len(chunk.frame) for chunk in chunks] == [2, 2, 1]
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2, 3], [4]]
    assert all(chunk.frame.columns.empty for chunk in chunks)
    assert result.shape == (5, 0)


def test_zero_column_projection_without_schema_inspects_once(tmp_path, monkeypatch):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"carrier": [1, 2, 3]}).to_csv(path, index=False)

    import marvis.validation.sample_chunks as module

    original = module.inspect_sample_schema
    calls = 0

    def inspect_once(selected_path):
        nonlocal calls
        calls += 1
        return original(selected_path)

    monkeypatch.setattr(module, "inspect_sample_schema", inspect_once)

    chunks = list(iter_sample_chunks(path, columns=(), chunk_size=2))

    assert calls == 1
    assert [chunk.row_ids.tolist() for chunk in chunks] == [[0, 1], [2]]


def test_zero_column_projection_rejects_schema_without_carrier(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    schema = replace(inspect_sample_schema(path), columns=())

    with pytest.raises(ValueError, match="no carrier column"):
        list(iter_sample_chunks(path, columns=(), chunk_size=2, schema=schema))


def test_read_selected_columns_checks_cancellation_once_per_chunk(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": range(5)}).to_csv(path, index=False)
    checks: list[int] = []

    result = read_selected_columns(
        path,
        columns=("x",),
        chunk_size=2,
        cancellation_check=lambda: checks.append(len(checks)),
    )

    assert checks == [0, 1, 2]
    assert result["x"].tolist() == list(range(5))


def test_read_selected_columns_propagates_cancellation_before_appending_chunk(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": range(5)}).to_csv(path, index=False)
    checks = 0

    def cancel_on_second_chunk() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        read_selected_columns(
            path,
            columns=("x",),
            chunk_size=2,
            cancellation_check=cancel_on_second_chunk,
        )

    assert checks == 2


def test_empty_sample_returns_empty_frame_with_requested_columns(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x,y\n", encoding="utf-8")
    schema = inspect_sample_schema(path)

    chunks = list(
        iter_sample_chunks(path, columns=("y", "x"), chunk_size=2, schema=schema)
    )
    result = read_selected_columns(
        path, columns=("y", "x"), chunk_size=2, schema=schema
    )

    assert chunks == []
    assert result.empty
    assert result.columns.tolist() == ["y", "x"]
