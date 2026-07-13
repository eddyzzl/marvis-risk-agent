from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import re
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from marvis.validation.sample_schema import (
    MAX_CSV_PREVIEW_BYTES,
    inspect_sample_schema,
    iter_sample_projection,
)


def _patch_xlsx_dimension(path, dimension: str) -> None:
    patched = path.with_name("patched.xlsx")
    with ZipFile(path, "r") as source, ZipFile(
        patched, "w", compression=ZIP_DEFLATED
    ) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                payload, replacements = re.subn(
                    br'<dimension ref="[^"]+"\s*/>',
                    f'<dimension ref="{dimension}"/>'.encode(),
                    payload,
                    count=1,
                )
                assert replacements == 1
            target.writestr(item, payload)
    patched.replace(path)


@pytest.fixture
def understated_dimension_xlsx(tmp_path):
    path = tmp_path / "understated.xlsx"
    pd.DataFrame({"x": range(10)}).to_excel(path, index=False)
    _patch_xlsx_dimension(path, "A1")
    return path


def test_csv_inspection_uses_bounded_encoding_fallback_and_streaming_hash(tmp_path):
    path = tmp_path / "sample.csv"
    payload = "年龄,标签\n20,0\n30,1\n".encode("gb18030")
    path.write_bytes(payload)

    schema = inspect_sample_schema(path)

    assert schema.columns == ("年龄", "标签")
    assert schema.preview_row_count == 2
    assert schema.row_count is None
    assert schema.encoding == "gb18030"
    assert schema.sha256 == sha256(payload).hexdigest()
    assert schema.path == str(path.resolve())


def test_csv_rejects_raw_duplicate_headers_before_pandas_mangling(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        inspect_sample_schema(path)


def test_csv_normalizes_unreferenced_blank_headers_without_blocking_projection(
    tmp_path,
):
    path = tmp_path / "blank-index-columns.csv"
    path.write_text(",,x,y\n1,1,10,0\n2,2,20,1\n", encoding="utf-8")

    schema = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(
            path,
            columns=("x", "y"),
            chunk_size=1,
            schema=schema,
        )
    )

    assert schema.columns[:2] == (
        "__marvis_unnamed_column_0__",
        "__marvis_unnamed_column_1__",
    )
    assert pd.concat(chunks, ignore_index=True).to_dict("list") == {
        "x": [10, 20],
        "y": [0, 1],
    }


def test_csv_projection_is_chunked_and_preserves_requested_order(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": range(5), "y": range(10, 15), "z": range(20, 25)}).to_csv(
        path, index=False
    )
    schema = inspect_sample_schema(path)

    chunks = list(
        iter_sample_projection(
            path, columns=("z", "x"), chunk_size=2, schema=schema
        )
    )

    assert [len(frame) for frame in chunks] == [2, 2, 1]
    assert all(tuple(frame.columns) == ("z", "x") for frame in chunks)
    assert pd.concat(chunks, ignore_index=True).to_dict("list") == {
        "z": list(range(20, 25)),
        "x": list(range(5)),
    }


def test_csv_rejects_header_crossing_bounded_preview_limit(tmp_path):
    path = tmp_path / "oversized-header.csv"
    path.write_bytes((b"a" * MAX_CSV_PREVIEW_BYTES) + b",b\n1,2\n")

    with pytest.raises(ValueError, match="header exceeds preview byte limit"):
        inspect_sample_schema(path)


def test_csv_truncation_keeps_only_complete_quote_aware_records(tmp_path):
    path = tmp_path / "quoted-boundary.csv"
    header = b"a,b\r\n"
    complete = b'1,"line one\nline two ""quoted"""\r\n'
    partial_prefix = b'2,"'
    filler = b"x" * (
        MAX_CSV_PREVIEW_BYTES
        - len(header)
        - len(complete)
        - len(partial_prefix)
        + 100
    )
    path.write_bytes(header + complete + partial_prefix + filler)

    schema = inspect_sample_schema(path)

    assert schema.columns == ("a", "b")
    assert schema.preview_row_count == 1


def test_csv_mid_field_quote_is_literal_like_python_csv_reader(tmp_path):
    path = tmp_path / "literal-quote.csv"
    path.write_bytes(b'a"b,c\n1,2\n')

    schema = inspect_sample_schema(path)

    assert schema.columns == ('a"b', "c")
    assert schema.preview_row_count == 1


def test_csv_truncation_keeps_complete_header_with_literal_mid_field_quote(
    tmp_path,
):
    path = tmp_path / "literal-quote-boundary.csv"
    header = b'a"b,c\r\n'
    complete = b"1,2\r\n"
    partial_prefix = b'3,"'
    filler = b"x" * (
        MAX_CSV_PREVIEW_BYTES
        - len(header)
        - len(complete)
        - len(partial_prefix)
        + 100
    )
    path.write_bytes(header + complete + partial_prefix + filler)

    schema = inspect_sample_schema(path)

    assert schema.columns == ('a"b', "c")
    assert schema.preview_row_count == 1


def test_csv_utf8_bom_allows_quoted_first_header_with_newline(tmp_path):
    path = tmp_path / "bom-quoted.csv"
    path.write_bytes(
        b'\xef\xbb\xbf"first\nheader",second\r\n1,2\r\n'
    )

    schema = inspect_sample_schema(path)

    assert schema.columns == ("first\nheader", "second")
    assert schema.preview_row_count == 1
    assert schema.encoding == "utf-8-sig"


def test_csv_utf8_bom_quoted_header_crossing_preview_limit_is_rejected(tmp_path):
    path = tmp_path / "bom-truncated-header.csv"
    prefix = b'\xef\xbb\xbf"first\n'
    filler = b"x" * (MAX_CSV_PREVIEW_BYTES - len(prefix) + 100)
    path.write_bytes(prefix + filler + b'",second\n1,2\n')

    with pytest.raises(ValueError, match="header exceeds preview byte limit"):
        inspect_sample_schema(path)


def test_parquet_uses_metadata_and_iterates_multiple_row_groups(tmp_path):
    path = tmp_path / "sample.parquet"
    table = pa.table({"x": range(7), "y": range(10, 17), "z": range(20, 27)})
    pq.write_table(table, path, row_group_size=2)

    schema = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(
            path, columns=("z", "x"), chunk_size=2, schema=schema
        )
    )

    assert schema.columns == ("x", "y", "z")
    assert schema.row_count == 7
    assert schema.preview_row_count == 0
    assert len(chunks) >= 4
    assert all(len(frame) <= 2 for frame in chunks)
    assert all(tuple(frame.columns) == ("z", "x") for frame in chunks)
    assert pd.concat(chunks, ignore_index=True)["x"].tolist() == list(range(7))


def test_parquet_normalizes_blank_field_name_in_dtype_contract(tmp_path):
    path = tmp_path / "blank-field.parquet"
    table = pa.table(
        {
            "": pa.array([1, 2], type=pa.int64()),
            "x": pa.array([10, 20], type=pa.int64()),
            "y": pa.array([0, 1], type=pa.int64()),
        }
    )
    pq.write_table(table, path)

    schema = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(
            path,
            columns=("x", "y"),
            chunk_size=1,
            schema=schema,
        )
    )

    assert schema.columns == ("__marvis_unnamed_column_0__", "x", "y")
    assert schema.dtypes == {
        "__marvis_unnamed_column_0__": "int64",
        "x": "int64",
        "y": "int64",
    }
    assert "" not in schema.dtypes
    assert pd.concat(chunks, ignore_index=True).to_dict("list") == {
        "x": [10, 20],
        "y": [0, 1],
    }


def test_feather_uses_ipc_schema_and_iterates_record_batches(tmp_path):
    path = tmp_path / "sample.feather"
    schema = pa.schema([("x", pa.int64()), ("y", pa.int64()), ("z", pa.int64())])
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema) as writer:
        writer.write_batch(
            pa.record_batch([[0, 1, 2], [10, 11, 12], [20, 21, 22]], schema=schema)
        )
        writer.write_batch(
            pa.record_batch([[3, 4], [13, 14], [23, 24]], schema=schema)
        )

    inspected = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(
            path, columns=("y", "x"), chunk_size=2, schema=inspected
        )
    )

    assert inspected.columns == ("x", "y", "z")
    assert inspected.row_count == 5
    assert [len(frame) for frame in chunks] == [2, 1, 2]
    assert all(tuple(frame.columns) == ("y", "x") for frame in chunks)


def test_feather_normalizes_blank_field_name_in_dtype_contract(tmp_path):
    path = tmp_path / "blank-field.feather"
    schema = pa.schema(
        [("", pa.int64()), ("x", pa.int64()), ("y", pa.int64())]
    )
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, schema) as writer:
        writer.write_batch(
            pa.record_batch([[1, 2], [10, 20], [0, 1]], schema=schema)
        )

    inspected = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(
            path,
            columns=("x", "y"),
            chunk_size=1,
            schema=inspected,
        )
    )

    assert inspected.columns == ("__marvis_unnamed_column_0__", "x", "y")
    assert inspected.dtypes == {
        "__marvis_unnamed_column_0__": "int64",
        "x": "int64",
        "y": "int64",
    }
    assert "" not in inspected.dtypes
    assert pd.concat(chunks, ignore_index=True).to_dict("list") == {
        "x": [10, 20],
        "y": [0, 1],
    }


def test_excel_accepts_only_nonempty_second_sheet_and_projection_uses_it(tmp_path):
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="empty", index=False, header=False)
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_excel(
            writer, sheet_name="sample_data", index=False
        )

    schema = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(path, columns=("y",), chunk_size=2, schema=schema)
    )

    assert schema.sheet_name == "sample_data"
    assert schema.columns == ("x", "y")
    assert schema.row_count == 3
    assert schema.preview_row_count == 3
    assert [frame["y"].tolist() for frame in chunks] == [[4, 5], [6]]


def test_excel_normalizes_non_string_headers_consistently_for_projection(tmp_path):
    path = tmp_path / "numeric-header.xlsx"
    pd.DataFrame([[2024, "score"], [1, 0.2], [2, 0.8]]).to_excel(
        path, index=False, header=False
    )

    schema = inspect_sample_schema(path)
    chunks = list(
        iter_sample_projection(path, columns=("2024",), chunk_size=1, schema=schema)
    )

    assert schema.columns == ("2024", "score")
    assert [frame["2024"].tolist() for frame in chunks] == [[1], [2]]


def test_excel_rejects_multiple_nonempty_sheets_with_actionable_message(tmp_path):
    path = tmp_path / "multiple.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="one", index=False)
        pd.DataFrame({"x": [2]}).to_excel(writer, sheet_name="two", index=False)

    with pytest.raises(
        ValueError,
        match="样本工作簿包含多个非空 sheet，请另存为单 sheet 样本后重新选择",
    ):
        inspect_sample_schema(path)


def test_excel_inspection_rejects_file_above_byte_cap_before_hashing(
    tmp_path, monkeypatch
):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(path, index=False)

    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda _path: pytest.fail("oversized Excel must be rejected before hashing"),
    )

    with pytest.raises(ValueError, match="size limit"):
        inspect_sample_schema(path, max_excel_upload_bytes=path.stat().st_size - 1)


def test_excel_inspection_rejects_selected_sheet_above_row_cap(tmp_path):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"x": [1, 2, 3]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="row limit"):
        inspect_sample_schema(path, max_excel_rows=2)


def test_excel_projection_rechecks_schema_row_and_file_size_caps(
    tmp_path, monkeypatch
):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"x": [1, 2, 3]}).to_excel(path, index=False)
    schema = inspect_sample_schema(path)

    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module.pd,
        "read_excel",
        lambda *args, **kwargs: pytest.fail(
            "Excel caps must be enforced before full projection"
        ),
    )

    with pytest.raises(ValueError, match="row limit"):
        list(
            iter_sample_projection(
                path,
                columns=("x",),
                chunk_size=2,
                schema=schema,
                max_excel_rows=2,
            )
        )
    with pytest.raises(ValueError, match="size limit"):
        list(
            iter_sample_projection(
                path,
                columns=("x",),
                chunk_size=2,
                schema=schema,
                max_excel_upload_bytes=path.stat().st_size - 1,
            )
        )


def test_excel_projection_enforces_post_load_row_cap_when_metadata_understates_rows(
    tmp_path, monkeypatch
):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"x": [1, 2, 3]}).to_excel(path, index=False)
    schema = replace(inspect_sample_schema(path), row_count=2)

    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module,
        "_scan_excel_sheets",
        lambda _path, *, max_rows: module._ExcelSheetScan(
            data_row_counts={schema.sheet_name: 2},
            nonempty_sheets=(schema.sheet_name,),
            overflow_sheets=frozenset(),
        ),
    )

    with pytest.raises(ValueError, match="row limit after loading"):
        list(
            iter_sample_projection(
                path,
                columns=("x",),
                chunk_size=2,
                schema=schema,
                max_excel_rows=2,
            )
        )


def test_xlsx_inspection_ignores_understated_dimension_before_pandas_load(
    understated_dimension_xlsx, monkeypatch
):
    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module.pd,
        "read_excel",
        lambda *args, **kwargs: pytest.fail(
            "actual XLSX rows must be capped before pandas reads the sheet"
        ),
    )

    with pytest.raises(ValueError, match="row limit"):
        inspect_sample_schema(understated_dimension_xlsx, max_excel_rows=2)


def test_xlsx_projection_ignores_understated_dimension_before_full_load(
    tmp_path, monkeypatch
):
    path = tmp_path / "understated-projection.xlsx"
    pd.DataFrame({"x": range(10)}).to_excel(path, index=False)
    schema = replace(inspect_sample_schema(path), row_count=2)
    _patch_xlsx_dimension(path, "A1")

    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module.pd,
        "read_excel",
        lambda *args, **kwargs: pytest.fail(
            "actual XLSX rows must be capped before full projection"
        ),
    )

    with pytest.raises(ValueError, match="row limit"):
        list(
            iter_sample_projection(
                path,
                columns=("x",),
                chunk_size=2,
                schema=schema,
                max_excel_rows=2,
            )
        )


def test_excel_rejects_raw_duplicate_headers(tmp_path):
    path = tmp_path / "bad.xlsx"
    rows = [["x", "x"], [1, 2]]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)
    with pytest.raises(ValueError, match="duplicate"):
        inspect_sample_schema(path)


def test_excel_normalizes_blank_header_for_named_projection(tmp_path):
    path = tmp_path / "blank.xlsx"
    pd.DataFrame([[None, "x"], [1, 2]]).to_excel(
        path, index=False, header=False
    )

    schema = inspect_sample_schema(path)
    projected = pd.concat(
        iter_sample_projection(
            path,
            columns=("x",),
            chunk_size=1,
            schema=schema,
        ),
        ignore_index=True,
    )

    assert schema.columns == ("__marvis_unnamed_column_0__", "x")
    assert projected.to_dict("list") == {"x": [2]}


def test_xls_dependency_or_parse_error_is_deterministic_and_bounded(tmp_path):
    path = tmp_path / "bad.xls"
    path.write_bytes(b"not an xls workbook")
    with pytest.raises(ValueError, match=r"^无法读取样本工作簿"):
        inspect_sample_schema(path)


def test_supplied_schema_prevents_reinspection_and_rehash(tmp_path, monkeypatch):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(path, index=False)
    schema = inspect_sample_schema(path)

    import marvis.validation.sample_schema as module

    monkeypatch.setattr(
        module,
        "inspect_sample_schema",
        lambda _path: pytest.fail("supplied schema must prevent reinspection"),
    )
    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda _path: pytest.fail("supplied schema must prevent rehash"),
    )

    chunks = list(
        iter_sample_projection(path, columns=("x",), chunk_size=2, schema=schema)
    )
    assert [len(frame) for frame in chunks] == [2, 1]


def test_supplied_schema_for_another_path_is_rejected_without_hashing(tmp_path):
    path = tmp_path / "sample.csv"
    other = tmp_path / "other.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    other.write_text("x\n2\n", encoding="utf-8")
    schema = inspect_sample_schema(other)

    with pytest.raises(ValueError, match="does not match sample path"):
        list(iter_sample_projection(path, columns=("x",), chunk_size=1, schema=schema))


def test_supplied_schema_requires_plausible_hash_without_rehashing(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    schema = replace(inspect_sample_schema(path), sha256="not-a-hash")

    with pytest.raises(ValueError, match="invalid SHA-256"):
        list(iter_sample_projection(path, columns=("x",), chunk_size=1, schema=schema))


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_projection_rejects_invalid_chunk_size(tmp_path, chunk_size):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be positive"):
        list(iter_sample_projection(path, columns=("x",), chunk_size=chunk_size))


def test_projection_rejects_missing_columns(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    schema = inspect_sample_schema(path)
    with pytest.raises(ValueError, match="missing columns: y"):
        list(iter_sample_projection(path, columns=("y",), chunk_size=1, schema=schema))


def test_projection_rejects_duplicate_requested_columns(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    schema = inspect_sample_schema(path)
    with pytest.raises(ValueError, match="duplicate requested"):
        list(
            iter_sample_projection(
                path, columns=("x", "x"), chunk_size=1, schema=schema
            )
        )


def test_unsupported_format_is_rejected(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported validation sample format"):
        inspect_sample_schema(path)


def test_empty_sample_workbook_is_rejected(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(path, index=False, header=False)
    with pytest.raises(ValueError, match="没有非空 sheet"):
        inspect_sample_schema(path)


def test_sample_schema_argument_must_be_sample_schema(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(TypeError, match="SampleSchema"):
        list(
            iter_sample_projection(
                path, columns=("x",), chunk_size=1, schema="wrong"  # type: ignore[arg-type]
            )
        )
