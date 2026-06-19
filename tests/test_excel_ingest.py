from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from marvis.data.errors import DataIngestError
from marvis.data.excel_ingest import (
    detect_header_rows,
    flatten_headers,
    ingest_sheet,
    list_sheets,
)


def test_ingest_sheet_handles_single_header_and_lists_sheets(tmp_path):
    workbook_path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame({"id": [1, 2], "score": [0.1, 0.2]}).to_excel(
            writer,
            sheet_name="Main",
            index=False,
        )
        pd.DataFrame({"id": [3], "score": [0.3]}).to_excel(
            writer,
            sheet_name="Feature",
            index=False,
        )

    out_path, report = ingest_sheet(workbook_path, "Main", tmp_path / "out")

    assert list_sheets(workbook_path) == ["Main", "Feature"]
    assert out_path == tmp_path / "out" / "Main.parquet"
    assert out_path.exists()
    assert report.sheet == "Main"
    assert report.header_rows == 1
    assert report.data_start_row == 1
    assert report.flattened_columns == ["id", "score"]
    assert report.original_shape == (3, 2)
    assert pd.read_parquet(out_path).to_dict("list") == {
        "id": [1, 2],
        "score": [0.1, 0.2],
    }


def test_ingest_sheet_flattens_merged_two_row_headers(tmp_path):
    workbook_path = tmp_path / "merged.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Merged"
    sheet.merge_cells("A1:B1")
    sheet["A1"] = "Customer"
    sheet["C1"] = "Outcome"
    sheet.append(["ID", "Phone", "Target"])
    sheet.append(["A1", "13800138000", 1])
    sheet.append(["B2", "13900139000", 0])
    workbook.save(workbook_path)

    out_path, report = ingest_sheet(workbook_path, "Merged", tmp_path / "out")

    assert report.header_rows == 2
    assert report.flattened_columns == [
        "Customer_ID",
        "Customer_Phone",
        "Outcome_Target",
    ]
    joined = pd.read_parquet(out_path)
    assert joined.columns.tolist() == report.flattened_columns
    assert joined["Customer_ID"].tolist() == ["A1", "B2"]


def test_header_detection_and_duplicate_disambiguation():
    raw = pd.DataFrame([
        ["id", "id", "group"],
        [1, 2, "A"],
        [3, 4, "B"],
    ])

    assert detect_header_rows(raw) == 1
    data, columns = flatten_headers(raw, 1)

    assert columns == ["id", "id_2", "group"]
    assert data.columns.tolist() == columns
    assert data.to_dict("records")[0] == {"id": 1, "id_2": 2, "group": "A"}


def test_ingest_sheet_rejects_empty_sheet(tmp_path):
    workbook_path = tmp_path / "empty.xlsx"
    workbook = Workbook()
    workbook.active.title = "Empty"
    workbook.save(workbook_path)

    with pytest.raises(DataIngestError):
        ingest_sheet(workbook_path, "Empty", tmp_path / "out")


def test_ingest_sheet_wraps_missing_sheet_errors(tmp_path):
    workbook_path = tmp_path / "book.xlsx"
    pd.DataFrame({"id": [1]}).to_excel(workbook_path, sheet_name="Present", index=False)

    with pytest.raises(DataIngestError):
        ingest_sheet(Path(workbook_path), "Missing", tmp_path / "out")
