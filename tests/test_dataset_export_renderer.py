from __future__ import annotations

from marvis.agent.renderers import render_tool_output


def _output() -> dict:
    return {
        "schema_version": "dataset-export-tool-result.v1",
        "input_hash": "a" * 64,
        "dataset_id": "dataset-current",
        "dataset_content_hash": "b" * 64,
        "workspace_revision": 4,
        "analysis_generation": 2,
        "semantic_mapping_hash": "c" * 64,
        "format": "xlsx",
        "row_count": 1234,
        "column_count": 12,
        "size_bytes": 2097152,
        "content_hash": "d" * 64,
        "options": {
            "text_columns": ["customer_id", "mobile"],
            "formula_escape": "apostrophe_prefix",
            "large_integer_policy": "text_when_more_than_15_digits",
            "encoding": None,
            "workbook_mode": "write_only",
        },
        "safety": {
            "formula_cells_escaped": 3,
            "text_column_cells_written": 2468,
            "csv_text_cells_coerced": 0,
            "large_integer_cells_as_text": 12,
            "decimal_cells_as_text": 5,
            "high_precision_decimal_cells_as_text": 2,
            "non_finite_cells_as_text": 0,
            "xlsx_control_characters_escaped": 1,
        },
        "artifact_id": "artifact-export-1",
        "download_url": (
            "/api/tasks/task-1/task-artifacts/artifact-export-1/download"
        ),
        "cached": False,
    }


def test_export_renderer_surfaces_persisted_identity_download_and_safety_evidence():
    text, tables = render_tool_output("export_dataset", _output())

    assert "Excel" in text
    assert "1,234 行 / 12 列" in text
    assert "2.00 MB" in text
    by_title = {table["title"]: table for table in tables}
    assert {"导出文件", "安全处理"} <= set(by_title)

    file_rows = by_title["导出文件"]["rows"]
    assert ["数据集", "dataset-current"] in file_rows
    assert ["数据集 SHA-256", "b" * 64] in file_rows
    assert ["文件 SHA-256", "d" * 64] in file_rows
    assert ["按文本导出的字段", "customer_id、mobile"] in file_rows
    assert ["产物", "artifact-export-1"] in file_rows
    assert [
        "下载地址",
        "/api/tasks/task-1/task-artifacts/artifact-export-1/download",
    ] in file_rows

    safety_rows = by_title["安全处理"]["rows"]
    assert ["公式注入转义", "3"] in safety_rows
    assert ["文本字段单元格", "2,468"] in safety_rows
    assert ["超长整数按文本", "12"] in safety_rows
    assert ["高精度小数按文本", "2"] in safety_rows
    assert ["Excel 控制字符转义", "1"] in safety_rows


def test_export_renderer_labels_csv_and_cached_evidence_without_recalculation():
    output = _output()
    output["format"] = "csv"
    output["size_bytes"] = 321
    output["options"] = {**output["options"], "text_columns": []}
    output["cached"] = True

    text, tables = render_tool_output("export_dataset", output)

    assert "CSV" in text
    assert "321 B" in text
    assert "复用已验证产物" in text
    file_rows = next(table for table in tables if table["title"] == "导出文件")[
        "rows"
    ]
    assert ["按文本导出的字段", "无"] in file_rows
