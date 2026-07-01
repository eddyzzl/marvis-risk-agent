"""Standalone feature-analysis Excel report writer (FEATURE form A)."""

from __future__ import annotations

import pytest
from openpyxl import Workbook
from openpyxl import load_workbook

from marvis.output.feature_report import render_feature_report


def test_render_feature_report_writes_metric_sheet(tmp_path):
    metrics = [
        {"feature": "x1", "iv": 0.42, "ks": 0.31, "auc": 0.71, "psi": 0.03, "missing_rate": 0.0, "lift_top_bin": 2.1},
        {"feature": "x2", "iv": 0.18, "ks": 0.20, "auc": 0.63, "psi": None, "missing_rate": 0.05, "lift_top_bin": 1.4},
    ]
    out = tmp_path / "feature_report.xlsx"

    render_feature_report(metrics, out)

    workbook = load_workbook(out)
    assert workbook.sheetnames == ["特征指标"]
    sheet = workbook["特征指标"]
    assert [cell.value for cell in sheet[1]] == ["特征", "IV", "KS", "AUC", "PSI", "缺失率", "头部lift"]
    assert sheet["A2"].value == "x1"
    assert sheet["A3"].value == "x2"
    # a missing metric renders as n/a, never silently blank
    assert sheet.cell(row=3, column=5).value == "n/a"


def test_render_feature_report_handles_empty_metrics(tmp_path):
    out = tmp_path / "empty.xlsx"
    render_feature_report([], out)
    sheet = load_workbook(out)["特征指标"]
    assert [cell.value for cell in sheet[1]] == ["特征", "IV", "KS", "AUC", "PSI", "缺失率", "头部lift"]
    assert sheet.max_row == 1  # header only


def test_render_feature_report_appends_optional_columns_only_when_present(tmp_path):
    """Head/tail lift + importance columns are written only when those keys ride in the
    metric rows (selected); a base-only report keeps its 7 columns."""
    out = tmp_path / "optional.xlsx"
    render_feature_report(
        [{
            "feature": "x1", "iv": 0.4, "ks": 0.3, "auc": 0.7, "psi": 0.02, "missing_rate": 0.0,
            "lift_top_bin": 2.0, "lift_head_5": 3.1, "lift_head_10": 2.6, "lift_tail_5": 0.2,
            "lift_tail_10": 0.4, "importance": 0.62,
        }],
        out,
    )
    header = [cell.value for cell in load_workbook(out)["特征指标"][1]]
    for col in ("头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%", "重要性"):
        assert col in header

    base_out = tmp_path / "base.xlsx"
    render_feature_report([{"feature": "x1", "iv": 0.4}], base_out)
    base_header = [cell.value for cell in load_workbook(base_out)["特征指标"][1]]
    assert "重要性" not in base_header and "头部lift5%" not in base_header


def test_render_feature_report_rolls_back_existing_file_when_save_fails(tmp_path, monkeypatch):
    out = tmp_path / "feature_report.xlsx"
    render_feature_report([{"feature": "old", "iv": 0.1}], out)
    original_bytes = out.read_bytes()
    original_save = Workbook.save

    def failing_save(self, filename):
        original_save(self, filename)
        raise RuntimeError("xlsx save failed")

    monkeypatch.setattr(Workbook, "save", failing_save)

    with pytest.raises(RuntimeError, match="xlsx save failed"):
        render_feature_report([{"feature": "new", "iv": 0.9}], out)

    assert out.read_bytes() == original_bytes
    assert not (tmp_path / ".staging").exists()
