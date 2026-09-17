from pathlib import Path

from openpyxl import load_workbook
import pytest

from marvis.output.validation_batch_excel import (
    OPS_SHEET_NAME,
    SUMMARY_SHEET_NAME,
    ValidationBatchSummaryRow,
    write_validation_batch_excel,
)


def _rows() -> list[ValidationBatchSummaryRow]:
    return [
        ValidationBatchSummaryRow(
            ordinal=1,
            item_id="item-pass",
            child_task_id="child-pass",
            model_name="分润复借T卡通用mob3模型",
            model_version="v1",
            oot_ks=0.339,
            oot_psi=0.03,
            pmml_status="pass",
            stress_risk="low",
            report_complete=True,
            outcome="pass",
            manual_review_url="",
            error_message="",
        ),
        ValidationBatchSummaryRow(
            ordinal=2,
            item_id="item-review",
            child_task_id="child-review",
            model_name="高风险机构多头识别模型",
            model_version="v2",
            oot_ks=0.54,
            oot_psi=0.12,
            pmml_status="pass",
            stress_risk="high",
            report_complete=True,
            outcome="manual_review",
            manual_review_url=(
                "http://127.0.0.1:8000/?task=parent&item=child-review"
            ),
            error_message="压力测试达到高风险阈值",
        ),
    ]


def test_batch_summary_excel_is_single_sheet_with_truthful_v2_gates(tmp_path: Path):
    output = tmp_path / "batch.xlsx"

    write_validation_batch_excel(
        batch_name="2026年8月模型验证批次",
        created_at="2026-08-01T09:00:00+08:00",
        rows=_rows(),
        output_path=output,
    )

    workbook = load_workbook(output)
    assert workbook.sheetnames == [SUMMARY_SHEET_NAME, OPS_SHEET_NAME]
    sheet = workbook[SUMMARY_SHEET_NAME]
    assert sheet["A1"].value == "模型验证：2026年8月验证2个模型"
    assert [cell.value for cell in sheet[3][:9]] == [
        "模型版本",
        "有效性 KS（OOT，仅展示不参与通过判定）",
        "稳定性 PSI（OOT vs train）",
        "PMML打分",
        "压力测试",
        "报告完整性",
        "处理结果",
        "人工校验",
        "失败/复核原因",
    ]
    assert sheet["B4"].value == pytest.approx(33.9)
    assert sheet["B4"].number_format == "0.0"
    assert sheet["D4"].value == "✓"
    assert sheet["E4"].value == "✓ 低风险"
    assert sheet["F4"].value == "✓"
    assert sheet["G4"].value == "✓ 通过"
    assert sheet["H4"].hyperlink is None


def test_batch_summary_links_non_pass_items_and_keeps_item_ids(tmp_path: Path):
    output = tmp_path / "batch.xlsx"
    write_validation_batch_excel(
        batch_name="批次",
        created_at="2026-08-01T09:00:00+08:00",
        rows=_rows(),
        output_path=output,
    )

    sheet = load_workbook(output)[SUMMARY_SHEET_NAME]
    assert sheet["C5"].fill.fgColor.rgb.endswith("FFEB9C")
    assert sheet["E5"].fill.fgColor.rgb.endswith("FFC7CE")
    assert sheet["G5"].value == "! 人工校验"
    assert sheet["H5"].value == "人工校验"
    assert sheet["H5"].hyperlink.target.endswith(
        "?task=parent&item=child-review"
    )
    assert sheet["J5"].value == "item-review"
    assert sheet["K5"].value == "child-review"
    assert sheet.column_dimensions["J"].hidden is True
    assert sheet.column_dimensions["K"].hidden is True

    values = [cell.value for row in sheet.iter_rows() for cell in row]
    assert any(isinstance(value, str) and "KS衰减范围：[0,10%)" in value for value in values)
    assert any(isinstance(value, str) and "PSI范围：[0.25,+∞)" in value for value in values)


def test_ops_sheet_matches_operator_workbook_layout(tmp_path: Path):
    output = tmp_path / "batch.xlsx"
    rows = [
        *_rows(),
        ValidationBatchSummaryRow(
            ordinal=3,
            item_id="item-fail",
            child_task_id="child-fail",
            model_name="失败模型",
            model_version="v1",
            oot_ks=None,
            oot_psi=None,
            pmml_status="fail",
            stress_risk=None,
            report_complete=False,
            outcome="failed",
            manual_review_url="",
            error_message="PMML 打分测试未通过",
        ),
    ]
    write_validation_batch_excel(
        batch_name="2026年8月模型验证批次",
        created_at="2026-08-01T09:00:00+08:00",
        rows=rows,
        output_path=output,
    )

    sheet = load_workbook(output)[OPS_SHEET_NAME]
    assert sheet["A1"].value == "模型验证：2026年8月验证3个模型"
    assert [cell.value for cell in sheet[2][:6]] == [
        "模型版本",
        "有效性 KS",
        "稳定性 PSI",
        "一致性",
        "可复现性",
        "完整性",
    ]
    assert sheet["A3"].value == "分润复借T卡通用mob3模型"
    assert sheet["B3"].value == pytest.approx(33.9)
    assert sheet["B3"].number_format == "0.0"
    assert sheet["C3"].value == pytest.approx(0.03)
    assert sheet["C3"].number_format == "0.00"
    assert sheet["D3"].value is None
    assert sheet["A5"].value == "失败模型"
    assert sheet["B5"].value is None
    images = sheet._images
    assert len(images) == 9
    anchors = [(image.anchor._from.col, image.anchor._from.row) for image in images]
    assert anchors[:3] == [(3, 2), (4, 2), (5, 2)]
    assert anchors[-3:] == [(3, 4), (4, 4), (5, 4)]
    sizes = [len(image._data()) for image in images]
    assert sizes.count(sizes[0]) == 6
    assert sizes[-1] != sizes[0]
