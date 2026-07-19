from __future__ import annotations

from marvis.agent.dataset_export import (
    build_dataset_export_request,
    detect_dataset_export_intent,
)


_COLUMNS = ("customer_id", "mobile", "amount", "bad")
_BUSINESS_NAMES = {
    "customer_id": "客户编号",
    "mobile": "手机号",
    "amount": "申请金额",
    "bad": "坏样本标签",
}


def _build(text: str):
    return build_dataset_export_request(
        text,
        columns=_COLUMNS,
        business_names=_BUSINESS_NAMES,
    )


def test_export_intent_requires_dataset_scope_and_export_action():
    assert detect_dataset_export_intent("把当前数据导出为 CSV")
    assert detect_dataset_export_intent("下载这份样本为 Excel")
    assert detect_dataset_export_intent("export current dataset as xlsx")

    assert not detect_dataset_export_intent("导出策略报告为 Excel")
    assert not detect_dataset_export_intent("基于当前数据导出策略报告为 Excel")
    assert not detect_dataset_export_intent("下载规则报告")
    assert not detect_dataset_export_intent("把策略规则导出为 CSV")
    assert not detect_dataset_export_intent("当前样本的规则报告下载为 CSV")
    assert not detect_dataset_export_intent("把当前数据的分析结果导出为Excel")
    assert not detect_dataset_export_intent("把当前数据的缺失值分析导出为Excel")
    assert not detect_dataset_export_intent("导出当前数据的数据字典为Excel")
    assert not detect_dataset_export_intent("把当前样本的统计概览下载为CSV")
    assert detect_dataset_export_intent("导出当前数据")
    assert detect_dataset_export_intent("下载这份样本")
    assert detect_dataset_export_intent("export current dataset")
    assert detect_dataset_export_intent("当前数据同时导出为 CSV 和 Excel")
    assert detect_dataset_export_intent("将当前数据导出为 JSON")
    assert not detect_dataset_export_intent("下载 CSV 模板")
    assert not detect_dataset_export_intent(None)
    assert detect_dataset_export_intent("导出当前数据为Excel用于后续报告")
    assert detect_dataset_export_intent("导出当前数据用于后续报告")


def test_formatless_non_data_exports_remain_excluded():
    assert not detect_dataset_export_intent("导出当前数据的分析结果")
    assert not detect_dataset_export_intent("导出当前数据的数据字典")
    assert not detect_dataset_export_intent("下载当前样本的统计概览")
    assert not detect_dataset_export_intent("基于当前数据导出策略报告")


def test_builds_closed_csv_and_xlsx_requests():
    csv_result = _build("把当前数据集导出为 CSV")
    xlsx_result = _build("下载当前样本为 Excel")

    assert csv_result.clarification is None
    assert csv_result.request is not None
    assert csv_result.request.format == "csv"
    assert csv_result.request.text_columns == ()
    assert xlsx_result.request is not None
    assert xlsx_result.request.format == "xlsx"
    assert xlsx_result.request.text_columns == ()


def test_resolves_original_and_business_field_names_for_text_export():
    result = _build(
        "将当前数据导出为 Excel，客户编号、mobile 这些列按文本导出"
    )

    assert result.clarification is None
    assert result.request is not None
    assert result.request.format == "xlsx"
    assert result.request.text_columns == ("customer_id", "mobile")

    compact = _build("当前数据导出为Excel 客户编号和手机号这些列按文本导出")
    english = _build(
        "export current dataset as csv, customer_id and mobile as text"
    )
    assert compact.request is not None
    assert compact.request.text_columns == ("customer_id", "mobile")
    assert english.request is not None
    assert english.request.text_columns == ("customer_id", "mobile")

    scoped = _build("将当前数据中的客户编号和手机号按文本导出为Excel")
    assert scoped.request is not None
    assert scoped.request.format == "xlsx"
    assert scoped.request.text_columns == ("customer_id", "mobile")


def test_missing_unknown_or_conflicting_format_requires_clarification():
    missing = _build("导出当前数据")
    unknown = _build("将当前数据导出为 JSON")
    conflicting = _build("当前数据同时导出为 CSV 和 Excel")

    assert missing.request is None
    assert "CSV" in (missing.clarification or "")
    assert "Excel" in (missing.clarification or "")
    assert unknown.request is None
    assert "CSV" in (unknown.clarification or "")
    assert "Excel" in (unknown.clarification or "")
    assert conflicting.request is None
    assert "一种" in (conflicting.clarification or "")


def test_unknown_or_ambiguous_text_column_fails_closed():
    unknown = _build("当前数据导出为 CSV，证件号按文本导出")
    ambiguous = build_dataset_export_request(
        "当前数据导出为 CSV，客户号按文本导出",
        columns=("customer_id", "legacy_customer_id"),
        business_names={
            "customer_id": "客户号",
            "legacy_customer_id": "客户号",
        },
    )

    assert unknown.request is None
    assert "证件号" in (unknown.clarification or "")
    assert ambiguous.request is None
    assert "原始字段名" in (ambiguous.clarification or "")


def test_build_rejects_non_dataset_export_scope_even_when_called_directly():
    result = _build("基于当前数据导出策略报告为 Excel")

    assert result.request is None
    assert "当前数据" in (result.clarification or "")

    analysis_result = _build("把当前数据的缺失值分析导出为Excel")
    assert analysis_result.request is None
    assert "分析" in (analysis_result.clarification or "")
