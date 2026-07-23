"""Deterministic projections of a canonical StrategyReportBundle V2.

This module is intentionally a renderer, not an analysis layer.  It validates
the self-authenticating bundle and projects the exact same facts to canonical
JSON, Markdown, and a formula-free XLSX workbook.  Missing values remain blank;
their typed availability and reason stay adjacent to the blank presentation
cell and in the evidence index.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from io import BytesIO
import json
import re
import unicodedata
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.report_bundle import (
    REPORT_CORE_SHEET_KEYS,
    canonical_strategy_report_bundle_json,
    validate_strategy_report_bundle,
)


STRATEGY_REPORT_OUTPUT_SCHEMA_VERSION = "strategy.report-output.v2"
STRATEGY_REPORT_SHEET_TITLES = {
    "00_summary": "结论与变更摘要",
    "01_current_state": "项目现状",
    "02_history": "历史策略",
    "03_sample": "样本与口径",
    "04_univariate_model": "单变量与模型",
    "05_candidates": "候选组合",
    "06_strategy": "最终策略明细",
    "07_waterfall_swap": "Waterfall 与 Swap",
    "08_impact": "逐月与分群影响",
    "09_economics": "收益与数据成本",
    "10_validation": "验证与稳定性",
    "11_evidence": "证据与版本",
}

_SECTION_PRIMARY_SHEETS = {
    "current_project": "01_current_state",
    "historical_versions": "02_history",
    "sample_design": "03_sample",
    "univariate_and_models": "04_univariate_model",
    "candidate_combinations": "05_candidates",
    "impact_assessment": "08_impact",
    "final_document": "06_strategy",
}
_SHEET_SECTION_KEYS = {
    "01_current_state": ("current_project",),
    "02_history": ("historical_versions",),
    "03_sample": ("sample_design",),
    "04_univariate_model": ("univariate_and_models",),
    "05_candidates": ("candidate_combinations",),
    "06_strategy": ("candidate_combinations", "final_document"),
    "07_waterfall_swap": ("impact_assessment",),
    "08_impact": ("impact_assessment",),
    "09_economics": ("impact_assessment",),
    "10_validation": ("impact_assessment", "final_document"),
}
_AVAILABILITY_LABELS = {
    "present": "已有可信值",
    "unavailable": "暂缺",
    "not_applicable": "本次不涉及",
    "not_matured": "尚未成熟",
}
_EFFECT_STAGE_LABELS = {
    "estimated": "预估",
    "backtested": "开发回测",
    "oot_validated": "OOT",
    "post_launch_observed": "上线后观察",
}
_REPORT_STATUS_LABELS = {
    "draft": "草稿",
    "partial": "部分完成",
    "final": "当前允许范围内已完成",
}

_FIXED_WORKBOOK_DATETIME = datetime(2000, 1, 1)
_FIXED_ZIP_DATETIME = (1980, 1, 1, 0, 0, 0)
_FORMULA_PREFIXES = frozenset("=+-@")
_ILLEGAL_PRESENTATION_CONTROL = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff\ufffe\uffff]"
)
_MARKDOWN_INLINE_META = re.compile(r"([`*_{}\[\]()!])")
_CORE_MODIFIED_TIMESTAMP = re.compile(
    rb"(<dcterms:modified\b[^>]*>)[^<]*(</dcterms:modified>)"
)
_MAX_XLSX_CELL_CHARACTERS = 32_767
_MAX_XLSX_ROWS = 1_048_576
_MAX_XLSX_COLUMNS = 16_384

_NAVY = "18324A"
_BLUE = "235A7A"
_PALE_BLUE = "EAF2F8"
_PALE_AMBER = "FFF3CD"
_PALE_RED = "F8D7DA"
_PALE_GREEN = "DDEFE3"
_WHITE = "FFFFFF"
_GRID = Side(style="thin", color="D9E1E8")
_BORDER = Border(left=_GRID, right=_GRID, top=_GRID, bottom=_GRID)


class StrategyReportOutputError(StrategyError):
    """A valid report bundle cannot be safely projected to an output format."""


def render_strategy_report_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, bytes]:
    """Render canonical JSON, Markdown, and deterministic XLSX bytes."""

    canonical = validate_strategy_report_bundle(bundle)
    return {
        "json": canonical_strategy_report_bundle_json(canonical).encode("utf-8"),
        "markdown": _render_markdown(canonical).encode("utf-8"),
        "xlsx": _render_xlsx(canonical),
    }


def render_strategy_report_bundle_json(bundle: Mapping[str, Any]) -> bytes:
    """Return the canonical machine-readable report manifest."""

    return canonical_strategy_report_bundle_json(bundle).encode("utf-8")


def render_strategy_report_bundle_markdown(bundle: Mapping[str, Any]) -> bytes:
    """Return the deterministic Markdown executive report."""

    return _render_markdown(validate_strategy_report_bundle(bundle)).encode("utf-8")


def render_strategy_report_bundle_xlsx(bundle: Mapping[str, Any]) -> bytes:
    """Return a deterministic, formula-free multi-sheet workbook."""

    return _render_xlsx(validate_strategy_report_bundle(bundle))


def _render_markdown(bundle: Mapping[str, Any]) -> str:
    title = _field_display_value(bundle["title"])
    lines = [
        f"# {_markdown_cell(title)}",
        "",
        "> 本报告是受治理结构化证据的投影。空白不等于 0，效果阶段不自动升级。",
        "",
        "| 报告属性 | 值 |",
        "|---|---|",
        f"| 报告 ID | {_markdown_cell(bundle['report_id'])} |",
        f"| 修订 | {bundle['report_revision']} |",
        f"| 状态 | {_markdown_cell(_REPORT_STATUS_LABELS[bundle['status']])} |",
        f"| 策略 ID | {_markdown_cell(bundle['strategy_id'] or '')} |",
        f"| 策略版本 | {_markdown_cell(bundle['strategy_version'] or '')} |",
        f"| 策略类型 | {_markdown_cell(bundle['strategy_type'] or '')} |",
        f"| 效果阶段 | {_markdown_cell(_stage_labels(bundle['effect_stages']))} |",
        f"| 生成时间 | {_markdown_cell(bundle['generated_at'])} |",
        f"| 数据分级 | {_markdown_cell(bundle['data_classification'])} |",
        f"| 内容 SHA-256 | {_markdown_cell(bundle['content_sha256'])} |",
        "",
    ]
    for index, section in enumerate(bundle["sections"], start=1):
        lines.extend(_markdown_section(index, section))

    lines.extend(
        [
            "## 证据、缺失信息与完整度",
            "",
            "### 完整度",
            "",
            "| 类别 | 状态 | 数量 |",
            "|---|---|---:|",
        ]
    )
    completeness = bundle["completeness_summary"]
    for key, count in completeness["field_counts"].items():
        lines.append(f"| 字段 | {_markdown_cell(key)} | {count} |")
    for key, count in completeness["missing_information_counts"].items():
        lines.append(f"| 缺失信息 | {_markdown_cell(key)} | {count} |")
    for key, count in completeness["blocking_counts"].items():
        lines.append(f"| 阻塞 | {_markdown_cell(key)} | {count} |")

    lines.extend(
        [
            "",
            "### 缺失信息",
            "",
            "| 字段 | 阻塞级别 | 状态 | 原因 | 已询问次数 |",
            "|---|---|---|---|---:|",
        ]
    )
    if bundle["missing_information"]:
        for item in bundle["missing_information"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(item["field_path"]),
                        _markdown_cell(item["blocking"]),
                        _markdown_cell(item["status"]),
                        _markdown_cell(item["reason"]),
                        str(item["asked_count"]),
                    )
                )
                + " |"
            )
    else:
        lines.append("|  |  |  |  | 0 |")

    lines.extend(
        [
            "",
            "### 来源索引",
            "",
            "| 位置 | 类型 | 引用 ID | 内容 SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for location, ref in _source_locations(bundle):
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_cell(location),
                    _markdown_cell(ref["kind"]),
                    _markdown_cell(ref["ref_id"]),
                    _markdown_cell(ref["content_hash"]),
                )
            )
            + " |"
        )

    # A final newline gives stable CLI/file rendering and is part of the
    # deterministic projection contract.
    return "\n".join(lines).rstrip() + "\n"


def _markdown_section(index: int, section: Mapping[str, Any]) -> list[str]:
    lines = [
        f"## {index}. {_markdown_cell(section['title'])}",
        "",
        f"- 状态：{_markdown_cell(_AVAILABILITY_LABELS[section['availability']])}",
    ]
    stages = sorted(
        {item["effect_stage"] for item in section["stage_evidence"]},
        key=("estimated", "backtested", "oot_validated", "post_launch_observed").index,
    )
    if stages:
        lines.append(f"- 效果阶段：{_markdown_cell(_stage_labels(stages))}")
    lines.append("")
    if section["summary_fields"]:
        lines.extend(
            [
                "| 字段 | 值 | 状态 | 说明 | 来源 |",
                "|---|---|---|---|---|",
            ]
        )
        for item in section["summary_fields"]:
            field = item["field"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(item["label"]),
                        _markdown_cell(_field_display_value(field)),
                        _markdown_cell(_AVAILABILITY_LABELS[field["availability"]]),
                        _markdown_cell(field["note"] or ""),
                        _markdown_cell(_compact_refs(field["source_refs"])),
                    )
                )
                + " |"
            )
        lines.append("")
    for table in section["tables"]:
        lines.extend(_markdown_table(table))
    if section["red_flags"]:
        lines.extend(
            [
                "### 红旗",
                "",
                "| 级别 | 编码 | 说明 |",
                "|---|---|---|",
            ]
        )
        for flag in section["red_flags"]:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _markdown_cell(flag["level"]),
                        _markdown_cell(flag["code"]),
                        _markdown_cell(flag["message"]),
                    )
                )
                + " |"
            )
        lines.append("")
    if (
        not section["summary_fields"]
        and not section["tables"]
        and not section["red_flags"]
    ):
        lines.extend(["该部分当前没有可展示的结构化事实。", ""])
    return lines


def _markdown_table(table: Mapping[str, Any]) -> list[str]:
    stage = (
        ""
        if table["effect_stage"] is None
        else f"（{_EFFECT_STAGE_LABELS[table['effect_stage']]}）"
    )
    headers: list[str] = []
    separators: list[str] = []
    for column in table["columns"]:
        label = column["label"]
        if column["unit"] is not None:
            label = f"{label} [{column['unit']}]"
        headers.extend((label, f"{label}状态", f"{label}说明"))
        separators.extend(("---", "---", "---"))
    lines = [
        f"### {_markdown_cell(table['title'])}{stage}",
        "",
        "| " + " | ".join(_markdown_cell(item) for item in headers) + " |",
        "|" + "|".join(separators) + "|",
    ]
    for row in table["rows"]:
        values: list[str] = []
        for column in table["columns"]:
            field = row["cells"][column["key"]]
            values.extend(
                (
                    _markdown_cell(
                        _table_field_display_value(field, column=column)
                    ),
                    _markdown_cell(_AVAILABILITY_LABELS[field["availability"]]),
                    _markdown_cell(field["note"] or ""),
                )
            )
        lines.append("| " + " | ".join(values) + " |")
    if not table["rows"]:
        lines.append("| " + " | ".join("" for _ in headers) + " |")
    lines.extend(
        [
            "",
            f"来源：{_markdown_cell(_compact_refs(table['source_refs']))}",
            "",
        ]
    )
    return lines


def _render_xlsx(bundle: Mapping[str, Any]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.creator = "MARVIS"
    workbook.properties.lastModifiedBy = "MARVIS"
    workbook.properties.title = str(
        _safe_text_projection(_field_display_value(bundle["title"]))
    )
    workbook.properties.description = STRATEGY_REPORT_OUTPUT_SCHEMA_VERSION
    workbook.properties.created = _FIXED_WORKBOOK_DATETIME
    workbook.properties.modified = _FIXED_WORKBOOK_DATETIME

    sections = {section["key"]: section for section in bundle["sections"]}
    tables_by_sheet: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for section in bundle["sections"]:
        for table in section["tables"]:
            tables_by_sheet.setdefault(table["sheet_key"], []).append(
                (section["key"], table)
            )
    appendix_keys = sorted(
        key for key in tables_by_sheet if key not in REPORT_CORE_SHEET_KEYS
    )
    try:
        for sheet_key in REPORT_CORE_SHEET_KEYS:
            sheet = workbook.create_sheet(sheet_key)
            if sheet_key == "00_summary":
                _write_summary_sheet(
                    sheet,
                    bundle=bundle,
                    tables=tables_by_sheet.get(sheet_key, ()),
                )
            elif sheet_key == "11_evidence":
                _write_evidence_sheet(sheet, bundle=bundle)
            else:
                _write_business_sheet(
                    sheet,
                    sheet_key=sheet_key,
                    sections=sections,
                    tables=tables_by_sheet.get(sheet_key, ()),
                )
        for sheet_key in appendix_keys:
            sheet = workbook.create_sheet(sheet_key)
            _write_appendix_sheet(
                sheet,
                sheet_key=sheet_key,
                tables=tables_by_sheet[sheet_key],
                sections=sections,
            )
        raw = BytesIO()
        workbook.save(raw)
    finally:
        workbook.close()
    return _canonicalize_xlsx_bytes(raw.getvalue())


def _write_summary_sheet(
    sheet: Any,
    *,
    bundle: Mapping[str, Any],
    tables: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    _prepare_sheet(sheet)
    row = _write_sheet_title(
        sheet,
        title=STRATEGY_REPORT_SHEET_TITLES["00_summary"],
        subtitle="受治理证据投影；空白不等于 0，效果阶段不自动升级",
    )
    row = _write_key_values(
        sheet,
        start_row=row,
        title="报告身份与状态",
        rows=(
            ("报告标题", _field_display_value(bundle["title"])),
            ("报告 ID", bundle["report_id"]),
            ("修订", bundle["report_revision"]),
            ("前一修订", bundle["previous_report_id"]),
            ("状态", _REPORT_STATUS_LABELS[bundle["status"]]),
            ("策略 ID", bundle["strategy_id"]),
            ("策略版本", bundle["strategy_version"]),
            ("策略类型", bundle["strategy_type"]),
            ("效果阶段", _stage_labels(bundle["effect_stages"])),
            ("生成时间", bundle["generated_at"]),
            ("数据分级", bundle["data_classification"]),
            ("Producer", bundle["producer_version"]),
            ("内容 SHA-256", bundle["content_sha256"]),
        ),
    )
    row = _write_section_heading(sheet, row, "七步完成状态")
    _write_header_row(sheet, row, ("步骤", "模块", "状态", "红旗数", "证据数"))
    row += 1
    for index, section in enumerate(bundle["sections"], start=1):
        _write_row(
            sheet,
            row,
            (
                index,
                section["title"],
                _AVAILABILITY_LABELS[section["availability"]],
                len(section["red_flags"]),
                len(section["source_refs"]),
            ),
        )
        row += 1
    for section_key in (
        "current_project",
        "candidate_combinations",
        "impact_assessment",
        "final_document",
    ):
        row = _write_summary_fields(
            sheet,
            start_row=row,
            section=next(
                item
                for item in bundle["sections"]
                if item["key"] == section_key
            ),
        )
    row += 1
    row = _write_section_heading(sheet, row, "完整度")
    _write_header_row(sheet, row, ("类别", "状态", "数量"))
    row += 1
    completeness = bundle["completeness_summary"]
    for category, counts in (
        ("字段", completeness["field_counts"]),
        ("缺失信息", completeness["missing_information_counts"]),
        ("阻塞", completeness["blocking_counts"]),
    ):
        for key, count in counts.items():
            _write_row(sheet, row, (category, key, count))
            row += 1
    row += 1
    row = _write_red_flags(
        sheet,
        start_row=row,
        sections=bundle["sections"],
    )
    row = _write_routed_tables(
        sheet,
        start_row=row,
        tables=tables,
    )
    _guard_sheet_bounds(sheet)
    _set_widths(sheet, {1: 24, 2: 70, 3: 24, 4: 18, 5: 18})


def _write_business_sheet(
    sheet: Any,
    *,
    sheet_key: str,
    sections: Mapping[str, Mapping[str, Any]],
    tables: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    _prepare_sheet(sheet)
    title = STRATEGY_REPORT_SHEET_TITLES[sheet_key]
    row = _write_sheet_title(
        sheet,
        title=title,
        subtitle=f"Sheet key: {sheet_key}",
    )
    relevant = [sections[key] for key in _SHEET_SECTION_KEYS[sheet_key]]
    row = _write_section_states(sheet, start_row=row, sections=relevant)
    for section in relevant:
        if _SECTION_PRIMARY_SHEETS[section["key"]] == sheet_key:
            row = _write_summary_fields(sheet, start_row=row, section=section)
            row = _write_red_flags(sheet, start_row=row, sections=(section,))
    row = _write_routed_tables(sheet, start_row=row, tables=tables)
    if not tables and all(
        _SECTION_PRIMARY_SHEETS[section["key"]] != sheet_key
        for section in relevant
    ):
        _write_row(sheet, row, ("当前没有路由到本 Sheet 的结构化表。",))
    _guard_sheet_bounds(sheet)
    _set_widths(sheet, {1: 24, 2: 40, 3: 18, 4: 42, 5: 54})


def _write_appendix_sheet(
    sheet: Any,
    *,
    sheet_key: str,
    tables: Sequence[tuple[str, Mapping[str, Any]]],
    sections: Mapping[str, Mapping[str, Any]],
) -> None:
    _prepare_sheet(sheet)
    row = _write_sheet_title(
        sheet,
        title=f"分析附件：{sheet_key}",
        subtitle="附件表仍引用同一受治理证据，不重新计算指标",
    )
    section_keys = tuple(dict.fromkeys(section_key for section_key, _ in tables))
    row = _write_section_states(
        sheet,
        start_row=row,
        sections=[sections[key] for key in section_keys],
    )
    _write_routed_tables(sheet, start_row=row, tables=tables)
    _guard_sheet_bounds(sheet)
    _set_widths(sheet, {1: 24, 2: 40, 3: 18, 4: 42, 5: 54})


def _write_evidence_sheet(sheet: Any, *, bundle: Mapping[str, Any]) -> None:
    _prepare_sheet(sheet)
    row = _write_sheet_title(
        sheet,
        title=STRATEGY_REPORT_SHEET_TITLES["11_evidence"],
        subtitle="完整 hash、缺失信息状态和字段级 lineage",
    )
    row = _write_section_heading(sheet, row, "缺失信息")
    _write_header_row(
        sheet,
        row,
        (
            "Missing ID",
            "字段",
            "阻塞级别",
            "状态",
            "原因",
            "问题",
            "已询问",
            "询问时间",
            "回答时间",
            "依赖 SHA-256",
            "回答来源",
        ),
    )
    row += 1
    for item in bundle["missing_information"]:
        _write_row(
            sheet,
            row,
            (
                item["missing_information_id"],
                item["field_path"],
                item["blocking"],
                item["status"],
                item["reason"],
                item["question"],
                item["asked_count"],
                item["asked_at"],
                item["answered_at"],
                item["dependency_hash"],
                _compact_refs(
                    []
                    if item["answer_source_ref"] is None
                    else [item["answer_source_ref"]]
                ),
            ),
        )
        row += 1
    if not bundle["missing_information"]:
        _write_row(sheet, row, ("", "", "", "", "", "", 0))
        row += 1

    row += 1
    row = _write_section_heading(sheet, row, "来源索引")
    _write_header_row(
        sheet,
        row,
        ("位置", "类型", "引用 ID", "内容 SHA-256"),
    )
    row += 1
    for location, ref in _source_locations(bundle):
        _write_row(
            sheet,
            row,
            (location, ref["kind"], ref["ref_id"], ref["content_hash"]),
        )
        row += 1

    row += 1
    row = _write_section_heading(sheet, row, "完整度")
    _write_header_row(sheet, row, ("类别", "状态", "数量"))
    row += 1
    completeness = bundle["completeness_summary"]
    for category, counts in (
        ("字段", completeness["field_counts"]),
        ("Section", completeness["section_counts"]),
        ("缺失信息", completeness["missing_information_counts"]),
        ("阻塞", completeness["blocking_counts"]),
    ):
        for key, count in counts.items():
            _write_row(sheet, row, (category, key, count))
            row += 1

    _guard_sheet_bounds(sheet)
    _set_widths(
        sheet,
        {
            1: 52,
            2: 32,
            3: 22,
            4: 20,
            5: 70,
            6: 70,
            7: 14,
            8: 28,
            9: 28,
            10: 68,
            11: 60,
        },
    )


def _write_section_states(
    sheet: Any,
    *,
    start_row: int,
    sections: Sequence[Mapping[str, Any]],
) -> int:
    row = _write_section_heading(sheet, start_row, "模块状态")
    _write_header_row(
        sheet,
        row,
        ("模块", "状态", "效果阶段", "红旗数", "来源"),
    )
    row += 1
    for section in sections:
        stages = sorted(
            {item["effect_stage"] for item in section["stage_evidence"]},
            key=(
                "estimated",
                "backtested",
                "oot_validated",
                "post_launch_observed",
            ).index,
        )
        _write_row(
            sheet,
            row,
            (
                section["title"],
                _AVAILABILITY_LABELS[section["availability"]],
                _stage_labels(stages),
                len(section["red_flags"]),
                _compact_refs(section["source_refs"]),
            ),
        )
        row += 1
    return row + 1


def _write_summary_fields(
    sheet: Any,
    *,
    start_row: int,
    section: Mapping[str, Any],
) -> int:
    if not section["summary_fields"]:
        return start_row
    row = _write_section_heading(sheet, start_row, f"{section['title']}摘要")
    _write_header_row(
        sheet,
        row,
        (
            "字段 ID",
            "字段",
            "值",
            "状态",
            "来源类型",
            "阻塞级别",
            "统计时点",
            "说明",
            "来源引用",
        ),
    )
    row += 1
    for item in section["summary_fields"]:
        field = item["field"]
        _write_row(
            sheet,
            row,
            (
                item["field_id"],
                item["label"],
                _field_display_value(field),
                _AVAILABILITY_LABELS[field["availability"]],
                field["origin"],
                field["blocking"],
                field["as_of"],
                field["note"],
                _compact_refs(field["source_refs"]),
            ),
        )
        _style_availability_cell(sheet.cell(row=row, column=4), field["availability"])
        row += 1
    return row + 1


def _write_red_flags(
    sheet: Any,
    *,
    start_row: int,
    sections: Sequence[Mapping[str, Any]],
) -> int:
    rows = [
        (section["title"], flag)
        for section in sections
        for flag in section["red_flags"]
    ]
    if not rows:
        return start_row
    row = _write_section_heading(sheet, start_row, "红旗与限制")
    _write_header_row(sheet, row, ("模块", "级别", "编码", "说明", "来源"))
    row += 1
    for section_title, flag in rows:
        _write_row(
            sheet,
            row,
            (
                section_title,
                flag["level"],
                flag["code"],
                flag["message"],
                _compact_refs(flag["source_refs"]),
            ),
        )
        _style_flag_cell(sheet.cell(row=row, column=2), flag["level"])
        row += 1
    return row + 1


def _write_routed_tables(
    sheet: Any,
    *,
    start_row: int,
    tables: Sequence[tuple[str, Mapping[str, Any]]],
) -> int:
    row = start_row
    for section_key, table in tables:
        row = _write_report_table(
            sheet,
            start_row=row,
            section_key=section_key,
            table=table,
        )
    return row


def _write_report_table(
    sheet: Any,
    *,
    start_row: int,
    section_key: str,
    table: Mapping[str, Any],
) -> int:
    expanded_columns = len(table["columns"]) * 4
    if expanded_columns > _MAX_XLSX_COLUMNS:
        raise StrategyReportOutputError(
            f"table {table['table_id']} exceeds Excel's column limit"
        )
    row = _write_section_heading(
        sheet,
        start_row,
        table["title"],
    )
    _write_row(
        sheet,
        row,
        (
            "Table ID",
            table["table_id"],
            "模块",
            section_key,
            "效果阶段",
            ""
            if table["effect_stage"] is None
            else _EFFECT_STAGE_LABELS[table["effect_stage"]],
            "粒度",
            table["granularity"],
            "内容类型",
            table["content_class"],
            "来源",
            _compact_refs(table["source_refs"]),
        ),
    )
    row += 1
    headers: list[str] = []
    for column in table["columns"]:
        label = column["label"]
        if column["unit"] is not None:
            label = f"{label} [{column['unit']}]"
        headers.extend((label, f"{label}状态", f"{label}说明", f"{label}来源"))
    _write_header_row(sheet, row, headers)
    row += 1
    for item in table["rows"]:
        values: list[object] = []
        value_columns: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]] = []
        for index, column in enumerate(table["columns"]):
            field = item["cells"][column["key"]]
            values.extend(
                (
                    _field_display_value(field),
                    _AVAILABILITY_LABELS[field["availability"]],
                    field["note"],
                    _compact_refs(field["source_refs"]),
                )
            )
            value_columns.append((index * 4 + 1, field, column))
        _write_row(sheet, row, values)
        for column_index, field, column in value_columns:
            _apply_number_format(
                sheet.cell(row=row, column=column_index),
                field=field,
                column=column,
            )
            _style_availability_cell(
                sheet.cell(row=row, column=column_index + 1),
                field["availability"],
            )
        row += 1
    if not table["rows"]:
        _write_row(sheet, row, tuple("" for _ in headers))
        row += 1
    return row + 2


def _write_sheet_title(sheet: Any, *, title: str, subtitle: str) -> int:
    sheet.cell(row=1, column=1, value=_xlsx_cell(title))
    sheet.cell(row=1, column=1).font = Font(
        name="Arial",
        size=18,
        bold=True,
        color=_WHITE,
    )
    sheet.cell(row=1, column=1).fill = PatternFill("solid", fgColor=_NAVY)
    sheet.cell(row=1, column=1).alignment = Alignment(vertical="center")
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=9)
    sheet.row_dimensions[1].height = 30
    sheet.cell(row=2, column=1, value=_xlsx_cell(subtitle))
    sheet.cell(row=2, column=1).font = Font(
        name="Arial",
        size=10,
        italic=True,
        color="536878",
    )
    sheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=9)
    return 4


def _write_key_values(
    sheet: Any,
    *,
    start_row: int,
    title: str,
    rows: Iterable[tuple[object, object]],
) -> int:
    row = _write_section_heading(sheet, start_row, title)
    _write_header_row(sheet, row, ("属性", "值"))
    row += 1
    for item in rows:
        _write_row(sheet, row, item)
        row += 1
    return row + 1


def _write_section_heading(sheet: Any, row: int, title: object) -> int:
    sheet.cell(row=row, column=1, value=_xlsx_cell(title))
    sheet.cell(row=row, column=1).font = Font(
        name="Arial",
        size=12,
        bold=True,
        color=_WHITE,
    )
    sheet.cell(row=row, column=1).fill = PatternFill("solid", fgColor=_BLUE)
    sheet.cell(row=row, column=1).alignment = Alignment(vertical="center")
    sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=9)
    return row + 1


def _write_header_row(sheet: Any, row: int, values: Sequence[object]) -> None:
    _write_row(sheet, row, values)
    for cell in sheet[row][: len(values)]:
        cell.font = Font(name="Arial", size=10, bold=True, color=_NAVY)
        cell.fill = PatternFill("solid", fgColor=_PALE_BLUE)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )


def _write_row(sheet: Any, row: int, values: Sequence[object]) -> None:
    if row > _MAX_XLSX_ROWS:
        raise StrategyReportOutputError("report exceeds Excel's row limit")
    if len(values) > _MAX_XLSX_COLUMNS:
        raise StrategyReportOutputError("report exceeds Excel's column limit")
    for column, value in enumerate(values, start=1):
        cell = sheet.cell(row=row, column=column, value=_xlsx_cell(value))
        cell.font = Font(name="Arial", size=10, color="1F2933")
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.border = _BORDER


def _prepare_sheet(sheet: Any) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A4"
    sheet.auto_filter.ref = None
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.outlinePr.summaryBelow = True


def _set_widths(sheet: Any, widths: Mapping[int, float]) -> None:
    for column, width in widths.items():
        sheet.column_dimensions[get_column_letter(column)].width = width
    for column in range(1, sheet.max_column + 1):
        letter = get_column_letter(column)
        if sheet.column_dimensions[letter].width == 13.0:
            sheet.column_dimensions[letter].width = 20


def _guard_sheet_bounds(sheet: Any) -> None:
    if sheet.max_row > _MAX_XLSX_ROWS:
        raise StrategyReportOutputError(
            f"sheet {sheet.title} exceeds Excel's row limit"
        )
    if sheet.max_column > _MAX_XLSX_COLUMNS:
        raise StrategyReportOutputError(
            f"sheet {sheet.title} exceeds Excel's column limit"
        )


def _field_display_value(field: Mapping[str, Any]) -> object:
    availability = field["availability"]
    if availability in {"unavailable", "not_matured"}:
        return ""
    if availability == "not_applicable":
        return "本次不涉及"
    return _presentation_value(field["value"])


def _table_field_display_value(
    field: Mapping[str, Any],
    *,
    column: Mapping[str, Any],
) -> object:
    value = _field_display_value(field)
    if field["availability"] != "present":
        return value
    raw = field["value"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return value
    precision = column["precision"]
    if precision is None:
        return value
    if column["unit"] == "%":
        return f"{raw * 100:.{precision}f}%"
    return f"{raw:.{precision}f}"


def _presentation_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _source_locations(
    bundle: Mapping[str, Any],
) -> Iterable[tuple[str, Mapping[str, str]]]:
    for ref in bundle["title"]["source_refs"]:
        yield "report.title", ref
    for field_name in (
        "dataset_refs",
        "strategy_artifact_refs",
        "tool_run_refs",
    ):
        for ref in bundle[field_name]:
            yield field_name, ref
    for section in bundle["sections"]:
        prefix = f"sections.{section['key']}"
        for ref in section["source_refs"]:
            yield prefix, ref
        for item in section["summary_fields"]:
            for ref in item["field"]["source_refs"]:
                yield f"{prefix}.summary_fields.{item['field_id']}", ref
        for table in section["tables"]:
            table_prefix = f"{prefix}.tables.{table['table_id']}"
            for ref in table["source_refs"]:
                yield table_prefix, ref
            for row in table["rows"]:
                for key, field in row["cells"].items():
                    for ref in field["source_refs"]:
                        yield f"{table_prefix}.rows.{row['row_id']}.{key}", ref
        for index, item in enumerate(section["stage_evidence"]):
            binding = item["binding"]
            for key, ref in binding.items():
                if key in {"kind", "effective_period"}:
                    continue
                yield f"{prefix}.stage_evidence[{index}].{key}", ref
        for flag in section["red_flags"]:
            for ref in flag["source_refs"]:
                yield f"{prefix}.red_flags.{flag['code']}", ref
    for item in bundle["missing_information"]:
        if item["answer_source_ref"] is not None:
            yield (
                f"missing_information.{item['missing_information_id']}.answer",
                item["answer_source_ref"],
            )


def _compact_refs(refs: Sequence[Mapping[str, str]]) -> str:
    return "; ".join(
        f"{item['kind']}:{item['ref_id']}@{item['content_hash']}"
        for item in refs
    )


def _stage_labels(stages: Sequence[str]) -> str:
    return " / ".join(_EFFECT_STAGE_LABELS[item] for item in stages)


def _apply_number_format(
    cell: Any,
    *,
    field: Mapping[str, Any],
    column: Mapping[str, Any],
) -> None:
    if field["availability"] != "present":
        return
    value = field["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    precision = column["precision"]
    if precision is None:
        return
    decimals = "" if precision == 0 else "." + ("0" * precision)
    if column["unit"] == "%":
        cell.number_format = f"0{decimals}%"
    else:
        cell.number_format = f"#,##0{decimals}"


def _style_availability_cell(cell: Any, availability: str) -> None:
    fill = {
        "present": _PALE_GREEN,
        "unavailable": _PALE_AMBER,
        "not_applicable": _PALE_BLUE,
        "not_matured": _PALE_AMBER,
    }[availability]
    cell.fill = PatternFill("solid", fgColor=fill)


def _style_flag_cell(cell: Any, level: str) -> None:
    fill = {
        "info": _PALE_BLUE,
        "amber": _PALE_AMBER,
        "red": _PALE_RED,
    }[level]
    cell.fill = PatternFill("solid", fgColor=fill)


def _xlsx_cell(value: object) -> object:
    value = _presentation_value(value)
    if value is None or isinstance(value, (bool, float)):
        return value
    if isinstance(value, int):
        # Excel has only 15 significant decimal digits.  Preserve longer
        # identifiers and counts as text instead of silently rounding them.
        return str(value) if len(str(abs(value))) > 15 else value
    if not isinstance(value, str):
        raise StrategyReportOutputError(
            f"report cell contains unsupported {type(value).__name__}"
        )
    text = _safe_text_projection(value)
    if _looks_like_formula(text):
        text = "'" + text
    if len(text) > _MAX_XLSX_CELL_CHARACTERS:
        raise StrategyReportOutputError(
            "report cell exceeds Excel's 32767 character limit"
        )
    return text


def _looks_like_formula(value: str) -> bool:
    for character in value:
        if character.isspace() or unicodedata.category(character).startswith("C"):
            continue
        return character in _FORMULA_PREFIXES
    return False


def _markdown_cell(value: object) -> str:
    text = _safe_text_projection(
        str(_presentation_value(value) if value is not None else "")
    )
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = _MARKDOWN_INLINE_META.sub(r"\\\1", escaped)
    return (
        escaped
        .replace("|", "\\|")
        .replace("\r\n", " / ")
        .replace("\n", " / ")
        .replace("\r", " / ")
    )


def _safe_text_projection(value: object) -> str:
    return _ILLEGAL_PRESENTATION_CONTROL.sub(
        lambda match: f"\\u{ord(match.group(0)):04x}",
        str(value),
    )


def _canonicalize_xlsx_bytes(raw: bytes) -> bytes:
    source = BytesIO(raw)
    destination = BytesIO()
    with ZipFile(source, "r") as input_archive:
        members = sorted(input_archive.infolist(), key=lambda item: item.filename)
        with ZipFile(
            destination,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as output_archive:
            for source_member in members:
                member = ZipInfo(
                    source_member.filename,
                    date_time=_FIXED_ZIP_DATETIME,
                )
                member.compress_type = ZIP_DEFLATED
                member.create_system = 0
                member.external_attr = 0
                member.internal_attr = 0
                member.comment = b""
                member.extra = b""
                payload = input_archive.read(source_member.filename)
                if source_member.filename == "docProps/core.xml":
                    payload, replacements = _CORE_MODIFIED_TIMESTAMP.subn(
                        rb"\g<1>2000-01-01T00:00:00Z\g<2>",
                        payload,
                    )
                    if replacements != 1:
                        raise StrategyReportOutputError(
                            "generated workbook has an invalid modified "
                            "timestamp contract"
                        )
                output_archive.writestr(
                    member,
                    payload,
                    compress_type=ZIP_DEFLATED,
                    compresslevel=9,
                )
    return destination.getvalue()


__all__ = [
    "STRATEGY_REPORT_OUTPUT_SCHEMA_VERSION",
    "STRATEGY_REPORT_SHEET_TITLES",
    "StrategyReportOutputError",
    "render_strategy_report_bundle",
    "render_strategy_report_bundle_json",
    "render_strategy_report_bundle_markdown",
    "render_strategy_report_bundle_xlsx",
]
