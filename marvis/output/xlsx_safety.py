"""Shared safety boundary for text written to generated XLSX workbooks."""

from __future__ import annotations

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


EXCEL_CELL_TEXT_LIMIT = 32_767
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def safe_xlsx_text(value: object) -> str:
    """Return bounded plain text that Excel cannot interpret as a formula."""

    text = ILLEGAL_CHARACTERS_RE.sub(" ", str(value))
    if text.lstrip().startswith(_FORMULA_PREFIXES):
        return "'" + text[: EXCEL_CELL_TEXT_LIMIT - 1]
    return text[:EXCEL_CELL_TEXT_LIMIT]


def safe_xlsx_cell(value: object):
    """Preserve scalar types while hardening every text-like value."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is not value:
            return safe_xlsx_cell(converted)
    return safe_xlsx_text(value)


__all__ = ["EXCEL_CELL_TEXT_LIMIT", "safe_xlsx_cell", "safe_xlsx_text"]
