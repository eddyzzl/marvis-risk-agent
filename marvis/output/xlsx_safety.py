"""Compatibility facade for the neutral spreadsheet safety boundary."""

from marvis.spreadsheet_safety import (
    EXCEL_CELL_TEXT_LIMIT,
    looks_like_excel_formula,
    safe_xlsx_cell,
    safe_xlsx_text,
)


__all__ = [
    "EXCEL_CELL_TEXT_LIMIT",
    "looks_like_excel_formula",
    "safe_xlsx_cell",
    "safe_xlsx_text",
]
