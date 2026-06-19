from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from marvis.data.errors import DataIngestError


MAX_HEADER_ROWS = 5
PREVIEW_ROWS = 25


@dataclass(frozen=True)
class IngestReport:
    sheet: str
    header_rows: int
    data_start_row: int
    flattened_columns: list[str]
    original_shape: tuple[int, int]


def list_sheets(path: Path) -> list[str]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def detect_header_rows(raw: pd.DataFrame) -> int:
    if raw.empty:
        raise DataIngestError("cannot detect headers for an empty sheet")
    limit = min(MAX_HEADER_ROWS, len(raw))
    for index in range(1, limit):
        if _looks_like_data_row(raw.iloc[index]):
            return max(index, 1)
    return 1


def flatten_headers(raw: pd.DataFrame, header_rows: int) -> tuple[pd.DataFrame, list[str]]:
    if raw.empty or raw.dropna(how="all").empty:
        raise DataIngestError("cannot flatten headers for an empty sheet")
    if header_rows < 1 or header_rows > len(raw):
        raise DataIngestError("header_rows is outside the sheet bounds")

    header_block = raw.iloc[:header_rows].copy()
    header_block = header_block.ffill(axis=1)
    flattened_columns = []
    for column_index in range(header_block.shape[1]):
        parts = [
            _header_part(header_block.iloc[row_index, column_index])
            for row_index in range(header_rows)
        ]
        parts = _dedupe_consecutive([part for part in parts if part])
        flattened_columns.append("_".join(parts) or f"col_{column_index}")
    flattened_columns = _disambiguate_duplicates(flattened_columns)

    data = raw.iloc[header_rows:].dropna(how="all").reset_index(drop=True)
    data.columns = flattened_columns
    return data, flattened_columns


def ingest_sheet(
    path: Path,
    sheet: str,
    out_dir: Path,
    *,
    header_rows: int | None = None,
) -> tuple[Path, IngestReport]:
    try:
        preview = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            nrows=PREVIEW_ROWS,
            engine="openpyxl",
        )
        full = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    except ValueError as exc:
        raise DataIngestError(f"cannot read sheet {sheet}: {exc}") from exc

    if full.empty or full.dropna(how="all").empty:
        raise DataIngestError(f"sheet is empty: {sheet}")
    resolved_header_rows = header_rows or detect_header_rows(preview)
    data, flattened_columns = flatten_headers(full, resolved_header_rows)
    if data.empty:
        raise DataIngestError(f"sheet has no data rows: {sheet}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_sheet_name(sheet)}.parquet"
    data.to_parquet(out_path, index=False)
    report = IngestReport(
        sheet=sheet,
        header_rows=resolved_header_rows,
        data_start_row=resolved_header_rows,
        flattened_columns=flattened_columns,
        original_shape=tuple(int(value) for value in full.shape),
    )
    return out_path, report


def _looks_like_data_row(row: pd.Series) -> bool:
    values = [value for value in row.tolist() if pd.notna(value) and str(value).strip()]
    if not values:
        return False
    typed = 0
    for value in values:
        if isinstance(value, pd.Timestamp):
            typed += 1
            continue
        if pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0]:
            typed += 1
            continue
        if pd.to_datetime(pd.Series([value]), errors="coerce").notna().iloc[0]:
            typed += 1
    return typed / len(values) >= 0.5


def _header_part(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower().startswith("unnamed:"):
        return ""
    return text


def _dedupe_consecutive(parts: list[str]) -> list[str]:
    deduped = []
    for part in parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return deduped


def _disambiguate_duplicates(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        result.append(name if count == 1 else f"{name}_{count}")
    return result


def _safe_sheet_name(sheet: str) -> str:
    cleaned = re.sub(r"[\\/:*?\\[\\]]+", "_", sheet).strip(" ._")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "sheet"


__all__ = [
    "MAX_HEADER_ROWS",
    "PREVIEW_ROWS",
    "IngestReport",
    "detect_header_rows",
    "flatten_headers",
    "ingest_sheet",
    "list_sheets",
]
