"""Safe, streaming exports of normalized Parquet datasets.

This module is deliberately a narrow execution kernel.  It reads immutable
Parquet in bounded Arrow record batches and writes either a UTF-8-BOM CSV or an
openpyxl write-only workbook.  It never constructs a pandas frame, never
silently truncates rows or columns, and publishes only a complete, hashed file
through an atomic sibling-path replace.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import Workbook

from marvis.data.errors import DataLayerError
from marvis.files import sha256_file


DATASET_EXPORT_RESULT_SCHEMA_VERSION = "dataset-export-result.v1"
_PRODUCER = {"name": "marvis.data.dataset_export", "version": "1"}
_EXCEL_MAX_ROWS = 1_048_576
_EXCEL_MAX_COLUMNS = 16_384
_EXCEL_MAX_DATA_ROWS = _EXCEL_MAX_ROWS - 1  # one row is reserved for headers
_FIXED_WORKBOOK_DATETIME = datetime(2000, 1, 1)
_FIXED_ZIP_DATETIME = (1980, 1, 1, 0, 0, 0)
_FORMULA_PREFIXES = frozenset("=+-@")
_XLSX_ILLEGAL_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXCEL_NUMERIC_TEXT = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?%?$"
)
_EXCEL_DATE_OR_TIME_TEXT = re.compile(
    r"^(?:"
    r"\d{1,4}[-/.]\d{1,2}(?:[-/.]\d{1,4})?"
    r"|\d{2,4}年\d{1,2}月(?:\d{1,2}日?)?"
    r"|\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[AP]M)?"
    r")$",
    re.IGNORECASE,
)
_HARD_CONFIG_MAXIMUMS = {
    "max_rows": 100_000_000,
    "max_columns": _EXCEL_MAX_COLUMNS,
    "batch_size": 65_536,
    "max_batch_cells": 10_000_000,
    "max_input_bytes": 100 * 1024**3,
    "max_output_bytes": 100 * 1024**3,
    "max_cell_characters": 32_767,
}


class DatasetExportInputError(DataLayerError):
    """The requested dataset export is invalid or unsafe."""


class DatasetExportExecutionError(DataLayerError):
    """A validated export could not be completed deterministically."""


class DatasetExportConfigError(ValueError):
    """An export limit is invalid or exceeds a hard safety ceiling."""


class DatasetExportBudgetError(DataLayerError):
    """The source or rendered output exceeds an explicit export budget."""

    def __init__(self, *, dimension: str, actual: int, limit: int) -> None:
        self.dimension = str(dimension)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(
            f"dataset export {self.dimension} budget exceeded: "
            f"actual={self.actual}, limit={self.limit}"
        )

    def to_detail(self) -> dict[str, object]:
        return {
            "kind": "dataset_export_budget_exceeded",
            "dimension": self.dimension,
            "actual": self.actual,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class DatasetExportConfig:
    """Explicit, hard-bounded streaming and artifact budgets.

    ``max_rows`` is format-independent.  XLSX additionally enforces Excel's
    fixed 1,048,576 total-row limit, including the header.  ``max_batch_cells``
    keeps wide datasets from turning a nominally small Arrow batch into a large
    in-memory Python object during cell conversion.
    """

    max_rows: int = 5_000_000
    max_columns: int = _EXCEL_MAX_COLUMNS
    batch_size: int = 4_096
    max_batch_cells: int = 1_000_000
    max_input_bytes: int = 20 * 1024**3
    max_output_bytes: int = 20 * 1024**3
    max_cell_characters: int = 32_767

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DatasetExportConfigError(
                    f"{name} must be an explicit positive integer"
                )
            maximum = _HARD_CONFIG_MAXIMUMS[name]
            if value > maximum:
                raise DatasetExportConfigError(f"{name} must be at most {maximum}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _SafetyCounts:
    formula_cells_escaped: int = 0
    text_column_cells_written: int = 0
    csv_text_cells_coerced: int = 0
    large_integer_cells_as_text: int = 0
    decimal_cells_as_text: int = 0
    high_precision_decimal_cells_as_text: int = 0
    non_finite_cells_as_text: int = 0
    xlsx_control_characters_escaped: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def export_dataset(
    input_path: Path,
    output_path: Path,
    *,
    format: str = "csv",
    temp_directory: Path,
    text_columns: Sequence[str] = (),
    config: DatasetExportConfig | None = None,
) -> dict[str, object]:
    """Stream normalized Parquet to a governed CSV or XLSX artifact.

    The destination must not exist.  A complete export is written to a unique
    sibling scratch file, checked against output budgets, hashed, validated as
    finite JSON evidence, and only then atomically moved into place.  Every
    failure path removes the scratch file.
    """

    source = Path(input_path)
    output = Path(output_path)
    temp_root = Path(temp_directory)
    effective_config = config or DatasetExportConfig()
    if not isinstance(effective_config, DatasetExportConfig):
        raise DatasetExportConfigError("config must be a DatasetExportConfig")
    export_format = _validate_paths_and_format(
        source,
        output,
        requested_format=format,
        config=effective_config,
    )

    try:
        parquet_file = pq.ParquetFile(source)
        schema = parquet_file.schema_arrow
        row_count = int(parquet_file.metadata.num_rows)
    except (OSError, pa.ArrowException) as exc:
        raise DatasetExportInputError(
            "input is not a readable normalized Parquet dataset"
        ) from exc

    columns = _validate_schema(schema)
    column_count = len(columns)
    _enforce_budget("rows", row_count, effective_config.max_rows)
    _enforce_budget("columns", column_count, effective_config.max_columns)
    if export_format == "xlsx":
        _enforce_budget("xlsx_rows", row_count, _EXCEL_MAX_DATA_ROWS)
        _enforce_budget("xlsx_columns", column_count, _EXCEL_MAX_COLUMNS)
    selected_text_columns = _validate_text_columns(text_columns, columns)

    # Batches remain bounded even for very wide schemas.  One row must always
    # fit because the schema-width budget has already been checked.
    batch_size = max(
        1,
        min(
            effective_config.batch_size,
            effective_config.max_batch_cells // column_count,
        ),
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    safety = _SafetyCounts()

    try:
        if export_format == "csv":
            written_rows = _write_csv(
                parquet_file,
                scratch,
                columns=columns,
                text_columns=selected_text_columns,
                batch_size=batch_size,
                safety=safety,
            )
        else:
            written_rows = _write_xlsx(
                parquet_file,
                scratch,
                columns=columns,
                text_columns=selected_text_columns,
                batch_size=batch_size,
                max_cell_characters=effective_config.max_cell_characters,
                safety=safety,
            )
        if written_rows != row_count:
            raise DatasetExportExecutionError(
                "Parquet row count changed while the export was being read"
            )

        size_bytes = scratch.stat().st_size
        _enforce_budget(
            "output_bytes", size_bytes, effective_config.max_output_bytes
        )
        content_hash = sha256_file(scratch)
        column_evidence = [
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "text_mode": field.name in selected_text_columns,
            }
            for field in columns
        ]
        result: dict[str, object] = {
            "schema_version": DATASET_EXPORT_RESULT_SCHEMA_VERSION,
            "producer": dict(_PRODUCER),
            "source": {
                "format": "parquet",
                "size_bytes": source.stat().st_size,
                "row_count": row_count,
                "column_count": column_count,
                "columns": column_evidence,
            },
            "output": {
                "path": str(output),
                "format": export_format,
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "hash_algorithm": "sha256",
                "row_count": row_count,
                "column_count": column_count,
                "columns": column_evidence,
            },
            "options": {
                "encoding": "utf-8-sig" if export_format == "csv" else None,
                "workbook_mode": "write_only" if export_format == "xlsx" else None,
                "sheet_name": "data" if export_format == "xlsx" else None,
                "null_representation": "empty_cell",
                "formula_escape": "apostrophe_prefix",
                "large_integer_policy": "text_when_more_than_15_digits",
                "decimal_policy": "exact_text_for_xlsx_and_precision_risky_csv",
                "text_columns": list(selected_text_columns),
                "effective_batch_size": batch_size,
                "config": effective_config.to_dict(),
            },
            "safety": safety.to_dict(),
        }
        _assert_json_safe(result)
        if _path_exists_including_broken_symlink(output):
            raise DatasetExportInputError(f"output path already exists: {output}")
        os.replace(scratch, output)
        return result
    except Exception:
        scratch.unlink(missing_ok=True)
        raise


def _validate_paths_and_format(
    source: Path,
    output: Path,
    *,
    requested_format: str,
    config: DatasetExportConfig,
) -> str:
    if source.suffix.lower() != ".parquet":
        raise DatasetExportInputError("export input must be normalized Parquet")
    if not source.exists() or not source.is_file():
        raise DatasetExportInputError(f"input Parquet does not exist: {source}")
    source_size = source.stat().st_size
    _enforce_budget("input_bytes", source_size, config.max_input_bytes)
    if not isinstance(requested_format, str) or requested_format not in {"csv", "xlsx"}:
        raise DatasetExportInputError("format must be csv or xlsx")
    expected_suffix = f".{requested_format}"
    if output.suffix.lower() != expected_suffix:
        raise DatasetExportInputError(
            f"output suffix must be {expected_suffix} for format={requested_format}"
        )
    try:
        same_path = source.resolve(strict=True) == output.resolve(strict=False)
    except OSError:
        same_path = source.absolute() == output.absolute()
    if same_path:
        raise DatasetExportInputError("output path must be different from input path")
    if _path_exists_including_broken_symlink(output):
        raise DatasetExportInputError(f"output path already exists: {output}")
    return requested_format


def _validate_schema(schema: pa.Schema) -> list[pa.Field]:
    columns = list(schema)
    if not columns:
        raise DatasetExportInputError("input Parquet must contain at least one column")
    names = [field.name for field in columns]
    if len(set(names)) != len(names):
        raise DatasetExportInputError("input Parquet has duplicate column names")
    for field in columns:
        if not _is_supported_type(field.type):
            raise DatasetExportInputError(
                f"unsupported Parquet column type for {field.name}: {field.type}"
            )
    return columns


def _is_supported_type(data_type: pa.DataType) -> bool:
    if pa.types.is_dictionary(data_type):
        return _is_supported_type(data_type.value_type)
    return any(
        predicate(data_type)
        for predicate in (
            pa.types.is_null,
            pa.types.is_boolean,
            pa.types.is_integer,
            pa.types.is_floating,
            pa.types.is_decimal,
            pa.types.is_string,
            pa.types.is_large_string,
            pa.types.is_date,
            pa.types.is_timestamp,
            pa.types.is_time,
            pa.types.is_duration,
        )
    )


def _validate_text_columns(
    value: Sequence[str],
    columns: list[pa.Field],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetExportInputError("text_columns must be an ordered sequence")
    names = {field.name for field in columns}
    selected: list[str] = []
    seen: set[str] = set()
    for raw_name in value:
        if not isinstance(raw_name, str) or not raw_name:
            raise DatasetExportInputError("text column names must be non-empty strings")
        if raw_name not in names:
            raise DatasetExportInputError(f"unknown text column: {raw_name}")
        if raw_name in seen:
            raise DatasetExportInputError(f"duplicate text column: {raw_name}")
        selected.append(raw_name)
        seen.add(raw_name)
    return tuple(selected)


def _write_csv(
    parquet_file: pq.ParquetFile,
    scratch: Path,
    *,
    columns: list[pa.Field],
    text_columns: tuple[str, ...],
    batch_size: int,
    safety: _SafetyCounts,
) -> int:
    text_set = set(text_columns)
    with scratch.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([_safe_string(field.name, safety=safety) for field in columns])
        written_rows = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=False):
            for row in batch.to_pylist():
                writer.writerow(
                    [
                        _csv_cell(
                            row[field.name],
                            force_text=field.name in text_set,
                            safety=safety,
                        )
                        for field in columns
                    ]
                )
                written_rows += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written_rows


def _write_xlsx(
    parquet_file: pq.ParquetFile,
    scratch: Path,
    *,
    columns: list[pa.Field],
    text_columns: tuple[str, ...],
    batch_size: int,
    max_cell_characters: int,
    safety: _SafetyCounts,
) -> int:
    text_set = set(text_columns)
    workbook = Workbook(write_only=True)
    workbook.properties.created = _FIXED_WORKBOOK_DATETIME
    workbook.properties.modified = _FIXED_WORKBOOK_DATETIME
    sheet = workbook.create_sheet("data")
    try:
        sheet.append(
            [
                _xlsx_string(
                    _safe_string(field.name, safety=safety),
                    max_characters=max_cell_characters,
                    safety=safety,
                )
                for field in columns
            ]
        )
        written_rows = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=False):
            for row in batch.to_pylist():
                sheet.append(
                    [
                        _xlsx_cell(
                            row[field.name],
                            force_text=field.name in text_set,
                            max_characters=max_cell_characters,
                            safety=safety,
                        )
                        for field in columns
                    ]
                )
                written_rows += 1
        workbook.save(scratch)
        _canonicalize_xlsx_archive(scratch)
        return written_rows
    finally:
        workbook.close()


def _csv_cell(value: Any, *, force_text: bool, safety: _SafetyCounts) -> str:
    if value is None:
        return ""
    if force_text:
        safety.text_column_cells_written += 1
    precision_marker_required = False
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int):
        text = str(value)
        if _integer_has_more_than_15_digits(value):
            safety.large_integer_cells_as_text += 1
            precision_marker_required = True
    elif isinstance(value, float):
        text = _finite_float_text(value, safety=safety)
    elif isinstance(value, Decimal):
        text = format(value, "f")
        if _decimal_exceeds_excel_precision(value):
            safety.high_precision_decimal_cells_as_text += 1
            precision_marker_required = True
    elif isinstance(value, (datetime, date, time)):
        text = value.isoformat()
    else:
        text = str(value)
    if precision_marker_required:
        text = _prefix_text_marker(text)
    # Only user-controlled strings can be spreadsheet formulas.  Native
    # numeric scalars such as -9999 and -1.5 must remain numeric text in CSV;
    # treating their minus sign as a formula prefix corrupts the export.
    if isinstance(value, str):
        text = _safe_string(text, safety=safety)
    if force_text and _looks_like_excel_auto_coercion(_stable_text(value)):
        safety.csv_text_cells_coerced += 1
        text = _prefix_text_marker(text)
    return text


def _canonicalize_xlsx_archive(path: Path) -> None:
    """Remove ZIP-container clock variance from an openpyxl workbook.

    Fixing core.xml's created/modified values is necessary but insufficient:
    ``zipfile`` also stamps every OOXML member with the wall clock.  Repacking
    members in name order with fixed metadata makes equal inputs byte-identical
    across processes and seconds, so the reported content hash is reproducible.
    """

    canonical = path.with_name(f".{path.name}.{uuid.uuid4().hex}.canonical")
    try:
        with ZipFile(path, "r") as source_archive:
            source_members = sorted(
                source_archive.infolist(), key=lambda member: member.filename
            )
            with ZipFile(
                canonical,
                "w",
                compression=ZIP_DEFLATED,
                compresslevel=9,
                allowZip64=True,
            ) as output_archive:
                for source_member in source_members:
                    payload = source_archive.read(source_member.filename)
                    member = ZipInfo(
                        filename=source_member.filename,
                        date_time=_FIXED_ZIP_DATETIME,
                    )
                    member.compress_type = ZIP_DEFLATED
                    member.create_system = 0
                    member.external_attr = 0
                    member.internal_attr = 0
                    member.comment = b""
                    member.extra = b""
                    output_archive.writestr(
                        member,
                        payload,
                        compress_type=ZIP_DEFLATED,
                        compresslevel=9,
                    )
        os.replace(canonical, path)
    finally:
        canonical.unlink(missing_ok=True)


def _xlsx_cell(
    value: Any,
    *,
    force_text: bool,
    max_characters: int,
    safety: _SafetyCounts,
) -> Any:
    if value is None:
        return None
    if force_text:
        safety.text_column_cells_written += 1
        if isinstance(value, int) and not isinstance(value, bool):
            if _integer_has_more_than_15_digits(value):
                safety.large_integer_cells_as_text += 1
        elif isinstance(value, Decimal):
            safety.decimal_cells_as_text += 1
            if _decimal_exceeds_excel_precision(value):
                safety.high_precision_decimal_cells_as_text += 1
        elif isinstance(value, float) and not math.isfinite(value):
            safety.non_finite_cells_as_text += 1
        return _xlsx_string(
            _safe_string(_stable_text(value), safety=safety),
            max_characters=max_characters,
            safety=safety,
        )
    if isinstance(value, str):
        return _xlsx_string(
            _safe_string(value, safety=safety),
            max_characters=max_characters,
            safety=safety,
        )
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if _integer_has_more_than_15_digits(value):
            safety.large_integer_cells_as_text += 1
            return str(value)
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return _finite_float_text(value, safety=safety)
    if isinstance(value, Decimal):
        safety.decimal_cells_as_text += 1
        if _decimal_exceeds_excel_precision(value):
            safety.high_precision_decimal_cells_as_text += 1
        return format(value, "f")
    if isinstance(value, datetime) and value.tzinfo is not None:
        return _xlsx_string(
            value.isoformat(),
            max_characters=max_characters,
            safety=safety,
        )
    if isinstance(value, (datetime, date, time)):
        return value
    return _xlsx_string(
        _safe_string(str(value), safety=safety),
        max_characters=max_characters,
        safety=safety,
    )


def _stable_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


def _finite_float_text(value: float, *, safety: _SafetyCounts) -> str:
    if math.isnan(value):
        safety.non_finite_cells_as_text += 1
        return "NaN"
    if math.isinf(value):
        safety.non_finite_cells_as_text += 1
        return "Infinity" if value > 0 else "-Infinity"
    return str(value)


def _safe_string(value: str, *, safety: _SafetyCounts) -> str:
    if _looks_like_formula(value):
        safety.formula_cells_escaped += 1
        return _prefix_text_marker(value)
    return value


def _looks_like_formula(value: str) -> bool:
    for character in value:
        if character.isspace() or unicodedata.category(character).startswith("C"):
            continue
        return character in _FORMULA_PREFIXES
    return False


def _looks_like_excel_auto_coercion(value: str) -> bool:
    candidate = _strip_excel_ignored_edges(value)
    if not candidate:
        return False
    return bool(
        _EXCEL_NUMERIC_TEXT.fullmatch(candidate)
        or _EXCEL_DATE_OR_TIME_TEXT.fullmatch(candidate)
        or candidate.upper() in {"TRUE", "FALSE"}
    )


def _strip_excel_ignored_edges(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and (
        value[start].isspace()
        or unicodedata.category(value[start]).startswith("C")
    ):
        start += 1
    while end > start and (
        value[end - 1].isspace()
        or unicodedata.category(value[end - 1]).startswith("C")
    ):
        end -= 1
    return value[start:end]


def _prefix_text_marker(value: str) -> str:
    return value if value.startswith("'") else "'" + value


def _xlsx_string(
    value: str,
    *,
    max_characters: int,
    safety: _SafetyCounts,
) -> str:
    if len(value) > max_characters:
        raise DatasetExportBudgetError(
            dimension="xlsx_cell_characters",
            actual=len(value),
            limit=max_characters,
        )

    def replace_control(match: re.Match[str]) -> str:
        safety.xlsx_control_characters_escaped += 1
        return f"\\u{ord(match.group(0)):04x}"

    return _XLSX_ILLEGAL_CONTROL.sub(replace_control, value)


def _integer_has_more_than_15_digits(value: int) -> bool:
    return len(str(abs(value))) > 15


def _decimal_exceeds_excel_precision(value: Decimal) -> bool:
    normalized_digits = value.as_tuple().digits
    first_nonzero = next(
        (index for index, digit in enumerate(normalized_digits) if digit != 0),
        len(normalized_digits) - 1,
    )
    return len(normalized_digits[first_nonzero:]) > 15


def _enforce_budget(dimension: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise DatasetExportBudgetError(
            dimension=dimension,
            actual=actual,
            limit=limit,
        )


def _path_exists_including_broken_symlink(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_json_safe(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise DatasetExportExecutionError(
            "dataset export evidence is not finite JSON"
        ) from exc


__all__ = [
    "DATASET_EXPORT_RESULT_SCHEMA_VERSION",
    "DatasetExportBudgetError",
    "DatasetExportConfig",
    "DatasetExportConfigError",
    "DatasetExportExecutionError",
    "DatasetExportInputError",
    "export_dataset",
]
