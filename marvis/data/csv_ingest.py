from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marvis.data.errors import DataIngestError


# GAP-1: CSVs exported from Chinese bank data warehouses / legacy Excel are very
# often GBK/GB18030-encoded rather than UTF-8; a hardcoded utf-8-sig read throws
# an opaque UnicodeDecodeError with no mention of "encoding" anywhere in the
# message. Try the common encodings in order of specificity -- utf-8-sig first
# (handles a BOM if present and is a strict superset check for plain utf-8),
# then gb18030 (a strict superset of gbk/gb2312, so it covers both), then
# latin-1 as a last resort (never raises a UnicodeDecodeError -- every byte
# maps to a codepoint -- so it always succeeds and is used purely as a
# not-silently-crash fallback).
ENCODING_FALLBACK_CHAIN: tuple[str, ...] = ("utf-8-sig", "gb18030", "latin-1")

# A float64 mantissa has ~15-17 significant decimal digits; an id-like integer
# column at or above this many digits is where trailing digits start silently
# getting rewritten once pandas' default type inference promotes the column to
# float64 (which happens as soon as the column contains any missing value).
LONG_ID_DIGIT_THRESHOLD = 15


@dataclass(frozen=True)
class CsvIngestReport:
    encoding_used: str
    long_id_columns: tuple[str, ...]


def sniff_long_id_columns(path: Path, *, encoding: str, sample_rows: int = 2000) -> tuple[str, ...]:
    """Detect columns whose sampled values look like long (>=15 digit) integer ids.

    Reads a small text sample with dtype=str (so no precision is lost while
    sampling) and flags any column where a majority of non-null sampled values
    are purely-digit strings at or above LONG_ID_DIGIT_THRESHOLD length. These
    columns must be read back as strings by the real parse to avoid float64
    truncating their trailing digits (the "18-digit national ID becomes a
    float and loses precision" failure mode).
    """
    try:
        sample = pd.read_csv(
            path,
            encoding=encoding,
            dtype=str,
            nrows=sample_rows,
            keep_default_na=True,
        )
    except (UnicodeDecodeError, pd.errors.ParserError, csv.Error):
        return ()
    flagged: list[str] = []
    for column in sample.columns:
        values = sample[column].dropna()
        if values.empty:
            continue
        digit_like = values.str.fullmatch(r"\d+")
        long_digit_like = digit_like & (values.str.len() >= LONG_ID_DIGIT_THRESHOLD)
        if digit_like.sum() == 0:
            continue
        # Require the column to be predominantly long-digit-shaped (not just a
        # handful of values that happen to look numeric) before flagging it.
        if long_digit_like.sum() / max(int(digit_like.sum()), 1) >= 0.9 and long_digit_like.sum() > 0:
            flagged.append(str(column))
    return tuple(flagged)


def read_csv_with_fallback_encoding(
    path: Path,
    *,
    encodings: tuple[str, ...] = ENCODING_FALLBACK_CHAIN,
    **read_csv_kwargs,
) -> tuple[pd.DataFrame, CsvIngestReport]:
    """Read a CSV trying each encoding in turn, with long-id dtype protection.

    First determines a working encoding (trying each candidate in order), then
    samples that encoding to detect long numeric-id-shaped columns and reads
    those columns back as strings so pandas' default float64 promotion cannot
    truncate their trailing digits. Raises DataIngestError with all attempted
    encodings listed if every candidate fails.
    """
    path = Path(path)
    errors: list[str] = []
    for encoding in encodings:
        try:
            long_id_columns = sniff_long_id_columns(path, encoding=encoding)
            dtype_overrides = {column: str for column in long_id_columns} or None
            frame = pd.read_csv(path, encoding=encoding, dtype=dtype_overrides, **read_csv_kwargs)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        return frame, CsvIngestReport(encoding_used=encoding, long_id_columns=long_id_columns)
    raise DataIngestError(
        "无法解析 CSV 文件编码 (tried "
        + ", ".join(encodings)
        + f"): {'; '.join(errors)}"
    )


__all__ = [
    "ENCODING_FALLBACK_CHAIN",
    "LONG_ID_DIGIT_THRESHOLD",
    "CsvIngestReport",
    "read_csv_with_fallback_encoding",
    "sniff_long_id_columns",
]
