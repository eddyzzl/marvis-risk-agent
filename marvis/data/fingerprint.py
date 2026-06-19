from __future__ import annotations

import re

import pandas as pd

from marvis.data.contracts import (
    DATE_FORMATS,
    HASH_ALGO_CANDIDATES,
    HASH_HEX_LENGTHS,
    ColumnFingerprint,
)


HASH_PATTERN = r"^[0-9a-fA-F]+$"
IDCARD_PATTERN = r"^\d{17}[\dXx]$"
PHONE_PATTERN = r"^1\d{10}$"


def fingerprint_column(
    series: pd.Series,
    *,
    sample_n: int = 1000,
    seed: int = 0,
) -> ColumnFingerprint:
    non_null = series.dropna()
    if non_null.empty:
        return _unknown_fingerprint()
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return ColumnFingerprint("date", None, None, False, None, None, "datetime")

    values = non_null.astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return _unknown_fingerprint()
    sample = _sample_values(values, sample_n=sample_n, seed=seed)
    length_mode = int(sample.str.len().mode().iloc[0])

    if _frac_match(sample, HASH_PATTERN) > 0.9 and length_mode in HASH_HEX_LENGTHS:
        return ColumnFingerprint(
            value_kind="hash",
            length_mode=length_mode,
            regex_pattern=rf"^[0-9a-fA-F]{{{length_mode}}}$",
            is_hashed=True,
            hash_type=HASH_HEX_LENGTHS[length_mode],
            hex_case=_detect_hex_case(sample),
            date_format=None,
        )
    if _frac_match(sample, IDCARD_PATTERN) > 0.9:
        return ColumnFingerprint(
            "raw_idcard",
            18,
            IDCARD_PATTERN,
            False,
            None,
            None,
            None,
        )
    if _frac_match(sample, PHONE_PATTERN) > 0.9:
        return ColumnFingerprint(
            "raw_phone",
            11,
            PHONE_PATTERN,
            False,
            None,
            None,
            None,
        )

    date_format = _detect_date_format(sample)
    if date_format is not None:
        return ColumnFingerprint("date", None, None, False, None, None, date_format)
    if _frac_numeric(sample) > 0.9:
        return ColumnFingerprint("numeric", None, None, False, None, None, None)
    return ColumnFingerprint("categorical", length_mode, None, False, None, None, None)


def candidate_match_methods(a: ColumnFingerprint, b: ColumnFingerprint) -> list[str]:
    if a.value_kind == b.value_kind and a.value_kind in {"raw_phone", "raw_idcard"}:
        return ["exact", "exact_lower"]
    if a.value_kind == "hash" and b.value_kind == "hash":
        return ["exact_lower"] if a.length_mode == b.length_mode else []

    kinds = {a.value_kind, b.value_kind}
    if "hash" in kinds and ("raw_phone" in kinds or "raw_idcard" in kinds):
        known = a.hash_type or b.hash_type
        ordered = ([known] if known else []) + [
            algorithm
            for algorithm in HASH_ALGO_CANDIDATES
            if algorithm != known
        ]
        return [f"hash:{algorithm}" for algorithm in ordered if algorithm]

    if a.value_kind == "date" and b.value_kind == "date":
        return ["date"]
    if a.value_kind == b.value_kind:
        return ["exact", "exact_lower"]
    return []


def _unknown_fingerprint() -> ColumnFingerprint:
    return ColumnFingerprint("unknown", None, None, False, None, None, None)


def _sample_values(series: pd.Series, *, sample_n: int, seed: int) -> pd.Series:
    if len(series) <= sample_n:
        return series
    return series.sample(n=int(sample_n), random_state=int(seed))


def _frac_match(series: pd.Series, pattern: str) -> float:
    if series.empty:
        return 0.0
    compiled = re.compile(pattern)
    matched = series.map(lambda value: bool(compiled.fullmatch(str(value))))
    return float(matched.mean())


def _detect_hex_case(sample: pd.Series) -> str:
    letters = "".join(re.sub(r"[^a-fA-F]", "", str(value)) for value in sample)
    if not letters:
        return "lower"
    if letters.lower() == letters:
        return "lower"
    if letters.upper() == letters:
        return "upper"
    return "mixed"


def _detect_date_format(sample: pd.Series) -> str | None:
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if float(parsed.notna().mean()) > 0.9:
            return fmt
    return None


def _frac_numeric(sample: pd.Series) -> float:
    parsed = pd.to_numeric(sample, errors="coerce")
    return float(parsed.notna().mean()) if not sample.empty else 0.0


__all__ = [
    "HASH_PATTERN",
    "IDCARD_PATTERN",
    "PHONE_PATTERN",
    "candidate_match_methods",
    "fingerprint_column",
]
