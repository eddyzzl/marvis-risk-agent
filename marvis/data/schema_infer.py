from __future__ import annotations

import re
from typing import Any

import pandas as pd

from marvis.data.contracts import ColumnFingerprint, ColumnProfile
from marvis.data.fingerprint import fingerprint_column


PROFILE_SAMPLE_VALUES = 5

PHONE_NAMES = (
    "phone",
    "mobile",
    "tel",
    "phone_no",
    "phone_md5",
    "mobile_md5",
    "tel_md5",
)
ID_NAMES = (
    "idcard",
    "id_number",
    "idnumber",
    "id_no",
    "cert_no",
    "cert",
    "identity",
    "id_md5",
    "idcard_md5",
)
DATE_NAMES = (
    "date",
    "dt",
    "day",
    "applydate",
    "apply_date",
    "huisudate",
    "data_date",
    "create_date",
    "created_at",
)
TARGET_NAMES = (
    "target",
    "label",
    "y",
    "bad",
    "is_bad",
    "default",
    "delinquent",
    "overdue",
)
SCORE_NAMES = ("score", "prob", "pd", "p_bad", "model_score")
AMOUNT_NAMES = ("amount", "amt", "loan_amount", "balance", "limit", "income")


def infer_column_profile(
    series: pd.Series,
    name: str,
    *,
    seed: int = 0,
) -> ColumnProfile:
    fingerprint = fingerprint_column(series, seed=seed)
    role = detect_semantic_role(name, fingerprint)
    samples = tuple(
        _desensitize(value, role)
        for value in series.dropna().head(PROFILE_SAMPLE_VALUES)
    )
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        semantic_role=role,
        fingerprint=fingerprint,
        null_rate=float(series.isna().mean()),
        cardinality=int(series.nunique(dropna=True)),
        sample_values=samples,
    )


def detect_semantic_role(name: str, fingerprint: ColumnFingerprint) -> str:
    if fingerprint.value_kind == "raw_phone" or _name_matches(name, PHONE_NAMES):
        return "phone"
    if fingerprint.value_kind == "raw_idcard" or _name_matches(name, ID_NAMES):
        return "idcard"
    if fingerprint.value_kind == "hash":
        if _name_matches(name, PHONE_NAMES):
            return "phone"
        if _name_matches(name, ID_NAMES):
            return "idcard"
        return "id"
    if fingerprint.value_kind == "date" or _name_matches(name, DATE_NAMES):
        return "date"
    if _name_matches(name, TARGET_NAMES):
        return "target"
    if _name_matches(name, SCORE_NAMES):
        return "score"
    if _name_matches(name, AMOUNT_NAMES):
        return "amount"
    return "numeric" if fingerprint.value_kind == "numeric" else "categorical"


def infer_dataset_schema(df: pd.DataFrame, *, seed: int = 0) -> list[ColumnProfile]:
    return [
        infer_column_profile(df[column], str(column), seed=seed)
        for column in df.columns
    ]


def detect_target_column(profiles: list[ColumnProfile], df: pd.DataFrame) -> str | None:
    candidates = [profile.name for profile in profiles if profile.semantic_role == "target"]
    if candidates:
        return candidates[0]
    for profile in profiles:
        if not _name_matches(profile.name, TARGET_NAMES):
            continue
        values = {_binary_value(value) for value in df[profile.name].dropna().unique()}
        if values and values <= {0, 1}:
            return profile.name
    return None


def _name_matches(name: str, keywords: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    tokens = set(normalized.split("_"))
    for keyword in keywords:
        normalized_keyword = _normalize_name(keyword)
        if normalized_keyword == "y":
            if normalized == "y" or "y" in tokens:
                return True
            continue
        if normalized_keyword in tokens or normalized_keyword in normalized:
            return True
    return False


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _desensitize(value: Any, role: str) -> object:
    if role == "phone":
        return _mask_text(value, keep_start=3, keep_end=2)
    if role == "idcard":
        return _mask_text(value, keep_start=4, keep_end=2)
    if role == "id":
        return _mask_text(value, keep_start=4, keep_end=4)
    return value


def _mask_text(value: Any, *, keep_start: int, keep_end: int) -> str:
    text = _mask_source_text(value)
    if len(text) <= keep_start + keep_end:
        return "*" * len(text)
    hidden = "*" * (len(text) - keep_start - keep_end)
    return f"{text[:keep_start]}{hidden}{text[-keep_end:]}"


def _mask_source_text(value: Any) -> str:
    if not isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            pass
        else:
            if number.is_integer():
                return str(int(number))
    return str(value).strip()


def _binary_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0:
        return 0
    if number == 1:
        return 1
    return None


__all__ = [
    "AMOUNT_NAMES",
    "DATE_NAMES",
    "ID_NAMES",
    "PHONE_NAMES",
    "PROFILE_SAMPLE_VALUES",
    "SCORE_NAMES",
    "TARGET_NAMES",
    "detect_semantic_role",
    "detect_target_column",
    "infer_column_profile",
    "infer_dataset_schema",
]
