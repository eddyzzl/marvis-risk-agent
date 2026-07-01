from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import re

import pandas as pd


SYNTHETIC_DEDUP_STRATEGIES = {"agg_mean", "agg_max"}
SYNTHETIC_DEDUP_WARNING = (
    "aggregate dedup strategies synthesize a derived feature row from same-key "
    "conflicts; the joined row may not correspond to a single source record"
)


def dataset_payload(dataset) -> dict:
    return {
        "id": dataset.id,
        "task_id": dataset.task_id,
        "role": dataset.role,
        "source_name": Path(dataset.source_path).name,
        "source_path": dataset.source_path,
        "format": dataset.format,
        "sheet": dataset.sheet,
        "row_count": dataset.row_count,
        "columns": [
            {
                "name": column.name,
                "semantic_role": column.semantic_role,
                "dtype": column.dtype,
                "is_hashed": column.fingerprint.is_hashed,
                "hash_type": column.fingerprint.hash_type,
            }
            for column in dataset.columns
        ],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


def join_plan_payload(plan) -> dict:
    return {
        "join_plan_id": plan.id,
        "anchor_dataset_id": plan.anchor_dataset_id,
        "status": plan.status,
        "joins": [
            {
                "feature_id": spec.feature_dataset_id,
                "key_pairs": [
                    {
                        "anchor_col": pair.anchor_col,
                        "feature_col": pair.feature_col,
                        "match_method": pair.match_method,
                        "transform_side": pair.transform_side,
                        "match_rate": pair.match_rate,
                        "resolved_by": pair.resolved_by,
                    }
                    for pair in spec.key_pairs
                ],
                "diagnostics": asdict(spec.diagnostics),
                "dedup_strategy": spec.dedup_strategy,
                "dedup_strategy_warning": (
                    SYNTHETIC_DEDUP_WARNING
                    if spec.dedup_strategy in SYNTHETIC_DEDUP_STRATEGIES
                    else None
                ),
                "confirmed": spec.confirmed,
            }
            for spec in plan.joins
        ],
    }


def dataset_preview_profiles(dataset) -> list[dict]:
    return [
        {
            "name": column.name,
            "dtype": column.dtype,
            "semantic_role": column.semantic_role,
            "null_rate": column.null_rate,
            "cardinality": column.cardinality,
            "sample_values": list(column.sample_values),
        }
        for column in dataset.columns
    ]


def masked_preview_records(frame: pd.DataFrame, dataset) -> list[dict]:
    role_by_column = {column.name: column.semantic_role for column in dataset.columns}
    rows = []
    for record in _nan_safe_records(frame):
        rows.append({
            str(column): _mask_preview_value(value, role_by_column.get(str(column)))
            for column, value in record.items()
        })
    return rows


def _nan_safe_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def _mask_preview_value(value, semantic_role: str | None):
    if value is None:
        return None
    if semantic_role == "phone":
        return _mask_preview_text(value, keep_start=3, keep_end=2)
    if semantic_role == "idcard":
        return _mask_preview_text(value, keep_start=4, keep_end=2)
    if semantic_role == "id":
        return _mask_preview_text(value, keep_start=4, keep_end=4)
    if semantic_role in {"categorical", "name"}:
        return _preview_token(value)
    if semantic_role not in {"amount", "date", "score", "target"} and _looks_like_sensitive_preview_identifier(value):
        return _mask_preview_text(value, keep_start=4, keep_end=4)
    return value


def _mask_preview_text(value, *, keep_start: int, keep_end: int) -> str:
    text = _mask_preview_source_text(value)
    if len(text) <= keep_start + keep_end:
        return "*" * len(text)
    hidden = "*" * (len(text) - keep_start - keep_end)
    return f"{text[:keep_start]}{hidden}{text[-keep_end:]}"


def _mask_preview_source_text(value) -> str:
    if not isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            pass
        else:
            if number.is_integer():
                return str(int(number))
    return str(value).strip()


def _preview_token(value) -> str:
    text = _mask_preview_source_text(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"value:{digest}"


def _looks_like_sensitive_preview_identifier(value) -> bool:
    text = re.sub(r"\D+", "", _mask_preview_source_text(value))
    return len(text) >= 12
