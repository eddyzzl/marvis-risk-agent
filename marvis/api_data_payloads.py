from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

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
        "content_hash": dataset.content_hash,
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


def dataset_preview_profiles(dataset, frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "name": column.name,
            "dtype": column.dtype,
            "semantic_role": column.semantic_role,
            "null_rate": column.null_rate,
            "cardinality": column.cardinality,
            "sample_values": [
                record[column.name]
                for record in dataset_preview_records(
                    frame[[column.name]].dropna().drop_duplicates().head(5)
                )
            ],
        }
        for column in dataset.columns
    ]


def dataset_preview_records(frame: pd.DataFrame) -> list[dict]:
    """Return the exact stored values for the local, user-requested preview."""

    return _nan_safe_records(frame)


def _nan_safe_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")
