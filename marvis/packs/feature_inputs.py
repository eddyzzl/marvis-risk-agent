"""Shared feature input handling at the Feature and Modeling pack boundary.

The candidate algorithms remain in marvis.feature.candidates; this adapter only
resolves the registered dataset and translates their hints into tool payloads.
"""
from __future__ import annotations

from marvis.feature.candidates import excluded_categorical_columns, suspected_categorical_columns


def flatten_feature_cols(features) -> list[str]:
    """Flatten one workflow union of feature lists, preserving order and uniqueness."""
    flat: list[str] = []
    seen: set[str] = set()
    for item in (features or []):
        candidates = item if isinstance(item, (list, tuple)) else [item]
        for candidate in candidates:
            name = str(candidate).strip()
            if name and name not in seen:
                seen.add(name)
                flat.append(name)
    return flat


def excluded_categorical_for_screen(
    runtime, dataset_id: str, requested_features: list, *,
    target_col: str, split_col: str | None,
) -> list[dict]:
    """Explain inferred exclusions; an explicit feature choice has no such hint."""
    if [str(item) for item in requested_features if str(item).strip()]:
        return []
    dataset = runtime.registry.get(str(dataset_id))
    excluded = excluded_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in excluded]


def suspected_categorical_for_screen(
    runtime, dataset_id: str, *, target_col: str, split_col: str | None,
) -> list[dict]:
    """Surface numeric nominal-code hints without changing selected features."""
    dataset = runtime.registry.get(str(dataset_id))
    suspected = suspected_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in suspected]
