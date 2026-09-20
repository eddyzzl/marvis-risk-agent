"""Shared, deterministic dataset context for task setup proposals."""

from __future__ import annotations

from pathlib import Path


def resolve_named_col(columns: list[str], requested: str | None, hints: tuple[str, ...]) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        low = column.lower()
        if any(hint in low for hint in hints):
            return column
    return ""


def dataset_name(registry, dataset) -> str:
    source_identity = getattr(registry, "source_identity", None)
    if callable(source_identity):
        try:
            identity = source_identity(dataset.id)
        except (KeyError, OSError, TypeError, ValueError):
            identity = None
        original_name = (
            str(identity.get("original_name") or "").strip()
            if isinstance(identity, dict)
            else ""
        )
        if original_name:
            return original_name
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))
