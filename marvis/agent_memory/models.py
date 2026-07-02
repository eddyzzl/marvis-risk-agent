from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MEMORY_TYPES = (
    "user_preference",
    "field_convention",
    "validation_pitfall",
    "task_experience",
    "model_experience",
    "join_experience",
    "skill_experience_reserved",
)
MEMORY_STATUSES = ("active", "disabled", "deleted", "rejected")
MODEL_EXPERIENCE_REQUIRED_FIELDS = (
    "ks",
    "auc",
    "psi",
    "month",
    "channel",
    "model_name",
    "model_version",
    "scope",
    "source_task_id",
    "important_feature_sources",
)
JOIN_EXPERIENCE_REQUIRED_FIELDS = (
    "match_rate",
    "anchor_rows",
    "joined_rows",
    "feature_table_count",
    "scope",
    "source_task_id",
)


def normalize_memory_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MEMORY_TYPES:
        raise ValueError(f"unsupported memory type: {value}")
    return normalized


def normalize_memory_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in MEMORY_STATUSES:
        raise ValueError(f"unsupported memory status: {value}")
    return normalized


def validate_model_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in MODEL_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required model_experience fields: {joined}")
    return payload


def validate_join_experience_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [
        field_name
        for field_name in JOIN_EXPERIENCE_REQUIRED_FIELDS
        if _is_missing(payload.get(field_name))
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required join_experience fields: {joined}")
    return payload


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    source_task_id: str | None = None
    source_message_id: str | None = None
    confidence: str = "medium"
    reason: str = ""

    def __post_init__(self) -> None:
        normalized_type = normalize_memory_type(self.memory_type)
        object.__setattr__(self, "memory_type", normalized_type)
        if normalized_type == "model_experience":
            validate_model_experience_payload(self.payload)
        elif normalized_type == "join_experience":
            validate_join_experience_payload(self.payload)
