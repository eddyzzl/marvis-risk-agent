from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import uuid
from typing import Any

from marvis.agent_memory.models import normalize_memory_type


CONFIDENCE_THRESHOLDS = {"high": 4, "medium": 2}
MAX_DISTILLED_SUMMARY_CHARS = 400
DISTILLATION_STATUSES = ("active", "rolled_back")


@dataclass(frozen=True)
class MemoryDistillation:
    id: str
    category: str
    scope_key: str
    distilled_summary: str
    structured: dict[str, Any] = field(default_factory=dict)
    source_memory_ids: tuple[str, ...] = ()
    support_count: int = 0
    confidence: str = "low"
    superseded_by: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_memory_type(self.category))
        object.__setattr__(self, "scope_key", str(self.scope_key))
        object.__setattr__(
            self,
            "distilled_summary",
            str(self.distilled_summary)[:MAX_DISTILLED_SUMMARY_CHARS],
        )
        object.__setattr__(
            self,
            "source_memory_ids",
            tuple(str(memory_id) for memory_id in self.source_memory_ids),
        )
        object.__setattr__(self, "support_count", int(self.support_count))
        object.__setattr__(self, "confidence", normalize_distillation_confidence(self.confidence))


def new_distillation(
    *,
    category: str,
    scope_key: str,
    distilled_summary: str,
    structured: dict[str, Any] | None = None,
    source_memory_ids: tuple[str, ...] = (),
    support_count: int = 0,
    confidence: str | None = None,
) -> MemoryDistillation:
    now = _now_iso()
    return MemoryDistillation(
        id=f"dist_{uuid.uuid4().hex}",
        category=category,
        scope_key=scope_key,
        distilled_summary=distilled_summary,
        structured=structured or {},
        source_memory_ids=source_memory_ids,
        support_count=support_count,
        confidence=confidence or confidence_from_support(support_count),
        superseded_by=None,
        created_at=now,
        updated_at=now,
    )


def confidence_from_support(support_count: int) -> str:
    support = int(support_count)
    if support >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if support >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def normalize_distillation_confidence(value: str) -> str:
    normalized = str(value or "low").strip().lower()
    if normalized not in {"high", "medium", "low"}:
        raise ValueError(f"unsupported distillation confidence: {value}")
    return normalized


def normalize_distillation_status(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in DISTILLATION_STATUSES:
        raise ValueError(f"unsupported distillation status: {value}")
    return normalized


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CONFIDENCE_THRESHOLDS",
    "DISTILLATION_STATUSES",
    "MAX_DISTILLED_SUMMARY_CHARS",
    "MemoryDistillation",
    "confidence_from_support",
    "new_distillation",
    "normalize_distillation_confidence",
    "normalize_distillation_status",
]
