from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marvis.drafts.errors import DraftStateError


DRAFT_STATUS_DRAFT = "draft"
DRAFT_STATUS_TESTED = "tested"
DRAFT_STATUS_PROMOTED = "promoted"
DRAFT_STATUS_REJECTED = "rejected"
DRAFT_STATUSES = (
    DRAFT_STATUS_DRAFT,
    DRAFT_STATUS_TESTED,
    DRAFT_STATUS_PROMOTED,
    DRAFT_STATUS_REJECTED,
)

_VALID_TRANSITIONS = {
    DRAFT_STATUS_DRAFT: {DRAFT_STATUS_TESTED, DRAFT_STATUS_REJECTED},
    DRAFT_STATUS_TESTED: {DRAFT_STATUS_PROMOTED, DRAFT_STATUS_REJECTED},
    DRAFT_STATUS_PROMOTED: set(),
    DRAFT_STATUS_REJECTED: set(),
}


@dataclass(frozen=True)
class LearningNote:
    id: str
    query: str
    sources: tuple[str, ...]
    distilled: str
    created_at: str


@dataclass(frozen=True)
class DraftTool:
    id: str
    task_id: str
    name: str
    summary: str
    code: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    determinism: str
    source: str
    learning_note_id: str | None
    status: str
    created_at: str


@dataclass(frozen=True)
class DraftRun:
    id: str
    draft_id: str
    task_id: str
    inputs_hash: str
    ok: bool
    output: dict[str, Any] | None
    error: str | None
    at: str


@dataclass(frozen=True)
class PromotionCheck:
    passed: bool
    problems: tuple[str, ...]
    test_result: dict[str, Any] | None


def assert_draft_status_transition(current: str, next_status: str) -> None:
    if current not in DRAFT_STATUSES or next_status not in DRAFT_STATUSES:
        raise DraftStateError(f"unknown draft status transition: {current} -> {next_status}")
    if current == next_status:
        return
    if next_status not in _VALID_TRANSITIONS[current]:
        raise DraftStateError(f"invalid draft status transition: {current} -> {next_status}")


__all__ = [
    "DRAFT_STATUS_DRAFT",
    "DRAFT_STATUS_PROMOTED",
    "DRAFT_STATUS_REJECTED",
    "DRAFT_STATUS_TESTED",
    "DRAFT_STATUSES",
    "DraftRun",
    "DraftTool",
    "LearningNote",
    "PromotionCheck",
    "assert_draft_status_transition",
]
