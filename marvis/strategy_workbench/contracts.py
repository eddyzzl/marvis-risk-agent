from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA = "strategy.workbench-draft.v1"
SEARCH_TOOLS = frozenset(
    {
        "strategy.search_voting_candidates",
        "strategy.search_cross_threshold_rules",
        "strategy.search_cross_matrix_candidates",
        "strategy.search_interactive_tree_split_candidates",
    }
)


class WorkbenchError(Exception):
    def __init__(self, code, status=409):
        super().__init__(code)
        self.code, self.status = code, status


class FieldDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    workflow: str = Field(pattern=r"^[a-z][a-z0-9_]{0,95}$")
    field: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    kind: Literal["checks", "select", "scalar", "narrative", "reference"]
    values: list[str] = Field(max_length=200)


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["strategy.workbench-draft.v1"] = SCHEMA
    fields: list[FieldDraft] = Field(max_length=240)
    open_workflows: list[str] = Field(max_length=240)
    scroll_top: int = Field(default=0, ge=0, le=100_000)


class SaveDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    draft: Draft


class ControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()
