"""Validated configuration; this module does not own execution or persistence."""

from __future__ import annotations
from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)


class MonitoringBinding(BaseModel):
    """An explicitly published immutable snapshot and its declared coverage.

    Coverage is a source assertion; temporal validation does not independently
    attest the source. Every outcome retains that distinction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, max_length=200)
    dataset_id: str = Field(min_length=1, max_length=200)
    dataset_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    time_col: str = Field(min_length=1, max_length=200)
    timezone: str
    complete_through: datetime
    target_id: str = Field(min_length=1, max_length=200)
    label_mode: Literal["required", "not_applicable"] = "required"
    target_col: str | None = Field(default=None, min_length=1, max_length=200)
    label_maturity_seconds: StrictInt | None = Field(default=None, ge=0, le=315_360_000)
    score_col: str | None = Field(default=None, min_length=1, max_length=200)
    max_source_rows: StrictInt = Field(default=200_000, ge=1, le=1_000_000)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, KeyError) as exc:
            raise ValueError("timezone must be an IANA timezone") from exc
        return value

    @field_validator("complete_through")
    @classmethod
    def validate_coverage(cls, value):
        if value.tzinfo is None:
            raise ValueError("complete_through must include timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_labels(self):
        if self.label_mode == "required" and not self.target_col:
            raise ValueError("required labels need target_col")
        if self.label_mode == "required" and self.label_maturity_seconds is None:
            raise ValueError("required labels need an explicit label_maturity_seconds")
        if self.label_mode == "not_applicable" and (
            self.target_col or self.label_maturity_seconds is not None
        ):
            raise ValueError(
                "not_applicable labels must not name target_col or maturity"
            )
        return self


class RecheckSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schedule_id: str = Field(min_length=1, max_length=200)
    period_key: str = Field(min_length=1, max_length=500)
    schedule_revision: StrictInt = Field(ge=1)
