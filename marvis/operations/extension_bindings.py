"""Explicit source and policy declarations for the installed periodic tools."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from marvis.operations.bindings import MonitoringBinding


class PortfolioBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: MonitoringBinding
    id_col: str = Field(min_length=1, max_length=200)
    bucket_col: str = Field(min_length=1, max_length=200)
    states: list[str] = Field(min_length=2, max_length=32)
    balance_col: str = Field(min_length=1, max_length=200)
    segment_col: str = Field(min_length=1, max_length=200)
    loss_state: str = Field(min_length=1, max_length=200)
    lgd: float = Field(ge=0, le=1, allow_inf_nan=False)
    horizon_months: StrictInt = Field(ge=1, le=120)
    lookback_seconds: StrictInt = Field(ge=1, le=315_360_000)
    experiment_id: str | None = Field(default=None, min_length=1, max_length=200)
    score_col: str | None = Field(default=None, min_length=1, max_length=200)
    generate_report: bool

    @model_validator(mode="after")
    def explicit_controls(self):
        if len(set(self.states)) != len(self.states) or any(
            not value.strip() or len(value) > 200 for value in self.states
        ):
            raise ValueError("explicit ordered unique delinquency states required")
        if self.loss_state not in self.states:
            raise ValueError("declared loss state must be in ordered states")
        if self.source.label_mode != "not_applicable":
            raise ValueError(
                "portfolio uses observed bucket snapshots, not a fabricated binary target"
            )
        if (self.experiment_id is None) != (self.score_col is None):
            raise ValueError(
                "score stability requires both authenticated experiment and score field"
            )
        return self


class DecisionFeedbackBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, max_length=200)
    environment: Literal["local-reference:production", "local-reference:shadow"]
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    complete_through: datetime
    label_maturity_seconds: StrictInt = Field(ge=0, le=315_360_000)
    max_source_rows: StrictInt = Field(default=200_000, ge=1, le=1_000_000)
    min_labeled_count: StrictInt = Field(ge=1, le=1_000_000)
    max_bad_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    max_fallback_rate: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("complete_through")
    @classmethod
    def aware_time(cls, value):
        if value.tzinfo is None:
            raise ValueError("feedback coverage watermark requires timezone")
        return value.astimezone(UTC)


class FeedbackImport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1, max_length=200)
    environment: Literal["local-reference:production", "local-reference:shadow"]
    dataset_id: str = Field(min_length=1, max_length=200)
    dataset_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    timezone: str = Field(min_length=1, max_length=100)
    request_id_col: str = Field(min_length=1, max_length=200)
    output_hash_col: str = Field(min_length=1, max_length=200)
    package_hash_col: str = Field(min_length=1, max_length=200)
    label_col: str = Field(min_length=1, max_length=200)
    event_at_col: str = Field(min_length=1, max_length=200)
    available_at_col: str = Field(min_length=1, max_length=200)
