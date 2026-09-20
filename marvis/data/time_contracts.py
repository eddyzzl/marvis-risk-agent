"""Immutable temporal declarations; registration time is never availability evidence."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marvis.decision_twin._canonical import content_hash, utc_datetime

Text = Annotated[str, Field(min_length=1, max_length=256, pattern=r"\S")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Assurance = Literal["verified", "inferred", "unknown"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TimestampEvidence(_FrozenContract):
    """A source claim, not an independent attestation of source authenticity."""

    basis: Literal["recorded", "inferred", "unknown"]
    source_ref: Text | None = None
    source_sha256: Sha256 | None = None
    rule_version: Text | None = None

    @model_validator(mode="after")
    def check_source(self):
        if self.basis != "unknown" and not (self.source_ref and self.source_sha256):
            raise ValueError("recorded/inferred timestamps require a frozen source reference")
        if self.basis == "inferred" and not self.rule_version:
            raise ValueError("inferred timestamps require rule_version")
        if self.basis == "recorded" and self.rule_version:
            raise ValueError("an inference rule cannot be recorded evidence")
        return self


class TimeColumn(_FrozenContract):
    column: Text
    timezone: Text
    evidence: TimestampEvidence

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("timezone must name an IANA zone") from exc
        return value


class DatasetTimeContract(_FrozenContract):
    schema_version: Literal["dataset-time.v1"] = "dataset-time.v1"
    dataset_id: Text
    content_hash: Sha256
    role: Literal["decision", "feature_snapshot"]
    row_id: Text
    entity_columns: tuple[Text, ...] = Field(min_length=1)
    decision_at: TimeColumn | None = None
    event_at: TimeColumn | None = None
    available_at: TimeColumn | None = None
    version_column: Text | None = None
    effective_from: TimeColumn | None = None
    effective_to: TimeColumn | None = None
    window_start: TimeColumn | None = None
    window_end: TimeColumn | None = None

    @model_validator(mode="after")
    def check_shape(self):
        if len(set(self.entity_columns)) != len(self.entity_columns):
            raise ValueError("entity_columns must be unique")
        if self.role == "decision":
            if self.decision_at is None:
                raise ValueError("decision requires decision_at")
            if any(getattr(self, name) is not None for name in (
                "event_at", "available_at", "version_column", "effective_from",
                "effective_to", "window_start", "window_end",
            )):
                raise ValueError("decision contract cannot declare feature snapshot semantics")
        elif self.event_at is None or self.version_column is None or self.decision_at:
            raise ValueError("feature_snapshot requires event_at/version, without decision_at")
        for start, end in ((self.effective_from, self.effective_to), (self.window_start, self.window_end)):
            if (start is None) != (end is None):
                raise ValueError("interval endpoints must be declared together")
        columns = [item.column for item in self.time_columns()]
        if len(set(columns)) != len(columns):
            raise ValueError("distinct time meanings require distinct columns")
        return self

    def time_columns(self) -> tuple[TimeColumn, ...]:
        return tuple(value for value in (
            self.decision_at, self.event_at, self.available_at, self.effective_from,
            self.effective_to, self.window_start, self.window_end,
        ) if value is not None)

    @property
    def contract_sha256(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class AsOfJoinSpec(_FrozenContract):
    schema_version: Literal["asof-selection.v1"] = "asof-selection.v1"
    as_of: datetime
    feature_columns: tuple[Text, ...] = Field(min_length=1)
    feature_prefix: Text = "asof__"
    mode: Literal["verified", "exploration"] = "verified"
    lookback_seconds: int | None = Field(default=None, ge=0)
    partition_pairs: tuple[tuple[Text, Text], ...] = ()
    require_match: bool = True
    max_source_rows: int = Field(default=200_000, ge=1, le=200_000)
    max_candidate_checks: int = Field(default=2_000_000, ge=1, le=10_000_000)

    @field_validator("as_of")
    @classmethod
    def aware_as_of(cls, value):
        if value.utcoffset() is None:
            raise ValueError("as_of must include a timezone")
        return utc_datetime(value, "as_of")

    @model_validator(mode="after")
    def unique_columns(self):
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be unique")
        if len(set(self.partition_pairs)) != len(self.partition_pairs):
            raise ValueError("partition_pairs must be unique")
        return self


class DatasetTimeStatus(_FrozenContract):
    assurance: Assurance
    artifact_id: str | None = None
    reasons: tuple[str, ...] = ()


def temporal_assurance(*contracts: DatasetTimeContract) -> DatasetTimeStatus:
    reasons = []
    bases = []
    for contract in contracts:
        if contract.role == "feature_snapshot" and contract.available_at is None:
            bases.append("unknown")
            reasons.append("historical_available_at_missing")
        for column in contract.time_columns():
            bases.append(column.evidence.basis)
            if column.evidence.basis != "recorded":
                reasons.append(f"{contract.role}.{column.column}:{column.evidence.basis}")
    assurance = "unknown" if "unknown" in bases else "inferred" if "inferred" in bases else "verified"
    return DatasetTimeStatus(assurance=assurance, reasons=tuple(reasons))
