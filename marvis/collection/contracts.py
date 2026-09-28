"""Explicit collection money and source semantics; no inferred action policy."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marvis.decision_twin._canonical import content_hash, iso_z, parse_datetime

Identity = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Minor = Annotated[int, Field(ge=0, le=10**15)]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    @property
    def content_hash(self):
        return content_hash(self.model_dump())


class CurrencyUnit(Contract):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    minor_unit_exponent: int = Field(ge=0, le=4)
    definition_source: str = Field(min_length=1, max_length=500)


class CollectionCase(Contract):
    schema_version: Literal["collection.case.v1"] = "collection.case.v1"
    case_id: Identity
    subject_namespace: Identity
    subject_token: Hash
    unit: CurrencyUnit
    opening_balance_minor: Minor
    opened_at: str
    source_artifact_id: str = Field(min_length=1, max_length=160)
    source_artifact_hash: Hash
    source_assurance: Literal["historical_import_unverified", "local_reference"]

    @field_validator("opened_at")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "opened_at"))


class Installment(Contract):
    installment_id: Identity
    due_at: str
    amount_minor: Annotated[int, Field(gt=0, le=10**15)]

    @field_validator("due_at")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "due_at"))


class InstallmentSchedule(Contract):
    schedule_id: Identity
    case_id: Identity
    unit: CurrencyUnit
    installments: list[Installment] = Field(min_length=1, max_length=600)
    terms_artifact_id: str = Field(min_length=1, max_length=160)
    terms_artifact_hash: Hash

    @model_validator(mode="after")
    def unique(self):
        if len({part.installment_id for part in self.installments}) != len(
            self.installments
        ):
            raise ValueError("installment identities must be unique")
        times = [parse_datetime(p.due_at, "due_at") for p in self.installments]
        if times != sorted(times):
            raise ValueError("installments must be ordered by due_at")
        return self


class FlowReference(Contract):
    source_id: Identity
    event_id: Identity


class InstallmentReference(Contract):
    schedule_id: Identity
    installment_id: Identity


class CashflowEvent(Contract):
    schema_version: Literal["collection.cashflow.v1"] = "collection.cashflow.v1"
    source_id: Identity
    event_id: Identity
    case_id: Identity
    kind: Literal["payment", "payment_reversal", "cost", "cost_reversal"]
    amount_minor: Annotated[int, Field(gt=0, le=10**15)]
    unit: CurrencyUnit
    event_at: str
    available_at: str | None
    reverses: FlowReference | None = None
    installment: InstallmentReference | None = None
    source_artifact_id: str = Field(min_length=1, max_length=160)
    source_artifact_hash: Hash
    source_assurance: Literal["historical_import_unverified", "local_reference"]

    @field_validator("event_at", "available_at")
    @classmethod
    def timestamp(cls, value):
        return None if value is None else iso_z(parse_datetime(value, "cashflow time"))

    @model_validator(mode="after")
    def coherent(self):
        reversal = self.kind.endswith("_reversal")
        if reversal != (self.reverses is not None):
            raise ValueError(
                "only reversal events require the original event reference"
            )
        if self.installment is not None and self.kind != "payment":
            raise ValueError(
                "only original payments allocate an installment; reversals inherit it"
            )
        if self.available_at and parse_datetime(
            self.available_at, "available_at"
        ) < parse_datetime(self.event_at, "event_at"):
            raise ValueError("cashflow cannot be available before its event")
        return self


class CashflowCoverage(Contract):
    source_id: Identity
    from_at: str
    through_at: str
    available_at: str
    status: Literal["complete", "partial", "unknown"]
    source_artifact_id: str = Field(min_length=1, max_length=160)
    source_artifact_hash: Hash
    # A watermark is a source declaration, never independent source validation.
    assurance: Literal["publisher_declared"] = "publisher_declared"

    @field_validator("from_at", "through_at", "available_at")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "coverage time"))

    @model_validator(mode="after")
    def coherent(self):
        start, end, available = (
            parse_datetime(getattr(self, k), k)
            for k in ("from_at", "through_at", "available_at")
        )
        if not start <= end <= available:
            raise ValueError("coverage must satisfy from <= through <= available")
        return self


class OutcomeMaturity(Contract):
    anchor: Literal["case_opened_at"]
    observation_days: int = Field(ge=1, le=36500)
    policy_artifact_id: str = Field(min_length=1, max_length=160)
    policy_artifact_hash: Hash


class ReconciliationRequest(Contract):
    case_id: Identity
    as_of: str
    knowledge_cutoff: str
    expected_sources: list[Identity] = Field(min_length=1, max_length=100)
    coverage: list[CashflowCoverage] = Field(default_factory=list, max_length=100)
    schedule_id: Identity | None = None
    maturity: OutcomeMaturity | None = None

    @field_validator("as_of", "knowledge_cutoff")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "reconciliation time"))

    @model_validator(mode="after")
    def coherent(self):
        if parse_datetime(self.as_of, "as_of") > parse_datetime(
            self.knowledge_cutoff, "knowledge_cutoff"
        ):
            raise ValueError("as_of must not exceed the knowledge cutoff")
        if len(set(self.expected_sources)) != len(self.expected_sources):
            raise ValueError("expected sources must be unique")
        sources = [item.source_id for item in self.coverage]
        if len(set(sources)) != len(sources) or not set(sources) <= set(
            self.expected_sources
        ):
            raise ValueError("coverage must uniquely bind expected sources")
        return self
