"""Typed collection intent and business-declared constraints, not permission.

These contracts contain no destination address or executable provider payload.
The strategy selects an intent; the governed executor must separately reserve
and recheck live constraints before any effect.
"""

from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator, model_serializer

from marvis.collection.contracts import Contract, CurrencyUnit, Hash, Identity, Minor
from marvis.decision_twin._canonical import iso_z, parse_datetime

Channel = Literal["phone", "sms", "in_app", "letter"]


class CollectionAction(Contract):
    schema_version: Literal["collection.action.v1"] = "collection.action.v1"
    kind: Literal["contact", "review", "hold"]
    policy_hash: Hash
    queue_id: Identity | None = None
    priority: int = Field(default=0, ge=0, le=1000)
    channel: Channel | None = None
    estimated_cost_minor: Minor | None = None

    @model_validator(mode="after")
    def coherent(self):
        if self.kind in {"contact", "review"} and self.queue_id is None:
            raise ValueError("contact and review require a queue")
        if self.kind == "contact":
            if self.channel is None or self.estimated_cost_minor is None:
                raise ValueError("contact requires a channel and declared unit cost")
        elif self.channel is not None or self.estimated_cost_minor is not None:
            raise ValueError("only contact may declare a channel and cost")
        if self.kind == "hold" and (self.queue_id is not None or self.priority != 0):
            raise ValueError("hold must not enqueue an action")
        return self


class ContactWindow(Contract):
    # ISO weekdays; intervals are [start,end) in the policy's IANA time zone.
    weekdays: list[Annotated[int, Field(ge=1, le=7)]] = Field(
        min_length=1, max_length=7
    )
    start: str = Field(pattern=r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]$")
    end: str = Field(pattern=r"^(?:(?:[01][0-9]|2[0-3]):[0-5][0-9]|24:00)$")

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.weekdays)) != len(self.weekdays) or self.start >= self.end:
            raise ValueError("use unique weekdays and split overnight windows")
        return self


class CollectionQueue(Contract):
    queue_id: Identity
    channels: list[Channel] = Field(default_factory=list, max_length=4)
    # A preview allocation bound, not a statement about live queue occupancy.
    max_batch_actions: int = Field(ge=1, le=10000)
    max_active_actions: int | None = Field(default=None, ge=1, le=10000)

    @model_serializer(mode="wrap")
    def execution_limit(self, handler):
        result = handler(self)
        if self.max_active_actions is None:
            result.pop("max_active_actions", None)
        return result

    @field_validator("channels")
    @classmethod
    def unique(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("queue channels must be unique")
        return value


class CollectionPolicy(Contract):
    schema_version: Literal["collection.policy.v1"] = "collection.policy.v1"
    policy_id: Identity
    revision: Identity
    unit: CurrencyUnit
    valid_from: str
    valid_until: str
    timezone: str = Field(min_length=1, max_length=100)
    contact_windows: list[ContactWindow] = Field(min_length=1, max_length=50)
    queues: list[CollectionQueue] = Field(min_length=1, max_length=100)
    frequency_window_seconds: int = Field(ge=1, le=366 * 86400)
    max_contacts_per_subject_window: int = Field(ge=1, le=10000)
    min_contact_interval_seconds: int = Field(ge=0, le=366 * 86400)
    max_estimated_batch_cost_minor: Minor
    max_estimated_active_cost_minor: Minor | None = None

    @model_serializer(mode="wrap")
    def execution_budget(self, handler):
        result = handler(self)
        if self.max_estimated_active_cost_minor is None:
            result.pop("max_estimated_active_cost_minor", None)
        return result

    basis_artifact_id: str = Field(min_length=1, max_length=160)
    basis_artifact_hash: Hash
    assurance: Literal["business_declared"] = "business_declared"

    @field_validator("valid_from", "valid_until")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "policy validity"))

    @field_validator("timezone")
    @classmethod
    def zone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("policy requires an IANA time zone") from exc
        return value

    @model_validator(mode="after")
    def coherent(self):
        if parse_datetime(self.valid_from, "valid_from") >= parse_datetime(
            self.valid_until, "valid_until"
        ):
            raise ValueError("policy validity must be a nonempty interval")
        if len({queue.queue_id for queue in self.queues}) != len(self.queues):
            raise ValueError("policy queue identities must be unique")
        return self


class CollectionCaseInput(Contract):
    case_id: Identity
    subject_namespace: Identity
    subject_token: Hash
    contact_permission: Literal["allowed", "prohibited", "unknown"]
    features: dict[str, str | int | float | bool | None]
    source_artifact_id: str = Field(min_length=1, max_length=160)
    source_artifact_hash: Hash


class ContactAttempt(Contract):
    attempt_id: Identity
    case_id: Identity
    attempted_at: str
    # Explicit identity only for a platform-recorded reference action. Ordinary
    # publisher attempt IDs are never inferred to be the same native action.
    reference_action_id: Hash | None = None

    @model_serializer(mode="wrap")
    def native_identity(self, handler):
        result = handler(self)
        if self.reference_action_id is None:
            result.pop("reference_action_id", None)
        return result

    available_at: str | None
    # Unknown effects consume capacity until positively reconciled. A failed
    # provider response alone cannot be declared failed_before_dispatch.
    state: Literal[
        "reserved",
        "dispatched",
        "completed",
        "unknown_effect",
        "cancelled_before_dispatch",
        "failed_before_dispatch",
    ]

    @field_validator("attempted_at", "available_at")
    @classmethod
    def timestamp(cls, value):
        return None if value is None else iso_z(parse_datetime(value, "attempt time"))

    @model_validator(mode="after")
    def coherent(self):
        if self.available_at and parse_datetime(
            self.available_at, "available_at"
        ) < parse_datetime(self.attempted_at, "attempted_at"):
            raise ValueError("attempt availability cannot precede the attempt")
        return self


class ContactHistory(Contract):
    subject_namespace: Identity
    subject_token: Hash
    from_at: str
    through_at: str
    available_at: str
    coverage: Literal["complete", "partial", "unknown"]
    attempts: list[ContactAttempt] = Field(default_factory=list, max_length=10000)
    source_artifact_id: str = Field(min_length=1, max_length=160)
    source_artifact_hash: Hash
    assurance: Literal["publisher_declared"] = "publisher_declared"

    @field_validator("from_at", "through_at", "available_at")
    @classmethod
    def timestamp(cls, value):
        return iso_z(parse_datetime(value, "history coverage"))

    @model_validator(mode="after")
    def coherent(self):
        if not (
            parse_datetime(self.from_at, "from_at")
            <= parse_datetime(self.through_at, "through_at")
            <= parse_datetime(self.available_at, "available_at")
        ):
            raise ValueError("history must satisfy from <= through <= available")
        if len({item.attempt_id for item in self.attempts}) != len(self.attempts):
            raise ValueError("attempt identities must be unique per subject")
        return self
