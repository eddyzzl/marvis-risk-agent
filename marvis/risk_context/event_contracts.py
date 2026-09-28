"""Frozen, deidentified event contracts; publisher claims are not platform history."""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marvis.decision_twin._canonical import canonical_json, content_hash


Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReferenceText = Annotated[str, Field(min_length=1, max_length=500, pattern=r"\S")]
EntityKind = Literal["subject", "device", "account"]
MAX_EVENTS = 100_000
MAX_FEATURES = 100


class EventError(ValueError):
    def __init__(self, code: str, status: int = 409):
        self.code, self.status = code, status
        super().__init__(code)


def at(value: str) -> datetime:
    if not isinstance(value, str):
        raise EventError("event_timestamp_required", 422)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventError("event_timestamp_invalid", 422) from exc
    if parsed.utcoffset() is None:
        raise EventError("event_timestamp_timezone_required", 422)
    try:
        return parsed.astimezone(UTC)
    except OverflowError as exc:
        raise EventError("event_timestamp_out_of_range", 422) from exc


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    @property
    def contract_hash(self):
        return content_hash(self.model_dump())


class OpaqueIdentity(Contract):
    namespace: Id
    token: Hash


class EventSourceContract(Contract):
    source_id: Id
    task_id: Id
    entity_namespaces: dict[EntityKind, Id]
    event_types: list[Id] = Field(min_length=1, max_length=64)
    numeric_fields: dict[Id, Id] = Field(default_factory=dict, max_length=100)
    publisher_ref: ReferenceText
    assurance: Literal["publisher_declared"] = "publisher_declared"

    @model_validator(mode="after")
    def scope(self):
        if "subject" not in self.entity_namespaces:
            raise ValueError("event source requires a subject namespace")
        if len(set(self.event_types)) != len(self.event_types):
            raise ValueError("event types must be unique")
        return self


class VersionedClaim(Contract):
    version: int = Field(ge=1, le=2**31 - 1)
    supersedes_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def lineage(self):
        expected = None if self.version == 1 else self.version - 1
        if self.supersedes_version != expected:
            raise ValueError(
                "version must explicitly supersede its immediate predecessor"
            )
        return self


class EventRecord(VersionedClaim):
    event_id: Id
    event_type: Id
    event_at: str
    available_at: str | None
    subject: OpaqueIdentity
    device: OpaqueIdentity | None = None
    account: OpaqueIdentity | None = None
    # Integer values with source-declared units preserve monetary precision.
    # Missing fields remain missing; no implicit null-to-zero conversion.
    values: dict[Id, int | None] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def chronology(self):
        occurred = at(self.event_at)
        if self.available_at is not None and occurred > at(self.available_at):
            raise ValueError("event cannot be available before it occurred")
        return self


class CoverageClaim(VersionedClaim):
    coverage_id: Id
    event_types: list[Id] = Field(min_length=1, max_length=64)
    start_exclusive: str
    through_inclusive: str
    declared_at: str
    available_at: str | None
    state: Literal["complete", "partial", "unknown"]
    # None explicitly means the complete source population, not an inferred scope.
    subject: OpaqueIdentity | None
    publisher_ref: ReferenceText

    @model_validator(mode="after")
    def chronology(self):
        if (
            not at(self.start_exclusive)
            < at(self.through_inclusive)
            <= at(self.declared_at)
        ):
            raise ValueError(
                "coverage must be a nonempty publisher-declared historical interval"
            )
        if self.available_at is not None and at(self.declared_at) > at(
            self.available_at
        ):
            raise ValueError("coverage cannot be available before declaration")
        if len(set(self.event_types)) != len(self.event_types):
            raise ValueError("coverage event types must be unique")
        return self


class EventImport(Contract):
    source_id: Id
    source_contract_hash: Hash
    dataset_id: Id
    expected_content_hash: Hash
    kind: Literal["events", "coverage"]
    json_column: Id


class EventSourceGrant(Contract):
    grant_id: Id
    task_id: Id
    source_id: Id
    source_contract_hash: Hash
    grantee_id: Id
    permissions: list[Literal["read", "write"]] = Field(min_length=1, max_length=2)
    purpose: ReferenceText
    basis_artifact_id: Id
    starts_at: str
    expires_at: str

    @model_validator(mode="after")
    def scope(self):
        if len(set(self.permissions)) != len(self.permissions) or at(
            self.starts_at
        ) >= at(self.expires_at):
            raise ValueError(
                "event grant requires unique permissions and a nonempty interval"
            )
        return self


class WindowFeature(Contract):
    name: Id
    operation: Literal["count", "sum", "distinct"]
    field: Id | None = None

    @model_validator(mode="after")
    def field_contract(self):
        if self.operation == "count" and self.field is not None:
            raise ValueError("count has no field")
        if self.operation != "count" and self.field is None:
            raise ValueError("sum and distinct require an explicit field")
        if self.operation == "distinct" and self.field not in {
            "subject",
            "device",
            "account",
        }:
            raise ValueError("distinct requires a typed, namespaced identity")
        return self


class RelationFeature(Contract):
    name: Id
    operation: Literal["neighbor_count", "shared_neighbor_count"]
    target_kind: EntityKind
    via_kind: EntityKind | None = None

    @model_validator(mode="after")
    def path(self):
        if (self.operation == "shared_neighbor_count") != (self.via_kind is not None):
            raise ValueError(
                "only shared relations require an explicit intermediate kind"
            )
        return self


class EventFeatureContract(Contract):
    schema_version: Literal["risk-event.features.v1"] = "risk-event.features.v1"
    source_id: Id
    source_contract_hash: Hash
    decision_at: str
    knowledge_cutoff: str
    availability_mode: Literal["platform_observed", "retrospective_declared"]
    window_seconds: int = Field(ge=1, le=36500 * 86400)
    interval: Literal["(t-window,t]"] = "(t-window,t]"
    include_current_event: bool
    current_event_id: Id | None
    event_types: list[Id] = Field(min_length=1, max_length=64)
    focus_kind: EntityKind
    focus: OpaqueIdentity
    window_features: list[WindowFeature] = Field(
        default_factory=list, max_length=MAX_FEATURES
    )
    relation_features: list[RelationFeature] = Field(
        default_factory=list, max_length=MAX_FEATURES
    )
    policy_ref: ReferenceText

    @model_validator(mode="after")
    def scope(self):
        decision, cutoff = at(self.decision_at), at(self.knowledge_cutoff)
        try:
            decision - timedelta(seconds=self.window_seconds)
        except OverflowError as exc:
            raise ValueError(
                "event window start is outside the supported calendar"
            ) from exc
        if self.availability_mode == "platform_observed" and cutoff > decision:
            raise ValueError("platform-observed features cannot use later ingestion")
        if not self.include_current_event and self.current_event_id is None:
            raise ValueError(
                "excluding the current event requires its explicit identity"
            )
        if len(set(self.event_types)) != len(self.event_types):
            raise ValueError("query event types must be unique")
        names = [
            feature.name for feature in [*self.window_features, *self.relation_features]
        ]
        if not names or len(names) > MAX_FEATURES or len(set(names)) != len(names):
            raise ValueError(
                "feature names must be unique and within the feature budget"
            )
        for feature in self.relation_features:
            if (
                feature.operation == "neighbor_count"
                and feature.target_kind == self.focus_kind
            ):
                raise ValueError(
                    "direct relations must connect different identity kinds"
                )
            if feature.via_kind in {self.focus_kind, feature.target_kind}:
                raise ValueError(
                    "shared relations require a distinct intermediate kind"
                )
        return self


class StoredClaim(Contract):
    source_id: Id
    source_contract_hash: Hash
    kind: Literal["events", "coverage"]
    claim: EventRecord | CoverageClaim
    content_hash: Hash
    ingested_at: str
    origin: Literal["authenticated_registered_dataset"] = (
        "authenticated_registered_dataset"
    )
    dataset_id: Id
    dataset_content_hash: Hash
    import_contract_hash: Hash

    @model_validator(mode="after")
    def binding(self):
        at(self.ingested_at)
        expected = EventRecord if self.kind == "events" else CoverageClaim
        if (
            not isinstance(self.claim, expected)
            or self.content_hash != self.claim.contract_hash
        ):
            raise ValueError("stored event claim binding mismatch")
        return self


class EventSnapshot(Contract):
    source: EventSourceContract
    contract: EventFeatureContract
    claims: list[StoredClaim] = Field(max_length=MAX_EVENTS)

    @model_validator(mode="after")
    def binding(self):
        if (
            self.source.source_id != self.contract.source_id
            or self.source.contract_hash != self.contract.source_contract_hash
        ):
            raise ValueError("snapshot source contract mismatch")
        if not set(self.contract.event_types) <= set(self.source.event_types):
            raise ValueError("query types escape source contract")
        if (
            self.source.entity_namespaces.get(self.contract.focus_kind)
            != self.contract.focus.namespace
        ):
            raise ValueError("query identity escapes source namespace")
        keys = set()
        for stored in self.claims:
            if (
                stored.source_id != self.source.source_id
                or stored.source_contract_hash != self.source.contract_hash
            ):
                raise ValueError("snapshot claim source mismatch")
            claim_id = (
                stored.claim.event_id
                if stored.kind == "events"
                else stored.claim.coverage_id
            )
            key = (stored.kind, claim_id, stored.claim.version)
            if key in keys:
                raise ValueError("snapshot has duplicate version identities")
            keys.add(key)
            validate_claim(self.source, stored.claim)
        return self


def validate_claim(source: EventSourceContract, claim: EventRecord | CoverageClaim):
    types = [claim.event_type] if isinstance(claim, EventRecord) else claim.event_types
    if not set(types) <= set(source.event_types):
        raise EventError("event_type_outside_source", 422)
    for kind in ("subject", "device", "account"):
        identity = getattr(claim, kind, None)
        if (
            identity is not None
            and source.entity_namespaces.get(kind) != identity.namespace
        ):
            raise EventError("event_identity_namespace_mismatch", 422)
    if isinstance(claim, EventRecord) and not set(claim.values) <= set(
        source.numeric_fields
    ):
        raise EventError("event_numeric_field_not_declared", 422)


__all__ = ["canonical_json", "content_hash"]
