from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Protocol

from marvis.decision_twin._canonical import (
    content_hash,
    finite_number,
    iso_z,
    required_text,
    utc_datetime,
)
from marvis.decision_twin.contracts import ReplayManifest


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CURRENCY_RE = re.compile(r"[A-Z]{3}")
_IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_UNIT_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_JSON_SCALARS = (str, int, float, bool, type(None))


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class ObservedField:
    """One scalar fact with field-level point-in-time provenance."""

    name: str
    value: str | int | float | bool | None
    visible_at: datetime
    source_sha256: str

    def __post_init__(self) -> None:
        name = required_text(self.name, "field name", max_length=80)
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError("field name must be a lower_snake_case identifier")
        if not isinstance(self.value, _JSON_SCALARS):
            raise ValueError("field value must be a JSON scalar")
        if isinstance(self.value, float):
            finite_number(self.value, f"{name} value")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "visible_at",
            utc_datetime(self.visible_at, f"{name} visible_at"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, f"{name} source_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "visible_at": iso_z(self.visible_at),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class ReplayFacts:
    """Decision facts passed to a trusted deterministic adapter."""

    record_id: str
    decision_at: datetime
    source_sha256: str
    fields: tuple[ObservedField, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", required_text(self.record_id, "record_id")
        )
        object.__setattr__(
            self,
            "decision_at",
            utc_datetime(self.decision_at, "decision_at"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, "source_sha256"),
        )
        fields = tuple(self.fields)
        if not fields or not all(isinstance(field, ObservedField) for field in fields):
            raise ValueError("fields must contain at least one ObservedField")
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        object.__setattr__(self, "fields", fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "decision_at": iso_z(self.decision_at),
            "source_sha256": self.source_sha256,
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class ProtectedGroupObservation:
    """Governed fairness-only attribute, never included in adapter facts."""

    attribute: str
    group: str
    governance_ref: str
    source_sha256: str

    def __post_init__(self) -> None:
        attribute = required_text(self.attribute, "protected attribute", max_length=80)
        if not _IDENTIFIER_RE.fullmatch(attribute):
            raise ValueError(
                "protected attribute must be a lower_snake_case identifier"
            )
        group = required_text(self.group, "protected group", max_length=80)
        if not _IDENTIFIER_RE.fullmatch(group):
            raise ValueError("protected group must be a lower_snake_case token")
        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "group", group)
        object.__setattr__(
            self,
            "governance_ref",
            required_text(self.governance_ref, "governance_ref"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, "protected group source_sha256"),
        )


@dataclass(frozen=True)
class ReplayRecord:
    facts: ReplayFacts
    protected_group: ProtectedGroupObservation

    def __post_init__(self) -> None:
        if not isinstance(self.facts, ReplayFacts):
            raise ValueError("facts must be ReplayFacts")
        if not isinstance(self.protected_group, ProtectedGroupObservation):
            raise ValueError("protected_group must be ProtectedGroupObservation")


@dataclass(frozen=True)
class TrustedAdapterIdentity:
    """Pinned identity for a non-LLM deterministic replay integration."""

    adapter_id: str
    version: str
    sha256: str
    execution_mode: str = "deterministic"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_id", required_text(self.adapter_id, "adapter_id")
        )
        object.__setattr__(self, "version", required_text(self.version, "version"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "adapter SHA-256"))
        if self.execution_mode != "deterministic":
            raise ValueError(
                "execution_mode must be deterministic; LLM replay is forbidden"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "adapter_id": self.adapter_id,
            "version": self.version,
            "sha256": self.sha256,
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True)
class AdapterDecision:
    """Deterministic adapter output; all financial values share one currency."""

    manifest_hash: str
    strategy_approved: bool
    approved: bool
    exposure: float
    ead: float
    projected_loss: float
    projected_profit: float
    currency: str
    stability: float
    operations_capacity: float
    operations_unit: str
    reason_codes: tuple[str, ...]
    manual_override_applied: bool = False
    manual_override_ref: str | None = None
    appeal_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_hash",
            _sha256(self.manifest_hash, "manifest_hash"),
        )
        if not isinstance(self.strategy_approved, bool) or not isinstance(
            self.approved, bool
        ):
            raise ValueError("strategy_approved and approved must be booleans")
        for field in ("exposure", "ead", "projected_loss", "operations_capacity"):
            value = finite_number(getattr(self, field), field)
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "projected_profit",
            finite_number(self.projected_profit, "projected_profit"),
        )
        stability = finite_number(self.stability, "stability")
        if not 0 <= stability <= 1:
            raise ValueError("stability must be between 0 and 1")
        object.__setattr__(self, "stability", stability)
        if not isinstance(self.currency, str) or not _CURRENCY_RE.fullmatch(
            self.currency
        ):
            raise ValueError("currency must be a three-letter uppercase code")
        if not isinstance(self.operations_unit, str) or not _UNIT_RE.fullmatch(
            self.operations_unit
        ):
            raise ValueError("operations_unit must be a lower_snake_case unit")
        reason_codes = tuple(self.reason_codes)
        if not reason_codes or any(
            not isinstance(code, str) or not _IDENTIFIER_RE.fullmatch(code)
            for code in reason_codes
        ):
            raise ValueError("reason_codes must contain lower_snake_case identifiers")
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("reason_codes must be unique")
        object.__setattr__(self, "reason_codes", reason_codes)
        if not isinstance(self.manual_override_applied, bool):
            raise ValueError("manual_override_applied must be a boolean")
        if self.manual_override_applied:
            object.__setattr__(
                self,
                "manual_override_ref",
                required_text(self.manual_override_ref, "manual_override_ref"),
            )
        elif self.manual_override_ref is not None:
            raise ValueError("manual_override_ref requires manual_override_applied")
        if self.strategy_approved != self.approved and not self.manual_override_applied:
            raise ValueError("decision changed without an explicit manual override")
        if self.appeal_ref is not None:
            object.__setattr__(
                self,
                "appeal_ref",
                required_text(self.appeal_ref, "appeal_ref"),
            )
        if not self.approved and any(
            value != 0.0 for value in (self.exposure, self.ead, self.projected_loss)
        ):
            raise ValueError("declined decisions cannot carry exposure, EAD, or loss")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "strategy_approved": self.strategy_approved,
            "approved": self.approved,
            "exposure": self.exposure,
            "ead": self.ead,
            "projected_loss": self.projected_loss,
            "projected_profit": self.projected_profit,
            "currency": self.currency,
            "stability": self.stability,
            "operations_capacity": self.operations_capacity,
            "operations_unit": self.operations_unit,
            "reason_codes": self.reason_codes,
            "manual_override_applied": self.manual_override_applied,
            "manual_override_ref": self.manual_override_ref,
            "appeal_ref": self.appeal_ref,
        }


class TrustedReplayAdapter(Protocol):
    identity: TrustedAdapterIdentity

    def replay(
        self,
        *,
        manifest: ReplayManifest,
        facts: ReplayFacts,
    ) -> AdapterDecision: ...


@dataclass(frozen=True)
class BindingLineage:
    name: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        name = required_text(self.name, "binding name", max_length=80)
        if not _IDENTIFIER_RE.fullmatch(name):
            raise ValueError("binding name must be a lower_snake_case identifier")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "version", required_text(self.version, "binding version")
        )
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, "binding SHA-256"),
        )


@dataclass(frozen=True)
class DecisionLineage:
    record_id: str
    manifest_hash: str
    adapter: TrustedAdapterIdentity
    bindings: tuple[BindingLineage, ...]
    input_sha256: str
    output_sha256: str
    reason_codes: tuple[str, ...]
    manual_override_ref: str | None
    appeal_ref: str | None
    protected_group_source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", required_text(self.record_id, "lineage record_id")
        )
        object.__setattr__(
            self,
            "manifest_hash",
            _sha256(self.manifest_hash, "lineage manifest_hash"),
        )
        if not isinstance(self.adapter, TrustedAdapterIdentity):
            raise ValueError("lineage adapter must be TrustedAdapterIdentity")
        bindings = tuple(self.bindings)
        if (
            not all(isinstance(binding, BindingLineage) for binding in bindings)
            or tuple(binding.name for binding in bindings)
            != ReplayManifest.REQUIRED_BINDINGS
        ):
            raise ValueError("lineage must contain complete replay bindings in order")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(
            self,
            "input_sha256",
            _sha256(self.input_sha256, "lineage input_sha256"),
        )
        object.__setattr__(
            self,
            "output_sha256",
            _sha256(self.output_sha256, "lineage output_sha256"),
        )
        reason_codes = tuple(self.reason_codes)
        if not reason_codes or any(
            not isinstance(code, str) or not _IDENTIFIER_RE.fullmatch(code)
            for code in reason_codes
        ):
            raise ValueError(
                "lineage reason_codes must be lower_snake_case identifiers"
            )
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("lineage reason_codes must be unique")
        object.__setattr__(self, "reason_codes", reason_codes)
        if self.manual_override_ref is not None:
            object.__setattr__(
                self,
                "manual_override_ref",
                required_text(self.manual_override_ref, "manual_override_ref"),
            )
        if self.appeal_ref is not None:
            object.__setattr__(
                self, "appeal_ref", required_text(self.appeal_ref, "appeal_ref")
            )
        object.__setattr__(
            self,
            "protected_group_source_sha256",
            _sha256(
                self.protected_group_source_sha256,
                "protected group source SHA-256",
            ),
        )


@dataclass(frozen=True)
class ReplayedDecision:
    record_id: str
    decision_at: datetime
    approved: bool
    exposure: float
    ead: float
    projected_loss: float
    projected_profit: float
    currency: str
    stability: float
    operations_capacity: float
    operations_unit: str
    protected_group: ProtectedGroupObservation
    lineage: DecisionLineage

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", required_text(self.record_id, "record_id")
        )
        object.__setattr__(
            self, "decision_at", utc_datetime(self.decision_at, "decision_at")
        )
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")
        for field in ("exposure", "ead", "projected_loss", "operations_capacity"):
            value = finite_number(getattr(self, field), field)
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "projected_profit",
            finite_number(self.projected_profit, "projected_profit"),
        )
        stability = finite_number(self.stability, "stability")
        if not 0 <= stability <= 1:
            raise ValueError("stability must be between 0 and 1")
        object.__setattr__(self, "stability", stability)
        if not isinstance(self.currency, str) or not _CURRENCY_RE.fullmatch(
            self.currency
        ):
            raise ValueError("currency must be a three-letter uppercase code")
        if not isinstance(self.operations_unit, str) or not _UNIT_RE.fullmatch(
            self.operations_unit
        ):
            raise ValueError("operations_unit must be a lower_snake_case unit")
        if not self.approved and any(
            value != 0.0 for value in (self.exposure, self.ead, self.projected_loss)
        ):
            raise ValueError("declined decisions cannot carry exposure, EAD, or loss")
        if not isinstance(self.protected_group, ProtectedGroupObservation):
            raise ValueError("protected_group must be ProtectedGroupObservation")
        if not isinstance(self.lineage, DecisionLineage):
            raise ValueError("lineage must be DecisionLineage")
        if self.lineage.record_id != self.record_id:
            raise ValueError("decision lineage record binding drifted")
        if (
            self.lineage.protected_group_source_sha256
            != self.protected_group.source_sha256
        ):
            raise ValueError("protected group lineage binding drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "decision_at": iso_z(self.decision_at),
            "approved": self.approved,
            "exposure": self.exposure,
            "ead": self.ead,
            "projected_loss": self.projected_loss,
            "projected_profit": self.projected_profit,
            "currency": self.currency,
            "stability": self.stability,
            "operations_capacity": self.operations_capacity,
            "operations_unit": self.operations_unit,
            "protected_group": {
                "attribute": self.protected_group.attribute,
                "group": self.protected_group.group,
                "governance_ref": self.protected_group.governance_ref,
                "source_sha256": self.protected_group.source_sha256,
            },
            "lineage": {
                "record_id": self.lineage.record_id,
                "manifest_hash": self.lineage.manifest_hash,
                "adapter": self.lineage.adapter.to_dict(),
                "bindings": [vars(binding) for binding in self.lineage.bindings],
                "input_sha256": self.lineage.input_sha256,
                "output_sha256": self.lineage.output_sha256,
                "reason_codes": self.lineage.reason_codes,
                "manual_override_ref": self.lineage.manual_override_ref,
                "appeal_ref": self.lineage.appeal_ref,
                "protected_group_source_sha256": (
                    self.lineage.protected_group_source_sha256
                ),
            },
        }


class ReplayEngine:
    """Point-in-time replay coordinator with no metric or LLM implementation."""

    def __init__(self, adapter: TrustedReplayAdapter) -> None:
        identity = getattr(adapter, "identity", None)
        replay = getattr(adapter, "replay", None)
        if not isinstance(identity, TrustedAdapterIdentity) or not callable(replay):
            raise ValueError(
                "adapter must expose a pinned TrustedAdapterIdentity and replay()"
            )
        self._adapter = adapter
        self._identity = identity

    def replay(
        self, manifest: ReplayManifest, record: ReplayRecord
    ) -> ReplayedDecision:
        if not isinstance(manifest, ReplayManifest):
            raise ValueError("manifest must be a ReplayManifest")
        if not isinstance(record, ReplayRecord):
            raise ValueError("record must be a ReplayRecord")
        facts = record.facts
        if facts.decision_at > manifest.as_of:
            raise ValueError("decision_at is after replay as_of")
        for field in facts.fields:
            if field.visible_at > facts.decision_at:
                raise ValueError(f"field {field.name} is visible after decision_at")
            if field.visible_at > manifest.as_of:
                raise ValueError(f"field {field.name} is visible after replay as_of")
        protected_name = record.protected_group.attribute.casefold()
        if any(field.name.casefold() == protected_name for field in facts.fields):
            raise ValueError(
                "protected attribute must not enter decision adapter facts"
            )
        output = self._adapter.replay(manifest=manifest, facts=facts)
        if not isinstance(output, AdapterDecision):
            raise ValueError("adapter must return AdapterDecision")
        if getattr(self._adapter, "identity", None) != self._identity:
            raise ValueError("adapter identity drifted during replay")
        if output.manifest_hash != manifest.manifest_hash:
            raise ValueError("adapter output manifest hash drifted")
        lineage = DecisionLineage(
            record_id=facts.record_id,
            manifest_hash=manifest.manifest_hash,
            adapter=self._identity,
            bindings=tuple(
                BindingLineage(name, binding.version, binding.sha256)
                for name, binding in manifest.bindings.items()
            ),
            input_sha256=content_hash(facts.to_dict()),
            output_sha256=content_hash(output.to_dict()),
            reason_codes=output.reason_codes,
            manual_override_ref=output.manual_override_ref,
            appeal_ref=output.appeal_ref,
            protected_group_source_sha256=record.protected_group.source_sha256,
        )
        return ReplayedDecision(
            record_id=facts.record_id,
            decision_at=facts.decision_at,
            approved=output.approved,
            exposure=output.exposure,
            ead=output.ead,
            projected_loss=output.projected_loss,
            projected_profit=output.projected_profit,
            currency=output.currency,
            stability=output.stability,
            operations_capacity=output.operations_capacity,
            operations_unit=output.operations_unit,
            protected_group=record.protected_group,
            lineage=lineage,
        )
