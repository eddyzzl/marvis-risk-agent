from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Any

from marvis.decision_twin._canonical import (
    finite_number,
    iso_z,
    required_text,
    utc_datetime,
)
from marvis.decision_twin.comparison import ScenarioKind, ScenarioReplay


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class MaturityPolicy:
    version: str
    window: timedelta
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", required_text(self.version, "version"))
        if not isinstance(self.window, timedelta) or self.window <= timedelta(0):
            raise ValueError("maturity window must be a positive timedelta")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "policy SHA-256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "window_seconds": self.window.total_seconds(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ObservedOutcome:
    """Matured accounting/performance outcome for an actually approved record."""

    record_id: str
    observed_at: datetime
    actual_loss: float
    actual_profit: float
    currency: str
    source_ref: str
    source_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "record_id", required_text(self.record_id, "record_id")
        )
        object.__setattr__(
            self,
            "observed_at",
            utc_datetime(self.observed_at, "observed_at"),
        )
        loss = finite_number(self.actual_loss, "actual_loss")
        if loss < 0:
            raise ValueError("actual_loss must be non-negative")
        object.__setattr__(self, "actual_loss", loss)
        object.__setattr__(
            self,
            "actual_profit",
            finite_number(self.actual_profit, "actual_profit"),
        )
        currency = required_text(self.currency, "currency", max_length=3)
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("currency must be a three-letter uppercase code")
        object.__setattr__(self, "currency", currency)
        object.__setattr__(
            self, "source_ref", required_text(self.source_ref, "source_ref")
        )
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, "source SHA-256"),
        )


@dataclass(frozen=True)
class ReconciliationRecord:
    record_id: str
    decision_at: datetime
    matured_at: datetime
    outcome_observed_at: datetime
    projected_loss: float
    actual_loss: float
    loss_delta: float
    projected_profit: float
    actual_profit: float
    profit_delta: float
    currency: str
    decision_output_sha256: str
    outcome_source_ref: str
    outcome_source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "decision_at": iso_z(self.decision_at),
            "matured_at": iso_z(self.matured_at),
            "outcome_observed_at": iso_z(self.outcome_observed_at),
            "projected_loss": self.projected_loss,
            "actual_loss": self.actual_loss,
            "loss_delta": self.loss_delta,
            "projected_profit": self.projected_profit,
            "actual_profit": self.actual_profit,
            "profit_delta": self.profit_delta,
            "currency": self.currency,
            "decision_output_sha256": self.decision_output_sha256,
            "outcome_source_ref": self.outcome_source_ref,
            "outcome_source_sha256": self.outcome_source_sha256,
        }


@dataclass(frozen=True)
class OutcomeReconciliation:
    manifest_hash: str
    maturity_policy: MaturityPolicy
    reconciled_at: datetime
    approved_denominator: int
    projected_loss: float
    actual_loss: float
    loss_delta: float
    projected_profit: float
    actual_profit: float
    profit_delta: float
    currency: str
    records: tuple[ReconciliationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "decision_twin.outcome_reconciliation.v1",
            "manifest_hash": self.manifest_hash,
            "maturity_policy": self.maturity_policy.to_dict(),
            "reconciled_at": iso_z(self.reconciled_at),
            "approved_denominator": self.approved_denominator,
            "projected_loss": self.projected_loss,
            "actual_loss": self.actual_loss,
            "loss_delta": self.loss_delta,
            "projected_profit": self.projected_profit,
            "actual_profit": self.actual_profit,
            "profit_delta": self.profit_delta,
            "currency": self.currency,
            "records": [record.to_dict() for record in self.records],
        }


class OutcomeReconciler:
    """Reconcile actuals only for matured, actually executed approvals."""

    def __init__(self, maturity_policy: MaturityPolicy) -> None:
        if not isinstance(maturity_policy, MaturityPolicy):
            raise ValueError("maturity_policy must be a MaturityPolicy")
        self._policy = maturity_policy

    def reconcile(
        self,
        *,
        scenario: ScenarioReplay,
        outcomes: tuple[ObservedOutcome, ...],
        reconciled_at: datetime,
    ) -> OutcomeReconciliation:
        if not isinstance(scenario, ScenarioReplay):
            raise ValueError("scenario must be ScenarioReplay")
        if (
            scenario.kind is not ScenarioKind.CHAMPION
            or not scenario.observed_execution
        ):
            raise ValueError("outcome reconciliation requires the observed champion")
        as_of = utc_datetime(reconciled_at, "reconciled_at")
        approved = {
            decision.record_id: decision
            for decision in scenario.decisions
            if decision.approved
        }
        for decision in approved.values():
            if as_of < decision.decision_at + self._policy.window:
                raise ValueError(f"decision {decision.record_id} is not mature")
        normalized_outcomes = tuple(outcomes)
        if not all(
            isinstance(outcome, ObservedOutcome) for outcome in normalized_outcomes
        ):
            raise ValueError("outcomes must contain ObservedOutcome records")
        outcome_by_id = {outcome.record_id: outcome for outcome in normalized_outcomes}
        if len(outcome_by_id) != len(normalized_outcomes):
            raise ValueError("outcome record ids must be unique")
        if set(outcome_by_id) != set(approved):
            raise ValueError(
                "outcomes must exactly match the approved observed population"
            )
        records: list[ReconciliationRecord] = []
        for record_id in sorted(approved):
            decision = approved[record_id]
            outcome = outcome_by_id[record_id]
            matured_at = decision.decision_at + self._policy.window
            if outcome.observed_at > as_of:
                raise ValueError(
                    "future observed outcome is not available at reconciliation"
                )
            if outcome.observed_at < matured_at:
                raise ValueError(
                    "outcome was observed before the maturity window closed"
                )
            if outcome.currency != decision.currency:
                raise ValueError("outcome currency unit drift")
            records.append(
                ReconciliationRecord(
                    record_id=record_id,
                    decision_at=decision.decision_at,
                    matured_at=matured_at,
                    outcome_observed_at=outcome.observed_at,
                    projected_loss=decision.projected_loss,
                    actual_loss=outcome.actual_loss,
                    loss_delta=outcome.actual_loss - decision.projected_loss,
                    projected_profit=decision.projected_profit,
                    actual_profit=outcome.actual_profit,
                    profit_delta=outcome.actual_profit - decision.projected_profit,
                    currency=decision.currency,
                    decision_output_sha256=decision.lineage.output_sha256,
                    outcome_source_ref=outcome.source_ref,
                    outcome_source_sha256=outcome.source_sha256,
                )
            )
        projected_loss = sum(record.projected_loss for record in records)
        actual_loss = sum(record.actual_loss for record in records)
        projected_profit = sum(record.projected_profit for record in records)
        actual_profit = sum(record.actual_profit for record in records)
        return OutcomeReconciliation(
            manifest_hash=scenario.manifest_hash,
            maturity_policy=self._policy,
            reconciled_at=as_of,
            approved_denominator=len(approved),
            projected_loss=projected_loss,
            actual_loss=actual_loss,
            loss_delta=actual_loss - projected_loss,
            projected_profit=projected_profit,
            actual_profit=actual_profit,
            profit_delta=actual_profit - projected_profit,
            currency=scenario.currency,
            records=tuple(records),
        )
