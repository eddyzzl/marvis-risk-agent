"""Trusted, read-only outcome lookup. These callbacks are platform code, not Tools."""

from dataclasses import dataclass
import json
import sqlite3
from typing import Callable

from marvis.orchestrator.evidence import payload_hash


@dataclass(frozen=True)
class VerificationTarget:
    kind: str
    producer: str
    binding_json: str

    @property
    def binding(self) -> dict:
        return json.loads(self.binding_json)

    @property
    def id(self) -> str:
        return payload_hash(
            {"kind": self.kind, "producer": self.producer, "binding": self.binding}
        )


@dataclass(frozen=True)
class OutcomeProof:
    outcome: str
    binding_hash: str
    receipt_id: str = ""
    receipt_hash: str = ""
    reason: str = ""
    output_json: str | None = None

    def __post_init__(self):
        if self.outcome not in {"applied", "not_applied_fenced", "unknown"}:
            raise ValueError("invalid verifier outcome")
        if self.outcome != "unknown" and (not self.receipt_id or not self.receipt_hash):
            raise ValueError("a resolved outcome requires an original producer receipt")


class OutcomeVerifierRegistry:
    """Only application composition can register a trusted verifier."""

    def __init__(self):
        self._verifiers: dict[tuple[str, str], tuple[str, Callable]] = {}

    def register(
        self, kind: str, producer: str, verifier_id: str, verifier: Callable
    ) -> None:
        key = (kind, producer)
        if key in self._verifiers:
            raise ValueError("outcome verifier already registered")
        self._verifiers[key] = (verifier_id, verifier)

    def supports(self, target: VerificationTarget) -> bool:
        return (target.kind, target.producer) in self._verifiers

    def verify(
        self, target: VerificationTarget, connection: sqlite3.Connection
    ) -> tuple[str, OutcomeProof]:
        configured = self._verifiers.get((target.kind, target.producer))
        if configured is None:
            return "unavailable", OutcomeProof(
                "unknown", target.id, reason="当前动作未接入可信核对器。"
            )
        verifier_id, verifier = configured
        connection.execute("PRAGMA query_only = ON")
        try:
            proof = verifier(target, connection)
        finally:
            connection.execute("PRAGMA query_only = OFF")
        if not isinstance(proof, OutcomeProof) or proof.binding_hash != target.id:
            raise ValueError(
                "verifier proof does not match the original execution binding"
            )
        return verifier_id, proof
