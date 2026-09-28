"""Trusted, read-only outcome lookup. These callbacks are platform code, not Tools."""

from contextlib import contextmanager
from contextvars import ContextVar
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
        self._verifiers: dict[
            tuple[str, str], tuple[str, Callable, Callable | None, Callable | None]
        ] = {}
        self._reader = ContextVar("reconciliation_reader", default=None)

    def register(
        self,
        kind: str,
        producer: str,
        verifier_id: str,
        verifier: Callable,
        *,
        read_guard: Callable | None = None,
        input_read_guard: Callable | None = None,
    ) -> None:
        key = (kind, producer)
        if key in self._verifiers:
            raise ValueError("outcome verifier already registered")
        self._verifiers[key] = (verifier_id, verifier, read_guard, input_read_guard)

    @contextmanager
    def reader(self, actor_id):
        # Per-request, not a mutable registry global or part of a frozen target.
        token = self._reader.set(actor_id)
        try:
            yield
        finally:
            self._reader.reset(token)

    def authorize_read(self, producer, run_id, connection):
        configured = self._verifiers.get(("tool", producer))
        if configured is None or configured[2] is None:
            return
        prior = connection.execute("PRAGMA query_only").fetchone()[0]
        connection.execute("PRAGMA query_only = ON")
        try:
            configured[2](run_id, connection, self._reader.get())
        finally:
            connection.execute(f"PRAGMA query_only = {int(prior)}")

    def authorize_inputs(self, producer, task_id, inputs, connection):
        configured = self._verifiers.get(("tool", producer))
        if configured is None or configured[3] is None:
            return
        prior = connection.execute("PRAGMA query_only").fetchone()[0]
        connection.execute("PRAGMA query_only = ON")
        try:
            configured[3](task_id, inputs, connection, self._reader.get())
        finally:
            connection.execute(f"PRAGMA query_only = {int(prior)}")

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
        verifier_id, verifier, _, _ = configured
        prior = connection.execute("PRAGMA query_only").fetchone()[0]
        connection.execute("PRAGMA query_only = ON")
        try:
            self.authorize_read(
                target.producer, target.binding.get("run_id"), connection
            )
            proof = verifier(target, connection)
        finally:
            connection.execute(f"PRAGMA query_only = {int(prior)}")
        if not isinstance(proof, OutcomeProof) or proof.binding_hash != target.id:
            raise ValueError(
                "verifier proof does not match the original execution binding"
            )
        return verifier_id, proof
