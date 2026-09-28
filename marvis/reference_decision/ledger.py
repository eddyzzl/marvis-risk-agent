from __future__ import annotations

from datetime import UTC, datetime
import json
import time
import uuid

from marvis.db_schema import connect
from marvis.reference_decision.contracts import DecisionError, canonical


class DecisionLedger:
    def __init__(self, db_path):
        self.db_path = db_path

    def existing(self, environment, request_id, input_hash):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM reference_decisions WHERE environment=? AND request_id=?",
                (environment, request_id),
            ).fetchone()
        if row is None:
            return None
        if row["input_hash"] != input_hash:
            raise DecisionError("idempotency_payload_conflict", 409)
        if row["status"] == "completed":
            return json.loads(row["response_json"])
        return None

    def pinned_package(self, environment, request_id, input_hash):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT package_hash, input_hash FROM reference_decisions WHERE environment=? AND request_id=?",
                (environment, request_id),
            ).fetchone()
        if row and row["input_hash"] != input_hash:
            raise DecisionError("idempotency_payload_conflict", 409)
        return row["package_hash"] if row else None

    def claim(self, environment, request_id, input_hash, package_hash, timeout):
        owner = uuid.uuid4().hex
        now = time.time()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM reference_decisions WHERE environment=? AND request_id=?",
                (environment, request_id),
            ).fetchone()
            if row:
                if row["input_hash"] != input_hash:
                    raise DecisionError("idempotency_payload_conflict", 409)
                if row["status"] == "completed":
                    return None, json.loads(row["response_json"])
                if row["lease_until"] > now:
                    raise DecisionError("decision_in_progress", 409)
                # Crashed work has no side effects; retry the originally pinned
                # package, with a new fenced owner. Never switch to a new head.
                if row["package_hash"] != package_hash:
                    raise DecisionError("idempotency_package_conflict", 409)
                conn.execute(
                    "UPDATE reference_decisions SET owner=?, lease_until=? WHERE environment=? AND request_id=?",
                    (owner, now + timeout + 15, environment, request_id),
                )
            else:
                # Persistent per-environment admission bound across workers.
                count = conn.execute(
                    "SELECT count(*) FROM reference_decisions WHERE environment=? AND status='running' AND lease_until>?",
                    (environment, now),
                ).fetchone()[0]
                recent = conn.execute(
                    "SELECT count(*) FROM reference_decisions WHERE environment=? AND created_at>?",
                    (environment, datetime.fromtimestamp(now - 60, UTC).isoformat()),
                ).fetchone()[0]
                if count >= 4 or recent >= 120:
                    raise DecisionError("reference_capacity_exceeded", 429)
                conn.execute(
                    "INSERT INTO reference_decisions VALUES (?,?,?,?,?,?,'running',NULL,?)",
                    (
                        environment,
                        request_id,
                        input_hash,
                        package_hash,
                        owner,
                        now + timeout + 15,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        return owner, None

    def finish(self, environment, request_id, owner, response):
        started = time.perf_counter()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            response["timing_ms"]["persistence_prepare"] = (
                time.perf_counter() - started
            ) * 1000
            changed = conn.execute(
                "UPDATE reference_decisions SET status='completed', response_json=? WHERE environment=? AND request_id=? AND owner=? AND status='running' AND lease_until>?",
                (
                    canonical(response),
                    environment,
                    request_id,
                    owner,
                    time.time(),
                ),
            )
            if changed.rowcount != 1:
                raise DecisionError("decision_lease_lost", 409)
        return response
