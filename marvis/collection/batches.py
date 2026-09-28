"""Freeze collection proposals before the existing governance effect boundary.

Only local reference execution may consume these batches. This module creates
proposals, never contacts a customer or treats a preview as an approval.
"""

from datetime import UTC, datetime
import hmac
import json
from typing import Literal

from pydantic import Field, field_validator

from marvis.collection.actions import (
    CollectionCaseInput,
    CollectionPolicy,
    ContactHistory,
)
from marvis.collection.contracts import CollectionCase, Contract, Identity
from marvis.collection.ledger import CollectionEvidenceError, CollectionLedger
from marvis.collection.planning import plan_collection_actions
from marvis.db_schema import connect
from marvis.decision_twin._canonical import (
    canonical_json,
    content_hash,
    iso_z,
    parse_datetime,
)
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.risk_context.source_repository import require_actor


class CollectionBatchRequest(Contract):
    batch_id: Identity
    execution_mode: Literal["local_reference"]
    strategy: dict
    policy: CollectionPolicy
    cases: list[CollectionCaseInput] = Field(min_length=1, max_length=10000)
    histories: list[ContactHistory] = Field(default_factory=list, max_length=10000)
    as_of: str
    knowledge_cutoff: str

    @field_validator("strategy")
    @classmethod
    def typed_strategy(cls, value):
        spec = parse_strategy_spec(value)
        if spec.strategy_type != "collection":
            raise ValueError("collection_strategy_required")
        return spec.to_dict()


def ensure_batch_schema(db_path):
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS collection_batches (
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            batch_id TEXT NOT NULL, request_json TEXT NOT NULL,
            request_hash TEXT NOT NULL, preview_json TEXT NOT NULL,
            preview_hash TEXT NOT NULL, actor_id TEXT NOT NULL,
            created_at TEXT NOT NULL, signature TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('proposed','queued','completed','cancelled')),
            revision INTEGER NOT NULL CHECK(revision>0),
            PRIMARY KEY(task_id,batch_id)
        );
        CREATE TRIGGER IF NOT EXISTS collection_batch_identity_immutable
        BEFORE UPDATE ON collection_batches WHEN
            NEW.task_id != OLD.task_id OR NEW.batch_id != OLD.batch_id OR
            NEW.request_json != OLD.request_json OR NEW.request_hash != OLD.request_hash OR
            NEW.preview_json != OLD.preview_json OR NEW.preview_hash != OLD.preview_hash OR
            NEW.actor_id != OLD.actor_id OR NEW.created_at != OLD.created_at OR
            NEW.signature != OLD.signature
        BEGIN SELECT RAISE(ABORT,'collection batch identity is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS collection_batch_no_delete
        BEFORE DELETE ON collection_batches
        WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id)
        BEGIN SELECT RAISE(ABORT,'collection batch identity is immutable'); END;
        """)


class CollectionBatchStore:
    def __init__(self, settings):
        self.settings = settings
        self.ledger = CollectionLedger(settings)
        ensure_batch_schema(settings.db_path)

    def _signature(
        self, task_id, batch_id, request_hash, preview_hash, actor_id, created_at
    ):
        return hmac.new(
            self.ledger.secret,
            canonical_json(
                [
                    "collection.batch.v1",
                    task_id,
                    batch_id,
                    request_hash,
                    preview_hash,
                    actor_id,
                    created_at,
                ]
            ).encode(),
            "sha256",
        ).hexdigest()

    def _decode(self, row):
        if row is None:
            raise CollectionEvidenceError("collection_batch_not_found")
        request, preview = (
            json.loads(row["request_json"]),
            json.loads(row["preview_json"]),
        )
        if (
            content_hash(request) != row["request_hash"]
            or preview.get("preview_hash") != row["preview_hash"]
            or content_hash(
                {key: value for key, value in preview.items() if key != "preview_hash"}
            )
            != row["preview_hash"]
            or request["batch_id"] != row["batch_id"]
            or not hmac.compare_digest(
                row["signature"],
                self._signature(
                    row["task_id"],
                    row["batch_id"],
                    row["request_hash"],
                    row["preview_hash"],
                    row["actor_id"],
                    row["created_at"],
                ),
            )
        ):
            raise CollectionEvidenceError("collection_batch_integrity_failed")
        return {
            "task_id": row["task_id"],
            "batch_id": row["batch_id"],
            "request_hash": row["request_hash"],
            "preview_hash": row["preview_hash"],
            "request": request,
            "preview": preview,
            "actor_id": row["actor_id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "revision": row["revision"],
            "execution_authorized": False,
        }

    def _sources(self, conn, task_id, request):
        references = {
            (request.policy.basis_artifact_id, request.policy.basis_artifact_hash),
            *(
                (item.source_artifact_id, item.source_artifact_hash)
                for item in request.cases
            ),
            *(
                (item.source_artifact_id, item.source_artifact_hash)
                for item in request.histories
            ),
        }
        for artifact_id, digest in sorted(references):
            self.ledger._source(conn, task_id, artifact_id, digest)
        for item in request.cases:
            case = CollectionCase.model_validate(
                self.ledger._get(conn, task_id, "case", item.case_id)
            )
            if (case.subject_namespace, case.subject_token) != (
                item.subject_namespace,
                item.subject_token,
            ):
                raise CollectionEvidenceError("collection_batch_case_subject_mismatch")
            if case.unit != request.policy.unit:
                raise CollectionEvidenceError("collection_batch_currency_unit_mismatch")
            if parse_datetime(case.opened_at, "opened_at") > parse_datetime(
                request.as_of, "as_of"
            ):
                raise CollectionEvidenceError("collection_batch_precedes_case_opening")

    def prepare(self, task_id, request: CollectionBatchRequest, actor_id):
        request = CollectionBatchRequest.model_validate(request.model_dump())
        encoded = canonical_json(request.model_dump())
        if len(encoded.encode()) > 16_000_000:
            raise CollectionEvidenceError("collection_batch_byte_budget_exceeded")
        # Validate role before potentially costly deterministic evaluation, then
        # recheck under the same transaction that freezes sources and identity.
        with connect(self.settings.db_path) as conn:
            require_actor(conn, actor_id, {"maker"})
        preview = plan_collection_actions(
            request.strategy,
            request.policy,
            request.cases,
            request.histories,
            as_of=request.as_of,
            knowledge_cutoff=request.knowledge_cutoff,
        )
        request_hash, preview_hash = (
            content_hash(request.model_dump()),
            preview["preview_hash"],
        )
        with connect(self.settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"maker"})
            self._sources(conn, task_id, request)
            row = conn.execute(
                "SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?",
                (task_id, request.batch_id),
            ).fetchone()
            if row is not None:
                old = self._decode(row)
                if old["request_hash"] != request_hash or old["actor_id"] != actor_id:
                    raise CollectionEvidenceError(
                        "collection_batch_idempotency_conflict"
                    )
                return old
            now = iso_z(datetime.now(UTC))
            signature = self._signature(
                task_id, request.batch_id, request_hash, preview_hash, actor_id, now
            )
            conn.execute(
                "INSERT INTO collection_batches VALUES (?,?,?,?,?,?,?,?,?,'proposed',1)",
                (
                    task_id,
                    request.batch_id,
                    encoded,
                    request_hash,
                    canonical_json(preview),
                    preview_hash,
                    actor_id,
                    now,
                    signature,
                ),
            )
            return self._decode(
                conn.execute(
                    "SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?",
                    (task_id, request.batch_id),
                ).fetchone()
            )

    def read(self, task_id, batch_id, actor_id):
        with connect(self.settings.db_path) as conn:
            role = require_actor(conn, actor_id, {"maker", "checker", "admin"})
            row = conn.execute(
                "SELECT * FROM collection_batches WHERE task_id=? AND batch_id=?",
                (task_id, batch_id),
            ).fetchone()
            result = self._decode(row)
            if role == "maker" and row["actor_id"] != actor_id:
                raise CollectionEvidenceError("collection_batch_actor_forbidden")
            return result
