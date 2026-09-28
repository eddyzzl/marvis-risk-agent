"""Collection-owned materials and case access; no application-wide task ACL.

Public writes always pass an in-transaction guard. Imported declarations remain
unverified history; access to a case never authorizes customer contact.
"""

import hashlib
import json

from pydantic import Field

from marvis.artifacts import ArtifactUnitOfWork
from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.contracts import (
    CashflowEvent,
    CollectionCase,
    Contract,
    Identity,
    InstallmentSchedule,
    ReconciliationRequest,
)
from marvis.collection.execution import CollectionExecutor, now
from marvis.collection.execution_state import (
    authenticated,
    batch_on_connection,
    final_for,
    queue_item,
    sign,
)
from marvis.collection.ledger import CollectionEvidenceError
from marvis.db_schema import connect
from marvis.decision_twin._canonical import canonical_json, content_hash
from marvis.governance.repository import GovernanceRepository
from marvis.risk_context.source_repository import require_actor


class CollectionMaterial(Contract):
    material_id: Identity
    description: str = Field(min_length=1, max_length=1000)
    declarations: dict


class CashflowImport(Contract):
    events: list[CashflowEvent] = Field(min_length=1, max_length=10000)


class CollectionService:
    def __init__(self, settings):
        self.settings = settings
        self.executor = CollectionExecutor(settings)
        self.batches = self.executor.batches
        self.ledger = self.batches.ledger
        self.secret = self.ledger.secret
        with connect(settings.db_path) as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS collection_ownership(
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                kind TEXT NOT NULL, identity TEXT NOT NULL,
                payload_json TEXT NOT NULL, signature TEXT NOT NULL,
                PRIMARY KEY(task_id,kind,identity));
            CREATE TRIGGER IF NOT EXISTS collection_ownership_no_update BEFORE UPDATE ON collection_ownership BEGIN SELECT RAISE(ABORT,'collection ownership immutable'); END;
            CREATE TRIGGER IF NOT EXISTS collection_ownership_no_delete BEFORE DELETE ON collection_ownership WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id) BEGIN SELECT RAISE(ABORT,'collection ownership immutable'); END;
            """)

    def _owner(self, conn, task_id, kind, identity):
        row = conn.execute(
            "SELECT * FROM collection_ownership WHERE task_id=? AND kind=? AND identity=?",
            (task_id, kind, identity),
        ).fetchone()
        value = authenticated(row, self.secret)
        if (value["task_id"], value["kind"], value["identity"]) != (
            task_id,
            kind,
            identity,
        ):
            raise CollectionEvidenceError("collection_ownership_integrity_failed")
        return value

    def _record_owner(self, conn, task_id, kind, identity, actor_id):
        value = {
            "task_id": task_id,
            "kind": kind,
            "identity": identity,
            "actor_id": actor_id,
            "recorded_at": now(),
        }
        conn.execute(
            "INSERT INTO collection_ownership VALUES(?,?,?,?,?)",
            (task_id, kind, identity, canonical_json(value), sign(self.secret, value)),
        )

    def _guard(self, task_id, actor_id, case_ids=(), *, write=False):
        def guard(conn):
            role = require_actor(
                conn, actor_id, {"maker"} if write else {"maker", "checker", "admin"}
            )
            for identity in sorted(set(case_ids)):
                owner = self._owner(conn, task_id, "case", identity)
                if (write or role == "maker") and owner["actor_id"] != actor_id:
                    raise CollectionEvidenceError("collection_case_actor_forbidden")

        return guard

    def material(self, task_id, value: CollectionMaterial, actor_id):
        value = CollectionMaterial.model_validate(value.model_dump())
        body = {
            "schema_version": "collection.declaration.v1",
            **value.model_dump(),
            "source_truth_verified": False,
        }
        raw = canonical_json(body).encode()
        if len(raw) > 16_000_000:
            raise CollectionEvidenceError("collection_material_byte_budget_exceeded")
        digest = hashlib.sha256(raw).hexdigest()
        path = (
            self.settings.tasks_dir
            / task_id
            / "collection"
            / ("material-" + content_hash(value.material_id) + ".json")
        )
        uow = ArtifactUnitOfWork()
        try:
            with connect(self.settings.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                require_actor(conn, actor_id, {"maker"})
                if (
                    conn.execute(
                        "SELECT 1 FROM tasks WHERE id=?", (task_id,)
                    ).fetchone()
                    is None
                ):
                    raise CollectionEvidenceError("collection_task_not_found")
                owner = conn.execute(
                    "SELECT 1 FROM collection_ownership WHERE task_id=? AND kind='material' AND identity=?",
                    (task_id, value.material_id),
                ).fetchone()
                if owner:
                    if (
                        self._owner(conn, task_id, "material", value.material_id)[
                            "actor_id"
                        ]
                        != actor_id
                    ):
                        raise CollectionEvidenceError(
                            "collection_material_actor_forbidden"
                        )
                    if (
                        not path.is_file()
                        or path.resolve() != path.absolute()
                        or path.read_bytes() != raw
                    ):
                        raise CollectionEvidenceError(
                            "collection_material_identity_conflict"
                        )
                else:
                    if path.exists() or path.parent.resolve() != path.parent.absolute():
                        raise CollectionEvidenceError(
                            "collection_material_path_conflict"
                        )
                    staged = uow.stage_file(path.parent, path.name)
                    staged.path.write_bytes(raw)
                    uow.promote_all()
                    self._record_owner(
                        conn, task_id, "material", value.material_id, actor_id
                    )
                artifact = self.executor.artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind="collection_declared_source",
                    path=str(path.relative_to(self.settings.workspace)),
                    content_hash=digest,
                    origin_tool="collection.declare_material",
                    provenance={
                        "material_id": value.material_id,
                        "actor_id": actor_id,
                        "source_truth_verified": False,
                    },
                )
            uow.commit()
        finally:
            uow.rollback()
        return {
            "source_artifact_id": artifact["id"],
            "source_artifact_hash": digest,
            "source_assurance": "historical_import_unverified",
        }

    def create_case(self, task_id, case: CollectionCase, actor_id):
        def guard(conn):
            require_actor(conn, actor_id, {"maker"})
            # A source reference must belong to this maker's explicit collection
            # declaration, not just any guessed artifact in the task.
            row = conn.execute(
                "SELECT * FROM task_artifacts WHERE task_id=? AND id=?",
                (task_id, case.source_artifact_id),
            ).fetchone()

            provenance = json.loads(row["provenance_json"]) if row else {}
            if not row or row["origin_tool"] != "collection.declare_material":
                raise CollectionEvidenceError("collection_case_owned_material_required")
            owner = self._owner(
                conn, task_id, "material", provenance.get("material_id")
            )
            if owner["actor_id"] != actor_id:
                raise CollectionEvidenceError("collection_material_actor_forbidden")
            existing = conn.execute(
                "SELECT 1 FROM collection_evidence WHERE task_id=? AND kind='case' AND identity=?",
                (task_id, case.case_id),
            ).fetchone()
            if existing:
                self._guard(task_id, actor_id, [case.case_id], write=True)(conn)
            else:
                self._record_owner(conn, task_id, "case", case.case_id, actor_id)

        return self.ledger.create_case(task_id, case, writer_guard=guard)

    def case(self, task_id, case_id, actor_id):
        with connect(self.settings.db_path) as conn:
            self._guard(task_id, actor_id, [case_id])(conn)
            return self.ledger._get(conn, task_id, "case", case_id)

    def schedule(self, task_id, value: InstallmentSchedule, actor_id):
        return self.ledger.register_schedule(
            task_id,
            value,
            writer_guard=self._guard(task_id, actor_id, [value.case_id], write=True),
        )

    def cashflows(self, task_id, value: CashflowImport, actor_id):
        return self.ledger.append(
            task_id,
            value.events,
            writer_guard=self._guard(
                task_id, actor_id, [e.case_id for e in value.events], write=True
            ),
        )

    def reconcile(self, task_id, value: ReconciliationRequest, actor_id):
        return self.ledger.reconcile(
            task_id,
            value,
            writer_guard=self._guard(task_id, actor_id, [value.case_id], write=True),
        )

    def report(self, task_id, case_id, digest, actor_id):
        self.case(task_id, case_id, actor_id)
        value = self.ledger.read_report(task_id, digest)
        if value["case_id"] != case_id:
            raise CollectionEvidenceError("collection_report_case_mismatch")
        return value

    def prepare(self, task_id, value: CollectionBatchRequest, actor_id):
        return self.batches.prepare(
            task_id,
            value,
            actor_id,
            writer_guard=self._guard(
                task_id, actor_id, [c.case_id for c in value.cases], write=True
            ),
        )

    def read(self, task_id, batch_id, actor_id):
        batch = self.batches.read(task_id, batch_id, actor_id)
        with connect(self.settings.db_path) as conn:
            require_actor(conn, actor_id, {"maker", "checker", "admin"})
            batch = batch_on_connection(conn, task_id, batch_id, self.secret)
            rows = conn.execute(
                "SELECT * FROM collection_queue_items WHERE task_id=? AND batch_id=? ORDER BY id",
                (task_id, batch_id),
            ).fetchall()
            actions = []
            for row in rows:
                item = queue_item(row, self.secret)
                final = final_for(conn, item, self.secret)
                actions.append(
                    {
                        **{
                            k: item[k]
                            for k in (
                                "id",
                                "case_id",
                                "queue_id",
                                "kind",
                                "assigned_to",
                                "estimated_cost_minor",
                                "platform_reserved_at",
                            )
                        },
                        "state": final["state"] if final else "queued",
                        "actual_cost_minor": None,
                        "customer_contacted": False,
                    }
                )
            effects = conn.execute(
                "SELECT * FROM collection_native_effects WHERE task_id=? AND batch_id=? ORDER BY rowid",
                (task_id, batch_id),
            ).fetchall()
            effect_ids = [row["effect_id"] for row in effects]
        outputs = [
            self.effect(task_id, batch_id, identity, actor_id)["output"]
            for identity in effect_ids
        ]
        return {
            **batch,
            "actions": actions,
            "effects": outputs,
            "execution_mode": "local_reference",
        }

    def effect(self, task_id, batch_id, effect_id, actor_id):
        self.batches.read(task_id, batch_id, actor_id)
        with connect(self.settings.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM collection_native_effects WHERE task_id=? AND batch_id=? AND effect_id=?",
                (task_id, batch_id, effect_id),
            ).fetchone()
            native = authenticated(row, self.secret)
            invocation_id = row["invocation_id"]
        outcome = GovernanceRepository(self.settings.db_path).verify_producer_outcome(
            invocation_id
        )
        if outcome["outcome"] != "applied" or outcome["output"] != native["output"]:
            raise CollectionEvidenceError(
                "collection_original_producer_receipt_unverified"
            )
        return native

    def list_cases(self, task_id, actor_id):
        with connect(self.settings.db_path) as conn:
            role = require_actor(conn, actor_id, {"maker", "checker", "admin"})
            rows = conn.execute(
                "SELECT * FROM collection_ownership WHERE task_id=? AND kind='case' ORDER BY identity LIMIT 1001",
                (task_id,),
            ).fetchall()
            if len(rows) > 1000:
                raise CollectionEvidenceError("collection_case_listing_budget_exceeded")
            result = []
            for row in rows:
                owner = self._owner(conn, task_id, "case", row["identity"])
                if role != "maker" or owner["actor_id"] == actor_id:
                    case = self.ledger._get(conn, task_id, "case", row["identity"])
                    result.append(
                        {
                            key: case[key]
                            for key in (
                                "case_id",
                                "unit",
                                "opening_balance_minor",
                                "opened_at",
                                "source_assurance",
                            )
                        }
                    )
        return result

    def list_batches(self, task_id, actor_id):
        with connect(self.settings.db_path) as conn:
            role = require_actor(conn, actor_id, {"maker", "checker", "admin"})
            rows = conn.execute(
                "SELECT batch_id,actor_id FROM collection_batches WHERE task_id=? ORDER BY created_at,batch_id LIMIT 1001",
                (task_id,),
            ).fetchall()
            if len(rows) > 1000:
                raise CollectionEvidenceError(
                    "collection_batch_listing_budget_exceeded"
                )
            result = []
            for row in rows:
                if role == "maker" and row["actor_id"] != actor_id:
                    continue
                value = batch_on_connection(conn, task_id, row["batch_id"], self.secret)
                result.append(
                    {
                        key: value[key]
                        for key in (
                            "batch_id",
                            "status",
                            "revision",
                            "request_hash",
                            "preview_hash",
                            "created_at",
                        )
                    }
                )
        return result
