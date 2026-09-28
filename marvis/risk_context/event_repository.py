"""Append-only task/source event authority with explicit multi-user source grants."""

from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path

from pydantic import ValidationError

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.repositories.datasets import DatasetRepository
from marvis.risk_context.event_contracts import (
    MAX_EVENTS,
    CoverageClaim,
    EventError,
    EventFeatureContract,
    EventImport,
    EventRecord,
    EventSnapshot,
    EventSourceContract,
    EventSourceGrant,
    StoredClaim,
    at,
    canonical_json,
    content_hash,
    validate_claim,
)
from marvis.risk_context.event_replay import evaluate_event_snapshot
from marvis.risk_context.source_repository import require_actor


MAX_CLAIM_BYTES = 65_536
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024


def _now():
    return datetime.now(UTC).isoformat()


def ensure_event_schema(db_path):
    """Own schema only; never changes the global SQLite user_version."""
    statements = [
        "CREATE TABLE IF NOT EXISTS event_schema(version INTEGER NOT NULL)",
        """CREATE TABLE IF NOT EXISTS event_sources(
            id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            payload TEXT NOT NULL, content_hash TEXT NOT NULL, signature TEXT NOT NULL,
            created_by TEXT NOT NULL, created_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS event_grants(
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            payload TEXT NOT NULL, signature TEXT NOT NULL, revoked_at TEXT,
            revoked_by TEXT, created_by TEXT NOT NULL, created_at TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS event_claims(
            source_id TEXT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
            kind TEXT NOT NULL, claim_id TEXT NOT NULL, version INTEGER NOT NULL,
            payload TEXT NOT NULL, content_hash TEXT NOT NULL, signature TEXT NOT NULL,
            ingested_at TEXT NOT NULL, PRIMARY KEY(source_id,kind,claim_id,version))""",
        """CREATE TABLE IF NOT EXISTS event_imports(
            source_id TEXT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
            contract_hash TEXT NOT NULL, payload TEXT NOT NULL, signature TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(source_id,contract_hash))""",
        """CREATE TABLE IF NOT EXISTS event_decisions(
            task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
            source_id TEXT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
            decision_id TEXT NOT NULL, contract_hash TEXT NOT NULL,
            payload TEXT NOT NULL, signature TEXT NOT NULL, receipt_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(task_id,decision_id))""",
        "CREATE INDEX IF NOT EXISTS event_claim_ingestion ON event_claims(source_id,ingested_at)",
    ]
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for statement in statements:
            conn.execute(statement)
        rows = conn.execute("SELECT version FROM event_schema").fetchall()
        if not rows:
            conn.execute("INSERT INTO event_schema(version) VALUES(1)")
        elif len(rows) != 1 or rows[0]["version"] != 1:
            raise EventError("event_schema_version_unsupported")
        for table in (
            "event_sources",
            "event_claims",
            "event_imports",
            "event_decisions",
        ):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_immutable BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT,'event evidence is immutable'); END"
            )
        for table in (
            "event_sources",
            "event_grants",
            "event_claims",
            "event_imports",
            "event_decisions",
        ):
            task_scope = (
                "SELECT 1 FROM tasks WHERE id=OLD.task_id"
                if table in {"event_sources", "event_grants", "event_decisions"}
                else "SELECT 1 FROM event_sources s JOIN tasks t ON t.id=s.task_id WHERE s.id=OLD.source_id"
            )
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} "
                f"WHEN EXISTS({task_scope}) BEGIN SELECT RAISE(ABORT,'event evidence is immutable'); END"
            )
        conn.execute("""CREATE TRIGGER IF NOT EXISTS event_grants_immutable
            BEFORE UPDATE ON event_grants WHEN
              NEW.payload != OLD.payload OR NEW.signature != OLD.signature
              OR NEW.id != OLD.id OR NEW.task_id != OLD.task_id OR NEW.source_id != OLD.source_id
              OR NEW.created_by != OLD.created_by OR NEW.created_at != OLD.created_at
              OR (OLD.revoked_at IS NOT NULL AND (NEW.revoked_at IS NOT OLD.revoked_at OR NEW.revoked_by IS NOT OLD.revoked_by))
            BEGIN SELECT RAISE(ABORT,'event grant identity is immutable'); END""")


def _decode(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise EventError("event_duplicate_json_key", 422)
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=pairs)


class EventRepository:
    def __init__(self, settings):
        self.settings, self.db_path = settings, settings.db_path
        ensure_event_schema(self.db_path)
        self.secret = settings.plugin_admin_token_path.read_text().strip().encode()
        if not self.secret:
            raise EventError("event_authentication_unavailable")
        self.registry = DatasetRegistry(
            DatasetRepository(self.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )

    def _sign(self, body):
        return hmac.new(
            self.secret,
            ("risk-event.v1:" + canonical_json(body)).encode(),
            hashlib.sha256,
        ).hexdigest()

    def _body(self, row):
        body = _decode(row["payload"])
        if not hmac.compare_digest(row["signature"], self._sign(body)):
            raise EventError("event_evidence_integrity_failed")
        return body

    def _source(self, conn, task_id, source_id):
        row = conn.execute(
            "SELECT * FROM event_sources WHERE task_id=? AND id=?", (task_id, source_id)
        ).fetchone()
        if row is None:
            raise EventError("event_source_not_found", 404)
        source = EventSourceContract.model_validate(self._body(row))
        if (
            source.source_id != source_id
            or source.task_id != task_id
            or source.contract_hash != row["content_hash"]
        ):
            raise EventError("event_source_binding_failed")
        return source

    def register_source(self, source: EventSourceContract, actor_id):
        source = EventSourceContract.model_validate(source.model_dump())
        body = source.model_dump()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            if not conn.execute(
                "SELECT 1 FROM tasks WHERE id=?", (source.task_id,)
            ).fetchone():
                raise EventError("event_task_not_found", 404)
            row = conn.execute(
                "SELECT * FROM event_sources WHERE id=?", (source.source_id,)
            ).fetchone()
            if row:
                if self._body(row) != body:
                    raise EventError("event_source_version_conflict")
            else:
                conn.execute(
                    "INSERT INTO event_sources VALUES(?,?,?,?,?,?,?)",
                    (
                        source.source_id,
                        source.task_id,
                        canonical_json(body),
                        source.contract_hash,
                        self._sign(body),
                        actor_id,
                        _now(),
                    ),
                )
        return {
            "source_id": source.source_id,
            "contract_hash": source.contract_hash,
            "task_id": source.task_id,
        }

    def _basis(self, conn, task_id, artifact_id):
        row = conn.execute(
            "SELECT * FROM task_artifacts WHERE task_id=? AND id=?",
            (task_id, artifact_id),
        ).fetchone()
        if row is None:
            raise EventError("event_grant_basis_not_found", 404)
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.settings.workspace / path
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(
                self.settings.tasks_dir.resolve() / task_id
            )
        ):
            raise EventError("event_grant_basis_path_invalid")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["content_hash"]:
            raise EventError("event_grant_basis_integrity_failed")
        return row["content_hash"]

    def create_grant(self, grant: EventSourceGrant, actor_id):
        grant = EventSourceGrant.model_validate(grant.model_dump())
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            require_actor(
                conn,
                grant.grantee_id,
                {"maker", "admin"}
                if "write" in grant.permissions
                else {"maker", "checker", "admin"},
            )
            source = self._source(conn, grant.task_id, grant.source_id)
            if source.contract_hash != grant.source_contract_hash or at(
                grant.expires_at
            ) <= at(_now()):
                raise EventError("event_grant_contract_or_expiry_invalid", 422)
            body = {
                "grant": grant.model_dump(),
                "basis_hash": self._basis(conn, grant.task_id, grant.basis_artifact_id),
            }
            row = conn.execute(
                "SELECT * FROM event_grants WHERE id=?", (grant.grant_id,)
            ).fetchone()
            if row:
                if self._body(row) != body or row["revoked_at"] is not None:
                    raise EventError("event_grant_version_conflict")
            else:
                conn.execute(
                    "INSERT INTO event_grants VALUES(?,?,?,?,?,NULL,NULL,?,?)",
                    (
                        grant.grant_id,
                        grant.source_id,
                        grant.task_id,
                        canonical_json(body),
                        self._sign(body),
                        actor_id,
                        _now(),
                    ),
                )
        return grant.model_dump()

    def revoke_grant(self, task_id, grant_id, actor_id):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            row = conn.execute(
                "SELECT * FROM event_grants WHERE id=? AND task_id=?",
                (grant_id, task_id),
            ).fetchone()
            if row is None:
                raise EventError("event_grant_not_found", 404)
            self._body(row)
            conn.execute(
                "UPDATE event_grants SET revoked_at=COALESCE(revoked_at,?),revoked_by=COALESCE(revoked_by,?) WHERE id=?",
                (_now(), actor_id, grant_id),
            )

    def _access(self, conn, task_id, source_id, grant_id, actor_id, permission):
        require_actor(
            conn,
            actor_id,
            {"maker", "admin"}
            if permission == "write"
            else {"maker", "checker", "admin"},
        )
        row = conn.execute(
            "SELECT * FROM event_grants WHERE id=? AND task_id=? AND source_id=?",
            (grant_id, task_id, source_id),
        ).fetchone()
        if row is None:
            raise EventError("event_grant_not_found", 404)
        body = self._body(row)
        grant = EventSourceGrant.model_validate(body["grant"])
        if (
            grant.task_id != task_id
            or grant.source_id != source_id
            or grant.grant_id != grant_id
            or grant.grantee_id != actor_id
            or permission not in grant.permissions
        ):
            raise EventError("event_grant_scope_forbidden", 403)
        if row["revoked_at"] is not None or not at(grant.starts_at) <= at(_now()) < at(
            grant.expires_at
        ):
            raise EventError("event_grant_not_active", 403)
        source = self._source(conn, task_id, source_id)
        if (
            source.contract_hash != grant.source_contract_hash
            or self._basis(conn, task_id, grant.basis_artifact_id) != body["basis_hash"]
        ):
            raise EventError("event_grant_binding_failed")
        return source

    def ingest_dataset(self, task_id, request: EventImport, *, grant_id, actor_id):
        request = EventImport.model_validate(request.model_dump())
        with connect(self.db_path) as conn:
            source = self._access(
                conn, task_id, request.source_id, grant_id, actor_id, "write"
            )
        if source.contract_hash != request.source_contract_hash:
            raise EventError("event_import_source_binding_failed")
        binding = self.registry.authenticate_dataset_binding(
            request.dataset_id,
            expected_task_id=task_id,
            expected_content_hash=request.expected_content_hash,
        )
        if binding.row_count > MAX_EVENTS:
            raise EventError("event_import_row_budget_exceeded", 422)
        frame = self.registry.read_authenticated_binding_snapshot(
            binding, columns=[request.json_column]
        )
        claims, size = [], 0
        model = EventRecord if request.kind == "events" else CoverageClaim
        for raw in frame[request.json_column].tolist():
            if not isinstance(raw, str):
                raise EventError("event_import_requires_json_strings", 422)
            count = len(raw.encode())
            size += count
            if count > MAX_CLAIM_BYTES or size > MAX_SNAPSHOT_BYTES:
                raise EventError("event_import_byte_budget_exceeded", 422)
            try:
                claim = model.model_validate(_decode(raw))
            except ValidationError as exc:
                # A malformed uploaded identity may contain raw personal data;
                # report a stable code rather than echoing the rejected value.
                raise EventError("event_claim_schema_invalid", 422) from exc
            validate_claim(source, claim)
            claims.append(claim)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._access(conn, task_id, request.source_id, grant_id, actor_id, "write")
            self.registry.verify_dataset_binding_on_connection(conn, binding)
            old = conn.execute(
                "SELECT * FROM event_imports WHERE source_id=? AND contract_hash=?",
                (request.source_id, request.contract_hash),
            ).fetchone()
            if old:
                result = self._body(old)
                if result["import_contract"] != request.model_dump():
                    raise EventError("event_import_binding_failed")
                return result
            inserted = 0
            now = _now()
            for claim in claims:
                claim_id = (
                    claim.event_id if request.kind == "events" else claim.coverage_id
                )
                prior = conn.execute(
                    "SELECT * FROM event_claims WHERE source_id=? AND kind=? AND claim_id=? ORDER BY version",
                    (request.source_id, request.kind, claim_id),
                ).fetchall()
                duplicate = False
                for row in prior:
                    stored = StoredClaim.model_validate(self._body(row))
                    if row["version"] == claim.version:
                        if stored.content_hash != claim.contract_hash:
                            raise EventError("event_version_content_conflict")
                        duplicate = True
                    if (
                        stored.claim.available_at is not None
                        and claim.available_at is not None
                    ):
                        before, after = (
                            (stored.claim, claim)
                            if stored.claim.version < claim.version
                            else (claim, stored.claim)
                        )
                        if at(before.available_at) > at(after.available_at):
                            raise EventError("event_version_availability_conflict")
                if duplicate:
                    continue
                stored = StoredClaim(
                    source_id=source.source_id,
                    source_contract_hash=source.contract_hash,
                    kind=request.kind,
                    claim=claim,
                    content_hash=claim.contract_hash,
                    ingested_at=now,
                    dataset_id=binding.dataset_id,
                    dataset_content_hash=binding.content_hash,
                    import_contract_hash=request.contract_hash,
                )
                body = stored.model_dump()
                conn.execute(
                    "INSERT INTO event_claims VALUES(?,?,?,?,?,?,?,?)",
                    (
                        source.source_id,
                        request.kind,
                        claim_id,
                        claim.version,
                        canonical_json(body),
                        claim.contract_hash,
                        self._sign(body),
                        now,
                    ),
                )
                inserted += 1
            result = {
                "source_id": source.source_id,
                "source_contract_hash": source.contract_hash,
                "import_contract": request.model_dump(),
                "row_count": len(claims),
                "inserted_count": inserted,
                "duplicate_count": len(claims) - inserted,
                "ingested_at": now,
                "origin": "authenticated_registered_dataset",
                "publisher_availability_independently_verified": False,
            }
            conn.execute(
                "INSERT INTO event_imports VALUES(?,?,?,?,?)",
                (
                    source.source_id,
                    request.contract_hash,
                    canonical_json(result),
                    self._sign(result),
                    now,
                ),
            )
        return result

    def evaluate(
        self,
        task_id,
        decision_id,
        contract: EventFeatureContract,
        *,
        grant_id,
        actor_id,
    ):
        contract = EventFeatureContract.model_validate(contract.model_dump())
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or len(decision_id) > 128
        ):
            raise EventError("event_decision_id_invalid", 422)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            source = self._access(
                conn, task_id, contract.source_id, grant_id, actor_id, "write"
            )
            old = conn.execute(
                "SELECT * FROM event_decisions WHERE task_id=? AND decision_id=?",
                (task_id, decision_id),
            ).fetchone()
            if old:
                body = self._decision_body(old, task_id, decision_id)
                if old["contract_hash"] != contract.contract_hash:
                    raise EventError("event_decision_contract_conflict")
                return body
            now = _now()
            if at(contract.knowledge_cutoff) > at(now) or at(contract.decision_at) > at(
                now
            ):
                raise EventError("event_decision_cutoff_in_future", 422)
            # SQL bounds by ingestion before loading; never LIMIT away real members.
            params = (contract.source_id, at(contract.knowledge_cutoff).isoformat())
            budget = conn.execute(
                "SELECT count(*),coalesce(sum(length(CAST(payload AS BLOB))),0) FROM event_claims WHERE source_id=? AND ingested_at<=?",
                params,
            ).fetchone()
            if budget[0] > MAX_EVENTS or budget[1] > MAX_SNAPSHOT_BYTES:
                raise EventError("event_snapshot_budget_exceeded", 422)
            rows = conn.execute(
                "SELECT * FROM event_claims WHERE source_id=? AND ingested_at<=? ORDER BY kind,claim_id,version",
                params,
            ).fetchall()
            claims = [StoredClaim.model_validate(self._body(row)) for row in rows]
            snapshot = EventSnapshot(source=source, contract=contract, claims=claims)
            result = evaluate_event_snapshot(snapshot)
            body = {
                "schema_version": "risk-event.decision_receipt.v1",
                "task_id": task_id,
                "decision_id": decision_id,
                "source_id": source.source_id,
                "snapshot": snapshot.model_dump(),
                "result": result,
                "created_at": now,
                "actor_id": actor_id,
                "grant_id": grant_id,
            }
            conn.execute(
                "INSERT INTO event_decisions VALUES(?,?,?,?,?,?,?,?)",
                (
                    task_id,
                    source.source_id,
                    decision_id,
                    contract.contract_hash,
                    canonical_json(body),
                    self._sign(body),
                    content_hash(body),
                    now,
                ),
            )
        return body

    def _decision_body(self, row, task_id, decision_id):
        body = self._body(row)
        if (
            body["task_id"] != task_id
            or body["decision_id"] != decision_id
            or body["source_id"] != row["source_id"]
            or content_hash(body) != row["receipt_hash"]
            or body["result"]["contract_hash"] != row["contract_hash"]
        ):
            raise EventError("event_decision_binding_failed")
        return body

    def replay(self, task_id, decision_id, *, grant_id, actor_id):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT * FROM event_decisions WHERE task_id=? AND decision_id=?",
                (task_id, decision_id),
            ).fetchone()
            if row is None:
                raise EventError("event_decision_not_found", 404)
            self._access(conn, task_id, row["source_id"], grant_id, actor_id, "read")
            body = self._decision_body(row, task_id, decision_id)
        result = evaluate_event_snapshot(EventSnapshot.model_validate(body["snapshot"]))
        if result != body["result"]:
            raise EventError("event_replay_result_drift")
        return body
