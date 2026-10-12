"""Server-issued event intents and registered, replayable native evidence."""

from datetime import UTC, datetime
from pathlib import Path

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db_schema import connect
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.risk_context.event_contracts import (
    Contract,
    EventError,
    EventFeatureContract,
    EventImport,
    EventSourceGrant,
    Hash,
    Id,
    at,
    canonical_json,
    content_hash,
)
from marvis.risk_context.event_repository import EventRepository
from marvis.risk_context.source_repository import require_actor

KIND = "risk_event_features"
ORIGIN = "risk_context.replay_events.v1"


class EventRequest(Contract):
    request_id: Id
    grant_id: Id
    contract: EventFeatureContract


class EventToolInput(Contract):
    request_id: Id
    proposal_hash: Hash
    contract: EventFeatureContract


class EventDatasetImport(Contract):
    grant_id: Id
    dataset: EventImport


class EventService:
    def __init__(self, settings):
        self.settings = settings
        self.repo = EventRepository(settings)
        self.artifacts = TaskArtifactRepository(settings.db_path)
        with connect(settings.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS event_requests(
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                request_id TEXT NOT NULL, payload TEXT NOT NULL,
                signature TEXT NOT NULL, PRIMARY KEY(task_id,request_id))""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS event_requests_immutable
                BEFORE UPDATE ON event_requests BEGIN
                SELECT RAISE(ABORT,'event intent is immutable'); END""")

            conn.execute("""CREATE TRIGGER IF NOT EXISTS event_requests_no_delete
                BEFORE DELETE ON event_requests
                WHEN EXISTS(SELECT 1 FROM tasks WHERE id=OLD.task_id)
                BEGIN SELECT RAISE(ABORT,'event intent is immutable'); END""")

    def _intent(self, conn, task_id, request_id):
        row = conn.execute(
            "SELECT * FROM event_requests WHERE task_id=? AND request_id=?",
            (task_id, request_id),
        ).fetchone()
        if row is None:
            raise EventError("event_request_not_found", 404)
        body = self.repo._body(row)
        request = EventRequest.model_validate(body["request"])
        if body["task_id"] != task_id or request.request_id != request_id:
            raise EventError("event_request_binding_failed")
        return body, request

    def discovery(self, task_id, actor_id):
        """Authenticate metadata without disclosing another principal's claims."""
        sources, grants, requests = [], [], []
        with connect(self.repo.db_path) as conn:
            role = require_actor(conn, actor_id, {"maker", "checker", "admin"})
            if not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
                raise EventError("event_task_not_found", 404)
            source_ids = set()
            readable = {}
            rows = conn.execute(
                "SELECT * FROM event_grants WHERE task_id=? ORDER BY created_at,id",
                (task_id,),
            ).fetchall()
            for row in rows:
                body = self.repo._body(row)
                grant = EventSourceGrant.model_validate(body["grant"])
                if role != "admin" and grant.grantee_id != actor_id:
                    continue
                if grant.task_id != task_id or grant.grant_id != row["id"] or grant.source_id != row["source_id"]:
                    raise EventError("event_grant_binding_failed")
                active = row["revoked_at"] is None and at(grant.starts_at) <= at(datetime.now(UTC).isoformat()) < at(grant.expires_at)
                grants.append({**grant.model_dump(), "state": "revoked" if row["revoked_at"] else "active" if active else "not_current", "owned": grant.grantee_id == actor_id})
                source_ids.add(grant.source_id)
                if active and grant.grantee_id == actor_id and "read" in grant.permissions:
                    readable[grant.source_id] = grant.grant_id
            if role == "admin":
                source_ids.update(row["id"] for row in conn.execute("SELECT id FROM event_sources WHERE task_id=?", (task_id,)))
            for source_id in sorted(source_ids):
                source = self.repo._source(conn, task_id, source_id)
                sources.append({**source.model_dump(), "contract_hash": source.contract_hash})
            for row in conn.execute("SELECT request_id FROM event_requests WHERE task_id=? ORDER BY request_id", (task_id,)):
                body, request = self._intent(conn, task_id, row["request_id"])
                grant_id = readable.get(request.contract.source_id)
                if grant_id is None:
                    continue
                self._access(conn, body, request, actor_id, grant_id, "read")
                requests.append({"request_id": request.request_id, "grant_id": grant_id, "proposal_hash": content_hash(body), "source_id": request.contract.source_id, "decision_at": request.contract.decision_at})
        return {"sources": sources, "grants": grants, "requests": requests}

    def _access(self, conn, body, request, actor_id, grant_id, permission):
        source = self.repo._access(
            conn,
            body["task_id"],
            request.contract.source_id,
            grant_id,
            actor_id,
            permission,
        )
        if source.contract_hash != request.contract.source_contract_hash:
            raise EventError("event_request_source_binding_failed")

    def prepare(self, task_id, request: EventRequest, actor_id):
        request = EventRequest.model_validate(request.model_dump())
        now = datetime.now(UTC).isoformat()
        if any(
            at(value) > at(now)
            for value in (
                request.contract.decision_at,
                request.contract.knowledge_cutoff,
            )
        ):
            raise EventError("event_decision_cutoff_in_future", 422)
        with connect(self.repo.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            body = {
                "task_id": task_id,
                "request": request.model_dump(),
                "actor_id": actor_id,
                "created_at": now,
            }
            for permission in ("read", "write"):
                self._access(
                    conn, body, request, actor_id, request.grant_id, permission
                )
            existing = conn.execute(
                "SELECT 1 FROM event_requests WHERE task_id=? AND request_id=?",
                (task_id, request.request_id),
            ).fetchone()
            if existing:
                body, old = self._intent(conn, task_id, request.request_id)
                if old != request or body["actor_id"] != actor_id:
                    raise EventError("event_request_id_conflict")
            else:
                conn.execute(
                    "INSERT INTO event_requests VALUES(?,?,?,?)",
                    (
                        task_id,
                        request.request_id,
                        canonical_json(body),
                        self.repo._sign(body),
                    ),
                )
        return self._proposal(body, request)

    def _proposal(self, body, request):
        return {
            "schema_version": "risk-event.proposal.v1",
            "request_id": request.request_id,
            "proposal_hash": content_hash(body),
            "contract": request.contract.model_dump(),
            "source_assurance": "publisher_declared",
            "workflow_id": "event_feature_replay",
            "automated_clearance": False,
        }

    def proposal(self, task_id, request_id, actor_id, grant_id):
        with connect(self.repo.db_path) as conn:
            body, request = self._intent(conn, task_id, request_id)
            self._access(conn, body, request, actor_id, grant_id, "read")
        return self._proposal(body, request)

    def authorize_inputs(self, conn, task_id, inputs, actor_id):
        """Read a pending plan's signed intent without executing its producer."""
        inputs = EventToolInput.model_validate(inputs)
        body, request = self._intent(conn, task_id, inputs.request_id)
        if (
            inputs.proposal_hash != content_hash(body)
            or inputs.contract != request.contract
        ):
            raise EventError("event_reviewed_contract_mismatch")
        self._access(conn, body, request, actor_id, request.grant_id, "read")

    def execute(self, task_id, inputs: EventToolInput, *, invocation=None):
        inputs = EventToolInput.model_validate(inputs.model_dump())
        with connect(self.repo.db_path) as conn:
            body, request = self._intent(conn, task_id, inputs.request_id)
            if (
                inputs.proposal_hash != content_hash(body)
                or inputs.contract != request.contract
            ):
                raise EventError("event_reviewed_contract_mismatch")
            self._access(
                conn, body, request, body["actor_id"], request.grant_id, "write"
            )
        # Only the authenticated repository can construct this snapshot. A later
        # import cannot enter the already frozen platform knowledge cutoff.
        receipt = self.repo.evaluate(
            task_id,
            "request:" + content_hash(body),
            request.contract,
            grant_id=request.grant_id,
            actor_id=body["actor_id"],
        )
        if (
            receipt["actor_id"] != body["actor_id"]
            or receipt["grant_id"] != request.grant_id
        ):
            raise EventError("event_request_receipt_binding_failed")
        artifact = self._publish(body, request, receipt, invocation=invocation)
        return self._summary(artifact, receipt, request.request_id, request.grant_id)

    def _path(self, body):
        task_root = self.settings.tasks_dir / body["task_id"]
        directory = task_root / "event_evidence"
        if (
            task_root.is_symlink()
            or directory.is_symlink()
            or not task_root.resolve().is_relative_to(self.settings.tasks_dir.resolve())
            or not directory.resolve().is_relative_to(task_root.resolve())
        ):
            raise EventError("event_artifact_path_invalid")
        return directory / (content_hash(body) + ".json")

    def _publish(self, body, request, receipt, *, invocation=None):
        payload = {"intent": body, "receipt": receipt}
        data = canonical_json(
            {"body": payload, "signature": self.repo._sign(payload)}
        ).encode()
        path = self._path(body)
        provenance = {
            "request_id": request.request_id,
            "proposal_hash": content_hash(body),
            "contract_hash": request.contract.contract_hash,
            "receipt_hash": content_hash(receipt),
        }
        uow = ArtifactUnitOfWork()
        try:
            # The database write lock also serializes same-path publication. Check
            # an existing file rather than replacing corrupted prior evidence.
            with connect(self.repo.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                self._access(
                    conn, body, request, body["actor_id"], request.grant_id, "write"
                )
                if path.exists() or path.is_symlink():
                    self._check_file(path, data)
                else:
                    stage = uow.stage_file(path.parent, path.name)
                    stage.path.write_bytes(data)
                    uow.promote_all()
                artifact = self.artifacts.register_on_connection(
                    conn,
                    task_id=body["task_id"],
                    kind=KIND,
                    path=str(path),
                    content_hash=content_hash(
                        {"body": payload, "signature": self.repo._sign(payload)}
                    ),
                    origin_tool=ORIGIN,
                    provenance=provenance,
                )
                if invocation is not None:
                    invocation.record(
                        conn,
                        uow,
                        artifact=artifact,
                        output=self._summary(
                            artifact, receipt, request.request_id, request.grant_id
                        ),
                        authority={
                            "actor_id": body["actor_id"],
                            "scopes": [
                                {
                                    "task_id": body["task_id"],
                                    "source_id": request.contract.source_id,
                                    "source_contract_hash": request.contract.source_contract_hash,
                                    "grant_id": request.grant_id,
                                }
                            ],
                        },
                    )
            uow.commit()
        except Exception:
            uow.rollback()
            raise
        return artifact

    def _check_file(self, path, expected):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(self.settings.tasks_dir.resolve())
            or path.read_bytes() != expected
        ):
            raise EventError("event_artifact_integrity_failed")

    def evidence(self, task_id, request_id, actor_id, grant_id):
        with connect(self.repo.db_path) as conn:
            body, request = self._intent(conn, task_id, request_id)
            self._access(conn, body, request, actor_id, grant_id, "read")
        receipt = self.repo.replay(
            task_id,
            "request:" + content_hash(body),
            actor_id=actor_id,
            grant_id=grant_id,
        )
        path = self._path(body)
        artifact = self.artifacts.get_for_task_kind_path(task_id, KIND, str(path))
        payload = {"intent": body, "receipt": receipt}
        envelope = {"body": payload, "signature": self.repo._sign(payload)}
        if (
            artifact is None
            or artifact["origin_tool"] != ORIGIN
            or artifact["content_hash"] != content_hash(envelope)
            or artifact["provenance"]
            != {
                "request_id": request_id,
                "proposal_hash": content_hash(body),
                "contract_hash": request.contract.contract_hash,
                "receipt_hash": content_hash(receipt),
            }
        ):
            raise EventError("event_artifact_binding_failed")
        self._check_file(Path(artifact["path"]), canonical_json(envelope).encode())
        return {
            "artifact_id": artifact["id"],
            "content_hash": artifact["content_hash"],
            "kind": KIND,
            "receipt": receipt,
            "proposal_hash": content_hash(body),
        }

    def read(self, task_id, request_id, actor_id, grant_id):
        evidence = self.evidence(task_id, request_id, actor_id, grant_id)
        return self._summary(
            {"id": evidence["artifact_id"]}, evidence["receipt"], request_id, grant_id
        )

    def _summary(self, artifact, receipt, request_id, grant_id):
        result = receipt["result"]
        return {
            "schema_version": "risk-event.summary.v1",
            "request_id": request_id,
            "evidence_url": f"/api/tasks/{receipt['task_id']}/risk-events/requests/{request_id}/evidence?grant_id={grant_id}",
            "artifact_id": artifact["id"],
            "task_id": receipt["task_id"],
            "decision_id": receipt["decision_id"],
            "contract_hash": result["contract_hash"],
            "snapshot_hash": result["snapshot_hash"],
            "status": result["status"],
            "next_action": result["next_action"],
            "availability_mode": result["availability_mode"],
            "source_assurance": "publisher_declared",
            "automated_clearance": False,
            "features": result["features"],
        }
