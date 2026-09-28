"""Authenticated source execution: durable intent before I/O, GET-only recovery."""

import hashlib
import hmac
import json
import time

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db_schema import connect
from marvis.risk_context.adapters.reference_http import call
from marvis.risk_context.source_contracts import (
    ProviderQuery,
    ProviderResult,
    SourceEnvelope,
    SourceError,
    SourceProfile,
    SourceQuery,
    assess,
    canonical_json,
    content_hash,
    instant,
    iso,
)
from marvis.risk_context.source_repository import SourceRepository, require_actor

KIND = "risk_source_receipt"
ORIGIN = "risk_context.source_query.v1"


class SourceService:
    def __init__(self, settings):
        self.settings = settings
        self.repo = SourceRepository(settings)

    def authorization_basis(self, task_id, basis, actor_id):
        body = {
            "task_id": task_id,
            "basis": basis.model_dump(),
            "actor_id": actor_id,
            "assurance": "admin_declaration_not_external_attestation",
        }
        envelope = {"body": body, "signature": self.repo.sign(body)}
        data = canonical_json(envelope).encode()
        path = (
            self.settings.tasks_dir
            / task_id
            / "source_evidence"
            / f"basis-{content_hash(basis.basis_id)}.json"
        )
        with connect(self.repo.db_path) as conn:
            require_actor(conn, actor_id, {"admin"})
            if not conn.execute(
                "SELECT 1 FROM tasks WHERE id=?", (task_id,)
            ).fetchone():
                raise SourceError("task_not_found", 404)
        return self._write_registered(
            task_id,
            path,
            data,
            "source_authorization_basis",
            {"declaration": basis.declaration},
            actor_id,
        )

    def _write_registered(self, task_id, path, data, kind, provenance, actor_id):
        uow = ArtifactUnitOfWork()
        staged = uow.stage_file(path.parent, path.name)
        staged.path.write_bytes(data)
        try:

            def register(conn):
                conn.execute("BEGIN IMMEDIATE")
                require_actor(conn, actor_id, {"admin"})
                return self.repo.artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=kind,
                    path=str(path),
                    content_hash=hashlib.sha256(data).hexdigest(),
                    origin_tool=ORIGIN,
                    provenance=provenance,
                )

            return uow.finalize_with_connection(
                self.repo.artifacts.transaction, register
            )
        except Exception:
            uow.rollback()
            raise

    def prepare(self, task_id, query: SourceQuery, actor_id):
        with connect(self.repo.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"maker"})
            grant = self.repo.grant_tx(conn, task_id, query.grant_id, actor_id)
            _, profile = self.repo.profile_tx(conn, grant.profile_id)
            if (profile.mode == "historical_file") != (query.historical is not None):
                raise SourceError("source_mode_mismatch", 422)
            existing = conn.execute(
                "SELECT 1 FROM source_requests WHERE task_id=? AND id=?",
                (task_id, query.request_id),
            ).fetchone()
            if existing:
                row, frozen = self.repo.get_tx(conn, task_id, query.request_id)
                if frozen["query"] != query.model_dump() or row["actor_id"] != actor_id:
                    raise SourceError("source_request_id_conflict")
            else:
                now = time.time()
                expires = instant(
                    iso(min(now + query.expires_in_seconds, instant(grant.expires_at)))
                )
                frozen = {
                    "task_id": task_id,
                    "query": query.model_dump(),
                    "grant": grant.model_dump(),
                    "profile": profile.model_dump(),
                    "actor_id": actor_id,
                    "created_at": iso(now),
                    "expires_at": iso(expires),
                }
                provider = ProviderQuery(
                    request_id=content_hash(frozen),
                    subject_namespace=grant.subject_namespace,
                    subject_token=grant.subject_token,
                    purpose=grant.purpose,
                    expires_at=iso(expires),
                )
                frozen["provider_query"] = provider.model_dump()
                conn.execute(
                    """INSERT INTO source_requests(task_id,id,profile_id,grant_id,actor_id,
                    contract,contract_hash,signature,created_at,expires_at,state) VALUES(?,?,?,?,?,?,?,?,?,?,'prepared')""",
                    (
                        task_id,
                        query.request_id,
                        profile.profile_id,
                        grant.grant_id,
                        actor_id,
                        canonical_json(frozen),
                        content_hash(frozen),
                        self.repo.sign(frozen),
                        now,
                        expires,
                    ),
                )
        return self.read(task_id, query.request_id, actor_id)

    def _historical(self, task_id, body):
        from marvis.data.backend import DataBackend
        from marvis.data.registry import DatasetRegistry
        from marvis.repositories.datasets import DatasetRepository

        history = body["query"]["historical"]
        registry = DatasetRegistry(
            DatasetRepository(self.repo.db_path),
            DataBackend(self.settings.datasets_dir),
            self.settings.datasets_dir,
        )
        binding = registry.authenticate_dataset_binding(
            history["dataset_id"],
            expected_task_id=task_id,
            expected_content_hash=history["expected_content_hash"],
        )
        if binding.row_count > 10000:
            raise SourceError("historical_source_row_limit", 422)
        columns = sorted({history["row_id_column"], history["response_column"]})
        if set(columns) - set(registry.authenticated_binding_column_names(binding)):
            raise SourceError("historical_source_columns_missing", 422)
        frame = registry.read_authenticated_binding_snapshot(binding, columns=columns)
        if {history["row_id_column"], history["response_column"]} - set(frame.columns):
            raise SourceError("historical_source_columns_missing", 422)
        selected = frame.loc[
            frame[history["row_id_column"]].astype(str) == history["row_id"]
        ]
        if len(selected) != 1:
            raise SourceError("historical_source_row_not_unique", 422)
        raw = selected.iloc[0][history["response_column"]]
        if not isinstance(raw, str) or len(raw.encode()) > 2_000_000:
            raise SourceError("schema_drift", 422)
        try:
            envelope = SourceEnvelope.model_validate_json(raw)
        except ValueError as exc:
            raise SourceError("schema_drift", 422) from exc
        return (
            {
                "request_id": body["provider_query"]["request_id"],
                "request_hash": content_hash(body["provider_query"]),
                "status": "found",
                "envelope": envelope.model_dump(),
            },
            raw.encode(),
            (registry, binding),
        )

    def execute(self, task_id, request_id):
        owner, row, body, readback = self.repo.claim(task_id, request_id)
        if owner is None:
            return self.read(task_id, request_id, row["actor_id"])
        profile = SourceProfile.model_validate(body["profile"])
        binding = None
        try:
            if profile.mode == "reference_http":
                token_path = (
                    self.repo.private / f"credential-{content_hash(profile.profile_id)}"
                )
                raw, wire_data = call(
                    profile,
                    token_path.read_text(),
                    body["provider_query"],
                    readback=readback,
                )
            else:
                raw, wire_data, binding = self._historical(task_id, body)
            try:
                result = ProviderResult.model_validate(raw)
            except ValueError as exc:
                raise SourceError("schema_drift") from exc
            if result.request_id != body["provider_query"][
                "request_id"
            ] or result.request_hash != content_hash(body["provider_query"]):
                raise SourceError("provider_request_binding_mismatch")
            if result.envelope and (
                result.envelope.subject_namespace != body["grant"]["subject_namespace"]
                or result.envelope.subject_token != body["grant"]["subject_token"]
            ):
                raise SourceError("provider_subject_binding_mismatch")
            self._publish(task_id, request_id, owner, body, result, binding, wire_data)
        except (SourceError, ValueError, OSError, RuntimeError, KeyError) as exc:
            code = (
                exc.code if isinstance(exc, SourceError) else "source_material_invalid"
            )
            uncertain = profile.mode == "reference_http" and code in {
                "unknown_effect",
                "unavailable",
                "source_material_invalid",
            }
            self.repo.finish_error(
                task_id, request_id, owner, code, uncertain=uncertain
            )
        return self.read(task_id, request_id, row["actor_id"])

    def _publish(self, task_id, request_id, owner, frozen, result, binding, wire_data):
        now = time.time()
        raw_data = canonical_json(result.model_dump()).encode()
        raw_hash = hashlib.sha256(raw_data).hexdigest()
        wire_hash = hashlib.sha256(wire_data).hexdigest()
        assessment = (
            assess(result.envelope, now)
            if result.envelope
            else {
                "status": "no_record",
                "missing_reasons": ["provider:no_record"],
                "finding_codes": [],
                "automated_clearance": False,
            }
        )
        history = frozen["query"]["historical"]
        membership = (
            None
            if history is None
            else {
                **{k: v for k, v in history.items() if k != "row_id"},
                "row_id_hash": content_hash(history["row_id"]),
            }
        )
        payload = {
            "task_id": task_id,
            "request_id": request_id,
            "contract_hash": content_hash(frozen),
            "response_hash": raw_hash,
            "wire_response_hash": wire_hash,
            "profile_id": frozen["profile"]["profile_id"],
            "origin": "reference_provider_claim"
            if frozen["profile"]["mode"] == "reference_http"
            else "historical_import_unverified",
            "external_authenticity": "not_independently_verified",
            "recorded_at": iso(now),
            "recorded_status": assessment["status"],
            "missing_reasons": assessment["missing_reasons"],
            "finding_codes": assessment["finding_codes"],
            "source_expires_at": result.envelope.expires_at
            if result.envelope
            else None,
            "dataset_binding": membership,
        }
        data = canonical_json(
            {"body": payload, "signature": self.repo.sign(payload)}
        ).encode()
        receipt_hash = hashlib.sha256(data).hexdigest()
        path = (
            self.settings.tasks_dir
            / task_id
            / "source_evidence"
            / f"{receipt_hash}.json"
        )
        raw_path = self.repo.private / f"response-{raw_hash}.json"
        uow = ArtifactUnitOfWork()
        staged = uow.stage_file(raw_path.parent, raw_path.name)
        staged.path.write_bytes(raw_data)
        staged.path.chmod(0o600)
        wire = uow.stage_file(self.repo.private, f"wire-{wire_hash}.bin")
        wire.path.write_bytes(wire_data)
        wire.path.chmod(0o600)
        uow.stage_file(path.parent, path.name).path.write_bytes(data)
        try:

            def register(conn):
                conn.execute("BEGIN IMMEDIATE")
                row, _ = self.repo.get_tx(conn, task_id, request_id)
                if row["owner"] != owner or row["state"] != "running":
                    raise SourceError("source_owner_superseded")
                # Revocation or expiry during outbound I/O must not make evidence usable.
                # Preserve what was observed, then read() computes current authorization.
                if binding:
                    binding[0].verify_dataset_binding_on_connection(conn, binding[1])
                record = self.repo.artifacts.register_on_connection(
                    conn,
                    task_id=task_id,
                    kind=KIND,
                    path=str(path),
                    content_hash=receipt_hash,
                    origin_tool=ORIGIN,
                    provenance={
                        "contract_hash": payload["contract_hash"],
                        "response_hash": raw_hash,
                        "request_id": request_id,
                    },
                )
                conn.execute(
                    "UPDATE source_requests SET state='completed',artifact_id=?,owner=NULL,lease_until=NULL,error_code=NULL WHERE task_id=? AND id=?",
                    (record["id"], task_id, request_id),
                )
                conn.execute(
                    "UPDATE source_attempts SET finished_at=?,outcome=? WHERE id=?",
                    (now, assessment["status"], owner),
                )
                conn.execute(
                    "UPDATE source_profiles SET failures=0,open_until=0 WHERE id=?",
                    (row["profile_id"],),
                )
                return record

            uow.finalize_with_connection(self.repo.artifacts.transaction, register)
        except Exception:
            uow.rollback()
            raise

    def _receipt(self, row):
        record = self.repo.artifacts.get_for_task(row["task_id"], row["artifact_id"])
        if not record:
            raise SourceError("source_receipt_not_found", 404)
        from pathlib import Path

        path = Path(record["path"])
        expected_dir = self.settings.tasks_dir / row["task_id"] / "source_evidence"
        if (
            path.parent.resolve() != expected_dir.resolve()
            or record["kind"] != KIND
            or record["origin_tool"] != ORIGIN
        ):
            raise SourceError("source_receipt_binding_failed")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["content_hash"]:
            raise SourceError("source_receipt_integrity_failed")
        envelope = json.loads(data)
        payload = envelope["body"]
        if not hmac.compare_digest(envelope["signature"], self.repo.sign(payload)):
            raise SourceError("source_receipt_authentication_failed")
        if (
            payload["task_id"] != row["task_id"]
            or payload["request_id"] != row["id"]
            or payload["contract_hash"] != row["contract_hash"]
        ):
            raise SourceError("source_receipt_binding_failed")
        for key, prefix, suffix in (
            ("response_hash", "response-", ".json"),
            ("wire_response_hash", "wire-", ".bin"),
        ):
            private_data = (
                self.repo.private / f"{prefix}{payload[key]}{suffix}"
            ).read_bytes()
            if (
                len(private_data) > 2_000_000
                or hashlib.sha256(private_data).hexdigest() != payload[key]
            ):
                raise SourceError("source_response_integrity_failed")
        return payload

    def read(self, task_id, request_id, actor_id):
        with connect(self.repo.db_path) as conn:
            row, _ = self.repo.get_tx(conn, task_id, request_id)
            self.repo.grant_tx(conn, task_id, row["grant_id"], actor_id, active=False)
            authorized = True
            try:
                self.repo.grant_tx(conn, task_id, row["grant_id"], actor_id)
            except SourceError:
                authorized = False
        payload = self._receipt(row) if row["artifact_id"] else None
        expired = bool(
            payload
            and payload["source_expires_at"]
            and instant(payload["source_expires_at"]) <= time.time()
        )
        return {
            "schema_version": "risk_source.summary.v1",
            "task_id": task_id,
            "request_id": request_id,
            "profile_id": row["profile_id"],
            "grant_id": row["grant_id"],
            "state": row["state"],
            "status": "unauthorized"
            if not authorized
            else (
                "expired"
                if expired
                else (
                    payload["recorded_status"]
                    if payload
                    else (row["error_code"] or row["state"])
                )
            ),
            "artifact_id": row["artifact_id"],
            "contract_hash": row["contract_hash"],
            "origin": payload["origin"] if payload else None,
            "external_authenticity": "not_independently_verified",
            "missing_reasons": payload["missing_reasons"] if payload else [],
            "finding_codes": payload["finding_codes"] if payload else [],
            "automated_clearance": False,
            "evidence_url": f"/api/tasks/{task_id}/risk-sources/requests/{request_id}/evidence"
            if payload
            else None,
            "next_action": "readback_same_request"
            if row["state"] == "unknown_effect"
            else (
                "execute_prepared_request"
                if row["state"] == "prepared"
                else "inspect_evidence"
            ),
        }

    def evidence(self, task_id, request_id, actor_id):
        summary = self.read(task_id, request_id, actor_id)
        with connect(self.repo.db_path) as conn:
            row, _ = self.repo.get_tx(conn, task_id, request_id)
        if not row["artifact_id"]:
            raise SourceError("source_evidence_unavailable", 404)
        receipt = self._receipt(row)
        data = (
            self.repo.private / f"response-{receipt['response_hash']}.json"
        ).read_bytes()
        if hashlib.sha256(data).hexdigest() != receipt["response_hash"]:
            raise SourceError("source_response_integrity_failed")
        wire_data = (
            self.repo.private / f"wire-{receipt['wire_response_hash']}.bin"
        ).read_bytes()
        if hashlib.sha256(wire_data).hexdigest() != receipt["wire_response_hash"]:
            raise SourceError("source_wire_integrity_failed")
        result = ProviderResult.model_validate_json(data)
        # Historical receipt visibility does not extend a revoked/expired permission
        # to the underlying business data. Recheck after loading the authenticated copy.
        try:
            with connect(self.repo.db_path) as conn:
                self.repo.grant_tx(conn, task_id, row["grant_id"], actor_id)
        except SourceError as exc:
            if exc.code not in {"grant_revoked", "grant_not_current"}:
                raise
            return {
                "summary": {**summary, "status": "unauthorized"},
                "receipt": receipt,
                "normalized_source": None,
                "assessment": None,
            }
        return {
            "summary": summary,
            "receipt": receipt,
            "normalized_source": result.envelope.model_dump()
            if result.envelope
            else None,
            "assessment": assess(result.envelope, time.time())
            if result.envelope
            else None,
        }

    def list_requests(self, task_id, actor_id):
        grants = {g["grant_id"] for g in self.repo.list_grants(task_id, actor_id)}
        with connect(self.repo.db_path) as conn:
            rows = conn.execute(
                "SELECT id,grant_id FROM source_requests WHERE task_id=? ORDER BY created_at DESC LIMIT 100",
                (task_id,),
            ).fetchall()
        return [
            self.read(task_id, row["id"], actor_id)
            for row in rows
            if row["grant_id"] in grants
        ]
