"""Server-issued source permissions and durable request admission."""

import hashlib
import hmac
import json
from pathlib import Path
import time
import uuid

from marvis.db_schema import connect
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.risk_context.source_contracts import (
    SourceError,
    SourceGrant,
    SourceProfile,
    canonical_json,
    content_hash,
    instant,
)
from marvis.risk_context.source_schema import ensure_source_schema


def require_actor(conn, actor_id, roles):
    row = conn.execute(
        """SELECT p.role,l.expires_at FROM production_principals p JOIN local_principals l
        ON p.local_principal_id=l.id WHERE l.id=? AND l.status='active' AND p.status='active'""",
        (actor_id,),
    ).fetchone()
    if not row or row["role"] not in roles or instant(row["expires_at"]) <= time.time():
        raise SourceError("source_role_forbidden", 403)
    return row["role"]


class SourceRepository:
    def __init__(self, settings):
        self.settings = settings
        self.db_path = settings.db_path
        ensure_source_schema(self.db_path)
        self.secret = settings.plugin_admin_token_path.read_text().strip().encode()
        if not self.secret:
            raise SourceError("source_authentication_unavailable")
        self.private = settings.workspace / "source_private"
        self.private.mkdir(mode=0o700, parents=True, exist_ok=True)

    def sign(self, body):
        return hmac.new(
            self.secret, canonical_json(body).encode(), hashlib.sha256
        ).hexdigest()

    def register_profile(self, request, actor_id):
        profile = request.profile
        token = request.token.get_secret_value() if request.token else ""
        if profile.mode == "reference_http" and (
            len(token) < 32
            or len(token) > 256
            or not token.isascii()
            or any(c.isspace() for c in token)
        ):
            raise SourceError("invalid_reference_credential", 422)
        if profile.mode == "historical_file" and token:
            raise SourceError("historical_profile_has_no_credential", 422)
        payload = canonical_json(profile.model_dump())
        token_path = self.private / f"credential-{content_hash(profile.profile_id)}"
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            old = conn.execute(
                "SELECT payload FROM source_profiles WHERE id=?", (profile.profile_id,)
            ).fetchone()
            if old:
                if old["payload"] != payload or (
                    token
                    and (
                        not token_path.exists()
                        or not hmac.compare_digest(token_path.read_text(), token)
                    )
                ):
                    raise SourceError("profile_version_conflict")
            else:
                # O_EXCL prevents an orphaned secret from being silently replaced.
                if token:
                    if token_path.exists():
                        if not hmac.compare_digest(token_path.read_text(), token):
                            raise SourceError("profile_credential_conflict")
                    else:
                        with token_path.open("x") as stream:
                            token_path.chmod(0o600)
                            stream.write(token)
                conn.execute(
                    "INSERT INTO source_profiles(id,payload,created_by,created_at) VALUES(?,?,?,?)",
                    (profile.profile_id, payload, actor_id, time.time()),
                )
        return profile.model_dump()

    def profile_tx(self, conn, profile_id):
        row = conn.execute(
            "SELECT * FROM source_profiles WHERE id=?", (profile_id,)
        ).fetchone()
        if row is None:
            raise SourceError("profile_not_found", 404)
        return row, SourceProfile.model_validate_json(row["payload"])

    def create_grant(self, grant: SourceGrant, actor_id):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            require_actor(conn, grant.grantee_id, {"maker"})
            self.profile_tx(conn, grant.profile_id)
            basis = conn.execute(
                "SELECT * FROM task_artifacts WHERE task_id=? AND id=? AND kind='source_authorization_basis'",
                (grant.task_id, grant.basis_artifact_id),
            ).fetchone()
            if not basis:
                raise SourceError("authorization_basis_required", 422)
            basis_path = Path(basis["path"])
            if (
                basis_path.parent.resolve()
                != (
                    self.settings.tasks_dir / grant.task_id / "source_evidence"
                ).resolve()
            ):
                raise SourceError("authorization_basis_binding_failed")
            basis_data = basis_path.read_bytes()
            if hashlib.sha256(basis_data).hexdigest() != basis["content_hash"]:
                raise SourceError("authorization_basis_integrity_failed")
            basis_envelope = json.loads(basis_data)
            if basis_envelope["body"][
                "task_id"
            ] != grant.task_id or not hmac.compare_digest(
                basis_envelope["signature"], self.sign(basis_envelope["body"])
            ):
                raise SourceError("authorization_basis_binding_failed")
            if instant(grant.expires_at) <= time.time():
                raise SourceError("grant_expired", 422)
            payload = canonical_json(grant.model_dump())
            old = conn.execute(
                "SELECT payload,revoked_at FROM source_grants WHERE id=?",
                (grant.grant_id,),
            ).fetchone()
            if old and (old["payload"] != payload or old["revoked_at"] is not None):
                raise SourceError("grant_version_conflict")
            if not old:
                conn.execute(
                    "INSERT INTO source_grants(id,task_id,payload,created_by,created_at) VALUES(?,?,?,?,?)",
                    (grant.grant_id, grant.task_id, payload, actor_id, time.time()),
                )
        return {
            "grant_id": grant.grant_id,
            "task_id": grant.task_id,
            "profile_id": grant.profile_id,
            "grantee_id": grant.grantee_id,
            "expires_at": grant.expires_at,
            "state": "active",
            "external_authorization_assurance": "admin_declared_not_independently_verified",
        }

    def revoke_grant(self, grant_id, actor_id):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            require_actor(conn, actor_id, {"admin"})
            if not conn.execute(
                "SELECT 1 FROM source_grants WHERE id=?", (grant_id,)
            ).fetchone():
                raise SourceError("grant_not_found", 404)
            conn.execute(
                "UPDATE source_grants SET revoked_at=COALESCE(revoked_at,?),revoked_by=COALESCE(revoked_by,?) WHERE id=?",
                (time.time(), actor_id, grant_id),
            )
        return {"grant_id": grant_id, "state": "revoked"}

    def grant_tx(self, conn, task_id, grant_id, actor_id, *, active=True):
        role = require_actor(conn, actor_id, {"maker", "admin"})
        row = conn.execute(
            "SELECT * FROM source_grants WHERE task_id=? AND id=?", (task_id, grant_id)
        ).fetchone()
        if row is None:
            raise SourceError("grant_not_found", 404)
        grant = SourceGrant.model_validate_json(row["payload"])
        if role != "admin" and actor_id != grant.grantee_id:
            raise SourceError("grant_scope_forbidden", 403)
        if active:
            now = time.time()
            if row["revoked_at"] is not None:
                raise SourceError("grant_revoked", 403)
            if not instant(grant.starts_at) <= now < instant(grant.expires_at):
                raise SourceError("grant_not_current", 403)
            require_actor(conn, grant.grantee_id, {"maker"})
        return grant

    def get_tx(self, conn, task_id, request_id):
        row = conn.execute(
            "SELECT * FROM source_requests WHERE task_id=? AND id=?",
            (task_id, request_id),
        ).fetchone()
        if row is None:
            raise SourceError("source_request_not_found", 404)
        body = json.loads(row["contract"])
        if content_hash(body) != row["contract_hash"] or not hmac.compare_digest(
            row["signature"], self.sign(body)
        ):
            raise SourceError("source_request_integrity_failed")
        if (
            body["task_id"] != task_id
            or body["query"]["request_id"] != request_id
            or body["actor_id"] != row["actor_id"]
            or body["profile"]["profile_id"] != row["profile_id"]
            or body["grant"]["grant_id"] != row["grant_id"]
            or instant(body["expires_at"]) != row["expires_at"]
        ):
            raise SourceError("source_request_binding_failed")
        return dict(row), body

    def claim(self, task_id, request_id):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row, body = self.get_tx(conn, task_id, request_id)
            self.grant_tx(conn, task_id, row["grant_id"], row["actor_id"])
            if row["state"] == "completed":
                return None, row, body, None
            now = time.time()
            if row["state"] == "running" and row["lease_until"] > now:
                raise SourceError("source_request_in_progress", 409)
            # Any previously started HTTP request is recoverable only by GET.
            readback = row["state"] in {"running", "unknown_effect"}
            if not readback and row["expires_at"] <= now:
                raise SourceError("source_request_expired", 410)
            profile_row, profile = self.profile_tx(conn, row["profile_id"])
            owner = uuid.uuid4().hex
            if profile.mode == "reference_http":
                if profile_row["open_until"] > now:
                    raise SourceError("source_circuit_open", 429)
                count = conn.execute(
                    "SELECT count(*) FROM source_attempts WHERE profile_id=? AND started_at>?",
                    (row["profile_id"], now - 60),
                ).fetchone()[0]
                if count >= profile.requests_per_minute:
                    raise SourceError("source_rate_limited", 429)
            method = (
                ("GET" if readback else "POST")
                if profile.mode == "reference_http"
                else "IMPORT"
            )
            conn.execute(
                "INSERT INTO source_attempts(id,task_id,request_id,profile_id,method,started_at) VALUES(?,?,?,?,?,?)",
                (owner, task_id, request_id, row["profile_id"], method, now),
            )
            conn.execute(
                "UPDATE source_requests SET state='running',owner=?,lease_until=?,error_code=NULL WHERE task_id=? AND id=?",
                (owner, now + 30, task_id, request_id),
            )
        return owner, row, body, readback

    def finish_error(self, task_id, request_id, owner, code, *, uncertain):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row, _ = self.get_tx(conn, task_id, request_id)
            if row["owner"] != owner or row["state"] != "running":
                raise SourceError("source_owner_superseded")
            profile_row, profile = self.profile_tx(conn, row["profile_id"])
            failures = profile_row["failures"] + 1
            now = time.time()
            conn.execute(
                "UPDATE source_profiles SET failures=?,open_until=? WHERE id=?",
                (
                    failures,
                    now + profile.breaker_seconds
                    if failures >= profile.breaker_threshold
                    else 0,
                    row["profile_id"],
                ),
            )
            conn.execute(
                "UPDATE source_attempts SET finished_at=?,outcome=? WHERE id=?",
                (now, code, owner),
            )
            conn.execute(
                "UPDATE source_requests SET state=?,error_code=?,owner=NULL,lease_until=NULL WHERE task_id=? AND id=?",
                (
                    "unknown_effect" if uncertain else "completed",
                    code,
                    task_id,
                    request_id,
                ),
            )

    def list_profiles(self, actor_id):
        with connect(self.db_path) as conn:
            require_actor(conn, actor_id, {"maker", "checker", "admin"})
            return [
                json.loads(row["payload"])
                for row in conn.execute(
                    "SELECT payload FROM source_profiles ORDER BY created_at,id"
                )
            ]

    def list_grants(self, task_id, actor_id):
        with connect(self.db_path) as conn:
            role = require_actor(conn, actor_id, {"maker", "admin"})
            rows = conn.execute(
                "SELECT * FROM source_grants WHERE task_id=? ORDER BY created_at DESC LIMIT 100",
                (task_id,),
            ).fetchall()
        items = []
        for row in rows:
            grant = SourceGrant.model_validate_json(row["payload"])
            if role != "admin" and grant.grantee_id != actor_id:
                continue
            items.append(
                {
                    "grant_id": grant.grant_id,
                    "profile_id": grant.profile_id,
                    "task_id": task_id,
                    "purpose": grant.purpose,
                    "expires_at": grant.expires_at,
                    "state": "revoked"
                    if row["revoked_at"]
                    else (
                        "active"
                        if instant(grant.starts_at)
                        <= time.time()
                        < instant(grant.expires_at)
                        else "not_current"
                    ),
                }
            )
        return items

    @property
    def artifacts(self):
        return TaskArtifactRepository(self.db_path)
