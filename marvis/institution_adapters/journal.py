"""Durable claims precede network I/O; every retry after uncertainty is GET-only."""
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import secrets

from marvis.reference_decision.contracts import canonical, digest

from .contracts import AdapterError


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory).absolute()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.resolve() != self.directory:
            raise AdapterError("adapter_journal_path_invalid", 503)
        key = self.directory / "authority.key"
        try:
            fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(secrets.token_bytes(32))
                handle.flush()
                os.fsync(handle.fileno())
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except FileExistsError:
            pass
        if key.is_symlink() or key.stat().st_mode & 0o077:
            raise AdapterError("adapter_authority_permissions_invalid", 503)
        self.secret = key.read_bytes()
        if len(self.secret) != 32:
            raise AdapterError("adapter_authority_unavailable", 503)
        self.path = self.directory / "operations.sqlite"
        with self.connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS operations(alias TEXT NOT NULL,id TEXT NOT NULL,task_id TEXT NOT NULL,actor_id TEXT NOT NULL,
                request_json TEXT NOT NULL,request_hash TEXT NOT NULL,request_mac TEXT NOT NULL,status TEXT NOT NULL,
                receipt_json TEXT,receipt_mac TEXT,completion_json TEXT,completion_mac TEXT,PRIMARY KEY(alias,id));
            CREATE TABLE IF NOT EXISTS grants(alias TEXT NOT NULL,id TEXT NOT NULL,body_json TEXT NOT NULL,body_mac TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(alias,id));
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        if self.path.is_symlink():
            raise AdapterError("adapter_journal_path_invalid", 503)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def sign(self, value):
        return hmac.new(self.secret, canonical(value).encode(), hashlib.sha256).hexdigest()

    def claim(self, alias, task_id, actor_id, request):
        value = request.model_dump()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM operations WHERE alias=? AND id=?", (alias, request.operation_id)).fetchone()
            if old:
                body = self.decode(old)
                if old["task_id"] != task_id or old["actor_id"] != actor_id or body != value:
                    raise AdapterError("operation_identity_conflict")
                return False
            signature = self.sign({"alias": alias, "task_id": task_id, "actor_id": actor_id, "request": value})
            conn.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,'claimed',NULL,NULL,NULL,NULL)", (alias, request.operation_id, task_id, actor_id, canonical(value), digest(value), signature))
        return True

    def decode(self, row):
        body = json.loads(row["request_json"])
        bound = {"alias": row["alias"], "task_id": row["task_id"], "actor_id": row["actor_id"], "request": body}
        if digest(body) != row["request_hash"] or not hmac.compare_digest(self.sign(bound), row["request_mac"]):
            raise AdapterError("operation_journal_integrity_failed")
        return body

    def read(self, alias, identity):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM operations WHERE alias=? AND id=?", (alias, identity)).fetchone()
        if row is None:
            raise AdapterError("operation_not_found", 404)
        self.decode(row)
        return dict(row)

    def uncertain(self, alias, identity):
        with self.connection() as conn:
            conn.execute("UPDATE operations SET status='unknown' WHERE alias=? AND id=? AND receipt_json IS NULL", (alias, identity))

    def receipt(self, alias, identity, value):
        body = value.model_dump()
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM operations WHERE alias=? AND id=?", (alias, identity)).fetchone()
            original = self.decode(row)
            if body["operation_id"] != identity or body["request_hash"] != digest(original) or body["kind"] != original["kind"]:
                raise AdapterError("remote_receipt_binding_mismatch")
            if row["receipt_json"] is not None and json.loads(row["receipt_json"]) != body:
                raise AdapterError("remote_receipt_drift")
            conn.execute("UPDATE operations SET status=?,receipt_json=?,receipt_mac=? WHERE alias=? AND id=?", ("completed" if body["state"] == "completed" else "rejected", canonical(body), self.sign(body), alias, identity))

    def result(self, row):
        if row["receipt_json"] is None:
            raise AdapterError("remote_effect_unknown")
        value = json.loads(row["receipt_json"])
        if not hmac.compare_digest(self.sign(value), row["receipt_mac"]):
            raise AdapterError("operation_receipt_integrity_failed")
        return value

    def finalize(self, alias, identity, completion):
        bound = {"alias": alias, "operation_id": identity, "completion": completion}
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM operations WHERE alias=? AND id=?", (alias, identity)).fetchone()
            self.result(row)
            if row["completion_json"] is not None and json.loads(row["completion_json"]) != completion:
                raise AdapterError("local_completion_drift")
            conn.execute("UPDATE operations SET completion_json=?,completion_mac=? WHERE alias=? AND id=?", (canonical(completion), self.sign(bound), alias, identity))

    def completion(self, row):
        if row["completion_json"] is None:
            return None
        value = json.loads(row["completion_json"])
        bound = {"alias": row["alias"], "operation_id": row["id"], "completion": value}
        if not hmac.compare_digest(self.sign(bound), row["completion_mac"]):
            raise AdapterError("local_completion_integrity_failed")
        return value

    def grant(self, alias, identity, body):
        bound = {"alias": alias, "grant_id": identity, "body": body, "revoked": 0}
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM grants WHERE alias=? AND id=?", (alias, identity)).fetchone()
            if old:
                current = self.read_grant(alias, identity)
                if current != body or old["revoked"]:
                    raise AdapterError("source_grant_identity_conflict")
                return
            conn.execute("INSERT INTO grants(alias,id,body_json,body_mac) VALUES(?,?,?,?)", (alias, identity, canonical(body), self.sign(bound)))

    def read_grant(self, alias, identity, *, allow_revoked=False):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM grants WHERE alias=? AND id=?", (alias, identity)).fetchone()
        if row is None:
            raise AdapterError("source_grant_unavailable", 403)
        value = json.loads(row["body_json"])
        bound = {"alias": alias, "grant_id": identity, "body": value, "revoked": row["revoked"]}
        if not hmac.compare_digest(self.sign(bound), row["body_mac"]):
            raise AdapterError("source_grant_integrity_failed")
        if row["revoked"] and not allow_revoked:
            raise AdapterError("source_grant_unavailable", 403)
        return value

    def revoke_grant(self, alias, identity):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM grants WHERE alias=? AND id=?", (alias, identity)).fetchone()
            if row is None:
                raise AdapterError("source_grant_not_found", 404)
            body = json.loads(row["body_json"])
            previous = {"alias": alias, "grant_id": identity, "body": body, "revoked": row["revoked"]}
            if not hmac.compare_digest(self.sign(previous), row["body_mac"]):
                raise AdapterError("source_grant_integrity_failed")
            bound = {"alias": alias, "grant_id": identity, "body": body, "revoked": 1}
            conn.execute("UPDATE grants SET revoked=1,body_mac=? WHERE alias=? AND id=?", (self.sign(bound), alias, identity))
