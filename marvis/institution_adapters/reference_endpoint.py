"""Independent loopback protocol endpoint with its own durable state.

It stores native models without deserializing them and serves only rule-only
deployments. These carriers prove local interoperability, never institution
production acceptance. Run separately using this module's CLI.
"""
import argparse
from contextlib import contextmanager
import base64
from datetime import UTC, datetime
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3

from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.evaluator import evaluate_strategy_row
from marvis.reference_decision.contracts import canonical, digest, validate_features
from marvis.risk_context.source_contracts import ProviderQuery, SourceEnvelope, instant

from .contracts import AdapterError, PROTOCOL, RemoteMutation, Hash
from pydantic import TypeAdapter


def write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    with temporary.open("xb") as handle:
        os.chmod(temporary, 0o600)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ReferenceState:
    def __init__(self, root, instance_id, records, drop_once=False, drop_kinds=(), drop_every=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.instance_id = instance_id
        self.drop_once = drop_once
        self.drop_kinds = frozenset(drop_kinds)
        self.drop_every = drop_every
        key = self.root / "access_token"
        if not key.exists():
            write_bytes(key, secrets.token_hex(32).encode())
        self.token = key.read_text().strip()
        self.records = [SourceEnvelope.model_validate(v).model_dump() for v in records]
        self.db = self.root / "remote.sqlite"
        with self.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS identity(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,request_hash TEXT NOT NULL,receipt_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS assets(hash TEXT PRIMARY KEY,manifest_json TEXT NOT NULL,state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS deployments(id TEXT PRIMARY KEY,bundle_hash TEXT NOT NULL,manifest_json TEXT NOT NULL,probe_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS heads(environment TEXT NOT NULL,slot TEXT NOT NULL,active_ref TEXT,PRIMARY KEY(environment,slot));
            CREATE TABLE IF NOT EXISTS effects(id TEXT PRIMARY KEY,kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY,method TEXT NOT NULL,resource TEXT NOT NULL,operation_id TEXT NOT NULL);
            """)
            old = conn.execute("SELECT id FROM identity").fetchone()
            if old and old[0] != instance_id:
                raise AdapterError("reference_instance_id_changed")
            conn.execute("INSERT OR IGNORE INTO identity VALUES(?)", (instance_id,))

    def record_request(self, method, resource, identity):
        with self.connect() as conn:
            conn.execute("INSERT INTO requests(method,resource,operation_id) VALUES(?,?,?)", (method, resource, identity))

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def registry(self, conn, identity):
        row = conn.execute("SELECT * FROM assets WHERE hash=?", (identity,)).fetchone()
        if row is None:
            raise AdapterError("remote_asset_missing", 404)
        manifest = json.loads(row["manifest_json"])
        if digest(manifest) != identity:
            raise AdapterError("remote_asset_manifest_changed")
        verified = []
        for item in manifest["files"]:
            path = self.root / "assets" / identity / item["sha256"]
            data = path.read_bytes()
            if path.is_symlink() or len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise AdapterError("remote_asset_file_changed")
            verified.append(item)
        return {"bundle_hash": identity, "manifest": manifest, "files_verified": verified, "state": row["state"], "storage_only": True}

    def deployment(self, conn, identity):
        row = conn.execute("SELECT * FROM deployments WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise AdapterError("remote_deployment_missing", 404)
        path = self.root / "deployments" / (identity + ".json")
        data = path.read_bytes()
        manifest = json.loads(data)
        if path.is_symlink() or digest(manifest) != row["bundle_hash"] or manifest != json.loads(row["manifest_json"]):
            raise AdapterError("remote_deployment_content_changed")
        spec = parse_strategy_spec(manifest["strategy_spec"])
        probe = json.loads(row["probe_json"])
        validate_features(probe, {"configuration": {"raw_schema": manifest["raw_schema"]}})
        decision = evaluate_strategy_row(probe, spec).to_dict()
        context = manifest["context"]
        head = conn.execute("SELECT active_ref FROM heads WHERE environment=? AND slot=?", (context["environment"], context["deployment_slot"])).fetchone()
        return {"bundle_hash": row["bundle_hash"], "manifest": manifest, "health_status": "healthy", "health_evidence_hash": digest({"bundle_hash": row["bundle_hash"], "probe": probe, "decision": decision}), "probe_decision": decision, "active_ref": None if head is None else head["active_ref"]}

    def apply(self, request):
        value, identity = request.model_dump(), request.operation_id
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM operations WHERE id=?", (identity,)).fetchone()
            if old:
                if old["request_hash"] != digest(value):
                    raise AdapterError("remote_operation_identity_conflict")
                return json.loads(old["receipt_json"]), False
            payload = request.payload
            if request.kind == "registry_upload":
                manifest = payload["manifest"]
                if digest(manifest) != payload["bundle_hash"] or digest(manifest["raw_schema"]) != manifest["schema_hash"] or manifest["storage_only"] is not True:
                    raise AdapterError("remote_registry_manifest_invalid")
                if set(payload["files"]) != {item["name"] for item in manifest["files"]}:
                    raise AdapterError("remote_registry_file_membership_invalid")
                for item in manifest["files"]:
                    TypeAdapter(Hash).validate_python(item["sha256"])
                    data = base64.b64decode(payload["files"][item["name"]], validate=True)
                    if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                        raise AdapterError("remote_registry_file_hash_invalid")
                    write_bytes(self.root / "assets" / payload["bundle_hash"] / item["sha256"], data)
                old_asset = conn.execute("SELECT manifest_json FROM assets WHERE hash=?", (payload["bundle_hash"],)).fetchone()
                if old_asset and json.loads(old_asset[0]) != manifest:
                    raise AdapterError("remote_registry_hash_conflict")
                conn.execute("INSERT OR IGNORE INTO assets VALUES(?,?,'uploaded')", (payload["bundle_hash"], canonical(manifest)))
                result = self.registry(conn, payload["bundle_hash"])
            elif request.kind in {"registry_promote", "registry_revoke"}:
                previous = self.registry(conn, payload["bundle_hash"])
                if previous["manifest"]["schema_hash"] != payload["schema_hash"] or previous["manifest"]["artifact_id"] != payload["model_artifact_id"]:
                    raise AdapterError("remote_registry_transition_binding_invalid")
                if request.kind == "registry_promote" and previous["state"] == "revoked":
                    raise AdapterError("remote_registry_revoked_version_immutable")
                state = "promoted" if request.kind == "registry_promote" else "revoked"
                conn.execute("UPDATE assets SET state=? WHERE hash=?", (state, payload["bundle_hash"]))
                result = {"bundle_hash": payload["bundle_hash"], "schema_hash": payload["schema_hash"], "state": state, "serving": False}
            elif request.kind == "deployment_install":
                manifest = payload["manifest"]
                if manifest["package_kind"] != "rule_only" or digest(manifest) != payload["bundle_hash"] or digest(manifest["raw_schema"]) != manifest["schema_hash"]:
                    raise AdapterError("remote_rule_package_binding_invalid")
                parse_strategy_spec(manifest["strategy_spec"])
                validate_features(payload["probe_features"], {"configuration": {"raw_schema": manifest["raw_schema"]}})
                write_bytes(self.root / "deployments" / (identity + ".json"), canonical(manifest).encode())
                conn.execute("INSERT INTO deployments VALUES(?,?,?,?)", (identity, payload["bundle_hash"], canonical(manifest), canonical(payload["probe_features"])))
                result = {"install_ref": identity, **self.deployment(conn, identity), "state": "installed_not_activated"}
            elif request.kind == "deployment_activate":
                observed = self.deployment(conn, payload["install_operation_id"])
                if observed["bundle_hash"] != payload["bundle_hash"] or observed["manifest"]["context"] != payload["context"]:
                    raise AdapterError("remote_activation_package_binding_invalid")
                context = payload["context"]
                if observed["active_ref"] != payload["expected_current_ref"]:
                    raise AdapterError("remote_active_head_conflict")
                conn.execute("INSERT INTO heads VALUES(?,?,?) ON CONFLICT(environment,slot) DO UPDATE SET active_ref=excluded.active_ref", (context["environment"], context["deployment_slot"], identity))
                result = {"active_ref": identity, "install_ref": payload["install_operation_id"], "bundle_hash": payload["bundle_hash"], "health_status": observed["health_status"], "health_evidence_hash": observed["health_evidence_hash"], "verified_at": datetime.now(UTC).isoformat()}
            elif request.kind == "deployment_rollback":
                context = payload["context"]
                head = conn.execute("SELECT active_ref FROM heads WHERE environment=? AND slot=?", (context["environment"], context["deployment_slot"])).fetchone()
                if head is None or head["active_ref"] != payload["expected_current_ref"]:
                    raise AdapterError("remote_rollback_head_conflict")
                if payload["restore_ref"] is not None:
                    prior = conn.execute("SELECT receipt_json FROM operations WHERE id=?", (payload["restore_ref"],)).fetchone()
                    if prior is None:
                        raise AdapterError("remote_rollback_predecessor_missing")
                    previous = json.loads(prior["receipt_json"])
                    install = previous["result"].get("active_ref")
                    if install != payload["restore_ref"] or previous["kind"] != "deployment_activate":
                        raise AdapterError("remote_rollback_predecessor_invalid")
                    observed = self.deployment(conn, previous["result"]["install_ref"])
                    prior_context = observed["manifest"]["context"]
                    if prior_context["environment"] != context["environment"] or prior_context["deployment_slot"] != context["deployment_slot"]:
                        raise AdapterError("remote_rollback_predecessor_scope_invalid")
                conn.execute("UPDATE heads SET active_ref=? WHERE environment=? AND slot=?", (payload["restore_ref"], context["environment"], context["deployment_slot"]))
                result = {"active_ref": payload["restore_ref"], "rolled_back_ref": payload["expected_current_ref"]}
            elif request.kind == "source_query":
                query = ProviderQuery.model_validate(payload["query"])
                if instant(query.expires_at) <= datetime.now(UTC).timestamp():
                    raise AdapterError("remote_source_query_expired")
                matches = [v for v in self.records if v["subject_namespace"] == query.subject_namespace and v["subject_token"] == query.subject_token]
                if len(matches) > 1:
                    raise AdapterError("remote_source_record_not_unique")
                result = {"response": {"request_id": query.request_id, "request_hash": digest(query.model_dump()), "status": "found" if matches else "no_record", "envelope": matches[0] if matches else None}, "recorded_at": datetime.now(UTC).isoformat()}
            else:
                raise AdapterError("remote_operation_kind_unsupported")
            receipt = {"protocol": PROTOCOL, "instance_id": self.instance_id, "operation_id": identity, "request_hash": digest(value), "kind": request.kind, "state": "completed", "result": result}
            conn.execute("INSERT INTO operations VALUES(?,?,?)", (identity, digest(value), canonical(receipt)))
            conn.execute("INSERT INTO effects VALUES(?,?)", (identity, request.kind))
        return receipt, True


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body):
            data = canonical(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + state.token)

        def do_GET(self):
            if not self.authorized():
                return self.reply(403, {"code": "unauthorized"})
            if self.path == "/v1/adapter/capabilities":
                return self.reply(200, {"protocol": PROTOCOL, "instance_id": state.instance_id, "registry_execution": "opaque_native_storage_only", "deployment_execution": "rule_only"})
            parts = self.path.split("/")
            if len(parts) != 5 or parts[:3] != ["", "v1", "adapter"]:
                return self.reply(404, {"code": "fixed_route_required"})
            resource, identity = parts[3:]
            state.record_request("GET", resource, identity)
            try:
                with state.connect() as conn:
                    if resource == "operations":
                        row = conn.execute("SELECT receipt_json FROM operations WHERE id=?", (identity,)).fetchone()
                        if row is None:
                            return self.reply(404, {"code": "unknown_effect"})
                        return self.reply(200, json.loads(row[0]))
                    if resource not in {"registry", "deployment"}:
                        return self.reply(404, {"code": "fixed_route_required"})
                    value = state.registry(conn, identity) if resource == "registry" else state.deployment(conn, identity)
                    return self.reply(200, {"protocol": PROTOCOL, "instance_id": state.instance_id, "resource": resource, "identity": identity, "payload": value})
            except Exception:
                return self.reply(409, {"code": "readback_verification_failed"})

        def do_POST(self):
            if not self.authorized():
                return self.reply(403, {"code": "unauthorized"})
            if self.path != "/v1/adapter/operations":
                return self.reply(404, {"code": "fixed_route_required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 64_000_000:
                    return self.reply(413, {"code": "byte_budget"})
                request = RemoteMutation.model_validate_json(self.rfile.read(size))
                state.record_request("POST", "operations", request.operation_id)
                receipt, fresh = state.apply(request)
                marker = state.root / ("dropped-operation-" + digest(request.operation_id) if state.drop_every else "dropped-" + request.kind if request.kind in state.drop_kinds else "dropped_once")
                if fresh and (state.drop_every or state.drop_once or request.kind in state.drop_kinds) and not marker.exists():
                    write_bytes(marker, b"committed response intentionally lost")
                    self.close_connection = True
                    return
                self.reply(201 if fresh else 200, receipt)
            except Exception:
                self.reply(422, {"code": "operation_rejected"})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--drop-after-commit-once", action="store_true")
    parser.add_argument("--drop-after-commit-kinds", default="")
    parser.add_argument("--drop-every-new-operation-response", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.records.read_bytes()) if args.records else []
    kinds = tuple(v for v in args.drop_after_commit_kinds.split(",") if v)
    valid = {"registry_upload", "registry_promote", "registry_revoke", "deployment_install", "deployment_activate", "deployment_rollback", "source_query"}
    if set(kinds) - valid:
        parser.error("unsupported fixed fault kind")
    state = ReferenceState(args.state_dir, args.instance_id, records, args.drop_after_commit_once, kinds, args.drop_every_new_operation_response)
    HTTPServer(("127.0.0.1", args.port), handler_for(state)).serve_forever()


if __name__ == "__main__":
    main()
