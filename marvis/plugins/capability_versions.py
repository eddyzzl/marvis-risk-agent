"""Publisher custody and immutable Plugin/Workflow versions in the platform DB."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from marvis.db_schema import connect
from marvis.plugins.publisher_trust import (
    CapabilityError,
    canonical,
    decode_key,
    digest,
    identifier,
    read_json,
    require,
    tree_hash,
    verify_signature,
)
from marvis.repositories.audit import _write_audit_row


def create_capability_schema(conn):
    statements = [
        """CREATE TABLE IF NOT EXISTS capability_settings(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            mode TEXT NOT NULL CHECK(mode IN ('trusted_local','governed')),
            revision INTEGER NOT NULL DEFAULT 0)""",
        "INSERT OR IGNORE INTO capability_settings VALUES(1,'trusted_local',0)",
        """CREATE TABLE IF NOT EXISTS capability_publishers(
            id TEXT PRIMARY KEY, display_name TEXT NOT NULL, public_key TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE, created_by TEXT NOT NULL, created_at REAL NOT NULL,
            revoked_at REAL, revoked_by TEXT, reason TEXT)""",
        """CREATE TABLE IF NOT EXISTS capability_versions(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('plugin','workflow')),
            name TEXT NOT NULL, version TEXT NOT NULL, publisher_id TEXT NOT NULL,
            content_hash TEXT NOT NULL, declaration_hash TEXT NOT NULL,
            envelope_json TEXT NOT NULL, declaration_json TEXT NOT NULL,
            permissions_json TEXT NOT NULL, material_path TEXT NOT NULL,
            installed_by TEXT NOT NULL, installed_at REAL NOT NULL,
            UNIQUE(kind,name,version), UNIQUE(kind,name,content_hash),
            FOREIGN KEY(publisher_id) REFERENCES capability_publishers(id))""",
        """CREATE TABLE IF NOT EXISTS capability_heads(
            kind TEXT NOT NULL, name TEXT NOT NULL, version_id TEXT NOT NULL,
            revision INTEGER NOT NULL, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
            PRIMARY KEY(kind,name), FOREIGN KEY(version_id) REFERENCES capability_versions(id))""",
        """CREATE TABLE IF NOT EXISTS capability_proposals(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
            version TEXT NOT NULL, publisher_id TEXT NOT NULL, content_hash TEXT NOT NULL,
            declaration_hash TEXT NOT NULL, envelope_json TEXT NOT NULL,
            declaration_json TEXT NOT NULL, permissions_json TEXT NOT NULL,
            material_path TEXT NOT NULL, expected_revision INTEGER NOT NULL,
            delta_json TEXT NOT NULL, actor TEXT NOT NULL, proposal_hash TEXT NOT NULL,
            created_at REAL NOT NULL, expires_at REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','confirmed')),
            receipt_json TEXT, confirmed_by TEXT)""",
        """CREATE TABLE IF NOT EXISTS capability_plan_bindings(
            plan_id TEXT PRIMARY KEY, workflow_version_id TEXT NOT NULL,
            FOREIGN KEY(plan_id) REFERENCES plans(id),
            FOREIGN KEY(workflow_version_id) REFERENCES capability_versions(id))""",
    ]
    for statement in statements:
        conn.execute(statement)
    for table in ("capability_versions", "capability_plan_bindings"):
        for action in ("UPDATE", "DELETE"):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_immutable "
                f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'capability immutable'); END"
            )
    conn.execute("""CREATE TRIGGER IF NOT EXISTS capability_publishers_key_immutable
        BEFORE UPDATE ON capability_publishers WHEN NEW.id IS NOT OLD.id
          OR NEW.public_key IS NOT OLD.public_key OR NEW.key_hash IS NOT OLD.key_hash
          OR NEW.created_by IS NOT OLD.created_by OR NEW.created_at IS NOT OLD.created_at
          OR (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS NOT OLD.revoked_at)
        BEGIN SELECT RAISE(ABORT,'publisher key immutable'); END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS capability_publishers_no_delete
        BEFORE DELETE ON capability_publishers BEGIN SELECT RAISE(ABORT,'publisher immutable'); END""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS capability_proposals_binding_immutable
        BEFORE UPDATE ON capability_proposals WHEN NEW.kind IS NOT OLD.kind
          OR NEW.name IS NOT OLD.name OR NEW.version IS NOT OLD.version
          OR NEW.publisher_id IS NOT OLD.publisher_id OR NEW.content_hash IS NOT OLD.content_hash
          OR NEW.declaration_hash IS NOT OLD.declaration_hash OR NEW.envelope_json IS NOT OLD.envelope_json
          OR NEW.declaration_json IS NOT OLD.declaration_json OR NEW.permissions_json IS NOT OLD.permissions_json
          OR NEW.material_path IS NOT OLD.material_path OR NEW.expected_revision IS NOT OLD.expected_revision
          OR NEW.delta_json IS NOT OLD.delta_json OR NEW.actor IS NOT OLD.actor
          OR NEW.proposal_hash IS NOT OLD.proposal_hash OR NEW.expires_at IS NOT OLD.expires_at
          OR OLD.status='confirmed'
        BEGIN SELECT RAISE(ABORT,'proposal binding immutable'); END""")


def _has_schema(conn):
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='capability_settings'"
        ).fetchone()
        is not None
    )


def capability_mode(conn) -> str:
    from marvis.institution.store import identity_mode
    if identity_mode(conn) == 'institutional':
        return 'governed'
    if not _has_schema(conn):
        return "trusted_local"
    return conn.execute(
        "SELECT mode FROM capability_settings WHERE singleton=1"
    ).fetchone()[0]


def governed_head(conn, kind, name):
    if not _has_schema(conn):
        return None
    return conn.execute(
        "SELECT * FROM capability_heads WHERE kind=? AND name=?", (kind, name)
    ).fetchone()


def require_legacy_registration_allowed(conn, manifest):
    if manifest.builtin:
        return
    require(
        capability_mode(conn) != "governed", "signed_capability_required", status=403
    )
    require(
        governed_head(conn, "plugin", manifest.name) is None,
        "governed_plugin_requires_version_proposal",
        status=403,
    )


def _audit(conn, kind, name, actor, detail):
    _write_audit_row(
        conn,
        kind="capability." + kind,
        target_ref=name,
        actor=actor,
        outcome="succeeded",
        detail=detail,
    )


class CapabilityStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.root = self.db_path.parent / ".capability_versions"

    def settings(self):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT mode,revision FROM capability_settings WHERE singleton=1"
            ).fetchone()
            return dict(row)

    def set_mode(self, mode, expected_revision, actor, reason):
        require(
            mode in {"trusted_local", "governed"}, "capability_invalid_mode", status=422
        )
        require(
            isinstance(expected_revision, int)
            and not isinstance(expected_revision, bool),
            "capability_invalid_revision",
            status=422,
        )
        require(
            isinstance(reason, str) and 0 < len(reason.strip()) <= 2000,
            "capability_reason_required",
            status=422,
        )
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            from marvis.institution.store import identity_mode
            require(mode != 'trusted_local' or identity_mode(conn) != 'institutional',
                    'institution_requires_signed_capabilities', status=403)
            current = conn.execute(
                "SELECT mode,revision FROM capability_settings WHERE singleton=1"
            ).fetchone()
            require(
                current["revision"] == expected_revision, "capability_mode_conflict"
            )
            if mode == "governed":
                unsigned = conn.execute("""SELECT name FROM plugins WHERE builtin=0 AND enabled=1
                    AND NOT EXISTS(SELECT 1 FROM capability_heads h WHERE h.kind='plugin' AND h.name=plugins.name)""").fetchall()
                require(not unsigned, "unsigned_enabled_plugins_must_be_disabled")
                skills = self.db_path.parent / "skills"
                for path in skills.glob("*.json"):
                    value = read_json(path)
                    if value.get("enabled", True) is not False:
                        require(
                            governed_head(
                                conn, "workflow", str(value.get("id", path.stem))
                            )
                            is not None,
                            "unsigned_active_workflows_must_be_disabled",
                        )
                for head in conn.execute(
                    "SELECT * FROM capability_heads WHERE enabled=1"
                ).fetchall():
                    self._version(conn, head["version_id"])
            conn.execute(
                "UPDATE capability_settings SET mode=?,revision=revision+1 WHERE singleton=1",
                (mode,),
            )
            _audit(
                conn,
                "mode",
                mode,
                actor,
                {
                    "reason": reason,
                    "previous": current["mode"],
                    "revision": expected_revision + 1,
                },
            )
        return self.settings()

    def add_publisher(self, publisher_id, display_name, public_key, actor):
        import hashlib

        identifier(publisher_id, "publisher")
        raw = decode_key(public_key)
        require(
            isinstance(display_name, str) and 0 < len(display_name.strip()) <= 120,
            "publisher_invalid_display_name",
            status=422,
        )
        with connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO capability_publishers(id,display_name,public_key,key_hash,created_by,created_at) VALUES(?,?,?,?,?,?)",
                    (
                        publisher_id,
                        display_name.strip(),
                        public_key,
                        hashlib.sha256(raw).hexdigest(),
                        actor,
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise CapabilityError("publisher_identity_or_key_exists") from exc
            _audit(
                conn,
                "publisher.add",
                publisher_id,
                actor,
                {"key_hash": hashlib.sha256(raw).hexdigest()},
            )
        return {
            "id": publisher_id,
            "display_name": display_name.strip(),
            "status": "trusted",
        }

    def publishers(self):
        with connect(self.db_path) as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT id,display_name,key_hash,revoked_at FROM capability_publishers ORDER BY id"
                )
            ]

    def revoke(self, publisher_id, actor, reason):
        require(
            isinstance(reason, str) and 0 < len(reason.strip()) <= 2000,
            "capability_reason_required",
            status=422,
        )
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._publisher(conn, publisher_id)
            require(row["revoked_at"] is None, "publisher_already_revoked")
            conn.execute(
                "UPDATE capability_publishers SET revoked_at=?,revoked_by=?,reason=? WHERE id=?",
                (time.time(), actor, reason, publisher_id),
            )
            _audit(conn, "publisher.revoke", publisher_id, actor, {"reason": reason})
        return {"id": publisher_id, "status": "revoked"}

    def _publisher(self, conn, publisher_id):
        row = conn.execute(
            "SELECT * FROM capability_publishers WHERE id=?", (publisher_id,)
        ).fetchone()
        require(row is not None, "publisher_not_trusted", status=403)
        return row

    def verify(self, envelope):
        require(
            isinstance(envelope, dict) and isinstance(envelope.get("body"), dict),
            "publisher_signature_missing",
            status=422,
        )
        with connect(self.db_path) as conn:
            publisher = self._publisher(conn, envelope["body"].get("publisher_id"))
            require(publisher["revoked_at"] is None, "publisher_revoked", status=403)
            return verify_signature(envelope, publisher["public_key"])

    def head(self, kind, name):
        with connect(self.db_path) as conn:
            row = governed_head(conn, kind, name)
            return dict(row) if row else None

    def managed(self, kind, name):
        return self.head(kind, name) is not None

    def versions(self):
        with connect(self.db_path) as conn:
            return [
                dict(row)
                for row in conn.execute("""SELECT v.id,v.kind,v.name,v.version,
                v.publisher_id,v.content_hash,v.declaration_hash,v.permissions_json,
                h.revision AS head_revision,h.enabled,h.version_id=v.id AS active,
                p.revoked_at IS NOT NULL AS publisher_revoked
                FROM capability_versions v JOIN capability_publishers p ON p.id=v.publisher_id
                LEFT JOIN capability_heads h ON h.kind=v.kind AND h.name=v.name
                ORDER BY v.kind,v.name,v.installed_at DESC""")
            ]

    def _version(self, conn, version_id, *, material=True):
        from marvis.institution.synchronization import authorize_synced_ref
        authorize_synced_ref(conn,'workflow:'+version_id)
        row = conn.execute(
            "SELECT * FROM capability_versions WHERE id=?", (version_id,)
        ).fetchone()
        require(row is not None, "capability_version_not_found", status=404)
        publisher = self._publisher(conn, row["publisher_id"])
        require(publisher["revoked_at"] is None, "publisher_revoked", status=403)
        envelope = json.loads(row["envelope_json"])
        body = verify_signature(envelope, publisher["public_key"])
        require(
            body["kind"] == row["kind"]
            and body["name"] == row["name"]
            and body["version"] == row["version"]
            and body["content_hash"] == row["content_hash"]
            and body["declaration_hash"] == row["declaration_hash"],
            "capability_version_identity_changed",
        )
        require(
            digest(json.loads(row["declaration_json"])) == row["declaration_hash"],
            "capability_declaration_changed",
        )
        if material:
            path = Path(row["material_path"])
            require(
                path.is_relative_to(self.root)
                and path.resolve().is_relative_to(self.root.resolve()),
                "capability_material_path_changed",
            )
            require(
                tree_hash(path) == row["content_hash"], "capability_material_changed"
            )
            require(
                read_json(path / "publisher-signature.json") == envelope,
                "capability_signature_changed",
            )
        return dict(row)

    def resolve(self, kind, name, release_version=""):
        with connect(self.db_path) as conn:
            head = governed_head(conn, kind, name)
            require(head is not None, "signed_capability_required", status=403)
            require(head["enabled"], "capability_disabled", status=403)
            if release_version:
                row = conn.execute(
                    "SELECT id FROM capability_versions WHERE kind=? AND name=? AND version=?",
                    (kind, name, release_version),
                ).fetchone()
                require(row is not None, "capability_version_not_found", status=404)
                return self._version(conn, row["id"])
            return self._version(conn, head["version_id"])

    def propose(
        self,
        *,
        body,
        envelope,
        declaration,
        permissions,
        material_path,
        expected_revision,
        actor,
    ):
        self.verify(envelope)
        require(
            body == envelope["body"]
            and digest(declaration) == body["declaration_hash"]
            and tree_hash(material_path) == body["content_hash"],
            "capability_binding_changed",
        )
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            head = governed_head(conn, body["kind"], body["name"])
            revision = head["revision"] if head else 0
            require(revision == expected_revision, "capability_head_conflict")
            old = (
                json.loads(
                    conn.execute(
                        "SELECT declaration_json FROM capability_versions WHERE id=?",
                        (head["version_id"],),
                    ).fetchone()[0]
                )
                if head
                else {}
            )
            old_permissions = (
                set(
                    json.loads(
                        conn.execute(
                            "SELECT permissions_json FROM capability_versions WHERE id=?",
                            (head["version_id"],),
                        ).fetchone()[0]
                    )
                )
                if head
                else set()
            )
            delta = {
                "added_permissions": sorted(set(permissions) - old_permissions),
                "removed_permissions": sorted(old_permissions - set(permissions)),
                "declaration_changed": old != declaration,
                "previous_declaration": old,
            }
            proposal_id = str(uuid.uuid4())
            binding = {
                "body": body,
                "expected_revision": revision,
                "delta": delta,
                "actor": actor,
            }
            proposal_hash = digest(binding)
            now = time.time()
            conn.execute(
                """INSERT INTO capability_proposals(id,kind,name,version,publisher_id,content_hash,
                declaration_hash,envelope_json,declaration_json,permissions_json,material_path,expected_revision,
                delta_json,actor,proposal_hash,created_at,expires_at,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')""",
                (
                    proposal_id,
                    body["kind"],
                    body["name"],
                    body["version"],
                    body["publisher_id"],
                    body["content_hash"],
                    body["declaration_hash"],
                    canonical(envelope).decode(),
                    canonical(declaration).decode(),
                    canonical(sorted(permissions)).decode(),
                    str(material_path),
                    revision,
                    canonical(delta).decode(),
                    actor,
                    proposal_hash,
                    now,
                    now + 900,
                ),
            )
            _audit(
                conn,
                "propose",
                proposal_id,
                actor,
                {
                    "kind": body["kind"],
                    "name": body["name"],
                    "version": body["version"],
                    "content_hash": body["content_hash"],
                    "proposal_hash": proposal_hash,
                },
            )
        return self.proposal(proposal_id)

    def proposal(self, proposal_id):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM capability_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            require(row is not None, "capability_proposal_not_found", status=404)
            return {
                key: row[key]
                for key in (
                    "id",
                    "kind",
                    "name",
                    "version",
                    "publisher_id",
                    "content_hash",
                    "declaration_hash",
                    "expected_revision",
                    "actor",
                    "proposal_hash",
                    "expires_at",
                    "status",
                )
            } | {
                "permissions": json.loads(row["permissions_json"]),
                "delta": json.loads(row["delta_json"]),
                "declaration": json.loads(row["declaration_json"]),
                "receipt": json.loads(row["receipt_json"])
                if row["receipt_json"]
                else None,
            }

    def confirm(
        self,
        proposal_id,
        *,
        proposal_hash,
        expected_revision,
        approved_permissions,
        actor,
        validator=None,
        on_connection=None,
    ):
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM capability_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            require(row is not None, "capability_proposal_not_found", status=404)
            require(
                row["actor"] == actor, "capability_proposal_actor_changed", status=403
            )
            require(
                row["proposal_hash"] == proposal_hash
                and row["expected_revision"] == expected_revision,
                "capability_proposal_binding_changed",
            )
            head = governed_head(conn, row["kind"], row["name"])
            permissions = json.loads(row["permissions_json"])
            require(
                isinstance(approved_permissions, list)
                and all(isinstance(value, str) for value in approved_permissions)
                and sorted(approved_permissions) == permissions
                and len(set(approved_permissions)) == len(approved_permissions),
                "capability_permissions_not_approved",
                status=403,
            )
            if row["status"] == "confirmed":
                receipt = json.loads(row["receipt_json"])
                require(
                    head is not None
                    and head["revision"] == receipt["revision"]
                    and head["version_id"] == receipt["version_id"],
                    "capability_head_conflict",
                )
                self._version(conn, head["version_id"])
                return receipt
            require(row["expires_at"] > time.time(), "capability_proposal_expired")
            require(
                (head["revision"] if head else 0) == expected_revision,
                "capability_head_conflict",
            )
            publisher = self._publisher(conn, row["publisher_id"])
            require(publisher["revoked_at"] is None, "publisher_revoked", status=403)
            envelope = json.loads(row["envelope_json"])
            verify_signature(envelope, publisher["public_key"])
            declaration = json.loads(row["declaration_json"])
            material_path = Path(row["material_path"])
            require(
                digest(declaration) == row["declaration_hash"]
                and tree_hash(material_path) == row["content_hash"]
                and read_json(material_path / "publisher-signature.json") == envelope,
                "capability_material_changed",
            )
            if validator:
                validator(row["kind"], declaration)
            if head is None:
                if row["kind"] == "plugin":
                    active = conn.execute(
                        """SELECT 1 FROM plan_steps s JOIN plans p ON p.id=s.plan_id
                        WHERE s.tool_plugin=? AND p.status NOT IN ('done','failed','cancelled') LIMIT 1""",
                        (row["name"],),
                    ).fetchone()
                else:
                    active = conn.execute(
                        """SELECT 1 FROM plans WHERE template_id=?
                        AND status NOT IN ('done','failed','cancelled') LIMIT 1""",
                        (row["name"],),
                    ).fetchone()
                require(
                    active is None, "capability_active_legacy_plan_requires_completion"
                )
            existing = conn.execute(
                "SELECT * FROM capability_versions WHERE kind=? AND name=? AND version=?",
                (row["kind"], row["name"], row["version"]),
            ).fetchone()
            if existing:
                require(
                    existing["content_hash"] == row["content_hash"]
                    and existing["declaration_hash"] == row["declaration_hash"]
                    and existing["publisher_id"] == row["publisher_id"],
                    "capability_version_identity_reused",
                )
                version_id = existing["id"]
                self._version(conn, version_id)
            else:
                version_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO capability_versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        version_id,
                        row["kind"],
                        row["name"],
                        row["version"],
                        row["publisher_id"],
                        row["content_hash"],
                        row["declaration_hash"],
                        row["envelope_json"],
                        row["declaration_json"],
                        row["permissions_json"],
                        row["material_path"],
                        actor,
                        time.time(),
                    ),
                )
            if row["kind"] == "plugin":
                from dataclasses import replace
                from marvis.plugins.manifest import parse_manifest
                from marvis.repositories.plugins import _upsert_plugin_row

                _upsert_plugin_row(
                    conn,
                    replace(parse_manifest(declaration), checksum=row["content_hash"]),
                    enabled=True,
                    governed=True,
                )
            conn.execute(
                """INSERT INTO capability_heads VALUES(?,?,?,?,1)
                ON CONFLICT(kind,name) DO UPDATE SET version_id=excluded.version_id,
                revision=excluded.revision,enabled=1""",
                (row["kind"], row["name"], version_id, expected_revision + 1),
            )
            receipt = {
                "kind": row["kind"],
                "name": row["name"],
                "version": row["version"],
                "version_id": version_id,
                "content_hash": row["content_hash"],
                "revision": expected_revision + 1,
            }
            conn.execute(
                "UPDATE capability_proposals SET status='confirmed',receipt_json=?,confirmed_by=? WHERE id=?",
                (canonical(receipt).decode(), actor, proposal_id),
            )
            _audit(
                conn,
                "activate",
                row["name"],
                actor,
                receipt
                | {"proposal_id": proposal_id, "approved_permissions": permissions},
            )
            if on_connection:
                on_connection(conn,receipt)
            return receipt

    def rollback_proposal(self, kind, name, release_version, expected_revision, actor):
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM capability_versions WHERE kind=? AND name=? AND version=?",
                (kind, name, release_version),
            ).fetchone()
            require(row is not None, "capability_version_not_found", status=404)
            target = self._version(conn, row["id"])
        return self.propose(
            body=json.loads(target["envelope_json"])["body"],
            envelope=json.loads(target["envelope_json"]),
            declaration=json.loads(target["declaration_json"]),
            permissions=json.loads(target["permissions_json"]),
            material_path=Path(target["material_path"]),
            expected_revision=expected_revision,
            actor=actor,
        )

    def set_enabled(self, kind, name, enabled, actor, expected_revision):
        require(isinstance(enabled, bool), "capability_invalid_enabled", status=422)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            head = governed_head(conn, kind, name)
            require(head is not None, "capability_version_not_found", status=404)
            require(head["revision"] == expected_revision, "capability_head_conflict")
            if enabled:
                self._version(conn, head["version_id"])
            conn.execute(
                "UPDATE capability_heads SET enabled=?,revision=revision+1 WHERE kind=? AND name=?",
                (int(enabled), kind, name),
            )
            if kind == "plugin":
                from marvis.repositories.plugins import _set_plugin_enabled_row

                _set_plugin_enabled_row(conn, name, enabled)
            _audit(
                conn,
                "enable" if enabled else "disable",
                name,
                actor,
                {"kind": kind, "revision": expected_revision + 1},
            )
        return self.head(kind, name)


def pin_ref(conn, ref):
    head = governed_head(conn, "plugin", ref.plugin)
    if head is None:
        if capability_mode(conn) == "governed":
            row = conn.execute(
                "SELECT builtin FROM plugins WHERE name=?", (ref.plugin,)
            ).fetchone()
            require(
                row is not None and bool(row["builtin"]),
                "signed_capability_required",
                status=403,
            )
        return ref
    require(head["enabled"], "capability_disabled", status=403)
    row = conn.execute(
        "SELECT version FROM capability_versions WHERE id=?", (head["version_id"],)
    ).fetchone()
    from marvis.plugins.manifest import ToolRef

    return ToolRef(ref.plugin, ref.tool, ref.version or row["version"])


def pin_step_payload(conn, step):
    from marvis.plugins.manifest import ToolRef

    for ref in [step["tool_ref"], *step.get("granted_tools", [])]:
        pinned = pin_ref(
            conn, ToolRef(ref["plugin"], ref["tool"], ref.get("version", ""))
        )
        ref["version"] = pinned.version
