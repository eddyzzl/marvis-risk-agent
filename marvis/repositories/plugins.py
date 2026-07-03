import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect
from marvis.plugins.errors import PluginNotFoundError
from marvis.plugins.manifest import PluginManifest, manifest_to_dict
from marvis.repositories.drafts import _set_draft_status_row


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PluginRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def upsert_plugin(self, manifest: PluginManifest, *, enabled: bool) -> None:
        with connect(self.db_path) as conn:
            _upsert_plugin_row(conn, manifest, enabled=enabled)

    def upsert_plugin_with_audit(
        self,
        manifest: PluginManifest,
        *,
        enabled: bool,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _upsert_plugin_row(conn, manifest, enabled=enabled)
            _write_audit_row(conn, **audit)

    def promote_draft_with_plugin_audits(
        self,
        manifest: PluginManifest,
        *,
        enabled: bool,
        draft_id: str,
        plugin_audit: dict,
        draft_audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _upsert_plugin_row(conn, manifest, enabled=enabled)
            _write_audit_row(conn, **plugin_audit)
            _set_draft_status_row(conn, draft_id, "promoted")
            _write_audit_row(conn, **draft_audit)

    def set_enabled(self, name: str, enabled: bool) -> None:
        with connect(self.db_path) as conn:
            _set_plugin_enabled_row(conn, name, enabled)

    def set_enabled_with_audit(self, name: str, enabled: bool, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _set_plugin_enabled_row(conn, name, enabled)
            _write_audit_row(conn, **audit)

    def delete_plugin(self, name: str) -> None:
        with connect(self.db_path) as conn:
            _delete_plugin_row(conn, name)

    def delete_plugin_with_audit(self, name: str, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _delete_plugin_row(conn, name)
            _write_audit_row(conn, **audit)

    def get_plugin(self, name: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT p.*, COUNT(t.name) AS tool_count
                  FROM plugins p
                  LEFT JOIN tools t ON t.plugin = p.name
                 WHERE p.name = ?
                 GROUP BY p.name
                """,
                (name,),
            ).fetchone()
        return _plugin_row_to_dict(row) if row is not None else None

    def list_plugins(self, *, include_disabled: bool = False) -> list[dict]:
        where = "" if include_disabled else "WHERE p.enabled = 1"
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT p.*, COUNT(t.name) AS tool_count
                  FROM plugins p
                  LEFT JOIN tools t ON t.plugin = p.name
                  {where}
                 GROUP BY p.name
                 ORDER BY p.name
                """
            ).fetchall()
        return [_plugin_row_to_dict(row) for row in rows]

    def list_tools(self) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT plugin, name, summary, input_schema_json,
                       output_schema_json, determinism, timeout_seconds,
                       failure_policy, side_effects_json, entrypoint,
                       memory_limit_mb
                  FROM tools
                 ORDER BY plugin, name
                """
            ).fetchall()
        return [_tool_row_to_dict(row) for row in rows]

    def write_audit(
        self,
        *,
        kind: str,
        target_ref: str,
        actor: str = "system",
        inputs_hash: str | None = None,
        outcome: str | None = None,
        detail: dict | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _write_audit_row(
                conn,
                kind=kind,
                target_ref=target_ref,
                actor=actor,
                inputs_hash=inputs_hash,
                outcome=outcome,
                detail=detail,
            )

    def list_audit(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        return _list_audit_rows(self.db_path, kind=kind, limit=limit, offset=offset)


def _plugin_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "name": row["name"],
        "version": row["version"],
        "display_name": row["display_name"],
        "description": row["description"],
        "module": row["module"],
        "manifest_json": row["manifest_json"],
        "checksum": row["checksum"],
        "builtin": bool(row["builtin"]),
        "enabled": bool(row["enabled"]),
        "installed_at": row["installed_at"],
        "tool_count": int(row["tool_count"]),
    }


def _tool_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "plugin": row["plugin"],
        "name": row["name"],
        "summary": row["summary"],
        "input_schema_json": row["input_schema_json"],
        "output_schema_json": row["output_schema_json"],
        "determinism": row["determinism"],
        "timeout_seconds": int(row["timeout_seconds"]),
        "failure_policy": row["failure_policy"],
        "side_effects_json": row["side_effects_json"],
        "entrypoint": row["entrypoint"],
        "memory_limit_mb": int(row["memory_limit_mb"]),
    }


def _audit_row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "actor": row["actor"],
        "target_ref": row["target_ref"],
        "inputs_hash": row["inputs_hash"],
        "outcome": row["outcome"],
        "detail": _load_json_object(row["detail_json"]),
        "at": row["at"],
    }


def _upsert_plugin_row(
    conn: sqlite3.Connection,
    manifest: PluginManifest,
    *,
    enabled: bool,
) -> None:
    manifest_json = json.dumps(
        manifest_to_dict(manifest),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    now = _now()
    conn.execute(
        """
        INSERT INTO plugins(
            name, version, display_name, description, module,
            manifest_json, checksum, builtin, enabled, installed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            version = excluded.version,
            display_name = excluded.display_name,
            description = excluded.description,
            module = excluded.module,
            manifest_json = excluded.manifest_json,
            checksum = excluded.checksum,
            builtin = excluded.builtin,
            enabled = excluded.enabled,
            installed_at = excluded.installed_at
        """,
        (
            manifest.name,
            manifest.version,
            manifest.display_name,
            manifest.description,
            manifest.module,
            manifest_json,
            manifest.checksum,
            int(manifest.builtin),
            int(enabled),
            now,
        ),
    )
    conn.execute("DELETE FROM tools WHERE plugin = ?", (manifest.name,))
    for tool in manifest.tools:
        conn.execute(
            """
            INSERT INTO tools(
                plugin, name, summary, input_schema_json,
                output_schema_json, determinism, timeout_seconds,
                failure_policy, side_effects_json, entrypoint,
                memory_limit_mb
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.name,
                tool.name,
                tool.summary,
                json.dumps(tool.input_schema, ensure_ascii=False, separators=(",", ":")),
                json.dumps(tool.output_schema, ensure_ascii=False, separators=(",", ":")),
                tool.determinism,
                tool.timeout_seconds,
                tool.failure_policy,
                json.dumps(list(tool.side_effects), ensure_ascii=False, separators=(",", ":")),
                tool.entrypoint,
                tool.memory_limit_mb,
            ),
        )


def _set_plugin_enabled_row(
    conn: sqlite3.Connection,
    name: str,
    enabled: bool,
) -> None:
    cursor = conn.execute(
        "UPDATE plugins SET enabled = ? WHERE name = ?",
        (int(enabled), name),
    )
    if cursor.rowcount == 0:
        raise PluginNotFoundError(name)


def _delete_plugin_row(conn: sqlite3.Connection, name: str) -> None:
    cursor = conn.execute("DELETE FROM plugins WHERE name = ?", (name,))
    if cursor.rowcount == 0:
        raise PluginNotFoundError(name)


def _write_audit_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    target_ref: str,
    actor: str = "system",
    inputs_hash: str | None = None,
    outcome: str | None = None,
    detail: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            kind,
            actor,
            target_ref,
            inputs_hash,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now(),
        ),
    )


def _list_audit_rows(
    db_path: Path,
    *,
    kind: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    bounded_limit = None if limit is None else max(1, int(limit))
    bounded_offset = max(0, int(offset))
    query = (
        "SELECT id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at "
        "FROM audit"
    )
    params: list[object] = []
    if kind is not None:
        query += " WHERE kind = ?"
        params.append(kind)
    query += " ORDER BY at, id"
    if bounded_limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([bounded_limit, bounded_offset])
    with connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_audit_row_to_dict(row) for row in rows]


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
