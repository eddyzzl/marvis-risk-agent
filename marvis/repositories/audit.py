import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect


def _now() -> str:
    return datetime.now(UTC).isoformat()


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


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        return {}
    return value
