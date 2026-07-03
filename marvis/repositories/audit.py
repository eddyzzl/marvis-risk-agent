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


def _build_audit_filters(
    *,
    kind: str | None,
    kind_prefix: str | None,
    target_ref: str | None,
    target_ref_prefix: str | None,
    task_id: str | None,
    after: str | None,
    before: str | None,
) -> tuple[str, list[object]]:
    """Shared WHERE-clause builder for _list_audit_rows/_count_audit_rows so the
    two stay in lockstep (pagination totals must reflect the same predicate as
    the page query). task_id matches records whose target_ref carries the task_id
    as a literal value *or* whose detail_json embeds "task_id":"<id>" -- neither
    convention is authoritative repo-wide (audit kinds use plan_id/step_id/
    experiment_id/etc. as target_ref), so this is a best-effort match, not a
    guaranteed-complete one.
    """
    clauses: list[str] = []
    params: list[object] = []
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind)
    if kind_prefix is not None:
        clauses.append("kind LIKE ? ESCAPE '\\'")
        params.append(_like_prefix(kind_prefix))
    if target_ref is not None:
        clauses.append("target_ref = ?")
        params.append(target_ref)
    if target_ref_prefix is not None:
        clauses.append("target_ref LIKE ? ESCAPE '\\'")
        params.append(_like_prefix(target_ref_prefix))
    if task_id is not None:
        clauses.append(
            "(target_ref = ? OR target_ref LIKE ? ESCAPE '\\' OR detail_json LIKE ? ESCAPE '\\')"
        )
        params.append(task_id)
        params.append(_like_prefix(f"{task_id}:"))
        params.append(_like_contains(f'"task_id":"{task_id}"'))
    if after is not None:
        clauses.append("at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("at <= ?")
        params.append(before)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _like_prefix(value: str) -> str:
    return f"{_escape_like(value)}%"


def _like_contains(value: str) -> str:
    return f"%{_escape_like(value)}%"


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _list_audit_rows(
    db_path: Path,
    *,
    kind: str | None = None,
    kind_prefix: str | None = None,
    target_ref: str | None = None,
    target_ref_prefix: str | None = None,
    task_id: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    bounded_limit = None if limit is None else max(1, int(limit))
    bounded_offset = max(0, int(offset))
    where, params = _build_audit_filters(
        kind=kind,
        kind_prefix=kind_prefix,
        target_ref=target_ref,
        target_ref_prefix=target_ref_prefix,
        task_id=task_id,
        after=after,
        before=before,
    )
    query = (
        "SELECT id, kind, actor, target_ref, inputs_hash, outcome, detail_json, at "
        "FROM audit" + where + " ORDER BY at, id"
    )
    if bounded_limit is not None:
        query += " LIMIT ? OFFSET ?"
        params = [*params, bounded_limit, bounded_offset]
    with connect(db_path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_audit_row_to_dict(row) for row in rows]


def _count_audit_rows(
    db_path: Path,
    *,
    kind: str | None = None,
    kind_prefix: str | None = None,
    target_ref: str | None = None,
    target_ref_prefix: str | None = None,
    task_id: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> int:
    where, params = _build_audit_filters(
        kind=kind,
        kind_prefix=kind_prefix,
        target_ref=target_ref,
        target_ref_prefix=target_ref_prefix,
        task_id=task_id,
        after=after,
        before=before,
    )
    query = "SELECT COUNT(*) FROM audit" + where
    with connect(db_path) as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return int(row[0]) if row is not None else 0


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
