import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect
from marvis.drafts.contracts import (
    DraftRun,
    DraftTool,
    LearningNote,
    assert_draft_status_transition,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class DraftRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def save_learning_note(self, note: LearningNote) -> None:
        with connect(self.db_path) as conn:
            _insert_learning_note_row(conn, note)

    def save_learning_note_with_audit(self, note: LearningNote, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_learning_note_row(conn, note)
            _write_audit_row(conn, **audit)

    def get_learning_note(self, note_id: str) -> LearningNote | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, query, sources_json, distilled, created_at
                  FROM learning_notes
                 WHERE id = ?
                """,
                (note_id,),
            ).fetchone()
        return None if row is None else _learning_note_from_row(row)

    def save_draft(self, draft: DraftTool) -> None:
        with connect(self.db_path) as conn:
            _insert_draft_tool_row(conn, draft)

    def save_draft_with_audit(self, draft: DraftTool, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _insert_draft_tool_row(conn, draft)
            _write_audit_row(conn, **audit)

    def get_draft(self, draft_id: str) -> DraftTool | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, name, summary, code, input_schema_json,
                       output_schema_json, determinism, source, learning_note_id,
                       status, created_at
                  FROM draft_tools
                 WHERE id = ?
                """,
                (draft_id,),
            ).fetchone()
        return None if row is None else _draft_tool_from_row(row)

    def list_drafts(
        self,
        task_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DraftTool]:
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        limit_clause = " LIMIT ? OFFSET ?" if bounded_limit is not None else ""
        with connect(self.db_path) as conn:
            if status is None:
                params: tuple = (task_id,)
                if bounded_limit is not None:
                    params = (*params, bounded_limit, bounded_offset)
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, name, summary, code, input_schema_json,
                           output_schema_json, determinism, source, learning_note_id,
                           status, created_at
                      FROM draft_tools
                     WHERE task_id = ?
                     ORDER BY created_at, id
                     {limit_clause}
                    """,
                    params,
                ).fetchall()
            else:
                params = (task_id, status)
                if bounded_limit is not None:
                    params = (*params, bounded_limit, bounded_offset)
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, name, summary, code, input_schema_json,
                           output_schema_json, determinism, source, learning_note_id,
                           status, created_at
                      FROM draft_tools
                     WHERE task_id = ? AND status = ?
                     ORDER BY created_at, id
                     {limit_clause}
                    """,
                    params,
                ).fetchall()
        return [_draft_tool_from_row(row) for row in rows]

    def list_all_drafts(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[DraftTool]:
        bounded_limit = None if limit is None else max(1, int(limit))
        bounded_offset = max(0, int(offset))
        limit_clause = " LIMIT ? OFFSET ?" if bounded_limit is not None else ""
        with connect(self.db_path) as conn:
            if status is None:
                params: tuple = ()
                if bounded_limit is not None:
                    params = (bounded_limit, bounded_offset)
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, name, summary, code, input_schema_json,
                           output_schema_json, determinism, source, learning_note_id,
                           status, created_at
                      FROM draft_tools
                     ORDER BY created_at, id
                     {limit_clause}
                    """,
                    params,
                ).fetchall()
            else:
                params = (status,)
                if bounded_limit is not None:
                    params = (*params, bounded_limit, bounded_offset)
                rows = conn.execute(
                    f"""
                    SELECT id, task_id, name, summary, code, input_schema_json,
                           output_schema_json, determinism, source, learning_note_id,
                           status, created_at
                      FROM draft_tools
                     WHERE status = ?
                     ORDER BY created_at, id
                     {limit_clause}
                    """,
                    params,
                ).fetchall()
        return [_draft_tool_from_row(row) for row in rows]

    def set_status(self, draft_id: str, status: str) -> None:
        with connect(self.db_path) as conn:
            _set_draft_status_row(conn, draft_id, status)

    def set_status_with_audit(self, draft_id: str, status: str, *, audit: dict) -> None:
        with connect(self.db_path) as conn:
            _set_draft_status_row(conn, draft_id, status)
            _write_audit_row(conn, **audit)

    def save_draft_run(self, run: DraftRun) -> None:
        with connect(self.db_path) as conn:
            _insert_draft_run_row(conn, run)

    def save_draft_run_with_status_audit(
        self,
        run: DraftRun,
        *,
        status: str | None = None,
        audit: dict,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_draft_run_row(conn, run)
            if status is not None:
                _set_draft_status_row(conn, run.draft_id, status)
            _write_audit_row(conn, **audit)

    def list_runs(self, draft_id: str) -> list[DraftRun]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, draft_id, task_id, inputs_hash, ok, output_json, error, at
                  FROM draft_runs
                 WHERE draft_id = ?
                 ORDER BY at, id
                """,
                (draft_id,),
            ).fetchall()
        return [_draft_run_from_row(row) for row in rows]


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


def _learning_note_insert_values(note: LearningNote) -> tuple:
    return (
        note.id,
        note.query,
        _dump_json_any(list(note.sources)),
        note.distilled,
        note.created_at,
    )


def _insert_learning_note_row(conn: sqlite3.Connection, note: LearningNote) -> None:
    conn.execute(
        """
        INSERT INTO learning_notes(
            id, query, sources_json, distilled, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        _learning_note_insert_values(note),
    )


def _learning_note_from_row(row: sqlite3.Row) -> LearningNote:
    return LearningNote(
        id=str(row["id"]),
        query=str(row["query"]),
        sources=tuple(str(item) for item in _load_json_array(row["sources_json"])),
        distilled=str(row["distilled"]),
        created_at=str(row["created_at"]),
    )


def _draft_tool_insert_values(draft: DraftTool) -> tuple:
    return (
        draft.id,
        draft.task_id,
        draft.name,
        draft.summary,
        draft.code,
        _dump_json_any(draft.input_schema),
        _dump_json_any(draft.output_schema),
        draft.determinism,
        draft.source,
        draft.learning_note_id,
        draft.status,
        draft.created_at,
    )


def _insert_draft_tool_row(conn: sqlite3.Connection, draft: DraftTool) -> None:
    conn.execute(
        """
        INSERT INTO draft_tools(
            id, task_id, name, summary, code, input_schema_json,
            output_schema_json, determinism, source, learning_note_id,
            status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _draft_tool_insert_values(draft),
    )


def _draft_tool_from_row(row: sqlite3.Row) -> DraftTool:
    return DraftTool(
        id=str(row["id"]),
        task_id=str(row["task_id"]),
        name=str(row["name"]),
        summary=str(row["summary"]),
        code=str(row["code"]),
        input_schema=_load_json_object(row["input_schema_json"]),
        output_schema=_load_json_object(row["output_schema_json"]),
        determinism=str(row["determinism"]),
        source=str(row["source"]),
        learning_note_id=_optional_str(row["learning_note_id"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
    )


def _set_draft_status_row(
    conn: sqlite3.Connection,
    draft_id: str,
    status: str,
) -> None:
    row = conn.execute(
        "SELECT status FROM draft_tools WHERE id = ?",
        (draft_id,),
    ).fetchone()
    if row is None:
        raise KeyError(draft_id)
    assert_draft_status_transition(str(row["status"]), status)
    cursor = conn.execute(
        "UPDATE draft_tools SET status = ? WHERE id = ?",
        (status, draft_id),
    )
    if cursor.rowcount == 0:
        raise KeyError(draft_id)


def _draft_run_insert_values(run: DraftRun) -> tuple:
    return (
        run.id,
        run.draft_id,
        run.task_id,
        run.inputs_hash,
        1 if run.ok else 0,
        None if run.output is None else _dump_json_any(run.output),
        run.error,
        run.at,
    )


def _insert_draft_run_row(conn: sqlite3.Connection, run: DraftRun) -> None:
    conn.execute(
        """
        INSERT INTO draft_runs(
            id, draft_id, task_id, inputs_hash, ok, output_json, error, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _draft_run_insert_values(run),
    )


def _draft_run_from_row(row: sqlite3.Row) -> DraftRun:
    output_json = row["output_json"]
    return DraftRun(
        id=str(row["id"]),
        draft_id=str(row["draft_id"]),
        task_id=str(row["task_id"]),
        inputs_hash=str(row["inputs_hash"]),
        ok=bool(row["ok"]),
        output=None if output_json is None else _load_json_object(output_json),
        error=_optional_str(row["error"]),
        at=str(row["at"]),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_array(raw: str | None) -> list:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
