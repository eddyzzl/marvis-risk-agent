from __future__ import annotations

import json
import uuid

from marvis.db_schema import connect, init_db
from marvis.repositories.tasks import maybe_write_task_cleanup_succeeded


def _write_delete_audit(conn, *, audit_id: str, task_id: str) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        ) VALUES (?, 'task.delete', 'reviewer', ?, NULL, 'db_purged', '{}', ?)
        """,
        (audit_id, task_id, "2026-08-01T12:00:00+00:00"),
    )


def test_cleanup_success_is_exactly_once_per_delete_generation(tmp_path) -> None:
    db_path = tmp_path / "workspace" / "marvis.sqlite"
    init_db(db_path)
    task_id = uuid.uuid4().hex
    first_delete_id = uuid.uuid4().hex
    second_delete_id = uuid.uuid4().hex

    with connect(db_path) as conn:
        _write_delete_audit(conn, audit_id=first_delete_id, task_id=task_id)
        assert maybe_write_task_cleanup_succeeded(conn, task_id=task_id) is True
        assert maybe_write_task_cleanup_succeeded(conn, task_id=task_id) is False

        _write_delete_audit(conn, audit_id=second_delete_id, task_id=task_id)
        assert maybe_write_task_cleanup_succeeded(conn, task_id=task_id) is True
        assert maybe_write_task_cleanup_succeeded(conn, task_id=task_id) is False

        rows = conn.execute(
            """
            SELECT detail_json
              FROM audit
             WHERE kind = 'task.cleanup'
               AND target_ref = ?
               AND outcome = 'succeeded'
             ORDER BY rowid
            """,
            (task_id,),
        ).fetchall()

    assert [json.loads(str(row["detail_json"]))["delete_audit_id"] for row in rows] == [
        first_delete_id,
        second_delete_id,
    ]
