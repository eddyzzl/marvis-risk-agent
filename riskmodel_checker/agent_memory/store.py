from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3
import uuid

from riskmodel_checker.agent_memory.models import (
    MemoryCandidate,
    normalize_memory_status,
    normalize_memory_type,
)
from riskmodel_checker.agent_memory.policy import (
    MemoryPolicyDecision,
    classify_memory_candidate,
)
from riskmodel_checker.db import _now, connect


AUDIT_EVENT_TYPES = (
    "create",
    "retrieve",
    "use",
    "disable",
    "enable",
    "delete",
    "reject",
)


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    memory_type: str
    status: str
    summary: str
    payload: dict[str, Any]
    source_task_id: str | None
    source_message_id: str | None
    confidence: str
    reason: str
    created_at: str
    updated_at: str
    deleted_at: str | None


class AgentMemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create(self, candidate: MemoryCandidate, *, task_id: str | None = None) -> MemoryEntry:
        decision = classify_memory_candidate(candidate)
        if not decision.allowed:
            return self.reject(candidate, decision, task_id=task_id)
        entry_id = uuid.uuid4().hex
        now = _now()
        payload_json = _dump_json_object(candidate.payload)
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_memory_entries
                (
                    id, memory_type, status, summary, payload_json,
                    source_task_id, source_message_id, confidence, reason,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    entry_id,
                    candidate.memory_type,
                    candidate.summary,
                    payload_json,
                    candidate.source_task_id,
                    candidate.source_message_id,
                    candidate.confidence,
                    candidate.reason,
                    now,
                    now,
                ),
            )
            self._append_event(
                conn,
                entry_id,
                "create",
                task_id=task_id or candidate.source_task_id,
                message_id=candidate.source_message_id,
                details={"memory_type": candidate.memory_type},
            )
            row = self._select_entry(conn, entry_id, include_deleted=True)
        return _row_to_entry(row)

    def reject(
        self,
        candidate: MemoryCandidate,
        decision: MemoryPolicyDecision,
        *,
        task_id: str | None = None,
    ) -> MemoryEntry:
        entry_id = uuid.uuid4().hex
        now = _now()
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO agent_memory_entries
                (
                    id, memory_type, status, summary, payload_json,
                    source_task_id, source_message_id, confidence, reason,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, 'rejected', '', '{}', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    candidate.memory_type,
                    candidate.source_task_id,
                    candidate.source_message_id,
                    candidate.confidence,
                    candidate.reason,
                    now,
                    now,
                    now,
                ),
            )
            self._append_event(
                conn,
                entry_id,
                "reject",
                task_id=task_id or candidate.source_task_id,
                message_id=candidate.source_message_id,
                details={
                    "memory_type": candidate.memory_type,
                    "reasons": list(decision.reasons),
                },
            )
            row = self._select_entry(conn, entry_id, include_deleted=True)
        return _row_to_entry(row)

    def get_entry(
        self,
        entry_id: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        include_deleted: bool = False,
        audit: bool = True,
    ) -> MemoryEntry:
        with connect(self.db_path) as conn:
            row = self._select_entry(conn, entry_id, include_deleted=include_deleted)
            if audit:
                self._append_event(
                    conn,
                    entry_id,
                    "retrieve",
                    task_id=task_id,
                    message_id=message_id,
                    details={},
                )
        return _row_to_entry(row)

    def list_entries(
        self,
        *,
        status: str | None = None,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is None:
            clauses.append("status = 'active'")
        else:
            clauses.append("status = ?")
            params.append(normalize_memory_status(status))
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(normalize_memory_type(memory_type))
        params.append(max(1, int(limit)))
        where_sql = " AND ".join(clauses)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                  FROM agent_memory_entries
                 WHERE {where_sql}
                 ORDER BY updated_at DESC, id DESC
                 LIMIT ?
                """,
                params,
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def list_events(self, entry_id: str) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, memory_id, event_type, task_id, message_id,
                       details_json, created_at
                  FROM agent_memory_events
                 WHERE memory_id = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (entry_id,),
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def record_use(
        self,
        entry_id: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        use_reason: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            self._select_entry(conn, entry_id, include_deleted=False)
            self._append_event(
                conn,
                entry_id,
                "use",
                task_id=task_id,
                message_id=message_id,
                details={"use_reason": use_reason},
            )

    def set_status(
        self,
        entry_id: str,
        status: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
    ) -> MemoryEntry:
        normalized_status = normalize_memory_status(status)
        if normalized_status == "deleted":
            return self.delete(entry_id, task_id=task_id, message_id=message_id)
        if normalized_status == "rejected":
            raise ValueError("use reject() to record rejected memory candidates")

        event_type = "enable" if normalized_status == "active" else "disable"
        now = _now()
        with connect(self.db_path) as conn:
            current = self._select_entry(conn, entry_id, include_deleted=False)
            if current["status"] == "rejected":
                raise ValueError("rejected memory entries are terminal")
            conn.execute(
                """
                UPDATE agent_memory_entries
                   SET status = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (normalized_status, now, entry_id),
            )
            self._append_event(
                conn,
                entry_id,
                event_type,
                task_id=task_id,
                message_id=message_id,
                details={},
            )
            row = self._select_entry(conn, entry_id, include_deleted=True)
        return _row_to_entry(row)

    def delete(
        self,
        entry_id: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
    ) -> MemoryEntry:
        now = _now()
        with connect(self.db_path) as conn:
            self._select_entry(conn, entry_id, include_deleted=False)
            conn.execute(
                """
                UPDATE agent_memory_entries
                   SET status = 'deleted',
                       summary = '',
                       payload_json = '{}',
                       updated_at = ?,
                       deleted_at = ?
                 WHERE id = ?
                """,
                (now, now, entry_id),
            )
            self._append_event(
                conn,
                entry_id,
                "delete",
                task_id=task_id,
                message_id=message_id,
                details={},
            )
            row = self._select_entry(conn, entry_id, include_deleted=True)
        return _row_to_entry(row)

    def _select_entry(
        self,
        conn: sqlite3.Connection,
        entry_id: str,
        *,
        include_deleted: bool,
    ) -> sqlite3.Row:
        if include_deleted:
            row = conn.execute(
                "SELECT * FROM agent_memory_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT *
                  FROM agent_memory_entries
                 WHERE id = ?
                   AND status != 'deleted'
                """,
                (entry_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Memory entry not found: {entry_id}")
        return row

    def _append_event(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        event_type: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if event_type not in AUDIT_EVENT_TYPES:
            raise ValueError(f"unsupported memory audit event: {event_type}")
        conn.execute(
            """
            INSERT INTO agent_memory_events
            (id, memory_id, event_type, task_id, message_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                memory_id,
                event_type,
                task_id,
                message_id,
                _dump_json_object(details or {}),
                _now(),
            ),
        )


def ensure_agent_memory_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_entries (
            id TEXT PRIMARY KEY,
            memory_type TEXT NOT NULL,
            status TEXT NOT NULL,
            summary TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            source_task_id TEXT,
            source_message_id TEXT,
            confidence TEXT NOT NULL DEFAULT 'medium',
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_memory_entries_status_type
            ON agent_memory_entries(status, memory_type, updated_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_memory_entries_source_task
            ON agent_memory_entries(source_task_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memory_events (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            event_type TEXT NOT NULL,
            task_id TEXT,
            message_id TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES agent_memory_entries(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_memory_events_memory
            ON agent_memory_events(memory_id, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_memory_events_type
            ON agent_memory_events(event_type, created_at)
        """
    )


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        memory_type=row["memory_type"],
        status=row["status"],
        summary=row["summary"],
        payload=_load_json_object(row["payload_json"]),
        source_task_id=row["source_task_id"],
        source_message_id=row["source_message_id"],
        confidence=row["confidence"],
        reason=row["reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
    )


def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "memory_id": row["memory_id"],
        "event_type": row["event_type"],
        "task_id": row["task_id"],
        "message_id": row["message_id"],
        "details": _load_json_object(row["details_json"]),
        "created_at": row["created_at"],
    }


def _dump_json_object(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _load_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}
