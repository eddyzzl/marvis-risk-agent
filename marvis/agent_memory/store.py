from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3
import uuid

from marvis.agent_memory.models import (
    MemoryCandidate,
    normalize_memory_status,
    normalize_memory_type,
)
from marvis.agent_memory.distillation import (
    MemoryDistillation,
    confidence_from_support,
    normalize_distillation_status,
)
from marvis.agent_memory.policy import (
    MemoryPolicyDecision,
    classify_memory_candidate,
)
from marvis.db import _now, connect
from marvis.redaction import redact_text, redact_value


AUDIT_EVENT_TYPES = (
    "create",
    "retrieve",
    "use",
    "disable",
    "enable",
    "delete",
    "reject",
)
DISTILLATION_AUDIT_EVENT_TYPES = (
    "create",
    "use",
    "supersede",
    "restore",
    "rollback",
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
        safe_summary = redact_text(candidate.summary)
        safe_payload = redact_value(candidate.payload)
        safe_reason = redact_text(candidate.reason)
        redacted_count = (
            int(safe_payload.redacted_count)
            + int(safe_summary != candidate.summary)
            + int(safe_reason != candidate.reason)
        )
        payload_json = _dump_json_object(safe_payload.value)
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
                    safe_summary,
                    payload_json,
                    candidate.source_task_id,
                    candidate.source_message_id,
                    candidate.confidence,
                    safe_reason,
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
                details={"memory_type": candidate.memory_type, "redacted_count": redacted_count},
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

    def record_retrievals(
        self,
        entry_ids: list[str],
        *,
        task_id: str | None = None,
        message_id: str | None = None,
    ) -> set[str]:
        ordered_ids = list(dict.fromkeys(str(entry_id) for entry_id in entry_ids if entry_id))
        if not ordered_ids:
            return set()
        placeholders = ",".join("?" for _ in ordered_ids)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id
                  FROM agent_memory_entries
                 WHERE status != 'deleted'
                   AND id IN ({placeholders})
                """,
                ordered_ids,
            ).fetchall()
            found_ids = {str(row["id"]) for row in rows}
            for entry_id in ordered_ids:
                if entry_id in found_ids:
                    self._append_event(
                        conn,
                        entry_id,
                        "retrieve",
                        task_id=task_id,
                        message_id=message_id,
                        details={},
                    )
        return found_ids

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

    def create_distillation(self, distillation: MemoryDistillation) -> MemoryDistillation:
        with connect(self.db_path) as conn:
            redacted_count = self._insert_distillation(conn, distillation)
            self._append_distillation_event(
                conn,
                distillation.id,
                "create",
                details={
                    "category": distillation.category,
                    "scope_key": distillation.scope_key,
                    "support_count": distillation.support_count,
                    "redacted_count": redacted_count,
                },
            )
            row = self._select_distillation(conn, distillation.id)
        return _row_to_distillation(row)

    def replace_active_distillation_with_audit(
        self,
        active_id: str,
        candidate: MemoryDistillation,
    ) -> MemoryDistillation:
        now = _now()
        with connect(self.db_path) as conn:
            self._select_distillation(conn, active_id)
            redacted_count = self._insert_distillation(conn, candidate)
            self._append_distillation_event(
                conn,
                candidate.id,
                "create",
                details={
                    "category": candidate.category,
                    "scope_key": candidate.scope_key,
                    "support_count": candidate.support_count,
                    "redacted_count": redacted_count,
                },
            )
            cursor = conn.execute(
                """
                UPDATE memory_distillations
                   SET superseded_by = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND status = 'active'
                   AND superseded_by IS NULL
                """,
                (candidate.id, now, active_id),
            )
            if cursor.rowcount == 0:
                raise RuntimeError("active distillation changed while replacing")
            self._append_distillation_event(
                conn,
                active_id,
                "supersede",
                details={"superseded_by": candidate.id},
            )
            row = self._select_distillation(conn, candidate.id)
        return _row_to_distillation(row)

    def get_distillation(self, distillation_id: str) -> MemoryDistillation:
        with connect(self.db_path) as conn:
            row = self._select_distillation(conn, distillation_id)
        return _row_to_distillation(row)

    def get_active_distillation(self, scope_key: str) -> MemoryDistillation | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                  FROM memory_distillations
                 WHERE scope_key = ?
                   AND status = 'active'
                   AND superseded_by IS NULL
                 ORDER BY updated_at DESC, id DESC
                 LIMIT 1
                """,
                (scope_key,),
            ).fetchone()
        return _row_to_distillation(row) if row is not None else None

    def list_distillations(
        self,
        *,
        category: str | None = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> list[MemoryDistillation]:
        clauses: list[str] = []
        params: list[Any] = []
        if not include_superseded:
            clauses.append("status = 'active'")
            clauses.append("superseded_by IS NULL")
        if category is not None:
            clauses.append("category = ?")
            params.append(normalize_memory_type(category))
        params.append(max(1, int(limit)))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                  FROM memory_distillations
                  {where_sql}
                 ORDER BY updated_at DESC, id DESC
                 LIMIT ?
                """,
                params,
            ).fetchall()
        return [_row_to_distillation(row) for row in rows]

    def set_superseded(self, distillation_id: str, *, by: str) -> None:
        now = _now()
        with connect(self.db_path) as conn:
            self._select_distillation(conn, distillation_id)
            conn.execute(
                """
                UPDATE memory_distillations
                   SET superseded_by = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (by, now, distillation_id),
            )
            self._append_distillation_event(
                conn,
                distillation_id,
                "supersede",
                details={"superseded_by": by},
            )

    def clear_superseded(self, distillation_id: str) -> None:
        now = _now()
        with connect(self.db_path) as conn:
            self._select_distillation(conn, distillation_id)
            conn.execute(
                """
                UPDATE memory_distillations
                   SET superseded_by = NULL,
                       updated_at = ?
                 WHERE id = ?
                """,
                (now, distillation_id),
            )
            self._append_distillation_event(
                conn,
                distillation_id,
                "restore",
                details={},
            )

    def rollback_active_distillation_with_audit(
        self,
        distillation_id: str,
        *,
        predecessor_id: str | None,
    ) -> MemoryDistillation:
        now = _now()
        with connect(self.db_path) as conn:
            self._select_distillation(conn, distillation_id)
            if predecessor_id is not None:
                self._select_distillation(conn, predecessor_id)
                cursor = conn.execute(
                    """
                    UPDATE memory_distillations
                       SET superseded_by = NULL,
                           updated_at = ?
                     WHERE id = ?
                       AND superseded_by = ?
                    """,
                    (now, predecessor_id, distillation_id),
                )
                if cursor.rowcount == 0:
                    raise RuntimeError("predecessor distillation changed while rolling back")
                self._append_distillation_event(
                    conn,
                    predecessor_id,
                    "restore",
                    details={},
                )
            cursor = conn.execute(
                """
                UPDATE memory_distillations
                   SET status = ?,
                       updated_at = ?
                 WHERE id = ?
                   AND status = 'active'
                   AND superseded_by IS NULL
                """,
                ("rolled_back", now, distillation_id),
            )
            if cursor.rowcount == 0:
                raise RuntimeError("active distillation changed while rolling back")
            self._append_distillation_event(
                conn,
                distillation_id,
                "rollback",
                details={"status": "rolled_back"},
            )
            row = self._select_distillation(conn, distillation_id)
        return _row_to_distillation(row)

    def update_distillation_support(self, distillation_id: str, support_count: int) -> MemoryDistillation:
        now = _now()
        support = int(support_count)
        with connect(self.db_path) as conn:
            self._select_distillation(conn, distillation_id)
            conn.execute(
                """
                UPDATE memory_distillations
                   SET support_count = ?,
                       confidence = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (support, confidence_from_support(support), now, distillation_id),
            )
            row = self._select_distillation(conn, distillation_id)
        return _row_to_distillation(row)

    def set_status_distillation(self, distillation_id: str, status: str) -> MemoryDistillation:
        normalized = normalize_distillation_status(status)
        now = _now()
        with connect(self.db_path) as conn:
            self._select_distillation(conn, distillation_id)
            conn.execute(
                """
                UPDATE memory_distillations
                   SET status = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (normalized, now, distillation_id),
            )
            self._append_distillation_event(
                conn,
                distillation_id,
                "rollback" if normalized == "rolled_back" else "restore",
                details={"status": normalized},
            )
            row = self._select_distillation(conn, distillation_id)
        return _row_to_distillation(row)

    def record_distillation_use(
        self,
        distillation_id: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        use_reason: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            row = self._select_distillation(conn, distillation_id)
            if row["status"] != "active":
                raise ValueError("rolled back memory distillations cannot be used")
            self._append_distillation_event(
                conn,
                distillation_id,
                "use",
                task_id=task_id,
                message_id=message_id,
                details={"use_reason": use_reason},
            )

    def list_distillation_events(self, distillation_id: str) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, distillation_id, event_type, task_id, message_id,
                       details_json, created_at
                  FROM memory_distillation_events
                 WHERE distillation_id = ?
                 ORDER BY created_at ASC, id ASC
                """,
                (distillation_id,),
            ).fetchall()
        return [_row_to_distillation_event(row) for row in rows]

    def find_superseded_by(self, distillation_id: str) -> MemoryDistillation | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                  FROM memory_distillations
                 WHERE superseded_by = ?
                 ORDER BY updated_at DESC, id DESC
                 LIMIT 1
                """,
                (distillation_id,),
            ).fetchone()
        return _row_to_distillation(row) if row is not None else None

    def search_distillations(
        self,
        query_context: dict[str, Any],
        *,
        active_only: bool = True,
        limit: int = 6,
    ) -> list[MemoryDistillation]:
        category = query_context.get("category")
        keywords = [
            str(item).lower()
            for item in query_context.get("keywords", [])
            if str(item).strip()
        ]
        scope_text = str(query_context.get("scope_key") or query_context.get("scope") or "").lower()
        clauses: list[str] = []
        params: list[Any] = []
        if active_only:
            clauses.extend(["status = 'active'", "superseded_by IS NULL"])
        if category:
            clauses.append("category = ?")
            params.append(normalize_memory_type(str(category)))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                  FROM memory_distillations
                  {where_sql}
                 ORDER BY updated_at DESC, id DESC
                """,
                params,
            ).fetchall()
        scored = [
            (distillation, _distillation_score(distillation, keywords, scope_text))
            for distillation in (_row_to_distillation(row) for row in rows)
        ]
        scored = [item for item in scored if item[1] > 0 or not (keywords or scope_text)]
        scored.sort(key=lambda item: (item[1], item[0].updated_at, item[0].id), reverse=True)
        return [item[0] for item in scored[: max(1, int(limit))]]

    def mark_consolidated(self, category: str, *, at: str) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO memory_consolidation_state (category, last_consolidated_at)
                VALUES (?, ?)
                ON CONFLICT(category) DO UPDATE SET last_consolidated_at = excluded.last_consolidated_at
                """,
                (normalize_memory_type(category), at),
            )

    def last_consolidated_at(self, category: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_consolidated_at FROM memory_consolidation_state WHERE category = ?",
                (normalize_memory_type(category),),
            ).fetchone()
        return str(row["last_consolidated_at"]) if row is not None else None

    def record_use(
        self,
        entry_id: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        use_reason: str = "",
    ) -> None:
        with connect(self.db_path) as conn:
            entry = self._select_entry(conn, entry_id, include_deleted=False)
            if entry["status"] == "rejected":
                raise ValueError("rejected memory entries are terminal")
            if entry["status"] != "active":
                raise ValueError("only active memory entries can be recorded as used")
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
            current = self._select_entry(conn, entry_id, include_deleted=False)
            if current["status"] == "rejected":
                # rejected is a terminal audit state: a rejected candidate (and its
                # rejection record) must not be overwritten by a delete, matching the
                # terminal guard already enforced in set_status / record_use.
                raise ValueError("rejected memory entries are terminal")
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

    def _select_distillation(
        self,
        conn: sqlite3.Connection,
        distillation_id: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM memory_distillations WHERE id = ?",
            (distillation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Memory distillation not found: {distillation_id}")
        return row

    def _insert_distillation(
        self,
        conn: sqlite3.Connection,
        distillation: MemoryDistillation,
    ) -> int:
        now = _now()
        created_at = distillation.created_at or now
        updated_at = distillation.updated_at or now
        safe_summary = redact_text(distillation.distilled_summary)
        safe_structured = redact_value(distillation.structured)
        redacted_count = int(safe_structured.redacted_count) + int(safe_summary != distillation.distilled_summary)
        conn.execute(
            """
            INSERT INTO memory_distillations
            (
                id, category, scope_key, distilled_summary, structured_json,
                source_memory_ids_json, support_count, confidence,
                superseded_by, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                distillation.id,
                normalize_memory_type(distillation.category),
                distillation.scope_key,
                safe_summary,
                _dump_json_object(safe_structured.value),
                json.dumps(list(distillation.source_memory_ids), ensure_ascii=False, separators=(",", ":")),
                int(distillation.support_count),
                distillation.confidence,
                distillation.superseded_by,
                distillation.status,
                created_at,
                updated_at,
            ),
        )
        return redacted_count

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

    def _append_distillation_event(
        self,
        conn: sqlite3.Connection,
        distillation_id: str,
        event_type: str,
        *,
        task_id: str | None = None,
        message_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if event_type not in DISTILLATION_AUDIT_EVENT_TYPES:
            raise ValueError(f"unsupported distillation audit event: {event_type}")
        conn.execute(
            """
            INSERT INTO memory_distillation_events
            (id, distillation_id, event_type, task_id, message_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                distillation_id,
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
            FOREIGN KEY(memory_id) REFERENCES agent_memory_entries(id)
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_distillations (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            distilled_summary TEXT NOT NULL,
            structured_json TEXT NOT NULL,
            source_memory_ids_json TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            confidence TEXT NOT NULL,
            superseded_by TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_distill_scope
            ON memory_distillations(scope_key, status)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_distill_category
            ON memory_distillations(category, status)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_distillation_events (
            id TEXT PRIMARY KEY,
            distillation_id TEXT,
            event_type TEXT NOT NULL,
            task_id TEXT,
            message_id TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(distillation_id) REFERENCES memory_distillations(id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_distill_events_distillation
            ON memory_distillation_events(distillation_id, created_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_distill_events_type
            ON memory_distillation_events(event_type, created_at)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_consolidation_state (
            category TEXT PRIMARY KEY,
            last_consolidated_at TEXT NOT NULL
        )
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


def _row_to_distillation(row: sqlite3.Row) -> MemoryDistillation:
    source_ids = json.loads(row["source_memory_ids_json"] or "[]")
    return MemoryDistillation(
        id=row["id"],
        category=row["category"],
        scope_key=row["scope_key"],
        distilled_summary=row["distilled_summary"],
        structured=_load_json_object(row["structured_json"]),
        source_memory_ids=tuple(str(item) for item in source_ids),
        support_count=int(row["support_count"]),
        confidence=row["confidence"],
        superseded_by=row["superseded_by"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _distillation_score(
    distillation: MemoryDistillation,
    keywords: list[str],
    scope_text: str,
) -> int:
    confidence_score = {"high": 300, "medium": 200, "low": 100}.get(distillation.confidence, 0)
    score = confidence_score + min(int(distillation.support_count), 50)
    match_score = 0
    searchable = (
        f"{distillation.category} {distillation.scope_key} {distillation.distilled_summary} "
        f"{json.dumps(distillation.structured, ensure_ascii=False, sort_keys=True)}"
    ).lower()
    if scope_text and scope_text in searchable:
        match_score += 80
    for keyword in keywords:
        if keyword in searchable:
            match_score += 40
    if (keywords or scope_text) and match_score == 0:
        return 0
    return score + match_score


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


def _row_to_distillation_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "distillation_id": row["distillation_id"],
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
