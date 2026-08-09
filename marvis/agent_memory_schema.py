"""Agent-memory DDL kept below both the DB facade and Agent runtime."""

from __future__ import annotations

import sqlite3


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


__all__ = ["ensure_agent_memory_schema"]
