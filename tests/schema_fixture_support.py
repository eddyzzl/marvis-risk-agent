"""Shared helpers for constructing honest historical-schema test fixtures."""

from __future__ import annotations

import sqlite3


def install_v1_plan_step_runs_predecessor(conn: sqlite3.Connection) -> None:
    """Install the v1 ``plan_step_runs`` shape required by later migrations.

    Migration-specific tests intentionally minimize the table they own. A
    database stamped v1 or later must nevertheless retain this unrelated
    baseline table so append-only migrations 31-33 can add receipt columns and
    immutability triggers. Foreign keys are omitted because these focused
    fixtures do not necessarily install the otherwise unrelated plan tables.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_step_runs (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            tool_ref TEXT NOT NULL,
            status TEXT NOT NULL,
            input_json TEXT NOT NULL DEFAULT '{}',
            output_ref TEXT,
            error TEXT,
            error_kind TEXT,
            duration_ms INTEGER,
            side_effects_json TEXT NOT NULL DEFAULT '[]',
            started_at TEXT NOT NULL,
            finished_at TEXT
        )
        """
    )
