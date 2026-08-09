"""Idempotent, standalone SQLite schema for local operations state."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect


OPERATIONS_SCHEMA_VERSION = 1


class OperationsSchemaVersionError(RuntimeError):
    """The database contains a newer operations schema than this runtime."""


_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations_schema_meta (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version INTEGER NOT NULL,
    installed_at TEXT NOT NULL
)
"""

_OPERATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations_schedules (
    schedule_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 1),
    schema_version TEXT NOT NULL,
    contract_json TEXT NOT NULL,
    contract_hash TEXT NOT NULL CHECK (length(contract_hash) = 64),
    created_at TEXT NOT NULL,
    PRIMARY KEY (schedule_id, revision)
);

CREATE INDEX IF NOT EXISTS operations_schedules_latest_idx
    ON operations_schedules(schedule_id, revision DESC);

CREATE TABLE IF NOT EXISTS operations_periods (
    schedule_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    schedule_revision INTEGER NOT NULL,
    period_index INTEGER NOT NULL CHECK (period_index >= 0),
    period_starts_at TEXT NOT NULL,
    period_ends_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('running', 'retry_wait', 'succeeded', 'terminal_failed')
    ),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TEXT,
    next_attempt_at TEXT,
    outcome_id TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (schedule_id, period_key),
    FOREIGN KEY (schedule_id, schedule_revision)
        REFERENCES operations_schedules(schedule_id, revision)
);

CREATE INDEX IF NOT EXISTS operations_periods_retry_idx
    ON operations_periods(state, next_attempt_at, schedule_id, period_ends_at);

CREATE TABLE IF NOT EXISTS operations_runs (
    run_id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    schedule_revision INTEGER NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    owner_id TEXT NOT NULL,
    lease_token TEXT NOT NULL UNIQUE,
    claimed_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    UNIQUE (schedule_id, period_key, attempt_number),
    FOREIGN KEY (schedule_id, period_key)
        REFERENCES operations_periods(schedule_id, period_key)
);

CREATE TABLE IF NOT EXISTS operations_outcomes (
    outcome_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    schedule_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    outcome_json TEXT NOT NULL,
    outcome_hash TEXT NOT NULL CHECK (length(outcome_hash) = 64),
    recorded_at TEXT NOT NULL,
    UNIQUE (schedule_id, period_key),
    FOREIGN KEY (run_id) REFERENCES operations_runs(run_id)
);

CREATE TABLE IF NOT EXISTS operations_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    schedule_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    run_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (schedule_id, period_key)
        REFERENCES operations_periods(schedule_id, period_key),
    FOREIGN KEY (run_id) REFERENCES operations_runs(run_id)
);

CREATE INDEX IF NOT EXISTS operations_events_period_idx
    ON operations_events(schedule_id, period_key, sequence);

CREATE TABLE IF NOT EXISTS operations_notification_outbox (
    notification_id TEXT PRIMARY KEY,
    outcome_id TEXT NOT NULL UNIQUE,
    schedule_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    run_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL CHECK (length(payload_hash) = 64),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'sending', 'sent', 'exhausted')
    ),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
    initial_backoff_seconds REAL NOT NULL CHECK (initial_backoff_seconds >= 0),
    backoff_multiplier REAL NOT NULL CHECK (backoff_multiplier > 0),
    max_backoff_seconds REAL NOT NULL CHECK (max_backoff_seconds >= 0),
    next_attempt_at TEXT,
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (outcome_id) REFERENCES operations_outcomes(outcome_id),
    FOREIGN KEY (run_id) REFERENCES operations_runs(run_id),
    FOREIGN KEY (schedule_id, period_key)
        REFERENCES operations_periods(schedule_id, period_key)
);

CREATE INDEX IF NOT EXISTS operations_notification_outbox_ready_idx
    ON operations_notification_outbox(status, next_attempt_at, created_at);

CREATE TRIGGER IF NOT EXISTS operations_schedules_no_update
BEFORE UPDATE ON operations_schedules
BEGIN
    SELECT RAISE(ABORT, 'operations_schedules is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_schedules_no_delete
BEFORE DELETE ON operations_schedules
BEGIN
    SELECT RAISE(ABORT, 'operations_schedules is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_runs_no_update
BEFORE UPDATE ON operations_runs
BEGIN
    SELECT RAISE(ABORT, 'operations_runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_runs_no_delete
BEFORE DELETE ON operations_runs
BEGIN
    SELECT RAISE(ABORT, 'operations_runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_outcomes_no_update
BEFORE UPDATE ON operations_outcomes
BEGIN
    SELECT RAISE(ABORT, 'operations_outcomes is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_outcomes_no_delete
BEFORE DELETE ON operations_outcomes
BEGIN
    SELECT RAISE(ABORT, 'operations_outcomes is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_events_no_update
BEFORE UPDATE ON operations_events
BEGIN
    SELECT RAISE(ABORT, 'operations_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_events_no_delete
BEFORE DELETE ON operations_events
BEGIN
    SELECT RAISE(ABORT, 'operations_events is append-only');
END;

CREATE TRIGGER IF NOT EXISTS operations_notification_outbox_payload_immutable
BEFORE UPDATE ON operations_notification_outbox
WHEN NEW.notification_id != OLD.notification_id
  OR NEW.outcome_id != OLD.outcome_id
  OR NEW.schedule_id != OLD.schedule_id
  OR NEW.period_key != OLD.period_key
  OR NEW.run_id != OLD.run_id
  OR NEW.schema_version != OLD.schema_version
  OR NEW.payload_json != OLD.payload_json
  OR NEW.payload_hash != OLD.payload_hash
BEGIN
    SELECT RAISE(ABORT, 'operations notification payload is immutable');
END;

CREATE TRIGGER IF NOT EXISTS operations_notification_outbox_no_delete
BEFORE DELETE ON operations_notification_outbox
BEGIN
    SELECT RAISE(ABORT, 'operations_notification_outbox is retained for audit');
END;
"""


def install_operations_schema(db_path: Path) -> None:
    """Install operations-owned tables without changing the main schema version."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(_META_SCHEMA)
        row = conn.execute(
            "SELECT schema_version FROM operations_schema_meta WHERE singleton_id = 1"
        ).fetchone()
        if row is not None and int(row["schema_version"]) > OPERATIONS_SCHEMA_VERSION:
            raise OperationsSchemaVersionError(
                "operations schema is newer than this runtime: "
                f"found {int(row['schema_version'])}, expected "
                f"{OPERATIONS_SCHEMA_VERSION}"
            )
        conn.executescript(_OPERATIONS_SCHEMA)
        conn.execute(
            """
            INSERT INTO operations_schema_meta(singleton_id, schema_version, installed_at)
            VALUES (1, ?, ?)
            ON CONFLICT(singleton_id) DO UPDATE SET
                schema_version = excluded.schema_version
            """,
            (OPERATIONS_SCHEMA_VERSION, datetime.now(UTC).isoformat()),
        )
