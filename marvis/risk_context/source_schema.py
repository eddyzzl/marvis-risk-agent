"""Additive source-only schema, initialized by the normal application and worker."""

from marvis.db_schema import connect


def ensure_source_schema(db_path):
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS source_profiles (
          id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_by TEXT NOT NULL,
          created_at REAL NOT NULL, failures INTEGER NOT NULL DEFAULT 0,
          open_until REAL NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS source_grants (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
          payload TEXT NOT NULL, created_by TEXT NOT NULL, created_at REAL NOT NULL,
          revoked_at REAL, revoked_by TEXT);
        CREATE TABLE IF NOT EXISTS source_requests (
          task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, id TEXT NOT NULL,
          profile_id TEXT NOT NULL REFERENCES source_profiles(id),
          grant_id TEXT NOT NULL REFERENCES source_grants(id),
          actor_id TEXT NOT NULL, contract TEXT NOT NULL, contract_hash TEXT NOT NULL,
          signature TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
          state TEXT NOT NULL, owner TEXT, lease_until REAL,
          artifact_id TEXT, error_code TEXT,
          PRIMARY KEY(task_id,id));
        CREATE TABLE IF NOT EXISTS source_attempts (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE, request_id TEXT NOT NULL,
          profile_id TEXT NOT NULL, method TEXT NOT NULL, started_at REAL NOT NULL,
          finished_at REAL, outcome TEXT);
        CREATE INDEX IF NOT EXISTS source_attempt_window
          ON source_attempts(profile_id,started_at);
        CREATE INDEX IF NOT EXISTS source_request_task ON source_requests(task_id,created_at);
        """)
