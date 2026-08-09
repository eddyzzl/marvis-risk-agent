from __future__ import annotations

import sqlite3

import pytest

from marvis.db_schema import SCHEMA_VERSION, connect, init_db


def test_production_governance_migration_and_audit_events_are_append_only(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        triggers = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }

    assert version == SCHEMA_VERSION
    assert {
        "production_principals",
        "production_promotion_requests",
        "production_promotion_approvals",
        "production_deployments",
        "production_environment_heads",
        "production_governance_events",
        "production_deployment_manifests",
        "production_activation_evidence",
    } <= tables
    assert {
        "trg_production_governance_events_no_update",
        "trg_production_governance_events_no_delete",
        "trg_production_promotion_approvals_no_update",
        "trg_production_promotion_approvals_no_delete",
        "trg_production_deployment_manifests_no_update",
        "trg_production_deployment_manifests_no_delete",
        "trg_production_activation_evidence_no_update",
        "trg_production_activation_evidence_no_delete",
    } <= triggers

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO production_governance_events(
                sequence, id, event_type, actor_principal_id, actor_role,
                target_type, target_id, environment, payload_json,
                previous_event_hash, event_hash, at
            ) VALUES (1, 'event-1', 'test', 'server', 'admin', 'test',
                      'target', NULL, '{}', ?, ?, '2026-08-01T00:00:00+00:00')
            """,
            ("0" * 64, "1" * 64),
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE production_governance_events SET payload_json = '{\"x\":1}'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM production_governance_events")
