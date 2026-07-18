from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import threading

import pytest

from marvis.db import TaskRepository, connect, init_db
from marvis.db_schema import SCHEMA_VERSION
from marvis.domain import TaskCreate
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestConflictError,
    PendingStrategyRequestDataError,
    PendingStrategyRequestNotFoundError,
    PendingStrategyRequestRepository,
)


def _task(db_path, tmp_path, name: str):
    return TaskRepository(db_path).create_task(
        TaskCreate(
            task_type="strategy",
            model_name=name,
            model_version="v1",
            validator="tester",
            source_dir=str(tmp_path),
        )
    )


def _draft() -> dict:
    return {
        "operation": "backtest",
        "strategy_type": "approval",
        "strategy_spec": {
            "schema_version": "strategy.dsl.v1",
            "strategy_type": "approval",
            "rules": [
                {
                    "id": "r1",
                    "priority": 1,
                    "when": {"field": "risk_score", "op": "lt", "value": 600},
                    "then": {"action": "reject", "reason_code": "secret-rule-marker"},
                }
            ],
            "default_action": {"action": "approve"},
        },
    }


def _identity() -> dict:
    return {
        "kind": "source",
        "source_path": "/task/sample.parquet",
        "size_bytes": 123,
        "mtime_ns": 456,
        "sha256": "a" * 64,
    }


def test_migration_007_adds_task_scoped_pending_strategy_requests(tmp_path):
    db_path = tmp_path / "schema-v6.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY)")
        conn.execute("PRAGMA user_version = 6")

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(pending_strategy_requests)"
            ).fetchall()
        }
        indexes = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list(pending_strategy_requests)"
            ).fetchall()
        }

    assert version == SCHEMA_VERSION
    assert {
        "id",
        "nonce",
        "task_id",
        "validated_draft_json",
        "dataset_identity_json",
        "target_col",
        "payload_sha256",
        "status",
        "created_at",
        "updated_at",
    } <= columns
    assert "idx_pending_strategy_requests_task_status" in indexes


def test_create_round_trips_canonical_payload_and_exposes_only_opaque_reference(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "task-a")
    repo = PendingStrategyRequestRepository(db_path)

    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )

    assert created.status == "pending"
    assert created.id != created.nonce
    assert len(created.id) == 32
    assert len(created.nonce) == 32
    assert len(created.payload_sha256) == 64
    assert created.to_metadata_reference() == {
        "request_id": created.id,
        "payload_sha256": created.payload_sha256,
    }
    assert repo.get(task.id, created.id) == created

    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT validated_draft_json, dataset_identity_json
              FROM pending_strategy_requests WHERE id = ?
            """,
            (created.id,),
        ).fetchone()
    for value in row:
        assert json.dumps(
            json.loads(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) == value


def test_get_and_transitions_are_strictly_task_scoped(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    owner = _task(db_path, tmp_path, "owner")
    other = _task(db_path, tmp_path, "other")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=owner.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )

    assert repo.get(other.id, created.id) is None
    with pytest.raises(PendingStrategyRequestNotFoundError):
        repo.consume(
            task_id=other.id,
            request_id=created.id,
            expected_payload_sha256=created.payload_sha256,
        )
    assert repo.get(owner.id, created.id).status == "pending"


def test_consume_is_one_shot_and_audit_does_not_duplicate_the_dsl(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "consume")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )

    consumed = repo.consume(
        task_id=task.id,
        request_id=created.id,
        expected_payload_sha256=created.payload_sha256,
    )

    assert consumed.status == "consumed"
    assert consumed.updated_at >= consumed.created_at
    assert repo.get(task.id, created.id) == consumed
    with pytest.raises(PendingStrategyRequestConflictError, match="already consumed"):
        repo.consume(
            task_id=task.id,
            request_id=created.id,
            expected_payload_sha256=created.payload_sha256,
        )

    with connect(db_path) as conn:
        audit_rows = conn.execute(
            """
            SELECT kind, inputs_hash, detail_json
              FROM audit WHERE target_ref = ? ORDER BY at, id
            """,
            (created.id,),
        ).fetchall()
    assert [row["kind"] for row in audit_rows] == [
        "strategy.request.create",
        "strategy.request.consume",
    ]
    assert {row["inputs_hash"] for row in audit_rows} == {created.payload_sha256}
    audit_json = "".join(row["detail_json"] for row in audit_rows)
    assert "secret-rule-marker" not in audit_json
    assert "risk_score" not in audit_json


@pytest.mark.parametrize("transition", ["cancel", "invalidate"])
def test_cancel_and_invalidate_are_terminal_one_shot_transitions(tmp_path, transition):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, transition)
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=None,
        target_col=None,
    )

    changed = getattr(repo, transition)(
        task_id=task.id,
        request_id=created.id,
        expected_payload_sha256=created.payload_sha256,
    )

    expected = "cancelled" if transition == "cancel" else "invalidated"
    assert changed.status == expected
    assert changed.dataset_identity is None
    with pytest.raises(PendingStrategyRequestConflictError, match=f"already {expected}"):
        repo.consume(
            task_id=task.id,
            request_id=created.id,
            expected_payload_sha256=created.payload_sha256,
        )


def test_wrong_payload_hash_cannot_change_pending_request(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "hash")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )

    with pytest.raises(PendingStrategyRequestConflictError, match="does not match"):
        repo.consume(
            task_id=task.id,
            request_id=created.id,
            expected_payload_sha256="0" * 64,
        )
    assert repo.get(task.id, created.id).status == "pending"


def test_stored_payload_tampering_is_detected_before_consumption(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "tamper")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE pending_strategy_requests SET validated_draft_json = ? WHERE id = ?",
            ('{"operation":"apply"}', created.id),
        )

    with pytest.raises(PendingStrategyRequestDataError, match="integrity"):
        repo.consume(
            task_id=task.id,
            request_id=created.id,
            expected_payload_sha256=created.payload_sha256,
        )


def test_concurrent_consumers_have_exactly_one_winner(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "race")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )
    barrier = threading.Barrier(2)

    def consume_once() -> str:
        barrier.wait()
        try:
            return repo.consume(
                task_id=task.id,
                request_id=created.id,
                expected_payload_sha256=created.payload_sha256,
            ).status
        except PendingStrategyRequestConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: consume_once(), range(2)))

    assert sorted(outcomes) == ["conflict", "consumed"]


def test_failed_start_release_is_retryable_and_concurrent_release_is_one_shot(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "release-race")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )
    repo.consume(
        task_id=task.id,
        request_id=created.id,
        expected_payload_sha256=created.payload_sha256,
    )
    barrier = threading.Barrier(2)

    def release_once() -> str:
        barrier.wait()
        try:
            return repo.release_after_failed_start(
                task_id=task.id,
                request_id=created.id,
                expected_payload_sha256=created.payload_sha256,
                existing_plan_ids=frozenset(),
            ).status
        except PendingStrategyRequestConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: release_once(), range(2)))

    assert sorted(outcomes) == ["conflict", "pending"]
    assert repo.get(task.id, created.id).status == "pending"


def test_failed_start_release_refuses_when_request_created_a_plan(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "release-plan-guard")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=_identity(),
        target_col="bad",
    )
    repo.consume(
        task_id=task.id,
        request_id=created.id,
        expected_payload_sha256=created.payload_sha256,
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plans(
                id, task_id, goal, source, template_id, autonomy_level,
                status, novel_mode, tier, replan_count, loop_events_json,
                success_criteria_json, created_at, updated_at
            )
            VALUES (?, ?, 'strategy', 'template', 'strategy_analysis', 1,
                    'validated', 'plan_ahead', 'balanced', 0, '[]', '[]', ?, ?)
            """,
            ("new-plan", task.id, created.created_at, created.updated_at),
        )

    with pytest.raises(
        PendingStrategyRequestConflictError,
        match="already created a plan",
    ):
        repo.release_after_failed_start(
            task_id=task.id,
            request_id=created.id,
            expected_payload_sha256=created.payload_sha256,
            existing_plan_ids=frozenset(),
        )

    assert repo.get(task.id, created.id).status == "consumed"


def test_task_delete_cascades_pending_requests(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "cascade")
    repo = PendingStrategyRequestRepository(db_path)
    created = repo.create(
        task_id=task.id,
        validated_draft=_draft(),
        dataset_identity=None,
        target_col=None,
    )
    with connect(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task.id,))

    assert repo.get(task.id, created.id) is None


def test_create_rejects_non_json_or_missing_task_without_partial_rows(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task = _task(db_path, tmp_path, "invalid")
    repo = PendingStrategyRequestRepository(db_path)

    with pytest.raises(PendingStrategyRequestDataError, match="valid JSON"):
        repo.create(
            task_id=task.id,
            validated_draft={"threshold": float("nan")},
            dataset_identity=None,
            target_col=None,
        )
    with pytest.raises(PendingStrategyRequestNotFoundError, match="task not found"):
        repo.create(
            task_id="missing-task",
            validated_draft=_draft(),
            dataset_identity=None,
            target_col=None,
        )

    with connect(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM pending_strategy_requests"
        ).fetchone()[0]
    assert total == 0
