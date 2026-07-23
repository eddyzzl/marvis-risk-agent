from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading

import pytest

import marvis.db_schema as db_schema_module
from marvis.db_schema import connect, init_db
from marvis.packs.strategy.project_context import (
    build_current_project_snapshot,
    build_report_field,
    build_strategy_project_context_revision,
    build_strategy_project_context_state,
    canonical_strategy_project_context_revision_json,
)
from marvis.repositories.strategy_project_context import (
    PROJECT_CONTEXT_HEAD_SCHEMA_VERSION,
    StrategyProjectContextConflictError,
    StrategyProjectContextDataError,
    StrategyProjectContextNotFoundError,
    StrategyProjectContextRepository,
)


_CREATED_AT = "2026-07-22T08:00:00+00:00"


def _seed_task(db_path: Path, task_id: str = "strategy-task-1", *, task_type="strategy"):
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, task_type, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                task_type,
                "project context fixture",
                "v1",
                "qa",
                "/tmp/project-context-fixture",
                "draft",
                "",
                _CREATED_AT,
                _CREATED_AT,
            ),
        )
    return task_id


def _unavailable_field() -> dict:
    return build_report_field(
        value=None,
        availability="unavailable",
        origin="repository",
        source_refs=[],
        note="No governed evidence in this fixture.",
    )


def _state(task_id: str, *, as_of: str = "2026-07-22") -> dict:
    snapshot = build_current_project_snapshot(
        task_id=task_id,
        as_of=as_of,
        scope=_unavailable_field(),
        dataset_refs=[],
        workspace_ref=None,
        champion_strategy_ref=None,
        status_fields={
            "volume": _unavailable_field(),
            "approval": _unavailable_field(),
            "risk": _unavailable_field(),
            "economics": _unavailable_field(),
        },
        metric_definition_refs=[],
        metric_observation_refs=[],
        monthly_observation_refs=[],
        segment_observation_refs=[],
        maturity_summary=_unavailable_field(),
        user_context_fields=[],
        red_flags=[],
        tool_run_refs=[],
    )
    return build_strategy_project_context_state(
        task_id=task_id,
        as_of=as_of,
        current_project_snapshot=snapshot,
        historical_strategy_reviews=[],
        missing_information_records=[],
        source_refs=[],
        red_flags=[],
    )


def _revision(
    state: dict,
    *,
    revision: int,
    parent_revision_id: str | None,
    parent_state_hash: str | None,
    operation_kind: str = "refresh",
) -> dict:
    return build_strategy_project_context_revision(
        state=state,
        revision=revision,
        parent_revision_id=parent_revision_id,
        parent_state_hash=parent_state_hash,
        operation_kind=operation_kind,
    )


def _save_initial(db_path: Path, task_id: str) -> tuple[StrategyProjectContextRepository, dict]:
    repo = StrategyProjectContextRepository(db_path)
    first = _revision(
        _state(task_id),
        revision=1,
        parent_revision_id=None,
        parent_state_hash=None,
    )
    assert repo.refresh(
        revision=first,
        expected_revision=0,
        expected_revision_id=None,
        expected_state_hash=None,
        created_at=_CREATED_AT,
    ) == first
    return repo, first


def test_migration_019_is_registered_idempotent_and_creates_guarded_ledger(tmp_path):
    db_path = tmp_path / "migration.sqlite"
    task_id = _seed_task(db_path)
    init_db(db_path)

    with connect(db_path) as conn:
        assert (
            conn.execute("PRAGMA user_version").fetchone()[0]
            == db_schema_module.SCHEMA_VERSION
            == 20
        )
        assert (
            19,
            db_schema_module._migration_019_strategy_project_context,
        ) in db_schema_module._MIGRATIONS
        assert db_schema_module._MIGRATIONS[-1] == (
            20,
            db_schema_module._migration_020_strategy_report_revisions,
        )
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' "
                "AND name LIKE 'trg_strategy_project_context_%'"
            )
        }
        task = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()

    assert {
        "strategy_project_context_heads",
        "strategy_project_context_revisions",
    } <= tables
    assert {
        "trg_strategy_project_context_revisions_parent",
        "trg_strategy_project_context_revisions_head_parent",
        "trg_strategy_project_context_heads_target_update",
        "trg_strategy_project_context_heads_immutable_fields",
        "trg_strategy_project_context_revisions_immutable_update",
        "trg_strategy_project_context_revisions_immutable_delete",
    } <= triggers
    assert task["id"] == task_id


def test_initial_refresh_roundtrips_and_exact_current_head_retry_is_idempotent(tmp_path):
    db_path = tmp_path / "initial.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)

    replay = repo.refresh(
        revision=first,
        expected_revision=0,
        expected_revision_id=None,
        expected_state_hash=None,
        created_at="2026-07-22T09:00:00+00:00",
    )

    assert replay == first
    assert repo.get_current(task_id) == first
    assert repo.get_revision(task_id, 1) == first
    assert repo.get_revision_by_id(task_id, first["revision_id"]) == first
    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_project_context_revisions"
        ).fetchone()[0] == 1
        head = conn.execute(
            "SELECT * FROM strategy_project_context_heads WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    assert head["schema_version"] == PROJECT_CONTEXT_HEAD_SCHEMA_VERSION
    assert tuple(
        head[key]
        for key in ("current_revision", "current_revision_id", "current_state_hash")
    ) == (1, first["revision_id"], first["state_hash"])


def test_exact_retry_is_stale_after_a_later_revision_advances_head(tmp_path):
    db_path = tmp_path / "stale-replay.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)
    second = _revision(
        _state(task_id, as_of="2026-07-23"),
        revision=2,
        parent_revision_id=first["revision_id"],
        parent_state_hash=first["state_hash"],
    )
    repo.refresh(
        revision=second,
        expected_revision=1,
        expected_revision_id=first["revision_id"],
        expected_state_hash=first["state_hash"],
    )

    with pytest.raises(
        StrategyProjectContextConflictError,
        match="no longer the current head",
    ):
        repo.refresh(
            revision=first,
            expected_revision=0,
            expected_revision_id=None,
            expected_state_hash=None,
        )


def test_no_change_refresh_returns_current_without_creating_revision(tmp_path):
    db_path = tmp_path / "no-change.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)
    no_change = _revision(
        _state(task_id),
        revision=2,
        parent_revision_id=first["revision_id"],
        parent_state_hash=first["state_hash"],
    )

    result = repo.refresh(
        revision=no_change,
        expected_revision=1,
        expected_revision_id=first["revision_id"],
        expected_state_hash=first["state_hash"],
    )

    assert result == first
    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_project_context_revisions"
        ).fetchone()[0] == 1
        head = conn.execute(
            "SELECT current_revision, current_revision_id, current_state_hash "
            "FROM strategy_project_context_heads WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    assert tuple(head) == (1, first["revision_id"], first["state_hash"])


def test_every_expected_head_component_participates_in_cas(tmp_path):
    db_path = tmp_path / "triple-cas.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)
    second = _revision(
        _state(task_id, as_of="2026-07-23"),
        revision=2,
        parent_revision_id=first["revision_id"],
        parent_state_hash=first["state_hash"],
    )
    stale_triples = (
        (2, first["revision_id"], first["state_hash"]),
        (1, "stale-revision-id", first["state_hash"]),
        (1, first["revision_id"], "f" * 64),
    )

    for expected_revision, expected_id, expected_hash in stale_triples:
        with pytest.raises(StrategyProjectContextConflictError, match="stale"):
            repo.refresh(
                revision=second,
                expected_revision=expected_revision,
                expected_revision_id=expected_id,
                expected_state_hash=expected_hash,
            )

    assert repo.get_current(task_id) == first


def test_concurrent_refresh_has_one_winner_and_one_conflict(tmp_path):
    db_path = tmp_path / "concurrent.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)
    candidates = [
        _revision(
            _state(task_id, as_of=as_of),
            revision=2,
            parent_revision_id=first["revision_id"],
            parent_state_hash=first["state_hash"],
            operation_kind=operation,
        )
        for as_of, operation in (
            ("2026-07-23", "scheduled_refresh"),
            ("2026-07-24", "user_refresh"),
        )
    ]
    barrier = threading.Barrier(2)
    records: list[dict] = []
    errors: list[BaseException] = []

    def writer(candidate: dict) -> None:
        barrier.wait()
        try:
            records.append(
                repo.refresh(
                    revision=candidate,
                    expected_revision=1,
                    expected_revision_id=first["revision_id"],
                    expected_state_hash=first["state_hash"],
                )
            )
        except BaseException as exc:  # noqa: BLE001 - captured for thread assertion
            errors.append(exc)

    writers = [threading.Thread(target=writer, args=(item,)) for item in candidates]
    for writer_thread in writers:
        writer_thread.start()
    for writer_thread in writers:
        writer_thread.join(timeout=10)

    assert all(not writer_thread.is_alive() for writer_thread in writers)
    assert len(records) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], StrategyProjectContextConflictError)
    assert repo.get_current(task_id) == records[0]
    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_project_context_revisions"
        ).fetchone()[0] == 2


def test_database_blocks_revision_and_head_tampering(tmp_path):
    db_path = tmp_path / "immutable.sqlite"
    task_id = _seed_task(db_path)
    _repo, first = _save_initial(db_path, task_id)
    second = _revision(
        _state(task_id, as_of="2026-07-23"),
        revision=2,
        parent_revision_id=first["revision_id"],
        parent_state_hash=first["state_hash"],
    )

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE strategy_project_context_revisions "
                "SET content_hash = ? WHERE revision_id = ?",
                ("f" * 64, first["revision_id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "DELETE FROM strategy_project_context_revisions "
                "WHERE revision_id = ?",
                (first["revision_id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE strategy_project_context_heads "
                "SET updated_at = ? WHERE task_id = ?",
                ("2026-07-22T10:00:00+00:00", task_id),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="parent|based on head",
        ):
            conn.execute(
                """
                INSERT INTO strategy_project_context_revisions(
                    revision_id, schema_version, producer_version, task_id,
                    revision, parent_revision_id, parent_state_hash,
                    operation_kind, operation_hash, revision_json, state_hash,
                    content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    second["revision_id"],
                    second["schema_version"],
                    second["producer_version"],
                    second["task_id"],
                    second["revision"],
                    second["parent_revision_id"],
                    "f" * 64,
                    second["operation_kind"],
                    second["operation_hash"],
                    canonical_strategy_project_context_revision_json(second),
                    second["state_hash"],
                    second["content_hash"],
                    _CREATED_AT,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="head mismatch"):
            conn.execute(
                """
                UPDATE strategy_project_context_heads
                   SET current_revision = 2, current_revision_id = ?,
                       current_state_hash = ?, updated_at = ?
                 WHERE task_id = ?
                """,
                (
                    "forged-revision",
                    "f" * 64,
                    "2026-07-22T10:00:00+00:00",
                    task_id,
                ),
            )


def test_repository_detects_persisted_tampering_after_trigger_removal(tmp_path):
    db_path = tmp_path / "tampered.sqlite"
    task_id = _seed_task(db_path)
    repo, first = _save_initial(db_path, task_id)
    noncanonical = json.dumps(first, ensure_ascii=False, sort_keys=True, indent=2)

    with connect(db_path) as conn:
        conn.execute(
            "DROP TRIGGER trg_strategy_project_context_revisions_immutable_update"
        )
        conn.execute(
            "UPDATE strategy_project_context_revisions "
            "SET revision_json = ? WHERE revision_id = ?",
            (noncanonical, first["revision_id"]),
        )

    with pytest.raises(StrategyProjectContextDataError, match="not canonical"):
        repo.get_current(task_id)


def test_task_delete_cascades_context_head_and_revisions(tmp_path):
    db_path = tmp_path / "cascade.sqlite"
    task_id = _seed_task(db_path)
    _repo, _first = _save_initial(db_path, task_id)

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="task-owned"):
            conn.execute(
                "DELETE FROM strategy_project_context_heads WHERE task_id = ?",
                (task_id,),
            )
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_project_context_heads"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_project_context_revisions"
        ).fetchone()[0] == 0


def test_repository_rejects_missing_and_non_strategy_tasks(tmp_path):
    db_path = tmp_path / "task-boundary.sqlite"
    non_strategy = _seed_task(db_path, "validation-task", task_type="validation")
    repo = StrategyProjectContextRepository(db_path)

    with pytest.raises(StrategyProjectContextNotFoundError, match="missing-task"):
        repo.refresh(
            revision=_revision(
                _state("missing-task"),
                revision=1,
                parent_revision_id=None,
                parent_state_hash=None,
            ),
            expected_revision=0,
            expected_revision_id=None,
            expected_state_hash=None,
        )
    with pytest.raises(StrategyProjectContextDataError, match="strategy task"):
        repo.refresh(
            revision=_revision(
                _state(non_strategy),
                revision=1,
                parent_revision_id=None,
                parent_state_hash=None,
            ),
            expected_revision=0,
            expected_revision_id=None,
            expected_state_hash=None,
        )
