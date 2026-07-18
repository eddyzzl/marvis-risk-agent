from __future__ import annotations

import hashlib
import sqlite3

import pytest

from marvis.db_schema import connect, init_db
from marvis.repositories.task_artifacts import (
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactRepository,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_task(db_path, task_id: str) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'artifact task', 'v1', 'tester', '/tmp/source',
                      'draft', '', '2026-07-18T00:00:00+00:00',
                      '2026-07-18T00:00:00+00:00')
            """,
            (task_id,),
        )


def _register(
    repo: TaskArtifactRepository,
    *,
    task_id: str = "task-1",
    kind: str = "strategy_plan",
    path: str = "outputs/strategy-plan.json",
    content_hash: str | None = None,
    origin_tool: str = "strategy.render_plan",
    provenance: dict | None = None,
):
    return repo.register(
        task_id=task_id,
        kind=kind,
        path=path,
        content_hash=content_hash or _sha("strategy-plan"),
        origin_tool=origin_tool,
        provenance=provenance or {"plan_id": "plan-1", "revision": 1},
        created_at="2026-07-18T01:02:03+00:00",
    )


def test_register_is_stable_idempotent_and_task_scoped(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    _seed_task(db_path, "task-2")
    repo = TaskArtifactRepository(db_path)

    first = _register(repo, provenance={"revision": 1, "plan_id": "plan-1"})
    replay = _register(repo, provenance={"plan_id": "plan-1", "revision": 1})

    assert replay == first
    assert first == {
        "id": first["id"],
        "task_id": "task-1",
        "kind": "strategy_plan",
        "path": "outputs/strategy-plan.json",
        "content_hash": _sha("strategy-plan"),
        "origin_tool": "strategy.render_plan",
        "provenance": {"plan_id": "plan-1", "revision": 1},
        "created_at": "2026-07-18T01:02:03+00:00",
    }
    assert repo.list_for_task("task-1") == [first]
    assert repo.list_for_task("task-2") == []
    assert repo.get_for_task("task-1", first["id"]) == first
    assert repo.get_for_task("task-2", first["id"]) is None
    assert repo.get_for_task("task-1", "missing") is None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"content_hash": _sha("drift")}, "content_hash"),
        ({"origin_tool": "strategy.other_tool"}, "origin_tool"),
        ({"provenance": {"plan_id": "plan-2", "revision": 1}}, "provenance"),
    ],
)
def test_register_rejects_drift_for_same_task_kind_and_path(
    tmp_path, override, match
):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    repo = TaskArtifactRepository(db_path)
    original = _register(repo)

    with pytest.raises(TaskArtifactConflictError, match=match):
        _register(repo, **override)

    assert repo.list_for_task("task-1") == [original]


def test_same_path_can_be_registered_for_another_kind_or_task(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    _seed_task(db_path, "task-2")
    repo = TaskArtifactRepository(db_path)

    first = _register(repo)
    another_kind = _register(repo, kind="strategy_report")
    another_task = _register(repo, task_id="task-2")

    assert len({first["id"], another_kind["id"], another_task["id"]}) == 3
    assert repo.list_for_task("task-1") == sorted(
        [first, another_kind], key=lambda row: (row["created_at"], row["id"])
    )
    assert repo.list_for_task("task-2") == [another_task]


def test_register_on_connection_participates_in_callers_transaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    repo = TaskArtifactRepository(db_path)

    with repo.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        registered = repo.register_on_connection(
            conn,
            task_id="task-1",
            kind="strategy_plan",
            path="outputs/strategy-plan.json",
            content_hash=_sha("strategy-plan"),
            origin_tool="strategy.render_plan",
            provenance={"plan_id": "plan-1", "revision": 1},
            created_at="2026-07-18T01:02:03+00:00",
        )
        conn.rollback()

    assert registered["task_id"] == "task-1"
    assert repo.list_for_task("task-1") == []


def test_stable_id_does_not_depend_on_database_or_created_at(tmp_path):
    ids = []
    for index in range(2):
        db_path = tmp_path / f"app-{index}.sqlite"
        _seed_task(db_path, "task-1")
        repo = TaskArtifactRepository(db_path)
        record = repo.register(
            task_id="task-1",
            kind="strategy_plan",
            path="outputs/strategy-plan.json",
            content_hash=_sha("strategy-plan"),
            origin_tool="strategy.render_plan",
            provenance={"plan_id": "plan-1", "revision": 1},
            created_at=f"2026-07-18T01:02:0{index}+00:00",
        )
        ids.append(record["id"])

    assert ids[0] == ids[1]


def test_register_validates_task_and_canonical_inputs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    repo = TaskArtifactRepository(db_path)

    with pytest.raises(KeyError, match="task not found"):
        _register(repo, task_id="missing")
    with pytest.raises(TaskArtifactDataError, match="content_hash"):
        _register(repo, content_hash="not-a-sha256")
    with pytest.raises(TaskArtifactDataError, match="provenance"):
        repo.register(
            task_id="task-1",
            kind="strategy_plan",
            path="outputs/strategy-plan.json",
            content_hash=_sha("strategy-plan"),
            origin_tool="strategy.render_plan",
            provenance=["not", "an", "object"],
        )


def test_registry_rows_are_immutable_and_follow_task_ownership(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_task(db_path, "task-1")
    repo = TaskArtifactRepository(db_path)
    record = _register(repo)

    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE task_artifacts SET path = 'outputs/drift.json' WHERE id = ?",
                (record["id"],),
            )
        conn.execute("DELETE FROM tasks WHERE id = 'task-1'")

    assert repo.list_for_task("task-1") == []
