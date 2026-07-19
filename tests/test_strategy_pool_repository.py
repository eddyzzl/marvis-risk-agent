from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sqlite3
import threading

import pytest

import marvis.repositories.strategy_pool as strategy_pool_repository_module
from marvis.db_schema import SCHEMA_VERSION, connect, init_db
from marvis.domain import TaskCreate
from marvis.repositories.strategy_pool import (
    ABSENT_POOL_REVISION,
    ABSENT_POOL_SNAPSHOT_HASH,
    POOL_ARTIFACT_KIND,
    POOL_SCHEMA_VERSION,
    SOURCE_ARTIFACT_KIND,
    StrategyCandidatePoolConflictError,
    StrategyCandidatePoolDataError,
    StrategyCandidatePoolRepository,
    strategy_pool_artifact_content_hash,
    strategy_pool_id,
    strategy_pool_operation_hash,
    strategy_pool_revision_id,
    strategy_pool_snapshot_hash,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed(tmp_path: Path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    tasks = TaskRepository(db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="candidate-pool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other = tasks.create_task(
        TaskCreate(
            model_name="other-pool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "other"),
            task_type="strategy",
            target_col="bad",
        )
    )
    artifacts = TaskArtifactRepository(db_path)
    source_hash = _sha("candidate-asset")
    source = artifacts.register(
        task_id=task.id,
        kind=SOURCE_ARTIFACT_KIND,
        path=str(tmp_path / task.id / "candidate-asset.json"),
        content_hash=source_hash,
        origin_tool="strategy.test_candidate_asset",
        provenance={
            "asset_id": "candidate-asset-a",
            "asset_hash": HASH_A,
            "candidate_id": "candidate-parent-a",
            "evidence_hash": HASH_B,
            "dataset_id": "dataset-a",
            "dataset_content_hash": HASH_C,
            "workspace_revision": 3,
            "workspace_generation": 2,
            "semantic_mapping_hash": HASH_A,
        },
    )
    return db_path, tasks, task, other, artifacts, source


def _entry(source: dict, *, suffix: str, position: int) -> dict:
    return {
        "entry_id": f"pool-entry-{suffix}",
        "rule_id": f"candidate-rule-{suffix}",
        "position": position,
        "source": {
            "artifact_id": source["id"],
            "kind": SOURCE_ARTIFACT_KIND,
            "content_hash": source["content_hash"],
            "asset_id": "candidate-asset-a",
            "asset_hash": HASH_A,
            "candidate_kind": "univariate_refinement",
            "fragment_id": f"candidate-fragment-{suffix}",
            "effect_id": f"candidate-effect-{suffix}",
            "effect_stage": "development",
            "validation_status": "unvalidated",
            "parent_candidate_id": "candidate-parent-a",
            "parent_evidence_hash": HASH_B,
            "evidence_identity": {
                "dataset_id": "dataset-a",
                "dataset_content_hash": HASH_C,
                "workspace_revision": 3,
                "workspace_generation": 2,
                "semantic_mapping_hash": HASH_A,
            },
        },
        "execution": {
            "condition": {"op": "eq", "field": "score", "value": suffix},
            "requirements": [],
        },
        "action": {
            "type": "reject",
            "value": "reject",
            "reason_code": f"R-{suffix}",
            "stop": True,
        },
        "enabled": True,
    }


def _snapshot(
    *,
    task_id: str,
    revision: int,
    parent_revision_id: str | None,
    entries: list[dict],
    kind: str,
    reason: str,
    strategy_type: str = "approval",
) -> dict:
    pool_id = strategy_pool_id(task_id, strategy_type)
    operation_hash = strategy_pool_operation_hash(
        pool_id=pool_id,
        parent_revision_id=parent_revision_id,
        kind=kind,
        reason=reason,
        default_action={
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        entries=entries,
        status="draft",
        validation_status="unvalidated",
    )
    body = {
        "schema_version": POOL_SCHEMA_VERSION,
        "pool_id": pool_id,
        "task_id": task_id,
        "strategy_type": strategy_type,
        "revision": revision,
        "revision_id": strategy_pool_revision_id(
            pool_id, parent_revision_id, operation_hash
        ),
        "parent_revision_id": parent_revision_id,
        "operation": {
            "kind": kind,
            "operation_hash": operation_hash,
            "reason": reason,
        },
        "default_action": {
            "type": "approval",
            "value": "approve",
            "reason_code": None,
            "stop": True,
        },
        "entries": entries,
        "status": "draft",
        "validation_status": "unvalidated",
    }
    return {**body, "snapshot_hash": strategy_pool_snapshot_hash(body)}


def _register_pool_artifact(
    artifacts: TaskArtifactRepository,
    tmp_path: Path,
    snapshot: dict,
) -> dict:
    return artifacts.register(
        task_id=snapshot["task_id"],
        kind=POOL_ARTIFACT_KIND,
        path=str(tmp_path / snapshot["task_id"] / f"{snapshot['revision_id']}.json"),
        content_hash=strategy_pool_artifact_content_hash(snapshot),
        origin_tool="strategy.update_candidate_pool",
        provenance={
            "pool_id": snapshot["pool_id"],
            "revision_id": snapshot["revision_id"],
            "snapshot_hash": snapshot["snapshot_hash"],
        },
    )


def _audit(snapshot: dict) -> dict:
    return {
        "kind": f"strategy.pool.{snapshot['operation']['kind']}",
        "target_ref": snapshot["revision_id"],
        "inputs_hash": snapshot["operation"]["operation_hash"],
        "outcome": "succeeded",
        "detail": {"reason": snapshot["operation"]["reason"]},
    }


def test_schema_v15_and_initial_snapshot_are_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    assert SCHEMA_VERSION == 15
    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 15

    entries = [
        _entry(source, suffix="a", position=0),
        _entry(source, suffix="b", position=1),
    ]
    snapshot = _snapshot(
        task_id=task.id,
        revision=1,
        parent_revision_id=None,
        entries=entries,
        kind="add",
        reason="add two fragments",
    )
    pool_artifact = _register_pool_artifact(artifacts, tmp_path, snapshot)
    repo = StrategyCandidatePoolRepository(db_path)
    created = repo.apply_snapshot(
        snapshot=snapshot,
        expected_revision=ABSENT_POOL_REVISION,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=pool_artifact["id"],
        artifact_content_hash=pool_artifact["content_hash"],
        audit=_audit(snapshot),
    )
    replayed = repo.apply_snapshot(
        snapshot=snapshot,
        expected_revision=ABSENT_POOL_REVISION,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=pool_artifact["id"],
        artifact_content_hash=pool_artifact["content_hash"],
        audit=_audit(snapshot),
    )

    assert created["created"] is True and created["replayed"] is False
    assert replayed["created"] is False and replayed["replayed"] is True
    assert created["snapshot_hash"] == snapshot["snapshot_hash"]
    assert repo.get_current(task.id, "approval") == snapshot
    assert repo.get_revision(task.id, "approval", 1) == snapshot
    for wrong_revision, wrong_hash in (
        (1, snapshot["snapshot_hash"]),
        (0, HASH_A),
    ):
        with pytest.raises(StrategyCandidatePoolConflictError, match="original parent"):
            repo.apply_snapshot(
                snapshot=snapshot,
                expected_revision=wrong_revision,
                expected_snapshot_hash=wrong_hash,
                artifact_id=pool_artifact["id"],
                artifact_content_hash=pool_artifact["content_hash"],
                audit=_audit(snapshot),
            )
    with connect(db_path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM strategy_candidate_pools").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_items"
            ).fetchone()[0]
            == 2
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE target_ref = ?",
                (snapshot["revision_id"],),
            ).fetchone()[0]
            == 1
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE strategy_candidate_pool_revisions SET status = 'draft'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE strategy_candidate_pool_items SET enabled = enabled")


def test_cas_revision_lineage_and_exact_retry_only_while_head(
    tmp_path: Path,
) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    repo = StrategyCandidatePoolRepository(db_path)
    first = _snapshot(
        task_id=task.id,
        revision=1,
        parent_revision_id=None,
        entries=[_entry(source, suffix="a", position=0)],
        kind="add",
        reason="first",
    )
    first_artifact = _register_pool_artifact(artifacts, tmp_path, first)
    repo.apply_snapshot(
        snapshot=first,
        expected_revision=0,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=first_artifact["id"],
        artifact_content_hash=first_artifact["content_hash"],
        audit=_audit(first),
    )

    second_entries = [
        _entry(source, suffix="b", position=0),
        _entry(source, suffix="a", position=1),
    ]
    second = _snapshot(
        task_id=task.id,
        revision=2,
        parent_revision_id=first["revision_id"],
        entries=second_entries,
        kind="reorder",
        reason="full order",
    )
    second_artifact = _register_pool_artifact(artifacts, tmp_path, second)
    repo.apply_snapshot(
        snapshot=second,
        expected_revision=1,
        expected_snapshot_hash=first["snapshot_hash"],
        artifact_id=second_artifact["id"],
        artifact_content_hash=second_artifact["content_hash"],
        audit=_audit(second),
    )
    assert repo.get_current(task.id, "approval") == second
    assert repo.get_revision_by_id(task.id, "approval", first["revision_id"]) == first

    with pytest.raises(
        StrategyCandidatePoolConflictError, match="no longer the current head"
    ):
        repo.apply_snapshot(
            snapshot=first,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            artifact_id=first_artifact["id"],
            artifact_content_hash=first_artifact["content_hash"],
            audit=_audit(first),
        )

    stale = _snapshot(
        task_id=task.id,
        revision=2,
        parent_revision_id=first["revision_id"],
        entries=[_entry(source, suffix="a", position=0)],
        kind="remove",
        reason="stale fork",
    )
    stale_artifact = _register_pool_artifact(artifacts, tmp_path, stale)
    with pytest.raises(StrategyCandidatePoolConflictError, match="stale"):
        repo.apply_snapshot(
            snapshot=stale,
            expected_revision=1,
            expected_snapshot_hash=first["snapshot_hash"],
            artifact_id=stale_artifact["id"],
            artifact_content_hash=stale_artifact["content_hash"],
            audit=_audit(stale),
        )


def test_task_can_own_independent_pool_heads_per_strategy_type(tmp_path: Path) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    repo = StrategyCandidatePoolRepository(db_path)
    snapshots = [
        _snapshot(
            task_id=task.id,
            strategy_type=strategy_type,
            revision=1,
            parent_revision_id=None,
            entries=[_entry(source, suffix=strategy_type, position=0)],
            kind="add",
            reason=f"initialize {strategy_type}",
        )
        for strategy_type in ("approval", "pricing")
    ]

    for snapshot in snapshots:
        artifact = _register_pool_artifact(artifacts, tmp_path, snapshot)
        repo.apply_snapshot(
            snapshot=snapshot,
            expected_revision=ABSENT_POOL_REVISION,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            artifact_id=artifact["id"],
            artifact_content_hash=artifact["content_hash"],
            audit=_audit(snapshot),
        )

    assert repo.get_current(task.id, "approval") == snapshots[0]
    assert repo.get_current(task.id, "pricing") == snapshots[1]
    with connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pools WHERE task_id = ?",
                (task.id,),
            ).fetchone()[0]
            == 2
        )


def test_task_and_artifact_ownership_and_projection_fail_closed(
    tmp_path: Path,
) -> None:
    db_path, _tasks, task, other, artifacts, source = _seed(tmp_path)
    repo = StrategyCandidatePoolRepository(db_path)
    snapshot = _snapshot(
        task_id=task.id,
        revision=1,
        parent_revision_id=None,
        entries=[_entry(source, suffix="a", position=0)],
        kind="add",
        reason="ownership",
    )
    foreign = artifacts.register(
        task_id=other.id,
        kind=POOL_ARTIFACT_KIND,
        path=str(tmp_path / other.id / "foreign-pool.json"),
        content_hash=strategy_pool_artifact_content_hash(snapshot),
        origin_tool="strategy.update_candidate_pool",
        provenance={"foreign": True},
    )
    with pytest.raises(KeyError, match="artifact not found for task"):
        repo.apply_snapshot(
            snapshot=snapshot,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            artifact_id=foreign["id"],
            artifact_content_hash=foreign["content_hash"],
            audit=_audit(snapshot),
        )
    assert repo.get_current(task.id, "approval") is None

    forged = deepcopy(snapshot)
    forged["entries"][0]["source"]["evidence_identity"]["workspace_generation"] = 99
    forged["operation"]["operation_hash"] = strategy_pool_operation_hash(
        pool_id=forged["pool_id"],
        parent_revision_id=None,
        kind="add",
        reason="ownership",
        default_action=forged["default_action"],
        entries=forged["entries"],
        status="draft",
        validation_status="unvalidated",
    )
    forged["revision_id"] = strategy_pool_revision_id(
        forged["pool_id"], None, forged["operation"]["operation_hash"]
    )
    body = {key: value for key, value in forged.items() if key != "snapshot_hash"}
    forged["snapshot_hash"] = strategy_pool_snapshot_hash(body)
    pool_artifact = _register_pool_artifact(artifacts, tmp_path, forged)
    with pytest.raises(StrategyCandidatePoolDataError, match="workspace_generation"):
        repo.apply_snapshot(
            snapshot=forged,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            artifact_id=pool_artifact["id"],
            artifact_content_hash=pool_artifact["content_hash"],
            audit=_audit(forged),
        )


def test_caller_owned_transaction_rolls_back_artifact_pool_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    snapshot = _snapshot(
        task_id=task.id,
        revision=1,
        parent_revision_id=None,
        entries=[_entry(source, suffix="a", position=0)],
        kind="add",
        reason="atomic",
    )
    artifact_hash = strategy_pool_artifact_content_hash(snapshot)
    repo = StrategyCandidatePoolRepository(db_path)

    def fail_audit_write(_conn: sqlite3.Connection, **_payload: object) -> None:
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(
        strategy_pool_repository_module,
        "_write_audit_row",
        fail_audit_write,
    )
    with pytest.raises(RuntimeError, match="injected audit failure"):
        with repo.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            artifact = artifacts.register_on_connection(
                conn,
                task_id=task.id,
                kind=POOL_ARTIFACT_KIND,
                path=str(tmp_path / task.id / "atomic-pool.json"),
                content_hash=artifact_hash,
                origin_tool="strategy.update_candidate_pool",
                provenance={"snapshot_hash": snapshot["snapshot_hash"]},
            )
            repo.apply_snapshot_on_connection(
                conn,
                snapshot=snapshot,
                expected_revision=0,
                expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
                artifact_id=artifact["id"],
                artifact_content_hash=artifact["content_hash"],
                audit=_audit(snapshot),
            )
    assert repo.get_current(task.id, "approval") is None
    with connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE kind = ?",
                (POOL_ARTIFACT_KIND,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind LIKE 'strategy.pool.%'"
            ).fetchone()[0]
            == 0
        )


def test_concurrent_divergent_initial_writes_have_one_cas_winner(
    tmp_path: Path,
) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    snapshots = [
        _snapshot(
            task_id=task.id,
            revision=1,
            parent_revision_id=None,
            entries=[_entry(source, suffix=suffix, position=0)],
            kind="add",
            reason=f"writer-{suffix}",
        )
        for suffix in ("a", "b")
    ]
    pool_artifacts = [
        _register_pool_artifact(artifacts, tmp_path, snapshot) for snapshot in snapshots
    ]
    barrier = threading.Barrier(2)
    results: list[dict] = []
    failures: list[BaseException] = []

    def write(index: int) -> None:
        try:
            barrier.wait(timeout=10)
            results.append(
                StrategyCandidatePoolRepository(db_path).apply_snapshot(
                    snapshot=snapshots[index],
                    expected_revision=0,
                    expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
                    artifact_id=pool_artifacts[index]["id"],
                    artifact_content_hash=pool_artifacts[index]["content_hash"],
                    audit=_audit(snapshots[index]),
                )
            )
        except BaseException as exc:  # asserted by the main test thread
            failures.append(exc)

    threads = [threading.Thread(target=write, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
        assert not thread.is_alive()

    assert len(results) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StrategyCandidatePoolConflictError)
    current = StrategyCandidatePoolRepository(db_path).get_current(task.id, "approval")
    assert current in snapshots


def test_task_delete_cascades_pool_revision_items_and_artifacts(
    tmp_path: Path,
) -> None:
    db_path, tasks, task, _other, artifacts, source = _seed(tmp_path)
    snapshot = _snapshot(
        task_id=task.id,
        revision=1,
        parent_revision_id=None,
        entries=[_entry(source, suffix="a", position=0)],
        kind="add",
        reason="delete task",
    )
    pool_artifact = _register_pool_artifact(artifacts, tmp_path, snapshot)
    repo = StrategyCandidatePoolRepository(db_path)
    repo.apply_snapshot(
        snapshot=snapshot,
        expected_revision=0,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=pool_artifact["id"],
        artifact_content_hash=pool_artifact["content_hash"],
        audit=_audit(snapshot),
    )
    second = _snapshot(
        task_id=task.id,
        revision=2,
        parent_revision_id=snapshot["revision_id"],
        entries=deepcopy(snapshot["entries"]),
        kind="reorder",
        reason="create child revision before deleting task",
    )
    second_artifact = _register_pool_artifact(artifacts, tmp_path, second)
    repo.apply_snapshot(
        snapshot=second,
        expected_revision=1,
        expected_snapshot_hash=snapshot["snapshot_hash"],
        artifact_id=second_artifact["id"],
        artifact_content_hash=second_artifact["content_hash"],
        audit=_audit(second),
    )

    tasks.delete_task(task.id)
    with connect(db_path) as conn:
        for table in (
            "strategy_candidate_pools",
            "strategy_candidate_pool_revisions",
            "strategy_candidate_pool_items",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE task_id = ?", (task.id,)
            ).fetchone()[0]
            == 0
        )
