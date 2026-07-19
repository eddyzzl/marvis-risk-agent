from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

import pytest

import marvis.db_schema as db_schema_module
import marvis.repositories.strategy_pool as strategy_pool_repository_module
from marvis.db_schema import SCHEMA_VERSION, connect, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
    verified_fragment_pool_parts,
)
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _legacy_v1_operation_hash(
    *,
    pool_id: str,
    entries: list[dict],
    default_action: dict,
    reason: str,
) -> str:
    payload = {
        "schema_version": "strategy.candidate-pool-operation.v1",
        "pool_id": pool_id,
        "parent_revision_id": None,
        "kind": "add_candidate",
        "reason": reason,
        "default_action": default_action,
        "entries": entries,
        "status": "draft",
        "validation_status": "unvalidated",
    }
    return _sha(_canonical_json(payload))


def _seed_legacy_v15_pool(tmp_path: Path):
    db_path = tmp_path / "legacy-v15.sqlite"
    with connect(db_path) as conn:
        for version, migration in db_schema_module._MIGRATIONS:
            if version > 15:
                break
            migration(conn)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()

    tasks = TaskRepository(db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="legacy-candidate-pool",
            model_version="v1",
            validator="qa",
            source_dir=str(tmp_path / "legacy-source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    artifacts = TaskArtifactRepository(db_path)
    source_bytes = b'{"schema_version":"legacy-candidate-asset-fixture.v1"}'
    source_path = tmp_path / task.id / "legacy-candidate.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    source = artifacts.register(
        task_id=task.id,
        kind=SOURCE_ARTIFACT_KIND,
        path=str(source_path),
        content_hash=hashlib.sha256(source_bytes).hexdigest(),
        origin_tool="strategy.refine_univariate_candidate",
        provenance={"schema_version": "legacy-fixture.v1"},
    )
    default_action = {
        "type": "approval",
        "value": "approve",
        "reason_code": None,
        "stop": True,
    }
    action = {
        "type": "reject",
        "value": "reject",
        "reason_code": "LEGACY-R1",
        "stop": True,
    }
    pool_id = strategy_pool_id(task.id, "approval")
    entry = {
        "entry_id": "pool-entry-legacy",
        "rule_id": "candidate-rule-legacy",
        "position": 0,
        "source": {
            "artifact_id": source["id"],
            "kind": SOURCE_ARTIFACT_KIND,
            "content_hash": source["content_hash"],
            "asset_id": "candidate-asset-legacy",
            "asset_hash": HASH_A,
            "candidate_kind": "univariate_refinement",
            "fragment_id": "candidate-rule-legacy",
            "effect_id": "candidate-effect-legacy",
            "effect_stage": "development",
            "validation_status": "unvalidated",
            "parent_candidate_id": "candidate-parent-legacy",
            "parent_evidence_hash": HASH_B,
            "evidence_identity": {
                "dataset_id": "dataset-legacy",
                "dataset_content_hash": HASH_C,
                "workspace_revision": 3,
                "workspace_generation": 2,
                "semantic_mapping_hash": HASH_A,
            },
        },
        "execution": {
            "condition": {
                "op": "compare",
                "field": "score",
                "operator": ">",
                "value": 700,
                "missing": "no_match",
            },
            "requirements": [],
        },
        "action": action,
        "enabled": True,
    }
    reason = "legacy v1 draft"
    operation_hash = _legacy_v1_operation_hash(
        pool_id=pool_id,
        entries=[entry],
        default_action=default_action,
        reason=reason,
    )
    revision_id = strategy_pool_revision_id(pool_id, None, operation_hash)
    body = {
        "schema_version": "strategy.candidate-pool.v1",
        "pool_id": pool_id,
        "task_id": task.id,
        "strategy_type": "approval",
        "revision": 1,
        "revision_id": revision_id,
        "parent_revision_id": None,
        "operation": {
            "kind": "add_candidate",
            "operation_hash": operation_hash,
            "reason": reason,
        },
        "default_action": default_action,
        "entries": [entry],
        "status": "draft",
        "validation_status": "unvalidated",
    }
    snapshot_hash = _sha(_canonical_json(body))
    snapshot = {**body, "snapshot_hash": snapshot_hash}
    pool_bytes = _canonical_json(snapshot).encode("utf-8")
    pool_path = tmp_path / task.id / "legacy-pool-v1.json"
    pool_path.write_bytes(pool_bytes)
    pool_artifact = artifacts.register(
        task_id=task.id,
        kind=POOL_ARTIFACT_KIND,
        path=str(pool_path),
        content_hash=hashlib.sha256(pool_bytes).hexdigest(),
        origin_tool="strategy.add_candidate_to_pool",
        provenance={
            "schema_version": "strategy.candidate-pool-artifact.v1",
            "pool_id": pool_id,
            "revision_id": revision_id,
            "snapshot_hash": snapshot_hash,
        },
    )
    old_absent_hash = (
        "9024538661b531de814a43e87e932bf39b4b87522525f7a7afea1bf5bf8968ee"
    )
    timestamp = "2026-07-19T00:00:00+00:00"
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO strategy_candidate_pools(
                id, schema_version, task_id, strategy_type, current_revision,
                current_revision_id, current_snapshot_hash, created_at, updated_at
            ) VALUES (?, 'strategy.candidate-pool-head.v1', ?, 'approval',
                      0, NULL, ?, ?, ?)
            """,
            (pool_id, task.id, old_absent_hash, timestamp, timestamp),
        )
        conn.execute(
            """
            INSERT INTO strategy_candidate_pool_revisions(
                id, schema_version, pool_id, task_id, strategy_type, revision,
                parent_revision_id, parent_snapshot_hash, operation_kind,
                operation_hash, operation_reason, default_action_json, status,
                validation_status, snapshot_json, snapshot_hash, artifact_id,
                artifact_content_hash, created_at
            ) VALUES (?, 'strategy.candidate-pool.v1', ?, ?, 'approval', 1,
                      NULL, ?, 'add_candidate', ?, ?, ?, 'draft', 'unvalidated',
                      ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                pool_id,
                task.id,
                old_absent_hash,
                operation_hash,
                reason,
                _canonical_json(default_action),
                _canonical_json(snapshot),
                snapshot_hash,
                pool_artifact["id"],
                pool_artifact["content_hash"],
                timestamp,
            ),
        )
        conn.execute(
            """
            INSERT INTO strategy_candidate_pool_items(
                revision_id, pool_id, task_id, position, entry_id, rule_id,
                source_artifact_id, source_kind, source_content_hash, asset_id,
                asset_hash, candidate_kind, fragment_id, effect_id, effect_stage,
                source_validation_status, parent_candidate_id,
                parent_evidence_hash, dataset_id, dataset_content_hash,
                workspace_revision, workspace_generation, semantic_mapping_hash,
                condition_json, requirements_json, action_json, enabled
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                revision_id,
                pool_id,
                task.id,
                entry["entry_id"],
                entry["rule_id"],
                source["id"],
                SOURCE_ARTIFACT_KIND,
                source["content_hash"],
                entry["source"]["asset_id"],
                entry["source"]["asset_hash"],
                entry["source"]["candidate_kind"],
                entry["source"]["fragment_id"],
                entry["source"]["effect_id"],
                entry["source"]["effect_stage"],
                entry["source"]["validation_status"],
                entry["source"]["parent_candidate_id"],
                entry["source"]["parent_evidence_hash"],
                entry["source"]["evidence_identity"]["dataset_id"],
                entry["source"]["evidence_identity"]["dataset_content_hash"],
                entry["source"]["evidence_identity"]["workspace_revision"],
                entry["source"]["evidence_identity"]["workspace_generation"],
                entry["source"]["evidence_identity"]["semantic_mapping_hash"],
                _canonical_json(entry["execution"]["condition"]),
                _canonical_json(entry["execution"]["requirements"]),
                _canonical_json(action),
            ),
        )
        conn.execute(
            """
            UPDATE strategy_candidate_pools
               SET current_revision = 1,
                   current_revision_id = ?,
                   current_snapshot_hash = ?,
                   updated_at = ?
             WHERE id = ?
            """,
            (revision_id, snapshot_hash, timestamp, pool_id),
        )
    return {
        "db_path": db_path,
        "tasks": tasks,
        "task": task,
        "artifacts": artifacts,
        "source": source,
        "source_path": source_path,
        "source_bytes": source_bytes,
        "pool_path": pool_path,
        "pool_bytes": pool_bytes,
        "pool_artifact": pool_artifact,
        "pool_id": pool_id,
        "revision_id": revision_id,
        "snapshot_hash": snapshot_hash,
    }


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


def _fragment_parts(
    source: dict,
    *,
    suffix: str,
    origin_tool: str = "strategy.test_candidate_asset",
) -> tuple[dict, str, dict]:
    fragment = build_verified_candidate_fragment(
        artifact={
            "artifact_id": source["id"],
            "artifact_kind": SOURCE_ARTIFACT_KIND,
            "artifact_schema_version": "strategy.candidate-asset-artifact.v1",
            "artifact_content_hash": source["content_hash"],
            "origin_tool": origin_tool,
        },
        asset={
            "schema_version": "strategy.candidate-asset.v1",
            "asset_id": "candidate-asset-a",
            "asset_hash": HASH_A,
            "asset_type": "univariate_refinement",
        },
        fragment_id=f"candidate-fragment-{suffix}",
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-{suffix}",
        condition={
            "op": "compare",
            "field": "score",
            "operator": "==",
            "value": suffix,
            "missing": "no_match",
        },
        requirements=[],
        effect_id=f"candidate-effect-{suffix}",
        evidence_id="candidate-parent-a",
        evidence_hash=HASH_B,
        evidence_identity={
            "dataset_id": "dataset-a",
            "dataset_content_hash": HASH_C,
            "workspace_revision": 3,
            "workspace_generation": 2,
            "semantic_mapping_hash": HASH_A,
            "sample_context_hash": HASH_B,
        },
    )
    return verified_fragment_pool_parts(fragment)


def _entry(
    source: dict,
    *,
    suffix: str,
    position: int,
    origin_tool: str = "strategy.test_candidate_asset",
) -> dict:
    generic_source, rule_id, execution = _fragment_parts(
        source, suffix=suffix, origin_tool=origin_tool
    )
    return {
        "entry_id": f"pool-entry-{suffix}",
        "rule_id": rule_id,
        "position": position,
        "source": generic_source,
        "execution": execution,
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


def test_schema_v17_and_initial_snapshot_are_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    db_path, _tasks, task, _other, artifacts, source = _seed(tmp_path)
    assert SCHEMA_VERSION == 18
    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 18

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

    tampered = deepcopy(snapshot)
    tampered["entries"][0]["source"]["fragment_hash"] = HASH_C
    tampered_body = {
        key: value for key, value in tampered.items() if key != "snapshot_hash"
    }
    with pytest.raises(StrategyCandidatePoolDataError, match="fragment_hash"):
        strategy_pool_snapshot_hash(tampered_body)


def test_migration_016_archives_real_v15_draft_and_starts_v2_cas_chain(
    tmp_path: Path,
) -> None:
    legacy = _seed_legacy_v15_pool(tmp_path)
    db_path = legacy["db_path"]
    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 15
        old_artifact = dict(
            conn.execute(
                "SELECT * FROM task_artifacts WHERE id = ?",
                (legacy["pool_artifact"]["id"],),
            ).fetchone()
        )

    init_db(db_path)
    init_db(db_path)
    repo = StrategyCandidatePoolRepository(db_path)
    archive = repo.get_archived_legacy_draft(legacy["task"].id, "approval")

    assert archive is not None
    assert archive["status"] == "archived_legacy_draft"
    assert archive["requires_rebuild"] is True
    assert archive["current_revision"] == 1
    assert archive["current_revision_id"] == legacy["revision_id"]
    assert archive["current_snapshot_hash"] == legacy["snapshot_hash"]
    assert archive["revision_count"] == archive["item_count"] == 1
    assert archive["revision_artifacts"] == [
        {
            "revision": 1,
            "revision_id": legacy["revision_id"],
            "snapshot_hash": legacy["snapshot_hash"],
            "artifact_id": legacy["pool_artifact"]["id"],
            "artifact_content_hash": legacy["pool_artifact"]["content_hash"],
        }
    ]
    assert repo.get_current(legacy["task"].id, "approval") is None
    assert legacy["source_path"].read_bytes() == legacy["source_bytes"]
    assert legacy["pool_path"].read_bytes() == legacy["pool_bytes"]

    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pools_v1_archive"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) "
                "FROM strategy_candidate_pool_revisions_v1_archive"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_items_v1_archive"
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT COUNT(*) FROM strategy_candidate_pools").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
            ).fetchone()[0]
            == 0
        )
        assert dict(
            conn.execute(
                "SELECT * FROM task_artifacts WHERE id = ?",
                (legacy["pool_artifact"]["id"],),
            ).fetchone()
        ) == old_artifact
        with pytest.raises(sqlite3.IntegrityError, match="archive is immutable"):
            conn.execute(
                "UPDATE strategy_candidate_pool_items_v1_archive "
                "SET enabled = enabled"
            )

    first = _snapshot(
        task_id=legacy["task"].id,
        revision=1,
        parent_revision_id=None,
        entries=[
            _entry(
                legacy["source"],
                suffix="v2-a",
                position=0,
                origin_tool="strategy.refine_univariate_candidate",
            )
        ],
        kind="add",
        reason="explicit v2 rebuild",
    )
    first_artifact = _register_pool_artifact(
        legacy["artifacts"], tmp_path, first
    )
    repo.apply_snapshot(
        snapshot=first,
        expected_revision=0,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=first_artifact["id"],
        artifact_content_hash=first_artifact["content_hash"],
        audit=_audit(first),
    )
    second = _snapshot(
        task_id=legacy["task"].id,
        revision=2,
        parent_revision_id=first["revision_id"],
        entries=deepcopy(first["entries"]),
        kind="reorder",
        reason="continue v2 chain",
    )
    second_artifact = _register_pool_artifact(
        legacy["artifacts"], tmp_path, second
    )
    repo.apply_snapshot(
        snapshot=second,
        expected_revision=1,
        expected_snapshot_hash=first["snapshot_hash"],
        artifact_id=second_artifact["id"],
        artifact_content_hash=second_artifact["content_hash"],
        audit=_audit(second),
    )
    assert repo.get_current(legacy["task"].id, "approval") == second
    assert repo.get_archived_legacy_draft(legacy["task"].id, "approval") == archive

    legacy["tasks"].delete_task(legacy["task"].id)
    with connect(db_path) as conn:
        for table in (
            "strategy_candidate_pools",
            "strategy_candidate_pool_revisions",
            "strategy_candidate_pool_items",
            "strategy_candidate_pools_v1_archive",
            "strategy_candidate_pool_revisions_v1_archive",
            "strategy_candidate_pool_items_v1_archive",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_artifacts WHERE task_id = ?",
                (legacy["task"].id,),
            ).fetchone()[0]
            == 0
        )


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
    with repo.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert repo.get_current_on_connection(conn, task.id, "approval") == second
        assert (
            repo.get_revision_by_id_on_connection(
                conn,
                task.id,
                "approval",
                first["revision_id"],
            )
            == first
        )

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
    forged_source, forged_rule_id, forged_execution = _fragment_parts(
        source,
        suffix="a",
        origin_tool="strategy.other_candidate",
    )
    forged["entries"][0]["source"] = forged_source
    forged["entries"][0]["rule_id"] = forged_rule_id
    forged["entries"][0]["execution"] = forged_execution
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
    with pytest.raises(StrategyCandidatePoolDataError, match="origin_tool"):
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
