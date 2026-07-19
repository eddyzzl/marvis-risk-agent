from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import sqlite3
import threading

import pytest

from marvis import db_schema as db_schema_module
from marvis.db_schema import connect, init_db
from marvis.packs.strategy.automatic_tree_apply import (
    _writer_contract as _kernel_writer_contract,
)
from marvis.repositories.automatic_tree_apply import (
    AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
    AUTOMATIC_TREE_APPLY_RESULT_SCHEMA_VERSION,
    AutomaticTreeApplyCommittedFacts,
    AutomaticTreeApplyConflictError,
    AutomaticTreeApplyDataError,
    AutomaticTreeApplyIdentity,
    AutomaticTreeApplyNotFoundError,
    AutomaticTreeApplyRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository


_CREATED_AT = "2026-07-19T05:00:00+00:00"
_ASSET_ID = "candidate-asset-" + "a" * 32


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


def _seed_task(db_path, task_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'strategy task', 'v1', 'tester', '/tmp/source',
                      'created', 'created', ?, ?)
            """,
            (task_id, _CREATED_AT, _CREATED_AT),
        )


def _seed_dataset(
    db_path,
    dataset_id: str,
    *,
    task_id: str,
    content_hash: str,
    path: str,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count,
                columns_json, has_target, target_col, created_at, content_hash
            ) VALUES (?, ?, 'derived', ?, 'parquet', 3, '[]', 0, NULL, ?, ?)
            """,
            (dataset_id, task_id, path, _CREATED_AT, content_hash),
        )


def _writer(*, engine_version: str = "19.0.1") -> dict[str, object]:
    writer = _kernel_writer_contract()
    writer["engine_version"] = engine_version
    return writer


def _result_payload(
    identity: AutomaticTreeApplyIdentity,
    committed: AutomaticTreeApplyCommittedFacts,
) -> dict:
    body = {
        "schema_version": AUTOMATIC_TREE_APPLY_RESULT_SCHEMA_VERSION,
        "producer_version": AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
        "source": {
            "content_hash": identity.source_dataset_hash,
            "row_count": 3,
        },
        "tree": {
            "result_hash": identity.tree_result_hash,
            "asset_id": identity.asset_id,
            "asset_hash": identity.asset_hash,
        },
        "output": {
            "content_hash": committed.result_dataset_hash,
            "row_count": 3,
            "schema": {
                "fields": [
                    {
                        "name": "feature_a",
                        "physical_type": "double",
                        "nullable": True,
                        "metadata_hash": None,
                    },
                    {
                        "name": identity.output_leaf_column,
                        "physical_type": "string",
                        "nullable": False,
                        "metadata_hash": None,
                    },
                    {
                        "name": identity.output_rule_column,
                        "physical_type": "string",
                        "nullable": False,
                        "metadata_hash": None,
                    },
                ],
                "metadata_hash": None,
            },
            "columns": {
                "leaf_id": identity.output_leaf_column,
                "rule_id": identity.output_rule_column,
            },
            "leaf_distribution": [
                {"leaf_id": "leaf-a", "rule_id": "rule-a", "row_count": 1},
                {"leaf_id": "leaf-b", "rule_id": "rule-b", "row_count": 2},
            ],
        },
        "writer": _writer(engine_version=identity.writer_version),
    }
    result_hash = _sha(_canonical_json(body))
    return {
        **body,
        "result_id": f"automatic-tree-apply-{result_hash[:32]}",
        "result_hash": result_hash,
    }


def _prepare(db_path):
    init_db(db_path)
    _seed_task(db_path, "task-1")
    _seed_task(db_path, "task-2")
    source_hash = _sha("source parquet bytes")
    result_hash = _sha("result parquet bytes")
    foreign_hash = _sha("foreign parquet bytes")
    _seed_dataset(
        db_path,
        "source",
        task_id="task-1",
        content_hash=source_hash,
        path="task-1/datasets/source.parquet",
    )
    _seed_dataset(
        db_path,
        "result",
        task_id="task-1",
        content_hash=result_hash,
        path="task-1/datasets/result.parquet",
    )
    _seed_dataset(
        db_path,
        "foreign-source",
        task_id="task-2",
        content_hash=foreign_hash,
        path="task-2/datasets/source.parquet",
    )
    asset_hash = _sha("automatic tree asset identity")
    tree_result_hash = _sha("weighted tree result")
    artifact_repo = TaskArtifactRepository(db_path)
    source_artifact = artifact_repo.register(
        task_id="task-1",
        kind="strategy_automatic_tree_asset_json",
        path="task-1/strategy_automatic_trees/tree.json",
        content_hash=_sha("canonical tree artifact bytes"),
        origin_tool="strategy.build_automatic_tree_candidate",
        provenance={
            "schema_version": "strategy.automatic-tree-asset-artifact.v1",
            "task_id": "task-1",
            "asset_id": _ASSET_ID,
            "asset_hash": asset_hash,
            "tree_result_hash": tree_result_hash,
        },
        created_at=_CREATED_AT,
    )
    foreign_source_artifact = artifact_repo.register(
        task_id="task-2",
        kind="strategy_automatic_tree_asset_json",
        path="task-2/strategy_automatic_trees/tree.json",
        content_hash=_sha("foreign canonical tree artifact bytes"),
        origin_tool="strategy.build_automatic_tree_candidate",
        provenance={
            "schema_version": "strategy.automatic-tree-asset-artifact.v1",
            "task_id": "task-2",
            "asset_id": _ASSET_ID,
            "asset_hash": asset_hash,
            "tree_result_hash": tree_result_hash,
        },
        created_at=_CREATED_AT,
    )
    kernel_writer = _kernel_writer_contract()
    identity = AutomaticTreeApplyIdentity(
        task_id="task-1",
        source_tree_artifact_id=source_artifact["id"],
        source_tree_artifact_hash=source_artifact["content_hash"],
        asset_id=_ASSET_ID,
        asset_hash=asset_hash,
        tree_result_hash=tree_result_hash,
        source_dataset_id="source",
        source_dataset_hash=source_hash,
        output_leaf_column="automatic_tree_leaf_id",
        output_rule_column="automatic_tree_rule_id",
        writer_contract="strategy.automatic-tree-apply-parquet-writer/1",
        writer_version=str(kernel_writer["engine_version"]),
    )
    evidence_artifact = artifact_repo.register(
        task_id="task-1",
        kind="strategy_automatic_tree_apply_evidence",
        path="task-1/strategy_automatic_tree_applies/evidence.json",
        content_hash=_sha("apply evidence artifact bytes"),
        origin_tool="strategy.apply_automatic_tree",
        provenance={"run_id": identity.run_id, "input_hash": identity.input_hash},
        created_at=_CREATED_AT,
    )
    foreign_evidence_artifact = artifact_repo.register(
        task_id="task-2",
        kind="strategy_automatic_tree_apply_evidence",
        path="task-2/strategy_automatic_tree_applies/evidence.json",
        content_hash=_sha("foreign apply evidence artifact bytes"),
        origin_tool="strategy.apply_automatic_tree",
        provenance={"run_id": identity.run_id, "input_hash": identity.input_hash},
        created_at=_CREATED_AT,
    )
    committed = AutomaticTreeApplyCommittedFacts(
        result_dataset_id="result",
        result_dataset_hash=result_hash,
        result_dataset_path="task-1/datasets/result.parquet",
        evidence_artifact_id=evidence_artifact["id"],
        evidence_artifact_hash=evidence_artifact["content_hash"],
        evidence_artifact_path=evidence_artifact["path"],
    )
    return {
        "repo": AutomaticTreeApplyRepository(db_path),
        "identity": identity,
        "committed": committed,
        "payload": _result_payload(identity, committed),
        "foreign_hash": foreign_hash,
        "foreign_source_artifact": foreign_source_artifact,
        "foreign_evidence_artifact": foreign_evidence_artifact,
    }


def test_identity_is_stable_strict_and_contains_every_byte_affecting_binding():
    identity = AutomaticTreeApplyIdentity(
        task_id="task-1",
        source_tree_artifact_id="artifact-1",
        source_tree_artifact_hash=_sha("artifact"),
        asset_id=_ASSET_ID,
        asset_hash=_sha("asset"),
        tree_result_hash=_sha("tree"),
        source_dataset_id="source",
        source_dataset_hash=_sha("source"),
        output_leaf_column="leaf_id",
        output_rule_column="rule_id",
        writer_contract="writer/1",
        writer_version="19.0.1",
    )
    assert (
        AutomaticTreeApplyIdentity(
            **{
                key: value
                for key, value in identity.__dict__.items()
                if key not in {"input_json", "input_hash", "run_id"}
            }
        )
        == identity
    )
    input_payload = json.loads(identity.input_json)
    assert "activate_result" not in input_payload
    assert "workspace_revision" not in input_payload

    variants = (
        {"task_id": "task-2"},
        {"source_tree_artifact_id": "artifact-2"},
        {"source_tree_artifact_hash": _sha("artifact-2")},
        {"asset_id": "candidate-asset-" + "b" * 32},
        {"asset_hash": _sha("asset-2")},
        {"tree_result_hash": _sha("tree-2")},
        {"source_dataset_id": "source-2"},
        {"source_dataset_hash": _sha("source-2")},
        {"output_leaf_column": "other_leaf"},
        {"output_rule_column": "other_rule"},
        {"writer_contract": "writer/2"},
        {"writer_version": "20.0.0"},
    )
    for override in variants:
        changed = replace(identity, **override)
        assert changed.input_hash != identity.input_hash
        assert changed.run_id != identity.run_id

    with pytest.raises(AutomaticTreeApplyDataError, match="distinct"):
        replace(identity, output_rule_column="LEAF_ID")
    with pytest.raises(AutomaticTreeApplyDataError, match="lowercase"):
        replace(identity, asset_hash=_sha("asset").upper())


def test_create_replays_exact_path_free_result_and_masks_foreign_task(tmp_path):
    prepared = _prepare(tmp_path / "app.sqlite")
    repo = prepared["repo"]
    identity = prepared["identity"]
    committed = prepared["committed"]
    payload = prepared["payload"]

    first = repo.create(
        identity,
        committed,
        result_payload=payload,
        created_at=_CREATED_AT,
    )
    replay = repo.record_succeeded(
        identity,
        committed,
        result_payload=deepcopy(payload),
        created_at="2026-07-19T06:00:00+00:00",
    )

    assert replay == first
    assert first.run_id == identity.run_id
    assert first.result_json == _canonical_json(payload)
    assert first.result_json_hash == _sha(first.result_json)
    assert first.result_payload == payload
    detached = first.result_payload
    detached["output"]["columns"]["leaf_id"] = "tampered"
    assert first.result_payload == payload
    assert repo.get_by_id("task-1", first.id) == first
    assert repo.get_by_input("task-1", identity.input_hash) == first
    assert repo.get_by_input(identity) == first
    with repo.transaction() as conn:
        assert repo.get_by_id_on_connection(conn, "task-1", first.id) == first
        assert repo.get_by_input_on_connection(conn, identity) == first
    assert repo.get_by_id("task-2", first.id) is None
    assert repo.get_by_input("task-2", identity.input_hash) is None

    with pytest.raises(AutomaticTreeApplyConflictError, match="committed facts"):
        repo.create(
            identity,
            replace(committed, result_dataset_path="task-1/datasets/other.parquet"),
            result_payload=payload,
        )

    with connect(tmp_path / "app.sqlite") as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE strategy_automatic_tree_apply_runs "
                "SET writer_version = 'tampered' WHERE id = ?",
                (first.id,),
            )


def test_connection_scoped_insert_rolls_back_with_outer_transaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    prepared = _prepare(db_path)
    repo = prepared["repo"]

    with pytest.raises(RuntimeError, match="rollback caller unit of work"):
        with repo.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            repo.record_succeeded_on_connection(
                conn,
                prepared["identity"],
                prepared["committed"],
                result_payload=prepared["payload"],
                created_at=_CREATED_AT,
            )
            raise RuntimeError("rollback caller unit of work")

    assert repo.get_by_id("task-1", prepared["identity"].run_id) is None


def test_create_requires_every_dataset_and_artifact_to_be_current_task_owned(
    tmp_path,
):
    prepared = _prepare(tmp_path / "app.sqlite")
    repo = prepared["repo"]
    identity = prepared["identity"]
    committed = prepared["committed"]

    foreign_source = prepared["foreign_source_artifact"]
    foreign_source_identity = replace(
        identity,
        source_tree_artifact_id=foreign_source["id"],
        source_tree_artifact_hash=foreign_source["content_hash"],
    )
    with pytest.raises(AutomaticTreeApplyNotFoundError, match="not found for task"):
        repo.create(
            foreign_source_identity,
            committed,
            result_payload=_result_payload(foreign_source_identity, committed),
        )

    foreign_evidence = prepared["foreign_evidence_artifact"]
    foreign_evidence_facts = replace(
        committed,
        evidence_artifact_id=foreign_evidence["id"],
        evidence_artifact_hash=foreign_evidence["content_hash"],
        evidence_artifact_path=foreign_evidence["path"],
    )
    with pytest.raises(AutomaticTreeApplyNotFoundError, match="not found for task"):
        repo.create(
            identity,
            foreign_evidence_facts,
            result_payload=_result_payload(identity, foreign_evidence_facts),
        )

    foreign_dataset_identity = replace(
        identity,
        source_dataset_id="foreign-source",
        source_dataset_hash=prepared["foreign_hash"],
    )
    with pytest.raises(AutomaticTreeApplyNotFoundError, match="not found for task"):
        repo.create(
            foreign_dataset_identity,
            committed,
            result_payload=_result_payload(foreign_dataset_identity, committed),
        )

    artifact_repo = TaskArtifactRepository(tmp_path / "app.sqlite")
    invalid_evidence_specs = (
        (
            "wrong-kind",
            "unrelated_artifact",
            "strategy.apply_automatic_tree",
            {"run_id": identity.run_id, "input_hash": identity.input_hash},
        ),
        (
            "wrong-origin",
            "strategy_automatic_tree_apply_evidence",
            "untrusted.tool",
            {"run_id": identity.run_id, "input_hash": identity.input_hash},
        ),
        (
            "wrong-binding",
            "strategy_automatic_tree_apply_evidence",
            "strategy.apply_automatic_tree",
            {"run_id": "atar_wrong", "input_hash": _sha("wrong input")},
        ),
    )
    for suffix, kind, origin, provenance in invalid_evidence_specs:
        artifact = artifact_repo.register(
            task_id="task-1",
            kind=kind,
            path=f"task-1/strategy_automatic_tree_applies/{suffix}.json",
            content_hash=_sha(f"{suffix} evidence bytes"),
            origin_tool=origin,
            provenance=provenance,
            created_at=_CREATED_AT,
        )
        invalid_facts = replace(
            committed,
            evidence_artifact_id=artifact["id"],
            evidence_artifact_hash=artifact["content_hash"],
            evidence_artifact_path=artifact["path"],
        )
        with pytest.raises(AutomaticTreeApplyConflictError, match="evidence artifact"):
            repo.create(
                identity,
                invalid_facts,
                result_payload=_result_payload(identity, invalid_facts),
            )

    with connect(tmp_path / "app.sqlite") as conn:
        conn.execute("UPDATE datasets SET row_count = 4 WHERE id = 'source'")
    with pytest.raises(AutomaticTreeApplyConflictError, match="row count"):
        repo.create(
            identity,
            committed,
            result_payload=_result_payload(identity, committed),
        )


def test_result_payload_is_exact_canonical_and_cross_bound(tmp_path):
    prepared = _prepare(tmp_path / "app.sqlite")
    repo = prepared["repo"]
    identity = prepared["identity"]
    committed = prepared["committed"]
    payload = prepared["payload"]
    mutations = (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["source"].update(content_hash=_sha("wrong source")),
        lambda value: value["tree"].update(result_hash=_sha("wrong tree")),
        lambda value: value["tree"].update(asset_id="candidate-asset-" + "b" * 32),
        lambda value: value["output"].update(content_hash=_sha("wrong result")),
        lambda value: value["output"]["columns"].update(leaf_id="wrong_leaf"),
        lambda value: value["writer"].update(engine_version="20.0.0"),
        lambda value: value.update(result_hash=_sha("wrong evidence")),
        lambda value: value["output"]["schema"]["fields"][0].update(
            physical_type=float("nan")
        ),
    )
    for mutate in mutations:
        tampered = deepcopy(payload)
        mutate(tampered)
        with pytest.raises(AutomaticTreeApplyDataError):
            repo.create(
                identity,
                committed,
                result_payload=tampered,
            )
    with connect(tmp_path / "app.sqlite") as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM strategy_automatic_tree_apply_runs"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize("corruption", ["noncanonical_json", "uppercase_hash", "data"])
def test_reads_fail_closed_on_raw_row_corruption(tmp_path, corruption):
    db_path = tmp_path / "app.sqlite"
    prepared = _prepare(db_path)
    record = prepared["repo"].create(
        prepared["identity"],
        prepared["committed"],
        result_payload=prepared["payload"],
        created_at=_CREATED_AT,
    )
    with connect(db_path) as conn:
        conn.execute(
            "DROP TRIGGER trg_strategy_automatic_tree_apply_runs_immutable_update"
        )
        if corruption == "noncanonical_json":
            raw = json.dumps(prepared["payload"], ensure_ascii=False, indent=2)
            conn.execute(
                "UPDATE strategy_automatic_tree_apply_runs "
                "SET result_json = ?, result_hash = ? WHERE id = ?",
                (raw, _sha(raw), record.id),
            )
        elif corruption == "uppercase_hash":
            conn.execute("PRAGMA ignore_check_constraints = ON")
            conn.execute(
                "UPDATE strategy_automatic_tree_apply_runs "
                "SET asset_hash = upper(asset_hash) WHERE id = ?",
                (record.id,),
            )
        else:
            conn.execute(
                "UPDATE strategy_automatic_tree_apply_runs "
                "SET writer_version = '20.0.0' WHERE id = ?",
                (record.id,),
            )

    with pytest.raises(AutomaticTreeApplyDataError, match="corrupt"):
        prepared["repo"].get_by_id("task-1", record.id)


def test_concurrent_same_input_has_one_winner_and_one_exact_replay(tmp_path):
    db_path = tmp_path / "app.sqlite"
    prepared = _prepare(db_path)
    barrier = threading.Barrier(2)
    records = []
    errors = []

    def invoke() -> None:
        try:
            barrier.wait(timeout=10)
            records.append(
                AutomaticTreeApplyRepository(db_path).create(
                    prepared["identity"],
                    prepared["committed"],
                    result_payload=deepcopy(prepared["payload"]),
                    created_at=_CREATED_AT,
                )
            )
        except BaseException as exc:  # captured for assertion in the main thread
            errors.append(exc)

    writers = [threading.Thread(target=invoke) for _ in range(2)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=15)

    assert all(not writer.is_alive() for writer in writers)
    assert errors == []
    assert len(records) == 2
    assert records[0] == records[1]
    with connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM strategy_automatic_tree_apply_runs"
        ).fetchone()[0]
    assert count == 1


def test_migration_017_is_additive_idempotent_and_guards_column_collision(tmp_path):
    db_path = tmp_path / "legacy_v16.sqlite"
    init_db(db_path)
    _seed_task(db_path, "preserved-task")
    with connect(db_path) as conn:
        conn.execute("DROP TABLE strategy_automatic_tree_apply_runs")
        conn.execute("PRAGMA user_version = 16")

    init_db(db_path)
    init_db(db_path)
    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 18
        assert conn.execute(
            "SELECT id FROM tasks WHERE id = 'preserved-task'"
        ).fetchone()
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(strategy_automatic_tree_apply_runs)"
            )
        }
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'strategy_automatic_tree_apply_runs'"
            )
        }
    assert {
        "id",
        "input_hash",
        "source_tree_artifact_id",
        "source_dataset_id",
        "output_leaf_column",
        "output_rule_column",
        "result_dataset_id",
        "evidence_artifact_id",
        "result_json",
    } <= columns
    assert triggers == {"trg_strategy_automatic_tree_apply_runs_immutable_update"}

    prepared = _prepare(tmp_path / "constraint.sqlite")
    record = prepared["repo"].create(
        prepared["identity"],
        prepared["committed"],
        result_payload=prepared["payload"],
        created_at=_CREATED_AT,
    )
    with connect(tmp_path / "constraint.sqlite") as conn:
        with pytest.raises(sqlite3.IntegrityError, match="output_leaf_column"):
            conn.execute(
                """
                INSERT INTO strategy_automatic_tree_apply_runs
                SELECT 'raw-case-collision', schema_version, task_id, ?,
                       source_tree_artifact_id, source_tree_artifact_hash,
                       asset_id, asset_hash, tree_result_hash,
                       source_dataset_id, source_dataset_hash,
                       'Leaf_Id', 'leaf_id', writer_contract, writer_version,
                       result_dataset_id, result_dataset_hash, result_dataset_path,
                       evidence_artifact_id, evidence_artifact_hash,
                       evidence_artifact_path, result_json, result_hash, created_at
                  FROM strategy_automatic_tree_apply_runs
                 WHERE id = ?
                """,
                (_sha("case collision input"), record.id),
            )


def test_schema_version_registry_includes_migration_017():
    assert db_schema_module.SCHEMA_VERSION == 18
    assert (
        17,
        db_schema_module._migration_017_automatic_tree_apply_runs,
    ) in db_schema_module._MIGRATIONS


def test_schema_version_registry_includes_migration_018():
    assert db_schema_module._MIGRATIONS[-1] == (
        18,
        db_schema_module._migration_018_strategy_dsl_content_hash,
    )
