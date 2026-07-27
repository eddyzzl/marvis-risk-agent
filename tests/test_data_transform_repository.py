from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db_schema import connect, init_db
from marvis.repositories.data_transform import (
    DataTransformConflictError,
    DataTransformDataError,
    DataTransformIdentity,
    DataTransformRepository,
    data_transform_artifact_provenance,
)
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


_CREATED_AT = "2026-07-19T03:00:00+00:00"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_task(db_path, task_id: str = "task-1") -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (?, 'data task', 'v1', 'tester', '/tmp/source',
                      'created', 'created', ?, ?)
            """,
            (task_id, _CREATED_AT, _CREATED_AT),
        )


def _seed_dataset(
    db_path,
    dataset_id: str,
    *,
    task_id: str = "task-1",
    content_hash: str | None = None,
) -> str:
    digest = content_hash or _sha(dataset_id)
    columns = [
        {
            "name": "customer_id",
            "dtype": "VARCHAR",
            "semantic_role": "id",
            "fingerprint": {
                "value_kind": "text",
                "length_mode": None,
                "regex_pattern": None,
                "is_hashed": False,
                "hash_type": None,
                "hex_case": None,
                "date_format": None,
            },
            "null_rate": 0.0,
            "cardinality": 2,
            "sample_values": [],
        }
    ]
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count,
                columns_json, has_target, target_col, created_at, content_hash
            ) VALUES (?, ?, 'derived', ?, 'parquet', 2, ?, 0, NULL, ?, ?)
            """,
            (
                dataset_id,
                task_id,
                f"{task_id}/transforms/{dataset_id}.parquet",
                json.dumps(columns, separators=(",", ":")),
                _CREATED_AT,
                digest,
            ),
        )
    return digest


def _identity(
    source_hash: str,
    *,
    workspace_revision: int,
    analysis_generation: int,
) -> DataTransformIdentity:
    return DataTransformIdentity(
        task_id="task-1",
        source_dataset_id="source",
        source_content_hash=source_hash,
        workspace_revision=workspace_revision,
        analysis_generation=analysis_generation,
        semantic_mapping_hash=_sha("semantics"),
        operations=(
            {"op": "rename_columns", "mapping": {"customer_id": "account_id"}},
        ),
        producer_version="marvis.data-transform/1",
    )


def _result_payload(identity: DataTransformIdentity, result_hash: str) -> dict:
    operations = json.loads(identity.operations_json)
    return {
        "schema_version": "data-transform-evidence.v1",
        "run_id": identity.run_id,
        "producer_version": identity.producer_version,
        "input_hash": identity.input_hash,
        "operations_hash": identity.operations_hash,
        "source": {
            "dataset_id": identity.source_dataset_id,
            "content_hash": identity.source_content_hash,
            "row_count": 2,
        },
        "result": {
            "dataset_id": "result",
            "content_hash": result_hash,
            "row_count": 2,
        },
        "transform": {
            "schema_version": "transform-result.v1",
            "execution": {
                "mode": "duckdb-single-thread-v1",
                "duckdb_threads": 1,
                "preserve_insertion_order": True,
            },
            "config": {},
            "operations": operations,
            "steps": [
                {
                    "step": 1,
                    "op": "rename_columns",
                    "row_count_before": 2,
                    "row_count_after": 2,
                    "row_delta": 0,
                    "columns_before": [],
                    "columns_after": [],
                    "impact": {},
                }
            ],
            "summary": {
                "row_count_before": 2,
                "row_count_after": 2,
                "row_delta": 0,
                "column_count_before": 1,
                "column_count_after": 1,
                "operation_count": 1,
            },
            "source": {"format": "parquet", "size_bytes": 1, "columns": []},
            "output": {
                "format": "parquet",
                "size_bytes": 1,
                "content_hash": result_hash,
                "hash_algorithm": "sha256",
                "row_count": 2,
                "column_count": 1,
                "columns": [],
            },
        },
        "semantic_migration": {
            "before_hash": identity.semantic_mapping_hash,
            "after_hash": _sha("result semantics"),
            "renamed_fields": {"customer_id": "account_id"},
            "dropped_fields": [],
            "dropped_protected_fields": [],
        },
        "workspace": {
            "source_revision": identity.workspace_revision,
            "result_revision": identity.workspace_revision + 1,
            "source_analysis_generation": identity.analysis_generation,
            "result_analysis_generation": identity.analysis_generation + 1,
        },
        "lineage": {
            "parent_dataset_id": identity.source_dataset_id,
            "child_dataset_id": "result",
            "relation_kind": "transform",
            "edge_order": 0,
        },
    }


def _prepare_succeeded_inputs(db_path):
    _seed_task(db_path)
    source_hash = _seed_dataset(db_path, "source")
    result_hash = _seed_dataset(db_path, "result")
    workspace_repo = DataWorkspaceRepository(db_path)
    source = workspace_repo.save(
        "task-1",
        DataWorkspaceDraft(
            active_dataset_id="source",
            active_dataset_content_hash=source_hash,
        ),
        expected_revision=0,
    )
    identity = _identity(
        source_hash,
        workspace_revision=source.revision,
        analysis_generation=source.analysis_generation,
    )
    payload = _result_payload(identity, result_hash)
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    evidence_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        result_workspace = workspace_repo.activate_derived_on_connection(
            conn,
            "task-1",
            expected_revision=source.revision,
            source_dataset_id="source",
            source_dataset_content_hash=source_hash,
            result_dataset_id="result",
            result_dataset_content_hash=result_hash,
            semantic_mapping=DataSemanticMapping(),
        )
        artifact = TaskArtifactRepository(db_path).register_on_connection(
            conn,
            task_id="task-1",
            kind="data_transform_evidence",
            path="task-1/transforms/evidence.json",
            content_hash=evidence_hash,
            origin_tool="data_ops.transform_dataset",
            provenance=data_transform_artifact_provenance(
                identity,
                result_dataset_id="result",
                result_content_hash=result_hash,
            ),
            created_at=_CREATED_AT,
        )
    return identity, result_hash, payload, evidence_hash, result_workspace, artifact


def test_transform_identity_is_canonical_and_page_revision_is_not_computational():
    source_hash = _sha("source")
    first = _identity(source_hash, workspace_revision=2, analysis_generation=1)
    page_only = _identity(source_hash, workspace_revision=7, analysis_generation=1)

    assert first.operations_hash == page_only.operations_hash
    assert first.input_hash == page_only.input_hash
    assert first.run_id == page_only.run_id
    assert first.operations_json == (
        '[{"mapping":{"customer_id":"account_id"},"op":"rename_columns"}]'
    )

    with pytest.raises(DataTransformDataError, match="canonical JSON"):
        DataTransformIdentity(
            task_id="task-1",
            source_dataset_id="source",
            source_content_hash=source_hash,
            workspace_revision=1,
            analysis_generation=1,
            semantic_mapping_hash=_sha("semantics"),
            operations=({"op": "fill_missing", "fills": [{"value": float("nan")}]},),
            producer_version="v1",
        )


def test_record_succeeded_persists_run_and_matching_lineage_atomically(tmp_path):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)

    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        record = repo.record_succeeded_on_connection(
            conn,
            identity,
            result_dataset_id="result",
            result_content_hash=result_hash,
            result_artifact_id=artifact["id"],
            result_payload=payload,
            result_workspace_revision=workspace.revision,
            result_analysis_generation=workspace.analysis_generation,
            created_at=_CREATED_AT,
        )

    assert record.id == identity.run_id
    assert record.input_hash == identity.input_hash
    assert record.result_hash == evidence_hash
    assert repo.get_for_task("task-1", record.id) == record
    assert repo.find_by_input_hash("task-1", identity.input_hash) == record
    lineage = repo.list_lineage("task-1")
    assert len(lineage) == 1
    assert lineage[0]["parent_dataset_id"] == "source"
    assert lineage[0]["child_dataset_id"] == "result"
    assert lineage[0]["transform_run_id"] == record.id

    with connect(db_path) as conn:
        replay = repo.record_succeeded_on_connection(
            conn,
            identity,
            result_dataset_id="result",
            result_content_hash=result_hash,
            result_artifact_id=artifact["id"],
            result_payload=payload,
            result_workspace_revision=workspace.revision,
            result_analysis_generation=workspace.analysis_generation,
            created_at=_CREATED_AT,
        )
    assert replay == record
    assert len(repo.list_lineage("task-1")) == 1


def test_record_succeeded_rejects_artifact_or_workspace_evidence_drift(tmp_path):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, _evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)

    with connect(db_path) as conn:
        wrong_artifact = TaskArtifactRepository(db_path).register_on_connection(
            conn,
            task_id="task-1",
            kind="data_transform_evidence",
            path="task-1/transforms/wrong-evidence.json",
            content_hash=_sha("wrong evidence"),
            origin_tool="untrusted.tool",
            provenance={"schema_version": "wrong.v1"},
            created_at=_CREATED_AT,
        )
    with connect(db_path) as conn:
        with pytest.raises(DataTransformDataError, match="artifact evidence"):
            repo.record_succeeded_on_connection(
                conn,
                identity,
                result_dataset_id="result",
                result_content_hash=result_hash,
                result_artifact_id=wrong_artifact["id"],
                result_payload=payload,
                result_workspace_revision=workspace.revision,
                result_analysis_generation=workspace.analysis_generation,
                created_at=_CREATED_AT,
            )

    with connect(db_path) as conn:
        conn.execute(
            "UPDATE data_workspaces SET analysis_generation = analysis_generation + 1 "
            "WHERE task_id = 'task-1'"
        )
    with connect(db_path) as conn:
        with pytest.raises(DataTransformConflictError, match="workspace"):
            repo.record_succeeded_on_connection(
                conn,
                identity,
                result_dataset_id="result",
                result_content_hash=result_hash,
                result_artifact_id=artifact["id"],
                result_payload=payload,
                result_workspace_revision=workspace.revision,
                result_analysis_generation=workspace.analysis_generation,
                created_at=_CREATED_AT,
            )


def test_record_succeeded_rejects_cross_field_evidence_mismatches(tmp_path):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, _evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)
    mismatches = (
        (("schema_version",), "wrong.v1", "schema_version"),
        (("run_id",), "dtr_wrong", "run_id"),
        (("producer_version",), "wrong-producer", "producer_version"),
        (("input_hash",), _sha("wrong input"), "input_hash"),
        (("operations_hash",), _sha("wrong operations"), "operations_hash"),
        (("source", "dataset_id"), "other-source", "source.dataset_id"),
        (("source", "content_hash"), _sha("other source"), "source.content_hash"),
        (("result", "dataset_id"), "other-result", "result.dataset_id"),
        (("result", "content_hash"), _sha("other result"), "result.content_hash"),
        (("transform", "schema_version"), "wrong.v1", "transform.schema_version"),
        (("transform", "execution"), {}, "transform.execution.mode"),
        (
            ("transform", "execution", "mode"),
            "duckdb-parallel-v1",
            "transform.execution.mode",
        ),
        (
            ("transform", "execution", "duckdb_threads"),
            True,
            "transform.execution.duckdb_threads",
        ),
        (
            ("transform", "execution", "duckdb_threads"),
            2,
            "transform.execution.duckdb_threads",
        ),
        (
            ("transform", "execution", "preserve_insertion_order"),
            False,
            "transform.execution.preserve_insertion_order",
        ),
        (("transform", "operations"), [], "transform.operations"),
        (("transform", "summary"), [], "transform.summary"),
        (
            ("transform", "summary", "row_count_after"),
            "2",
            "transform.summary.row_count_after",
        ),
        (("transform", "steps"), {}, "transform.steps"),
        (
            ("transform", "steps", 0, "row_count_before"),
            "2",
            "transform.steps\\[0\\].row_count_before",
        ),
        (
            ("semantic_migration", "before_hash"),
            _sha("wrong semantics"),
            "semantic_migration.before_hash",
        ),
        (
            ("workspace", "source_revision"),
            identity.workspace_revision + 1,
            "workspace.source_revision",
        ),
        (
            ("workspace", "result_revision"),
            workspace.revision + 1,
            "workspace.result_revision",
        ),
        (
            ("workspace", "source_analysis_generation"),
            identity.analysis_generation + 1,
            "workspace.source_analysis_generation",
        ),
        (
            ("workspace", "result_analysis_generation"),
            workspace.analysis_generation + 1,
            "workspace.result_analysis_generation",
        ),
        (("lineage", "parent_dataset_id"), "other-source", "lineage.parent_dataset_id"),
        (("lineage", "child_dataset_id"), "other-result", "lineage.child_dataset_id"),
        (("lineage", "relation_kind"), "copy", "lineage.relation_kind"),
        (("lineage", "edge_order"), 1, "lineage.edge_order"),
    )

    for path, value, expected_error in mismatches:
        tampered = deepcopy(payload)
        parent = tampered
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        with connect(db_path) as conn:
            with pytest.raises(DataTransformDataError, match=expected_error):
                repo.record_succeeded_on_connection(
                    conn,
                    identity,
                    result_dataset_id="result",
                    result_content_hash=result_hash,
                    result_artifact_id=artifact["id"],
                    result_payload=tampered,
                    result_workspace_revision=workspace.revision,
                    result_analysis_generation=workspace.analysis_generation,
                    created_at=_CREATED_AT,
                )


def test_get_transform_fails_closed_when_result_payload_is_rehashed_but_tampered(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, _evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)
    with connect(db_path) as conn:
        record = repo.record_succeeded_on_connection(
            conn,
            identity,
            result_dataset_id="result",
            result_content_hash=result_hash,
            result_artifact_id=artifact["id"],
            result_payload=payload,
            result_workspace_revision=workspace.revision,
            result_analysis_generation=workspace.analysis_generation,
            created_at=_CREATED_AT,
        )

    tampered = deepcopy(payload)
    tampered["lineage"]["child_dataset_id"] = "source"
    tampered_json = json.dumps(
        tampered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with connect(db_path) as conn:
        conn.execute("DROP TRIGGER trg_data_transform_runs_immutable")
        conn.execute(
            "UPDATE data_transform_runs SET result_json = ?, result_hash = ? WHERE id = ?",
            (tampered_json, _sha(tampered_json), record.id),
        )

    with pytest.raises(DataTransformDataError, match="lineage.child_dataset_id"):
        repo.get_for_task("task-1", record.id)


def test_get_transform_fails_closed_when_execution_evidence_is_rehashed_but_tampered(
    tmp_path,
):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, _evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)
    with connect(db_path) as conn:
        record = repo.record_succeeded_on_connection(
            conn,
            identity,
            result_dataset_id="result",
            result_content_hash=result_hash,
            result_artifact_id=artifact["id"],
            result_payload=payload,
            result_workspace_revision=workspace.revision,
            result_analysis_generation=workspace.analysis_generation,
            created_at=_CREATED_AT,
        )
        conn.execute("DROP TRIGGER trg_data_transform_runs_immutable")

    mutations = (
        (None, "transform.execution"),
        ({"mode": "duckdb-parallel-v1", "duckdb_threads": 2}, "execution.mode"),
        (
            {
                "mode": "duckdb-single-thread-v1",
                "duckdb_threads": True,
                "preserve_insertion_order": True,
            },
            "execution.duckdb_threads",
        ),
        (
            {
                "mode": "duckdb-single-thread-v1",
                "duckdb_threads": 1,
                "preserve_insertion_order": False,
            },
            "execution.preserve_insertion_order",
        ),
    )
    for execution, expected_error in mutations:
        tampered = deepcopy(payload)
        if execution is None:
            del tampered["transform"]["execution"]
        else:
            tampered["transform"]["execution"] = execution
        tampered_json = json.dumps(
            tampered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with connect(db_path) as conn:
            conn.execute(
                "UPDATE data_transform_runs SET result_json = ?, result_hash = ? "
                "WHERE id = ?",
                (tampered_json, _sha(tampered_json), record.id),
            )
        with pytest.raises(DataTransformDataError, match=expected_error):
            repo.get_for_task("task-1", record.id)


def test_transform_run_and_lineage_are_immutable_and_transactional(tmp_path):
    db_path = tmp_path / "app.sqlite"
    identity, result_hash, payload, _evidence_hash, workspace, artifact = (
        _prepare_succeeded_inputs(db_path)
    )
    repo = DataTransformRepository(db_path)

    with pytest.raises(RuntimeError, match="rollback after insert"):
        with connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            repo.record_succeeded_on_connection(
                conn,
                identity,
                result_dataset_id="result",
                result_content_hash=result_hash,
                result_artifact_id=artifact["id"],
                result_payload=payload,
                result_workspace_revision=workspace.revision,
                result_analysis_generation=workspace.analysis_generation,
                created_at=_CREATED_AT,
            )
            raise RuntimeError("rollback after insert")

    assert repo.get_for_task("task-1", identity.run_id) is None
    assert repo.list_lineage("task-1") == []

    with connect(db_path) as conn:
        record = repo.record_succeeded_on_connection(
            conn,
            identity,
            result_dataset_id="result",
            result_content_hash=result_hash,
            result_artifact_id=artifact["id"],
            result_payload=payload,
            result_workspace_revision=workspace.revision,
            result_analysis_generation=workspace.analysis_generation,
            created_at=_CREATED_AT,
        )
    with connect(db_path) as conn:
        with pytest.raises(Exception, match="immutable"):
            conn.execute(
                "UPDATE data_transform_runs SET producer_version = 'tampered' WHERE id = ?",
                (record.id,),
            )
        with pytest.raises(Exception, match="immutable"):
            conn.execute(
                "UPDATE dataset_lineage_edges SET edge_order = 2 "
                "WHERE transform_run_id = ?",
                (record.id,),
            )
