"""Immutable task-scoped data-transform runs and dataset lineage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from marvis.db_schema import connect


DATA_TRANSFORM_SCHEMA_VERSION = "data-transform.v1"
DATA_TRANSFORM_INPUT_SCHEMA_VERSION = "data-transform-input.v1"
DATA_TRANSFORM_ARTIFACT_SCHEMA_VERSION = "data-transform-artifact.v1"
DATA_TRANSFORM_EVIDENCE_SCHEMA_VERSION = "data-transform-evidence.v1"
TRANSFORM_RESULT_SCHEMA_VERSION = "transform-result.v1"
TRANSFORM_EXECUTION_MODE = "duckdb-single-thread-v1"
DATASET_LINEAGE_SCHEMA_VERSION = "dataset-lineage.v1"
DATA_TRANSFORM_ARTIFACT_KIND = "data_transform_evidence"
DATA_TRANSFORM_ORIGIN_TOOL = "data_ops.transform_dataset"
_RUN_ID_NAMESPACE = "marvis.data_transform_run.v1"
_EDGE_ID_NAMESPACE = "marvis.dataset_lineage_edge.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataTransformDataError(ValueError):
    """Supplied or persisted transform evidence violates the contract."""


class DataTransformConflictError(RuntimeError):
    """Live task, dataset, artifact, or workspace evidence drifted."""


@dataclass(frozen=True)
class DataTransformIdentity:
    task_id: str
    source_dataset_id: str
    source_content_hash: str
    workspace_revision: int
    analysis_generation: int
    semantic_mapping_hash: str
    operations: Sequence[Mapping[str, Any]]
    producer_version: str
    operations_json: str = field(init=False)
    operations_hash: str = field(init=False)
    input_hash: str = field(init=False)
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        task_id = _canonical_text(self.task_id, field_name="task_id")
        source_dataset_id = _canonical_text(
            self.source_dataset_id,
            field_name="source_dataset_id",
        )
        source_content_hash = _sha256(
            self.source_content_hash,
            field_name="source_content_hash",
        )
        workspace_revision = _non_negative_int(
            self.workspace_revision,
            field_name="workspace_revision",
        )
        analysis_generation = _non_negative_int(
            self.analysis_generation,
            field_name="analysis_generation",
        )
        semantic_mapping_hash = _sha256(
            self.semantic_mapping_hash,
            field_name="semantic_mapping_hash",
        )
        producer_version = _canonical_text(
            self.producer_version,
            field_name="producer_version",
        )
        operations_json = _canonical_operations(self.operations)
        normalized_operations = tuple(json.loads(operations_json))
        operations_hash = _digest(operations_json)
        # Page/selection-only workspace revisions do not change the computation.
        # The originating revision remains immutable provenance, while dataset,
        # analysis generation, semantic hash, operations and producer own cache
        # identity.
        input_payload = {
            "schema_version": DATA_TRANSFORM_INPUT_SCHEMA_VERSION,
            "task_id": task_id,
            "source_dataset_id": source_dataset_id,
            "source_content_hash": source_content_hash,
            "analysis_generation": analysis_generation,
            "semantic_mapping_hash": semantic_mapping_hash,
            "operations_hash": operations_hash,
            "producer_version": producer_version,
        }
        input_hash = _digest(_canonical_json(input_payload, field_name="input"))
        run_id = _stable_id(
            prefix="dtr_",
            namespace=_RUN_ID_NAMESPACE,
            parts=(task_id, input_hash),
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "source_dataset_id", source_dataset_id)
        object.__setattr__(self, "source_content_hash", source_content_hash)
        object.__setattr__(self, "workspace_revision", workspace_revision)
        object.__setattr__(self, "analysis_generation", analysis_generation)
        object.__setattr__(self, "semantic_mapping_hash", semantic_mapping_hash)
        object.__setattr__(self, "operations", normalized_operations)
        object.__setattr__(self, "producer_version", producer_version)
        object.__setattr__(self, "operations_json", operations_json)
        object.__setattr__(self, "operations_hash", operations_hash)
        object.__setattr__(self, "input_hash", input_hash)
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True)
class DataTransformRecord:
    id: str
    schema_version: str
    task_id: str
    source_dataset_id: str
    source_content_hash: str
    workspace_revision: int
    analysis_generation: int
    semantic_mapping_hash: str
    operations_json: str
    operations_hash: str
    producer_version: str
    input_hash: str
    result_dataset_id: str
    result_content_hash: str
    result_artifact_id: str
    result_json: str
    result_hash: str
    result_workspace_revision: int
    result_analysis_generation: int
    created_at: str

    @property
    def operations(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self.operations_json))

    @property
    def result_payload(self) -> dict[str, Any]:
        value = json.loads(self.result_json)
        if not isinstance(value, dict):  # pragma: no cover - constructor validates
            raise DataTransformDataError("persisted result payload must be an object")
        return value


def data_transform_artifact_provenance(
    identity: DataTransformIdentity,
    *,
    result_dataset_id: str,
    result_content_hash: str,
) -> dict[str, Any]:
    if not isinstance(identity, DataTransformIdentity):
        raise DataTransformDataError("identity must be DataTransformIdentity")
    return {
        "schema_version": DATA_TRANSFORM_ARTIFACT_SCHEMA_VERSION,
        "run_id": identity.run_id,
        "task_id": identity.task_id,
        "source_dataset_id": identity.source_dataset_id,
        "source_content_hash": identity.source_content_hash,
        "result_dataset_id": _canonical_text(
            result_dataset_id,
            field_name="result_dataset_id",
        ),
        "result_content_hash": _sha256(
            result_content_hash,
            field_name="result_content_hash",
        ),
        "operations_hash": identity.operations_hash,
        "input_hash": identity.input_hash,
        "producer_version": identity.producer_version,
    }


class DataTransformRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        return connect(self.db_path)

    def get_for_task(
        self,
        task_id: str,
        run_id: str,
    ) -> DataTransformRecord | None:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        normalized_run_id = _canonical_text(run_id, field_name="run_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM data_transform_runs WHERE task_id = ? AND id = ?",
                (normalized_task_id, normalized_run_id),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def find_by_input_hash(
        self,
        task_id: str,
        input_hash: str,
    ) -> DataTransformRecord | None:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        normalized_hash = _sha256(input_hash, field_name="input_hash")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM data_transform_runs WHERE task_id = ? AND input_hash = ?",
                (normalized_task_id, normalized_hash),
            ).fetchone()
        return None if row is None else _record_from_row(row)

    def list_lineage(self, task_id: str) -> list[dict[str, Any]]:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM dataset_lineage_edges
                 WHERE task_id = ?
                 ORDER BY created_at, edge_order, id
                """,
                (normalized_task_id,),
            ).fetchall()
        return [_lineage_from_row(row) for row in rows]

    def record_succeeded_on_connection(
        self,
        conn: sqlite3.Connection,
        identity: DataTransformIdentity,
        *,
        result_dataset_id: str,
        result_content_hash: str,
        result_artifact_id: str,
        result_payload: Mapping[str, Any],
        result_workspace_revision: int,
        result_analysis_generation: int,
        created_at: str | None = None,
    ) -> DataTransformRecord:
        """Insert a successful run and its lineage within the caller's UoW."""

        if not isinstance(identity, DataTransformIdentity):
            raise DataTransformDataError("identity must be DataTransformIdentity")
        result_dataset = _canonical_text(
            result_dataset_id,
            field_name="result_dataset_id",
        )
        result_content = _sha256(
            result_content_hash,
            field_name="result_content_hash",
        )
        artifact_id = _canonical_text(
            result_artifact_id,
            field_name="result_artifact_id",
        )
        result_json = _canonical_json(result_payload, field_name="result_payload")
        result_hash = _digest(result_json)
        result_revision = _non_negative_int(
            result_workspace_revision,
            field_name="result_workspace_revision",
        )
        result_generation = _non_negative_int(
            result_analysis_generation,
            field_name="result_analysis_generation",
        )
        if result_revision <= identity.workspace_revision:
            raise DataTransformDataError(
                "result_workspace_revision must be newer than the source revision"
            )
        if result_generation != identity.analysis_generation + 1:
            raise DataTransformDataError(
                "result_analysis_generation must increment source generation once"
            )
        normalized_payload = json.loads(result_json)
        _validate_result_payload(
            normalized_payload,
            identity=identity,
            result_dataset_id=result_dataset,
            result_content_hash=result_content,
            result_workspace_revision=result_revision,
            result_analysis_generation=result_generation,
        )
        timestamp = _timestamp(created_at or _now())

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM data_transform_runs WHERE task_id = ? AND input_hash = ?",
            (identity.task_id, identity.input_hash),
        ).fetchone()
        if existing is not None:
            record = _record_from_row(existing)
            _require_exact_replay(
                record,
                identity=identity,
                result_dataset_id=result_dataset,
                result_content_hash=result_content,
                result_artifact_id=artifact_id,
                result_json=result_json,
                result_hash=result_hash,
                result_workspace_revision=result_revision,
                result_analysis_generation=result_generation,
            )
            return record

        collision = conn.execute(
            "SELECT 1 FROM data_transform_runs WHERE id = ?",
            (identity.run_id,),
        ).fetchone()
        if collision is not None:
            raise DataTransformConflictError("stable transform run id collision")

        _require_dataset(
            conn,
            dataset_id=identity.source_dataset_id,
            task_id=identity.task_id,
            content_hash=identity.source_content_hash,
            label="source",
        )
        _require_dataset(
            conn,
            dataset_id=result_dataset,
            task_id=identity.task_id,
            content_hash=result_content,
            label="result",
        )
        _require_result_workspace(
            conn,
            task_id=identity.task_id,
            result_dataset_id=result_dataset,
            result_content_hash=result_content,
            revision=result_revision,
            analysis_generation=result_generation,
        )
        _require_artifact(
            conn,
            identity=identity,
            artifact_id=artifact_id,
            result_dataset_id=result_dataset,
            result_content_hash=result_content,
            result_hash=result_hash,
        )

        conn.execute(
            """
            INSERT INTO data_transform_runs(
                id, schema_version, task_id, source_dataset_id,
                source_content_hash, workspace_revision, analysis_generation,
                semantic_mapping_hash, operations_json, operations_hash,
                producer_version, input_hash, result_dataset_id,
                result_content_hash, result_artifact_id, result_json,
                result_hash, result_workspace_revision,
                result_analysis_generation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity.run_id,
                DATA_TRANSFORM_SCHEMA_VERSION,
                identity.task_id,
                identity.source_dataset_id,
                identity.source_content_hash,
                identity.workspace_revision,
                identity.analysis_generation,
                identity.semantic_mapping_hash,
                identity.operations_json,
                identity.operations_hash,
                identity.producer_version,
                identity.input_hash,
                result_dataset,
                result_content,
                artifact_id,
                result_json,
                result_hash,
                result_revision,
                result_generation,
                timestamp,
            ),
        )
        edge_id = _stable_id(
            prefix="dle_",
            namespace=_EDGE_ID_NAMESPACE,
            parts=(
                identity.task_id,
                identity.source_dataset_id,
                result_dataset,
                identity.run_id,
                "0",
            ),
        )
        conn.execute(
            """
            INSERT INTO dataset_lineage_edges(
                id, schema_version, task_id, parent_dataset_id,
                child_dataset_id, transform_run_id, relation_kind,
                edge_order, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'transform', 0, ?)
            """,
            (
                edge_id,
                DATASET_LINEAGE_SCHEMA_VERSION,
                identity.task_id,
                identity.source_dataset_id,
                result_dataset,
                identity.run_id,
                timestamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM data_transform_runs WHERE id = ?",
            (identity.run_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - defensive after successful INSERT
            raise DataTransformDataError("transform run disappeared after insert")
        return _record_from_row(row)


def _require_dataset(
    conn: sqlite3.Connection,
    *,
    dataset_id: str,
    task_id: str,
    content_hash: str,
    label: str,
) -> None:
    row = conn.execute(
        "SELECT task_id, content_hash FROM datasets WHERE id = ?",
        (dataset_id,),
    ).fetchone()
    if row is None:
        raise DataTransformDataError(f"{label} dataset not found: {dataset_id}")
    if str(row["task_id"]) != task_id:
        raise DataTransformDataError(f"{label} dataset is not owned by task")
    registered = row["content_hash"]
    if not isinstance(registered, str) or not hmac.compare_digest(
        registered,
        content_hash,
    ):
        raise DataTransformDataError(f"{label} dataset content hash mismatch")


def _require_result_workspace(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    result_dataset_id: str,
    result_content_hash: str,
    revision: int,
    analysis_generation: int,
) -> None:
    row = conn.execute(
        "SELECT active_dataset_id, active_dataset_content_hash, revision, "
        "analysis_generation FROM data_workspaces WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        raise DataTransformConflictError("result workspace is missing")
    active_hash = row["active_dataset_content_hash"]
    if (
        str(row["active_dataset_id"]) != result_dataset_id
        or not isinstance(active_hash, str)
        or not hmac.compare_digest(active_hash, result_content_hash)
        or int(row["revision"]) != revision
        or int(row["analysis_generation"]) != analysis_generation
    ):
        raise DataTransformConflictError("result workspace evidence drifted")


def _require_artifact(
    conn: sqlite3.Connection,
    *,
    identity: DataTransformIdentity,
    artifact_id: str,
    result_dataset_id: str,
    result_content_hash: str,
    result_hash: str,
) -> None:
    row = conn.execute(
        "SELECT * FROM task_artifacts WHERE id = ? AND task_id = ?",
        (artifact_id, identity.task_id),
    ).fetchone()
    expected_provenance = data_transform_artifact_provenance(
        identity,
        result_dataset_id=result_dataset_id,
        result_content_hash=result_content_hash,
    )
    expected_json = _canonical_json(
        expected_provenance,
        field_name="artifact provenance",
    )
    if row is None:
        raise DataTransformDataError("transform artifact evidence is missing")
    if (
        str(row["kind"]) != DATA_TRANSFORM_ARTIFACT_KIND
        or str(row["origin_tool"]) != DATA_TRANSFORM_ORIGIN_TOOL
        or str(row["content_hash"]) != result_hash
        or str(row["provenance_json"]) != expected_json
    ):
        raise DataTransformDataError("transform artifact evidence does not match run")


def _require_exact_replay(
    record: DataTransformRecord,
    *,
    identity: DataTransformIdentity,
    result_dataset_id: str,
    result_content_hash: str,
    result_artifact_id: str,
    result_json: str,
    result_hash: str,
    result_workspace_revision: int,
    result_analysis_generation: int,
) -> None:
    expected = (
        identity.run_id,
        identity.task_id,
        identity.source_dataset_id,
        identity.source_content_hash,
        identity.analysis_generation,
        identity.semantic_mapping_hash,
        identity.operations_json,
        identity.operations_hash,
        identity.producer_version,
        identity.input_hash,
        result_dataset_id,
        result_content_hash,
        result_artifact_id,
        result_json,
        result_hash,
        result_workspace_revision,
        result_analysis_generation,
    )
    actual = (
        record.id,
        record.task_id,
        record.source_dataset_id,
        record.source_content_hash,
        record.analysis_generation,
        record.semantic_mapping_hash,
        record.operations_json,
        record.operations_hash,
        record.producer_version,
        record.input_hash,
        record.result_dataset_id,
        record.result_content_hash,
        record.result_artifact_id,
        record.result_json,
        record.result_hash,
        record.result_workspace_revision,
        record.result_analysis_generation,
    )
    if actual != expected:
        raise DataTransformConflictError(
            "transform input hash replay has different persisted evidence"
        )


def _record_from_row(row: sqlite3.Row) -> DataTransformRecord:
    try:
        operations_json = str(row["operations_json"])
        operations = json.loads(operations_json)
        if not isinstance(operations, list) or not operations:
            raise ValueError("operations must be a non-empty array")
        if _canonical_operations(operations) != operations_json:
            raise ValueError("operations_json is not canonical")
        result_json = str(row["result_json"])
        result_payload = json.loads(result_json)
        if not isinstance(result_payload, dict):
            raise ValueError("result_json must be an object")
        if _canonical_json(result_payload, field_name="result") != result_json:
            raise ValueError("result_json is not canonical")
        identity = DataTransformIdentity(
            task_id=row["task_id"],
            source_dataset_id=row["source_dataset_id"],
            source_content_hash=row["source_content_hash"],
            workspace_revision=row["workspace_revision"],
            analysis_generation=row["analysis_generation"],
            semantic_mapping_hash=row["semantic_mapping_hash"],
            operations=operations,
            producer_version=row["producer_version"],
        )
        if str(row["schema_version"]) != DATA_TRANSFORM_SCHEMA_VERSION:
            raise ValueError("unsupported schema version")
        if str(row["id"]) != identity.run_id:
            raise ValueError("run id does not match input identity")
        if str(row["operations_hash"]) != identity.operations_hash:
            raise ValueError("operations hash mismatch")
        if str(row["input_hash"]) != identity.input_hash:
            raise ValueError("input hash mismatch")
        if str(row["result_hash"]) != _digest(result_json):
            raise ValueError("result hash mismatch")
        result_dataset_id = _canonical_text(
            row["result_dataset_id"],
            field_name="result_dataset_id",
        )
        result_content_hash = _sha256(
            row["result_content_hash"],
            field_name="result_content_hash",
        )
        result_workspace_revision = _non_negative_int(
            row["result_workspace_revision"],
            field_name="result_workspace_revision",
        )
        result_analysis_generation = _non_negative_int(
            row["result_analysis_generation"],
            field_name="result_analysis_generation",
        )
        if result_workspace_revision <= identity.workspace_revision:
            raise ValueError("result workspace revision is not newer")
        if result_analysis_generation != identity.analysis_generation + 1:
            raise ValueError("result analysis generation did not increment once")
        _validate_result_payload(
            result_payload,
            identity=identity,
            result_dataset_id=result_dataset_id,
            result_content_hash=result_content_hash,
            result_workspace_revision=result_workspace_revision,
            result_analysis_generation=result_analysis_generation,
        )
        return DataTransformRecord(
            id=identity.run_id,
            schema_version=DATA_TRANSFORM_SCHEMA_VERSION,
            task_id=identity.task_id,
            source_dataset_id=identity.source_dataset_id,
            source_content_hash=identity.source_content_hash,
            workspace_revision=identity.workspace_revision,
            analysis_generation=identity.analysis_generation,
            semantic_mapping_hash=identity.semantic_mapping_hash,
            operations_json=identity.operations_json,
            operations_hash=identity.operations_hash,
            producer_version=identity.producer_version,
            input_hash=identity.input_hash,
            result_dataset_id=result_dataset_id,
            result_content_hash=result_content_hash,
            result_artifact_id=_canonical_text(
                row["result_artifact_id"],
                field_name="result_artifact_id",
            ),
            result_json=result_json,
            result_hash=_sha256(row["result_hash"], field_name="result_hash"),
            result_workspace_revision=result_workspace_revision,
            result_analysis_generation=result_analysis_generation,
            created_at=_timestamp(row["created_at"]),
        )
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, DataTransformDataError):
            raise
        raise DataTransformDataError("corrupt data transform record") from exc


def _validate_result_payload(
    payload: object,
    *,
    identity: DataTransformIdentity,
    result_dataset_id: str,
    result_content_hash: str,
    result_workspace_revision: int,
    result_analysis_generation: int,
) -> None:
    """Bind transform evidence to its immutable identity and result row."""

    evidence = _evidence_object(payload, field_name="result_payload")
    _require_evidence_value(
        evidence.get("schema_version"),
        DATA_TRANSFORM_EVIDENCE_SCHEMA_VERSION,
        field_name="schema_version",
    )
    _require_evidence_value(
        evidence.get("run_id"),
        identity.run_id,
        field_name="run_id",
    )
    _require_evidence_value(
        evidence.get("producer_version"),
        identity.producer_version,
        field_name="producer_version",
    )
    _require_evidence_hash(
        evidence.get("input_hash"),
        identity.input_hash,
        field_name="input_hash",
    )
    _require_evidence_hash(
        evidence.get("operations_hash"),
        identity.operations_hash,
        field_name="operations_hash",
    )

    source = _evidence_object(evidence.get("source"), field_name="source")
    _require_evidence_value(
        source.get("dataset_id"),
        identity.source_dataset_id,
        field_name="source.dataset_id",
    )
    _require_evidence_hash(
        source.get("content_hash"),
        identity.source_content_hash,
        field_name="source.content_hash",
    )
    source_rows = _non_negative_int(
        source.get("row_count"),
        field_name="source.row_count",
    )

    result = _evidence_object(evidence.get("result"), field_name="result")
    _require_evidence_value(
        result.get("dataset_id"),
        result_dataset_id,
        field_name="result.dataset_id",
    )
    _require_evidence_hash(
        result.get("content_hash"),
        result_content_hash,
        field_name="result.content_hash",
    )
    result_rows = _non_negative_int(
        result.get("row_count"),
        field_name="result.row_count",
    )

    transform = _evidence_object(
        evidence.get("transform"),
        field_name="transform",
    )
    _require_evidence_value(
        transform.get("schema_version"),
        TRANSFORM_RESULT_SCHEMA_VERSION,
        field_name="transform.schema_version",
    )
    execution = _evidence_object(
        transform.get("execution"),
        field_name="transform.execution",
    )
    _require_evidence_value(
        execution.get("mode"),
        TRANSFORM_EXECUTION_MODE,
        field_name="transform.execution.mode",
    )
    _require_evidence_integer(
        execution.get("duckdb_threads"),
        1,
        field_name="transform.execution.duckdb_threads",
    )
    _require_evidence_value(
        execution.get("preserve_insertion_order"),
        True,
        field_name="transform.execution.preserve_insertion_order",
    )
    transform_operations = transform.get("operations")
    try:
        transform_operations_json = _canonical_operations(transform_operations)
    except DataTransformDataError as exc:
        raise DataTransformDataError(
            "transform.operations must be the canonical run operations"
        ) from exc
    if not hmac.compare_digest(transform_operations_json, identity.operations_json):
        raise DataTransformDataError(
            "transform.operations must match the run identity"
        )
    steps = _evidence_array(transform.get("steps"), field_name="transform.steps")
    summary = _evidence_object(
        transform.get("summary"),
        field_name="transform.summary",
    )
    _validate_transform_summary(
        summary,
        source_rows=source_rows,
        result_rows=result_rows,
        operation_count=len(identity.operations),
    )
    _validate_transform_steps(
        steps,
        operations=identity.operations,
        source_rows=source_rows,
        result_rows=result_rows,
    )
    output = _evidence_object(
        transform.get("output"),
        field_name="transform.output",
    )
    _require_evidence_hash(
        output.get("content_hash"),
        result_content_hash,
        field_name="transform.output.content_hash",
    )
    _require_evidence_value(
        _non_negative_int(
            output.get("row_count"),
            field_name="transform.output.row_count",
        ),
        result_rows,
        field_name="transform.output.row_count",
    )

    semantic = _evidence_object(
        evidence.get("semantic_migration"),
        field_name="semantic_migration",
    )
    _require_evidence_hash(
        semantic.get("before_hash"),
        identity.semantic_mapping_hash,
        field_name="semantic_migration.before_hash",
    )

    workspace = _evidence_object(
        evidence.get("workspace"),
        field_name="workspace",
    )
    _require_evidence_integer(
        workspace.get("source_revision"),
        identity.workspace_revision,
        field_name="workspace.source_revision",
    )
    _require_evidence_integer(
        workspace.get("result_revision"),
        result_workspace_revision,
        field_name="workspace.result_revision",
    )
    _require_evidence_integer(
        workspace.get("source_analysis_generation"),
        identity.analysis_generation,
        field_name="workspace.source_analysis_generation",
    )
    _require_evidence_integer(
        workspace.get("result_analysis_generation"),
        result_analysis_generation,
        field_name="workspace.result_analysis_generation",
    )

    lineage = _evidence_object(
        evidence.get("lineage"),
        field_name="lineage",
    )
    _require_evidence_value(
        lineage.get("parent_dataset_id"),
        identity.source_dataset_id,
        field_name="lineage.parent_dataset_id",
    )
    _require_evidence_value(
        lineage.get("child_dataset_id"),
        result_dataset_id,
        field_name="lineage.child_dataset_id",
    )
    _require_evidence_value(
        lineage.get("relation_kind"),
        "transform",
        field_name="lineage.relation_kind",
    )
    _require_evidence_integer(
        lineage.get("edge_order"),
        0,
        field_name="lineage.edge_order",
    )


def _validate_transform_summary(
    summary: Mapping[str, Any],
    *,
    source_rows: int,
    result_rows: int,
    operation_count: int,
) -> None:
    expected = {
        "row_count_before": source_rows,
        "row_count_after": result_rows,
        "row_delta": result_rows - source_rows,
        "operation_count": operation_count,
    }
    for name, expected_value in expected.items():
        value = summary.get(name)
        if name == "row_delta":
            actual = _integer(value, field_name=f"transform.summary.{name}")
        else:
            actual = _non_negative_int(
                value,
                field_name=f"transform.summary.{name}",
            )
        _require_evidence_value(
            actual,
            expected_value,
            field_name=f"transform.summary.{name}",
        )
    for name in ("column_count_before", "column_count_after"):
        _non_negative_int(
            summary.get(name),
            field_name=f"transform.summary.{name}",
        )


def _validate_transform_steps(
    steps: Sequence[object],
    *,
    operations: Sequence[Mapping[str, Any]],
    source_rows: int,
    result_rows: int,
) -> None:
    if len(steps) != len(operations):
        raise DataTransformDataError(
            "transform.steps length must match transform.operations"
        )
    prior_rows = source_rows
    for index, (raw_step, operation) in enumerate(zip(steps, operations), start=1):
        field = f"transform.steps[{index - 1}]"
        step = _evidence_object(raw_step, field_name=field)
        _require_evidence_integer(step.get("step"), index, field_name=f"{field}.step")
        expected_op = _canonical_text(
            operation.get("op"),
            field_name=f"transform.operations[{index - 1}].op",
        )
        actual_op = _canonical_text(step.get("op"), field_name=f"{field}.op")
        _require_evidence_value(
            actual_op,
            expected_op,
            field_name=f"{field}.op",
        )
        before = _non_negative_int(
            step.get("row_count_before"),
            field_name=f"{field}.row_count_before",
        )
        after = _non_negative_int(
            step.get("row_count_after"),
            field_name=f"{field}.row_count_after",
        )
        delta = _integer(
            step.get("row_delta"),
            field_name=f"{field}.row_delta",
        )
        _require_evidence_value(
            before,
            prior_rows,
            field_name=f"{field}.row_count_before",
        )
        _require_evidence_value(
            delta,
            after - before,
            field_name=f"{field}.row_delta",
        )
        _evidence_array(
            step.get("columns_before"),
            field_name=f"{field}.columns_before",
        )
        _evidence_array(
            step.get("columns_after"),
            field_name=f"{field}.columns_after",
        )
        _evidence_object(step.get("impact"), field_name=f"{field}.impact")
        prior_rows = after
    _require_evidence_value(
        prior_rows,
        result_rows,
        field_name="transform.steps final row_count_after",
    )


def _evidence_object(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataTransformDataError(f"{field_name} must be an object")
    return value


def _evidence_array(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise DataTransformDataError(f"{field_name} must be an array")
    return value


def _require_evidence_hash(
    value: object,
    expected: str,
    *,
    field_name: str,
) -> None:
    actual = _sha256(value, field_name=field_name)
    if not hmac.compare_digest(actual, expected):
        raise DataTransformDataError(f"{field_name} does not match the run")


def _require_evidence_integer(
    value: object,
    expected: int,
    *,
    field_name: str,
) -> None:
    actual = _non_negative_int(value, field_name=field_name)
    _require_evidence_value(actual, expected, field_name=field_name)


def _require_evidence_value(
    value: object,
    expected: object,
    *,
    field_name: str,
) -> None:
    if value != expected or type(value) is not type(expected):
        raise DataTransformDataError(f"{field_name} does not match the run")


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataTransformDataError(f"{field_name} must be an integer")
    return value


def _lineage_from_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        schema_version = str(row["schema_version"])
        relation_kind = str(row["relation_kind"])
        if schema_version != DATASET_LINEAGE_SCHEMA_VERSION:
            raise ValueError("unsupported lineage schema")
        if relation_kind != "transform":
            raise ValueError("unsupported lineage relation")
        return {
            "id": _canonical_text(row["id"], field_name="lineage id"),
            "schema_version": schema_version,
            "task_id": _canonical_text(row["task_id"], field_name="task_id"),
            "parent_dataset_id": _canonical_text(
                row["parent_dataset_id"],
                field_name="parent_dataset_id",
            ),
            "child_dataset_id": _canonical_text(
                row["child_dataset_id"],
                field_name="child_dataset_id",
            ),
            "transform_run_id": _canonical_text(
                row["transform_run_id"],
                field_name="transform_run_id",
            ),
            "relation_kind": relation_kind,
            "edge_order": _non_negative_int(
                row["edge_order"],
                field_name="edge_order",
            ),
            "created_at": _timestamp(row["created_at"]),
        }
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, DataTransformDataError):
            raise
        raise DataTransformDataError("corrupt dataset lineage record") from exc


def _canonical_operations(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise DataTransformDataError("operations must be a non-empty array")
    operations = list(value)
    if not operations or not all(isinstance(item, Mapping) for item in operations):
        raise DataTransformDataError("operations must be a non-empty object array")
    return _canonical_json(operations, field_name="operations")


def _canonical_json(value: object, *, field_name: str) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataTransformDataError(
            f"{field_name} must be canonical JSON"
        ) from exc
    return payload


def _canonical_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise DataTransformDataError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataTransformDataError(
            f"{field_name} must be a 64-character lowercase SHA-256 hex"
        )
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataTransformDataError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _timestamp(value: object) -> str:
    text = _canonical_text(value, field_name="timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataTransformDataError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise DataTransformDataError("timestamp must include timezone")
    return text


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_id(
    *,
    prefix: str,
    namespace: str,
    parts: tuple[str, ...],
) -> str:
    payload = "\x00".join((namespace, *parts))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "DATASET_LINEAGE_SCHEMA_VERSION",
    "DATA_TRANSFORM_ARTIFACT_KIND",
    "DATA_TRANSFORM_ARTIFACT_SCHEMA_VERSION",
    "DATA_TRANSFORM_EVIDENCE_SCHEMA_VERSION",
    "DATA_TRANSFORM_INPUT_SCHEMA_VERSION",
    "DATA_TRANSFORM_ORIGIN_TOOL",
    "DATA_TRANSFORM_SCHEMA_VERSION",
    "DataTransformConflictError",
    "DataTransformDataError",
    "DataTransformIdentity",
    "DataTransformRecord",
    "DataTransformRepository",
    "data_transform_artifact_provenance",
]
