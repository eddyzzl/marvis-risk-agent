"""Immutable committed facts for full automatic-tree dataset writeback.

The caller owns file promotion, dataset/artifact registration, optional
workspace activation and audit.  This repository only binds the already
committed task-owned rows inside the caller's SQLite transaction and resolves
an exact computational replay by ``(task_id, input_hash)``.
"""

from __future__ import annotations

from collections.abc import Mapping
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


AUTOMATIC_TREE_APPLY_RUN_SCHEMA_VERSION = "strategy.automatic-tree-apply-run.v1"
AUTOMATIC_TREE_APPLY_INPUT_SCHEMA_VERSION = "strategy.automatic-tree-apply-input.v1"
AUTOMATIC_TREE_APPLY_RESULT_SCHEMA_VERSION = "strategy.automatic-tree-apply-result.v1"
AUTOMATIC_TREE_APPLY_PRODUCER_VERSION = "strategy.automatic-tree-apply/1"
AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND = "strategy_automatic_tree_asset_json"
AUTOMATIC_TREE_SOURCE_ORIGIN_TOOL = "strategy.build_automatic_tree_candidate"
AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND = "strategy_automatic_tree_apply_evidence"
AUTOMATIC_TREE_APPLY_ORIGIN_TOOL = "strategy.apply_automatic_tree"

_RUN_ID_NAMESPACE = "marvis.strategy.automatic_tree_apply_run.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_OUTPUT_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RESULT_ID_RE = re.compile(r"^automatic-tree-apply-[0-9a-f]{32}$")
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "producer_version",
        "result_id",
        "source",
        "tree",
        "output",
        "writer",
        "result_hash",
    }
)
_WRITER_FIELDS = frozenset(
    {
        "contract",
        "engine",
        "engine_version",
        "threads",
        "preserve_insertion_order",
        "batch_rows",
        "max_decoded_batch_bytes",
        "row_group_rows",
        "write_batch_rows",
        "parquet_version",
        "data_page_version",
        "compression",
        "compression_level",
        "dictionary_encoding",
        "dictionary_page_bytes",
        "write_statistics",
        "byte_stream_split",
        "use_deprecated_int96_timestamps",
        "use_compliant_nested_type",
        "store_arrow_schema",
        "write_page_index",
        "write_page_checksum",
        "source_schema_metadata",
        "appended_id_type",
    }
)


class AutomaticTreeApplyDataError(ValueError):
    """Supplied or persisted apply-run facts violate the exact contract."""


class AutomaticTreeApplyConflictError(RuntimeError):
    """The same logical run or one of its durable bindings has drifted."""


class AutomaticTreeApplyNotFoundError(KeyError):
    """A task-owned entity is absent or intentionally masked as absent."""


@dataclass(frozen=True)
class AutomaticTreeApplyIdentity:
    task_id: str
    source_tree_artifact_id: str
    source_tree_artifact_hash: str
    asset_id: str
    asset_hash: str
    tree_result_hash: str
    source_dataset_id: str
    source_dataset_hash: str
    output_leaf_column: str
    output_rule_column: str
    writer_contract: str
    writer_version: str
    input_json: str = field(init=False)
    input_hash: str = field(init=False)
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        task_id = _canonical_text(self.task_id, field_name="task_id")
        source_tree_artifact_id = _canonical_text(
            self.source_tree_artifact_id,
            field_name="source_tree_artifact_id",
        )
        source_tree_artifact_hash = _sha256(
            self.source_tree_artifact_hash,
            field_name="source_tree_artifact_hash",
        )
        asset_id = _asset_id(self.asset_id)
        asset_hash = _sha256(self.asset_hash, field_name="asset_hash")
        tree_result_hash = _sha256(
            self.tree_result_hash,
            field_name="tree_result_hash",
        )
        source_dataset_id = _canonical_text(
            self.source_dataset_id,
            field_name="source_dataset_id",
        )
        source_dataset_hash = _sha256(
            self.source_dataset_hash,
            field_name="source_dataset_hash",
        )
        output_leaf_column = _output_column(
            self.output_leaf_column,
            field_name="output_leaf_column",
        )
        output_rule_column = _output_column(
            self.output_rule_column,
            field_name="output_rule_column",
        )
        if output_leaf_column.lower() == output_rule_column.lower():
            raise AutomaticTreeApplyDataError(
                "output leaf/rule columns must be distinct case-insensitively"
            )
        writer_contract = _canonical_text(
            self.writer_contract,
            field_name="writer_contract",
        )
        writer_version = _canonical_text(
            self.writer_version,
            field_name="writer_version",
        )
        input_payload = {
            "schema_version": AUTOMATIC_TREE_APPLY_INPUT_SCHEMA_VERSION,
            "task_id": task_id,
            "source_tree_artifact": {
                "id": source_tree_artifact_id,
                "content_hash": source_tree_artifact_hash,
            },
            "tree_asset": {
                "asset_id": asset_id,
                "asset_hash": asset_hash,
                "result_hash": tree_result_hash,
            },
            "source_dataset": {
                "id": source_dataset_id,
                "content_hash": source_dataset_hash,
            },
            "output_columns": {
                "leaf_id": output_leaf_column,
                "rule_id": output_rule_column,
            },
            "writer": {
                "contract": writer_contract,
                "version": writer_version,
            },
        }
        input_json = _canonical_json(input_payload, field_name="input")
        input_hash = _digest(input_json)
        run_id = _stable_id(task_id=task_id, input_hash=input_hash)

        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(
            self,
            "source_tree_artifact_id",
            source_tree_artifact_id,
        )
        object.__setattr__(
            self,
            "source_tree_artifact_hash",
            source_tree_artifact_hash,
        )
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "asset_hash", asset_hash)
        object.__setattr__(self, "tree_result_hash", tree_result_hash)
        object.__setattr__(self, "source_dataset_id", source_dataset_id)
        object.__setattr__(self, "source_dataset_hash", source_dataset_hash)
        object.__setattr__(self, "output_leaf_column", output_leaf_column)
        object.__setattr__(self, "output_rule_column", output_rule_column)
        object.__setattr__(self, "writer_contract", writer_contract)
        object.__setattr__(self, "writer_version", writer_version)
        object.__setattr__(self, "input_json", input_json)
        object.__setattr__(self, "input_hash", input_hash)
        object.__setattr__(self, "run_id", run_id)


@dataclass(frozen=True)
class AutomaticTreeApplyCommittedFacts:
    result_dataset_id: str
    result_dataset_hash: str
    result_dataset_path: str
    evidence_artifact_id: str
    evidence_artifact_hash: str
    evidence_artifact_path: str

    def __post_init__(self) -> None:
        result_dataset_id = _canonical_text(
            self.result_dataset_id,
            field_name="result_dataset_id",
        )
        result_dataset_hash = _sha256(
            self.result_dataset_hash,
            field_name="result_dataset_hash",
        )
        result_dataset_path = _canonical_text(
            self.result_dataset_path,
            field_name="result_dataset_path",
        )
        evidence_artifact_id = _canonical_text(
            self.evidence_artifact_id,
            field_name="evidence_artifact_id",
        )
        evidence_artifact_hash = _sha256(
            self.evidence_artifact_hash,
            field_name="evidence_artifact_hash",
        )
        evidence_artifact_path = _canonical_text(
            self.evidence_artifact_path,
            field_name="evidence_artifact_path",
        )
        object.__setattr__(self, "result_dataset_id", result_dataset_id)
        object.__setattr__(self, "result_dataset_hash", result_dataset_hash)
        object.__setattr__(self, "result_dataset_path", result_dataset_path)
        object.__setattr__(self, "evidence_artifact_id", evidence_artifact_id)
        object.__setattr__(self, "evidence_artifact_hash", evidence_artifact_hash)
        object.__setattr__(self, "evidence_artifact_path", evidence_artifact_path)


@dataclass(frozen=True)
class AutomaticTreeApplyRecord:
    id: str
    schema_version: str
    identity: AutomaticTreeApplyIdentity
    committed: AutomaticTreeApplyCommittedFacts
    result_json: str
    result_hash: str
    created_at: str

    @property
    def run_id(self) -> str:
        return self.id

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @property
    def input_hash(self) -> str:
        return self.identity.input_hash

    @property
    def result_json_hash(self) -> str:
        return self.result_hash

    @property
    def result_payload(self) -> dict[str, Any]:
        value = json.loads(self.result_json)
        if not isinstance(value, dict):  # pragma: no cover - row parser validates
            raise AutomaticTreeApplyDataError("persisted result must be an object")
        return value


class AutomaticTreeApplyRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def create(
        self,
        identity: AutomaticTreeApplyIdentity,
        committed: AutomaticTreeApplyCommittedFacts,
        *,
        result_payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> AutomaticTreeApplyRecord:
        """Create one run or return the exact prior committed replay."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.insert_on_connection(
                conn,
                identity,
                committed,
                result_payload=result_payload,
                created_at=created_at,
            )

    def record_succeeded(
        self,
        identity: AutomaticTreeApplyIdentity,
        committed: AutomaticTreeApplyCommittedFacts,
        *,
        result_payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> AutomaticTreeApplyRecord:
        """Domain spelling for ``create`` used by the writeback Tool."""

        return self.create(
            identity,
            committed,
            result_payload=result_payload,
            created_at=created_at,
        )

    def create_on_connection(
        self,
        conn: sqlite3.Connection,
        identity: AutomaticTreeApplyIdentity,
        committed: AutomaticTreeApplyCommittedFacts,
        *,
        result_payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> AutomaticTreeApplyRecord:
        """Compatibility spelling for the connection-scoped insert seam."""

        return self.insert_on_connection(
            conn,
            identity,
            committed,
            result_payload=result_payload,
            created_at=created_at,
        )

    def record_succeeded_on_connection(
        self,
        conn: sqlite3.Connection,
        identity: AutomaticTreeApplyIdentity,
        committed: AutomaticTreeApplyCommittedFacts,
        *,
        result_payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> AutomaticTreeApplyRecord:
        """Domain spelling for the caller-owned transaction seam."""

        return self.insert_on_connection(
            conn,
            identity,
            committed,
            result_payload=result_payload,
            created_at=created_at,
        )

    def insert_on_connection(
        self,
        conn: sqlite3.Connection,
        identity: AutomaticTreeApplyIdentity,
        committed: AutomaticTreeApplyCommittedFacts,
        *,
        result_payload: Mapping[str, Any],
        created_at: str | None = None,
    ) -> AutomaticTreeApplyRecord:
        """Insert inside the caller transaction without committing it."""

        normalized_identity = _identity(identity)
        normalized_committed = _committed(committed)
        if (
            normalized_committed.result_dataset_id
            == normalized_identity.source_dataset_id
        ):
            raise AutomaticTreeApplyDataError(
                "result_dataset_id must differ from source_dataset_id"
            )
        if (
            normalized_committed.evidence_artifact_id
            == normalized_identity.source_tree_artifact_id
        ):
            raise AutomaticTreeApplyDataError(
                "evidence artifact must differ from source tree artifact"
            )
        timestamp = _timestamp(created_at or _now())
        result_json = canonical_automatic_tree_apply_result_json(
            result_payload,
            identity=normalized_identity,
            committed=normalized_committed,
        )
        result_hash = _digest(result_json)

        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        existing = _select_by_input(
            conn,
            task_id=normalized_identity.task_id,
            input_hash=normalized_identity.input_hash,
        )
        if existing is not None:
            record = _validated_record(conn, existing)
            _require_exact_replay(
                record,
                identity=normalized_identity,
                committed=normalized_committed,
                result_json=result_json,
                result_hash=result_hash,
            )
            return record

        _require_live_bindings(
            conn,
            identity=normalized_identity,
            committed=normalized_committed,
            result_payload=json.loads(result_json),
        )
        collision = conn.execute(
            "SELECT 1 FROM strategy_automatic_tree_apply_runs WHERE id = ?",
            (normalized_identity.run_id,),
        ).fetchone()
        if collision is not None:
            raise AutomaticTreeApplyConflictError(
                "stable automatic-tree apply run id collision"
            )

        try:
            conn.execute(
                """
                INSERT INTO strategy_automatic_tree_apply_runs(
                    id, schema_version, task_id, input_hash,
                    source_tree_artifact_id, source_tree_artifact_hash,
                    asset_id, asset_hash, tree_result_hash,
                    source_dataset_id, source_dataset_hash,
                    output_leaf_column, output_rule_column,
                    writer_contract, writer_version,
                    result_dataset_id, result_dataset_hash, result_dataset_path,
                    evidence_artifact_id, evidence_artifact_hash,
                    evidence_artifact_path, result_json, result_hash, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    normalized_identity.run_id,
                    AUTOMATIC_TREE_APPLY_RUN_SCHEMA_VERSION,
                    normalized_identity.task_id,
                    normalized_identity.input_hash,
                    normalized_identity.source_tree_artifact_id,
                    normalized_identity.source_tree_artifact_hash,
                    normalized_identity.asset_id,
                    normalized_identity.asset_hash,
                    normalized_identity.tree_result_hash,
                    normalized_identity.source_dataset_id,
                    normalized_identity.source_dataset_hash,
                    normalized_identity.output_leaf_column,
                    normalized_identity.output_rule_column,
                    normalized_identity.writer_contract,
                    normalized_identity.writer_version,
                    normalized_committed.result_dataset_id,
                    normalized_committed.result_dataset_hash,
                    normalized_committed.result_dataset_path,
                    normalized_committed.evidence_artifact_id,
                    normalized_committed.evidence_artifact_hash,
                    normalized_committed.evidence_artifact_path,
                    result_json,
                    result_hash,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            replay = _select_by_input(
                conn,
                task_id=normalized_identity.task_id,
                input_hash=normalized_identity.input_hash,
            )
            if replay is None:
                raise AutomaticTreeApplyConflictError(
                    "could not persist automatic-tree apply run"
                ) from exc
            record = _validated_record(conn, replay)
            _require_exact_replay(
                record,
                identity=normalized_identity,
                committed=normalized_committed,
                result_json=result_json,
                result_hash=result_hash,
            )
            return record

        row = conn.execute(
            "SELECT * FROM strategy_automatic_tree_apply_runs WHERE id = ?",
            (normalized_identity.run_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - defensive after successful INSERT
            raise AutomaticTreeApplyDataError(
                "automatic-tree apply run disappeared after insert"
            )
        return _validated_record(conn, row)

    def get_by_id(
        self,
        task_id: str,
        run_id: str,
    ) -> AutomaticTreeApplyRecord | None:
        with connect(self.db_path) as conn:
            return self.get_by_id_on_connection(conn, task_id, run_id)

    def get_by_id_on_connection(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        run_id: str,
    ) -> AutomaticTreeApplyRecord | None:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        normalized_run_id = _canonical_text(run_id, field_name="run_id")
        row = conn.execute(
            """
            SELECT * FROM strategy_automatic_tree_apply_runs
             WHERE task_id = ? AND id = ?
            """,
            (normalized_task_id, normalized_run_id),
        ).fetchone()
        return None if row is None else _validated_record(conn, row)

    def get_by_input(
        self,
        task_id: str | AutomaticTreeApplyIdentity,
        input_hash: str | None = None,
    ) -> AutomaticTreeApplyRecord | None:
        with connect(self.db_path) as conn:
            return self.get_by_input_on_connection(conn, task_id, input_hash)

    def get_by_input_hash(
        self,
        task_id: str | AutomaticTreeApplyIdentity,
        input_hash: str | None = None,
    ) -> AutomaticTreeApplyRecord | None:
        return self.get_by_input(task_id, input_hash)

    def get_by_input_on_connection(
        self,
        conn: sqlite3.Connection,
        task_id: str | AutomaticTreeApplyIdentity,
        input_hash: str | None = None,
    ) -> AutomaticTreeApplyRecord | None:
        expected: AutomaticTreeApplyIdentity | None = None
        if isinstance(task_id, AutomaticTreeApplyIdentity):
            if input_hash is not None:
                raise AutomaticTreeApplyDataError(
                    "input_hash must be omitted when identity is supplied"
                )
            expected = task_id
            normalized_task_id = expected.task_id
            normalized_input_hash = expected.input_hash
        else:
            normalized_task_id = _canonical_text(task_id, field_name="task_id")
            normalized_input_hash = _sha256(input_hash, field_name="input_hash")
        row = _select_by_input(
            conn,
            task_id=normalized_task_id,
            input_hash=normalized_input_hash,
        )
        if row is None:
            return None
        record = _validated_record(conn, row)
        if expected is not None:
            _require_same_identity(record.identity, expected)
        return record


def canonical_automatic_tree_apply_result_json(
    result_payload: Mapping[str, Any],
    *,
    identity: AutomaticTreeApplyIdentity,
    committed: AutomaticTreeApplyCommittedFacts,
) -> str:
    normalized_identity = _identity(identity)
    normalized_committed = _committed(committed)
    normalized = _normalize_result_payload(
        result_payload,
        identity=normalized_identity,
        committed=normalized_committed,
    )
    return _canonical_json(normalized, field_name="result")


def _normalize_result_payload(
    value: object,
    *,
    identity: AutomaticTreeApplyIdentity,
    committed: AutomaticTreeApplyCommittedFacts,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticTreeApplyDataError("result_payload must be a JSON object")
    try:
        normalized = json.loads(_canonical_json(value, field_name="result_payload"))
    except json.JSONDecodeError as exc:  # pragma: no cover - dumps produced it
        raise AutomaticTreeApplyDataError(
            "result_payload must be a JSON object"
        ) from exc
    if not isinstance(normalized, dict):
        raise AutomaticTreeApplyDataError("result_payload must be a JSON object")
    _exact_fields(normalized, _RESULT_FIELDS, field_name="result_payload")
    if normalized["schema_version"] != AUTOMATIC_TREE_APPLY_RESULT_SCHEMA_VERSION:
        raise AutomaticTreeApplyDataError(
            "result_payload.schema_version is unsupported"
        )
    if normalized["producer_version"] != AUTOMATIC_TREE_APPLY_PRODUCER_VERSION:
        raise AutomaticTreeApplyDataError(
            "result_payload.producer_version is unsupported"
        )

    source = _object(normalized["source"], field_name="result_payload.source")
    _exact_fields(
        source,
        frozenset({"content_hash", "row_count"}),
        field_name="result_payload.source",
    )
    source_hash = _sha256(
        source["content_hash"],
        field_name="result_payload.source.content_hash",
    )
    source_rows = _non_negative_int(
        source["row_count"],
        field_name="result_payload.source.row_count",
    )
    if not hmac.compare_digest(source_hash, identity.source_dataset_hash):
        raise AutomaticTreeApplyDataError(
            "result_payload source hash does not match the run identity"
        )

    tree = _object(normalized["tree"], field_name="result_payload.tree")
    _exact_fields(
        tree,
        frozenset({"result_hash", "asset_id", "asset_hash"}),
        field_name="result_payload.tree",
    )
    tree_result_hash = _sha256(
        tree["result_hash"],
        field_name="result_payload.tree.result_hash",
    )
    tree_asset_id = _asset_id(tree["asset_id"])
    tree_asset_hash = _sha256(
        tree["asset_hash"],
        field_name="result_payload.tree.asset_hash",
    )
    if (
        not hmac.compare_digest(tree_result_hash, identity.tree_result_hash)
        or tree_asset_id != identity.asset_id
        or not hmac.compare_digest(tree_asset_hash, identity.asset_hash)
    ):
        raise AutomaticTreeApplyDataError(
            "result_payload tree identity does not match the run identity"
        )

    output = _object(normalized["output"], field_name="result_payload.output")
    _exact_fields(
        output,
        frozenset(
            {
                "content_hash",
                "row_count",
                "schema",
                "columns",
                "leaf_distribution",
            }
        ),
        field_name="result_payload.output",
    )
    output_hash = _sha256(
        output["content_hash"],
        field_name="result_payload.output.content_hash",
    )
    output_rows = _non_negative_int(
        output["row_count"],
        field_name="result_payload.output.row_count",
    )
    if not hmac.compare_digest(output_hash, committed.result_dataset_hash):
        raise AutomaticTreeApplyDataError(
            "result_payload output hash does not match the result dataset"
        )
    if output_rows != source_rows:
        raise AutomaticTreeApplyDataError(
            "result_payload output row count must equal source row count"
        )
    columns = _object(
        output["columns"],
        field_name="result_payload.output.columns",
    )
    _exact_fields(
        columns,
        frozenset({"leaf_id", "rule_id"}),
        field_name="result_payload.output.columns",
    )
    output_leaf = _output_column(
        columns["leaf_id"],
        field_name="result_payload.output.columns.leaf_id",
    )
    output_rule = _output_column(
        columns["rule_id"],
        field_name="result_payload.output.columns.rule_id",
    )
    if (
        output_leaf != identity.output_leaf_column
        or output_rule != identity.output_rule_column
    ):
        raise AutomaticTreeApplyDataError(
            "result_payload output columns do not match the run identity"
        )
    _validate_result_schema(
        output["schema"],
        output_leaf_column=output_leaf,
        output_rule_column=output_rule,
    )
    _validate_leaf_distribution(
        output["leaf_distribution"],
        expected_rows=source_rows,
    )

    writer = _object(normalized["writer"], field_name="result_payload.writer")
    _validate_writer_contract(writer, identity=identity)

    computation_hash = _sha256(
        normalized["result_hash"],
        field_name="result_payload.result_hash",
    )
    evidence_body = {
        key: normalized[key]
        for key in (
            "schema_version",
            "producer_version",
            "source",
            "tree",
            "output",
            "writer",
        )
    }
    expected_computation_hash = _digest(
        _canonical_json(evidence_body, field_name="result_payload evidence")
    )
    if not hmac.compare_digest(computation_hash, expected_computation_hash):
        raise AutomaticTreeApplyDataError(
            "result_payload.result_hash does not authenticate the evidence"
        )
    result_id = normalized["result_id"]
    if (
        not isinstance(result_id, str)
        or _RESULT_ID_RE.fullmatch(result_id) is None
        or result_id != f"automatic-tree-apply-{computation_hash[:32]}"
    ):
        raise AutomaticTreeApplyDataError(
            "result_payload.result_id does not match result_hash"
        )
    return normalized


def _validate_result_schema(
    value: object,
    *,
    output_leaf_column: str,
    output_rule_column: str,
) -> None:
    schema = _object(value, field_name="result_payload.output.schema")
    _exact_fields(
        schema,
        frozenset({"fields", "metadata_hash"}),
        field_name="result_payload.output.schema",
    )
    _optional_sha256(
        schema["metadata_hash"],
        field_name="result_payload.output.schema.metadata_hash",
    )
    fields = schema["fields"]
    if not isinstance(fields, list) or len(fields) < 2:
        raise AutomaticTreeApplyDataError(
            "result_payload.output.schema.fields must contain output fields"
        )
    normalized_fields: list[dict[str, Any]] = []
    for index, item in enumerate(fields):
        field_name = f"result_payload.output.schema.fields[{index}]"
        field_value = _object(item, field_name=field_name)
        _exact_fields(
            field_value,
            frozenset({"name", "physical_type", "nullable", "metadata_hash"}),
            field_name=field_name,
        )
        normalized_fields.append(
            {
                "name": _canonical_text(
                    field_value["name"],
                    field_name=f"{field_name}.name",
                ),
                "physical_type": _canonical_text(
                    field_value["physical_type"],
                    field_name=f"{field_name}.physical_type",
                ),
                "nullable": _boolean(
                    field_value["nullable"],
                    field_name=f"{field_name}.nullable",
                ),
                "metadata_hash": _optional_sha256(
                    field_value["metadata_hash"],
                    field_name=f"{field_name}.metadata_hash",
                ),
            }
        )
    leaf_field, rule_field = normalized_fields[-2:]
    expected = (
        (leaf_field, output_leaf_column),
        (rule_field, output_rule_column),
    )
    for field_value, name in expected:
        if (
            field_value["name"] != name
            or field_value["physical_type"] != "string"
            or field_value["nullable"] is not False
        ):
            raise AutomaticTreeApplyDataError(
                "result_payload output schema does not bind the appended columns"
            )


def _validate_leaf_distribution(value: object, *, expected_rows: int) -> None:
    if not isinstance(value, list) or not value:
        raise AutomaticTreeApplyDataError(
            "result_payload.output.leaf_distribution must be a non-empty array"
        )
    total = 0
    leaf_ids: set[str] = set()
    rule_ids: set[str] = set()
    for index, item in enumerate(value):
        field_name = f"result_payload.output.leaf_distribution[{index}]"
        distribution = _object(item, field_name=field_name)
        _exact_fields(
            distribution,
            frozenset({"leaf_id", "rule_id", "row_count"}),
            field_name=field_name,
        )
        leaf_id = _canonical_text(
            distribution["leaf_id"],
            field_name=f"{field_name}.leaf_id",
        )
        rule_id = _canonical_text(
            distribution["rule_id"],
            field_name=f"{field_name}.rule_id",
        )
        if leaf_id in leaf_ids or rule_id in rule_ids:
            raise AutomaticTreeApplyDataError(
                "result_payload leaf distribution ids must be unique"
            )
        leaf_ids.add(leaf_id)
        rule_ids.add(rule_id)
        total += _non_negative_int(
            distribution["row_count"],
            field_name=f"{field_name}.row_count",
        )
    if total != expected_rows:
        raise AutomaticTreeApplyDataError(
            "result_payload leaf distribution must conserve source rows"
        )


def _validate_writer_contract(
    writer: Mapping[str, Any],
    *,
    identity: AutomaticTreeApplyIdentity,
) -> None:
    _exact_fields(
        writer,
        _WRITER_FIELDS,
        field_name="result_payload.writer",
    )
    expected_values: dict[str, object] = {
        "contract": identity.writer_contract,
        "engine": "pyarrow.parquet",
        "engine_version": identity.writer_version,
        "threads": 1,
        "preserve_insertion_order": True,
        "batch_rows": 8_192,
        "max_decoded_batch_bytes": 256 * 1024 * 1024,
        "row_group_rows": 8_192,
        "write_batch_rows": 1_024,
        "parquet_version": "2.6",
        "data_page_version": "1.0",
        "compression": "zstd",
        "compression_level": 3,
        "dictionary_encoding": True,
        "dictionary_page_bytes": 1_048_576,
        "write_statistics": True,
        "byte_stream_split": False,
        "use_deprecated_int96_timestamps": False,
        "use_compliant_nested_type": True,
        "store_arrow_schema": True,
        "write_page_index": False,
        "write_page_checksum": False,
        "source_schema_metadata": "preserved",
        "appended_id_type": "utf8_non_null",
    }
    for field_name, expected in expected_values.items():
        actual = writer[field_name]
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and actual == expected
            )
        else:
            matches = isinstance(actual, str) and actual == expected
        if not matches:
            raise AutomaticTreeApplyDataError(
                f"result_payload.writer.{field_name} violates the writer contract"
            )


def _select_by_input(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    input_hash: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM strategy_automatic_tree_apply_runs
         WHERE task_id = ? AND input_hash = ?
        """,
        (task_id, input_hash),
    ).fetchone()


def _validated_record(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> AutomaticTreeApplyRecord:
    record = _record_from_row(row)
    _require_live_bindings(
        conn,
        identity=record.identity,
        committed=record.committed,
        result_payload=record.result_payload,
    )
    return record


def _record_from_row(row: sqlite3.Row) -> AutomaticTreeApplyRecord:
    try:
        if str(row["schema_version"]) != AUTOMATIC_TREE_APPLY_RUN_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        identity = AutomaticTreeApplyIdentity(
            task_id=row["task_id"],
            source_tree_artifact_id=row["source_tree_artifact_id"],
            source_tree_artifact_hash=row["source_tree_artifact_hash"],
            asset_id=row["asset_id"],
            asset_hash=row["asset_hash"],
            tree_result_hash=row["tree_result_hash"],
            source_dataset_id=row["source_dataset_id"],
            source_dataset_hash=row["source_dataset_hash"],
            output_leaf_column=row["output_leaf_column"],
            output_rule_column=row["output_rule_column"],
            writer_contract=row["writer_contract"],
            writer_version=row["writer_version"],
        )
        if str(row["id"]) != identity.run_id:
            raise ValueError("run id does not match canonical identity")
        if not hmac.compare_digest(str(row["input_hash"]), identity.input_hash):
            raise ValueError("input hash does not match canonical identity")
        committed = AutomaticTreeApplyCommittedFacts(
            result_dataset_id=row["result_dataset_id"],
            result_dataset_hash=row["result_dataset_hash"],
            result_dataset_path=row["result_dataset_path"],
            evidence_artifact_id=row["evidence_artifact_id"],
            evidence_artifact_hash=row["evidence_artifact_hash"],
            evidence_artifact_path=row["evidence_artifact_path"],
        )
        if committed.result_dataset_id == identity.source_dataset_id:
            raise ValueError("source/result datasets must differ")
        if committed.evidence_artifact_id == identity.source_tree_artifact_id:
            raise ValueError("source/evidence artifacts must differ")
        raw_result_json = str(row["result_json"])
        parsed_result = json.loads(raw_result_json)
        if not isinstance(parsed_result, dict):
            raise ValueError("result_json must be an object")
        canonical_result_json = _canonical_json(
            parsed_result,
            field_name="result_json",
        )
        expected_result_json = canonical_automatic_tree_apply_result_json(
            parsed_result,
            identity=identity,
            committed=committed,
        )
        if raw_result_json != canonical_result_json:
            raise ValueError("result_json is not canonical")
        if raw_result_json != expected_result_json:
            raise ValueError("result_json does not exactly match committed facts")
        result_hash = _sha256(row["result_hash"], field_name="result_hash")
        if not hmac.compare_digest(result_hash, _digest(raw_result_json)):
            raise ValueError("result_hash does not authenticate result_json")
        return AutomaticTreeApplyRecord(
            id=identity.run_id,
            schema_version=AUTOMATIC_TREE_APPLY_RUN_SCHEMA_VERSION,
            identity=identity,
            committed=committed,
            result_json=raw_result_json,
            result_hash=result_hash,
            created_at=_timestamp(row["created_at"]),
        )
    except (
        AutomaticTreeApplyDataError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise AutomaticTreeApplyDataError("corrupt automatic-tree apply run") from exc


def _require_exact_replay(
    record: AutomaticTreeApplyRecord,
    *,
    identity: AutomaticTreeApplyIdentity,
    committed: AutomaticTreeApplyCommittedFacts,
    result_json: str,
    result_hash: str,
) -> None:
    _require_same_identity(record.identity, identity)
    if (
        record.committed != committed
        or record.result_json != result_json
        or not hmac.compare_digest(record.result_hash, result_hash)
    ):
        raise AutomaticTreeApplyConflictError(
            "automatic-tree apply input replay has different committed facts"
        )


def _require_same_identity(
    persisted: AutomaticTreeApplyIdentity,
    requested: AutomaticTreeApplyIdentity,
) -> None:
    if persisted != requested:
        raise AutomaticTreeApplyConflictError(
            "input_hash maps to a different automatic-tree apply identity"
        )


def _require_live_bindings(
    conn: sqlite3.Connection,
    *,
    identity: AutomaticTreeApplyIdentity,
    committed: AutomaticTreeApplyCommittedFacts,
    result_payload: Mapping[str, Any],
) -> None:
    task = conn.execute(
        "SELECT 1 FROM tasks WHERE id = ?",
        (identity.task_id,),
    ).fetchone()
    if task is None:
        raise AutomaticTreeApplyNotFoundError("task-owned apply inputs not found")
    source_evidence = _object(
        result_payload.get("source"),
        field_name="result_payload.source",
    )
    output_evidence = _object(
        result_payload.get("output"),
        field_name="result_payload.output",
    )
    _require_dataset(
        conn,
        task_id=identity.task_id,
        dataset_id=identity.source_dataset_id,
        content_hash=identity.source_dataset_hash,
        path=None,
        row_count=_non_negative_int(
            source_evidence.get("row_count"),
            field_name="result_payload.source.row_count",
        ),
        label="source dataset",
    )
    _require_artifact(
        conn,
        task_id=identity.task_id,
        artifact_id=identity.source_tree_artifact_id,
        content_hash=identity.source_tree_artifact_hash,
        path=None,
        label="source tree artifact",
        expected_kind=AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        expected_origin=AUTOMATIC_TREE_SOURCE_ORIGIN_TOOL,
        expected_provenance={
            "task_id": identity.task_id,
            "asset_id": identity.asset_id,
            "asset_hash": identity.asset_hash,
            "tree_result_hash": identity.tree_result_hash,
        },
    )
    _require_dataset(
        conn,
        task_id=identity.task_id,
        dataset_id=committed.result_dataset_id,
        content_hash=committed.result_dataset_hash,
        path=committed.result_dataset_path,
        row_count=_non_negative_int(
            output_evidence.get("row_count"),
            field_name="result_payload.output.row_count",
        ),
        label="result dataset",
    )
    _require_artifact(
        conn,
        task_id=identity.task_id,
        artifact_id=committed.evidence_artifact_id,
        content_hash=committed.evidence_artifact_hash,
        path=committed.evidence_artifact_path,
        label="evidence artifact",
        expected_kind=AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND,
        expected_origin=AUTOMATIC_TREE_APPLY_ORIGIN_TOOL,
        expected_provenance={
            "run_id": identity.run_id,
            "input_hash": identity.input_hash,
        },
    )


def _require_dataset(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    dataset_id: str,
    content_hash: str,
    path: str | None,
    row_count: int,
    label: str,
) -> None:
    row = conn.execute(
        """
        SELECT content_hash, source_path, row_count FROM datasets
         WHERE id = ? AND task_id = ?
        """,
        (dataset_id, task_id),
    ).fetchone()
    if row is None:
        raise AutomaticTreeApplyNotFoundError(f"{label} not found for task")
    registered_hash = row["content_hash"]
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        content_hash,
    ):
        raise AutomaticTreeApplyConflictError(f"{label} content hash drifted")
    if path is not None and str(row["source_path"]) != path:
        raise AutomaticTreeApplyConflictError(f"{label} path drifted")
    if (
        isinstance(row["row_count"], bool)
        or not isinstance(row["row_count"], int)
        or int(row["row_count"]) != row_count
    ):
        raise AutomaticTreeApplyConflictError(f"{label} row count drifted")


def _require_artifact(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    artifact_id: str,
    content_hash: str,
    path: str | None,
    label: str,
    expected_kind: str,
    expected_origin: str,
    expected_provenance: Mapping[str, str],
) -> None:
    row = conn.execute(
        """
        SELECT kind, path, content_hash, origin_tool, provenance_json
          FROM task_artifacts
         WHERE id = ? AND task_id = ?
        """,
        (artifact_id, task_id),
    ).fetchone()
    if row is None:
        raise AutomaticTreeApplyNotFoundError(f"{label} not found for task")
    registered_hash = row["content_hash"]
    if not isinstance(registered_hash, str) or not hmac.compare_digest(
        registered_hash,
        content_hash,
    ):
        raise AutomaticTreeApplyConflictError(f"{label} content hash drifted")
    if path is not None and str(row["path"]) != path:
        raise AutomaticTreeApplyConflictError(f"{label} path drifted")

    raw_provenance = row["provenance_json"]
    if not isinstance(raw_provenance, str):
        raise AutomaticTreeApplyDataError(f"{label} provenance is corrupt")
    try:
        provenance = json.loads(raw_provenance)
    except json.JSONDecodeError as exc:
        raise AutomaticTreeApplyDataError(f"{label} provenance is corrupt") from exc
    if (
        not isinstance(provenance, dict)
        or _canonical_json(provenance, field_name=f"{label} provenance")
        != raw_provenance
    ):
        raise AutomaticTreeApplyDataError(f"{label} provenance is corrupt")
    if (
        str(row["kind"]) != expected_kind
        or str(row["origin_tool"]) != expected_origin
        or any(
            provenance.get(key) != value for key, value in expected_provenance.items()
        )
    ):
        raise AutomaticTreeApplyConflictError(
            f"{label} does not bind the requested apply run"
        )


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field_name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise AutomaticTreeApplyDataError(
            f"{field_name} must contain exact fields: " + "; ".join(detail)
        )


def _object(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutomaticTreeApplyDataError(f"{field_name} must be an object")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutomaticTreeApplyDataError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise AutomaticTreeApplyDataError(f"{field_name} must be a boolean")
    return value


def _optional_sha256(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field_name=field_name)


def _identity(value: object) -> AutomaticTreeApplyIdentity:
    if not isinstance(value, AutomaticTreeApplyIdentity):
        raise AutomaticTreeApplyDataError("identity must be AutomaticTreeApplyIdentity")
    return value


def _committed(value: object) -> AutomaticTreeApplyCommittedFacts:
    if not isinstance(value, AutomaticTreeApplyCommittedFacts):
        raise AutomaticTreeApplyDataError(
            "committed must be AutomaticTreeApplyCommittedFacts"
        )
    return value


def _canonical_json(value: object, *, field_name: str) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AutomaticTreeApplyDataError(
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
        raise AutomaticTreeApplyDataError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AutomaticTreeApplyDataError(
            f"{field_name} must be a 64-character lowercase SHA-256 hex"
        )
    return value


def _asset_id(value: object) -> str:
    if not isinstance(value, str) or _ASSET_ID_RE.fullmatch(value) is None:
        raise AutomaticTreeApplyDataError(
            "asset_id must be a canonical automatic-tree asset id"
        )
    return value


def _output_column(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _OUTPUT_COLUMN_RE.fullmatch(value) is None:
        raise AutomaticTreeApplyDataError(
            f"{field_name} must be a safe ASCII identifier of at most 64 characters"
        )
    return value


def _timestamp(value: object) -> str:
    text = _canonical_text(value, field_name="created_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AutomaticTreeApplyDataError(
            "created_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise AutomaticTreeApplyDataError("created_at must include timezone")
    return text


def _stable_id(*, task_id: str, input_hash: str) -> str:
    payload = "\x00".join((_RUN_ID_NAMESPACE, task_id, input_hash))
    return "atar_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "AUTOMATIC_TREE_APPLY_EVIDENCE_ARTIFACT_KIND",
    "AUTOMATIC_TREE_APPLY_INPUT_SCHEMA_VERSION",
    "AUTOMATIC_TREE_APPLY_ORIGIN_TOOL",
    "AUTOMATIC_TREE_APPLY_PRODUCER_VERSION",
    "AUTOMATIC_TREE_APPLY_RESULT_SCHEMA_VERSION",
    "AUTOMATIC_TREE_APPLY_RUN_SCHEMA_VERSION",
    "AutomaticTreeApplyCommittedFacts",
    "AutomaticTreeApplyConflictError",
    "AutomaticTreeApplyDataError",
    "AutomaticTreeApplyIdentity",
    "AutomaticTreeApplyNotFoundError",
    "AutomaticTreeApplyRecord",
    "AutomaticTreeApplyRepository",
    "canonical_automatic_tree_apply_result_json",
]
