"""Evidence-bound persistence for deterministic, task-scoped data analysis.

The repository stores execution state, not analytical values.  Results live in
the immutable :mod:`marvis.repositories.task_artifacts` registry and a run may
be marked successful only on the same SQLite connection that registered the
artifact.  Computational identity deliberately excludes ``workspace_revision``:
page/selection-only workspace saves advance that revision but do not invalidate
analysis.  The originating revision remains immutable provenance, while dataset
hash, analysis generation, semantic mapping, normalized config, and producer
version determine reuse.
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
from types import MappingProxyType
from typing import Any

from marvis.data.workspace import (
    data_semantic_mapping_from_dict,
    data_semantic_mapping_hash,
    data_semantic_mapping_to_dict,
)
from marvis.db_schema import connect


DATA_ANALYSIS_SCHEMA_VERSION = "data-analysis.v1"
DATA_ANALYSIS_ARTIFACT_KIND = "data_analysis"
DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL = "data.analysis_service"
DATA_ANALYSIS_JOB_KIND = "data_analysis"
DATA_ANALYSIS_STATUSES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
)

_STATUS_SET = frozenset(DATA_ANALYSIS_STATUSES)
_TERMINAL_RETRYABLE = frozenset({"failed", "cancelled"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_NAMESPACE = "marvis.data-analysis-run.v1"
_QUEUED_JOB_STATUSES = frozenset({"queued"})
_RUNNING_JOB_STATUSES = frozenset({"running"})


class DataAnalysisDataError(ValueError):
    """Supplied or persisted analysis evidence violates the contract."""


class DataAnalysisNotFoundError(KeyError):
    """A task-owned run, dataset, job, or artifact was not found."""


class DataAnalysisConflictError(RuntimeError):
    """An idempotent identity replay conflicts with persisted state."""


class DataAnalysisTransitionError(DataAnalysisConflictError):
    """A compare-and-swap execution-state transition is no longer valid."""


class DataAnalysisStaleIdentityError(DataAnalysisConflictError):
    """The request was built from a workspace snapshot that is no longer current."""


@dataclass(frozen=True)
class DataAnalysisIdentity:
    """Canonical computational identity plus its originating workspace revision."""

    task_id: str
    dataset_id: str
    dataset_content_hash: str
    workspace_revision: int
    analysis_generation: int
    semantic_mapping_hash: str
    config: Mapping[str, Any]
    producer_version: str
    schema_version: str = DATA_ANALYSIS_SCHEMA_VERSION
    config_json: str = field(init=False)
    config_hash: str = field(init=False)
    input_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != DATA_ANALYSIS_SCHEMA_VERSION:
            raise DataAnalysisDataError(
                f"schema_version must be {DATA_ANALYSIS_SCHEMA_VERSION}"
            )
        task_id = _required_text(self.task_id, field_name="task_id")
        dataset_id = _required_text(self.dataset_id, field_name="dataset_id")
        dataset_hash = _sha256(
            self.dataset_content_hash,
            field_name="dataset_content_hash",
        )
        revision = _non_negative_int(
            self.workspace_revision,
            field_name="workspace_revision",
        )
        generation = _non_negative_int(
            self.analysis_generation,
            field_name="analysis_generation",
        )
        mapping_hash = _sha256(
            self.semantic_mapping_hash,
            field_name="semantic_mapping_hash",
        )
        producer = _required_text(
            self.producer_version,
            field_name="producer_version",
        )
        normalized_config, config_json = _canonical_config(self.config)
        config_hash = _digest(config_json)
        input_hash = _compute_input_hash(
            schema_version=self.schema_version,
            task_id=task_id,
            dataset_id=dataset_id,
            dataset_content_hash=dataset_hash,
            analysis_generation=generation,
            semantic_mapping_hash=mapping_hash,
            config_json=config_json,
            config_hash=config_hash,
            producer_version=producer,
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_content_hash", dataset_hash)
        object.__setattr__(self, "workspace_revision", revision)
        object.__setattr__(self, "analysis_generation", generation)
        object.__setattr__(self, "semantic_mapping_hash", mapping_hash)
        object.__setattr__(self, "config", MappingProxyType(normalized_config))
        object.__setattr__(self, "producer_version", producer)
        object.__setattr__(self, "config_json", config_json)
        object.__setattr__(self, "config_hash", config_hash)
        object.__setattr__(self, "input_hash", input_hash)


@dataclass(frozen=True)
class DataAnalysisRunRecord:
    id: str
    identity: DataAnalysisIdentity
    job_id: str | None
    status: str
    result_artifact_id: str | None
    result_content_hash: str | None
    error_kind: str | None
    error_message: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None

    @property
    def schema_version(self) -> str:
        return self.identity.schema_version

    @property
    def task_id(self) -> str:
        return self.identity.task_id

    @property
    def dataset_id(self) -> str:
        return self.identity.dataset_id

    @property
    def dataset_content_hash(self) -> str:
        return self.identity.dataset_content_hash

    @property
    def workspace_revision(self) -> int:
        return self.identity.workspace_revision

    @property
    def analysis_generation(self) -> int:
        return self.identity.analysis_generation

    @property
    def semantic_mapping_hash(self) -> str:
        return self.identity.semantic_mapping_hash

    @property
    def config(self) -> Mapping[str, Any]:
        return self.identity.config

    @property
    def config_json(self) -> str:
        return self.identity.config_json

    @property
    def config_hash(self) -> str:
        return self.identity.config_hash

    @property
    def producer_version(self) -> str:
        return self.identity.producer_version

    @property
    def input_hash(self) -> str:
        return self.identity.input_hash

    @property
    def result_artifact_content_hash(self) -> str | None:
        """Unambiguous alias used by artifact-oriented consumers."""

        return self.result_content_hash


# The longer alias reads naturally at call sites that distinguish identities
# from persisted records.  Keep both names as one contract, not two runtimes.
DataAnalysisRunIdentity = DataAnalysisIdentity
DataAnalysisRun = DataAnalysisRunRecord


class DataAnalysisRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def transaction(self):
        """Return a configured connection for a caller-owned unit of work."""

        return connect(self.db_path)

    def create_or_get(
        self,
        identity: DataAnalysisIdentity,
        job_id: str | None = None,
        *,
        retry: bool = False,
    ) -> DataAnalysisRunRecord:
        """Create a queued run or return its exact computational replay.

        ``retry=True`` is explicit and only resets a failed/cancelled run.  A
        successful result is immutable; callers change config or producer
        version when they intentionally need a distinct computation.
        """

        normalized_identity = _identity(identity)
        normalized_job_id = _optional_text(job_id, field_name="job_id")
        if not isinstance(retry, bool):
            raise DataAnalysisDataError("retry must be a boolean")

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _validate_live_identity(conn, normalized_identity)
            _validate_job(
                conn,
                normalized_identity.task_id,
                normalized_job_id,
                allowed_statuses=_QUEUED_JOB_STATUSES,
                action="create or retry",
            )
            row = _select_by_input_hash(
                conn,
                task_id=normalized_identity.task_id,
                input_hash=normalized_identity.input_hash,
            )
            if row is not None:
                record = _validated_record(conn, row)
                _require_same_computation(record.identity, normalized_identity)
                if retry:
                    return _retry_record(
                        conn,
                        record,
                        job_id=normalized_job_id,
                    )
                _require_same_job(record, normalized_job_id)
                return record
            if retry:
                raise DataAnalysisNotFoundError(
                    "data analysis run not found for retry"
                )

            timestamp = _now()
            run_id = _stable_run_id(normalized_identity)
            collision = conn.execute(
                "SELECT * FROM data_analysis_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if collision is not None:
                raise DataAnalysisConflictError(
                    "stable data analysis run id collided with another identity"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO data_analysis_runs(
                        id, schema_version, task_id, dataset_id,
                        dataset_content_hash, workspace_revision,
                        analysis_generation, semantic_mapping_hash,
                        config_json, config_hash, producer_version, input_hash,
                        job_id, status, result_artifact_id, result_content_hash,
                        error_kind, error_message, created_at, updated_at,
                        started_at, completed_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued',
                        NULL, NULL, NULL, NULL, ?, ?, NULL, NULL
                    )
                    """,
                    (
                        run_id,
                        normalized_identity.schema_version,
                        normalized_identity.task_id,
                        normalized_identity.dataset_id,
                        normalized_identity.dataset_content_hash,
                        normalized_identity.workspace_revision,
                        normalized_identity.analysis_generation,
                        normalized_identity.semantic_mapping_hash,
                        normalized_identity.config_json,
                        normalized_identity.config_hash,
                        normalized_identity.producer_version,
                        normalized_identity.input_hash,
                        normalized_job_id,
                        timestamp,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                # BEGIN IMMEDIATE normally serializes this path.  Preserve an
                # explicit typed boundary if a caller changes transaction mode
                # or a future SQLite adapter surfaces a uniqueness race here.
                replay = _select_by_input_hash(
                    conn,
                    task_id=normalized_identity.task_id,
                    input_hash=normalized_identity.input_hash,
                )
                if replay is None:
                    raise DataAnalysisConflictError(
                        "could not create data analysis run"
                    ) from exc
                record = _validated_record(conn, replay)
                _require_same_computation(record.identity, normalized_identity)
                _require_same_job(record, normalized_job_id)
                return record

            row = conn.execute(
                "SELECT * FROM data_analysis_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            assert row is not None
            return _validated_record(conn, row)

    def retry(
        self,
        identity: DataAnalysisIdentity,
        job_id: str | None = None,
    ) -> DataAnalysisRunRecord:
        """Explicitly reset the same failed/cancelled identity to ``queued``."""

        return self.create_or_get(identity, job_id=job_id, retry=True)

    def get_for_task(
        self,
        task_id: str,
        run_id: str,
    ) -> DataAnalysisRunRecord | None:
        normalized_task_id = _required_text(task_id, field_name="task_id")
        normalized_run_id = _required_text(run_id, field_name="run_id")
        with connect(self.db_path) as conn:
            row = _select_for_task(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )
            return None if row is None else _validated_record(conn, row)

    def get_by_input_hash(
        self,
        task_id: str | DataAnalysisIdentity,
        input_hash: str | None = None,
    ) -> DataAnalysisRunRecord | None:
        """Read by unique compute hash, optionally asserting the full identity."""

        expected_identity: DataAnalysisIdentity | None = None
        if isinstance(task_id, DataAnalysisIdentity):
            if input_hash is not None:
                raise DataAnalysisDataError(
                    "input_hash must be omitted when identity is supplied"
                )
            expected_identity = _identity(task_id)
            normalized_task_id = expected_identity.task_id
            normalized_input_hash = expected_identity.input_hash
        else:
            normalized_task_id = _required_text(task_id, field_name="task_id")
            normalized_input_hash = _sha256(input_hash, field_name="input_hash")

        with connect(self.db_path) as conn:
            row = _select_by_input_hash(
                conn,
                task_id=normalized_task_id,
                input_hash=normalized_input_hash,
            )
            if row is None:
                return None
            record = _validated_record(conn, row)
            if expected_identity is not None:
                _require_same_computation(record.identity, expected_identity)
            return record

    def current(
        self,
        identity: DataAnalysisIdentity,
    ) -> DataAnalysisRunRecord | None:
        """Return the run matching the current live computational identity."""

        normalized_identity = _identity(identity)
        with connect(self.db_path) as conn:
            _validate_live_identity(conn, normalized_identity)
            row = _select_by_input_hash(
                conn,
                task_id=normalized_identity.task_id,
                input_hash=normalized_identity.input_hash,
            )
            if row is None:
                return None
            record = _validated_record(conn, row)
            _require_same_computation(record.identity, normalized_identity)
            return record

    # Compatibility-friendly spelling for consumers that prefer an explicit
    # getter name.  It delegates to the same full-identity check.
    get_current = current

    def list_for_task(self, task_id: str) -> list[DataAnalysisRunRecord]:
        normalized_task_id = _required_text(task_id, field_name="task_id")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM data_analysis_runs
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (normalized_task_id,),
            ).fetchall()
            return [_validated_record(conn, row) for row in rows]

    def attach_job(
        self,
        *,
        task_id: str,
        run_id: str,
        job_id: str,
    ) -> DataAnalysisRunRecord:
        """Attach one task-owned job without claiming the queued run yet."""

        normalized_task_id = _required_text(task_id, field_name="task_id")
        normalized_run_id = _required_text(run_id, field_name="run_id")
        normalized_job_id = _required_text(job_id, field_name="job_id")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _validate_job(
                conn,
                normalized_task_id,
                normalized_job_id,
                allowed_statuses=_QUEUED_JOB_STATUSES,
                action="attach",
            )
            record = _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )
            if record.status != "queued":
                raise DataAnalysisTransitionError(
                    "data analysis run must be queued before a job can attach"
                )
            if record.job_id == normalized_job_id:
                return record
            if record.job_id is not None:
                raise DataAnalysisConflictError(
                    "data analysis run is already bound to a different job_id"
                )
            cursor = conn.execute(
                """
                UPDATE data_analysis_runs
                   SET job_id = ?, updated_at = ?
                 WHERE task_id = ? AND id = ?
                   AND status = 'queued' AND job_id IS NULL
                """,
                (
                    normalized_job_id,
                    _now(),
                    normalized_task_id,
                    normalized_run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DataAnalysisTransitionError(
                    "data analysis run is no longer an unattached queued run"
                )
            return _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )

    def mark_running(
        self,
        *,
        task_id: str,
        run_id: str,
        job_id: str | None = None,
    ) -> DataAnalysisRunRecord:
        """CAS ``queued`` to ``running`` and optionally bind its task job."""

        normalized_task_id = _required_text(task_id, field_name="task_id")
        normalized_run_id = _required_text(run_id, field_name="run_id")
        normalized_job_id = _optional_text(job_id, field_name="job_id")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )
            if record.status != "queued":
                raise DataAnalysisTransitionError(
                    "data analysis run must be queued before it can start"
                )
            if (
                normalized_job_id is not None
                and record.job_id is not None
                and record.job_id != normalized_job_id
            ):
                raise DataAnalysisConflictError(
                    "data analysis run is already bound to a different job_id"
                )
            resolved_job_id = normalized_job_id or record.job_id
            _validate_job(
                conn,
                normalized_task_id,
                resolved_job_id,
                allowed_statuses=_RUNNING_JOB_STATUSES,
                action="start",
            )
            timestamp = _now()
            cursor = conn.execute(
                """
                UPDATE data_analysis_runs
                   SET job_id = ?, status = 'running', started_at = ?, updated_at = ?
                 WHERE task_id = ? AND id = ? AND status = 'queued'
                """,
                (
                    resolved_job_id,
                    timestamp,
                    timestamp,
                    normalized_task_id,
                    normalized_run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise DataAnalysisTransitionError(
                    "data analysis run is no longer queued"
                )
            return _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )

    def complete_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        run_id: str,
        result_artifact_id: str,
        result_content_hash: str,
    ) -> DataAnalysisRunRecord:
        """CAS ``running`` to ``succeeded`` beside artifact registration.

        The caller owns commit/rollback.  The artifact id is the immutable
        registry row id, never a filesystem path.
        """

        normalized_task_id = _required_text(task_id, field_name="task_id")
        normalized_run_id = _required_text(run_id, field_name="run_id")
        normalized_artifact_id = _required_text(
            result_artifact_id,
            field_name="result_artifact_id",
        )
        normalized_result_hash = _sha256(
            result_content_hash,
            field_name="result_content_hash",
        )
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        record = _required_run(
            conn,
            task_id=normalized_task_id,
            run_id=normalized_run_id,
        )
        if record.status != "running":
            raise DataAnalysisTransitionError(
                "data analysis run must be running before completion"
            )
        _validate_job(
            conn,
            normalized_task_id,
            record.job_id,
            allowed_statuses=_RUNNING_JOB_STATUSES,
            action="complete",
        )

        artifact = conn.execute(
            """
            SELECT id, task_id, kind, content_hash, origin_tool, provenance_json
              FROM task_artifacts
             WHERE task_id = ? AND id = ?
            """,
            (normalized_task_id, normalized_artifact_id),
        ).fetchone()
        if artifact is None:
            raise DataAnalysisNotFoundError("artifact not found for task")
        _validate_analysis_artifact(
            artifact,
            record=record,
            expected_content_hash=normalized_result_hash,
        )

        timestamp = _now()
        cursor = conn.execute(
            """
            UPDATE data_analysis_runs
               SET status = 'succeeded', result_artifact_id = ?,
                   result_content_hash = ?, completed_at = ?, updated_at = ?
             WHERE task_id = ? AND id = ? AND status = 'running'
            """,
            (
                normalized_artifact_id,
                normalized_result_hash,
                timestamp,
                timestamp,
                normalized_task_id,
                normalized_run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise DataAnalysisTransitionError(
                "data analysis run is no longer running"
            )
        return _required_run(
            conn,
            task_id=normalized_task_id,
            run_id=normalized_run_id,
        )

    def fail(
        self,
        *,
        task_id: str,
        run_id: str,
        error_kind: str,
        error_message: str,
        expected_job_id: str | None = None,
    ) -> DataAnalysisRunRecord:
        return self._finish_without_result(
            task_id=task_id,
            run_id=run_id,
            status="failed",
            error_kind=error_kind,
            error_message=error_message,
            expected_job_id=expected_job_id,
        )

    def cancel(
        self,
        *,
        task_id: str,
        run_id: str,
        error_kind: str = "cancelled",
        error_message: str = "cancelled",
    ) -> DataAnalysisRunRecord:
        return self._finish_without_result(
            task_id=task_id,
            run_id=run_id,
            status="cancelled",
            error_kind=error_kind,
            error_message=error_message,
            expected_job_id=None,
        )

    def _finish_without_result(
        self,
        *,
        task_id: str,
        run_id: str,
        status: str,
        error_kind: str,
        error_message: str,
        expected_job_id: str | None,
    ) -> DataAnalysisRunRecord:
        normalized_task_id = _required_text(task_id, field_name="task_id")
        normalized_run_id = _required_text(run_id, field_name="run_id")
        normalized_error_kind = _required_text(
            error_kind,
            field_name="error_kind",
        )
        normalized_error_message = _required_text(
            error_message,
            field_name="error_message",
        )
        normalized_expected_job_id = _optional_text(
            expected_job_id,
            field_name="expected_job_id",
        )
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            record = _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )
            if record.status not in {"queued", "running"}:
                raise DataAnalysisTransitionError(
                    f"data analysis run must be queued or running before {status}"
                )
            if (
                normalized_expected_job_id is not None
                and record.job_id != normalized_expected_job_id
            ):
                raise DataAnalysisTransitionError(
                    "data analysis run is no longer bound to the expected job"
                )
            timestamp = _now()
            sql = """
                UPDATE data_analysis_runs
                   SET status = ?, error_kind = ?, error_message = ?,
                       completed_at = ?, updated_at = ?
                 WHERE task_id = ? AND id = ?
                   AND status IN ('queued', 'running')
            """
            params: tuple[object, ...] = (
                status,
                normalized_error_kind,
                normalized_error_message,
                timestamp,
                timestamp,
                normalized_task_id,
                normalized_run_id,
            )
            if normalized_expected_job_id is not None:
                sql += " AND job_id = ?"
                params += (normalized_expected_job_id,)
            cursor = conn.execute(sql, params)
            if cursor.rowcount != 1:
                raise DataAnalysisTransitionError(
                    f"data analysis run is no longer eligible to become {status}"
                )
            return _required_run(
                conn,
                task_id=normalized_task_id,
                run_id=normalized_run_id,
            )


def canonical_data_analysis_config_json(config: Mapping[str, Any]) -> str:
    """Return the one accepted JSON encoding for an analysis config."""

    return _canonical_config(config)[1]


def canonical_data_analysis_config_hash(config: Mapping[str, Any]) -> str:
    return _digest(canonical_data_analysis_config_json(config))


def data_analysis_input_hash(identity: DataAnalysisIdentity) -> str:
    return _identity(identity).input_hash


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _identity(value: object) -> DataAnalysisIdentity:
    if not isinstance(value, DataAnalysisIdentity):
        raise DataAnalysisDataError("identity must be a DataAnalysisIdentity")
    # Reconstruct from immutable canonical fields.  This catches callers that
    # used object.__setattr__ or mutated a nested config after construction.
    try:
        live_config_json = _canonical_config(value.config)[1]
        reconstructed = DataAnalysisIdentity(
            schema_version=value.schema_version,
            task_id=value.task_id,
            dataset_id=value.dataset_id,
            dataset_content_hash=value.dataset_content_hash,
            workspace_revision=value.workspace_revision,
            analysis_generation=value.analysis_generation,
            semantic_mapping_hash=value.semantic_mapping_hash,
            config=json.loads(value.config_json),
            producer_version=value.producer_version,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataAnalysisDataError("identity is not canonical") from exc
    if (
        live_config_json != value.config_json
        or value.config_json != reconstructed.config_json
        or value.config_hash != reconstructed.config_hash
        or value.input_hash != reconstructed.input_hash
    ):
        raise DataAnalysisDataError("identity hashes are not canonical")
    return reconstructed


def _canonical_config(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise DataAnalysisDataError("config must be a JSON object")
    raw = dict(value)
    try:
        _validate_json_value(raw, field_name="config")
        payload = _canonical_json(raw)
        normalized = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataAnalysisDataError("config must be a canonical JSON object") from exc
    if not isinstance(normalized, dict):
        raise DataAnalysisDataError("config must be a JSON object")
    return normalized, payload


def _validate_json_value(value: object, *, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, field_name=f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            _validate_json_value(item, field_name=f"{field_name}.{key}")
        return
    raise TypeError(f"{field_name} contains a non-JSON value")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compute_input_hash(
    *,
    schema_version: str,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    analysis_generation: int,
    semantic_mapping_hash: str,
    config_json: str,
    config_hash: str,
    producer_version: str,
) -> str:
    # workspace_revision is provenance, not compute identity.  Page and selected
    # field changes advance it without changing any analytical input.
    payload = {
        "schema_version": schema_version,
        "task_id": task_id,
        "dataset_id": dataset_id,
        "dataset_content_hash": dataset_content_hash,
        "analysis_generation": analysis_generation,
        "semantic_mapping_hash": semantic_mapping_hash,
        "config_json": config_json,
        "config_hash": config_hash,
        "producer_version": producer_version,
    }
    return _digest(_canonical_json(payload))


def _stable_run_id(identity: DataAnalysisIdentity) -> str:
    payload = _canonical_json(
        {
            "task_id": identity.task_id,
            "input_hash": identity.input_hash,
        }
    )
    return _digest(f"{_RUN_ID_NAMESPACE}:{payload}")


def _required_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise DataAnalysisDataError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name=field_name)


def _sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DataAnalysisDataError(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataAnalysisDataError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _timestamp(value: object, *, field_name: str) -> tuple[str, datetime]:
    raw = _required_text(value, field_name=field_name)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataAnalysisDataError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise DataAnalysisDataError(f"{field_name} must include a timezone")
    return raw, parsed


def _optional_timestamp(
    value: object,
    *,
    field_name: str,
) -> tuple[str | None, datetime | None]:
    if value is None:
        return None, None
    return _timestamp(value, field_name=field_name)


def _validate_live_identity(
    conn: sqlite3.Connection,
    identity: DataAnalysisIdentity,
) -> None:
    task = conn.execute(
        "SELECT 1 FROM tasks WHERE id = ?",
        (identity.task_id,),
    ).fetchone()
    if task is None:
        raise DataAnalysisNotFoundError("task not found")

    dataset = conn.execute(
        "SELECT task_id, content_hash FROM datasets WHERE id = ?",
        (identity.dataset_id,),
    ).fetchone()
    if dataset is None or str(dataset["task_id"]) != identity.task_id:
        raise DataAnalysisNotFoundError("dataset not found for task")
    try:
        registered_hash = _sha256(
            dataset["content_hash"],
            field_name="registered dataset content hash",
        )
    except (IndexError, KeyError, TypeError, DataAnalysisDataError) as exc:
        raise DataAnalysisDataError(
            "dataset has no canonical registered content hash"
        ) from exc
    if not hmac.compare_digest(registered_hash, identity.dataset_content_hash):
        raise DataAnalysisDataError(
            "dataset registered content hash does not match analysis identity"
        )

    workspace = conn.execute(
        "SELECT * FROM data_workspaces WHERE task_id = ?",
        (identity.task_id,),
    ).fetchone()
    if workspace is None:
        raise DataAnalysisStaleIdentityError(
            "task has no persisted data workspace for analysis"
        )
    try:
        workspace_revision = _non_negative_int(
            workspace["revision"],
            field_name="persisted workspace revision",
        )
        workspace_generation = _non_negative_int(
            workspace["analysis_generation"],
            field_name="persisted analysis_generation",
        )
        workspace_dataset_id = _required_text(
            workspace["active_dataset_id"],
            field_name="persisted active_dataset_id",
        )
        workspace_dataset_hash = _sha256(
            workspace["active_dataset_content_hash"],
            field_name="persisted active_dataset_content_hash",
        )
        raw_mapping = workspace["semantic_mapping_json"]
        if not isinstance(raw_mapping, str):
            raise DataAnalysisDataError(
                "persisted semantic_mapping_json must be text"
            )
        mapping_payload = json.loads(raw_mapping)
        mapping = data_semantic_mapping_from_dict(mapping_payload)
        canonical_mapping = _canonical_json(data_semantic_mapping_to_dict(mapping))
        if raw_mapping != canonical_mapping:
            raise DataAnalysisDataError(
                "persisted semantic_mapping_json is not canonical"
            )
        workspace_mapping_hash = data_semantic_mapping_hash(mapping)
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise DataAnalysisDataError("persisted data workspace is corrupt") from exc

    if workspace_revision != identity.workspace_revision:
        raise DataAnalysisStaleIdentityError(
            "workspace revision changed: "
            f"expected {identity.workspace_revision}, found {workspace_revision}"
        )
    if workspace_dataset_id != identity.dataset_id or not hmac.compare_digest(
        workspace_dataset_hash,
        identity.dataset_content_hash,
    ):
        raise DataAnalysisStaleIdentityError(
            "active workspace dataset changed before analysis started"
        )
    if workspace_generation != identity.analysis_generation:
        raise DataAnalysisStaleIdentityError(
            "analysis generation changed before analysis started"
        )
    if not hmac.compare_digest(
        workspace_mapping_hash,
        identity.semantic_mapping_hash,
    ):
        raise DataAnalysisStaleIdentityError(
            "semantic mapping changed before analysis started"
        )


def _validate_job(
    conn: sqlite3.Connection,
    task_id: str,
    job_id: str | None,
    *,
    allowed_statuses: frozenset[str],
    action: str,
) -> None:
    if job_id is None:
        return
    row = conn.execute(
        "SELECT task_id, kind, status FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None or str(row["task_id"]) != task_id:
        raise DataAnalysisNotFoundError("job not found for task")
    if str(row["kind"]) != DATA_ANALYSIS_JOB_KIND:
        raise DataAnalysisDataError(
            "job kind must be data_analysis for a data analysis run"
        )
    status = str(row["status"])
    if status not in allowed_statuses:
        expected = (
            "running"
            if allowed_statuses == _RUNNING_JOB_STATUSES
            else "queued"
        )
        raise DataAnalysisTransitionError(
            f"data analysis job must be {expected} before {action}"
        )


def _select_for_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM data_analysis_runs WHERE task_id = ? AND id = ?",
        (task_id, run_id),
    ).fetchone()


def _select_by_input_hash(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    input_hash: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM data_analysis_runs WHERE task_id = ? AND input_hash = ?",
        (task_id, input_hash),
    ).fetchone()


def _required_run(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: str,
) -> DataAnalysisRunRecord:
    row = _select_for_task(conn, task_id=task_id, run_id=run_id)
    if row is None:
        raise DataAnalysisNotFoundError("run not found for task")
    return _validated_record(conn, row)


def _validated_record(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> DataAnalysisRunRecord:
    record = _record_from_row(row)
    _validate_record_references(conn, record)
    return record


def _record_from_row(row: sqlite3.Row) -> DataAnalysisRunRecord:
    try:
        raw_config = row["config_json"]
        if not isinstance(raw_config, str):
            raise DataAnalysisDataError("config_json must be text")
        config = json.loads(raw_config)
        canonical_config, canonical_config_json = _canonical_config(config)
        if raw_config != canonical_config_json:
            raise DataAnalysisDataError("config_json is not canonical")

        identity = DataAnalysisIdentity(
            schema_version=row["schema_version"],
            task_id=row["task_id"],
            dataset_id=row["dataset_id"],
            dataset_content_hash=row["dataset_content_hash"],
            workspace_revision=row["workspace_revision"],
            analysis_generation=row["analysis_generation"],
            semantic_mapping_hash=row["semantic_mapping_hash"],
            config=canonical_config,
            producer_version=row["producer_version"],
        )
        stored_config_hash = _sha256(
            row["config_hash"],
            field_name="persisted config_hash",
        )
        stored_input_hash = _sha256(
            row["input_hash"],
            field_name="persisted input_hash",
        )
        if not hmac.compare_digest(stored_config_hash, identity.config_hash):
            raise DataAnalysisDataError("config_hash does not match config_json")
        if not hmac.compare_digest(stored_input_hash, identity.input_hash):
            raise DataAnalysisDataError("input_hash does not match run identity")

        run_id = _sha256(row["id"], field_name="persisted run id")
        if not hmac.compare_digest(run_id, _stable_run_id(identity)):
            raise DataAnalysisDataError("run id does not match run identity")
        job_id = _optional_text(row["job_id"], field_name="persisted job_id")
        status = _required_text(row["status"], field_name="persisted status")
        if status not in _STATUS_SET:
            raise DataAnalysisDataError("persisted status is unsupported")
        result_artifact_id = _optional_text(
            row["result_artifact_id"],
            field_name="persisted result_artifact_id",
        )
        result_content_hash = (
            None
            if row["result_content_hash"] is None
            else _sha256(
                row["result_content_hash"],
                field_name="persisted result_content_hash",
            )
        )
        error_kind = _optional_text(
            row["error_kind"],
            field_name="persisted error_kind",
        )
        error_message = _optional_text(
            row["error_message"],
            field_name="persisted error_message",
        )
        created_at, created = _timestamp(
            row["created_at"],
            field_name="persisted created_at",
        )
        updated_at, updated = _timestamp(
            row["updated_at"],
            field_name="persisted updated_at",
        )
        started_at, started = _optional_timestamp(
            row["started_at"],
            field_name="persisted started_at",
        )
        completed_at, completed = _optional_timestamp(
            row["completed_at"],
            field_name="persisted completed_at",
        )
        if updated < created:
            raise DataAnalysisDataError("updated_at precedes created_at")
        if started is not None and started < created:
            raise DataAnalysisDataError("started_at precedes created_at")
        if completed is not None and completed < created:
            raise DataAnalysisDataError("completed_at precedes created_at")
        if started is not None and completed is not None and completed < started:
            raise DataAnalysisDataError("completed_at precedes started_at")
        _validate_status_shape(
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            result_artifact_id=result_artifact_id,
            result_content_hash=result_content_hash,
            error_kind=error_kind,
            error_message=error_message,
        )
        return DataAnalysisRunRecord(
            id=run_id,
            identity=identity,
            job_id=job_id,
            status=status,
            result_artifact_id=result_artifact_id,
            result_content_hash=result_content_hash,
            error_kind=error_kind,
            error_message=error_message,
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            completed_at=completed_at,
        )
    except (
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raw_id = row["id"] if "id" in row.keys() else "unknown"
        raise DataAnalysisDataError(
            f"corrupt data analysis run: {raw_id}"
        ) from exc


def _validate_status_shape(
    *,
    status: str,
    started_at: str | None,
    completed_at: str | None,
    result_artifact_id: str | None,
    result_content_hash: str | None,
    error_kind: str | None,
    error_message: str | None,
) -> None:
    result_pair = (result_artifact_id is None) == (result_content_hash is None)
    error_pair = (error_kind is None) == (error_message is None)
    if not result_pair or not error_pair:
        raise DataAnalysisDataError("persisted run has incomplete evidence fields")
    valid = (
        status == "queued"
        and started_at is None
        and completed_at is None
        and result_artifact_id is None
        and error_kind is None
    ) or (
        status == "running"
        and started_at is not None
        and completed_at is None
        and result_artifact_id is None
        and error_kind is None
    ) or (
        status == "succeeded"
        and started_at is not None
        and completed_at is not None
        and result_artifact_id is not None
        and error_kind is None
    ) or (
        status in _TERMINAL_RETRYABLE
        and completed_at is not None
        and result_artifact_id is None
        and error_kind is not None
    )
    if not valid:
        raise DataAnalysisDataError(
            "persisted run status does not match its evidence fields"
        )


def _validate_record_references(
    conn: sqlite3.Connection,
    record: DataAnalysisRunRecord,
) -> None:
    dataset = conn.execute(
        "SELECT task_id, content_hash FROM datasets WHERE id = ?",
        (record.dataset_id,),
    ).fetchone()
    if dataset is None or str(dataset["task_id"]) != record.task_id:
        raise DataAnalysisDataError(
            "persisted run dataset is not owned by its task"
        )
    try:
        dataset_hash = _sha256(
            dataset["content_hash"],
            field_name="persisted dataset content hash",
        )
    except (IndexError, KeyError, TypeError, DataAnalysisDataError) as exc:
        raise DataAnalysisDataError(
            "persisted run dataset has no canonical content hash"
        ) from exc
    if not hmac.compare_digest(dataset_hash, record.dataset_content_hash):
        raise DataAnalysisDataError(
            "persisted run dataset content hash no longer matches registry"
        )

    if record.job_id is not None:
        job = conn.execute(
            "SELECT task_id, kind FROM jobs WHERE id = ?",
            (record.job_id,),
        ).fetchone()
        if job is None or str(job["task_id"]) != record.task_id:
            raise DataAnalysisDataError(
                "persisted run job is not owned by its task"
            )
        if str(job["kind"]) != DATA_ANALYSIS_JOB_KIND:
            raise DataAnalysisDataError(
                "persisted run job kind is not data_analysis"
            )

    if record.status == "succeeded":
        artifact = conn.execute(
            """
            SELECT task_id, kind, content_hash, origin_tool, provenance_json
              FROM task_artifacts
             WHERE id = ?
            """,
            (record.result_artifact_id,),
        ).fetchone()
        if artifact is None or str(artifact["task_id"]) != record.task_id:
            raise DataAnalysisDataError(
                "persisted result artifact is not owned by its task"
            )
        assert record.result_content_hash is not None
        _validate_analysis_artifact(
            artifact,
            record=record,
            expected_content_hash=record.result_content_hash,
        )


def _validate_analysis_artifact(
    artifact: sqlite3.Row,
    *,
    record: DataAnalysisRunRecord,
    expected_content_hash: str,
) -> None:
    if str(artifact["kind"]) != DATA_ANALYSIS_ARTIFACT_KIND:
        raise DataAnalysisDataError(
            "registered artifact kind must be data_analysis"
        )
    if str(artifact["origin_tool"]) != DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL:
        raise DataAnalysisDataError(
            "registered artifact origin does not match the analysis service"
        )
    try:
        registered_hash = _sha256(
            artifact["content_hash"],
            field_name="registered artifact content hash",
        )
    except (IndexError, KeyError, TypeError, DataAnalysisDataError) as exc:
        raise DataAnalysisDataError(
            "registered artifact has a corrupt content hash"
        ) from exc
    if not hmac.compare_digest(registered_hash, expected_content_hash):
        raise DataAnalysisDataError(
            "result content hash does not match the registered artifact"
        )

    raw_provenance = artifact["provenance_json"]
    try:
        if not isinstance(raw_provenance, str):
            raise TypeError("provenance_json must be text")
        provenance = json.loads(raw_provenance)
        if not isinstance(provenance, dict):
            raise TypeError("provenance_json must be an object")
        if raw_provenance != _canonical_json(provenance):
            raise ValueError("provenance_json must be canonical")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataAnalysisDataError(
            "registered artifact provenance is not canonical"
        ) from exc

    expected_provenance = {
        "schema_version": record.schema_version,
        "task_id": record.task_id,
        "dataset_id": record.dataset_id,
        "dataset_content_hash": record.dataset_content_hash,
        "analysis_generation": record.analysis_generation,
        "semantic_mapping_hash": record.semantic_mapping_hash,
        "config_hash": record.config_hash,
        "producer_version": record.producer_version,
        "input_hash": record.input_hash,
    }
    if provenance != expected_provenance:
        raise DataAnalysisDataError(
            "registered artifact provenance does not match the analysis run"
        )


def _require_same_computation(
    persisted: DataAnalysisIdentity,
    requested: DataAnalysisIdentity,
) -> None:
    persisted_fields = (
        persisted.schema_version,
        persisted.task_id,
        persisted.dataset_id,
        persisted.dataset_content_hash,
        persisted.analysis_generation,
        persisted.semantic_mapping_hash,
        persisted.config_json,
        persisted.config_hash,
        persisted.producer_version,
        persisted.input_hash,
    )
    requested_fields = (
        requested.schema_version,
        requested.task_id,
        requested.dataset_id,
        requested.dataset_content_hash,
        requested.analysis_generation,
        requested.semantic_mapping_hash,
        requested.config_json,
        requested.config_hash,
        requested.producer_version,
        requested.input_hash,
    )
    if persisted_fields != requested_fields:
        raise DataAnalysisDataError(
            "input_hash maps to a different data analysis identity"
        )


def _require_same_job(
    record: DataAnalysisRunRecord,
    requested_job_id: str | None,
) -> None:
    # Omission means "do not attach/assert a job".  This lets an idempotent API
    # replay observe a run after mark_running atomically attached its async job.
    if requested_job_id is not None and record.job_id != requested_job_id:
        raise DataAnalysisConflictError(
            "data analysis replay job_id does not match persisted run"
        )


def _retry_record(
    conn: sqlite3.Connection,
    record: DataAnalysisRunRecord,
    *,
    job_id: str | None,
) -> DataAnalysisRunRecord:
    if record.status not in _TERMINAL_RETRYABLE:
        raise DataAnalysisTransitionError(
            "only a failed or cancelled data analysis run can be retried"
        )
    timestamp = _now()
    cursor = conn.execute(
        """
        UPDATE data_analysis_runs
           SET job_id = ?, status = 'queued', result_artifact_id = NULL,
               result_content_hash = NULL, error_kind = NULL,
               error_message = NULL, started_at = NULL, completed_at = NULL,
               updated_at = ?
         WHERE task_id = ? AND id = ?
           AND status IN ('failed', 'cancelled')
        """,
        (job_id, timestamp, record.task_id, record.id),
    )
    if cursor.rowcount != 1:
        raise DataAnalysisTransitionError(
            "data analysis run is no longer failed or cancelled"
        )
    return _required_run(
        conn,
        task_id=record.task_id,
        run_id=record.id,
    )


__all__ = [
    "DATA_ANALYSIS_ARTIFACT_KIND",
    "DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL",
    "DATA_ANALYSIS_JOB_KIND",
    "DATA_ANALYSIS_SCHEMA_VERSION",
    "DATA_ANALYSIS_STATUSES",
    "DataAnalysisConflictError",
    "DataAnalysisDataError",
    "DataAnalysisIdentity",
    "DataAnalysisNotFoundError",
    "DataAnalysisRepository",
    "DataAnalysisRun",
    "DataAnalysisRunIdentity",
    "DataAnalysisRunRecord",
    "DataAnalysisStaleIdentityError",
    "DataAnalysisTransitionError",
    "canonical_data_analysis_config_hash",
    "canonical_data_analysis_config_json",
    "data_analysis_input_hash",
]
