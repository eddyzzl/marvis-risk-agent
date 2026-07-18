"""Shared orchestration for task-owned deterministic data analysis.

Manual HTTP requests and Agent workflows can use the same service boundary:
request normalization and cache identity happen before a Task job is created;
execution revalidates the live workspace and physical dataset; artifact
promotion, registry insertion, successful run transition, and completion audit
share one transaction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
from pathlib import Path
import threading
from urllib.parse import quote

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DUCKDB_TEMP_DIR_NAME, DataBackend
from marvis.data.descriptive import (
    DATA_ANALYSIS_SCHEMA_VERSION,
    DescriptiveBudgetError,
    DescriptiveConfig,
    DescriptiveInputError,
    TaggedScalar,
    analyze_parquet,
)
from marvis.data.errors import DatasetContentDriftError
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceSnapshot,
    data_semantic_mapping_hash,
    data_semantic_mapping_to_dict,
)
from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.job_heartbeat import heartbeat_job
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.data_analysis import (
    DATA_ANALYSIS_ARTIFACT_KIND as REPOSITORY_DATA_ANALYSIS_ARTIFACT_KIND,
    DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
    DATA_ANALYSIS_JOB_KIND as REPOSITORY_DATA_ANALYSIS_JOB_KIND,
    DataAnalysisConflictError,
    DataAnalysisDataError,
    DataAnalysisIdentity,
    DataAnalysisRepository,
    DataAnalysisRunRecord,
    DataAnalysisTransitionError,
)
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
)
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.tasks import TaskRepository
from marvis.state_machine import ConflictError


logger = logging.getLogger(__name__)

DATA_ANALYSIS_SECTIONS = (
    "overview",
    "target",
    "missing",
    "distribution",
    "correlation",
)
DATA_ANALYSIS_JOB_KIND = REPOSITORY_DATA_ANALYSIS_JOB_KIND
DATA_ANALYSIS_ARTIFACT_KIND = REPOSITORY_DATA_ANALYSIS_ARTIFACT_KIND

_SECTION_SET = frozenset(DATA_ANALYSIS_SECTIONS)
_SENSITIVE_FIELD_ROLES = frozenset({"phone", "idcard", "id", "name"})
_IDENTITY_CONFIG_KEYS = frozenset({"sections", "columns", "config"})
_TOKEN_NAMESPACE = "marvis.data-analysis-token.v1"
_DISPATCH_LOCK_GUARD = threading.Lock()
_DISPATCH_LOCKS: dict[tuple[str, str], threading.Lock] = {}


class DataAnalysisServiceError(RuntimeError):
    """Base error for the shared analysis orchestration boundary."""


class DataAnalysisRequestError(DataAnalysisServiceError, ValueError):
    """The requested sections, columns, or active workspace are invalid."""


class DataAnalysisNotFoundError(DataAnalysisServiceError):
    """A task-owned resource is absent or intentionally concealed."""


class DataAnalysisWorkspacePreconditionError(DataAnalysisServiceError):
    """If-Match did not identify the current workspace revision."""


class DataAnalysisRetryRequiredError(DataAnalysisServiceError):
    """A terminal failed/cancelled identity requires explicit retry consent."""

    def __init__(self, record: DataAnalysisRunRecord) -> None:
        self.record = record
        super().__init__(
            "failed or cancelled data analysis requires explicit retry=true"
        )


class DataAnalysisActiveJobError(DataAnalysisServiceError):
    """Another task job owns the task's single active execution slot."""


class DataAnalysisExecutionStaleError(DataAnalysisServiceError):
    """Live dataset/semantic computation identity changed after dispatch."""


class DataAnalysisArtifactError(DataAnalysisServiceError):
    """A successful run's artifact is missing, unowned, or integrity-drifted."""


@dataclass(frozen=True)
class DataAnalysisRequest:
    sections: Sequence[str]
    columns: Sequence[str] | None
    config: DescriptiveConfig
    retry: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sections, (str, bytes)):
            raise DataAnalysisRequestError("sections must be a sequence")
        raw_sections = tuple(self.sections)
        if not raw_sections:
            raise DataAnalysisRequestError("sections must not be empty")
        if any(not isinstance(section, str) for section in raw_sections):
            raise DataAnalysisRequestError("sections must contain strings")
        if len(set(raw_sections)) != len(raw_sections):
            raise DataAnalysisRequestError("sections must not contain duplicates")
        unknown_sections = sorted(set(raw_sections) - _SECTION_SET)
        if unknown_sections:
            raise DataAnalysisRequestError(
                "unsupported data analysis section(s): "
                + ", ".join(unknown_sections)
            )
        sections = tuple(
            section for section in DATA_ANALYSIS_SECTIONS if section in raw_sections
        )

        columns: tuple[str, ...] | None
        if self.columns is None:
            columns = None
        else:
            if isinstance(self.columns, (str, bytes)):
                raise DataAnalysisRequestError("columns must be a sequence")
            columns = tuple(self.columns)
            if not columns:
                raise DataAnalysisRequestError("columns must be null or non-empty")
            for column in columns:
                _canonical_text(column, field_name="columns item")
            if len(set(columns)) != len(columns):
                raise DataAnalysisRequestError("columns must not contain duplicates")
        if not isinstance(self.config, DescriptiveConfig):
            raise DataAnalysisRequestError("config must be a DescriptiveConfig")
        if not isinstance(self.retry, bool):
            raise DataAnalysisRequestError("retry must be a boolean")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "columns", columns)

    def identity_config(self) -> dict[str, object]:
        return {
            "sections": list(self.sections),
            "columns": None if self.columns is None else list(self.columns),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_identity_config(
        cls,
        payload: object,
        *,
        retry: bool = False,
    ) -> DataAnalysisRequest:
        if not isinstance(payload, Mapping) or set(payload) != _IDENTITY_CONFIG_KEYS:
            raise DataAnalysisRequestError("persisted analysis request is invalid")
        config_payload = payload["config"]
        if not isinstance(config_payload, Mapping):
            raise DataAnalysisRequestError("persisted analysis config is invalid")
        try:
            config = DescriptiveConfig(**dict(config_payload))
        except (TypeError, ValueError) as exc:
            raise DataAnalysisRequestError(
                "persisted analysis config is invalid"
            ) from exc
        return cls(
            sections=payload["sections"],
            columns=payload["columns"],
            config=config,
            retry=retry,
        )


@dataclass(frozen=True)
class DataAnalysisDispatch:
    record: DataAnalysisRunRecord
    should_execute: bool
    cached: bool
    result: dict[str, object] | None = None
    result_artifact_id: str | None = None
    download_url: str | None = None

    @property
    def job_id(self) -> str | None:
        return self.record.job_id

    @property
    def http_status(self) -> int:
        return 200 if self.record.status == "succeeded" else 202


@dataclass(frozen=True)
class DataAnalysisRunView:
    record: DataAnalysisRunRecord
    result: dict[str, object] | None = None
    result_artifact_id: str | None = None
    download_url: str | None = None


@dataclass(frozen=True)
class _AnalysisContext:
    snapshot: DataWorkspaceSnapshot
    request: DataAnalysisRequest
    identity: DataAnalysisIdentity
    dataset_path: Path
    mapping: DataSemanticMapping
    sensitive_columns: frozenset[str]


class DataAnalysisService:
    def __init__(
        self,
        settings,
        *,
        analyzer: Callable[..., dict[str, object]] | None = None,
    ) -> None:
        self.settings = settings
        self.analysis_repo = DataAnalysisRepository(settings.db_path)
        self.task_repo = TaskRepository(settings.db_path)
        self.artifact_repo = TaskArtifactRepository(settings.db_path)
        self.workspace_repo = DataWorkspaceRepository(settings.db_path)
        self.dataset_registry = DatasetRegistry(
            DatasetRepository(settings.db_path),
            DataBackend(settings.datasets_dir),
            settings.datasets_dir,
        )
        self._analyzer = analyzer or analyze_parquet

    def request_analysis(
        self,
        task_id: str,
        *,
        expected_workspace_revision: int,
        request: DataAnalysisRequest,
    ) -> DataAnalysisDispatch:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        if (
            isinstance(expected_workspace_revision, bool)
            or not isinstance(expected_workspace_revision, int)
            or expected_workspace_revision < 0
        ):
            raise DataAnalysisRequestError(
                "expected_workspace_revision must be a non-negative integer"
            )
        if not isinstance(request, DataAnalysisRequest):
            raise DataAnalysisRequestError(
                "request must be a DataAnalysisRequest"
            )

        lock = _dispatch_lock(self.settings.db_path, normalized_task_id)
        with lock:
            context = self._request_context(
                normalized_task_id,
                expected_workspace_revision=expected_workspace_revision,
                request=request,
            )
            try:
                current = self.analysis_repo.current(context.identity)
            except DataAnalysisDataError as exc:
                raise DataAnalysisArtifactError(
                    "data analysis cache registry drifted"
                ) from exc
            if current is not None and current.status == "succeeded":
                view = self._view_from_record(current)
                return DataAnalysisDispatch(
                    record=current,
                    should_execute=False,
                    cached=True,
                    result=view.result,
                    result_artifact_id=view.result_artifact_id,
                    download_url=view.download_url,
                )
            if current is not None and current.status in {"failed", "cancelled"}:
                if not request.retry:
                    raise DataAnalysisRetryRequiredError(current)
                current = self.analysis_repo.retry(context.identity)
            elif current is None:
                current = self.analysis_repo.create_or_get(context.identity)

            if current.status == "running":
                if self._has_active_attached_job(current):
                    return DataAnalysisDispatch(
                        record=current,
                        should_execute=False,
                        cached=False,
                    )
                current = self._strand_run_for_retry(current)
                if not request.retry:
                    raise DataAnalysisRetryRequiredError(current)
                current = self.analysis_repo.retry(context.identity)
            if current.status != "queued":
                raise DataAnalysisConflictError(
                    f"unsupported data analysis dispatch state: {current.status}"
                )
            if current.job_id is not None:
                if self._has_active_attached_job(current):
                    return DataAnalysisDispatch(
                        record=current,
                        should_execute=False,
                        cached=False,
                    )
                current = self._strand_run_for_retry(current)
                if not request.retry:
                    raise DataAnalysisRetryRequiredError(current)
                current = self.analysis_repo.retry(context.identity)

            try:
                job_id = self._start_and_attach_job(current)
            except DataAnalysisActiveJobError:
                refreshed = self.analysis_repo.get_for_task(
                    normalized_task_id,
                    current.id,
                )
                if refreshed is not None and self._has_active_attached_job(
                    refreshed
                ):
                    return DataAnalysisDispatch(
                        record=refreshed,
                        should_execute=False,
                        cached=False,
                    )
                raise
            attached = self.analysis_repo.get_for_task(
                normalized_task_id,
                current.id,
            )
            assert attached is not None
            assert attached.job_id == job_id
            return DataAnalysisDispatch(
                record=attached,
                should_execute=True,
                cached=False,
            )

    def _has_active_attached_job(self, record: DataAnalysisRunRecord) -> bool:
        if record.job_id is None or record.status not in {"queued", "running"}:
            return False
        job = self.task_repo.get_job(record.job_id)
        return bool(
            job is not None
            and job["task_id"] == record.task_id
            and job["kind"] == DATA_ANALYSIS_JOB_KIND
            and job["status"] in {"queued", "running"}
        )

    def run_job(self, *, task_id: str, run_id: str, job_id: str) -> None:
        """Execute one already-dispatched job and close both state machines."""

        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        normalized_run_id = _canonical_text(run_id, field_name="run_id")
        normalized_job_id = _canonical_text(job_id, field_name="job_id")
        record = self.analysis_repo.get_for_task(
            normalized_task_id,
            normalized_run_id,
        )
        if (
            record is None
            or record.job_id != normalized_job_id
            or record.status not in {"queued", "running"}
        ):
            return
        job = self.task_repo.get_job(normalized_job_id)
        if (
            job is None
            or job["task_id"] != normalized_task_id
            or job["kind"] != DATA_ANALYSIS_JOB_KIND
        ):
            return
        if job["status"] == "running":
            return
        if job["status"] != "queued":
            self._fail_active_run(
                task_id=normalized_task_id,
                run_id=normalized_run_id,
                error_kind="data_analysis_job_not_active",
                error_message="task job was no longer queued when execution started",
                expected_job_id=normalized_job_id,
            )
            return
        if self.task_repo.mark_job_running(normalized_job_id) is False:
            latest_job = self.task_repo.get_job(normalized_job_id)
            if latest_job is not None and latest_job["status"] not in {
                "queued",
                "running",
            }:
                self._fail_active_run(
                    task_id=normalized_task_id,
                    run_id=normalized_run_id,
                    error_kind="data_analysis_job_not_active",
                    error_message=(
                        "task job was no longer queued when execution started"
                    ),
                    expected_job_id=normalized_job_id,
                )
            return

        try:
            self.analysis_repo.mark_running(
                task_id=normalized_task_id,
                run_id=normalized_run_id,
                job_id=normalized_job_id,
            )
            with heartbeat_job(self.task_repo, normalized_job_id):
                self._execute_running(
                    task_id=normalized_task_id,
                    run_id=normalized_run_id,
                )
        except Exception as exc:
            error_kind, error_message = _execution_error(exc)
            self._fail_active_run(
                task_id=normalized_task_id,
                run_id=normalized_run_id,
                error_kind=error_kind,
                error_message=error_message,
                expected_job_id=normalized_job_id,
            )
            self.task_repo.finish_job(
                normalized_job_id,
                status="failed",
                error_name=error_kind,
                error_value=error_message,
                traceback="",
            )
            return

    def fail_dispatch(
        self,
        *,
        task_id: str,
        run_id: str,
        job_id: str,
        error_kind: str,
        error_message: str,
    ) -> None:
        """Release a queued dispatch when background registration itself fails."""

        self._fail_active_run(
            task_id=task_id,
            run_id=run_id,
            error_kind=error_kind,
            error_message=error_message,
            expected_job_id=job_id,
        )
        self.task_repo.finish_job(
            job_id,
            status="failed",
            error_name=error_kind,
            error_value=_safe_error_message(error_message),
            traceback="",
        )

    def get_run(
        self,
        task_id: str,
        run_id: str,
    ) -> DataAnalysisRunView | None:
        normalized_task_id = _canonical_text(task_id, field_name="task_id")
        normalized_run_id = _canonical_text(run_id, field_name="run_id")
        try:
            record = self.analysis_repo.get_for_task(
                normalized_task_id,
                normalized_run_id,
            )
        except DataAnalysisDataError as exc:
            raise DataAnalysisArtifactError(
                "data analysis artifact registry drifted"
            ) from exc
        return None if record is None else self._view_from_record(record)

    def _request_context(
        self,
        task_id: str,
        *,
        expected_workspace_revision: int,
        request: DataAnalysisRequest,
    ) -> _AnalysisContext:
        try:
            self.task_repo.get_task(task_id)
        except KeyError as exc:
            raise DataAnalysisNotFoundError("task not found") from exc
        try:
            snapshot = self.workspace_repo.get_or_default(task_id)
        except (DataWorkspaceDataError, DataWorkspaceDatasetNotFound) as exc:
            raise DataAnalysisRequestError("data workspace is unavailable") from exc
        if snapshot.revision != expected_workspace_revision:
            raise DataAnalysisWorkspacePreconditionError(
                "stale data workspace revision: "
                f"expected {expected_workspace_revision}, found {snapshot.revision}"
            )
        return self._context_from_snapshot(snapshot, request=request)

    def _context_from_snapshot(
        self,
        snapshot: DataWorkspaceSnapshot,
        *,
        request: DataAnalysisRequest,
    ) -> _AnalysisContext:
        dataset_id = snapshot.active_dataset_id
        dataset_hash = snapshot.active_dataset_content_hash
        if dataset_id is None or dataset_hash is None:
            raise DataAnalysisRequestError(
                "data analysis requires an active verified dataset"
            )
        try:
            dataset = self.dataset_registry.get(dataset_id)
        except KeyError as exc:
            raise DataAnalysisRequestError("active dataset not found") from exc
        if dataset.task_id != snapshot.task_id:
            raise DataAnalysisRequestError("active dataset not found")
        if dataset.content_hash != dataset_hash:
            raise DataAnalysisRequestError(
                "active dataset hash does not match the workspace"
            )
        try:
            dataset_path = self.dataset_registry.resolve_verified_path(dataset_id)
        except DatasetContentDriftError as exc:
            raise DataAnalysisRequestError(
                "active dataset integrity verification failed"
            ) from exc

        available_columns = tuple(column.name for column in dataset.columns)
        if request.columns is None:
            normalized_columns = None
        else:
            selected = set(request.columns)
            unknown = sorted(selected - set(available_columns))
            if unknown:
                raise DataAnalysisRequestError(
                    "unknown data analysis column(s): " + ", ".join(unknown)
                )
            normalized_columns = tuple(
                column for column in available_columns if column in selected
            )
        normalized_request = DataAnalysisRequest(
            sections=request.sections,
            columns=normalized_columns,
            config=request.config,
            retry=request.retry,
        )
        mapping = snapshot.semantic_mapping
        sensitive_columns = frozenset(
            {
                str(column.name)
                for column in dataset.columns
                if str(column.semantic_role) in _SENSITIVE_FIELD_ROLES
            }
            | {
                str(name)
                for name, role in mapping.field_roles.items()
                if str(role) in _SENSITIVE_FIELD_ROLES
            }
        )
        identity = DataAnalysisIdentity(
            task_id=snapshot.task_id,
            dataset_id=dataset_id,
            dataset_content_hash=dataset_hash,
            workspace_revision=snapshot.revision,
            analysis_generation=snapshot.analysis_generation,
            semantic_mapping_hash=data_semantic_mapping_hash(mapping),
            config=normalized_request.identity_config(),
            producer_version=DATA_ANALYSIS_SCHEMA_VERSION,
        )
        return _AnalysisContext(
            snapshot=snapshot,
            request=normalized_request,
            identity=identity,
            dataset_path=dataset_path,
            mapping=mapping,
            sensitive_columns=sensitive_columns,
        )

    def _start_and_attach_job(self, record: DataAnalysisRunRecord) -> str:
        try:
            job_id = self.task_repo.start_job(record.task_id, DATA_ANALYSIS_JOB_KIND)
        except ConflictError as exc:
            raise DataAnalysisActiveJobError(
                "task already has an active job"
            ) from exc
        try:
            attached = self.analysis_repo.attach_job(
                task_id=record.task_id,
                run_id=record.id,
                job_id=job_id,
            )
            self._write_run_audit(
                kind="data.analysis.started",
                record=attached,
                outcome="queued",
            )
        except Exception as exc:
            self.task_repo.finish_job(
                job_id,
                status="failed",
                error_name=exc.__class__.__name__,
                error_value=_safe_error_message(str(exc)),
                traceback="",
            )
            self._fail_active_run(
                task_id=record.task_id,
                run_id=record.id,
                error_kind="data_analysis_dispatch_failed",
                error_message="data analysis dispatch could not attach its task job",
                expected_job_id=job_id,
            )
            raise
        return job_id

    def _strand_run_for_retry(
        self,
        record: DataAnalysisRunRecord,
    ) -> DataAnalysisRunRecord:
        failed = self.analysis_repo.fail(
            task_id=record.task_id,
            run_id=record.id,
            error_kind="data_analysis_job_lost",
            error_message="attached task job ended before analysis completed",
            expected_job_id=record.job_id,
        )
        self._write_run_audit(
            kind="data.analysis.failed",
            record=failed,
            outcome="failed",
        )
        return failed

    def _execute_running(self, *, task_id: str, run_id: str) -> DataAnalysisRunView:
        record = self.analysis_repo.get_for_task(task_id, run_id)
        if record is None:
            raise DataAnalysisExecutionStaleError("data analysis run is missing")
        if record.status != "running":
            raise DataAnalysisTransitionError(
                "data analysis run must be running during execution"
            )
        request = DataAnalysisRequest.from_identity_config(record.config)
        try:
            snapshot = self.workspace_repo.get_or_default(task_id)
            context = self._context_from_snapshot(snapshot, request=request)
        except (DataAnalysisRequestError, DataWorkspaceDataError) as exc:
            raise DataAnalysisExecutionStaleError(
                "live data workspace no longer matches the dispatched analysis"
            ) from exc
        if not hmac.compare_digest(context.identity.input_hash, record.input_hash):
            raise DataAnalysisExecutionStaleError(
                "dataset or semantic analysis identity changed before execution"
            )
        current = self.analysis_repo.current(context.identity)
        if current is None or current.id != record.id:
            raise DataAnalysisExecutionStaleError(
                "data analysis run is not current for the live workspace"
            )

        sanitizers = _value_sanitizers(
            task_id=record.task_id,
            dataset_hash=record.dataset_content_hash,
            sensitive_columns=context.sensitive_columns,
        )
        analysis = self._analyzer(
            context.dataset_path,
            temp_directory=(
                self.settings.datasets_dir.parent / DUCKDB_TEMP_DIR_NAME
            ),
            target_column=context.mapping.target_col,
            columns=context.request.columns,
            config=context.request.config,
            value_sanitizers=sanitizers,
        )
        analysis = _suppress_sensitive_analysis_values(
            analysis,
            sensitive_columns=context.sensitive_columns,
        )
        envelope = _result_envelope(
            record=record,
            request=context.request,
            mapping=context.mapping,
            analysis=analysis,
        )
        return self._persist_result(record=record, envelope=envelope)

    def _persist_result(
        self,
        *,
        record: DataAnalysisRunRecord,
        envelope: dict[str, object],
    ) -> DataAnalysisRunView:
        raw = _canonical_json(envelope)
        relative_path = Path(
            "tasks",
            record.task_id,
            "data_analysis",
            f"{record.input_hash}.json",
        )
        artifact_root = self.settings.tasks_dir / record.task_id / "data_analysis"
        uow = ArtifactUnitOfWork()
        staged = uow.stage_file(artifact_root, f"{record.input_hash}.json")
        try:
            staged.path.write_text(raw, encoding="utf-8")
            content_hash = sha256_file(staged.path)
            provenance = _artifact_provenance(record)

            def commit_result(conn):
                artifact = self.artifact_repo.register_on_connection(
                    conn,
                    task_id=record.task_id,
                    kind=DATA_ANALYSIS_ARTIFACT_KIND,
                    path=relative_path.as_posix(),
                    content_hash=content_hash,
                    origin_tool=DATA_ANALYSIS_ARTIFACT_ORIGIN_TOOL,
                    provenance=provenance,
                )
                completed = self.analysis_repo.complete_on_connection(
                    conn,
                    task_id=record.task_id,
                    run_id=record.id,
                    result_artifact_id=str(artifact["id"]),
                    result_content_hash=content_hash,
                )
                if record.job_id is None or not self.task_repo.finish_job_on_connection(
                    conn,
                    record.job_id,
                    status="succeeded",
                    expected_status="running",
                ):
                    raise DataAnalysisTransitionError(
                        "bound data analysis job is no longer running"
                    )
                _write_audit_row(
                    conn,
                    kind="data.analysis.completed",
                    target_ref=record.id,
                    inputs_hash=record.input_hash,
                    outcome="succeeded",
                    detail=_audit_detail(
                        completed,
                        artifact_id=str(artifact["id"]),
                    ),
                )
                return completed, str(artifact["id"])

            completed, artifact_id = uow.finalize_with_connection(
                self.analysis_repo.transaction,
                commit_result,
            )
        except Exception:
            uow.rollback()
            raise
        return DataAnalysisRunView(
            record=completed,
            result=envelope,
            result_artifact_id=artifact_id,
            download_url=_download_url(record.task_id, artifact_id),
        )

    def _fail_active_run(
        self,
        *,
        task_id: str,
        run_id: str,
        error_kind: str,
        error_message: str,
        expected_job_id: str | None = None,
    ) -> DataAnalysisRunRecord | None:
        safe_kind = _canonical_error_kind(error_kind)
        safe_message = _safe_error_message(error_message)
        record = self.analysis_repo.get_for_task(task_id, run_id)
        if record is None or record.status not in {"queued", "running"}:
            return record
        if expected_job_id is not None and record.job_id != expected_job_id:
            return record
        try:
            failed = self.analysis_repo.fail(
                task_id=task_id,
                run_id=run_id,
                error_kind=safe_kind,
                error_message=safe_message,
                expected_job_id=expected_job_id,
            )
        except (DataAnalysisConflictError, DataAnalysisTransitionError):
            return self.analysis_repo.get_for_task(task_id, run_id)
        self._write_run_audit(
            kind="data.analysis.failed",
            record=failed,
            outcome="failed",
        )
        return failed

    def _write_run_audit(
        self,
        *,
        kind: str,
        record: DataAnalysisRunRecord,
        outcome: str,
    ) -> None:
        with connect(self.settings.db_path) as conn:
            _write_audit_row(
                conn,
                kind=kind,
                target_ref=record.id,
                inputs_hash=record.input_hash,
                outcome=outcome,
                detail=_audit_detail(record),
            )

    def _view_from_record(
        self,
        record: DataAnalysisRunRecord,
    ) -> DataAnalysisRunView:
        if record.status != "succeeded":
            return DataAnalysisRunView(record=record)
        artifact_id = record.result_artifact_id
        if artifact_id is None:
            raise DataAnalysisArtifactError(
                "successful data analysis has no artifact id"
            )
        artifact = self.artifact_repo.get_for_task(record.task_id, artifact_id)
        if artifact is None:
            raise DataAnalysisArtifactError("data analysis artifact not found")
        expected_relative = Path(
            "tasks",
            record.task_id,
            "data_analysis",
            f"{record.input_hash}.json",
        ).as_posix()
        if artifact["path"] != expected_relative:
            raise DataAnalysisArtifactError("data analysis artifact path drifted")
        if artifact["kind"] != DATA_ANALYSIS_ARTIFACT_KIND:
            raise DataAnalysisArtifactError("data analysis artifact kind drifted")
        if artifact["provenance"] != _artifact_provenance(record):
            raise DataAnalysisArtifactError(
                "data analysis artifact provenance drifted"
            )
        candidate = _verified_artifact_path(
            self.settings,
            task_id=record.task_id,
            relative_path=expected_relative,
        )
        raw_bytes = candidate.read_bytes()
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        artifact_hash = artifact.get("content_hash")
        if (
            record.result_content_hash is None
            or not isinstance(artifact_hash, str)
            or not hmac.compare_digest(artifact_hash, record.result_content_hash)
            or not hmac.compare_digest(actual_hash, record.result_content_hash)
        ):
            raise DataAnalysisArtifactError(
                "data analysis artifact content hash drifted"
            )
        try:
            raw = raw_bytes.decode("utf-8")
            result = json.loads(raw)
            if not isinstance(result, dict) or raw != _canonical_json(result):
                raise ValueError("artifact is not canonical JSON")
            identity = result.get("identity")
            if (
                not isinstance(identity, dict)
                or identity.get("input_hash") != record.input_hash
            ):
                raise ValueError("artifact identity does not match run")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataAnalysisArtifactError(
                "data analysis artifact payload is corrupt"
            ) from exc
        return DataAnalysisRunView(
            record=record,
            result=result,
            result_artifact_id=artifact_id,
            download_url=_download_url(record.task_id, artifact_id),
        )


def _dispatch_lock(db_path: Path, task_id: str) -> threading.Lock:
    key = (str(Path(db_path).resolve()), task_id)
    with _DISPATCH_LOCK_GUARD:
        return _DISPATCH_LOCKS.setdefault(key, threading.Lock())


def _result_envelope(
    *,
    record: DataAnalysisRunRecord,
    request: DataAnalysisRequest,
    mapping: DataSemanticMapping,
    analysis: Mapping[str, object],
) -> dict[str, object]:
    if analysis.get("schema_version") != DATA_ANALYSIS_SCHEMA_VERSION:
        raise DescriptiveInputError("descriptive result schema_version is invalid")
    fields = analysis.get("fields")
    if not isinstance(fields, list):
        raise DescriptiveInputError("descriptive result fields are invalid")
    projected: dict[str, object] = {}
    for section in request.sections:
        if section == "overview":
            projected[section] = analysis["dataset"]
        elif section == "target":
            projected[section] = analysis["target_distribution"]
        elif section == "missing":
            projected[section] = [
                {
                    "name": field["name"],
                    "row_count": field["row_count"],
                    "null_count": field["null_count"],
                    "null_rate": field["null_rate"],
                }
                for field in fields
            ]
        elif section == "distribution":
            projected[section] = fields
        elif section == "correlation":
            projected[section] = analysis["correlations"]
    envelope = {
        "schema_version": DATA_ANALYSIS_SCHEMA_VERSION,
        "identity": {
            "task_id": record.task_id,
            "dataset_id": record.dataset_id,
            "dataset_content_hash": record.dataset_content_hash,
            "workspace_revision": record.workspace_revision,
            "analysis_generation": record.analysis_generation,
            "semantic_mapping_hash": record.semantic_mapping_hash,
            "config_hash": record.config_hash,
            "producer_version": record.producer_version,
            "input_hash": record.input_hash,
        },
        "request": request.identity_config(),
        "semantics": data_semantic_mapping_to_dict(mapping),
        "analysis": projected,
    }
    _canonical_json(envelope)
    return envelope


def _value_sanitizers(
    *,
    task_id: str,
    dataset_hash: str,
    sensitive_columns: frozenset[str],
) -> dict[str, Callable[[TaggedScalar], Mapping[str, object]]]:
    sanitizers: dict[str, Callable[[TaggedScalar], Mapping[str, object]]] = {}
    for field_name in sorted(sensitive_columns):

        def sanitize(value: TaggedScalar, *, field: str = field_name):
            canonical_value = _canonical_json(value)
            token = hashlib.sha256(
                (
                    f"{_TOKEN_NAMESPACE}\x00{task_id}\x00{dataset_hash}\x00"
                    f"{field}\x00{canonical_value}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            return {"type": "string", "value": f"token:{token}"}

        sanitizers[field_name] = sanitize
    return sanitizers


def _suppress_sensitive_analysis_values(
    report: dict[str, object],
    *,
    sensitive_columns: frozenset[str],
) -> dict[str, object]:
    """Remove numeric distributions and correlation cells for sensitive fields."""

    if not sensitive_columns:
        return report
    sanitized = deepcopy(report)
    fields = sanitized.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if (
                not isinstance(field, dict)
                or str(field.get("name")) not in sensitive_columns
            ):
                continue
            field["numeric"] = None
            field["histogram"] = None
            field["sensitive_value_policy"] = (
                "frequency_tokenized_numeric_distribution_suppressed"
            )

    correlations = sanitized.get("correlations")
    if not isinstance(correlations, dict):
        return sanitized
    names = [str(name) for name in correlations.get("columns") or []]
    kept_indices = [
        index for index, name in enumerate(names) if name not in sensitive_columns
    ]
    correlations["columns"] = [names[index] for index in kept_indices]
    for matrix_name in ("values", "pair_counts", "reasons"):
        matrix = correlations.get(matrix_name)
        if not isinstance(matrix, list):
            continue
        correlations[matrix_name] = [
            [row[column_index] for column_index in kept_indices]
            for row_index in kept_indices
            if row_index < len(matrix)
            and isinstance((row := matrix[row_index]), list)
            and all(column_index < len(row) for column_index in kept_indices)
        ]
    return sanitized


def _artifact_provenance(record: DataAnalysisRunRecord) -> dict[str, object]:
    return {
        "schema_version": DATA_ANALYSIS_SCHEMA_VERSION,
        "task_id": record.task_id,
        "dataset_id": record.dataset_id,
        "dataset_content_hash": record.dataset_content_hash,
        "analysis_generation": record.analysis_generation,
        "semantic_mapping_hash": record.semantic_mapping_hash,
        "config_hash": record.config_hash,
        "producer_version": record.producer_version,
        "input_hash": record.input_hash,
    }


def _audit_detail(
    record: DataAnalysisRunRecord,
    *,
    artifact_id: str | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {
        "task_id": record.task_id,
        "input_hash": record.input_hash,
        "dataset_content_hash": record.dataset_content_hash,
        "semantic_mapping_hash": record.semantic_mapping_hash,
        "config_hash": record.config_hash,
    }
    if artifact_id is not None:
        detail["artifact_id"] = artifact_id
    return detail


def _verified_artifact_path(settings, *, task_id: str, relative_path: str) -> Path:
    try:
        declared_root = (settings.tasks_dir / task_id).absolute()
        if declared_root.is_symlink():
            raise OSError("task root is a symlink")
        candidate = (settings.workspace / relative_path).absolute()
        candidate.relative_to(declared_root)
        cursor = declared_root
        for part in candidate.relative_to(declared_root).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise OSError("artifact path contains a symlink")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(declared_root.resolve(strict=True))
        if not resolved.is_file():
            raise OSError("artifact is not a file")
        return resolved
    except (OSError, RuntimeError, ValueError) as exc:
        raise DataAnalysisArtifactError("data analysis artifact not found") from exc


def _download_url(task_id: str, artifact_id: str) -> str:
    return (
        f"/api/tasks/{quote(task_id, safe='')}/task-artifacts/"
        f"{quote(artifact_id, safe='')}/download"
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataAnalysisArtifactError(
            "data analysis result is not strict JSON"
        ) from exc


def _canonical_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise DataAnalysisRequestError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _canonical_error_kind(value: object) -> str:
    raw = str(value or "data_analysis_failed").strip().lower()
    normalized = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in raw
    ).strip("_")
    return normalized[:96] or "data_analysis_failed"


def _safe_error_message(value: object) -> str:
    message = str(value or "data analysis failed").replace("\x00", "").strip()
    return (message or "data analysis failed")[:1000]


def _execution_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, DataAnalysisExecutionStaleError):
        return "stale_data_analysis_identity", _safe_error_message(str(exc))
    if isinstance(exc, DatasetContentDriftError):
        return "dataset_content_drift", "active dataset integrity verification failed"
    if isinstance(exc, DescriptiveBudgetError):
        detail = exc.to_detail()
        return str(detail["kind"]), _safe_error_message(str(exc))
    if isinstance(exc, DescriptiveInputError):
        return "descriptive_input_error", _safe_error_message(str(exc))
    return "data_analysis_failed", "data analysis failed"


__all__ = [
    "DATA_ANALYSIS_ARTIFACT_KIND",
    "DATA_ANALYSIS_JOB_KIND",
    "DATA_ANALYSIS_SECTIONS",
    "DataAnalysisActiveJobError",
    "DataAnalysisArtifactError",
    "DataAnalysisDispatch",
    "DataAnalysisExecutionStaleError",
    "DataAnalysisNotFoundError",
    "DataAnalysisRequest",
    "DataAnalysisRequestError",
    "DataAnalysisRetryRequiredError",
    "DataAnalysisRunView",
    "DataAnalysisService",
    "DataAnalysisServiceError",
    "DataAnalysisWorkspacePreconditionError",
]
