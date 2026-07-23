from __future__ import annotations

import traceback
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from marvis.errors import (
    bad_request,
    conflict,
    not_found,
    payload_too_large,
    precondition_failed,
    precondition_required,
    unprocessable,
)

from marvis.api_data_payloads import (
    dataset_payload,
    dataset_preview_profiles,
    dataset_preview_records,
    join_plan_payload,
)
from marvis.api_schemas import (
    DataWorkspaceSnapshotResponse,
    DataWorkspaceUpdateRequest,
)
from marvis.artifacts import ArtifactUnitOfWork
from marvis.api_stage_helpers import start_task_job
from marvis.api_task_helpers import dispatch_platform_hook
from marvis.job_cancellation import (
    clear_pending_job_cancellation,
    JobCancelled,
    register_job_cancellation,
    unregister_job_cancellation,
)
from marvis.job_heartbeat import heartbeat_job
from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import HASH_ALGO_CANDIDATES, KeyPair
from marvis.data.errors import (
    DataBackendError,
    DatasetContentDriftError,
    DataIngestError,
    DatasetTooLargeError,
    DedupRequiredError,
    FanOutError,
    InvalidDatasetPathError,
    JoinNotConfirmedError,
)
from marvis.data.excel_ingest import (
    detect_excel_container_format,
    detect_excel_container_format_from_prefix,
    ingest_sheet,
    list_sheets,
    new_excel_artifact_name,
    require_excel_format,
)
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    DataWorkspaceSnapshot,
)
from marvis.db import DatasetRepository, TaskRepository
from marvis.repositories.data_workspace import (
    DataWorkspaceDataError,
    DataWorkspaceDatasetNotFound,
    DataWorkspaceRepository,
    DataWorkspaceRevisionConflict,
)


router = APIRouter(prefix="/api", tags=["data"])
DATASET_ROLES = {"sample", "feature", "derived", "performance", "unknown"}
DATASET_PREVIEW_MAX_ROWS = 500
DEDUP_STRATEGIES = {None, "first", "last", "agg_mean", "agg_max"}
# TST-2: chunk size for streaming an upload to disk instead of reading the
# whole file into memory with UploadFile.read()/file.file.read().
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024
# TST-2 (roadmap-1e): local-path registration only accepts these extensions --
# an explicit whitelist, not "whatever register_from_upload happens to parse".
DATASET_PATH_SUFFIXES = {".csv", ".xls", ".xlsx", ".xlsm", ".parquet"}
_EXCEL_UPLOAD_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
_MATCH_METHODS = frozenset({
    "exact",
    "exact_lower",
    "date",
    *(f"hash:{algorithm}" for algorithm in HASH_ALGO_CANDIDATES),
})
_TRANSFORM_SIDES = frozenset({"anchor", "feature", "both"})


def _upload_artifact_name(filename: str | None) -> str:
    source_name = Path(filename or "upload").name
    source_path = Path(source_name)
    stem = source_path.stem or "upload"
    return f"{stem}_{uuid4().hex[:8]}{source_path.suffix.lower()}"


def _max_upload_bytes_for_suffix(settings, suffix: str) -> int:
    if suffix in _EXCEL_UPLOAD_SUFFIXES:
        return settings.max_excel_upload_bytes
    return settings.max_csv_upload_bytes


def _verified_excel_format(path: Path, settings) -> str | None:
    container_format = detect_excel_container_format(path)
    if container_format is None:
        return None
    size_bytes = path.stat().st_size
    if size_bytes > settings.max_excel_upload_bytes:
        raise DatasetTooLargeError(
            reason="检测为 Excel 的文件大小超过上限",
            limit=settings.max_excel_upload_bytes,
            actual=size_bytes,
        )
    return require_excel_format(path)


def _reject_by_content_length(request: Request, max_bytes: int) -> None:
    """Fast pre-check: if the client sent a Content-Length header, reject
    obviously oversized requests before any bytes are read. Content-Length
    covers the whole multipart body (form fields + boundaries), which is
    always >= the uploaded file's own size, so this can only reject requests
    that are already too large -- it can never wrongly *accept* an oversized
    file, and is not itself sufficient (a client can omit or lie about
    Content-Length), which is why the streaming write below re-checks the
    actual bytes written as the authoritative guard.
    """
    content_length = request.headers.get("content-length")
    if content_length is None:
        return
    try:
        declared_bytes = int(content_length)
    except ValueError:
        return
    if declared_bytes > max_bytes:
        raise DatasetTooLargeError(
            reason="上传内容大小超过上限",
            limit=max_bytes,
            actual=declared_bytes,
        )


def _stream_upload_to_path(
    file: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    max_excel_bytes: int | None = None,
) -> int:
    """Stream ``file`` to ``destination`` in fixed-size chunks instead of
    ``file.file.read()`` (whole-file-into-memory). Sync read of the underlying
    SpooledTemporaryFile: this endpoint is a plain `def` so FastAPI already
    runs it in a worker thread (PERF-1); `file.file` is the raw BinaryIO, no
    event loop needed. Enforces `max_bytes` against the cumulative bytes
    actually written -- the authoritative guard (Content-Length can be absent
    or spoofed, see `_reject_by_content_length`).
    """
    total_bytes = 0
    effective_max_bytes = max_bytes
    detected_excel = False
    with destination.open("wb") as output:
        while True:
            chunk = file.file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            if total_bytes == 0 and max_excel_bytes is not None:
                detected_excel = (
                    detect_excel_container_format_from_prefix(chunk) is not None
                )
                if detected_excel:
                    effective_max_bytes = min(effective_max_bytes, max_excel_bytes)
            total_bytes += len(chunk)
            if total_bytes > effective_max_bytes:
                raise DatasetTooLargeError(
                    reason=(
                        "检测为 Excel 的上传文件大小超过上限"
                        if detected_excel
                        else "上传文件大小超过上限"
                    ),
                    limit=effective_max_bytes,
                    actual=total_bytes,
                )
            output.write(chunk)
    return total_bytes


def _resolve_local_dataset_path(raw_path: str) -> Path:
    """TST-2 (roadmap-1e): validate a client-supplied local absolute path for
    server-side dataset registration.

    Guards, in order: non-empty; absolute (a relative path is ambiguous about
    the server's cwd and is rejected outright); resolves cleanly (catches
    embedded NUL bytes / OS-level malformed paths); exists and is a regular
    file, not a symlink to elsewhere or a directory (checked on the *resolved*
    path so a symlink pointing e.g. at /etc/passwd cannot slip through);
    extension is in the whitelist. The resolved path is deliberately allowed
    to live anywhere on the local filesystem (this endpoint is loopback-only,
    see the module docstring on the router below) -- it is copied into
    ``datasets_root`` rather than registered in place, so no traversal check
    against ``datasets_root`` is needed here; the copy destination itself is
    always a freshly generated name under the task's own upload directory.
    """
    value = str(raw_path or "").strip()
    if not value:
        raise InvalidDatasetPathError("path is required")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise InvalidDatasetPathError(f"path must be absolute: {value}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise InvalidDatasetPathError(f"path does not exist: {value}") from exc
    if not resolved.is_file():
        raise InvalidDatasetPathError(f"path is not a regular file: {value}")
    if resolved.suffix.lower() not in DATASET_PATH_SUFFIXES:
        raise InvalidDatasetPathError(
            "unsupported file extension: "
            f"{resolved.suffix or '(none)'} (allowed: {sorted(DATASET_PATH_SUFFIXES)})"
        )
    return resolved


def _copy_local_dataset_path(
    source: Path,
    destination: Path,
    *,
    max_bytes: int,
    max_excel_bytes: int | None = None,
) -> int:
    """Chunked copy (mirrors ``_stream_upload_to_path``) so a large local file
    is never fully buffered in memory, and so it is subject to the same size
    guardrail as an HTTP upload of the same file type."""
    total_bytes = 0
    effective_max_bytes = max_bytes
    detected_excel = False
    with source.open("rb") as input_file, destination.open("wb") as output:
        while True:
            chunk = input_file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            if total_bytes == 0 and max_excel_bytes is not None:
                detected_excel = (
                    detect_excel_container_format_from_prefix(chunk) is not None
                )
                if detected_excel:
                    effective_max_bytes = min(effective_max_bytes, max_excel_bytes)
            total_bytes += len(chunk)
            if total_bytes > effective_max_bytes:
                raise DatasetTooLargeError(
                    reason=(
                        "检测为 Excel 的本地路径文件大小超过上限"
                        if detected_excel
                        else "本地路径注册文件大小超过上限"
                    ),
                    limit=effective_max_bytes,
                    actual=total_bytes,
                )
            output.write(chunk)
    return total_bytes


def _data_runtime(request: Request):
    settings = request.app.state.settings
    datasets_root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    repo = DatasetRepository(settings.db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(repo, backend, datasets_root)
    aligner = ColumnAligner(backend)
    join_engine = JoinEngine(backend, aligner, registry, repo)
    return repo, backend, registry, join_engine


def _require_task(request: Request, task_id: str) -> None:
    try:
        TaskRepository(request.app.state.settings.db_path).get_task(task_id)
    except KeyError as exc:
        raise not_found("task not found") from exc


def _data_workspace_payload(
    snapshot: DataWorkspaceSnapshot,
) -> DataWorkspaceSnapshotResponse:
    mapping = snapshot.semantic_mapping
    return DataWorkspaceSnapshotResponse(
        schema_version=snapshot.schema_version,
        task_id=snapshot.task_id,
        revision=snapshot.revision,
        active_dataset_id=snapshot.active_dataset_id,
        active_dataset_content_hash=snapshot.active_dataset_content_hash,
        analysis_generation=snapshot.analysis_generation,
        page=snapshot.page,
        selected_field=snapshot.selected_field,
        semantic_mapping={
            "target_col": mapping.target_col,
            "field_roles": dict(mapping.field_roles),
            "business_names": dict(mapping.business_names),
        },
        updated_at=snapshot.updated_at,
    )


def _verified_task_dataset(
    request: Request,
    *,
    task_id: str,
    dataset_id: str,
):
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    try:
        dataset = registry.get(dataset_id)
    except KeyError as exc:
        raise not_found("dataset not found") from exc
    if dataset.task_id != task_id:
        raise not_found("dataset not found")
    try:
        registry.resolve_verified_path(dataset_id)
    except DatasetContentDriftError as exc:
        raise conflict(str(exc)) from exc
    return dataset


def _parse_if_match(if_match: str | None) -> int:
    if if_match is None:
        raise precondition_required("If-Match header is required")
    if not if_match.isascii() or not if_match.isdecimal():
        raise bad_request("If-Match must be a non-negative integer")
    try:
        expected_revision = int(if_match)
    except ValueError as exc:
        raise bad_request("If-Match must be a non-negative integer") from exc
    if expected_revision < 0:
        raise bad_request("If-Match must be a non-negative integer")
    return expected_revision


def _join_async_requested(payload: dict) -> bool:
    return any(
        _coerce_bool(payload.get(key))
        for key in ("async", "async_execute", "background")
    )


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _coerce_key_pairs(raw_pairs: list, *, anchor, feature) -> list[KeyPair]:
    anchor_columns = {column.name for column in anchor.columns}
    feature_columns = {column.name for column in feature.columns}
    pairs = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise unprocessable("key_pairs must contain objects")
        anchor_col = str(item.get("anchor_col") or "")
        feature_col = str(item.get("feature_col") or "")
        if anchor_col not in anchor_columns or feature_col not in feature_columns:
            raise unprocessable("key_pairs contain unknown columns")
        match_method = str(item.get("match_method") or "exact").strip()
        transform_side = str(item.get("transform_side") or "both").strip()
        if match_method not in _MATCH_METHODS:
            raise unprocessable("invalid key pair match_method")
        if transform_side not in _TRANSFORM_SIDES:
            raise unprocessable("invalid key pair transform_side")
        pairs.append(
            KeyPair(
                anchor_col=anchor_col,
                feature_col=feature_col,
                match_method=match_method,
                transform_side=transform_side,
                # Client-supplied deterministic evidence is never authoritative.
                match_rate=0.0,
                resolved_by="user",
            )
        )
    return pairs


@router.get("/tasks/{task_id}/datasets")
def list_task_datasets(task_id: str, request: Request) -> dict:
    _require_task(request, task_id)
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    return {
        "datasets": [
            dataset_payload(dataset)
            for dataset in registry.list_for_task(task_id)
        ]
    }


@router.get(
    "/tasks/{task_id}/data-workspace",
    response_model=DataWorkspaceSnapshotResponse,
)
def get_data_workspace(
    task_id: str,
    request: Request,
) -> DataWorkspaceSnapshotResponse:
    _require_task(request, task_id)
    try:
        snapshot = DataWorkspaceRepository(
            request.app.state.settings.db_path
        ).get_or_default(task_id)
    except DataWorkspaceDatasetNotFound as exc:
        raise not_found("dataset not found") from exc
    except KeyError as exc:
        raise not_found("task not found") from exc
    if snapshot.active_dataset_id is not None:
        _verified_task_dataset(
            request,
            task_id=task_id,
            dataset_id=snapshot.active_dataset_id,
        )
    return _data_workspace_payload(snapshot)


@router.put(
    "/tasks/{task_id}/data-workspace",
    response_model=DataWorkspaceSnapshotResponse,
)
def update_data_workspace(
    task_id: str,
    payload: DataWorkspaceUpdateRequest,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> DataWorkspaceSnapshotResponse:
    _require_task(request, task_id)
    expected_revision = _parse_if_match(if_match)
    if payload.active_dataset_id is not None:
        _verified_task_dataset(
            request,
            task_id=task_id,
            dataset_id=payload.active_dataset_id,
        )
    repo = DataWorkspaceRepository(request.app.state.settings.db_path)
    try:
        draft = DataWorkspaceDraft(
            active_dataset_id=payload.active_dataset_id,
            active_dataset_content_hash=payload.active_dataset_content_hash,
            page=payload.page,
            selected_field=payload.selected_field,
            semantic_mapping=DataSemanticMapping(
                target_col=payload.semantic_mapping.target_col,
                field_roles=payload.semantic_mapping.field_roles,
                business_names=payload.semantic_mapping.business_names,
            ),
        )
        snapshot = repo.save(
            task_id,
            draft,
            expected_revision=expected_revision,
        )
    except DataWorkspaceRevisionConflict as exc:
        raise precondition_failed(str(exc)) from exc
    except DataWorkspaceDatasetNotFound as exc:
        raise not_found("dataset not found") from exc
    except KeyError as exc:
        raise not_found("task not found") from exc
    except DataWorkspaceDataError as exc:
        raise unprocessable(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise unprocessable(str(exc)) from exc
    return _data_workspace_payload(snapshot)


@router.post("/tasks/{task_id}/datasets/upload", status_code=201)
def upload_task_dataset(
    task_id: str,
    request: Request,
    file: UploadFile = File(...),
    role: str = Form("unknown"),
    sheet: str | None = Form(None),
) -> dict:
    _require_task(request, task_id)
    if role not in DATASET_ROLES:
        raise unprocessable("invalid dataset role")
    settings = request.app.state.settings
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    upload_suffix = Path(file.filename or "").suffix.lower()
    max_upload_bytes = _max_upload_bytes_for_suffix(settings, upload_suffix)
    try:
        _reject_by_content_length(request, max_upload_bytes)
    except DatasetTooLargeError as exc:
        raise payload_too_large(str(exc)) from exc
    upload_dir = settings.datasets_dir / task_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_uow = ArtifactUnitOfWork()
    upload_artifact = upload_uow.stage_file(upload_dir, _upload_artifact_name(file.filename))
    # TST-2: stream to disk in fixed-size chunks instead of file.file.read()
    # (whole-file-into-memory) -- see _stream_upload_to_path. The cumulative
    # byte count is the authoritative size guard; Content-Length above is only
    # a fast pre-check and can be absent or spoofed.
    try:
        _stream_upload_to_path(
            file,
            upload_artifact.path,
            max_bytes=max_upload_bytes,
            max_excel_bytes=settings.max_excel_upload_bytes,
        )
    except DatasetTooLargeError as exc:
        upload_uow.rollback()
        raise payload_too_large(str(exc)) from exc
    upload_path = upload_artifact.final_path
    try:
        upload_uow.promote_all()
        suffix = upload_path.suffix.lower()
        detected_excel_format = _verified_excel_format(upload_path, settings)
        if detected_excel_format is not None or suffix in _EXCEL_UPLOAD_SUFFIXES:
            sheets = list_sheets(upload_path)
            if sheet:
                if sheet not in sheets:
                    raise unprocessable("sheet not found")
                sheets = [sheet]
            datasets = []
            reports = []
            out_dir = settings.datasets_dir / task_id / "excel"
            out_dir.mkdir(parents=True, exist_ok=True)
            uow = ArtifactUnitOfWork()
            staged_sheets = []
            try:
                with tempfile.TemporaryDirectory(prefix=".excel_ingest_", dir=out_dir) as scratch:
                    scratch_dir = Path(scratch)
                    for sheet_name in sheets:
                        parquet_path, report = ingest_sheet(
                            upload_path,
                            sheet_name,
                            scratch_dir,
                            max_rows=settings.max_excel_rows,
                        )
                        artifact = uow.stage_file(
                            out_dir,
                            new_excel_artifact_name(report.sheet),
                        )
                        shutil.move(parquet_path, artifact.path)
                        staged_sheets.append((artifact.final_path, report))
                        excel_warnings = []
                        if report.suspected_truncated_id_columns:
                            excel_warnings.append(
                                "疑似长数字 ID 列已被 Excel 存为数值并截断精度，建议改为文本格式重新导入："
                                + ", ".join(report.suspected_truncated_id_columns)
                            )
                        reports.append({
                            "sheet": report.sheet,
                            "header_rows": report.header_rows,
                            "data_start_row": report.data_start_row,
                            "flattened_columns": report.flattened_columns,
                            "warnings": excel_warnings,
                        })
                datasets = uow.finalize_with_connection(
                    registry.transaction,
                    lambda conn: [
                        registry.register_existing_on_connection(
                            conn,
                            parquet_path,
                            task_id=task_id,
                            role=role,
                        )
                        for parquet_path, _report in staged_sheets
                    ],
                )
            except Exception:
                uow.rollback()
                raise
        else:
            datasets = [
                registry.register_from_upload(
                    task_id,
                    upload_path,
                    role=role,
                    max_excel_rows=settings.max_excel_rows,
                )
            ]
            reports = []
            csv_report = registry.last_csv_ingest_report
            if csv_report is not None:
                csv_warnings = []
                if csv_report.encoding_used != "utf-8-sig":
                    csv_warnings.append(f"文件按 {csv_report.encoding_used} 编码解析（非 UTF-8）。")
                if csv_report.long_id_columns:
                    csv_warnings.append(
                        "以下长数字 ID 列已按文本读取，避免精度截断："
                        + ", ".join(csv_report.long_id_columns)
                    )
                reports.append({
                    "encoding_used": csv_report.encoding_used,
                    "long_id_columns": list(csv_report.long_id_columns),
                    "warnings": csv_warnings,
                })
    except HTTPException:
        upload_uow.rollback()
        raise
    except DatasetTooLargeError as exc:
        upload_uow.rollback()
        raise payload_too_large(str(exc)) from exc
    except (DataBackendError, DataIngestError, ValueError) as exc:
        upload_uow.rollback()
        raise bad_request(str(exc)) from exc
    except Exception:
        upload_uow.rollback()
        raise
    upload_uow.commit()
    for dataset in datasets:
        dispatch_platform_hook(
            getattr(request.app.state, "hook_dispatcher", None),
            "dataset.registered",
            {
                "task_id": task_id,
                "dataset_id": dataset.id,
                "role": dataset.role,
            },
            task_id=task_id,
        )
    return {
        "datasets": [dataset_payload(dataset) for dataset in datasets],
        "reports": reports,
    }


@router.post("/tasks/{task_id}/datasets/register-path", status_code=201)
def register_task_dataset_from_path(
    task_id: str,
    request: Request,
    payload: dict = Body(default_factory=dict),
) -> dict:
    """TST-2 (roadmap-1e): register a dataset directly from a local absolute
    path on the machine running MARVIS, skipping an HTTP upload entirely --
    for single-machine deployments where the file already lives on disk.

    Loopback-only: this is a POST, and the app-wide ``_local_access_guard``
    middleware (marvis/app.py) already rejects every non-GET/HEAD/OPTIONS
    request from a non-local client with 403 before it reaches any router --
    the same guard every other write endpoint in this module relies on, so no
    additional per-endpoint check is added here (importing app.py's
    ``_is_local_client`` back into this module would be a circular import,
    since app.py imports this router).
    """
    _require_task(request, task_id)
    role = str(payload.get("role") or "unknown")
    if role not in DATASET_ROLES:
        raise unprocessable("invalid dataset role")
    settings = request.app.state.settings
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    try:
        source_path = _resolve_local_dataset_path(str(payload.get("path") or ""))
    except InvalidDatasetPathError as exc:
        raise unprocessable(str(exc)) from exc

    upload_dir = settings.datasets_dir / task_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_uow = ArtifactUnitOfWork()
    upload_artifact = upload_uow.stage_file(
        upload_dir, _upload_artifact_name(source_path.name)
    )
    max_bytes = _max_upload_bytes_for_suffix(settings, source_path.suffix.lower())
    try:
        _copy_local_dataset_path(
            source_path,
            upload_artifact.path,
            max_bytes=max_bytes,
            max_excel_bytes=settings.max_excel_upload_bytes,
        )
    except DatasetTooLargeError as exc:
        upload_uow.rollback()
        raise payload_too_large(str(exc)) from exc
    except OSError as exc:
        upload_uow.rollback()
        raise unprocessable(f"cannot read path: {exc}") from exc
    upload_path = upload_artifact.final_path
    try:
        upload_uow.promote_all()
        _verified_excel_format(upload_path, settings)
        # register_from_upload already gives GAP-7 content-hash dedup / idempotency
        # for free -- registering the same local path twice reuses the existing
        # dataset's parquet instead of duplicating it, exactly like a repeat HTTP
        # upload of the same file would.
        dataset = registry.register_from_upload(
            task_id,
            upload_path,
            role=role,
            max_excel_rows=settings.max_excel_rows,
            audit_factory=lambda registered: {
                "kind": "dataset.registered_from_path",
                "target_ref": registered.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": task_id,
                    "dataset_id": registered.id,
                    "role": registered.role,
                    "source_path": str(source_path),
                    "content_hash": registered.content_hash,
                },
            },
        )
    except HTTPException:
        upload_uow.rollback()
        raise
    except DatasetTooLargeError as exc:
        upload_uow.rollback()
        raise payload_too_large(str(exc)) from exc
    except (DataBackendError, DataIngestError, ValueError) as exc:
        upload_uow.rollback()
        raise bad_request(str(exc)) from exc
    except Exception:
        upload_uow.rollback()
        raise
    upload_uow.commit()
    dispatch_platform_hook(
        getattr(request.app.state, "hook_dispatcher", None),
        "dataset.registered",
        {
            "task_id": task_id,
            "dataset_id": dataset.id,
            "role": dataset.role,
        },
        task_id=task_id,
    )
    return {"datasets": [dataset_payload(dataset)]}


def _dataset_preview_payload(
    dataset_id: str,
    request: Request,
    rows: int,
    *,
    task_id: str | None = None,
) -> dict:
    if rows < 1 or rows > DATASET_PREVIEW_MAX_ROWS:
        raise unprocessable("rows is outside allowed range")
    _repo_data, backend, registry, _join_engine = _data_runtime(request)
    try:
        dataset = registry.get(dataset_id)
    except KeyError as exc:
        raise not_found("dataset not found") from exc
    if task_id is not None and dataset.task_id != task_id:
        raise not_found("dataset not found")
    try:
        path = registry.resolve_verified_path(dataset_id)
    except KeyError as exc:
        raise not_found("dataset not found") from exc
    except DatasetContentDriftError as exc:
        raise conflict(str(exc)) from exc
    frame = backend.read_frame(path, nrows=rows + 1)
    truncated = len(frame) > rows or dataset.row_count > rows
    frame = frame.head(rows)
    return {
        "columns": [str(column) for column in frame.columns],
        "column_profiles": dataset_preview_profiles(dataset, frame),
        "rows": dataset_preview_records(frame),
        "truncated": truncated,
    }


@router.get("/tasks/{task_id}/datasets/{dataset_id}/preview")
def preview_task_dataset(
    task_id: str,
    dataset_id: str,
    request: Request,
    rows: int = 50,
) -> dict:
    _require_task(request, task_id)
    return _dataset_preview_payload(dataset_id, request, rows, task_id=task_id)


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, request: Request, rows: int = 50) -> dict:
    return _dataset_preview_payload(dataset_id, request, rows)


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, request: Request) -> FileResponse:
    """Download the exact registered dataset artifact produced by a workflow."""
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    try:
        path = registry.resolve_path(dataset_id)
    except KeyError as exc:
        raise not_found("dataset not found") from exc
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
    )


@router.post("/tasks/{task_id}/joins/propose", status_code=201)
def propose_join(
    task_id: str,
    request: Request,
    payload: dict = Body(default_factory=dict),
) -> dict:
    _require_task(request, task_id)
    anchor_id = str(payload.get("anchor_dataset_id") or payload.get("anchor_id") or "")
    feature_ids = [
        str(item)
        for item in (
            payload.get("feature_dataset_ids")
            or payload.get("feature_ids")
            or []
        )
    ]
    if not anchor_id or not feature_ids:
        raise unprocessable("anchor_dataset_id and feature_dataset_ids are required")
    _repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        anchor = registry.get(anchor_id)
        features = [registry.get(feature_id) for feature_id in feature_ids]
    except KeyError as exc:
        raise not_found("dataset not found") from exc
    if anchor.task_id != task_id or any(feature.task_id != task_id for feature in features):
        raise not_found("dataset not found")
    plan = join_engine.propose_join_plan(anchor_id, feature_ids, task_id)
    return join_plan_payload(plan)


@router.get("/joins/{join_plan_id}")
def get_join_plan(join_plan_id: str, request: Request) -> dict:
    repo_data, _backend, _registry, _join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
    except KeyError as exc:
        raise not_found("join plan not found") from exc
    return join_plan_payload(plan)


@router.post("/joins/{join_plan_id}/confirm")
def confirm_join_plan(
    join_plan_id: str,
    request: Request,
    payload: dict = Body(default_factory=dict),
) -> dict:
    feature_id = str(payload.get("feature_id") or payload.get("feature_dataset_id") or "")
    if not feature_id:
        raise unprocessable("feature_id is required")
    dedup_strategy = payload.get("dedup_strategy")
    if dedup_strategy not in DEDUP_STRATEGIES:
        raise unprocessable("invalid dedup_strategy")
    confirmed = bool(payload.get("confirmed", True))
    repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
        spec = next(item for item in plan.joins if item.feature_dataset_id == feature_id)
    except (KeyError, StopIteration) as exc:
        raise not_found("join plan or feature not found") from exc
    if payload.get("key_pairs"):
        anchor = registry.get(plan.anchor_dataset_id)
        feature = registry.get(feature_id)
        spec.key_pairs = join_engine.recompute_key_pairs(
            anchor,
            registry.resolve_path(anchor.id),
            feature,
            registry.resolve_path(feature.id),
            _coerce_key_pairs(payload["key_pairs"], anchor=anchor, feature=feature),
            seed=0,
        )
        spec.diagnostics = join_engine.diagnose_join(
            anchor,
            registry.resolve_path(anchor.id),
            feature,
            registry.resolve_path(feature.id),
            spec.key_pairs,
            seed=0,
            recompute_match=True,
        )
        repo_data.update_join_spec(plan.id, spec)
    try:
        if confirmed:
            join_engine.confirm_join_spec(
                join_plan_id,
                feature_id,
                dedup_strategy=dedup_strategy,
            )
        else:
            spec.confirmed = False
            spec.dedup_strategy = dedup_strategy
            repo_data.update_join_spec(plan.id, spec)
    except DedupRequiredError as exc:
        raise conflict(str(exc)) from exc
    if confirmed:
        dispatch_platform_hook(
            getattr(request.app.state, "hook_dispatcher", None),
            "join.confirmed",
            {
                "task_id": plan.task_id,
                "join_plan_id": join_plan_id,
                "feature_id": feature_id,
                "confirmed": True,
            },
            task_id=plan.task_id,
        )
    return join_plan_payload(repo_data.load_join_plan(join_plan_id))


@router.post("/joins/{join_plan_id}/execute")
def execute_join_plan(
    join_plan_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    response: Response,
    payload: dict = Body(default_factory=dict),
) -> dict:
    repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
    except KeyError as exc:
        raise not_found("join plan not found") from exc
    if plan.status == "executed":
        raise conflict("join plan already executed")
    anchor = registry.get(plan.anchor_dataset_id)
    if _join_async_requested(payload):
        task_repo = TaskRepository(request.app.state.settings.db_path)
        job_id = start_task_job(task_repo, plan.task_id, "join")
        background_tasks.add_task(
            _run_join_execute_job,
            job_id,
            request.app.state.settings.db_path,
            request.app.state.settings.datasets_dir,
            join_plan_id,
        )
        response.status_code = 202
        return {
            "status": "accepted",
            "job_id": job_id,
            "task_id": plan.task_id,
            "join_plan_id": join_plan_id,
            "anchor_rows": anchor.row_count,
            "message": "join execution dispatched; poll GET /api/tasks/{task_id}",
        }
    task_repo = TaskRepository(request.app.state.settings.db_path)
    job_id = start_task_job(task_repo, plan.task_id, "join")
    if task_repo.mark_job_running(job_id) is False:
        clear_pending_job_cancellation(job_id)
        raise conflict("join execution job is no longer active")
    cancel_token = register_job_cancellation(job_id)
    try:
        with heartbeat_job(task_repo, job_id):
            result = join_engine.execute_join_plan(
                join_plan_id,
                out_dir=request.app.state.settings.datasets_dir / plan.task_id / "joins",
                cancel_token=cancel_token,
            )
    except JobCancelled:
        task_repo.finish_job(job_id, status="cancelled")
        raise conflict("join execution cancelled") from None
    except (JoinNotConfirmedError, DedupRequiredError, FanOutError) as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise conflict(str(exc)) from exc
    except DataBackendError as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise bad_request(str(exc)) from exc
    except Exception as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    else:
        task_repo.finish_job(job_id, status="succeeded")
    finally:
        unregister_job_cancellation(job_id, cancel_token)
    return {
        "result_dataset_id": result.id,
        "anchor_rows": anchor.row_count,
        "joined_rows": result.row_count,
        "fan_out": False,
        "warnings": [],
    }


def _run_join_execute_job(
    job_id: str,
    db_path: Path,
    datasets_dir: Path,
    join_plan_id: str,
) -> None:
    task_repo = TaskRepository(db_path)
    if task_repo.mark_job_running(job_id) is False:
        clear_pending_job_cancellation(job_id)
        return
    repo_data = DatasetRepository(db_path)
    backend = DataBackend(datasets_dir)
    registry = DatasetRegistry(repo_data, backend, datasets_dir)
    join_engine = JoinEngine(backend, ColumnAligner(backend), registry, repo_data)
    cancel_token = register_job_cancellation(job_id)
    try:
        with heartbeat_job(task_repo, job_id):
            plan = repo_data.load_join_plan(join_plan_id)
            join_engine.execute_join_plan(
                join_plan_id,
                out_dir=Path(datasets_dir) / plan.task_id / "joins",
                cancel_token=cancel_token,
            )
    except JobCancelled:
        task_repo.finish_job(job_id, status="cancelled")
    except Exception as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
    else:
        task_repo.finish_job(job_id, status="succeeded")
    finally:
        unregister_job_cancellation(job_id, cancel_token)
