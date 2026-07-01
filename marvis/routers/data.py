from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from marvis.api_data_payloads import (
    dataset_payload,
    dataset_preview_profiles,
    join_plan_payload,
    masked_preview_records,
)
from marvis.artifacts import ArtifactUnitOfWork
from marvis.api_task_helpers import dispatch_platform_hook
from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import KeyPair
from marvis.data.errors import (
    DataBackendError,
    DataIngestError,
    DedupRequiredError,
    FanOutError,
    JoinNotConfirmedError,
)
from marvis.data.excel_ingest import ingest_sheet, list_sheets
from marvis.data.join_engine import JoinEngine
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository


router = APIRouter(prefix="/api", tags=["data"])
DATASET_ROLES = {"sample", "feature", "derived", "unknown"}
DATASET_PREVIEW_MAX_ROWS = 500
DEDUP_STRATEGIES = {None, "first", "last", "agg_mean", "agg_max"}


def _upload_artifact_name(filename: str | None) -> str:
    source_name = Path(filename or "upload").name
    source_path = Path(source_name)
    stem = source_path.stem or "upload"
    return f"{stem}_{uuid4().hex[:8]}{source_path.suffix.lower()}"


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
        raise HTTPException(status_code=404, detail="task not found") from exc


def _coerce_key_pairs(raw_pairs: list, *, anchor, feature) -> list[KeyPair]:
    anchor_columns = {column.name for column in anchor.columns}
    feature_columns = {column.name for column in feature.columns}
    pairs = []
    for item in raw_pairs:
        anchor_col = str(item.get("anchor_col") or "")
        feature_col = str(item.get("feature_col") or "")
        if anchor_col not in anchor_columns or feature_col not in feature_columns:
            raise HTTPException(status_code=422, detail="key_pairs contain unknown columns")
        pairs.append(
            KeyPair(
                anchor_col=anchor_col,
                feature_col=feature_col,
                match_method=str(item.get("match_method") or "exact"),
                transform_side=str(item.get("transform_side") or "both"),
                match_rate=float(item.get("match_rate") or 0.0),
                resolved_by=str(item.get("resolved_by") or "user"),
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


@router.post("/tasks/{task_id}/datasets/upload", status_code=201)
async def upload_task_dataset(
    task_id: str,
    request: Request,
    file: UploadFile = File(...),
    role: str = Form("unknown"),
    sheet: str | None = Form(None),
) -> dict:
    _require_task(request, task_id)
    if role not in DATASET_ROLES:
        raise HTTPException(status_code=422, detail="invalid dataset role")
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    upload_dir = request.app.state.settings.datasets_dir / task_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_uow = ArtifactUnitOfWork()
    upload_artifact = upload_uow.stage_file(upload_dir, _upload_artifact_name(file.filename))
    upload_artifact.path.write_bytes(await file.read())
    upload_path = upload_artifact.final_path
    try:
        upload_uow.promote_all()
        suffix = upload_path.suffix.lower()
        if suffix in {".xlsx", ".xlsm"}:
            sheets = list_sheets(upload_path)
            if sheet:
                if sheet not in sheets:
                    raise HTTPException(status_code=422, detail="sheet not found")
                sheets = [sheet]
            datasets = []
            reports = []
            out_dir = request.app.state.settings.datasets_dir / task_id / "excel"
            out_dir.mkdir(parents=True, exist_ok=True)
            uow = ArtifactUnitOfWork()
            staged_sheets = []
            try:
                with tempfile.TemporaryDirectory(prefix=".excel_ingest_", dir=out_dir) as scratch:
                    scratch_dir = Path(scratch)
                    for sheet_name in sheets:
                        parquet_path, report = ingest_sheet(upload_path, sheet_name, scratch_dir)
                        artifact = uow.stage_file(out_dir, parquet_path.name)
                        shutil.move(parquet_path, artifact.path)
                        staged_sheets.append((artifact.final_path, report))
                        reports.append({
                            "sheet": report.sheet,
                            "header_rows": report.header_rows,
                            "data_start_row": report.data_start_row,
                            "flattened_columns": report.flattened_columns,
                            "warnings": [],
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
            datasets = [registry.register_from_upload(task_id, upload_path, role=role)]
            reports = []
    except HTTPException:
        upload_uow.rollback()
        raise
    except (DataBackendError, DataIngestError, ValueError) as exc:
        upload_uow.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, request: Request, rows: int = 50) -> dict:
    if rows < 1 or rows > DATASET_PREVIEW_MAX_ROWS:
        raise HTTPException(status_code=422, detail="rows is outside allowed range")
    _repo_data, backend, registry, _join_engine = _data_runtime(request)
    try:
        path = registry.resolve_path(dataset_id)
        dataset = registry.get(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    frame = backend.read_frame(path, nrows=rows + 1)
    truncated = len(frame) > rows or dataset.row_count > rows
    frame = frame.head(rows)
    return {
        "columns": [str(column) for column in frame.columns],
        "column_profiles": dataset_preview_profiles(dataset),
        "rows": masked_preview_records(frame, dataset),
        "truncated": truncated,
    }


@router.post("/tasks/{task_id}/joins/propose", status_code=201)
async def propose_join(task_id: str, request: Request) -> dict:
    _require_task(request, task_id)
    payload = await request.json()
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
        raise HTTPException(
            status_code=422,
            detail="anchor_dataset_id and feature_dataset_ids are required",
        )
    _repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        anchor = registry.get(anchor_id)
        features = [registry.get(feature_id) for feature_id in feature_ids]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset not found") from exc
    if anchor.task_id != task_id or any(feature.task_id != task_id for feature in features):
        raise HTTPException(status_code=404, detail="dataset not found")
    plan = join_engine.propose_join_plan(anchor_id, feature_ids, task_id)
    return join_plan_payload(plan)


@router.get("/joins/{join_plan_id}")
def get_join_plan(join_plan_id: str, request: Request) -> dict:
    repo_data, _backend, _registry, _join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="join plan not found") from exc
    return join_plan_payload(plan)


@router.post("/joins/{join_plan_id}/confirm")
async def confirm_join_plan(join_plan_id: str, request: Request) -> dict:
    payload = await request.json()
    feature_id = str(payload.get("feature_id") or payload.get("feature_dataset_id") or "")
    if not feature_id:
        raise HTTPException(status_code=422, detail="feature_id is required")
    dedup_strategy = payload.get("dedup_strategy")
    if dedup_strategy not in DEDUP_STRATEGIES:
        raise HTTPException(status_code=422, detail="invalid dedup_strategy")
    confirmed = bool(payload.get("confirmed", True))
    repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
        spec = next(item for item in plan.joins if item.feature_dataset_id == feature_id)
    except (KeyError, StopIteration) as exc:
        raise HTTPException(status_code=404, detail="join plan or feature not found") from exc
    if payload.get("key_pairs"):
        anchor = registry.get(plan.anchor_dataset_id)
        feature = registry.get(feature_id)
        spec.key_pairs = _coerce_key_pairs(payload["key_pairs"], anchor=anchor, feature=feature)
        spec.diagnostics = join_engine.diagnose_join(
            anchor,
            registry.resolve_path(anchor.id),
            feature,
            registry.resolve_path(feature.id),
            spec.key_pairs,
            seed=0,
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
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
def execute_join_plan(join_plan_id: str, request: Request) -> dict:
    repo_data, _backend, registry, join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="join plan not found") from exc
    if plan.status == "executed":
        raise HTTPException(status_code=409, detail="join plan already executed")
    anchor = registry.get(plan.anchor_dataset_id)
    try:
        result = join_engine.execute_join_plan(
            join_plan_id,
            out_dir=request.app.state.settings.datasets_dir / plan.task_id / "joins",
        )
    except (JoinNotConfirmedError, DedupRequiredError, FanOutError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DataBackendError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "result_dataset_id": result.id,
        "anchor_rows": anchor.row_count,
        "joined_rows": result.row_count,
        "fan_out": False,
        "warnings": [],
    }
