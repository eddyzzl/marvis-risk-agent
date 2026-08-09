from __future__ import annotations

from dataclasses import asdict
from ipaddress import ip_address
import logging
from pathlib import Path, PurePosixPath
import shutil
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, Response

from marvis.api_task_helpers import (
    normalize_source_dir,
    validate_model_identifier,
)
from marvis.api_task_payloads import task_report_download_filename
from marvis.data.task_filesystem_gc import (
    DATASET_IDENTITY_DIR,
    DATASET_TASK_DIR,
    TASK_DIR,
    VALIDATION_BATCH_SOURCE_DIR,
    TaskFilesystemGcTarget,
)
from marvis.data.validation_batch_upload_gc import (
    ValidationBatchUploadClaimError,
)
from marvis.repositories.tasks import TaskRepository
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VALIDATION_BATCH,
    TaskCreate,
)
from marvis.errors import conflict, not_found, payload_too_large, unprocessable
from marvis.model_algorithms import normalize_algorithm
from marvis.notebooks import close_live_notebook_session
from marvis.repositories.validation_batches import (
    ValidationBatchRepository,
    ValidationBatchStateConflict,
)
from marvis.repositories.validation_contracts import ValidationContractRepository
from marvis.safe_paths import assert_within, lexical_child_path, resolve_fixed_task_output
from marvis.state_machine import ConflictError
from marvis.validation_batch_runner import (
    mark_validation_batch_parent_running,
    run_validation_batch_job,
)
from marvis.validation_batch_ingress import MATERIAL_UPLOAD_MAX_FILES
from marvis.validation_batch_schemas import (
    CreateValidationBatchRequest,
    StartValidationBatchRequest,
)
from marvis.validation_materials import resolve_validation_material_paths


router = APIRouter(prefix="/api/validation-batches", tags=["validation-batches"])
logger = logging.getLogger(__name__)
_MATERIAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_MATERIAL_UPLOAD_FILES = MATERIAL_UPLOAD_MAX_FILES
_EXCEL_MATERIAL_SUFFIXES = frozenset({".xls", ".xlsx", ".xlsm"})


def _batch_repo(request: Request) -> ValidationBatchRepository:
    return ValidationBatchRepository(request.app.state.settings.db_path)


@router.post("/material-uploads", status_code=201)
async def upload_validation_batch_materials(
    request: Request,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(default=None),
) -> dict:
    """Stage one model's files under an opaque, batch-only ownership token."""

    if not files:
        raise unprocessable("at least one material file is required")
    if len(files) > _MAX_MATERIAL_UPLOAD_FILES:
        raise unprocessable(
            f"too many material files: max_files={_MAX_MATERIAL_UPLOAD_FILES}"
        )
    if relative_paths and len(relative_paths) != len(files):
        raise unprocessable("relative_paths count must match uploaded files count")

    upload_gc = request.app.state.validation_batch_upload_gc
    try:
        upload_gc.sweep_expired(limit=100)
    except Exception:
        # Existing expired rows remain durable and retryable; an opportunistic
        # sweep failure must not make a new, independently owned upload fail.
        logger.exception("validation batch expired upload sweep failed")
    upload_token, upload_dir = _new_batch_upload_dir(
        request.app.state.settings.workspace
    )
    settings = request.app.state.settings
    max_batch_bytes = settings.max_csv_upload_bytes
    batch_size_bytes = 0
    saved_files: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    try:
        for index, upload in enumerate(files):
            raw_relative_path = (
                relative_paths[index]
                if relative_paths and index < len(relative_paths)
                else upload.filename
            )
            relative_path = _validate_upload_relative_path(raw_relative_path)
            relative_path_text = relative_path.as_posix()
            if relative_path_text in seen_paths:
                raise unprocessable(
                    f"duplicate upload path: {relative_path_text}"
                )
            seen_paths.add(relative_path_text)
            destination = (upload_dir / Path(*relative_path.parts)).resolve()
            try:
                destination = assert_within(upload_dir, destination)
            except PermissionError as exc:
                raise unprocessable(
                    f"invalid upload path: {relative_path_text}"
                ) from exc
            max_file_bytes = _material_upload_file_limit(
                settings,
                relative_path,
            )
            size_bytes = await _save_upload_file(
                upload,
                destination,
                relative_path=relative_path_text,
                max_file_bytes=max_file_bytes,
                current_batch_bytes=batch_size_bytes,
                max_batch_bytes=max_batch_bytes,
            )
            batch_size_bytes += size_bytes
            saved_files.append(
                {
                    "relative_path": relative_path_text,
                    "size_bytes": size_bytes,
                }
            )
        upload_gc.register(upload_token)
    except Exception:
        _remove_batch_owned_path(
            request.app.state.settings.workspace,
            upload_dir,
        )
        raise
    return {
        "upload_token": upload_token,
        "source_dir": str(upload_dir),
        "files": saved_files,
    }


@router.delete("/material-uploads/{upload_token}", status_code=204)
def discard_validation_batch_materials(
    upload_token: str,
    request: Request,
) -> Response:
    try:
        token = _validate_batch_upload_token(upload_token)
        request.app.state.validation_batch_upload_gc.discard(token)
    except ValueError as exc:
        raise unprocessable(str(exc)) from exc
    return Response(status_code=204)


@router.post("", status_code=201)
def create_validation_batch(
    payload: CreateValidationBatchRequest,
    request: Request,
) -> dict:
    validate_model_identifier("batch_name", payload.batch_name)
    parent_source = _new_batch_source_dir(request.app.state.settings.workspace)
    child_payloads: list[TaskCreate] = []
    seen_materials: dict[Path, tuple[str, str]] = {}
    claimed_uploads: dict[str, Path] = {}
    reservation_owner = uuid.uuid4().hex
    try:
        requested_upload_sources: dict[str, str] = {}
        for item in payload.items:
            if item.upload_token is None:
                continue
            token = _validate_batch_upload_token(item.upload_token)
            if token in requested_upload_sources:
                raise unprocessable("同一批次不能重复使用 upload_token")
            requested_upload_sources[token] = item.source_dir

        if requested_upload_sources:
            try:
                claimed_uploads = (
                    request.app.state.validation_batch_upload_gc.reserve(
                        tuple(requested_upload_sources),
                        reservation_owner,
                    )
                )
            except ValidationBatchUploadClaimError as exc:
                raise unprocessable(str(exc)) from exc

        for token, requested_source_dir in requested_upload_sources.items():
            requested_path = _claimed_batch_upload_dir(
                request.app.state.settings.workspace,
                token,
                requested_source_dir,
            )
            if requested_path != claimed_uploads[token]:
                raise unprocessable(
                    "source_dir does not match durable validation batch upload ownership"
                )

        prepared_items: list[dict[str, object]] = []
        for ordinal, item in enumerate(payload.items, start=1):
            validate_model_identifier("model_name", item.model_name)
            if item.model_version:
                validate_model_identifier("model_version", item.model_version)
            try:
                algorithm = normalize_algorithm(item.algorithm, allow_empty=True)
            except ValueError as exc:
                raise unprocessable(str(exc)) from exc
            if item.upload_token is not None:
                source_dir = claimed_uploads[item.upload_token]
            else:
                source_dir = normalize_source_dir(
                    item.source_dir,
                    request.app.state.settings,
                )
            try:
                materials = resolve_validation_material_paths(
                    source_dir=source_dir,
                    notebook_path=item.notebook_path,
                    sample_path=item.sample_path,
                    pmml_path=item.pmml_path,
                    dictionary_path=item.dictionary_path,
                )
            except ValueError as exc:
                raise unprocessable(str(exc)) from exc
            for role, path in (
                ("Notebook", materials.notebook),
                ("样本", materials.sample),
                ("PMML", materials.pmml),
                ("数据字典", materials.dictionary),
            ):
                previous = seen_materials.get(path)
                if previous is not None:
                    previous_model, previous_role = previous
                    raise unprocessable(
                        "同一批次的模型材料路径不能复用："
                        f"{item.model_name} 的{role}与 {previous_model} 的{previous_role}相同"
                    )
                seen_materials[path] = (item.model_name, role)

            prepared_items.append(
                {
                    "ordinal": ordinal,
                    "item": item,
                    "algorithm": algorithm,
                    "source_dir": source_dir,
                    "materials": materials,
                }
            )

        for prepared in prepared_items:
            item = prepared["item"]
            source_dir = prepared["source_dir"]
            materials = prepared["materials"]
            if item.upload_token is not None:
                destination = parent_source / f"{int(prepared['ordinal']):02d}"
                relative_materials = {
                    "notebook": materials.notebook.relative_to(source_dir),
                    "sample": materials.sample.relative_to(source_dir),
                    "pmml": materials.pmml.relative_to(source_dir),
                    "dictionary": materials.dictionary.relative_to(source_dir),
                }
                source_dir.rename(destination)
                source_dir = destination
                materials = type(materials)(
                    notebook=destination / relative_materials["notebook"],
                    sample=destination / relative_materials["sample"],
                    pmml=destination / relative_materials["pmml"],
                    dictionary=destination / relative_materials["dictionary"],
                )
            child_payloads.append(TaskCreate(
                task_type=TASK_TYPE_VALIDATION,
                model_name=item.model_name.strip(),
                model_version=item.model_version.strip(),
                validator=payload.validator.strip(),
                source_dir=str(source_dir),
                algorithm=str(prepared["algorithm"]),
                run_mode="agent",
                target_col=item.target_col,
                score_col=item.score_col,
                split_col=item.split_col,
                time_col=item.time_col,
                notebook_path=str(materials.notebook),
                sample_path=str(materials.sample),
                pmml_path=str(materials.pmml),
                dictionary_path=str(materials.dictionary),
            ))

        batch = _batch_repo(request).create_batch(
            TaskCreate(
                task_type=TASK_TYPE_VALIDATION_BATCH,
                model_name=payload.batch_name.strip(),
                model_version="",
                validator=payload.validator.strip(),
                source_dir=str(parent_source),
                run_mode="agent",
            ),
            child_payloads,
            parent_source_relative_path=(
                f"validation-batches/{parent_source.name}"
            ),
            claimed_upload_tokens=tuple(claimed_uploads),
            claimed_upload_owner=reservation_owner,
        )
    except Exception:
        _remove_batch_owned_path(
            request.app.state.settings.workspace,
            parent_source,
        )
        for upload_token in claimed_uploads:
            try:
                request.app.state.validation_batch_upload_gc.abandon_reservation(
                    upload_token,
                    reservation_owner,
                )
            except Exception:
                logger.exception(
                    "failed to abandon validation batch upload reservation after create failure: %s",
                    upload_token,
                )
        raise

    try:
        TaskRepository(request.app.state.settings.db_path).add_agent_message(
            batch.parent_task_id,
            role="assistant",
            stage="batch_created",
            content=(
                f"已创建模型验证批次，共 {batch.item_count} 个模型。"
                "平台会逐项隔离执行；单个模型失败不会中断同批次其他模型。"
            ),
            metadata={"batch_status": batch.status, "item_count": batch.item_count},
        )
    except Exception as exc:
        # The parent/children/material ownership transaction has committed.
        # An optional conversational welcome must not turn that success into a
        # retryable-looking 500 that could make the UI create a duplicate batch.
        logger.warning(
            "validation batch created but welcome message persistence failed for %s: %s",
            batch.parent_task_id,
            exc,
        )
    return _batch_payload(request, batch.parent_task_id)


@router.get("/{parent_task_id}")
def get_validation_batch(parent_task_id: str, request: Request) -> dict:
    try:
        return _batch_payload(request, parent_task_id)
    except KeyError as exc:
        raise not_found("validation batch not found") from exc


@router.get("/{parent_task_id}/purge-preview")
def preview_validation_batch_purge(
    parent_task_id: str,
    request: Request,
) -> dict:
    try:
        preview = _batch_repo(request).purge_preview(parent_task_id)
        parent = TaskRepository(
            request.app.state.settings.db_path
        ).get_task(parent_task_id)
    except KeyError as exc:
        raise not_found("validation batch not found") from exc
    owned_material_tree_count = _owned_batch_material_tree_count(
        parent.source_dir,
        request.app.state.settings,
    )
    purge_summary = dict(preview["purge_summary"])
    purge_summary.update(
        {
            "child_tasks": len(preview["child_task_ids"]),
            "owned_batch_material_trees": owned_material_tree_count,
        }
    )
    # Match the generic task purge-preview contract: ``purge_summary`` is the
    # flat, user-displayable count map.  Internal paths and per-task summaries
    # stay server-side; batch-only scope is exposed as explicit safe counts.
    return {
        "parent_task_id": parent_task_id,
        "task_count": preview["task_count"],
        "child_task_count": len(preview["child_task_ids"]),
        "active_job_count": preview["active_job_count"],
        "owned_batch_material_tree_count": owned_material_tree_count,
        "purge_summary": purge_summary,
    }


@router.delete("/{parent_task_id}", status_code=204)
def delete_validation_batch(parent_task_id: str, request: Request) -> Response:
    repo = _batch_repo(request)
    task_repo = TaskRepository(request.app.state.settings.db_path)
    try:
        preview = repo.purge_preview(parent_task_id)
        tasks = {
            task_id: task_repo.get_task(task_id)
            for task_id in preview["family_task_ids"]
        }
    except KeyError as exc:
        raise not_found("validation batch not found") from exc

    settings = request.app.state.settings
    targets_by_task: dict[str, tuple[TaskFilesystemGcTarget, ...]] = {}
    for task_id, task in tasks.items():
        targets = list(_task_directory_gc_targets(task_id, settings))
        if task_id == parent_task_id:
            batch_source = _owned_batch_source_relative_path(task.source_dir, settings)
            if batch_source is not None:
                targets.append(
                    TaskFilesystemGcTarget(
                        VALIDATION_BATCH_SOURCE_DIR,
                        batch_source,
                    )
                )
        targets_by_task[task_id] = tuple(targets)

    datasets_root = getattr(settings, "datasets_dir", None)

    def validate_dataset_source_path(relative_path: str) -> None:
        if datasets_root is not None:
            lexical_child_path(datasets_root, relative_path)

    try:
        result = repo.purge_batch(
            parent_task_id,
            validate_dataset_source_path=validate_dataset_source_path,
            task_fs_targets_by_task=targets_by_task,
        )
    except KeyError as exc:
        raise not_found("validation batch not found") from exc
    except PermissionError as exc:
        raise unprocessable(
            "dataset source path escapes the datasets directory"
        ) from exc
    except ConflictError as exc:
        raise conflict(str(exc)) from exc

    for task_id in result["family_task_ids"]:
        try:
            close_live_notebook_session(task_id)
        except Exception as exc:
            # DB purge and durable GC intent already committed.  Keep the HTTP
            # result truthful and let process/session teardown finish best-effort.
            logger.warning(
                "notebook session close failed after validation batch purge for %s: %s",
                task_id,
                exc,
            )
    _sweep_batch_dataset_sources(request, result)
    _sweep_batch_task_filesystems(request, result)
    return Response(status_code=204)


@router.get("/{parent_task_id}/summary/download")
def download_validation_batch_summary(
    parent_task_id: str,
    request: Request,
) -> FileResponse:
    repo = _batch_repo(request)
    try:
        batch = repo.get_batch(parent_task_id)
    except KeyError as exc:
        raise not_found("validation batch not found") from exc
    path = _current_batch_summary_path(request, batch)
    if path is None:
        raise not_found("validation batch summary not generated")
    parent = TaskRepository(
        request.app.state.settings.db_path
    ).get_task(parent_task_id)
    return FileResponse(
        path,
        media_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        filename=task_report_download_filename(parent, "_批次汇总.xlsx"),
    )


@router.post("/{parent_task_id}/start", status_code=202)
def start_validation_batch(
    parent_task_id: str,
    payload: StartValidationBatchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _batch_repo(request)
    task_repo = TaskRepository(request.app.state.settings.db_path)
    try:
        batch = repo.get_batch(parent_task_id)
    except KeyError as exc:
        raise not_found("validation batch not found") from exc
    if batch.status not in {
        "created",
        "awaiting_confirmation",
        "partial_failure",
        "failed",
    }:
        raise conflict(f"validation batch is not startable: {batch.status}")
    if batch.status in {"awaiting_confirmation", "partial_failure", "failed"}:
        contract_repo = ValidationContractRepository(
            request.app.state.settings.db_path
        )
        pending_items = [
            item
            for item in repo.list_items(parent_task_id)
            if item.status == "awaiting_confirmation"
            and (
                (record := contract_repo.get(item.child_task_id)) is None
                or record.status != "ready"
            )
        ]
        if pending_items:
            raise conflict(
                f"批次仍有 {len(pending_items)} 个模型等待确认输入合同"
            )

    try:
        job_id = task_repo.start_job(parent_task_id, "validation_batch")
    except ConflictError as exc:
        raise conflict(str(exc)) from exc
    try:
        _batch, previous_status = repo.claim_start(parent_task_id)
    except KeyError as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value="validation batch not found",
        )
        raise not_found("validation batch not found") from exc
    except ValidationBatchStateConflict as exc:
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise conflict(str(exc)) from exc

    try:
        mark_validation_batch_parent_running(task_repo, parent_task_id)
    except Exception as exc:
        try:
            repo.restore_start_claim(batch)
        except Exception:
            pass
        task_repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value="unable to start validation batch parent task",
        )
        raise conflict("validation batch parent task is not startable") from exc

    retry_failed_items = previous_status in {"partial_failure", "failed"}
    job_runner = getattr(
        request.app.state,
        "validation_batch_job_runner",
        run_validation_batch_job,
    )
    background_tasks.add_task(
        job_runner,
        job_id=job_id,
        settings=request.app.state.settings,
        parent_task_id=parent_task_id,
        model_id=payload.model_id,
        effort=payload.effort,
        manual_review_base_url=_trusted_manual_review_base_url(request),
        retry_failed_items=retry_failed_items,
    )
    return {
        "parent_task_id": parent_task_id,
        "job_id": job_id,
        "status": "accepted",
        "batch_url": f"/api/validation-batches/{parent_task_id}",
        "message": "validation batch dispatched; poll batch_url for progress",
    }


def _batch_payload(request: Request, parent_task_id: str) -> dict:
    repo = _batch_repo(request)
    contract_repo = ValidationContractRepository(
        request.app.state.settings.db_path
    )
    batch = repo.get_batch(parent_task_id)
    batch_payload = asdict(batch)
    batch_payload.pop("summary_path", None)
    batch_payload["summary_download_url"] = (
        f"/api/validation-batches/{parent_task_id}/summary/download"
        if _current_batch_summary_path(request, batch) is not None
        else ""
    )
    return {
        "batch": batch_payload,
        "items": [
            _batch_item_payload(request, parent_task_id, item, contract_repo)
            for item in repo.list_items(parent_task_id)
        ],
    }


def _batch_item_payload(
    request: Request,
    parent_task_id: str,
    item,
    contract_repo: ValidationContractRepository,
) -> dict:
    tasks_dir = request.app.state.settings.tasks_dir
    return {
        **asdict(item),
        "input_contract_status": (
            contract.status
            if (contract := contract_repo.get(item.child_task_id)) is not None
            else ""
        ),
        "input_contract_url": (
            f"/api/tasks/{item.child_task_id}/validation-input-contract"
        ),
        "manual_review_url": (
            f"/?task={parent_task_id}&item={item.child_task_id}"
        ),
        "word_report_download_url": (
            f"/api/tasks/{item.child_task_id}/report/download"
            if resolve_fixed_task_output(
                tasks_dir,
                item.child_task_id,
                "validation_report.docx",
            )
            is not None
            else ""
        ),
        "analysis_download_url": (
            f"/api/tasks/{item.child_task_id}/analysis/download"
            if resolve_fixed_task_output(
                tasks_dir,
                item.child_task_id,
                "validation.xlsx",
            )
            is not None
            else ""
        ),
    }


def _batch_summary_path(request: Request, parent_task_id: str) -> Path:
    return (
        request.app.state.settings.tasks_dir
        / parent_task_id
        / "outputs"
        / "validation_batch_summary.xlsx"
    )


def _current_batch_summary_path(request: Request, batch) -> Path | None:
    expected_path = _batch_summary_path(request, batch.parent_task_id)
    if (
        batch.status not in {"completed", "partial_failure", "failed"}
        or not batch.summary_path
        or Path(batch.summary_path) != expected_path
    ):
        return None
    return resolve_fixed_task_output(
        request.app.state.settings.tasks_dir,
        batch.parent_task_id,
        "validation_batch_summary.xlsx",
    )


def _trusted_manual_review_base_url(request: Request) -> str:
    hostname = str(request.url.hostname or "").lower()
    if hostname == "localhost":
        return str(request.base_url).rstrip("/")
    try:
        if ip_address(hostname).is_loopback:
            return str(request.base_url).rstrip("/")
    except ValueError:
        pass
    return ""


def _new_batch_source_dir(workspace: Path) -> Path:
    root = _batch_material_root(workspace)
    for _ in range(10):
        path = root / uuid.uuid4().hex
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("failed to allocate validation batch source directory")


def _batch_material_root(workspace: Path) -> Path:
    uploads_root = Path(workspace).resolve() / "material_uploads"
    if uploads_root.is_symlink():
        raise RuntimeError("material uploads root must not be a symlink")
    uploads_root.mkdir(parents=True, exist_ok=True)
    if uploads_root.resolve() != uploads_root.absolute():
        raise RuntimeError("material uploads root resolved outside owned path")

    root = uploads_root / "validation-batches"
    if root.is_symlink():
        raise RuntimeError("validation batch material root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root.absolute():
        raise RuntimeError("validation batch material root resolved outside owned path")
    return root


def _batch_staging_root(workspace: Path) -> Path:
    root = _batch_material_root(workspace) / "staging"
    if root.is_symlink():
        raise RuntimeError("validation batch staging root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root.absolute():
        raise RuntimeError("validation batch staging root resolved outside owned path")
    return root


def _new_batch_upload_dir(workspace: Path) -> tuple[str, Path]:
    root = _batch_staging_root(workspace)
    for _ in range(10):
        upload_token = uuid.uuid4().hex
        path = root / upload_token
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return upload_token, path
    raise RuntimeError("failed to allocate validation batch upload directory")


def _validate_batch_upload_token(upload_token: str) -> str:
    token = str(upload_token or "")
    if (
        len(token) != 32
        or token != token.lower()
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise unprocessable("invalid validation batch upload_token")
    return token


def _batch_staging_dir_for_token(workspace: Path, upload_token: str) -> Path:
    token = _validate_batch_upload_token(upload_token)
    return _batch_staging_root(workspace) / token


def _claimed_batch_upload_dir(
    workspace: Path,
    upload_token: str,
    requested_source_dir: str,
) -> Path:
    expected = _batch_staging_dir_for_token(workspace, upload_token)
    if expected.is_symlink() or not expected.is_dir():
        raise unprocessable("validation batch upload_token is missing or already claimed")
    requested = Path(requested_source_dir).expanduser().absolute()
    if requested != expected.absolute():
        raise unprocessable("source_dir does not match validation batch upload_token")
    try:
        resolved = expected.resolve(strict=True)
    except OSError as exc:
        raise unprocessable("validation batch upload_token is unavailable") from exc
    if resolved != expected.absolute():
        raise unprocessable("validation batch upload_token resolved outside staging")
    return resolved


def _validate_upload_relative_path(raw_path: str | None) -> PurePosixPath:
    value = str(raw_path or "").replace("\\", "/").strip()
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise unprocessable(f"invalid upload path: {value}")
    return path


def _material_upload_file_limit(settings, relative_path: PurePosixPath) -> int:
    if relative_path.suffix.lower() in _EXCEL_MATERIAL_SUFFIXES:
        return settings.max_excel_upload_bytes
    return settings.max_csv_upload_bytes


async def _save_upload_file(
    upload: UploadFile,
    destination: Path,
    *,
    relative_path: str,
    max_file_bytes: int,
    current_batch_bytes: int,
    max_batch_bytes: int,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size_bytes = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(_MATERIAL_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_file_bytes:
                    raise payload_too_large(
                        "验证批次材料单文件大小超过上限："
                        f"file={relative_path}, limit_bytes={max_file_bytes}, "
                        f"actual_bytes={size_bytes}"
                    )
                total_bytes = current_batch_bytes + size_bytes
                if total_bytes > max_batch_bytes:
                    raise payload_too_large(
                        "验证批次材料总大小超过上限："
                        f"limit_bytes={max_batch_bytes}, actual_bytes={total_bytes}"
                    )
                output.write(chunk)
    finally:
        await upload.close()
    return size_bytes


def _remove_batch_owned_path(workspace: Path, path: Path) -> None:
    try:
        root = _batch_material_root(workspace).absolute()
    except RuntimeError as exc:
        logger.error("refused validation batch cleanup through unsafe root: %s", exc)
        return
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        logger.error("refused validation batch cleanup outside owned root: %s", candidate)
        return
    parts = relative.parts
    token = parts[-1] if parts else ""
    controlled = (
        (len(parts) == 1 or (len(parts) == 2 and parts[0] == "staging"))
        and len(token) == 32
        and token == token.lower()
        and all(character in "0123456789abcdef" for character in token)
    )
    if not controlled:
        logger.error("refused malformed validation batch cleanup target: %s", candidate)
        return
    try:
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.exists():
            shutil.rmtree(candidate)
    except OSError as exc:
        logger.warning("validation batch owned path cleanup failed for %s: %s", candidate, exc)


def _task_directory_gc_targets(task_id: str, settings) -> tuple[TaskFilesystemGcTarget, ...]:
    lexical_child_path(settings.tasks_dir, task_id)
    lexical_child_path(settings.datasets_dir, task_id)
    lexical_child_path(settings.datasets_dir, f"{task_id}/.source-identities")
    return (
        TaskFilesystemGcTarget(TASK_DIR, task_id),
        TaskFilesystemGcTarget(
            DATASET_IDENTITY_DIR,
            f"{task_id}/.source-identities",
        ),
        TaskFilesystemGcTarget(DATASET_TASK_DIR, task_id),
    )


def _owned_batch_source_relative_path(source_dir: str, settings) -> str | None:
    root = _batch_material_root(settings.workspace).absolute()
    source = Path(str(source_dir or "")).absolute()
    if source.parent != root:
        return None
    token = source.name
    if (
        len(token) != 32
        or token != token.lower()
        or any(character not in "0123456789abcdef" for character in token)
    ):
        return None
    relative = f"validation-batches/{token}"
    try:
        lexical_child_path(
            Path(settings.workspace).resolve() / "material_uploads",
            relative,
        )
    except PermissionError:
        return None
    return relative


def _owned_batch_material_tree_count(source_dir: str, settings) -> int:
    """Return one when the owned batch tree contains uploaded material.

    A manually configured batch still owns an empty bookkeeping directory, but
    its external source paths are never deletion targets and must not be
    described as uploaded material in the confirmation dialog.
    """

    if _owned_batch_source_relative_path(source_dir, settings) is None:
        return 0
    source = Path(str(source_dir or "")).absolute()
    try:
        return int(source.is_dir() and next(source.iterdir(), None) is not None)
    except OSError:
        # Fail closed in the user-facing preview: an unreadable owned tree may
        # still contain sensitive material, so disclose it as part of scope.
        return 1


def _sweep_batch_dataset_sources(request: Request, result: dict) -> None:
    source_paths = tuple(result.get("dataset_source_paths", ()))
    if not source_paths:
        return
    collector = getattr(request.app.state, "dataset_source_gc", None)
    if collector is None:
        logger.warning(
            "dataset source GC unavailable after validation batch purge; "
            "durable candidates remain queued"
        )
        return
    try:
        report = collector.sweep(
            source_paths=source_paths,
            limit=min(len(source_paths), 500),
            force=True,
        )
    except Exception as exc:
        logger.warning(
            "dataset source GC failed after validation batch purge; "
            "durable candidates remain queued: %s",
            exc,
        )
        return
    if report.deferred or report.quarantined:
        logger.warning(
            "dataset source GC deferred after validation batch purge: "
            "deferred=%d quarantined=%d",
            report.deferred,
            report.quarantined,
        )


def _sweep_batch_task_filesystems(request: Request, result: dict) -> None:
    collector = getattr(request.app.state, "task_filesystem_gc", None)
    candidate_count = int(result.get("task_fs_gc_candidates", 0))
    if not candidate_count:
        return
    if collector is None:
        logger.warning(
            "task filesystem GC unavailable after validation batch purge; "
            "durable candidates remain queued"
        )
        return
    for task_id in result.get("family_task_ids", ()):
        try:
            report = collector.sweep(
                origin_task_id=task_id,
                limit=min(candidate_count, 500),
                force=True,
            )
        except Exception as exc:
            logger.warning(
                "task filesystem GC failed after validation batch purge for %s; "
                "durable candidates remain queued: %s",
                task_id,
                exc,
            )
            continue
        if report.deferred or report.quarantined:
            logger.warning(
                "task filesystem GC deferred after validation batch purge for %s: "
                "deferred=%d quarantined=%d",
                task_id,
                report.deferred,
                report.quarantined,
            )


__all__ = ["router"]
