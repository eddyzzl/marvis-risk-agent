from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath
import shutil

from fastapi import APIRouter, File, Request, UploadFile
from marvis.errors import conflict, payload_too_large, unprocessable

from marvis.api_scan_helpers import (
    REQUIRED_SCAN_MATERIALS,
    SourceDirectoryScanError,
    material_candidates_payload,
    perform_scan_task,
    scan_hook_payload,
    validate_material_selection,
)
from marvis.api_schemas import MaterialSelectionRequest
from marvis.api_stage_helpers import start_task_job
from marvis.api_task_payloads import task_payload
from marvis.api_task_helpers import (
    dispatch_platform_hook,
    get_task_or_404,
    reject_if_task_has_active_job,
    validation_batch_parent_id,
)
from marvis.repositories.tasks import TaskRepository
from marvis.domain import (
    TASK_TYPE_VALIDATION,
    TASK_TYPE_VALIDATION_BATCH,
    TaskRecord,
    TaskStatus,
)
from marvis.job_heartbeat import heartbeat_job
from marvis.repositories.validation_contracts import ValidationContractRepository


router = APIRouter(prefix="/api", tags=["scans"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _material_source_admission(request, task):
    from marvis.modeling_holdout.guards import assert_public_path, record_exposure
    settings = request.app.state.settings
    assert_public_path(settings.workspace, task.source_dir)
    return lambda path: record_exposure(settings.db_path, settings.workspace, path)


def _reject_validation_batch_task(
    repo: TaskRepository,
    task: TaskRecord,
) -> None:
    if task.task_type == TASK_TYPE_VALIDATION_BATCH:
        raise unprocessable(
            "validation_batch 父任务不能使用通用模型验证扫描；"
            "请使用专用批次 API /api/validation-batches。"
        )
    if validation_batch_parent_id(repo, task.id) is not None:
        raise conflict(
            "批次内模型验证子任务不能通过通用扫描或材料选择接口重跑；"
            "请从专用批次流程继续。"
        )


MATERIAL_SUPPLEMENT_MAX_FILES = 20
MATERIAL_SUPPLEMENT_CHUNK_SIZE = 1024 * 1024
_EXCEL_SUPPLEMENT_SUFFIXES = frozenset({".xls", ".xlsx", ".xlsm"})


def _supplement_upload_dir_parent(request: Request, task: TaskRecord) -> Path | None:
    """The task's own upload directory, when its materials were uploaded.

    Only a directory the platform created under ``material_uploads`` accepts
    supplementary files; a task bound to the user's own folder is never
    written into (the user adds the file there and refreshes the list).
    """
    uploads_root = (
        Path(request.app.state.settings.workspace).resolve() / "material_uploads"
    )
    try:
        source_dir = Path(task.source_dir).resolve()
    except (TypeError, ValueError, OSError):
        return None
    if source_dir == uploads_root or not source_dir.is_relative_to(uploads_root):
        return None
    return source_dir if source_dir.is_dir() else None


def _material_payload_with_upload_flag(request: Request, task: TaskRecord, payload: dict) -> dict:
    return {
        **payload,
        "supplement_upload_supported": _supplement_upload_dir_parent(request, task) is not None,
    }


def _new_supplement_dir(source_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for index in range(100):
        name = f"补充材料-{stamp}" if index == 0 else f"补充材料-{stamp}-{index}"
        candidate = source_dir / name
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise conflict("补充材料目录分配失败，请稍后重试。")


def _supplement_file_name(raw_name: str | None) -> str:
    name = PurePosixPath(str(raw_name or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."} or name.startswith("."):
        raise unprocessable(f"文件名无效：{raw_name or ''}")
    return name


@router.get("/tasks/{task_id}/materials")
def task_material_candidates(
    task_id: str, request: Request, pmml_path: str | None = None
) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    _reject_validation_batch_task(repo, task)
    try:
        payload = material_candidates_payload(task, pmml_path_override=pmml_path, source_admission=_material_source_admission(request, task))
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise unprocessable(f"source dir invalid: {exc}") from exc
    return _material_payload_with_upload_flag(request, task, payload)


@router.post("/tasks/{task_id}/materials/files", status_code=201)
async def upload_task_supplement_materials(
    task_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
) -> dict:
    """Add corrected materials (e.g. a dictionary with importance) to a task.

    Eval round 1 (S1): a failed material check said "补齐后重新上传", but the
    task had nowhere to upload. Files land in a new 补充材料-<time> folder of
    the task's upload directory, so earlier files stay untouched; selecting
    them still goes through PUT /materials (which invalidates old evidence).
    """
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    if task.task_type != TASK_TYPE_VALIDATION:
        raise unprocessable("只有模型验证任务可以补传验证材料。")
    _reject_validation_batch_task(repo, task)
    reject_if_task_has_active_job(repo, task_id)
    if task.status in {TaskStatus.RUNNING, TaskStatus.COMPUTING_METRICS}:
        raise conflict(f"任务正在运行，不能补传材料：{task.status.value}")
    source_dir = _supplement_upload_dir_parent(request, task)
    if source_dir is None:
        raise conflict("这个任务的材料来自本地目录；请把补好的文件放进该目录，再点「刷新列表」。")
    if not files:
        raise unprocessable("请至少选择一个文件。")
    if len(files) > MATERIAL_SUPPLEMENT_MAX_FILES:
        raise unprocessable(f"一次最多补传 {MATERIAL_SUPPLEMENT_MAX_FILES} 个文件。")
    names = [_supplement_file_name(upload.filename) for upload in files]
    if len(set(names)) != len(names):
        raise unprocessable("同一次补传里有重名文件，请改名后再传。")
    settings = request.app.state.settings
    target_dir = _new_supplement_dir(source_dir)
    saved: list[dict] = []
    try:
        for upload, name in zip(files, names):
            limit = (
                settings.max_excel_upload_bytes
                if PurePosixPath(name).suffix.lower() in _EXCEL_SUPPLEMENT_SUFFIXES
                else settings.max_csv_upload_bytes
            )
            destination = target_dir / name
            size_bytes = 0
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(MATERIAL_SUPPLEMENT_CHUNK_SIZE)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > limit:
                        raise payload_too_large(
                            f"文件超过上传上限：{name}（上限 {limit // (1024 * 1024)} MB）"
                        )
                    output.write(chunk)
            await upload.close()
            saved.append({
                "relative_path": destination.relative_to(source_dir).as_posix(),
                "size_bytes": size_bytes,
            })
    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    task = repo.get_task(task_id)
    try:
        payload = material_candidates_payload(
            task, source_admission=_material_source_admission(request, task)
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise unprocessable(f"source dir invalid: {exc}") from exc
    return {
        "files": saved,
        "materials": _material_payload_with_upload_flag(request, task, payload),
    }


@router.put("/tasks/{task_id}/materials")
def update_task_materials(
    task_id: str,
    payload: MaterialSelectionRequest,
    request: Request,
) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    _reject_validation_batch_task(repo, task)
    job_id = start_task_job(repo, task_id, "material_selection")
    if not repo.mark_job_running(job_id):
        repo.finish_job(job_id, status="failed")
        raise conflict("material selection job could not be started")
    try:
        with heartbeat_job(repo, job_id):
            task = repo.get_task(task_id)
            if task.status in {
                TaskStatus.RUNNING,
                TaskStatus.COMPUTING_METRICS,
            }:
                raise conflict(
                    f"cannot update materials for task in status {task.status.value}"
                )
            selection = validate_material_selection(
                task,
                {
                    "notebook_path": payload.notebook_path,
                    "sample_path": payload.sample_path,
                    "pmml_path": payload.pmml_path,
                    "dictionary_path": payload.dictionary_path,
                },
            )
            if (
                _is_v2_validation_task(task)
                and hasattr(repo, "transaction")
                and hasattr(repo, "update_material_paths_on_connection")
            ):
                with repo.transaction() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    current = conn.execute(
                        """
                        SELECT notebook_path, sample_path, pmml_path, dictionary_path
                          FROM tasks
                         WHERE id = ?
                        """,
                        (task_id,),
                    ).fetchone()
                    if current is None:
                        raise KeyError(f"Task not found: {task_id}")
                    selection_changed = any(
                        str(current[field] or "") != selection[field]
                        for _role, _label, field in REQUIRED_SCAN_MATERIALS
                    )
                    repo.update_material_paths_on_connection(
                        conn,
                        task_id,
                        **selection,
                        begin_immediate=False,
                    )
                    if selection_changed:
                        ValidationContractRepository(
                            request.app.state.settings.db_path
                        ).invalidate_for_material_change_on_connection(conn, task_id)
                task = repo.get_task(task_id)
            else:
                task = repo.update_material_paths(task_id, **selection)
            response = {
                "task": task_payload(repo, task, request.app.state.settings.tasks_dir),
                "materials": material_candidates_payload(task, source_admission=_material_source_admission(request, task)),
            }
    except ValueError as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise unprocessable(str(exc)) from exc
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise
    repo.finish_job(job_id, status="succeeded")
    return response


@router.post("/tasks/{task_id}/scan")
def scan_task(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    _reject_validation_batch_task(repo, task)
    job_id = start_task_job(repo, task_id, "scan")
    if not repo.mark_job_running(job_id):
        repo.finish_job(job_id, status="failed")
        raise conflict("scan job could not be started")
    try:
        with heartbeat_job(repo, job_id):
            task = repo.get_task(task_id)
            if task.status in {
                TaskStatus.RUNNING,
                TaskStatus.COMPUTING_METRICS,
            }:
                raise conflict(f"cannot scan task in status {task.status.value}")
            payload = perform_scan_task(repo, task, request.app.state.settings)
        if payload.get("status") == TaskStatus.SCANNED.value:
            dispatch_platform_hook(
                getattr(request.app.state, "hook_dispatcher", None),
                "task.scanned",
                scan_hook_payload(payload),
                task_id=task_id,
            )
        contract_payload = payload.get("validation_input_contract")
        if (
            _is_v2_validation_task(task)
            and isinstance(contract_payload, dict)
            and contract_payload.get("status") == "blocked"
        ):
            contract_check = next(
                (
                    check
                    for check in payload.get("checks", [])
                    if isinstance(check, dict)
                    and check.get("id") == "validation_input_contract"
                ),
                {},
            )
            detail = str(contract_check.get("message") or "")
            raise unprocessable(
                "validation materials invalid: "
                + (detail or "validation input contract is blocked")
            )
    except (FileNotFoundError, NotADirectoryError, SourceDirectoryScanError) as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        # scan_source_dir limit and source-dir errors are client-side invalid input.
        raise unprocessable(f"source dir invalid: {exc}") from exc
    except ValueError as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        if _is_v2_validation_task(task):
            raise unprocessable(f"validation materials invalid: {exc}") from exc
        raise unprocessable(f"source dir invalid: {exc}") from exc
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
        )
        raise
    repo.finish_job(job_id, status="succeeded")
    return payload


def _is_v2_validation_task(task) -> bool:
    return (
        task.task_type == TASK_TYPE_VALIDATION
        and task.validation_workflow_version == 2
    )
