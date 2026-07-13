from __future__ import annotations

from fastapi import APIRouter, Request
from marvis.errors import conflict, unprocessable

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
)
from marvis.db import TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION, TaskStatus
from marvis.job_heartbeat import heartbeat_job
from marvis.repositories.validation_contracts import ValidationContractRepository


router = APIRouter(prefix="/api", tags=["scans"])


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/materials")
def task_material_candidates(
    task_id: str, request: Request, pmml_path: str | None = None
) -> dict:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    try:
        return material_candidates_payload(task, pmml_path_override=pmml_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise unprocessable(f"source dir invalid: {exc}") from exc


@router.put("/tasks/{task_id}/materials")
def update_task_materials(
    task_id: str,
    payload: MaterialSelectionRequest,
    request: Request,
) -> dict:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
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
                TaskStatus.WRITING_ARTIFACTS,
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
                "materials": material_candidates_payload(task),
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
    get_task_or_404(repo, task_id)
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
