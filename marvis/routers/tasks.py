import logging
import shutil

from fastapi import APIRouter, HTTPException, Request, Response

from marvis.api_schemas import CreateTaskRequest
from marvis.api_task_helpers import (
    dispatch_platform_hook,
    get_task_or_404,
    normalize_source_dir,
    normalized_capability_tier,
    normalized_target_type,
    reject_if_task_has_active_job,
    task_hook_payload,
    validate_model_identifier,
)
from marvis.api_task_payloads import task_payload
from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.model_algorithms import normalize_algorithm
from marvis.notebooks import close_live_notebook_session
from marvis.safe_paths import assert_within


router = APIRouter(prefix="/api", tags=["tasks"])
logger = logging.getLogger(__name__)


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks")
def list_tasks(
    request: Request,
    response: Response,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict]:
    repo = _repo(request)
    bounded_limit = None if limit is None else max(1, min(int(limit), 500))
    bounded_offset = max(0, int(offset))
    query_limit = bounded_limit + 1 if bounded_limit is not None else None
    tasks = repo.list_tasks(limit=query_limit, offset=bounded_offset)
    has_more = False
    if bounded_limit is not None and len(tasks) > bounded_limit:
        has_more = True
        tasks = tasks[:bounded_limit]
    if bounded_limit is not None or bounded_offset:
        response.headers["X-Result-Limit"] = "" if bounded_limit is None else str(bounded_limit)
        response.headers["X-Result-Offset"] = str(bounded_offset)
        response.headers["X-Result-Has-More"] = "true" if has_more else "false"
    return [
        task_payload(repo, task, request.app.state.settings.tasks_dir)
        for task in tasks
    ]


@router.post("/tasks")
def create_task(payload: CreateTaskRequest, request: Request) -> dict:
    validate_model_identifier("model_name", payload.model_name)
    if payload.model_version:
        validate_model_identifier("model_version", payload.model_version)
    try:
        algorithm = normalize_algorithm(payload.algorithm, allow_empty=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Normalize source_dir once at write time so pipeline.py and /scan agree on
    # the canonical absolute path.
    normalized_source_dir = str(
        normalize_source_dir(payload.source_dir, request.app.state.settings)
    )
    repo = _repo(request)
    task = repo.create_task(
        TaskCreate(
            task_type=payload.task_type,
            model_name=payload.model_name,
            model_version=payload.model_version,
            validator=payload.validator,
            source_dir=normalized_source_dir,
            algorithm=algorithm,
            run_mode=payload.run_mode,
            target_col=payload.target_col,
            score_col=payload.score_col,
            split_col=payload.split_col,
            time_col=payload.time_col,
            feature_columns=payload.feature_columns,
            target_type=normalized_target_type(payload.target_type),
            recipes=payload.recipes,
            sample_weight_col=str(payload.sample_weight_col or "").strip(),
            metrics=payload.metrics,
            capability_tier=normalized_capability_tier(payload.capability_tier),
            notebook_path=payload.notebook_path,
            sample_path=payload.sample_path,
            pmml_path=payload.pmml_path,
            dictionary_path=payload.dictionary_path,
            report_values=payload.report_values,
        )
    )
    dispatch_platform_hook(
        getattr(request.app.state, "hook_dispatcher", None),
        "task.created",
        task_hook_payload(task),
        task_id=task.id,
    )
    return task_payload(repo, task, request.app.state.settings.tasks_dir)


@router.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    return task_payload(
        repo,
        get_task_or_404(repo, task_id),
        request.app.state.settings.tasks_dir,
    )


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, request: Request) -> None:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    reject_if_task_has_active_job(repo, task_id)

    settings = request.app.state.settings
    task_dir = assert_within(settings.tasks_dir, settings.tasks_dir / task_id)
    close_live_notebook_session(task_id)
    try:
        repo.delete_task(task_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}",
        ) from exc
    try:
        if task_dir.exists():
            shutil.rmtree(task_dir)
    except OSError as exc:
        logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
