import logging
import shutil

from fastapi import APIRouter, Request, Response
from marvis.errors import not_found, unprocessable

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
from marvis.api_task_payloads import list_task_payloads, task_payload
from marvis.db import TaskRepository
from marvis.domain import TaskCreate
from marvis.model_algorithms import normalize_algorithm
from marvis.notebooks import close_live_notebook_session
from marvis.safe_paths import assert_within


router = APIRouter(prefix="/api", tags=["tasks"])
logger = logging.getLogger(__name__)


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


def _job_payload(job: dict | None) -> dict | None:
    if job is None:
        return None
    keys = (
        "id",
        "task_id",
        "kind",
        "status",
        "progress_message",
        "error_name",
        "error_value",
        "created_at",
        "started_at",
        "finished_at",
        "log_path",
    )
    return {key: job.get(key) for key in keys if key in job}


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
    return list_task_payloads(repo, tasks, request.app.state.settings.tasks_dir)


@router.post("/tasks")
def create_task(payload: CreateTaskRequest, request: Request) -> dict:
    validate_model_identifier("model_name", payload.model_name)
    if payload.model_version:
        validate_model_identifier("model_version", payload.model_version)
    try:
        algorithm = normalize_algorithm(payload.algorithm, allow_empty=True)
    except ValueError as exc:
        raise unprocessable(str(exc)) from exc
    if payload.oot_ks_min is not None and not (0.0 <= payload.oot_ks_min <= 1.0):
        raise unprocessable("oot_ks_min 必须是 0 到 1 之间的数字。")
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
            oot_ks_min=payload.oot_ks_min,
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


@router.get("/tasks/{task_id}/jobs/latest")
def get_latest_task_job(task_id: str, request: Request, kind: str | None = None) -> dict:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    normalized_kind = str(kind or "").strip() or None
    return {"job": _job_payload(repo.get_latest_job(task_id, kind=normalized_kind))}


@router.get("/tasks/{task_id}/purge-preview")
def purge_preview(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    try:
        summary = repo.purge_preview(task_id)
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc
    return {"task_id": task_id, "purge_summary": summary}


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, request: Request) -> None:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    reject_if_task_has_active_job(repo, task_id)

    settings = request.app.state.settings
    task_dir = assert_within(settings.tasks_dir, settings.tasks_dir / task_id)
    close_live_notebook_session(task_id)
    try:
        summary = repo.purge_task(task_id)
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc
    try:
        if task_dir.exists():
            shutil.rmtree(task_dir)
    except OSError as exc:
        logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
    datasets_root = getattr(settings, "datasets_dir", None)
    if datasets_root is not None:
        # Only the dataset files this task exclusively owned are safe to remove --
        # purge_task already excluded source_paths still referenced by another
        # task's dataset row (GAP-7 content-fingerprint reuse shares parquet files
        # across tasks). Remove files individually rather than rmtree'ing the whole
        # datasets/<task_id>/ subtree, since a dataset row reused by this task may
        # point at a file physically stored under a *different* task's directory.
        for relative_path in summary.get("dataset_source_paths", []):
            try:
                dataset_path = assert_within(datasets_root, datasets_root / relative_path)
            except ValueError:
                continue
            try:
                if dataset_path.exists():
                    dataset_path.unlink()
            except OSError as exc:
                logger.warning(
                    "dataset file cleanup failed for %s (%s): %s",
                    task_id,
                    relative_path,
                    exc,
                )
        task_datasets_dir = datasets_root / task_id
        try:
            if task_datasets_dir.exists() and not any(task_datasets_dir.rglob("*")):
                shutil.rmtree(task_datasets_dir)
        except OSError as exc:
            logger.warning("datasets dir cleanup failed for %s: %s", task_id, exc)
