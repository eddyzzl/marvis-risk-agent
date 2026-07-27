import logging
from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Request, Response
from marvis.errors import conflict, not_found, unprocessable

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
from marvis.domain import (
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    StrategyProfitInput,
    StrategyTaskInput,
    TaskCreate,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.notebooks import close_live_notebook_session
from marvis.state_machine import ConflictError


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
    if payload.strategy_input is not None and payload.task_type != TASK_TYPE_STRATEGY:
        raise unprocessable("strategy_input 只能用于 strategy 类型任务。")
    # Risk-analysis intake is intentionally conversation-first: an Agent task
    # may exist before the user has uploaded any data. Give that task a safe,
    # empty workspace-owned material directory instead of letting Path("")
    # resolve to the server cwd. Every other flow keeps the existing material
    # requirement.
    source_dir_path = _create_source_dir(payload, request.app.state.settings)
    normalized_source_dir = str(source_dir_path)
    created_intake_dir = not str(payload.source_dir or "").strip()
    repo = _repo(request)
    try:
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
                strategy_input=_strategy_task_input(payload),
                metrics=payload.metrics,
                capability_tier=normalized_capability_tier(payload.capability_tier),
                notebook_path=payload.notebook_path,
                sample_path=payload.sample_path,
                pmml_path=payload.pmml_path,
                dictionary_path=payload.dictionary_path,
                report_values=payload.report_values,
            )
        )
    except Exception:
        if created_intake_dir:
            try:
                source_dir_path.rmdir()
            except OSError as exc:
                logger.warning(
                    "failed to clean unclaimed risk intake dir %s: %s",
                    source_dir_path,
                    exc,
                )
        raise
    dispatch_platform_hook(
        getattr(request.app.state, "hook_dispatcher", None),
        "task.created",
        task_hook_payload(task),
        task_id=task.id,
    )
    return task_payload(repo, task, request.app.state.settings.tasks_dir)


def _create_source_dir(payload: CreateTaskRequest, settings) -> Path:
    raw = str(payload.source_dir or "").strip()
    if raw:
        return normalize_source_dir(raw, settings)
    if payload.task_type != TASK_TYPE_VINTAGE or payload.run_mode != "agent":
        raise unprocessable("source_dir is required")
    intake_dir = (
        Path(settings.workspace).resolve()
        / "material_uploads"
        / f"risk-intake-{uuid.uuid4().hex}"
    )
    intake_dir.mkdir(parents=True, exist_ok=False)
    return normalize_source_dir(str(intake_dir), settings)


def _strategy_task_input(payload: CreateTaskRequest) -> StrategyTaskInput | None:
    contract = payload.strategy_input
    if contract is None:
        return None
    profit = (
        StrategyProfitInput(**contract.profit.model_dump())
        if contract.profit is not None
        else None
    )
    return StrategyTaskInput(
        entry_mode=contract.entry_mode,
        strategy_type=contract.strategy_type,
        objective=contract.objective,
        max_bad_rate=contract.max_bad_rate,
        min_approval_rate=contract.min_approval_rate,
        baseline_strategy_id=contract.baseline_strategy_id,
        profit=profit,
    )


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
    task = get_task_or_404(repo, task_id)
    reject_if_task_has_active_job(repo, task_id)

    settings = request.app.state.settings
    task_dir = _lexical_child_path(settings.tasks_dir, task_id)
    datasets_root = getattr(settings, "datasets_dir", None)
    owned_intake_dir = _unshared_task_owned_risk_intake_dir(
        repo,
        task,
        settings,
    )

    def validate_dataset_source_path(relative_path: str) -> None:
        if datasets_root is not None:
            _lexical_child_path(datasets_root, relative_path)

    try:
        summary = repo.purge_task(
            task_id,
            validate_dataset_source_path=validate_dataset_source_path,
        )
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc
    except PermissionError as exc:
        raise unprocessable("dataset source path escapes the datasets directory") from exc
    except ConflictError as exc:
        raise conflict(str(exc)) from exc
    close_live_notebook_session(task_id)
    try:
        if task_dir.is_symlink():
            task_dir.unlink()
        elif task_dir.exists():
            shutil.rmtree(task_dir)
    except OSError as exc:
        logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
    if datasets_root is not None:
        # Only the dataset files this task exclusively owned are safe to remove --
        # purge_task already excluded source_paths still referenced by another
        # task's dataset row (GAP-7 content-fingerprint reuse shares parquet files
        # across tasks). Remove files individually rather than rmtree'ing the whole
        # datasets/<task_id>/ subtree, since a dataset row reused by this task may
        # point at a file physically stored under a *different* task's directory.
        for relative_path in summary.get("dataset_source_paths", []):
            try:
                dataset_path = _lexical_child_path(datasets_root, relative_path)
            except PermissionError:
                continue
            try:
                if dataset_path.is_symlink() or dataset_path.is_file():
                    dataset_path.unlink()
            except OSError as exc:
                logger.warning(
                    "dataset file cleanup failed for %s (%s): %s",
                    task_id,
                    relative_path,
                    exc,
                )
        task_datasets_dir = datasets_root / task_id
        # Original-upload identity sidecars are task-local metadata used to
        # reconcile retries exactly. They are never shared through parquet
        # content-hash deduplication, so removing only this controlled child is
        # safe after the task's dataset rows have been purged.
        try:
            source_identity_dir = _lexical_child_path(
                datasets_root,
                f"{task_id}/.source-identities",
            )
            if source_identity_dir.is_symlink():
                source_identity_dir.unlink()
            elif source_identity_dir.is_dir():
                shutil.rmtree(source_identity_dir)
        except (OSError, PermissionError) as exc:
            logger.warning("source identity cleanup failed for %s: %s", task_id, exc)
        try:
            if task_datasets_dir.is_symlink():
                task_datasets_dir.unlink()
            elif task_datasets_dir.exists() and not any(task_datasets_dir.rglob("*")):
                shutil.rmtree(task_datasets_dir)
        except OSError as exc:
            logger.warning("datasets dir cleanup failed for %s: %s", task_id, exc)
    if owned_intake_dir is not None:
        try:
            if owned_intake_dir.is_symlink():
                owned_intake_dir.unlink()
            elif owned_intake_dir.exists():
                shutil.rmtree(owned_intake_dir)
        except OSError as exc:
            logger.warning("risk intake dir cleanup failed for %s: %s", task_id, exc)


def _unshared_task_owned_risk_intake_dir(repo, task, settings) -> Path | None:
    if task.task_type != TASK_TYPE_VINTAGE:
        return None
    uploads_root = (Path(settings.workspace).resolve() / "material_uploads").resolve()
    source = Path(str(task.source_dir or "")).absolute()
    try:
        source_parent = source.parent.resolve()
    except OSError:
        return None
    if (
        source_parent != uploads_root
        or not source.name.startswith("risk-intake-")
    ):
        return None
    resolved = None if source.is_symlink() else source.resolve()
    if resolved is not None and resolved.parent != uploads_root:
        return None
    for other in repo.list_tasks():
        if other.id == task.id:
            continue
        other_source = Path(str(other.source_dir or "")).absolute()
        if other_source == source:
            return None
        try:
            if resolved is not None and other_source.resolve() == resolved:
                return None
        except OSError:
            continue
    return source


def _lexical_child_path(root: Path, relative_path: str) -> Path:
    """Return a child path without resolving its final symlink.

    Resolving the final component before deletion turns a symlink into its
    target and can delete another task's data. Intermediate symlinks are
    rejected because unlinking a descendant would still follow them.
    """
    root_path = Path(root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PermissionError(f"path escapes root: {relative_path}")
    current = root_path
    for part in relative.parts[:-1]:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            raise PermissionError(f"path traverses symlink: {relative_path}")
    return root_path.joinpath(*relative.parts)
