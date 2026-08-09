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
from marvis.data.task_filesystem_gc import (
    DATASET_IDENTITY_DIR,
    DATASET_TASK_DIR,
    RISK_INTAKE_DIR,
    TASK_DIR,
    TaskFilesystemGcTarget,
)
from marvis.domain import (
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TASK_TYPE_VALIDATION_BATCH,
    StrategyProfitInput,
    StrategyTaskInput,
    TaskCreate,
    TaskRecord,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.notebooks import close_live_notebook_session
from marvis.repositories.tasks import TaskRepository
from marvis.safe_paths import lexical_child_path
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


def _reject_if_validation_batch_managed(
    repo: TaskRepository,
    task: TaskRecord,
) -> None:
    if task.task_type == TASK_TYPE_VALIDATION_BATCH:
        raise conflict(
            "批次父任务由批次生命周期统一管理，"
            "不能通过通用任务清理接口预览或删除。"
        )
    transaction = getattr(repo, "transaction", None)
    if transaction is None:
        return
    with transaction() as conn:
        child = conn.execute(
            "SELECT parent_task_id FROM validation_batch_items WHERE child_task_id = ?",
            (task.id,),
        ).fetchone()
    if child is not None:
        raise conflict(
            "批次内模型验证子任务由批次生命周期统一管理，"
            "不能单独预览清理或删除。"
        )


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
    tasks = repo.list_tasks(
        limit=query_limit,
        offset=bounded_offset,
        include_batch_children=False,
    )
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
    if payload.task_type == TASK_TYPE_VALIDATION_BATCH:
        raise unprocessable(
            "validation_batch 父任务只能通过专用批次 API "
            "POST /api/validation-batches 创建。"
        )
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
    repo = _repo(request)
    source_dir_path: Path | None = None
    created_intake_dir = not str(payload.source_dir or "").strip()
    try:
        transaction = getattr(repo, "transaction", None)
        create_on_connection = getattr(repo, "create_task_on_connection", None)
        if callable(transaction) and callable(create_on_connection):
            with transaction() as conn:
                # Directory creation/validation and the task insert share the
                # same writer lock used by task filesystem GC.  The collector
                # therefore cannot recheck, delete, and race a later insert.
                conn.execute("BEGIN IMMEDIATE")
                source_dir_path = _create_source_dir(
                    payload,
                    request.app.state.settings,
                )
                task = create_on_connection(
                    conn,
                    _task_create_contract(
                        payload,
                        algorithm=algorithm,
                        source_dir=source_dir_path,
                    ),
                )
                if created_intake_dir:
                    repo.record_task_filesystem_provision_on_connection(
                        conn,
                        task,
                        target_type=RISK_INTAKE_DIR,
                        relative_path=source_dir_path.name,
                    )
        else:
            # Lightweight repository doubles used by route unit tests do not
            # own SQLite transactions. Production TaskRepository always takes
            # the locked branch above.
            source_dir_path = _create_source_dir(
                payload,
                request.app.state.settings,
            )
            task = repo.create_task(
                _task_create_contract(
                    payload,
                    algorithm=algorithm,
                    source_dir=source_dir_path,
                )
            )
    except Exception:
        if created_intake_dir and source_dir_path is not None:
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
        source_dir = normalize_source_dir(raw, settings)
        if not source_dir.is_dir():
            raise unprocessable("source_dir must be an existing directory")
        return source_dir
    if payload.task_type != TASK_TYPE_VINTAGE or payload.run_mode != "agent":
        raise unprocessable("source_dir is required")
    intake_dir = (
        Path(settings.workspace).resolve()
        / "material_uploads"
        / f"risk-intake-{uuid.uuid4().hex}"
    )
    intake_dir.mkdir(parents=True, exist_ok=False)
    return normalize_source_dir(str(intake_dir), settings)


def _task_create_contract(
    payload: CreateTaskRequest,
    *,
    algorithm: str,
    source_dir: Path,
) -> TaskCreate:
    return TaskCreate(
        task_type=payload.task_type,
        model_name=payload.model_name,
        model_version=payload.model_version,
        validator=payload.validator,
        source_dir=str(source_dir),
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
    task = get_task_or_404(repo, task_id)
    _reject_if_validation_batch_managed(repo, task)
    try:
        summary = repo.purge_preview(task_id)
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc
    return {"task_id": task_id, "purge_summary": summary}


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, request: Request) -> None:
    repo = _repo(request)
    task = get_task_or_404(repo, task_id)
    _reject_if_validation_batch_managed(repo, task)
    reject_if_task_has_active_job(repo, task_id)

    settings = request.app.state.settings
    task_dir = lexical_child_path(settings.tasks_dir, task_id)
    datasets_root = getattr(settings, "datasets_dir", None)
    supports_durable_task_fs_gc = callable(
        getattr(repo, "purge_task_on_connection", None)
    )
    task_fs_targets = (
        _task_filesystem_gc_targets(task, settings)
        if supports_durable_task_fs_gc
        else ()
    )

    def validate_dataset_source_path(relative_path: str) -> None:
        if datasets_root is not None:
            lexical_child_path(datasets_root, relative_path)

    try:
        purge_kwargs = {
            "validate_dataset_source_path": validate_dataset_source_path,
        }
        if supports_durable_task_fs_gc:
            purge_kwargs["task_fs_targets"] = task_fs_targets
        summary = repo.purge_task(task_id, **purge_kwargs)
    except KeyError as exc:
        raise not_found(f"Task not found: {task_id}") from exc
    except PermissionError as exc:
        raise unprocessable("dataset source path escapes the datasets directory") from exc
    except ConflictError as exc:
        raise conflict(str(exc)) from exc
    close_live_notebook_session(task_id)
    if not supports_durable_task_fs_gc:
        # Compatibility for lightweight route test doubles. Production
        # TaskRepository always persists typed cleanup candidates above.
        try:
            if task_dir.is_symlink():
                task_dir.unlink()
            elif task_dir.exists():
                shutil.rmtree(task_dir)
        except OSError as exc:
            logger.warning("task dir cleanup failed for %s: %s", task_id, exc)
    if datasets_root is not None:
        source_paths = tuple(summary.get("dataset_source_paths", ()))
        collector = getattr(request.app.state, "dataset_source_gc", None)
        if collector is None:
            logger.warning(
                "dataset source GC is unavailable after task purge for %s; "
                "durable candidates remain queued",
                task_id,
            )
        elif source_paths:
            try:
                gc_report = collector.sweep(
                    source_paths=source_paths,
                    limit=min(len(source_paths), 500),
                    force=True,
                )
            except Exception as exc:
                logger.warning(
                    "dataset source GC failed after task purge for %s; "
                    "durable candidates remain queued: %s",
                    task_id,
                    exc,
                )
            else:
                if gc_report.deferred or gc_report.quarantined:
                    logger.warning(
                        "dataset source GC deferred after task purge for %s: "
                        "deferred=%d quarantined=%d",
                        task_id,
                        gc_report.deferred,
                        gc_report.quarantined,
                    )
    if supports_durable_task_fs_gc:
        collector = getattr(request.app.state, "task_filesystem_gc", None)
        candidate_count = int(summary.get("task_fs_gc_candidates", 0))
        if collector is None:
            logger.warning(
                "task filesystem GC is unavailable after task purge for %s; "
                "durable candidates remain queued",
                task_id,
            )
        elif candidate_count:
            try:
                gc_report = collector.sweep(
                    origin_task_id=task_id,
                    limit=min(candidate_count, 500),
                    force=True,
                )
            except Exception as exc:
                logger.warning(
                    "task filesystem GC failed after task purge for %s; "
                    "durable candidates remain queued: %s",
                    task_id,
                    exc,
                )
            else:
                if gc_report.deferred or gc_report.quarantined:
                    logger.warning(
                        "task filesystem GC deferred after task purge for %s: "
                        "deferred=%d quarantined=%d",
                        task_id,
                        gc_report.deferred,
                        gc_report.quarantined,
                    )


def _task_filesystem_gc_targets(task: TaskRecord, settings) -> tuple:
    task_id = str(task.id)
    lexical_child_path(settings.tasks_dir, task_id)
    lexical_child_path(settings.datasets_dir, task_id)
    lexical_child_path(settings.datasets_dir, f"{task_id}/.source-identities")
    targets = [
        TaskFilesystemGcTarget(TASK_DIR, task_id),
        TaskFilesystemGcTarget(
            DATASET_IDENTITY_DIR,
            f"{task_id}/.source-identities",
        ),
        TaskFilesystemGcTarget(DATASET_TASK_DIR, task_id),
    ]
    risk_intake = _task_owned_risk_intake_relative_path(task, settings)
    if risk_intake is not None:
        targets.append(TaskFilesystemGcTarget(RISK_INTAKE_DIR, risk_intake))
    return tuple(targets)


def _task_owned_risk_intake_relative_path(
    task: TaskRecord,
    settings,
) -> str | None:
    if task.task_type != TASK_TYPE_VINTAGE:
        return None
    uploads_root = (Path(settings.workspace).resolve() / "material_uploads").resolve()
    source = Path(str(task.source_dir or "")).absolute()
    try:
        source_parent = source.parent.resolve()
    except OSError:
        return None
    if source_parent != uploads_root or not source.name.startswith("risk-intake-"):
        return None
    try:
        lexical_child_path(uploads_root, source.name)
    except PermissionError:
        return None
    return source.name
