from collections.abc import Callable
from dataclasses import asdict, replace
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import traceback
from urllib.parse import unquote
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd

from marvis.agent.orchestrator import (
    AgentValidationCancelled,
    agent_next_stage,
    clear_agent_cancellation,
    is_metrics_failure,
    raise_if_agent_cancelled,
    request_agent_cancellation,
)
from marvis.agent.service import (
    REQUIRED_AGENT_REPORT_KEYS,
    agent_conclusions_confirmed,
    agent_rerun_stage,
    answer_chat_message,
    compose_agent_start_message,
    failure_summary,
    generate_word_conclusions,
    is_agent_advance_intent,
    is_stop_validation_intent,
    summarize_stage,
)
from marvis.agent_memory.consolidation import ConsolidationScheduler
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.retrieval import (
    MemoryQuery,
    retrieve_with_distillations,
)
from marvis.agent_memory.extractors import extract_user_preference
from marvis.agent_memory.store import AgentMemoryStore
from marvis.branding import load_branding
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
from marvis.agent.auto_drive import decide_gate
from marvis.agent.feature_setup import FeatureSetupError, build_feature_proposal
from marvis.agent.join_setup import JoinSetupError, build_join_proposal
from marvis.agent.modeling_setup import ModelingSetupError, build_modeling_proposal
from marvis.agent.plan_driver import PlanDriver, is_confirm
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_VALIDATION,
    FileArtifact,
    FileRole,
    TaskCreate,
    TaskRecord,
    TaskStatus,
)
from marvis.execution_environment import (
    load_execution_environment,
)
from marvis.files import scan_source_dir
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_settings import (
    LLMSettingsError,
    resolve_llm_model,
)
from marvis.memory_policy import load_memory_policy
from marvis.api_schemas import (
    AgentMessageRequest,
    AgentModelRequest,
    AgentReportDraftConfirmRequest,
    CreateTaskRequest,
    ReportFieldsUpdateRequest,
    ValidateRequest,
)
from marvis.api_settings import router as settings_router
from marvis.api_task_payloads import (
    normalized_status_reason as _normalized_status_reason,
    task_payload as _task_payload,
    task_report_download_filename as _task_report_download_filename,
)
from marvis.notebook_contract import (
    NotebookContractError,
    precheck_notebook_contract,
)
from marvis.notebook_cancellation import (
    clear_pending_notebook_cancellation,
    request_notebook_cancellation,
)
from marvis.notebooks import close_live_notebook_session, get_live_notebook_session
from marvis.notebook_steps import notebook_step_preview
from marvis.model_algorithms import normalize_algorithm
from marvis.pipeline import (
    NOTEBOOK_STAGE_FAILURE_PREFIX,
    REPORT_STAGE_FAILURE_PREFIX,
    SCAN_STAGE_FAILURE_PREFIX,
    PipelineSettings,
    _clear_generated_artifacts,
    _metrics_cancel_marker_path,
    run_metrics_stage,
    run_notebook_stage,
    run_report_stage,
    run_staged_pipeline,
)
from marvis.metric_tables import metric_table_sections_from_payload
from marvis.report_fields import report_field_payload
from marvis.report_texts import computed_report_text_values_from_payload
from marvis.safe_paths import assert_within
from marvis.state_machine import ConflictError, IllegalTransition
from marvis.output.word_preview import docx_to_html_preview
from marvis.validation.overfitting import overfitting_check_from_validation_results


router = APIRouter(prefix="/api")
router.include_router(settings_router)
logger = logging.getLogger(__name__)
MODEL_ID_RE = re.compile(r"^[\w一-鿿\- ]{1,64}$", re.UNICODE)
AGENT_STOP_ACK_CONTENT = "已停止当前动作，请问有什么指示？"
AGENT_STOP_STATUS_MESSAGE = "已停止当前动作"
MATERIAL_UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_MATERIAL_UPLOAD_FILES = 2000


REQUIRED_SCAN_MATERIALS = (
    (FileRole.NOTEBOOK, "Notebook 文件", "notebook_path"),
    (FileRole.SAMPLE, "样本数据", "sample_path"),
    (FileRole.MODEL_PMML, "PMML 模型", "pmml_path"),
    (FileRole.DATA_DICTIONARY, "数据字典", "dictionary_path"),
)

RMC_CONTRACT_NAME_LABELS = {
    "RMC_SAMPLE_DF": "RMC_SAMPLE_DF（样本 DataFrame）",
    "RMC_SCORE_FN": "RMC_SCORE_FN（模型打分函数）",
    "RMC_TARGET_COL": "RMC_TARGET_COL（目标列）",
    "RMC_ALGORITHM": "RMC_ALGORITHM（模型算法）",
}
SCAN_FAILURE_PREFIX = SCAN_STAGE_FAILURE_PREFIX
ACTIVE_JOB_DETAIL = "task already has an active stage"
AGENT_ACCEPTANCE_NORMAL = "normal"
AGENT_ACCEPTANCE_AUTO = "auto_accept"
AGENT_ACCEPTANCE_MODES = {AGENT_ACCEPTANCE_NORMAL, AGENT_ACCEPTANCE_AUTO}
DATASET_ROLES = {"sample", "feature", "derived", "unknown"}
DATASET_PREVIEW_MAX_ROWS = 500
DEDUP_STRATEGIES = {None, "first", "last", "agg_mean", "agg_max"}


def _format_notebook_contract_error(exc: NotebookContractError) -> str:
    message = str(exc)
    missing_prefix = "Notebook contract check failed before execution: missing "
    if message.startswith(missing_prefix):
        missing_names = [
            name.strip()
            for name in message.removeprefix(missing_prefix).split(",")
            if name.strip()
        ]
        missing_text = "、".join(
            RMC_CONTRACT_NAME_LABELS.get(name, name) for name in missing_names
        )
        return f"Notebook RMC 契约检查失败：缺少 {missing_text}。请在 Notebook 顶层定义后重新扫描。"
    return f"Notebook RMC 契约检查失败：{message}"


def _scan_error_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    return [check for check in checks if check.get("status") == "error"]


def _scan_status_message(checks: list[dict[str, str]]) -> str:
    messages = [
        check.get("message", "")
        for check in _scan_error_checks(checks)
        if check.get("message")
    ]
    if messages:
        return f"{SCAN_FAILURE_PREFIX}{'；'.join(messages)}"
    return "材料扫描完成。"


def _is_scan_failure(task: TaskRecord) -> bool:
    return task.status == TaskStatus.FAILED and task.status_message.startswith(
        SCAN_FAILURE_PREFIX
    )


def _is_metrics_failure(task: TaskRecord) -> bool:
    return is_metrics_failure(task)


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


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
        _repo(request).get_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


def _dataset_payload(dataset) -> dict:
    return {
        "id": dataset.id,
        "task_id": dataset.task_id,
        "role": dataset.role,
        "source_name": Path(dataset.source_path).name,
        "source_path": dataset.source_path,
        "format": dataset.format,
        "sheet": dataset.sheet,
        "row_count": dataset.row_count,
        "columns": [
            {
                "name": column.name,
                "semantic_role": column.semantic_role,
                "dtype": column.dtype,
                "is_hashed": column.fingerprint.is_hashed,
                "hash_type": column.fingerprint.hash_type,
            }
            for column in dataset.columns
        ],
        "has_target": dataset.has_target,
        "target_col": dataset.target_col,
    }


def _join_plan_payload(plan) -> dict:
    return {
        "join_plan_id": plan.id,
        "anchor_dataset_id": plan.anchor_dataset_id,
        "status": plan.status,
        "joins": [
            {
                "feature_id": spec.feature_dataset_id,
                "key_pairs": [
                    {
                        "anchor_col": pair.anchor_col,
                        "feature_col": pair.feature_col,
                        "match_method": pair.match_method,
                        "transform_side": pair.transform_side,
                        "match_rate": pair.match_rate,
                        "resolved_by": pair.resolved_by,
                    }
                    for pair in spec.key_pairs
                ],
                "diagnostics": asdict(spec.diagnostics),
                "dedup_strategy": spec.dedup_strategy,
                "confirmed": spec.confirmed,
            }
            for spec in plan.joins
        ],
    }


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


def _nan_safe_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict("records")


def _dataset_preview_profiles(dataset) -> list[dict]:
    return [
        {
            "name": column.name,
            "dtype": column.dtype,
            "semantic_role": column.semantic_role,
            "null_rate": column.null_rate,
            "cardinality": column.cardinality,
            "sample_values": list(column.sample_values),
        }
        for column in dataset.columns
    ]


def _masked_preview_records(frame: pd.DataFrame, dataset) -> list[dict]:
    role_by_column = {column.name: column.semantic_role for column in dataset.columns}
    rows = []
    for record in _nan_safe_records(frame):
        rows.append({
            str(column): _mask_preview_value(value, role_by_column.get(str(column)))
            for column, value in record.items()
        })
    return rows


def _mask_preview_value(value, semantic_role: str | None):
    if value is None:
        return None
    if semantic_role == "phone":
        return _mask_preview_text(value, keep_start=3, keep_end=2)
    if semantic_role == "idcard":
        return _mask_preview_text(value, keep_start=4, keep_end=2)
    if semantic_role == "id":
        return _mask_preview_text(value, keep_start=4, keep_end=4)
    if semantic_role in {"categorical", "name"}:
        # Person names (姓名) are PII — anonymize to an opaque token, never surface raw.
        return _preview_token(value)
    if semantic_role not in {"amount", "date", "score", "target"} and _looks_like_sensitive_preview_identifier(value):
        return _mask_preview_text(value, keep_start=4, keep_end=4)
    return value


def _mask_preview_text(value, *, keep_start: int, keep_end: int) -> str:
    text = _mask_preview_source_text(value)
    if len(text) <= keep_start + keep_end:
        return "*" * len(text)
    hidden = "*" * (len(text) - keep_start - keep_end)
    return f"{text[:keep_start]}{hidden}{text[-keep_end:]}"


def _mask_preview_source_text(value) -> str:
    if not isinstance(value, str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            pass
        else:
            if number.is_integer():
                return str(int(number))
    return str(value).strip()


def _preview_token(value) -> str:
    text = _mask_preview_source_text(value)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"value:{digest}"


def _looks_like_sensitive_preview_identifier(value) -> bool:
    text = re.sub(r"\D+", "", _mask_preview_source_text(value))
    return len(text) >= 12


def _reject_if_task_has_active_job(repo: TaskRepository, task_id: str) -> None:
    if repo.task_has_active_job(task_id):
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL)


def _start_task_job(repo: TaskRepository, task_id: str, kind: str) -> str:
    try:
        return repo.start_job(task_id, kind)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=ACTIVE_JOB_DETAIL) from exc


def _fail_queued_job(repo: TaskRepository, job_id: str, exc: Exception) -> None:
    repo.finish_job(
        job_id,
        status="failed",
        error_name=exc.__class__.__name__,
        error_value=str(exc),
        traceback="",
    )


def _dispatch_platform_hook(
    hook_dispatcher,
    event: str | None,
    payload: dict,
    *,
    task_id: str | None,
) -> None:
    if hook_dispatcher is None or not event or not task_id:
        return
    try:
        hook_dispatcher.dispatch(event, payload, task_id=task_id)
    except Exception as exc:
        logger.warning("platform hook dispatch failed for %s/%s: %s", event, task_id, exc)


def _run_stage_job(
    job_id: str,
    db_path: Path,
    stage_func,
    kwargs: dict,
    *,
    success_agent_notice: str | None = None,
    hook_dispatcher=None,
    before_hook_event: str | None = None,
    after_hook_event: str | None = None,
) -> None:
    repo = TaskRepository(db_path)
    repo.mark_job_running(job_id)
    task_id = kwargs.get("task_id")
    task_id_text = str(task_id) if task_id else None
    _dispatch_platform_hook(
        hook_dispatcher,
        before_hook_event,
        _stage_hook_payload(job_id, task_id_text),
        task_id=task_id_text,
    )
    try:
        stage_func(**kwargs)
    except Exception as exc:
        repo.finish_job(
            job_id,
            status="failed",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    else:
        job_status = (
            "cancelled"
            if _stage_returned_cancelled_task(repo, task_id)
            else "succeeded"
        )
        repo.finish_job(job_id, status=job_status)
        if job_status == "succeeded":
            _dispatch_platform_hook(
                hook_dispatcher,
                after_hook_event,
                _stage_hook_payload(job_id, task_id_text, status=job_status),
                task_id=task_id_text,
            )
        if job_status == "succeeded" and success_agent_notice == "word_report_ready":
            _add_agent_report_ready_message(repo, task_id)


def _stage_hook_payload(
    job_id: str,
    task_id: str | None,
    *,
    status: str | None = None,
) -> dict:
    payload = {"job_id": job_id}
    if task_id:
        payload["task_id"] = task_id
    if status:
        payload["status"] = status
    return payload


def _task_hook_payload(task: TaskRecord) -> dict:
    payload = {
        "task_id": task.id,
        "task_type": task.task_type,
        "status": task.status.value,
        "run_mode": task.run_mode,
        "algorithm": task.algorithm,
    }
    if getattr(task, "target_type", ""):
        payload["target_type"] = task.target_type
    return payload


def _scan_hook_payload(payload: dict) -> dict:
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    failed_codes = [
        str(check.get("code") or "")
        for check in checks
        if isinstance(check, dict) and check.get("status") != "pass"
    ]
    return {
        "task_id": str(payload.get("task_id") or ""),
        "status": str(payload.get("status") or ""),
        "status_message": str(payload.get("status_message") or ""),
        "check_count": len(checks),
        "failed_check_codes": [code for code in failed_codes if code],
    }


def _add_agent_report_ready_message(repo: TaskRepository, task_id: str | None) -> None:
    if not task_id:
        return
    task = repo.get_task(task_id)
    if task.run_mode != "agent":
        return
    messages = repo.list_agent_messages(task_id)
    latest_confirmed_index = max(
        (
            index
            for index, message in enumerate(messages)
            if message.get("stage") == "word_conclusion_confirmed"
        ),
        default=-1,
    )
    if latest_confirmed_index < 0:
        return
    if any(
        message.get("stage") == "word_report_ready"
        for message in messages[latest_confirmed_index + 1 :]
    ):
        return
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_report_ready",
        content=(
            "报告已生成。右侧步骤里的“预览”可以在线查看 Word，"
            "“下载Word”用于下载验证报告，“下载Excel”用于下载指标分析明细。"
        ),
        metadata={"report_ready": True},
    )


def _stage_returned_cancelled_task(repo: TaskRepository, task_id: str | None) -> bool:
    if not task_id:
        return False
    try:
        task = repo.get_task(task_id)
    except Exception:
        return False
    return _normalized_status_reason(task.status_reason_code) == (
        TASK_STATUS_REASON_USER_CANCELLED
    )


def _get_task_or_404(repo: TaskRepository, task_id: str) -> TaskRecord:
    try:
        return repo.get_task(task_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}",
        ) from exc


def _validate_model_identifier(field_name: str, value: str) -> None:
    if not MODEL_ID_RE.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} contains illegal characters",
        )


def _normalize_source_dir(source_dir: str, settings) -> Path:
    resolved = Path(source_dir).expanduser().resolve()
    allowed_roots = _allowed_material_roots(settings)
    if not any(_path_is_within(root, resolved) for root in allowed_roots):
        allowed = "、".join(str(root) for root in allowed_roots)
        raise HTTPException(
            status_code=422,
            detail=(
                f"source_dir must be under an allowed material root: {allowed}. "
                "Set RMC_MATERIAL_ROOTS to allow another local material directory."
            ),
        )
    return resolved


def _allowed_material_roots(settings) -> tuple[Path, ...]:
    roots = [settings.workspace, Path.home()]
    extra_roots = os.environ.get("RMC_MATERIAL_ROOTS", "")
    roots.extend(Path(raw).expanduser() for raw in extra_roots.split(os.pathsep) if raw)
    resolved: list[Path] = []
    for root in roots:
        candidate = root.resolve()
        if candidate not in resolved:
            resolved.append(candidate)
    return tuple(resolved)


def _path_is_within(root: Path, candidate: Path) -> bool:
    try:
        assert_within(root, candidate)
    except PermissionError:
        return False
    return True


def _new_material_upload_dir(settings) -> Path:
    uploads_root = Path(settings.workspace).resolve() / "material_uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        candidate = uploads_root / uuid4().hex
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise HTTPException(status_code=500, detail="failed to allocate material upload directory")


def _validate_upload_relative_path(raw_path: str | None) -> PurePosixPath:
    value = str(raw_path or "").replace("\\", "/").strip()
    if not value:
        raise HTTPException(status_code=422, detail="invalid upload path: empty filename")
    path = PurePosixPath(value)
    invalid = (
        path.is_absolute()
        or path.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    )
    if invalid:
        raise HTTPException(status_code=422, detail=f"invalid upload path: {value}")
    return path


async def _save_upload_file(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size_bytes = 0
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(MATERIAL_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            size_bytes += len(chunk)
            output.write(chunk)
    await upload.close()
    return size_bytes


@router.get("/branding")
def get_branding(request: Request) -> dict[str, object]:
    return load_branding(request.app.state.settings.workspace)


@router.get("/tasks")
def list_tasks(request: Request) -> list[dict]:
    repo = _repo(request)
    return [
        _task_payload(repo, task, request.app.state.settings.tasks_dir)
        for task in repo.list_tasks()
    ]


def _normalized_capability_tier(value: str | None) -> str:
    """Validate a per-task capability tier name (conservative/balanced/aggressive);
    unknown or empty → '' so the driver falls back to the global settings default.
    Never raises — task creation stays lenient about this autonomy-only knob."""
    from marvis.orchestrator.capability import TIERS

    name = str(value or "").strip().lower()
    return name if name in TIERS else ""


def _normalized_target_type(value: str | None) -> str:
    name = str(value or "").strip().lower()
    return name if name in {"binary", "continuous", "multiclass"} else ""


def _task_tier(request, task) -> str:
    """The capability tier name for a task's plan: the per-task pick if set, else the
    global settings default (spec §5.1). Affects only the autonomy budget
    (max_replan_iterations) — never gates/determinism/safety."""
    from marvis.orchestrator.capability import tier_from_settings

    if getattr(task, "capability_tier", ""):
        return task.capability_tier
    try:
        return tier_from_settings(request.app.state.settings).name
    except Exception:
        return "balanced"


@router.post("/tasks")
def create_task(payload: CreateTaskRequest, request: Request) -> dict:
    _validate_model_identifier("model_name", payload.model_name)
    if payload.model_version:
        _validate_model_identifier("model_version", payload.model_version)
    try:
        algorithm = normalize_algorithm(payload.algorithm, allow_empty=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Normalize source_dir once at write time so pipeline.py and /scan agree
    # on the canonical absolute path (expanduser handles ~, resolve drops ..)
    normalized_source_dir = str(
        _normalize_source_dir(payload.source_dir, request.app.state.settings)
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
            target_type=_normalized_target_type(payload.target_type),
            recipes=payload.recipes,
            metrics=payload.metrics,
            capability_tier=_normalized_capability_tier(payload.capability_tier),
            notebook_path=payload.notebook_path,
            sample_path=payload.sample_path,
            pmml_path=payload.pmml_path,
            dictionary_path=payload.dictionary_path,
            report_values=payload.report_values,
        )
    )
    _dispatch_platform_hook(
        getattr(request.app.state, "hook_dispatcher", None),
        "task.created",
        _task_hook_payload(task),
        task_id=task.id,
    )
    return _task_payload(repo, task, request.app.state.settings.tasks_dir)


@router.post("/material-uploads", status_code=201)
async def upload_materials(
    request: Request,
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(default=None),
) -> dict:
    if not files:
        raise HTTPException(status_code=422, detail="at least one material file is required")
    if len(files) > MAX_MATERIAL_UPLOAD_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"too many material files: max_files={MAX_MATERIAL_UPLOAD_FILES}",
        )
    if relative_paths and len(relative_paths) != len(files):
        raise HTTPException(
            status_code=422,
            detail="relative_paths count must match uploaded files count",
        )

    upload_dir = _new_material_upload_dir(request.app.state.settings)
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
                raise HTTPException(
                    status_code=422,
                    detail=f"duplicate upload path: {relative_path_text}",
                )
            seen_paths.add(relative_path_text)

            destination = (upload_dir / Path(*relative_path.parts)).resolve()
            try:
                destination = assert_within(upload_dir, destination)
            except PermissionError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"invalid upload path: {relative_path_text}",
                ) from exc
            size_bytes = await _save_upload_file(upload, destination)
            saved_files.append(
                {
                    "relative_path": relative_path_text,
                    "size_bytes": size_bytes,
                }
            )
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    return {"source_dir": str(upload_dir), "files": saved_files}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    return _task_payload(
        repo,
        _get_task_or_404(repo, task_id),
        request.app.state.settings.tasks_dir,
    )


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, request: Request) -> None:
    repo = _repo(request)
    _get_task_or_404(repo, task_id)
    _reject_if_task_has_active_job(repo, task_id)

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


@router.get("/tasks/{task_id}/datasets")
def list_task_datasets(task_id: str, request: Request) -> dict:
    _require_task(request, task_id)
    _repo_data, _backend, registry, _join_engine = _data_runtime(request)
    return {
        "datasets": [
            _dataset_payload(dataset)
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
    filename = Path(file.filename or "upload").name
    upload_path = upload_dir / filename
    upload_path.write_bytes(await file.read())
    suffix = upload_path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xlsm"}:
            sheets = list_sheets(upload_path)
            if sheet:
                if sheet not in sheets:
                    raise HTTPException(status_code=422, detail="sheet not found")
                sheets = [sheet]
            datasets = []
            reports = []
            out_dir = request.app.state.settings.datasets_dir / task_id / "excel"
            for sheet_name in sheets:
                parquet_path, report = ingest_sheet(upload_path, sheet_name, out_dir)
                dataset = registry.register_existing(
                    parquet_path,
                    task_id=task_id,
                    role=role,
                )
                datasets.append(dataset)
                reports.append({
                    "sheet": report.sheet,
                    "header_rows": report.header_rows,
                    "data_start_row": report.data_start_row,
                    "flattened_columns": report.flattened_columns,
                    "warnings": [],
                })
        else:
            datasets = [registry.register_from_upload(task_id, upload_path, role=role)]
            reports = []
    except HTTPException:
        raise
    except (DataBackendError, DataIngestError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for dataset in datasets:
        _dispatch_platform_hook(
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
        "datasets": [_dataset_payload(dataset) for dataset in datasets],
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
        "column_profiles": _dataset_preview_profiles(dataset),
        "rows": _masked_preview_records(frame, dataset),
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
    return _join_plan_payload(plan)


@router.get("/joins/{join_plan_id}")
def get_join_plan(join_plan_id: str, request: Request) -> dict:
    repo_data, _backend, _registry, _join_engine = _data_runtime(request)
    try:
        plan = repo_data.load_join_plan(join_plan_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="join plan not found") from exc
    return _join_plan_payload(plan)


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
        _dispatch_platform_hook(
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
    return _join_plan_payload(repo_data.load_join_plan(join_plan_id))


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


@router.post("/tasks/{task_id}/scan")
def scan_task(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _reject_if_task_has_active_job(repo, task_id)
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        raise HTTPException(
            status_code=409,
            detail=f"cannot scan task in status {task.status.value}",
        )
    try:
        payload = _perform_scan_task(repo, task, request.app.state.settings)
        if payload.get("status") == TaskStatus.SCANNED.value:
            _dispatch_platform_hook(
                getattr(request.app.state, "hook_dispatcher", None),
                "task.scanned",
                _scan_hook_payload(payload),
                task_id=task_id,
            )
        return payload
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        # ValueError covers scan-limit breaches (max_files / max_depth) from
        # scan_source_dir; all three are client-side "bad source dir" conditions
        # and must return 422 rather than crashing into a 500.
        raise HTTPException(status_code=422, detail=f"source dir invalid: {exc}") from exc


@router.get("/tasks/{task_id}/report-fields")
def get_report_fields(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    values, revision = repo.get_report_values(task_id)
    payload = _validation_results_payload_for_task(request, task)
    return report_field_payload(
        task,
        values,
        revision,
        metric_values=_metric_values_from_payload(payload),
        metric_table_sections=_metric_table_sections_from_payload(payload),
    )


@router.get("/tasks/{task_id}/evidence")
def get_task_evidence(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    _get_task_or_404(repo, task_id)
    settings = request.app.state.settings
    task_dir = settings.tasks_dir / task_id
    notebook_steps = _read_json(task_dir / "execution" / "notebook_steps.json")
    scan_result = _read_json(task_dir / "execution" / "scan_result.json")
    contract = _read_json(task_dir / "execution" / "runtime_contract.json")
    notebook_reproducibility = _read_json(task_dir / "outputs" / "reproducibility_result.json")
    results = _read_json(task_dir / "outputs" / "validation_results.json")
    environment = load_execution_environment(settings.workspace)
    if scan_result and notebook_steps:
        scan_result = {
            **scan_result,
            "notebook_steps": notebook_steps.get("steps", []),
        }
    return {
        "scan": scan_result,
        "notebook_steps": (notebook_steps or {}).get("steps", []),
        "notebook_cells": (notebook_steps or {}).get("cells", []),
        "contract": contract or {},
        "reproducibility": (results or {}).get("reproducibility", {}) or notebook_reproducibility or {},
        "execution_environment": asdict(environment),
    }


@router.get("/tasks/{task_id}/agent/messages")
def get_agent_messages(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    _get_task_or_404(repo, task_id)
    return {"messages": repo.list_agent_messages(task_id)}


@router.post("/tasks/{task_id}/agent/start", status_code=202)
def start_agent_task(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    _require_wired_agent_task_type(task)
    if task.task_type in _DRIVER_AGENT_TASK_TYPES:
        agent_client = _resolve_driver_agent_client(request, task, payload)
        return _dispatch_driver_turn(
            request, repo, task, user_text=None, agent_client=agent_client,
            acceptance_mode=payload.acceptance_mode,
        )
    model_profile = _resolve_agent_model(request, payload.model_id, payload.effort)
    return _dispatch_agent_validation_job(
        repo=repo,
        task=task,
        settings=request.app.state.settings,
        model_profile=model_profile,
        acceptance_mode=payload.acceptance_mode,
        background_tasks=background_tasks,
    )


@router.post("/tasks/{task_id}/agent/messages", status_code=202)
def post_agent_message(
    task_id: str,
    payload: AgentMessageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    _require_wired_agent_task_type(task)
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="message content is required")
    if is_stop_validation_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "stop"},
        )
        _capture_user_preference_memory(request, task_id, user_message)
        return _handle_agent_stop_message(repo, task)
    if task.task_type in _DRIVER_AGENT_TASK_TYPES:
        agent_client = _resolve_driver_agent_client(request, task, payload)
        return _dispatch_driver_turn(
            request, repo, task, user_text=content, agent_client=agent_client,
            acceptance_mode=payload.acceptance_mode, selection=payload.selection,
            dedup_strategies=payload.dedup_strategies,
        )
    conversation = repo.list_agent_messages(task_id)
    pending_report_draft = _latest_pending_agent_report_draft(conversation)
    if _is_agent_report_confirm_intent(content) and pending_report_draft:
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={"intent": "confirm_report"},
        )
        _capture_user_preference_memory(request, task_id, user_message)
        return _confirm_agent_report_conclusions(
            repo=repo,
            task=task,
            task_id=task_id,
            settings=request.app.state.settings,
            text_values=pending_report_draft["values"],
            expected_revision=pending_report_draft["report_revision"],
            background_tasks=background_tasks,
            hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        )
    model_profile = _resolve_agent_model(request, payload.model_id, payload.effort)
    rerun_stage = agent_rerun_stage(content)
    if rerun_stage:
        _reject_if_task_has_active_job(repo, task_id)
        _require_agent_rerun_stage_reached(task, rerun_stage)
        rerun_intent = (
            "regenerate_report_draft"
            if rerun_stage == "word_conclusion_draft"
            and _is_agent_report_regenerate_intent(content)
            else "rerun_stage"
        )
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={
                **_model_metadata(model_profile),
                "intent": rerun_intent,
                "target_stage": rerun_stage,
            },
        )
        _capture_user_preference_memory(request, task_id, user_message)
        task = _reset_agent_task_for_rerun(repo, task_id, rerun_stage)
        return _dispatch_agent_validation_job(
            repo=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=payload.acceptance_mode,
            background_tasks=background_tasks,
            forced_stage=rerun_stage,
            stage_instruction=content,
        )
    if _is_agent_report_regenerate_intent(content) and pending_report_draft:
        _reject_if_task_has_active_job(repo, task_id)
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata={**_model_metadata(model_profile), "intent": "regenerate_report_draft"},
        )
        _capture_user_preference_memory(request, task_id, user_message)
        return _dispatch_agent_validation_job(
            repo=repo,
            task=task,
            settings=request.app.state.settings,
            model_profile=model_profile,
            acceptance_mode=payload.acceptance_mode,
            background_tasks=background_tasks,
        )
    if not is_agent_advance_intent(content):
        user_message = repo.add_agent_message(
            task_id,
            role="user",
            stage="chat",
            content=content,
            metadata=_model_metadata(model_profile),
        )
        _capture_user_preference_memory(request, task_id, user_message)
        conversation = repo.list_agent_messages(task_id)
        evidence = _agent_chat_evidence(request, repo, task, conversation)
        memory_context = _agent_memory_context(
            request,
            task,
            stage="chat",
            user_message=content,
            evidence=evidence,
        )
        message = _add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: answer_chat_message(
                task=task,
                user_message=content,
                conversation=conversation,
                evidence=evidence,
                memory_context=memory_context,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
        )
        _audit_agent_memory_use(request, message, task_id=task_id)
        return {"task_id": task_id, "status": "message_saved", "messages": repo.list_agent_messages(task_id)}
    user_message = repo.add_agent_message(
        task_id,
        role="user",
        stage="chat",
        content=content,
        metadata={**_model_metadata(model_profile), "intent": "advance"},
    )
    _capture_user_preference_memory(request, task_id, user_message)
    return _dispatch_agent_validation_job(
        repo=repo,
        task=task,
        settings=request.app.state.settings,
        model_profile=model_profile,
        acceptance_mode=payload.acceptance_mode,
        background_tasks=background_tasks,
    )


@router.post("/tasks/{task_id}/agent/stop", status_code=202)
def stop_agent_action(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    return _handle_agent_stop_message(repo, task)


@router.get("/agent-memory")
def list_agent_memory(
    request: Request,
    memory_type: str | None = None,
    status: str | None = None,
    source_task_id: str | None = None,
    model_name: str | None = None,
    channel: str | None = None,
    month: str | None = None,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entries = store.list_entries(status=status, memory_type=memory_type, limit=500)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid memory filter: {exc}") from exc
    items = [_memory_entry_payload(entry) for entry in entries]
    items = [
        item
        for item in items
        if _memory_api_filter_match(
            item,
            source_task_id=source_task_id,
            model_name=model_name,
            channel=channel,
            month=month,
        )
    ]
    return {"items": items}


@router.get("/agent-memory/distillations")
def list_agent_memory_distillations(
    request: Request,
    category: str | None = None,
    include_superseded: bool = False,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        distillations = store.list_distillations(
            category=category,
            include_superseded=include_superseded,
            limit=500,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid distillation filter: {exc}") from exc
    return {"items": [_memory_distillation_payload(item) for item in distillations]}


@router.post("/agent-memory/consolidate")
def consolidate_agent_memory(
    request: Request,
    category: str | None = None,
) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    scheduler = ConsolidationScheduler(
        DistillationEngine(store),
        EvolutionManager(store),
        store,
        async_mode=False,
    )
    try:
        result = scheduler.consolidate_all([category] if category else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"consolidated": result}


@router.get("/agent-memory/distillations/{distillation_id}")
def get_agent_memory_distillation(distillation_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        distillation = store.get_distillation(distillation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory distillation not found") from exc
    return _memory_distillation_detail(store, distillation)


@router.post("/agent-memory/distillations/{distillation_id}/rollback")
def rollback_agent_memory_distillation(distillation_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    predecessor = store.find_superseded_by(distillation_id)
    try:
        EvolutionManager(store).rollback(distillation_id)
        distillation = store.get_distillation(distillation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory distillation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    restored = (
        _memory_distillation_payload(store.get_distillation(predecessor.id))
        if predecessor is not None
        else None
    )
    return {
        "distillation": _memory_distillation_payload(distillation),
        "restored": restored,
    }


@router.get("/agent-memory/{memory_id}")
def get_agent_memory(memory_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.get_entry(memory_id, include_deleted=True, audit=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    return {
        "memory": _memory_entry_payload(entry),
        "events": store.list_events(memory_id),
    }


@router.post("/agent-memory/{memory_id}/disable")
def disable_agent_memory(memory_id: str, request: Request) -> dict:
    return _set_agent_memory_status(request, memory_id, "disabled")


@router.post("/agent-memory/{memory_id}/enable")
def enable_agent_memory(memory_id: str, request: Request) -> dict:
    return _set_agent_memory_status(request, memory_id, "active")


@router.delete("/agent-memory/{memory_id}")
def delete_agent_memory(memory_id: str, request: Request) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.delete(memory_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    return {"memory": _memory_entry_payload(entry), "events": store.list_events(memory_id)}


@router.get("/tasks/{task_id}/agent/messages/{message_id}/memory-references")
def get_agent_message_memory_references(
    task_id: str,
    message_id: str,
    request: Request,
) -> dict:
    repo = _repo(request)
    _get_task_or_404(repo, task_id)
    for message in repo.list_agent_messages(task_id):
        if message.get("id") == message_id:
            references = (message.get("metadata") or {}).get("memory_references")
            return {
                "task_id": task_id,
                "message_id": message_id,
                "memory_references": references if isinstance(references, list) else [],
            }
    raise HTTPException(status_code=404, detail="Agent message not found")


@router.post("/tasks/{task_id}/agent/summarize")
def summarize_agent_task(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    model_profile = _resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = _agent_evidence(request, task_id)
    memory_context = _agent_memory_context(
        request,
        task,
        stage="metrics",
        evidence=evidence,
    )
    content, metadata = summarize_stage(
        task=task,
        stage="metrics",
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
        fallback="已读取当前验证证据。请结合分数一致性、效果稳定性和压力测试明细复核模型表现。",
    )
    metadata.update(_model_metadata(model_profile))
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="summary",
        content=content,
        metadata=metadata,
    )
    _audit_agent_memory_use(request, message, task_id=task_id)
    return {"message": message, "messages": repo.list_agent_messages(task_id)}


@router.post("/tasks/{task_id}/agent/report-draft")
def draft_agent_report_conclusions(
    task_id: str,
    payload: AgentModelRequest,
    request: Request,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    model_profile = _resolve_agent_model(request, payload.model_id, payload.effort)
    evidence = _agent_evidence(request, task_id)
    memory_context = _agent_memory_context(
        request,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
    )
    values, metadata = generate_word_conclusions(
        task=task,
        evidence=evidence,
        memory_context=memory_context,
        model_profile=model_profile,
    )
    metadata.update(_model_metadata(model_profile))
    _, report_revision = repo.get_report_values(task_id)
    message = repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_draft",
        content=_format_conclusion_values(values),
        metadata={**metadata, "draft_values": values, "report_revision": report_revision},
    )
    _audit_agent_memory_use(request, message, task_id=task_id)
    return {
        "message": message,
        "text_values": values,
        "messages": repo.list_agent_messages(task_id),
    }


@router.post("/tasks/{task_id}/agent/report-draft/confirm", status_code=202)
def confirm_agent_report_conclusions(
    task_id: str,
    payload: AgentReportDraftConfirmRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_agent_task(task)
    return _confirm_agent_report_conclusions(
        repo=repo,
        task=task,
        task_id=task_id,
        settings=request.app.state.settings,
        text_values=payload.text_values,
        expected_revision=payload.revision,
        background_tasks=background_tasks,
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
    )


def _confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    task: TaskRecord,
    task_id: str,
    settings,
    text_values: dict[str, str],
    expected_revision: int | None,
    background_tasks: BackgroundTasks,
    model_profile: dict | None = None,
    hook_dispatcher=None,
) -> dict:
    latest_task = _get_task_or_404(repo, task_id)
    if latest_task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate report in status {latest_task.status.value}",
        )
    if expected_revision is None:
        _, expected_revision = repo.get_report_values(task_id)
    job_id = _start_task_job(repo, task_id, "report")
    try:
        revision = repo.update_agent_report_conclusions(
            task_id,
            text_values,
            expected_revision=expected_revision,
        )
    except ConflictError as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    metadata = {
        "revision": revision,
        "confirmed_keys": sorted(text_values),
    }
    if model_profile:
        metadata.update(_model_metadata(model_profile))
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已确认，将开始生成最终 Word 报告。",
        metadata=metadata,
    )
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        settings.db_path,
        run_report_stage,
        {
            "task_id": task_id,
            "settings": _agent_pipeline_settings(settings, latest_task)
            if latest_task.run_mode == "agent"
            else PipelineSettings(
                workspace=settings.workspace,
                db_path=settings.db_path,
                report_template_path=settings.report_template_path,
                feature_columns=latest_task.feature_columns,
                notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
            ),
        },
        success_agent_notice="word_report_ready",
        hook_dispatcher=hook_dispatcher,
        before_hook_event="report.before_generate",
        after_hook_event="report.after_generate",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "revision": revision,
        "message": "agent conclusions confirmed; word report stage dispatched",
        "messages": repo.list_agent_messages(task_id),
    }


@router.put("/tasks/{task_id}/report-fields")
def update_report_fields(
    task_id: str,
    payload: ReportFieldsUpdateRequest,
    request: Request,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    if if_match is None:
        raise HTTPException(status_code=428, detail="If-Match header is required")
    try:
        expected_revision = int(if_match)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="If-Match must be an integer",
        ) from exc
    try:
        revision = repo.update_report_values(
            task_id,
            payload.text_values,
            expected_revision=expected_revision,
        )
        values, _ = repo.get_report_values(task_id)
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    results_payload = _validation_results_payload_for_task(request, task)
    return report_field_payload(
        task,
        values,
        revision,
        metric_values=_metric_values_from_payload(results_payload),
    )


@router.post("/tasks/{task_id}/notebook", status_code=202)
def run_task_notebook(
    task_id: str,
    payload: ValidateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    job_id = _start_task_job(repo, task_id, "notebook")
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"cannot run notebook in status {task.status.value}",
        )
    if _is_scan_failure(task):
        detail = task.status_message.removeprefix(SCAN_FAILURE_PREFIX)
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"材料扫描未完整通过：{detail}",
        )
    try:
        repo.update_status(
            task_id,
            TaskStatus.RUNNING,
            "notebook queued",
            expected={
                TaskStatus.SCANNED,
                TaskStatus.FAILED,
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            },
        )
    except IllegalTransition as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(
            status_code=409,
            detail=f"cannot run notebook in status {exc.current.value}",
        ) from exc
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_notebook_stage,
        {
            "task_id": task_id,
            "settings": _pipeline_settings(request, task, payload.feature_columns),
            "stage_claimed": True,
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        after_hook_event="notebook.completed",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "notebook stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/notebook/cancel", status_code=202)
def cancel_task_notebook(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    if task.status != TaskStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel notebook in status {task.status.value}",
        )
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "notebook cancellation requested; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/metrics/cancel", status_code=202)
def cancel_task_metrics(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    if task.status != TaskStatus.COMPUTING_METRICS:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel metrics in status {task.status.value}",
        )
    _write_metrics_cancel_marker(request.app.state.settings.tasks_dir / task_id)
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "metrics cancellation requested; poll GET /api/tasks/{task_id}",
    }


def _write_metrics_cancel_marker(task_dir: Path) -> None:
    marker_path = _metrics_cancel_marker_path(task_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("cancelled\n", encoding="utf-8")


@router.post("/tasks/{task_id}/report/cancel", status_code=202)
def cancel_task_report(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel report in status {task.status.value}",
        )
    if repo.get_active_job_kind(task_id) != "report":
        raise HTTPException(status_code=409, detail="task has no active report job")
    request_notebook_cancellation(task_id)
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "report cancellation requested; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/metrics", status_code=202)
def run_task_metrics(
    task_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    job_id = _start_task_job(repo, task_id, "metrics")
    metrics_retry = _is_metrics_failure(task)
    if (
        task.status
        in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}
        and get_live_notebook_session(task_id) is None
    ):
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail="live notebook kernel is not available; rerun notebook stage before metrics",
        )
    if task.status in {
        TaskStatus.CREATED,
        TaskStatus.SCANNED,
        TaskStatus.RUNNING,
        TaskStatus.COMPUTING_METRICS,
    }:
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate metrics in status {task.status.value}",
        )
    if task.status == TaskStatus.FAILED and not metrics_retry:
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate metrics in status {task.status.value}",
        )
    try:
        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            "metrics queued",
            expected={
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
                TaskStatus.FAILED,
            }
            if metrics_retry
            else {
                TaskStatus.EXECUTED,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            },
        )
    except IllegalTransition as exc:
        _fail_queued_job(repo, job_id, exc)
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate metrics in status {exc.current.value}",
        ) from exc
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_metrics_stage,
        {
            "task_id": task_id,
            "settings": _pipeline_settings(request, task, None),
            "stage_claimed": True,
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        after_hook_event="validation.completed",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "metrics stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.post("/tasks/{task_id}/report", status_code=202)
def run_task_report(
    task_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_confirmed_agent_conclusions(repo, task)
    job_id = _start_task_job(repo, task_id, "report")
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"cannot generate report in status {task.status.value}",
        )
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        request.app.state.settings.db_path,
        run_report_stage,
        {
            "task_id": task_id,
            "settings": _pipeline_settings(request, task, None),
        },
        hook_dispatcher=getattr(request.app.state, "hook_dispatcher", None),
        before_hook_event="report.before_generate",
        after_hook_event="report.after_generate",
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "word report stage dispatched; poll GET /api/tasks/{task_id}",
    }


@router.get("/tasks/{task_id}/report/download")
def download_task_report(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_confirmed_agent_conclusions(repo, task)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(status_code=404, detail="report not generated")
    report_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation_report.docx"
    )
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=_task_report_download_filename(task, ".docx"),
    )


@router.get("/tasks/{task_id}/report/preview")
def preview_task_report(task_id: str, request: Request) -> HTMLResponse:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    _require_confirmed_agent_conclusions(repo, task)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise HTTPException(status_code=404, detail="report not generated")
    report_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation_report.docx"
    )
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return HTMLResponse(docx_to_html_preview(report_path))


@router.get("/tasks/{task_id}/analysis/download")
def download_task_analysis(task_id: str, request: Request) -> FileResponse:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    if task.status not in {
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    } and not (
        task.status == TaskStatus.FAILED
        and task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
    ):
        raise HTTPException(status_code=404, detail="analysis not generated")
    analysis_path = (
        request.app.state.settings.tasks_dir
        / task_id
        / "outputs"
        / "validation.xlsx"
    )
    if not analysis_path.exists():
        raise HTTPException(status_code=404, detail="analysis not generated")
    return FileResponse(
        analysis_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=_task_report_download_filename(task, ".xlsx"),
    )


@router.get("/tasks/{task_id}/driver-report/download")
def download_driver_report(task_id: str, request: Request) -> FileResponse:
    """Download a driver task's generated Excel report (model development / feature
    analysis). The file path comes from the plan step's ``report_path`` output and is
    validated to live inside the task's outputs directory (no path traversal)."""
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    report_path = _latest_driver_report_path(request.app.state, task_id)
    if report_path is None or not report_path.exists():
        raise HTTPException(status_code=404, detail="report not generated")
    return FileResponse(
        report_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=_task_report_download_filename(task, ".xlsx"),
    )


def _latest_driver_report_path(state, task_id: str):
    """The most recent ``report_path`` produced by a plan step for the task, validated
    to be inside the task's outputs dir. None if no report has been generated."""
    plan_repo = state.plan_repo
    outputs_dir = (Path(state.settings.tasks_dir) / task_id / "outputs").resolve()
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        for step in sorted(plan.steps, key=lambda s: -(int(getattr(s, "index", 0) or 0))):
            try:
                output = plan_repo.load_step_output(step.id)
            except KeyError:
                continue
            raw = (output or {}).get("report_path")
            if not raw:
                continue
            path = Path(str(raw)).resolve()
            try:
                path.relative_to(outputs_dir)
            except ValueError:
                continue  # reject anything outside the task outputs dir
            return path
    return None


@router.get("/artifacts/{artifact_path:path}/preview")
def preview_artifact(artifact_path: str, request: Request):
    path = _resolve_task_artifact_path(request, artifact_path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return HTMLResponse(docx_to_html_preview(path))
    if suffix in {".html", ".htm"}:
        return FileResponse(path, media_type="text/html", filename=path.name)
    if suffix == ".pdf":
        return FileResponse(path, media_type="application/pdf", filename=path.name)
    raise HTTPException(status_code=404, detail="artifact preview not available")


@router.get("/artifacts/{artifact_path:path}")
def download_artifact(artifact_path: str, request: Request) -> FileResponse:
    path = _resolve_task_artifact_path(request, artifact_path)
    return FileResponse(path, filename=path.name)


@router.post("/tasks/{task_id}/validate", status_code=202)
def validate_task(
    task_id: str,
    payload: ValidateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    repo = _repo(request)
    task = _get_task_or_404(repo, task_id)
    job_id = _start_task_job(repo, task_id, "pipeline")
    if task.status in {
        TaskStatus.RUNNING,
        TaskStatus.EXECUTED,
        TaskStatus.COMPUTING_METRICS,
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    }:
        repo.finish_job(job_id, status="failed")
        raise HTTPException(
            status_code=409,
            detail=f"cannot validate task in status {task.status.value}",
        )
    settings = request.app.state.settings
    background_tasks.add_task(
        _run_stage_job,
        job_id,
        settings.db_path,
        run_staged_pipeline,
        {
            "task_id": task_id,
            "settings": PipelineSettings(
                workspace=settings.workspace,
                db_path=settings.db_path,
                report_template_path=settings.report_template_path,
                feature_columns=payload.feature_columns or task.feature_columns,
                notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
            ),
        },
    )
    return {
        "task_id": task_id,
        "status": "accepted",
        "message": "pipeline dispatched; poll GET /api/tasks/{task_id} for terminal status",
    }


def _perform_scan_task(repo: TaskRepository, task: TaskRecord, settings) -> dict:
    # source_dir is already normalized at task-create time (see create_task)
    artifacts = scan_source_dir(Path(task.source_dir))
    checks = _scan_preflight_checks(task, artifacts)
    execution_dir = settings.tasks_dir / task.id / "execution"
    _clear_generated_artifacts(settings.tasks_dir / task.id, stage="scan")
    execution_dir.mkdir(parents=True, exist_ok=True)
    notebook_steps = _scan_notebook_steps(settings, task, artifacts)
    scan_status = TaskStatus.FAILED if _scan_error_checks(checks) else TaskStatus.SCANNED
    scan_message = _scan_status_message(checks)
    payload = {
        "task_id": task.id,
        "status": scan_status.value,
        "status_message": scan_message,
        "artifacts": [_artifact_payload(artifact) for artifact in artifacts],
        "ambiguities": _artifact_ambiguities(artifacts),
        "checks": checks,
        "notebook_steps": notebook_steps,
    }
    (execution_dir / "scan_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    close_live_notebook_session(task.id)
    repo.update_status(
        task.id,
        scan_status,
        scan_message,
        expected={
            TaskStatus.CREATED,
            TaskStatus.SCANNED,
            TaskStatus.FAILED,
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        },
    )
    return payload


def _run_agent_validation_job(
    job_id: str,
    settings,
    task_id: str,
    model_profile: dict,
    opening_message_id: str | None = None,
    stage: str | None = None,
    stage_message_id: str | None = None,
    acceptance_mode: str | None = None,
    stage_instruction: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    repo.mark_job_running(job_id)
    auto_accept = _agent_auto_accept(acceptance_mode)
    try:
        current_stage = stage
        current_opening_message_id = opening_message_id
        current_stage_message_id = stage_message_id
        current_stage_instruction = stage_instruction
        while True:
            task = repo.get_task(task_id)
            current_stage = current_stage or _agent_next_stage(repo, task)
            if current_stage is None:
                if current_opening_message_id or not auto_accept:
                    _finalize_agent_opening_message(
                        repo,
                        task_id=task_id,
                        message_id=current_opening_message_id,
                        model_profile=model_profile,
                        content=(
                            "当前没有可继续执行的下一步。你可以继续询问已生成的验证结果，"
                            "或确认报告结论后生成 Word。"
                        ),
                    )
                repo.finish_job(job_id, status="succeeded")
                return
            _raise_if_agent_cancelled(task_id)
            _open_agent_stage(
                repo,
                task=task,
                task_id=task_id,
                stage=current_stage,
                model_profile=model_profile,
                opening_message_id=current_opening_message_id,
                auto_accept=auto_accept,
            )
            _raise_if_agent_cancelled(task_id)
            if current_stage == "scan":
                stage_succeeded = _run_agent_scan_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "reproducibility":
                stage_succeeded = _run_agent_reproducibility_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "metrics":
                stage_succeeded = _run_agent_metrics_stage(
                    repo, settings, task_id, model_profile, auto_accept=auto_accept
                )
            elif current_stage == "word_conclusion_draft":
                stage_succeeded = _run_agent_word_conclusion_stage(
                    repo,
                    settings,
                    task_id,
                    model_profile,
                    draft_message_id=current_stage_message_id,
                    auto_accept=auto_accept,
                    rewrite_instruction=current_stage_instruction,
                )
            else:
                raise RuntimeError(f"unknown agent stage: {current_stage}")
            if not stage_succeeded:
                repo.finish_job(job_id, status="failed")
                return
            if not auto_accept:
                repo.finish_job(job_id, status="succeeded")
                return
            current_stage = _agent_next_stage(repo, repo.get_task(task_id))
            current_opening_message_id = None
            current_stage_message_id = None
            current_stage_instruction = None
            if current_stage is None:
                repo.finish_job(job_id, status="succeeded")
                return
    except AgentValidationCancelled as exc:
        _mark_agent_cancelled(repo, task_id)
        if not _agent_has_stop_ack_message(repo, task_id):
            repo.add_agent_message(
                task_id,
                role="assistant",
                stage="chat",
                content=AGENT_STOP_ACK_CONTENT,
                metadata={"cancelled": True, "intent": "stop", "cancel_requested": True},
            )
        repo.finish_job(
            job_id,
            status="cancelled",
            error_name=exc.__class__.__name__,
            error_value=str(exc),
            traceback="",
        )
    except Exception as exc:
        try:
            task = repo.get_task(task_id)
            error_detail = f"{exc.__class__.__name__}: {exc}"
            _add_and_stream_agent_message(
                repo,
                task_id,
                stage="failure",
                model_profile=model_profile,
                producer=lambda on_delta: failure_summary(
                    task=task,
                    stage="Agent 执行",
                    error=error_detail,
                    model_profile=model_profile,
                    on_delta=on_delta,
                ),
            )
        finally:
            repo.finish_job(
                job_id,
                status="failed",
                error_name=exc.__class__.__name__,
                error_value=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
    finally:
        _clear_agent_cancellation(task_id)


def _agent_next_stage(repo: TaskRepository, task: TaskRecord) -> str | None:
    return agent_next_stage(repo, task, scan_failure_prefix=SCAN_FAILURE_PREFIX)


def _reset_agent_task_for_rerun(
    repo: TaskRepository,
    task_id: str,
    stage: str,
) -> TaskRecord:
    target_status = {
        "scan": TaskStatus.CREATED,
        "reproducibility": TaskStatus.SCANNED,
        "metrics": TaskStatus.EXECUTED,
        "word_conclusion_draft": TaskStatus.WRITING_ARTIFACTS,
    }.get(stage)
    if target_status is None:
        raise HTTPException(status_code=422, detail=f"unknown rerun stage: {stage}")
    repo.reset_status_for_agent_rerun(
        task_id,
        target_status,
        f"agent rerun requested: {stage}",
        clear_agent_report_conclusions=True,
    )
    if stage in {"scan", "reproducibility"}:
        close_live_notebook_session(task_id)
    return repo.get_task(task_id)


def _agent_pipeline_settings(settings, task: TaskRecord) -> PipelineSettings:
    return PipelineSettings(
        workspace=settings.workspace,
        db_path=settings.db_path,
        report_template_path=settings.report_template_path,
        feature_columns=task.feature_columns,
        notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
    )


def _open_agent_stage(
    repo: TaskRepository,
    *,
    task: TaskRecord,
    task_id: str,
    stage: str,
    model_profile: dict,
    opening_message_id: str | None,
    auto_accept: bool = False,
) -> None:
    # Scan is the entry stage; the next message is the agent's substantive
    # opening (compose_agent_start_message) so a separate "接下来开始执行..."
    # banner here is redundant chatter. The banner stays for later stages
    # where it follows the previous stage's wrap-up.
    if auto_accept and stage != "scan":
        _add_agent_auto_stage_start_message(
            repo,
            task_id=task_id,
            stage=stage,
            model_profile=model_profile,
        )
    if stage == "scan":
        if opening_message_id:
            _stream_agent_message(
                repo,
                opening_message_id,
                task_id=task_id,
                model_profile=model_profile,
                producer=lambda on_delta: compose_agent_start_message(
                    task=task,
                    model_profile=model_profile,
                    on_delta=on_delta,
                ),
            )
            return
        _add_and_stream_agent_message(
            repo,
            task_id,
            stage="chat",
            model_profile=model_profile,
            producer=lambda on_delta: compose_agent_start_message(
                task=task,
                model_profile=model_profile,
                on_delta=on_delta,
            ),
        )
        return
    if auto_accept:
        return
    _finalize_agent_opening_message(
        repo,
        task_id=task_id,
        message_id=opening_message_id,
        model_profile=model_profile,
        content=_agent_stage_opening_text(stage),
    )


def _add_agent_auto_stage_start_message(
    repo: TaskRepository,
    *,
    task_id: str,
    stage: str,
    model_profile: dict,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=f"接下来开始执行{_agent_stage_label(stage)}。",
        metadata={
            **_model_metadata(model_profile),
            "auto_accept": True,
            "auto_stage_start": stage,
            "streaming": False,
        },
    )


def _finalize_agent_opening_message(
    repo: TaskRepository,
    *,
    task_id: str,
    message_id: str | None,
    model_profile: dict,
    content: str,
) -> None:
    metadata = {**_model_metadata(model_profile), "streaming": False}
    if message_id:
        repo.update_agent_message(message_id, content=content, metadata=metadata)
        return
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=content,
        metadata=metadata,
    )


def _agent_stage_opening_text(stage: str) -> str:
    if stage == "reproducibility":
        return "收到，我将继续执行模型可复现性验证，运行 Notebook 并检查代码模型分数与提交 PMML 分数的一致性。"
    if stage == "metrics":
        return "收到，我将继续执行模型效果与稳定性验证，计算 KS、PSI、分箱和压力测试等指标。"
    if stage == "word_conclusion_draft":
        return "收到，我将基于已完成的验证结果起草 Word 报告中的三段结论，完成后会等你确认。"
    return "收到，我将继续执行下一步验证。"


def _run_agent_scan_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="scan",
        content=(
            "正在调用材料识别工具 scan_materials：读取材料目录，识别 Notebook、样本数据、"
            "PMML 模型和数据字典，并检查 Notebook RMC 契约。"
        ),
        metadata={
            **_model_metadata(model_profile),
            "tool_call": {
                "name": "scan_materials",
                "stage": "scan",
            },
        },
    )
    _raise_if_agent_cancelled(task_id)
    scan_payload = _perform_scan_task(repo, task, settings)
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="材料完备性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    _add_and_stream_agent_message(
        repo,
        task_id,
        stage="scan",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="scan",
            evidence=scan_payload,
            model_profile=model_profile,
            fallback="材料扫描完成，平台已识别必需验证材料。",
            on_delta=on_delta,
        ),
    )
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(repo, task_id, model_profile, next_stage="reproducibility")
    return True


def _run_agent_reproducibility_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    repo.update_status(
        task_id,
        TaskStatus.RUNNING,
        "agent notebook queued",
        expected={TaskStatus.SCANNED, TaskStatus.FAILED},
    )
    _raise_if_agent_cancelled(task_id)
    run_notebook_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, task),
        stage_claimed=True,
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="模型可复现性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="reproducibility",
        evidence=evidence,
    )
    message = _add_and_stream_agent_message(
        repo,
        task_id,
        stage="reproducibility",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="reproducibility",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback="分数一致性阶段已完成，请查看可复现性证据明细。",
            on_delta=on_delta,
        ),
    )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(repo, task_id, model_profile, next_stage="metrics")
    return True


def _run_agent_metrics_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    *,
    auto_accept: bool = False,
) -> bool:
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED and _is_metrics_failure(task):
        expected_statuses = {
            TaskStatus.FAILED,
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    else:
        expected_statuses = {
            TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.SUCCEEDED,
            TaskStatus.REVIEW_REQUIRED,
        }
    repo.update_status(
        task_id,
        TaskStatus.COMPUTING_METRICS,
        "agent metrics queued",
        expected=expected_statuses,
    )
    _raise_if_agent_cancelled(task_id)
    run_metrics_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, task),
        stage_claimed=True,
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="效果和稳定性",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="metrics",
        evidence=evidence,
    )
    message = _add_and_stream_agent_message(
        repo,
        task_id,
        stage="metrics",
        model_profile=model_profile,
        producer=lambda on_delta: summarize_stage(
            task=task,
            stage="metrics",
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            fallback="效果、稳定性和 Excel 指标产物已生成，请结合 OOT KS、PSI 和压力测试明细复核。",
            on_delta=on_delta,
        ),
    )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    _raise_if_agent_cancelled(task_id)
    if not auto_accept:
        _add_agent_continue_prompt(
            repo, task_id, model_profile, next_stage="word_conclusion_draft"
        )
    return True


def _run_agent_word_conclusion_stage(
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    draft_message_id: str | None = None,
    *,
    auto_accept: bool = False,
    rewrite_instruction: str | None = None,
) -> bool:
    task = repo.get_task(task_id)
    evidence = _agent_evidence_from_settings(settings, task_id)
    memory_store = AgentMemoryStore(settings.db_path)
    memory_context = _agent_memory_context_from_store(
        memory_store,
        task,
        stage="word_conclusion_draft",
        evidence=evidence,
        user_message=rewrite_instruction or "",
    )
    draft_result: dict[str, object] = {}

    def produce_draft(_on_delta):
        _, report_revision = repo.get_report_values(task_id)
        values, metadata = generate_word_conclusions(
            task=task,
            evidence=evidence,
            memory_context=memory_context,
            model_profile=model_profile,
            user_instruction=rewrite_instruction,
        )
        draft_result["values"] = values
        draft_result["report_revision"] = report_revision
        return (
            _format_conclusion_values(values),
            {**metadata, "draft_values": values, "report_revision": report_revision},
        )

    if draft_message_id:
        message = _stream_agent_message(
            repo,
            draft_message_id,
            task_id=task_id,
            model_profile=model_profile,
            producer=produce_draft,
        )
    else:
        message = _add_and_stream_agent_message(
            repo,
            task_id,
            stage="word_conclusion_draft",
            model_profile=model_profile,
            producer=produce_draft,
        )
    _audit_agent_memory_use_from_store(memory_store, message, task_id=task_id)
    if auto_accept:
        return _auto_confirm_agent_report_conclusions(
            repo=repo,
            settings=settings,
            task_id=task_id,
            model_profile=model_profile,
            values=draft_result.get("values"),
            expected_revision=draft_result.get("report_revision"),
        )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="三段 Word 结论草稿已生成。请先查看；需要写入 Word 时，请直接回复“确认”。",
        metadata={**_model_metadata(model_profile), "awaiting_confirmation": True},
    )
    return True


def _auto_confirm_agent_report_conclusions(
    *,
    repo: TaskRepository,
    settings,
    task_id: str,
    model_profile: dict,
    values: object,
    expected_revision: object,
) -> bool:
    if (
        not isinstance(values, dict)
        or not agent_conclusions_confirmed(values)
        or not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
    ):
        raise RuntimeError("agent report draft is incomplete; cannot auto-confirm report")
    revision = repo.update_agent_report_conclusions(
        task_id,
        {
            key: str(values.get(key) or "").strip()
            for key in REQUIRED_AGENT_REPORT_KEYS
        },
        expected_revision=expected_revision,
    )
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="word_conclusion_confirmed",
        content="三段报告结论已自动确认，正在生成最终 Word 报告。",
        metadata={
            **_model_metadata(model_profile),
            "revision": revision,
            "confirmed_keys": sorted(REQUIRED_AGENT_REPORT_KEYS),
            "auto_accept": True,
        },
    )
    _raise_if_agent_cancelled(task_id)
    run_report_stage(
        task_id=task_id,
        settings=_agent_pipeline_settings(settings, repo.get_task(task_id)),
    )
    _raise_if_agent_cancelled(task_id)
    task = repo.get_task(task_id)
    if task.status == TaskStatus.FAILED:
        _add_agent_failure_summary(
            repo,
            task_id=task_id,
            task=task,
            stage_label="报告生成",
            error=task.status_message,
            model_profile=model_profile,
        )
        return False
    _add_agent_report_ready_message(repo, task_id)
    return True


def _add_agent_failure_summary(
    repo: TaskRepository,
    *,
    task_id: str,
    task: TaskRecord,
    stage_label: str,
    error: str,
    model_profile: dict,
) -> None:
    _add_and_stream_agent_message(
        repo,
        task_id,
        stage="failure",
        model_profile=model_profile,
        producer=lambda on_delta: failure_summary(
            task=task,
            stage=stage_label,
            error=error,
            model_profile=model_profile,
            on_delta=on_delta,
        ),
    )


def _add_agent_continue_prompt(
    repo: TaskRepository,
    task_id: str,
    model_profile: dict,
    *,
    next_stage: str,
) -> None:
    repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content=(
            f"是否继续执行【{_agent_stage_label(next_stage)}】？"
            "你可以先继续提问；需要继续时，请明确回复“继续”。"
        ),
        metadata={**_model_metadata(model_profile), "awaiting_next_stage": next_stage},
    )


def _agent_stage_label(stage: str) -> str:
    if stage == "scan":
        return "模型材料完备性验证"
    if stage == "reproducibility":
        return "模型可复现性验证"
    if stage == "metrics":
        return "模型效果&稳定性验证"
    if stage == "word_conclusion_draft":
        return "报告结论草稿生成"
    return "下一步验证"


def _normalize_agent_report_command(content: str) -> str:
    return "".join(str(content or "").lower().split()).strip("。.!！?？")


def _is_agent_report_confirm_intent(content: str) -> bool:
    command = _normalize_agent_report_command(content)
    return command in {
        "确认",
        "确认写入",
        "写入报告",
        "确认生成报告",
        "生成报告",
        "生成word",
        "生成word报告",
        "可以写入",
    }


def _is_agent_report_regenerate_intent(content: str) -> bool:
    command = _normalize_agent_report_command(content)
    return command in {
        "重新生成",
        "重新生成报告",
        "重新生成word",
        "重新生成word报告",
        "重新生成草稿",
        "重新生成结论",
        "重新生成三段总结",
        "重写报告",
        "重新起草",
        "再生成",
        "再生成报告",
        "再写一版",
    }


def _latest_pending_agent_report_draft(messages: list[dict]) -> dict:
    for message in reversed(messages):
        if message.get("stage") == "word_conclusion_confirmed":
            return {}
        if message.get("role") != "assistant":
            continue
        if message.get("stage") != "word_conclusion_draft":
            continue
        metadata = message.get("metadata") or {}
        draft_values = metadata.get("draft_values")
        report_revision = metadata.get("report_revision")
        if (
            isinstance(draft_values, dict)
            and agent_conclusions_confirmed(draft_values)
            and isinstance(report_revision, int)
            and not isinstance(report_revision, bool)
        ):
            return {
                "message_id": message.get("id"),
                "report_revision": report_revision,
                "values": {
                    key: str(draft_values.get(key) or "").strip()
                    for key in REQUIRED_AGENT_REPORT_KEYS
                },
            }
    return {}


def _handle_agent_stop_message(repo: TaskRepository, task: TaskRecord) -> dict:
    if not _agent_has_cancellable_work(repo, task.id):
        message = repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content="当前没有正在执行的 Agent 任务，无需停止。需要继续验证时可以重新发送指令。",
            metadata={"intent": "stop", "active_job": None},
        )
        return {
            "task_id": task.id,
            "status": "message_saved",
            "message": message["content"],
            "messages": repo.list_agent_messages(task.id),
        }
    _request_agent_cancellation(task.id)
    request_notebook_cancellation(task.id)
    _mark_agent_cancelled(repo, task.id)
    # Guard against a duplicate ack when stop is sent twice in quick succession
    # (the background job cancel path already dedupes via the same check).
    if _agent_has_stop_ack_message(repo, task.id):
        ack_content = AGENT_STOP_ACK_CONTENT
    else:
        ack_content = repo.add_agent_message(
            task.id,
            role="assistant",
            stage="chat",
            content=AGENT_STOP_ACK_CONTENT,
            metadata={"intent": "stop", "cancel_requested": True},
        )["content"]
    return {
        "task_id": task.id,
        "status": "cancel_requested",
        "message": ack_content,
        "messages": repo.list_agent_messages(task.id),
    }


def _require_agent_rerun_stage_reached(task: TaskRecord, stage: str) -> None:
    if stage == "scan":
        return
    if _agent_rerun_stage_reached(task, stage):
        return
    raise HTTPException(
        status_code=409,
        detail="尚未执行到该阶段，不能重新执行；请先按顺序完成前置验证步骤。",
    )


def _agent_rerun_stage_reached(task: TaskRecord, stage: str) -> bool:
    status = task.status
    if stage == "reproducibility":
        return (
            status
            in {
                TaskStatus.SCANNED,
                TaskStatus.RUNNING,
                TaskStatus.EXECUTED,
                TaskStatus.COMPUTING_METRICS,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or task.status_message.startswith(NOTEBOOK_STAGE_FAILURE_PREFIX)
            or is_metrics_failure(task)
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    if stage == "metrics":
        return (
            status
            in {
                TaskStatus.EXECUTED,
                TaskStatus.COMPUTING_METRICS,
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or is_metrics_failure(task)
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    if stage == "word_conclusion_draft":
        return (
            status
            in {
                TaskStatus.WRITING_ARTIFACTS,
                TaskStatus.SUCCEEDED,
                TaskStatus.REVIEW_REQUIRED,
            }
            or task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
        )
    return False


def _agent_has_cancellable_work(repo: TaskRepository, task_id: str) -> bool:
    if repo.get_active_job_kind(task_id) == "agent":
        return True
    return any(
        message.get("role") == "assistant"
        and bool((message.get("metadata") or {}).get("streaming"))
        for message in repo.list_agent_messages(task_id)
    )


def _agent_has_stop_ack_message(repo: TaskRepository, task_id: str) -> bool:
    for message in repo.list_agent_messages(task_id):
        metadata = message.get("metadata") or {}
        if (
            message.get("role") == "assistant"
            and metadata.get("intent") == "stop"
            and metadata.get("cancel_requested") is True
        ):
            return True
    return False


def _modeling_data_runtime(settings):
    datasets_root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(data_repo, backend, datasets_root)
    return backend, registry


# A plan in one of these statuses is finished — re-engaging the task must build a
# FRESH plan, not resume the old one (resuming a terminal plan just replays its final
# done/failed message forever).
_TERMINAL_PLAN_STATUS_VALUES = frozenset({"done", "failed", "cancelled"})


def _active_plan(plan_repo, task_id: str):
    """The latest NON-terminal plan for the task, or None. Used so a driver turn
    resumes an in-flight plan but starts a new one once the previous plan finished."""
    for plan in reversed(plan_repo.list_plans_for_task(task_id)):
        status = getattr(plan.status, "value", plan.status)
        if status not in _TERMINAL_PLAN_STATUS_VALUES:
            return plan
    return None


def _run_join_driver_turn(request, repo: TaskRepository, task: TaskRecord, *, user_text: str | None, selection: list | None = None, dedup_strategies: dict | None = None) -> dict:
    """Drive a data_join task through the generic PlanDriver, synchronously.

    The deterministic JOIN flow (propose -> confirm -> execute, with a forced
    confirmation gate) is short, so it runs inline on the app's already-wired plan
    engine — same thread as the request, so the repos' per-call sqlite connections
    stay safe and there is no engine to reconstruct. First turn: discover/register
    the task's data files and propose anchor/feature roles, then start the plan
    (runs to the confirm gate). Later turns: resume the task's plan at its gate.
    No LLM is required (the reviewer degrades gracefully). Messages are append-only.
    """
    state = request.app.state
    plan_repo = state.plan_repo
    driver = PlanDriver(
        plan_repo, state.plan_executor, planner=state.planner, validator=state.plan_validator,
        llm_client=_driver_llm_client(request, task),
    )
    if user_text is not None:
        # The C1 control form posts a structured "[C1]{...}" payload — show a
        # friendly line in the transcript instead of the raw JSON (the original
        # text is still parsed below for the role assignment).
        display = "已确认文件角色与目标列。" if user_text.startswith("[C1]") else user_text
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=display, metadata={"intent": "data_join"}
        )
    try:
        active = _active_plan(plan_repo, task.id)
        if active is not None:
            # An in-flight plan exists → resume at the C2 diagnostics gate.
            turn = driver.resume(plan_id=active.id, user_text=user_text or "", selection=selection, dedup_strategies=dedup_strategies)
            _append_driver_messages(repo, task.id, turn)
            return _join_turn_response(repo, task.id)
        # No plan yet → C1 file-role assignment gate (spec §3).
        conversation = repo.list_agent_messages(task.id)
        c1_state = _latest_c1_state(conversation)
        _, registry = _modeling_data_runtime(state.settings)
        if c1_state is None:
            # First turn: propose roles + target, pause for the user to confirm/adjust.
            proposal = build_join_proposal(registry, task.id, task.source_dir)
            _append_c1_message(repo, task.id, proposal)
            return _join_turn_response(repo, task.id)
        # User is replying to C1: finalize roles, then build + run the join plan.
        assignment = _parse_c1_reply(user_text, c1_state)
        if assignment is None:
            repo.add_agent_message(
                task.id, role="assistant", stage="chat",
                content="请确认文件角色与目标列:无误就回复「确认」,或用下方控件调整后点「确认角色」。",
                metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
            )
            return _join_turn_response(repo, task.id)
        if not assignment["anchor_id"]:
            return _append_join_error(repo, task.id, "请先指定样本锚表(通常是含目标列的那张),再确认。")
        if not assignment["feature_ids"]:
            repo.add_agent_message(
                task.id, role="assistant", stage="chat",
                content="已确认样本表与目标列。只有一张表,无需拼接(数据拼接阶段已跳过)。",
                metadata={"join_skip": True},
            )
            return _join_turn_response(repo, task.id)
        turn = driver.start(
            task_id=task.id, template_id="data_join",
            slots={"anchor_id": assignment["anchor_id"], "feature_ids": assignment["feature_ids"]},
            tier=_task_tier(request, task),
        )
        _append_driver_messages(repo, task.id, turn)
        return _join_turn_response(repo, task.id)
    except JoinSetupError as exc:
        return _append_join_error(repo, task.id, str(exc))
    except Exception as exc:  # surface as an assistant message rather than a 500
        return _append_join_error(repo, task.id, f"数据拼接出错：{exc}")


def _append_driver_messages(repo: TaskRepository, task_id: str, turn) -> None:
    for message in turn.messages:
        repo.add_agent_message(
            task_id, role="assistant", stage="chat", content=message.content, metadata=dict(message.metadata)
        )


def _join_turn_response(repo: TaskRepository, task_id: str) -> dict:
    return {"task_id": task_id, "status": "ok", "messages": repo.list_agent_messages(task_id)}


def _latest_c1_state(conversation: list[dict]) -> dict | None:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            c1 = (message.get("metadata") or {}).get("join_c1")
            if isinstance(c1, dict):
                return c1
    return None


def _c1_table(c1_state: dict) -> list[dict]:
    rows = [
        [
            f.get("name", ""),
            str(f.get("row_count", "")),
            str(f.get("n_cols", "")),
            "是" if f.get("has_target") else "否",
            f.get("candidate_target") or "—",
            "样本主表" if f.get("proposed_role") == "anchor" else "特征表",
        ]
        for f in c1_state.get("files") or []
    ]
    return [{
        "title": "输入文件(请确认角色与目标列)",
        "columns": ["文件", "行数", "列数", "含目标列", "候选目标列", "提议角色"],
        "rows": rows,
    }]


def _append_c1_message(repo: TaskRepository, task_id: str, proposal) -> None:
    files = proposal.files
    anchor = next((f for f in files if f.proposed_role == "anchor"), None)
    feature_names = [f.name for f in files if f.proposed_role == "feature"]
    if proposal.skip:
        text = (
            f"我发现 {len(files)} 个数据文件。提议**样本主表 = `{anchor.name if anchor else '?'}`**"
            + (f",目标列 = `{proposal.target_col}`" if proposal.target_col else "(未识别目标列,请指定)")
            + "。只有一张表,确认后将跳过拼接。请确认,或用下方控件调整。"
        )
    else:
        text = (
            f"我发现 {len(files)} 个数据文件,先确认每张的**角色与目标列**(样本是锚,只贴列不改行,**1:1**):\n"
            f"- 提议**样本主表** = `{anchor.name if anchor else '?'}`"
            + (f"(目标列 `{proposal.target_col}`)" if proposal.target_col else "(未识别目标列,请指定)")
            + "\n- 提议**特征表** = "
            + (", ".join(f"`{name}`" for name in feature_names) or "(无)")
            + "\n确认无误回复「确认」;要改就用下方控件选好角色/目标列后点「确认角色」。"
        )
    c1_state = {
        "files": [
            {
                "dataset_id": f.dataset_id,
                "name": f.name,
                "row_count": f.row_count,
                "n_cols": f.n_cols,
                "has_target": f.has_target,
                "candidate_target": f.candidate_target,
                "proposed_role": f.proposed_role,
                "columns": f.columns,
            }
            for f in files
        ],
        "anchor_id": proposal.anchor_id,
        "feature_ids": proposal.feature_ids,
        "target_col": proposal.target_col,
        "skip": proposal.skip,
    }
    repo.add_agent_message(
        task_id, role="assistant", stage="chat", content=text,
        metadata={"join_c1": c1_state, "tables": _c1_table(c1_state)},
    )


def _parse_c1_reply(user_text: str | None, c1_state: dict) -> dict | None:
    text = (user_text or "").strip()
    # Structured assignment from the C1 control form: "[C1]{json}".
    if text.startswith("[C1]"):
        try:
            payload = json.loads(text[len("[C1]"):])
        except (ValueError, TypeError):
            return None
        anchor_id = payload.get("anchor_id")
        feature_ids = [
            fid for fid in (payload.get("feature_ids") or []) if fid and fid != anchor_id
        ]
        return {
            "anchor_id": anchor_id,
            "feature_ids": feature_ids,
            "target_col": payload.get("target_col"),
        }
    # Plain confirmation → use the proposed roles as-is.
    if is_confirm(text):
        return {
            "anchor_id": c1_state.get("anchor_id"),
            "feature_ids": list(c1_state.get("feature_ids") or []),
            "target_col": c1_state.get("target_col"),
        }
    return None


def _append_join_error(repo: TaskRepository, task_id: str, detail: str) -> dict:
    repo.add_agent_message(task_id, role="assistant", stage="chat", content=detail, metadata={"error": True})
    return {"task_id": task_id, "status": "error", "messages": repo.list_agent_messages(task_id)}


def _run_feature_driver_turn(request, repo: TaskRepository, task: TaskRecord, *, user_text: str | None, selection: list | None = None, dedup_strategies: dict | None = None) -> dict:
    """Drive a feature_analysis task: compute the selected per-feature metrics and
    show the wide table (spec §1 form A — standalone analysis report, no screening
    gate). Runs synchronously on the wired plan engine; manual mode = the user reads
    the table (no LLM). One step, no gate, so the plan completes in a single run.
    """
    state = request.app.state
    plan_repo = state.plan_repo
    driver = PlanDriver(
        plan_repo, state.plan_executor, planner=state.planner, validator=state.plan_validator,
        llm_client=_driver_llm_client(request, task),
    )
    if user_text is not None:
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=user_text, metadata={"intent": "feature_analysis"}
        )
    try:
        active = _active_plan(plan_repo, task.id)
        if active is not None:
            turn = driver.resume(plan_id=active.id, user_text=user_text or "", selection=selection, dedup_strategies=dedup_strategies)
            _append_driver_messages(repo, task.id, turn)
            return _join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(state.settings)
        proposal = build_feature_proposal(
            registry, backend, task.id, task.source_dir, metrics=_feature_metrics(task)
        )
        repo.add_agent_message(
            task.id, role="assistant", stage="chat",
            content=(
                f"分析数据集 `{proposal.dataset_name}`(目标列 `{proposal.target_col}`,"
                f"{len(proposal.features)} 个候选特征):"
            ),
            metadata={"intent": "feature_analysis"},
        )
        turn = driver.start(
            task_id=task.id, template_id=proposal.template_id,
            slots=proposal.template_slots(),
            tier=_task_tier(request, task),
        )
        _append_driver_messages(repo, task.id, turn)
        return _join_turn_response(repo, task.id)
    except FeatureSetupError as exc:
        return _append_join_error(repo, task.id, str(exc))
    except Exception as exc:  # surface as an assistant message rather than a 500
        return _append_join_error(repo, task.id, f"特征分析出错：{exc}")


def _feature_metrics(task: TaskRecord) -> list[str]:
    """Optional feature metrics the user selected at creation (e.g. ``vif``), stored
    on the task. Empty → base per-feature metrics only (spec §2: 选了才算)."""
    return [str(item).strip() for item in (getattr(task, "metrics", None) or []) if str(item).strip()]


def _modeling_recipes(task: TaskRecord) -> list[str] | None:
    """Recipes the user chose for a modeling task (manual-mode multi-select, stored on
    the task). None → modeling_setup recommends / defaults (the agent-mode path, where
    no options are shown). Multiple → train each and pick the best (G2 multi-algorithm)."""
    recipes = [str(item).strip() for item in (getattr(task, "recipes", None) or []) if str(item).strip()]
    return recipes or None


def _modeling_target_type(task: TaskRecord) -> str | None:
    target_type = str(getattr(task, "target_type", "") or "").strip()
    return target_type or None


def _run_modeling_driver_turn(request, repo: TaskRepository, task: TaskRecord, *, user_text: str | None, selection: list | None = None, dedup_strategies: dict | None = None) -> dict:
    """Drive a modeling task through the generic PlanDriver: leakage-aware screen ->
    [confirm features] -> tune+train -> [confirm model] -> model-development report.
    Replaces the bespoke ModelingSession prototype. Synchronous on the wired plan
    engine; manual mode = the user confirms each gate via a control button (no LLM).
    """
    state = request.app.state
    plan_repo = state.plan_repo
    driver = PlanDriver(
        plan_repo, state.plan_executor, planner=state.planner, validator=state.plan_validator,
        llm_client=_driver_llm_client(request, task),
    )
    if user_text is not None:
        repo.add_agent_message(
            task.id, role="user", stage="chat", content=user_text, metadata={"intent": "modeling"}
        )
    try:
        active = _active_plan(plan_repo, task.id)
        if active is not None:
            turn = driver.resume(plan_id=active.id, user_text=user_text or "", selection=selection, dedup_strategies=dedup_strategies)
            _append_driver_messages(repo, task.id, turn)
            return _join_turn_response(repo, task.id)
        backend, registry = _modeling_data_runtime(state.settings)
        proposal = build_modeling_proposal(
            registry, backend, task.id, task.source_dir,
            target_type=_modeling_target_type(task),
            recipes=_modeling_recipes(task),
        )
        counts = proposal.counts
        bad = f"(坏率 {proposal.bad_rate:.2%})" if proposal.bad_rate is not None else ""
        note_text = ("\n" + " ".join(proposal.notes)) if proposal.notes else ""
        repo.add_agent_message(
            task.id, role="assistant", stage="chat",
            content=(
                f"开始建模:样本 `{proposal.dataset_name}`,目标列 `{proposal.target_col}`{bad},"
                f"切分 `{proposal.split_col}` train/test/oot="
                f"{counts.get('train', 0)}/{counts.get('test', 0)}/{counts.get('oot', 0)},"
                f"候选特征 {len(proposal.feature_cols)} 个。先做泄漏感知特征筛选,随后请确认特征集。"
                f"{note_text}"
            ),
            metadata={"intent": "modeling"},
        )
        turn = driver.start(
            task_id=task.id,
            template_id=proposal.template_id,
            slots=proposal.template_slots(),
            tier=_task_tier(request, task),
        )
        _append_driver_messages(repo, task.id, turn)
        return _join_turn_response(repo, task.id)
    except ModelingSetupError as exc:
        return _append_join_error(repo, task.id, str(exc))
    except Exception as exc:  # surface as an assistant message rather than a 500
        return _append_join_error(repo, task.id, f"建模出错：{exc}")


# --- agent-mode auto-drive ---------------------------------------------------
# Manual mode = the user clicks 「确认」 at each gate. Agent mode = an LLM operates
# those same gates. A configured LLM is therefore MANDATORY in agent mode (the user's
# invariant: no LLM must error, never silently run the manual flow). The deterministic
# turn functions above are reused verbatim — agent mode just feeds them the 「确认」 the
# LLM decides on, gate after gate, until the flow finishes / halts / hits the cap.
_AGENT_MAX_GATES = 8

# Maps a driver task_type to its synchronous turn function (all share the
# (request, repo, task, *, user_text) signature).
_DRIVER_TURN_FUNCS = {
    TASK_TYPE_MODELING: _run_modeling_driver_turn,
    TASK_TYPE_DATA_JOIN: _run_join_driver_turn,
    TASK_TYPE_FEATURE_ANALYSIS: _run_feature_driver_turn,
}


def _resolve_driver_agent_client(request, task: TaskRecord, payload):
    """Agent mode hands the manual gate-controls to an LLM, so a configured LLM is
    mandatory: returns the client, or raises HTTP 409 when none is configured (never
    silently runs the manual flow). Manual mode operates the gates by hand → None."""
    if task.run_mode != "agent":
        return None
    profile = _resolve_agent_model(
        request, getattr(payload, "model_id", None), getattr(payload, "effort", None)
    )
    return OpenAICompatibleLLMClient(profile)


def _driver_llm_client(request, task: TaskRecord):
    """The PlanDriver's LLM for agent-mode free-text gate instructions (adjust /
    replan). None in manual mode or when no LLM is configured — the driver then
    degrades to the canned gate hint (manual gates use control buttons, not text)."""
    if task.run_mode != "agent":
        return None
    try:
        return OpenAICompatibleLLMClient(resolve_llm_model(request.app.state.settings.workspace, None))
    except LLMSettingsError:
        return None


def _latest_open_gate(messages: list[dict]) -> dict | None:
    """The latest assistant message iff it is a confirmation gate awaiting a reply —
    the plan-level overview gate (``metadata.kind == 'plan_overview'``), a per-step
    driver gate (``metadata.kind == 'gate'``), or the JOIN C1 file-role gate
    (``metadata.join_c1``). A terminal/error/skip message is not an open gate."""
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    if last_assistant is None:
        return None
    meta = last_assistant.get("metadata") or {}
    if meta.get("error") or meta.get("join_skip"):
        return None
    if meta.get("kind") in ("gate", "plan_overview") or "join_c1" in meta:
        return last_assistant
    return None


def _agent_autodrive_turn(request, repo: TaskRepository, task: TaskRecord, *, client) -> None:
    """Let the LLM operate the gates the user would click in manual mode: while the
    latest message is an open gate, ask the LLM to confirm/halt and re-run the turn
    with its reply. Bounded by ``_AGENT_MAX_GATES`` so a confirm loop can't run away."""
    turn_fn = _DRIVER_TURN_FUNCS[task.task_type]
    for _ in range(_AGENT_MAX_GATES):
        gate = _latest_open_gate(repo.list_agent_messages(task.id))
        if gate is None:
            return  # flow finished (done / skip / error) — nothing left to operate
        try:
            decision = decide_gate(client, gate=gate)
        except LLMClientError as exc:
            repo.add_agent_message(
                task.id, role="assistant", stage="chat",
                content=f"⚠️ 自动决策失败（{exc}），请手动确认或重试。",
                metadata={"intent": "agent_error"},
            )
            return
        repo.add_agent_message(
            task.id, role="assistant", stage="chat", content=f"🤖 {decision['reason']}",
            metadata={"intent": "agent_decision", "action": decision["action"]},
        )
        if decision["action"] != "confirm":
            return  # halt — leave the gate for the user
        turn_fn(request, repo, task, user_text="确认")


def _dispatch_driver_turn(
    request, repo: TaskRepository, task: TaskRecord, *, user_text: str | None,
    agent_client, acceptance_mode: str | None = None, selection: list | None = None,
    dedup_strategies: dict | None = None,
) -> dict:
    """Run one driver turn. ``acceptance_mode`` controls the agent-mode behavior at
    gates (spec §6, two 受控度): AUTO(自动审查) lets the LLM auto-drive ALL gates;
    NORMAL(默认权限) runs a single turn and STOPS at the first gate for the user to
    confirm — even with an LLM configured. Manual mode (agent_client None) always
    stops at the gate for the control button. ``selection`` carries an edited feature
    set from the §4 screening table; ``dedup_strategies`` carries the per-feature dedup
    map from the §4 join dedup picker."""
    result = _DRIVER_TURN_FUNCS[task.task_type](
        request, repo, task, user_text=user_text, selection=selection,
        dedup_strategies=dedup_strategies,
    )
    if agent_client is not None and _agent_auto_accept(acceptance_mode):
        _agent_autodrive_turn(request, repo, task, client=agent_client)
        return _join_turn_response(repo, task.id)
    return result


def _dispatch_agent_validation_job(
    *,
    repo: TaskRepository,
    task: TaskRecord,
    settings,
    model_profile: dict,
    acceptance_mode: str | None = None,
    background_tasks: BackgroundTasks,
    forced_stage: str | None = None,
    stage_instruction: str | None = None,
) -> dict:
    _clear_agent_cancellation(task.id)
    normalized_acceptance_mode = _normalize_agent_acceptance_mode(acceptance_mode)
    auto_accept = _agent_auto_accept(normalized_acceptance_mode)
    stage = forced_stage or _agent_next_stage(repo, task)
    job_id = _start_task_job(repo, task.id, "agent")
    should_create_opening_message = not (auto_accept and stage and stage != "scan")
    opening_message = (
        _add_streaming_agent_message(
            repo,
            task.id,
            stage="chat",
            model_profile=model_profile,
        )
        if should_create_opening_message
        else None
    )
    if opening_message and stage and stage != "scan":
        _finalize_agent_opening_message(
            repo,
            task_id=task.id,
            message_id=opening_message["id"],
            model_profile=model_profile,
            content=_agent_stage_opening_text(stage),
        )
    stage_message = None
    if stage == "word_conclusion_draft":
        stage_message = _add_streaming_agent_message(
            repo,
            task.id,
            stage="word_conclusion_draft",
            model_profile=model_profile,
        )
    background_tasks.add_task(
        _run_agent_validation_job,
        job_id,
        settings,
        task.id,
        model_profile,
        opening_message["id"] if opening_message else None,
        stage,
        stage_message["id"] if stage_message else None,
        normalized_acceptance_mode,
        stage_instruction,
    )
    return {
        "task_id": task.id,
        "status": "accepted",
        "stage": stage,
        "acceptance_mode": normalized_acceptance_mode,
        "message": "agent validation dispatched; poll task and agent messages",
        "messages": repo.list_agent_messages(task.id),
    }


def _request_agent_cancellation(task_id: str) -> None:
    request_agent_cancellation(task_id)


def _clear_agent_cancellation(task_id: str) -> None:
    clear_agent_cancellation(task_id)
    clear_pending_notebook_cancellation(task_id)


def _agent_cancellation_requested(task_id: str) -> bool:
    from marvis.agent.orchestrator import agent_cancellation_requested

    return agent_cancellation_requested(task_id)


def _raise_if_agent_cancelled(task_id: str) -> None:
    raise_if_agent_cancelled(task_id)


def _mark_agent_cancelled(repo: TaskRepository, task_id: str) -> None:
    try:
        task = repo.get_task(task_id)
        resume_status_by_current = {
            TaskStatus.SCANNED: TaskStatus.SCANNED,
            TaskStatus.RUNNING: TaskStatus.SCANNED,
            TaskStatus.COMPUTING_METRICS: TaskStatus.EXECUTED,
            TaskStatus.WRITING_ARTIFACTS: TaskStatus.REVIEW_REQUIRED,
        }
        resume_status = resume_status_by_current.get(task.status)
        if resume_status is None:
            return
        if task.status == resume_status:
            repo.update_status_message(
                task_id,
                AGENT_STOP_STATUS_MESSAGE,
                reason_code=TASK_STATUS_REASON_USER_CANCELLED,
            )
            return
        repo.update_status(
            task_id,
            resume_status,
            AGENT_STOP_STATUS_MESSAGE,
            expected=task.status,
            reason_code=TASK_STATUS_REASON_USER_CANCELLED,
        )
    except Exception:
        pass


def _agent_evidence(request: Request, task_id: str) -> dict:
    return _agent_evidence_from_settings(request.app.state.settings, task_id)


def _agent_chat_evidence(
    request: Request,
    repo: TaskRepository,
    task: TaskRecord,
    conversation: list[dict],
) -> dict:
    evidence = _agent_evidence(request, task.id)
    values, revision = repo.get_report_values(task.id)
    payload = _validation_results_payload_for_task(request, task)
    report_payload = report_field_payload(
        task,
        values,
        revision,
        metric_values=_metric_values_from_payload(payload),
    )
    evidence["report_fields"] = {
        "revision": report_payload["revision"],
        "text_values": report_payload["text_values"],
        "metric_values": report_payload["metric_values"],
    }
    evidence["visible_stage_summaries"] = [
        {
            "stage": message["stage"],
            "content": message["content"],
        }
        for message in conversation[-16:]
        if message.get("role") == "assistant"
        and message.get("stage") != "chat"
        and str(message.get("content") or "").strip()
    ]
    return evidence


def _agent_memory_context(
    request: Request,
    task: TaskRecord,
    *,
    stage: str,
    user_message: str = "",
    evidence: dict | None = None,
) -> dict | None:
    return _agent_memory_context_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        task,
        stage=stage,
        user_message=user_message,
        evidence=evidence,
    )


def _capture_user_preference_memory(
    request: Request,
    task_id: str,
    message: dict,
) -> None:
    # Memory policy gate (auto_distill): this function performs AUTOMATIC per-turn
    # capture of user messages as memory candidates. When the flag is OFF the user
    # has disabled automatic distillation, so capture must be a no-op. (The flag
    # governs only automatic capture; the explicit user-triggered
    # POST /agent-memory/consolidate endpoint stays functional regardless.)
    if not load_memory_policy(request.app.state.settings.workspace).auto_distill:
        return
    candidate = extract_user_preference(
        {
            "content": message.get("content"),
            "id": message.get("id"),
        }
    )
    if candidate is None:
        return
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        store.create(
            replace(
                candidate,
                source_task_id=task_id,
                source_message_id=str(message.get("id") or ""),
            )
        )
    except Exception as exc:
        logger.warning(
            "failed to save user preference memory for task %s: %s",
            task_id,
            exc,
        )
        return


def _set_agent_memory_status(request: Request, memory_id: str, status: str) -> dict:
    store = AgentMemoryStore(request.app.state.settings.db_path)
    try:
        entry = store.set_status(memory_id, status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"memory": _memory_entry_payload(entry), "events": store.list_events(memory_id)}


def _memory_entry_payload(entry) -> dict:
    return {
        "id": entry.id,
        "memory_type": entry.memory_type,
        "status": entry.status,
        "summary": entry.summary,
        "payload": entry.payload,
        "source_task_id": entry.source_task_id,
        "source_message_id": entry.source_message_id,
        "confidence": entry.confidence,
        "reason": entry.reason,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "deleted_at": entry.deleted_at,
    }


def _memory_distillation_payload(distillation) -> dict:
    return {
        "id": distillation.id,
        "kind": "distillation",
        "category": distillation.category,
        "memory_type": distillation.category,
        "scope_key": distillation.scope_key,
        "status": distillation.status,
        "summary": distillation.distilled_summary,
        "payload": distillation.structured,
        "source_memory_ids": list(distillation.source_memory_ids),
        "support_count": distillation.support_count,
        "confidence": distillation.confidence,
        "superseded_by": distillation.superseded_by,
        "created_at": distillation.created_at,
        "updated_at": distillation.updated_at,
    }


def _memory_distillation_detail(
    store: AgentMemoryStore,
    distillation,
) -> dict:
    source_memories = []
    for memory_id in distillation.source_memory_ids:
        try:
            source = store.get_entry(memory_id, include_deleted=True, audit=False)
        except KeyError:
            continue
        source_memories.append(_memory_entry_payload(source))
    predecessor = store.find_superseded_by(distillation.id)
    return {
        "distillation": _memory_distillation_payload(distillation),
        "source_memories": source_memories,
        "predecessor": (
            _memory_distillation_payload(predecessor)
            if predecessor is not None
            else None
        ),
        "events": store.list_distillation_events(distillation.id),
    }


def _memory_api_filter_match(
    item: dict,
    *,
    source_task_id: str | None,
    model_name: str | None,
    channel: str | None,
    month: str | None,
) -> bool:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    checks = (
        (source_task_id, item.get("source_task_id")),
        (model_name, payload.get("model_name")),
        (channel, payload.get("channel")),
        (month, payload.get("month")),
    )
    return all(
        expected in (None, "") or str(actual or "") == str(expected)
        for expected, actual in checks
    )


def _agent_memory_context_from_store(
    store: AgentMemoryStore,
    task: TaskRecord,
    *,
    stage: str,
    user_message: str = "",
    evidence: dict | None = None,
) -> dict | None:
    # Memory policy gate (reference_cross_task): when this flag is OFF the user
    # has disabled injecting prior-task agent memory into the prompt context, so
    # we short-circuit here and inject nothing. Gating in this single function
    # covers every caller (all turn handlers retrieve memory through this path).
    # The workspace is derived from the store's db_path (workspace/marvis.sqlite).
    workspace = Path(store.db_path).parent
    if not load_memory_policy(workspace).reference_cross_task:
        return None
    query = _agent_memory_query(task, user_message=user_message, evidence=evidence)
    packets = retrieve_with_distillations(store, query, limit=6)
    if not packets:
        return None
    raw_packets: list[tuple[str, dict]] = []
    for packet in packets:
        if packet.get("kind") == "distillation":
            continue
        memory_id = packet.get("id")
        if memory_id:
            raw_packets.append((str(memory_id), packet))
    found_ids = store.record_retrievals(
        [memory_id for memory_id, _ in raw_packets],
        task_id=task.id,
    )
    memories = [
        packet
        for packet in packets
        if packet.get("kind") == "distillation"
        or str(packet.get("id") or "") in found_ids
    ]
    if not memories:
        return None
    return {
        "scope": "cross_task_agent_memory",
        "stage": stage,
        "memories": memories,
    }


def _agent_memory_query(
    task: TaskRecord,
    *,
    user_message: str = "",
    evidence: dict | None = None,
) -> MemoryQuery:
    validation_results = (evidence or {}).get("validation_results")
    dimensions = _agent_memory_dimensions_from_validation_results(validation_results)
    return MemoryQuery(
        model_name=task.model_name or dimensions.get("model_name"),
        scope=dimensions.get("scope"),
        channel=dimensions.get("channel"),
        month=dimensions.get("month"),
        keywords=_agent_memory_keywords(task, user_message, dimensions),
    )


def _agent_memory_dimensions_from_validation_results(value) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    dimensions: dict[str, str] = {}
    for key in ("model_name", "model_version", "scope", "channel", "month"):
        item = value.get(key)
        if item not in (None, ""):
            dimensions[key] = str(item)
    basic_info = value.get("basic_info")
    if isinstance(basic_info, dict):
        for source_key, target_key in (
            ("model_name", "model_name"),
            ("model_version", "model_version"),
            ("model_scope", "scope"),
            ("scope", "scope"),
            ("channel", "channel"),
            ("month", "month"),
        ):
            item = basic_info.get(source_key)
            if target_key not in dimensions and item not in (None, ""):
                dimensions[target_key] = str(item)
    return dimensions


def _agent_memory_keywords(
    task: TaskRecord,
    user_message: str,
    dimensions: dict[str, str],
) -> tuple[str, ...]:
    values = [
        task.model_name,
        task.model_version,
        task.algorithm,
        dimensions.get("scope"),
        dimensions.get("channel"),
        dimensions.get("month"),
    ]
    compact_message = "".join(str(user_message or "").split())
    for marker in (
        "A卡",
        "B卡",
        "C卡",
        "额度",
        "利率",
        "前筛",
        "KS",
        "AUC",
        "PSI",
        "bad_flag",
        "RMC_SAMPLE_DF",
    ):
        if marker.lower() in compact_message.lower():
            values.append(marker)
    return tuple(
        dict.fromkeys(str(value).strip() for value in values if str(value or "").strip())
    )


def _audit_agent_memory_use(request: Request, message: dict, *, task_id: str) -> None:
    _audit_agent_memory_use_from_store(
        AgentMemoryStore(request.app.state.settings.db_path),
        message,
        task_id=task_id,
    )


def _audit_agent_memory_use_from_store(
    store: AgentMemoryStore,
    message: dict,
    *,
    task_id: str,
) -> None:
    metadata = message.get("metadata") or {}
    references = metadata.get("memory_references")
    if not isinstance(references, list):
        return
    for reference in references:
        if not isinstance(reference, dict):
            continue
        memory_id = reference.get("id")
        if not memory_id:
            continue
        if reference.get("kind") == "distillation":
            try:
                store.record_distillation_use(
                    str(memory_id),
                    task_id=task_id,
                    message_id=message.get("id"),
                    use_reason=str(reference.get("use_reason") or "agent"),
                )
            except (KeyError, ValueError):
                continue
            continue
        try:
            store.record_use(
                str(memory_id),
                task_id=task_id,
                message_id=message.get("id"),
                use_reason=str(reference.get("use_reason") or "agent"),
            )
        except KeyError:
            continue


def _agent_evidence_from_settings(settings, task_id: str) -> dict:
    task_dir = settings.tasks_dir / task_id
    validation_results = _read_json(task_dir / "outputs" / "validation_results.json")
    return {
        "scan": _read_json(task_dir / "execution" / "scan_result.json"),
        "notebook_steps": _read_json(task_dir / "execution" / "notebook_steps.json"),
        "contract": _read_json(task_dir / "execution" / "runtime_contract.json"),
        "reproducibility": _read_json(task_dir / "outputs" / "reproducibility_result.json"),
        "validation_results": _agent_validation_results_with_overfitting_check(validation_results),
    }


def _agent_validation_results_with_overfitting_check(validation_results):
    if not isinstance(validation_results, dict):
        return validation_results
    return {
        **validation_results,
        "overfitting_check": overfitting_check_from_validation_results(validation_results),
    }


# Driver-based task types run their deterministic plan flow through the agent
# endpoints in BOTH manual (user operates the controls, no LLM) and agent (an LLM
# operates them) mode. Validation, in contrast, only uses these endpoints in agent
# mode — its manual mode is the separate scan/notebook flow.
_DRIVER_AGENT_TASK_TYPES = frozenset(
    {TASK_TYPE_DATA_JOIN, TASK_TYPE_FEATURE_ANALYSIS, TASK_TYPE_MODELING}
)


def _require_agent_task(task: TaskRecord) -> None:
    if task.run_mode == "agent":
        return
    if task.run_mode == "manual" and task.task_type in _DRIVER_AGENT_TASK_TYPES:
        return
    raise HTTPException(status_code=409, detail="task is not in Agent mode")


# Agent task types that have a wired conversational backend today.
# validation -> validation agent (default path); modeling -> modeling agent.
# Other welcome-page entries (data_join / feature_analysis / strategy / vintage)
# are advertised in the create flow but their dedicated agents are not built yet.
# This is an ALLOWLIST, not a denylist: any task_type not listed here rejects
# explicitly instead of silently falling through to the validation agent on a
# goal prompt written for a different workflow. New task types must be added here
# as their agent flow is wired (see docs/plans/v2-completion-plan.md SS8 step 0).
_WIRED_AGENT_TASK_TYPES = frozenset(
    {TASK_TYPE_VALIDATION, TASK_TYPE_MODELING, TASK_TYPE_DATA_JOIN, TASK_TYPE_FEATURE_ANALYSIS}
)


def _require_wired_agent_task_type(task: TaskRecord) -> None:
    if task.task_type not in _WIRED_AGENT_TASK_TYPES:
        raise HTTPException(
            status_code=501,
            detail=(
                f"任务类型 '{task.task_type}' 的 Agent 流程尚未接入"
                "（当前仅支持 模型验证 / 模型开发 / 数据拼接 / 特征分析）"
            ),
        )


_VALID_EFFORTS = ("low", "medium", "high")


def _normalize_effort(effort: str | None) -> str:
    value = str(effort or "").strip().lower()
    return value if value in _VALID_EFFORTS else "high"


def _normalize_agent_acceptance_mode(mode: str | None) -> str:
    value = str(mode or "").strip().lower().replace("-", "_")
    return value if value in AGENT_ACCEPTANCE_MODES else AGENT_ACCEPTANCE_NORMAL


def _agent_auto_accept(mode: str | None) -> bool:
    return _normalize_agent_acceptance_mode(mode) == AGENT_ACCEPTANCE_AUTO


def _resolve_agent_model(
    request: Request, model_id: str | None, effort: str | None = None
) -> dict:
    try:
        profile = resolve_llm_model(request.app.state.settings.workspace, model_id)
    except LLMSettingsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # An explicit per-request effort wins; otherwise fall back to the value
    # persisted in the model profile (so the UI-configured effort is honored).
    if effort is not None:
        profile["reasoning_effort"] = _normalize_effort(effort)
    else:
        profile["reasoning_effort"] = _normalize_effort(profile.get("reasoning_effort"))
    return profile


def _model_metadata(model_profile: dict) -> dict:
    return {
        "model_id": model_profile.get("model_id"),
        "display_name": model_profile.get("display_name"),
        "model_name": model_profile.get("model_name"),
    }


def _add_streaming_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
) -> dict:
    return repo.add_agent_message(
        task_id,
        role="assistant",
        stage=stage,
        content="",
        metadata={**_model_metadata(model_profile), "streaming": True},
    )


def _add_and_stream_agent_message(
    repo: TaskRepository,
    task_id: str,
    *,
    stage: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    message = _add_streaming_agent_message(
        repo,
        task_id,
        stage=stage,
        model_profile=model_profile,
    )
    return _stream_agent_message(
        repo,
        message["id"],
        task_id=task_id,
        model_profile=model_profile,
        producer=producer,
    )


def _stream_agent_message(
    repo: TaskRepository,
    message_id: str,
    *,
    task_id: str,
    model_profile: dict,
    producer: Callable[[Callable[[str], None]], tuple[str, dict]],
) -> dict:
    parts: list[str] = []
    streaming_metadata = {**_model_metadata(model_profile), "streaming": True}

    def on_delta(delta: str) -> None:
        if not delta:
            return
        _raise_if_agent_cancelled(task_id)
        parts.append(delta)
        repo.update_agent_message(
            message_id,
            content="".join(parts),
            metadata=streaming_metadata,
        )

    try:
        _raise_if_agent_cancelled(task_id)
        content, metadata = producer(on_delta)
        _raise_if_agent_cancelled(task_id)
        final_metadata = {
            **metadata,
            **_model_metadata(model_profile),
            "streaming": False,
        }
        if parts:
            final_metadata["streamed"] = True
        _raise_if_agent_cancelled(task_id)
        return repo.update_agent_message(
            message_id,
            content=content,
            metadata=final_metadata,
        )
    except AgentValidationCancelled:
        cancelled_metadata = {
            **_model_metadata(model_profile),
            "streaming": False,
            "cancelled": True,
        }
        if parts:
            cancelled_metadata["streamed"] = True
        repo.update_agent_message(
            message_id,
            content="".join(parts),
            metadata=cancelled_metadata,
        )
        raise


def _require_confirmed_agent_conclusions(repo: TaskRepository, task: TaskRecord) -> None:
    if task.run_mode != "agent":
        return
    values, _ = repo.get_report_values(task.id)
    if agent_conclusions_confirmed(values):
        return
    raise HTTPException(
        status_code=409,
        detail="请先确认三段报告结论，确认后将生成 Word 报告",
    )


def _format_conclusion_values(values: dict[str, str]) -> str:
    labels = {
        "TEXT:pressure_test_summary": "压力测试总结",
        "TEXT:pressure_impact_recommendation": "压力影响建议",
        "TEXT:final_validation_conclusion": "最终验证结论",
    }
    ordered_keys = [
        "TEXT:pressure_test_summary",
        "TEXT:pressure_impact_recommendation",
        "TEXT:final_validation_conclusion",
    ]
    ordered_keys.extend(key for key in values if key not in labels)
    return "\n\n".join(
        f"{labels.get(key, key)}\n{value}"
        for key in ordered_keys
        if (value := values.get(key))
    )


def _pipeline_settings(
    request: Request,
    task: TaskRecord,
    feature_columns: list[str] | None,
) -> PipelineSettings:
    settings = request.app.state.settings
    return PipelineSettings(
        workspace=settings.workspace,
        db_path=settings.db_path,
        report_template_path=settings.report_template_path,
        feature_columns=feature_columns or task.feature_columns,
        notebook_kernel_name=load_execution_environment(settings.workspace).kernel_name,
    )


def _artifact_payload(artifact) -> dict:
    return {
        "role": artifact.role.value,
        "path": str(artifact.path),
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "risk_notes": artifact.risk_notes,
    }


def _artifact_ambiguities(artifacts) -> list[str]:
    role_counts: dict[str, int] = {}
    for artifact in artifacts:
        role_counts[artifact.role.value] = role_counts.get(artifact.role.value, 0) + 1
    return [
        f"{role} has {count} candidates; configure explicit path before validation"
        for role, count in sorted(role_counts.items())
        if count > 1
    ]


def _scan_notebook_steps(
    settings,
    task: TaskRecord,
    artifacts: list[FileArtifact],
) -> list[dict]:
    notebook_path, error = _resolve_scan_material(
        task=task,
        artifacts=artifacts,
        role=FileRole.NOTEBOOK,
        label="Notebook 文件",
        task_field="notebook_path",
    )
    if error or notebook_path is None:
        return []
    try:
        steps = notebook_step_preview(notebook_path)
    except Exception:
        return []
    execution_dir = settings.tasks_dir / task.id / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    (execution_dir / "notebook_steps.json").write_text(
        json.dumps({"steps": steps, "cells": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return steps


def _scan_preflight_checks(
    task: TaskRecord,
    artifacts: list[FileArtifact],
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    resolved_paths: dict[FileRole, Path] = {}
    for role, label, task_field in REQUIRED_SCAN_MATERIALS:
        path, error = _resolve_scan_material(
            task=task,
            artifacts=artifacts,
            role=role,
            label=label,
            task_field=task_field,
        )
        check_id = f"material_{role.value}"
        if error:
            checks.append(
                {
                    "id": check_id,
                    "label": label,
                    "status": "error",
                    "message": error,
                }
            )
            continue
        assert path is not None
        resolved_paths[role] = path
        checks.append(
            {
                "id": check_id,
                "label": label,
                "status": "success",
                "message": f"已识别：{path.name}",
            }
        )

    notebook_path = resolved_paths.get(FileRole.NOTEBOOK)
    if notebook_path is not None:
        try:
            result = precheck_notebook_contract(notebook_path)
        except NotebookContractError as exc:
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "error",
                    "message": _format_notebook_contract_error(exc),
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "error",
                    "message": f"Notebook 契约检查失败：{exc.__class__.__name__}: {exc}",
                }
            )
        else:
            target_note = f"，目标列 {result.target_col}" if result.target_col else ""
            algorithm_note = f"，算法 {result.algorithm}" if result.algorithm else ""
            checks.append(
                {
                    "id": "notebook_contract",
                    "label": "Notebook RMC 契约",
                    "status": "success",
                    "message": (
                        "已定义 RMC_SAMPLE_DF / RMC_SCORE_FN / RMC_TARGET_COL / "
                        f"RMC_ALGORITHM{target_note}{algorithm_note}"
                    ),
                }
            )
    return checks


def _resolve_scan_material(
    *,
    task: TaskRecord,
    artifacts: list[FileArtifact],
    role: FileRole,
    label: str,
    task_field: str,
) -> tuple[Path | None, str | None]:
    source_dir = Path(task.source_dir).resolve()
    explicit_value = getattr(task, task_field, None)
    if explicit_value:
        raw_path = Path(explicit_value)
        candidate = raw_path if raw_path.is_absolute() else source_dir / raw_path
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            return None, f"配置的 {label} 路径不存在：{candidate}"
        try:
            resolved = assert_within(source_dir, resolved)
        except PermissionError:
            return None, f"配置的 {label} 必须位于材料目录内：{resolved}"
        return resolved, None

    candidates = [artifact.path for artifact in artifacts if artifact.role == role]
    if not candidates:
        return None, f"缺少必需材料：{label}"
    if len(candidates) > 1:
        candidate_text = "、".join(path.name for path in candidates[:5])
        return None, f"{label} 有 {len(candidates)} 个候选，请在创建任务时显式指定：{candidate_text}"
    return candidates[0], None


def _metric_values_from_payload(payload: dict | None) -> dict[str, str]:
    if payload is None:
        return {}
    return computed_report_text_values_from_payload(payload)


def _metric_table_sections_from_payload(payload: dict | None) -> list[dict]:
    if payload is None:
        return []
    return metric_table_sections_from_payload(payload)


def _validation_results_payload_for_task(request: Request, task: TaskRecord) -> dict | None:
    if task.status not in {
        TaskStatus.WRITING_ARTIFACTS,
        TaskStatus.SUCCEEDED,
        TaskStatus.REVIEW_REQUIRED,
    } and not (
        task.status == TaskStatus.FAILED
        and task.status_message.startswith(REPORT_STAGE_FAILURE_PREFIX)
    ):
        return None
    result_path = (
        request.app.state.settings.tasks_dir
        / task.id
        / "outputs"
        / "validation_results.json"
    )
    if not result_path.exists():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_task_artifact_path(request: Request, artifact_path: str) -> Path:
    raw = unquote(str(artifact_path or ""))
    if not raw or raw.startswith(("/", "\\")):
        raise HTTPException(status_code=404, detail="artifact not found")
    root = request.app.state.settings.tasks_dir.resolve()
    candidate = (request.app.state.settings.workspace / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return candidate


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
