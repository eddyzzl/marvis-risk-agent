from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

from marvis.agent_memory.extractors import (
    extract_field_convention,
    extract_model_experience,
    extract_task_experience,
    extract_validation_pitfall,
)
from marvis.agent_memory.store import AgentMemoryStore
from marvis.artifacts import ArtifactUnitOfWork
from marvis.db import TaskRepository
from marvis.memory_policy import load_memory_policy
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    FileArtifact,
    FileRole,
    TaskRecord,
    TaskStatus,
)
from marvis.files import scan_source_dir
from marvis.execution_environment import (
    load_execution_environment,
    validate_execution_environment,
)
from marvis.notebook_cancellation import (
    NotebookCancelled,
    NotebookCancellationToken,
    register_notebook_cancellation,
    unregister_notebook_cancellation,
)
from marvis.notebook_contract import RuntimeContract, load_runtime_contract
from marvis.notebooks import (
    AppendedCellExecutionPolicy,
    NotebookExecutionSession,
    close_live_notebook_session,
    get_live_notebook_session,
    prepare_execution_notebook_v3,
    register_live_notebook_session,
    run_notebook,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.output.word import write_validation_word
from marvis.validation.results import (
    ConsistencyStatus,
    validation_results_from_dict,
)
from marvis.state_machine import IllegalTransition


@dataclass(frozen=True)
class PipelineSettings:
    workspace: Path
    db_path: Path
    report_template_path: Path
    feature_columns: list[str] = field(default_factory=list)
    notebook_kernel_name: str = "python3"
    notebook_memory_limit_mb: int | None = 4096
    notebook_isolated_execution: bool = True
    allow_legacy_live_notebook_execution: bool = False
    bin_count: int = 10
    random_sample_size: int = 1000
    random_seed: int = 42
    data_dict_feature_col: str = "特征名"
    data_dict_category_col: str = "类别"


class PipelineError(Exception):
    pass


class PipelineCancelled(PipelineError):
    def __init__(self, message: str, resume_status: TaskStatus) -> None:
        super().__init__(message)
        self.resume_status = resume_status


RMC_PMML_SCORE_COL = "__rmc_submitted_pmml_score__"
VALIDATION_RESULTS_PICKLE = "validation_results.pkl"
REPRODUCIBILITY_RESULT_JSON = "reproducibility_result.json"
METRICS_CANCEL_MARKER = "metrics_cancel.requested"
METRICS_OUTPUT_FILENAMES = (
    "validation_results.json",
    VALIDATION_RESULTS_PICKLE,
    "validation.xlsx",
    "metrics_notebook.log",
)
SCAN_STAGE_FAILURE_PREFIX = "材料扫描失败："
NOTEBOOK_STAGE_FAILURE_PREFIX = "模型可复现性验证失败："
METRICS_STAGE_FAILURE_PREFIX = "模型效果&稳定性验证失败："
REPORT_STAGE_FAILURE_PREFIX = "报告输出失败："
LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE = (
    "legacy live notebook execution requires notebook_isolated_execution=False "
    "and allow_legacy_live_notebook_execution=True"
)
V1_VALIDATION_APPENDED_CELL_KINDS = (
    "repro-pmml",
    "repro-compare",
    "metrics-prepare",
    "metrics-score",
    "metrics-basic",
    "metrics-ks",
    "metrics-psi",
    "metrics-binning",
    "metrics-stress",
    "metrics-output",
)
V1_VALIDATION_APPENDED_EXECUTION_POLICY = AppendedCellExecutionPolicy(
    scope="v1-validation-post-notebook",
    reason="run MARVIS-generated reproducibility and validation metric cells in the validated notebook kernel",
    allowed_marvis_kinds=V1_VALIDATION_APPENDED_CELL_KINDS,
)


def _metrics_cancel_marker_path(task_dir: Path) -> Path:
    return task_dir / "execution" / METRICS_CANCEL_MARKER


def _require_legacy_live_notebook_execution(settings: PipelineSettings) -> None:
    if settings.notebook_isolated_execution or not settings.allow_legacy_live_notebook_execution:
        raise PipelineError(LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE)


def run_notebook_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    execution_dir = task_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = _scan_artifacts(task) if stage_claimed else _scan_step(repo, task)
        _clear_generated_artifacts(task_dir, stage="notebook")
        notebook_path = _required_path(
            task, artifacts, FileRole.NOTEBOOK, "notebook", "notebook_path"
        )
        sample_path = _required_path(
            task, artifacts, FileRole.SAMPLE, "sample", "sample_path"
        )
        input_pmml_path = _required_path(
            task, artifacts, FileRole.MODEL_PMML, "input PMML", "pmml_path"
        )
        kernel_name = _execution_kernel_name(settings)
        cancellation_token = register_notebook_cancellation(task_id)
        live_session: NotebookExecutionSession | None = None
        try:
            close_live_notebook_session(task_id)
            if settings.notebook_isolated_execution:
                _notebook_step_v3(
                    repo=repo,
                    task=task,
                    source_notebook=notebook_path,
                    sample_path=sample_path,
                    execution_dir=execution_dir,
                    contract_meta_path=execution_dir / "runtime_contract.json",
                    code_scores_path=execution_dir / "code_model_scores.csv",
                    feature_importance_path=execution_dir / "feature_importance.csv",
                    model_params_path=execution_dir / "model_params.json",
                    notebook_steps_path=execution_dir / "notebook_steps.json",
                    kernel_name=kernel_name,
                    notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
                    stage_claimed=stage_claimed,
                    cancellation_token=cancellation_token,
                    keep_alive=False,
                    isolated=True,
                    mark_executed=False,
                    extra_code_cells=_build_reproducibility_cell_sources(
                        package_root=_package_root_for_notebook(),
                        task=task,
                        settings=settings,
                        input_pmml_path=input_pmml_path,
                        contract_meta_path=execution_dir / "runtime_contract.json",
                        output_path=task_dir / "outputs" / REPRODUCIBILITY_RESULT_JSON,
                    ),
                )
                if repo.get_task(task_id).status != TaskStatus.RUNNING:
                    return
                contract = load_runtime_contract(execution_dir / "runtime_contract.json")
                _sync_task_algorithm(repo, task, contract.algorithm)
                output_path = task_dir / "outputs" / REPRODUCIBILITY_RESULT_JSON
                if not output_path.exists():
                    raise PipelineError(
                        "notebook reproducibility evidence did not produce output"
                    )
                repo.update_status(
                    task_id,
                    TaskStatus.EXECUTED,
                    message="notebook executed",
                    expected=TaskStatus.RUNNING,
                )
                return

            _require_legacy_live_notebook_execution(settings)
            live_session = _notebook_step_v3(
                repo=repo,
                task=task,
                source_notebook=notebook_path,
                sample_path=sample_path,
                execution_dir=execution_dir,
                contract_meta_path=execution_dir / "runtime_contract.json",
                code_scores_path=execution_dir / "code_model_scores.csv",
                feature_importance_path=execution_dir / "feature_importance.csv",
                model_params_path=execution_dir / "model_params.json",
                notebook_steps_path=execution_dir / "notebook_steps.json",
                kernel_name=kernel_name,
                notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
                stage_claimed=stage_claimed,
                cancellation_token=cancellation_token,
                keep_alive=True,
                mark_executed=False,
            )
            if live_session is not None:
                contract = load_runtime_contract(execution_dir / "runtime_contract.json")
                task = _sync_task_algorithm(repo, task, contract.algorithm)
                _write_reproducibility_result_in_session(
                    session=live_session,
                    task=task,
                    settings=settings,
                    input_pmml_path=input_pmml_path,
                    contract_meta_path=execution_dir / "runtime_contract.json",
                    output_path=task_dir / "outputs" / REPRODUCIBILITY_RESULT_JSON,
                )
                repo.update_status(
                    task_id,
                    TaskStatus.EXECUTED,
                    message="notebook executed",
                    expected=TaskStatus.RUNNING,
                )
                register_live_notebook_session(task_id, live_session)
        except PipelineCancelled as exc:
            if live_session is not None:
                live_session.close()
            _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
            return
        except Exception:
            if live_session is not None:
                live_session.close()
            raise
        finally:
            unregister_notebook_cancellation(task_id, cancellation_token)
    except PipelineCancelled as exc:
        _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
        return
    except PipelineError as exc:
        message = _stage_failure_message(NOTEBOOK_STAGE_FAILURE_PREFIX, str(exc))
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(str(exc), default="notebook"),
            message=message,
        )
        raise
    except Exception as exc:
        message = _stage_failure_message(
            NOTEBOOK_STAGE_FAILURE_PREFIX,
            f"{exc.__class__.__name__}: {exc}",
        )
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(str(exc), default="notebook"),
            message=message,
        )
        raise


def run_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    metrics_work_dir = outputs_dir / ".metrics-stage-work"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_uow: ArtifactUnitOfWork | None = None
    try:
        if not stage_claimed and task.status != TaskStatus.EXECUTED:
            raise PipelineError(
                f"metrics require executed notebook; current status is {task.status.value}"
            )
        artifacts = _scan_artifacts(task)
        input_pmml_path = _required_path(
            task, artifacts, FileRole.MODEL_PMML, "input PMML", "pmml_path"
        )
        dictionary_path = _required_path(
            task,
            artifacts,
            FileRole.DATA_DICTIONARY,
            "data dictionary",
            "dictionary_path",
        )
        if not stage_claimed:
            repo.update_status(
                task_id,
                TaskStatus.COMPUTING_METRICS,
                message="computing metrics",
                expected=TaskStatus.EXECUTED,
            )
        cancellation_token = register_notebook_cancellation(task_id)
        try:
            live_session = get_live_notebook_session(task_id)
            if settings.notebook_isolated_execution:
                if live_session is not None:
                    close_live_notebook_session(task_id)
                    live_session = None
            elif not settings.allow_legacy_live_notebook_execution:
                if live_session is not None:
                    close_live_notebook_session(task_id)
                raise PipelineError(LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE)
            elif live_session is None:
                raise PipelineError(
                    "live notebook kernel is not available; rerun notebook stage before metrics"
                )
            _unlink_if_exists(_metrics_cancel_marker_path(task_dir))
            _remove_dir_if_exists(metrics_work_dir)
            metrics_work_dir.mkdir(parents=True, exist_ok=True)
            contract = load_runtime_contract(execution_dir / "runtime_contract.json")
            task = _sync_task_algorithm(repo, task, contract.algorithm)
            model_meta_path = execution_dir / "model_meta.json"
            _write_model_meta_from_contract(contract, model_meta_path)
            if live_session is None:
                notebook_path = _required_path(
                    task, artifacts, FileRole.NOTEBOOK, "notebook", "notebook_path"
                )
                sample_path = _required_path(
                    task, artifacts, FileRole.SAMPLE, "sample", "sample_path"
                )
                _notebook_step_v3(
                    repo=repo,
                    task=task,
                    source_notebook=notebook_path,
                    sample_path=sample_path,
                    execution_dir=execution_dir,
                    contract_meta_path=execution_dir / "runtime_contract.json",
                    code_scores_path=execution_dir / "code_model_scores.csv",
                    feature_importance_path=execution_dir / "feature_importance.csv",
                    model_params_path=execution_dir / "model_params.json",
                    notebook_steps_path=execution_dir / "notebook_steps.json",
                    kernel_name=_execution_kernel_name(settings),
                    notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
                    stage_claimed=True,
                    cancellation_token=cancellation_token,
                    keep_alive=False,
                    isolated=True,
                    mark_executed=False,
                    cancel_message="metrics cancelled",
                    cancel_resume_status=TaskStatus.EXECUTED,
                    extra_code_cells=_build_metrics_cell_sources(
                        package_root=_package_root_for_notebook(),
                        task=task,
                        settings=settings,
                        dictionary_path=dictionary_path,
                        input_pmml_path=input_pmml_path,
                        contract=contract,
                        model_meta_path=model_meta_path,
                        reproducibility_json_path=outputs_dir
                        / REPRODUCIBILITY_RESULT_JSON,
                        results_json_path=metrics_work_dir / "validation_results.json",
                        excel_path=metrics_work_dir / "validation.xlsx",
                    ),
                )
                if repo.get_task(task_id).status != TaskStatus.COMPUTING_METRICS:
                    return
                _require_metrics_outputs(metrics_work_dir)
            else:
                previous_token = getattr(live_session, "cancellation_token", None)
                setattr(live_session, "cancellation_token", cancellation_token)
                live_client = getattr(live_session, "client", None)
                if live_client is not None:
                    cancellation_token.bind_client(live_client)
                try:
                    _write_metrics_results_in_session(
                        session=live_session,
                        task=task,
                        settings=settings,
                        dictionary_path=dictionary_path,
                        input_pmml_path=input_pmml_path,
                        contract=contract,
                        model_meta_path=model_meta_path,
                        outputs_dir=metrics_work_dir,
                        reproducibility_json_path=outputs_dir / REPRODUCIBILITY_RESULT_JSON,
                    )
                finally:
                    setattr(live_session, "cancellation_token", previous_token)
        finally:
            unregister_notebook_cancellation(task_id, cancellation_token)
        close_live_notebook_session(task_id)
        _require_metrics_outputs(metrics_work_dir)
        metrics_uow = _stage_metrics_outputs_for_commit(
            task_dir=task_dir,
            outputs_dir=outputs_dir,
            metrics_work_dir=metrics_work_dir,
        )
        metrics_uow.finalize_with_connection(
            repo.transaction,
            lambda conn: repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.WRITING_ARTIFACTS,
                message="metrics and excel generated",
                expected=TaskStatus.COMPUTING_METRICS,
                begin_immediate=True,
            ),
        )
        _capture_agent_memory_for_metrics_success(
            repo=repo,
            task_id=task_id,
            outputs_dir=outputs_dir,
        )
    except PipelineCancelled as exc:
        _rollback_artifact_uow(metrics_uow)
        _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
        return
    except PipelineError as exc:
        _rollback_artifact_uow(metrics_uow)
        message = _stage_failure_message(METRICS_STAGE_FAILURE_PREFIX, str(exc))
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(str(exc), default="execution"),
            message=message,
        )
        raise
    except Exception as exc:
        _rollback_artifact_uow(metrics_uow)
        message = _stage_failure_message(
            METRICS_STAGE_FAILURE_PREFIX,
            f"{exc.__class__.__name__}: {exc}",
        )
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(str(exc), default="execution"),
            message=message,
        )
        raise
    finally:
        _remove_dir_if_exists(metrics_work_dir)


def run_report_stage(*, task_id: str, settings: PipelineSettings) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    report_path = outputs_dir / "validation_report.docx"
    temp_report_path = outputs_dir / ".validation_report.docx.tmp"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cancellation_token = register_notebook_cancellation(task_id)
    report_uow: ArtifactUnitOfWork | None = None
    try:
        cancellation_token.raise_if_cancelled()
        if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
            raise PipelineError(
                f"word output requires generated metrics; current status is {task.status.value}"
            )
        _unlink_if_exists(temp_report_path)
        report_uow = ArtifactUnitOfWork()
        staged_report = report_uow.stage_file(outputs_dir, report_path.name)
        staged_images = report_uow.stage_directory(task_dir, images_dir.name)
        results = _load_validation_results(outputs_dir)
        report_values, _ = repo.get_report_values(task_id)
        word_result = write_validation_word(
            results,
            template_path=settings.report_template_path,
            output_path=staged_report.path,
            image_output_dir=staged_images.path,
            report_values=report_values,
        )
        cancellation_token.raise_if_cancelled()
        if word_result.unresolved_placeholders:
            if task.status != TaskStatus.REVIEW_REQUIRED:
                try:
                    report_uow.finalize_with_connection(
                        repo.transaction,
                        lambda conn: repo.update_status_on_connection(
                            conn,
                            task_id,
                            TaskStatus.REVIEW_REQUIRED,
                            message="报告已生成，需人工复核",
                            expected=TaskStatus.WRITING_ARTIFACTS,
                            begin_immediate=True,
                        ),
                    )
                except IllegalTransition:
                    # Another concurrent report request already advanced the task;
                    # this run lost the race, so leave the winner's result intact.
                    return
            else:
                report_uow.finalize(lambda: None)
            return
        terminal_status = (
            TaskStatus.REVIEW_REQUIRED
            if results.reproducibility.summary.status is ConsistencyStatus.FAIL
            else TaskStatus.SUCCEEDED
        )
        try:
            report_uow.finalize_with_connection(
                repo.transaction,
                lambda conn: repo.update_status_on_connection(
                    conn,
                    task_id,
                    terminal_status,
                    message=(
                        "验证已完成，需人工复核报告"
                        if terminal_status is TaskStatus.REVIEW_REQUIRED
                        else "pipeline succeeded"
                    ),
                    expected={TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED},
                    begin_immediate=True,
                ),
            )
        except IllegalTransition:
            # Lost a concurrent report race; the other worker already finalized
            # the task. Do not fall through to _mark_failed and clobber it.
            return
    except NotebookCancelled:
        _rollback_report_uow(report_uow)
        _unlink_if_exists(temp_report_path)
        _mark_cancelled(repo, task_id, TaskStatus.REVIEW_REQUIRED, "report cancelled")
        return
    except PipelineError as exc:
        _rollback_report_uow(report_uow)
        _unlink_if_exists(temp_report_path)
        message = _stage_failure_message(REPORT_STAGE_FAILURE_PREFIX, str(exc))
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind="report",
            message=message,
        )
        raise
    except Exception as exc:
        _rollback_report_uow(report_uow)
        _unlink_if_exists(temp_report_path)
        message = _stage_failure_message(
            REPORT_STAGE_FAILURE_PREFIX,
            f"{exc.__class__.__name__}: {exc}",
        )
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind="report",
            message=message,
        )
        raise
    finally:
        unregister_notebook_cancellation(task_id, cancellation_token)


def _rollback_report_uow(uow: ArtifactUnitOfWork | None) -> None:
    _rollback_artifact_uow(uow)


def _rollback_artifact_uow(uow: ArtifactUnitOfWork | None) -> None:
    if uow is None:
        return
    try:
        uow.rollback()
    except Exception:
        pass


def _stage_metrics_outputs_for_commit(
    *,
    task_dir: Path,
    outputs_dir: Path,
    metrics_work_dir: Path,
) -> ArtifactUnitOfWork:
    uow = ArtifactUnitOfWork()
    for name in METRICS_OUTPUT_FILENAMES:
        source = metrics_work_dir / name
        destination = outputs_dir / name
        if source.exists() or source.is_symlink():
            staged = uow.stage_file(outputs_dir, name)
            shutil.move(str(source), staged.path)
        elif destination.exists() or destination.is_symlink():
            uow.remove_path(destination)
    report_path = outputs_dir / "validation_report.docx"
    if report_path.exists() or report_path.is_symlink():
        uow.remove_path(report_path)
    images_dir = task_dir / "images"
    if images_dir.exists() or images_dir.is_symlink():
        uow.remove_path(images_dir)
    return uow


def _write_reproducibility_result_in_session(
    *,
    session: NotebookExecutionSession,
    task: TaskRecord,
    settings: PipelineSettings,
    input_pmml_path: Path,
    contract_meta_path: Path,
    output_path: Path,
    cancel_message: str = "notebook cancelled",
    cancel_resume_status: TaskStatus = TaskStatus.SCANNED,
) -> None:
    cell_sources = _build_reproducibility_cell_sources(
        package_root=_package_root_for_notebook(),
        task=task,
        settings=settings,
        input_pmml_path=input_pmml_path,
        contract_meta_path=contract_meta_path,
        output_path=output_path,
    )
    planned_cells = _append_injected_cells(session, cell_sources)
    for kind, source, cell_index in planned_cells:
        _execute_injected_cell(
            session=session,
            source=source,
            cell_index=cell_index,
            log_path=output_path.parent / "reproducibility_notebook.log",
            metadata_kind=kind,
            failure_label="notebook reproducibility evidence",
            cancel_message=cancel_message,
            cancel_resume_status=cancel_resume_status,
        )
    if not output_path.exists():
        raise PipelineError("notebook reproducibility evidence did not produce output")


def _write_metrics_results_in_session(
    *,
    session: NotebookExecutionSession,
    task: TaskRecord,
    settings: PipelineSettings,
    dictionary_path: Path,
    input_pmml_path: Path,
    contract: RuntimeContract,
    model_meta_path: Path,
    outputs_dir: Path,
    reproducibility_json_path: Path | None = None,
) -> None:
    if reproducibility_json_path is None:
        reproducibility_json_path = outputs_dir / REPRODUCIBILITY_RESULT_JSON
    cell_sources = _build_metrics_cell_sources(
        package_root=_package_root_for_notebook(),
        task=task,
        settings=settings,
        dictionary_path=dictionary_path,
        input_pmml_path=input_pmml_path,
        contract=contract,
        model_meta_path=model_meta_path,
        reproducibility_json_path=reproducibility_json_path,
        results_json_path=outputs_dir / "validation_results.json",
        excel_path=outputs_dir / "validation.xlsx",
    )
    planned_cells = _append_injected_cells(session, cell_sources)
    for kind, source, cell_index in planned_cells:
        _execute_injected_cell(
            session=session,
            source=source,
            cell_index=cell_index,
            log_path=outputs_dir / "metrics_notebook.log",
            metadata_kind=kind,
            failure_label="notebook metrics",
            cancel_message="metrics cancelled",
            cancel_resume_status=TaskStatus.EXECUTED,
        )
    required = [
        outputs_dir / "validation_results.json",
        outputs_dir / "validation.xlsx",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise PipelineError("notebook metrics did not produce: " + ", ".join(missing))


def _require_metrics_outputs(outputs_dir: Path) -> None:
    required = [
        outputs_dir / "validation_results.json",
        outputs_dir / "validation.xlsx",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise PipelineError("notebook metrics did not produce: " + ", ".join(missing))


def _append_injected_cells(
    session: NotebookExecutionSession,
    cell_sources: list[tuple[str, str]],
) -> list[tuple[str, str, int | None]]:
    append_code_cell = getattr(session, "append_code_cell", None)
    execute_existing_code_cell = getattr(session, "execute_existing_code_cell", None)
    if not callable(append_code_cell) or not callable(execute_existing_code_cell):
        return [(kind, source, None) for kind, source in cell_sources]
    return [
        (
            kind,
            source,
            append_code_cell(
                source,
                metadata={"marvis": kind},
                record_progress=True,
            ),
        )
        for kind, source in cell_sources
    ]


def _execute_injected_cell(
    *,
    session: NotebookExecutionSession,
    source: str,
    cell_index: int | None = None,
    log_path: Path,
    metadata_kind: str,
    failure_label: str,
    cancel_message: str,
    cancel_resume_status: TaskStatus,
) -> None:
    execute_existing_code_cell = getattr(session, "execute_existing_code_cell", None)
    if cell_index is not None and callable(execute_existing_code_cell):
        result = execute_existing_code_cell(
            cell_index,
            log_path=log_path,
            record_progress=True,
        )
    else:
        result = session.execute_code_cell(
            source,
            log_path=log_path,
            metadata={"marvis": metadata_kind},
            record_progress=True,
        )
    if getattr(result, "cancelled", False):
        raise PipelineCancelled(cancel_message, cancel_resume_status)
    if not result.succeeded:
        raise PipelineError(
            f"{failure_label} failed at cell {result.failed_cell_index}: "
            f"{result.error_name}: {result.error_value}"
        )


def _package_root_for_notebook() -> Path:
    return Path(__file__).resolve().parents[1]


def _notebook_package_prelude(package_root: Path) -> list[str]:
    package_root_text = Path(package_root).as_posix()
    return [
        "import sys as _rmc_sys",
        f"_rmc_package_root = {package_root_text!r}",
        "if _rmc_package_root not in _rmc_sys.path:",
        "    _rmc_sys.path.insert(0, _rmc_package_root)",
    ]


def _build_reproducibility_cell_sources(
    *,
    package_root: Path,
    task: TaskRecord,
    settings: PipelineSettings,
    input_pmml_path: Path,
    contract_meta_path: Path,
    output_path: Path,
) -> list[tuple[str, str]]:
    payload = {
        "input_pmml_path": str(input_pmml_path),
        "contract_meta_path": str(contract_meta_path),
        "output_path": str(output_path),
        "random_sample_size": settings.random_sample_size,
        "random_seed": settings.random_seed,
        "bin_count": settings.bin_count,
        "fallback_split_col": task.split_col,
        "fallback_time_col": task.time_col,
    }
    pmml_lines = [
        "# Injected by marvis v3 (reproducibility evidence).",
        *_notebook_package_prelude(package_root),
        "import json as _rmc_json",
        "from pathlib import Path as _RmcPath",
        "import pandas as _rmc_pd",
        "from marvis.validation.config import ValidationConfig as _RmcValidationConfig",
        "from marvis.validation.in_memory_scores import load_code_model_scores as _rmc_load_code_model_scores",
        "from marvis.validation.pmml_scoring import load_pmml_scorer as _rmc_load_pmml_scorer",
        "from marvis.validation.reproducibility import _code_scores_by_row_index as _rmc_code_scores_by_row_index",
        f"_rmc_payload = {_json_literal(payload)}",
        "if 'RMC_SAMPLE_DF' not in globals() or not isinstance(RMC_SAMPLE_DF, _rmc_pd.DataFrame):",
        "    raise NameError('RMC_SAMPLE_DF must be defined as a pandas DataFrame before reproducibility evidence')",
        "_rmc_contract = _rmc_json.loads(_RmcPath(_rmc_payload['contract_meta_path']).read_text(encoding='utf-8'))",
        "_rmc_config = _RmcValidationConfig(",
        "    target_col=str(_rmc_contract['target_col']),",
        f"    score_col={RMC_PMML_SCORE_COL!r},",
        "    split_col=_rmc_contract.get('split_col') or _rmc_payload['fallback_split_col'],",
        "    time_col=_rmc_contract.get('time_col') or _rmc_payload['fallback_time_col'],",
        "    feature_columns=[],",
        "    bin_count=int(_rmc_payload['bin_count']),",
        "    random_sample_size=int(_rmc_payload['random_sample_size']),",
        "    random_seed=int(_rmc_payload['random_seed']),",
        "    score_decimal_places=int(_rmc_contract.get('score_decimal_places') or 6),",
        ")",
        "_rmc_repro_sample = RMC_SAMPLE_DF.copy().reset_index(drop=True)",
        "_rmc_repro_take = min(_rmc_config.random_sample_size, len(_rmc_repro_sample))",
        "_rmc_repro_drawn = _rmc_repro_sample.sample(",
        "    n=_rmc_repro_take,",
        "    random_state=_rmc_config.random_seed,",
        ")",
        "_rmc_repro_code_scores = _rmc_code_scores_by_row_index(",
        "    _rmc_load_code_model_scores(_RmcPath(_rmc_contract['code_model_scores_path']))",
        ")",
        "_rmc_repro_missing_indexes = [",
        "    idx for idx in _rmc_repro_drawn.index",
        "    if idx not in _rmc_repro_code_scores.index",
        "]",
        "if _rmc_repro_missing_indexes:",
        "    _rmc_preview = ', '.join(str(idx) for idx in _rmc_repro_missing_indexes[:10])",
        "    raise ValueError(f'missing code-model scores for sampled rows: {_rmc_preview}')",
        "_rmc_repro_scores_code = _rmc_repro_code_scores.loc[_rmc_repro_drawn.index].astype(float).tolist()",
        "_rmc_repro_pmml_scorer = _rmc_load_pmml_scorer(",
        "    _RmcPath(_rmc_payload['input_pmml_path']),",
        "    positive_output_field=str(_rmc_contract.get('pmml_output_field') or 'probability_1'),",
        ")",
        "_rmc_repro_scores_pmml = _rmc_repro_pmml_scorer.score(_rmc_repro_drawn.copy())",
        "if len(_rmc_repro_scores_pmml) != len(_rmc_repro_drawn):",
        "    raise ValueError(",
        "        f'submitted PMML scorer returned {len(_rmc_repro_scores_pmml)} scores for {len(_rmc_repro_drawn)} rows'",
        "    )",
    ]
    compare_lines = [
        "# Injected by marvis v3 (reproducibility compare).",
        *_notebook_package_prelude(package_root),
        "import json as _rmc_json",
        "from dataclasses import asdict as _rmc_asdict",
        "from pathlib import Path as _RmcPath",
        "from marvis.validation.reproducibility import _nullable_score as _rmc_nullable_score",
        "from marvis.validation.reproducibility import scores_match_at_precision as _rmc_scores_match_at_precision",
        "from marvis.validation.results import ConsistencyStatus as _RmcConsistencyStatus",
        "from marvis.validation.results import ConsistencySummary as _RmcConsistencySummary",
        "from marvis.validation.results import ReproducibilityResult as _RmcReproducibilityResult",
        "from marvis.validation.results import ScoreCompareRow as _RmcScoreCompareRow",
        f"_rmc_payload = {_json_literal(payload)}",
        "_rmc_rows = []",
        "_rmc_match_count = 0",
        "_rmc_mismatch_count = 0",
        "_rmc_max_abs_diff = 0.0",
        "for _rmc_row_index, _rmc_code_score, _rmc_pmml_score in zip(",
        "    _rmc_repro_drawn.index,",
        "    _rmc_repro_scores_code,",
        "    _rmc_repro_scores_pmml,",
        "):",
        "    _rmc_code_score_float = float(_rmc_code_score)",
        "    _rmc_pmml_score_float = _rmc_nullable_score(_rmc_pmml_score)",
        "    _rmc_abs_diff = None if _rmc_pmml_score_float is None else abs(_rmc_code_score_float - _rmc_pmml_score_float)",
        "    _rmc_matched = False if _rmc_pmml_score_float is None else _rmc_scores_match_at_precision(",
        "        _rmc_code_score_float,",
        "        _rmc_pmml_score_float,",
        "        _rmc_config.score_decimal_places,",
        "    )",
        "    _rmc_rows.append(_RmcScoreCompareRow(",
        "        row_index=_rmc_row_index,",
        "        score_code_model=_rmc_code_score_float,",
        "        score_submitted_pmml=_rmc_pmml_score_float,",
        "        abs_diff=_rmc_abs_diff,",
        "        matched=_rmc_matched,",
        "    ))",
        "    if _rmc_abs_diff is not None:",
        "        _rmc_max_abs_diff = max(_rmc_max_abs_diff, _rmc_abs_diff)",
        "    if _rmc_matched:",
        "        _rmc_match_count += 1",
        "    else:",
        "        _rmc_mismatch_count += 1",
        "_rmc_result = _RmcReproducibilityResult(",
        "    sample_size=_rmc_repro_take,",
        "    seed=_rmc_config.random_seed,",
        "    rows=_rmc_rows,",
        "    summary=_RmcConsistencySummary(",
        "        match_count=_rmc_match_count,",
        "        mismatch_count=_rmc_mismatch_count,",
        "        max_abs_diff=float(_rmc_max_abs_diff),",
        "        status=_RmcConsistencyStatus.PASS if _rmc_mismatch_count == 0 else _RmcConsistencyStatus.FAIL,",
        "    ),",
        ")",
        "_rmc_output_path = _RmcPath(_rmc_payload['output_path'])",
        "_rmc_output_path.parent.mkdir(parents=True, exist_ok=True)",
        "_rmc_output_path.write_text(",
        "    _rmc_json.dumps(_rmc_asdict(_rmc_result), ensure_ascii=False, indent=2),",
        "    encoding='utf-8',",
        ")",
    ]
    return [
        ("repro-pmml", "\n".join(pmml_lines)),
        ("repro-compare", "\n".join(compare_lines)),
    ]


def _build_metrics_cell_source(
    *,
    package_root: Path,
    task: TaskRecord,
    settings: PipelineSettings,
    dictionary_path: Path,
    input_pmml_path: Path,
    contract: RuntimeContract,
    model_meta_path: Path,
    reproducibility_json_path: Path,
    results_json_path: Path,
    excel_path: Path,
) -> str:
    return "\n\n".join(
        source
        for _, source in _build_metrics_cell_sources(
            package_root=package_root,
            task=task,
            settings=settings,
            dictionary_path=dictionary_path,
            input_pmml_path=input_pmml_path,
            contract=contract,
            model_meta_path=model_meta_path,
            reproducibility_json_path=reproducibility_json_path,
            results_json_path=results_json_path,
            excel_path=excel_path,
        )
    )


def _build_metrics_cell_sources(
    *,
    package_root: Path,
    task: TaskRecord,
    settings: PipelineSettings,
    dictionary_path: Path,
    input_pmml_path: Path,
    contract: RuntimeContract,
    model_meta_path: Path,
    reproducibility_json_path: Path,
    results_json_path: Path,
    excel_path: Path,
) -> list[tuple[str, str]]:
    payload = {
        "model_name": task.model_name,
        "model_version": task.model_version,
        "algorithm": _algorithm(contract.algorithm),
        "dictionary_path": str(dictionary_path),
        "input_pmml_path": str(input_pmml_path),
        "model_meta_path": str(model_meta_path),
        "reproducibility_json_path": str(reproducibility_json_path),
        "results_json_path": str(results_json_path),
        "excel_path": str(excel_path),
        "metrics_cancel_path": str(_metrics_cancel_marker_path(results_json_path.parent.parent)),
        "target_col": contract.target_col,
        "split_col": contract.split_col or task.split_col or "",
        "time_col": contract.time_col or task.time_col or "",
        "pmml_output_field": contract.pmml_output_field,
        "score_decimal_places": contract.score_decimal_places,
        "code_scores_path": str(contract.code_model_scores_path),
        "bin_count": settings.bin_count,
        "random_sample_size": settings.random_sample_size,
        "random_seed": settings.random_seed,
        "data_dict_feature_col": settings.data_dict_feature_col,
        "data_dict_category_col": settings.data_dict_category_col,
    }
    prepare_lines = [
        "# Injected by marvis v3 (metrics).",
        *_notebook_package_prelude(package_root),
        "import json as _rmc_json",
        "from dataclasses import asdict as _rmc_asdict",
        "from pathlib import Path as _RmcPath",
        "import pandas as _rmc_pd",
        "from marvis.output.excel import write_validation_excel as _rmc_write_validation_excel",
        "from marvis.validation.checks import finite_score_series as _rmc_finite_score_series",
        "from marvis.validation.config import ValidationConfig as _RmcValidationConfig",
        "from marvis.validation.effectiveness import build_effectiveness_result as _rmc_build_effectiveness_result",
        "from marvis.validation.effectiveness import compute_bin_tables as _rmc_compute_bin_tables",
        "from marvis.validation.effectiveness import compute_monthly_ks as _rmc_compute_monthly_ks",
        "from marvis.validation.effectiveness import compute_monthly_psi as _rmc_compute_monthly_psi",
        "from marvis.validation.effectiveness import compute_overall_ks as _rmc_compute_overall_ks",
        "from marvis.validation.effectiveness import compute_overall_psi as _rmc_compute_overall_psi",
        "from marvis.validation.effectiveness import compute_psi_stability_table as _rmc_compute_psi_stability_table",
        "from marvis.validation.effectiveness import compute_roc_ks_curves as _rmc_compute_roc_ks_curves",
        "from marvis.validation.effectiveness import prepare_effectiveness_context as _rmc_prepare_effectiveness_context",
        "from marvis.validation.engine import _filter_feature_categories as _rmc_filter_feature_categories",
        "from marvis.validation.engine import _model_features as _rmc_model_features",
        "from marvis.validation.in_memory_scores import load_code_model_scores as _rmc_load_code_model_scores",
        "from marvis.validation.results import ValidationResults as _RmcValidationResults",
        "from marvis.validation.results import ConsistencyStatus as _RmcConsistencyStatus",
        "from marvis.validation.results import ConsistencySummary as _RmcConsistencySummary",
        "from marvis.validation.results import ReproducibilityResult as _RmcReproducibilityResult",
        "from marvis.validation.results import ScoreCompareRow as _RmcScoreCompareRow",
        "from marvis.validation.sample_stats import run_basic_info as _rmc_run_basic_info",
        "from marvis.validation.stress_test import load_feature_categories as _rmc_load_feature_categories",
        "from marvis.validation.stress_test import run_stress_test as _rmc_run_stress_test",
        f"_rmc_payload = {_json_literal(payload)}",
        "def _rmc_load_dictionary(path_value):",
        "    path = _RmcPath(path_value)",
        "    suffix = path.suffix.lower()",
        "    if suffix == '.csv':",
        "        return _rmc_pd.read_csv(path)",
        "    if suffix == '.xlsx':",
        "        return _rmc_pd.read_excel(path)",
        "    raise ValueError(f'unsupported data dictionary format: {suffix}')",
        "def _rmc_reproducibility_from_json(path_value):",
        "    payload = _rmc_json.loads(_RmcPath(path_value).read_text(encoding='utf-8'))",
        "    summary = payload['summary']",
        "    return _RmcReproducibilityResult(",
        "        sample_size=int(payload['sample_size']),",
        "        seed=int(payload['seed']),",
        "        rows=[",
        "            _RmcScoreCompareRow(",
        "                row_index=row.get('row_index', 0),",
        "                score_code_model=row.get('score_code_model'),",
        "                score_submitted_pmml=row.get('score_submitted_pmml'),",
        "                abs_diff=row.get('abs_diff'),",
        "                matched=bool(row.get('matched')),",
        "            )",
        "            for row in payload.get('rows', [])",
        "        ],",
        "        summary=_RmcConsistencySummary(",
        "            match_count=int(summary['match_count']),",
        "            mismatch_count=int(summary['mismatch_count']),",
        "            max_abs_diff=float(summary['max_abs_diff']),",
        "            status=_RmcConsistencyStatus(summary['status']),",
        "        ),",
        "    )",
        "def _rmc_raise_if_metrics_cancelled():",
        "    if _RmcPath(_rmc_payload['metrics_cancel_path']).exists():",
        "        raise KeyboardInterrupt('metrics cancelled')",
        "if 'RMC_SAMPLE_DF' not in globals() or not isinstance(RMC_SAMPLE_DF, _rmc_pd.DataFrame):",
        "    raise NameError('RMC_SAMPLE_DF must be defined as a pandas DataFrame before metrics')",
        "_rmc_sample = RMC_SAMPLE_DF.copy()",
        "_rmc_missing_cols = [",
        "    f\"{label}='{column}'\"",
        "    for label, column in {",
        "        'target_col': _rmc_payload['target_col'],",
        "        'split_col': _rmc_payload['split_col'],",
        "        'time_col': _rmc_payload['time_col'],",
        "    }.items()",
        "    if column and column not in _rmc_sample.columns",
        "]",
        "if _rmc_missing_cols:",
        "    raise ValueError('sample column check failed: ' + ', '.join(_rmc_missing_cols))",
        "_rmc_dictionary = _rmc_load_dictionary(_rmc_payload['dictionary_path'])",
        "_rmc_missing_dict_cols = [",
        "    col for col in (",
        "        _rmc_payload['data_dict_feature_col'],",
        "        _rmc_payload['data_dict_category_col'],",
        "    )",
        "    if col not in _rmc_dictionary.columns",
        "]",
        "if _rmc_missing_dict_cols:",
        "    raise ValueError('data dictionary missing columns: ' + ', '.join(sorted(_rmc_missing_dict_cols)))",
        "_rmc_config = _RmcValidationConfig(",
        "    target_col=_rmc_payload['target_col'],",
        f"    score_col={RMC_PMML_SCORE_COL!r},",
        "    split_col=_rmc_payload['split_col'],",
        "    time_col=_rmc_payload['time_col'],",
        "    feature_columns=[],",
        "    bin_count=int(_rmc_payload['bin_count']),",
        "    random_sample_size=int(_rmc_payload['random_sample_size']),",
        "    random_seed=int(_rmc_payload['random_seed']),",
        "    score_decimal_places=int(_rmc_payload['score_decimal_places']),",
        "    data_dict_feature_col=_rmc_payload['data_dict_feature_col'],",
        "    data_dict_category_col=_rmc_payload['data_dict_category_col'],",
        ")",
        "_rmc_code_scores = _rmc_load_code_model_scores(_RmcPath(_rmc_payload['code_scores_path']))",
        "_rmc_reproducibility = _rmc_reproducibility_from_json(_rmc_payload['reproducibility_json_path'])",
    ]
    score_lines = [
        "# Injected by marvis v3 (metrics score).",
        "def _rmc_score_with_notebook(dataframe):",
        "    if 'RMC_SCORE_FN' not in globals() or not callable(RMC_SCORE_FN):",
        "        raise NameError('RMC_SCORE_FN must be defined before metrics')",
        "    values = RMC_SCORE_FN(dataframe.copy())",
        "    if isinstance(values, _rmc_pd.DataFrame):",
        "        if len(values.columns) != 1:",
        "            raise ValueError('RMC_SCORE_FN must return one score column')",
        "        values = values.iloc[:, 0]",
        "    if isinstance(values, _rmc_pd.Series):",
        "        scores = values.astype(float).tolist()",
        "    else:",
        "        scores = _rmc_pd.Series(values).astype(float).tolist()",
        "    if len(scores) != len(dataframe):",
        "        raise ValueError(f'RMC_SCORE_FN returned {len(scores)} scores for {len(dataframe)} rows')",
        "    return _rmc_finite_score_series(scores, index=dataframe.index, label='RMC_SCORE_FN').tolist()",
        "_rmc_model_scores = _rmc_score_with_notebook(_rmc_sample)",
        "_rmc_sample_scored = _rmc_sample.copy()",
        "_rmc_sample_scored[_rmc_config.score_col] = _rmc_pd.Series(_rmc_model_scores, index=_rmc_sample_scored.index, dtype=float)",
        "class _RmcNotebookScorer:",
        "    def score(self, dataframe):",
        "        if len(dataframe) == len(_rmc_sample) and dataframe.index.equals(_rmc_sample.index):",
        "            return list(_rmc_model_scores)",
        "        return _rmc_score_with_notebook(dataframe)",
    ]
    basic_lines = [
        "# Injected by marvis v3 (metrics basic info).",
        "_rmc_basic_info = _rmc_run_basic_info(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        "    model_meta_path=_RmcPath(_rmc_payload['model_meta_path']),",
        ")",
    ]
    ks_lines = [
        "# Injected by marvis v3 (metrics KS).",
        "_rmc_effectiveness_context = _rmc_prepare_effectiveness_context(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        ")",
        "_rmc_effectiveness_overall = _rmc_compute_overall_ks(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        ")",
        "_rmc_monthly_ks = _rmc_compute_monthly_ks(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        ")",
        "_rmc_roc_ks_curves = _rmc_compute_roc_ks_curves(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        ")",
    ]
    psi_lines = [
        "# Injected by marvis v3 (metrics PSI).",
        "_rmc_effectiveness_overall = _rmc_compute_overall_psi(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        "    context=_rmc_effectiveness_context,",
        "    overall=_rmc_effectiveness_overall,",
        ")",
        "_rmc_monthly_psi = _rmc_compute_monthly_psi(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        "    context=_rmc_effectiveness_context,",
        ")",
        "_rmc_psi_stability_table = _rmc_compute_psi_stability_table(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        ")",
    ]
    binning_lines = [
        "# Injected by marvis v3 (metrics binning).",
        "_rmc_bin_tables = _rmc_compute_bin_tables(",
        "    sample=_rmc_sample_scored,",
        "    config=_rmc_config,",
        "    context=_rmc_effectiveness_context,",
        ")",
        "_rmc_effectiveness = _rmc_build_effectiveness_result(",
        "    overall=_rmc_effectiveness_overall,",
        "    bin_tables=_rmc_bin_tables,",
        "    monthly_ks=_rmc_monthly_ks,",
        "    monthly_psi=_rmc_monthly_psi,",
        "    psi_stability_table=_rmc_psi_stability_table,",
        "    roc_ks_curves=_rmc_roc_ks_curves,",
        ")",
    ]
    stress_lines = [
        "# Injected by marvis v3 (metrics stress).",
        "_rmc_feature_categories = _rmc_load_feature_categories(",
        "    _rmc_dictionary,",
        "    feature_col=_rmc_config.data_dict_feature_col,",
        "    category_col=_rmc_config.data_dict_category_col,",
        ")",
        "_rmc_feature_categories = _rmc_filter_feature_categories(",
        "    _rmc_feature_categories,",
        "    model_features=_rmc_model_features(_rmc_config, _rmc_basic_info.feature_importance),",
        ")",
        "if not _rmc_config.split_col:",
        "    raise ValueError(",
        "        'split_col is not configured; set RMC_SPLIT_COL in the notebook or '",
        "        'configure the task split column before running metrics'",
        "    )",
        "_rmc_oot_sample = _rmc_sample[_rmc_sample[_rmc_config.split_col] == _rmc_config.split_values['oot']]",
        "_rmc_stress_test = _rmc_run_stress_test(",
        "    oot_sample=_rmc_oot_sample,",
        "    config=_rmc_config,",
        "    feature_categories=_rmc_feature_categories,",
        "    input_scorer=_RmcNotebookScorer(),",
        "    cancellation_check=_rmc_raise_if_metrics_cancelled,",
        ")",
    ]
    output_lines = [
        "# Injected by marvis v3 (metrics output).",
        "_rmc_results = _RmcValidationResults(",
        "    model_name=_rmc_payload['model_name'],",
        "    model_version=_rmc_payload['model_version'],",
        "    algorithm=_rmc_payload['algorithm'],",
        "    target_type='binary',",
        "    reproducibility=_rmc_reproducibility,",
        "    basic_info=_rmc_basic_info,",
        "    effectiveness=_rmc_effectiveness,",
        "    stress_test=_rmc_stress_test,",
        ")",
        "_rmc_results_json_path = _RmcPath(_rmc_payload['results_json_path'])",
        "_rmc_results_json_path.parent.mkdir(parents=True, exist_ok=True)",
        "_rmc_results_json_path.write_text(",
        "    _rmc_json.dumps(_rmc_asdict(_rmc_results), ensure_ascii=False, indent=2),",
        "    encoding='utf-8',",
        ")",
        "_rmc_write_validation_excel(_rmc_results, _RmcPath(_rmc_payload['excel_path']))",
    ]
    return [
        ("metrics-prepare", "\n".join(prepare_lines)),
        ("metrics-score", "\n".join(score_lines)),
        ("metrics-basic", "\n".join(basic_lines)),
        ("metrics-ks", "\n".join(ks_lines)),
        ("metrics-psi", "\n".join(psi_lines)),
        ("metrics-binning", "\n".join(binning_lines)),
        ("metrics-stress", "\n".join(stress_lines)),
        ("metrics-output", "\n".join(output_lines)),
    ]


def _json_literal(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def run_staged_pipeline(*, task_id: str, settings: PipelineSettings) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if task.status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise PipelineError(
            f"task already {task.status.value}; reset task before rerunning pipeline"
        )

    run_notebook_stage(task_id=task_id, settings=settings)
    task = repo.get_task(task_id)
    if task.status is not TaskStatus.EXECUTED:
        return

    run_metrics_stage(task_id=task_id, settings=settings)
    task = repo.get_task(task_id)
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        return

    run_report_stage(task_id=task_id, settings=settings)


def run_pipeline(*, task_id: str, settings: PipelineSettings) -> None:
    if settings.notebook_isolated_execution:
        run_staged_pipeline(task_id=task_id, settings=settings)
        return
    _require_legacy_live_notebook_execution(settings)

    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if task.status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise PipelineError(
            f"task already {task.status.value}; reset task before rerunning pipeline"
        )
    task_dir = settings.workspace / "tasks" / task_id
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    execution_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    live_session: NotebookExecutionSession | None = None
    failure_prefix = NOTEBOOK_STAGE_FAILURE_PREFIX

    try:
        _clear_generated_artifacts(task_dir, stage="scan")
        execution_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        artifacts = _scan_step(repo, task)
        notebook_path = _required_path(
            task, artifacts, FileRole.NOTEBOOK, "notebook", "notebook_path"
        )
        sample_path = _required_path(
            task, artifacts, FileRole.SAMPLE, "sample", "sample_path"
        )
        input_pmml_path = _required_path(
            task, artifacts, FileRole.MODEL_PMML, "input PMML", "pmml_path"
        )
        dictionary_path = _required_path(
            task,
            artifacts,
            FileRole.DATA_DICTIONARY,
            "data dictionary",
            "dictionary_path",
        )

        kernel_name = _execution_kernel_name(settings)

        contract_meta_path = execution_dir / "runtime_contract.json"
        code_scores_path = execution_dir / "code_model_scores.csv"
        feature_importance_path = execution_dir / "feature_importance.csv"
        model_params_path = execution_dir / "model_params.json"
        notebook_steps_path = execution_dir / "notebook_steps.json"
        live_session = _notebook_step_v3(
            repo=repo,
            task=task,
            source_notebook=notebook_path,
            sample_path=sample_path,
            execution_dir=execution_dir,
            contract_meta_path=contract_meta_path,
            code_scores_path=code_scores_path,
            feature_importance_path=feature_importance_path,
            model_params_path=model_params_path,
            notebook_steps_path=notebook_steps_path,
            kernel_name=kernel_name,
            notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
            keep_alive=True,
        )
        if live_session is None:
            return
        contract = load_runtime_contract(contract_meta_path)
        task = _sync_task_algorithm(repo, task, contract.algorithm)
        _write_reproducibility_result_in_session(
            session=live_session,
            task=task,
            settings=settings,
            input_pmml_path=input_pmml_path,
            contract_meta_path=contract_meta_path,
            output_path=outputs_dir / REPRODUCIBILITY_RESULT_JSON,
        )
        model_meta_path = execution_dir / "model_meta.json"
        _write_model_meta_from_contract(contract, model_meta_path)

        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            message="computing metrics",
            expected=TaskStatus.EXECUTED,
        )
        failure_prefix = METRICS_STAGE_FAILURE_PREFIX
        _write_metrics_results_in_session(
            session=live_session,
            task=task,
            settings=settings,
            dictionary_path=dictionary_path,
            input_pmml_path=input_pmml_path,
            contract=contract,
            model_meta_path=model_meta_path,
            reproducibility_json_path=outputs_dir / REPRODUCIBILITY_RESULT_JSON,
            outputs_dir=outputs_dir,
        )
        live_session.close()
        live_session = None

        repo.update_status(
            task_id,
            TaskStatus.WRITING_ARTIFACTS,
            message="writing artifacts",
            expected=TaskStatus.COMPUTING_METRICS,
        )
        failure_prefix = REPORT_STAGE_FAILURE_PREFIX
        results = _load_validation_results(outputs_dir)
        report_values, _ = repo.get_report_values(task_id)
        word_result = write_validation_word(
            results,
            template_path=settings.report_template_path,
            output_path=outputs_dir / "validation_report.docx",
            image_output_dir=images_dir,
            report_values=report_values,
        )

        if word_result.unresolved_placeholders:
            repo.update_status(
                task_id,
                TaskStatus.REVIEW_REQUIRED,
                message="报告已生成，需人工复核",
                expected=TaskStatus.WRITING_ARTIFACTS,
            )
            return

        terminal_status = (
            TaskStatus.REVIEW_REQUIRED
            if results.reproducibility.summary.status is ConsistencyStatus.FAIL
            else TaskStatus.SUCCEEDED
        )
        repo.update_status(
            task_id,
            terminal_status,
            message=(
                "验证已完成，需人工复核报告"
                if terminal_status is TaskStatus.REVIEW_REQUIRED
                else "pipeline succeeded"
            ),
            expected=TaskStatus.WRITING_ARTIFACTS,
        )
    except PipelineError as exc:
        _mark_failed(repo, task_id, _stage_failure_message(failure_prefix, str(exc)))
        raise
    except Exception as exc:
        _mark_failed(
            repo,
            task_id,
            _stage_failure_message(failure_prefix, f"{exc.__class__.__name__}: {exc}"),
        )
        raise
    finally:
        if live_session is not None:
            live_session.close()


def _scan_step(repo: TaskRepository, task: TaskRecord) -> list[FileArtifact]:
    artifacts = _scan_artifacts(task)
    repo.update_status(
        task.id,
        TaskStatus.SCANNED,
        message="source scanned",
        expected={TaskStatus.CREATED, TaskStatus.SCANNED, TaskStatus.FAILED},
    )
    return artifacts


def _scan_artifacts(task: TaskRecord) -> list[FileArtifact]:
    try:
        return scan_source_dir(Path(task.source_dir))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise PipelineError(f"source dir invalid: {exc}") from exc
    except ValueError as exc:
        # scan-limit breaches (max_files / max_depth) raise ValueError. Tag them
        # with the scan-stage prefix so the notebook/metrics/report except handlers
        # keep the failure attributed to the scan stage instead of mislabeling it
        # as a notebook failure (see _stage_failure_message + _is_scan_failure).
        raise PipelineError(f"{SCAN_STAGE_FAILURE_PREFIX}{exc}") from exc


def _required_path(
    task: TaskRecord,
    artifacts: list[FileArtifact],
    role: FileRole,
    label: str,
    task_field: str,
) -> Path:
    source_dir = Path(task.source_dir).resolve()
    explicit_value = getattr(task, task_field, None)
    if explicit_value:
        return _resolve_explicit_input_path(
            explicit_value=explicit_value,
            source_dir=source_dir,
            label=label,
            task_field=task_field,
        )

    candidates = [artifact.path for artifact in artifacts if artifact.role == role]
    if not candidates:
        raise PipelineError(f"missing required input: {label}")
    if len(candidates) == 1:
        return candidates[0]
    candidate_text = ", ".join(str(path) for path in candidates)
    raise PipelineError(
        f"{label} role ambiguous: {candidate_text}; configure {task_field}"
    )


def _resolve_explicit_input_path(
    *,
    explicit_value: str | Path,
    source_dir: Path,
    label: str,
    task_field: str,
) -> Path:
    raw_path = Path(explicit_value)
    path = raw_path if raw_path.is_absolute() else source_dir / raw_path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PipelineError(
            f"configured {task_field} for {label} does not exist: {path}"
        ) from exc

    try:
        resolved.relative_to(source_dir)
    except ValueError as exc:
        raise PipelineError(
            f"configured {task_field} for {label} must be inside source_dir: {resolved}"
        ) from exc
    return resolved


def _feature_columns(settings: PipelineSettings, task: TaskRecord) -> list[str]:
    if settings.feature_columns:
        return settings.feature_columns
    task_feature_columns = getattr(task, "feature_columns", None)
    return list(task_feature_columns or [])


def _load_sample(
    sample_path: Path,
    *,
    fallback_python: Path | str | None = None,
) -> pd.DataFrame:
    suffix = sample_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(sample_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(sample_path)
    if suffix == ".feather":
        return _load_arrow_sample(
            sample_path,
            suffix=suffix,
            fallback_python=fallback_python,
            read_with_platform=pd.read_feather,
        )
    if suffix == ".parquet":
        return _load_arrow_sample(
            sample_path,
            suffix=suffix,
            fallback_python=fallback_python,
            read_with_platform=pd.read_parquet,
        )
    raise PipelineError(f"unsupported sample format: {suffix}")


def _load_arrow_sample(
    sample_path: Path,
    *,
    suffix: str,
    fallback_python: Path | str | None,
    read_with_platform,
) -> pd.DataFrame:
    try:
        return read_with_platform(sample_path)
    except ImportError as exc:
        if not fallback_python or _same_python(fallback_python, sys.executable):
            raise
        try:
            return _load_arrow_sample_with_python(
                sample_path,
                suffix=suffix,
                python_executable=Path(fallback_python),
            )
        except Exception as fallback_exc:
            raise PipelineError(
                "failed to load sample with both platform Python and selected "
                f"execution Python; platform error: {exc}; fallback error: "
                f"{fallback_exc.__class__.__name__}: {fallback_exc}"
            ) from fallback_exc


def _load_arrow_sample_with_python(
    sample_path: Path,
    *,
    suffix: str,
    python_executable: Path,
) -> pd.DataFrame:
    code = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "import pandas as pd",
            "input_path = Path(sys.argv[1])",
            "output_path = Path(sys.argv[2])",
            "suffix = sys.argv[3]",
            "if suffix == '.feather':",
            "    df = pd.read_feather(input_path)",
            "elif suffix == '.parquet':",
            "    df = pd.read_parquet(input_path)",
            "else:",
            "    raise ValueError(f'unsupported arrow suffix: {suffix}')",
            "df.to_pickle(output_path)",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="marvis-sample-") as tmp_dir:
        pickle_path = Path(tmp_dir) / "sample.pkl"
        completed = subprocess.run(
            [str(python_executable), "-c", code, str(sample_path), str(pickle_path), suffix],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )
        if completed.returncode != 0:
            raise PipelineError(
                (completed.stderr or completed.stdout or "selected Python failed").strip()
            )
        return pd.read_pickle(pickle_path)


def _same_python(left: Path | str, right: Path | str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _execution_kernel_name(settings: PipelineSettings) -> str:
    environment = load_execution_environment(settings.workspace)
    validation = validate_execution_environment(environment)
    if not validation.ok:
        raise PipelineError(f"execution environment invalid: {validation.message}")
    return validation.kernel_name or settings.notebook_kernel_name


def _write_model_meta_from_contract(contract: RuntimeContract, output_path: Path) -> Path:
    feature_importance: list[dict] = []
    if contract.feature_importance_path and contract.feature_importance_path.exists():
        feature_importance = pd.read_csv(contract.feature_importance_path).to_dict(
            orient="records"
        )

    hyperparameters: dict = {}
    if contract.model_params_path and contract.model_params_path.exists():
        hyperparameters = json.loads(contract.model_params_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "algorithm": _algorithm(contract.algorithm),
                "feature_importance": feature_importance,
                "hyperparameters": hyperparameters,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _clear_generated_artifacts(task_dir: Path, *, stage: str) -> None:
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    if stage == "scan":
        _remove_dir_if_exists(execution_dir)
        _remove_dir_if_exists(outputs_dir)
        _remove_dir_if_exists(images_dir)
        return

    if stage == "notebook":
        for name in (
            "prepared.ipynb",
            "executed.ipynb",
            "notebook.log",
            "runtime_contract.json",
            "code_model_scores.csv",
            "feature_importance.csv",
            "model_params.json",
            "model_meta.json",
            "notebook_steps.json",
            METRICS_CANCEL_MARKER,
        ):
            _unlink_if_exists(execution_dir / name)
        for name in (
            REPRODUCIBILITY_RESULT_JSON,
            "validation_results.json",
            VALIDATION_RESULTS_PICKLE,
            "validation.xlsx",
            "validation_report.docx",
            "reproducibility_notebook.log",
            "metrics_notebook.log",
        ):
            _unlink_if_exists(outputs_dir / name)
        _remove_dir_if_exists(images_dir)
        return

    if stage == "metrics":
        _unlink_if_exists(_metrics_cancel_marker_path(task_dir))
        for name in (
            "validation_results.json",
            VALIDATION_RESULTS_PICKLE,
            "validation.xlsx",
            "validation_report.docx",
            "metrics_notebook.log",
        ):
            _unlink_if_exists(outputs_dir / name)
        _remove_dir_if_exists(images_dir)
        return

    if stage == "report":
        _unlink_if_exists(outputs_dir / "validation_report.docx")
        _remove_dir_if_exists(images_dir)
        return

    raise ValueError(f"unknown artifact cleanup stage: {stage}")


def _unlink_if_exists(path: Path) -> None:
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except FileNotFoundError:
        return


def _remove_dir_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _load_validation_results(outputs_dir: Path):
    json_path = outputs_dir / "validation_results.json"
    if not json_path.exists():
        raise PipelineError("metrics output missing: validation_results.json")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError("metrics output invalid: validation_results.json") from exc
    if not isinstance(payload, dict):
        raise PipelineError("metrics output invalid: validation_results.json")
    return validation_results_from_dict(payload)


def _capture_agent_memory_for_metrics_success(
    *,
    repo: TaskRepository,
    task_id: str,
    outputs_dir: Path,
) -> None:
    # Gate on the user-facing "自动沉淀任务经验" (auto_distill) memory policy:
    # when off, no automatic capture happens on this pipeline surface either.
    if not load_memory_policy(repo.db_path.parent).auto_distill:
        return
    try:
        task = repo.get_task(task_id)
        payload = _read_validation_results_payload(outputs_dir)
        store = AgentMemoryStore(repo.db_path)
        for candidate in (
            extract_model_experience(
                _memory_model_experience_payload(task=task, results=payload)
            ),
            extract_field_convention(_memory_field_convention_payload(task)),
        ):
            if candidate is not None:
                store.create(candidate, task_id=task_id)
    except Exception:
        return


def _capture_agent_memory_for_failure(
    *,
    repo: TaskRepository,
    task_id: str,
    failure_kind: str,
    message: str,
) -> None:
    # Gate on auto_distill (see _capture_agent_memory_for_metrics_success).
    if not load_memory_policy(repo.db_path.parent).auto_distill:
        return
    try:
        store = AgentMemoryStore(repo.db_path)
        payload = {
            "task_id": task_id,
            "status": "failed",
            "summary": message,
            "failures": [{"kind": failure_kind, "message": message}],
        }
        for candidate in [
            *extract_validation_pitfall(payload),
            extract_task_experience(payload),
        ]:
            if candidate is not None:
                store.create(candidate, task_id=task_id)
    except Exception:
        return


def _read_validation_results_payload(outputs_dir: Path) -> dict:
    path = outputs_dir / "validation_results.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _memory_model_experience_payload(
    *,
    task: TaskRecord,
    results: dict,
) -> dict:
    metrics_row = _memory_preferred_overall_row(results)
    return {
        "task_id": task.id,
        "source_task_id": task.id,
        "model_name": results.get("model_name") or task.model_name,
        "model_version": results.get("model_version") or task.model_version,
        "scope": results.get("scope") or f"{task.model_name}验证任务",
        "channel": results.get("channel") or "未标注",
        "month": results.get("month") or _memory_latest_month(results) or "未标注",
        "metrics": {
            "ks": metrics_row.get("ks"),
            "auc": metrics_row.get("auc"),
            "psi": metrics_row.get("psi_vs_train") or metrics_row.get("psi"),
        },
        "important_feature_sources": _memory_important_feature_sources(results),
    }


def _memory_field_convention_payload(task: TaskRecord) -> dict:
    return {
        "task_id": task.id,
        "target_col": task.target_col,
        "score_col": task.score_col,
        "split_col": task.split_col,
        "time_col": task.time_col,
    }


def _memory_preferred_overall_row(results: dict) -> dict:
    overall = ((results.get("effectiveness") or {}).get("overall") or [])
    rows = [row for row in overall if isinstance(row, dict)]
    for split_name in ("oot", "test", "train"):
        for row in rows:
            if str(row.get("split") or "").strip().lower() == split_name:
                return row
    return rows[0] if rows else {}


def _memory_latest_month(results: dict) -> str:
    monthly_sources = (
        (results.get("effectiveness") or {}).get("monthly_ks") or [],
        (results.get("basic_info") or {}).get("monthly_distribution") or [],
    )
    months: list[str] = []
    for rows in monthly_sources:
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("month") not in (None, ""):
                months.append(str(row["month"]))
    return sorted(months)[-1] if months else ""


def _memory_important_feature_sources(results: dict) -> list[str]:
    feature_importance = (results.get("basic_info") or {}).get("feature_importance") or []
    sources: list[str] = []
    for row in feature_importance:
        if not isinstance(row, dict):
            continue
        category = str(row.get("category") or row.get("类别") or "").strip()
        if category:
            sources.append(category)
    return list(dict.fromkeys(sources)) or ["未标注"]


def _memory_failure_kind(message: str, *, default: str) -> str:
    text = str(message or "").lower()
    if (
        SCAN_STAGE_FAILURE_PREFIX.lower() in text
        or "too many files" in text
        or "too deep" in text
        or "source dir invalid" in text
    ):
        return "scan"
    if "pmml" in text:
        return "pmml"
    if (
        "field" in text
        or "column" in text
        or "字段" in text
        or "split_col" in text
        or "target_col" in text
        or "score_col" in text
        or "time_col" in text
    ):
        return "field"
    if "report" in text or "报告" in text or "word" in text:
        return "report"
    if "notebook" in text or "kernel" in text or "rmc_" in text:
        return "notebook"
    return default


def _notebook_step_v3(
    *,
    repo: TaskRepository,
    task: TaskRecord,
    source_notebook: Path,
    sample_path: Path,
    execution_dir: Path,
    contract_meta_path: Path,
    code_scores_path: Path,
    feature_importance_path: Path,
    model_params_path: Path,
    notebook_steps_path: Path,
    kernel_name: str,
    notebook_memory_limit_mb: int | None = None,
    stage_claimed: bool = False,
    cancellation_token: NotebookCancellationToken | None = None,
    keep_alive: bool = False,
    isolated: bool = False,
    mark_executed: bool = True,
    extra_code_cells: list[tuple[str, str]] | None = None,
    cancel_message: str = "notebook cancelled",
    cancel_resume_status: TaskStatus = TaskStatus.SCANNED,
) -> NotebookExecutionSession | None:
    prepared = execution_dir / "prepared.ipynb"
    executed = execution_dir / "executed.ipynb"
    log = execution_dir / "notebook.log"
    if not stage_claimed:
        repo.update_status(
            task.id,
            TaskStatus.RUNNING,
            message="notebook running",
            expected={
                TaskStatus.SCANNED,
                TaskStatus.RUNNING,
                TaskStatus.EXECUTED,
                TaskStatus.FAILED,
            },
        )
    prepare_execution_notebook_v3(
        source_notebook=source_notebook,
        output_notebook=prepared,
        sample_path=sample_path,
        contract_meta_path=contract_meta_path,
        code_scores_path=code_scores_path,
        feature_importance_path=feature_importance_path,
        model_params_path=model_params_path,
        extra_code_cells=extra_code_cells,
    )
    live_session: NotebookExecutionSession | None = None
    if keep_alive:
        live_session = NotebookExecutionSession(
            notebook_path=prepared,
            executed_path=executed,
            log_path=log,
            kernel_name=kernel_name,
            progress_path=notebook_steps_path,
            execution_cwd=source_notebook.parent,
            cancellation_token=cancellation_token,
            memory_limit_mb=notebook_memory_limit_mb,
            allow_appended_execution=True,
            appended_execution_policy=V1_VALIDATION_APPENDED_EXECUTION_POLICY,
        )
        result = live_session.execute_notebook(keep_alive=True)
    else:
        result = run_notebook(
            notebook_path=prepared,
            executed_path=executed,
            log_path=log,
            kernel_name=kernel_name,
            progress_path=notebook_steps_path,
            execution_cwd=source_notebook.parent,
            cancellation_token=cancellation_token,
            memory_limit_mb=notebook_memory_limit_mb,
            isolated=isolated,
        )
    if result.step_events is not None:
        notebook_steps_path.parent.mkdir(parents=True, exist_ok=True)
        notebook_steps_path.write_text(
            json.dumps(result.step_events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if result.cancelled:
        if live_session is not None:
            live_session.close()
        _mark_cancelled(repo, task.id, cancel_resume_status, cancel_message)
        return
    if not result.succeeded:
        if live_session is not None:
            live_session.close()
        raise PipelineError(
            f"notebook failed at cell {result.failed_cell_index}: "
            f"{result.error_name}: {result.error_value}"
        )
    if not contract_meta_path.exists():
        raise PipelineError("notebook runtime contract did not produce metadata")
    if not code_scores_path.exists():
        raise PipelineError("notebook runtime contract did not produce code-model scores")
    if mark_executed:
        repo.update_status(
            task.id,
            TaskStatus.EXECUTED,
            message="notebook executed",
            expected=TaskStatus.RUNNING,
        )
    return live_session


def _mark_failed(repo: TaskRepository, task_id: str, message: str) -> None:
    try:
        if repo.get_task(task_id).status == TaskStatus.FAILED:
            return
        repo.update_status(task_id, TaskStatus.FAILED, message=message, expected=None)
    except Exception:
        pass


def _mark_cancelled(
    repo: TaskRepository,
    task_id: str,
    resume_status: TaskStatus,
    message: str,
) -> None:
    try:
        current = repo.get_task(task_id).status
        if current == resume_status:
            repo.update_status_message(
                task_id,
                message,
                reason_code=TASK_STATUS_REASON_USER_CANCELLED,
            )
            return
        repo.update_status(
            task_id,
            resume_status,
            message=message,
            expected=None,
            reason_code=TASK_STATUS_REASON_USER_CANCELLED,
        )
    except Exception:
        pass


def _stage_failure_message(prefix: str, message: str) -> str:
    # A scan-stage failure can surface while a later stage's prefix is active
    # (the notebook/metrics/report try-blocks re-scan artifacts). Preserve an
    # already scan-prefixed message so _is_scan_failure keeps the right attribution.
    if message.startswith(prefix) or message.startswith(SCAN_STAGE_FAILURE_PREFIX):
        return message
    return f"{prefix}{message}"


def _sync_task_algorithm(
    repo: TaskRepository,
    task: TaskRecord,
    algorithm: str,
) -> TaskRecord:
    normalized = _algorithm(algorithm)
    if task.algorithm == normalized:
        return task
    return repo.update_algorithm(task.id, normalized)


def _algorithm(value: str) -> str:
    return normalize_algorithm(value)
