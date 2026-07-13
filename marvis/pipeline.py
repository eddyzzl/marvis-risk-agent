"""Validation pipeline orchestration: notebook -> metrics -> report stages.

This module is the pipeline's public facade and orchestration layer. Per
ARCH-6, the god-file was split into focused submodules:
  - marvis.pipeline_errors: PipelineError / PipelineCancelled (shared base
    exception types, to avoid a circular import between this module and its
    submodules).
  - marvis.pipeline_cellgen: notebook injected-cell source builders
    (reproducibility + metrics cell Python source strings).
  - marvis.pipeline_io: path resolution, sample loading, and generated-
    artifact hygiene helpers.
  - marvis.pipeline_memory: agent-memory capture on success/failure, gated
    on the auto_distill policy (INV-4).

Everything that stays here either IS a stage entry point (run_notebook_stage,
run_metrics_stage, run_report_stage, run_staged_pipeline, run_pipeline) or is
called, unqualified, from one of those entry points or from another function
that stays here -- several of these names (_notebook_step_v3, _scan_step,
_write_reproducibility_result_in_session, _write_metrics_results_in_session,
_load_validation_results, load_runtime_contract, write_validation_word,
scan_source_dir) are monkeypatched by test_pipeline_v2.py via
`monkeypatch.setattr("marvis.pipeline.<name>", ...)`, which only rebinds the
name in THIS module's namespace -- so the functions that reference them
unqualified must stay physically defined here for the patches to take
effect. All submodule names are re-exported below for backward compatibility
with the existing `from marvis.pipeline import ...` surface used throughout
the codebase and test suite.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import json
import logging
import math
import shutil
import time
from pathlib import Path

import pandas as pd  # noqa: F401 - re-exported for marvis.pipeline.pd.* monkeypatch compatibility (test_pipeline_v2.py)

from marvis.agent_memory.extractors import (
    extract_field_convention,
    extract_model_experience,
    extract_task_experience,
    extract_validation_pitfall,
)
from marvis.agent_memory.store import AgentMemoryStore
from marvis.artifacts import ArtifactUnitOfWork
from marvis.db import TaskRepository
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    FileArtifact,
    FileRole,
    TaskRecord,
    TaskStatus,
)
from marvis.files import scan_source_dir, write_json_atomic
from marvis.job_cancellation import (
    JobCancelled,
    register_job_cancellation,
    unregister_job_cancellation,
)
from marvis.memory_policy import load_memory_policy
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
from marvis.output.word import write_validation_word
from marvis.pipeline_cellgen import (
    RMC_PMML_SCORE_COL as RMC_PMML_SCORE_COL,
    _build_deferred_contract_resolution_lines as _build_deferred_contract_resolution_lines,  # noqa: F401
    _build_metrics_cell_source as _build_metrics_cell_source,  # noqa: F401
    _build_metrics_cell_sources,
    _build_reproducibility_cell_sources,
    _build_stress_scenario_score_cell_sources,
    _json_literal as _json_literal,  # noqa: F401
    _notebook_package_prelude as _notebook_package_prelude,  # noqa: F401
    _package_root_for_notebook,
)
from marvis.pipeline_errors import PipelineCancelled, PipelineError
from marvis.pipeline_io import (
    SCAN_STAGE_FAILURE_PREFIX as SCAN_STAGE_FAILURE_PREFIX,
    STRESS_SCENARIO_SCORES_JSON,
    VALIDATION_RESULTS_PICKLE as VALIDATION_RESULTS_PICKLE,
    _algorithm as _algorithm,  # noqa: F401
    _clear_generated_artifacts,
    _execution_kernel_name,
    _feature_columns as _feature_columns,  # noqa: F401
    _load_arrow_sample as _load_arrow_sample,  # noqa: F401
    _load_arrow_sample_with_python as _load_arrow_sample_with_python,  # noqa: F401
    _load_sample as _load_sample,  # noqa: F401
    _load_validation_results,
    _metrics_cancel_marker_path,
    _metrics_work_dir_prepared,
    _notebook_execution_artifacts_complete,
    _remove_dir_if_exists,
    _require_metrics_outputs,
    _required_path as _required_path,  # noqa: F401
    _resolve_explicit_input_path as _resolve_explicit_input_path,  # noqa: F401
    _same_python as _same_python,  # noqa: F401
    _stage_failure_message,
    _sync_task_algorithm,
    _truthy_env,
    _unlink_if_exists,
    _write_model_meta_from_contract,
)
from marvis.pipeline_memory import (
    _memory_failure_kind,
    _memory_field_convention_payload,
    _memory_important_feature_sources as _memory_important_feature_sources,  # noqa: F401
    _memory_latest_month as _memory_latest_month,  # noqa: F401
    _memory_model_experience_payload,
    _memory_preferred_overall_row as _memory_preferred_overall_row,  # noqa: F401
    _read_validation_results_payload,
)
from marvis.state_machine import IllegalTransition
from marvis.repositories.validation_contracts import (
    ValidationContractRepository,
    require_confirmed_validation_input_contract,
)
from marvis.validation.pmml_score_artifacts import (
    build_pmml_scoring_identity,
    load_or_run_pmml_scoring,
    raise_if_cancelled,
    run_pmml_scoring,
    sha256_file_cancellable,
    validate_pmml_score_artifact,
)
from marvis.validation.pmml_scoring import TASK_PMML_SCORERS
from marvis.validation.pmml_stress import run_pmml_stress
from marvis.validation.stress_test import require_complete_stress_result
from marvis.validation.platform_metrics import (
    compute_platform_validation_results,
    validation_config_from_input_contract,
    write_platform_validation_metrics,
    write_reproducibility_result,
)
from marvis.validation.results import (
    ConsistencyStatus,
    pmml_scoring_result_from_dict,
    pmml_scoring_result_to_dict,
    validation_results_to_dict,
    validation_results_from_dict as validation_results_from_dict,  # noqa: F401
)
from marvis.validation_materials import resolve_selected_validation_materials

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSettings:
    workspace: Path
    db_path: Path
    report_template_path: Path
    feature_columns: list[str] = field(default_factory=list)
    notebook_kernel_name: str = "python3"
    # None = no soft RSS cap on the notebook kernel (the new default; real
    # datasets are large). The execution-environment setting overrides this.
    notebook_memory_limit_mb: int | None = None
    notebook_isolated_execution: bool = True
    allow_legacy_live_notebook_execution: bool = False
    bin_count: int = 10
    random_sample_size: int = 1000
    random_seed: int = 42
    data_dict_feature_col: str = "特征名"
    data_dict_category_col: str = "类别"
    pmml_scoring_chunk_size: int = 10_000


REPRODUCIBILITY_RESULT_JSON = "reproducibility_result.json"
METRICS_CANCEL_MARKER = "metrics_cancel.requested"
METRICS_OUTPUT_FILENAMES = (
    "validation_results.json",
    VALIDATION_RESULTS_PICKLE,
    "validation.xlsx",
    "metrics_notebook.log",
)
NOTEBOOK_STAGE_FAILURE_PREFIX = "模型可复现性验证失败："
METRICS_STAGE_FAILURE_PREFIX = "模型效果&稳定性验证失败："
REPORT_STAGE_FAILURE_PREFIX = "报告输出失败："
LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE = (
    "legacy live notebook execution requires notebook_isolated_execution=False "
    "and allow_legacy_live_notebook_execution=True plus "
    "MARVIS_ALLOW_LEGACY_LIVE_NOTEBOOK_EXECUTION=1"
)
LEGACY_LIVE_NOTEBOOK_ENV_VAR = "MARVIS_ALLOW_LEGACY_LIVE_NOTEBOOK_EXECUTION"
V1_VALIDATION_APPENDED_CELL_KINDS = (
    "stress-scores",
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


def _require_legacy_live_notebook_execution(settings: PipelineSettings) -> None:
    if not legacy_live_notebook_execution_allowed(settings):
        raise PipelineError(LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE)


def legacy_live_notebook_execution_allowed(settings: PipelineSettings) -> bool:
    if settings.notebook_isolated_execution or not settings.allow_legacy_live_notebook_execution:
        return False
    return _truthy_env(LEGACY_LIVE_NOTEBOOK_ENV_VAR)


def run_notebook_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
    also_prepare_metrics: bool = False,
    cancellation_job_id: str | None = None,
) -> None:
    """Run the notebook (reproducibility) stage.

    `also_prepare_metrics` is purely additive and defaults to False, which
    preserves this function's existing behavior exactly. When True (used
    only by `run_staged_pipeline`'s default isolated-mode path), the metrics
    injected cells are appended to the SAME isolated subprocess run as the
    reproducibility cells, and their outputs are written straight into
    `outputs/.metrics-stage-work`. This avoids a second full notebook
    execution (PERF-3): `run_metrics_stage`, invoked right after, detects
    those pre-populated outputs and skips its own notebook re-run. A
    metrics-cell failure in this merged run is still attributed with
    METRICS_STAGE_FAILURE_PREFIX (not the notebook prefix) by checking
    whether the notebook's own execution artifacts already completed.
    """
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    execution_dir = task_dir / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    merge_metrics = (
        also_prepare_metrics
        and settings.notebook_isolated_execution
        and not stage_claimed
    )
    logger.info(
        "notebook stage starting task_id=%s merge_metrics=%s stage_claimed=%s",
        task_id, merge_metrics, stage_claimed,
    )
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
        cancellation_token = register_notebook_cancellation(
            task_id,
            job_id=cancellation_job_id,
        )
        live_session: NotebookExecutionSession | None = None
        try:
            close_live_notebook_session(task_id)
            if settings.notebook_isolated_execution:
                outputs_dir = task_dir / "outputs"
                metrics_work_dir = outputs_dir / ".metrics-stage-work"
                extra_code_cells: list[tuple[str, str]] = []
                dictionary_path: Path | None = None
                if merge_metrics:
                    dictionary_path = _required_path(
                        task,
                        artifacts,
                        FileRole.DATA_DICTIONARY,
                        "data dictionary",
                        "dictionary_path",
                    )
                    _remove_dir_if_exists(metrics_work_dir)
                    metrics_work_dir.mkdir(parents=True, exist_ok=True)
                    extra_code_cells = _build_stress_scenario_score_cell_sources(
                        task=task,
                        settings=settings,
                        dictionary_path=dictionary_path,
                        contract_meta_path=execution_dir / "runtime_contract.json",
                        output_path=execution_dir / STRESS_SCENARIO_SCORES_JSON,
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
                    kernel_name=kernel_name,
                    notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
                    stage_claimed=stage_claimed,
                    cancellation_token=cancellation_token,
                    keep_alive=False,
                    isolated=True,
                    mark_executed=False,
                    extra_code_cells=extra_code_cells,
                )
                if repo.get_task(task_id).status != TaskStatus.RUNNING:
                    return
                contract = load_runtime_contract(execution_dir / "runtime_contract.json")
                task = _sync_task_algorithm(repo, task, contract.algorithm)
                output_path = outputs_dir / REPRODUCIBILITY_RESULT_JSON
                write_reproducibility_result(
                    task=task,
                    contract=contract,
                    settings=settings,
                    input_pmml_path=input_pmml_path,
                    output_path=output_path,
                    fallback_sample_path=sample_path,
                )
                if not output_path.exists():
                    raise PipelineError(
                        "notebook reproducibility evidence did not produce output"
                    )
                if merge_metrics:
                    if dictionary_path is None:
                        raise PipelineError("data dictionary is required for metrics")
                    model_meta_path = _write_model_meta_from_contract(
                        contract,
                        execution_dir / "model_meta.json",
                    )
                    write_platform_validation_metrics(
                        task=task,
                        contract=contract,
                        settings=settings,
                        dictionary_path=dictionary_path,
                        model_meta_path=model_meta_path,
                        reproducibility_json_path=output_path,
                        results_json_path=metrics_work_dir / "validation_results.json",
                        excel_path=metrics_work_dir / "validation.xlsx",
                        stress_scores_path=execution_dir / STRESS_SCENARIO_SCORES_JSON,
                        fallback_sample_path=sample_path,
                    )
                    _require_metrics_outputs(metrics_work_dir)
                repo.update_status(
                    task_id,
                    TaskStatus.EXECUTED,
                    message="notebook executed",
                    expected=TaskStatus.RUNNING,
                )
                logger.info(
                    "notebook stage executed task_id=%s isolated=True merge_metrics=%s",
                    task_id, merge_metrics,
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
        logger.info("notebook stage cancelled task_id=%s", task_id)
        _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
        return
    except PipelineError as exc:
        failure_prefix = _notebook_stage_failure_prefix(merge_metrics, execution_dir)
        message = _stage_failure_message(failure_prefix, str(exc))
        logger.error("notebook stage failed task_id=%s error=%s", task_id, exc)
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(
                str(exc),
                default="notebook" if failure_prefix == NOTEBOOK_STAGE_FAILURE_PREFIX else "execution",
            ),
            message=message,
        )
        raise
    except Exception as exc:
        failure_prefix = _notebook_stage_failure_prefix(merge_metrics, execution_dir)
        message = _stage_failure_message(
            failure_prefix,
            f"{exc.__class__.__name__}: {exc}",
        )
        logger.error(
            "notebook stage failed unexpectedly task_id=%s error_type=%s",
            task_id, exc.__class__.__name__, exc_info=True,
        )
        _mark_failed(repo, task_id, message)
        _capture_agent_memory_for_failure(
            repo=repo,
            task_id=task_id,
            failure_kind=_memory_failure_kind(
                str(exc),
                default="notebook" if failure_prefix == NOTEBOOK_STAGE_FAILURE_PREFIX else "execution",
            ),
            message=message,
        )
        raise


def _notebook_stage_failure_prefix(merge_metrics: bool, execution_dir: Path) -> str:
    """When notebook+metrics cells ran in one merged subprocess call
    (PERF-3), a failure after the notebook's own contract/scores/model_meta
    artifacts are already complete on disk must have happened in a metrics
    cell (those run after the tail cell writes those artifacts), so it
    should be attributed as a metrics failure, not a notebook failure -
    otherwise is_metrics_failure() would not recognize it as resumable via
    the metrics-only retry path."""
    if merge_metrics and _notebook_execution_artifacts_complete(execution_dir):
        return METRICS_STAGE_FAILURE_PREFIX
    return NOTEBOOK_STAGE_FAILURE_PREFIX


class _ValidationIdentityInvalid(PipelineError):
    """Selected materials or persisted score evidence no longer match."""


@contextmanager
def _pmml_job_cancellation(
    job_id: str | None,
    cancellation_check: Callable[[], None] | None = None,
) -> Iterator[Callable[[], None] | None]:
    if not job_id:
        yield cancellation_check
        return
    token = register_job_cancellation(job_id)

    def combined_check() -> None:
        if cancellation_check is not None:
            cancellation_check()
        token.raise_if_cancelled()

    try:
        yield combined_check
    finally:
        unregister_job_cancellation(job_id, token)


def _current_validation_material_hashes(
    materials,
    cancellation_check: Callable[[], None] | None,
) -> dict[str, str]:
    return {
        "notebook": sha256_file_cancellable(
            materials.notebook, cancellation_check
        ),
        "sample": sha256_file_cancellable(materials.sample, cancellation_check),
        "pmml": sha256_file_cancellable(materials.pmml, cancellation_check),
        "dictionary": sha256_file_cancellable(
            materials.dictionary, cancellation_check
        ),
    }


def _require_current_validation_materials(
    *,
    contract,
    materials,
    cancellation_check: Callable[[], None] | None,
) -> dict[str, str]:
    current = _current_validation_material_hashes(materials, cancellation_check)
    if current != contract.material_hashes:
        raise _ValidationIdentityInvalid(
            "selected validation materials changed; rescan and reconfirm"
        )
    return current


def _invalidate_v2_validation_identity(
    repo: TaskRepository,
    *,
    task_id: str,
    message: str,
) -> None:
    contracts = ValidationContractRepository(repo.db_path)
    with repo.transaction() as conn:
        current = repo.get_task(task_id).status
        if current is TaskStatus.COMPUTING_METRICS:
            repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.EXECUTED,
                message=message,
                expected=TaskStatus.COMPUTING_METRICS,
            )
            current = TaskStatus.EXECUTED
        if current is not TaskStatus.SCANNED:
            repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.SCANNED,
                message=message,
                expected=current,
            )
        contracts.invalidate_for_material_change_on_connection(conn, task_id)


def _restore_scoring_resume_state(
    repo: TaskRepository,
    *,
    task_id: str,
    message: str,
) -> None:
    current = repo.get_task(task_id).status
    if current in {TaskStatus.RUNNING, TaskStatus.EXECUTED}:
        repo.update_status(
            task_id,
            TaskStatus.SCANNED,
            message=message,
            expected=current,
        )


def _restore_metrics_resume_state(
    repo: TaskRepository,
    *,
    task_id: str,
    message: str,
) -> None:
    if repo.get_task(task_id).status is TaskStatus.COMPUTING_METRICS:
        repo.update_status(
            task_id,
            TaskStatus.EXECUTED,
            message=message,
            expected=TaskStatus.COMPUTING_METRICS,
        )


def _remove_empty_directory_chain(path: Path, *, stop_at: Path) -> None:
    current = Path(path)
    boundary = Path(stop_at)
    while current != boundary.parent:
        try:
            current.rmdir()
        except OSError:
            return
        if current == boundary:
            return
        current = current.parent


def run_pmml_scoring_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if task.validation_workflow_version != 2:
        raise PipelineError("PMML scoring stage requires a v2 validation task")
    if not stage_claimed:
        repo.update_status(
            task_id,
            TaskStatus.RUNNING,
            message="PMML打分测试进行中",
            expected=TaskStatus.SCANNED,
        )
    try:
        with _pmml_job_cancellation(
            cancellation_job_id,
            cancellation_check,
        ) as active_cancellation_check:
            _execute_pmml_scoring_stage(
                task_id=task_id,
                settings=settings,
                cancellation_check=active_cancellation_check,
            )
    except JobCancelled:
        _mark_cancelled(repo, task_id, TaskStatus.SCANNED, "PMML打分测试已取消")
    except _ValidationIdentityInvalid as exc:
        _invalidate_v2_validation_identity(
            repo, task_id=task_id, message=str(exc)
        )
        raise
    except Exception:
        _restore_scoring_resume_state(
            repo, task_id=task_id, message="PMML打分测试失败，可重试"
        )
        raise


def _execute_pmml_scoring_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    cancellation_check: Callable[[], None] | None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    raise_if_cancelled(cancellation_check)
    try:
        materials = resolve_selected_validation_materials(task)
        record = require_confirmed_validation_input_contract(
            ValidationContractRepository(settings.db_path), task_id
        )
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    contract = record.contract
    current_hashes = _require_current_validation_materials(
        contract=contract,
        materials=materials,
        cancellation_check=cancellation_check,
    )
    identity = build_pmml_scoring_identity(
        contract=contract,
        pmml_sha256=current_hashes["pmml"],
        sample_sha256=current_hashes["sample"],
        chunk_size=settings.pmml_scoring_chunk_size,
    )
    scoring_cache = settings.workspace / "cache" / "pmml_scoring"
    outputs_dir = settings.workspace / "tasks" / task_id / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    work_root = settings.workspace / "cache" / "pmml_scoring_work"
    working_score = work_root / task_id / f"{identity.cache_key}.parquet"
    working_score.parent.mkdir(parents=True, exist_ok=True)
    uow = ArtifactUnitOfWork()
    try:
        staged_scores = uow.stage_file(outputs_dir, "pmml_scores.parquet")

        def produce() -> object:
            scorer = TASK_PMML_SCORERS.get(
                task_id=task_id,
                pmml_path=materials.pmml,
                pmml_sha256=current_hashes["pmml"],
                output_field=identity.output_field,
            )
            return run_pmml_scoring(
                contract=contract,
                sample_path=materials.sample,
                pmml_path=materials.pmml,
                score_path=working_score,
                chunk_size=settings.pmml_scoring_chunk_size,
                scorer=scorer,
                pmml_sha256=current_hashes["pmml"],
                sample_sha256=current_hashes["sample"],
                cancellation_check=cancellation_check,
            )

        result = load_or_run_pmml_scoring(
            cache_dir=scoring_cache,
            cache_key=identity.cache_key,
            runner=produce,  # type: ignore[arg-type]
            materialize_path=staged_scores.path,
            cancellation_check=cancellation_check,
        )
        persisted_result = replace(
            result,
            score_artifact_path="pmml_scores.parquet",
        )
        # Strict round-trip before publishing orchestration evidence.
        payload = pmml_scoring_result_to_dict(persisted_result)
        pmml_scoring_result_from_dict(payload)
        staged_result = uow.stage_file(outputs_dir, "pmml_scoring_result.json")
        staged_result.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise_if_cancelled(cancellation_check)
        uow.finalize_with_connection(
            repo.transaction,
            lambda conn: repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.EXECUTED,
                message="PMML打分测试完成",
                expected=TaskStatus.RUNNING,
                begin_immediate=True,
            ),
        )
    except BaseException:
        uow.rollback()
        raise
    finally:
        working_score.unlink(missing_ok=True)
        _remove_empty_directory_chain(working_score.parent, stop_at=work_root)


def run_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> None:
    task = TaskRepository(settings.db_path).get_task(task_id)
    if task.validation_workflow_version == 2:
        return _run_v2_metrics_stage(
            task_id=task_id,
            settings=settings,
            stage_claimed=stage_claimed,
            cancellation_job_id=cancellation_job_id,
            cancellation_check=cancellation_check,
        )
    return _run_legacy_metrics_stage(
        task_id=task_id,
        settings=settings,
        stage_claimed=stage_claimed,
        cancellation_job_id=cancellation_job_id,
    )


def run_legacy_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
) -> None:
    """Explicit compatibility entry point for the built-in V1 tool pack."""

    return _run_legacy_metrics_stage(
        task_id=task_id,
        settings=settings,
        stage_claimed=stage_claimed,
        cancellation_job_id=cancellation_job_id,
    )


def _run_v2_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool,
    cancellation_job_id: str | None,
    cancellation_check: Callable[[], None] | None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if not stage_claimed:
        if task.status is not TaskStatus.EXECUTED:
            raise PipelineError(
                "PMML metrics require completed scoring; "
                f"current status is {task.status.value}"
            )
        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            message="模型压力测试进行中",
            expected=TaskStatus.EXECUTED,
        )
    try:
        with _pmml_job_cancellation(
            cancellation_job_id,
            cancellation_check,
        ) as active_cancellation_check:
            _execute_v2_metrics_stage(
                task_id=task_id,
                settings=settings,
                cancellation_check=active_cancellation_check,
            )
    except JobCancelled:
        _mark_cancelled(repo, task_id, TaskStatus.EXECUTED, "模型压力测试已取消")
    except _ValidationIdentityInvalid as exc:
        _invalidate_v2_validation_identity(
            repo, task_id=task_id, message=str(exc)
        )
        raise
    except Exception:
        _restore_metrics_resume_state(
            repo, task_id=task_id, message="模型效果或模型压力测试失败，可重试"
        )
        raise


def _execute_v2_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    cancellation_check: Callable[[], None] | None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    raise_if_cancelled(cancellation_check)
    try:
        materials = resolve_selected_validation_materials(task)
        record = require_confirmed_validation_input_contract(
            ValidationContractRepository(settings.db_path), task_id
        )
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc
    contract = record.contract
    current_hashes = _require_current_validation_materials(
        contract=contract,
        materials=materials,
        cancellation_check=cancellation_check,
    )
    outputs_dir = settings.workspace / "tasks" / task_id / "outputs"
    score_path = outputs_dir / "pmml_scores.parquet"
    result_path = outputs_dir / "pmml_scoring_result.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        persisted_scoring = pmml_scoring_result_from_dict(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _ValidationIdentityInvalid(
            "PMML scoring evidence is invalid; rerun PMML打分测试"
        ) from exc
    if persisted_scoring.score_artifact_path != "pmml_scores.parquet":
        raise _ValidationIdentityInvalid(
            "PMML scoring evidence path is invalid; rerun PMML打分测试"
        )
    identity = build_pmml_scoring_identity(
        contract=contract,
        pmml_sha256=current_hashes["pmml"],
        sample_sha256=current_hashes["sample"],
        chunk_size=settings.pmml_scoring_chunk_size,
    )
    if (
        persisted_scoring.cache_key != identity.cache_key
        or persisted_scoring.pmml_sha256 != current_hashes["pmml"]
        or persisted_scoring.sample_sha256 != current_hashes["sample"]
        or persisted_scoring.output_field != identity.output_field
        or persisted_scoring.engine_version != identity.engine_version
    ):
        raise _ValidationIdentityInvalid(
            "PMML scoring evidence does not match selected materials"
        )
    runtime_scoring = replace(
        persisted_scoring,
        score_artifact_path=str(score_path),
    )
    try:
        runtime_scoring = validate_pmml_score_artifact(
            runtime_scoring,
            score_path,
            expected_cache_key=identity.cache_key,
            cancellation_check=cancellation_check,
        )
    except ValueError as exc:
        raise _ValidationIdentityInvalid(
            "task-local PMML score artifact is invalid; rerun PMML打分测试"
        ) from exc

    work_dir = outputs_dir / ".pmml-metrics-stage-work"
    _remove_dir_if_exists(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)
    uow: ArtifactUnitOfWork | None = None
    try:
        config = validation_config_from_input_contract(contract, settings)

        def scorer_factory():
            return TASK_PMML_SCORERS.get(
                task_id=task_id,
                pmml_path=materials.pmml,
                pmml_sha256=current_hashes["pmml"],
                output_field=identity.output_field,
            )

        metadata = contract.require_feature_metadata()
        stress = run_pmml_stress(
            contract=contract,
            config=config,
            sample_path=materials.sample,
            baseline_score_path=score_path,
            scoring_result=runtime_scoring,
            scenario_dir=work_dir / "stress",
            feature_categories=metadata.per_category_raw_fields,
            scorer_factory=scorer_factory,
            chunk_size=settings.pmml_scoring_chunk_size,
            cancellation_check=cancellation_check,
            category_source_counts={
                "dictionary": len(metadata.rows),
                "unresolved": 0,
            },
            baseline_cache_key=runtime_scoring.cache_key,
            cache_dir=settings.workspace / "cache" / "pmml_stress",
        )
        require_complete_stress_result(stress)
        results = compute_platform_validation_results(
            task=task,
            contract=contract,
            sample_path=materials.sample,
            score_path=score_path,
            scoring_result=runtime_scoring,
            metadata_resolution=metadata,
            stress_test=stress,
            settings=settings,
            cancellation_check=cancellation_check,
        )
        persisted_results = replace(
            results,
            pmml_scoring=replace(
                runtime_scoring,
                score_artifact_path="pmml_scores.parquet",
            ),
        )
        results_payload = validation_results_to_dict(persisted_results)
        validation_results_from_dict(results_payload)
        work_result = work_dir / "validation_results.json"
        work_result.write_text(
            json.dumps(results_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise_if_cancelled(cancellation_check)
        uow = ArtifactUnitOfWork()
        staged_result = uow.stage_file(outputs_dir, "validation_results.json")
        shutil.move(str(work_result), staged_result.path)
        raise_if_cancelled(cancellation_check)
        uow.finalize_with_connection(
            repo.transaction,
            lambda conn: repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.WRITING_ARTIFACTS,
                message="模型效果与模型压力测试完成",
                expected=TaskStatus.COMPUTING_METRICS,
                begin_immediate=True,
            ),
        )
    except BaseException:
        if uow is not None:
            uow.rollback()
        raise
    finally:
        _remove_dir_if_exists(work_dir)


def _run_legacy_metrics_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    stage_claimed: bool = False,
    cancellation_job_id: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    execution_dir = task_dir / "execution"
    outputs_dir = task_dir / "outputs"
    metrics_work_dir = outputs_dir / ".metrics-stage-work"
    metrics_steps_path = execution_dir / "metrics_steps.json"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_uow: ArtifactUnitOfWork | None = None
    logger.info("metrics stage starting task_id=%s stage_claimed=%s", task_id, stage_claimed)
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
        cancellation_token = register_notebook_cancellation(
            task_id,
            job_id=cancellation_job_id,
        )
        try:
            live_session = get_live_notebook_session(task_id)
            if settings.notebook_isolated_execution:
                if live_session is not None:
                    close_live_notebook_session(task_id)
                    live_session = None
            elif not legacy_live_notebook_execution_allowed(settings):
                if live_session is not None:
                    close_live_notebook_session(task_id)
                raise PipelineError(LEGACY_LIVE_NOTEBOOK_DISABLED_MESSAGE)
            elif live_session is None:
                raise PipelineError(
                    "live notebook kernel is not available; rerun notebook stage before metrics"
                )
            _unlink_if_exists(_metrics_cancel_marker_path(task_dir))
            if settings.notebook_isolated_execution and _metrics_work_dir_prepared(
                metrics_work_dir
            ):
                # PERF-3: the notebook stage of run_staged_pipeline's default
                # consecutive path already executed the metrics cells in the
                # same isolated subprocess run as the reproducibility cells
                # (see run_notebook_stage's also_prepare_metrics) and left
                # valid outputs here. Skip re-running the notebook entirely
                # and fall through to the shared finalization below.
                contract = load_runtime_contract(execution_dir / "runtime_contract.json")
                task = _sync_task_algorithm(repo, task, contract.algorithm)
                metrics_uow = ArtifactUnitOfWork()
            else:
                _remove_dir_if_exists(metrics_work_dir)
                _unlink_if_exists(metrics_steps_path)
                metrics_work_dir.mkdir(parents=True, exist_ok=True)
                contract = load_runtime_contract(execution_dir / "runtime_contract.json")
                task = _sync_task_algorithm(repo, task, contract.algorithm)
                metrics_uow = ArtifactUnitOfWork()
                model_meta_path = _stage_model_meta_from_contract(
                    metrics_uow,
                    contract,
                    execution_dir,
                )
                if live_session is None:
                    sample_path = _required_path(
                        task, artifacts, FileRole.SAMPLE, "sample", "sample_path"
                    )
                    stress_scores_path = execution_dir / STRESS_SCENARIO_SCORES_JSON
                    if not _stress_scores_artifact_valid(stress_scores_path):
                        _unlink_if_exists(stress_scores_path)
                        notebook_path = _required_path(
                            task,
                            artifacts,
                            FileRole.NOTEBOOK,
                            "notebook",
                            "notebook_path",
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
                            notebook_steps_path=metrics_steps_path,
                            kernel_name=_execution_kernel_name(settings),
                            notebook_memory_limit_mb=settings.notebook_memory_limit_mb,
                            stage_claimed=True,
                            cancellation_token=cancellation_token,
                            keep_alive=False,
                            isolated=True,
                            mark_executed=False,
                            cancel_message="metrics cancelled",
                            cancel_resume_status=TaskStatus.EXECUTED,
                            extra_code_cells=_build_stress_scenario_score_cell_sources(
                                task=task,
                                settings=settings,
                                dictionary_path=dictionary_path,
                                contract_meta_path=execution_dir / "runtime_contract.json",
                                output_path=stress_scores_path,
                            ),
                        )
                        if repo.get_task(task_id).status != TaskStatus.COMPUTING_METRICS:
                            _rollback_artifact_uow(metrics_uow)
                            return
                        contract = load_runtime_contract(
                            execution_dir / "runtime_contract.json"
                        )
                        task = _sync_task_algorithm(repo, task, contract.algorithm)
                        model_meta_path = _stage_model_meta_from_contract(
                            metrics_uow,
                            contract,
                            execution_dir,
                        )
                    # Metrics may be retried while the stress-score artifact from
                    # the previous attempt is still valid. Rebuild reproducibility
                    # evidence on every attempt so freshly computed metrics cannot
                    # be paired with a stale Notebook-vs-PMML comparison.
                    staged_reproducibility = metrics_uow.stage_file(
                        outputs_dir,
                        REPRODUCIBILITY_RESULT_JSON,
                    )
                    write_reproducibility_result(
                        task=task,
                        contract=contract,
                        settings=settings,
                        input_pmml_path=input_pmml_path,
                        output_path=staged_reproducibility.path,
                        fallback_sample_path=sample_path,
                    )
                    write_platform_validation_metrics(
                        task=task,
                        contract=contract,
                        settings=settings,
                        dictionary_path=dictionary_path,
                        model_meta_path=model_meta_path,
                        reproducibility_json_path=staged_reproducibility.path,
                        results_json_path=metrics_work_dir / "validation_results.json",
                        excel_path=metrics_work_dir / "validation.xlsx",
                        stress_scores_path=stress_scores_path,
                        fallback_sample_path=sample_path,
                    )
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
                            task_dir=task_dir,
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
            uow=metrics_uow,
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
        logger.info("metrics stage finished task_id=%s", task_id)
        _capture_agent_memory_for_metrics_success(
            repo=repo,
            task_id=task_id,
            outputs_dir=outputs_dir,
        )
    except PipelineCancelled as exc:
        logger.info("metrics stage cancelled task_id=%s", task_id)
        _rollback_artifact_uow(metrics_uow)
        _mark_cancelled(repo, task_id, exc.resume_status, str(exc))
        return
    except PipelineError as exc:
        logger.error("metrics stage failed task_id=%s error=%s", task_id, exc)
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
        logger.error(
            "metrics stage failed unexpectedly task_id=%s error_type=%s",
            task_id, exc.__class__.__name__, exc_info=True,
        )
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


def run_report_stage(
    *,
    task_id: str,
    settings: PipelineSettings,
    cancellation_job_id: str | None = None,
) -> None:
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    task_dir = settings.workspace / "tasks" / task_id
    outputs_dir = task_dir / "outputs"
    images_dir = task_dir / "images"
    report_path = outputs_dir / "validation_report.docx"
    temp_report_path = outputs_dir / ".validation_report.docx.tmp"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cancellation_token = register_notebook_cancellation(
        task_id,
        job_id=cancellation_job_id,
    )
    report_uow: ArtifactUnitOfWork | None = None
    logger.info("report stage starting task_id=%s", task_id)
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
        terminal_status = _terminal_validation_status(task, results)
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
        logger.info("report stage finished task_id=%s terminal_status=%s", task_id, terminal_status.value)
    except NotebookCancelled:
        logger.info("report stage cancelled task_id=%s", task_id)
        _rollback_report_uow(report_uow)
        _unlink_if_exists(temp_report_path)
        _mark_cancelled(repo, task_id, TaskStatus.REVIEW_REQUIRED, "report cancelled")
        return
    except PipelineError as exc:
        logger.error("report stage failed task_id=%s error=%s", task_id, exc)
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
        logger.error(
            "report stage failed unexpectedly task_id=%s error_type=%s",
            task_id, exc.__class__.__name__, exc_info=True,
        )
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


def _terminal_validation_status(task: TaskRecord, results) -> TaskStatus:
    if results.reproducibility is None:
        return TaskStatus.SUCCEEDED
    return (
        TaskStatus.REVIEW_REQUIRED
        if results.reproducibility.summary.status is ConsistencyStatus.FAIL
        else TaskStatus.SUCCEEDED
    )


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
    uow: ArtifactUnitOfWork | None = None,
) -> ArtifactUnitOfWork:
    if uow is None:
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


def _stage_model_meta_from_contract(
    uow: ArtifactUnitOfWork,
    contract: RuntimeContract,
    execution_dir: Path,
) -> Path:
    artifact = uow.stage_file(execution_dir, "model_meta.json")
    return _write_model_meta_from_contract(contract, artifact.path)


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
    task_dir: Path,
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
        task_dir=task_dir,
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


def run_staged_pipeline(
    *,
    task_id: str,
    settings: PipelineSettings,
    cancellation_job_id: str | None = None,
) -> None:
    task = TaskRepository(settings.db_path).get_task(task_id)
    if task.validation_workflow_version == 2:
        run_pmml_scoring_stage(
            task_id=task_id,
            settings=settings,
            cancellation_job_id=cancellation_job_id,
        )
        task = TaskRepository(settings.db_path).get_task(task_id)
        if task.status is not TaskStatus.EXECUTED:
            return
        run_metrics_stage(
            task_id=task_id,
            settings=settings,
            cancellation_job_id=cancellation_job_id,
        )
        task = TaskRepository(settings.db_path).get_task(task_id)
        if task.status not in {
            TaskStatus.WRITING_ARTIFACTS,
            TaskStatus.REVIEW_REQUIRED,
        }:
            return
        run_report_stage(
            task_id=task_id,
            settings=settings,
            cancellation_job_id=cancellation_job_id,
        )
        return
    return _run_legacy_staged_pipeline(
        task_id=task_id,
        settings=settings,
        cancellation_job_id=cancellation_job_id,
    )


def _run_legacy_staged_pipeline(
    *,
    task_id: str,
    settings: PipelineSettings,
    cancellation_job_id: str | None = None,
) -> None:
    logger.info("staged pipeline starting task_id=%s", task_id)
    repo = TaskRepository(settings.db_path)
    task = repo.get_task(task_id)
    if task.status in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        raise PipelineError(
            f"task already {task.status.value}; reset task before rerunning pipeline"
        )

    # PERF-3: in isolated mode, run the metrics cells in the same subprocess
    # call as the notebook stage so the user notebook executes once instead
    # of twice. run_metrics_stage below then detects the pre-populated
    # outputs and skips its own notebook re-run. This only applies to this
    # default consecutive path; standalone notebook/metrics retries via the
    # API are unaffected (also_prepare_metrics defaults to False there).
    run_notebook_stage(
        task_id=task_id,
        settings=settings,
        also_prepare_metrics=settings.notebook_isolated_execution,
        cancellation_job_id=cancellation_job_id,
    )
    task = repo.get_task(task_id)
    if task.status is not TaskStatus.EXECUTED:
        logger.info(
            "staged pipeline stopping after notebook stage task_id=%s status=%s",
            task_id, task.status.value,
        )
        return

    run_metrics_stage(
        task_id=task_id,
        settings=settings,
        cancellation_job_id=cancellation_job_id,
    )
    task = repo.get_task(task_id)
    if task.status not in {TaskStatus.WRITING_ARTIFACTS, TaskStatus.REVIEW_REQUIRED}:
        logger.info(
            "staged pipeline stopping after metrics stage task_id=%s status=%s",
            task_id, task.status.value,
        )
        return

    run_report_stage(
        task_id=task_id,
        settings=settings,
        cancellation_job_id=cancellation_job_id,
    )
    logger.info("staged pipeline finished task_id=%s", task_id)


def _stress_scores_artifact_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("categories"), list):
        return False
    if payload.get("schema_version") != "marvis.validation_stress_scores.v2":
        return False
    feature_categories = payload.get("feature_categories")
    unclassified = payload.get("unclassified_features")
    source_counts = payload.get("source_counts")
    conflicts = payload.get("conflicts")
    if not isinstance(feature_categories, dict):
        return False
    if not _valid_stress_feature_categories(feature_categories):
        return False
    if not _valid_unique_strings(unclassified):
        return False
    if not isinstance(source_counts, dict) or set(source_counts) != {
        "notebook",
        "dictionary",
        "unresolved",
    }:
        return False
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in source_counts.values()
    ):
        return False
    if not isinstance(conflicts, list) or conflicts:
        return False
    classified_features = [
        feature for features in feature_categories.values() for feature in features
    ]
    if len(classified_features) != len(set(classified_features)):
        return False
    if set(classified_features) & set(unclassified):
        return False
    if sum(source_counts.values()) != len(classified_features) + len(unclassified):
        return False
    if source_counts["unresolved"] != len(unclassified):
        return False
    rows_by_category: dict[str, dict] = {}
    for row in payload["categories"]:
        if not isinstance(row, dict):
            return False
        category = str(row.get("category") or "").strip()
        if not category or category in rows_by_category:
            return False
        rows_by_category[category] = row
    if set(rows_by_category) != set(feature_categories):
        return False
    for category, row in rows_by_category.items():
        dropped_features = row.get("dropped_features")
        if dropped_features != feature_categories[category]:
            return False
        status = row.get("status")
        if status not in {None, "completed", "skipped", "error"}:
            return False
        row_indexes = row.get("row_index")
        scores = row.get("scores")
        if not isinstance(row_indexes, list) or not isinstance(scores, list):
            return False
        if len(row_indexes) != len(scores):
            return False
        try:
            if any(
                isinstance(score, bool) or not math.isfinite(float(score))
                for score in scores
            ):
                return False
        except (TypeError, ValueError):
            return False
        error = row.get("error")
        if status == "error" and not str(error or "").strip():
            return False
        if status in {"completed", "skipped"} and error not in {None, ""}:
            return False
    return True


def _valid_stress_feature_categories(value: dict) -> bool:
    for category, features in value.items():
        if not isinstance(category, str) or not category.strip():
            return False
        if not _valid_unique_strings(features) or not features:
            return False
    return True


def _valid_unique_strings(value: object) -> bool:
    if not isinstance(value, list):
        return False
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return False
    return len(value) == len(set(value))


def run_pipeline(*, task_id: str, settings: PipelineSettings) -> None:
    task = TaskRepository(settings.db_path).get_task(task_id)
    if task.validation_workflow_version == 2:
        run_staged_pipeline(task_id=task_id, settings=settings)
        return
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
    metrics_work_dir = outputs_dir / ".metrics-stage-work"
    execution_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    live_session: NotebookExecutionSession | None = None
    metrics_uow: ArtifactUnitOfWork | None = None
    failure_prefix = NOTEBOOK_STAGE_FAILURE_PREFIX

    logger.info("legacy live-notebook pipeline starting task_id=%s", task_id)
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
        metrics_uow = ArtifactUnitOfWork()
        model_meta_path = _stage_model_meta_from_contract(
            metrics_uow,
            contract,
            execution_dir,
        )

        repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            message="computing metrics",
            expected=TaskStatus.EXECUTED,
        )
        failure_prefix = METRICS_STAGE_FAILURE_PREFIX
        _remove_dir_if_exists(metrics_work_dir)
        metrics_work_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics_results_in_session(
            session=live_session,
            task_dir=task_dir,
            task=task,
            settings=settings,
            dictionary_path=dictionary_path,
            input_pmml_path=input_pmml_path,
            contract=contract,
            model_meta_path=model_meta_path,
            reproducibility_json_path=outputs_dir / REPRODUCIBILITY_RESULT_JSON,
            outputs_dir=metrics_work_dir,
        )
        live_session.close()
        live_session = None

        _require_metrics_outputs(metrics_work_dir)
        metrics_uow = _stage_metrics_outputs_for_commit(
            task_dir=task_dir,
            outputs_dir=outputs_dir,
            metrics_work_dir=metrics_work_dir,
            uow=metrics_uow,
        )
        metrics_uow.finalize_with_connection(
            repo.transaction,
            lambda conn: repo.update_status_on_connection(
                conn,
                task_id,
                TaskStatus.WRITING_ARTIFACTS,
                message="writing artifacts",
                expected=TaskStatus.COMPUTING_METRICS,
                begin_immediate=True,
            ),
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

        terminal_status = _terminal_validation_status(task, results)
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
        logger.error("legacy live-notebook pipeline failed task_id=%s error=%s", task_id, exc)
        _rollback_artifact_uow(metrics_uow)
        _mark_failed(repo, task_id, _stage_failure_message(failure_prefix, str(exc)))
        raise
    except Exception as exc:
        logger.error(
            "legacy live-notebook pipeline failed unexpectedly task_id=%s error_type=%s",
            task_id, exc.__class__.__name__, exc_info=True,
        )
        _rollback_artifact_uow(metrics_uow)
        _mark_failed(
            repo,
            task_id,
            _stage_failure_message(failure_prefix, f"{exc.__class__.__name__}: {exc}"),
        )
        raise
    finally:
        if live_session is not None:
            live_session.close()
        _remove_dir_if_exists(metrics_work_dir)


def _scan_step(repo: TaskRepository, task: TaskRecord) -> list[FileArtifact]:
    artifacts = _scan_artifacts(task)
    logger.debug("source scan complete task_id=%s artifact_count=%d", task.id, len(artifacts))
    repo.update_status(
        task.id,
        TaskStatus.SCANNED,
        message="source scanned",
        expected={TaskStatus.CREATED, TaskStatus.SCANNED, TaskStatus.FAILED},
    )
    return artifacts


def _scan_artifacts(task: TaskRecord) -> list[FileArtifact]:
    task_id = getattr(task, "id", None)
    try:
        return scan_source_dir(Path(task.source_dir))
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("source dir invalid task_id=%s error=%s", task_id, exc)
        raise PipelineError(f"source dir invalid: {exc}") from exc
    except ValueError as exc:
        # scan-limit breaches (max_files / max_depth) raise ValueError. Tag them
        # with the scan-stage prefix so the notebook/metrics/report except handlers
        # keep the failure attributed to the scan stage instead of mislabeling it
        # as a notebook failure (see _stage_failure_message + _is_scan_failure).
        logger.error("source scan limit breach task_id=%s error=%s", task_id, exc)
        raise PipelineError(f"{SCAN_STAGE_FAILURE_PREFIX}{exc}") from exc


def _capture_agent_memory_for_metrics_success(
    *,
    repo: TaskRepository,
    task_id: str,
    outputs_dir: Path,
) -> None:
    # Gate on the user-facing "自动沉淀任务经验" (auto_distill) memory policy:
    # when off, no automatic capture happens on this pipeline surface either.
    if not load_memory_policy(repo.db_path.parent).auto_distill:
        logger.debug("auto_distill disabled; skipping memory capture task_id=%s", task_id)
        return
    try:
        task = repo.get_task(task_id)
        payload = _read_validation_results_payload(outputs_dir)
        store = AgentMemoryStore(repo.db_path)
        created = 0
        for candidate in (
            extract_model_experience(
                _memory_model_experience_payload(task=task, results=payload)
            ),
            extract_field_convention(_memory_field_convention_payload(task)),
        ):
            if candidate is not None:
                store.create(candidate, task_id=task_id)
                created += 1
        logger.info(
            "agent memory captured on metrics success task_id=%s entries=%d",
            task_id, created,
        )
    except Exception:
        logger.warning(
            "agent memory capture failed on metrics success task_id=%s", task_id, exc_info=True
        )
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
        logger.debug("auto_distill disabled; skipping failure memory capture task_id=%s", task_id)
        return
    store = AgentMemoryStore(repo.db_path)
    _downgrade_task_memory_on_failure(store, task_id=task_id, reason=f"task_failed:{failure_kind}")
    try:
        payload = {
            "task_id": task_id,
            "status": "failed",
            "summary": message,
            "failures": [{"kind": failure_kind, "message": message}],
        }
        created = 0
        for candidate in [
            *extract_validation_pitfall(payload),
            extract_task_experience(payload),
        ]:
            if candidate is not None:
                store.create(candidate, task_id=task_id)
                created += 1
        logger.info(
            "agent memory captured on failure task_id=%s failure_kind=%s entries=%d",
            task_id, failure_kind, created,
        )
    except Exception:
        logger.warning(
            "agent memory capture failed on failure path task_id=%s", task_id, exc_info=True
        )
        return


def _downgrade_task_memory_on_failure(
    store: AgentMemoryStore,
    *,
    task_id: str,
    reason: str,
) -> None:
    # MEM-7 negative feedback loop: when a task reaches its FAILED terminal
    # state, every active memory entry that task itself produced earlier in
    # its lifecycle (field conventions, model experience, etc. captured before
    # it ultimately failed) gets a negative_feedback audit event and a
    # one-tier confidence downgrade -- a weak prior tied to a failed run is
    # worse than no prior. This runs before the failure-record candidates
    # below are created, so the task_experience/validation_pitfall entries
    # that describe *this* failure are not immediately self-downgraded. This
    # is a pure retrieval-ranking signal -- it never touches deterministic
    # metrics (INV-4). Kept in its own try/except so a downgrade failure
    # (or a store stub in tests that only implements a subset of the API)
    # never blocks the failure-record candidates below from being captured.
    try:
        entries = store.list_entries(source_task_id=task_id, limit=200)
    except Exception:
        return
    for entry in entries:
        try:
            store.record_negative_feedback(
                entry.id,
                task_id=task_id,
                reason=reason,
            )
        except (KeyError, ValueError):
            continue


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
    logger.debug(
        "notebook execution starting task_id=%s kernel=%s isolated=%s keep_alive=%s",
        task.id, kernel_name, isolated, keep_alive,
    )
    _notebook_step_started_at = time.monotonic()
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
        runtime_sample_path=execution_dir / "runtime_sample.csv",
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
    _notebook_step_elapsed_s = time.monotonic() - _notebook_step_started_at
    if result.step_events is not None:
        write_json_atomic(notebook_steps_path, result.step_events)
    if result.cancelled:
        logger.info(
            "notebook execution cancelled task_id=%s elapsed_s=%.1f",
            task.id, _notebook_step_elapsed_s,
        )
        if live_session is not None:
            live_session.close()
        _mark_cancelled(repo, task.id, cancel_resume_status, cancel_message)
        return
    if not result.succeeded:
        logger.error(
            "notebook execution failed task_id=%s cell=%s error=%s elapsed_s=%.1f",
            task.id, result.failed_cell_index, result.error_name, _notebook_step_elapsed_s,
        )
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
    logger.debug(
        "notebook execution succeeded task_id=%s elapsed_s=%.1f",
        task.id, _notebook_step_elapsed_s,
    )
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
