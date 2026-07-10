"""Path resolution, sample loading, and artifact-hygiene helpers for the
validation pipeline.

Extracted from marvis/pipeline.py (ARCH-6) -- pure/self-contained helpers
with no monkeypatch coupling to marvis.pipeline's module namespace (verified
against the full test suite's monkeypatch.setattr("marvis.pipeline....", ...)
targets before the split). No behavior change.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from marvis.domain import FileArtifact, FileRole, TaskRecord
from marvis.execution_environment import (
    load_execution_environment,
    validate_execution_environment,
)
from marvis.model_algorithms import normalize_algorithm
from marvis.notebook_contract import RuntimeContract
from marvis.notebooks import _notebook_worker_env
from marvis.pipeline_errors import PipelineError
from marvis.validation.results import validation_results_from_dict

if TYPE_CHECKING:
    from marvis.db import TaskRepository
    from marvis.pipeline import PipelineSettings

logger = logging.getLogger(__name__)

VALIDATION_RESULTS_PICKLE = "validation_results.pkl"
REPRODUCIBILITY_RESULT_JSON = "reproducibility_result.json"
METRICS_CANCEL_MARKER = "metrics_cancel.requested"
STRESS_SCENARIO_SCORES_JSON = "stress_scenario_scores.json"
SCAN_STAGE_FAILURE_PREFIX = "材料扫描失败："

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
        logger.warning(
            "platform python missing arrow support for %s sample; retrying with fallback python",
            suffix,
        )
        try:
            result = _load_arrow_sample_with_python(
                sample_path,
                suffix=suffix,
                python_executable=Path(fallback_python),
            )
        except Exception as fallback_exc:
            logger.error(
                "arrow sample fallback load failed: platform_error=%s fallback_error=%s",
                exc.__class__.__name__,
                fallback_exc.__class__.__name__,
            )
            raise PipelineError(
                "failed to load sample with both platform Python and selected "
                f"execution Python; platform error: {exc}; fallback error: "
                f"{fallback_exc.__class__.__name__}: {fallback_exc}"
            ) from fallback_exc
        logger.info("arrow sample fallback load succeeded rows=%d", len(result))
        return result


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
            env=_notebook_worker_env(),
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
        logger.error("execution environment invalid: %s", validation.message)
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
    logger.debug("clearing generated artifacts stage=%s task_dir=%s", stage, task_dir.name)
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
            "metrics_steps.json",
            STRESS_SCENARIO_SCORES_JSON,
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
        _remove_dir_if_exists(outputs_dir / ".metrics-stage-work")
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


def _metrics_cancel_marker_path(task_dir: Path) -> Path:
    return task_dir / "execution" / METRICS_CANCEL_MARKER


def _require_metrics_outputs(outputs_dir: Path) -> None:
    required = [
        outputs_dir / "validation_results.json",
        outputs_dir / "validation.xlsx",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise PipelineError("notebook metrics did not produce: " + ", ".join(missing))


def _metrics_work_dir_prepared(metrics_work_dir: Path) -> bool:
    """True when a prior merged notebook+metrics subprocess run (PERF-3,
    see run_notebook_stage's also_prepare_metrics) already wrote valid
    metrics outputs here, so run_metrics_stage can skip re-executing the
    notebook."""
    return (
        (metrics_work_dir / "validation_results.json").exists()
        and (metrics_work_dir / "validation.xlsx").exists()
    )


def _notebook_execution_artifacts_complete(execution_dir: Path) -> bool:
    return (
        (execution_dir / "code_model_scores.csv").exists()
        and (execution_dir / "runtime_contract.json").exists()
        and (execution_dir / "model_meta.json").exists()
    )


def _algorithm(value: str) -> str:
    return normalize_algorithm(value)


def _sync_task_algorithm(
    repo: TaskRepository,
    task: TaskRecord,
    algorithm: str,
) -> TaskRecord:
    normalized = _algorithm(algorithm)
    if task.algorithm == normalized:
        return task
    return repo.update_algorithm(task.id, normalized)


def _stage_failure_message(prefix: str, message: str) -> str:
    # A scan-stage failure can surface while a later stage's prefix is active
    # (the notebook/metrics/report try-blocks re-scan artifacts). Preserve an
    # already scan-prefixed message so _is_scan_failure keeps the right attribution.
    if message.startswith(prefix) or message.startswith(SCAN_STAGE_FAILURE_PREFIX):
        return message
    return f"{prefix}{message}"


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}
