from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat

from marvis.artifacts import TransactionalDirectoryStore
from marvis.db import DatasetRepository, TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION, TaskCreate, TaskStatus
from marvis.packs.modeling.artifact import export_pmml, persist_model_meta
from marvis.packs.modeling.contracts import Experiment, ModelArtifact
from marvis.packs.modeling.errors import ModelingError


HANDOFF_DIR_NAME = "validation_handoff"
CHALLENGER_BACKTEST_DIR_NAME = "challenger_backtest"
MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"
SCORING_NOTEBOOK_NAME = "scoring_notebook.ipynb"
DICTIONARY_NAME = "dictionary.csv"
CHALLENGER_BACKTEST_PLAN_JSON = "challenger_backtest_plan.json"
CHALLENGER_BACKTEST_PLAN_MD = "challenger_backtest_plan.md"


def handoff_to_validation(
    experiment_store,
    artifact: ModelArtifact,
    *,
    sample_dataset_id: str,
    settings,
) -> str:
    if not artifact.experiment_id:
        raise ModelingError("artifact experiment_id is required for validation handoff")
    experiment = experiment_store.get(artifact.experiment_id)
    if experiment.artifact_id and experiment.artifact_id != artifact.id:
        raise ModelingError(
            f"artifact {artifact.id} is not attached to experiment {experiment.id}"
        )
    sample_path = _sample_dataset_path(settings, sample_dataset_id)
    artifact_base_dir = _artifact_base_dir(settings, experiment)
    model_path = _resolve_artifact_path(artifact.model_path, base_dir=artifact_base_dir)
    if not model_path.exists():
        raise ModelingError(f"model file does not exist: {model_path}")
    pmml_path = _ensure_pmml_path(
        experiment_store,
        artifact,
        experiment=experiment,
        sample_path=sample_path,
        artifact_base_dir=artifact_base_dir,
    )

    material_dir = _material_dir(settings, experiment, artifact)
    staged_materials = TransactionalDirectoryStore(material_dir.parent).stage(material_dir.name)
    staged_materials.path.mkdir(parents=True, exist_ok=True)
    sample_material_name = f"sample{sample_path.suffix or '.parquet'}"
    model_material_name = f"model{model_path.suffix or '.joblib'}"
    shutil.copy2(sample_path, staged_materials.path / sample_material_name)
    shutil.copy2(pmml_path, staged_materials.path / "model.pmml")
    shutil.copy2(model_path, staged_materials.path / model_material_name)
    calibration_material_name = _copy_calibration_payload(
        artifact,
        artifact_base_dir=artifact_base_dir,
        material_dir=staged_materials.path,
    )
    _write_dictionary(staged_materials.path / DICTIONARY_NAME, artifact.feature_list)
    _write_scoring_notebook(
        staged_materials.path / SCORING_NOTEBOOK_NAME,
        artifact=artifact,
        experiment=experiment,
        model_filename=model_material_name,
        calibration_filename=calibration_material_name,
    )

    source_task = _source_task(settings, experiment.task_id)
    payload = TaskCreate(
        task_type=TASK_TYPE_VALIDATION,
        model_name=source_task.model_name if source_task else f"{experiment.recipe_id} model",
        model_version=artifact.id,
        validator=source_task.validator if source_task else "MARVIS Modeling",
        source_dir=str(material_dir.resolve()),
        algorithm=artifact.algorithm,
        run_mode="agent",
        target_col=experiment.config.target_col,
        score_col=_score_col(experiment.config.params),
        split_col=experiment.config.split_col,
        time_col=_time_col(experiment.config.params),
        feature_columns=list(artifact.feature_list),
        notebook_path=SCORING_NOTEBOOK_NAME,
        sample_path=sample_material_name,
        pmml_path="model.pmml",
        dictionary_path=DICTIONARY_NAME,
        report_values={},
    )
    try:
        staged_materials.activate()
        validation_task = TaskRepository(settings.db_path).create_validation_handoff_with_audit(
            payload,
            experiment_id=artifact.experiment_id,
            experiment_status="handed_off",
            audit_factory=lambda record: {
                "kind": "modeling.validation_handoff.create",
                "target_ref": record.id,
                "outcome": "succeeded",
                "detail": {
                    "experiment_id": artifact.experiment_id,
                    "artifact_id": artifact.id,
                    "sample_dataset_id": sample_dataset_id,
                    "source_dir": str(material_dir.resolve()),
                },
            },
        )
        staged_materials.commit()
    except Exception:
        staged_materials.rollback()
        raise
    return validation_task.id


def mark_validated_from_validation_task(
    experiment_store,
    artifact: ModelArtifact,
    *,
    validation_task_id: str,
    settings,
) -> bool:
    task = TaskRepository(settings.db_path).get_task(validation_task_id)
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.REVIEW_REQUIRED}:
        return False
    experiment_store.set_status(artifact.experiment_id, "validated")
    return True


def create_challenger_backtest_task(
    experiment_store,
    artifact: ModelArtifact,
    *,
    sample_dataset_id: str,
    settings,
    selection_policy_decision: dict | None = None,
) -> dict[str, str]:
    if not artifact.experiment_id:
        raise ModelingError("artifact experiment_id is required for challenger/backtest task")
    experiment = experiment_store.get(artifact.experiment_id)
    if experiment.artifact_id and experiment.artifact_id != artifact.id:
        raise ModelingError(
            f"artifact {artifact.id} is not attached to experiment {experiment.id}"
        )
    sample_path = _sample_dataset_path(settings, sample_dataset_id)
    artifact_base_dir = _artifact_base_dir(settings, experiment)
    model_path = _resolve_artifact_path(artifact.model_path, base_dir=artifact_base_dir)
    if not model_path.exists():
        raise ModelingError(f"model file does not exist: {model_path}")
    pmml_path = _ensure_pmml_path(
        experiment_store,
        artifact,
        experiment=experiment,
        sample_path=sample_path,
        artifact_base_dir=artifact_base_dir,
    )

    material_dir = _challenger_backtest_dir(settings, experiment, artifact)
    staged_materials = TransactionalDirectoryStore(material_dir.parent).stage(material_dir.name)
    staged_materials.path.mkdir(parents=True, exist_ok=True)
    sample_material_name = f"sample{sample_path.suffix or '.parquet'}"
    model_material_name = f"model{model_path.suffix or '.joblib'}"
    shutil.copy2(sample_path, staged_materials.path / sample_material_name)
    shutil.copy2(pmml_path, staged_materials.path / "model.pmml")
    shutil.copy2(model_path, staged_materials.path / model_material_name)
    calibration_material_name = _copy_calibration_payload(
        artifact,
        artifact_base_dir=artifact_base_dir,
        material_dir=staged_materials.path,
    )
    _write_dictionary(staged_materials.path / DICTIONARY_NAME, artifact.feature_list)
    _write_scoring_notebook(
        staged_materials.path / SCORING_NOTEBOOK_NAME,
        artifact=artifact,
        experiment=experiment,
        model_filename=model_material_name,
        calibration_filename=calibration_material_name,
    )
    plan_payload = _challenger_backtest_payload(
        experiment=experiment,
        artifact=artifact,
        sample_dataset_id=sample_dataset_id,
        sample_material_name=sample_material_name,
        model_material_name=model_material_name,
        selection_policy_decision=selection_policy_decision or {},
    )
    (staged_materials.path / CHALLENGER_BACKTEST_PLAN_JSON).write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (staged_materials.path / CHALLENGER_BACKTEST_PLAN_MD).write_text(
        _challenger_backtest_markdown(plan_payload),
        encoding="utf-8",
    )

    source_task = _source_task(settings, experiment.task_id)
    payload = TaskCreate(
        task_type=TASK_TYPE_VALIDATION,
        model_name=source_task.model_name if source_task else f"{experiment.recipe_id} model",
        model_version=f"{artifact.id}-challenger-backtest",
        validator="MARVIS Challenger Backtest",
        source_dir=str(material_dir.resolve()),
        algorithm=artifact.algorithm,
        run_mode="agent",
        target_col=experiment.config.target_col,
        score_col=_score_col(experiment.config.params),
        split_col=experiment.config.split_col,
        time_col=_time_col(experiment.config.params),
        feature_columns=list(artifact.feature_list),
        notebook_path=SCORING_NOTEBOOK_NAME,
        sample_path=sample_material_name,
        pmml_path="model.pmml",
        dictionary_path=DICTIONARY_NAME,
        report_values={
            "TEXT:task_kind": "modeling_challenger_backtest",
            "TEXT:source_experiment_id": artifact.experiment_id,
            "TEXT:source_artifact_id": artifact.id,
            "TEXT:sample_dataset_id": sample_dataset_id,
            "TEXT:plan_path": CHALLENGER_BACKTEST_PLAN_JSON,
        },
    )
    try:
        staged_materials.activate()
        task = TaskRepository(settings.db_path).create_task_with_audit(
            payload,
            audit_factory=lambda record: {
                "kind": "modeling.challenger_backtest.create",
                "target_ref": record.id,
                "outcome": "succeeded",
                "detail": {
                    "experiment_id": artifact.experiment_id,
                    "artifact_id": artifact.id,
                    "sample_dataset_id": sample_dataset_id,
                    "source_dir": str(material_dir.resolve()),
                    "plan_path": CHALLENGER_BACKTEST_PLAN_JSON,
                },
            },
        )
        staged_materials.commit()
    except Exception:
        staged_materials.rollback()
        raise
    return {
        "task_id": task.id,
        "package_path": str((material_dir / CHALLENGER_BACKTEST_PLAN_JSON).resolve()),
        "markdown_path": str((material_dir / CHALLENGER_BACKTEST_PLAN_MD).resolve()),
    }


def _sample_dataset_path(settings, dataset_id: str) -> Path:
    dataset = DatasetRepository(settings.db_path).get_dataset(dataset_id)
    if dataset is None:
        raise ModelingError(f"sample dataset not found: {dataset_id}")
    path = settings.datasets_dir / dataset.source_path
    if not path.exists():
        raise ModelingError(f"sample dataset file does not exist: {path}")
    return path


def _artifact_base_dir(settings, experiment: Experiment) -> Path:
    return Path(settings.tasks_dir) / experiment.task_id / MODELING_ARTIFACTS_DIR_NAME


def _material_dir(settings, experiment: Experiment, artifact: ModelArtifact) -> Path:
    return Path(settings.tasks_dir) / experiment.task_id / HANDOFF_DIR_NAME / artifact.id


def _challenger_backtest_dir(settings, experiment: Experiment, artifact: ModelArtifact) -> Path:
    return Path(settings.tasks_dir) / experiment.task_id / CHALLENGER_BACKTEST_DIR_NAME / artifact.id


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _ensure_pmml_path(
    experiment_store,
    artifact: ModelArtifact,
    *,
    experiment: Experiment,
    sample_path: Path,
    artifact_base_dir: Path,
) -> Path:
    if artifact.pmml_path:
        path = _resolve_artifact_path(artifact.pmml_path, base_dir=artifact_base_dir)
        if not path.exists():
            raise ModelingError(f"PMML file does not exist: {path}")
        persist_model_meta(artifact_base_dir, artifact, config=experiment.config)
        return path

    out_path = artifact_base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        sample_path,
        out_path,
        base_dir=artifact_base_dir,
        target_col=experiment.config.target_col,
    )
    try:
        updated_artifact = replace(artifact, pmml_path=pmml_path.name)
        persist_model_meta(artifact_base_dir, updated_artifact, config=experiment.config)
        experiment_store.set_artifact_pmml_path(artifact.id, pmml_path.name)
    except Exception:
        pmml_path.unlink(missing_ok=True)
        try:
            persist_model_meta(artifact_base_dir, artifact, config=experiment.config)
        except Exception:
            pass
        raise
    return pmml_path


def _copy_calibration_payload(
    artifact: ModelArtifact,
    *,
    artifact_base_dir: Path,
    material_dir: Path,
) -> str | None:
    calibration = _calibration_metadata(artifact)
    if not calibration:
        return None
    source_name = calibration.get("path")
    if not source_name:
        return None
    source_path = _resolve_artifact_path(str(source_name), base_dir=artifact_base_dir)
    if not source_path.exists():
        raise ModelingError(f"calibration file does not exist: {source_path}")
    material_name = f"calibration{source_path.suffix or '.joblib'}"
    shutil.copy2(source_path, material_dir / material_name)
    return material_name


def _calibration_metadata(artifact: ModelArtifact) -> dict[str, Any] | None:
    calibration = (artifact.params or {}).get("calibration")
    return calibration if isinstance(calibration, dict) else None


def _source_task(settings, task_id: str):
    try:
        return TaskRepository(settings.db_path).get_task(task_id)
    except KeyError:
        return None


def _score_col(params: dict[str, Any]) -> str:
    return str(params.get("score_col") or "pred")


def _time_col(params: dict[str, Any]) -> str:
    return str(params.get("time_col") or "apply_month")


def _write_dictionary(path: Path, feature_list: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["特征名", "类别"])
        writer.writeheader()
        for feature in feature_list:
            writer.writerow({"特征名": feature, "类别": "建模特征"})


def _write_scoring_notebook(
    path: Path,
    *,
    artifact: ModelArtifact,
    experiment: Experiment,
    model_filename: str,
    calibration_filename: str | None,
) -> None:
    source = _scoring_notebook_source(
        artifact=artifact,
        experiment=experiment,
        model_filename=model_filename,
        calibration_filename=calibration_filename,
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, path)


def _challenger_backtest_payload(
    *,
    experiment: Experiment,
    artifact: ModelArtifact,
    sample_dataset_id: str,
    sample_material_name: str,
    model_material_name: str,
    selection_policy_decision: dict,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "modeling_challenger_backtest",
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "algorithm": artifact.algorithm,
        "recipe": experiment.recipe_id,
        "target_type": experiment.config.target_type,
        "dataset_id": experiment.config.dataset_id,
        "sample_dataset_id": sample_dataset_id,
        "target_col": experiment.config.target_col,
        "split_col": experiment.config.split_col,
        "time_col": _time_col(experiment.config.params),
        "score_col": _score_col(experiment.config.params),
        "features": list(artifact.feature_list),
        "feature_count": len(artifact.feature_list),
        "metrics": asdict(experiment.metrics) if experiment.metrics else {},
        "selection_policy_decision": selection_policy_decision,
        "materials": {
            "sample_path": sample_material_name,
            "native_model_path": model_material_name,
            "pmml_path": "model.pmml",
            "notebook_path": SCORING_NOTEBOOK_NAME,
            "dictionary_path": DICTIONARY_NAME,
        },
        "recommended_checks": [
            "compare selected model against current champion or prior production model",
            "run OOT/backtest by split and time bucket",
            "review PSI, KS/AUC drift, calibration, and reject/fairness slices",
            "record business override reason when selecting a non-recommended challenger",
        ],
    }


def _challenger_backtest_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    lines = [
        "# Challenger / Backtest 任务包",
        "",
        "## 模型",
        "",
        f"- 实验ID: `{_md_value(payload.get('experiment_id'))}`",
        f"- 产物ID: `{_md_value(payload.get('artifact_id'))}`",
        f"- 算法: `{_md_value(payload.get('algorithm'))}`",
        f"- 样本集: `{_md_value(payload.get('sample_dataset_id'))}`",
        f"- 目标列: `{_md_value(payload.get('target_col'))}`",
        f"- 分割列: `{_md_value(payload.get('split_col'))}`",
        f"- 时间列: `{_md_value(payload.get('time_col'))}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key in sorted(k for k in metrics if k.startswith(("test_", "oot_", "psi_"))):
        lines.append(f"| {_md_cell(key)} | {_md_cell(metrics.get(key))} |")
    if len(lines) >= 2 and lines[-1] == "| --- | ---: |":
        lines.append("| - | - |")
    lines.extend([
        "",
        "## 建议检查",
        "",
    ])
    for item in payload.get("recommended_checks") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 物料", ""])
    materials = payload.get("materials") if isinstance(payload.get("materials"), dict) else {}
    for key, value in materials.items():
        lines.append(f"- {key}: `{_md_value(value)}`")
    return "\n".join(lines) + "\n"


def _md_value(value: Any) -> str:
    return str(value if value is not None else "-").replace("`", "'")


def _md_cell(value: Any) -> str:
    return _md_value(value).replace("|", "\\|").replace("\n", " ")


def _scoring_notebook_source(
    *,
    artifact: ModelArtifact,
    experiment: Experiment,
    model_filename: str,
    calibration_filename: str | None,
) -> str:
    features_json = json.dumps(list(artifact.feature_list), ensure_ascii=False)
    params_json = json.dumps(artifact.params, ensure_ascii=False, default=str)
    calibration = _calibration_metadata(artifact) or {}
    calibration_filename_json = json.dumps(calibration_filename, ensure_ascii=False)
    calibration_method_json = json.dumps(calibration.get("method"), ensure_ascii=False)
    score_version_json = json.dumps(
        "calibrated" if calibration_filename else "raw",
        ensure_ascii=False,
    )
    pmml_includes_calibration_python = repr(
        bool(calibration.get("pmml_includes_calibration", False)),
    )
    target_json = json.dumps(experiment.config.target_col, ensure_ascii=False)
    split_json = json.dumps(experiment.config.split_col, ensure_ascii=False)
    time_json = json.dumps(_time_col(experiment.config.params), ensure_ascii=False)
    algorithm_json = json.dumps(artifact.algorithm, ensure_ascii=False)
    return "\n".join(
        [
            "import json",
            "from pathlib import Path",
            "",
            "import joblib",
            "import numpy as np",
            "import pandas as pd",
            "from marvis.feature.encode import woe_encode",
            "",
            f"RMC_FEATURES = json.loads({features_json!r})",
            f"RMC_MODEL_PARAMS = json.loads({params_json!r})",
            f"RMC_TARGET_COL = {target_json}",
            f"RMC_ALGORITHM = {algorithm_json}",
            f"RMC_SPLIT_COL = {split_json}",
            f"RMC_TIME_COL = {time_json}",
            "RMC_PMML_OUTPUT_FIELD = 'probability_1'",
            f"RMC_CALIBRATION_FILENAME = {calibration_filename_json}",
            f"RMC_CALIBRATION_METHOD = {calibration_method_json}",
            f"RMC_SCORE_VERSION = {score_version_json}",
            f"RMC_PMML_INCLUDES_CALIBRATION = {pmml_includes_calibration_python}",
            "RMC_SCORE_DECIMAL_PLACES = 6",
            "",
            "def _rmc_read_sample(path):",
            "    path = Path(path)",
            "    if path.suffix.lower() == '.parquet':",
            "        return pd.read_parquet(path)",
            "    if path.suffix.lower() == '.csv':",
            "        return pd.read_csv(path)",
            "    raise ValueError(f'unsupported sample format: {path.suffix}')",
            "",
            "RMC_SAMPLE_DF = _rmc_read_sample(RMC_SAMPLE_PATH)",
            f"_RMC_MODEL = joblib.load(Path({model_filename!r}))",
            "_RMC_CALIBRATION = joblib.load(Path(RMC_CALIBRATION_FILENAME)) if RMC_CALIBRATION_FILENAME else None",
            "",
            "def _rmc_apply_calibration(scores):",
            "    if _RMC_CALIBRATION is None:",
            "        return np.asarray(scores, dtype=float)",
            "    method = str(_RMC_CALIBRATION.get('method') or RMC_CALIBRATION_METHOD)",
            "    calibrator = _RMC_CALIBRATION['calibrator']",
            "    values = np.asarray(scores, dtype=float)",
            "    if method == 'sigmoid':",
            "        return calibrator.predict_proba(values.reshape(-1, 1))[:, 1]",
            "    if method == 'isotonic':",
            "        return calibrator.predict(values)",
            "    raise ValueError(f'unsupported calibration method: {method}')",
            "",
            "def RMC_SCORE_FN(dataframe):",
            "    if isinstance(_RMC_MODEL, dict) and 'model' in _RMC_MODEL and 'woe_maps' in _RMC_MODEL:",
            "        encoded = pd.DataFrame(index=dataframe.index)",
            "        for feature in RMC_FEATURES:",
            "            encoded[feature] = woe_encode(dataframe, feature, _RMC_MODEL['woe_maps'][feature]).to_numpy(dtype=float)",
            "        scores = _RMC_MODEL['model'].predict_proba(encoded)[:, 1]",
            "    else:",
            "        scores = _RMC_MODEL.predict_proba(dataframe[RMC_FEATURES])[:, 1]",
            "    return np.clip(_rmc_apply_calibration(scores), 0.0, 1.0)",
        ]
    )


__all__ = [
    "create_challenger_backtest_task",
    "handoff_to_validation",
    "mark_validated_from_validation_task",
]
