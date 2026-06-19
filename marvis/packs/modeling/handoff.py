from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import nbformat

from marvis.db import DatasetRepository, TaskRepository
from marvis.domain import TASK_TYPE_VALIDATION, TaskCreate, TaskStatus
from marvis.packs.modeling.artifact import export_pmml
from marvis.packs.modeling.contracts import Experiment, ModelArtifact
from marvis.packs.modeling.errors import ModelingError


HANDOFF_DIR_NAME = "validation_handoff"
MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"
SCORING_NOTEBOOK_NAME = "scoring_notebook.ipynb"
DICTIONARY_NAME = "dictionary.csv"


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
    if artifact.algorithm != "lr":
        raise ModelingError(
            f"validation handoff currently requires lr PMML export, got: {artifact.algorithm}"
        )

    sample_path = _sample_dataset_path(settings, sample_dataset_id)
    artifact_base_dir = _artifact_base_dir(settings, experiment)
    model_path = _resolve_artifact_path(artifact.model_path, base_dir=artifact_base_dir)
    if not model_path.exists():
        raise ModelingError(f"model file does not exist: {model_path}")
    pmml_path = _ensure_pmml_path(
        experiment_store,
        artifact,
        sample_path=sample_path,
        artifact_base_dir=artifact_base_dir,
    )

    material_dir = _material_dir(settings, experiment, artifact)
    material_dir.mkdir(parents=True, exist_ok=True)
    sample_material_name = f"sample{sample_path.suffix or '.parquet'}"
    model_material_name = f"model{model_path.suffix or '.joblib'}"
    shutil.copy2(sample_path, material_dir / sample_material_name)
    shutil.copy2(pmml_path, material_dir / "model.pmml")
    shutil.copy2(model_path, material_dir / model_material_name)
    _write_dictionary(material_dir / DICTIONARY_NAME, artifact.feature_list)
    _write_scoring_notebook(
        material_dir / SCORING_NOTEBOOK_NAME,
        artifact=artifact,
        experiment=experiment,
        model_filename=model_material_name,
    )

    source_task = _source_task(settings, experiment.task_id)
    validation_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
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
    )
    experiment_store.set_status(artifact.experiment_id, "handed_off")
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


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _ensure_pmml_path(
    experiment_store,
    artifact: ModelArtifact,
    *,
    sample_path: Path,
    artifact_base_dir: Path,
) -> Path:
    if artifact.pmml_path:
        path = _resolve_artifact_path(artifact.pmml_path, base_dir=artifact_base_dir)
        if not path.exists():
            raise ModelingError(f"PMML file does not exist: {path}")
        return path

    out_path = artifact_base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        sample_path,
        out_path,
        base_dir=artifact_base_dir,
    )
    experiment_store.set_artifact_pmml_path(artifact.id, pmml_path.name)
    return pmml_path


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
) -> None:
    source = _scoring_notebook_source(
        artifact=artifact,
        experiment=experiment,
        model_filename=model_filename,
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, path)


def _scoring_notebook_source(
    *,
    artifact: ModelArtifact,
    experiment: Experiment,
    model_filename: str,
) -> str:
    features_json = json.dumps(list(artifact.feature_list), ensure_ascii=False)
    params_json = json.dumps(artifact.params, ensure_ascii=False, default=str)
    target_json = json.dumps(experiment.config.target_col, ensure_ascii=False)
    split_json = json.dumps(experiment.config.split_col, ensure_ascii=False)
    time_json = json.dumps(_time_col(experiment.config.params), ensure_ascii=False)
    algorithm_json = json.dumps(artifact.algorithm, ensure_ascii=False)
    model_json = json.dumps(model_filename, ensure_ascii=False)
    return "\n".join(
        [
            "import json",
            "from pathlib import Path",
            "",
            "import joblib",
            "import pandas as pd",
            "",
            f"RMC_FEATURES = json.loads({features_json!r})",
            f"RMC_MODEL_PARAMS = json.loads({params_json!r})",
            f"RMC_TARGET_COL = {target_json}",
            f"RMC_ALGORITHM = {algorithm_json}",
            f"RMC_SPLIT_COL = {split_json}",
            f"RMC_TIME_COL = {time_json}",
            "RMC_PMML_OUTPUT_FIELD = 'probability_1'",
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
            f"_RMC_MODEL = joblib.load(Path({model_json!r}))",
            "",
            "def RMC_SCORE_FN(dataframe):",
            "    return _RMC_MODEL.predict_proba(dataframe[RMC_FEATURES])[:, 1]",
        ]
    )


__all__ = ["handoff_to_validation", "mark_validated_from_validation_task"]
