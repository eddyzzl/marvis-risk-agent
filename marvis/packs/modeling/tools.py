from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.packs.modeling.artifact import export_pmml
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.handoff import handoff_to_validation
from marvis.packs.modeling.readiness import check_data_quality, modeling_readiness
from marvis.packs.modeling.prepare import prepare_modeling_frame
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.select import select_features
from marvis.packs.modeling.errors import ModelingError
from marvis.settings import build_settings


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"


def tool_check_data_quality(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    issues = check_data_quality(
        runtime.backend,
        dataset,
        runtime.registry.resolve_path(dataset.id),
        target_col=_optional_str(inputs.get("target_col")),
    )
    return {"issues": [_jsonable(issue) for issue in issues]}


def tool_modeling_readiness(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return _jsonable(
        modeling_readiness(
            runtime.backend,
            dataset,
            runtime.registry.resolve_path(dataset.id),
            target_col=str(inputs["target_col"]),
            split_col=_optional_str(inputs.get("split_col")),
        )
    )


def tool_prepare_modeling_frame(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        str(inputs["dataset_id"]),
        target_col=str(inputs["target_col"]),
        feature_cols=[str(item) for item in inputs["feature_cols"]],
        split_col=_optional_str(inputs.get("split_col")),
        split_config=inputs.get("split_config") or {},
        seed=int(inputs.get("seed") if inputs.get("seed") is not None else ctx.seed or 0),
    )
    split_col = _optional_str(inputs.get("split_col")) or "split"
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(result.id), columns=[split_col])
    counts = {
        str(key): int(value)
        for key, value in frame[split_col].value_counts().sort_index().items()
    }
    return {"result_dataset_id": result.id, "split_counts": counts}


def tool_select_features(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    result = select_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=[str(item) for item in inputs["features"]],
        target_col=str(inputs["target_col"]),
        iv_min=float(inputs.get("iv_min", 0.02)),
        corr_max=float(inputs.get("corr_max", 0.8)),
        vif_max=float(inputs.get("vif_max", 10.0)),
        top_k=_optional_int(inputs.get("top_k")),
        seed=int(ctx.seed or 0),
    )
    return {
        "selected": list(result.selected),
        "dropped": [[feature, reason] for feature, reason in result.dropped],
        "scores": _jsonable(result.scores),
    }


def tool_train_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipe = str(inputs["recipe"])
    config = TrainConfig(
        dataset_id=dataset.id,
        features=tuple(str(item) for item in inputs["features"]),
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        params=dict(inputs.get("params") or {}),
        seed=int(inputs["seed"]),
        early_stopping_rounds=_optional_int(inputs.get("early_stopping_rounds")),
        recipe_id=recipe,
    )
    if inputs.get("scenario"):
        config = apply_scenario(config, str(inputs["scenario"]))
        recipe = config.recipe_id or recipe

    experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
    try:
        result = _train_recipe(
            recipe,
            runtime.backend,
            runtime.registry.resolve_path(dataset.id),
            config,
            out_dir=_artifact_base_dir(runtime.settings, ctx.task_id),
        )
        runtime.experiments.attach_result(experiment_id, result)
    except Exception:
        runtime.experiments.set_status(experiment_id, "failed")
        raise

    experiment = runtime.experiments.get(experiment_id)
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact after training: {experiment_id}")
    artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {experiment.artifact_id}")
    return {
        "experiment_id": experiment_id,
        "artifact_id": artifact.id,
        "metrics": _jsonable(experiment.metrics),
        "feature_importance": _jsonable(result.feature_importance),
    }


def tool_compare_experiments(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    return _jsonable(runtime.experiments.compare([str(item) for item in inputs["experiment_ids"]]))


def tool_export_pmml(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    pmml_path = _pmml_path(runtime, artifact)
    return {"pmml_path": str(pmml_path)}


def tool_handoff_to_validation(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    validation_task_id = handoff_to_validation(
        runtime.experiments,
        artifact,
        sample_dataset_id=str(inputs["sample_dataset_id"]),
        settings=runtime.settings,
    )
    return {"validation_task_id": validation_task_id}


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.experiments = ExperimentStore(self.settings.db_path)
        self.modeling_repo = ModelingRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _train_recipe(
    recipe: str,
    backend,
    dataset_path: Path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    if recipe == "lgb":
        return train_lgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "xgb":
        return train_xgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lr":
        return train_lr(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "scorecard":
        return train_scorecard(backend, dataset_path, config, out_dir=out_dir)
    raise ModelingError(f"unsupported modeling recipe: {recipe}")


def _artifact(runtime: _Runtime, artifact_id: str) -> ModelArtifact:
    artifact = runtime.modeling_repo.get_model_artifact(artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {artifact_id}")
    return artifact


def _pmml_path(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    if artifact.pmml_path:
        existing = _resolve_artifact_path(artifact.pmml_path, base_dir=base_dir)
        if existing.exists():
            return existing
    dataset = runtime.registry.get(experiment.config.dataset_id)
    out_path = base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        runtime.registry.resolve_path(dataset.id),
        out_path,
        base_dir=base_dir,
    )
    runtime.experiments.set_artifact_pmml_path(artifact.id, pmml_path.name)
    return pmml_path


def _artifact_base_dir(settings, task_id: str) -> Path:
    return Path(settings.tasks_dir) / task_id / MODELING_ARTIFACTS_DIR_NAME


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _jsonable(value: Any):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "tool_check_data_quality",
    "tool_compare_experiments",
    "tool_export_pmml",
    "tool_handoff_to_validation",
    "tool_modeling_readiness",
    "tool_prepare_modeling_frame",
    "tool_select_features",
    "tool_train_model",
]
