from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    compute_model_metrics,
    model_params,
    sample_weight_values,
    split_modeling_frame,
)


def train_catboost(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("catboost").default_params,
        **model_params(config.params),
        "random_seed": config.seed,
        "thread_count": 1,
        "allow_writing_files": False,
    }
    iterations = int(params.pop("iterations", params.pop("num_boost_round", 50)))
    model = CatBoostClassifier(**params, iterations=iterations)
    features = list(config.features)
    model.fit(
        train[features],
        train[config.target_col].to_numpy(dtype=int),
        sample_weight=sample_weight_values(train, config),
        eval_set=(test[features], test[config.target_col].to_numpy(dtype=int)),
        verbose=False,
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[features])[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_catboost_model(
        model,
        config,
        out_dir,
        artifact_params({**params, "iterations": iterations}, config),
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_catboost_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_catboost_model(
    model: CatBoostClassifier,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.pkl"
    joblib.dump(model, out_dir / model_path)
    return ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="catboost",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def _catboost_importance(
    model: CatBoostClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    values = np.asarray(model.get_feature_importance(), dtype=float)
    return tuple((feature, float(value)) for feature, value in zip(features, values, strict=True))


__all__ = ["train_catboost"]
