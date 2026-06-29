from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.artifact import persist_model_meta, write_artifact_file
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    compute_multiclass_model_metrics,
    split_modeling_frame,
)


def train_lgb_multiclass(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """MODELING §8.3 multiclass (credit risk grade / rating) recipe.

    Trains a LightGBM ``objective="multiclass"`` model where K is the number of
    distinct training-target classes. Classes use the sorted distinct training values
    so the column→class mapping is deterministic. predict returns an N×K probability
    matrix consumed by ``compute_multiclass_model_metrics`` for macro_auc/logloss/
    accuracy; binary KS/AUC and regression RMSE/MAE fields stay None."""
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    classes = _resolve_classes(train[config.target_col])
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    params = {
        **get_recipe("lgb_multiclass").default_params,
        **config.params,
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "seed": config.seed,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    num_boost_round = int(params.pop("num_boost_round", 50))
    dtrain = lgb.Dataset(
        train[list(config.features)],
        label=_encode_labels(train[config.target_col], class_to_index),
    )
    dvalid = lgb.Dataset(
        test[list(config.features)],
        label=_encode_labels(test[config.target_col], class_to_index),
        reference=dtrain,
    )
    callbacks = []
    if config.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(config.early_stopping_rounds, verbose=False))
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        callbacks=callbacks,
    )

    def predict(data):
        proba = np.asarray(model.predict(data[list(config.features)]), dtype=float)
        # A single-class degenerate model can collapse to a 1-D vector; restore N×K.
        if proba.ndim == 1:
            proba = proba.reshape(len(data), len(classes))
        return proba

    metrics, per_class = compute_multiclass_model_metrics(
        predict,
        train,
        test,
        oot,
        config,
        classes,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_lgb_multiclass_model(
        model,
        config,
        out_dir,
        {**params, "num_boost_round": num_boost_round},
        classes,
        per_class,
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _resolve_classes(target) -> tuple:
    """Sorted distinct training-target classes (deterministic column order)."""
    values = [value for value in target.tolist() if value is not None and not _is_nan(value)]
    return tuple(sorted(set(values)))


def _is_nan(value) -> bool:
    return isinstance(value, float) and value != value


def _encode_labels(target, class_to_index: dict) -> np.ndarray:
    return np.array([class_to_index[value] for value in target.tolist()], dtype=int)


def _jsonable_params(params: dict, classes: tuple) -> dict:
    """Serialise params so tuples become lists and class labels are JSON-safe."""
    cleaned: dict = {}
    for key, value in params.items():
        cleaned[str(key)] = list(value) if isinstance(value, tuple) else value
    cleaned["classes"] = [_jsonable_scalar(cls) for cls in classes]
    return cleaned


def _jsonable_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _save_lgb_multiclass_model(
    model: lgb.Booster,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
    classes: tuple,
    per_class: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.txt"
    write_artifact_file(out_dir, model_path, model.save_model)
    stored_params = _jsonable_params(params, classes)
    stored_params["per_class"] = per_class
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="lgb_multiclass",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=stored_params,
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _lgb_importance(
    model: lgb.Booster,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.feature_importance(importance_type="gain")
    return tuple((feature, float(value)) for feature, value in zip(features, gains, strict=True))


__all__ = ["train_lgb_multiclass"]
