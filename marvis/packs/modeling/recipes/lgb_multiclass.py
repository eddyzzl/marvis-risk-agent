from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from marvis.packs.modeling.artifact import (
    persist_model_meta,
    points_direction_for_algorithm,
    score_direction_for_algorithm,
    write_artifact_file,
)
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_TRAIN_NUM_THREADS
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    carve_early_stop_fold_for_config,
    compute_multiclass_model_metrics,
    model_params,
    pop_boost_rounds,
    sample_weight_values,
    split_modeling_frame,
    training_frame_columns,
)
from marvis.packs.modeling.recipes.nonbinary_common import (
    encode_multiclass_target,
    resolve_multiclass_classes,
    resolve_multiclass_splits,
    strict_json_classes,
)


def train_lgb_multiclass(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """MODELING §8.3 multiclass (credit risk grade / rating) recipe.

    Trains a LightGBM ``objective="multiclass"`` model where K is the number of
    distinct training-target classes. Classes use the sorted distinct training values
    so the column→class mapping is deterministic. predict returns an N×K probability
    matrix consumed by ``compute_multiclass_model_metrics`` for macro_auc/logloss/
    accuracy; binary KS/AUC and regression RMSE/MAE fields stay None."""
    frame = backend.read_frame(
        dataset_path, columns=training_frame_columns(backend, dataset_path, config)
    )
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_multiclass_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    classes = resolve_multiclass_classes(train[config.target_col])

    params = {
        **get_recipe("lgb_multiclass").default_params,
        **model_params(config.params),
        "objective": "multiclass",
        "num_class": len(classes),
        "metric": "multi_logloss",
        "seed": config.seed,
        # TUNE-6: sourced from defaults.py -- see lgb.py's train_lgb for the
        # single-source rationale shared across every tree recipe's direct-train path.
        "num_threads": DEFAULT_TRAIN_NUM_THREADS,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    num_boost_round = pop_boost_rounds(params, default=50)
    fit_train = train
    if config.early_stopping_rounds:
        # Keep the formal test split out of round-count selection.  Metrics below
        # still evaluate the untouched test set exactly once.
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
    else:
        valid = test
    dtrain = lgb.Dataset(
        fit_train[list(config.features)],
        label=encode_multiclass_target(fit_train[config.target_col], classes),
        weight=sample_weight_values(fit_train, config),
    )
    dvalid = lgb.Dataset(
        valid[list(config.features)],
        label=encode_multiclass_target(valid[config.target_col], classes),
        weight=sample_weight_values(valid, config),
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


def _jsonable_params(params: dict, classes: tuple) -> dict:
    """Serialise params so tuples become lists and class labels are JSON-safe."""
    cleaned: dict = {}
    for key, value in params.items():
        cleaned[str(key)] = list(value) if isinstance(value, tuple) else value
    cleaned["classes"] = strict_json_classes(classes)
    return cleaned


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
    stored_params = artifact_params(_jsonable_params(params, classes), config)
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
        score_direction=score_direction_for_algorithm("lgb_multiclass"),
        points_direction=points_direction_for_algorithm("lgb_multiclass"),
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
