from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.artifact import (
    persist_model_meta,
    points_direction_for_algorithm,
    score_direction_for_algorithm,
    write_artifact_file,
)
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    carve_early_stop_fold_for_config,
    cat_feature_indices,
    compute_model_metrics,
    model_params,
    pop_boost_rounds,
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
    features = list(config.features)
    cat_features = cat_feature_indices(train, features, params.pop("cat_features", None))
    iterations = pop_boost_rounds(
        params, default=50, primary="iterations", aliases=("num_boost_round", "n_estimators")
    )
    od_wait = params.pop("od_wait", None)
    fit_train = train
    if config.early_stopping_rounds:
        # SEL-4/TUNE-3: early stopping watches a fold carved from train, not test --
        # test stays a one-time, unbiased comparison set instead of also picking the
        # round count.
        params.setdefault("od_type", "Iter")
        od_wait = int(config.early_stopping_rounds)
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
    else:
        valid = test
    model = CatBoostClassifier(**params, iterations=iterations, od_wait=od_wait, cat_features=cat_features or None)
    model.fit(
        fit_train[features],
        fit_train[config.target_col].to_numpy(dtype=int),
        sample_weight=sample_weight_values(fit_train, config),
        eval_set=(valid[features], valid[config.target_col].to_numpy(dtype=int)),
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
    resolved_iterations = int(model.get_best_iteration() or model.tree_count_ - 1) + 1 if config.early_stopping_rounds else iterations
    artifact = _save_catboost_model(
        model,
        config,
        out_dir,
        artifact_params(
            {
                **params,
                "iterations": iterations,
                "best_iteration": resolved_iterations,
                "cat_features": [features[index] for index in cat_features],
            },
            config,
        ),
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
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="catboost",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm("catboost"),
        points_direction=points_direction_for_algorithm("catboost"),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _catboost_importance(
    model: CatBoostClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    values = np.asarray(model.get_feature_importance(), dtype=float)
    return tuple((feature, float(value)) for feature, value in zip(features, values, strict=True))


__all__ = ["train_catboost"]
