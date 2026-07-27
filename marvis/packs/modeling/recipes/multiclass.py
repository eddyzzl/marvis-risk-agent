from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marvis.packs.modeling.contracts import TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_TRAIN_NUM_THREADS
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
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
    save_joblib_nonbinary_artifact,
    sklearn_linear_importance,
    xgb_importance,
)


def train_xgb_multiclass(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    classes = resolve_multiclass_classes(train[config.target_col])
    params = {
        **get_recipe("xgb_multiclass").default_params,
        **model_params(config.params),
        "objective": "multi:softprob",
        "num_class": len(classes),
        "random_state": config.seed,
        "n_jobs": DEFAULT_TRAIN_NUM_THREADS,
    }
    num_boost_round = pop_boost_rounds(params, default=50)
    fit_train = train
    if config.early_stopping_rounds:
        params["early_stopping_rounds"] = int(config.early_stopping_rounds)
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
    else:
        valid = test
    features = list(config.features)
    model = xgb.XGBClassifier(**params, n_estimators=num_boost_round)
    valid_weight = sample_weight_values(valid, config)
    model.fit(
        fit_train[features],
        encode_multiclass_target(fit_train[config.target_col], classes),
        sample_weight=sample_weight_values(fit_train, config),
        eval_set=[
            (
                valid[features],
                encode_multiclass_target(valid[config.target_col], classes),
            )
        ],
        sample_weight_eval_set=[valid_weight] if valid_weight is not None else None,
        verbose=False,
    )

    def predict(data):
        return np.asarray(model.predict_proba(data[features]), dtype=float)

    metrics, per_class = compute_multiclass_model_metrics(
        predict,
        train,
        test,
        oot,
        config,
        classes,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="xgb_multiclass",
        config=config,
        out_dir=out_dir,
        params={**params, "num_boost_round": num_boost_round},
        classes=classes,
        per_class=per_class,
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=xgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def train_lr_multiclass(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    classes = resolve_multiclass_classes(train[config.target_col])
    params = {
        **get_recipe("lr_multiclass").default_params,
        **model_params(config.params),
        "random_state": config.seed,
    }
    features = list(config.features)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(**params)),
    ])
    model.fit(
        train[features],
        encode_multiclass_target(train[config.target_col], classes),
        lr__sample_weight=sample_weight_values(train, config),
    )

    def predict(data):
        return np.asarray(model.predict_proba(data[features]), dtype=float)

    metrics, per_class = compute_multiclass_model_metrics(
        predict,
        train,
        test,
        oot,
        config,
        classes,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="lr_multiclass",
        config=config,
        out_dir=out_dir,
        params=params,
        classes=classes,
        per_class=per_class,
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=sklearn_linear_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def train_mlp_multiclass(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    classes = resolve_multiclass_classes(train[config.target_col])
    params = {
        **get_recipe("mlp_multiclass").default_params,
        **model_params(config.params),
        "random_state": config.seed,
    }
    params["hidden_layer_sizes"] = tuple(
        params.get("hidden_layer_sizes") or (32, 16)
    )
    features = list(config.features)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(**params)),
    ])
    model.fit(
        train[features],
        encode_multiclass_target(train[config.target_col], classes),
        mlp__sample_weight=sample_weight_values(train, config),
    )

    def predict(data):
        return np.asarray(model.predict_proba(data[features]), dtype=float)

    metrics, per_class = compute_multiclass_model_metrics(
        predict,
        train,
        test,
        oot,
        config,
        classes,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="mlp_multiclass",
        config=config,
        out_dir=out_dir,
        params=params,
        classes=classes,
        per_class=per_class,
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=(),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _resolved_splits(backend, dataset_path, config: TrainConfig):
    frame = backend.read_frame(
        dataset_path,
        columns=training_frame_columns(backend, dataset_path, config),
    )
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_multiclass_splits(
        train,
        test,
        oot,
        target_col=config.target_col,
        drop_nan_labels=config.drop_nan_labels,
    )
    return train, test, oot, oot_has_labels, audit


__all__ = [
    "train_lr_multiclass",
    "train_mlp_multiclass",
    "train_xgb_multiclass",
]
