from __future__ import annotations

from pathlib import Path

import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.contracts import TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_TRAIN_NUM_THREADS
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    carve_early_stop_fold_for_config,
    compute_regression_metrics,
    model_params,
    pop_boost_rounds,
    sample_weight_values,
    split_modeling_frame,
    training_frame_columns,
)
from marvis.packs.modeling.recipes.nonbinary_common import (
    save_joblib_nonbinary_artifact,
    sklearn_linear_importance,
    xgb_importance,
)


def train_xgb_regressor(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    params = {
        **get_recipe("xgb_regressor").default_params,
        **model_params(config.params),
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
    model = xgb.XGBRegressor(**params, n_estimators=num_boost_round)
    model.fit(
        fit_train[features],
        fit_train[config.target_col].to_numpy(dtype=float),
        sample_weight=sample_weight_values(fit_train, config),
        eval_set=[(valid[features], valid[config.target_col].to_numpy(dtype=float))],
        sample_weight_eval_set=(
            [sample_weight_values(valid, config)]
            if sample_weight_values(valid, config) is not None
            else None
        ),
        verbose=False,
    )
    metrics = compute_regression_metrics(
        lambda data: model.predict(data[features]),
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="xgb_regressor",
        config=config,
        out_dir=out_dir,
        params={**params, "num_boost_round": num_boost_round},
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=xgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def train_lr_regressor(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    params = {
        **get_recipe("lr_regressor").default_params,
        **model_params(config.params),
        "random_state": config.seed,
    }
    # Ridge has no random_state; keep the seed in artifact metadata, not in the
    # estimator constructor.
    params.pop("random_state", None)
    features = list(config.features)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("regressor", Ridge(**params)),
    ])
    model.fit(
        train[features],
        train[config.target_col].to_numpy(dtype=float),
        regressor__sample_weight=sample_weight_values(train, config),
    )
    metrics = compute_regression_metrics(
        lambda data: model.predict(data[features]),
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="lr_regressor",
        config=config,
        out_dir=out_dir,
        params=params,
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=sklearn_linear_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def train_mlp_regressor(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    train, test, oot, oot_has_labels, audit = _resolved_splits(
        backend, dataset_path, config
    )
    params = {
        **get_recipe("mlp_regressor").default_params,
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
        ("mlp", MLPRegressor(**params)),
    ])
    model.fit(
        train[features],
        train[config.target_col].to_numpy(dtype=float),
        mlp__sample_weight=sample_weight_values(train, config),
    )
    metrics = compute_regression_metrics(
        lambda data: model.predict(data[features]),
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = save_joblib_nonbinary_artifact(
        model,
        algorithm="mlp_regressor",
        config=config,
        out_dir=out_dir,
        params=params,
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
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train,
        test,
        oot,
        target_col=config.target_col,
        drop_nan_labels=config.drop_nan_labels,
    )
    return train, test, oot, oot_has_labels, audit


__all__ = [
    "train_lr_regressor",
    "train_mlp_regressor",
    "train_xgb_regressor",
]
