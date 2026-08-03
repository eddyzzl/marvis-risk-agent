from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb

from marvis.data.labels import resolve_modeling_splits
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
    compute_regression_metrics,
    model_params,
    pop_boost_rounds,
    sample_weight_values,
    split_modeling_frame,
    training_frame_columns,
)


def train_lgb_regressor(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(
        dataset_path, columns=training_frame_columns(backend, dataset_path, config)
    )
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    supplied_params = model_params(config.params)
    params = {
        **get_recipe("lgb_regressor").default_params,
        **supplied_params,
        "seed": config.seed,
        # TUNE-6: sourced from defaults.py -- see lgb.py's train_lgb for the
        # single-source rationale shared across every tree recipe's direct-train path.
        "num_threads": DEFAULT_TRAIN_NUM_THREADS,
        "deterministic": True,
    }
    if "force_col_wise" not in supplied_params and "force_row_wise" not in supplied_params:
        params["force_row_wise"] = True
    elif bool(params.get("force_col_wise")):
        # Tuning deliberately emits force_col_wise. Preserve that selected
        # mode and discard a conflicting row-wise flag instead of making
        # LightGBM abort before the final fit.
        params.pop("force_row_wise", None)
    elif bool(params.get("force_row_wise")):
        params.pop("force_col_wise", None)
    num_boost_round = pop_boost_rounds(params, default=20)
    fit_train = train
    if config.early_stopping_rounds:
        # Keep the formal test split out of round-count selection.  Metrics below
        # still evaluate the untouched test set exactly once.
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
    else:
        valid = test
    dtrain = lgb.Dataset(
        fit_train[list(config.features)],
        label=fit_train[config.target_col].to_numpy(dtype=float),
        weight=sample_weight_values(fit_train, config),
    )
    dvalid = lgb.Dataset(
        valid[list(config.features)],
        label=valid[config.target_col].to_numpy(dtype=float),
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
    metrics = compute_regression_metrics(
        lambda data: model.predict(data[list(config.features)]),
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_lgb_regressor_model(
        model,
        config,
        out_dir,
        {**params, "num_boost_round": num_boost_round},
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_lgb_regressor_model(
    model: lgb.Booster,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.txt"
    write_artifact_file(out_dir, model_path, model.save_model)
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="lgb_regressor",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=artifact_params(dict(params), config),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm("lgb_regressor"),
        points_direction=points_direction_for_algorithm("lgb_regressor"),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _lgb_importance(
    model: lgb.Booster,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.feature_importance(importance_type="gain")
    return tuple((feature, float(value)) for feature, value in zip(features, gains, strict=True))


__all__ = ["train_lgb_regressor"]
