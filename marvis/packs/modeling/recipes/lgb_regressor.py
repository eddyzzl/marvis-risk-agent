from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import compute_regression_metrics, split_modeling_frame


def train_lgb_regressor(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("lgb_regressor").default_params,
        **config.params,
        "seed": config.seed,
        "num_threads": 1,
        "deterministic": True,
    }
    num_boost_round = int(params.pop("num_boost_round", 20))
    dtrain = lgb.Dataset(
        train[list(config.features)],
        label=train[config.target_col].to_numpy(dtype=float),
    )
    dvalid = lgb.Dataset(
        test[list(config.features)],
        label=test[config.target_col].to_numpy(dtype=float),
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
    model.save_model(out_dir / model_path)
    return ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="lgb_regressor",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def _lgb_importance(
    model: lgb.Booster,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.feature_importance(importance_type="gain")
    return tuple((feature, float(value)) for feature, value in zip(features, gains, strict=True))


__all__ = ["train_lgb_regressor"]
