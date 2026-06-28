from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb

from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import compute_model_metrics, split_modeling_frame


def train_lgb(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    params = {
        **get_recipe("lgb").default_params,
        **config.params,
        "random_state": config.seed,
        "n_jobs": 1,
        "deterministic": True,
    }
    num_boost_round = int(params.pop("num_boost_round", 20))
    callbacks = []
    if config.early_stopping_rounds:
        callbacks.append(lgb.early_stopping(config.early_stopping_rounds, verbose=False))
    model = lgb.LGBMClassifier(
        **params,
        n_estimators=num_boost_round,
    )
    model.fit(
        train[list(config.features)],
        train[config.target_col],
        eval_set=[(test[list(config.features)], test[config.target_col])],
        callbacks=callbacks,
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[list(config.features)])[:, 1],
        train,
        test,
        oot,
        config,
    )
    artifact = _save_lgb_model(model, config, out_dir, {**params, "num_boost_round": num_boost_round})
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lgb_importance(model, config.features),
        experiment_id="",
    )


def _save_lgb_model(
    model: lgb.LGBMClassifier,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    joblib.dump(model, out_dir / model_path)
    return ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="lgb",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def _lgb_importance(
    model: lgb.LGBMClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.booster_.feature_importance(importance_type="gain")
    return tuple((feature, float(value)) for feature, value in zip(features, gains, strict=True))


__all__ = ["train_lgb"]
