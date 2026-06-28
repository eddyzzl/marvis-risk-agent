from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import xgboost as xgb

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import compute_model_metrics, split_modeling_frame


def train_xgb(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("xgb").default_params,
        **config.params,
        "random_state": config.seed,
        "n_jobs": 1,
    }
    num_boost_round = int(params.pop("num_boost_round", 20))
    if config.early_stopping_rounds:
        params["early_stopping_rounds"] = int(config.early_stopping_rounds)
    model = xgb.XGBClassifier(
        **params,
        n_estimators=num_boost_round,
    )
    model.fit(
        train[list(config.features)],
        train[config.target_col].to_numpy(dtype=int),
        eval_set=[(test[list(config.features)], test[config.target_col].to_numpy(dtype=int))],
        verbose=False,
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[list(config.features)])[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_xgb_model(model, config, out_dir, {**params, "num_boost_round": num_boost_round})
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_xgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_xgb_model(
    model: xgb.XGBClassifier,
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
        algorithm="xgb",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def _xgb_importance(
    model: xgb.XGBClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.get_booster().get_score(importance_type="gain")
    return tuple((feature, float(gains.get(feature, 0.0))) for feature in features)


__all__ = ["train_xgb"]
