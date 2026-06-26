from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

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
        "seed": config.seed,
        "nthread": 1,
    }
    num_boost_round = int(params.pop("num_boost_round", 20))
    dtrain = _dmatrix(train, config)
    dtest = _dmatrix(test, config)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtest, "test")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )
    metrics = compute_model_metrics(
        lambda data: model.predict(_dmatrix(data, config)),
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


def _dmatrix(frame, config: TrainConfig) -> xgb.DMatrix:
    # Float labels: train/test are label-resolved upstream; OOT may be scoring-only, where
    # the label is unused by predict() and must never be coerced into a class.
    return xgb.DMatrix(
        frame[list(config.features)],
        label=frame[config.target_col].to_numpy(dtype=float),
        feature_names=list(config.features),
    )


def _save_xgb_model(
    model: xgb.Booster,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.json"
    model.save_model(out_dir / model_path)
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
    model: xgb.Booster,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.get_score(importance_type="gain")
    return tuple((feature, float(gains.get(feature, 0.0))) for feature in features)


__all__ = ["train_xgb"]
