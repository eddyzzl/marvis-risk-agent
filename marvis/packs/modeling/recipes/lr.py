from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import compute_model_metrics, split_modeling_frame


def train_lr(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    params = {
        **get_recipe("lr").default_params,
        **config.params,
        "random_state": config.seed,
    }
    model = LogisticRegression(**params)
    model.fit(train[list(config.features)], train[config.target_col].to_numpy(dtype=int))
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[list(config.features)])[:, 1],
        train,
        test,
        oot,
        config,
    )
    artifact = _save_lr_model(model, config, out_dir, params)
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lr_importance(model, config.features),
        experiment_id="",
    )


def _save_lr_model(
    model: LogisticRegression,
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
        algorithm="lr",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
    )


def _lr_importance(
    model: LogisticRegression,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    coefficients = np.abs(model.coef_[0])
    pairs = sorted(
        zip(features, coefficients, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return tuple((feature, float(value)) for feature, value in pairs)


__all__ = ["train_lr"]
