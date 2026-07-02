from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    compute_model_metrics,
    model_params,
    sample_weight_values,
    split_modeling_frame,
)


def train_lr(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """LR recipe. Unlike scorecard (WOE-encoded), raw LR needs numeric, imputed,
    standardized inputs -- without impute, missing data crashes sklearn's fit();
    without standardization, the L2 penalty (default C=1.0) is dominated by
    whichever feature happens to have the largest raw scale (e.g. an amount
    column vs. a 0-1 ratio), distorting the fitted coefficients' relative
    importance (PREP-6/SEL-3). Pipeline mirrors mlp.py's impute->scale->estimator
    shape so both non-WOE linear/DNN recipes share the same deterministic,
    artifact-replayable preprocessing contract. random_state is pinned to
    config.seed for reproducibility."""
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("lr").default_params,
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
        train[config.target_col].to_numpy(dtype=int),
        lr__sample_weight=sample_weight_values(train, config),
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[features])[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_lr_model(model, config, out_dir, artifact_params(params, config))
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lr_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_lr_model(
    model: Pipeline,
    config: TrainConfig,
    out_dir: Path,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="lr",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm("lr"),
        points_direction=points_direction_for_algorithm("lr"),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _lr_importance(
    model: Pipeline,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    coefficients = np.abs(model.named_steps["lr"].coef_[0])
    pairs = sorted(
        zip(features, coefficients, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return tuple((feature, float(value)) for feature, value in pairs)


__all__ = ["train_lr"]
