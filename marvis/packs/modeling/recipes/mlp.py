from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
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


def train_mlp(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    """DNN (sklearn MLP) recipe. MLP needs numeric, imputed, scaled inputs, so the model is
    a deterministic impute→scale→MLP pipeline that consumes the same raw frame as the other
    recipes. random_state is pinned to config.seed for reproducibility."""
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("mlp").default_params,
        **model_params(config.params),
        "random_state": config.seed,
    }
    params["hidden_layer_sizes"] = tuple(params.get("hidden_layer_sizes") or (32, 16))
    features = list(config.features)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(**params)),
    ])
    model.fit(
        train[features],
        train[config.target_col].to_numpy(dtype=int),
        mlp__sample_weight=sample_weight_values(train, config),
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[features])[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_mlp_model(model, config, out_dir, artifact_params(params, config))
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=(),  # an MLP has no native per-feature importance
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_mlp_model(model: Pipeline, config: TrainConfig, out_dir: Path, params: dict) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="mlp",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params={key: (list(value) if isinstance(value, tuple) else value) for key, value in params.items()},
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm("mlp"),
        points_direction=points_direction_for_algorithm("mlp"),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


__all__ = ["train_mlp"]
