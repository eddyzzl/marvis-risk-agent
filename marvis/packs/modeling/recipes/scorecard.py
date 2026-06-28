from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from marvis.data.labels import resolve_modeling_splits
from marvis.feature.binning import chimerge_edges
from marvis.feature.encode import woe_encode
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    compute_model_metrics,
    sample_weight_values,
    split_modeling_frame,
)


def train_scorecard(
    backend,
    dataset_path,
    config: TrainConfig,
    *,
    out_dir: Path,
    base_score: int = 600,
    pdo: int = 50,
    base_odds: float = 50,
) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    max_bins = int(config.params.get("scorecard_max_bins", 6))
    woe_maps = _fit_woe_maps(train, config, max_bins=max_bins)
    train_woe = _encode_with_woe(train, config, woe_maps)
    lr_params = {**_lr_params(config), "random_state": config.seed}
    model = LogisticRegression(**lr_params)
    model.fit(
        train_woe,
        train[config.target_col].to_numpy(dtype=int),
        sample_weight=sample_weight_values(train, config),
    )

    metrics = compute_model_metrics(
        lambda data: model.predict_proba(_encode_with_woe(data, config, woe_maps))[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    artifact = _save_scorecard_model(
        model,
        config,
        out_dir,
        woe_maps,
        params=artifact_params({
            **lr_params,
            "base_score": base_score,
            "pdo": pdo,
            "base_odds": base_odds,
            "factor": float(factor),
            "offset": float(offset),
            "scorecard_max_bins": max_bins,
        }, config),
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lr_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _fit_woe_maps(
    train: pd.DataFrame,
    config: TrainConfig,
    *,
    max_bins: int,
) -> dict:
    target = train[config.target_col].to_numpy(dtype=int)
    maps = {}
    for feature in config.features:
        values = train[feature].to_numpy(dtype=float)
        edges = chimerge_edges(values, target, max_bins=max_bins)
        binning = compute_woe_iv(values, target, edges, feature=feature)
        maps[feature] = woe_result_from_binning(binning)
    return maps


def _encode_with_woe(
    frame: pd.DataFrame,
    config: TrainConfig,
    woe_maps: dict,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for feature in config.features:
        encoded[feature] = woe_encode(frame, feature, woe_maps[feature]).to_numpy(dtype=float)
    return encoded


def _lr_params(config: TrainConfig) -> dict:
    allowed = {
        "C",
        "class_weight",
        "dual",
        "fit_intercept",
        "intercept_scaling",
        "l1_ratio",
        "max_iter",
        "multi_class",
        "n_jobs",
        "penalty",
        "solver",
        "tol",
        "verbose",
        "warm_start",
    }
    return {
        **get_recipe("scorecard").default_params,
        **{key: value for key, value in config.params.items() if key in allowed},
    }


def _save_scorecard_model(
    model: LogisticRegression,
    config: TrainConfig,
    out_dir: Path,
    woe_maps: dict,
    *,
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    serializable_woe = {feature: asdict(woe) for feature, woe in woe_maps.items()}
    joblib.dump(
        {
            "model": model,
            "woe_maps": woe_maps,
            "params": params,
            "features": tuple(config.features),
        },
        out_dir / model_path,
    )
    return ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="scorecard",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=serializable_woe,
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


__all__ = ["train_scorecard"]
