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
from marvis.feature.binning import chimerge_edges, monotonic_direction, monotonic_edges
from marvis.feature.encode import woe_encode
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.packs.modeling.artifact import persist_model_meta, write_artifact_file
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
    enforce_monotonic = bool(config.params.get("enforce_monotonic", True))
    requested_monotonic_direction = str(config.params.get("monotonic_direction") or "auto")
    woe_maps, binnings, monotonic_directions = _fit_woe_maps(
        train,
        config,
        max_bins=max_bins,
        enforce_monotonic=enforce_monotonic,
        direction=requested_monotonic_direction,
    )
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
    scorecard_base_points = float(offset - factor * float(model.intercept_[0]))
    scorecard_table = _scorecard_table(
        model,
        config,
        binnings,
        monotonic_directions=monotonic_directions,
        factor=float(factor),
        base_points=scorecard_base_points,
        base_score=base_score,
        pdo=pdo,
        base_odds=base_odds,
        offset=float(offset),
    )
    artifact = _save_scorecard_model(
        model,
        config,
        out_dir,
        woe_maps,
        scorecard_table=scorecard_table,
        params=artifact_params({
            **lr_params,
            "base_score": base_score,
            "pdo": pdo,
            "base_odds": base_odds,
            "factor": float(factor),
            "offset": float(offset),
            "scorecard_base_points": scorecard_base_points,
            "scorecard_max_bins": max_bins,
            "enforce_monotonic": enforce_monotonic,
            "monotonic_direction": requested_monotonic_direction,
            "monotonic_directions": monotonic_directions,
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
    enforce_monotonic: bool,
    direction: str,
) -> tuple[dict, dict, dict[str, str]]:
    target = train[config.target_col].to_numpy(dtype=int)
    maps = {}
    binnings = {}
    directions = {}
    for feature in config.features:
        values = train[feature].to_numpy(dtype=float)
        # PREP-9: minimum bin share (5%) keeps WOE estimates stable across time
        # periods instead of leaving a chimerge-surviving 1-2% bin to drift on OOT.
        edges = chimerge_edges(values, target, max_bins=max_bins, min_bin_pct=0.05)
        if enforce_monotonic:
            resolved_direction = monotonic_direction(values, target, edges, direction=direction)
            edges = monotonic_edges(values, target, edges, direction=resolved_direction)
            directions[feature] = resolved_direction
        binning = compute_woe_iv(values, target, edges, feature=feature)
        maps[feature] = woe_result_from_binning(binning)
        binnings[feature] = binning
    return maps, binnings, directions


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
    scorecard_table: tuple[dict, ...],
    params: dict,
) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = f"artifact_{uuid.uuid4().hex}"
    model_path = f"{artifact_id}.joblib"
    serializable_woe = {feature: asdict(woe) for feature, woe in woe_maps.items()}
    payload = {
        "model": model,
        "woe_maps": woe_maps,
        "params": params,
        "features": tuple(config.features),
        "scorecard_table": list(scorecard_table),
    }
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(payload, path))
    artifact = ModelArtifact(
        id=artifact_id,
        experiment_id="",
        algorithm="scorecard",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=serializable_woe,
        created_at=datetime.now(UTC).isoformat(),
        scorecard_table=scorecard_table,
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _scorecard_table(
    model: LogisticRegression,
    config: TrainConfig,
    binnings: dict,
    *,
    monotonic_directions: dict[str, str],
    factor: float,
    base_points: float,
    base_score: int,
    pdo: int,
    base_odds: float,
    offset: float,
) -> tuple[dict, ...]:
    coefficients = {
        feature: float(coef)
        for feature, coef in zip(config.features, model.coef_[0], strict=True)
    }
    rows: list[dict] = [{
        "feature": "__base__",
        "bin_index": -999,
        "bin_label": "base_points",
        "lower": None,
        "upper": None,
        "count": None,
        "bad_count": None,
        "good_count": None,
        "bad_rate": None,
        "woe": None,
        "iv_contribution": None,
        "coefficient": None,
        "monotonic_direction": None,
        "points": float(base_points),
        "base_score": int(base_score),
        "pdo": int(pdo),
        "base_odds": float(base_odds),
        "factor": float(factor),
        "offset": float(offset),
    }]
    for feature in config.features:
        coefficient = coefficients[feature]
        binning = binnings[feature]
        for bin_row in (*binning.bins, *((binning.na_bin,) if binning.na_bin else ())):
            rows.append({
                "feature": feature,
                "bin_index": int(bin_row.index),
                "bin_label": _bin_label(bin_row.lower, bin_row.upper),
                "lower": _finite_or_none(bin_row.lower),
                "upper": _finite_or_none(bin_row.upper),
                "count": int(bin_row.count),
                "bad_count": int(bin_row.bad_count),
                "good_count": int(bin_row.good_count),
                "bad_rate": float(bin_row.bad_rate),
                "woe": float(bin_row.woe),
                "iv_contribution": float(bin_row.iv_contribution),
                "coefficient": coefficient,
                "monotonic_direction": monotonic_directions.get(feature),
                "points": float(-factor * coefficient * float(bin_row.woe)),
            })
    return tuple(rows)


def _bin_label(lower: float, upper: float) -> str:
    if np.isnan(lower) or np.isnan(upper):
        return "missing"
    left = "-inf" if np.isneginf(lower) else f"{lower:.6g}"
    right = "inf" if np.isposinf(upper) else f"{upper:.6g}"
    return f"[{left}, {right})"


def _finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


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
