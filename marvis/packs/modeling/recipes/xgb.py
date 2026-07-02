from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import xgboost as xgb

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.artifact import persist_model_meta, write_artifact_file
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_TRAIN_NUM_THREADS
from marvis.packs.modeling.recipes import get_recipe
from marvis.packs.modeling.recipes.common import (
    artifact_params,
    carve_early_stop_fold_for_config,
    compute_model_metrics,
    model_params,
    normalized_monotone_constraints,
    resolve_auto_scale_pos_weight,
    sample_weight_values,
    split_modeling_frame,
)


def train_xgb(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("xgb").default_params,
        **model_params(config.params),
        "random_state": config.seed,
        # TUNE-6: sourced from defaults.py -- see lgb.py's train_lgb for the
        # single-source rationale shared across every tree recipe's direct-train path.
        "n_jobs": DEFAULT_TRAIN_NUM_THREADS,
    }
    params = resolve_auto_scale_pos_weight(params, train, config)
    constraints = normalized_monotone_constraints(config)
    params.pop("monotone_constraints", None)
    params.pop("monotonic_constraints", None)
    if constraints is not None:
        params["monotone_constraints"] = f"({','.join(str(value) for value in constraints)})"
    num_boost_round = int(params.pop("num_boost_round", 20))
    fit_train = train
    if config.early_stopping_rounds:
        # SEL-4/TUNE-3: early stopping watches a fold carved from train, not test --
        # test stays a one-time, unbiased comparison set instead of also picking the
        # round count.
        params["early_stopping_rounds"] = int(config.early_stopping_rounds)
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
    else:
        valid = test
    model = xgb.XGBClassifier(
        **params,
        n_estimators=num_boost_round,
    )
    valid_weight = sample_weight_values(valid, config)
    model.fit(
        fit_train[list(config.features)],
        fit_train[config.target_col].to_numpy(dtype=int),
        sample_weight=sample_weight_values(fit_train, config),
        eval_set=[(valid[list(config.features)], valid[config.target_col].to_numpy(dtype=int))],
        sample_weight_eval_set=[valid_weight] if valid_weight is not None else None,
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
    resolved_rounds = (
        int(getattr(model, "best_iteration", num_boost_round - 1)) + 1
        if config.early_stopping_rounds
        else num_boost_round
    )
    artifact = _save_xgb_model(
        model,
        config,
        out_dir,
        artifact_params(
            {**params, "num_boost_round": num_boost_round, "best_iteration": resolved_rounds},
            config,
        ),
    )
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
    model_path = f"{artifact_id}.pkl"
    write_artifact_file(out_dir, model_path, lambda path: joblib.dump(model, path))
    artifact = ModelArtifact(
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
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _xgb_importance(
    model: xgb.XGBClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.get_booster().get_score(importance_type="gain")
    return tuple((feature, float(gains.get(feature, 0.0))) for feature in features)


__all__ = ["train_xgb"]
