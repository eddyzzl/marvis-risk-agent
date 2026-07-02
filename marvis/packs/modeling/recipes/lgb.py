from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb

from marvis.data.labels import resolve_modeling_splits
from marvis.packs.modeling.artifact import (
    persist_model_meta,
    points_direction_for_algorithm,
    score_direction_for_algorithm,
    write_artifact_file,
)
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


def train_lgb(backend, dataset_path, config: TrainConfig, *, out_dir: Path) -> TrainResult:
    frame = backend.read_frame(dataset_path)
    train, test, oot = split_modeling_frame(frame, config)
    train, test, oot, oot_has_labels, audit = resolve_modeling_splits(
        train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels,
    )
    params = {
        **get_recipe("lgb").default_params,
        **model_params(config.params),
        "random_state": config.seed,
        # TUNE-6: sourced from defaults.py (single source shared with tune.py's
        # DEFAULT_TUNE_NUM_THREADS) instead of a bare literal -- both training
        # paths' thread counts now come from one place, even though their
        # defaults intentionally differ (single-thread here for cross-machine
        # determinism; tune.py defaults to full-core parallelism for search speed).
        "n_jobs": DEFAULT_TRAIN_NUM_THREADS,
        "deterministic": True,
        # TUNE-6: pins the row/col-wise split so deterministic=True's guarantee
        # actually holds (LightGBM's documented determinism contract requires it).
        "force_row_wise": True,
    }
    params = resolve_auto_scale_pos_weight(params, train, config)
    constraints = normalized_monotone_constraints(config)
    params.pop("monotone_constraints", None)
    params.pop("monotonic_constraints", None)
    if constraints is not None:
        params["monotone_constraints"] = list(constraints)
    num_boost_round = int(params.pop("num_boost_round", 20))
    callbacks = []
    fit_train = train
    if config.early_stopping_rounds:
        # SEL-4/TUNE-3: early stopping watches a fold carved from train, not test --
        # test stays a one-time, unbiased comparison set instead of also picking the
        # round count.
        fit_train, valid = carve_early_stop_fold_for_config(train, config)
        callbacks.append(lgb.early_stopping(config.early_stopping_rounds, verbose=False))
    else:
        valid = test
    model = lgb.LGBMClassifier(
        **params,
        n_estimators=num_boost_round,
    )
    valid_weight = sample_weight_values(valid, config)
    model.fit(
        fit_train[list(config.features)],
        fit_train[config.target_col].to_numpy(dtype=int),
        sample_weight=sample_weight_values(fit_train, config),
        eval_set=[(valid[list(config.features)], valid[config.target_col].to_numpy(dtype=int))],
        eval_sample_weight=[valid_weight] if valid_weight is not None else None,
        callbacks=callbacks,
    )
    metrics = compute_model_metrics(
        lambda data: model.predict_proba(data[list(config.features)])[:, 1],
        train,
        test,
        oot,
        config,
        oot_has_labels=oot_has_labels,
    )
    artifact = _save_lgb_model(
        model,
        config,
        out_dir,
        artifact_params({**params, "num_boost_round": num_boost_round}, config),
    )
    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        feature_importance=_lgb_importance(model, config.features),
        experiment_id="",
        nan_labels_dropped=audit["total_dropped"],
    )


def _save_lgb_model(
    model: lgb.LGBMClassifier,
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
        algorithm="lgb",
        model_path=model_path,
        pmml_path=None,
        feature_list=tuple(config.features),
        params=dict(params),
        woe_maps=None,
        created_at=datetime.now(UTC).isoformat(),
        score_direction=score_direction_for_algorithm("lgb"),
        points_direction=points_direction_for_algorithm("lgb"),
    )
    persist_model_meta(out_dir, artifact, config=config)
    return artifact


def _lgb_importance(
    model: lgb.LGBMClassifier,
    features: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    gains = model.booster_.feature_importance(importance_type="gain")
    return tuple((feature, float(value)) for feature, value in zip(features, gains, strict=True))


__all__ = ["train_lgb"]
