from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd

from marvis.artifacts import TransactionalArtifactStore
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.feature.candidates import candidate_numeric_features
from marvis.feature.metrics import feature_metrics
from marvis.feature.encode import woe_encode
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.artifact import export_pmml, load_model, persist_model_meta, write_artifact_file
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.handoff import handoff_to_validation
from marvis.packs.modeling.report_compute import (
    BusinessColumns,
    build_feature_dictionary,
    compute_amount_bin_table,
    compute_sample_analysis,
    compute_vintage_report,
    resolve_report_sections,
    stress_low_pricing,
)
from marvis.packs.modeling.readiness import check_data_quality, modeling_readiness
from marvis.packs.modeling.prepare import SPLIT_COLUMN, prepare_modeling_frame
from marvis.packs.modeling.reject_inference import reject_inference
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.packs.modeling.recipes.catboost import train_catboost
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.mlp import train_mlp
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.feature.screen import screen_features, screen_features_non_binary
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.select import select_features
from marvis.packs.modeling.tune import tune_hyperparameters
from marvis.packs.modeling.errors import ModelingError
from marvis.settings import build_settings
from marvis.validation.config import ValidationConfig
from marvis.validation.stress_test import run_stress_test


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"
MODEL_REPORT_SCORE_COL = "__model_score__"
SCORECARD_POINTS_COL = "__scorecard_points__"
PMML_SUPPORTED_ALGORITHMS = frozenset({"lr", "lgb", "xgb", "scorecard"})
CALIBRATION_PARAMS_KEY = "calibration"
SUPPORTED_MODELING_RECIPES = frozenset({
    "lgb",
    "xgb",
    "catboost",
    "lr",
    "scorecard",
    "mlp",
    "lgb_regressor",
    "lgb_multiclass",
})
BINARY_MODELING_RECIPES = frozenset({"lgb", "xgb", "catboost", "lr", "scorecard", "mlp"})
CONTINUOUS_MODELING_RECIPES = frozenset({"lgb_regressor"})
MULTICLASS_MODELING_RECIPES = frozenset({"lgb_multiclass"})


def tool_check_data_quality(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    issues = check_data_quality(
        runtime.backend,
        dataset,
        runtime.registry.resolve_path(dataset.id),
        target_col=_optional_str(inputs.get("target_col")),
    )
    return {"issues": [_jsonable(issue) for issue in issues]}


def tool_modeling_readiness(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    return _jsonable(
        modeling_readiness(
            runtime.backend,
            dataset,
            runtime.registry.resolve_path(dataset.id),
            target_col=str(inputs["target_col"]),
            split_col=_optional_str(inputs.get("split_col")),
        )
    )


def tool_reject_inference(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id))
    result = reject_inference(
        frame,
        target_col=str(inputs["target_col"]),
        decision_col=str(inputs["decision_col"]),
        method=str(inputs.get("method") or "parceling"),
        score_col=_optional_str(inputs.get("score_col")),
        reject_bad_rate=_optional_float(inputs.get("reject_bad_rate")),
        reject_weight=float(inputs.get("reject_weight") or 1.0),
    )
    out_dir = runtime.datasets_root / str(ctx.task_id) / "modeling"
    artifact = TransactionalArtifactStore(out_dir).stage(f"reject_inference_{uuid.uuid4().hex}.parquet")
    try:
        result.frame.to_parquet(artifact.path, index=False)
        final_path = artifact.promote()
        registered = runtime.registry.register_existing_with_audit(
            final_path,
            audit_factory=lambda registered_dataset: {
                "kind": "modeling.reject_inference.created",
                "target_ref": registered_dataset.id,
                "outcome": "succeeded",
                "detail": {
                    "source_dataset_id": dataset.id,
                    "method": str(inputs.get("method") or "parceling"),
                    "target_col": str(inputs["target_col"]),
                    "decision_col": str(inputs["decision_col"]),
                    "sample_weight_col": result.sample_weight_col,
                },
            },
            task_id=str(ctx.task_id),
            role="reject_inference",
            anchor_target=dataset.id,
            seed=_effective_seed(inputs, ctx),
        )
        artifact.commit()
    except Exception:
        artifact.rollback()
        raise
    return {
        "result_dataset_id": registered.id,
        "target_col": result.target_col,
        "sample_weight_col": result.sample_weight_col,
        "diagnostics": _jsonable(result.diagnostics),
    }


def tool_prepare_modeling_frame(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    split_col = _optional_str(inputs.get("split_col"))
    feature_cols = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("feature_cols") or [],
        target_col=str(inputs["target_col"]),
        split_col=split_col,
    )
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        dataset.id,
        target_col=str(inputs["target_col"]),
        feature_cols=feature_cols,
        split_col=split_col,
        split_config=inputs.get("split_config") or {},
        passthrough_cols=[str(item) for item in inputs.get("passthrough_cols") or [] if str(item).strip()],
        seed=_effective_seed(inputs, ctx),
        audit_kind="modeling.dataset.derived",
        audit_detail={"tool": "prepare_modeling_frame"},
    )
    split_col = split_col or "split"
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(result.id), columns=[split_col])
    counts = {
        str(key): int(value)
        for key, value in frame[split_col].value_counts().sort_index().items()
    }
    return {
        "result_dataset_id": result.id,
        "split_counts": counts,
        "split_col": split_col,
        "split_values": {key: key for key in counts},
        "holdout_values": ["oot"] if "oot" in counts else [],
        "feature_cols": feature_cols,
    }


def tool_make_split(inputs: dict, ctx) -> dict:
    """MODELING G1 split gate: build a derived modeling frame from an arbitrary
    rule set (e.g. channel A → train, channel B before a cutoff → test) plus the
    existing random/time fallback, then return per-split counts and, when month or
    channel columns are present, a per-split × per-group distribution table for the
    confirmation gate UI."""
    runtime = _runtime(ctx)
    # split_col present → pass the EXISTING split through unchanged (the gate just surfaces
    # it for review); absent → generate from split_config (rules / time-OOT / grouped-random
    # fallback). prepare_modeling_frame keeps the passed-through column's name, and names a
    # generated column SPLIT_COLUMN, so the effective name is one or the other.
    split_col = str(inputs["split_col"]) if inputs.get("split_col") else None
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    feature_cols = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("feature_cols") or [],
        target_col=str(inputs["target_col"]),
        split_col=split_col,
    )
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        dataset.id,
        target_col=str(inputs["target_col"]),
        feature_cols=feature_cols,
        split_col=split_col,
        split_config=inputs.get("split_config") or {},
        passthrough_cols=[str(item) for item in inputs.get("passthrough_cols") or [] if str(item).strip()],
        seed=_effective_seed(inputs, ctx),
        audit_kind="modeling.dataset.derived",
        audit_detail={"tool": "make_split"},
    )
    effective_split_col = split_col or SPLIT_COLUMN
    split_frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(result.id), columns=[effective_split_col]
    )
    dataset_path = runtime.registry.resolve_path(dataset.id)
    source_columns = [profile.name for profile in dataset.columns] or runtime.backend.column_names(dataset_path)
    group_columns = _detect_group_columns(source_columns)
    source_frame = (
        runtime.backend.read_frame(dataset_path, columns=group_columns)
        if group_columns
        else pd.DataFrame(index=split_frame.index)
    )
    sample_analysis = _split_sample_analysis(split_frame[effective_split_col], source_frame)
    split_counts = sample_analysis["split_counts"]
    return {
        "result_dataset_id": result.id,
        "split_col": effective_split_col,
        "split_values": {key: key for key in split_counts},
        "holdout_values": ["oot"] if "oot" in split_counts else [],
        "feature_cols": feature_cols,
        "sample_analysis": _json_safe(sample_analysis),
    }


_GROUP_COLUMN_HINTS = ("month", "channel", "渠道", "月", "split_month")


def _split_sample_analysis(split_series: pd.Series, source_frame: pd.DataFrame) -> dict:
    """Row counts per split plus, for each detected month/channel-like column, a
    per-split × per-group count table. The split frame and the source frame share row
    order (prepare_modeling_frame preserves it), so we align by position."""
    splits = [str(value) for value in split_series.tolist()]
    counts: dict[str, int] = {}
    for split in splits:
        counts[split] = counts.get(split, 0) + 1
    group_tables: dict[str, dict] = {}
    for column in _detect_group_columns(source_frame.columns):
        values = source_frame[column].astype("object").where(source_frame[column].notna(), None)
        table: dict[str, dict[str, int]] = {}
        for split, value in zip(splits, values.tolist()):
            key = "(missing)" if value is None else str(value)
            row = table.setdefault(split, {})
            row[key] = row.get(key, 0) + 1
        group_tables[str(column)] = {split: dict(sorted(row.items())) for split, row in table.items()}
    return {
        "split_counts": dict(sorted(counts.items())),
        "total_rows": len(splits),
        "group_distributions": group_tables,
    }


def _detect_group_columns(columns) -> list[str]:
    return [str(column) for column in columns if any(hint in str(column) for hint in _GROUP_COLUMN_HINTS)]


def tool_select_features(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
    )
    split_col = _optional_str(inputs.get("split_col"))
    result = select_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        iv_min=float(inputs.get("iv_min", 0.02)),
        corr_max=float(inputs.get("corr_max", 0.8)),
        vif_max=float(inputs.get("vif_max", 10.0)),
        top_k=_optional_int(inputs.get("top_k")),
        seed=_effective_seed(inputs, ctx),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        space=str(inputs.get("space") or "raw"),
        split_col=split_col,
        split_value=inputs.get("split_value"),
        scorecard_max_bins=int(inputs.get("scorecard_max_bins") or 6),
        enforce_monotonic=bool(inputs.get("enforce_monotonic", True)),
        monotonic_direction_request=str(inputs.get("monotonic_direction") or "auto"),
        sign_check=bool(inputs.get("sign_check", True)),
    )
    return {
        "selected": list(result.selected),
        "dropped": [[feature, reason] for feature, reason in result.dropped],
        "scores": _jsonable(result.scores),
        "nan_labels_dropped": result.nan_labels_dropped,
        "warnings": list(result.warnings),
    }


def tool_screen_features(inputs: dict, ctx) -> dict:
    # feature_ks is a binary-only statistic; a continuous target would miscompute/crash
    # it, so for a non-binary target skip the leakage screen and keep every candidate.
    if str(inputs.get("target_type", "binary")) != "binary":
        return _screen_features_non_binary(inputs, ctx)
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
    )
    holdout = inputs.get("holdout_values")
    result = screen_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
        holdout_values=tuple(str(v) for v in holdout) if holdout else ("oot",),
        leakage_ks=float(inputs.get("leakage_ks", 0.40)),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=_optional_int(inputs.get("top_k")),
        batch_size=int(inputs.get("batch_size", 500)),
    )
    return {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
    }


def _screen_features_non_binary(inputs: dict, ctx) -> dict:
    """Non-binary (continuous/multiclass) screen: the binary-only leakage KS screen is skipped,
    but unusable columns are still dropped into ``unusable`` (mirroring the binary screen) —
    constant (unique_count<=1) or mostly-missing (missing_rate>=max_missing_rate) — and the
    rest are kept as selected (ks=None)."""
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
    )
    holdout = inputs.get("holdout_values")
    result = screen_features_non_binary(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
        holdout_values=tuple(str(v) for v in holdout) if holdout else ("oot",),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=_optional_int(inputs.get("top_k")),
    )
    return {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [],
        "suspected": [],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "note": "非二分类目标：跳过泄漏KS筛选，已剔除常量/高缺失列",
    }


def tool_choose_modeling_spec(inputs: dict, ctx) -> dict:
    recipes = _normalize_recipe_list(inputs.get("recipes") or [inputs.get("recipe") or "lgb"])
    target_type = _normalize_modeling_target_type(inputs.get("target_type")) or _target_type_from_recipes(recipes)
    derived_target_type = _target_type_from_recipes(recipes)
    if target_type != derived_target_type:
        raise ModelingError(
            f"target_type `{target_type}` does not match recipes `{', '.join(recipes)}`"
        )
    primary_recipe = "lgb" if "lgb" in recipes else recipes[0]
    sample_weight_col = str(inputs.get("sample_weight_col") or "").strip()
    sample_weight_candidates = _unique_strings([
        sample_weight_col,
        *(inputs.get("sample_weight_candidates") or []),
    ])
    sample_weight_diagnostics = [
        dict(item)
        for item in (inputs.get("sample_weight_diagnostics") or [])
        if isinstance(item, dict)
    ]
    target_col = str(inputs.get("target_col") or "").strip()
    features = _unique_strings(inputs.get("features") or [])
    warnings: list[str] = []
    if sample_weight_col and sample_weight_col == target_col:
        raise ModelingError("sample_weight_col cannot be the target column")
    if sample_weight_col and sample_weight_col in features:
        features = [feature for feature in features if feature != sample_weight_col]
        warnings.append("样本权重列已从入模特征中移除。")
    n_trials = int(inputs.get("n_trials") or 12)
    if n_trials < 1:
        raise ModelingError("n_trials must be at least 1")
    params = _training_params(inputs)
    if sample_weight_col:
        params["sample_weight_col"] = sample_weight_col
    metric_policy = _metric_policy_for_target_type(target_type)
    return {
        "target_type": target_type,
        "recipe": primary_recipe,
        "recipes": recipes,
        "feature_cols": features,
        "feature_count": len(features),
        "sample_weight_col": sample_weight_col,
        "sample_weight_candidates": sample_weight_candidates,
        "sample_weight_diagnostics": _jsonable(sample_weight_diagnostics),
        "seed": _effective_seed(inputs, ctx),
        "n_trials": n_trials,
        "params": _jsonable(params),
        "metric_policy": metric_policy,
        "eligible_algorithms": _eligible_algorithms(target_type),
        "disabled_algorithms": _disabled_algorithms(target_type),
        "pmml_supported_algorithms": sorted(PMML_SUPPORTED_ALGORITHMS),
        "warnings": warnings,
        "reason": (
            f"目标类型 `{target_type}`,候选算法 {'/'.join(recipes)},"
            f"主调参算法 `{primary_recipe}`,选择指标 {metric_policy}。"
        ),
    }


def tool_configure_tuning(inputs: dict, ctx) -> dict:
    recipe = str(inputs.get("recipe") or "lgb")
    target_type = str(inputs.get("target_type") or "binary")
    n_trials = int(inputs.get("n_trials") or 12)
    if n_trials < 1:
        raise ModelingError("n_trials must be at least 1")
    sample_weight_col = str(inputs.get("sample_weight_col") or "").strip()
    seed = _effective_seed(inputs, ctx)
    tune_enabled = recipe == "lgb"
    params = _training_params(inputs)
    return {
        "recipe": recipe,
        "target_type": target_type,
        "tune_enabled": tune_enabled,
        "n_trials": n_trials if tune_enabled else 0,
        "sample_weight_col": sample_weight_col,
        "seed": seed,
        "params": _jsonable(params),
        "reason": "LightGBM 使用有界随机搜索。" if tune_enabled else f"{recipe} 暂不执行随机搜索,使用算法默认参数。",
    }


def tool_tune_hyperparameters(inputs: dict, ctx) -> dict:
    # The random search is LightGBM-specific. For other recipes there is no lgb
    # search to run (lr/scorecard have their own knobs, not a random search; xgb
    # tuning is a later slice), so we skip tuning and let train_model use the
    # recipe's own defaults.
    recipe = str(inputs.get("recipe") or "lgb")
    configured_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, configured_params)
    base_params = {**configured_params, **control_params}
    if recipe != "lgb":
        return {"best_params": _jsonable(base_params), "best_metrics": {}, "n_trials": 0, "trials": []}
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    result = tune_hyperparameters(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=[str(item) for item in inputs["features"]],
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        n_trials=int(inputs.get("n_trials", 40)),
        seed=_effective_seed(inputs, ctx),
        early_stopping_rounds=int(inputs.get("early_stopping_rounds", 100)),
        max_boost_round=int(inputs.get("max_boost_round", 3000)),
        overfit_penalty=float(inputs.get("overfit_penalty", 0.5)),
        sample_weight_col=control_params.get("sample_weight_col", ""),
        base_params=base_params,
    )
    best_params = {**control_params, **result.best_params}
    return {
        "best_params": _jsonable(best_params),
        "best_metrics": _jsonable(result.best_metrics),
        "n_trials": result.n_trials,
        "trials": _jsonable(result.trials),
    }


def tool_train_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipe = str(inputs["recipe"])
    config = TrainConfig(
        dataset_id=dataset.id,
        features=tuple(str(item) for item in inputs["features"]),
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        params=_training_params(inputs),
        seed=int(inputs["seed"]),
        early_stopping_rounds=_optional_int(inputs.get("early_stopping_rounds")),
        recipe_id=recipe,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    if inputs.get("scenario"):
        config = apply_scenario(config, str(inputs["scenario"]))
        recipe = config.recipe_id or recipe

    experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
    artifact_dir = _artifact_base_dir(runtime.settings, ctx.task_id)
    meta_snapshot = _snapshot_latest_model_meta(artifact_dir)
    result = None
    try:
        result = _train_recipe(
            recipe,
            runtime.backend,
            runtime.registry.resolve_path(dataset.id),
            config,
            out_dir=artifact_dir,
        )
        runtime.experiments.attach_result(experiment_id, result)
    except Exception:
        if result is not None:
            _cleanup_unattached_artifact(result.artifact, artifact_dir, meta_snapshot)
        runtime.experiments.set_status(experiment_id, "failed")
        raise

    experiment = runtime.experiments.get(experiment_id)
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact after training: {experiment_id}")
    artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {experiment.artifact_id}")
    return {
        "experiment_id": experiment_id,
        "artifact_id": artifact.id,
        "metrics": _jsonable(experiment.metrics),
        "feature_importance": _jsonable(result.feature_importance),
        "nan_labels_dropped": result.nan_labels_dropped,
    }


def tool_train_models(inputs: dict, ctx) -> dict:
    """Train each requested recipe and return all experiments plus the best by OOT KS
    (test KS fallback). lgb uses the tuned params; other recipes train with their own
    defaults. The single-recipe case (recipes=[lgb]) behaves like train_model."""
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipes = [str(item) for item in inputs["recipes"]]
    tuned_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, tuned_params)
    features = tuple(str(item) for item in inputs["features"])
    target_col = str(inputs["target_col"])
    split_col = str(inputs["split_col"])
    split_values = dict(inputs["split_values"])
    seed = int(inputs["seed"])
    drop_nan = bool(inputs.get("drop_nan_labels"))
    target_type = str(inputs.get("target_type", "binary"))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    training_dataset = TrainingDataset.load(runtime.backend, dataset_path)
    training_backend = training_dataset.backend_adapter(runtime.backend)

    experiments: list[dict] = []
    for recipe in recipes:
        config = TrainConfig(
            dataset_id=dataset.id,
            features=features,
            target_col=target_col,
            split_col=split_col,
            split_values=split_values,
            # only the lgb recipe consumes the tuned params; others use their defaults
            params={**tuned_params, **control_params} if recipe == "lgb" else dict(control_params),
            seed=seed,
            early_stopping_rounds=None,
            recipe_id=recipe,
            target_type=target_type,
            drop_nan_labels=drop_nan,
        )
        experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
        artifact_dir = _artifact_base_dir(runtime.settings, ctx.task_id)
        meta_snapshot = _snapshot_latest_model_meta(artifact_dir)
        result = None
        try:
            result = _train_recipe(
                recipe,
                training_backend,
                dataset_path,
                config,
                out_dir=artifact_dir,
            )
            runtime.experiments.attach_result(experiment_id, result)
        except Exception:
            if result is not None:
                _cleanup_unattached_artifact(result.artifact, artifact_dir, meta_snapshot)
            runtime.experiments.set_status(experiment_id, "failed")
            raise
        experiment = runtime.experiments.get(experiment_id)
        experiments.append({
            "experiment_id": experiment_id,
            "recipe": recipe,
            "metrics": _jsonable(experiment.metrics) or {},
        })

    best, selection_metric = _pick_best_experiment(experiments, target_type=target_type)
    return {
        "experiments": experiments,
        "experiment_ids": [exp["experiment_id"] for exp in experiments],
        "best_experiment_id": best["experiment_id"],
        "best_recipe": best["recipe"],
        "target_type": target_type,
        "selection_metric": selection_metric,
    }


def _pick_best_experiment(experiments: list[dict], *, target_type: str = "binary") -> tuple[dict, str]:
    """Pick the best experiment with the metric family that matches the target.

    Binary maximizes OOT/test KS; regression minimizes OOT/test RMSE; multiclass
    maximizes OOT/test macro-AUC, falling back to minimizing logloss.
    """
    target_type = str(target_type or "binary")
    if target_type == "continuous":
        metric_keys = ("oot_rmse", "test_rmse")

        def score(experiment: dict) -> float:
            metrics = experiment.get("metrics") or {}
            for key in metric_keys:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    return -float(value)
            return float("-inf")

        return max(experiments, key=score), "oot_rmse"
    if target_type == "multiclass":
        auc_keys = ("oot_macro_auc", "test_macro_auc")
        logloss_keys = ("oot_logloss", "test_logloss")

        def score(experiment: dict) -> float:
            metrics = experiment.get("metrics") or {}
            for key in auc_keys:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for key in logloss_keys:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    return -float(value)
            return float("-inf")

        return max(experiments, key=score), "oot_macro_auc"

    def score(experiment: dict) -> float:
        metrics = experiment.get("metrics") or {}
        for key in ("oot_ks", "test_ks"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return float("-inf")

    return max(experiments, key=score), "oot_ks"


def tool_compare_experiments(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    compared = runtime.experiments.compare([str(item) for item in inputs["experiment_ids"]])
    rows = [row for row in compared.get("experiments", []) if isinstance(row, dict)]
    _attach_capabilities_to_comparison_rows(runtime, rows)
    _attach_policy_profile_to_comparison_rows(runtime, rows)
    return _jsonable(compared)


def tool_select_experiment(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment_ids = [str(item) for item in inputs.get("experiment_ids") or [] if str(item).strip()]
    if not experiment_ids:
        raise ModelingError("experiment_ids must not be empty")
    target_type = str(inputs.get("target_type") or "").strip()
    if not target_type:
        target_type = getattr(runtime.experiments.get(experiment_ids[0]).config, "target_type", "binary")
    compared = runtime.experiments.compare(experiment_ids)
    rows = [row for row in compared.get("experiments", []) if isinstance(row, dict)]
    _attach_capabilities_to_comparison_rows(runtime, rows)
    _attach_policy_profile_to_comparison_rows(runtime, rows)
    selection_policy = _normalize_selection_policy(inputs.get("selection_policy"))
    selected_id = str(inputs.get("selected_experiment_id") or "").strip()
    if selected_id:
        selected = next((row for row in rows if row.get("id") == selected_id), None)
        if selected is None:
            raise ModelingError(f"selected_experiment_id is not in candidates: {selected_id}")
        selection_metric = str(inputs.get("selection_metric") or "manual")
        selection_reason = "用户指定实验。"
        policy_decision = _selection_policy_decision(selected, selection_policy, explicit=True)
        if policy_decision["status"] == "blocked":
            raise ModelingError(_selection_policy_block_message(selected_id, policy_decision))
    else:
        selected, selection_metric, policy_decision = _pick_best_comparison_row_with_policy(
            rows,
            target_type=target_type,
            policy=selection_policy,
        )
        selected_id = str(selected.get("id") or "")
        if _selection_policy_requested(selection_policy) and policy_decision["status"] == "accepted":
            selection_reason = f"按 {selection_metric} 在满足交付/审批策略的候选中自动选择。"
            if policy_decision.get("selected_by_preference"):
                selection_reason = f"按 {selection_metric} 在评分卡优先候选中自动选择。"
        elif _selection_policy_requested(selection_policy) and policy_decision["status"] == "overridden":
            selection_reason = f"按 {selection_metric} 自动选择;未满足全部交付/审批策略,已按 override_reason 放行。"
        elif _delivery_ready(selected):
            selection_reason = f"按 {selection_metric} 在 PMML/验证移交可用候选中自动选择。"
        else:
            selection_reason = f"按 {selection_metric} 自动选择。"
    artifact_id = str(selected.get("artifact_id") or "")
    if not artifact_id:
        raise ModelingError(f"selected experiment has no artifact: {selected_id}")
    artifact = _artifact(runtime, artifact_id)
    experiment = runtime.experiments.get(selected_id)
    capabilities = _artifact_capabilities(
        artifact,
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    return {
        "selected_experiment_id": selected_id,
        "artifact_id": artifact_id,
        "recipe": selected.get("recipe") or experiment.recipe_id,
        "target_type": target_type,
        "selection_metric": selection_metric,
        "selection_reason": selection_reason,
        "metrics": {k: v for k, v in selected.items() if _is_metric_key(k) and v is not None},
        "capabilities": capabilities,
        "policy_profile": selected.get("policy_profile") or {},
        "policy_decision": policy_decision,
        "scorecard_table": selected.get("scorecard_table") or [],
        "model_params": selected.get("model_params") or {},
        "experiments": rows,
    }


def _pick_best_comparison_row_with_policy(
    rows: list[dict],
    *,
    target_type: str,
    policy: dict,
) -> tuple[dict, str, dict]:
    if not _selection_policy_requested(policy):
        selected, metric = _pick_best_comparison_row(rows, target_type=target_type)
        return selected, metric, _selection_policy_decision(selected, policy, explicit=False)

    compliant = [row for row in rows if not _selection_policy_violations(row, policy)]
    if compliant:
        candidates = compliant
    elif _selection_policy_has_hard_requirements(policy) and not policy.get("allow_policy_override"):
        raise ModelingError(_no_policy_candidate_message(rows, policy))
    else:
        candidates = rows

    selected_by_preference = False
    if policy.get("prefer_scorecard"):
        scorecard_candidates = [
            row for row in candidates
            if _row_policy_profile(row).get("scorecard")
        ]
        if scorecard_candidates:
            candidates = scorecard_candidates
            selected_by_preference = True

    selected, metric = _pick_best_comparison_row(candidates, target_type=target_type)
    decision = _selection_policy_decision(selected, policy, explicit=False)
    decision["evaluated_candidates"] = len(rows)
    decision["policy_candidate_count"] = len(compliant)
    decision["selected_by_preference"] = selected_by_preference
    if decision["status"] == "blocked":
        raise ModelingError(_selection_policy_block_message(str(selected.get("id") or ""), decision))
    return selected, metric, decision


def _pick_best_comparison_row(rows: list[dict], *, target_type: str) -> tuple[dict, str]:
    if not rows:
        raise ModelingError("experiment_ids must resolve to experiments")
    delivery_ready = [row for row in rows if _delivery_ready(row)]
    if delivery_ready:
        rows = delivery_ready
    target_type = str(target_type or "binary")
    if target_type == "continuous":
        return max(rows, key=lambda row: _score_first(row, ("oot_rmse", "test_rmse"), minimize=True)), "oot_rmse"
    if target_type == "multiclass":
        auc_best = max(rows, key=lambda row: _score_first(row, ("oot_macro_auc", "test_macro_auc")))
        if _score_first(auc_best, ("oot_macro_auc", "test_macro_auc")) != float("-inf"):
            return auc_best, "oot_macro_auc"
        return max(rows, key=lambda row: _score_first(row, ("oot_logloss", "test_logloss"), minimize=True)), "oot_logloss"
    return max(rows, key=lambda row: _score_first(row, ("oot_ks", "test_ks"))), "oot_ks"


def _attach_capabilities_to_comparison_rows(runtime: _Runtime, rows: list[dict]) -> None:
    for row in rows:
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            continue
        artifact = runtime.modeling_repo.get_model_artifact(str(artifact_id))
        if artifact is None:
            continue
        experiment = runtime.experiments.get(artifact.experiment_id)
        row["capabilities"] = _artifact_capabilities(
            artifact,
            base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
        )


def _attach_policy_profile_to_comparison_rows(runtime: _Runtime, rows: list[dict]) -> None:
    for row in rows:
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            row["policy_profile"] = _row_policy_profile(row)
            continue
        artifact = runtime.modeling_repo.get_model_artifact(str(artifact_id))
        if artifact is None:
            row["policy_profile"] = _row_policy_profile(row)
            continue
        row["feature_count"] = len(artifact.feature_list)
        row["feature_list"] = list(artifact.feature_list)
        row["model_params"] = dict(artifact.params or {})
        row["scorecard_table"] = _scorecard_table_rows(artifact)
        row["policy_profile"] = _row_policy_profile(row)


def _delivery_ready(row: dict) -> bool:
    caps = row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {}
    return bool(caps.get("pmml_supported") and caps.get("handoff_supported"))


def _normalize_selection_policy(raw) -> dict:
    source = raw if isinstance(raw, dict) else {}
    policy = {
        "require_pmml": _policy_bool(source.get("require_pmml")),
        "require_handoff": _policy_bool(source.get("require_handoff")),
        "require_scorecard": _policy_bool(source.get("require_scorecard")),
        "require_monotonicity": _policy_bool(source.get("require_monotonicity")),
        "prefer_scorecard": _policy_bool(source.get("prefer_scorecard")),
        "allow_policy_override": _policy_bool(source.get("allow_policy_override")),
        "override_reason": str(source.get("override_reason") or "").strip(),
    }
    max_feature_count = _positive_int_or_none(source.get("max_feature_count"))
    if max_feature_count is not None:
        policy["max_feature_count"] = max_feature_count
    max_oot_psi = _nonnegative_float_or_none(source.get("max_oot_psi"))
    if max_oot_psi is not None:
        policy["max_oot_psi"] = max_oot_psi
    return policy


def _policy_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _selection_policy_requested(policy: dict) -> bool:
    return any(
        bool(policy.get(key))
        for key in (
            "require_pmml",
            "require_handoff",
            "require_scorecard",
            "require_monotonicity",
            "prefer_scorecard",
        )
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None


def _selection_policy_has_hard_requirements(policy: dict) -> bool:
    return any(
        bool(policy.get(key))
        for key in (
            "require_pmml",
            "require_handoff",
            "require_scorecard",
            "require_monotonicity",
        )
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None


def _selection_policy_decision(row: dict, policy: dict, *, explicit: bool) -> dict:
    profile = _row_policy_profile(row)
    requested = _selection_policy_requested(policy)
    violations = _selection_policy_violations(row, policy)
    missing_override_reason = bool(violations and policy.get("allow_policy_override") and not policy.get("override_reason"))
    if missing_override_reason:
        violations = [
            *violations,
            {
                "code": "override_reason_required",
                "message": "策略 override 必须填写 override_reason。",
            },
        ]
    if not requested:
        status = "not_requested"
    elif violations and policy.get("allow_policy_override") and not missing_override_reason:
        status = "overridden"
    elif violations:
        status = "blocked"
    else:
        status = "accepted"
    return {
        "status": status,
        "explicit_selection": bool(explicit),
        "selected_experiment_id": str(row.get("id") or ""),
        "policy": {
            key: value
            for key, value in policy.items()
            if value not in (None, "", False)
        },
        "profile": profile,
        "violations": violations,
        "override_reason": policy.get("override_reason") if status == "overridden" else "",
    }


def _selection_policy_violations(row: dict, policy: dict) -> list[dict]:
    if not _selection_policy_requested(policy):
        return []
    profile = _row_policy_profile(row)
    violations: list[dict] = []
    if policy.get("require_pmml") and not profile.get("pmml_supported"):
        violations.append({
            "code": "require_pmml",
            "message": "要求最终模型支持 PMML 导出,但该候选不支持。",
        })
    if policy.get("require_handoff") and not profile.get("handoff_supported"):
        violations.append({
            "code": "require_handoff",
            "message": "要求最终模型支持验证移交,但该候选不支持。",
        })
    if policy.get("require_scorecard") and not profile.get("scorecard"):
        violations.append({
            "code": "require_scorecard",
            "message": "要求最终模型为评分卡,但该候选不是评分卡。",
        })
    if policy.get("require_monotonicity") and not profile.get("monotonicity_declared"):
        violations.append({
            "code": "require_monotonicity",
            "message": "要求声明单调约束,但该候选缺少单调性证据。",
        })
    max_feature_count = policy.get("max_feature_count")
    feature_count = profile.get("feature_count")
    if isinstance(max_feature_count, int) and isinstance(feature_count, int) and feature_count > max_feature_count:
        violations.append({
            "code": "max_feature_count",
            "message": f"要求特征数不超过 {max_feature_count},但该候选有 {feature_count} 个特征。",
        })
    max_oot_psi = policy.get("max_oot_psi")
    oot_psi = profile.get("psi_oot_vs_train")
    if isinstance(max_oot_psi, (int, float)) and isinstance(oot_psi, (int, float)) and oot_psi > float(max_oot_psi):
        violations.append({
            "code": "max_oot_psi",
            "message": (
                f"要求 OOT PSI 不超过 {_format_number_token(float(max_oot_psi))},"
                f"但该候选为 {_format_number_token(float(oot_psi))}。"
            ),
        })
    return violations


def _selection_policy_block_message(experiment_id: str, decision: dict) -> str:
    reasons = "; ".join(
        f"{item.get('code')}: {item.get('message') or ''}".strip()
        for item in decision.get("violations", [])
        if isinstance(item, dict)
    )
    suffix = f" {reasons}" if reasons else ""
    return (
        f"selected_experiment_id violates selection_policy: {experiment_id}.{suffix} "
        "Set allow_policy_override=true with override_reason to keep this candidate."
    )


def _no_policy_candidate_message(rows: list[dict], policy: dict) -> str:
    details = []
    for row in rows[:5]:
        violations = _selection_policy_violations(row, policy)
        if not violations:
            continue
        details.append(
            f"{row.get('id') or '?'}: "
            + ", ".join(str(item.get("code") or "?") for item in violations if isinstance(item, dict))
        )
    suffix = f" Candidates: {'; '.join(details)}" if details else ""
    return (
        "no experiment satisfies selection_policy. "
        "Relax the policy, retrain a compliant candidate, or set allow_policy_override=true "
        f"with override_reason.{suffix}"
    )


def _row_policy_profile(row: dict) -> dict:
    item = row if isinstance(row, dict) else {}
    caps = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    scorecard_rows = item.get("scorecard_table") if isinstance(item.get("scorecard_table"), list) else []
    recipe = str(item.get("recipe") or "")
    features = item.get("feature_list") if isinstance(item.get("feature_list"), list) else item.get("features")
    feature_count = item.get("feature_count")
    if not isinstance(feature_count, int) and isinstance(features, list):
        feature_count = len(features)
    profile = {
        "recipe": recipe,
        "scorecard": recipe == "scorecard" or bool(scorecard_rows),
        "scorecard_table_rows": len(scorecard_rows),
        "monotonicity_declared": _row_has_monotonic_policy(item, scorecard_rows),
        "pmml_supported": bool(caps.get("pmml_supported")),
        "handoff_supported": bool(caps.get("handoff_supported")),
        "native_model_supported": bool(caps.get("native_model_supported")),
        "feature_count": feature_count if isinstance(feature_count, int) else None,
    }
    psi_oot = item.get("psi_oot_vs_train")
    if isinstance(psi_oot, (int, float)):
        profile["psi_oot_vs_train"] = float(psi_oot)
    return profile


def _row_has_monotonic_policy(item: dict, scorecard_rows: list) -> bool:
    for key in ("monotonic_constraints", "monotone_constraints", "monotonic_directions"):
        value = item.get(key)
        if isinstance(value, (dict, list, tuple)) and len(value) > 0:
            return True
        if isinstance(value, str) and value.strip():
            return True
    for container_key in ("params", "model_params", "fixed_params"):
        value = item.get(container_key)
        if isinstance(value, dict) and _row_has_monotonic_policy(value, []):
            return True
    for row in scorecard_rows:
        if isinstance(row, dict) and str(row.get("monotonic_direction") or "").strip():
            return True
    return False


def _positive_int_or_none(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _nonnegative_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _score_first(row: dict, keys: tuple[str, ...], *, minimize: bool = False) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            number = float(value)
            return -number if minimize else number
    return float("-inf")


def _is_metric_key(key: str) -> bool:
    return key.startswith(("train_", "test_", "oot_", "psi_")) or key == "overfit_flag"


def tool_calibrate_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    experiment = runtime.experiments.get(artifact.experiment_id)
    config = experiment.config
    if getattr(config, "target_type", "binary") != "binary":
        raise ModelingError("probability calibration is only supported for binary models")

    method = str(inputs.get("method") or "sigmoid").strip().lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ModelingError(f"unsupported calibration method: {method}")
    n_bins = int(inputs.get("n_bins") or 10)
    min_samples = int(inputs.get("min_samples") or 30)
    if n_bins < 2:
        raise ModelingError("n_bins must be at least 2")
    if min_samples < 1:
        raise ModelingError("min_samples must be at least 1")

    dataset_id = str(inputs.get("dataset_id") or config.dataset_id)
    dataset = runtime.registry.get(dataset_id)
    target_col = str(inputs.get("target_col") or config.target_col)
    split_col = str(inputs.get("split_col") or config.split_col)
    split_name = str(inputs.get("split") or "test")
    split_value = inputs.get("split_value", config.split_values.get(split_name, split_name))
    frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(dataset.id),
        columns=_unique_columns([*artifact.feature_list, target_col, split_col]),
    )
    sample = frame[frame[split_col] == split_value].copy()
    if sample.empty:
        raise ModelingError(f"calibration split has no rows: {split_col}={split_value}")

    scorer = _ModelArtifactScorer(
        artifact,
        base_dir=_artifact_model_base_dir(runtime, artifact),
        load_calibration=False,
    )
    raw_scores = np.asarray(scorer.score(sample, use_calibration=False), dtype=float)
    labels = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw_scores) & np.isfinite(labels) & np.isin(labels, [0.0, 1.0])
    raw_scores = raw_scores[valid]
    labels = labels[valid].astype(int)
    if labels.size < min_samples:
        raise ModelingError(
            f"calibration sample has {labels.size} valid labeled rows; require at least {min_samples}"
        )
    if np.unique(labels).size < 2:
        raise ModelingError("calibration sample must contain both positive and negative labels")

    calibrator = _fit_calibrator(method, raw_scores, labels)
    calibrated_scores = _apply_calibrator(method, calibrator, raw_scores)
    raw_metrics = _calibration_metrics(labels, raw_scores, n_bins=n_bins)
    calibrated_metrics = _calibration_metrics(labels, calibrated_scores, n_bins=n_bins)
    reliability_curve = _calibration_curve_rows(
        labels,
        raw_scores,
        calibrated_scores,
        n_bins=n_bins,
    )

    base_dir = _artifact_model_base_dir(runtime, artifact)
    calibration_path = f"{artifact.id}.calibration.{method}.joblib"
    calibration_payload = {
        "method": method,
        "calibrator": calibrator,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_artifact_file(base_dir, calibration_path, lambda path: joblib.dump(calibration_payload, path))
    calibration = {
        "method": method,
        "path": calibration_path,
        "dataset_id": dataset.id,
        "target_col": target_col,
        "split_col": split_col,
        "split": split_name,
        "split_value": split_value,
        "sample_count": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "brier_raw": raw_metrics["brier"],
        "brier_calibrated": calibrated_metrics["brier"],
        "ece_raw": raw_metrics["ece"],
        "ece_calibrated": calibrated_metrics["ece"],
        "n_bins": n_bins,
        "pmml_includes_calibration": False,
        "reliability_curve": reliability_curve,
    }
    params = {**dict(artifact.params or {}), CALIBRATION_PARAMS_KEY: calibration}
    updated_artifact = replace(artifact, params=params)
    try:
        persist_model_meta(base_dir, updated_artifact, config=config)
        runtime.modeling_repo.set_model_artifact_params_with_audit(
            artifact.id,
            params,
            audit={
                "kind": "modeling.artifact.calibrate",
                "target_ref": artifact.id,
                "outcome": "succeeded",
                "detail": {
                    "method": method,
                    "dataset_id": dataset.id,
                    "sample_count": int(labels.size),
                    "calibration_path": calibration_path,
                },
            },
        )
    except Exception:
        (base_dir / calibration_path).unlink(missing_ok=True)
        try:
            persist_model_meta(base_dir, artifact, config=config)
        except Exception:
            pass
        raise
    return {
        "artifact_id": artifact.id,
        "method": method,
        "calibration_path": str(base_dir / calibration_path),
        "split": split_name,
        "split_value": split_value,
        "sample_count": int(labels.size),
        "brier_raw": raw_metrics["brier"],
        "brier_calibrated": calibrated_metrics["brier"],
        "ece_raw": raw_metrics["ece"],
        "ece_calibrated": calibrated_metrics["ece"],
        "pmml_includes_calibration": False,
        "reliability_curve": reliability_curve,
    }


def tool_export_pmml(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    experiment = runtime.experiments.get(artifact.experiment_id)
    _require_pmml_supported(
        artifact,
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    pmml_path = _pmml_path(runtime, artifact)
    return {"pmml_path": str(pmml_path)}


def tool_handoff_to_validation(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    _require_pmml_supported(
        artifact,
        operation="validation handoff",
        base_dir=_artifact_base_dir(runtime.settings, experiment.task_id),
    )
    validation_task_id = handoff_to_validation(
        runtime.experiments,
        artifact,
        sample_dataset_id=str(inputs["sample_dataset_id"]),
        settings=runtime.settings,
    )
    return {"validation_task_id": validation_task_id}


def tool_post_training_action(inputs: dict, ctx) -> dict:
    """Close the modeling workflow with safe delivery actions.

    PMML export and V1 validation handoff are compatibility deliverables, not a
    reason to fail native-only models. Unsupported actions are returned as
    ``skipped`` with a reason so the user can still use the native artifact/report.
    """
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    capabilities = _artifact_capabilities(artifact, base_dir=base_dir)
    selection_policy_decision = _approval_policy_decision(inputs.get("selection_policy_decision"))
    requested_actions = [
        str(item)
        for item in (inputs.get("actions") or ["export_pmml", "handoff_to_validation"])
        if str(item).strip()
    ]
    actions: list[dict] = []
    pmml_path = ""
    validation_task_id = ""
    reason = str(capabilities.get("reason") or "")

    if "export_pmml" in requested_actions:
        if capabilities.get("pmml_supported"):
            pmml_path = str(_pmml_path(runtime, artifact))
            actions.append({"action": "export_pmml", "status": "succeeded", "pmml_path": pmml_path})
        else:
            actions.append({"action": "export_pmml", "status": "skipped", "reason": reason})

    if "handoff_to_validation" in requested_actions:
        sample_dataset_id = str(inputs.get("sample_dataset_id") or "").strip()
        if capabilities.get("handoff_supported") and sample_dataset_id:
            validation_task_id = handoff_to_validation(
                runtime.experiments,
                artifact,
                sample_dataset_id=sample_dataset_id,
                settings=runtime.settings,
            )
            actions.append({
                "action": "handoff_to_validation",
                "status": "succeeded",
                "validation_task_id": validation_task_id,
            })
        else:
            actions.append({
                "action": "handoff_to_validation",
                "status": "skipped",
                "reason": reason or "sample_dataset_id is required for validation handoff",
            })

    approval_package_path = str(_write_approval_package(
        base_dir,
        experiment=experiment,
        artifact=artifact,
        capabilities=capabilities,
        actions=actions,
        sample_dataset_id=str(inputs.get("sample_dataset_id") or ""),
        pmml_path=pmml_path,
        validation_task_id=validation_task_id,
        selection_policy_decision=selection_policy_decision,
    ))
    return {
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "native_model_path": str(artifact.model_path),
        "pmml_path": pmml_path,
        "validation_task_id": validation_task_id,
        "approval_package_path": approval_package_path,
        "capabilities": capabilities,
        "actions": actions,
    }


def _approval_policy_decision(value) -> dict:
    decision = value if isinstance(value, dict) else {}
    if not decision:
        return {}
    violations = []
    for item in decision.get("violations") or []:
        if not isinstance(item, dict):
            continue
        violations.append({
            "code": str(item.get("code") or ""),
            "message": str(item.get("message") or ""),
        })
    return {
        "status": str(decision.get("status") or ""),
        "explicit_selection": bool(decision.get("explicit_selection")),
        "selected_experiment_id": str(decision.get("selected_experiment_id") or ""),
        "policy": _json_safe(decision.get("policy") if isinstance(decision.get("policy"), dict) else {}),
        "profile": _json_safe(decision.get("profile") if isinstance(decision.get("profile"), dict) else {}),
        "violations": violations,
        "override_reason": str(decision.get("override_reason") or ""),
    }


def _write_approval_package(
    base_dir: Path,
    *,
    experiment,
    artifact: ModelArtifact,
    capabilities: dict,
    actions: list[dict],
    sample_dataset_id: str,
    pmml_path: str,
    validation_task_id: str,
    selection_policy_decision: dict,
) -> Path:
    config = experiment.config
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "recipe": experiment.recipe_id,
        "algorithm": artifact.algorithm,
        "target_type": getattr(config, "target_type", "binary"),
        "dataset_id": getattr(config, "dataset_id", ""),
        "sample_dataset_id": sample_dataset_id,
        "target_col": getattr(config, "target_col", ""),
        "split_col": getattr(config, "split_col", ""),
        "split_values": _json_safe(getattr(config, "split_values", {})),
        "seed": getattr(config, "seed", None),
        "features": list(artifact.feature_list),
        "feature_count": len(artifact.feature_list),
        "metrics": _json_safe(experiment.metrics),
        "capabilities": _json_safe(capabilities),
        "selection_policy_decision": selection_policy_decision,
        "delivery_actions": _json_safe(actions),
        "artifacts": {
            "native_model_path": artifact.model_path,
            "pmml_path": pmml_path,
            "validation_task_id": validation_task_id,
        },
        "scorecard_table": _json_safe(_scorecard_table_rows(artifact)),
        "model_params": _json_safe(artifact.params),
    }
    filename = f"{artifact.id}.approval_package.json"
    return write_artifact_file(
        base_dir,
        filename,
        lambda path: path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        ),
    )


def tool_generate_model_report(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    report_path = Path(runtime.settings.tasks_dir) / ctx.task_id / "outputs" / "model_report.xlsx"
    return _generate_model_report_for(runtime, ctx, experiment, inputs, report_path)


def tool_generate_model_reports(inputs: dict, ctx) -> dict:
    """MODELING §5 multi-version fan-out: render one report per requested experiment
    using version-specific output paths. Each report reuses the single-report pipeline.
    report_path mirrors the first report so the existing download endpoint stays
    compatible."""
    runtime = _runtime(ctx)
    experiment_ids = [str(item) for item in inputs.get("experiment_ids") or []]
    if not experiment_ids:
        raise ModelingError("experiment_ids must not be empty")
    outputs_dir = Path(runtime.settings.tasks_dir) / ctx.task_id / "outputs"
    reports: list[dict] = []
    for experiment_id in experiment_ids:
        experiment = runtime.experiments.get(experiment_id)
        recipe = str(experiment.recipe_id)
        report_path = outputs_dir / _report_filename(recipe, experiment_id)
        generated = _generate_model_report_for(runtime, ctx, experiment, inputs, report_path)
        reports.append({
            "experiment_id": experiment_id,
            "recipe": recipe,
            "report_path": generated["report_path"],
        })
    return {
        "reports": reports,
        "report_path": reports[0]["report_path"] if reports else "",
    }


_REPORT_FILENAME_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _report_filename(recipe: str, experiment_id: str) -> str:
    safe_recipe = _REPORT_FILENAME_UNSAFE_RE.sub("_", recipe).strip("_") or "model"
    safe_id = _REPORT_FILENAME_UNSAFE_RE.sub("_", experiment_id)[:8]
    return f"model_report_{safe_recipe}_{safe_id}.xlsx"


def _generate_model_report_for(runtime: _Runtime, ctx, experiment, inputs: dict, report_path: Path) -> dict:
    # The full report is binary-credit-specific (bad-rate / Vintage / OOT bins / stress).
    # For a non-binary target (regression / multiclass) write a compact metrics report so
    # the flow finishes with a downloadable artifact instead of crashing on binary-only math.
    if getattr(experiment.config, "target_type", "binary") != "binary":
        from marvis.output.model_report_minimal import render_minimal_model_report

        statuses = [
            {"section": "汇总", "status": "ok"},
            {"section": "模型指标", "status": "ok"},
        ]
        try:
            render_minimal_model_report(experiment, report_path)
            _write_model_report_audit(
                runtime,
                experiment=experiment,
                report_path=report_path,
                section_status=statuses,
            )
        except Exception:
            report_path.unlink(missing_ok=True)
            raise
        return {
            "report_path": str(report_path),
            "section_status": statuses,
            "scorecard_table": [],
            "score_bands": [],
        }
    artifact = _artifact(runtime, experiment.artifact_id) if experiment.artifact_id else None
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    business = _business_columns(inputs.get("business_columns") or {})
    statuses = resolve_report_sections(
        business,
        _optional_str(inputs.get("feature_dictionary_id")),
    )
    sample = None
    if _section_available(statuses, "sample_analysis") and business.loan_month_col:
        sample = compute_sample_analysis(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            target_col=experiment.config.target_col,
            business=business,
            mob_cols=business.mob_observe_cols,
        )
    vintage = None
    if _section_available(statuses, "vintage") and business.loan_month_col:
        vintage = compute_vintage_report(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            mob_observe_cols=business.mob_observe_cols,
            amount_col=business.loan_amount_col,
        )

    report_dataset_path, score_col = _report_scored_dataset(
        runtime,
        dataset_path,
        artifact,
        experiment.config,
        task_id=ctx.task_id,
    )
    low_pricing = None
    if _section_available(statuses, "low_pricing") and business.interest_rate_col:
        low_pricing = stress_low_pricing(
            runtime.backend,
            report_dataset_path,
            score_col=score_col,
            target_col=experiment.config.target_col,
            interest_rate_col=business.interest_rate_col,
            low_pricing_threshold=None,
        )
    oot_bin = _report_bin_table(
        runtime,
        report_dataset_path,
        score_col=score_col,
        target_col=experiment.config.target_col,
        config=experiment.config,
        business=business,
    )
    feature_dictionary_id = _optional_str(inputs.get("feature_dictionary_id"))
    feature_dictionary = (
        build_feature_dictionary(runtime.backend, feature_dictionary_id, runtime.registry)
        if feature_dictionary_id
        else {}
    )
    feature_importance = _feature_importance_rows(artifact, feature_dictionary=feature_dictionary)
    scorecard_table = _scorecard_table_rows(artifact)
    score_band_col = (
        SCORECARD_POINTS_COL
        if artifact is not None and artifact.algorithm == "scorecard"
        else score_col
    )
    score_bands = _score_band_rows(
        runtime,
        report_dataset_path,
        score_col=score_band_col,
        target_col=experiment.config.target_col,
        config=experiment.config,
    )
    stress_product_removal = _stress_product_removal(runtime, dataset_path, artifact, experiment.config, feature_dictionary)
    split_profile = _dataset_split_profile(
        runtime,
        dataset_path,
        experiment.config,
        window_col=business.loan_month_col,
    )
    calibration = _artifact_calibration_rows(artifact)
    structured_summary = _report_structured_summary(
        project_meta=dict(inputs.get("project_meta") or {}),
        dataset_split=_dataset_split_rows(experiment.metrics, split_profile=split_profile),
        stability=_stability_rows(experiment.metrics),
        sample_analysis=sample,
        vintage=vintage,
        feature_importance=feature_importance,
        scorecard_table=scorecard_table,
        score_bands=score_bands,
        calibration=calibration,
        univariate=_univariate_rows(runtime, dataset_path, artifact, experiment.config),
        oot_bin_table=oot_bin,
        stress_product_removal=stress_product_removal,
        stress_low_pricing=low_pricing,
        section_status=statuses,
    )
    narratives = _guard_no_invented_numbers(
        _draft_report_narratives(
            structured_summary,
            llm_factory=_report_llm_factory(runtime.settings.workspace, _optional_str(inputs.get("model_id"))),
        ),
        structured_summary,
    )
    scored_dataset_path = report_dataset_path if report_dataset_path != dataset_path else None
    try:
        render_model_report(
            ModelReportPayload(
                project_meta=structured_summary["project_meta"],
                dataset_split=structured_summary["dataset_split"],
                stability=structured_summary["stability"],
                sample_analysis=sample,
                vintage=vintage,
                feature_importance=structured_summary["feature_importance"],
                scorecard_table=structured_summary["scorecard_table"],
                score_bands=structured_summary["score_bands"],
                calibration=structured_summary["calibration"],
                univariate=structured_summary["univariate"],
                oot_bin_table=oot_bin,
                stress_product_removal=stress_product_removal,
                stress_low_pricing=low_pricing,
                narratives=narratives,
                section_status=statuses,
            ),
            report_path,
        )
        _write_model_report_audit(
            runtime,
            experiment=experiment,
            report_path=report_path,
            section_status=statuses,
            scored_dataset_path=scored_dataset_path,
        )
    except Exception:
        _cleanup_model_report_outputs(
            report_path=report_path,
            scored_dataset_path=scored_dataset_path,
        )
        raise
    return {
        "report_path": str(report_path),
        "section_status": [_jsonable(status) for status in statuses],
        "scorecard_table": structured_summary["scorecard_table"],
        "score_bands": structured_summary["score_bands"],
        "calibration": structured_summary["calibration"],
    }


def _write_model_report_audit(
    runtime: _Runtime,
    *,
    experiment,
    report_path: Path,
    section_status: list[dict],
    scored_dataset_path: Path | None = None,
) -> None:
    artifact_id = experiment.artifact_id or ""
    runtime.repo.write_audit(
        kind="modeling.report.generated",
        target_ref=experiment.id,
        outcome="succeeded",
        detail={
            "artifact_id": artifact_id,
            "report_path": str(report_path),
            "scored_dataset_path": str(scored_dataset_path) if scored_dataset_path else "",
            "section_status": [_jsonable(status) for status in section_status],
        },
    )


def _cleanup_model_report_outputs(
    *,
    report_path: Path,
    scored_dataset_path: Path | None,
) -> None:
    report_path.unlink(missing_ok=True)
    if scored_dataset_path is not None and scored_dataset_path.name == "model_report_scored.parquet":
        scored_dataset_path.unlink(missing_ok=True)


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.experiments = ExperimentStore(self.settings.db_path)
        self.modeling_repo = ModelingRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _resolve_feature_cols(
    runtime: _Runtime,
    dataset_id: str,
    features,
    *,
    target_col: str,
    split_col: str | None = None,
) -> list[str]:
    provided = [str(item) for item in (features or []) if str(item).strip()]
    if provided:
        return provided
    dataset = runtime.registry.get(str(dataset_id))
    inferred = candidate_numeric_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=str(target_col),
        split_col=split_col,
    )
    if not inferred:
        raise ModelingError("未找到可用候选特征列;请检查拼接结果或指定特征列。")
    return inferred


def _train_recipe(
    recipe: str,
    backend,
    dataset_path: Path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    if recipe == "lgb":
        return train_lgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lgb_regressor":
        return train_lgb_regressor(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lgb_multiclass":
        return train_lgb_multiclass(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "xgb":
        return train_xgb(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "catboost":
        return train_catboost(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "lr":
        return train_lr(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "scorecard":
        return train_scorecard(backend, dataset_path, config, out_dir=out_dir)
    if recipe == "mlp":
        return train_mlp(backend, dataset_path, config, out_dir=out_dir)
    raise ModelingError(f"unsupported modeling recipe: {recipe}")


def _artifact(runtime: _Runtime, artifact_id: str) -> ModelArtifact:
    artifact = runtime.modeling_repo.get_model_artifact(artifact_id)
    if artifact is None:
        raise ModelingError(f"model artifact not found: {artifact_id}")
    return artifact


def _artifact_capabilities(artifact: ModelArtifact, *, base_dir: Path | None = None) -> dict:
    pmml_supported, payload_reason = _pmml_payload_support(artifact, base_dir=base_dir)
    if pmml_supported:
        reason = None
    elif payload_reason:
        reason = payload_reason
    elif artifact.algorithm == "catboost":
        reason = (
            "CatBoost 可保留原生 .pkl 模型和报告;当前 sklearn2pmml/JPMML "
            "不支持 CatBoostClassifier 直接导出 PMML,因此验证移交需使用 lr/lgb/xgb/scorecard。"
        )
    else:
        reason = (
            f"当前 PMML 导出/验证移交支持 lr/lgb/xgb/scorecard;"
            f"{artifact.algorithm} 可保留原生模型文件和报告。"
        )
    return {
        "pmml_supported": pmml_supported,
        "handoff_supported": pmml_supported,
        "native_model_supported": True,
        "reason": reason,
    }


def _require_pmml_supported(
    artifact: ModelArtifact,
    *,
    operation: str = "PMML export",
    base_dir: Path | None = None,
) -> None:
    supported, reason = _pmml_payload_support(artifact, base_dir=base_dir)
    if not supported:
        raise ModelingError(
            f"{operation} currently supports lr/lgb/xgb/scorecard only; got: {artifact.algorithm}. "
            f"{reason or 'Use the native model artifact/report, or retrain/export a supported binary model for V1 validation handoff.'}"
        )


def _pmml_payload_support(artifact: ModelArtifact, *, base_dir: Path | None) -> tuple[bool, str | None]:
    if artifact.algorithm not in PMML_SUPPORTED_ALGORITHMS:
        return False, None
    if base_dir is None:
        return True, None
    try:
        model = load_model(artifact, base_dir=base_dir)
    except Exception as exc:
        return False, f"模型文件无法加载,不能导出 PMML:{exc}"
    if artifact.algorithm == "scorecard":
        if isinstance(model, dict) and "model" in model and "woe_maps" in model:
            return True, None
        return False, "评分卡 PMML 导出需要包含 model 与 woe_maps 的 scorecard payload。"
    if hasattr(model, "fit") and (hasattr(model, "predict_proba") or hasattr(model, "predict")):
        return True, None
    return False, (
        "当前 PMML 导出仅支持 sklearn 兼容模型对象；原生 LightGBM/XGBoost Booster "
        "请保留原生模型或使用专门 JPMML 导出链路。"
    )


def _pmml_path(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    if artifact.pmml_path:
        existing = _resolve_artifact_path(artifact.pmml_path, base_dir=base_dir)
        if existing.exists():
            persist_model_meta(base_dir, artifact, config=experiment.config)
            return existing
    dataset = runtime.registry.get(experiment.config.dataset_id)
    out_path = base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        runtime.registry.resolve_path(dataset.id),
        out_path,
        base_dir=base_dir,
        target_col=experiment.config.target_col,
    )
    try:
        updated_artifact = replace(artifact, pmml_path=pmml_path.name)
        persist_model_meta(base_dir, updated_artifact, config=experiment.config)
        runtime.experiments.set_artifact_pmml_path(artifact.id, pmml_path.name)
    except Exception:
        pmml_path.unlink(missing_ok=True)
        try:
            persist_model_meta(base_dir, artifact, config=experiment.config)
        except Exception:
            pass
        raise
    return pmml_path


def _fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray):
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(x, y)
        return calibrator
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(np.asarray(scores, dtype=float), y)
        return calibrator
    raise ModelingError(f"unsupported calibration method: {method}")


def _apply_calibrator(method: str, calibrator, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if method == "sigmoid":
        calibrated = calibrator.predict_proba(values.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(values)
    else:
        raise ModelingError(f"unsupported calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def _calibration_metrics(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _expected_calibration_error(y, p, n_bins=n_bins),
    }


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> float:
    rows = _calibration_bin_rows(labels, scores, n_bins=n_bins, score_type="")
    total = sum(int(row["sample_count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            (int(row["sample_count"]) / total) * abs(float(row["calibration_gap"]))
            for row in rows
        )
    )


def _calibration_curve_rows(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    *,
    n_bins: int,
) -> list[dict]:
    return [
        *_calibration_bin_rows(labels, raw_scores, n_bins=n_bins, score_type="raw"),
        *_calibration_bin_rows(labels, calibrated_scores, n_bins=n_bins, score_type="calibrated"),
    ]


def _calibration_bin_rows(labels: np.ndarray, scores: np.ndarray, *, n_bins: int, score_type: str) -> list[dict]:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict] = []
    for index in range(int(n_bins)):
        lower = edges[index]
        upper = edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(p[mask]))
        bad_rate = float(np.mean(y[mask]))
        rows.append({
            "score_type": score_type,
            "bin": index + 1,
            "prob_lower": float(lower),
            "prob_upper": float(upper),
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y[mask] == 1.0)),
            "avg_predicted_pd": avg_pred,
            "observed_bad_rate": bad_rate,
            "calibration_gap": avg_pred - bad_rate,
            "abs_gap": abs(avg_pred - bad_rate),
        })
    return rows


def _artifact_calibration_rows(artifact: ModelArtifact | None) -> list[dict]:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return []
    rows = []
    rows.append({
        "score_type": "summary",
        "method": calibration.get("method"),
        "split": calibration.get("split"),
        "split_value": calibration.get("split_value"),
        "sample_count": calibration.get("sample_count"),
        "positive_count": calibration.get("positive_count"),
        "brier_raw": calibration.get("brier_raw"),
        "brier_calibrated": calibration.get("brier_calibrated"),
        "ece_raw": calibration.get("ece_raw"),
        "ece_calibrated": calibration.get("ece_calibrated"),
        "pmml_includes_calibration": calibration.get("pmml_includes_calibration", False),
        "bin": None,
        "prob_lower": None,
        "prob_upper": None,
        "avg_predicted_pd": None,
        "observed_bad_rate": None,
        "calibration_gap": None,
        "abs_gap": None,
    })
    for row in calibration.get("reliability_curve") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "score_type": row.get("score_type"),
            "method": calibration.get("method"),
            "split": calibration.get("split"),
            "split_value": calibration.get("split_value"),
            "sample_count": row.get("sample_count"),
            "positive_count": row.get("positive_count"),
            "brier_raw": None,
            "brier_calibrated": None,
            "ece_raw": None,
            "ece_calibrated": None,
            "pmml_includes_calibration": calibration.get("pmml_includes_calibration", False),
            "bin": row.get("bin"),
            "prob_lower": row.get("prob_lower"),
            "prob_upper": row.get("prob_upper"),
            "avg_predicted_pd": row.get("avg_predicted_pd"),
            "observed_bad_rate": row.get("observed_bad_rate"),
            "calibration_gap": row.get("calibration_gap"),
            "abs_gap": row.get("abs_gap"),
        })
    return rows


def _artifact_calibration_metadata(artifact: ModelArtifact | None) -> dict | None:
    params = getattr(artifact, "params", None)
    if not isinstance(params, dict):
        return None
    calibration = params.get(CALIBRATION_PARAMS_KEY)
    return calibration if isinstance(calibration, dict) else None


def _load_calibration_payload(artifact: ModelArtifact, *, base_dir: Path) -> dict | None:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return None
    calibration_path = _optional_str(calibration.get("path"))
    if not calibration_path:
        return None
    path = _resolve_artifact_path(calibration_path, base_dir=base_dir)
    if not path.exists():
        raise ModelingError(f"calibration file does not exist: {calibration_path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "method" not in payload or "calibrator" not in payload:
        raise ModelingError(f"invalid calibration payload: {calibration_path}")
    return payload


def _artifact_base_dir(settings, task_id: str) -> Path:
    return Path(settings.tasks_dir) / task_id / MODELING_ARTIFACTS_DIR_NAME


def _snapshot_latest_model_meta(base_dir: Path) -> bytes | None:
    path = Path(base_dir) / "model_meta.json"
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _cleanup_unattached_artifact(artifact: ModelArtifact, base_dir: Path, meta_snapshot: bytes | None) -> None:
    base = Path(base_dir)
    for relative in (artifact.model_path, artifact.pmml_path, f"{artifact.id}.model_meta.json"):
        if not relative:
            continue
        try:
            _resolve_artifact_path(str(relative), base_dir=base).unlink(missing_ok=True)
        except OSError:
            pass
    latest = base / "model_meta.json"
    try:
        if meta_snapshot is None:
            latest.unlink(missing_ok=True)
        else:
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_bytes(meta_snapshot)
    except OSError:
        pass


def _resolve_artifact_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _effective_seed(inputs: dict, ctx) -> int:
    if inputs.get("seed") is not None:
        return int(inputs["seed"])
    if getattr(ctx, "seed", None) is not None:
        return int(ctx.seed)
    return DEFAULT_RANDOM_SEED


def _training_params(inputs: dict) -> dict:
    params = dict(inputs.get("params") or {})
    return {**params, **_training_control_params(inputs, params)}


def _training_control_params(inputs: dict, params: dict | None = None) -> dict:
    params = dict(params or {})
    controls = {}
    for key in ("sample_weight_col", "sample_weight_column", "weight_col"):
        value = inputs.get(key, params.get(key))
        if value not in (None, ""):
            controls["sample_weight_col"] = str(value).strip()
            break
    constraints = inputs.get(
        "monotone_constraints",
        inputs.get(
            "monotonic_constraints",
            params.get("monotone_constraints", params.get("monotonic_constraints")),
        ),
    )
    if constraints not in (None, ""):
        controls["monotone_constraints"] = constraints
    return controls


def _normalize_recipe_list(value) -> list[str]:
    recipes = _unique_strings(value if isinstance(value, list) else [value])
    if not recipes:
        recipes = ["lgb"]
    unsupported = [recipe for recipe in recipes if recipe not in SUPPORTED_MODELING_RECIPES]
    if unsupported:
        raise ModelingError(
            f"unsupported modeling recipe(s): {', '.join(unsupported)}; "
            f"available: {', '.join(sorted(SUPPORTED_MODELING_RECIPES))}"
        )
    _target_type_from_recipes(recipes)
    return recipes


def _target_type_from_recipes(recipes: list[str]) -> str:
    has_binary = any(recipe in BINARY_MODELING_RECIPES for recipe in recipes)
    has_continuous = any(recipe in CONTINUOUS_MODELING_RECIPES for recipe in recipes)
    has_multiclass = any(recipe in MULTICLASS_MODELING_RECIPES for recipe in recipes)
    family_count = sum(1 for flag in (has_binary, has_continuous, has_multiclass) if flag)
    if family_count > 1:
        raise ModelingError("binary, continuous, and multiclass recipes cannot be mixed in one modeling spec")
    if has_continuous:
        return "continuous"
    if has_multiclass:
        return "multiclass"
    return "binary"


def _normalize_modeling_target_type(value) -> str | None:
    target_type = str(value or "").strip().lower()
    if not target_type:
        return None
    if target_type not in {"binary", "continuous", "multiclass"}:
        raise ModelingError(f"unsupported target_type: {target_type}")
    return target_type


def _metric_policy_for_target_type(target_type: str) -> str:
    if target_type == "continuous":
        return "lower OOT RMSE, fallback lower test RMSE"
    if target_type == "multiclass":
        return "higher OOT macro-AUC, fallback higher test macro-AUC then lower logloss"
    return "higher OOT KS, fallback higher test KS"


def _eligible_algorithms(target_type: str) -> list[str]:
    if target_type == "continuous":
        return sorted(CONTINUOUS_MODELING_RECIPES)
    if target_type == "multiclass":
        return sorted(MULTICLASS_MODELING_RECIPES)
    return sorted(BINARY_MODELING_RECIPES)


def _disabled_algorithms(target_type: str) -> list[dict]:
    disabled = []
    eligible = set(_eligible_algorithms(target_type))
    for recipe in sorted(SUPPORTED_MODELING_RECIPES - eligible):
        disabled.append({
            "recipe": recipe,
            "reason": f"recipe target family does not match `{target_type}`",
        })
    return disabled


def _unique_strings(values) -> list[str]:
    return [value for value in dict.fromkeys(str(item).strip() for item in (values or [])) if value]


def _business_columns(payload: dict) -> BusinessColumns:
    return BusinessColumns(
        loan_month_col=_optional_str(payload.get("loan_month_col")),
        interest_rate_col=_optional_str(payload.get("interest_rate_col")),
        loan_amount_col=_optional_str(payload.get("loan_amount_col")),
        term_col=_optional_str(payload.get("term_col")),
        drawdown_amount_col=_optional_str(payload.get("drawdown_amount_col")),
        credit_limit_col=_optional_str(payload.get("credit_limit_col")),
        mob_observe_cols=tuple(str(item) for item in payload.get("mob_observe_cols") or ()),
    )


def _section_available(statuses, section: str) -> bool:
    return any(status.section == section and status.available for status in statuses)


def _dataset_split_rows(metrics, *, split_profile: dict[str, dict] | None = None) -> list[dict]:
    if metrics is None:
        return []
    split_profile = split_profile or {}
    if metrics.train_rmse is not None:
        return [
            {
                "split": "train",
                **split_profile.get("train", {}),
                "rmse": metrics.train_rmse,
                "mae": metrics.train_mae,
                "r2": metrics.train_r2,
            },
            {
                "split": "test",
                **split_profile.get("test", {}),
                "rmse": metrics.test_rmse,
                "mae": metrics.test_mae,
                "r2": metrics.test_r2,
            },
            {
                "split": "oot",
                **split_profile.get("oot", {}),
                "rmse": metrics.oot_rmse,
                "mae": metrics.oot_mae,
                "r2": metrics.oot_r2,
            },
        ]
    return [
        {"split": "train", **split_profile.get("train", {}), "ks": metrics.train_ks, "auc": metrics.train_auc},
        {"split": "test", **split_profile.get("test", {}), "ks": metrics.test_ks, "auc": metrics.test_auc},
        {"split": "oot", **split_profile.get("oot", {}), "ks": metrics.oot_ks, "auc": metrics.oot_auc},
    ]


def _dataset_split_profile(
    runtime: _Runtime,
    dataset_path: Path,
    config: TrainConfig,
    *,
    window_col: str | None = None,
) -> dict[str, dict]:
    frame = runtime.backend.read_frame(dataset_path, columns=_unique_columns([config.split_col, config.target_col, window_col]))
    target = pd.to_numeric(frame[config.target_col], errors="coerce")
    binary_target = target.dropna().isin([0, 1]).all()
    profile = {}
    for split in ("train", "test", "oot"):
        split_value = config.split_values.get(split, split)
        split_mask = frame[config.split_col] == split_value
        group_target = target[split_mask]
        row = {"sample_count": int(len(group_target))}
        if binary_target:
            row["bad_rate"] = _ratio(float((group_target == 1).sum()), float(len(group_target)))
        if window_col and window_col in frame.columns:
            window_values = sorted(str(value) for value in frame.loc[split_mask, window_col].dropna().unique())
            if window_values:
                row["window_start"] = window_values[0]
                row["window_end"] = window_values[-1]
        profile[split] = row
    return profile


def _unique_columns(values) -> list[str]:
    columns = []
    for value in values:
        if value and str(value) not in columns:
            columns.append(str(value))
    return columns


def _stability_rows(metrics) -> list[dict]:
    if metrics is None:
        return []
    if metrics.train_rmse is not None:
        return [
            {"metric": "rmse_test_minus_train", "value": metrics.overfit_train_test_gap},
            {"metric": "rmse_oot_minus_train", "value": metrics.overfit_train_oot_gap},
            {"metric": "overfit_flag", "value": metrics.overfit_flag},
        ]
    return [
        {"metric": "psi_test_vs_train", "value": metrics.psi_test_vs_train},
        {"metric": "psi_oot_vs_train", "value": metrics.psi_oot_vs_train},
        {"metric": "overfit_flag", "value": metrics.overfit_flag},
    ]


def _feature_importance_rows(artifact: ModelArtifact | None, *, feature_dictionary: dict | None = None) -> list[dict]:
    if artifact is None:
        return []
    dictionary = feature_dictionary or {}
    metadata_keys = ("含义", "产品名称", "厂商名称")
    importance_pairs = artifact.feature_importance or tuple((feature, 0.0) for feature in artifact.feature_list)
    total_importance = sum(float(importance) for _, importance in importance_pairs)
    cumulative_importance = 0.0
    rows = []
    for feature, importance in importance_pairs:
        importance_value = float(importance)
        cumulative_importance += importance_value
        row = {
            "feature": feature,
            "importance": importance_value,
            "importance_pct": _ratio(importance_value, total_importance),
            "cumulative_importance_pct": _ratio(cumulative_importance, total_importance),
        }
        if dictionary:
            metadata = dictionary.get(str(feature))
            row.update({
                key: metadata.get(key) if isinstance(metadata, dict) and metadata.get(key) not in ("",) else None
                for key in metadata_keys
            })
        rows.append(row)
    return rows


def _scorecard_table_rows(artifact: ModelArtifact | None) -> list[dict]:
    if artifact is None or artifact.algorithm != "scorecard":
        return []
    if not artifact.scorecard_table:
        return [{
            "feature": "__missing__",
            "bin_label": "旧 artifact 未包含评分卡表,需重训或回填后查看 points 明细",
            "points": None,
        }]
    return [dict(row) for row in artifact.scorecard_table]


def _score_band_rows(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    config: TrainConfig,
    bin_count: int = 10,
) -> list[dict]:
    columns = _unique_columns([score_col, target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    from marvis.validation.binning import assign_bins, equal_frequency_bin_edges

    rows: list[dict] = []
    for split_name, split_value in config.split_values.items():
        split_frame = frame[frame[config.split_col] == split_value]
        if split_frame.empty:
            continue
        scores = pd.to_numeric(split_frame[score_col], errors="coerce").to_numpy(dtype=float)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            continue
        edges = equal_frequency_bin_edges(finite_scores, int(bin_count))
        assigned = assign_bins(scores, edges)
        labels = pd.to_numeric(split_frame[target_col], errors="coerce").to_numpy(dtype=float)
        for bin_index in range(1, len(edges)):
            mask = assigned == bin_index
            if not np.any(mask):
                continue
            label_mask = mask & np.isfinite(labels)
            labeled_count = int(np.sum(label_mask))
            bad_count = int(np.sum(labels[label_mask].astype(int))) if labeled_count else None
            rows.append({
                "split": split_name,
                "bin": int(bin_index),
                "score_lower": float(edges[bin_index - 1]) if np.isfinite(edges[bin_index - 1]) else None,
                "score_upper": float(edges[bin_index]) if np.isfinite(edges[bin_index]) else None,
                "sample_count": int(np.sum(mask)),
                "labeled_count": labeled_count,
                "bad_count": bad_count,
                "bad_rate": (bad_count / labeled_count) if labeled_count and bad_count is not None else None,
                "avg_score": float(np.mean(scores[mask])),
            })
    return rows


def _univariate_rows(runtime: _Runtime, dataset_path: Path, artifact, config: TrainConfig) -> list[dict]:
    if artifact is None:
        return []
    frame = runtime.backend.read_frame(dataset_path, columns=[*artifact.feature_list, config.target_col, config.split_col])
    rows = []
    for feature in artifact.feature_list:
        for split_name, split_value in config.split_values.items():
            split_frame = frame[frame[config.split_col] == split_value]
            if split_frame.empty:
                continue
            target_series = pd.to_numeric(split_frame[config.target_col], errors="coerce")
            if target_series.notna().sum() == 0:
                # Scoring-only split (no labels): skip univariate label metrics for it.
                continue
            metrics = feature_metrics(
                split_frame[feature].to_numpy(dtype=float),
                target_series.to_numpy(dtype=float),
                feature=feature,
            )
            rows.append({
                "feature": feature,
                "split": split_name,
                "iv": metrics.iv,
                "ks": metrics.ks,
                "auc": metrics.auc,
                "sample_count": int(len(split_frame)),
                "coverage": 1.0 - metrics.missing_rate,
                "missing_rate": metrics.missing_rate,
                "unique_count": metrics.unique_count,
            })
    return rows


def _stress_product_removal(
    runtime: _Runtime,
    dataset_path: Path,
    artifact: ModelArtifact | None,
    config: TrainConfig,
    feature_dictionary: dict,
) -> dict:
    if artifact is None or not feature_dictionary:
        return {}
    categories = _stress_feature_categories(feature_dictionary, artifact.feature_list)
    if not categories:
        return {}
    columns = _unique_columns([*artifact.feature_list, config.target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    oot_value = config.split_values.get("oot", "oot")
    oot_sample = frame[frame[config.split_col] == oot_value]
    if oot_sample.empty:
        return {"baseline": {"status": "skipped", "reason": "OOT sample is required for stress test"}}
    result = run_stress_test(
        oot_sample=oot_sample,
        config=ValidationConfig(
            target_col=config.target_col,
            score_col=MODEL_REPORT_SCORE_COL,
            split_col=config.split_col,
            time_col=str(config.params.get("time_col") or "apply_month"),
            feature_columns=list(artifact.feature_list),
            bin_count=10,
            random_seed=config.seed,
            split_values={key: str(value) for key, value in config.split_values.items()},
        ),
        feature_categories=categories,
        input_scorer=_ModelArtifactScorer(artifact, base_dir=_artifact_model_base_dir(runtime, artifact)),
    )
    return _stress_product_rows(result)


def _report_scored_dataset(
    runtime: _Runtime,
    dataset_path: Path,
    artifact: ModelArtifact | None,
    config: TrainConfig,
    *,
    task_id: str,
) -> tuple[Path, str]:
    columns = runtime.backend.column_names(dataset_path)
    if "score" in columns:
        return dataset_path, "score"
    if artifact is None:
        return dataset_path, _report_score_col(runtime, dataset_path, artifact, config)

    frame = runtime.backend.read_frame(dataset_path)
    scorer = _ModelArtifactScorer(artifact, base_dir=_artifact_model_base_dir(runtime, artifact))
    frame[MODEL_REPORT_SCORE_COL] = scorer.score(frame)
    scorecard_points = scorer.scorecard_points(frame)
    if scorecard_points is not None:
        frame[SCORECARD_POINTS_COL] = scorecard_points
    out_path = Path(runtime.settings.tasks_dir) / task_id / "outputs" / "model_report_scored.parquet"
    artifact = TransactionalArtifactStore(out_path.parent).stage(out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
        final_path = artifact.promote()
        artifact.commit()
        return final_path, MODEL_REPORT_SCORE_COL
    except Exception:
        artifact.rollback()
        raise


def _artifact_model_base_dir(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    return _artifact_base_dir(runtime.settings, experiment.task_id)


def _stress_feature_categories(feature_dictionary: dict, feature_list: tuple[str, ...]) -> dict[str, list[str]]:
    allowed = set(feature_list)
    categories: dict[str, list[str]] = {}
    for feature, metadata in feature_dictionary.items():
        if feature not in allowed or not isinstance(metadata, dict):
            continue
        product = _optional_str(metadata.get("产品名称"))
        if not product:
            continue
        categories.setdefault(product, []).append(str(feature))
    return categories


def _stress_product_rows(result) -> dict:
    rows = {
        "baseline": {
            "status": result.status,
            "sample_count": result.baseline.sample_count,
            "ks": result.baseline.ks,
            "dropped_features": "",
            "dropped_feature_count": "",
            "ks_after": "",
            "ks_delta": "",
            "psi_vs_baseline": "",
            "error": "",
        }
    }
    for row in result.per_category:
        rows[row.category] = {
            "status": row.status,
            "sample_count": result.baseline.sample_count,
            "ks": result.baseline.ks,
            "dropped_features": ", ".join(row.dropped_features),
            "dropped_feature_count": len(row.dropped_features),
            "ks_after": row.ks_after,
            "ks_delta": row.ks_delta,
            "psi_vs_baseline": row.psi_vs_baseline,
            "error": row.error or "",
        }
    return rows


class _ModelArtifactScorer:
    def __init__(self, artifact: ModelArtifact, *, base_dir: Path, load_calibration: bool = True):
        self.artifact = artifact
        self.base_dir = Path(base_dir)
        self.model = load_model(artifact, base_dir=base_dir)
        self.calibration = (
            _load_calibration_payload(artifact, base_dir=self.base_dir)
            if load_calibration
            else None
        )

    def score(self, dataframe: pd.DataFrame, *, use_calibration: bool = True) -> list[float]:
        scores = np.asarray(self.raw_score(dataframe), dtype=float)
        if use_calibration and self.calibration is not None:
            scores = _apply_calibrator(str(self.calibration["method"]), self.calibration["calibrator"], scores)
        return [float(value) for value in scores]

    def raw_score(self, dataframe: pd.DataFrame) -> list[float]:
        features = list(self.artifact.feature_list)
        if self.artifact.algorithm == "xgb" and not hasattr(self.model, "predict_proba"):
            import xgboost as xgb

            matrix = xgb.DMatrix(dataframe[features], feature_names=features)
            return [float(value) for value in self.model.predict(matrix)]
        if self.artifact.algorithm == "scorecard" and isinstance(self.model, dict):
            encoded = pd.DataFrame(index=dataframe.index)
            woe_maps = self.model["woe_maps"]
            for feature in features:
                encoded[feature] = woe_encode(dataframe, feature, woe_maps[feature]).to_numpy(dtype=float)
            return [float(value) for value in self.model["model"].predict_proba(encoded)[:, 1]]
        if hasattr(self.model, "predict_proba"):
            return [float(value) for value in self.model.predict_proba(dataframe[features])[:, 1]]
        return [float(value) for value in self.model.predict(dataframe[features])]

    def scorecard_points(self, dataframe: pd.DataFrame) -> list[float] | None:
        if self.artifact.algorithm != "scorecard" or not isinstance(self.model, dict):
            return None
        params = dict(self.model.get("params") or {})
        if "factor" not in params or "offset" not in params:
            return None
        features = list(self.artifact.feature_list)
        encoded = pd.DataFrame(index=dataframe.index)
        woe_maps = self.model["woe_maps"]
        for feature in features:
            encoded[feature] = woe_encode(dataframe, feature, woe_maps[feature]).to_numpy(dtype=float)
        logits = (
            float(self.model["model"].intercept_[0])
            + encoded.to_numpy(dtype=float) @ self.model["model"].coef_[0]
        )
        scores = float(params["offset"]) - float(params["factor"]) * logits
        return [float(value) for value in scores]


def _report_score_col(runtime: _Runtime, dataset_path: Path, artifact, config: TrainConfig) -> str:
    columns = runtime.backend.column_names(dataset_path)
    if "score" in columns:
        return "score"
    if artifact and artifact.feature_list:
        return artifact.feature_list[0]
    return config.features[0]


def _report_bin_table(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    config: TrainConfig,
    business: BusinessColumns,
) -> list[dict]:
    columns = _unique_columns([score_col, target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    oot_value = config.split_values.get("oot", "oot")
    oot_frame = frame[frame[config.split_col] == oot_value]
    if oot_frame.empty:
        return []
    from marvis.validation.binning import equal_frequency_bin_edges

    edges = equal_frequency_bin_edges(oot_frame[score_col].to_numpy(dtype=float), 10)
    return compute_amount_bin_table(
        runtime.backend,
        dataset_path,
        score_col=score_col,
        target_col=target_col,
        edges=edges,
        business=business,
        filters={config.split_col: oot_value},
    )


def _report_structured_summary(**payload) -> dict:
    return _jsonable(payload)


REPORT_NARRATIVE_SYS = (
    "你为信贷风控建模报告起草章节文字。只能解释用户提供的结构化摘要，"
    "不得编造任何数字、百分比、阈值、金额或样本量。输出 JSON object。"
)
REPORT_NARRATIVE_KEYS = ("sample", "vintage", "model", "stress")
REPORT_NUMERIC_EVIDENCE_KEYS = (
    "dataset_split",
    "stability",
    "sample_analysis",
    "vintage",
    "feature_importance",
    "scorecard_table",
    "score_bands",
    "calibration",
    "univariate",
    "oot_bin_table",
    "stress_product_removal",
    "stress_low_pricing",
)


def _draft_report_narratives(structured_summary: dict, *, llm_factory=None) -> dict:
    fallback = _fallback_report_narratives()
    if llm_factory is None:
        return fallback
    try:
        raw = llm_factory().complete(
            system_prompt=REPORT_NARRATIVE_SYS,
            user_prompt=_report_narrative_prompt(structured_summary),
            response_format={"type": "json_object"},
            stream=False,
        )
        payload = json.loads(str(raw))
    except (LLMClientError, LLMSettingsError, json.JSONDecodeError, TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {
        key: str(payload.get(key) or fallback[key])
        for key in REPORT_NARRATIVE_KEYS
    }


def _fallback_report_narratives() -> dict:
    return {
        "sample": "样本分析基于平台聚合结果生成。",
        "vintage": "Vintage 结论基于平台计算曲线生成。",
        "model": "模型结论基于平台指标与特征重要性生成。",
        "stress": "压力测试结论基于平台压测结果生成。",
    }


def _report_narrative_prompt(structured_summary: dict) -> str:
    return (
        "请基于以下结构化摘要，输出 JSON："
        "{sample, vintage, model, stress}。\n"
        "要求：只写文字解释；所有数字必须来自摘要原文；缺少数据时说明缺业务数据。\n\n"
        f"结构化摘要：\n{json.dumps(structured_summary, ensure_ascii=False, sort_keys=True)}"
    )


def _report_llm_factory(workspace: Path, model_id: str | None):
    def factory():
        return OpenAICompatibleLLMClient(resolve_llm_model(workspace, model_id))

    return factory


_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")


def _guard_no_invented_numbers(narratives: dict, structured_summary: dict) -> dict:
    allowed = _allowed_number_tokens(_report_numeric_evidence(structured_summary))
    guarded: dict[str, str] = {}
    for key, value in narratives.items():
        text = str(value)
        guarded[str(key)] = _NUMBER_TOKEN_RE.sub(
            lambda match: match.group(0) if _number_token_allowed(match.group(0), allowed) else "[平台未提供该数字]",
            text,
        )
    return guarded


def _report_numeric_evidence(structured_summary: dict) -> dict:
    return {
        key: structured_summary.get(key)
        for key in REPORT_NUMERIC_EVIDENCE_KEYS
        if key in structured_summary
    }


def _allowed_number_tokens(value) -> set[str]:
    tokens: set[str] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float, np.integer, np.floating)):
            numeric = float(item)
            tokens.add(_format_number_token(numeric))
            tokens.add(str(item))
            return
        if isinstance(item, str):
            for match in _NUMBER_TOKEN_RE.finditer(item):
                tokens.add(match.group(0))

    visit(value)
    return {token for token in tokens if token}


def _number_token_allowed(token: str, allowed: set[str]) -> bool:
    if token in allowed:
        return True
    if token.endswith("%"):
        return False
    try:
        numeric = float(token)
    except ValueError:
        return False
    return _format_number_token(numeric) in allowed


def _format_number_token(value: float) -> str:
    return f"{value:.12g}"


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _jsonable(value: Any):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any):
    """Like _jsonable, but additionally maps NaN/inf to None so the payload is strict
    JSON (no NaN/Infinity tokens) for the make_split sample analysis."""
    cleaned = _jsonable(value)
    return _strip_non_finite(cleaned)


def _strip_non_finite(value: Any):
    if isinstance(value, dict):
        return {key: _strip_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_non_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = [
    "tool_calibrate_model",
    "tool_check_data_quality",
    "tool_compare_experiments",
    "tool_export_pmml",
    "tool_handoff_to_validation",
    "tool_generate_model_report",
    "tool_generate_model_reports",
    "tool_make_split",
    "tool_modeling_readiness",
    "tool_prepare_modeling_frame",
    "tool_reject_inference",
    "tool_select_features",
    "tool_train_model",
]
