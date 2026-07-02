from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork, TransactionalArtifactStore
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.feature.candidates import (
    candidate_numeric_features,
    excluded_categorical_columns,
    suspected_categorical_columns,
)
from marvis.feature.metrics import feature_metrics
from marvis.feature.encode import woe_encode
from marvis.feature.screen import sentinel_screen_notice
from marvis.feature.preprocessing import (
    apply_preprocessing_steps,
    read_preprocessing_chain,
    sidecar_path,
)
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.artifact import (
    export_pmml,
    load_model,
    persist_model_meta,
    validate_scorecard_pmml_payload,
)
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.defaults import DEFAULT_RANDOM_SEED
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.handoff import create_challenger_backtest_task, handoff_to_validation
from marvis.modeling_policy_signals import has_monotonic_policy, monotonic_policy_profile
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
from marvis.packs.modeling.tune import DEFAULT_TRIAL_BUDGET, tune_hyperparameters
from marvis.packs.modeling.errors import ModelingError, ReportScoreMissingError
from marvis.settings import build_settings
from marvis.validation.config import ValidationConfig
from marvis.validation.stress_test import run_stress_test


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"
MODEL_REPORT_SCORE_COL = "__model_score__"
SCORECARD_POINTS_COL = "__scorecard_points__"
PMML_SUPPORTED_ALGORITHMS = frozenset({"lr", "lgb", "xgb", "scorecard"})
CALIBRATION_PARAMS_KEY = "calibration"
CHALLENGER_COMPARISON_VERSION = "champion_challenger_v1"
MODEL_CARD_VERSION = "model_card_v1"
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
MONITORING_POLICY_VERSION = "model_monitoring_v1"
_POLICY_METRIC_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_POLICY_METRIC_THRESHOLD_SHORTCUTS = {
    "min_oot_ks": ("oot_ks", "min"),
    "min_test_ks": ("test_ks", "min"),
    "min_oot_auc": ("oot_auc", "min"),
    "min_test_auc": ("test_auc", "min"),
    "min_oot_macro_auc": ("oot_macro_auc", "min"),
    "min_test_macro_auc": ("test_macro_auc", "min"),
    "max_oot_rmse": ("oot_rmse", "max"),
    "max_test_rmse": ("test_rmse", "max"),
    "max_oot_logloss": ("oot_logloss", "max"),
    "max_test_logloss": ("test_logloss", "max"),
}
DEFAULT_MONITORING_THRESHOLDS = {
    "psi_test_vs_train": {
        "label": "Test PSI vs Train",
        "metric": "psi_test_vs_train",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "psi_oot_vs_train": {
        "label": "OOT PSI vs Train",
        "metric": "psi_oot_vs_train",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "overfit_train_test_gap": {
        "label": "Train/Test KS gap",
        "metric": "overfit_train_test_gap",
        "direction": "max",
        "warn": 0.08,
        "fail": 0.12,
    },
    "overfit_train_oot_gap": {
        "label": "Train/OOT KS gap",
        "metric": "overfit_train_oot_gap",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.15,
    },
    "oot_ks": {
        "label": "OOT KS",
        "metric": "oot_ks",
        "direction": "min",
        "warn": 0.25,
        "fail": 0.20,
    },
    "oot_auc": {
        "label": "OOT AUC",
        "metric": "oot_auc",
        "direction": "min",
        "warn": 0.65,
        "fail": 0.60,
    },
    "oot_macro_auc": {
        "label": "OOT Macro AUC",
        "metric": "oot_macro_auc",
        "direction": "min",
        "warn": 0.65,
        "fail": 0.60,
    },
    "oot_rmse": {
        "label": "OOT RMSE",
        "metric": "oot_rmse",
        "direction": "max",
        "warn": None,
        "fail": None,
    },
}
DEFAULT_MONITORING_CHECKS_BY_TARGET = {
    "binary": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_ks",
        "oot_auc",
    ),
    "continuous": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_rmse",
    ),
    "multiclass": (
        "psi_test_vs_train",
        "psi_oot_vs_train",
        "overfit_train_test_gap",
        "overfit_train_oot_gap",
        "oot_macro_auc",
    ),
}


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
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_dir, f"reject_inference_{uuid.uuid4().hex}.parquet")
    try:
        result.frame.to_parquet(artifact.path, index=False)
        def audit_factory(registered_dataset):
            return {
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
            }

        registered = uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.registry.register_existing_with_audit_on_connection(
                conn,
                artifact.final_path,
                audit_factory=audit_factory,
                task_id=str(ctx.task_id),
                role="reject_inference",
                anchor_target=dataset.id,
                seed=_effective_seed(inputs, ctx),
            ),
        )
    except Exception:
        uow.rollback()
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
    holdout = inputs.get("holdout_values")
    result = select_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        target_type=str(inputs.get("target_type", "binary")),
        iv_min=float(inputs.get("iv_min", 0.02)),
        corr_max=float(inputs.get("corr_max", 0.8)),
        vif_max=float(inputs.get("vif_max", 10.0)),
        top_k=_optional_int(inputs.get("top_k")),
        seed=_effective_seed(inputs, ctx),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        space=str(inputs.get("space") or "raw"),
        split_col=split_col,
        split_value=inputs.get("split_value"),
        holdout_values=tuple(str(v) for v in holdout) if holdout else ("test", "oot"),
        allow_full_fit=bool(inputs.get("allow_full_fit")),
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
        "fit_rows": result.fit_rows,
        "fit_split": result.fit_split,
    }


def tool_screen_features(inputs: dict, ctx) -> dict:
    # feature_ks is a binary-only statistic; a continuous target would miscompute/crash
    # it, so for a non-binary target skip the leakage screen and keep every candidate.
    if str(inputs.get("target_type", "binary")) != "binary":
        return _screen_features_non_binary(inputs, ctx)
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    requested_features = inputs.get("features") or []
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
    )
    excluded_categorical = _excluded_categorical_for_screen(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=_optional_str(inputs.get("split_col")),
    )
    suspected_categorical = _suspected_categorical_for_screen(
        runtime,
        dataset.id,
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
        max_ks_decay=float(inputs["max_ks_decay"]) if inputs.get("max_ks_decay") is not None else None,
    )
    payload = {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "excluded_categorical": excluded_categorical,
    }
    if suspected_categorical:
        payload["suspected_categorical"] = suspected_categorical
    if result.split_shift:
        payload["split_shift"] = [[feature, delta, reason] for feature, delta, reason in result.split_shift]
    if result.leakage_watch:
        payload["leakage_watch"] = [[feature, ks, reason] for feature, ks, reason in result.leakage_watch]
    if result.ks_decay_watch:
        payload["ks_decay_watch"] = [[feature, decay, reason] for feature, decay, reason in result.ks_decay_watch]
    if result.sentinel_columns:
        payload["sentinel_columns"] = _jsonable(result.sentinel_columns)
        payload["sentinel_notice"] = sentinel_screen_notice(result.sentinel_columns)
    return payload


def _excluded_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    requested_features: list,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """String/object columns silently dropped by candidate inference (PREP-3/FS-3).

    Only meaningful when ``features`` was NOT explicitly provided — an explicit
    feature list is the caller's own choice, not an inference the platform made
    on their behalf, so there is nothing to surface."""
    if [str(item) for item in requested_features if str(item).strip()]:
        return []
    dataset = runtime.registry.get(str(dataset_id))
    excluded = excluded_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in excluded]


def _suspected_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """Numeric columns that look like nominal codes rather than continuous measures
    (PREP-5), e.g. a zip/industry code — surfaced as a screen-gate hint, always (even
    with an explicit feature list) since these columns keep being modeled as continuous
    numeric today; nothing about candidate inference or the selected set changes."""
    dataset = runtime.registry.get(str(dataset_id))
    suspected = suspected_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in suspected]


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
        target_type=str(inputs.get("target_type") or "continuous"),
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
    n_trials_override = _optional_int(inputs.get("n_trials"))
    if n_trials_override is not None and n_trials_override < 1:
        raise ModelingError("n_trials must be at least 1")
    # Per-recipe tuning budget (TUNE-1/SEL-2): every recipe gets its own trial
    # count from DEFAULT_TRIAL_BUDGET (tree recipes 40, lr/scorecard/mlp 12) so a
    # multi-algorithm comparison tunes every candidate, not just lgb. An explicit
    # `n_trials` override applies uniformly to every recipe in the request (the
    # single-recipe case behaves exactly like before: one scalar budget).
    n_trials_by_recipe = {
        item: (n_trials_override if n_trials_override is not None else DEFAULT_TRIAL_BUDGET.get(item, 40))
        for item in recipes
    }
    n_trials = n_trials_by_recipe.get(primary_recipe, 40)
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
        "n_trials_by_recipe": n_trials_by_recipe,
        "params": _jsonable(params),
        "metric_policy": metric_policy,
        "eligible_algorithms": _eligible_algorithms(target_type),
        "disabled_algorithms": _disabled_algorithms(target_type),
        "pmml_supported_algorithms": sorted(PMML_SUPPORTED_ALGORITHMS),
        "warnings": warnings,
        "reason": (
            f"目标类型 `{target_type}`,候选算法 {'/'.join(recipes)},"
            f"主调参算法 `{primary_recipe}`,选择指标 {metric_policy}。"
            f"调参预算(按算法):{', '.join(f'{k}={v}' for k, v in n_trials_by_recipe.items())}。"
        ),
    }


def tool_configure_tuning(inputs: dict, ctx) -> dict:
    """Prepare the tuning configuration for one or more recipes (TUNE-1/SEL-2).

    Every recipe in BINARY_MODELING_RECIPES (lgb/xgb/catboost/lr/scorecard/mlp)
    now runs the two-stage search in tune.py, each with its own budget — total
    search cost is the SUM of each recipe's n_trials (tree recipes default 40,
    lr/scorecard/mlp default 12; see DEFAULT_TRIAL_BUDGET). ``recipe`` stays the
    single-recipe entry point for back-compat: it degrades to a one-element
    ``recipes`` list. An explicit ``n_trials`` overrides every listed recipe's
    budget uniformly; per-recipe overrides can be passed via ``n_trials_by_recipe``.
    """
    recipe = str(inputs.get("recipe") or "lgb")
    recipes = _unique_strings(inputs.get("recipes") or [recipe]) or [recipe]
    target_type = str(inputs.get("target_type") or "binary")
    n_trials_override = _optional_int(inputs.get("n_trials"))
    if n_trials_override is not None and n_trials_override < 1:
        raise ModelingError("n_trials must be at least 1")
    explicit_budgets = {
        str(k): int(v)
        for k, v in dict(inputs.get("n_trials_by_recipe") or {}).items()
        if v is not None
    }
    for item, value in explicit_budgets.items():
        if value < 1:
            raise ModelingError("n_trials must be at least 1")
    sample_weight_col = str(inputs.get("sample_weight_col") or "").strip()
    seed = _effective_seed(inputs, ctx)
    tunable = [item for item in recipes if item in DEFAULT_TRIAL_BUDGET]
    budgets = {
        item: explicit_budgets.get(
            item,
            n_trials_override if n_trials_override is not None else DEFAULT_TRIAL_BUDGET.get(item, 40),
        )
        for item in tunable
    }
    tune_enabled = bool(tunable)
    total_budget = sum(budgets.values())
    params = _training_params(inputs)
    budget_note = ', '.join(f'{item}={budgets[item]}' for item in tunable)
    non_tunable = [item for item in recipes if item not in DEFAULT_TRIAL_BUDGET]
    reason = (
        f"{'/'.join(tunable)} 使用有界两阶段随机搜索(按算法预算:{budget_note};"
        f"多算法总预算=Σ各配方预算={total_budget} 轮)。"
        if tunable else "所选算法暂不支持随机搜索,使用算法默认参数。"
    )
    if non_tunable:
        reason += f" {'/'.join(non_tunable)} 不参与调参,使用算法默认参数。"
    return {
        "recipe": recipe,
        "recipes": recipes,
        "target_type": target_type,
        "tune_enabled": tune_enabled,
        "n_trials": budgets.get(recipe, 0),
        "n_trials_by_recipe": budgets,
        "total_n_trials": total_budget,
        "sample_weight_col": sample_weight_col,
        "seed": seed,
        "params": _jsonable(params),
        "reason": reason,
    }


def tool_tune_hyperparameters(inputs: dict, ctx) -> dict:
    """Two-stage random search, generalised to every BINARY_MODELING_RECIPES
    family (TUNE-1/SEL-2): lgb/xgb/catboost get tree-recipe spaces with early
    stopping against the test split; lr/scorecard/mlp get smaller spaces
    (regularization strength, scorecard bin granularity, mlp architecture).

    ``recipe`` (single, back-compat) stays the default entry point: with one
    recipe, ``best_params``/``best_metrics``/``trials``/``n_trials`` are the
    flat, single-recipe shape unchanged from the historical lgb-only contract.
    Pass ``recipes`` (list) to tune several algorithms in one call — each gets
    its own budget from ``n_trials_by_recipe`` (falling back to
    DEFAULT_TRIAL_BUDGET), and the output additionally carries ``per_recipe``
    (full per-algorithm detail) plus a ``best_params``/``trials`` dict keyed by
    recipe id for ``train_models`` to consume.
    """
    recipe = str(inputs.get("recipe") or "lgb")
    recipes = _unique_strings(inputs.get("recipes") or [recipe]) or [recipe]
    configured_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, configured_params)
    base_params = {**configured_params, **control_params}
    n_trials_override = _optional_int(inputs.get("n_trials"))
    explicit_budgets = {
        str(k): int(v)
        for k, v in dict(inputs.get("n_trials_by_recipe") or {}).items()
        if v is not None
    }

    def _budget_for(item: str) -> int:
        if item in explicit_budgets:
            return explicit_budgets[item]
        if n_trials_override is not None:
            return n_trials_override
        return DEFAULT_TRIAL_BUDGET.get(item, 40)

    non_tunable = [item for item in recipes if item not in DEFAULT_TRIAL_BUDGET]
    tunable = [item for item in recipes if item in DEFAULT_TRIAL_BUDGET]
    per_recipe: dict[str, dict] = {}
    for item in non_tunable:
        per_recipe[item] = {"best_params": _jsonable(base_params), "best_metrics": {}, "n_trials": 0, "trials": []}

    if tunable:
        runtime = _runtime(ctx)
        dataset = runtime.registry.get(str(inputs["dataset_id"]))
        dataset_path = runtime.registry.resolve_path(dataset.id)
        seed = _effective_seed(inputs, ctx)
        for item in tunable:
            result = tune_hyperparameters(
                runtime.backend,
                dataset_path,
                features=[str(f) for f in inputs["features"]],
                target_col=str(inputs["target_col"]),
                split_col=str(inputs["split_col"]),
                split_values=dict(inputs["split_values"]),
                recipe=item,
                n_trials=_budget_for(item),
                # Per-recipe deterministic seed derivation: same base seed always
                # reproduces the same trial sequence per recipe, but different
                # recipes don't share identical RNG draws.
                seed=_recipe_seed(seed, item),
                early_stopping_rounds=int(inputs.get("early_stopping_rounds", 100)),
                max_boost_round=int(inputs.get("max_boost_round", 3000)),
                overfit_penalty=float(inputs.get("overfit_penalty", 0.5)),
                sample_weight_col=control_params.get("sample_weight_col", ""),
                base_params=base_params,
                drop_nan_labels=bool(inputs.get("drop_nan_labels")),
            )
            best_params = {**control_params, **result.best_params}
            per_recipe[item] = {
                "best_params": _jsonable(best_params),
                "best_metrics": _jsonable(result.best_metrics),
                "n_trials": result.n_trials,
                "trials": _jsonable(result.trials),
                "nan_labels_dropped": result.nan_labels_dropped,
            }

    total_nan_dropped = max(
        (int(item.get("nan_labels_dropped") or 0) for item in per_recipe.values()),
        default=0,
    )
    if len(recipes) == 1:
        # Single-recipe back-compat shape: flat best_params/trials, exactly like
        # the historical lgb-only contract.
        only = per_recipe[recipes[0]]
        return {
            "best_params": only["best_params"],
            "best_metrics": only["best_metrics"],
            "n_trials": only["n_trials"],
            "trials": only["trials"],
            "nan_labels_dropped": only.get("nan_labels_dropped", 0),
            "per_recipe": _jsonable(per_recipe),
        }
    return {
        "best_params": {item: per_recipe[item]["best_params"] for item in recipes},
        "best_metrics": {item: per_recipe[item]["best_metrics"] for item in recipes},
        "n_trials": sum(per_recipe[item]["n_trials"] for item in recipes),
        "trials": [trial for item in recipes for trial in per_recipe[item]["trials"]],
        "nan_labels_dropped": total_nan_dropped,
        "per_recipe": _jsonable(per_recipe),
    }


def tool_train_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipe = str(inputs["recipe"])
    train_params = _training_params(inputs)
    preprocessing_steps = _preprocessing_steps_for_training(runtime, dataset.id)
    if preprocessing_steps:
        train_params["preprocessing_steps"] = preprocessing_steps
    elif not _preprocessing_chain_traceable(runtime, dataset.id):
        train_params["preprocessing_chain_traceable"] = False
    config = TrainConfig(
        dataset_id=dataset.id,
        features=tuple(str(item) for item in inputs["features"]),
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]),
        params=train_params,
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


#: Tree recipes that fit on a boosting-round ceiling and support early stopping
#: in train_models' multi-algorithm comparison (TUNE-1/SEL-2 fair-arena policy).
_EARLY_STOPPED_TREE_RECIPES = frozenset({"lgb", "xgb", "catboost"})

#: Early-stopping round count used in train_models when a tree recipe's params
#: were not produced by tune_hyperparameters (e.g. a manually-fixed param dict) —
#: mirrors tune.py's own default so an untuned tree recipe still trains to a
#: real ceiling instead of starving at the recipe's bare default round count.
_TRAIN_MODELS_EARLY_STOPPING_ROUNDS = 100


def _params_by_recipe(tuned_params: dict, recipes: list[str]) -> dict[str, dict] | None:
    """Detect whether ``tuned_params`` is a per-recipe-keyed dict (as produced by
    tool_tune_hyperparameters when called with multiple ``recipes``) vs. the
    legacy flat-params shape (single dict of hyperparameters applied only to the
    lgb slot). A dict counts as per-recipe-keyed when every one of its keys is a
    requested recipe id and every value is itself a dict — real hyperparameter
    names never collide with recipe ids."""
    if not tuned_params or not all(isinstance(v, dict) for v in tuned_params.values()):
        return None
    if not set(tuned_params.keys()) <= set(recipes):
        return None
    return {k: dict(v) for k, v in tuned_params.items()}


def tool_train_models(inputs: dict, ctx) -> dict:
    """Train each requested recipe and return all experiments plus the champion picked by
    overfit-penalized test KS (OOT is reported only, never used to select — mirrors
    tune_hyperparameters' "OOT reports only" policy, DOM-9).

    Fair multi-algorithm arena (TUNE-1/SEL-2): every recipe trains with its own
    tuned params (when ``params`` is the per-recipe dict tool_tune_hyperparameters
    produces for multi-recipe runs) or the legacy flat dict (back-compat: applies
    only to the lgb slot, exactly like before). Tree recipes (lgb/xgb/catboost)
    always train with early stopping against the test split — either the round
    count tuning already resolved, or a default early-stopping window when no
    tuned params were supplied for that recipe. The single-recipe case
    (recipes=[lgb]) behaves like train_model."""
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    recipes = [str(item) for item in inputs["recipes"]]
    tuned_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, tuned_params)
    per_recipe_params = _params_by_recipe(tuned_params, recipes)
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
    preprocessing_steps = read_preprocessing_chain(dataset_path)
    preprocessing_chain_traceable = bool(preprocessing_steps) or sidecar_path(dataset_path).exists()

    experiments: list[dict] = []
    for recipe in recipes:
        if per_recipe_params is not None:
            recipe_params = {**per_recipe_params.get(recipe, {}), **control_params}
        elif recipe == "lgb":
            # legacy flat-params shape: only the lgb slot consumes it (unchanged
            # single-recipe / lgb-only-tuned back-compat behaviour).
            recipe_params = {**tuned_params, **control_params}
        else:
            recipe_params = dict(control_params)
        if preprocessing_steps:
            recipe_params["preprocessing_steps"] = preprocessing_steps
        elif not preprocessing_chain_traceable:
            recipe_params["preprocessing_chain_traceable"] = False
        early_stopping_rounds = (
            _TRAIN_MODELS_EARLY_STOPPING_ROUNDS
            if recipe in _EARLY_STOPPED_TREE_RECIPES
            else None
        )
        config = TrainConfig(
            dataset_id=dataset.id,
            features=features,
            target_col=target_col,
            split_col=split_col,
            split_values=split_values,
            params=recipe_params,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
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


#: Overfit penalty applied to the binary champion-selection score, matching
#: tune.py's ``_trial_score`` objective (``test_ks - penalty * max(0, train_ks - test_ks)``).
_CHAMPION_OVERFIT_PENALTY = 0.5

#: Binary champion selection metric name/basis: OOT is reported but never used to pick
#: a winner (mirrors tune_hyperparameters' "OOT reports only" policy — DOM-9).
BINARY_SELECTION_METRIC = "test_ks(overfit-penalized)"


def _overfit_penalized_test_ks(metrics: dict) -> float:
    """``test_ks - penalty * max(0, train_ks - test_ks)``; ``-inf`` when test_ks is missing.

    OOT is intentionally excluded from the score — using it for champion selection would
    contradict tune_hyperparameters' explicit "OOT metrics are reported for transparency
    but are not used for hyperparameter selection" policy (DOM-9).
    """
    test_ks = metrics.get("test_ks")
    if not isinstance(test_ks, (int, float)):
        return float("-inf")
    train_ks = metrics.get("train_ks")
    gap = float(train_ks) - float(test_ks) if isinstance(train_ks, (int, float)) else 0.0
    return float(test_ks) - _CHAMPION_OVERFIT_PENALTY * max(0.0, gap)


def _pick_best_experiment(experiments: list[dict], *, target_type: str = "binary") -> tuple[dict, str]:
    """Pick the best experiment with the metric family that matches the target.

    Binary maximizes the overfit-penalized test KS (OOT is reported, not selected on —
    DOM-9); regression minimizes OOT/test RMSE; multiclass maximizes OOT/test macro-AUC,
    falling back to minimizing logloss.
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
        return _overfit_penalized_test_ks(experiment.get("metrics") or {})

    return max(experiments, key=score), BINARY_SELECTION_METRIC


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
    runtime.experiments.set_status(selected_id, "selected")
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
    policy = _normalize_selection_policy(policy)
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
    """Pick the best comparison row. Binary maximizes the overfit-penalized test KS —
    OOT is reported but not used for selection, matching tune_hyperparameters (DOM-9)."""
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
    return max(rows, key=_overfit_penalized_test_ks), BINARY_SELECTION_METRIC


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
    metric_thresholds = _normalize_selection_policy_metric_thresholds(source)
    if metric_thresholds:
        policy["metric_thresholds"] = metric_thresholds
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
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None or bool(policy.get("metric_thresholds"))


def _selection_policy_has_hard_requirements(policy: dict) -> bool:
    return any(
        bool(policy.get(key))
        for key in (
            "require_pmml",
            "require_handoff",
            "require_scorecard",
            "require_monotonicity",
        )
    ) or policy.get("max_feature_count") is not None or policy.get("max_oot_psi") is not None or bool(policy.get("metric_thresholds"))


def _selection_policy_decision(row: dict, policy: dict, *, explicit: bool) -> dict:
    policy = _normalize_selection_policy(policy)
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
        missing = profile.get("monotonicity_missing_features")
        if isinstance(missing, list) and missing:
            missing_text = ", ".join(str(item) for item in missing[:8])
            if len(missing) > 8:
                missing_text += ", ..."
            message = f"要求完整单调性证据,但以下特征缺少方向: {missing_text}。"
        else:
            message = "要求声明单调约束,但该候选缺少单调性证据。"
        violations.append({
            "code": "require_monotonicity",
            "message": message,
        })
    max_feature_count = policy.get("max_feature_count")
    feature_count = profile.get("feature_count")
    if isinstance(max_feature_count, int):
        if not isinstance(feature_count, int):
            violations.append({
                "code": "max_feature_count_missing",
                "message": f"要求特征数不超过 {max_feature_count},但该候选缺少特征数证据。",
            })
        elif feature_count > max_feature_count:
            violations.append({
                "code": "max_feature_count",
                "message": f"要求特征数不超过 {max_feature_count},但该候选有 {feature_count} 个特征。",
            })
    max_oot_psi = policy.get("max_oot_psi")
    oot_psi = profile.get("policy_psi_oot_vs_train")
    if isinstance(max_oot_psi, (int, float)):
        if not isinstance(oot_psi, (int, float)):
            violations.append({
                "code": "max_oot_psi_missing",
                "message": (
                    f"要求 OOT PSI 不超过 {_format_number_token(float(max_oot_psi))},"
                    "但该候选缺少 OOT PSI 证据。"
                ),
            })
        elif oot_psi > float(max_oot_psi):
            psi_source = str(profile.get("policy_psi_source") or "psi_oot_vs_train")
            psi_label = "加权 OOT PSI" if psi_source == "weighted_psi_oot_vs_train" else "OOT PSI"
            violations.append({
                "code": "max_oot_psi",
                "message": (
                    f"要求 OOT PSI 不超过 {_format_number_token(float(max_oot_psi))},"
                    f"但该候选{psi_label}为 {_format_number_token(float(oot_psi))}。"
                ),
            })
    violations.extend(_selection_policy_metric_threshold_violations(row, policy.get("metric_thresholds")))
    return violations


def _normalize_selection_policy_metric_thresholds(source: dict) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    raw_thresholds = source.get("metric_thresholds")
    if isinstance(raw_thresholds, dict):
        for raw_metric, raw_spec in raw_thresholds.items():
            metric = _policy_metric_name(raw_metric)
            if not metric or not isinstance(raw_spec, dict):
                continue
            spec: dict[str, float] = {}
            minimum = _finite_float_or_none(raw_spec.get("min"))
            maximum = _finite_float_or_none(raw_spec.get("max"))
            if minimum is not None:
                spec["min"] = minimum
            if maximum is not None:
                spec["max"] = maximum
            if spec:
                thresholds[metric] = spec
    for key, (metric, direction) in _POLICY_METRIC_THRESHOLD_SHORTCUTS.items():
        value = _finite_float_or_none(source.get(key))
        if value is None:
            continue
        thresholds.setdefault(metric, {})[direction] = value
    return thresholds


def _selection_policy_metric_threshold_violations(row: dict, thresholds) -> list[dict]:
    if not isinstance(thresholds, dict) or not thresholds:
        return []
    violations: list[dict] = []
    for metric in sorted(thresholds):
        spec = thresholds.get(metric)
        if not isinstance(spec, dict):
            continue
        value = _finite_float_or_none(row.get(metric))
        if value is None:
            violations.append({
                "code": "metric_threshold_missing",
                "metric": metric,
                "message": f"要求指标 `{metric}` 满足阈值,但该候选缺少该指标证据。",
            })
            continue
        minimum = spec.get("min")
        maximum = spec.get("max")
        if isinstance(minimum, (int, float)) and value < float(minimum):
            violations.append({
                "code": "metric_min_threshold",
                "metric": metric,
                "message": (
                    f"要求 `{metric}` 不低于 {_format_number_token(float(minimum))},"
                    f"但该候选为 {_format_number_token(float(value))}。"
                ),
            })
        if isinstance(maximum, (int, float)) and value > float(maximum):
            violations.append({
                "code": "metric_max_threshold",
                "metric": metric,
                "message": (
                    f"要求 `{metric}` 不超过 {_format_number_token(float(maximum))},"
                    f"但该候选为 {_format_number_token(float(value))}。"
                ),
            })
    return violations


def _policy_metric_name(value) -> str:
    metric = str(value or "").strip()
    if not _POLICY_METRIC_NAME.fullmatch(metric):
        return ""
    return metric


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
    monotonic = monotonic_policy_profile(item, scorecard_rows)
    profile = {
        "recipe": recipe,
        "scorecard": recipe == "scorecard" or bool(scorecard_rows),
        "scorecard_table_rows": len(scorecard_rows),
        "monotonicity_declared": bool(monotonic.get("monotonicity_declared")),
        "monotonicity_coverage": monotonic.get("monotonicity_coverage"),
        "monotonicity_missing_features": monotonic.get("monotonicity_missing_features") or [],
        "monotonicity_constrained_features": monotonic.get("monotonicity_constrained_features") or [],
        "pmml_supported": bool(caps.get("pmml_supported")),
        "handoff_supported": bool(caps.get("handoff_supported")),
        "native_model_supported": bool(caps.get("native_model_supported")),
        "feature_count": feature_count if isinstance(feature_count, int) else None,
    }
    psi_oot = _finite_float_or_none(item.get("psi_oot_vs_train"))
    weighted_psi_oot = _finite_float_or_none(item.get("weighted_psi_oot_vs_train"))
    if psi_oot is not None:
        profile["psi_oot_vs_train"] = psi_oot
    if weighted_psi_oot is not None:
        profile["weighted_psi_oot_vs_train"] = weighted_psi_oot
        profile["policy_psi_oot_vs_train"] = weighted_psi_oot
        profile["policy_psi_source"] = "weighted_psi_oot_vs_train"
    elif psi_oot is not None:
        profile["policy_psi_oot_vs_train"] = psi_oot
        profile["policy_psi_source"] = "psi_oot_vs_train"
    return profile


def _row_has_monotonic_policy(item: dict, scorecard_rows: list) -> bool:
    return has_monotonic_policy(item, scorecard_rows)


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
    return number if np.isfinite(number) and number >= 0 else None


def _finite_float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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
    uow = ArtifactUnitOfWork()
    calibration_artifact = uow.stage_file(base_dir, calibration_path)
    try:
        joblib.dump(calibration_payload, calibration_artifact.path)
    except Exception:
        uow.rollback()
        raise
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
        persist_model_meta(base_dir, updated_artifact, config=config, uow=uow)
    except Exception:
        uow.rollback()
        raise
    audit = {
        "kind": "modeling.artifact.calibrate",
        "target_ref": artifact.id,
        "outcome": "succeeded",
        "detail": {
            "method": method,
            "dataset_id": dataset.id,
            "sample_count": int(labels.size),
            "calibration_path": calibration_path,
        },
    }
    set_params_on_connection = getattr(
        runtime.modeling_repo,
        "set_model_artifact_params_with_audit_on_connection",
        None,
    )
    transaction = getattr(runtime.modeling_repo, "transaction", None)
    if callable(set_params_on_connection) and callable(transaction):
        uow.finalize_with_connection(
            transaction,
            lambda conn: set_params_on_connection(conn, artifact.id, params, audit=audit),
        )
    else:
        uow.finalize(
            lambda: runtime.modeling_repo.set_model_artifact_params_with_audit(
                artifact.id,
                params,
                audit=audit,
            )
        )
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
    monitoring_policy = _monitoring_policy_payload(
        experiment=experiment,
        artifact=artifact,
        source=inputs.get("monitoring_policy"),
        selection_policy_decision=selection_policy_decision,
    )
    challenger_comparison = _challenger_comparison_payload(
        runtime=runtime,
        experiment=experiment,
        artifact=artifact,
        source=inputs.get("champion_reference"),
    )
    requested_actions = [
        str(item)
        for item in (
            inputs.get("actions")
            or ["export_pmml", "handoff_to_validation", "create_challenger_backtest"]
        )
        if str(item).strip()
    ]
    actions: list[dict] = []
    pmml_path = ""
    validation_task_id = ""
    challenger_task_id = ""
    challenger_package_path = ""
    challenger_package_markdown_path = ""
    reason = str(capabilities.get("reason") or "")

    if "export_pmml" in requested_actions:
        if capabilities.get("pmml_supported"):
            pmml_path = str(_pmml_path(runtime, artifact))
            action = {"action": "export_pmml", "status": "succeeded", "pmml_path": pmml_path}
            note = _pmml_delivery_note(capabilities)
            if note:
                action["reason"] = note
            actions.append(action)
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
                "reason": _handoff_delivery_note(capabilities),
            })
        else:
            actions.append({
                "action": "handoff_to_validation",
                "status": "skipped",
                "reason": reason or "sample_dataset_id is required for validation handoff",
            })

    if "create_challenger_backtest" in requested_actions:
        sample_dataset_id = str(inputs.get("sample_dataset_id") or "").strip()
        if capabilities.get("handoff_supported") and sample_dataset_id:
            challenger = create_challenger_backtest_task(
                runtime.experiments,
                artifact,
                sample_dataset_id=sample_dataset_id,
                settings=runtime.settings,
                selection_policy_decision=selection_policy_decision,
                monitoring_policy=monitoring_policy,
                challenger_comparison=challenger_comparison,
            )
            challenger_task_id = challenger["task_id"]
            challenger_package_path = challenger["package_path"]
            challenger_package_markdown_path = challenger["markdown_path"]
            actions.append({
                "action": "create_challenger_backtest",
                "status": "succeeded",
                "challenger_task_id": challenger_task_id,
                "package_path": challenger_package_path,
                "markdown_path": challenger_package_markdown_path,
            })
        else:
            actions.append({
                "action": "create_challenger_backtest",
                "status": "skipped",
                "reason": reason or "sample_dataset_id and PMML-capable model are required",
            })

    model_card = _model_card_payload(
        experiment=experiment,
        artifact=artifact,
        capabilities=capabilities,
        actions=actions,
        sample_dataset_id=str(inputs.get("sample_dataset_id") or ""),
        pmml_path=pmml_path,
        validation_task_id=validation_task_id,
        challenger_task_id=challenger_task_id,
        challenger_package_path=challenger_package_path,
        challenger_package_markdown_path=challenger_package_markdown_path,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
    )
    approval_package = _write_approval_package(
        base_dir,
        experiment=experiment,
        artifact=artifact,
        capabilities=capabilities,
        actions=actions,
        sample_dataset_id=str(inputs.get("sample_dataset_id") or ""),
        pmml_path=pmml_path,
        validation_task_id=validation_task_id,
        challenger_task_id=challenger_task_id,
        challenger_package_path=challenger_package_path,
        challenger_package_markdown_path=challenger_package_markdown_path,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
        model_card=model_card,
    )
    return {
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "native_model_path": str(artifact.model_path),
        "pmml_path": pmml_path,
        "validation_task_id": validation_task_id,
        "challenger_task_id": challenger_task_id,
        "challenger_package_path": challenger_package_path,
        "challenger_package_markdown_path": challenger_package_markdown_path,
        "approval_package_path": str(approval_package["json_path"]),
        "approval_package_markdown_path": str(approval_package["markdown_path"]),
        "monitoring_policy_path": str(approval_package["monitoring_policy_path"]),
        "monitoring_policy_markdown_path": str(approval_package["monitoring_policy_markdown_path"]),
        "monitoring_policy": monitoring_policy,
        "challenger_comparison_path": str(approval_package.get("challenger_comparison_path") or ""),
        "challenger_comparison_markdown_path": str(
            approval_package.get("challenger_comparison_markdown_path") or ""
        ),
        "challenger_comparison": challenger_comparison,
        "model_card_path": str(approval_package["model_card_path"]),
        "model_card_markdown_path": str(approval_package["model_card_markdown_path"]),
        "model_card": model_card,
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


def _monitoring_policy_payload(
    *,
    experiment,
    artifact: ModelArtifact,
    source,
    selection_policy_decision: dict,
) -> dict:
    source_policy = source if isinstance(source, dict) else {}
    target_type = str(getattr(experiment.config, "target_type", "binary") or "binary")
    thresholds = _monitoring_thresholds(source_policy.get("thresholds"), target_type=target_type)
    baseline_metrics = _json_safe(experiment.metrics) or {}
    checks = [
        _monitoring_check_payload(check_id, spec, baseline_metrics)
        for check_id, spec in thresholds.items()
    ]
    overall_status = _monitoring_overall_status(checks)
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
    return {
        "schema_version": 1,
        "policy_version": str(source_policy.get("policy_version") or MONITORING_POLICY_VERSION),
        "created_at": datetime.now(UTC).isoformat(),
        "status": overall_status,
        "recommendation": _monitoring_recommendation(overall_status),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "recipe": experiment.recipe_id,
        "algorithm": artifact.algorithm,
        "target_type": target_type,
        "dataset_id": getattr(experiment.config, "dataset_id", ""),
        "target_col": getattr(experiment.config, "target_col", ""),
        "split_col": getattr(experiment.config, "split_col", ""),
        "baseline_metrics": baseline_metrics,
        "checks": checks,
        "sample_weight_policy": sample_weight_policy,
        "selection_policy_status": str(selection_policy_decision.get("status") or ""),
        "review_cadence": str(source_policy.get("review_cadence") or "monthly"),
        "owner": str(source_policy.get("owner") or "model_risk"),
        "notes": str(source_policy.get("notes") or ""),
    }


def _monitoring_thresholds(source, *, target_type: str) -> dict:
    default_keys = DEFAULT_MONITORING_CHECKS_BY_TARGET.get(target_type, DEFAULT_MONITORING_CHECKS_BY_TARGET["binary"])
    thresholds = {
        key: dict(DEFAULT_MONITORING_THRESHOLDS[key])
        for key in default_keys
        if key in DEFAULT_MONITORING_THRESHOLDS
    }
    if not isinstance(source, dict):
        return thresholds
    for key, override in source.items():
        if not isinstance(override, dict):
            continue
        normalized_key = str(key)
        base = thresholds.get(normalized_key, {
            "label": normalized_key,
            "metric": normalized_key,
            "direction": "max",
            "warn": None,
            "fail": None,
        })
        merged = dict(base)
        for field in ("label", "metric", "direction", "warn", "fail"):
            if field in override:
                merged[field] = override[field]
        thresholds[normalized_key] = merged
    return thresholds


def _monitoring_check_payload(check_id: str, spec: dict, metrics: dict) -> dict:
    metric = str(spec.get("metric") or check_id)
    value = metrics.get(metric)
    direction = str(spec.get("direction") or "max")
    warn = _optional_float(spec.get("warn"))
    fail = _optional_float(spec.get("fail"))
    status, message = _monitoring_check_status(value, direction=direction, warn=warn, fail=fail)
    return {
        "id": str(check_id),
        "label": str(spec.get("label") or check_id),
        "metric": metric,
        "value": value,
        "direction": direction,
        "warn": warn,
        "fail": fail,
        "status": status,
        "message": message,
    }


def _monitoring_check_status(value, *, direction: str, warn: float | None, fail: float | None) -> tuple[str, str]:
    numeric = _optional_float(value)
    if numeric is None:
        return "missing", "指标缺失，需在监控任务中补充基线或跳过原因"
    if warn is None and fail is None:
        return "needs_policy", "缺少业务阈值，需配置 warn/fail 阈值后纳入自动判断"
    if direction == "min":
        if fail is not None and numeric < fail:
            return "fail", f"{_format_number_token(numeric)} 低于 fail 阈值 {_format_number_token(fail)}"
        if warn is not None and numeric < warn:
            return "warn", f"{_format_number_token(numeric)} 低于 warn 阈值 {_format_number_token(warn)}"
    else:
        if fail is not None and numeric > fail:
            return "fail", f"{_format_number_token(numeric)} 高于 fail 阈值 {_format_number_token(fail)}"
        if warn is not None and numeric > warn:
            return "warn", f"{_format_number_token(numeric)} 高于 warn 阈值 {_format_number_token(warn)}"
    return "pass", "基线指标在监控阈值内"


def _monitoring_overall_status(checks: list[dict]) -> str:
    statuses = {str(item.get("status") or "") for item in checks}
    if "fail" in statuses:
        return "fail"
    if statuses & {"warn", "missing", "needs_policy"}:
        return "warn"
    return "pass"


def _monitoring_recommendation(status: str) -> str:
    if status == "pass":
        return "可进入常规监控"
    if status == "fail":
        return "需模型风险复核后再交付"
    return "需补充监控阈值或业务说明"


def _sample_weight_policy_payload(*, experiment, artifact: ModelArtifact) -> dict:
    config_params = getattr(getattr(experiment, "config", None), "params", {})
    config_col = _sample_weight_col_from_params(config_params)
    artifact_col = _sample_weight_col_from_params(artifact.params)
    sample_weight_col = artifact_col or config_col
    used = bool(sample_weight_col)
    source = "artifact_params" if artifact_col else "train_config_params" if config_col else "none"
    approval_items = [
        "训练未使用样本权重；如后续引入拒绝推断、成本权重或抽样校正，需要重新执行筛选、调参、训练和审批。"
    ]
    monitoring_checks: list[dict] = []
    if used:
        approval_items = [
            "确认样本权重列的业务定义、生成逻辑、适用样本范围和取值边界。",
            "审批时同时查看加权与非加权验证指标，确认模型收益不是仅由权重口径驱动。",
            "上线监控需跟踪权重列可用率、非正值占比和分布漂移，权重口径变化时触发重新审批。",
        ]
        monitoring_checks = [
            {
                "id": "sample_weight_availability",
                "metric": f"{sample_weight_col}.missing_or_non_positive_rate",
                "status": "needs_baseline",
                "recommendation": "配置缺失、非正值和异常高权重占比阈值。",
            },
            {
                "id": "sample_weight_distribution",
                "metric": f"{sample_weight_col}.population_stability",
                "status": "needs_baseline",
                "recommendation": "配置训练基线分布并按月监控 PSI/分位数漂移。",
            },
        ]
    return _json_safe({
        "schema_version": 1,
        "used": used,
        "sample_weight_col": sample_weight_col,
        "source": source,
        "approval_policy": {
            "requires_manual_review": used,
            "review_items": approval_items,
        },
        "monitoring_defaults": {
            "requires_monitoring": used,
            "review_cadence": "monthly" if used else "standard",
            "checks": monitoring_checks,
        },
    })


def _sample_weight_col_from_params(params) -> str:
    if not isinstance(params, dict):
        return ""
    for key in ("sample_weight_col", "sample_weight_column", "weight_col"):
        value = str(params.get(key) or "").strip()
        if value:
            return value
    return ""


def _challenger_comparison_payload(
    *,
    runtime: _Runtime,
    experiment,
    artifact: ModelArtifact,
    source,
) -> dict:
    champion = _resolve_champion_reference(runtime, source)
    if not champion and source is None:
        champion = _previous_selected_champion_reference(
            runtime=runtime,
            experiment=experiment,
            artifact=artifact,
        )
    if not champion:
        return {}
    challenger_metrics = _json_safe(experiment.metrics) or {}
    rows = _challenger_metric_comparisons(
        champion.get("metrics") if isinstance(champion.get("metrics"), dict) else {},
        challenger_metrics if isinstance(challenger_metrics, dict) else {},
    )
    summary = _challenger_comparison_summary(rows)
    status = _challenger_comparison_status(summary)
    return _json_safe({
        "schema_version": 1,
        "comparison_version": CHALLENGER_COMPARISON_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "recommendation": _challenger_comparison_recommendation(status, summary),
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "target_type": getattr(experiment.config, "target_type", "binary"),
        "dataset_id": getattr(experiment.config, "dataset_id", ""),
        "challenger": {
            "label": "challenger",
            "experiment_id": experiment.id,
            "artifact_id": artifact.id,
            "recipe": experiment.recipe_id,
            "algorithm": artifact.algorithm,
            "metrics": challenger_metrics,
        },
        "champion": champion,
        "metric_comparisons": rows,
        "summary": summary,
    })


def _resolve_champion_reference(runtime: _Runtime, source) -> dict:
    if not isinstance(source, dict) or source.get("enabled") is False:
        return {}
    experiment_id = str(source.get("experiment_id") or "").strip()
    explicit_metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    label = str(source.get("label") or "prior_champion").strip() or "prior_champion"
    champion_experiment = None
    champion_artifact = None
    if experiment_id:
        try:
            champion_experiment = runtime.experiments.get(experiment_id)
        except KeyError as exc:
            raise ModelingError(f"champion_reference experiment_id not found: {experiment_id}") from exc
        if champion_experiment.artifact_id:
            champion_artifact = _artifact(runtime, champion_experiment.artifact_id)
    if not experiment_id and not explicit_metrics and not str(source.get("artifact_id") or "").strip():
        return {}
    metrics = explicit_metrics or (
        _json_safe(champion_experiment.metrics) if champion_experiment is not None else {}
    )
    return _json_safe({
        "label": label,
        "experiment_id": experiment_id,
        "artifact_id": str(
            source.get("artifact_id")
            or getattr(champion_experiment, "artifact_id", "")
            or ""
        ),
        "recipe": str(source.get("recipe") or getattr(champion_experiment, "recipe_id", "") or ""),
        "algorithm": str(source.get("algorithm") or getattr(champion_artifact, "algorithm", "") or ""),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "notes": str(source.get("notes") or ""),
    })


def _previous_selected_champion_reference(
    *,
    runtime: _Runtime,
    experiment,
    artifact: ModelArtifact,
) -> dict:
    statuses = {"selected", "validated", "delivered", "approved", "champion"}
    current_created_at = str(getattr(experiment, "created_at", "") or "")
    candidates = []
    for candidate in runtime.experiments.list_for_task(experiment.task_id):
        if candidate.id == experiment.id or candidate.artifact_id == artifact.id:
            continue
        if candidate.metrics is None or str(candidate.status or "") not in statuses:
            continue
        candidate_created_at = str(getattr(candidate, "created_at", "") or "")
        if current_created_at and candidate_created_at and candidate_created_at >= current_created_at:
            continue
        candidates.append(candidate)
    if not candidates:
        return {}
    champion_experiment = max(
        candidates,
        key=lambda item: (str(getattr(item, "created_at", "") or ""), item.id),
    )
    champion_artifact = (
        _artifact(runtime, champion_experiment.artifact_id)
        if champion_experiment.artifact_id
        else None
    )
    return _json_safe({
        "label": "previous_selected_experiment",
        "experiment_id": champion_experiment.id,
        "artifact_id": str(champion_experiment.artifact_id or ""),
        "recipe": champion_experiment.recipe_id,
        "algorithm": str(getattr(champion_artifact, "algorithm", "") or ""),
        "metrics": _json_safe(champion_experiment.metrics) or {},
        "notes": "Auto-resolved from an earlier selected experiment in the same task.",
    })


def _challenger_metric_comparisons(champion_metrics: dict, challenger_metrics: dict) -> list[dict]:
    metric_keys = sorted(
        {
            key
            for key in set(champion_metrics) | set(challenger_metrics)
            if _is_metric_key(str(key))
        },
        key=_challenger_metric_sort_key,
    )
    rows: list[dict] = []
    for key in metric_keys:
        metric = str(key)
        champion_value = _numeric_metric(champion_metrics.get(metric))
        challenger_value = _numeric_metric(challenger_metrics.get(metric))
        direction = _metric_better_direction(metric)
        verdict = "missing"
        delta = None
        if champion_value is not None and challenger_value is not None:
            delta = challenger_value - champion_value
            verdict = _metric_verdict(delta, direction)
        rows.append({
            "metric": metric,
            "champion_value": champion_value,
            "challenger_value": challenger_value,
            "delta": delta,
            "direction": direction,
            "verdict": verdict,
        })
    return rows


def _challenger_metric_sort_key(metric: str) -> tuple[int, str]:
    preferred = [
        "oot_ks",
        "test_ks",
        "oot_auc",
        "test_auc",
        "oot_macro_auc",
        "test_macro_auc",
        "oot_accuracy",
        "test_accuracy",
        "oot_r2",
        "test_r2",
        "oot_rmse",
        "test_rmse",
        "oot_mae",
        "test_mae",
        "oot_logloss",
        "test_logloss",
        "psi_oot_vs_train",
        "psi_test_vs_train",
    ]
    try:
        return (preferred.index(metric), metric)
    except ValueError:
        return (len(preferred), metric)


def _numeric_metric(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
        return float(value)
    return None


def _metric_better_direction(metric: str) -> str:
    lowered = metric.lower()
    if (
        lowered.startswith("psi_")
        or "rmse" in lowered
        or "mae" in lowered
        or "logloss" in lowered
        or "loss" in lowered
        or "brier" in lowered
        or "ece" in lowered
        or "gap" in lowered
    ):
        return "lower"
    return "higher"


def _metric_verdict(delta: float, direction: str) -> str:
    if abs(delta) <= 1e-12:
        return "same"
    if direction == "lower":
        return "improved" if delta < 0 else "declined"
    return "improved" if delta > 0 else "declined"


def _challenger_comparison_summary(rows: list[dict]) -> dict:
    comparable = [item for item in rows if item.get("verdict") != "missing"]
    return {
        "metric_count": len(rows),
        "comparable_metric_count": len(comparable),
        "improved_count": sum(1 for item in comparable if item.get("verdict") == "improved"),
        "declined_count": sum(1 for item in comparable if item.get("verdict") == "declined"),
        "same_count": sum(1 for item in comparable if item.get("verdict") == "same"),
        "missing_count": sum(1 for item in rows if item.get("verdict") == "missing"),
    }


def _challenger_comparison_status(summary: dict) -> str:
    if int(summary.get("comparable_metric_count") or 0) <= 0:
        return "missing"
    if int(summary.get("declined_count") or 0) > 0:
        return "warn"
    return "pass"


def _challenger_comparison_recommendation(status: str, summary: dict) -> str:
    if status == "pass":
        return "Challenger 不弱于 Champion 的已配置指标"
    if status == "warn":
        return (
            "Challenger 有指标弱于 Champion, 需业务复核差异 "
            f"({summary.get('declined_count', 0)} 项下降)"
        )
    return "缺少可比较的 Champion/Challenger 指标, 需补充生产模型基线"


def _model_card_payload(
    *,
    experiment,
    artifact: ModelArtifact,
    capabilities: dict,
    actions: list[dict],
    sample_dataset_id: str,
    pmml_path: str,
    validation_task_id: str,
    challenger_task_id: str,
    challenger_package_path: str,
    challenger_package_markdown_path: str,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
) -> dict:
    config = experiment.config
    metrics = _json_safe(experiment.metrics) or {}
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
    selection_policy = (
        selection_policy_decision.get("policy")
        if isinstance(selection_policy_decision.get("policy"), dict)
        else {}
    )
    selection_policy_requirements = [
        {"requirement": label, "configured": value}
        for label, value in _selection_policy_requirement_markdown_rows(selection_policy)
    ]
    limitations = _model_card_limitations(
        capabilities=capabilities,
        selection_policy_decision=selection_policy_decision,
        monitoring_policy=monitoring_policy,
        challenger_comparison=challenger_comparison,
    )
    return _json_safe({
        "schema_version": 1,
        "card_version": MODEL_CARD_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "title": f"{artifact.algorithm} model card",
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
        "feature_count": len(artifact.feature_list),
        "feature_preview": list(artifact.feature_list[:30]),
        "sample_weight_col": str(sample_weight_policy.get("sample_weight_col") or ""),
        "training": {
            "sample_weight": sample_weight_policy,
        },
        "key_metrics": _model_card_key_metrics(metrics),
        "governance": {
            "selection_policy_status": str(selection_policy_decision.get("status") or "not_requested"),
            "selection_policy": _json_safe(selection_policy),
            "selection_policy_requirements": selection_policy_requirements,
            "selection_policy_violations": _json_safe(selection_policy_decision.get("violations") or []),
            "selection_policy_override_reason": str(selection_policy_decision.get("override_reason") or ""),
            "monitoring_status": str(monitoring_policy.get("status") or "not_configured"),
            "monitoring_recommendation": str(monitoring_policy.get("recommendation") or ""),
            "champion_comparison_status": str(challenger_comparison.get("status") or "not_configured"),
            "champion_comparison_recommendation": str(challenger_comparison.get("recommendation") or ""),
        },
        "delivery": {
            "native_model_path": str(artifact.model_path or ""),
            "pmml_path": str(pmml_path or ""),
            "pmml_includes_calibration": capabilities.get("pmml_includes_calibration", True),
            "calibration": _json_safe(capabilities.get("calibration") or {}),
            "validation_task_id": str(validation_task_id or ""),
            "challenger_task_id": str(challenger_task_id or ""),
            "challenger_package_path": str(challenger_package_path or ""),
            "challenger_package_markdown_path": str(challenger_package_markdown_path or ""),
            "export_pmml_status": _model_card_action_status(actions, "export_pmml"),
            "validation_handoff_status": _model_card_action_status(actions, "handoff_to_validation"),
            "challenger_backtest_status": _model_card_action_status(actions, "create_challenger_backtest"),
        },
        "capabilities": _json_safe(capabilities),
        "limitations": limitations,
        "next_review_actions": _model_card_next_review_actions(
            limitations,
            monitoring_policy,
            challenger_comparison,
            sample_weight_policy,
        ),
    })


def _model_card_key_metrics(metrics: dict) -> list[dict]:
    rows = []
    for metric in [
        "oot_ks",
        "test_ks",
        "train_ks",
        "oot_auc",
        "test_auc",
        "oot_macro_auc",
        "test_macro_auc",
        "oot_accuracy",
        "test_accuracy",
        "oot_rmse",
        "test_rmse",
        "oot_mae",
        "test_mae",
        "oot_logloss",
        "test_logloss",
        "psi_oot_vs_train",
        "psi_test_vs_train",
        "overfit_flag",
    ]:
        if metric in metrics and metrics.get(metric) is not None:
            rows.append({"metric": metric, "value": metrics.get(metric)})
    return rows


def _model_card_action_status(actions: list[dict], action: str) -> str:
    for item in actions:
        if isinstance(item, dict) and item.get("action") == action:
            return str(item.get("status") or "")
    return "not_requested"


def _model_card_limitations(
    *,
    capabilities: dict,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
) -> list[str]:
    limitations: list[str] = [
        str(item)
        for item in (capabilities.get("limitations") or [])
        if str(item)
    ]
    if not capabilities.get("pmml_supported"):
        reason = str(capabilities.get("reason") or "PMML export is not supported for this artifact.")
        limitations.append(reason)
    calibration = capabilities.get("calibration") if isinstance(capabilities.get("calibration"), dict) else {}
    if calibration and capabilities.get("pmml_includes_calibration") is False:
        method = str(calibration.get("method") or "unknown")
        limitations.append(
            f"模型已进行 {method} 概率校准，但 PMML 产物不包含校准器；"
            "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
        )
    policy_status = str(selection_policy_decision.get("status") or "")
    if policy_status in {"blocked", "overridden"}:
        limitations.append(f"Selection policy status is {policy_status}.")
    violations = [
        str(item.get("message") or item.get("code") or "")
        for item in (selection_policy_decision.get("violations") or [])
        if isinstance(item, dict)
    ]
    limitations.extend(item for item in violations if item)
    monitoring_status = str(monitoring_policy.get("status") or "")
    if monitoring_status in {"warn", "fail", "missing", "needs_policy"}:
        limitations.append(str(monitoring_policy.get("recommendation") or "Monitoring policy needs review."))
    comparison_status = str(challenger_comparison.get("status") or "")
    if comparison_status in {"warn", "missing"}:
        limitations.append(
            str(challenger_comparison.get("recommendation") or "Champion comparison needs review.")
        )
    return _unique_strings([item for item in limitations if item])


def _model_card_next_review_actions(
    limitations: list[str],
    monitoring_policy: dict,
    challenger_comparison: dict,
    sample_weight_policy: dict | None = None,
) -> list[str]:
    actions = ["确认模型卡、审批包、监控策略与交付产物路径一致。"]
    if limitations:
        actions.append("逐项复核模型限制与放行说明。")
    if isinstance(sample_weight_policy, dict) and sample_weight_policy.get("used"):
        actions.append("复核样本权重业务口径、加权/非加权指标差异和上线权重列监控阈值。")
    if monitoring_policy:
        actions.append("按监控策略配置上线后的漂移/稳定性复核。")
    if challenger_comparison:
        actions.append("结合 Champion 对比结论确认是否接受 Challenger。")
    return actions


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
    challenger_task_id: str,
    challenger_package_path: str,
    challenger_package_markdown_path: str,
    selection_policy_decision: dict,
    monitoring_policy: dict,
    challenger_comparison: dict,
    model_card: dict,
) -> dict[str, Path | None]:
    uow = ArtifactUnitOfWork()
    json_artifact = uow.stage_file(base_dir, f"{artifact.id}.approval_package.json")
    markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.approval_package.md")
    monitoring_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.monitoring_policy.json")
    monitoring_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.monitoring_policy.md")
    model_card_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.model_card.json")
    model_card_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.model_card.md")
    comparison_json_artifact = None
    comparison_markdown_artifact = None
    if challenger_comparison:
        comparison_json_artifact = uow.stage_file(base_dir, f"{artifact.id}.champion_comparison.json")
        comparison_markdown_artifact = uow.stage_file(base_dir, f"{artifact.id}.champion_comparison.md")
    config = experiment.config
    sample_weight_policy = _sample_weight_policy_payload(experiment=experiment, artifact=artifact)
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
        "sample_weight_col": str(sample_weight_policy.get("sample_weight_col") or ""),
        "training": {
            "sample_weight": sample_weight_policy,
        },
        "features": list(artifact.feature_list),
        "feature_count": len(artifact.feature_list),
        "metrics": _json_safe(experiment.metrics),
        "capabilities": _json_safe(capabilities),
        "selection_policy_decision": selection_policy_decision,
        "monitoring_policy": _json_safe(monitoring_policy),
        "challenger_comparison": _json_safe(challenger_comparison),
        "model_card": _json_safe(model_card),
        "delivery_actions": _json_safe(actions),
        "artifacts": {
            "native_model_path": str(artifact.model_path or ""),
            "pmml_path": str(pmml_path or ""),
            "validation_task_id": str(validation_task_id or ""),
            "challenger_task_id": str(challenger_task_id or ""),
            "challenger_package_path": str(challenger_package_path or ""),
            "challenger_package_markdown_path": str(challenger_package_markdown_path or ""),
            "model_card_path": str(model_card_json_artifact.final_path),
            "model_card_markdown_path": str(model_card_markdown_artifact.final_path),
            "challenger_comparison_path": (
                str(comparison_json_artifact.final_path) if comparison_json_artifact else ""
            ),
            "challenger_comparison_markdown_path": (
                str(comparison_markdown_artifact.final_path) if comparison_markdown_artifact else ""
            ),
        },
        "scorecard_table": _json_safe(_scorecard_table_rows(artifact)),
        "model_params": _json_safe(artifact.params),
    }
    safe_payload = _json_safe(payload)
    try:
        json_artifact.path.write_text(
            json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        markdown_artifact.path.write_text(
            _approval_package_markdown(safe_payload),
            encoding="utf-8",
        )
        monitoring_json_artifact.path.write_text(
            json.dumps(safe_payload["monitoring_policy"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        monitoring_markdown_artifact.path.write_text(
            _monitoring_policy_markdown(safe_payload["monitoring_policy"]),
            encoding="utf-8",
        )
        model_card_json_artifact.path.write_text(
            json.dumps(safe_payload["model_card"], ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_card_markdown_artifact.path.write_text(
            _model_card_markdown(safe_payload["model_card"]),
            encoding="utf-8",
        )
        if comparison_json_artifact and comparison_markdown_artifact:
            comparison_json_artifact.path.write_text(
                json.dumps(
                    safe_payload["challenger_comparison"],
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            comparison_markdown_artifact.path.write_text(
                _challenger_comparison_markdown(safe_payload["challenger_comparison"]),
                encoding="utf-8",
            )
        return uow.finalize(lambda: {
            "json_path": json_artifact.final_path,
            "markdown_path": markdown_artifact.final_path,
            "monitoring_policy_path": monitoring_json_artifact.final_path,
            "monitoring_policy_markdown_path": monitoring_markdown_artifact.final_path,
            "model_card_path": model_card_json_artifact.final_path,
            "model_card_markdown_path": model_card_markdown_artifact.final_path,
            "challenger_comparison_path": (
                comparison_json_artifact.final_path if comparison_json_artifact else None
            ),
            "challenger_comparison_markdown_path": (
                comparison_markdown_artifact.final_path if comparison_markdown_artifact else None
            ),
        })
    except Exception:
        uow.rollback()
        raise


def _approval_package_markdown(payload: dict) -> str:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    policy = (
        payload.get("selection_policy_decision")
        if isinstance(payload.get("selection_policy_decision"), dict)
        else {}
    )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    actions = [item for item in (payload.get("delivery_actions") or []) if isinstance(item, dict)]
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    limitations = [
        str(item)
        for item in (capabilities.get("limitations") or [])
        if str(item)
    ]
    monitoring = payload.get("monitoring_policy") if isinstance(payload.get("monitoring_policy"), dict) else {}
    comparison = (
        payload.get("challenger_comparison")
        if isinstance(payload.get("challenger_comparison"), dict)
        else {}
    )
    features = [str(item) for item in (payload.get("features") or []) if str(item)]
    violations = [item for item in (policy.get("violations") or []) if isinstance(item, dict)]
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    sample_weight = (
        training.get("sample_weight")
        if isinstance(training.get("sample_weight"), dict)
        else {}
    )
    lines = [
        "# 模型审批包",
        "",
        "## 基本信息",
        "",
        f"- 实验ID: `{_md_inline(payload.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(payload.get('artifact_id'))}`",
        f"- 算法: `{_md_inline(payload.get('algorithm'))}`",
        f"- 目标类型: `{_md_inline(payload.get('target_type'))}`",
        f"- 目标列: `{_md_inline(payload.get('target_col'))}`",
        f"- 样本集: `{_md_inline(payload.get('sample_dataset_id') or payload.get('dataset_id'))}`",
        f"- 特征数: {_md_inline(payload.get('feature_count'))}",
        f"- 样本权重: `{_md_inline(payload.get('sample_weight_col') or '未使用')}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    metric_rows = [
        (key, value)
        for key, value in metrics.items()
        if key.startswith(("test_", "oot_", "psi_")) or key in {"overfit_flag"}
    ]
    if metric_rows:
        for key, value in sorted(metric_rows):
            lines.append(f"| {_md_cell(key)} | {_md_cell(_metric_display(value))} |")
    else:
        lines.append("| - | - |")
    lines.extend([
        "",
        "## 策略执行",
        "",
        f"- 状态: `{_md_inline(policy.get('status') or 'not_requested')}`",
        f"- Override原因: {_md_inline(policy.get('override_reason') or '-')}",
    ])
    policy_requirement_rows = _selection_policy_requirement_markdown_rows(policy.get("policy"))
    if policy_requirement_rows:
        lines.extend(["", "| 策略要求 | 配置 |", "| --- | --- |"])
        for label, value in policy_requirement_rows:
            lines.append(f"| {_md_cell(label)} | {_md_cell(value)} |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重治理"))
    if violations:
        lines.extend(["", "| 违规项 | 说明 |", "| --- | --- |"])
        for item in violations:
            lines.append(
                f"| {_md_cell(item.get('code') or '-')} | {_md_cell(item.get('message') or '-')} |"
            )
    if limitations:
        lines.extend(["", "## 交付限制", ""])
        for item in limitations:
            lines.append(f"- {_md_inline(item)}")
    if monitoring:
        lines.extend([
            "",
            "## 监控策略",
            "",
            f"- 版本: `{_md_inline(monitoring.get('policy_version'))}`",
            f"- 状态: `{_md_inline(monitoring.get('status'))}`",
            f"- 建议: {_md_inline(monitoring.get('recommendation') or '-')}",
        ])
        monitor_checks = [
            item for item in (monitoring.get("checks") or [])
            if isinstance(item, dict)
        ][:8]
        if monitor_checks:
            lines.extend(["", "| 检查项 | 状态 | 当前值 | 阈值 |", "| --- | --- | ---: | --- |"])
            for item in monitor_checks:
                threshold = _monitoring_threshold_display(item)
                lines.append(
                    f"| {_md_cell(item.get('label') or item.get('id') or '-')} | "
                    f"{_md_cell(item.get('status') or '-')} | "
                    f"{_md_cell(_metric_display(item.get('value')))} | "
                    f"{_md_cell(threshold)} |"
                )
    if comparison:
        lines.extend([
            "",
            "## Champion对比",
            "",
            f"- 状态: `{_md_inline(comparison.get('status'))}`",
            f"- 建议: {_md_inline(comparison.get('recommendation') or '-')}",
        ])
        champion = comparison.get("champion") if isinstance(comparison.get("champion"), dict) else {}
        if champion:
            lines.extend([
                f"- Champion: `{_md_inline(champion.get('label') or 'prior_champion')}`",
                f"- Champion实验: `{_md_inline(champion.get('experiment_id') or '-')}`",
            ])
        rows = [
            item for item in (comparison.get("metric_comparisons") or [])
            if isinstance(item, dict)
        ][:12]
        if rows:
            lines.extend([
                "",
                "| 指标 | Champion | Challenger | 差异 | 方向 | 结论 |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ])
            for item in rows:
                lines.append(
                    f"| {_md_cell(item.get('metric') or '-')} | "
                    f"{_md_cell(_metric_display(item.get('champion_value')))} | "
                    f"{_md_cell(_metric_display(item.get('challenger_value')))} | "
                    f"{_md_cell(_metric_display(item.get('delta')))} | "
                    f"{_md_cell(item.get('direction') or '-')} | "
                    f"{_md_cell(item.get('verdict') or '-')} |"
                )
    lines.extend([
        "",
        "## 交付产物",
        "",
        "| 类型 | 路径/任务 |",
        "| --- | --- |",
        f"| 原生模型 | `{_md_cell(artifacts.get('native_model_path') or '-')}` |",
        f"| PMML | `{_md_cell(artifacts.get('pmml_path') or '-')}` |",
        f"| 验证任务 | `{_md_cell(artifacts.get('validation_task_id') or '-')}` |",
        f"| Challenger/Backtest任务 | `{_md_cell(artifacts.get('challenger_task_id') or '-')}` |",
        f"| Challenger/Backtest包 | `{_md_cell(artifacts.get('challenger_package_markdown_path') or artifacts.get('challenger_package_path') or '-')}` |",
        f"| 模型卡 | `{_md_cell(artifacts.get('model_card_markdown_path') or artifacts.get('model_card_path') or '-')}` |",
        f"| Champion对比 | `{_md_cell(artifacts.get('challenger_comparison_markdown_path') or artifacts.get('challenger_comparison_path') or '-')}` |",
    ])
    if actions:
        lines.extend(["", "## 交付动作", "", "| 动作 | 状态 | 说明 |", "| --- | --- | --- |"])
        for item in actions:
            lines.append(
                f"| {_md_cell(item.get('action') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(item.get('reason') or item.get('pmml_path') or item.get('validation_task_id') or item.get('challenger_task_id') or '-')} |"
            )
    lines.extend(["", "## 入模特征", ""])
    preview = features[:50]
    if preview:
        for feature in preview:
            lines.append(f"- `{_md_inline(feature)}`")
        if len(features) > len(preview):
            lines.append(f"- ... 另有 {len(features) - len(preview)} 个特征")
    else:
        lines.append("- -")
    return "\n".join(lines) + "\n"


def _selection_policy_requirement_markdown_rows(policy) -> list[tuple[str, str]]:
    if not isinstance(policy, dict) or not policy:
        return []
    rows: list[tuple[str, str]] = []
    boolean_labels = {
        "require_pmml": "要求 PMML",
        "require_handoff": "要求验证移交",
        "require_scorecard": "要求评分卡",
        "require_monotonicity": "要求单调性证据",
        "prefer_scorecard": "优先评分卡",
        "allow_policy_override": "允许策略 override",
    }
    for key, label in boolean_labels.items():
        if key in policy:
            rows.append((label, "是" if policy.get(key) else "否"))
    if policy.get("max_feature_count") is not None:
        rows.append(("最大特征数", _metric_display(policy.get("max_feature_count"))))
    if policy.get("max_oot_psi") is not None:
        rows.append(("最大 OOT PSI", _metric_display(policy.get("max_oot_psi"))))
    metric_thresholds = policy.get("metric_thresholds")
    if isinstance(metric_thresholds, dict):
        for metric in sorted(metric_thresholds):
            spec = metric_thresholds.get(metric)
            if not isinstance(spec, dict):
                continue
            parts = []
            if spec.get("min") is not None:
                parts.append(f">= {_metric_display(spec.get('min'))}")
            if spec.get("max") is not None:
                parts.append(f"<= {_metric_display(spec.get('max'))}")
            if parts:
                rows.append((f"指标 {metric}", " 且 ".join(parts)))
    return rows


def _sample_weight_policy_markdown_section(policy: dict, *, heading: str) -> list[str]:
    approval = policy.get("approval_policy") if isinstance(policy.get("approval_policy"), dict) else {}
    monitoring = (
        policy.get("monitoring_defaults")
        if isinstance(policy.get("monitoring_defaults"), dict)
        else {}
    )
    review_items = [str(item) for item in (approval.get("review_items") or []) if str(item)]
    monitor_checks = [
        item for item in (monitoring.get("checks") or [])
        if isinstance(item, dict)
    ]
    lines = [
        "",
        heading,
        "",
        f"- 是否使用: `{_md_inline('是' if policy.get('used') else '否')}`",
        f"- 权重列: `{_md_inline(policy.get('sample_weight_col') or '未使用')}`",
        f"- 来源: `{_md_inline(policy.get('source') or 'none')}`",
        f"- 需要人工复核: `{_md_inline('是' if approval.get('requires_manual_review') else '否')}`",
    ]
    if review_items:
        lines.extend(["", "### 审批复核项", ""])
        for item in review_items:
            lines.append(f"- {_md_inline(item)}")
    if monitor_checks:
        lines.extend([
            "",
            "### 监控默认项",
            "",
            "| 检查项 | 指标 | 状态 | 建议 |",
            "| --- | --- | --- | --- |",
        ])
        for item in monitor_checks:
            lines.append(
                f"| {_md_cell(item.get('id') or '-')} | "
                f"{_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(item.get('recommendation') or '-')} |"
            )
    return lines


def _monitoring_policy_markdown(policy: dict) -> str:
    checks = [item for item in (policy.get("checks") or []) if isinstance(item, dict)]
    sample_weight = (
        policy.get("sample_weight_policy")
        if isinstance(policy.get("sample_weight_policy"), dict)
        else {}
    )
    lines = [
        "# 模型监控策略",
        "",
        "## 基本信息",
        "",
        f"- 策略版本: `{_md_inline(policy.get('policy_version'))}`",
        f"- 状态: `{_md_inline(policy.get('status'))}`",
        f"- 建议: {_md_inline(policy.get('recommendation') or '-')}",
        f"- 实验ID: `{_md_inline(policy.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(policy.get('artifact_id'))}`",
        f"- 目标类型: `{_md_inline(policy.get('target_type'))}`",
        f"- 复核频率: `{_md_inline(policy.get('review_cadence'))}`",
        f"- Owner: `{_md_inline(policy.get('owner'))}`",
        "",
        "## 阈值检查",
        "",
        "| 检查项 | 指标 | 状态 | 当前值 | 阈值 | 说明 |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    if checks:
        for item in checks:
            lines.append(
                f"| {_md_cell(item.get('label') or item.get('id') or '-')} | "
                f"{_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(item.get('status') or '-')} | "
                f"{_md_cell(_metric_display(item.get('value')))} | "
                f"{_md_cell(_monitoring_threshold_display(item))} | "
                f"{_md_cell(item.get('message') or '-')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重监控"))
    if policy.get("notes"):
        lines.extend(["", "## 备注", "", str(policy.get("notes"))])
    return "\n".join(lines) + "\n"


def _model_card_markdown(card: dict) -> str:
    key_metrics = [
        item for item in (card.get("key_metrics") or [])
        if isinstance(item, dict)
    ]
    governance = card.get("governance") if isinstance(card.get("governance"), dict) else {}
    selection_requirements = [
        item for item in (governance.get("selection_policy_requirements") or [])
        if isinstance(item, dict)
    ]
    selection_violations = [
        item for item in (governance.get("selection_policy_violations") or [])
        if isinstance(item, dict)
    ]
    delivery = card.get("delivery") if isinstance(card.get("delivery"), dict) else {}
    calibration = delivery.get("calibration") if isinstance(delivery.get("calibration"), dict) else {}
    training = card.get("training") if isinstance(card.get("training"), dict) else {}
    sample_weight = (
        training.get("sample_weight")
        if isinstance(training.get("sample_weight"), dict)
        else {}
    )
    limitations = [str(item) for item in (card.get("limitations") or []) if str(item)]
    review_actions = [str(item) for item in (card.get("next_review_actions") or []) if str(item)]
    feature_preview = [str(item) for item in (card.get("feature_preview") or []) if str(item)]
    lines = [
        "# 模型卡",
        "",
        "## 基本信息",
        "",
        f"- 模型卡版本: `{_md_inline(card.get('card_version'))}`",
        f"- 实验ID: `{_md_inline(card.get('experiment_id'))}`",
        f"- 产物ID: `{_md_inline(card.get('artifact_id'))}`",
        f"- 算法: `{_md_inline(card.get('algorithm'))}`",
        f"- 目标类型: `{_md_inline(card.get('target_type'))}`",
        f"- 目标列: `{_md_inline(card.get('target_col'))}`",
        f"- 样本集: `{_md_inline(card.get('sample_dataset_id') or card.get('dataset_id'))}`",
        f"- 特征数: {_md_inline(card.get('feature_count'))}",
        f"- 样本权重: `{_md_inline(card.get('sample_weight_col') or '未使用')}`",
        f"- 概率校准: `{_md_inline(calibration.get('method') if calibration else '未校准')}`",
        "",
        "## 关键指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    if key_metrics:
        for item in key_metrics:
            lines.append(
                f"| {_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(_metric_display(item.get('value')))} |"
            )
    else:
        lines.append("| - | - |")
    if sample_weight:
        lines.extend(_sample_weight_policy_markdown_section(sample_weight, heading="## 样本权重治理"))
    lines.extend([
        "",
        "## 治理状态",
        "",
        f"- 选择策略: `{_md_inline(governance.get('selection_policy_status') or 'not_requested')}`",
        f"- 监控策略: `{_md_inline(governance.get('monitoring_status') or 'not_configured')}`",
        f"- Champion对比: `{_md_inline(governance.get('champion_comparison_status') or 'not_configured')}`",
        f"- Override原因: {_md_inline(governance.get('selection_policy_override_reason') or '-')}",
        f"- 监控建议: {_md_inline(governance.get('monitoring_recommendation') or '-')}",
        f"- 对比建议: {_md_inline(governance.get('champion_comparison_recommendation') or '-')}",
    ])
    if selection_requirements:
        lines.extend([
            "",
            "### 选择策略要求",
            "",
            "| 策略要求 | 配置 |",
            "| --- | --- |",
        ])
        for item in selection_requirements:
            lines.append(
                f"| {_md_cell(item.get('requirement') or '-')} | "
                f"{_md_cell(item.get('configured') or '-')} |"
            )
    if selection_violations:
        lines.extend([
            "",
            "### 选择策略违规项",
            "",
            "| 违规项 | 说明 |",
            "| --- | --- |",
        ])
        for item in selection_violations:
            lines.append(
                f"| {_md_cell(item.get('code') or '-')} | "
                f"{_md_cell(item.get('message') or '-')} |"
            )
    lines.extend([
        "",
        "## 交付状态",
        "",
        "| 产物/动作 | 状态或路径 |",
        "| --- | --- |",
        f"| 原生模型 | `{_md_cell(delivery.get('native_model_path') or '-')}` |",
        f"| PMML | `{_md_cell(delivery.get('pmml_path') or delivery.get('export_pmml_status') or '-')}` |",
        f"| PMML包含校准 | `{_md_cell(delivery.get('pmml_includes_calibration'))}` |",
        f"| 验证移交 | `{_md_cell(delivery.get('validation_task_id') or delivery.get('validation_handoff_status') or '-')}` |",
        f"| Challenger/Backtest | `{_md_cell(delivery.get('challenger_task_id') or delivery.get('challenger_backtest_status') or '-')}` |",
    ])
    lines.extend(["", "## 限制与复核", ""])
    if limitations:
        for item in limitations:
            lines.append(f"- {_md_inline(item)}")
    else:
        lines.append("- 暂无已记录限制")
    lines.extend(["", "## 后续动作", ""])
    if review_actions:
        for item in review_actions:
            lines.append(f"- {_md_inline(item)}")
    else:
        lines.append("- -")
    lines.extend(["", "## 特征预览", ""])
    if feature_preview:
        for feature in feature_preview:
            lines.append(f"- `{_md_inline(feature)}`")
    else:
        lines.append("- -")
    return "\n".join(lines) + "\n"


def _challenger_comparison_markdown(comparison: dict) -> str:
    champion = comparison.get("champion") if isinstance(comparison.get("champion"), dict) else {}
    challenger = (
        comparison.get("challenger")
        if isinstance(comparison.get("challenger"), dict)
        else {}
    )
    summary = comparison.get("summary") if isinstance(comparison.get("summary"), dict) else {}
    rows = [
        item for item in (comparison.get("metric_comparisons") or [])
        if isinstance(item, dict)
    ]
    lines = [
        "# Champion / Challenger 对比",
        "",
        "## 基本信息",
        "",
        f"- 对比版本: `{_md_inline(comparison.get('comparison_version'))}`",
        f"- 状态: `{_md_inline(comparison.get('status'))}`",
        f"- 建议: {_md_inline(comparison.get('recommendation') or '-')}",
        f"- Champion: `{_md_inline(champion.get('label') or 'prior_champion')}`",
        f"- Champion实验: `{_md_inline(champion.get('experiment_id') or '-')}`",
        f"- Challenger实验: `{_md_inline(challenger.get('experiment_id') or comparison.get('experiment_id'))}`",
        f"- Challenger产物: `{_md_inline(challenger.get('artifact_id') or comparison.get('artifact_id'))}`",
        "",
        "## 汇总",
        "",
        f"- 可比指标: {_md_inline(summary.get('comparable_metric_count') or 0)}/{_md_inline(summary.get('metric_count') or 0)}",
        f"- 优于Champion: {_md_inline(summary.get('improved_count') or 0)}",
        f"- 弱于Champion: {_md_inline(summary.get('declined_count') or 0)}",
        f"- 持平: {_md_inline(summary.get('same_count') or 0)}",
        "",
        "## 指标差异",
        "",
        "| 指标 | Champion | Challenger | 差异 | 趋势 | 结论 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    if rows:
        for item in rows:
            lines.append(
                f"| {_md_cell(item.get('metric') or '-')} | "
                f"{_md_cell(_metric_display(item.get('champion_value')))} | "
                f"{_md_cell(_metric_display(item.get('challenger_value')))} | "
                f"{_md_cell(_metric_display(item.get('delta')))} | "
                f"{_md_cell(item.get('direction') or '-')} | "
                f"{_md_cell(item.get('verdict') or '-')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - |")
    if champion.get("notes"):
        lines.extend(["", "## Champion备注", "", _md_inline(champion.get("notes"))])
    return "\n".join(lines) + "\n"


def _monitoring_threshold_display(item: dict) -> str:
    direction = str(item.get("direction") or "max")
    warn = item.get("warn")
    fail = item.get("fail")
    prefix = "<=" if direction != "min" else ">="
    if warn is None and fail is None:
        return "需配置"
    parts = []
    if warn is not None:
        parts.append(f"warn {prefix} {_metric_display(warn)}")
    if fail is not None:
        parts.append(f"fail {prefix} {_metric_display(fail)}")
    return "; ".join(parts)


def _metric_display(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return _format_number_token(float(value))
    return "-" if value is None else str(value)


def _md_inline(value) -> str:
    return str(value if value is not None else "-").replace("`", "'")


def _md_cell(value) -> str:
    return _md_inline(value).replace("|", "\\|").replace("\n", " ")


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

    report_dataset_path, score_col, report_frame = _report_scored_dataset(
        runtime,
        dataset_path,
        artifact,
        experiment.config,
        task_id=ctx.task_id,
        experiment_id=experiment.id,
        dataset_id=dataset.id,
    )
    report_runtime = _cached_dataset_runtime(runtime, report_dataset_path, frame=report_frame)
    low_pricing = None
    if _section_available(statuses, "low_pricing") and business.interest_rate_col:
        low_pricing = stress_low_pricing(
            report_runtime.backend,
            report_dataset_path,
            score_col=score_col,
            target_col=experiment.config.target_col,
            interest_rate_col=business.interest_rate_col,
            low_pricing_threshold=None,
        )
    oot_bin = _report_bin_table(
        report_runtime,
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
        report_runtime,
        report_dataset_path,
        score_col=score_band_col,
        target_col=experiment.config.target_col,
        config=experiment.config,
    )
    stress_product_removal = _stress_product_removal(
        report_runtime,
        report_dataset_path,
        artifact,
        experiment.config,
        feature_dictionary,
    )
    split_profile = _dataset_split_profile(
        report_runtime,
        report_dataset_path,
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
        univariate=_univariate_rows(report_runtime, report_dataset_path, artifact, experiment.config),
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
    provided = _flatten_feature_cols(features)
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


def _flatten_feature_cols(features) -> list[str]:
    """Flatten a features input that may be a union of lists (FS-5): a workflow can pass
    ``features=[<base cols>, <$ref new_columns>]`` which resolves to nested lists; screen
    them together as one de-duplicated flat set (input order preserved)."""
    flat: list[str] = []
    seen: set[str] = set()
    for item in (features or []):
        candidates = item if isinstance(item, (list, tuple)) else [item]
        for candidate in candidates:
            name = str(candidate).strip()
            if name and name not in seen:
                seen.add(name)
                flat.append(name)
    return flat


def _preprocessing_steps_for_training(runtime: "_Runtime", dataset_id: str) -> list[dict]:
    """The accumulated preprocessing chain (PREP-2) for the modeling input dataset, read
    from its lineage sidecar. Empty when the dataset has no traceable chain (e.g. a
    historical dataset registered before this mechanism, or one built without any
    impute/cap/normalize/onehot step) — the resulting model artifact then has no
    preprocessing_steps and scoring-time replay is a no-op, matching pre-PREP-2 behavior."""
    try:
        dataset_path = runtime.registry.resolve_path(str(dataset_id))
    except KeyError:
        return []
    return read_preprocessing_chain(dataset_path)


def _preprocessing_chain_traceable(runtime: "_Runtime", dataset_id: str) -> bool:
    """Whether the modeling input dataset carries a preprocessing lineage sidecar at
    all (PREP-2). False means the dataset predates this mechanism or was never derived
    through a chain-tracking FEATURE/prepare_modeling_frame call — the model card
    flags this explicitly ("预处理链不可追溯") rather than silently implying the model
    has zero preprocessing."""
    try:
        dataset_path = runtime.registry.resolve_path(str(dataset_id))
    except KeyError:
        return False
    return sidecar_path(dataset_path).exists()


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
    calibration = _artifact_calibration_for_capabilities(artifact)
    reason = None if pmml_supported else _unsupported_pmml_reason(artifact, payload_reason)
    limitations = _artifact_delivery_limitations(
        artifact,
        pmml_supported=pmml_supported,
        unsupported_reason=reason,
        calibration=calibration,
    )
    return {
        "pmml_supported": pmml_supported,
        "handoff_supported": pmml_supported,
        "native_model_supported": True,
        "reason": reason,
        "calibrated": bool(calibration),
        "calibration": calibration,
        "pmml_includes_calibration": (
            bool(calibration.get("pmml_includes_calibration"))
            if calibration
            else True
        ),
        "limitations": limitations,
    }


def _unsupported_pmml_reason(artifact: ModelArtifact, payload_reason: str | None) -> str:
    if payload_reason:
        return payload_reason
    if artifact.algorithm == "catboost":
        return (
            "CatBoost 可保留原生 .pkl 模型和报告;当前 sklearn2pmml/JPMML "
            "不支持 CatBoostClassifier 直接导出 PMML,因此验证移交需使用 lr/lgb/xgb/scorecard。"
        )
    return (
        f"当前 PMML 导出/验证移交支持 lr/lgb/xgb/scorecard;"
        f"{artifact.algorithm} 可保留原生模型文件和报告。"
    )


def _artifact_calibration_for_capabilities(artifact: ModelArtifact) -> dict:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return {}
    keys = (
        "method",
        "split",
        "split_value",
        "sample_count",
        "positive_count",
        "brier_raw",
        "brier_calibrated",
        "ece_raw",
        "ece_calibrated",
        "pmml_includes_calibration",
        "path",
    )
    return _json_safe({key: calibration.get(key) for key in keys if key in calibration}) or {}


def _artifact_delivery_limitations(
    artifact: ModelArtifact,
    *,
    pmml_supported: bool,
    unsupported_reason: str | None,
    calibration: dict,
) -> list[str]:
    limitations: list[str] = []
    if not pmml_supported:
        limitations.append(unsupported_reason or _unsupported_pmml_reason(artifact, None))
    if calibration and calibration.get("pmml_includes_calibration") is False:
        method = str(calibration.get("method") or "unknown")
        limitations.append(
            f"模型已进行 {method} 概率校准，但 PMML 产物不包含校准器；"
            "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
        )
    if (artifact.params or {}).get("preprocessing_steps"):
        # PREP-2: PMML export intentionally does not embed the generic feature-pack
        # preprocessing layer (impute/cap/normalize/onehot) — out of scope for this
        # fix; only the platform scorer/handoff notebook replay it. Mirrors the
        # calibration limitation above so the model card/capabilities never imply
        # PMML alone reproduces this model's scores on new raw data.
        limitations.append(
            "模型训练前经过 impute/cap/normalize/onehot 等预处理；PMML 产物不包含该预处理层，"
            "对新数据打分须使用平台内 scorer 或验证移交 Notebook（会自动重放预处理链）。"
        )
    elif (artifact.params or {}).get("preprocessing_chain_traceable") is False:
        # PREP-2: the input dataset carried no lineage sidecar at all (predates this
        # mechanism, or was never derived through a chain-tracking FEATURE/
        # prepare_modeling_frame call) — flag explicitly rather than silently implying
        # this model has zero preprocessing.
        limitations.append("预处理链不可追溯：训练数据集无预处理血缘记录，无法确认打分重放是否完整。")
    return _unique_strings(limitations)


def _pmml_delivery_note(capabilities: dict) -> str:
    limitations = [
        str(item)
        for item in capabilities.get("limitations") or []
        if str(item)
    ]
    return " ".join(limitations)


def _handoff_delivery_note(capabilities: dict) -> str:
    calibration = capabilities.get("calibration") if isinstance(capabilities.get("calibration"), dict) else {}
    if calibration and capabilities.get("pmml_includes_calibration") is False:
        return "验证移交 Notebook 会加载 calibration.joblib 应用校准。"
    return ""


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
        try:
            validate_scorecard_pmml_payload(model, feature_list=list(artifact.feature_list))
        except ModelingError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"评分卡 PMML 预检失败:{exc}"
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


def _recipe_seed(seed: int, recipe: str) -> int:
    """Deterministic per-recipe seed derivation (TUNE-1): every recipe's search
    gets its own seed so trial sequences don't collide across algorithms, but the
    derivation is a pure function of (seed, recipe) — same base seed always
    reproduces the same per-recipe trial sequence."""
    digest = hashlib.sha256(f"{seed}:{recipe}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


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
    # Binary champion selection uses test KS (overfit-penalized); OOT is reported only,
    # never used to pick a winner — mirrors tune_hyperparameters' policy (DOM-9).
    return "higher overfit-penalized test KS; OOT reported only, not used for selection"


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
    experiment_id: str,
    dataset_id: str,
) -> tuple[Path, str, pd.DataFrame | None]:
    columns = runtime.backend.column_names(dataset_path)
    if "score" in columns:
        return dataset_path, "score", None
    if artifact is None:
        # No trained artifact and no explicit `score` column: there is no real model
        # score to report on. Previously this silently substituted the first feature
        # column as a fake "score", producing a plausible-looking but semantically wrong
        # formal report (DOM-10) — fail loudly instead.
        raise ReportScoreMissingError(experiment_id=experiment_id, dataset_id=dataset_id)

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
        return final_path, MODEL_REPORT_SCORE_COL, frame
    except Exception:
        artifact.rollback()
        raise


def _cached_dataset_runtime(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    frame: pd.DataFrame | None = None,
):
    dataset = (
        TrainingDataset(path=Path(dataset_path), frame=frame)
        if frame is not None
        else TrainingDataset.load(runtime.backend, dataset_path)
    )
    proxy = SimpleNamespace(**vars(runtime))
    proxy.backend = dataset.backend_adapter(runtime.backend)
    return proxy


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
    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        base_dir: Path,
        load_calibration: bool = True,
        replay_preprocessing: bool = False,
    ):
        self.artifact = artifact
        self.base_dir = Path(base_dir)
        self.model = load_model(artifact, base_dir=base_dir)
        self.calibration = (
            _load_calibration_payload(artifact, base_dir=self.base_dir)
            if load_calibration
            else None
        )
        # PREP-2: replay is opt-in. Existing report/stress-test/calibration call sites
        # score the SAME already-transformed modeling frame the model was trained on
        # (impute/cap/normalize already applied in place), so replaying again would
        # double-apply a non-idempotent transform like zscore/minmax normalize and
        # silently corrupt those scores. Only a caller scoring genuinely new raw data
        # (e.g. a future score_dataset tool) should pass replay_preprocessing=True.
        self.replay_preprocessing = bool(replay_preprocessing)

    def score(self, dataframe: pd.DataFrame, *, use_calibration: bool = True) -> list[float]:
        scores = np.asarray(self.raw_score(dataframe), dtype=float)
        if use_calibration and self.calibration is not None:
            scores = _apply_calibrator(str(self.calibration["method"]), self.calibration["calibrator"], scores)
        return [float(value) for value in scores]

    def raw_score(self, dataframe: pd.DataFrame) -> list[float]:
        dataframe = self._replay_preprocessing(dataframe)
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
        dataframe = self._replay_preprocessing(dataframe)
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

    def _replay_preprocessing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Replay this artifact's persisted preprocessing chain (PREP-2) before scoring,
        so predict-time input matches the exact impute/cap/normalize/onehot transforms
        the model was trained on — the fix for silent scoring drift on new raw data.
        No-op unless the caller opted in via replay_preprocessing=True (see __init__),
        or when the artifact carries no chain (e.g. a pre-PREP-2 artifact, or one
        trained straight off a dataset with no traceable lineage)."""
        if not self.replay_preprocessing:
            return dataframe
        steps = self.artifact.params.get("preprocessing_steps") if self.artifact.params else None
        if not steps:
            return dataframe
        return apply_preprocessing_steps(dataframe, list(steps))


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
