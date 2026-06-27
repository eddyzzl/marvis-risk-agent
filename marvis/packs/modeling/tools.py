from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, ModelingRepository
from marvis.feature.metrics import feature_metrics
from marvis.feature.encode import woe_encode
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.artifact import export_pmml, load_model
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
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
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.mlp import train_mlp
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.feature.screen import screen_features
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.select import select_features
from marvis.packs.modeling.tune import tune_hyperparameters
from marvis.packs.modeling.errors import ModelingError
from marvis.settings import build_settings
from marvis.validation.config import ValidationConfig
from marvis.validation.stress_test import run_stress_test


MODELING_ARTIFACTS_DIR_NAME = "modeling_artifacts"
MODEL_REPORT_SCORE_COL = "__model_score__"


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


def tool_prepare_modeling_frame(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        str(inputs["dataset_id"]),
        target_col=str(inputs["target_col"]),
        feature_cols=[str(item) for item in inputs["feature_cols"]],
        split_col=_optional_str(inputs.get("split_col")),
        split_config=inputs.get("split_config") or {},
        seed=int(inputs.get("seed") if inputs.get("seed") is not None else ctx.seed or 0),
    )
    split_col = _optional_str(inputs.get("split_col")) or "split"
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(result.id), columns=[split_col])
    counts = {
        str(key): int(value)
        for key, value in frame[split_col].value_counts().sort_index().items()
    }
    return {"result_dataset_id": result.id, "split_counts": counts}


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
    result = prepare_modeling_frame(
        runtime.registry,
        runtime.backend,
        str(inputs["dataset_id"]),
        target_col=str(inputs["target_col"]),
        feature_cols=[str(item) for item in inputs["feature_cols"]],
        split_col=split_col,
        split_config=inputs.get("split_config") or {},
        seed=int(inputs.get("seed") if inputs.get("seed") is not None else ctx.seed or 0),
    )
    effective_split_col = split_col or SPLIT_COLUMN
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    source_frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id))
    split_frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(result.id), columns=[effective_split_col]
    )
    sample_analysis = _split_sample_analysis(split_frame[effective_split_col], source_frame)
    return {
        "result_dataset_id": result.id,
        "split_col": effective_split_col,
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
    result = select_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=[str(item) for item in inputs["features"]],
        target_col=str(inputs["target_col"]),
        iv_min=float(inputs.get("iv_min", 0.02)),
        corr_max=float(inputs.get("corr_max", 0.8)),
        vif_max=float(inputs.get("vif_max", 10.0)),
        top_k=_optional_int(inputs.get("top_k")),
        seed=int(ctx.seed or 0),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    return {
        "selected": list(result.selected),
        "dropped": [[feature, reason] for feature, reason in result.dropped],
        "scores": _jsonable(result.scores),
        "nan_labels_dropped": result.nan_labels_dropped,
    }


def tool_screen_features(inputs: dict, ctx) -> dict:
    # feature_ks is a binary-only statistic; a continuous target would miscompute/crash
    # it, so for a non-binary target skip the leakage screen and keep every candidate.
    if str(inputs.get("target_type", "binary")) != "binary":
        return _screen_features_non_binary(inputs, ctx)
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    holdout = inputs.get("holdout_values")
    result = screen_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=[str(item) for item in inputs["features"]],
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
    """Non-binary (continuous) screen: skip the binary-only leakage KS screen, keep every
    candidate as selected with ks=None, reporting only missing_rate / unique_count."""
    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    features = [str(item) for item in inputs["features"]]
    target_col = str(inputs["target_col"])
    feats = [feature for feature in dict.fromkeys(features) if feature != target_col]
    frame = (
        runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=feats)
        if feats
        else None
    )
    scores: dict[str, dict] = {}
    for feature in feats:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        missing_rate = float(1.0 - finite.mean()) if values.size else 1.0
        unique = int(np.unique(values[finite]).size)
        scores[feature] = {"ks": None, "missing_rate": missing_rate, "unique_count": unique}
    return {
        "selected": list(feats),
        "ranked": [[feature, None] for feature in feats],
        "leakage": [],
        "suspected": [],
        "unusable": [],
        "scores": _jsonable(scores),
        "n_screened": len(feats),
        "note": "非二分类目标：跳过泄漏KS筛选，保留全部候选特征",
    }


def tool_tune_hyperparameters(inputs: dict, ctx) -> dict:
    # The random search is LightGBM-specific. For other recipes there is no lgb
    # search to run (lr/scorecard have their own knobs, not a random search; xgb
    # tuning is a later slice), so we skip tuning and let train_model use the
    # recipe's own defaults.
    recipe = str(inputs.get("recipe") or "lgb")
    if recipe != "lgb":
        return {"best_params": {}, "best_metrics": {}, "n_trials": 0, "trials": []}
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
        seed=int(inputs.get("seed") if inputs.get("seed") is not None else ctx.seed or 0),
        early_stopping_rounds=int(inputs.get("early_stopping_rounds", 100)),
        max_boost_round=int(inputs.get("max_boost_round", 3000)),
        overfit_penalty=float(inputs.get("overfit_penalty", 0.5)),
    )
    return {
        "best_params": _jsonable(result.best_params),
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
        params=dict(inputs.get("params") or {}),
        seed=int(inputs["seed"]),
        early_stopping_rounds=_optional_int(inputs.get("early_stopping_rounds")),
        recipe_id=recipe,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    if inputs.get("scenario"):
        config = apply_scenario(config, str(inputs["scenario"]))
        recipe = config.recipe_id or recipe

    experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
    try:
        result = _train_recipe(
            recipe,
            runtime.backend,
            runtime.registry.resolve_path(dataset.id),
            config,
            out_dir=_artifact_base_dir(runtime.settings, ctx.task_id),
        )
        runtime.experiments.attach_result(experiment_id, result)
    except Exception:
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
    features = tuple(str(item) for item in inputs["features"])
    target_col = str(inputs["target_col"])
    split_col = str(inputs["split_col"])
    split_values = dict(inputs["split_values"])
    seed = int(inputs["seed"])
    drop_nan = bool(inputs.get("drop_nan_labels"))
    target_type = str(inputs.get("target_type", "binary"))

    experiments: list[dict] = []
    for recipe in recipes:
        config = TrainConfig(
            dataset_id=dataset.id,
            features=features,
            target_col=target_col,
            split_col=split_col,
            split_values=split_values,
            # only the lgb recipe consumes the tuned params; others use their defaults
            params=dict(tuned_params) if recipe == "lgb" else {},
            seed=seed,
            early_stopping_rounds=None,
            recipe_id=recipe,
            target_type=target_type,
            drop_nan_labels=drop_nan,
        )
        experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
        try:
            result = _train_recipe(
                recipe,
                runtime.backend,
                runtime.registry.resolve_path(dataset.id),
                config,
                out_dir=_artifact_base_dir(runtime.settings, ctx.task_id),
            )
            runtime.experiments.attach_result(experiment_id, result)
        except Exception:
            runtime.experiments.set_status(experiment_id, "failed")
            raise
        experiment = runtime.experiments.get(experiment_id)
        experiments.append({
            "experiment_id": experiment_id,
            "recipe": recipe,
            "metrics": _jsonable(experiment.metrics) or {},
        })

    best = _pick_best_experiment(experiments)
    return {
        "experiments": experiments,
        "experiment_ids": [exp["experiment_id"] for exp in experiments],
        "best_experiment_id": best["experiment_id"],
        "best_recipe": best["recipe"],
    }


def _pick_best_experiment(experiments: list[dict]) -> dict:
    """Best by OOT KS, falling back to test KS, then first. OOT is the unbiased
    final metric; test KS is the fallback when there is no OOT set."""
    def score(experiment: dict) -> float:
        metrics = experiment.get("metrics") or {}
        for key in ("oot_ks", "test_ks"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return float("-inf")

    return max(experiments, key=score)


def tool_compare_experiments(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    return _jsonable(runtime.experiments.compare([str(item) for item in inputs["experiment_ids"]]))


def tool_export_pmml(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    pmml_path = _pmml_path(runtime, artifact)
    return {"pmml_path": str(pmml_path)}


def tool_handoff_to_validation(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    validation_task_id = handoff_to_validation(
        runtime.experiments,
        artifact,
        sample_dataset_id=str(inputs["sample_dataset_id"]),
        settings=runtime.settings,
    )
    return {"validation_task_id": validation_task_id}


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

        render_minimal_model_report(experiment, report_path)
        return {
            "report_path": str(report_path),
            "section_status": [
                {"section": "汇总", "status": "ok"},
                {"section": "模型指标", "status": "ok"},
            ],
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
    stress_product_removal = _stress_product_removal(runtime, dataset_path, artifact, experiment.config, feature_dictionary)
    split_profile = _dataset_split_profile(
        runtime,
        dataset_path,
        experiment.config,
        window_col=business.loan_month_col,
    )
    structured_summary = _report_structured_summary(
        project_meta=dict(inputs.get("project_meta") or {}),
        dataset_split=_dataset_split_rows(experiment.metrics, split_profile=split_profile),
        stability=_stability_rows(experiment.metrics),
        sample_analysis=sample,
        vintage=vintage,
        feature_importance=feature_importance,
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
    render_model_report(
        ModelReportPayload(
            project_meta=structured_summary["project_meta"],
            dataset_split=structured_summary["dataset_split"],
            stability=structured_summary["stability"],
            sample_analysis=sample,
            vintage=vintage,
            feature_importance=structured_summary["feature_importance"],
            univariate=structured_summary["univariate"],
            oot_bin_table=oot_bin,
            stress_product_removal=stress_product_removal,
            stress_low_pricing=low_pricing,
            narratives=narratives,
            section_status=statuses,
        ),
        report_path,
    )
    return {
        "report_path": str(report_path),
        "section_status": [_jsonable(status) for status in statuses],
    }


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


def _pmml_path(runtime: _Runtime, artifact: ModelArtifact) -> Path:
    experiment = runtime.experiments.get(artifact.experiment_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)
    if artifact.pmml_path:
        existing = _resolve_artifact_path(artifact.pmml_path, base_dir=base_dir)
        if existing.exists():
            return existing
    dataset = runtime.registry.get(experiment.config.dataset_id)
    out_path = base_dir / f"{artifact.id}.pmml"
    pmml_path = export_pmml(
        artifact,
        runtime.registry.resolve_path(dataset.id),
        out_path,
        base_dir=base_dir,
    )
    runtime.experiments.set_artifact_pmml_path(artifact.id, pmml_path.name)
    return pmml_path


def _artifact_base_dir(settings, task_id: str) -> Path:
    return Path(settings.tasks_dir) / task_id / MODELING_ARTIFACTS_DIR_NAME


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
    out_path = Path(runtime.settings.tasks_dir) / task_id / "outputs" / "model_report_scored.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path, MODEL_REPORT_SCORE_COL


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
    def __init__(self, artifact: ModelArtifact, *, base_dir: Path):
        self.artifact = artifact
        self.model = load_model(artifact, base_dir=base_dir)

    def score(self, dataframe: pd.DataFrame) -> list[float]:
        features = list(self.artifact.feature_list)
        if self.artifact.algorithm == "xgb":
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
    "tool_check_data_quality",
    "tool_compare_experiments",
    "tool_export_pmml",
    "tool_handoff_to_validation",
    "tool_generate_model_report",
    "tool_generate_model_reports",
    "tool_make_split",
    "tool_modeling_readiness",
    "tool_prepare_modeling_frame",
    "tool_select_features",
    "tool_train_model",
]
