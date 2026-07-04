from __future__ import annotations

import numpy as np
import pandas as pd
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from marvis.feature.preprocessing import read_preprocessing_chain, sidecar_path
from marvis.packs.modeling.artifact import persist_model_meta
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.catboost import train_catboost
from marvis.packs.modeling.recipes.common import REFIT_ON_TRAIN_PLUS_TEST_PARAM_KEY, training_frame_columns
from marvis.packs.modeling.recipes.ensemble import train_ensemble
from marvis.packs.modeling.recipes.lgb import train_lgb
from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass
from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor
from marvis.packs.modeling.recipes.lr import train_lr
from marvis.packs.modeling.recipes.mlp import train_mlp
from marvis.packs.modeling.recipes.scorecard import train_scorecard
from marvis.packs.modeling.recipes.xgb import train_xgb
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.packs.modeling.tune import DEFAULT_TRIAL_BUDGET, tune_hyperparameters
from marvis.validation.binning import bin_distribution, equal_frequency_bin_edges
from pathlib import Path

from marvis.packs.modeling._common import _cleanup_unattached_artifact, _effective_seed, _jsonable, _optional_int, _recipe_seed, _snapshot_latest_model_meta, _training_control_params, _training_params, _unique_columns, _unique_strings
from marvis.packs.modeling._runtime import _Runtime, _artifact_base_dir, _runtime
from marvis.packs.modeling.scoring import _ModelArtifactScorer


def tool_configure_tuning(inputs: dict, ctx) -> dict:
    """Prepare the tuning configuration for one or more recipes (TUNE-1/SEL-2).

    Every recipe in BINARY_MODELING_RECIPES (lgb/xgb/catboost/lr/scorecard/mlp)
    now runs the two-stage search in tune.py, each with its own budget — total
    search cost is the SUM of each recipe's n_trials (tree recipes default 40,
    lr/scorecard/mlp default 12; see DEFAULT_TRIAL_BUDGET). ``recipe`` stays the
    single-recipe entry point for back-compat: it degrades to a one-element
    ``recipes`` list. An explicit ``n_trials`` overrides every listed recipe's
    budget uniformly; per-recipe overrides can be passed via ``n_trials_by_recipe``.

    ``cv_folds`` (TUNE-3, optional, default None -- single split): when set (>=2),
    every recipe's search additionally scores trials via grouped cross-validation
    instead of a single train/test split; recommended for small samples where a
    single split's KS is noisy. Costs roughly ``cv_folds``x the runtime.
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
    cv_folds = _optional_int(inputs.get("cv_folds"))
    if cv_folds is not None and cv_folds < 2:
        raise ModelingError("cv_folds must be at least 2")
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
    if cv_folds:
        reason += f" 已启用 {cv_folds} 折分组交叉验证,每轮 trial 耗时约为单一切分的 {cv_folds} 倍。"
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
        "cv_folds": cv_folds,
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

    ``cv_folds`` (TUNE-3, optional, default None -- single split): when set (>=2),
    every requested recipe's search scores trials via grouped cross-validation
    over train instead of the single train/test split; recommended for small
    samples where a single split's KS is noisy. Applies uniformly to every
    recipe in ``recipes``. Costs roughly ``cv_folds``x the runtime.
    """
    recipe = str(inputs.get("recipe") or "lgb")
    recipes = _unique_strings(inputs.get("recipes") or [recipe]) or [recipe]
    configured_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, configured_params)
    base_params = {**configured_params, **control_params}
    n_trials_override = _optional_int(inputs.get("n_trials"))
    cv_folds = _optional_int(inputs.get("cv_folds"))
    if cv_folds is not None and cv_folds < 2:
        raise ModelingError("cv_folds must be at least 2")
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
                cv_folds=cv_folds,
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
    # DOM-6: an explicit eval_metric input (e.g. "response_lift" for a marketing/
    # recall scenario) drives champion selection below; every experiment's own
    # TrainConfig also records it so compare_experiments/select_experiment can
    # recover it later without the caller having to repeat it.
    eval_metric = str(inputs.get("eval_metric") or "ks_auc").strip() or "ks_auc"
    dataset_path = runtime.registry.resolve_path(dataset.id)
    training_dataset = TrainingDataset.load(runtime.backend, dataset_path)
    training_backend = training_dataset.backend_adapter(runtime.backend)
    preprocessing_steps = read_preprocessing_chain(dataset_path)
    preprocessing_chain_traceable = bool(preprocessing_steps) or sidecar_path(dataset_path).exists()

    experiments: list[dict] = []
    failed: list[dict] = []
    last_exc: Exception | None = None
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
            eval_metric=eval_metric,
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
        except Exception as exc:
            # TUNE-8/SEL-3: one recipe's failure (e.g. a data issue only that
            # algorithm chokes on) no longer aborts the whole multi-algorithm
            # comparison -- it's recorded as a failed candidate and the batch
            # continues, so the other recipes' results are never lost.
            if result is not None:
                _cleanup_unattached_artifact(result.artifact, artifact_dir, meta_snapshot)
            runtime.experiments.set_status(experiment_id, "failed")
            failed.append({
                "experiment_id": experiment_id,
                "recipe": recipe,
                "error": f"{type(exc).__name__}: {exc}",
            })
            last_exc = exc
            continue
        experiment = runtime.experiments.get(experiment_id)
        experiments.append({
            "experiment_id": experiment_id,
            "recipe": recipe,
            "metrics": _jsonable(experiment.metrics) or {},
        })

    if not experiments:
        # Every recipe failed: nothing survived to compare, so this must be a hard
        # error, not a silently empty result -- re-raise the last recipe's original
        # exception (not a generic wrapper) so infrastructure failures (e.g. an
        # audit-log write failure, as opposed to a genuine per-recipe training
        # issue) still surface with their real type/message for callers/tests
        # that match on it.
        if last_exc is not None:
            raise last_exc
        raise ModelingError("all requested recipes failed to train")
    best, selection_metric = _pick_best_experiment(
        experiments, target_type=target_type, eval_metric=eval_metric
    )
    return {
        "experiments": experiments,
        "experiment_ids": [exp["experiment_id"] for exp in experiments],
        "best_experiment_id": best["experiment_id"],
        "best_recipe": best["recipe"],
        "target_type": target_type,
        "eval_metric": eval_metric,
        "selection_metric": selection_metric,
        "failed": failed,
    }


#: Overfit penalty applied to the binary champion-selection score, matching
#: tune.py's ``_trial_score`` objective (``test_ks - penalty * max(0, train_ks - test_ks)``).
_CHAMPION_OVERFIT_PENALTY = 0.5


#: Binary champion selection metric name/basis: OOT is reported but never used to pick
#: a winner (mirrors tune_hyperparameters' "OOT reports only" policy — DOM-9).
BINARY_SELECTION_METRIC = "test_ks(overfit-penalized)"


#: DOM-6: champion selection metric name/basis when a scenario declares
#: eval_metric="response_lift" (marketing/recall templates) -- test-only, no OOT
#: peeking, matching BINARY_SELECTION_METRIC's DOM-9 policy. No train reading is
#: computed for lift, so (unlike KS) there is no overfit penalty term to subtract.
RESPONSE_LIFT_SELECTION_METRIC = "test_lift_head_10"


def _overfit_penalized_test_ks(metrics: dict) -> float:
    """``test_ks - penalty * max(0, train_ks - test_ks)``; ``-inf`` when test_ks is missing.

    TUNE-5: uses ``weighted_test_ks``/``weighted_train_ks`` when the experiment has
    them (i.e. it trained with a sample_weight_col) instead of the unweighted
    reading — a model trained against a weighted population must also be
    *compared* against the weighted population, or champion selection silently
    optimises a different objective than training did. Falls back to the
    unweighted KS when no weighted metric is present (the historical, unweighted
    contract, unchanged).

    OOT is intentionally excluded from the score — using it for champion selection would
    contradict tune_hyperparameters' explicit "OOT metrics are reported for transparency
    but are not used for hyperparameter selection" policy (DOM-9).
    """
    test_ks = metrics.get("weighted_test_ks")
    if not isinstance(test_ks, (int, float)):
        test_ks = metrics.get("test_ks")
    if not isinstance(test_ks, (int, float)):
        return float("-inf")
    train_ks = metrics.get("weighted_train_ks")
    if not isinstance(train_ks, (int, float)):
        train_ks = metrics.get("train_ks")
    gap = float(train_ks) - float(test_ks) if isinstance(train_ks, (int, float)) else 0.0
    return float(test_ks) - _CHAMPION_OVERFIT_PENALTY * max(0.0, gap)


def _response_lift_score(metrics: dict) -> float:
    """DOM-6: ``test_lift_head_10`` (top-decile response lift), ``-inf`` when missing.

    Mirrors ``_overfit_penalized_test_ks``'s DOM-9 "test only, OOT reports but
    never selects" policy -- head_tail_lift's OOT reading is still surfaced on the
    comparison row for transparency, it just never drives the winner.
    """
    value = metrics.get("test_lift_head_10")
    if not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _binary_selection_score_and_metric(eval_metric: str) -> tuple[Callable[[dict], float], str]:
    """DOM-6: resolve a binary target's champion-selection scoring function and its
    metric label from the scenario's declared ``eval_metric`` -- ``response_lift``
    (marketing/recall scenario templates) selects by top-decile test lift instead
    of KS; every other value (including the default ``ks_auc``) keeps the
    pre-existing overfit-penalized test KS behaviour unchanged."""
    if str(eval_metric or "").strip() == "response_lift":
        return _response_lift_score, RESPONSE_LIFT_SELECTION_METRIC
    return _overfit_penalized_test_ks, BINARY_SELECTION_METRIC


def _pick_best_experiment(
    experiments: list[dict], *, target_type: str = "binary", eval_metric: str = "ks_auc"
) -> tuple[dict, str]:
    """Pick the best experiment with the metric family that matches the target.

    Binary maximizes the overfit-penalized test KS by default (OOT is reported,
    not selected on — DOM-9); when ``eval_metric="response_lift"`` (marketing/
    recall scenario templates, DOM-6) it instead maximizes test top-decile lift.
    Regression minimizes OOT/test RMSE; multiclass maximizes OOT/test macro-AUC,
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

    metric_score, selection_metric = _binary_selection_score_and_metric(eval_metric)

    def score(experiment: dict) -> float:
        return metric_score(experiment.get("metrics") or {})

    return max(experiments, key=score), selection_metric


def _resolve_scenario_eval_metric(runtime: _Runtime, experiment_ids: list[str], override: str) -> str:
    """DOM-6: resolve the eval_metric that should drive champion selection for a set
    of experiments. An explicit ``eval_metric`` tool input always wins; otherwise
    read it off the first resolvable experiment's stored ``TrainConfig.eval_metric``
    (populated by ``apply_scenario`` at train time, e.g. "response_lift" for the
    marketing/recall scenario templates) -- every candidate compared/selected
    together came from the same training run, so they share one scenario. Falls
    back to the platform default ``"ks_auc"`` when neither is available."""
    if override:
        return override
    for experiment_id in experiment_ids:
        try:
            experiment = runtime.experiments.get(experiment_id)
        except KeyError:
            continue
        eval_metric = getattr(experiment.config, "eval_metric", None)
        if eval_metric:
            return str(eval_metric)
    return "ks_auc"


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


def _refit_champion_on_train_plus_test(
    runtime: "_Runtime",
    *,
    task_id: str,
    experiment,
    recipe: str,
) -> tuple[TrainConfig, TrainResult] | None:
    """Retrain the champion's frozen hyperparameters on train+test combined (TUNE-4).

    The champion selected by ``select_experiment`` only ever saw the train split
    (~50-70% of labeled rows once test+OOT are carved out) -- test's information is
    otherwise permanently wasted on the delivered artifact. This freezes the
    champion's resolved params (incl. ``num_boost_round``/``iterations`` for tree
    recipes, scaled by ``best_iteration * 1/(1 - test_fraction)`` so the combined
    fit gets a comparable number of boosting rounds for its larger training set),
    disables early stopping (there is no more held-out fold to watch), and refits
    on train ∪ test. OOT is never touched -- it stays the pre-refit population,
    scored fresh against the refit model, so before/after OOT metrics are
    directly comparable.

    Returns ``(refit_config, result)`` on success, or ``None`` when the recipe/
    config isn't refittable this way (no ``dataset_id``/split_values on the
    experiment's config, e.g. a very old record) rather than raising -- refit is
    an enhancement, never a hard blocker.
    """
    config = experiment.config
    dataset_id = getattr(config, "dataset_id", "") or ""
    split_values = dict(getattr(config, "split_values", {}) or {})
    if not dataset_id or "train" not in split_values or "test" not in split_values:
        return None
    try:
        dataset_path = runtime.registry.resolve_path(dataset_id)
    except KeyError:
        return None
    split_col = str(config.split_col)
    # A missing split_col is a graceful "can't refit" (returns None below, caller
    # keeps the original candidate) rather than an error -- checked against the
    # dataset's actual columns BEFORE requesting the projection, so this still
    # degrades the same way it always has instead of read_frame's column
    # validation raising on a column that was never going to be used anyway.
    if split_col not in runtime.backend.column_names(dataset_path):
        return None
    # LT-6: refit only ever consumes config.features + target_col/split_col (the
    # frame is copied into a scratch parquet and re-read by _train_recipe below,
    # which itself now projects to the SAME column set via training_frame_columns) --
    # never any other column from the source dataset, so project the read instead of
    # pulling the full modeling frame. A missing config.features/target_col entry
    # still surfaces as a hard failure either way (previously a KeyError deep inside
    # the refit recipe's model.fit; now a DataSecurityError from this read) -- both
    # are caught by select_tools.py's broad `except Exception` around this call, so
    # the "refit failed, kept original candidate" outcome is unchanged either way.
    frame = runtime.backend.read_frame(
        dataset_path, columns=training_frame_columns(runtime.backend, dataset_path, config)
    )
    if split_col not in frame.columns:
        return None
    train_mask = frame[split_col] == split_values["train"]
    test_mask = frame[split_col] == split_values["test"]
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    if n_train == 0 or n_test == 0:
        return None
    test_fraction = n_test / (n_train + n_test)

    # Combined rows become "train"; a small deterministic slice is carved back out
    # to satisfy split_modeling_frame's non-empty-test contract (early stopping is
    # off below, so this slice is never fit on). compute_model_metrics DOES compute
    # the full test_* family on it, but because the slice is a random 5% drawn from
    # the same train+test population it is in-distribution and optimistically biased;
    # select_tools._apply_champion_refit (D14) relabels those refit_holdout_* and
    # excludes them from the headline / model card / monitoring baseline -- only the
    # honest train_/OOT_ metrics from this refit are surfaced as held-out results.
    combined_idx = frame.index[train_mask | test_mask]
    rng = np.random.RandomState(int(config.seed))
    shuffled = combined_idx.to_numpy().copy()
    rng.shuffle(shuffled)
    holdout_n = max(1, min(len(shuffled) - 1, round(len(shuffled) * 0.05)))
    scratch = frame.copy()
    scratch[split_col] = scratch[split_col].astype(object)
    scratch.loc[combined_idx, split_col] = "__refit_train__"
    scratch.loc[shuffled[:holdout_n], split_col] = "__refit_holdout__"
    scratch_split_values = {
        "train": "__refit_train__",
        "test": "__refit_holdout__",
        **({"oot": split_values["oot"]} if "oot" in split_values else {}),
    }

    frozen_params = dict(config.params)
    resolved_artifact = runtime.modeling_repo.get_model_artifact(experiment.artifact_id)
    if resolved_artifact is not None:
        for key in ("num_boost_round", "iterations"):
            raw = resolved_artifact.params.get(key)
            if raw is None:
                continue
            scaled = max(1, round(int(raw) * (1.0 / max(1e-6, 1.0 - test_fraction))))
            frozen_params[key] = scaled

    scratch_dir = _artifact_base_dir(runtime.settings, task_id) / "_refit_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_path = scratch_dir / f"{uuid.uuid4().hex}.parquet"
    try:
        scratch.to_parquet(scratch_path, index=False)
        refit_params = dict(frozen_params)
        refit_params[REFIT_ON_TRAIN_PLUS_TEST_PARAM_KEY] = True
        refit_config = TrainConfig(
            dataset_id=dataset_id,
            features=tuple(config.features),
            target_col=str(config.target_col),
            split_col=split_col,
            split_values=scratch_split_values,
            params=refit_params,
            seed=int(config.seed),
            early_stopping_rounds=None,
            recipe_id=recipe,
            target_type=getattr(config, "target_type", "binary"),
            drop_nan_labels=bool(getattr(config, "drop_nan_labels", False)),
        )
        artifact_dir = _artifact_base_dir(runtime.settings, task_id)
        result = _train_recipe(recipe, runtime.backend, scratch_path, refit_config, out_dir=artifact_dir)
        return refit_config, result
    finally:
        scratch_path.unlink(missing_ok=True)


#: S1b: number of equal-frequency score bins in the training-time baseline
#: distribution snapshot -- matches the platform's existing OOT bin-table
#: convention (_report_bin_table above / DEFAULT_IV_BINS-independent, a fixed
#: monitoring-grade granularity rather than the IV-binning knob).
BASELINE_SCORE_BIN_COUNT = 10


def _compute_baseline_distributions(
    backend,
    dataset_path: Path,
    config: TrainConfig,
    artifact: ModelArtifact,
    *,
    base_dir: Path,
) -> dict | None:
    """S1b: snapshot the training-time score distribution (equal-frequency bin
    edges + per-split bin proportions) and in-model feature distributions, so a
    later monitor_run has a deterministic reference to compare new data against
    (DOM-3's monitoring-policy execution gap). Computed once, at training time,
    from the same dataset_path/config the artifact was just trained on -- this
    covers train_model, train_models, and the champion refit path uniformly
    since all three route through this function.

    Returns None (never raises) when the frame carries no usable ``train`` split
    to build a reference from -- callers must treat that as "no baseline could be
    computed", not silently skip persisting the field."""
    split_col = str(config.split_col)
    try:
        columns = _unique_columns([*artifact.feature_list, split_col])
        frame = backend.read_frame(dataset_path, columns=columns)
    except Exception:
        return None
    if split_col not in frame.columns:
        return None
    train_value = config.split_values.get("train", "train")
    train_frame = frame[frame[split_col] == train_value]
    if train_frame.empty:
        return None

    try:
        scorer = _ModelArtifactScorer(artifact, base_dir=base_dir, load_calibration=False)
        train_scores = np.asarray(scorer.score(train_frame, use_calibration=False), dtype=float)
    except Exception:
        return None
    finite_train_scores = train_scores[np.isfinite(train_scores)]
    if finite_train_scores.size == 0:
        return None
    edges = equal_frequency_bin_edges(finite_train_scores, BASELINE_SCORE_BIN_COUNT)

    score_distribution: dict[str, dict] = {
        "train": {
            "sample_count": int(finite_train_scores.size),
            "bin_proportions": [float(value) for value in bin_distribution(finite_train_scores, edges)],
        }
    }
    for split_name in ("test", "oot"):
        split_value = config.split_values.get(split_name)
        if split_value is None:
            continue
        split_frame = frame[frame[split_col] == split_value]
        if split_frame.empty:
            continue
        try:
            split_scores = np.asarray(scorer.score(split_frame, use_calibration=False), dtype=float)
        except Exception:
            continue
        finite_split_scores = split_scores[np.isfinite(split_scores)]
        if finite_split_scores.size == 0:
            continue
        score_distribution[split_name] = {
            "sample_count": int(finite_split_scores.size),
            "bin_proportions": [float(value) for value in bin_distribution(finite_split_scores, edges)],
        }

    feature_distributions: dict[str, dict] = {}
    for feature in artifact.feature_list:
        if feature not in train_frame.columns:
            continue
        values = pd.to_numeric(train_frame[feature], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        quantiles = np.quantile(finite_values, np.linspace(0.0, 1.0, BASELINE_SCORE_BIN_COUNT + 1))
        feature_distributions[str(feature)] = {
            "sample_count": int(finite_values.size),
            "missing_rate": float(1.0 - finite_values.size / values.size) if values.size else 0.0,
            "quantile_edges": [float(value) for value in quantiles],
            # FIN-3 #2: store the feature's ACTUAL train-time bin proportions under
            # these quantile edges. Equal-frequency edges make the distribution
            # uniform only when values are distinct; a feature with many repeated
            # values collapses np.unique edges so the surviving bins are NOT equal-
            # sized, and monitor_run's CSI uniform(1/bin) expectation would be wrong.
            # bin_distribution is computed against the same collapsed edges, so its
            # length matches quantile_edges (len(edges)-1) and gives the true baseline
            # occupancy. Older snapshots lack this key; monitor_run falls back to
            # uniform for them (see _monitor_run_feature_csi_checks).
            "bin_proportions": [
                float(value) for value in bin_distribution(finite_values, quantiles)
            ],
        }

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "bin_count": BASELINE_SCORE_BIN_COUNT,
        "score_edges": [float(value) for value in edges],
        "score_direction": artifact.score_direction,
        "score_distribution": score_distribution,
        "feature_distributions": feature_distributions,
    }


def _train_recipe(
    recipe: str,
    backend,
    dataset_path: Path,
    config: TrainConfig,
    *,
    out_dir: Path,
) -> TrainResult:
    if recipe == "lgb":
        result = train_lgb(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lgb_regressor":
        result = train_lgb_regressor(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lgb_multiclass":
        result = train_lgb_multiclass(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "xgb":
        result = train_xgb(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "catboost":
        result = train_catboost(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lr":
        result = train_lr(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "scorecard":
        result = train_scorecard(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "mlp":
        result = train_mlp(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "ensemble":
        result = train_ensemble(backend, dataset_path, config, out_dir=out_dir)
    else:
        raise ModelingError(f"unsupported modeling recipe: {recipe}")
    return _attach_baseline_distributions(backend, dataset_path, config, result, out_dir=out_dir)


def _attach_baseline_distributions(
    backend,
    dataset_path: Path,
    config: TrainConfig,
    result: TrainResult,
    *,
    out_dir: Path,
) -> TrainResult:
    """S1b: compute and persist the training-time baseline distribution snapshot
    (see _compute_baseline_distributions) onto the freshly-trained artifact, both
    channels (DB field + .model_meta.json), mirroring the S1a score_direction
    double-channel persistence paradigm. Scoped to binary target_type -- score()
    on a multiclass Booster returns a 2D array _compute_baseline_distributions
    cannot reduce to a single score distribution, and a continuous regressor's
    raw output isn't a PD/points product monitor_run's PSI checks are meant for.
    Never lets a computation failure break training: on any error the artifact is
    persisted exactly as before, with baseline_distributions left None."""
    if getattr(config, "target_type", "binary") != "binary":
        return result
    try:
        baseline = _compute_baseline_distributions(
            backend, dataset_path, config, result.artifact, base_dir=out_dir
        )
    except Exception:
        baseline = None
    if baseline is None:
        return result
    updated_artifact = replace(result.artifact, baseline_distributions=baseline)
    persist_model_meta(out_dir, updated_artifact, config=config)
    return replace(result, artifact=updated_artifact)
