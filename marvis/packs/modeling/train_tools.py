from __future__ import annotations

import gc
import numpy as np
import pandas as pd
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from marvis.feature.preprocessing import read_preprocessing_chain, sidecar_path
from marvis.files import sha256_file
from marvis.modeling_limits import normalize_n_trials, normalize_n_trials_by_recipe
from marvis.packs.modeling.artifact import persist_model_meta
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig, TrainResult
from marvis.packs.modeling.errors import ModelingError, SpecialValueDecisionRequiredError
from marvis.packs.modeling.recipes.common import (
    REFIT_ON_TRAIN_PLUS_TEST_PARAM_KEY,
    SPECIAL_VALUE_GOVERNANCE_PARAM_KEY,
    training_frame_columns,
)
from marvis.packs.modeling.scenarios import apply_scenario
from marvis.packs.modeling.special_value_tools import (
    SPECIAL_VALUE_POLICY_VERSION,
    special_value_decision_fingerprint,
)
from marvis.packs.modeling.training_dataset import TrainingDataset
from marvis.packs.modeling.tune import DEFAULT_TRIAL_BUDGET
from marvis.packs.modeling.tune_checkpoint import (
    TUNE_CHECKPOINT_DIR_NAME,
    TUNE_CHECKPOINT_CONTRACT_VERSION,
    TuneCheckpointStore,
    build_tune_checkpoint_identity,
    checkpoint_identity_hash,
    dataset_content_identity,
)
from marvis.packs.modeling.tune_isolation import (
    run_tuning_recipe_isolated as _run_tuning_recipe_isolated,
)
from marvis.validation.binning import bin_distribution, equal_frequency_bin_edges
from pathlib import Path

from marvis.packs.modeling._common import (
    _cleanup_unattached_artifact,
    _effective_seed,
    _jsonable,
    _normalize_modeling_target_type,
    _normalize_recipe_list,
    _optional_int,
    _recipe_seed,
    _snapshot_latest_model_meta,
    _target_type_from_recipes,
    _training_control_params,
    _training_params,
    _unique_columns,
)
from marvis.packs.modeling._runtime import (
    _Runtime,
    _artifact_base_dir,
    _runtime,
    _task_dataset,
)
from marvis.packs.modeling.scoring import _ModelArtifactScorer


def _validated_target_type(recipes: list[str], requested) -> str:
    """Return the recipe-derived target family and reject mismatched requests."""

    derived = _target_type_from_recipes(recipes)
    explicit = _normalize_modeling_target_type(requested)
    if explicit is not None and explicit != derived:
        raise ModelingError(
            f"target_type `{explicit}` does not match recipes `{', '.join(recipes)}`"
        )
    return explicit or derived


def _assert_sentinel_preprocessing_governed(
    *,
    dataset_id,
    dataset_content_hash,
    features,
    sentinel_columns,
    preprocessing_steps,
    governance,
) -> None:
    """Defense-in-depth: verify every selected detected column has real evidence.

    The primary contract lives in ``resolve_special_values``.  This check prevents
    direct tune/train calls from bypassing it and rejects a generic lineage boolean
    or forged governance record without the corresponding exact replay step.
    """

    if not isinstance(sentinel_columns, dict) or not sentinel_columns:
        return
    selected = {str(feature) for feature in features}
    relevant = {
        str(column): _sentinel_values(rows)
        for column, rows in sentinel_columns.items()
        if str(column) in selected
    }
    if not relevant:
        return

    governed: dict[str, list[set[float]]] = {}
    for step in preprocessing_steps or []:
        if not isinstance(step, dict) or str(step.get("kind") or "") != "sentinel":
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            continue
        declared_columns = {str(column) for column in step.get("columns") or []}
        for column, values in params.items():
            if str(column) not in declared_columns:
                continue
            normalized: set[float] = set()
            for value in values if isinstance(values, (list, tuple)) else [values]:
                try:
                    normalized.add(float(value))
                except (TypeError, ValueError):
                    continue
            governed.setdefault(str(column), []).append(normalized)

    governance = governance if isinstance(governance, dict) else {}
    problems: dict[str, str] = {}
    for column, values in relevant.items():
        evidence = governance.get(column)
        if not isinstance(evidence, dict):
            problems[column] = "missing_governance_evidence"
            continue
        action = str(evidence.get("action") or "")
        fingerprints = [
            str(evidence.get(key) or "").strip()
            for key in ("fingerprint", "decision_fingerprint")
            if str(evidence.get(key) or "").strip()
        ]
        expected_fingerprint = special_value_decision_fingerprint(evidence)
        evidence_values = {
            float(value)
            for value in evidence.get("detected_values") or []
        }
        if str(evidence.get("policy_version") or "") != SPECIAL_VALUE_POLICY_VERSION:
            problems[column] = "unsupported_governance_policy_version"
        elif str(evidence.get("column") or "") != column:
            problems[column] = "governance_column_mismatch"
        elif evidence_values != values:
            problems[column] = "governance_values_do_not_match_detection"
        elif not fingerprints:
            problems[column] = "missing_decision_fingerprint"
        elif any(value != expected_fingerprint for value in fingerprints):
            problems[column] = "invalid_decision_fingerprint"
        elif str(evidence.get("resolved_dataset_id") or "") != str(dataset_id):
            problems[column] = "governance_resolved_dataset_mismatch"
        elif action == "mask":
            if values not in governed.get(column, []):
                problems[column] = "missing_exact_sentinel_preprocessing_step"
        elif action == "retain":
            if str(evidence.get("source_dataset_id") or "") != str(dataset_id):
                problems[column] = "retain_source_dataset_mismatch"
            elif str(evidence.get("source_dataset_content_hash") or "") != str(
                dataset_content_hash
            ):
                problems[column] = "retain_source_dataset_hash_mismatch"
            elif evidence.get("confirmed") is not True:
                problems[column] = "retain_not_explicitly_confirmed"
            elif not str(evidence.get("reason") or "").strip():
                problems[column] = "retain_requires_reason"
        else:
            problems[column] = f"selected_feature_has_invalid_action:{action or 'missing'}"
    if problems:
        raise SpecialValueDecisionRequiredError(
            columns=sorted(problems),
            sentinel_columns={
                column: sentinel_columns[column]
                for column in problems
            },
            problems=problems,
        )


def _sentinel_values(rows) -> set[float]:
    values: set[float] = set()
    for row in rows if isinstance(rows, (list, tuple)) else []:
        value = row[0] if isinstance(row, (list, tuple)) and row else row
        try:
            values.add(float(value))
        except (TypeError, ValueError):
            continue
    return values


def tool_configure_tuning(inputs: dict, ctx) -> dict:
    """Prepare the tuning configuration for one or more recipes (TUNE-1/SEL-2).

    Every supported binary, regression, and multiclass recipe now runs the
    bounded search in tune.py, each with its own budget — total search cost is
    the SUM of each recipe's n_trials (product default: one real trial per
    recipe; see DEFAULT_TRIAL_BUDGET). ``recipe`` stays the
    single-recipe entry point for back-compat: it degrades to a one-element
    ``recipes`` list. An explicit ``n_trials`` overrides every listed recipe's
    budget uniformly; per-recipe overrides can be passed via ``n_trials_by_recipe``.

    ``cv_folds`` (TUNE-3, optional, default None -- single split): when set (>=2),
    every recipe's search additionally scores trials via grouped cross-validation
    instead of a single train/test split; recommended for small samples where a
    single split's KS is noisy. Costs roughly ``cv_folds``x the runtime.
    """
    recipe = str(inputs.get("recipe") or "lgb")
    recipes = _normalize_recipe_list(inputs.get("recipes") or [recipe])
    target_type = _validated_target_type(recipes, inputs.get("target_type"))
    try:
        n_trials_override = normalize_n_trials(
            inputs.get("n_trials"),
            optional=True,
        )
        explicit_budgets = normalize_n_trials_by_recipe(
            inputs.get("n_trials_by_recipe")
        )
    except ValueError as exc:
        raise ModelingError(str(exc)) from exc
    cv_folds = _optional_int(inputs.get("cv_folds"))
    if cv_folds is not None and cv_folds < 2:
        raise ModelingError("cv_folds must be at least 2")
    sample_weight_col = str(inputs.get("sample_weight_col") or "").strip()
    seed = _effective_seed(inputs, ctx)
    tunable = [item for item in recipes if item in DEFAULT_TRIAL_BUDGET]
    budgets = {
        item: explicit_budgets.get(
            item,
            n_trials_override if n_trials_override is not None else DEFAULT_TRIAL_BUDGET.get(item, 1),
        )
        for item in tunable
    }
    tune_enabled = bool(tunable)
    total_budget = sum(budgets.values())
    params = _training_params(inputs)
    budget_note = ', '.join(f'{item}={budgets[item]}' for item in tunable)
    non_tunable = [item for item in recipes if item not in DEFAULT_TRIAL_BUDGET]
    reason = (
        f"{'/'.join(tunable)} 使用有界随机搜索(预算>1时自动进入粗搜+细搜;"
        f"按算法预算:{budget_note};"
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
    """Bounded random search across target families (TUNE-1/SEL-2):
    binary lgb/xgb/catboost get tree-recipe spaces with early
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
    recipes = _normalize_recipe_list(inputs.get("recipes") or [recipe])
    _validated_target_type(recipes, inputs.get("target_type"))
    configured_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, configured_params)
    base_params = {**configured_params, **control_params}
    sentinel_columns = inputs.get("sentinel_columns")
    runtime = None
    dataset = None
    if isinstance(sentinel_columns, dict) and sentinel_columns:
        runtime = _runtime(ctx)
        dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
        dataset_path = runtime.registry.resolve_path(dataset.id)
        _assert_sentinel_preprocessing_governed(
            dataset_id=dataset.id,
            dataset_content_hash=sha256_file(dataset_path),
            features=inputs.get("features") or [],
            sentinel_columns=sentinel_columns,
            preprocessing_steps=read_preprocessing_chain(dataset_path),
            governance=inputs.get("special_value_governance"),
        )
    try:
        n_trials_override = normalize_n_trials(
            inputs.get("n_trials"),
            optional=True,
        )
        explicit_budgets = normalize_n_trials_by_recipe(
            inputs.get("n_trials_by_recipe")
        )
    except ValueError as exc:
        raise ModelingError(str(exc)) from exc
    cv_folds = _optional_int(inputs.get("cv_folds"))
    if cv_folds is not None and cv_folds < 2:
        raise ModelingError("cv_folds must be at least 2")
    def _budget_for(item: str) -> int:
        if item in explicit_budgets:
            return explicit_budgets[item]
        if n_trials_override is not None:
            return n_trials_override
        return DEFAULT_TRIAL_BUDGET.get(item, 1)

    non_tunable = [item for item in recipes if item not in DEFAULT_TRIAL_BUDGET]
    tunable = [item for item in recipes if item in DEFAULT_TRIAL_BUDGET]
    budgets = {item: _budget_for(item) for item in tunable}
    total_trials = sum(budgets.values())
    per_recipe: dict[str, dict] = {}
    best_by_algorithm: dict[str, dict] = {}
    for item in non_tunable:
        per_recipe[item] = {"best_params": _jsonable(base_params), "best_metrics": {}, "n_trials": 0, "trials": []}

    if tunable:
        _report_model_tuning_progress(
            ctx,
            {
                "kind": "model_tuning",
                "algorithm": tunable[0],
                "algorithm_index": 1,
                "algorithm_total": len(tunable),
                "trial": 0,
                "trial_total": budgets[tunable[0]],
                "stage": "preparing",
                "completed_trials": 0,
                "total_trials": total_trials,
                "percent": 0.0,
                "best_by_algorithm": {},
            },
        )
        runtime = runtime or _runtime(ctx)
        dataset = dataset or _task_dataset(runtime, ctx, inputs["dataset_id"])
        dataset_path = runtime.registry.resolve_path(dataset.id)
        seed = _effective_seed(inputs, ctx)
        requested_features = [str(f) for f in inputs["features"]]
        early_stopping_rounds = int(inputs.get("early_stopping_rounds", 100))
        max_boost_round = int(inputs.get("max_boost_round", 3000))
        overfit_penalty = float(inputs.get("overfit_penalty", 0.5))
        drop_nan_labels = bool(inputs.get("drop_nan_labels"))
        force_recompute = bool(inputs.get("force_recompute"))
        checkpoint_store = None
        dataset_content_hash = ""
        settings = getattr(runtime, "settings", None)
        if settings is not None and getattr(settings, "tasks_dir", None) is not None:
            checkpoint_store = TuneCheckpointStore(
                _artifact_base_dir(settings, str(ctx.task_id)) / TUNE_CHECKPOINT_DIR_NAME
            )
            dataset_content_hash = dataset_content_identity(dataset, dataset_path)
        completed_before = 0
        for algorithm_offset, item in enumerate(tunable):
            algorithm_index = algorithm_offset + 1
            recipe_seed = _recipe_seed(seed, item)
            checkpoint_identity = None
            if checkpoint_store is not None:
                checkpoint_identity = build_tune_checkpoint_identity(
                    task_id=str(ctx.task_id),
                    dataset_id=dataset.id,
                    dataset_content_hash=dataset_content_hash,
                    features=requested_features,
                    target_col=str(inputs["target_col"]),
                    split_col=str(inputs["split_col"]),
                    split_values=dict(inputs["split_values"]),
                    sample_weight_col=str(control_params.get("sample_weight_col") or ""),
                    drop_nan_labels=drop_nan_labels,
                    recipe=item,
                    n_trials=budgets[item],
                    seed=recipe_seed,
                    cv_folds=cv_folds,
                    early_stopping_rounds=early_stopping_rounds,
                    max_boost_round=max_boost_round,
                    overfit_penalty=overfit_penalty,
                    base_params=_jsonable(base_params),
                    control_params=_jsonable(control_params),
                )

            def on_trial(event: dict, *, _item=item, _index=algorithm_index, _before=completed_before) -> None:
                current_best = {
                    "selection_score": event.get("best_selection_score"),
                    "test_ks": event.get("best_test_ks"),
                }
                best_by_algorithm[_item] = current_best
                local_trial = int(event.get("trial") or 0)
                payload = {
                    **event,
                    "algorithm": _item,
                    "algorithm_index": _index,
                    "algorithm_total": len(tunable),
                    "trial_total": budgets[_item],
                    "completed_trials": _before + local_trial,
                    "total_trials": total_trials,
                    "percent": round(
                        100.0 * (_before + local_trial) / max(1, total_trials),
                        2,
                    ),
                    "best_by_algorithm": {
                        algorithm: dict(metrics)
                        for algorithm, metrics in best_by_algorithm.items()
                    },
                }
                _report_model_tuning_progress(ctx, payload)

            if algorithm_offset > 0:
                _report_model_tuning_progress(
                    ctx,
                    {
                        "kind": "model_tuning",
                        "algorithm": item,
                        "algorithm_index": algorithm_index,
                        "algorithm_total": len(tunable),
                        "trial": 0,
                        "trial_total": budgets[item],
                        "stage": "preparing",
                        "completed_trials": completed_before,
                        "total_trials": total_trials,
                        "percent": round(
                            100.0 * completed_before / max(1, total_trials),
                            2,
                        ),
                        "best_by_algorithm": {
                            algorithm: dict(metrics)
                            for algorithm, metrics in best_by_algorithm.items()
                        },
                    },
                )
            cached = (
                checkpoint_store.load(item, checkpoint_identity)
                if (
                    not force_recompute
                    and checkpoint_store is not None
                    and checkpoint_identity is not None
                )
                else None
            )
            if cached is not None:
                per_recipe[item] = cached
                cached_best = _tuning_progress_best(cached)
                best_by_algorithm[item] = cached_best
                cached_trials = int(cached.get("n_trials") or budgets[item])
                completed_before += cached_trials
                _report_model_tuning_progress(
                    ctx,
                    {
                        "kind": "model_tuning",
                        "algorithm": item,
                        "algorithm_index": algorithm_index,
                        "algorithm_total": len(tunable),
                        "trial": cached_trials,
                        "trial_total": budgets[item],
                        "stage": "checkpoint_hit",
                        "cache_hit": True,
                        "completed_trials": completed_before,
                        "total_trials": total_trials,
                        "percent": round(
                            100.0 * completed_before / max(1, total_trials),
                            2,
                        ),
                        "best_selection_score": cached_best.get("selection_score"),
                        "best_test_ks": cached_best.get("test_ks"),
                        "best_by_algorithm": {
                            algorithm: dict(metrics)
                            for algorithm, metrics in best_by_algorithm.items()
                        },
                    },
                )
                _write_tuning_checkpoint_audit(
                    runtime,
                    ctx,
                    event="hit",
                    recipe=item,
                    identity=checkpoint_identity,
                    cache_hit=True,
                    force_recompute=force_recompute,
                    completed_trials=completed_before,
                    total_trials=total_trials,
                    n_trials=cached_trials,
                )
                continue
            # PERF/P0: each recipe owns a fresh interpreter.  Native learner
            # allocators (notably LightGBM/XGBoost) retain high-water RSS for
            # the lifetime of a process even after every Python object is
            # released; running all recipes in this aggregate worker therefore
            # starved CatBoost against the 4 GiB process-tree ceiling.  The
            # child receives the exact same seed/args and streams trial events
            # back through on_trial; after it exits, the OS reliably releases
            # its entire compact frame and native heap before the next recipe.
            isolated = _run_tuning_recipe_isolated(
                {
                    "dataset_id": dataset.id,
                    "features": requested_features,
                    "target_col": str(inputs["target_col"]),
                    "split_col": str(inputs["split_col"]),
                    "split_values": dict(inputs["split_values"]),
                    "recipe": item,
                    "n_trials": _budget_for(item),
                    # Per-recipe deterministic seed derivation: same base seed
                    # always reproduces the same trial sequence per recipe.
                    "seed": recipe_seed,
                    "early_stopping_rounds": early_stopping_rounds,
                    "max_boost_round": max_boost_round,
                    "overfit_penalty": overfit_penalty,
                    "sample_weight_col": control_params.get("sample_weight_col", ""),
                    "base_params": _jsonable(base_params),
                    "drop_nan_labels": drop_nan_labels,
                    "cv_folds": cv_folds,
                },
                ctx=ctx,
                progress_callback=on_trial,
            )
            best_params = {**control_params, **dict(isolated["best_params"])}
            per_recipe[item] = {
                "best_params": _jsonable(best_params),
                "best_metrics": _jsonable(isolated["best_metrics"]),
                "n_trials": int(isolated["n_trials"]),
                "trials": _jsonable(isolated["trials"]),
                "nan_labels_dropped": int(isolated.get("nan_labels_dropped") or 0),
            }
            if item not in best_by_algorithm:
                best_by_algorithm[item] = {
                    "selection_score": None,
                    "test_ks": isolated["best_metrics"].get("test_ks"),
                }
            completed_before += int(isolated["n_trials"])
            if checkpoint_store is not None and checkpoint_identity is not None:
                checkpoint_path = checkpoint_store.save(
                    item,
                    checkpoint_identity,
                    per_recipe[item],
                )
                _report_model_tuning_progress(
                    ctx,
                    {
                        "kind": "model_tuning",
                        "algorithm": item,
                        "algorithm_index": algorithm_index,
                        "algorithm_total": len(tunable),
                        "trial": int(isolated["n_trials"]),
                        "trial_total": budgets[item],
                        "stage": "checkpoint_saved",
                        "cache_hit": False,
                        "checkpoint_saved": True,
                        "completed_trials": completed_before,
                        "total_trials": total_trials,
                        "percent": round(
                            100.0 * completed_before / max(1, total_trials),
                            2,
                        ),
                        "best_by_algorithm": {
                            algorithm: dict(metrics)
                            for algorithm, metrics in best_by_algorithm.items()
                        },
                    },
                )
                _write_tuning_checkpoint_audit(
                    runtime,
                    ctx,
                    event="saved",
                    recipe=item,
                    identity=checkpoint_identity,
                    cache_hit=False,
                    force_recompute=force_recompute,
                    completed_trials=completed_before,
                    total_trials=total_trials,
                    n_trials=int(isolated["n_trials"]),
                    checkpoint_name=checkpoint_path.name,
                )

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


def _report_model_tuning_progress(ctx, payload: dict) -> None:
    """Best-effort bridge from deterministic tuning to worker telemetry."""

    report = getattr(ctx, "report_progress", None)
    if not callable(report):
        return
    try:
        report(dict(payload))
    except Exception:
        # Progress is an observation channel.  A custom/test context may not
        # implement the ToolContext non-throwing guarantee, so defend here too.
        return


def _write_tuning_checkpoint_audit(
    runtime,
    ctx,
    *,
    event: str,
    recipe: str,
    identity: dict | None,
    cache_hit: bool,
    force_recompute: bool,
    completed_trials: int,
    total_trials: int,
    n_trials: int,
    checkpoint_name: str | None = None,
) -> None:
    """Persist a formal, correlatable record for checkpoint reuse/save.

    PackRuntime always exposes ``repo.write_audit`` in production.  The
    callable guard keeps the deterministic helper usable with deliberately
    minimal unit-test runtimes; when an audit repository is present, failures
    remain fail-closed just like every other governed audit write.
    """

    write_audit = getattr(getattr(runtime, "repo", None), "write_audit", None)
    if not callable(write_audit) or not isinstance(identity, dict):
        return
    dataset_identity = identity.get("dataset")
    dataset_identity = dataset_identity if isinstance(dataset_identity, dict) else {}
    runtime_fingerprint = identity.get("runtime_fingerprint")
    runtime_fingerprint = (
        runtime_fingerprint if isinstance(runtime_fingerprint, dict) else {}
    )
    detail = {
        "contract_version": TUNE_CHECKPOINT_CONTRACT_VERSION,
        "event": str(event),
        "task_id": str(ctx.task_id),
        "recipe": str(recipe),
        "cache_hit": bool(cache_hit),
        "force_recompute": bool(force_recompute),
        "checkpoint_identity_hash": checkpoint_identity_hash(identity),
        "dataset_id": str(dataset_identity.get("id") or ""),
        "dataset_content_hash": str(dataset_identity.get("content_hash") or ""),
        "runtime_fingerprint": str(runtime_fingerprint.get("fingerprint") or ""),
        "n_trials": int(n_trials),
        "completed_trials": int(completed_trials),
        "total_trials": int(total_trials),
        "checkpoint_name": str(checkpoint_name or f"{recipe}.json"),
    }
    write_audit(
        kind=f"modeling.tuning_checkpoint.{event}",
        target_ref=f"{ctx.task_id}:{recipe}",
        outcome="succeeded",
        detail=detail,
    )


def _tuning_progress_best(result: dict) -> dict:
    """Recover the same progress headline from a completed checkpoint."""

    trials = [item for item in result.get("trials", []) if isinstance(item, dict)]
    scored = [item for item in trials if isinstance(item.get("score"), (int, float))]
    best = max(scored, key=lambda item: float(item["score"])) if scored else {}
    metrics = result.get("best_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return {
        "selection_score": best.get("score"),
        "test_ks": best.get("test_ks", metrics.get("test_ks")),
    }


def tool_train_model(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    recipe = str(inputs["recipe"])
    target_type = _validated_target_type([recipe], inputs.get("target_type"))
    train_params = _training_params(inputs)
    preprocessing_steps = _preprocessing_steps_for_training(runtime, dataset.id)
    governance = inputs.get("special_value_governance")
    _assert_sentinel_preprocessing_governed(
        dataset_id=dataset.id,
        dataset_content_hash=sha256_file(runtime.registry.resolve_path(dataset.id)),
        features=inputs["features"],
        sentinel_columns=inputs.get("sentinel_columns"),
        preprocessing_steps=preprocessing_steps,
        governance=governance,
    )
    if preprocessing_steps:
        train_params["preprocessing_steps"] = preprocessing_steps
    elif not _preprocessing_chain_traceable(runtime, dataset.id):
        train_params["preprocessing_chain_traceable"] = False
    if isinstance(governance, dict) and governance:
        train_params[SPECIAL_VALUE_GOVERNANCE_PARAM_KEY] = dict(governance)
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
        target_type=target_type,
        eval_metric=str(inputs.get("eval_metric") or "ks_auc").strip() or "ks_auc",
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


def _reusable_trained_experiment(
    runtime,
    *,
    task_id: str,
    recipe: str,
    config: TrainConfig,
    artifact_dir: Path,
):
    """Return the newest exact, readable result from an interrupted batch.

    ``train_models`` commits each recipe independently.  If the aggregate
    worker is interrupted after one recipe finishes, a retry should continue
    with the remaining recipes instead of creating duplicate experiments and
    retraining already-persisted models.  Reuse is deliberately strict: same
    task, recipe and full immutable TrainConfig, a trained experiment with
    metrics, a registered artifact, and a model file that still exists.
    """

    list_for_task = getattr(runtime.experiments, "list_for_task", None)
    get_artifact = getattr(
        getattr(runtime, "modeling_repo", None),
        "get_model_artifact",
        None,
    )
    if not callable(list_for_task) or not callable(get_artifact):
        return None
    try:
        candidates = list_for_task(str(task_id))
    except Exception:
        return None
    for experiment in reversed(candidates):
        if (
            experiment.recipe_id != recipe
            or experiment.status != "trained"
            or experiment.config != config
            or experiment.metrics is None
            or not experiment.artifact_id
        ):
            continue
        artifact = get_artifact(str(experiment.artifact_id))
        if artifact is None:
            continue
        model_path = Path(str(artifact.model_path))
        if not model_path.is_absolute():
            model_path = artifact_dir / model_path
        if model_path.is_file():
            return experiment
    return None


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
    dataset = _task_dataset(runtime, ctx, inputs["dataset_id"])
    recipes = _normalize_recipe_list(inputs["recipes"])
    tuned_params = dict(inputs.get("params") or {})
    control_params = _training_control_params(inputs, tuned_params)
    per_recipe_params = _params_by_recipe(tuned_params, recipes)
    features = tuple(str(item) for item in inputs["features"])
    target_col = str(inputs["target_col"])
    split_col = str(inputs["split_col"])
    split_values = dict(inputs["split_values"])
    seed = int(inputs["seed"])
    drop_nan = bool(inputs.get("drop_nan_labels"))
    target_type = _validated_target_type(recipes, inputs.get("target_type"))
    # DOM-6: an explicit eval_metric input (e.g. "response_lift" for a marketing/
    # recall scenario) drives champion selection below; every experiment's own
    # TrainConfig also records it so compare_experiments/select_experiment can
    # recover it later without the caller having to repeat it.
    eval_metric = str(inputs.get("eval_metric") or "ks_auc").strip() or "ks_auc"
    dataset_path = runtime.registry.resolve_path(dataset.id)
    preprocessing_steps = read_preprocessing_chain(dataset_path)
    governance = inputs.get("special_value_governance")
    _assert_sentinel_preprocessing_governed(
        dataset_id=dataset.id,
        dataset_content_hash=sha256_file(dataset_path),
        features=features,
        sentinel_columns=inputs.get("sentinel_columns"),
        preprocessing_steps=preprocessing_steps,
        governance=governance,
    )
    training_dataset = None
    training_backend = None
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
        if isinstance(governance, dict) and governance:
            recipe_params[SPECIAL_VALUE_GOVERNANCE_PARAM_KEY] = dict(governance)
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
        artifact_dir = _artifact_base_dir(runtime.settings, ctx.task_id)
        reusable = _reusable_trained_experiment(
            runtime,
            task_id=str(ctx.task_id),
            recipe=recipe,
            config=config,
            artifact_dir=artifact_dir,
        )
        if reusable is not None:
            experiments.append({
                "experiment_id": reusable.id,
                "recipe": recipe,
                "metrics": _jsonable(reusable.metrics) or {},
            })
            continue
        if training_backend is None:
            training_dataset = TrainingDataset.load_compact(
                runtime.backend,
                dataset_path,
                features=features,
                target_col=target_col,
                split_col=split_col,
                extra_columns=[
                    str(control_params.get("sample_weight_col") or ""),
                    *[
                        str(column)
                        for column in (control_params.get("valid_group_cols") or [])
                        if str(column)
                    ],
                ],
            )
            training_backend = training_dataset.backend_adapter(runtime.backend)
        experiment_id = runtime.experiments.create(ctx.task_id, recipe, config)
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
    Regression minimizes test RMSE; multiclass maximizes test macro-AUC,
    falling back to minimizing test logloss. OOT is report-only for every
    target family and never participates in champion selection.
    """
    target_type = str(target_type or "binary")
    if target_type == "continuous":
        def score(experiment: dict) -> float:
            metrics = experiment.get("metrics") or {}
            value = metrics.get("test_rmse")
            if isinstance(value, (int, float)):
                return -float(value)
            return float("-inf")

        return max(experiments, key=score), "test_rmse"
    if target_type == "multiclass":
        def score(experiment: dict) -> float:
            metrics = experiment.get("metrics") or {}
            value = metrics.get("test_macro_auc")
            if isinstance(value, (int, float)):
                return float(value)
            value = metrics.get("test_logloss")
            if isinstance(value, (int, float)):
                return -float(value)
            return float("-inf")

        winner = max(experiments, key=score)
        metric = (
            "test_macro_auc"
            if isinstance((winner.get("metrics") or {}).get("test_macro_auc"), (int, float))
            else "test_logloss"
        )
        return winner, metric

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
    # frame is written into a scratch parquet and re-read by _train_recipe below,
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
    # ``frame`` is a private, projected read used only by this refit. Mutate its
    # split column in place instead of cloning every feature block: on a
    # 750k x 187 sample the historical ``frame.copy()`` doubled the resident
    # modeling frame immediately before the learner allocated its own matrices.
    # The registered source parquet is never modified.
    frame[split_col] = frame[split_col].astype(object)
    frame.loc[combined_idx, split_col] = "__refit_train__"
    frame.loc[shuffled[:holdout_n], split_col] = "__refit_holdout__"
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
        frame.to_parquet(scratch_path, index=False)
        # _train_recipe re-reads scratch_path and native learners allocate their
        # own contiguous matrices. Ensure the source projection and its row-sized
        # masks/index arrays are gone before that allocation begins.
        del frame, train_mask, test_mask, combined_idx, shuffled
        gc.collect()
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
        from marvis.packs.modeling.recipes.lgb import train_lgb

        result = train_lgb(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lgb_regressor":
        from marvis.packs.modeling.recipes.lgb_regressor import train_lgb_regressor

        result = train_lgb_regressor(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "xgb_regressor":
        from marvis.packs.modeling.recipes.xgb_regressor import train_xgb_regressor

        result = train_xgb_regressor(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lr_regressor":
        from marvis.packs.modeling.recipes.lr_regressor import train_lr_regressor

        result = train_lr_regressor(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "mlp_regressor":
        from marvis.packs.modeling.recipes.mlp_regressor import train_mlp_regressor

        result = train_mlp_regressor(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lgb_multiclass":
        from marvis.packs.modeling.recipes.lgb_multiclass import train_lgb_multiclass

        result = train_lgb_multiclass(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "xgb_multiclass":
        from marvis.packs.modeling.recipes.xgb_multiclass import train_xgb_multiclass

        result = train_xgb_multiclass(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lr_multiclass":
        from marvis.packs.modeling.recipes.lr_multiclass import train_lr_multiclass

        result = train_lr_multiclass(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "mlp_multiclass":
        from marvis.packs.modeling.recipes.mlp_multiclass import train_mlp_multiclass

        result = train_mlp_multiclass(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "xgb":
        from marvis.packs.modeling.recipes.xgb import train_xgb

        result = train_xgb(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "catboost":
        from marvis.packs.modeling.recipes.catboost import train_catboost

        result = train_catboost(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "lr":
        from marvis.packs.modeling.recipes.lr import train_lr

        result = train_lr(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "scorecard":
        from marvis.packs.modeling.recipes.scorecard import train_scorecard

        result = train_scorecard(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "mlp":
        from marvis.packs.modeling.recipes.mlp import train_mlp

        result = train_mlp(backend, dataset_path, config, out_dir=out_dir)
    elif recipe == "ensemble":
        from marvis.packs.modeling.recipes.ensemble import train_ensemble

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
