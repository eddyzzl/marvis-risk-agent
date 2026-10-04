"""Cross-validation whose feature-selection algorithm fits inside each fold."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from marvis.data.labels import resolve_labeled_frame
from marvis.data.preprocessing_validation import selected_supervised_fit_members
from marvis.data.fitted_input_time import validate_fitted_input_time
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import carve_early_stop_fold, normalize_monotone_constraints_value
from marvis.packs.modeling.recipes.nonbinary_common import resolve_multiclass_classes, resolve_multiclass_splits


def _weight_hint(frame, target_col, weight_col):
    from marvis.packs.modeling.tune import _sample_weight, _weighted_count

    target = frame[target_col].to_numpy(dtype=float)
    weights = _sample_weight(frame, weight_col)
    pos, neg = (_weighted_count(target, weights, label) for label in (1.0, 0.0))
    return float(neg / pos) if pos > 0 and neg > 0 else 1.0


def _resolve_weight(params, frame, target_col, weight_col):
    params = deepcopy(params)
    if str(params.get("scale_pos_weight") or "").lower() == "auto":
        params["scale_pos_weight"] = _weight_hint(frame, target_col, weight_col)
    return params


def _parameter_intent(params, candidates):
    params = deepcopy(params or {})
    for key in ("monotone_constraints", "monotonic_constraints"):
        raw = params.get(key)
        if raw is not None and raw != "":
            directions = normalize_monotone_constraints_value(raw, features=candidates)
            params[key] = dict(zip(candidates, directions, strict=True))
    if any(params.get(key) not in (None, False, 0, "") for key in (
        "early_stopping", "early_stopping_round", "early_stopping_rounds",
        "early_stopping_min_delta", "n_iter_no_change", "od_type", "od_wait", "od_pval",
        "use_best_model",
    )):
        raise ModelingError("fold selection requires platform-owned early stopping")
    return params


def _merge_records(records, intent, *, target_type, penalty):
    if target_type == "binary":
        scores = [row.get("weighted_test_ks", row["test_ks"]) for row in records]
        selection_metric = "cv_mean_ks_minus_std_penalty"
        score = float(np.mean(scores) - penalty * np.std(scores))
    else:
        key = "test_rmse" if target_type == "continuous" else "test_logloss"
        scores = [row[key] for row in records]
        selection_metric = f"cv_mean_{key[5:]}_plus_std_penalty"
        score = -float(np.mean(scores) + penalty * np.std(scores))
    merged = {"params": deepcopy(intent), "score": score, "selection_metric": selection_metric,
        "search_stage": records[0]["search_stage"], "cv_fold_metrics": deepcopy(records),
        "cv_fold_scores": scores, "cv_score_std": float(np.std(scores))}
    for key in set.intersection(*(set(row) for row in records)) - {
        "params", "score", "best_iteration", "search_stage", "selection_metric"}:
        values = [row[key] for row in records]
        if all(isinstance(value, (int, float, np.number)) for value in values):
            merged[key] = float(np.mean(values))
        elif all(value is None for value in values):
            merged[key] = None
    if target_type == "binary":
        merged["cv_fold_test_ks"] = scores
        merged["test_ks_std"] = float(np.std(scores))
    rounds = [row["best_iteration"] for row in records if row.get("best_iteration") is not None]
    if rounds:
        if len(rounds) != len(records) or any(type(value) is not int or value < 1 for value in rounds):
            raise ModelingError("every fold must provide a positive early-stopping iteration count")
        merged["best_iteration"] = int(math.ceil(float(np.median(rounds))))
    return merged


def prepare_fold_layout(session, plan, *, recipe, seed, cv_folds, base_params=None, drop_nan_labels=False):
    """Revalidate actual memberships before either cache reuse or learner setup."""
    from marvis.packs.modeling import tune

    if type(cv_folds) is not int or cv_folds < 2:
        raise ModelingError("fold selection requires cv_folds >= 2")
    if recipe not in tune.DEFAULT_TRIAL_BUDGET:
        raise ModelingError("fold selection requires a supported tuning recipe")
    target_type = ("multiclass" if recipe in tune._MULTICLASS_TUNING_RECIPES else
                   "continuous" if recipe in tune._REGRESSION_TUNING_RECIPES else "binary")
    groups = tune._group_cols_from_params(base_params) or []
    target_col = plan["target_col"]
    controls = session.training_controls(split_col=plan["split_col"],
        train_value=plan["split_values"]["train"], group_columns=groups)
    if target_type == "multiclass":
        # Reuse the label gate with training rows only; the duplicate frame is
        # not a validation population and its audit count is not accumulated.
        train, _, _, _, audit = resolve_multiclass_splits(controls, controls, None,
            target_col=target_col, drop_nan_labels=drop_nan_labels)
        dropped = len(controls) - len(train)
    else:
        train, dropped = resolve_labeled_frame(controls, target_col=target_col,
                                              drop_nan_labels=drop_nan_labels)
    fixed_intent = _parameter_intent(base_params, plan["candidates"])
    folds = tune._cv_folds(train, cv_folds=cv_folds, seed=seed, group_cols=groups)
    guard = selected_supervised_fit_members(session.registry, session.source.id, plan["candidates"])
    tree = recipe in tune._EARLY_STOPPED_TREE_RECIPES or recipe.startswith(("lgb_", "xgb_"))
    layout = []
    for fold, test_positions in enumerate(folds):
        outer = train.iloc[np.concatenate([positions for i, positions in enumerate(folds) if i != fold])]
        heldout = train.iloc[test_positions]
        fit, valid = (carve_early_stop_fold(outer, seed=seed + fold + 1, group_cols=groups)
                      if tree else (outer, heldout))
        guard.check(heldout.index, context=f"fold selection CV {fold + 1}")
        if tree:
            guard.check(valid.index, context=f"fold selection early-stopping {fold + 1}")
        fit_mask = np.zeros(session.source.row_count, dtype=bool)
        eval_mask = fit_mask.copy()
        fit_mask[fit.index] = True
        eval_mask[np.r_[valid.index, heldout.index]] = True
        time_evidence = validate_fitted_input_time(session.registry, session.source.id,
            plan["candidates"], fit_mask=fit_mask, evaluation_mask=eval_mask, requires_labels=True,
            evaluation_roles=["cv_holdout", "early_stopping"] if tree else ["cv_holdout"])
        layout.append((outer, heldout, fit, valid, time_evidence))
    return train, dropped, target_type, tree, fixed_intent, layout


def tune_fold_selection(session, plan, *, recipe, seed, cv_folds, n_trials,
                        base_params=None, early_stopping_rounds=100, max_boost_round=3000,
                        overfit_penalty=.5, drop_nan_labels=False, coarse_fraction=.6,
                        progress_callback=None):
    """Use explicit native folds; no outer test or OOT labels enter this search."""
    from marvis.packs.modeling import tune

    train, dropped, target_type, tree, fixed_intent, layout = prepare_fold_layout(session, plan,
        recipe=recipe, seed=seed, cv_folds=cv_folds, base_params=base_params, drop_nan_labels=drop_nan_labels)
    target_col, weight_col = plan["target_col"], plan["sample_weight_col"]
    fold_runners, selections = [], []
    for fold, (outer, heldout, fit, valid, time_evidence) in enumerate(layout):
        selected = session.prepare_fold(fit_positions=fit.index.to_numpy(),
            train_positions=outer.index.to_numpy(), valid_positions=valid.index.to_numpy(),
            test_positions=heldout.index.to_numpy())
        selections.append({"selection": selected.evidence, "input_time": time_evidence})
        fixed = _resolve_weight(fixed_intent, selected.fit, target_col, weight_col)
        wtr, wte, wfit, wvalid = (tune._sample_weight(frame, weight_col)
            for frame in (selected.train, selected.test, selected.fit, selected.valid))
        if target_type == "binary":
            coarse_sampler, fine_sampler, runner = tune._recipe_search_hooks(recipe,
                train=selected.train, test=selected.test, oot=None, fit_train=selected.fit,
                valid=selected.valid, feats=list(selected.features), target_col=target_col,
                ytr=selected.train[target_col].to_numpy(dtype=float), yte=selected.test[target_col].to_numpy(dtype=float),
                yfit=selected.fit[target_col].to_numpy(dtype=float), yva=selected.valid[target_col].to_numpy(dtype=float),
                wtr=wtr, wte=wte, woot=None, wfit=wfit, wva=wvalid, seed=seed + fold + 1,
                early_stopping_rounds=early_stopping_rounds, max_boost_round=max_boost_round,
                overfit_penalty=overfit_penalty, oot_has_labels=False, base_params=fixed,
                pos_weight_hint=_weight_hint(selected.fit, target_col, weight_col))
        else:
            coarse_sampler, fine_sampler = tune._nonbinary_samplers(recipe)
            if recipe.startswith("lgb_"):
                fixed = tune._normalize_lgb_monotone_constraints(fixed, list(selected.features))
            elif recipe.startswith("xgb_"):
                fixed = tune._normalize_xgb_monotone_constraints(fixed, list(selected.features))
            classes = resolve_multiclass_classes(selected.fit[target_col]) if target_type == "multiclass" else None

            def runner(sampled, stage, *, selected=selected, fixed=fixed, classes=classes,
                       wtr=wtr, wte=wte, wfit=wfit, wvalid=wvalid, fold=fold):
                return tune._run_nonbinary_trial(recipe, sampled, stage,
                    fixed_params=tune._base_params_without_controls(fixed), seed=seed + fold + 1,
                    train=selected.train, test=selected.test, oot=None, fit_train=selected.fit,
                    valid=selected.valid, feats=list(selected.features), target_col=target_col, classes=classes,
                    early_stopping_rounds=early_stopping_rounds, max_boost_round=max_boost_round,
                    overfit_penalty=overfit_penalty, oot_has_labels=False,
                    wtr=wtr, wte=wte, woot=None, wfit=wfit, wvalid=wvalid)
        fold_runners.append((runner, selected.fit))

    total = max(1, int(n_trials) if n_trials is not None else tune.DEFAULT_TRIAL_BUDGET[recipe])
    n_coarse = min(total, max(1, round(total * coarse_fraction)))
    rngs = {"coarse": np.random.RandomState(seed), "fine": np.random.RandomState(seed + 1)}
    trials, best, anchor = [], None, None
    for trial in range(total):
        stage = "coarse" if trial < n_coarse else "fine"
        sampled = (coarse_sampler(rngs[stage], 0.0) if stage == "coarse"
                   else fine_sampler(rngs[stage], 0.0, anchor))
        # Zero is only an internal sampler slot; the stored policy is explicit
        # and each fold resolves it using its own actual fit labels/weights.
        if sampled.get("scale_pos_weight") == 0.0:
            sampled["scale_pos_weight"] = "auto"
        records = [runner(_resolve_weight(sampled, fit, target_col, weight_col), stage)
                   for runner, fit in fold_runners]
        record = _merge_records(records, {**sampled, **fixed_intent},
                                target_type=target_type, penalty=overfit_penalty)
        record["sampled_parameters"] = deepcopy(sampled)
        trials.append(record)
        if best is None or record["score"] > best["score"]:
            best = record
        if trial == n_coarse - 1:
            anchor = deepcopy(best["sampled_parameters"])
        tune._report_trial_progress(progress_callback, recipe=recipe, record=record,
            best=best, trial=trial + 1, trial_total=total)
    final = session.prepare_final_fit(train.index.to_numpy())
    final_params = _resolve_weight(best["params"], final.train, target_col, weight_col)
    if recipe == "lgb" or recipe.startswith("lgb_"):
        final_params = tune._normalize_lgb_monotone_constraints(final_params, list(final.features))
    elif recipe == "xgb" or recipe.startswith("xgb_"):
        final_params = tune._normalize_xgb_monotone_constraints(final_params, list(final.features))
    if recipe == "catboost":
        final_params["use_best_model"] = False
    if tree:
        final_params["iterations" if recipe == "catboost" else "num_boost_round"] = best["best_iteration"]
    evidence = {"schema_version": "marvis.fold_tuning.v1", "plan": deepcopy(plan),
        "recipe": recipe, "seed": seed, "cv_folds": cv_folds, "folds": selections,
        "final_selection": final.evidence, "parameter_intent": deepcopy(best["params"]),
        "round_aggregation": plan["round_aggregation"], "final_early_stopping": False,
        "inner_selection_scope": "supplied_candidate_algorithm_fit_only",
        "upstream_candidate_selection": "unknown", "historical_availability": "unknown",
        "oot_evaluation": "not_used_for_selection_or_tuning",
        "oot_physical_read_isolation": "not_established_shared_parquet"}
    metrics = {key: deepcopy(value) for key, value in best.items()
               if key not in {"params", "cv_fold_metrics", "sampled_parameters"}}
    metrics["selection_score"] = best["score"]
    return tune.TuneResult(best_params=final_params, best_metrics=metrics, trials=tuple(trials),
        n_trials=total, nan_labels_dropped=int(dropped), recipe=recipe,
        selected_features=final.features, fold_selection_evidence=evidence)
