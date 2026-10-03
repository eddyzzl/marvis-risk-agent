"""Check fitted preprocessing against the runtime's actual inner holdouts."""

from __future__ import annotations

import numpy as np

from marvis.data.labels import resolve_modeling_splits
from marvis.data.preprocessing_validation import selected_supervised_fit_members
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.recipes.common import _resolve_valid_group_cols, carve_early_stop_fold, split_modeling_frame


def validate_tuning_preprocessing(registry, inputs):
    recipe = str(inputs["recipe"])
    config = TrainConfig(
        dataset_id=str(inputs["dataset_id"]), features=tuple(inputs["features"]),
        target_col=str(inputs["target_col"]), split_col=str(inputs["split_col"]),
        split_values=dict(inputs["split_values"]), params=dict(inputs.get("base_params") or {}),
        seed=int(inputs["seed"]), early_stopping_rounds=int(inputs.get("early_stopping_rounds", 100)),
        recipe_id=recipe, target_type="multiclass" if recipe.endswith("_multiclass") else "binary",
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    validate_inner_preprocessing(registry, config, cv_folds=inputs.get("cv_folds"), tuning=True)


def validate_inner_preprocessing(registry, config, *, cv_folds=None, tuning=False, frame=None):
    recipe = str(config.recipe_id or "")
    seeds = [config.seed]
    if recipe == "ensemble":
        from marvis.packs.modeling.recipes.ensemble import DEFAULT_ENSEMBLE_N_MEMBERS, _member_seed
        recipe = str(config.params.get("base_recipe") or "lgb").strip()
        count = int(config.params.get("n_members") or DEFAULT_ENSEMBLE_N_MEMBERS)
        seeds = (_member_seed(config.seed, index) for index in range(count))
    tree = recipe in {"lgb", "xgb", "catboost"} or recipe.startswith(("lgb_", "xgb_"))
    early_stop = tree and (tuning or bool(config.early_stopping_rounds))
    private_early_stop = (recipe == "mlp" or recipe.startswith("mlp_")) and bool(config.params.get("early_stopping"))
    if not cv_folds and not early_stop and not private_early_stop:
        return
    guard = selected_supervised_fit_members(registry, config.dataset_id, config.features)
    if not guard.masks:
        return
    if private_early_stop:
        # The estimator creates this split privately after initialization. Do
        # not substitute the platform's different tree validation partition.
        guard.require_known_heldout(context="learner-private early-stopping")
    if tuning:
        from marvis.packs.modeling.tune import _group_cols_from_params
        group_cols = _group_cols_from_params(config.params) or []
    else:
        group_cols = _resolve_valid_group_cols(config)
    if frame is None:
        names = registry.authenticated_parquet_column_names(config.dataset_id)
        columns = list(dict.fromkeys([config.target_col, config.split_col, *[name for name in group_cols if name in names]]))
        frame = registry.read_authenticated_parquet_snapshot(config.dataset_id, columns=columns)
    train, test, oot = split_modeling_frame(frame, config)
    if config.target_type == "multiclass":
        from marvis.packs.modeling.recipes.nonbinary_common import resolve_multiclass_splits
        resolver = resolve_multiclass_splits
    else:
        resolver = resolve_modeling_splits
    train, _, _, _, _ = resolver(train, test, oot, target_col=config.target_col, drop_nan_labels=config.drop_nan_labels)
    if cv_folds:
        from marvis.packs.modeling.tune import _cv_folds
        folds = _cv_folds(train, cv_folds=cv_folds, seed=config.seed, group_cols=group_cols)
        for fold, positions in enumerate(folds):
            guard.check(train.iloc[positions].index, context=f"CV fold {fold + 1}")
            if early_stop:
                fit_positions = np.concatenate([item for number, item in enumerate(folds) if number != fold])
                _, valid = carve_early_stop_fold(train.iloc[fit_positions], seed=config.seed + fold + 1, group_cols=group_cols)
                guard.check(valid.index, context=f"CV fold {fold + 1} early-stopping")
    elif early_stop:
        for seed in seeds:
            _, valid = carve_early_stop_fold(train, seed=seed, group_cols=group_cols)
            guard.check(valid.index, context="early-stopping")
