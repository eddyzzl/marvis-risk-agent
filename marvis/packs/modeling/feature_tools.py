from __future__ import annotations

from marvis.data.data_dictionary import first_data_dictionary_id, load_business_names
from marvis.feature.candidates import excluded_categorical_columns, suspected_categorical_columns
from marvis.feature.screen import screen_features, screen_features_non_binary, sentinel_screen_notice
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.select import select_features
from marvis.packs.modeling.tune import DEFAULT_TRIAL_BUDGET

from marvis.packs.modeling._common import PMML_SUPPORTED_ALGORITHMS, _disabled_algorithms, _effective_seed, _eligible_algorithms, _jsonable, _metric_policy_for_target_type, _normalize_modeling_target_type, _normalize_recipe_list, _optional_int, _optional_str, _target_type_from_recipes, _training_params, _unique_strings
from marvis.packs.modeling._runtime import _Runtime, _resolve_feature_cols, _runtime


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
        max_feature_psi=float(inputs["max_feature_psi"]) if inputs.get("max_feature_psi") is not None else None,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    payload = {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "nan_labels_dropped": result.nan_labels_dropped,
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
    if result.psi_watch:
        payload["psi_watch"] = [[feature, psi, reason] for feature, psi, reason in result.psi_watch]
    if result.sentinel_columns:
        payload["sentinel_columns"] = _jsonable(result.sentinel_columns)
        payload["sentinel_notice"] = sentinel_screen_notice(result.sentinel_columns)
    dictionary = _screen_dictionary(runtime, ctx)
    if dictionary:
        payload["dictionary"] = dictionary
    return payload


def _screen_dictionary(runtime: "_Runtime", ctx) -> dict:
    """GAP-4: compact {column: business_name} map for the screen gate's "业务含义"
    column + LLM gate context, when the task has a registered data dictionary.
    Best-effort — {} when no dictionary is registered or it can't be parsed."""
    dictionary_id = first_data_dictionary_id(runtime.registry.list_for_task(ctx.task_id))
    if not dictionary_id:
        return {}
    return load_business_names(runtime.backend, runtime.registry, dictionary_id)


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
    payload = {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [],
        "suspected": [],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "note": "非二分类目标：跳过泄漏KS筛选，已剔除常量/高缺失列",
    }
    dictionary = _screen_dictionary(runtime, ctx)
    if dictionary:
        payload["dictionary"] = dictionary
    return payload


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
    cv_folds = _optional_int(inputs.get("cv_folds"))
    if cv_folds is not None and cv_folds < 2:
        raise ModelingError("cv_folds must be at least 2")
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
        "cv_folds": cv_folds,
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
