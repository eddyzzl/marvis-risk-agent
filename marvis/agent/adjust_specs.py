"""Typed adjustment parameter specs for plan-gate recomputation."""

from __future__ import annotations

import math

from marvis.modeling_limits import normalize_n_trials

UNIT_INTERVAL_ADJUST_PARAMS = frozenset({"leakage_ks", "max_missing_rate"})
POSITIVE_INT_ADJUST_PARAMS = frozenset({"n_trials", "num_boost_round"})
NONNEGATIVE_INT_ADJUST_PARAMS = frozenset({"seed"})
SAMPLE_WEIGHT_ADJUST_PARAMS = frozenset({"sample_weight_col"})
MODELING_SETUP_ADJUST_PARAMS = frozenset({"target_type", "recipes", "sample_weight_col"})
TUNING_ADJUST_PARAMS = frozenset({"n_trials", "num_boost_round"})
# The FS-1 multivariate-refinement gate ("精选特征", select_features): iv_min/corr_max
# loosen or effectively bypass the funnel (iv_min=0 + corr_max=1.0 lets everything
# through) without a bespoke "enabled" flag — same generic adjust mechanism as the
# screen gate's leakage_ks/max_missing_rate.
SELECT_ADJUST_PARAMS = frozenset({"iv_min", "corr_max"})
# The G1 split gate ("特征筛选", which depends on the "切分样本"/make_split step) lets
# users override the default split — e.g. switch a time-extrapolated OOT (SEL-1) back
# to random, or move the OOT time boundary — by replacing the whole split_config dict
# make_split was run with (test_size / oot_by_time / oot_size / random_oot / group_cols
# / rules; see marvis/packs/modeling/prepare.py::_make_split).
SPLIT_ADJUST_PARAMS = frozenset({"split_config"})
JOIN_KEY_ADJUST_PARAMS = frozenset({"key_overrides"})
FEATURE_BINNING_ADJUST_PARAMS = frozenset({"features", "bins"})
SPECIAL_VALUE_ADJUST_PARAMS = frozenset({"decisions"})
SUPPORTED_MODELING_RECIPES = frozenset(
    {
        "lgb",
        "xgb",
        "catboost",
        "lr",
        "scorecard",
        "mlp",
        "lgb_regressor",
        "xgb_regressor",
        "lr_regressor",
        "mlp_regressor",
        "lgb_multiclass",
        "xgb_multiclass",
        "lr_multiclass",
        "mlp_multiclass",
        "ensemble",
    }
)

_MODELING_RECIPE_ALIASES = {
    "cat": "catboost",
    "cat_boost": "catboost",
    "cat-boost": "catboost",
    "xgboost": "xgb",
    "xg_boost": "xgb",
    "xg-boost": "xgb",
    "lightgbm": "lgb",
    "light_gbm": "lgb",
    "light-gbm": "lgb",
    "logistic": "lr",
    "logistic_regression": "lr",
    "logistic-regression": "lr",
}

_TIME_OOT_METHOD_ALIASES = frozenset(
    {"time", "time_oot", "time_outer", "time_out", "temporal"}
)
_RANDOM_OOT_METHOD_ALIASES = frozenset(
    {"random", "random_oot", "random_holdout", "random_outer"}
)
_NO_OOT_METHOD_ALIASES = frozenset(
    {"none", "no_oot", "without_oot", "train_test"}
)
_SPLIT_CONFIG_FIELDS = frozenset(
    {
        "test_size",
        "oot_size",
        "oot_by_time",
        "random_oot",
        "group_cols",
        "rules",
    }
)


def normalize_adjust_params(params: dict | None) -> dict:
    """Canonicalize safe aliases emitted by either the UI or an LLM router.

    Structured gate input is still validated after normalization.  This keeps
    common product names such as ``CatBoost`` and the shorthand ``cat`` from
    reaching a tool schema as invalid recipe identifiers, while unknown names
    remain untouched so validation can reject them before any step is reset.
    """

    normalized = dict(params or {})
    recipes = normalized.get("recipes")
    if isinstance(recipes, list):
        clean: list[object] = []
        for recipe in recipes:
            if not isinstance(recipe, str):
                clean.append(recipe)
                continue
            token = recipe.strip().lower().replace(" ", "_")
            clean.append(_MODELING_RECIPE_ALIASES.get(token, token))
        normalized["recipes"] = clean
    if isinstance(normalized.get("split_config"), dict):
        normalized["split_config"] = _normalize_split_config(
            normalized["split_config"]
        )
        # ``split_col`` means an already-materialized partition column.  It is
        # not a modeling split-mode control, but some router responses copied
        # the requested date column into this top-level slot.  The canonical
        # split_config alone is enough to override an uploaded partition and
        # generate the governed ``split`` column.
        normalized.pop("split_col", None)
    return normalized


def _normalize_split_config(value: dict) -> dict:
    config = dict(value)
    _normalize_split_ratio_alias(config, "test_ratio", "test_size")
    _normalize_split_ratio_alias(config, "oot_ratio", "oot_size")
    _normalize_derived_train_ratio(config)
    raw_method = config.get("method")
    method = (
        str(raw_method).strip().lower().replace("-", "_").replace(" ", "_")
        if isinstance(raw_method, str)
        else ""
    )
    alias_column = next(
        (
            str(config.get(key)).strip()
            for key in ("date_col", "time_col", "column")
            if isinstance(config.get(key), str)
            and str(config.get(key)).strip()
        ),
        "",
    )

    if method in _TIME_OOT_METHOD_ALIASES and alias_column:
        config["oot_by_time"] = alias_column
        config.pop("random_oot", None)
        config.pop("method", None)
    elif method in _RANDOM_OOT_METHOD_ALIASES:
        config["random_oot"] = True
        config.pop("oot_by_time", None)
        config.pop("method", None)
    elif method in _NO_OOT_METHOD_ALIASES:
        # Keep an explicit false marker so a free-text "no OOT" adjustment is
        # non-empty and therefore overrides an uploaded split column.
        config["random_oot"] = False
        config.pop("oot_by_time", None)
        config.pop("method", None)
    elif not method and alias_column:
        config["oot_by_time"] = alias_column
        config.pop("random_oot", None)

    if "oot_by_time" in config and config.get("oot_by_time") is not None:
        config["oot_by_time"] = str(config["oot_by_time"]).strip()
    for key in ("date_col", "time_col", "column"):
        config.pop(key, None)
    return config


def _normalize_split_ratio_alias(
    config: dict,
    alias: str,
    canonical: str,
) -> None:
    if alias not in config:
        return
    if canonical not in config:
        config[canonical] = config.pop(alias)
        return
    alias_value = _finite_number(config.get(alias))
    canonical_value = _finite_number(config.get(canonical))
    if (
        alias_value is not None
        and canonical_value is not None
        and math.isclose(alias_value, canonical_value, rel_tol=1e-9, abs_tol=1e-9)
    ):
        config.pop(alias, None)


def _normalize_derived_train_ratio(config: dict) -> None:
    if "train_ratio" not in config:
        return
    ratios = [
        _finite_number(config.get(key))
        for key in ("train_ratio", "test_size", "oot_size")
    ]
    if all(value is not None for value in ratios) and math.isclose(
        sum(ratios),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        # Train share is derived by the split tool. Keep it only long enough to
        # verify a complete LLM-emitted 3-way ratio, never as an executable key.
        config.pop("train_ratio", None)


def has_screen_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & UNIT_INTERVAL_ADJUST_PARAMS)
    )


def has_sample_weight_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & SAMPLE_WEIGHT_ADJUST_PARAMS)
    )


def has_modeling_setup_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & MODELING_SETUP_ADJUST_PARAMS)
    )


def has_tuning_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & TUNING_ADJUST_PARAMS)
    )


def has_split_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & SPLIT_ADJUST_PARAMS)
    )


def has_select_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & SELECT_ADJUST_PARAMS)
    )


def has_join_key_adjust(params: dict | None) -> bool:
    return bool(isinstance(params, dict) and "key_overrides" in params)


def has_feature_binning_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & FEATURE_BINNING_ADJUST_PARAMS)
    )


def has_special_value_adjust(params: dict | None) -> bool:
    return bool(
        isinstance(params, dict)
        and (set(str(key) for key in params) & SPECIAL_VALUE_ADJUST_PARAMS)
    )


def adjust_param_error(params: dict | None) -> str | None:
    for key, value in (params or {}).items():
        if key == "target_type":
            if str(value or "").strip() not in {"binary", "continuous", "multiclass"}:
                return "target_type 必须是 binary、continuous 或 multiclass，未重算。"
        if key == "recipes":
            if not isinstance(value, list) or not value:
                return "recipes 必须是非空算法列表，未重算。"
            clean = [str(item).strip() for item in value if str(item).strip()]
            if len(clean) != len(value) or any(len(item) > 64 or "\x00" in item for item in clean):
                return "recipes 包含无效算法名，未重算。"
            unknown = [item for item in clean if item not in SUPPORTED_MODELING_RECIPES]
            if unknown:
                supported = "、".join(sorted(SUPPORTED_MODELING_RECIPES))
                return f"不支持算法 {', '.join(unknown)}；可选算法为 {supported}，未重算。"
        if key in UNIT_INTERVAL_ADJUST_PARAMS or key in SELECT_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 0 or number > 1:
                return f"{key} 必须是 0 到 1 之间的数字，未重算。"
        if key == "n_trials":
            try:
                normalize_n_trials(value)
            except ValueError:
                return "n_trials 必须是 1 到 200 之间的整数，未重算。"
        elif key in POSITIVE_INT_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 1 or int(number) != number:
                return f"{key} 必须是正整数，未重算。"
        if key in NONNEGATIVE_INT_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 0 or int(number) != number:
                return f"{key} 必须是非负整数，未重算。"
        if key in SAMPLE_WEIGHT_ADJUST_PARAMS:
            if value is None:
                continue
            if not isinstance(value, str):
                return f"{key} 必须是列名字符串，未重算。"
            text = value.strip()
            if len(text) > 128 or "\x00" in text:
                return f"{key} 不是有效列名，未重算。"
        if key in SPLIT_ADJUST_PARAMS:
            error = _split_config_error(value)
            if error:
                return error
        if key in JOIN_KEY_ADJUST_PARAMS:
            if not isinstance(value, dict) or not value:
                return "key_overrides 必须包含至少一张特征表的拼接键选择，未重算。"
            for feature_id, columns in value.items():
                if not isinstance(feature_id, str) or not feature_id.strip() or "\x00" in feature_id:
                    return "key_overrides 包含无效特征表编号，未重算。"
                if not isinstance(columns, list) or not columns:
                    return f"特征表 {feature_id} 至少选择一个拼接键，未重算。"
                clean = [str(column).strip() for column in columns]
                if any(not column or len(column) > 128 or "\x00" in column for column in clean):
                    return f"特征表 {feature_id} 包含无效拼接键，未重算。"
                if len(set(clean)) != len(clean):
                    return f"特征表 {feature_id} 的拼接键不能重复，未重算。"
        if key == "features":
            if not isinstance(value, list):
                return "features 必须是特征名列表，未执行分箱。"
            clean = [str(item).strip() for item in value]
            if any(not item or len(item) > 128 or "\x00" in item for item in clean):
                return "features 包含无效特征名，未执行分箱。"
            if len(set(clean)) != len(clean):
                return "features 不能包含重复特征，未执行分箱。"
        if key == "bins":
            number = _finite_number(value)
            if number is None or int(number) != number or number < 3 or number > 20:
                return "bins 必须是 3 到 20 之间的整数，未执行分箱。"
        if key == "decisions":
            if not isinstance(value, dict):
                return "decisions 必须是按特征列组织的特殊值治理决策，未执行。"
            for raw_column, raw_decision in value.items():
                if not isinstance(raw_column, str):
                    return "decisions 包含无效特征名，未执行。"
                column = raw_column.strip()
                if not column or len(column) > 128 or "\x00" in column:
                    return "decisions 包含无效特征名，未执行。"
                if not isinstance(raw_decision, dict):
                    return f"特征 {column} 的治理决策必须是对象，未执行。"
                unexpected = sorted(
                    str(field)
                    for field in set(raw_decision)
                    - {"action", "values", "confirmed", "reason"}
                )
                if unexpected:
                    return (
                        f"特征 {column} 的治理决策包含不支持字段 "
                        f"{', '.join(unexpected)}，未执行。"
                    )
                action = str(raw_decision.get("action") or "").strip().lower()
                if action not in {"mask", "retain", "drop"}:
                    return f"特征 {column} 的 action 必须是 mask、retain 或 drop，未执行。"
                values = raw_decision.get("values")
                if values is not None:
                    if not isinstance(values, list) or not values:
                        return f"特征 {column} 的 values 必须是非空数字列表，未执行。"
                    if any(_finite_number(item) is None for item in values):
                        return f"特征 {column} 的 values 包含非有限数字，未执行。"
                if action == "retain":
                    if raw_decision.get("confirmed") is not True:
                        return f"保留特征 {column} 的特殊值需要显式确认，未执行。"
                    if not str(raw_decision.get("reason") or "").strip():
                        return f"保留特征 {column} 的特殊值需要填写理由，未执行。"
    return None


def _split_config_error(value) -> str | None:
    if not isinstance(value, dict):
        return "split_config 必须是对象，未重算。"
    for alias, canonical in (
        ("test_ratio", "test_size"),
        ("oot_ratio", "oot_size"),
    ):
        if alias in value:
            return (
                f"split_config.{alias} 与 {canonical} 冲突或不是有效比例，"
                "未重算。"
            )
    if "train_ratio" in value:
        if "test_size" not in value or "oot_size" not in value:
            return (
                "split_config.train_ratio 只能用于同时校验 test_size 和 "
                "oot_size，未重算。"
            )
        return (
            "split_config.train_ratio、test_size、oot_size 之和必须约等于 1，"
            "未重算。"
        )
    unexpected = sorted(str(key) for key in set(value) - _SPLIT_CONFIG_FIELDS)
    if unexpected:
        return (
            "split_config 包含不支持字段 "
            f"{', '.join(unexpected)}，未重算。"
        )
    if "test_size" in value:
        number = _finite_number(value.get("test_size"))
        if number is None or number < 0 or number > 1:
            return "split_config.test_size 必须是 0 到 1 之间的数字，未重算。"
    if "oot_size" in value:
        number = _finite_number(value.get("oot_size"))
        if number is None or number < 0 or number > 1:
            return "split_config.oot_size 必须是 0 到 1 之间的数字，未重算。"
    if "oot_by_time" in value and value.get("oot_by_time") is not None:
        if not isinstance(value["oot_by_time"], str) or not value["oot_by_time"].strip():
            return "split_config.oot_by_time 必须是列名字符串，未重算。"
    if "random_oot" in value and not isinstance(value["random_oot"], bool):
        return "split_config.random_oot 必须是布尔值，未重算。"
    if "group_cols" in value and value.get("group_cols") is not None:
        cols = value["group_cols"]
        if not isinstance(cols, list) or not all(isinstance(item, str) for item in cols):
            return "split_config.group_cols 必须是列名字符串列表，未重算。"
    return None


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "adjust_param_error",
    "normalize_adjust_params",
    "has_modeling_setup_adjust",
    "has_join_key_adjust",
    "has_feature_binning_adjust",
    "has_special_value_adjust",
    "has_sample_weight_adjust",
    "has_screen_adjust",
    "has_select_adjust",
    "has_split_adjust",
    "has_tuning_adjust",
]
