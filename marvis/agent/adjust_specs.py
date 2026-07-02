"""Typed adjustment parameter specs for plan-gate recomputation."""

from __future__ import annotations

import math

UNIT_INTERVAL_ADJUST_PARAMS = frozenset({"leakage_ks", "max_missing_rate"})
POSITIVE_INT_ADJUST_PARAMS = frozenset({"n_trials", "num_boost_round"})
NONNEGATIVE_INT_ADJUST_PARAMS = frozenset({"seed"})
SAMPLE_WEIGHT_ADJUST_PARAMS = frozenset({"sample_weight_col"})
MODELING_SETUP_ADJUST_PARAMS = frozenset({"target_type", "recipes", "sample_weight_col"})
TUNING_ADJUST_PARAMS = frozenset({"n_trials", "num_boost_round"})
# The G1 split gate ("特征筛选", which depends on the "切分样本"/make_split step) lets
# users override the default split — e.g. switch a time-extrapolated OOT (SEL-1) back
# to random, or move the OOT time boundary — by replacing the whole split_config dict
# make_split was run with (test_size / oot_by_time / oot_size / random_oot / group_cols
# / rules; see marvis/packs/modeling/prepare.py::_make_split).
SPLIT_ADJUST_PARAMS = frozenset({"split_config"})


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


def adjust_param_error(params: dict | None) -> str | None:
    for key, value in (params or {}).items():
        if key == "target_type":
            if str(value or "").strip() not in {"binary", "continuous", "multiclass"}:
                return "target_type 必须是 binary、continuous 或 multiclass,未重算。"
        if key == "recipes":
            if not isinstance(value, list) or not value:
                return "recipes 必须是非空算法列表,未重算。"
            clean = [str(item).strip() for item in value if str(item).strip()]
            if len(clean) != len(value) or any(len(item) > 64 or "\x00" in item for item in clean):
                return "recipes 包含无效算法名,未重算。"
        if key in UNIT_INTERVAL_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 0 or number > 1:
                return f"{key} 必须是 0 到 1 之间的数字,未重算。"
        if key in POSITIVE_INT_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 1 or int(number) != number:
                return f"{key} 必须是正整数,未重算。"
        if key in NONNEGATIVE_INT_ADJUST_PARAMS:
            number = _finite_number(value)
            if number is None or number < 0 or int(number) != number:
                return f"{key} 必须是非负整数,未重算。"
        if key in SAMPLE_WEIGHT_ADJUST_PARAMS:
            if value is None:
                continue
            if not isinstance(value, str):
                return f"{key} 必须是列名字符串,未重算。"
            text = value.strip()
            if len(text) > 128 or "\x00" in text:
                return f"{key} 不是有效列名,未重算。"
        if key in SPLIT_ADJUST_PARAMS:
            error = _split_config_error(value)
            if error:
                return error
    return None


def _split_config_error(value) -> str | None:
    if not isinstance(value, dict):
        return "split_config 必须是对象,未重算。"
    if "test_size" in value:
        number = _finite_number(value.get("test_size"))
        if number is None or number < 0 or number > 1:
            return "split_config.test_size 必须是 0 到 1 之间的数字,未重算。"
    if "oot_size" in value:
        number = _finite_number(value.get("oot_size"))
        if number is None or number < 0 or number > 1:
            return "split_config.oot_size 必须是 0 到 1 之间的数字,未重算。"
    if "oot_by_time" in value and value.get("oot_by_time") is not None:
        if not isinstance(value["oot_by_time"], str) or not value["oot_by_time"].strip():
            return "split_config.oot_by_time 必须是列名字符串,未重算。"
    if "random_oot" in value and not isinstance(value["random_oot"], bool):
        return "split_config.random_oot 必须是布尔值,未重算。"
    if "group_cols" in value and value.get("group_cols") is not None:
        cols = value["group_cols"]
        if not isinstance(cols, list) or not all(isinstance(item, str) for item in cols):
            return "split_config.group_cols 必须是列名字符串列表,未重算。"
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
    "has_modeling_setup_adjust",
    "has_sample_weight_adjust",
    "has_screen_adjust",
    "has_split_adjust",
    "has_tuning_adjust",
]
