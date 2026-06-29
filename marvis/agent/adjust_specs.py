"""Typed adjustment parameter specs for plan-gate recomputation."""

from __future__ import annotations

import math

UNIT_INTERVAL_ADJUST_PARAMS = frozenset({"leakage_ks", "max_missing_rate"})
POSITIVE_INT_ADJUST_PARAMS = frozenset({"n_trials", "num_boost_round"})
NONNEGATIVE_INT_ADJUST_PARAMS = frozenset({"seed"})
SAMPLE_WEIGHT_ADJUST_PARAMS = frozenset({"sample_weight_col"})
MODELING_SETUP_ADJUST_PARAMS = frozenset({"target_type", "recipes", "sample_weight_col"})


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
]
