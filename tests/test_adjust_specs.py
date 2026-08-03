from __future__ import annotations

import pytest

from marvis.agent.adjust_specs import (
    adjust_param_error,
    normalize_adjust_params,
)


def test_normalize_adjust_params_canonicalizes_time_oot_router_aliases() -> None:
    normalized = normalize_adjust_params(
        {
            "split_col": "apply_month",
            "split_config": {
                "method": "time_outer",
                "date_col": "apply_month",
            },
            "recipes": ["LR", "XGBoost", "LightGBM"],
        }
    )

    assert normalized == {
        "split_config": {"oot_by_time": "apply_month"},
        "recipes": ["lr", "xgb", "lgb"],
    }
    assert adjust_param_error(normalized) is None


def test_normalize_adjust_params_canonicalizes_random_and_no_oot_aliases() -> None:
    assert normalize_adjust_params(
        {"split_config": {"method": "random", "oot_size": 0.2}}
    ) == {
        "split_config": {"random_oot": True, "oot_size": 0.2}
    }
    assert normalize_adjust_params(
        {"split_config": {"method": "none"}}
    ) == {
        "split_config": {"random_oot": False}
    }


def test_normalize_adjust_params_canonicalizes_split_ratio_aliases() -> None:
    normalized = normalize_adjust_params(
        {
            "split_config": {
                "method": "random",
                "test_ratio": 0.2,
                "oot_ratio": 0.2,
                "train_ratio": 0.6,
            }
        }
    )

    assert normalized == {
        "split_config": {
            "random_oot": True,
            "test_size": 0.2,
            "oot_size": 0.2,
        }
    }
    assert adjust_param_error(normalized) is None


def test_split_ratio_alias_must_agree_with_existing_canonical_value() -> None:
    consistent = normalize_adjust_params(
        {"split_config": {"test_size": 0.2, "test_ratio": 0.2}}
    )
    assert consistent == {"split_config": {"test_size": 0.2}}

    conflicting = normalize_adjust_params(
        {"split_config": {"test_size": 0.25, "test_ratio": 0.2}}
    )
    assert "test_ratio 与 test_size 冲突" in str(
        adjust_param_error(conflicting)
    )


def test_train_ratio_is_only_a_consistency_check() -> None:
    inconsistent = normalize_adjust_params(
        {
            "split_config": {
                "test_ratio": 0.2,
                "oot_ratio": 0.2,
                "train_ratio": 0.5,
            }
        }
    )
    assert "train_ratio、test_size、oot_size 之和必须约等于 1" in str(
        adjust_param_error(inconsistent)
    )

    under_specified = normalize_adjust_params(
        {"split_config": {"test_ratio": 0.2, "train_ratio": 0.8}}
    )
    assert "train_ratio 只能用于同时校验 test_size 和 oot_size" in str(
        adjust_param_error(under_specified)
    )


def test_split_config_rejects_unknown_fields_after_normalization() -> None:
    normalized = normalize_adjust_params(
        {"split_config": {"strategy": "invented"}}
    )

    assert "不支持字段 strategy" in str(adjust_param_error(normalized))


@pytest.mark.parametrize("value", [0, 201, 1.5, True, "7"])
def test_n_trials_adjust_rejects_unsafe_or_coerced_values(value) -> None:
    assert "1 到 200" in str(adjust_param_error({"n_trials": value}))


@pytest.mark.parametrize("value", [1, 200])
def test_n_trials_adjust_accepts_inclusive_bounds(value) -> None:
    assert adjust_param_error({"n_trials": value}) is None
