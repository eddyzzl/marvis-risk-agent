"""Deterministic red-flag checklists for modeling gates (AGT-9): pure functions
computed straight from platform tool outputs (INV-1: the LLM re-checks, it
never computes)."""

from __future__ import annotations

from marvis.agent.modeling_red_flags import select_experiment_red_flags, tuning_setup_red_flags


def _split_output(counts: dict, total_rows: int | None = None) -> dict:
    return {
        "sample_analysis": {
            "split_counts": counts,
            "total_rows": total_rows if total_rows is not None else sum(counts.values()),
        }
    }


# -- tuning_setup_red_flags (调参配置门) --------------------------------------


def test_tuning_setup_flags_small_sample_size():
    flags = tuning_setup_red_flags(
        split_output=_split_output({"train": 2000, "test": 500, "oot": 500}),
        modeling_spec_output={"feature_count": 20},
    )
    assert any("样本量偏小" in flag for flag in flags)
    assert not any("OOT" in flag for flag in flags)
    assert not any("特征数" in flag for flag in flags)


def test_tuning_setup_flags_missing_oot():
    flags = tuning_setup_red_flags(
        split_output=_split_output({"train": 8000, "test": 2000, "oot": 0}),
        modeling_spec_output={"feature_count": 20},
    )
    assert any("缺少 OOT" in flag for flag in flags)


def test_tuning_setup_flags_too_many_features():
    flags = tuning_setup_red_flags(
        split_output=_split_output({"train": 8000, "test": 2000, "oot": 1000}),
        modeling_spec_output={"feature_count": 250},
    )
    assert any("入模候选特征数偏多" in flag and "250" in flag for flag in flags)


def test_tuning_setup_flags_clean_inputs_produce_no_flags():
    flags = tuning_setup_red_flags(
        split_output=_split_output({"train": 8000, "test": 2000, "oot": 1000}),
        modeling_spec_output={"feature_count": 30},
    )
    assert flags == []


def test_tuning_setup_flags_tolerates_missing_outputs():
    assert tuning_setup_red_flags(split_output=None, modeling_spec_output=None) == []


# -- select_experiment_red_flags (选实验门) ------------------------------------


def test_select_experiment_flags_train_test_overfit_gap():
    tune_output = {"trials": [{"train_ks": 0.52, "test_ks": 0.40}]}
    flags = select_experiment_red_flags(tune_output=tune_output, train_models_output=None)
    assert any("过拟合迹象" in flag and "0.120" in flag for flag in flags)


def test_select_experiment_flags_no_overfit_gap_below_threshold():
    tune_output = {"trials": [{"train_ks": 0.45, "test_ks": 0.40}]}
    flags = select_experiment_red_flags(tune_output=tune_output, train_models_output=None)
    assert not any("过拟合迹象" in flag for flag in flags)


def test_select_experiment_flags_thin_champion_runner_up_gap():
    train_models_output = {
        "experiments": [
            {"recipe": "lgb", "metrics": {"test_ks": 0.401}},
            {"recipe": "xgb", "metrics": {"test_ks": 0.400}},
        ],
    }
    flags = select_experiment_red_flags(tune_output=None, train_models_output=train_models_output)
    assert any("冠军与亚军" in flag for flag in flags)


def test_select_experiment_flags_healthy_champion_margin_no_flag():
    train_models_output = {
        "experiments": [
            {"recipe": "lgb", "metrics": {"test_ks": 0.45}},
            {"recipe": "xgb", "metrics": {"test_ks": 0.30}},
        ],
    }
    flags = select_experiment_red_flags(tune_output=None, train_models_output=train_models_output)
    assert not any("冠军与亚军" in flag for flag in flags)


def test_select_experiment_flags_weighted_unweighted_mismatch():
    train_models_output = {
        "experiments": [
            {"recipe": "lgb", "metrics": {"test_ks": 0.45, "weighted_test_ks": 0.30}},
            {"recipe": "xgb", "metrics": {"test_ks": 0.40, "weighted_test_ks": 0.50}},
        ],
    }
    flags = select_experiment_red_flags(tune_output=None, train_models_output=train_models_output)
    assert any("加权" in flag and "未加权" in flag and "不一致" in flag for flag in flags)


def test_select_experiment_flags_weighted_unweighted_agreement_no_flag():
    train_models_output = {
        "experiments": [
            {"recipe": "lgb", "metrics": {"test_ks": 0.45, "weighted_test_ks": 0.50}},
            {"recipe": "xgb", "metrics": {"test_ks": 0.40, "weighted_test_ks": 0.30}},
        ],
    }
    flags = select_experiment_red_flags(tune_output=None, train_models_output=train_models_output)
    assert not any("加权" in flag and "不一致" in flag for flag in flags)


def test_select_experiment_flags_any_failed_candidate():
    train_models_output = {
        "experiments": [{"recipe": "lgb", "metrics": {"test_ks": 0.40}}],
        "failed": [{"recipe": "catboost", "error": "ValueError: bad data"}],
    }
    flags = select_experiment_red_flags(tune_output=None, train_models_output=train_models_output)
    assert any("训练阶段有 1 个候选算法失败" in flag and "catboost" in flag for flag in flags)


def test_select_experiment_flags_clean_inputs_produce_no_flags():
    tune_output = {"trials": [{"train_ks": 0.42, "test_ks": 0.40}]}
    train_models_output = {
        "experiments": [
            {"recipe": "lgb", "metrics": {"test_ks": 0.40}},
            {"recipe": "xgb", "metrics": {"test_ks": 0.30}},
        ],
        "failed": [],
    }
    flags = select_experiment_red_flags(tune_output=tune_output, train_models_output=train_models_output)
    assert flags == []


def test_select_experiment_flags_tolerates_missing_outputs():
    assert select_experiment_red_flags(tune_output=None, train_models_output=None) == []
