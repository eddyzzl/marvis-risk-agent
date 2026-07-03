import pytest

from marvis.db import init_db
from marvis.packs.modeling.contracts import TrainConfig
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.experiment import ExperimentStore
from marvis.packs.modeling.scenarios import (
    apply_scenario,
    get_scenario,
    list_scenarios,
)


def _config(**params) -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params=params,
        seed=31,
        early_stopping_rounds=None,
    )


def test_list_scenarios_exposes_explicit_modeling_templates():
    scenarios = list_scenarios()

    assert [scenario.id for scenario in scenarios] == [
        "loan_pre_a",
        "pre_screen",
        "loan_in",
        "loan_post",
        "marketing",
        "transaction",
        "recall",
        "income",
        "credit_limit",
        "pricing",
    ]
    assert get_scenario("loan_pre_a").default_recipe == "scorecard"
    assert get_scenario("marketing").eval_metric == "response_lift"
    with pytest.raises(KeyError):
        get_scenario("unknown")


def test_apply_scenario_merges_params_and_preserves_user_overrides():
    config = _config(recipe="scorecard", max_depth=5, learning_rate=0.03)

    applied = apply_scenario(config, "loan_pre_a")

    assert applied.params["max_depth"] == 5
    assert applied.params["learning_rate"] == 0.03
    assert applied.recipe_id == "scorecard"
    assert applied.scenario_id == "loan_pre_a"
    assert applied.target_type == "binary"
    assert applied.eval_metric == "ks_auc"


def test_apply_scenario_marks_income_as_continuous_regression():
    config = _config(recipe="lgb_regressor", objective="huber")

    applied = apply_scenario(config, "income")

    assert applied.params["objective"] == "huber"
    assert applied.recipe_id == "lgb_regressor"
    assert applied.scenario_id == "income"
    assert applied.target_type == "continuous"
    assert applied.eval_metric == "rmse_mae"


def test_apply_scenario_rejects_recipe_target_type_mismatch():
    with pytest.raises(ModelingError, match="binary scenario requires a classification recipe"):
        apply_scenario(_config(recipe="lgb_regressor"), "loan_pre_a")
    with pytest.raises(ModelingError, match="continuous scenario requires a regression recipe"):
        apply_scenario(_config(recipe="lgb"), "income")
    # a multiclass recipe is also not a binary classification recipe
    with pytest.raises(ModelingError, match="binary scenario requires a classification recipe"):
        apply_scenario(_config(recipe="lgb_multiclass"), "loan_pre_a")


def test_assert_recipe_matches_target_handles_multiclass():
    from marvis.packs.modeling.scenarios import _assert_recipe_matches_target

    # multiclass target_type requires a multiclass recipe
    _assert_recipe_matches_target("lgb_multiclass", "multiclass")
    with pytest.raises(ModelingError, match="multiclass scenario requires a multiclass recipe"):
        _assert_recipe_matches_target("lgb", "multiclass")
    with pytest.raises(ModelingError, match="multiclass scenario requires a multiclass recipe"):
        _assert_recipe_matches_target("lgb_regressor", "multiclass")


def test_scenario_metadata_persists_with_experiment_config(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = ExperimentStore(db_path)
    applied = apply_scenario(_config(recipe="scorecard"), "loan_pre_a")

    experiment_id = store.create("task-1", applied.recipe_id or "scorecard", applied)
    loaded = store.get(experiment_id)

    assert loaded.config.scenario_id == "loan_pre_a"
    assert loaded.config.recipe_id == "scorecard"
    assert loaded.config.target_type == "binary"
    assert loaded.config.eval_metric == "ks_auc"
