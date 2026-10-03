"""A supervised outer fit must not leak labels into inner validation folds."""

import pandas as pd
import pytest

from marvis.data.preprocessing_evidence import load_preprocessing_state
from marvis.feature.preprocessing import write_preprocessing_chain
from marvis.packs.modeling.recipes.common import carve_early_stop_fold
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


@pytest.fixture
def scenario(tmp_path):
    runner, registry, _, _ = _runtime(tmp_path)
    frame = pd.DataFrame({
        "segment": [f"g{i % 5}" for i in range(120)],
        "x": [float(i % 11) for i in range(120)],
        "y": [i % 2 for i in range(120)],
        "split": ["train"] * 90 + ["test"] * 15 + ["oot"] * 15,
    })
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    result = runner.invoke(ToolRef("feature", "woe_encode_categorical"), {
        "dataset_id": source.id, "features": ["segment"], "target_col": "y",
        "split_col": "split", "min_count": 1,
    }, task_id="task-feature")
    assert result.ok, result.error
    return runner, registry, source, result.output["result_dataset_id"]


def _inputs(dataset_id, **overrides):
    return {
        "dataset_id": dataset_id, "features": ["segment_woe"], "target_col": "y",
        "split_col": "split", "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "seed": 7, **overrides,
    }


def test_cv_rejects_supervised_preprocessing_fitted_on_heldout_members(scenario):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), _inputs(
        dataset_id, recipe="lr", cv_folds=3, n_trials=1,
    ), task_id="task-feature")
    assert not result.ok
    assert "supervised preprocessing" in str(result.error)
    assert "heldout" in str(result.error)
    assert "原始" in str(result.error)


def test_isolated_recipe_worker_checks_membership_without_the_aggregator(scenario):
    from marvis.packs.modeling.tune_isolation import IsolatedRecipeTuningError, run_tuning_recipe_isolated
    from marvis.plugins.contracts import ToolContext

    _, registry, _, dataset_id = scenario
    with pytest.raises(IsolatedRecipeTuningError, match="supervised preprocessing.*heldout"):
        run_tuning_recipe_isolated(_inputs(
            dataset_id, recipe="lr", cv_folds=3, n_trials=1,
            early_stopping_rounds=2, max_boost_round=4, overfit_penalty=0.5,
        ), ctx=ToolContext(
            task_id="task-feature", seed=7, datasets_root=registry.datasets_root,
            workspace=registry.datasets_root.parent,
        ), progress_callback=lambda event: None)


def test_early_stopping_rejects_outer_supervised_fit(scenario):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", "train_model"), _inputs(
        dataset_id, recipe="lgb", early_stopping_rounds=2, params={"num_boost_round": 4},
    ), task_id="task-feature")
    assert not result.ok
    assert "supervised preprocessing" in str(result.error)
    assert "heldout" in str(result.error)


def test_unused_woe_does_not_block_raw_feature_cv(scenario):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), _inputs(
        dataset_id, features=["x"], recipe="lr", cv_folds=3, n_trials=1,
    ), task_id="task-feature")
    assert result.ok, result.error
    assert result.output["n_trials"] == 1


def test_row_local_descendant_still_depends_on_supervised_fit(scenario):
    runner, _, _, dataset_id = scenario
    derived = runner.invoke(ToolRef("feature", "cross_features"), {
        "dataset_id": dataset_id,
        "recipe": [{"kind": "cross", "a": "segment_woe", "b": "x", "ops": ["add"]}],
    }, task_id="task-feature")
    assert derived.ok, derived.error
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), _inputs(
        derived.output["result_dataset_id"], features=["segment_woe_add_x"],
        recipe="lr", cv_folds=3, n_trials=1,
    ), task_id="task-feature")
    assert not result.ok
    assert "supervised preprocessing" in str(result.error)


def test_raw_scorecard_recipe_can_fit_woe_inside_each_cv_fold(scenario):
    runner, _, source, _ = scenario
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), _inputs(
        source.id, features=["x"], recipe="scorecard", cv_folds=3, n_trials=1,
    ), task_id="task-feature")
    assert result.ok, result.error
    assert result.output["n_trials"] == 1


def test_exact_grouped_early_stop_members_can_be_excluded_from_preprocessing_fit(scenario, tmp_path):
    runner, registry, source, _ = scenario
    frame = registry.read_authenticated_parquet_snapshot(source.id)
    frame["group"] = [f"person{i // 5}" for i in range(len(frame))]
    train = frame[frame.split == "train"]
    fit, _ = carve_early_stop_fold(train, seed=7, group_cols=["group"])
    frame["fit_split"] = "test"
    frame.loc[fit.index, "fit_split"] = "train"
    path = tmp_path / "inner_fit.csv"
    frame.to_csv(path, index=False)
    dataset = registry.register_from_upload("task-feature", path, role="sample")
    encoded = runner.invoke(ToolRef("feature", "woe_encode_categorical"), {
        "dataset_id": dataset.id, "features": ["segment"], "target_col": "y",
        "split_col": "fit_split", "min_count": 1,
    }, task_id="task-feature")
    assert encoded.ok, encoded.error
    result = runner.invoke(ToolRef("modeling", "train_model"), _inputs(
        encoded.output["result_dataset_id"], recipe="lgb", early_stopping_rounds=2,
        params={"num_boost_round": 4, "valid_group_cols": ["group"]},
    ), task_id="task-feature")
    assert result.ok, result.error


def test_legacy_supervised_mapping_cannot_claim_independent_cv(scenario, tmp_path):
    runner, registry, _, dataset_id = scenario
    state = load_preprocessing_state(registry, dataset_id)
    path = tmp_path / "legacy.csv"
    registry.read_authenticated_parquet_snapshot(dataset_id).to_csv(path, index=False)
    legacy = registry.register_from_upload("task-feature", path, role="sample")
    # A legacy sidecar supplies scoring parameters but no authenticated fit members.
    write_preprocessing_chain(registry.resolve_path(legacy.id), state.steps)
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), _inputs(
        legacy.id, recipe="lr", cv_folds=3, n_trials=1,
    ), task_id="task-feature")
    assert not result.ok
    assert "fitting membership is unknown" in str(result.error)


def test_train_models_default_early_stop_cannot_bypass_member_check(scenario):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", "train_models"), _inputs(
        dataset_id, recipes=["lgb"], params={"num_boost_round": 4},
    ), task_id="task-feature")
    assert not result.ok
    assert "early-stopping heldout" in str(result.error)


def test_ensemble_member_early_stop_cannot_bypass_member_check(scenario):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", "train_model"), _inputs(
        dataset_id, recipe="ensemble", early_stopping_rounds=2,
        params={"base_recipe": "lgb", "n_members": 2, "num_boost_round": 4},
    ), task_id="task-feature")
    assert not result.ok
    assert "early-stopping heldout" in str(result.error)


@pytest.mark.parametrize("tool", ["train_model", "tune_hyperparameters"])
def test_learner_private_validation_cannot_bypass_supervised_membership_check(scenario, tool):
    runner, _, _, dataset_id = scenario
    result = runner.invoke(ToolRef("modeling", tool), _inputs(
        dataset_id, recipe="mlp", **({"n_trials": 1} if tool == "tune_hyperparameters" else {}),
        params={"early_stopping": True, "max_iter": 2, "hidden_layer_sizes": [4]},
    ), task_id="task-feature")
    assert not result.ok
    assert "early-stopping membership is unknown" in str(result.error)


def test_unneeded_ensemble_guard_does_not_materialize_member_seeds(monkeypatch):
    from marvis.packs.modeling.contracts import TrainConfig
    from marvis.packs.modeling.preprocessing_validation import validate_inner_preprocessing
    from marvis.packs.modeling.recipes import ensemble

    def unexpected_seed(*args):
        pytest.fail("no seed should be derived without an inner holdout")

    monkeypatch.setattr(ensemble, "_member_seed", unexpected_seed)
    validate_inner_preprocessing(None, TrainConfig(
        dataset_id="raw", features=("x",), target_col="y", split_col="split",
        split_values={"train": "train", "test": "test"},
        params={"base_recipe": "lgb", "n_members": 10**12}, seed=7,
        early_stopping_rounds=None, recipe_id="ensemble",
    ))


def test_batch_training_keeps_exact_group_identity_when_group_is_also_a_feature(tmp_path):
    import joblib
    from marvis.packs.modeling._runtime import _artifact_base_dir
    from marvis.repositories.modeling import ModelingRepository
    from marvis.settings import build_settings

    runner, registry, repo, _ = _runtime(tmp_path)
    frame = pd.DataFrame({
        "segment": [f"g{i % 5}" for i in range(180)], "y": [i % 2 for i in range(180)],
        "group": [16777216 + i for i in range(180)],
        "split": ["train"] * 120 + ["test"] * 30 + ["oot"] * 30,
    })
    fit, _ = carve_early_stop_fold(frame[frame.split == "train"], seed=7, group_cols=["group"])
    frame["fit_split"] = "test"
    frame.loc[fit.index, "fit_split"] = "train"
    path = tmp_path / "groups.csv"
    frame.to_csv(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    encoded = runner.invoke(ToolRef("feature", "woe_encode_categorical"), {
        "dataset_id": source.id, "features": ["segment"], "target_col": "y",
        "split_col": "fit_split", "min_count": 1,
    }, task_id="task-feature")
    assert encoded.ok, encoded.error
    result = runner.invoke(ToolRef("modeling", "train_models"), _inputs(
        encoded.output["result_dataset_id"], features=["segment_woe", "group"], recipes=["lgb"],
        params={"num_boost_round": 4, "valid_group_cols": ["group"]},
    ), task_id="task-feature")
    assert result.ok, result.error
    modeling = ModelingRepository(repo.db_path)
    experiment = modeling.get_experiment(result.output["experiments"][0]["experiment_id"])
    artifact = modeling.get_model_artifact(experiment.artifact_id)
    model = joblib.load(_artifact_base_dir(build_settings(registry.datasets_root.parent), "task-feature") / artifact.model_path)
    group_info = model.booster_.dump_model()["feature_infos"]["group"]
    assert group_info["max_value"] == fit.group.max()
    assert group_info["min_value"] == fit.group.min()
