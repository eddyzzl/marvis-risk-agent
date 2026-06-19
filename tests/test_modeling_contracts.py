from dataclasses import asdict

import marvis.packs.modeling as modeling_contracts
from marvis.packs.modeling import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    ModelRecipe,
    TrainConfig,
    TrainResult,
)


def _train_config() -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("score", "income", "age"),
        target_col="bad",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_leaves": 16},
        seed=42,
        early_stopping_rounds=20,
    )


def _metrics() -> ModelMetrics:
    return ModelMetrics(
        train_ks=0.42,
        test_ks=0.38,
        oot_ks=0.35,
        train_auc=0.76,
        test_auc=0.73,
        oot_auc=0.71,
        psi_test_vs_train=0.03,
        psi_oot_vs_train=0.07,
        overfit_train_test_gap=0.095,
        overfit_train_oot_gap=0.07,
        overfit_flag=False,
    )


def test_model_recipe_contract_round_trips():
    recipe = ModelRecipe(
        id="lgb",
        algorithm="lightgbm",
        default_params={"objective": "binary", "learning_rate": 0.05},
        param_space={"num_leaves": [16, 32]},
        requires_woe=False,
    )

    payload = asdict(recipe)

    assert payload["id"] == "lgb"
    assert payload["default_params"]["objective"] == "binary"
    assert payload["param_space"]["num_leaves"] == [16, 32]
    assert payload["requires_woe"] is False


def test_train_result_contract_keeps_metrics_and_artifact_separate():
    artifact = ModelArtifact(
        id="artifact-1",
        experiment_id="experiment-1",
        algorithm="lgb",
        model_path="models/artifact-1/model.txt",
        pmml_path=None,
        feature_list=("score", "income"),
        params={"num_leaves": 16},
        woe_maps=None,
        created_at="2026-06-19T00:00:00Z",
    )
    result = TrainResult(
        artifact=artifact,
        metrics=_metrics(),
        feature_importance=(("score", 0.7), ("income", 0.3)),
        experiment_id="experiment-1",
    )

    payload = asdict(result)

    assert payload["artifact"]["model_path"] == "models/artifact-1/model.txt"
    assert payload["artifact"]["pmml_path"] is None
    assert payload["metrics"]["overfit_train_test_gap"] == 0.095
    assert payload["metrics"]["overfit_train_oot_gap"] == 0.07
    assert payload["feature_importance"][0] == ("score", 0.7)


def test_experiment_contract_allows_created_and_trained_states():
    created = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lgb",
        config=_train_config(),
        metrics=None,
        artifact_id=None,
        status="created",
        created_at="2026-06-19T00:00:00Z",
    )
    trained = Experiment(
        id="experiment-2",
        task_id="task-1",
        recipe_id="scorecard",
        config=_train_config(),
        metrics=_metrics(),
        artifact_id="artifact-1",
        status="trained",
        created_at="2026-06-19T01:00:00Z",
    )

    assert created.metrics is None
    assert created.artifact_id is None
    assert asdict(trained)["metrics"]["test_auc"] == 0.73
    assert asdict(trained)["config"]["seed"] == 42


def test_modeling_package_exports_contract_surface():
    assert modeling_contracts.ModelRecipe is ModelRecipe
    assert modeling_contracts.TrainConfig is TrainConfig
    assert modeling_contracts.ModelMetrics is ModelMetrics
    assert modeling_contracts.ModelArtifact is ModelArtifact
    assert modeling_contracts.TrainResult is TrainResult
    assert modeling_contracts.Experiment is Experiment
    assert "ModelMetrics" in modeling_contracts.__all__
