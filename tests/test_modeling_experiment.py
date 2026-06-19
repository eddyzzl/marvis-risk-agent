import pytest

from marvis.db import ModelingRepository, init_db
from marvis.packs.modeling import ModelArtifact, ModelMetrics, TrainConfig, TrainResult
from marvis.packs.modeling.experiment import ExperimentStore


def _config(recipe_param: str = "lr") -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("x1", "x2"),
        target_col="y",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"recipe_param": recipe_param},
        seed=17,
        early_stopping_rounds=None,
    )


def _metrics(test_ks: float, *, overfit: bool = False) -> ModelMetrics:
    return ModelMetrics(
        train_ks=0.42,
        test_ks=test_ks,
        oot_ks=0.35,
        train_auc=0.78,
        test_auc=0.74,
        oot_auc=0.71,
        psi_test_vs_train=0.03,
        psi_oot_vs_train=0.07,
        overfit_train_test_gap=0.08,
        overfit_train_oot_gap=0.07,
        overfit_flag=overfit,
    )


def _result(artifact_id: str, metrics: ModelMetrics) -> TrainResult:
    return TrainResult(
        artifact=ModelArtifact(
            id=artifact_id,
            experiment_id="",
            algorithm="lr",
            model_path=f"{artifact_id}.joblib",
            pmml_path=None,
            feature_list=("x1", "x2"),
            params={"C": 1.0},
            woe_maps=None,
            created_at="2026-06-19T00:01:00Z",
        ),
        metrics=metrics,
        feature_importance=(("x1", 0.7), ("x2", 0.3)),
        experiment_id="",
    )


def test_experiment_store_create_get_list_and_status(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = ExperimentStore(db_path)

    experiment_id = store.create("task-1", "lr", _config())
    store.set_status(experiment_id, "failed")
    experiment = store.get(experiment_id)

    assert experiment.id == experiment_id
    assert experiment.task_id == "task-1"
    assert experiment.recipe_id == "lr"
    assert experiment.status == "failed"
    assert experiment.metrics is None
    assert store.list_for_task("task-1") == [experiment]
    with pytest.raises(KeyError):
        store.get("missing")


def test_experiment_store_attach_result_persists_artifact_and_metrics(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = ExperimentStore(db_path)
    experiment_id = store.create("task-1", "lr", _config())

    store.attach_result(experiment_id, _result("artifact-1", _metrics(0.37)))

    experiment = store.get(experiment_id)
    artifact = ModelingRepository(db_path).get_model_artifact("artifact-1")
    assert experiment.status == "trained"
    assert experiment.artifact_id == "artifact-1"
    assert experiment.metrics is not None
    assert experiment.metrics.test_ks == 0.37
    assert artifact is not None
    assert artifact.experiment_id == experiment_id


def test_experiment_store_compare_returns_metric_rows_in_requested_order(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = ExperimentStore(db_path)
    first = store.create("task-1", "lr", _config("first"))
    second = store.create("task-1", "scorecard", _config("second"))
    store.attach_result(first, _result("artifact-1", _metrics(0.37)))
    store.attach_result(second, _result("artifact-2", _metrics(0.33, overfit=True)))

    comparison = store.compare([second, first])

    assert [row["id"] for row in comparison["experiments"]] == [second, first]
    assert comparison["experiments"][0]["recipe"] == "scorecard"
    assert comparison["experiments"][0]["test_ks"] == 0.33
    assert comparison["experiments"][0]["overfit_flag"] is True
    assert comparison["experiments"][1]["recipe"] == "lr"
