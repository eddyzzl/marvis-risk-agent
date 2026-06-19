from marvis.db import ModelingRepository, connect, init_db
from marvis.packs.modeling import (
    Experiment,
    ModelArtifact,
    ModelMetrics,
    TrainConfig,
)


def _config() -> TrainConfig:
    return TrainConfig(
        dataset_id="dataset-1",
        features=("score", "income"),
        target_col="bad",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={"num_leaves": 16},
        seed=7,
        early_stopping_rounds=10,
    )


def _metrics() -> ModelMetrics:
    return ModelMetrics(
        train_ks=0.41,
        test_ks=0.37,
        oot_ks=0.35,
        train_auc=0.78,
        test_auc=0.74,
        oot_auc=0.72,
        psi_test_vs_train=0.04,
        psi_oot_vs_train=0.08,
        overfit_train_test_gap=0.0976,
        overfit_train_oot_gap=0.06,
        overfit_flag=True,
    )


def test_modeling_repository_round_trips_experiment_and_artifact(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lgb",
        config=_config(),
        metrics=_metrics(),
        artifact_id="artifact-1",
        status="trained",
        created_at="2026-06-19T00:00:00Z",
    )
    artifact = ModelArtifact(
        id="artifact-1",
        experiment_id=experiment.id,
        algorithm="lgb",
        model_path="models/artifact-1/model.txt",
        pmml_path=None,
        feature_list=("score", "income"),
        params={"num_leaves": 16},
        woe_maps=None,
        created_at="2026-06-19T00:01:00Z",
    )

    repo.create_experiment(experiment)
    repo.create_model_artifact(artifact)

    assert repo.get_experiment(experiment.id) == experiment
    assert repo.get_model_artifact(artifact.id) == artifact
    assert repo.list_experiments("task-1") == [experiment]


def test_model_artifacts_cascade_when_experiment_is_deleted(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lr",
        config=_config(),
        metrics=None,
        artifact_id=None,
        status="created",
        created_at="2026-06-19T00:00:00Z",
    )
    artifact = ModelArtifact(
        id="artifact-1",
        experiment_id=experiment.id,
        algorithm="lr",
        model_path="models/artifact-1/model.pkl",
        pmml_path="models/artifact-1/model.pmml",
        feature_list=("score",),
        params={"C": 1.0},
        woe_maps={"score": {"0": -0.2}},
        created_at="2026-06-19T00:01:00Z",
    )
    repo.create_experiment(experiment)
    repo.create_model_artifact(artifact)

    with connect(db_path) as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment.id,))

    assert repo.get_experiment(experiment.id) is None
    assert repo.get_model_artifact(artifact.id) is None
