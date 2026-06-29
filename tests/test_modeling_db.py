import pytest

import marvis.db as db_module
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
        feature_importance=(("score", 0.7), ("income", 0.3)),
        scorecard_table=(
            {
                "feature": "score",
                "bin_index": 1,
                "bin_label": "[0, 1)",
                "points": 12.3,
            },
        ),
    )

    repo.create_experiment(experiment)
    repo.create_model_artifact(artifact)

    assert repo.get_experiment(experiment.id) == experiment
    assert repo.get_model_artifact(artifact.id) == artifact
    assert repo.list_experiments("task-1") == [experiment]


def test_modeling_repository_attaches_artifact_and_audit_atomically(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lgb",
        config=_config(),
        metrics=None,
        artifact_id=None,
        status="created",
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

    repo.attach_experiment_result_with_artifact_and_audit(
        experiment.id,
        artifact=artifact,
        metrics=_metrics(),
        audit={
            "kind": "modeling.experiment.trained",
            "target_ref": experiment.id,
            "outcome": "succeeded",
            "detail": {"artifact_id": artifact.id},
        },
    )

    loaded = repo.get_experiment(experiment.id)
    assert loaded.status == "trained"
    assert loaded.artifact_id == artifact.id
    assert repo.get_model_artifact(artifact.id) == artifact
    audit = db_module.PluginRepository(db_path).list_audit(kind="modeling.experiment.trained")[0]
    assert audit["target_ref"] == experiment.id
    assert audit["detail"]["artifact_id"] == artifact.id


def test_modeling_repository_rolls_back_artifact_attach_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lgb",
        config=_config(),
        metrics=None,
        artifact_id=None,
        status="created",
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

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.attach_experiment_result_with_artifact_and_audit(
            experiment.id,
            artifact=artifact,
            metrics=_metrics(),
            audit={
                "kind": "modeling.experiment.trained",
                "target_ref": experiment.id,
                "outcome": "succeeded",
            },
        )

    loaded = repo.get_experiment(experiment.id)
    assert loaded.status == "created"
    assert loaded.artifact_id is None
    assert loaded.metrics is None
    assert repo.get_model_artifact(artifact.id) is None


def test_modeling_repository_rolls_back_artifact_params_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = ModelingRepository(db_path)
    experiment = Experiment(
        id="experiment-1",
        task_id="task-1",
        recipe_id="lgb",
        config=_config(),
        metrics=None,
        artifact_id=None,
        status="created",
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

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.set_model_artifact_params_with_audit(
            artifact.id,
            {"num_leaves": 16, "calibration": {"method": "sigmoid"}},
            audit={
                "kind": "modeling.artifact.calibrate",
                "target_ref": artifact.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_model_artifact(artifact.id).params == {"num_leaves": 16}


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


def test_model_artifact_migration_adds_empty_scorecard_table_for_old_rows(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                artifact_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE model_artifacts (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                model_path TEXT NOT NULL,
                pmml_path TEXT,
                feature_list_json TEXT NOT NULL,
                feature_importance_json TEXT NOT NULL DEFAULT '[]',
                params_json TEXT NOT NULL,
                woe_maps_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO experiments(
                id, task_id, recipe_id, config_json, metrics_json, artifact_id, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "experiment-legacy",
                "task-1",
                "scorecard",
                '{"dataset_id":"ds","features":["score"],"target_col":"bad","split_col":"split","split_values":{"train":"train","test":"test","oot":"oot"},"params":{},"seed":7,"early_stopping_rounds":null}',
                None,
                "artifact-legacy",
                "trained",
                "2026-06-19T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO model_artifacts(
                id, experiment_id, algorithm, model_path, pmml_path,
                feature_list_json, feature_importance_json, params_json, woe_maps_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact-legacy",
                "experiment-legacy",
                "scorecard",
                "artifact.joblib",
                None,
                '["score"]',
                "[]",
                "{}",
                None,
                "2026-06-19T00:01:00Z",
            ),
        )

    init_db(db_path)
    artifact = ModelingRepository(db_path).get_model_artifact("artifact-legacy")

    assert artifact is not None
    assert artifact.scorecard_table == ()
