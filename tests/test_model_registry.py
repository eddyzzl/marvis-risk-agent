from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import ModelingRepository
from marvis.packs.modeling import Experiment, ModelArtifact, ModelMetrics, TrainConfig


def _client(tmp_path):
    app = create_app(tmp_path / "workspace")
    return TestClient(app), app.state.settings


def _create_task(client, model_name: str = "A-card"):
    response = client.post(
        "/api/tasks",
        json={
            "model_name": model_name,
            "model_version": "v1",
            "validator": "qa",
            "source_dir": str(client.app.state.settings.workspace),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


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


def _seed_experiment(settings, task_id: str, experiment_id: str, *, status: str = "trained"):
    modeling_repo = ModelingRepository(settings.db_path)
    experiment = Experiment(
        id=experiment_id,
        task_id=task_id,
        recipe_id="lgb",
        config=_config(),
        metrics=_metrics() if status == "trained" else None,
        artifact_id=f"{experiment_id}-artifact" if status == "trained" else None,
        status=status,
        created_at="2026-06-19T00:00:00Z",
    )
    modeling_repo.create_experiment(experiment)
    if status == "trained":
        artifact = ModelArtifact(
            id=f"{experiment_id}-artifact",
            experiment_id=experiment_id,
            algorithm="lgb",
            model_path=f"models/{experiment_id}/model.txt",
            pmml_path=None,
            feature_list=("score", "income"),
            params={"num_leaves": 16},
            woe_maps=None,
            created_at="2026-06-19T00:01:00Z",
            feature_importance=(("score", 0.7), ("income", 0.3)),
        )
        modeling_repo.create_model_artifact(artifact)
    return experiment


def test_list_experiments_returns_cross_task_registry_with_task_identity(tmp_path):
    client, settings = _client(tmp_path)
    task_a = _create_task(client, "A-card")
    task_b = _create_task(client, "B-card")
    _seed_experiment(settings, task_a["id"], "experiment-a", status="trained")
    _seed_experiment(settings, task_b["id"], "experiment-b", status="pending")

    response = client.get("/api/experiments")

    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body] == ["experiment-b", "experiment-a"]
    trained_row = next(row for row in body if row["id"] == "experiment-a")
    assert trained_row["task_id"] == task_a["id"]
    assert trained_row["task_model_name"] == "A-card"
    assert trained_row["train_ks"] == 0.41
    assert trained_row["oot_ks"] == 0.35


def test_list_experiments_filters_by_status_and_paginates(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    _seed_experiment(settings, task["id"], "experiment-a", status="trained")
    _seed_experiment(settings, task["id"], "experiment-b", status="pending")

    trained_only = client.get("/api/experiments", params={"status": "trained"})
    assert trained_only.status_code == 200
    assert [row["id"] for row in trained_only.json()] == ["experiment-a"]

    paginated = client.get("/api/experiments", params={"limit": 1, "offset": 0})
    assert paginated.status_code == 200
    assert len(paginated.json()) == 1
    assert paginated.headers["X-Result-Has-More"] == "true"


def test_get_experiment_detail_includes_metrics_and_artifacts(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    _seed_experiment(settings, task["id"], "experiment-a", status="trained")

    response = client.get("/api/experiments/experiment-a")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "experiment-a"
    assert body["task_id"] == task["id"]
    assert body["metrics"]["train_ks"] == 0.41
    assert len(body["artifacts"]) == 1
    assert body["artifacts"][0]["id"] == "experiment-a-artifact"
    assert body["artifacts"][0]["algorithm"] == "lgb"


def test_get_experiment_detail_returns_404_for_unknown_experiment(tmp_path):
    client, _settings = _client(tmp_path)

    response = client.get("/api/experiments/does-not-exist")

    assert response.status_code == 404


def test_list_task_experiments_scopes_to_single_task(tmp_path):
    client, settings = _client(tmp_path)
    task_a = _create_task(client, "A-card")
    task_b = _create_task(client, "B-card")
    _seed_experiment(settings, task_a["id"], "experiment-a", status="trained")
    _seed_experiment(settings, task_b["id"], "experiment-b", status="trained")

    response = client.get(f"/api/tasks/{task_a['id']}/experiments")

    assert response.status_code == 200
    body = response.json()
    assert [row["id"] for row in body["experiments"]] == ["experiment-a"]


def test_list_task_experiments_supports_limit_offset_and_reports_total(tmp_path):
    # LT-13: limit/offset are opt-in -- the scoping test above (no params)
    # keeps getting the full per-task experiment history with no pagination
    # keys in the response.
    client, settings = _client(tmp_path)
    task = _create_task(client)
    for index in range(3):
        _seed_experiment(settings, task["id"], f"experiment-{index}", status="trained")

    first_page = client.get(
        f"/api/tasks/{task['id']}/experiments", params={"limit": 2, "offset": 0}
    )
    assert first_page.status_code == 200, first_page.text
    body = first_page.json()
    assert [row["id"] for row in body["experiments"]] == ["experiment-0", "experiment-1"]
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert body["has_more"] is True

    second_page = client.get(
        f"/api/tasks/{task['id']}/experiments", params={"limit": 2, "offset": 2}
    )
    body2 = second_page.json()
    assert [row["id"] for row in body2["experiments"]] == ["experiment-2"]
    assert body2["total"] == 3
    assert body2["has_more"] is False


def test_list_task_experiments_limit_is_bounded_at_maximum(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    _seed_experiment(settings, task["id"], "experiment-only", status="trained")

    response = client.get(
        f"/api/tasks/{task['id']}/experiments", params={"limit": 999999}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["limit"] == 500
    assert [row["id"] for row in body["experiments"]] == ["experiment-only"]


def test_list_task_experiments_returns_404_for_unknown_task(tmp_path):
    client, _settings = _client(tmp_path)

    response = client.get("/api/tasks/does-not-exist/experiments")

    assert response.status_code == 404
