from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from marvis.data import analysis_service as analysis_service_module
from marvis.app import create_app
from marvis.db import TaskRepository, connect
from marvis.domain import TaskCreate
from marvis.repositories.data_analysis import (
    DataAnalysisNotFoundError,
    DataAnalysisStaleIdentityError,
)
from marvis.routers.data_analysis import router as data_analysis_router


def _create_task(app):
    return TaskRepository(app.state.settings.db_path).create_task(
        TaskCreate(
            model_name="data analysis",
            model_version="v1",
            validator="owner",
            source_dir=str(app.state.settings.workspace),
            task_type="strategy",
        )
    )


def _prepare_workspace(client: TestClient, task_id: str) -> dict:
    upload = client.post(
        f"/api/tasks/{task_id}/datasets/upload",
        data={"role": "sample"},
        files={
            "file": (
                "sample.csv",
                (
                    "phone,customer_name,amount,bad\n"
                    "13800138000,Alice,100,0\n"
                    "13900139000,Bob,200,1\n"
                    "13800138000,Alice,300,0\n"
                ).encode(),
                "text/csv",
            )
        },
    )
    assert upload.status_code == 201
    dataset = upload.json()["datasets"][0]
    activated = client.put(
        f"/api/tasks/{task_id}/data-workspace",
        headers={"If-Match": "0"},
        json={
            "active_dataset_id": dataset["id"],
            "active_dataset_content_hash": dataset["content_hash"],
            "page": "overview",
            "selected_field": None,
            "semantic_mapping": {
                "target_col": None,
                "field_roles": {},
                "business_names": {},
            },
        },
    )
    assert activated.status_code == 200
    saved = client.put(
        f"/api/tasks/{task_id}/data-workspace",
        headers={"If-Match": str(activated.json()["revision"])},
        json={
            "active_dataset_id": dataset["id"],
            "active_dataset_content_hash": dataset["content_hash"],
            "page": "statistics",
            "selected_field": "amount",
            "semantic_mapping": {
                "target_col": "bad",
                "field_roles": {
                    "phone": "phone",
                    "customer_name": "name",
                    "amount": "amount",
                    "bad": "target",
                },
                "business_names": {"amount": "申请金额"},
            },
        },
    )
    assert saved.status_code == 200
    return saved.json()


def _body(*, retry: bool = False) -> dict:
    return {
        "sections": [
            "overview",
            "target",
            "missing",
            "distribution",
            "correlation",
        ],
        "columns": ["phone", "customer_name", "amount", "bad"],
        "config": {
            "frequency_top_k": 10,
            "low_cardinality_threshold": 10,
            "histogram_bins": 4,
        },
        "retry": retry,
    }


def test_data_analysis_routes_are_dedicated_and_app_wired(tmp_path):
    routes = {
        (route.path, tuple(sorted(route.methods or []))): route.endpoint.__module__
        for route in data_analysis_router.routes
    }
    assert routes[("/api/tasks/{task_id}/data-analysis", ("POST",))] == (
        "marvis.routers.data_analysis"
    )
    assert routes[
        ("/api/tasks/{task_id}/data-analysis/{run_id}", ("GET",))
    ] == "marvis.routers.data_analysis"

    app_paths = create_app(tmp_path).openapi()["paths"]
    assert "post" in app_paths["/api/tasks/{task_id}/data-analysis"]


def test_post_requires_exact_if_match_and_strict_request(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = _create_task(app)
    workspace = _prepare_workspace(client, task.id)

    missing = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        json=_body(),
    )
    malformed = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": "not-an-int"},
        json=_body(),
    )
    stale = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"] - 1)},
        json=_body(),
    )
    extra = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"])},
        json={**_body(), "raw_values": True},
    )
    coerced = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"])},
        json={**_body(), "config": {"histogram_bins": "4"}},
    )

    assert missing.status_code == 428
    assert malformed.status_code == 400
    assert stale.status_code == 412
    assert extra.status_code == 422
    assert coerced.status_code == 422


def test_post_dispatches_get_returns_result_and_succeeded_cache_is_job_free(
    tmp_path,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = _create_task(app)
    workspace = _prepare_workspace(client, task.id)
    headers = {"If-Match": str(workspace["revision"])}

    accepted = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers=headers,
        json=_body(),
    )

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "queued"
    assert accepted.json()["run_id"]
    assert accepted.json()["job_id"]
    run_id = accepted.json()["run_id"]

    completed = client.get(f"/api/tasks/{task.id}/data-analysis/{run_id}")
    assert completed.status_code == 200
    payload = completed.json()
    assert payload["status"] == "succeeded"
    assert payload["result_artifact_id"]
    assert payload["download_url"].endswith("/download")
    assert payload["result"]["schema_version"] == "data-analysis.v1"
    assert "13800138000" not in completed.text
    assert "Alice" not in completed.text

    cached = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers=headers,
        json=_body(),
    )
    assert cached.status_code == 200
    assert cached.json()["status"] == "succeeded"
    assert cached.json()["run_id"] == run_id
    assert cached.json()["cached"] is True
    assert cached.json()["result"] == payload["result"]
    assert TaskRepository(app.state.settings.db_path).get_latest_job(
        task.id,
        kind="data_analysis",
    )["id"] == accepted.json()["job_id"]

    download = client.get(payload["download_url"])
    assert download.status_code == 200
    assert download.json() == payload["result"]

    with connect(app.state.settings.db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET path = ? WHERE id = ?",
            (
                f"tasks/{task.id}/data_analysis/drifted.json",
                payload["result_artifact_id"],
            ),
        )
    drifted_cache = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers=headers,
        json=_body(),
    )
    assert drifted_cache.status_code == 409


def test_columns_are_optional_and_unknown_analyzer_error_is_generic(
    tmp_path,
    monkeypatch,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = _create_task(app)
    workspace = _prepare_workspace(client, task.id)
    sensitive_error = "analyzer leaked 13800138000 Alice"

    def fail_with_sensitive_text(*_args, **_kwargs):
        raise RuntimeError(sensitive_error)

    monkeypatch.setattr(
        analysis_service_module,
        "analyze_parquet",
        fail_with_sensitive_text,
    )
    body = _body()
    del body["columns"]
    accepted = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"])},
        json=body,
    )

    assert accepted.status_code == 202
    failed = client.get(
        f"/api/tasks/{task.id}/data-analysis/{accepted.json()['run_id']}"
    )
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    assert failed.json()["error"] == {
        "kind": "data_analysis_failed",
        "message": "data analysis failed",
    }
    assert sensitive_error not in failed.text


def test_get_conceals_cross_task_run(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = _create_task(app)
    other = _create_task(app)
    workspace = _prepare_workspace(client, task.id)
    accepted = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"])},
        json=_body(),
    )

    concealed = client.get(
        f"/api/tasks/{other.id}/data-analysis/{accepted.json()['run_id']}"
    )
    missing = client.get(f"/api/tasks/{task.id}/data-analysis/missing")

    assert concealed.status_code == 404
    assert missing.status_code == 404


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (DataAnalysisStaleIdentityError("workspace changed"), 412),
        (DataAnalysisNotFoundError("task deleted"), 404),
    ],
)
def test_dispatch_races_preserve_precondition_and_not_found_taxonomy(
    tmp_path,
    monkeypatch,
    error,
    expected_status,
):
    app = create_app(tmp_path)
    client = TestClient(app)
    task = _create_task(app)
    workspace = _prepare_workspace(client, task.id)

    def fail_raced_dispatch(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(
        analysis_service_module.DataAnalysisService,
        "request_analysis",
        fail_raced_dispatch,
    )
    response = client.post(
        f"/api/tasks/{task.id}/data-analysis",
        headers={"If-Match": str(workspace["revision"])},
        json=_body(),
    )

    assert response.status_code == expected_status
