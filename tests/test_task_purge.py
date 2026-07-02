import pytest

import pandas as pd
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository


def _client(tmp_path):
    app = create_app(tmp_path / "workspace")
    settings = app.state.settings
    return TestClient(app), settings


def _create_task(client):
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "A card",
            "model_version": "v1",
            "validator": "qa",
            "source_dir": str(client.app.state.settings.workspace),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_dataset(client, task_id: str, frame: pd.DataFrame, *, name: str = "sample.csv"):
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    response = client.post(
        f"/api/tasks/{task_id}/datasets/upload",
        data={"role": "sample"},
        files={"file": (name, csv_bytes, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()["datasets"][0]


def test_purge_preview_endpoint_lists_expected_counts_without_deleting(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(client, task["id"], pd.DataFrame({"acct_id": ["A1", "B2"]}))
    dataset_path = settings.datasets_dir / dataset["source_path"]
    assert dataset_path.exists()

    preview = client.get(f"/api/tasks/{task['id']}/purge-preview")

    assert preview.status_code == 200
    body = preview.json()
    assert body["task_id"] == task["id"]
    assert body["purge_summary"]["datasets"] == 1
    assert "dataset_source_paths" not in body["purge_summary"]
    # dry-run leaves everything in place
    assert dataset_path.exists()
    assert TaskRepository(settings.db_path).get_task(task["id"]).id == task["id"]


def test_purge_preview_returns_404_for_unknown_task(tmp_path):
    client, _settings = _client(tmp_path)

    response = client.get("/api/tasks/does-not-exist/purge-preview")

    assert response.status_code == 404


def test_delete_task_removes_dataset_files_and_writes_delete_audit(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(client)
    dataset = _upload_dataset(client, task["id"], pd.DataFrame({"acct_id": ["A1", "B2"]}))
    dataset_path = settings.datasets_dir / dataset["source_path"]
    assert dataset_path.exists()

    response = client.delete(f"/api/tasks/{task['id']}")

    assert response.status_code == 204
    assert not dataset_path.exists()
    assert not (settings.tasks_dir / task["id"]).exists()

    from marvis.repositories.audit import _list_audit_rows

    audit_rows = _list_audit_rows(settings.db_path, kind="task.delete")
    assert len(audit_rows) == 1
    assert audit_rows[0]["target_ref"] == task["id"]
    assert audit_rows[0]["detail"]["purge_summary"]["datasets"] == 1

    with pytest.raises(KeyError):
        TaskRepository(settings.db_path).get_task(task["id"])


def test_delete_task_keeps_dataset_file_still_referenced_by_another_task(tmp_path):
    # GAP-2 x GAP-7: uses the real content-fingerprint dedup upload path (see
    # GET /api/tasks/{id}/purge-preview and register_from_upload) so both tasks'
    # dataset rows point at the same physical parquet file.
    client, settings = _client(tmp_path)
    owner_task = _create_task(client)
    frame = pd.DataFrame({"acct_id": ["A1", "B2"]})
    dataset = _upload_dataset(client, owner_task["id"], frame)
    dataset_path = settings.datasets_dir / dataset["source_path"]
    assert dataset_path.exists()

    other_task = _create_task(client)
    other_dataset = _upload_dataset(client, other_task["id"], frame)
    assert other_dataset["source_path"] == dataset["source_path"]

    response = client.delete(f"/api/tasks/{owner_task['id']}")

    assert response.status_code == 204
    # the physical file is still referenced by other_task's dataset row
    assert dataset_path.exists()

    second_response = client.delete(f"/api/tasks/{other_task['id']}")

    assert second_response.status_code == 204
    # once the last referencing task is gone, the file is finally removed
    assert not dataset_path.exists()
