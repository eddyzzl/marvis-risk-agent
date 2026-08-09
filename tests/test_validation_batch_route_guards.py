from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


def _batch_item(root: Path, name: str) -> dict:
    source_dir = root / name
    source_dir.mkdir(parents=True)
    paths = {
        "notebook_path": source_dir / f"{name}.ipynb",
        "sample_path": source_dir / f"{name}.csv",
        "pmml_path": source_dir / f"{name}.pmml",
        "dictionary_path": source_dir / f"{name}.xlsx",
    }
    for path in paths.values():
        path.touch()
    return {
        "model_name": name,
        "model_version": "v1",
        "source_dir": str(source_dir),
        **{key: str(value) for key, value in paths.items()},
    }


def _create_batch(client: TestClient, root: Path) -> tuple[str, str]:
    response = client.post(
        "/api/validation-batches",
        json={
            "batch_name": "受保护的模型验证批次",
            "validator": "qa",
            "items": [_batch_item(root, "模型A")],
        },
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["batch"]["parent_task_id"], payload["items"][0]["child_task_id"]


def _output_sentinel(client: TestClient, task_id: str) -> Path:
    path = Path(client.app.state.settings.tasks_dir) / task_id / "outputs" / "keep.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("batch-owned-output", encoding="utf-8")
    return path


def test_generic_task_create_rejects_validation_batch_parent(tmp_path: Path):
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/tasks",
        json={
            "task_type": "validation_batch",
            "model_name": "禁止孤立创建的批次",
            "validator": "qa",
            "source_dir": str(tmp_path),
            "run_mode": "agent",
        },
    )

    assert response.status_code == 422
    assert "/api/validation-batches" in response.json()["detail"]
    assert client.get("/api/tasks").json() == []


def test_generic_scan_rejects_validation_batch_parent_before_starting_job(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, _child_task_id = _create_batch(client, tmp_path / "materials")

    response = client.post(f"/api/tasks/{parent_task_id}/scan")

    assert response.status_code == 422
    assert "/api/validation-batches" in response.json()["detail"]
    assert client.get(f"/api/tasks/{parent_task_id}/jobs/latest").json() == {
        "job": None
    }


def test_generic_scan_rejects_validation_batch_child_before_job_or_output_cleanup(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    sentinel = _output_sentinel(client, child_task_id)
    batch_before = client.get(f"/api/validation-batches/{parent_task_id}").json()

    response = client.post(f"/api/tasks/{child_task_id}/scan")

    assert response.status_code == 409
    assert "批次内" in response.json()["detail"]
    assert client.get(f"/api/tasks/{child_task_id}/jobs/latest").json() == {
        "job": None
    }
    assert sentinel.read_text(encoding="utf-8") == "batch-owned-output"
    assert client.get(f"/api/validation-batches/{parent_task_id}").json() == batch_before


def test_generic_material_candidates_reject_validation_batch_parent(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    parent_task_id, _child_task_id = _create_batch(client, tmp_path / "materials")

    response = client.get(f"/api/tasks/{parent_task_id}/materials")

    assert response.status_code == 422
    assert "/api/validation-batches" in response.json()["detail"]


def test_generic_material_candidates_reject_validation_batch_child(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    batch_before = client.get(f"/api/validation-batches/{parent_task_id}").json()

    response = client.get(f"/api/tasks/{child_task_id}/materials")

    assert response.status_code == 409
    assert "批次内" in response.json()["detail"]
    assert client.get(f"/api/validation-batches/{parent_task_id}").json() == batch_before


def test_generic_material_update_rejects_batch_parent_before_starting_job(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, _child_task_id = _create_batch(client, tmp_path / "materials")

    response = client.put(
        f"/api/tasks/{parent_task_id}/materials",
        json={
            "notebook_path": "model.ipynb",
            "sample_path": "sample.csv",
            "pmml_path": "model.pmml",
            "dictionary_path": "dictionary.xlsx",
        },
    )

    assert response.status_code == 422
    assert "/api/validation-batches" in response.json()["detail"]
    assert client.get(f"/api/tasks/{parent_task_id}/jobs/latest").json() == {
        "job": None
    }


def test_generic_material_update_rejects_batch_child_before_job_or_output_cleanup(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    sentinel = _output_sentinel(client, child_task_id)
    batch_before = client.get(f"/api/validation-batches/{parent_task_id}").json()

    response = client.put(
        f"/api/tasks/{child_task_id}/materials",
        json={
            "notebook_path": "模型A.ipynb",
            "sample_path": "模型A.csv",
            "pmml_path": "模型A.pmml",
            "dictionary_path": "模型A.xlsx",
        },
    )

    assert response.status_code == 409
    assert "批次内" in response.json()["detail"]
    assert client.get(f"/api/tasks/{child_task_id}/jobs/latest").json() == {
        "job": None
    }
    assert sentinel.read_text(encoding="utf-8") == "batch-owned-output"
    assert client.get(f"/api/validation-batches/{parent_task_id}").json() == batch_before


@pytest.mark.parametrize(
    ("stage_path", "payload"),
    [
        ("notebook", {}),
        ("metrics", None),
        ("report", None),
        ("validate", {}),
    ],
)
def test_generic_validation_stage_rejects_batch_parent_before_starting_job(
    tmp_path: Path,
    stage_path: str,
    payload: dict | None,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, _child_task_id = _create_batch(client, tmp_path / "materials")

    response = client.request(
        "POST",
        f"/api/tasks/{parent_task_id}/{stage_path}",
        json=payload,
    )

    assert response.status_code == 422
    assert "/api/validation-batches" in response.json()["detail"]
    assert client.get(f"/api/tasks/{parent_task_id}/jobs/latest").json() == {
        "job": None
    }


@pytest.mark.parametrize(
    ("stage_path", "payload"),
    [
        ("notebook", {}),
        ("metrics", None),
        ("report", None),
        ("validate", {}),
    ],
)
def test_generic_validation_stage_rejects_batch_child_before_job_or_cleanup(
    tmp_path: Path,
    stage_path: str,
    payload: dict | None,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    sentinel = _output_sentinel(client, child_task_id)
    batch_before = client.get(f"/api/validation-batches/{parent_task_id}").json()

    response = client.request(
        "POST",
        f"/api/tasks/{child_task_id}/{stage_path}",
        json=payload,
    )

    assert response.status_code == 409
    assert "批次内" in response.json()["detail"]
    assert client.get(f"/api/tasks/{child_task_id}/jobs/latest").json() == {
        "job": None
    }
    assert sentinel.read_text(encoding="utf-8") == "batch-owned-output"
    assert client.get(f"/api/validation-batches/{parent_task_id}").json() == batch_before


@pytest.mark.parametrize("target_kind", ["parent", "child"])
def test_generic_purge_preview_rejects_batch_task(
    tmp_path: Path,
    target_kind: str,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    target_task_id = parent_task_id if target_kind == "parent" else child_task_id

    response = client.get(f"/api/tasks/{target_task_id}/purge-preview")

    assert response.status_code == 409
    assert "批次" in response.json()["detail"]
    assert client.get(f"/api/tasks/{parent_task_id}").status_code == 200
    assert client.get(f"/api/tasks/{child_task_id}").status_code == 200
    batch = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert batch["batch"]["item_count"] == 1
    assert [item["child_task_id"] for item in batch["items"]] == [child_task_id]


@pytest.mark.parametrize("target_kind", ["parent", "child"])
def test_generic_delete_rejects_batch_task_without_breaking_membership(
    tmp_path: Path,
    target_kind: str,
):
    client = TestClient(create_app(tmp_path))
    parent_task_id, child_task_id = _create_batch(client, tmp_path / "materials")
    target_task_id = parent_task_id if target_kind == "parent" else child_task_id

    response = client.delete(f"/api/tasks/{target_task_id}")

    assert response.status_code == 409
    assert "批次" in response.json()["detail"]
    assert client.get(f"/api/tasks/{parent_task_id}").status_code == 200
    assert client.get(f"/api/tasks/{child_task_id}").status_code == 200
    batch = client.get(f"/api/validation-batches/{parent_task_id}").json()
    assert batch["batch"]["item_count"] == 1
    assert [item["child_task_id"] for item in batch["items"]] == [child_task_id]
