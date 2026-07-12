from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.validation_materials import resolve_selected_validation_materials
from tests.validation_material_builders import (
    create_api_validation_task,
    write_validation_material_bundle,
)


def _selected_material_task(root):
    names = {
        "notebook_path": "model.ipynb",
        "sample_path": "sample.parquet",
        "pmml_path": "model.pmml",
        "dictionary_path": "metadata.csv",
    }
    for name in names.values():
        (root / name).write_bytes(name.encode("utf-8"))
    return SimpleNamespace(source_dir=str(root), **names)


def test_selected_validation_materials_are_resolved_under_source_directory(tmp_path):
    task = _selected_material_task(tmp_path)

    resolved = resolve_selected_validation_materials(task)

    assert resolved.notebook == (tmp_path / "model.ipynb").resolve()
    assert resolved.sample == (tmp_path / "sample.parquet").resolve()
    assert resolved.pmml == (tmp_path / "model.pmml").resolve()
    assert resolved.dictionary == (tmp_path / "metadata.csv").resolve()


def test_selected_validation_materials_reject_escape_and_non_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    task = _selected_material_task(root)
    outside = tmp_path / "outside.pmml"
    outside.write_bytes(b"outside")
    task.pmml_path = str(outside)

    with pytest.raises(ValueError, match="escapes source directory"):
        resolve_selected_validation_materials(task)

    task.pmml_path = "folder.pmml"
    (root / "folder.pmml").mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        resolve_selected_validation_materials(task)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace")) as value:
        yield value


def _valid_confirmation(*, revision: int) -> dict:
    return {
        "revision": revision,
        "target_col": "y",
        "positive_label": 1,
        "negative_label": 0,
        "split_col": "split",
        "split_value_mapping": {
            "train": "train",
            "test": "test",
            "oot": "oot",
        },
        "time_col": "apply_month",
        "time_granularity": "month",
        "pmml_output_field": "probability_1",
        "model_params": {},
        "metadata_sheet": None,
        "feature_col": "feature",
        "category_col": "category",
        "importance_col": "importance",
        "transformations": [],
    }


def _create_scanned_task(client, *, sample=None, dictionary=None):
    root = client.app.state.settings.workspace / "validation-bundle"
    bundle = write_validation_material_bundle(
        root,
        notebook_source=(
            "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
            "RMC_TIME_COL='apply_month'\n"
            "RMC_PMML_OUTPUT_FIELD='probability_1'\nRMC_MODEL_PARAMS={}\n"
        ),
        sample=sample,
        dictionary=dictionary,
    )
    task_id, scan = create_api_validation_task(client, bundle)
    return task_id, scan, bundle


def test_contract_api_confirms_scanned_ambiguity(client):
    task_id, scan, _bundle = _create_scanned_task(client)
    scanned = scan["validation_input_contract"]
    assert scanned["status"] == "pending_confirmation"
    assert scanned["needs_confirmation"] is True
    assert scanned["read_only"] is True

    before = client.get(
        f"/api/tasks/{task_id}/validation-input-contract"
    )
    assert before.status_code == 200
    confirmed = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(revision=before.json()["revision"]),
    )

    assert confirmed.status_code == 200, confirmed.text
    payload = confirmed.json()
    assert payload["status"] == "ready"
    assert payload["revision"] == before.json()["revision"] + 1
    assert payload["contract"]["confirmed"]["algorithm"]


def test_ready_contract_rejects_reconfirmation_without_starting_work(client):
    task_id, scan, _bundle = _create_scanned_task(client)
    first = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(
            revision=scan["validation_input_contract"]["revision"]
        ),
    )
    assert first.status_code == 200, first.text

    second = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(revision=first.json()["revision"]),
    )

    assert second.status_code == 409
    assert "pending" in second.json()["detail"]
    assert TaskRepository(client.app.state.settings.db_path).task_has_active_job(
        task_id
    ) is False


def test_changing_selected_material_atomically_invalidates_ready_contract(client):
    task_id, scan, bundle = _create_scanned_task(client)
    ready = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(
            revision=scan["validation_input_contract"]["revision"]
        ),
    )
    assert ready.status_code == 200, ready.text
    replacement = bundle.pmml_path.with_name("replacement.pmml")
    replacement.write_bytes(bundle.pmml_path.read_bytes())

    changed = client.put(
        f"/api/tasks/{task_id}/materials",
        json={
            "notebook_path": bundle.notebook_path.name,
            "sample_path": bundle.sample_path.name,
            "pmml_path": replacement.name,
            "dictionary_path": bundle.dictionary_path.name,
        },
    )

    assert changed.status_code == 200, changed.text
    invalidated = client.get(
        f"/api/tasks/{task_id}/validation-input-contract"
    ).json()
    assert invalidated["status"] == "blocked"
    assert invalidated["revision"] == ready.json()["revision"] + 1
    assert invalidated["contract"]["confirmed"] == {}
    assert "rescan required" in invalidated["contract"]["conflicts"][0]
    for endpoint in ("notebook", "metrics", "report", "validate"):
        response = client.post(
            f"/api/tasks/{task_id}/{endpoint}",
            json={} if endpoint in {"notebook", "validate"} else None,
        )
        assert response.status_code == 422, (endpoint, response.text)
    assert TaskRepository(client.app.state.settings.db_path).task_has_active_job(
        task_id
    ) is False


def test_resubmitting_identical_material_selection_keeps_ready_contract(client):
    task_id, scan, bundle = _create_scanned_task(client)
    ready = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(
            revision=scan["validation_input_contract"]["revision"]
        ),
    )
    assert ready.status_code == 200, ready.text

    unchanged = client.put(
        f"/api/tasks/{task_id}/materials",
        json={
            "notebook_path": bundle.notebook_path.name,
            "sample_path": bundle.sample_path.name,
            "pmml_path": bundle.pmml_path.name,
            "dictionary_path": bundle.dictionary_path.name,
        },
    )

    assert unchanged.status_code == 200, unchanged.text
    stored = client.get(
        f"/api/tasks/{task_id}/validation-input-contract"
    ).json()
    assert stored["status"] == "ready"
    assert stored["revision"] == ready.json()["revision"]


def test_contract_api_preserves_numeric_split_scalars(client):
    sample = pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 1.0, 0.0, 1.0],
            "y": [0, 1, 0, 1],
            "split": [0, 1, 2, 2],
            "apply_month": ["202601", "202602", "202603", "202603"],
        }
    )
    task_id, scan, _bundle = _create_scanned_task(client, sample=sample)
    revision = scan["validation_input_contract"]["revision"]
    confirmation = _valid_confirmation(revision=revision)
    confirmation["split_value_mapping"] = {"train": 0, "test": 1, "oot": 2}

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=confirmation,
    )

    assert response.status_code == 200, response.text
    mapping = response.json()["contract"]["confirmed"]["split_value_mapping"]
    assert mapping == {"train": 0, "test": 1, "oot": 2}
    assert all(type(value) is int for value in mapping.values())


def test_contract_api_rejects_stale_revision_before_material_reads(client):
    task_id, scan, bundle = _create_scanned_task(client)
    payload = _valid_confirmation(
        revision=scan["validation_input_contract"]["revision"]
    )
    first = client.put(
        f"/api/tasks/{task_id}/validation-input-contract", json=payload
    )
    assert first.status_code == 200, first.text
    bundle.sample_path.unlink()

    stale = client.put(
        f"/api/tasks/{task_id}/validation-input-contract", json=payload
    )

    assert stale.status_code == 409
    assert "revision conflict" in stale.json()["detail"]


def test_contract_api_rejects_confirmation_while_task_has_active_job(client):
    task_id, scan, _bundle = _create_scanned_task(client)
    repo = TaskRepository(client.app.state.settings.db_path)
    repo.start_job(task_id, "validation")

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(
            revision=scan["validation_input_contract"]["revision"]
        ),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "task already has an active stage"


def test_contract_api_rehashes_selected_materials_before_commit(client):
    task_id, scan, bundle = _create_scanned_task(client)
    bundle.pmml_path.write_bytes(bundle.pmml_path.read_bytes() + b"\n")

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract",
        json=_valid_confirmation(
            revision=scan["validation_input_contract"]["revision"]
        ),
    )

    assert response.status_code == 422
    assert "changed; rescan" in response.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positive_label", 99, "positive label"),
        ("split_value_mapping", {"train": "train", "test": "test"}, "oot"),
        ("importance_col", "missing_importance", "feature metadata"),
        (
            "transformations",
            [
                {
                    "operation": "python_eval",
                    "output_field": "x",
                    "input_fields": ["x1"],
                    "params": {},
                }
            ],
            "unsupported transformation",
        ),
    ],
)
def test_confirmation_validation_errors_are_structured_422(
    client, field, value, message
):
    task_id, scan, _bundle = _create_scanned_task(client)
    payload = _valid_confirmation(
        revision=scan["validation_input_contract"]["revision"]
    )
    payload[field] = value

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract", json=payload
    )

    assert response.status_code == 422
    assert message.lower() in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(revision=True),
        lambda payload: payload.update(revision=0),
        lambda payload: payload.update(revision="1"),
        lambda payload: payload.pop("model_params"),
        lambda payload: payload.update(unexpected=True),
    ],
)
def test_confirmation_request_schema_is_strict(client, mutate):
    task_id, scan, _bundle = _create_scanned_task(client)
    payload = _valid_confirmation(
        revision=scan["validation_input_contract"]["revision"]
    )
    mutate(payload)

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract", json=payload
    )

    assert response.status_code == 422


def test_confirmation_api_rejects_deep_transformation_params_as_bounded_422(client):
    task_id, scan, _bundle = _create_scanned_task(client)
    payload = _valid_confirmation(
        revision=scan["validation_input_contract"]["revision"]
    )
    nested: object = "leaf"
    for _ in range(70):
        nested = {"nested": nested}
    payload["transformations"] = [
        {
            "operation": "copy",
            "output_field": "derived",
            "input_fields": ["x1"],
            "params": {"nested": nested},
        }
    ]

    response = client.put(
        f"/api/tasks/{task_id}/validation-input-contract", json=payload
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "JSON value exceeds maximum depth"


def test_blocked_scan_is_persisted_and_returns_validation_material_422(client):
    dictionary = pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["内部", "征信"]}
    )
    root = client.app.state.settings.workspace / "blocked-bundle"
    bundle = write_validation_material_bundle(
        root,
        notebook_source="RMC_TARGET_COL='y'",
        dictionary=dictionary,
    )
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "validation",
            "model_name": "fixture",
            "validator": "pytest",
            "source_dir": str(root),
        },
    )
    task_id = created.json()["id"]
    selected = client.put(
        f"/api/tasks/{task_id}/materials",
        json={
            "notebook_path": bundle.notebook_path.name,
            "sample_path": bundle.sample_path.name,
            "pmml_path": bundle.pmml_path.name,
            "dictionary_path": bundle.dictionary_path.name,
        },
    )
    assert selected.status_code == 200, selected.text

    response = client.post(f"/api/tasks/{task_id}/scan")

    assert response.status_code == 422
    assert response.json()["detail"].startswith("validation materials invalid:")
    stored = client.get(
        f"/api/tasks/{task_id}/validation-input-contract"
    )
    assert stored.status_code == 200
    assert stored.json()["status"] == "blocked"


def test_confirmation_for_missing_task_is_404(client):
    response = client.put(
        "/api/tasks/missing/validation-input-contract",
        json=_valid_confirmation(revision=1),
    )

    assert response.status_code == 404
