"""In-task supplementary materials (eval round 1, P0-5).

A failed material check told the user to fix the dictionary and upload it
again, but the task had nowhere to upload. Corrected files now go into a new
补充材料-<time> folder of the task's own upload directory; a task bound to the
user's own folder is never written into.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from tests.validation_material_builders import write_validation_material_bundle

NOTEBOOK = (
    "RMC_TARGET_COL='y'\nRMC_SPLIT_COL='split'\n"
    "RMC_TIME_COL='apply_month'\n"
    "RMC_PMML_OUTPUT_FIELD='probability_1'\nRMC_MODEL_PARAMS={}\n"
)
NO_IMPORTANCE = pd.DataFrame({"feature": ["x1", "x2"], "category": ["内部", "征信"]})


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "workspace")) as value:
        yield value


def _task(client, root):
    bundle = write_validation_material_bundle(root, notebook_source=NOTEBOOK, dictionary=NO_IMPORTANCE)
    created = client.post(
        "/api/tasks",
        json={
            "task_type": "validation",
            "model_name": "fixture",
            "validator": "pytest",
            "source_dir": str(root),
            "run_mode": "agent",
        },
    )
    assert created.status_code == 200, created.text
    task_id = str(created.json()["id"])
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
    return task_id, bundle


def _fixed_dictionary() -> bytes:
    buffer = io.StringIO()
    pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["内部", "征信"], "importance": [0.6, 0.4]}
    ).to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def _uploaded_task(client):
    root = client.app.state.settings.workspace / "material_uploads" / "abc123"
    return _task(client, root)


def test_uploaded_task_takes_a_corrected_dictionary_next_to_the_old_one(client):
    task_id, bundle = _uploaded_task(client)
    before = client.get(f"/api/tasks/{task_id}/materials")
    assert before.json()["supplement_upload_supported"] is True
    old_dictionary = bundle.dictionary_path.read_bytes()

    response = client.post(
        f"/api/tasks/{task_id}/materials/files",
        files=[("files", ("metadata_v2.csv", _fixed_dictionary(), "text/csv"))],
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    [saved] = payload["files"]
    folder, name = saved["relative_path"].split("/")
    assert folder.startswith("补充材料-") and name == "metadata_v2.csv"
    assert (bundle.root / folder / name).read_bytes() == _fixed_dictionary()
    # The earlier file stays as it was; nothing is selected behind the user's back.
    assert bundle.dictionary_path.read_bytes() == old_dictionary
    assert payload["materials"]["selection"]["dictionary_path"] == bundle.dictionary_path.name
    dictionaries = {
        item["relative_path"]: item.get("metadata_compatibility", {}).get("status")
        for item in payload["materials"]["candidates"]["data_dictionary"]
    }
    assert dictionaries[saved["relative_path"]] == "compatible"
    assert dictionaries[bundle.dictionary_path.name] == "incompatible"

    # Choosing it still goes through the normal material selection.
    chosen = client.put(
        f"/api/tasks/{task_id}/materials",
        json={
            "notebook_path": bundle.notebook_path.name,
            "sample_path": bundle.sample_path.name,
            "pmml_path": bundle.pmml_path.name,
            "dictionary_path": saved["relative_path"],
        },
    )
    assert chosen.status_code == 200, chosen.text
    assert chosen.json()["materials"]["selection"]["dictionary_path"] == saved["relative_path"]


def test_task_bound_to_a_local_folder_is_never_written_into(client):
    # A folder the user pointed the task at (inside an allowed material root,
    # but not one the platform created for an upload).
    root = client.app.state.settings.workspace / "user-folder"
    task_id, _bundle = _task(client, root)
    before = sorted(path.name for path in root.iterdir())

    listing = client.get(f"/api/tasks/{task_id}/materials")
    response = client.post(
        f"/api/tasks/{task_id}/materials/files",
        files=[("files", ("metadata_v2.csv", _fixed_dictionary(), "text/csv"))],
    )

    assert listing.json()["supplement_upload_supported"] is False
    assert response.status_code == 409
    assert "本地目录" in response.json()["detail"]
    assert sorted(path.name for path in root.iterdir()) == before


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ([("files", (".hidden.csv", b"x\n1\n", "text/csv"))], "文件名无效"),
        (
            [
                ("files", ("a.csv", b"x\n1\n", "text/csv")),
                ("files", ("a.csv", b"x\n2\n", "text/csv")),
            ],
            "重名",
        ),
    ],
)
def test_bad_supplement_files_are_refused_without_leaving_a_folder(client, files, message):
    task_id, bundle = _uploaded_task(client)

    response = client.post(f"/api/tasks/{task_id}/materials/files", files=files)

    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert not [path for path in bundle.root.iterdir() if path.name.startswith("补充材料-")]


def test_supplement_upload_waits_for_an_active_stage(client):
    task_id, bundle = _uploaded_task(client)
    TaskRepository(client.app.state.settings.db_path).start_job(task_id, "validation")

    response = client.post(
        f"/api/tasks/{task_id}/materials/files",
        files=[("files", ("metadata_v2.csv", _fixed_dictionary(), "text/csv"))],
    )

    assert response.status_code == 409
    assert not [path for path in bundle.root.iterdir() if path.name.startswith("补充材料-")]
