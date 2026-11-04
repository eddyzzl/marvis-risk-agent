"""V3 5.7: historical material mapping extends from strategy tasks to data
processing tasks, with the same locating, proposal and signed confirmation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.routers.plugins import ensure_plugin_admin_token
from marvis.routers.strategy_history_mapping import router


def _runtime(tmp_path: Path, task_type: str):
    source = tmp_path / "materials"
    source.mkdir()
    (source / "main.csv").write_text("apply_id,bad_flag\nA1,0\n", encoding="utf-8")
    (source / "历史报告.csv").write_text("period,bad_rate\n2025-01,2.5%\n", encoding="utf-8")
    full = create_app(tmp_path)
    with TestClient(full) as client:
        created = client.post("/api/tasks", json={
            "model_name": "history", "validator": "qa", "source_dir": str(source),
            "task_type": task_type, "run_mode": "agent",
        })
        assert created.status_code == 200, created.text
        task_id = created.json()["id"]
    settings = full.state.settings
    ensure_plugin_admin_token(settings.plugin_admin_token_path)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)

    @app.middleware("http")
    async def actor(request, call_next):
        request.state.private_client_authenticated = True
        request.state.local_principal = SimpleNamespace(id="maker")
        return await call_next(request)

    return TestClient(app), f"/api/tasks/{task_id}/strategy/history-mapping"


def test_data_processing_task_registers_maps_and_confirms_a_history_material(tmp_path):
    client, base = _runtime(tmp_path, "data_join")
    candidates = client.get(base + "/sources/candidates")
    assert candidates.status_code == 200, candidates.text
    assert candidates.json()["filenames"] == ["main.csv", "历史报告.csv"]
    assert client.get(base + "/sources").json()["sources"] == []

    registered = client.post(base + "/sources", json={"filename": "历史报告.csv"})
    assert registered.status_code == 200, registered.text
    source_id = registered.json()["artifact_id"]
    again = client.post(base + "/sources", json={"filename": "历史报告.csv"})
    assert again.json() == {**registered.json(), "status": "already_registered"}
    sources = client.get(base + "/sources").json()["sources"]
    assert [(item["artifact_id"], item["status"]) for item in sources] == [(source_id, "ready")]

    cells = client.get(base + f"/sources/{source_id}/cells").json()
    cell = next(row for row in cells["cells"] if row["locator"] == {"format": "csv", "row": 2, "column": 2})
    binding = {"binding_key": "b1", "source_id": source_id, "source_hash": cells["source_hash"],
               "locator": cell["locator"], "cell_hash": cell["cell_hash"],
               "target": {"kind": "metric", "name": "bad_rate", "unit": "percent", "period_format": "unknown"},
               "period_key": None, "version_key": None}
    head = client.get(base).json()["head"]
    proposal = client.post(base + "/proposals", json={"expected_head": head, "bindings": [binding]})
    assert proposal.status_code == 200, proposal.text
    assert proposal.json()["bindings"][0]["value"] == 2.5
    saved = client.post(base + "/confirm", json={"proposal_token": proposal.json()["proposal_token"]})
    assert saved.status_code == 200, saved.text
    assert saved.json()["mapping"]["validation_status"] == "historical_reported_unvalidated"
    assert client.get(base).json()["head"]["revision"] == 1


def test_material_registration_is_bounded_to_data_tasks_and_locatable_files(tmp_path):
    client, base = _runtime(tmp_path, "data_join")
    assert client.post(base + "/sources", json={"filename": "../outside.csv"}).status_code == 422
    assert client.post(base + "/sources", json={"filename": "notes.md"}).json()["detail"]["code"] == "history_source_suffix_unsupported"
    assert client.post(base + "/sources", json={"filename": "missing.csv"}).json()["detail"]["code"] == "history_source_unavailable"


def test_other_task_types_still_cannot_use_history_mapping(tmp_path):
    client, base = _runtime(tmp_path, "feature_analysis")
    response = client.get(base)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "history_strategy_task_required"
