from dataclasses import replace
from types import SimpleNamespace
import json

from fastapi.testclient import TestClient
import pytest

from marvis.data.field_directions import (
    ApprovedFieldDirections, approved_field_directions, confirm_field_directions,
    confirmed_field_directions,
)
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.transform_semantics import migrate_transform_semantics
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.llm_outbound import OutboundContextError
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.datasets import DatasetRepository
from marvis.routers.field_directions import router
from tests.test_data_workspace_api import _app, _create_task, _upload_csv, _put_workspace, _workspace_draft
from marvis.db import init_db
from marvis.settings import build_settings


def _setup(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    app = _app(settings)
    app.include_router(router)
    @app.middleware("http")
    async def actor(request, call_next):
        identity = request.headers.get("x-test-actor")
        request.state.private_client_authenticated = bool(identity)
        request.state.local_principal = SimpleNamespace(id=identity) if identity else None
        return await call_next(request)
    client = TestClient(app, headers={"x-test-actor": "maker-1"})
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    assert _put_workspace(client, task.id, _workspace_draft(dataset), revision=0).status_code == 200
    draft = _workspace_draft(dataset, business_names={"loan_amount": "risk decreases automatically"})
    assert _put_workspace(client, task.id, draft, revision=1).status_code == 200
    return client, settings, task, dataset


def _base(task):
    return f"/api/tasks/{task.id}/data-workspace/field-directions"


def _confirm_body(client, task, directions):
    projection = client.get(_base(task)).json()
    return {"expected_revision": projection["revision"],
        "expected_semantic_mapping_hash": projection["semantic_mapping_hash"],
        "dataset_id": projection["dataset_id"], "dataset_content_hash": projection["dataset_content_hash"],
        "directions": directions}


def test_actual_http_confirmation_restart_replay_alias_edit_and_sealed_outbound(tmp_path, monkeypatch):
    from marvis.llm_client import OpenAICompatibleLLMClient
    monkeypatch.setattr(OpenAICompatibleLLMClient, "complete", lambda *a, **kw: pytest.fail("semantic declarations must not request a model"))
    client, settings, task, dataset = _setup(tmp_path)
    initial = client.get(_base(task)).json()
    assert all(row["direction"] == "unknown" and not row["actor_confirmed"] for row in initial["fields"])
    body = _confirm_body(client, task, {"loan_amount": "lower_is_riskier", "mobile": "unknown"})
    response = client.post(_base(task) + "/confirm", json=body)
    assert response.status_code == 200, response.text
    known = next(row for row in response.json()["fields"] if row["column"] == "loan_amount")
    assert known["direction"] == "lower_is_riskier" and known["source"] == "user_confirmed"
    assert "actor_id" not in json.dumps(response.json())
    repository = DataWorkspaceRepository(settings.db_path)
    snapshot = repository.get_or_default(task.id)
    declaration = snapshot.semantic_mapping.risk_directions["loan_amount"]
    assert declaration.actor_id == "maker-1" and declaration.source_workspace_revision == 2
    replay = client.post(_base(task) + "/confirm", json=_confirm_body(client, task, body["directions"]))
    assert replay.status_code == 200 and replay.json()["revision"] == snapshot.revision
    # Normal semantic edits retain exact server-owned provenance, including old clients.
    draft = _workspace_draft(dataset, business_names={"loan_amount": "new opposite sounding alias"})
    saved = _put_workspace(client, task.id, draft, revision=snapshot.revision)
    assert saved.status_code == 200
    restarted = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    assert restarted.semantic_mapping.risk_directions["loan_amount"] == declaration
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    sealed = approved_field_directions(registry, task.id)
    payload = sealed.payload()
    assert next(row for row in payload["fields"] if row["column"] == "loan_amount")["direction"] == "lower_is_riskier"
    assert "new opposite sounding alias" not in json.dumps(payload)
    assert "maker-1" not in json.dumps(payload)
    with pytest.raises(OutboundContextError):
        ApprovedFieldDirections(sealed.binding_json, object()).payload()


def test_http_confirmation_auth_cas_source_and_unknown_never_infer(tmp_path):
    client, settings, task, dataset = _setup(tmp_path)
    body = _confirm_body(client, task, {"loan_amount": "higher_is_riskier"})
    assert client.post(_base(task) + "/confirm", json={**body, "actor_id": "other"}).status_code == 422
    assert client.post(_base(task) + "/confirm", json=body, headers={"x-test-actor": ""}).status_code == 403
    assert client.post(_base(task) + "/confirm", json={**body, "directions": {"risk decreases automatically": "lower_is_riskier"}}).status_code == 409
    assert client.post(_base(task) + "/confirm", json={**body, "expected_semantic_mapping_hash": "0" * 64}).status_code == 412
    assert client.post(_base(task) + "/confirm", json={**body, "directions": {"loan_amount": "inferred"}}).status_code == 409
    assert client.post(_base(task) + "/confirm", json={**body, "directions": {"loan_amount": "unknown"}}).status_code == 200
    unknown = next(row for row in client.get(_base(task)).json()["fields"] if row["column"] == "loan_amount")
    assert unknown["direction"] == "unknown" and unknown["actor_confirmed"]
    assert client.post(_base(task) + "/confirm", json=body).status_code == 412
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task.id)
    draft = _workspace_draft(dataset)
    declaration = json.loads(json.dumps(client.get(f"/api/tasks/{task.id}/data-workspace").json()["semantic_mapping"]["risk_directions"]))
    declaration["loan_amount"]["direction"] = "lower_is_riskier"
    draft["semantic_mapping"]["risk_directions"] = declaration
    assert _put_workspace(client, task.id, draft, revision=snapshot.revision).status_code == 422
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    path = registry.resolve_verified_path(dataset["id"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"changed")
        assert client.post(_base(task) + "/confirm", json=_confirm_body_from_snapshot(snapshot, {"loan_amount": "lower_is_riskier"})).status_code == 409
        assert client.get(_base(task)).status_code == 409
    finally:
        path.write_bytes(original)
    assert DataWorkspaceRepository(settings.db_path).get_or_default(task.id) == snapshot


def _confirm_body_from_snapshot(snapshot, directions):
    return {"expected_revision": snapshot.revision,
        "expected_semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping),
        "dataset_id": snapshot.active_dataset_id, "dataset_content_hash": snapshot.active_dataset_content_hash,
        "directions": directions}


def test_source_change_after_write_rolls_back_confirm_and_rename_requires_new_binding(tmp_path, monkeypatch):
    client, settings, task, dataset = _setup(tmp_path)
    repository = DataWorkspaceRepository(settings.db_path)
    before = repository.get_or_default(task.id)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    path = registry.resolve_verified_path(dataset["id"])
    original = path.read_bytes()
    resolve = DatasetRegistry.resolve_verified_path
    calls = []
    def changed(self, identity):
        calls.append(identity)
        if len(calls) == 2:
            path.write_bytes(original + b"changed-after-write")
        return resolve(self, identity)
    monkeypatch.setattr(DatasetRegistry, "resolve_verified_path", changed)
    try:
        with pytest.raises(Exception):
            confirm_field_directions(settings, task_id=task.id, actor_id="maker-1", **_confirm_body_from_snapshot(before, {"loan_amount": "higher_is_riskier"}))
    finally:
        path.write_bytes(original)
    assert len(calls) == 2 and repository.get_or_default(task.id) == before
    monkeypatch.setattr(DatasetRegistry, "resolve_verified_path", resolve)
    assert client.post(_base(task) + "/confirm", json=_confirm_body(client, task, {"loan_amount": "higher_is_riskier"})).status_code == 200
    snapshot = repository.get_or_default(task.id)
    migration = migrate_transform_semantics(snapshot.semantic_mapping, selected_field="loan_amount",
        operations=[{"op": "rename_columns", "mapping": {"loan_amount": "amount"}}],
        source_columns=["mobile", "bad_flag", "loan_amount"], result_columns=["mobile", "bad_flag", "amount"])
    copied = replace(snapshot, active_dataset_id="new-derived", active_dataset_content_hash="f" * 64,
        semantic_mapping=migration.semantic_mapping)
    changed_dataset = SimpleNamespace(id="new-derived", content_hash="f" * 64, columns=[SimpleNamespace(name="amount")])
    row = confirmed_field_directions(copied, changed_dataset)[0]
    assert row["direction"] == "unknown" and row["status"] == "requires_confirmation"
    assert migration.semantic_mapping.risk_directions["amount"] == snapshot.semantic_mapping.risk_directions["loan_amount"]


def test_actual_native_profile_preserves_confirmed_direction_with_current_semantic_hash(tmp_path):
    from pathlib import Path
    import sys
    from marvis.plugins.loader import load_builtin_packs
    from marvis.plugins.registry import PluginRegistry, ToolRegistry
    from marvis.plugins.runner import ToolRunner
    from marvis.plugins.manifest import ToolRef
    from marvis.repositories.plugins import PluginRepository
    from tests.test_data_ops_profile_dataset import _profile_runtime
    case = _profile_runtime(tmp_path)
    snapshot = confirm_field_directions(case.settings, task_id="task-1", actor_id="maker-1",
        **_confirm_body_from_snapshot(case.snapshot, {"amount": "non_monotonic"}))
    repo = PluginRepository(case.settings.db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis/packs")
    runner = ToolRunner(ToolRegistry(registry), repo, python_executable=sys.executable,
        datasets_root=case.settings.datasets_dir, workspace=case.settings.workspace)
    result = runner.invoke(ToolRef("data_ops", "profile_dataset"), {**case.inputs,
        "workspace_revision": snapshot.revision,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping)}, task_id="task-1")
    assert result.ok, result.error
    output = result.output["semantics"]["risk_directions"]["amount"]
    assert output["direction"] == "non_monotonic" and output["actor_id"] == "maker-1"
    assert result.output["semantic_mapping_hash"] != case.inputs["semantic_mapping_hash"]
