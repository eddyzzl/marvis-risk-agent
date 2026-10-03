"""The typed monitoring entry never executes before its existing overview gate."""
import copy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marvis.app import create_app
from marvis.api_schemas import AgentMessageRequest
from marvis.repositories.plans import PlanRepository
from marvis.repositories.tasks import TaskRepository
from marvis.domain import TaskCreate
from marvis.plugins.manifest import ToolRef
from test_modeling_monitor import _train_lr_experiment
from test_modeling_pack import _runtime


@pytest.fixture
def scenario(tmp_path):
    runner, _, registry, _, settings, _ = _runtime(tmp_path)
    task = TaskRepository(settings.db_path).create_task(TaskCreate(
        task_type="modeling", run_mode="agent", model_name="监控入口测试", model_version="dev",
        source_dir=str(tmp_path), validator="tester", target_col="y",
    ))
    trained, frame = _train_lr_experiment(runner, registry, tmp_path, task)
    experiment_id = trained.output["experiment_id"]
    selected = runner.invoke(ToolRef("modeling", "select_experiment"), {
        "experiment_ids": [experiment_id], "selected_experiment_id": experiment_id,
        "target_type": "binary", "refit_on_train_plus_test": False,
    }, task_id=task.id)
    assert selected.ok, selected.error
    path = tmp_path / "new-period.parquet"
    frame.drop(columns=["split"]).to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="monitoring.input")
    return SimpleNamespace(settings=settings, task=task, dataset=dataset, experiment_id=experiment_id)


def _request(client, s, **fields):
    route = f"/api/tasks/{s.task.id}/data-workspace"
    workspace = client.get(route).json()
    response = client.put(route, headers={"If-Match": str(workspace["revision"])}, json={
        "active_dataset_id": s.dataset.id, "active_dataset_content_hash": s.dataset.content_hash,
        "page": "overview", "selected_field": None,
        "semantic_mapping": {"target_col": None, "field_roles": {}, "business_names": {}},
    })
    assert response.status_code == 200, response.text
    workspace = response.json()
    return {"experiment_id": s.experiment_id, "dataset_id": s.dataset.id,
            "expected_content_hash": s.dataset.content_hash,
            "workspace_revision": workspace["revision"], "analysis_generation": workspace["analysis_generation"],
            "target_col": None, **fields}


@pytest.mark.parametrize("target", [None, "y"])
def test_proposal_preserves_unknown_maturity_and_waits_without_a_model_call(scenario, target):
    s = scenario
    app = create_app(s.settings)
    with TestClient(app) as client:
        body = _request(client, s, target_col=target)
        response = client.post(f"/api/tasks/{s.task.id}/agent/messages", json={
            "content": "检查已选模型在这份数据上的风险，技术阈值不代表业务通过。",
            "model_monitoring_request": body, "acceptance_mode": "auto_accept",
        })
        assert response.status_code == 202, response.text
        plan = client.get(f"/api/tasks/{s.task.id}/plans").json()["plans"][0]
        assert plan["template_id"] == "monitoring_run" and plan["status"] == "validated"
        assert all(step["status"] == "pending" and not step.get("output_ref") for step in plan["steps"])
        repo = PlanRepository(s.settings.db_path)
        assert all(not repo.list_step_runs(step["id"]) for step in plan["steps"])
        messages = response.json()["messages"]
        overview = next(m for m in messages if "model_monitoring_proposal" in m.get("metadata", {}))
        proposal = overview["metadata"]["model_monitoring_proposal"]
        assert proposal["label_maturity_assurance"] == "unknown"
        assert proposal["business_acceptance"] == "not_established"
        assert proposal["threshold_source"] == "platform_default_technical_thresholds"
        assert proposal["label_mode"] == ("unlabeled_drift_only" if target is None else "declared_label_column")
        assert plan["steps"][0]["inputs"]["monitoring_binding"] == proposal["monitoring_binding"]
        assert plan["steps"][1]["inputs"]["monitoring_policy"] == {"thresholds": proposal["thresholds"]}


def test_workspace_edit_after_overview_cannot_execute_the_old_binding(scenario):
    s = scenario
    with TestClient(create_app(s.settings)) as client:
        body = _request(client, s)
        response = client.post(f"/api/tasks/{s.task.id}/agent/messages", json={
            "content": "监控已选模型", "model_monitoring_request": body,
        })
        assert response.status_code == 202, response.text
        plan = client.get(f"/api/tasks/{s.task.id}/plans").json()["plans"][0]
        route = f"/api/tasks/{s.task.id}/data-workspace"
        workspace = client.get(route).json()
        changed = {k: workspace[k] for k in (
            "active_dataset_id", "active_dataset_content_hash", "page", "selected_field", "semantic_mapping")}
        changed["page"] = "fields"
        assert client.put(route, headers={"If-Match": str(workspace["revision"])}, json=changed).status_code == 200
        response = client.post(f"/api/tasks/{s.task.id}/agent/messages", json={
            "content": "确认执行这个计划", "ui_action": "start_plan", "expected_plan_id": plan["id"],
            **plan["confirmation_snapshot"], "acceptance_mode": "manual_review",
        })
        assert response.status_code == 202, response.text
        current = client.get(f"/api/plans/{plan['id']}").json()["plan"]
        assert current["status"] == "failed"
        assert not any(step.get("output_ref") for step in current["steps"])


def test_monitoring_contract_rejects_identity_coercion_and_extra_business_claims():
    body = {"content": "监控", "model_monitoring_request": {
        "experiment_id": "model", "dataset_id": "new", "expected_content_hash": "a" * 64,
        "workspace_revision": 0, "analysis_generation": 0,
    }}
    for field, value in (("workspace_revision", True), ("analysis_generation", "0"),
                         ("business_acceptance", True), ("thresholds", {})):
        altered = copy.deepcopy(body)
        altered["model_monitoring_request"][field] = value
        with pytest.raises(ValidationError):
            AgentMessageRequest.model_validate(altered)


def test_source_change_before_plan_transaction_rolls_back_with_actionable_conflict(scenario, monkeypatch):
    from marvis.agent.turn_handlers import model_monitoring
    from marvis.packs.modeling.errors import ModelingError

    s = scenario
    prepare = model_monitoring.prepare_model_monitoring
    calls = 0
    def changed(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ModelingError("monitoring_workspace_changed")
        return prepare(*args)
    monkeypatch.setattr(model_monitoring, "prepare_model_monitoring", changed)
    with TestClient(create_app(s.settings)) as client:
        body = _request(client, s)
        response = client.post(f"/api/tasks/{s.task.id}/agent/messages", json={
            "content": "检查模型漂移", "model_monitoring_request": body,
        })
        assert response.status_code == 409 and "重建计划" in response.text
        assert client.get(f"/api/tasks/{s.task.id}/plans").json()["plans"] == []
