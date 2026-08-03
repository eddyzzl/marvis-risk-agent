from types import SimpleNamespace

from fastapi.testclient import TestClient

from marvis.agent import validation_app_service as service
from marvis.app import create_app
from marvis.db import TaskRepository


def test_explicit_driver_request_model_is_reused_by_strategy_compiler(
    tmp_path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "request-model-routing",
            "validator": "pytest",
            "source_dir": str(source_dir),
            "task_type": "strategy",
            "run_mode": "agent",
            "target_col": "bad",
        },
    )
    assert response.status_code == 200, response.text

    task_id = response.json()["id"]
    repo = TaskRepository(client.app.state.settings.db_path)
    selected_client = object()
    default_router_client = object()
    observed = {}

    monkeypatch.setattr(
        service,
        "driver_llm_client",
        lambda _request, _task: default_router_client,
    )

    def capture_runtime(runtime, _repo, task, **_kwargs):
        observed["task_id"] = task.id
        observed["llm_client"] = runtime.llm_client
        return {"task_id": task.id, "status": "message_saved", "messages": []}

    monkeypatch.setattr(service, "dispatch_plan_driver_turn", capture_runtime)
    request = SimpleNamespace(
        app=client.app,
        state=SimpleNamespace(local_principal=None),
    )

    result = service.dispatch_driver_turn(
        request,
        repo,
        repo.get_task(task_id),
        user_text="请创建 SampleDesign",
        agent_client=selected_client,
        recovery_model_id="deepseek-v4-pro",
        recovery_effort="high",
    )

    assert result["status"] == "message_saved"
    assert observed == {
        "task_id": task_id,
        "llm_client": selected_client,
    }


def test_driver_without_explicit_request_model_keeps_router_role_client(
    tmp_path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "default-router-routing",
            "validator": "pytest",
            "source_dir": str(source_dir),
            "task_type": "strategy",
            "run_mode": "agent",
            "target_col": "bad",
        },
    )
    assert response.status_code == 200, response.text

    task_id = response.json()["id"]
    repo = TaskRepository(client.app.state.settings.db_path)
    gate_client = object()
    default_router_client = object()
    observed = {}

    monkeypatch.setattr(
        service,
        "driver_llm_client",
        lambda _request, _task: default_router_client,
    )

    def capture_runtime(runtime, _repo, task, **_kwargs):
        observed["llm_client"] = runtime.llm_client
        return {"task_id": task.id, "status": "message_saved", "messages": []}

    monkeypatch.setattr(service, "dispatch_plan_driver_turn", capture_runtime)
    request = SimpleNamespace(
        app=client.app,
        state=SimpleNamespace(local_principal=None),
    )

    service.dispatch_driver_turn(
        request,
        repo,
        repo.get_task(task_id),
        user_text="继续",
        agent_client=gate_client,
    )

    assert observed["llm_client"] is default_router_client
