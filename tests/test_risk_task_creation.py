from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import DatasetRepository, TaskRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_agent_risk_task_can_be_created_before_materials(client):
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "风险收益分析",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )

    assert response.status_code == 200, response.text
    source_dir = Path(response.json()["source_dir"])
    assert source_dir.is_dir()
    assert source_dir.parent == client.app.state.settings.workspace / "material_uploads"
    assert source_dir.name.startswith("risk-intake-")


def test_deleting_agent_risk_task_removes_its_generated_intake_directory(client):
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "待删除风险分析",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )
    task_id = created.json()["id"]
    source_dir = Path(created.json()["source_dir"])
    (source_dir / "temporary-note.txt").write_text("task-owned", encoding="utf-8")

    deleted = client.delete(f"/api/tasks/{task_id}")

    assert deleted.status_code == 204, deleted.text
    assert not source_dir.exists()


def test_failed_agent_risk_task_creation_removes_unclaimed_intake_directory(
    client,
    monkeypatch,
):
    uploads_root = client.app.state.settings.workspace / "material_uploads"

    def fail_create_task(*_args, **_kwargs):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(
        TaskRepository,
        "create_task_on_connection",
        fail_create_task,
    )

    with pytest.raises(RuntimeError, match="database write failed"):
        client.post(
            "/api/tasks",
            json={
                "model_name": "创建失败的风险分析",
                "validator": "qa",
                "source_dir": "",
                "task_type": "vintage",
                "run_mode": "agent",
            },
        )

    assert list(uploads_root.glob("risk-intake-*")) == []


def test_empty_material_dir_is_rejected_for_other_tasks(client):
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "缺材料建模",
            "validator": "qa",
            "source_dir": "",
            "task_type": "modeling",
            "run_mode": "agent",
        },
    )

    assert response.status_code == 422
    assert "source_dir is required" in response.text


def test_agent_risk_task_start_asks_goal_without_llm_or_materials(client):
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "先访谈",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert started.status_code == 202, started.text
    assistant = [
        message
        for message in started.json()["messages"]
        if message["role"] == "assistant"
    ][-1]
    assert "你想分析什么" in assistant["content"]
    assert "VTG终值与年化不良" in assistant["content"]
    assert "收益测算" in assistant["content"]
    assert assistant["metadata"]["risk_analysis_intake"]["phase"] == "ask_goal"


def test_agent_risk_kind_uses_two_pass_semantics_without_menu_keywords(
    client,
    monkeypatch,
):
    callers: list[str] = []

    class FakeLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            callers.append(kwargs["caller"])
            return json.dumps(
                {
                    "intent": "risk_profitability",
                    "evidence_quote": "到底能赚多少钱",
                    "reason": "用户要拆解收入与成本并测算盈利。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_action": False,
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.OpenAICompatibleLLMClient",
        FakeLLMClient,
    )
    configured = client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "测试路由模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "intent-test",
                    "api_key": "secret",
                }
            ],
        },
    )
    assert configured.status_code == 200, configured.text
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "语义风险分析",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )
    task_id = created.json()["id"]
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "我想知道这批客户到底能赚多少钱，顺便拆一下收入成本",
            "model_id": "m1",
        },
    )

    assert response.status_code == 202, response.text
    assistant = [
        message for message in response.json()["messages"] if message["role"] == "assistant"
    ][-1]
    intake = assistant["metadata"]["risk_analysis_intake"]
    assert intake["phase"] == "request_materials"
    assert intake["analysis_kind"] == "profitability"
    assert "canonical economics" in assistant["content"]
    assert callers == ["semantic_intent_router", "semantic_intent_reviewer"]


def test_semantic_route_is_discarded_when_task_state_changes_during_review(
    client,
    monkeypatch,
):
    task_ref: dict[str, str] = {}

    class MutatingLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            if kwargs["caller"] == "semantic_intent_reviewer":
                TaskRepository(client.app.state.settings.db_path).add_agent_message(
                    task_ref["id"],
                    role="assistant",
                    stage="chat",
                    content="并发状态更新",
                    metadata={"kind": "concurrent_update"},
                )
            return json.dumps(
                {
                    "intent": "risk_profitability",
                    "evidence_quote": "赚多少钱",
                    "reason": "用户要测算收益",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_action": False,
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.OpenAICompatibleLLMClient",
        MutatingLLMClient,
    )
    assert client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "并发复核模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "intent-test",
                    "api_key": "secret",
                }
            ],
        },
    ).status_code == 200
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "语义并发防护",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )
    task_ref["id"] = created.json()["id"]
    assert client.post(
        f"/api/tasks/{task_ref['id']}/agent/start",
        json={},
    ).status_code == 202

    response = client.post(
        f"/api/tasks/{task_ref['id']}/agent/messages",
        json={"content": "我想知道能赚多少钱", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    last = response.json()["messages"][-1]
    assert last["metadata"]["code"] == "semantic_intent_clarification"
    assert "状态已变化" in last["metadata"]["reason"]
    intakes = [
        message["metadata"]["risk_analysis_intake"]
        for message in response.json()["messages"]
        if "risk_analysis_intake" in message["metadata"]
    ]
    assert intakes[-1]["phase"] == "ask_goal"
    assert client.get(f"/api/tasks/{task_ref['id']}/plans").json()["plans"] == []


@pytest.mark.parametrize("mutation", ("workspace", "task", "dataset"))
def test_semantic_route_is_discarded_when_authenticated_data_state_changes(
    client,
    monkeypatch,
    mutation: str,
):
    created = client.post(
        "/api/tasks",
        json={
            "model_name": f"语义数据并发防护-{mutation}",
            "validator": "qa",
            "source_dir": "",
            "task_type": "vintage",
            "run_mode": "agent",
        },
    )
    task_id = created.json()["id"]
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    first_source = settings.workspace / f"semantic-{mutation}-first.csv"
    second_source = settings.workspace / f"semantic-{mutation}-second.csv"
    first_source.write_text("id,amount,bad\n1,111,0\n", encoding="utf-8")
    second_source.write_text("id,amount,bad\n2,999,1\n", encoding="utf-8")
    first = registry.register_from_upload(task_id, first_source, role="sample")
    second = registry.register_from_upload(task_id, second_source, role="sample")

    workspace_repo = DataWorkspaceRepository(settings.db_path)
    workspace = workspace_repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=first.id,
            active_dataset_content_hash=first.content_hash,
        ),
        expected_revision=0,
    )
    workspace_repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=first.id,
            active_dataset_content_hash=first.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={"bad": "target"},
            ),
        ),
        expected_revision=workspace.revision,
    )
    assert client.post(f"/api/tasks/{task_id}/agent/start", json={}).status_code == 202

    mutation_applied = False

    class MutatingLLMClient:
        def __init__(self, profile):
            self.profile = profile

        def complete(self, **kwargs):
            nonlocal mutation_applied
            if kwargs["caller"] == "semantic_intent_reviewer" and not mutation_applied:
                if mutation == "workspace":
                    current = workspace_repo.get_or_default(task_id)
                    workspace_repo.save(
                        task_id,
                        DataWorkspaceDraft(
                            active_dataset_id=second.id,
                            active_dataset_content_hash=second.content_hash,
                        ),
                        expected_revision=current.revision,
                    )
                elif mutation == "task":
                    TaskRepository(settings.db_path).update_target_col(
                        task_id,
                        "changed_bad",
                    )
                else:
                    registry.set_role(first.id, "derived")
                mutation_applied = True
            return json.dumps(
                {
                    "intent": "risk_profitability",
                    "evidence_quote": "赚多少钱",
                    "reason": "用户要测算收益",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_action": False,
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.validation_app_service.OpenAICompatibleLLMClient",
        MutatingLLMClient,
    )
    assert client.put(
        "/api/settings/llm",
        json={
            "default_model_id": "m1",
            "models": [
                {
                    "model_id": "m1",
                    "enabled": True,
                    "display_name": "数据并发复核模型",
                    "provider": "OpenAI Compatible",
                    "api_base_url": "https://example.test/v1",
                    "model_name": "intent-test",
                    "api_key": "secret",
                }
            ],
        },
    ).status_code == 200

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "我想知道能赚多少钱", "model_id": "m1"},
    )

    assert response.status_code == 202, response.text
    assert mutation_applied is True
    assert response.json()["status"] == "clarification_required"
    last = response.json()["messages"][-1]
    assert last["metadata"]["code"] == "semantic_intent_clarification"
    assert "状态已变化" in last["metadata"]["reason"]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
