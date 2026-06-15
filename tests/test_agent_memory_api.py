from pathlib import Path

from fastapi.testclient import TestClient

from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.domain import TaskCreate


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _model_payload(**overrides):
    payload = {
        "ks": 30,
        "auc": 0.72,
        "psi": 0.08,
        "month": "202601",
        "channel": "自营",
        "model_name": "A卡模型",
        "model_version": "V2026",
        "scope": "贷前A卡",
        "source_task_id": "task-history",
        "important_feature_sources": ["征信"],
    }
    payload.update(overrides)
    return payload


def _create_model_memory(tmp_path: Path, **payload_overrides):
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    return store.create(
        MemoryCandidate(
            memory_type="model_experience",
            summary="A卡模型V2026在202601自营渠道KS为30。",
            payload=_model_payload(**payload_overrides),
            source_task_id=payload_overrides.get("source_task_id", "task-history"),
            confidence="high",
        )
    )


def test_memory_api_lists_memories_with_type_status_and_payload_filters(tmp_path):
    client = _client(tmp_path)
    matched = _create_model_memory(tmp_path)
    _create_model_memory(
        tmp_path,
        model_name="B卡模型",
        channel="联合贷",
        month="202512",
        source_task_id="task-b",
    )

    response = client.get(
        "/api/agent-memory",
        params={
            "memory_type": "model_experience",
            "status": "active",
            "model_name": "A卡模型",
            "channel": "自营",
            "month": "202601",
        },
    )

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [item["id"] for item in items] == [matched.id]
    assert items[0]["payload"]["ks"] == 30


def test_memory_api_rejects_invalid_filters_with_422(tmp_path):
    client = _client(tmp_path)

    response = client.get(
        "/api/agent-memory",
        params={"memory_type": "unknown_memory", "status": "active"},
    )

    assert response.status_code == 422
    assert "invalid memory filter" in response.json()["detail"]


def test_memory_api_gets_one_memory_with_audit_events(tmp_path):
    client = _client(tmp_path)
    memory = _create_model_memory(tmp_path)

    response = client.get(f"/api/agent-memory/{memory.id}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["memory"]["id"] == memory.id
    assert [event["event_type"] for event in payload["events"]] == [
        "create",
        "retrieve",
    ]


def test_memory_api_can_disable_enable_and_delete_memory(tmp_path):
    client = _client(tmp_path)
    memory = _create_model_memory(tmp_path)

    disabled = client.post(f"/api/agent-memory/{memory.id}/disable")
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["memory"]["status"] == "disabled"
    assert client.get("/api/agent-memory").json()["items"] == []

    enabled = client.post(f"/api/agent-memory/{memory.id}/enable")
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["memory"]["status"] == "active"

    deleted = client.delete(f"/api/agent-memory/{memory.id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["memory"]["status"] == "deleted"
    assert deleted.json()["memory"]["summary"] == ""
    assert deleted.json()["memory"]["payload"] == {}


def test_memory_api_cannot_enable_rejected_memory(tmp_path):
    client = _client(tmp_path)
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    rejected = store.create(
        MemoryCandidate(
            memory_type="task_experience",
            summary="OPENAI_API_KEY=sk-test-secret-value",
            payload={"raw": "OPENAI_API_KEY=sk-test-secret-value"},
            source_task_id="task-secret",
        )
    )

    response = client.post(f"/api/agent-memory/{rejected.id}/enable")

    assert response.status_code == 422
    assert "terminal" in response.json()["detail"]
    assert store.get_entry(rejected.id, include_deleted=True, audit=False).status == "rejected"


def test_memory_api_lists_references_attached_to_agent_message(tmp_path):
    client = _client(tmp_path)
    memory = _create_model_memory(tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    task = repo.create_task(
        TaskCreate(
            model_name="A卡模型",
            model_version="V2026",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="上一版KS更低。",
        metadata={
            "memory_references": [
                {
                    "id": memory.id,
                    "memory_type": "model_experience",
                    "source_task_id": memory.source_task_id,
                    "confidence": "high",
                    "use_reason": "chat",
                }
            ]
        },
    )

    response = client.get(
        f"/api/tasks/{task.id}/agent/messages/{message['id']}/memory-references"
    )

    assert response.status_code == 200, response.text
    assert response.json()["memory_references"][0]["id"] == memory.id
