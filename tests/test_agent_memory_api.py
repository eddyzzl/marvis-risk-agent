from pathlib import Path

from fastapi.testclient import TestClient

from marvis.api import _agent_memory_context_from_store, _audit_agent_memory_use_from_store
from marvis.agent_memory.distillation import new_distillation
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_agent_memory_routes_live_outside_api_module():
    from marvis.routers import agent_memory

    assert any(route.path == "/api/agent-memory" for route in agent_memory.router.routes)
    assert all(
        route.endpoint.__module__ == "marvis.routers.agent_memory"
        for route in agent_memory.router.routes
    )


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
    assert response.json()["has_more"] is False


def test_memory_api_payload_filter_scans_past_first_storage_page(tmp_path):
    client = _client(tmp_path)
    matched = _create_model_memory(tmp_path, model_name="深页模型")
    for index in range(501):
        _create_model_memory(
            tmp_path,
            model_name=f"非匹配模型{index}",
            source_task_id=f"task-noise-{index}",
        )

    response = client.get(
        "/api/agent-memory",
        params={
            "memory_type": "model_experience",
            "model_name": "深页模型",
            "limit": 1,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [item["id"] for item in payload["items"]] == [matched.id]
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is False


def test_memory_api_supports_limit_offset_pagination(tmp_path):
    client = _client(tmp_path)
    for index in range(3):
        _create_model_memory(tmp_path, model_name=f"A卡模型{index}")

    first_page = client.get("/api/agent-memory", params={"limit": 1})
    second_page = client.get("/api/agent-memory", params={"limit": 1, "offset": 1})

    assert first_page.status_code == 200, first_page.text
    assert second_page.status_code == 200, second_page.text
    first_payload = first_page.json()
    second_payload = second_page.json()
    assert len(first_payload["items"]) == 1
    assert len(second_payload["items"]) == 1
    assert first_payload["items"][0]["id"] != second_payload["items"][0]["id"]
    assert first_payload["has_more"] is True
    assert first_payload["limit"] == 1
    assert first_payload["offset"] == 0
    assert second_payload["offset"] == 1


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


def test_memory_api_manages_distillations_and_rolls_back_superseding_version(tmp_path):
    client = _client(tmp_path)
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    source = store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="A卡验证里目标字段常用 bad_flag。",
            payload={"target_col": "bad_flag"},
            source_task_id="task-history",
            confidence="high",
        )
    )
    old = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="目标字段常见取值包括 bad_flag。",
            structured={"fields": {"target_col": ["bad_flag"]}},
            source_memory_ids=(source.id,),
            support_count=2,
        )
    )
    new = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="A卡目标字段常见取值包括 bad_flag 和 overdue_flag。",
            structured={"fields": {"target_col": ["bad_flag", "overdue_flag"]}},
            source_memory_ids=(source.id,),
            support_count=4,
        )
    )
    store.set_superseded(old.id, by=new.id)

    listed = client.get("/api/agent-memory/distillations")
    assert listed.status_code == 200, listed.text
    assert [item["id"] for item in listed.json()["items"]] == [new.id]

    detail = client.get(f"/api/agent-memory/distillations/{new.id}")
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["distillation"]["status"] == "active"
    assert payload["predecessor"]["id"] == old.id
    assert payload["source_memories"][0]["id"] == source.id
    assert payload["events"][0]["event_type"] == "create"

    rollback = client.post(f"/api/agent-memory/distillations/{new.id}/rollback")
    assert rollback.status_code == 200, rollback.text
    rollback_payload = rollback.json()
    assert rollback_payload["distillation"]["status"] == "rolled_back"
    assert rollback_payload["restored"]["id"] == old.id
    assert rollback_payload["restored"]["superseded_by"] is None

    relisted = client.get("/api/agent-memory/distillations")
    assert [item["id"] for item in relisted.json()["items"]] == [old.id]
    history = client.get(
        "/api/agent-memory/distillations",
        params={"include_superseded": True},
    )
    statuses = {item["id"]: item["status"] for item in history.json()["items"]}
    assert statuses[new.id] == "rolled_back"
    assert statuses[old.id] == "active"


def test_memory_distillation_api_supports_limit_offset_pagination(tmp_path):
    client = _client(tmp_path)
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    source = store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="A卡验证里目标字段常用 bad_flag。",
            payload={"target_col": "bad_flag"},
            source_task_id="task-history",
            confidence="high",
        )
    )
    for index in range(3):
        store.create_distillation(
            new_distillation(
                category="field_convention",
                scope_key=f"field_convention:target_col:{index}",
                distilled_summary=f"目标字段常见取值 {index}。",
                structured={"fields": {"target_col": ["bad_flag"]}},
                source_memory_ids=(source.id,),
                support_count=2,
            )
        )

    first_page = client.get("/api/agent-memory/distillations", params={"limit": 1})
    second_page = client.get(
        "/api/agent-memory/distillations",
        params={"limit": 1, "offset": 1},
    )

    assert first_page.status_code == 200, first_page.text
    assert second_page.status_code == 200, second_page.text
    first_payload = first_page.json()
    second_payload = second_page.json()
    assert len(first_payload["items"]) == 1
    assert len(second_payload["items"]) == 1
    assert first_payload["items"][0]["id"] != second_payload["items"][0]["id"]
    assert first_payload["has_more"] is True
    assert first_payload["limit"] == 1
    assert first_payload["offset"] == 0
    assert second_payload["offset"] == 1


def test_memory_distillation_references_are_use_audited(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡模型",
            model_version="V2026",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    distillation = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="A卡坏样本字段常见取值包括 bad_flag。",
            structured={"fields": {"target_col": ["bad_flag"]}},
            source_memory_ids=("mem-a", "mem-b", "mem-c", "mem-d"),
            support_count=4,
        )
    )
    message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="历史字段口径显示 bad_flag 常作为坏样本字段。",
        metadata={
            "memory_references": [
                {
                    "kind": "distillation",
                    "id": distillation.id,
                    "memory_type": "field_convention",
                    "confidence": "high",
                    "use_reason": "chat",
                }
            ]
        },
    )

    _audit_agent_memory_use_from_store(store, message, task_id=task.id)

    events = store.list_distillation_events(distillation.id)
    assert [event["event_type"] for event in events] == ["create", "use"]
    assert events[-1]["task_id"] == task.id
    assert events[-1]["message_id"] == message["id"]
    assert events[-1]["details"] == {"use_reason": "chat"}


def test_agent_message_memory_reference_route_uses_direct_lookup(tmp_path, monkeypatch):
    client = _client(tmp_path)
    db_path = tmp_path / "marvis.sqlite"
    repo = TaskRepository(db_path)
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
        content="历史字段口径显示 bad_flag 常作为坏样本字段。",
        metadata={
            "memory_references": [
                {
                    "kind": "raw",
                    "id": "mem-a",
                    "memory_type": "field_convention",
                    "confidence": "high",
                    "use_reason": "chat",
                }
            ]
        },
    )

    def fail_list_messages(*_args, **_kwargs):
        raise AssertionError("memory-reference route should not scan all Agent messages")

    monkeypatch.setattr(TaskRepository, "list_agent_messages", fail_list_messages)

    response = client.get(
        f"/api/tasks/{task.id}/agent/messages/{message['id']}/memory-references"
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["task_id"] == task.id
    assert payload["message_id"] == message["id"]
    assert payload["memory_references"][0]["id"] == "mem-a"


def test_memory_api_can_trigger_manual_consolidation(tmp_path):
    client = _client(tmp_path)
    store = AgentMemoryStore(tmp_path / "marvis.sqlite")
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
            confidence="high",
        )
    )

    response = client.post(
        "/api/agent-memory/consolidate",
        params={"category": "field_convention"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["consolidated"] == {"field_convention": 1}
    listed = client.get(
        "/api/agent-memory/distillations",
        params={"category": "field_convention"},
    )
    assert listed.json()["items"][0]["scope_key"] == "field_convention:target_col"


def test_agent_memory_context_uses_distillations_without_raw_memories(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡模型",
            model_version="V2026",
            validator="qa",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    distilled = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="A卡坏样本字段常见取值包括 bad_flag。",
            structured={"fields": {"target_col": ["bad_flag"]}},
            source_memory_ids=("mem-a", "mem-b", "mem-c", "mem-d"),
            support_count=4,
        )
    )

    context = _agent_memory_context_from_store(
        store,
        task,
        stage="chat",
        user_message="bad_flag 字段怎么处理？",
    )

    assert context is not None
    assert context["memories"][0]["kind"] == "distillation"
    assert context["memories"][0]["id"] == distilled.id
    assert context["memories"][0]["support_count"] == 4


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
