from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


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
