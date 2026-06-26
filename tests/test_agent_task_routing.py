"""Routing guard: unwired agent task types reject explicitly (HTTP 501).

Covers the C3 blocker surfaced by the V2 plan review. feature_analysis / strategy /
vintage are advertised as agent entries in the create flow but have no dedicated agent
backend yet. Their /agent/start and /agent/messages must fail loudly rather than
silently fall through to the validation agent on a foreign goal prompt.

data_join IS wired (PlanDriver + data_join template) — its end-to-end behavior is
covered in test_data_join_api.py, so it is no longer in the unwired parametrization.
When each remaining type's agent flow is wired, add it to _WIRED_AGENT_TASK_TYPES in
api.py and move it out of the unwired parametrization here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _make_agent_task(client: TestClient, tmp_path: Path, task_type: str) -> str:
    src = tmp_path / f"src_{task_type}"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cust_id": [1, 2, 3], "x": [0.1, 0.2, 0.3]}).to_csv(
        src / "data.csv", index=False
    )
    resp = client.post(
        "/api/tasks",
        json={
            "model_name": f"{task_type}-task",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": task_type,
            "run_mode": "agent",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


@pytest.mark.parametrize(
    "task_type", ["strategy", "vintage"]
)
def test_unwired_agent_start_rejects_explicitly(client, tmp_path, task_type):
    task_id = _make_agent_task(client, tmp_path, task_type)
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 501, resp.text
    assert "尚未接入" in resp.json()["detail"]


@pytest.mark.parametrize("task_type", ["strategy"])
def test_unwired_agent_message_rejects_explicitly(client, tmp_path, task_type):
    task_id = _make_agent_task(client, tmp_path, task_type)
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "开始吧"}
    )
    assert resp.status_code == 501, resp.text
    assert "尚未接入" in resp.json()["detail"]
