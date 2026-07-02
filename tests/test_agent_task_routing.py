"""Routing guard for driver-backed agent task types.

data_join / feature_analysis / strategy / vintage are wired through PlanDriver.
These tests pin the late-added strategy and vintage entries so they do not
regress to 501 placeholders.
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


def test_strategy_agent_start_builds_plan_and_reaches_strategy_gate(client, tmp_path):
    src = tmp_path / "strategy"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "bad": [1, 0, 0, 0, 1, 0],
        "score": [580, 620, 730, 760, 590, 800],
    }).to_csv(src / "strategy.csv", index=False)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "策略回测",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    start_messages = started.json()["messages"]
    assert any("开始策略分析" in message["content"] for message in start_messages)
    assert start_messages[-1]["metadata"]["kind"] == "plan_overview"

    confirmed = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert confirmed.status_code == 202, confirmed.text
    gate = confirmed.json()["messages"][-1]
    assert gate["metadata"]["kind"] == "gate"
    assert "策略候选已生成" in gate["content"]
    assert any(
        table["title"] == "策略规则（按顺序命中）"
        for table in gate["metadata"]["tables"]
    )

    finished = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert finished.status_code == 202, finished.text
    done = finished.json()["messages"][-1]
    assert "策略权衡视图完成" in done["content"]
    assert any(table["title"] == "cutoff 权衡点" for table in done["metadata"]["tables"])


def test_vintage_agent_start_builds_plan_and_returns_curve(client, tmp_path):
    src = tmp_path / "vintage"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cohort": ["202601", "202601", "202602", "202602"],
        "mob": [0, 1, 0, 1],
        "bad": [0, 1, 0, 0],
    }).to_csv(src / "vintage.csv", index=False)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "Vintage 分析",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "vintage",
            "run_mode": "manual",
            "target_col": "bad",
            "time_col": "cohort",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    assert any("开始 Vintage 风险分析" in message["content"] for message in started.json()["messages"])

    confirmed = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert confirmed.status_code == 202, confirmed.text
    done = confirmed.json()["messages"][-1]
    assert "Vintage 曲线完成" in done["content"]
    assert any(
        table["title"] == "Vintage 累计坏账率"
        for table in done["metadata"]["tables"]
    )
