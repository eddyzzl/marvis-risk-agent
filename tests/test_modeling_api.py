"""End-to-end HTTP test of the modeling entry through the generic PlanDriver.

task_type='modeling' is routed to the plan-conversation driver (NOT the retired
ModelingSession prototype): leakage-aware screen -> [confirm features] -> tune +
train -> [confirm model] -> model-development report. No LLM is configured — this
is the no-LLM manual scenario, confirming each gate with the "确认" content.
This test runs real screening/tuning/training, so it is slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


def _sample_dir(root: Path, n: int = 4000) -> Path:
    src = root / "modeling_material"
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    s1, s2, s3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.9 * s1 + 0.7 * s2 - 0.6 * s3 - 1.3)))
    y = (rng.uniform(size=n) < p).astype(float)
    split = np.array(["train"] * n, dtype=object)
    split[int(n * 0.5):int(n * 0.7)] = "test"
    split[int(n * 0.7):] = "oot"
    pd.DataFrame({
        "cust_id": np.arange(n),
        "sig1": s1, "sig2": s2, "sig3": s3,
        "long_y": y, "model_flag": split,
    }).to_parquet(src / "sample.parquet")
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_modeling_end_to_end(client: TestClient, tmp_path: Path):
    src = _sample_dir(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "建模驱动验证",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "modeling",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    # turn 0 — start: show the plan overview, pause at the plan-level 开始 gate
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]

    # turn 1 — 开始: screen features, pause at the confirm-features gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    gate1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert gate1["metadata"].get("kind") == "gate"
    assert "特征筛选完成" in gate1["content"]

    # turn 1 — confirm features: tune + train, pause at the confirm-model gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert gate2["metadata"].get("kind") == "gate"
    assert "训练完成" in gate2["content"]
    gate2_tables = gate2["metadata"].get("tables", [])
    # the model metrics table is shown at the gate
    assert any(t["title"] == "模型指标" for t in gate2_tables)
    # the trials leaderboard (G4) is shown at the model gate too
    assert any(t["title"].startswith("trials 排行") for t in gate2_tables)

    # turn 2 — confirm model: generate the model-development report, done
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "报告已生成" in done["content"]


def test_modeling_without_split_auto_generates_grouped_split(client: TestClient, tmp_path: Path):
    """No split column → make_split generates a grouped train/test split (no fabricated
    OOT) and modeling proceeds to the plan-overview gate, instead of erroring."""
    src = tmp_path / "nosplit"
    src.mkdir()
    n = 200
    rng = np.random.RandomState(3)
    pd.DataFrame({
        "cust_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "long_y": (rng.uniform(size=n) < 0.3).astype(float),
    }).to_parquet(src / "s.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "无切分", "validator": "qa", "source_dir": str(src),
        "task_type": "modeling", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    assistants = [m for m in msgs if m["role"] == "assistant"]
    # no error; it auto-split and reached the plan-overview gate
    assert not any((m.get("metadata") or {}).get("error") for m in assistants)
    assert any("已自动" in m["content"] and "train/test" in m["content"] for m in assistants)
    assert any((m.get("metadata") or {}).get("kind") == "plan_overview" for m in assistants)


def test_modeling_honors_selected_algorithms(client: TestClient, tmp_path: Path):
    """The user-selected algorithms (manual-mode multi-select → `recipes`) drive the
    modeling recipes instead of the lgb default (G2: 算法多选)."""
    src = _sample_dir(tmp_path, n=600)
    task_id = client.post("/api/tasks", json={
        "model_name": "选算法", "validator": "qa", "source_dir": str(src),
        "task_type": "modeling", "run_mode": "manual", "recipes": ["lr", "lgb"],
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    opening = next(m for m in msgs if m["role"] == "assistant")
    # the chosen algorithms are noted (multi-algorithm compare, not the lgb default only)
    assert "算法" in opening["content"]
    assert "lr" in opening["content"] and "lgb" in opening["content"]
    assert "取最优" in opening["content"]
