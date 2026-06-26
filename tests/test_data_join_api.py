"""End-to-end HTTP test of the data-join entry through the generic PlanDriver.

Drives the real FastAPI endpoints (/agent/start, /agent/messages) for a
task_type='data_join' task whose materials are two joinable tables (an anchor
sample with a label + a feature table keyed by md5(mobile)). No LLM is
configured — this is exactly the no-LLM preview scenario the manual-first build
targets. Asserts the driver pauses at the forced-confirm gate (showing join
diagnostics) and, on confirmation, executes a 1:1-anchored join.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


def _join_dir(root: Path, n: int = 50) -> Path:
    src = root / "join_material"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    pd.DataFrame({
        "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
        "balance": list(range(n)),
    }).to_parquet(src / "features.parquet")
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_data_join_conversation_end_to_end(client: TestClient, tmp_path: Path):
    src = _join_dir(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "拼接测试",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    # turn 0 — start: C1 file-role assignment gate (propose anchor/feature + target)
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    c1 = _last_assistant(msgs)
    assert "样本主表" in c1["content"]
    assert c1["metadata"]["join_c1"]["anchor_id"]  # proposal carried for the form
    assert any(t["title"].startswith("输入文件") for t in c1["metadata"].get("tables", []))

    # turn 1 — confirm C1 roles: build the join plan, pause at the plan-overview gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]

    # turn 2 — 开始: run the plan, pause at the C2 diagnostics gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]
    assert any(t["title"].startswith("拼接诊断") for t in gate["metadata"].get("tables", []))

    # turn 3 — confirm C2: confirm_join + execute_join run, anchor preserved 1:1
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


def test_data_join_c1_form_assignment_drives_the_join(client: TestClient, tmp_path: Path):
    """The C1 control form posts a structured [C1] assignment; the driver honors it."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接表单", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    state = c1["metadata"]["join_c1"]
    anchor = state["anchor_id"]
    features = state["feature_ids"]
    payload = json.dumps({"anchor_id": anchor, "feature_ids": features, "target_col": state["target_col"]})
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": f"[C1]{payload}"})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]
    # 开始 → run to the C2 diagnostics gate
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]


def test_data_join_single_file_confirms_then_skips(client: TestClient, tmp_path: Path):
    src = tmp_path / "single_material"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(src / "only.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "单文件", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    # C1 still confirms the sample + target even with one file (spec §1 / §3 single-file degenerate)
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert c1["metadata"]["join_c1"]["skip"] is True
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    skip = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "无需拼接" in skip["content"]


def test_strategy_task_still_returns_501(client: TestClient, tmp_path: Path):
    """Routing allowlist: an un-wired agent task type must still error explicitly,
    not silently fall through to the validation agent."""
    src = tmp_path / "strat"
    src.mkdir()
    pd.DataFrame({"x": [1, 2]}).to_parquet(src / "s.parquet")
    resp = client.post("/api/tasks", json={
        "model_name": "策略",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "strategy",
        "run_mode": "agent",
    })
    task_id = resp.json()["id"]
    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 501
