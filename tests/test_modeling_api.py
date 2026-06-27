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


def _multiclass_dir(root: Path, n: int = 900) -> Path:
    """A 3-class (credit grade) sample for the §8.3 multiclass conversational flow."""
    src = root / "multiclass_material"
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    grade = rng.randint(0, 3, n)
    split = np.array(["train"] * n, dtype=object)
    split[int(n * 0.55):int(n * 0.75)] = "test"
    split[int(n * 0.75):] = "oot"
    pd.DataFrame({
        "f1": grade + rng.uniform(size=n) * 0.5,
        "f2": (2 - grade) + rng.uniform(size=n) * 0.5,
        "grade": grade, "model_flag": split,
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

    # turn 1 — 开始: make the split, pause at the G1 split-review gate (spec §2)
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    split_gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert split_gate["metadata"].get("kind") == "gate"
    assert "样本切分完成" in split_gate["content"]

    # turn 2 — confirm the split: screen features, pause at the confirm-features gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    gate1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert gate1["metadata"].get("kind") == "gate"
    assert "特征筛选完成" in gate1["content"]
    # the screening gate carries the structured §4 payload (the frontend table consumes it)
    screen = gate1["metadata"].get("screen")
    assert screen is not None and screen["selected"], gate1["metadata"]
    assert screen["step_id"] and screen["thresholds"]["leakage_ks"] > 0
    proposed = screen["selected"]
    # edit the selection (drop the last proposed feature) to exercise the override path
    chosen = proposed[:-1] if len(proposed) > 1 else proposed

    # turn 1 — confirm features WITH an edited selection: override the screen's set,
    # then tune + train on exactly the chosen features, pause at the confirm-model gate
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "确认", "selection": chosen}
    )
    assert resp.status_code == 202, resp.text
    # the screen step's stored output now reflects the user's edited selection
    overridden = client.app.state.plan_repo.load_step_output(screen["step_id"])["selected"]
    assert overridden == chosen
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


def test_modeling_multiclass_completes_end_to_end(client: TestClient, tmp_path: Path):
    """§8.3 multiclass runs through the WHOLE conversational flow (split → screen → tune →
    train → report). Guards two real bugs fixed together: (1) a non-lgb primary recipe skips
    the lgb random search and returns empty best_params — the 调参 gate must accept that
    (not 'nonempty'); (2) the binary 7-sheet report would crash on a multiclass model — a
    minimal report is written instead so the plan completes with a downloadable artifact."""
    src = _multiclass_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "评级", "validator": "qa", "source_dir": str(src),
        "task_type": "modeling", "run_mode": "manual", "recipes": ["lgb_multiclass"],
    }).json()["id"]

    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    # target_type derived from lgb_multiclass — surfaced in the opening setup message
    assert any("多分类任务" in m["content"] for m in msgs if m["role"] == "assistant")

    # 开始 → split gate, 确认 → feature gate, 确认 → model gate (tune skipped, model trained)
    for content in ["开始", "确认", "确认"]:
        resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": content})
        assert resp.status_code == 202, resp.text
    model_gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert model_gate["metadata"].get("kind") == "gate"
    assert "训练完成" in model_gate["content"]  # tune-skip no longer fails the flow

    # 确认 → minimal non-binary report, plan done
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "报告已生成" in done["content"]
    assert not done["metadata"].get("error")


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


def test_modeling_honors_per_task_capability_tier(client: TestClient, tmp_path: Path):
    """The per-task capability tier (TIER-IA, spec §5.1) flows into the plan, setting
    its autonomy (replan) budget; absent → the global default (balanced)."""
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "档位", "validator": "qa", "source_dir": str(src),
        "task_type": "modeling", "run_mode": "manual", "recipes": ["lgb"],
        "capability_tier": "conservative",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert plans, "no plan was built"
    assert plans[0].tier == "conservative"

    # an unknown tier is normalized away → the plan falls back to the default
    default_id = client.post("/api/tasks", json={
        "model_name": "默认档", "validator": "qa", "source_dir": str(src),
        "task_type": "modeling", "run_mode": "manual", "recipes": ["lgb"],
        "capability_tier": "not-a-tier",
    }).json()["id"]
    client.post(f"/api/tasks/{default_id}/agent/start", json={})
    default_plans = client.app.state.plan_repo.list_plans_for_task(default_id)
    assert default_plans and default_plans[0].tier == "balanced"
