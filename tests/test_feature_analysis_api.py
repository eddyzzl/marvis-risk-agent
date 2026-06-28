"""End-to-end HTTP test of the standalone feature-analysis entry (spec §1 form A).

Drives /agent/start for a task_type='feature_analysis' task whose material is a
single sample (target + numeric features). No LLM is configured — this is the
no-LLM manual scenario: the driver computes the per-feature metrics and returns
the wide table in one synchronous run, no screening gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


def _sample_dir(root: Path, n: int = 3000) -> Path:
    src = root / "feature_material"
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    s1, s2, s3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.8 * s1 + 0.6 * s2 - 0.5 * s3 - 1.2)))
    y = (rng.uniform(size=n) < p).astype(float)
    pd.DataFrame({
        "cust_id": np.arange(n),
        "sig1": s1, "sig2": s2, "sig3": s3,
        "long_y": y,
    }).to_parquet(src / "sample.parquet")
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_feature_analysis_end_to_end(client: TestClient, tmp_path: Path):
    src = _sample_dir(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "特征分析验证",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]
    # 开始 → compute the per-feature metrics, then generate the Excel report → DONE
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    done = _last_assistant(msgs)
    assert "特征分析完成" in done["content"]
    # the downloadable Excel feature-analysis report (form A) was generated
    assert "特征分析报告已生成" in done["content"]
    table = next((t for t in done["metadata"].get("tables", []) if t["title"] == "特征指标"), None)
    assert table is not None
    # one row per analysed feature (sig1/sig2/sig3; cust_id/long_y excluded)
    feature_names = {row[0] for row in table["rows"]}
    assert {"sig1", "sig2", "sig3"} <= feature_names
    assert "long_y" not in feature_names and "cust_id" not in feature_names
    # No optional metric selected → no collinear / VIF section computed (spec §2).
    titles = {t["title"] for t in done["metadata"].get("tables", [])}
    assert "VIF(共线性)" not in titles


def test_feature_analysis_with_vif_metric_shows_collinear_section(client: TestClient, tmp_path: Path):
    """Selecting the VIF metric at creation computes + surfaces the 共线性 section.

    Drives the full chain: create payload metrics=["vif"] → task → feature setup →
    template slot → compute_feature_metrics(metrics=["vif"]) → driver render.
    """
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "共线分析", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
        "metrics": ["vif"],
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    titles = {t["title"] for t in done["metadata"].get("tables", [])}
    assert "特征指标" in titles  # base metrics still present
    assert "VIF(共线性)" in titles  # the selected optional metric was computed + shown


def test_feature_analysis_multiple_files_runs_join_then_feature_analysis(client: TestClient, tmp_path: Path):
    src = _sample_dir(tmp_path, n=200)
    pd.DataFrame({
        "cust_id": np.arange(200),
        "external_score": np.linspace(0, 1, 200),
    }).to_parquet(src / "feature_table.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "多表特征",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
    }).json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    join_gate = _last_assistant(messages)
    assert join_gate["metadata"].get("kind") == "gate"
    assert "拼接诊断完成" in join_gate["content"]

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "特征分析完成" in done["content"]
    table = next(t for t in done["metadata"]["tables"] if t["title"] == "特征指标")
    feature_names = {row[0] for row in table["rows"]}
    assert "external_score" in feature_names

    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert plans and plans[-1].template_id == "feature_analysis_with_join"


def test_feature_analysis_with_head_tail_lift_adds_columns(client: TestClient, tmp_path: Path):
    """Selecting head/tail lift adds the risk-aware 头部/尾部 lift columns to the wide
    table; without it those columns are absent (base table keeps its 7 columns)."""
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "头尾lift", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
        "metrics": ["head_tail_lift"],
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    table = next(t for t in done["metadata"]["tables"] if t["title"] == "特征指标")
    for col in ("头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%"):
        assert col in table["columns"]


def test_feature_analysis_with_importance_adds_column(client: TestClient, tmp_path: Path):
    """Selecting feature importance trains one pinned LGB model and adds the 重要性
    column; the per-feature importances are present (a fraction of total gain)."""
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "重要性", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
        "metrics": ["importance"],
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    table = next(t for t in done["metadata"]["tables"] if t["title"] == "特征指标")
    assert "重要性" in table["columns"]
    assert "头部lift5%" not in table["columns"]  # other optional metrics stay off


def test_finished_task_builds_fresh_plan_not_replay(client: TestClient, tmp_path: Path):
    """Re-engaging a task whose plan already finished must build a NEW plan, not
    resume the terminal one (which would just replay its final message forever)."""
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "重启验证", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
    }).json()["id"]
    # start shows the plan overview; 开始 then runs the single-step plan to DONE.
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    plans_after_start = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans_after_start) == 1
    assert plans_after_start[0]["status"] == "done"

    # Re-engaging the finished task builds a second, fresh plan rather than resuming
    # the terminal one.
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "再分析一次"})
    plans_after_reengage = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans_after_reengage) == 2
    assert plans_after_reengage[-1]["id"] != plans_after_start[0]["id"]


def test_feature_report_is_downloadable_after_run(client: TestClient, tmp_path: Path):
    """The generated feature-analysis Excel report is downloadable via the driver-report
    endpoint once the flow has run; 404 before it exists."""
    src = _sample_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "下载报告", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
    }).json()["id"]

    # before the report exists → 404
    assert client.get(f"/api/tasks/{task_id}/driver-report/download").status_code == 404

    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})

    resp = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert resp.status_code == 200, resp.text
    assert "spreadsheetml" in resp.headers["content-type"]
    assert resp.content[:2] == b"PK"  # a real .xlsx (zip) file


def test_feature_analysis_without_target_reports_error(client: TestClient, tmp_path: Path):
    src = tmp_path / "no_target"
    src.mkdir()
    pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}).to_parquet(src / "x.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "无标签", "validator": "qa", "source_dir": str(src),
        "task_type": "feature_analysis", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    assert "目标列" in _last_assistant(msgs)["content"]
