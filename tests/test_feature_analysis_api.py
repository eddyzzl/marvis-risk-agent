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

from marvis.agent.renderers import _render_feature_metrics
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


def _ambiguous_target_dir(root: Path, n: int = 300) -> Path:
    src = root / "ambiguous_feature_material"
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    x1 = rng.normal(size=n)
    pd.DataFrame({
        "x1": x1,
        "label_a": (x1 > 0).astype(int),
        "label_b": (x1 < 0.5).astype(int),
    }).to_parquet(src / "sample.parquet")
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


def _finish_optional_binning(
    client: TestClient,
    task_id: str,
    *,
    features: list[str] | None = None,
    bins: int = 10,
) -> dict:
    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    gate = _last_assistant(messages)
    assert gate["metadata"].get("feature_binning") is not None
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={
        "content": "确认",
        "ui_action": "confirm_feature_binning",
        "expected_step_id": gate["metadata"]["step_id"],
        "adjust_params": {"features": features or [], "bins": bins},
    })
    assert response.status_code == 202, response.text
    return _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )


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
    assert "手动模式请点击「开始执行」" in overview["content"]
    assert "Agent 模式请回复「开始」或「继续」" in overview["content"]
    # 开始 → compute per-feature metrics → stop at the optional binning HITL gate.
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    bin_gate = _last_assistant(msgs)
    assert bin_gate["metadata"]["kind"] == "gate"
    assert bin_gate["metadata"]["feature_binning"]["default_bins"] == 10
    assert {item["feature"] for item in bin_gate["metadata"]["feature_binning"]["features"]} >= {
        "sig1", "sig2", "sig3"
    }
    assert "report_download" not in bin_gate["metadata"]

    # Select two features, request five bins, then generate the report.
    gate_step_id = bin_gate["metadata"]["step_id"]
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={
        "content": "确认",
        "ui_action": "confirm_feature_binning",
        "expected_step_id": gate_step_id,
        "adjust_params": {"features": ["sig1", "sig2"], "bins": 5},
    })
    assert response.status_code == 202, response.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "特征分析完成" in done["content"]
    # the downloadable Excel feature-analysis report (form A) was generated
    assert "特征分析报告已生成" in done["content"]
    table = next((t for t in done["metadata"].get("tables", []) if t["title"] == "特征指标"), None)
    assert table is not None
    # one row per analysed feature (sig1/sig2/sig3; cust_id/long_y excluded)
    feature_names = {row[0] for row in table["rows"]}
    assert {"sig1", "sig2", "sig3"} <= feature_names
    assert "long_y" not in feature_names and "cust_id" not in feature_names
    rendered_tables = {item["title"]: item for item in done["metadata"]["tables"]}
    assert "Agent建议" in rendered_tables["Agent 特征建议"]["columns"]
    # FEATURE §2 defaults are IV/KS/AUC/coverage only. Optional PSI and lift
    # families are neither computed nor rendered until selected.
    assert "PSI 稳定性" not in rendered_tables
    assert "头尾 Lift（风险方向）" not in rendered_tables
    assert done["metadata"]["report_download"]["download_url"].endswith(
        "/driver-report/download"
    )
    bin_table = next(t for t in done["metadata"]["tables"] if t["title"] == "分箱分析")
    assert {row[0] for row in bin_table["rows"]} == {"sig1", "sig2"}
    assert all(isinstance(row[2], int) for row in bin_table["rows"])
    for column in ("单一值率", "零值率", "缺失率"):
        assert column in rendered_tables["数据质量"]["columns"]
    # VIF remains opt-in; the default metrics do not train a model.
    titles = {t["title"] for t in done["metadata"].get("tables", [])}
    assert "VIF（共线性）" not in titles


def test_feature_analysis_ambiguous_target_can_be_selected_and_persists(
    client: TestClient,
    tmp_path: Path,
):
    src = _ambiguous_target_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "目标列选择",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})

    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始"},
    )
    assert first.status_code == 202, first.text
    gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert gate["metadata"]["feature_target_choice"]["candidates"] == [
        "label_a",
        "label_b",
    ]
    assert gate["metadata"]["join_c1"]["target_col"] is None

    selected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "使用 label_a 作为目标列"},
    )
    assert selected.status_code == 202, selected.text
    # Selecting the target completes setup and produces the manual-mode plan;
    # executing that plan remains a separate explicit manual action.
    started = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始"},
    )
    assert started.status_code == 202, started.text
    bin_gate = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    assert bin_gate["metadata"]["feature_binning"]
    task = client.get(f"/api/tasks/{task_id}").json()
    assert task["target_col"] == "label_a"
    done = _finish_optional_binning(client, task_id)
    assert "特征分析完成" in done["content"]


def test_feature_analysis_honors_explicit_target_without_clarification(
    client: TestClient,
    tmp_path: Path,
):
    src = _ambiguous_target_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "显式目标列",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
        "target_col": "label_b",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})

    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    assert not any(
        (message.get("metadata") or {}).get("feature_target_choice")
        for message in messages
    )
    assert _last_assistant(messages)["metadata"]["feature_binning"]


def test_feature_analysis_explicit_empty_metrics_stays_empty(
    client: TestClient,
    tmp_path: Path,
):
    src = _sample_dir(tmp_path, n=300)
    task_id = client.post("/api/tasks", json={
        "model_name": "明确不选指标",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
        "metrics": [],
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    done = _finish_optional_binning(client, task_id)

    metrics = next(
        table for table in done["metadata"]["tables"]
        if table["title"] == "特征指标"
    )
    assert metrics["columns"] == ["特征"]


def test_feature_renderer_only_shows_selected_metrics_and_formats_psi_reasons():
    _text, tables = _render_feature_metrics({
        "selected_metrics": ["ks", "psi_month_first", "psi_split"],
        "metrics": [
            {
                "feature": "x1",
                "ks": 0.31,
                "psi_month_first": None,
                "psi_month_first_reason": {
                    "code": "missing_dependency",
                    "message": "未识别到时间列。",
                },
                "psi_split": 0.12,
                "psi_split_series": [
                    {"base": "train", "compare": "test", "psi": 0.08},
                    {"base": "train", "compare": "oot", "psi": 0.12},
                ],
                "recommendation": "候选",
                "recommendation_reason": "KS 有信号。",
            }
        ],
    })

    by_title = {table["title"]: table for table in tables}
    assert by_title["特征指标"]["columns"] == ["特征", "KS"]
    psi = by_title["PSI 稳定性"]
    assert psi["columns"] == [
        "特征",
        "月度PSI(首月基准)",
        "月度PSI(首月基准)说明",
        "样本集PSI",
        "样本集PSI说明",
    ]
    assert psi["rows"][0][2] == "未识别到时间列。"
    assert by_title["PSI 明细"]["rows"][1] == [
        "x1",
        "样本集PSI",
        "train",
        "oot",
        "0.1200",
    ]
    for absent in ("数据质量", "头尾 Lift（风险方向）", "含义方向一致性"):
        assert absent not in by_title


def test_feature_binning_gate_accepts_natural_language_selection(client: TestClient, tmp_path: Path):
    src = _sample_dir(tmp_path, n=300)
    task_id = client.post("/api/tasks", json={
        "model_name": "自然语言分箱",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        # The same PlanDriver text adapter is used by Agent mode after routing;
        # manual mode keeps this test independent of an external LLM config.
        "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "请对 sig1 和 sig3 做 6 箱分箱分析"},
    )
    assert response.status_code == 202, response.text
    done = _last_assistant(
        client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    )
    bin_table = next(table for table in done["metadata"]["tables"] if table["title"] == "分箱分析")
    assert {row[0] for row in bin_table["rows"]} == {"sig1", "sig3"}
    assert {row[1] for row in bin_table["rows"]} == {6}


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
    done = _finish_optional_binning(client, task_id)
    titles = {t["title"] for t in done["metadata"].get("tables", [])}
    assert "特征指标" in titles  # base metrics still present
    assert "VIF（共线性）" in titles  # the selected optional metric was computed + shown


@pytest.mark.slow
def test_feature_analysis_multiple_files_runs_join_then_feature_analysis(client: TestClient, tmp_path: Path):
    src = _sample_dir(tmp_path, n=200)
    pd.DataFrame({
        "cust_id": np.arange(200),
        "usedate": 20260101 + np.arange(200),
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
    assert "手动模式请点击「开始执行」" in overview["content"]
    assert "Agent 模式请回复「开始」或「继续」" in overview["content"]

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    join_gate = _last_assistant(messages)
    assert join_gate["metadata"].get("kind") == "gate"
    assert "拼接诊断完成" in join_gate["content"]

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _finish_optional_binning(client, task_id)
    assert "特征分析完成" in done["content"]
    table = next(t for t in done["metadata"]["tables"] if t["title"] == "特征指标")
    feature_names = {row[0] for row in table["rows"]}
    assert "external_score" in feature_names
    assert "usedate" not in feature_names

    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert plans and plans[-1].template_id == "feature_analysis_with_join"


def test_feature_analysis_plain_named_excel_is_included_in_join_inputs(
    client: TestClient, tmp_path: Path
):
    src = _sample_dir(tmp_path, n=40)
    pd.DataFrame({"cust_id": np.arange(40), "external_score": np.arange(40)}).to_parquet(
        src / "feature_table.parquet"
    )
    pd.DataFrame({"cust_id": np.arange(40), "bureau_score": np.arange(40)}).to_excel(
        src / "vars.xlsx", index=False
    )
    task_id = client.post("/api/tasks", json={
        "model_name": "三文件特征",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "feature_analysis",
        "run_mode": "manual",
    }).json()["id"]

    response = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert response.status_code == 202, response.text
    plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    diagnose = next(step for step in plan.steps if step.title == "拼接诊断")
    assert len(diagnose.inputs["feature_ids"]) == 2


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
    done = _finish_optional_binning(client, task_id)
    table = next(t for t in done["metadata"]["tables"] if t["title"] == "头尾 Lift（风险方向）")
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
    done = _finish_optional_binning(client, task_id)
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
    _finish_optional_binning(client, task_id)
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
    _finish_optional_binning(client, task_id)

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
