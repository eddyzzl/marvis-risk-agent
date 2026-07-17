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

from marvis.agent.strategy_setup import resolve_strategy_intent
from marvis.agent.turn_handlers import _strategy_success_criteria
from marvis.app import create_app
from marvis.domain import TASK_TYPE_STRATEGY, TaskRecord, TaskStatus


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _strategy_source(tmp_path: Path) -> Path:
    src = tmp_path / "strategy"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "bad": [1, 0, 0, 0, 1, 0],
        "score": [580, 620, 730, 760, 590, 800],
    }).to_csv(src / "strategy.csv", index=False)
    return src


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("额度准入策略", "full_development"),
        ("快速策略分析", "quick_analysis"),
        ("规则挖掘后快速策略分析", "rule_mining"),
        ("策略监控并检查规则策略", "monitoring"),
        ("额度定价", "limit_pricing"),
        ("定价矩阵", "limit_pricing"),
        ("limit pricing", "limit_pricing"),
        ("组合分析", "portfolio_analysis"),
    ],
)
def test_strategy_intent_taxonomy_and_priority(goal, expected):
    assert resolve_strategy_intent(None, goal) == expected


def test_strategy_agent_start_defaults_to_full_development_plan(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "额度准入策略",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {
                "objective": "max_approval",
                "max_bad_rate": 0.20,
            },
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "ok"
    assert any(
        message.get("metadata", {}).get("intent") == "full_development"
        for message in started.json()["messages"]
    )
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[-1]["template_id"] == "strategy_development"
    assert plans[-1]["success_criteria"] == [
        {"metric": "approved_bad_rate", "max": 0.20}
    ]
    assert [step["title"] for step in plans[-1]["steps"]] == [
        "权衡扫描",
        "设计分数带",
        "构造策略",
        "回测策略",
        "对比基线",
        "挑战者报告",
        "采纳策略",
        "策略文档",
    ]


def test_strategy_limit_pricing_intent_redirects_without_approval_plan(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "额度准入策略",
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

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "请做额度定价矩阵"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "clarification_required"
    assert payload["intent"] == "limit_pricing"
    assert payload["code"] == "strategy_limit_pricing_workflow_planned"
    assert payload["planned_v2_workflow"] == "limit_pricing_matrix"
    clarification = payload["clarification"]
    assert clarification["intent"] == "limit_pricing"
    assert clarification["planned_v2_workflow"] == "limit_pricing_matrix"
    metadata = payload["messages"][-1]["metadata"]
    assert metadata["intent"] == "limit_pricing"
    assert metadata["code"] == "strategy_limit_pricing_workflow_planned"
    assert metadata["planned_v2_workflow"] == "limit_pricing_matrix"
    assert "V2" in payload["messages"][-1]["content"]
    assert "V3" not in payload["messages"][-1]["content"]
    assert "V4" not in payload["messages"][-1]["content"]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_strategy_portfolio_intent_redirects_to_portfolio_task_without_plan(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "额度准入策略",
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

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始组合分析"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "clarification_required"
    assert payload["intent"] == "portfolio_analysis"
    assert payload["code"] == "strategy_portfolio_task_redirect"
    assert payload["suggested_task_type"] == "portfolio"
    clarification = payload["clarification"]
    assert clarification["intent"] == "portfolio_analysis"
    assert clarification["suggested_task_type"] == "portfolio"
    metadata = payload["messages"][-1]["metadata"]
    assert metadata["intent"] == "portfolio_analysis"
    assert metadata["code"] == "strategy_portfolio_task_redirect"
    assert metadata["suggested_task_type"] == "portfolio"
    assert "V2" in payload["messages"][-1]["content"]
    assert "V3" not in payload["messages"][-1]["content"]
    assert "V4" not in payload["messages"][-1]["content"]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_strategy_development_without_business_constraints_clarifies_without_plan(
    client, tmp_path
):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "缺经营口径的策略",
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
    assert started.json()["status"] == "clarification_required"
    clarification = started.json()["clarification"]
    assert clarification["code"] == "strategy_business_inputs_required"
    assert clarification["current_input"] is None
    assert started.json()["messages"][-1]["metadata"]["current_input"] is None
    assert set(clarification["missing_fields"]) == {
        "objective",
        "max_bad_rate_or_min_approval_rate",
    }
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_strategy_clarification_preserves_partial_business_contract(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "已有目标待补约束的策略",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {"objective": "max_approval"},
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "clarification_required"
    expected = {
        "entry_mode": "strategy_development",
        "objective": "max_approval",
        "max_bad_rate": None,
        "min_approval_rate": None,
        "baseline_strategy_id": None,
        "profit": None,
    }
    clarification = started.json()["clarification"]
    assert clarification["missing_fields"] == ["max_bad_rate_or_min_approval_rate"]
    assert clarification["current_input"] == expected
    metadata = started.json()["messages"][-1]["metadata"]
    assert metadata["current_input"] == expected
    assert metadata["clarification"]["current_input"] == expected
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_max_profit_strategy_requires_complete_profit_contract(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "利润目标策略",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {
                "objective": "max_profit",
                "max_bad_rate": 0.20,
            },
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert started.status_code == 202, started.text
    assert started.json()["status"] == "clarification_required"
    missing = set(started.json()["clarification"]["missing_fields"])
    assert missing == {
        "profit.ead_col",
        "profit.pd_col",
        "profit.annual_rate",
        "profit.funding_rate",
        "profit.lgd",
        "profit.operating_cost_per_loan",
        "profit.term_months",
    }
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_explicit_quick_strategy_analysis_keeps_lightweight_workflow(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "兼容轻量策略",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {"entry_mode": "strategy_analysis"},
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    start_messages = started.json()["messages"]
    assert any("开始策略分析" in message["content"] for message in start_messages)
    assert start_messages[-1]["metadata"]["kind"] == "plan_overview"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[-1]["template_id"] == "strategy_analysis"

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


def test_quick_strategy_analysis_phrase_overrides_development_default(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "临时策略预览",
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

    started = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "快速策略分析"},
    )

    assert started.status_code == 202, started.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[-1]["template_id"] == "strategy_analysis"


def test_start_strategy_development_message_uses_full_product_route(client, tmp_path):
    src = _strategy_source(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "对话式策略任务",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {
                "objective": "max_approval",
                "min_approval_rate": 0.60,
            },
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始策略开发"},
    )

    assert started.status_code == 202, started.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[-1]["template_id"] == "strategy_development"
    assert plans[-1]["success_criteria"] == [
        {"metric": "approval_rate", "min": 0.60}
    ]


def test_strategy_rule_mining_goal_routes_to_rule_strategy_template(client, tmp_path):
    src = tmp_path / "rules"
    src.mkdir(parents=True, exist_ok=True)
    # bad concentrated where f1 is low -> mining returns candidate reject rules.
    pd.DataFrame({
        "f1":  list(range(10, 210, 10)),
        "f2":  [i % 3 for i in range(20)],
        "bad": [1 if v <= 60 else 0 for v in range(10, 210, 10)],
    }).to_csv(src / "rules.csv", index=False)
    created = client.post(
        "/api/tasks",
        json={
            # a rule-mining goal in the task name routes strategy setup to the
            # rule_strategy template instead of the default strategy_analysis.
            "model_name": "规则挖掘",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    assert any("开始规则策略挖掘" in message["content"] for message in started.json()["messages"])

    confirmed = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert confirmed.status_code == 202, confirmed.text
    gate = confirmed.json()["messages"][-1]
    assert gate["metadata"]["kind"] == "gate"
    assert "规则挖掘完成" in gate["content"]
    assert any(
        table["title"] == "候选规则（按 lift 降序）"
        for table in gate["metadata"]["tables"]
    )


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
    # A1: the vintage kernel always accumulates the bad column across MOBs; on a
    # snapshot/ever-bad flag that silently double-counts. The strategy vintage tool now
    # refuses to guess the cumulation basis, so an undeclared label_semantics halts at a
    # gate (mirrors the NaN-label gate) offering both concrete semantics to the user.
    assert "label_semantics" in done["content"]
    assert "incremental" in done["content"] and "snapshot" in done["content"]
    envelope = done["metadata"].get("failure_envelope") or {}
    assert envelope.get("error_kind") == "label_semantics_not_declared"


def _bare_strategy_task(**overrides) -> TaskRecord:
    base = dict(
        id="task-strategy-sc",
        model_name="策略",
        model_version="v1",
        validator="qa",
        source_dir="",
        algorithm="",
        run_mode="agent",
        target_col="bad",
        score_col="score",
        split_col="",
        time_col="",
        feature_columns=[],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.SCANNED,
        status_message="",
        created_at="",
        updated_at="",
        task_type=TASK_TYPE_STRATEGY,
    )
    base.update(overrides)
    return TaskRecord(**base)


def test_strategy_success_criteria_none_when_task_has_no_optional_fields():
    # Mirrors _modeling_success_criteria's oot_ks_min default: absent optional
    # fields inject no criterion at all (never a hard-coded platform default).
    assert _strategy_success_criteria(_bare_strategy_task()) is None


def test_strategy_success_criteria_builds_bad_rate_max_and_approval_min():
    task = _bare_strategy_task()
    object.__setattr__(task, "strategy_bad_rate_max", 0.05)
    object.__setattr__(task, "strategy_approval_min", 0.6)

    criteria = _strategy_success_criteria(task)

    assert criteria == [
        {"metric": "approved_bad_rate", "max": 0.05},
        {"metric": "approval_rate", "min": 0.6},
    ]


def test_strategy_success_criteria_builds_only_the_field_that_is_set():
    task = _bare_strategy_task()
    object.__setattr__(task, "strategy_bad_rate_max", 0.1)

    criteria = _strategy_success_criteria(task)

    assert criteria == [{"metric": "approved_bad_rate", "max": 0.1}]
