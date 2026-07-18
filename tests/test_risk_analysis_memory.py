from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.agent.memory_bridge import capture_agent_memory_for_driver_done
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db
from marvis.domain import TASK_TYPE_VINTAGE
from marvis.memory_policy import MemoryPolicySettings, save_memory_policy
from marvis.settings import build_settings


def _risk_report(tmp_path: Path) -> dict:
    return {
        "analysis_kind": "vintage_and_profitability",
        "product_scope": ["白条消费", "白条取现"],
        "as_of_period": "2026-05-31",
        "headline_metrics": {
            "annualized_bad_rate": 0.0353,
            "net_yield": 0.000295,
        },
        "assumptions": ["DPD30+终值按长期回收调整", "年化周转按360天口径"],
        "key_points": ["生息资产风险成本高于免息资产"],
        "red_flags": [{"code": "thin_cohort", "message": "部分新放款月份观察期不足"}],
        "column_map": {
            "loan_month": "放款月",
            "loan_amount": "放款额(元)",
        },
        "report_path": str(tmp_path / "tasks" / "task-risk-1" / "risk-analysis.xlsx"),
    }


def _task(task_id: str = "task-risk-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=task_id,
        task_type=TASK_TYPE_VINTAGE,
        model_name="风险分析",
    )


def _capture(settings, task, report: dict) -> None:
    capture_agent_memory_for_driver_done(
        settings,
        task,
        done_message_content="风险分析报告已生成",
        done_message_metadata={"risk_analysis_report": report},
    )


def test_vintage_done_writes_governed_risk_analysis_memory(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)

    _capture(settings, _task(), _risk_report(tmp_path))

    entries = AgentMemoryStore(settings.db_path).list_entries(
        memory_type="risk_analysis_experience"
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.source_task_id == "task-risk-1"
    assert entry.payload == {
        "analysis_kind": "vintage_and_profitability",
        "source_task_id": "task-risk-1",
        "product_scope": ["白条消费", "白条取现"],
        "as_of_period": "2026-05-31",
        "headline_metrics": {
            "annualized_bad_rate": 0.0353,
            "net_yield": 0.000295,
        },
        "assumptions": ["DPD30+终值按长期回收调整", "年化周转按360天口径"],
        "key_points": ["生息资产风险成本高于免息资产"],
        "red_flags": ["thin_cohort"],
        "column_map": {
            "loan_month": "放款月",
            "loan_amount": "放款额(元)",
        },
        "report_file": "risk-analysis.xlsx",
    }
    assert "/" not in entry.payload["report_file"]


def test_vintage_done_respects_auto_distill_disable_gate(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    save_memory_policy(
        settings.workspace,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )

    _capture(settings, _task(), _risk_report(tmp_path))

    assert (
        AgentMemoryStore(settings.db_path).list_entries(
            memory_type="risk_analysis_experience"
        )
        == []
    )


@pytest.mark.parametrize("invalid_kind", ["raw_rows", "pii", "absolute_report_file"])
def test_risk_analysis_memory_rejects_sensitive_or_path_payloads(
    tmp_path: Path,
    invalid_kind: str,
):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    report = deepcopy(_risk_report(tmp_path))
    if invalid_kind == "raw_rows":
        report["raw_rows"] = [{"loan_id": "L-001", "dpd": 45}]
    elif invalid_kind == "pii":
        report["key_points"] = ["客户手机号: 13800138000"]
    else:
        report.pop("report_path")
        report["report_file"] = "/Users/example/uploads/risk-analysis.xlsx"

    _capture(settings, _task(), report)

    assert (
        AgentMemoryStore(settings.db_path).list_entries(
            memory_type="risk_analysis_experience"
        )
        == []
    )


def test_repeated_vintage_done_reuses_store_dedup_for_same_task_payload(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task()
    report = _risk_report(tmp_path)

    _capture(settings, task, report)
    _capture(settings, task, report)

    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="risk_analysis_experience")
    assert len(entries) == 1
    create_events = [
        event
        for event in store.list_events(entries[0].id)
        if event["event_type"] == "create"
    ]
    assert len(create_events) == 2
    assert create_events[-1]["details"]["dedup"] is True


def test_risk_analysis_memory_bounds_large_product_and_flag_lists(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    report = _risk_report(tmp_path)
    report["product_scope"] = [f"产品{i}" for i in range(12)]
    report["red_flags"] = [f"flag_{i}" for i in range(20)]
    report["headline_metrics"] = {f"metric_{i}": i for i in range(20)}

    _capture(settings, _task(), report)

    entry = AgentMemoryStore(settings.db_path).list_entries(
        memory_type="risk_analysis_experience"
    )[0]
    assert entry.payload["product_scope"] == [f"产品{i}" for i in range(8)]
    assert entry.payload["red_flags"] == [f"flag_{i}" for i in range(12)]
    assert list(entry.payload["headline_metrics"]) == [f"metric_{i}" for i in range(16)]


def test_profitability_memory_keeps_bounded_headline_when_optional_scenario_metrics_absent(
    tmp_path: Path,
):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    report = _risk_report(tmp_path)
    report["analysis_kind"] = "profitability"
    report["headline_metrics"] = {
        "product_count": 1,
        "analysis_slice_count": 1,
        "negative_product_count": 0,
        "lowest_net_yield": 0.01,
        "lowest_net_yield_product": "白条消费",
        "lowest_net_yield_as_of_period": "2025-12-04",
        "lowest_net_yield_scenario": "基准",
        "highest_net_yield": 0.01,
        "highest_net_yield_product": "白条消费",
        "highest_net_yield_as_of_period": "2025-12-04",
        "highest_net_yield_scenario": "基准",
        "max_cost_rate": 0.04,
        "max_cost_component": "risk_cost_rate",
        "max_cost_product": "白条消费",
        "max_cost_as_of_period": "2025-12-04",
        "max_cost_scenario": "基准",
        "largest_scenario_net_yield_spread": None,
        "largest_scenario_net_yield_spread_product": None,
    }

    _capture(settings, _task(), report)

    entries = AgentMemoryStore(settings.db_path).list_entries(
        memory_type="risk_analysis_experience"
    )
    assert len(entries) == 1
    assert len(entries[0].payload["headline_metrics"]) == 16
    assert (
        "largest_scenario_net_yield_spread"
        not in entries[0].payload["headline_metrics"]
    )


def test_risk_analysis_memory_truncates_long_safe_summary_text_instead_of_dropping_it(
    tmp_path: Path,
):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    report = _risk_report(tmp_path)
    report["product_scope"] = ["产品" * 60]
    report["assumptions"] = ["口径" * 150]
    report["key_points"] = ["发现" * 180]
    report["red_flags"] = ["风险" * 80]
    report["column_map"] = {"canonical": "来源字段" * 80}
    report["headline_metrics"]["highest_annualized_product"] = "产品" * 100

    _capture(settings, _task(), report)

    entry = AgentMemoryStore(settings.db_path).list_entries(
        memory_type="risk_analysis_experience"
    )[0]
    assert len(entry.payload["product_scope"][0]) == 80
    assert len(entry.payload["assumptions"][0]) == 200
    assert len(entry.payload["key_points"][0]) == 240
    assert len(entry.payload["red_flags"][0]) == 80
    assert len(entry.payload["column_map"]["canonical"]) == 120
    assert len(entry.payload["headline_metrics"]["highest_annualized_product"]) == 120


def test_risk_analysis_memory_has_minimal_distillation_support(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    _capture(settings, _task(), _risk_report(tmp_path))

    distilled = DistillationEngine(AgentMemoryStore(settings.db_path)).distill_category(
        "risk_analysis_experience"
    )

    assert len(distilled) == 1
    assert distilled[0].category == "risk_analysis_experience"
    assert distilled[0].structured["headline_metrics"]["annualized_bad_rate"] == 0.0353
    assert distilled[0].structured["red_flags"] == ["thin_cohort"]
