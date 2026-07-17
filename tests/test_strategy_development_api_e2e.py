"""Real product-entry E2E for governed strategy development and adoption.

This intentionally starts at ``POST /api/tasks`` instead of constructing a
PlanDriver or supplying template slots directly. It protects the route the user
actually takes through task creation, the Agent HTTP surface, mandatory gates,
evidence-bound adoption, and persisted deliverables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import PluginRepository
from marvis.orchestrator.contracts import PlanStatus, StepStatus
from marvis.repositories.strategy import StrategyRepository


def _strategy_source(tmp_path: Path) -> Path:
    source = tmp_path / "strategy-materials"
    source.mkdir(parents=True, exist_ok=True)
    # Higher score is safer. The first score band contains all three bads, so a
    # max-approval objective constrained to 5% approved bad rate has a stable,
    # non-trivial reject boundary.
    pd.DataFrame(
        {
            "score": list(range(100, 2100, 100)),
            "bad": [1, 1, 1, *([0] * 17)],
        }
    ).to_csv(source / "strategy.csv", index=False)
    return source


def _latest_message(payload: dict, *, kind: str) -> dict:
    return next(
        message
        for message in reversed(payload["messages"])
        if message.get("metadata", {}).get("kind") == kind
    )


def _confirm_gate(client: TestClient, task_id: str, gate: dict) -> dict:
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认",
            "expected_step_id": gate["metadata"]["step_id"],
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


@pytest.mark.slow
@pytest.mark.e2e
def test_strategy_development_product_entry_requires_evidence_bound_adoption_reason(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    source = _strategy_source(tmp_path)
    reason = "风险策略委员会批准在验证样本上采纳该 5% 坏率约束方案"

    created = client.post(
        "/api/tasks",
        json={
            "model_name": "额度准入策略开发",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {
                "entry_mode": "strategy_development",
                "objective": "max_approval",
                "max_bad_rate": 0.05,
            },
        },
    )
    assert created.status_code == 200, created.text
    task = created.json()
    task_id = task["id"]
    assert task["strategy_input"] == {
        "entry_mode": "strategy_development",
        "objective": "max_approval",
        "max_bad_rate": 0.05,
        "min_approval_rate": None,
        "baseline_strategy_id": None,
        "profit": None,
    }

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    overview = _latest_message(started.json(), kind="plan_overview")
    plan_id = overview["metadata"]["plan_id"]
    plans_response = client.get(f"/api/tasks/{task_id}/plans")
    assert plans_response.status_code == 200, plans_response.text
    plans = plans_response.json()["plans"]
    assert plans[-1]["id"] == plan_id
    assert plans[-1]["template_id"] == "strategy_development"

    began = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始"},
    )
    assert began.status_code == 202, began.text
    bands_gate = _latest_message(began.json(), kind="gate")
    assert bands_gate["metadata"]["gate_source_tool"] == "design_cutoff_bands"

    backtest_payload = _confirm_gate(client, task_id, bands_gate)
    backtest_gate = _latest_message(backtest_payload, kind="gate")
    assert backtest_gate["metadata"]["gate_source_tool"] == "backtest_strategy"

    adoption_payload = _confirm_gate(client, task_id, backtest_gate)
    adoption_gate = _latest_message(adoption_payload, kind="gate")
    assert adoption_gate["metadata"]["gate_source_tool"] == "adopt_strategy"
    adoption_step_id = adoption_gate["metadata"]["step_id"]
    editable_schema = adoption_gate["metadata"]["editable_input_schema"]
    assert editable_schema["required"] == ["adoption_reason"]
    assert editable_schema["additionalProperties"] is False
    reason_schema = editable_schema["properties"]["adoption_reason"]
    assert reason_schema["type"] == "string"
    assert reason_schema["title"] == "采纳理由"
    assert reason_schema["description"] == "说明基于当前策略与回测证据采纳该版本的业务理由。"
    assert reason_schema["minLength"] >= 2

    strategy_repo = StrategyRepository(client.app.state.settings.db_path)
    before_reject = strategy_repo.list_meta_for_task(task_id)
    assert len(before_reject) == 1
    assert before_reject[0]["status"] == "draft"
    assert before_reject[0]["adoption_reason"] is None

    # A bare confirmation cannot inherit a setup placeholder or silently adopt.
    rejected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认采纳",
            "expected_step_id": adoption_step_id,
        },
    )
    assert rejected.status_code == 409, rejected.text
    assert "采纳理由" in rejected.json()["detail"]

    rejected_plan = client.app.state.plan_repo.load_plan(plan_id)
    rejected_adopt_step = next(
        step for step in rejected_plan.steps if step.id == adoption_step_id
    )
    assert rejected_plan.status == PlanStatus.AWAITING_CONFIRM
    assert rejected_adopt_step.status == StepStatus.AWAITING_CONFIRM
    assert rejected_adopt_step.output_ref is None
    after_reject = strategy_repo.list_meta_for_task(task_id)
    assert [item["status"] for item in after_reject] == ["draft"]
    assert PluginRepository(client.app.state.settings.db_path).list_audit(
        kind="strategy.adopt"
    ) == []

    # The editable reason and confirmation travel in one request, bound to the
    # exact gate token. This request performs the adoption and finishes the doc.
    completed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认采纳",
            "expected_step_id": adoption_step_id,
            "adjust_params": {"adoption_reason": reason},
        },
    )
    assert completed.status_code == 202, completed.text
    assert any(
        "计划已全部完成" in message["content"]
        for message in completed.json()["messages"]
    )

    final_plan = client.app.state.plan_repo.load_plan(plan_id)
    assert final_plan.status == PlanStatus.DONE
    final_adopt_step = next(
        step for step in final_plan.steps if step.id == adoption_step_id
    )
    assert final_adopt_step.status == StepStatus.DONE
    assert final_adopt_step.inputs["adoption_reason"] == reason
    adopt_output = client.app.state.plan_repo.load_step_output(adoption_step_id)
    strategy_id = adopt_output["strategy_id"]

    meta = strategy_repo.get_strategy_meta(strategy_id)
    assert meta is not None
    assert meta["status"] == "adopted"
    assert meta["adoption_reason"] == reason

    adoption_audits = PluginRepository(
        client.app.state.settings.db_path
    ).list_audit(kind="strategy.adopt")
    assert len(adoption_audits) == 1
    assert adoption_audits[0]["target_ref"] == strategy_id
    assert adoption_audits[0]["outcome"] == "succeeded"
    assert adoption_audits[0]["detail"]["adoption_reason"] == reason

    artifacts = strategy_repo.list_strategy_artifacts(strategy_id)
    assert {artifact["kind"] for artifact in artifacts} == {
        "decision_table_csv",
        "monitoring_plan_json",
        "strategy_doc_md",
    }
    for artifact in artifacts:
        assert Path(artifact["path"]).is_file()

    doc_artifact = next(
        artifact for artifact in artifacts if artifact["kind"] == "strategy_doc_md"
    )
    doc_text = Path(doc_artifact["path"]).read_text(encoding="utf-8")
    assert reason in doc_text
    assert "已采纳" in doc_text
