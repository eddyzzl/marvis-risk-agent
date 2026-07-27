"""Slow smoke coverage for NL -> governed standard strategy analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


class _WorkflowLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _install_llm(monkeypatch, llm: _WorkflowLLM) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


def _task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "customer_id": ["a", "a", "b", "b", "c", "c"],
            "month": [
                "2026-01",
                "2026-02",
                "2026-01",
                "2026-02",
                "2026-01",
                "2026-02",
            ],
            "status": ["M0", "M1", "M0", "M0", "M1", "M2+"],
            "segment": ["new", "new", "old", "old", "new", "new"],
            "balance": [1000, 900, 2000, 1900, 800, 700],
            "score": [780, 760, 720, 700, 660, 620],
            "ead": [1000, 900, 2000, 1900, 800, 700],
            "pd": [0.02, 0.04, 0.08, 0.10, 0.16, 0.22],
            "bad": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "标准策略分析 E2E",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.parametrize(
    ("utterance", "payload", "template_id", "artifact_kinds"),
    [
        (
            "按客群计算利润和 ROA",
            {
                "request_kind": "standard_workflow",
                "workflow": "profit_calc",
                "workflow_inputs": {
                    "segment_col": "segment",
                    "ead_col": "ead",
                    "pd_col": "pd",
                    "profit_params": {
                        "annual_rate": 0.18,
                        "funding_rate": 0.04,
                        "lgd": 0.5,
                        "operating_cost_per_loan": 10,
                        "term_months": 12,
                    },
                },
            },
            "strategy_profit_analysis",
            {"profit_csv", "profit_markdown"},
        ),
        (
            "计算客户状态滚动率矩阵",
            {
                "request_kind": "standard_workflow",
                "workflow": "roll_rate_matrix",
                "workflow_inputs": {
                    "id_col": "customer_id",
                    "time_col": "month",
                    "status_col": "status",
                    "states": ["M0", "M1", "M2+"],
                    "balance_col": "balance",
                    "observation_semantics": "adjacent_observation",
                },
            },
            "strategy_roll_rate_analysis",
            {"roll_rate_csv", "roll_rate_markdown"},
        ),
        (
            "测算额度利率定价矩阵",
            {
                "request_kind": "standard_workflow",
                "workflow": "limit_pricing_matrix",
                "workflow_inputs": {
                    "score_col": "score",
                    "pd_col": "pd",
                    "n_bands": 2,
                    "limit_grid": [1000, 2000],
                    "rate_grid": [0.12, 0.18],
                    "lgd": 0.5,
                    "funding_rate": 0.04,
                    "term_months": 12,
                    "cost_per_loan": 10,
                    "el_ead_max": 0.2,
                },
            },
            "strategy_limit_pricing_analysis",
            {"limit_pricing_csv", "limit_pricing_markdown"},
        ),
    ],
)
def test_natural_language_standard_workflow_executes_and_exports(
    tmp_path: Path,
    monkeypatch,
    utterance: str,
    payload: dict,
    template_id: str,
    artifact_kinds: set[str],
) -> None:
    client = TestClient(create_app(tmp_path / template_id))
    task_id = _task(client, tmp_path / template_id)
    sample_design_ref = None
    if template_id == "strategy_limit_pricing_analysis":
        sample_design_ref = materialize_mature_strategy_sample_design(
            client,
            task_id,
            monkeypatch,
        )
    llm = _WorkflowLLM(payload)
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": utterance},
    )
    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    plan = next(item for item in plans if item["template_id"] == template_id)
    assert plan["template_id"] == template_id
    assert plan["status"] == "done"
    assert all(step["status"] == "done" for step in plan["steps"])

    stored_plan = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored_plan.status.value == "done"
    if sample_design_ref is not None:
        assert all(
            step.inputs["sample_design_ref"] == sample_design_ref
            for step in stored_plan.steps
        )
    output = client.app.state.plan_repo.load_step_output(stored_plan.steps[-1].id)
    artifacts = output["artifacts"]
    assert {artifact["kind"] for artifact in artifacts} == artifact_kinds
    assert all(artifact.get("artifact_id") for artifact in artifacts)
    assert all("path" not in artifact for artifact in artifacts)
    listed = client.get(f"/api/tasks/{task_id}/task-artifacts")
    assert listed.status_code == 200, listed.text
    registered = listed.json()["artifacts"]
    output_ids = {artifact["artifact_id"] for artifact in artifacts}
    registered_outputs = [
        artifact for artifact in registered if artifact["id"] in output_ids
    ]
    assert {artifact["kind"] for artifact in registered_outputs} == artifact_kinds
    assert {artifact["id"] for artifact in registered_outputs} == output_ids
    assert str(client.app.state.settings.tasks_dir) not in listed.text
    for artifact in registered:
        downloaded = client.get(artifact["download_url"])
        assert downloaded.status_code == 200, downloaded.text
    assert len(llm.calls) == 1
