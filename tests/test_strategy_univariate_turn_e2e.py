"""Focused NL -> Agent plan -> deterministic Candidate Lab smoke test."""

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


class _UnivariateLLM:
    def __init__(self, workflow_inputs: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.workflow_inputs = workflow_inputs

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_analysis",
                "workflow_inputs": self.workflow_inputs
                or {
                        "features": ["score", "segment"],
                        "bin_count": 3,
                        "min_bin_pct": 0.02,
                        "loan_amount_col": "loan_amount",
                        "overdue_amount_col": "overdue_amount",
                        "sentinel_values": ["UNKNOWN"],
                    },
            },
            ensure_ascii=False,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_univariate_candidate_analysis_runs_without_effect_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
            "segment": ["UNKNOWN", "A", "A", "B", "B", "C"] * 2,
            "loan_amount": [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
            "overdue_amount": [0, 0, 5, 0, 10, 0, 15, 0, 20, 0, 25, 30],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言单变量候选分析",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    sample_design_ref = materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    llm = _UnivariateLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "对 score 和 segment 做单变量分析"},
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans, opened.json()["messages"][-1]["content"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design",
        "strategy_univariate_candidate_analysis",
    ]
    downstream = plans[1]
    assert downstream["status"] == "done"
    assert len(downstream["steps"]) == 1
    assert downstream["steps"][0]["status"] == "done"
    assert downstream["steps"][0]["needs_confirmation"] is False

    stored = client.app.state.plan_repo.load_plan(downstream["id"])
    assert stored.steps[0].inputs["sample_design_ref"] == sample_design_ref
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["validation_status"] == "unvalidated"
    assert output["available_method_count"] == 5
    assert {artifact["kind"] for artifact in output["artifacts"]} == {
        "strategy_candidate_json",
        "strategy_candidate_xlsx",
    }
    listed = client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"]
    assert {artifact["id"] for artifact in listed} == {
        sample_design_ref["artifact_id"]
    } | {artifact["artifact_id"] for artifact in output["artifacts"]}
    assert all(
        client.get(artifact["download_url"]).status_code == 200 for artifact in listed
    )
    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "development / unvalidated" in assistant_text
    assert len(llm.calls) == 1


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_manual_cutpoints_reach_the_real_toolrunner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言手工切点",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    sample_design_ref = materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    llm = _UnivariateLLM(
        {
            "features": ["score"],
            "methods": ["manual"],
            "bin_count": 3,
            "min_bin_pct": 0.02,
            "sentinel_values": [],
            "manual_breakpoints": {"score": [200, 400]},
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 score 做单变量分析，score manual 切点 [200, 400]"
            )
        },
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design",
        "strategy_univariate_candidate_analysis",
    ]
    stored = client.app.state.plan_repo.load_plan(plans[1]["id"])
    assert stored.status == "done"
    assert stored.steps[0].inputs["sample_design_ref"] == sample_design_ref
    assert stored.steps[0].inputs["manual_breakpoints"] == {
        "score": [200.0, 400.0]
    }
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["schema_version"] == "strategy.univariate-candidate-tool.v2"
    assert output["candidate_evidence"]["analysis"]["parameters"][
        "manual_breakpoints"
    ] == {"score": [200.0, 400.0]}
    assert len(llm.calls) == 1
