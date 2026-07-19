"""Focused NL -> evidence -> immutable candidate asset smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


class _RefinementLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_refinement",
                "workflow_inputs": {
                    "feature": "score",
                    "method": "equal_width",
                    "bin_count": 3,
                    "min_bin_pct": 0.02,
                    "loan_amount_col": "loan_amount",
                    "overdue_amount_col": "overdue_amount",
                    "selection": {"risk_threshold": {"operator": ">=", "value": 0.5}},
                    "selection_reason": "保留观测坏率达到 50% 的风险箱",
                },
            },
            ensure_ascii=False,
        )


def _strategy_task(tmp_path: Path) -> tuple[TestClient, str]:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
            "loan_amount": [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
            "overdue_amount": [0, 0, 5, 0, 10, 0, 15, 0, 20, 0, 25, 30],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言候选选择与合并",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert created.status_code == 200, created.text
    return client, created.json()["id"]


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_candidate_refinement_emits_downloadable_unvalidated_asset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, task_id = _strategy_task(tmp_path)
    llm = _RefinementLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "选择 score 等距分析中观测坏率大于等于 50% 的候选箱"},
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_univariate_candidate_refinement"
    ]
    assert plans[0]["status"] == "done"
    assert [step["status"] for step in plans[0]["steps"]] == ["done", "done"]
    assert all(step["needs_confirmation"] is False for step in plans[0]["steps"])

    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    analysis_output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    output = client.app.state.plan_repo.load_step_output(stored.steps[1].id)
    assert output["validation_status"] == "unvalidated"
    assert output["effect_stage"] == "development"
    assert output["parent_candidate_id"] == analysis_output["candidate_id"]
    assert output["parent_evidence_hash"] == analysis_output["evidence_hash"]
    assert output["feature"] == "score"
    assert output["method"] == "equal_width"
    assert output["asset_id"]
    assert output["asset_hash"]
    assert output["effect_id"]
    assert output["candidate_asset"]["rule"]["condition"]["op"] in {
        "compare",
        "between",
        "or",
    }
    assert [artifact["kind"] for artifact in output["artifacts"]] == [
        "strategy_candidate_asset_json"
    ]
    listed = client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"]
    assert {artifact["id"] for artifact in listed} == {
        artifact["artifact_id"]
        for artifact in analysis_output["artifacts"] + output["artifacts"]
    }
    assert client.get(output["artifacts"][0]["download_url"]).status_code == 200
    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "候选选择与合并完成" in assistant_text
    assert "development / unvalidated" in assistant_text
    assert len(llm.calls) == 1


@pytest.mark.slow
@pytest.mark.e2e
def test_explicit_source_bins_consume_the_exact_candidate_the_user_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client, task_id = _strategy_task(tmp_path)
    analysis_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {
                "features": ["score"],
                "methods": ["equal_width"],
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: analysis_llm,
    )
    analyzed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "分析 score 的等距单变量效果"},
    )
    assert analyzed.status_code == 202, analyzed.text
    first_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[0]
    analysis_output = client.app.state.plan_repo.load_step_output(
        first_plan.steps[0].id
    )
    candidate_id = analysis_output["candidate_id"]

    refinement_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": {
                "feature": "score",
                "method": "equal_width",
                "source_candidate_id": candidate_id,
                "merge_groups": [["regular:1", "regular:2"]],
                "selection": {"source_bin_ids": ["regular:1", "regular:2"]},
                "selection_reason": "合并并保留高风险两个箱",
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: refinement_llm,
    )
    refined = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"在 {candidate_id} 中合并 regular:1 和 regular:2，"
                "并选择 regular:1 和 regular:2 的候选箱"
            )
        },
    )

    assert refined.status_code == 202, refined.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_univariate_candidate_analysis",
        "strategy_univariate_candidate_refinement_existing",
    ]
    assert len(plans[1]["steps"]) == 1
    assert plans[1]["steps"][0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[1]["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["parent_candidate_id"] == candidate_id
    assert output["parent_evidence_hash"] == analysis_output["evidence_hash"]
    assert output["candidate_asset"]["refinement"]["edited_bin_count"] == 2
    assert output["candidate_asset"]["selection"]["source_bin_ids"] == [
        "regular:1",
        "regular:2",
    ]
    assert len(analysis_llm.calls) == 1
    assert len(refinement_llm.calls) == 1
