"""Manual controls and natural language share one Candidate Lab kernel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent import turn_handlers
from marvis.app import create_app
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


class _ParityLLM:
    def __init__(self, workflow_inputs: dict) -> None:
        self.workflow_inputs = workflow_inputs
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_analysis",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_manual_and_natural_language_requests_have_canonical_execution_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [300 + index * 20 for index in range(24)],
            "age": [20 + index * 2 for index in range(24)],
            "loan_amount": [1000.0 + index * 50 for index in range(24)],
            "overdue_amount": [
                50.0 + index if index < 12 else 0.0 for index in range(24)
            ],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "Candidate Lab manual NL parity",
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
    materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )

    workflow_inputs = {
        "features": ["score", "age"],
        "methods": ["equal_width"],
        "bin_count": 3,
        "min_bin_pct": 0.02,
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "sentinel_values": [],
    }
    llm = _ParityLLM(workflow_inputs)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    original_prepare = turn_handlers._prepare_and_run_validated_strategy_request
    canonical_drafts: list[dict] = []

    def capture_prepare(runtime, repo, task, draft, **kwargs):
        canonical_drafts.append(draft.to_dict())
        return original_prepare(runtime, repo, task, draft, **kwargs)

    monkeypatch.setattr(
        turn_handlers,
        "_prepare_and_run_validated_strategy_request",
        capture_prepare,
    )

    manual = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "人工界面执行 score 和 age 单变量分析",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_analysis",
                "workflow_inputs": workflow_inputs,
            },
        },
    )
    assert manual.status_code == 202, manual.text
    assert llm.calls == []

    natural = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "对 score 和 age 做单变量分析，两个字段都用等距分箱，3 箱；"
                "放款金额列 loan_amount，逾期金额列 overdue_amount。"
            )
        },
    )
    assert natural.status_code == 202, natural.text
    assert len(llm.calls) == 1

    assert canonical_drafts == [
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": workflow_inputs,
        },
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": workflow_inputs,
        },
    ]
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    candidate_plans = [
        plan
        for plan in plans
        if plan["template_id"] == "strategy_univariate_candidate_analysis"
    ]
    assert len(candidate_plans) == 2
    assert all(plan["status"] == "done" for plan in candidate_plans)
    manual_plan = client.app.state.plan_repo.load_plan(candidate_plans[0]["id"])
    natural_plan = client.app.state.plan_repo.load_plan(candidate_plans[1]["id"])
    assert manual_plan.template_id == natural_plan.template_id
    assert manual_plan.steps[0].tool_ref == natural_plan.steps[0].tool_ref
    assert manual_plan.steps[0].inputs == natural_plan.steps[0].inputs
    manual_output = client.app.state.plan_repo.load_step_output(
        manual_plan.steps[0].id
    )
    natural_output = client.app.state.plan_repo.load_step_output(
        natural_plan.steps[0].id
    )
    assert manual_output["candidate_id"] == natural_output["candidate_id"]
    assert manual_output["evidence_hash"] == natural_output["evidence_hash"]

    def artifact_hashes(output: dict) -> dict[tuple[str, str], str]:
        return {
            (artifact["kind"], Path(artifact["filename"]).suffix): artifact[
                "content_hash"
            ]
            for artifact in output["artifacts"]
        }

    assert artifact_hashes(manual_output) == artifact_hashes(natural_output)

    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    request_sources = [
        message["metadata"]["request_source"]
        for message in messages
        if message["role"] == "user"
        and (message.get("metadata") or {}).get("intent") == "strategy_request"
    ]
    assert request_sources == ["manual_ui", "agent_nl"]
