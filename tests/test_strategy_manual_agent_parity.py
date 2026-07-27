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
    def __init__(
        self,
        workflow_inputs: dict,
        *,
        workflow: str = "univariate_candidate_analysis",
    ) -> None:
        self.workflow_inputs = workflow_inputs
        self.workflow = workflow
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": self.workflow,
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


@pytest.mark.slow
@pytest.mark.e2e
def test_manual_and_natural_language_refinement_share_the_same_cutpoint_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [300 + index * 20 for index in range(24)],
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
            "model_name": "Candidate refinement manual NL parity",
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
        "feature": "score",
        "method": "manual",
        "manual_breakpoints": {"score": [420, 600]},
        "selection": {"risk_threshold": {"operator": ">=", "value": 0.5}},
    }
    llm = _ParityLLM(
        workflow_inputs,
        workflow="univariate_candidate_refinement",
    )
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
            "content": "人工界面按指定切点重跑并选择风险箱",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_refinement",
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
                "对 score 重新分析，score manual 切点 [420, 600]，"
                "并选择坏率大于等于 50% 的候选箱"
            )
        },
    )
    assert natural.status_code == 202, natural.text
    assert len(llm.calls) == 1

    expected_draft = {
        "request_kind": "standard_workflow",
        "workflow": "univariate_candidate_refinement",
        "workflow_inputs": {
            **workflow_inputs,
            "features": ["score"],
            "methods": ["manual"],
            "bin_count": 10,
            "min_bin_pct": 0.02,
            "sentinel_values": [],
            "manual_breakpoints": {"score": [420.0, 600.0]},
            "merge_groups": [],
        },
    }
    assert canonical_drafts == [expected_draft, expected_draft]

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    refinement_plans = [
        plan
        for plan in plans
        if plan["template_id"] == "strategy_univariate_candidate_refinement"
    ]
    assert len(refinement_plans) == 2
    assert all(plan["status"] == "done" for plan in refinement_plans)
    manual_plan = client.app.state.plan_repo.load_plan(refinement_plans[0]["id"])
    natural_plan = client.app.state.plan_repo.load_plan(refinement_plans[1]["id"])
    assert [step.tool_ref for step in manual_plan.steps] == [
        step.tool_ref for step in natural_plan.steps
    ]

    def normalize_plan_refs(value):
        if isinstance(value, dict):
            return {key: normalize_plan_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize_plan_refs(item) for item in value]
        if isinstance(value, str) and value.startswith("$ref:") and ".output" in value:
            return "$ref:<step>" + value[value.index(".output") :]
        return value

    assert [normalize_plan_refs(step.inputs) for step in manual_plan.steps] == [
        normalize_plan_refs(step.inputs) for step in natural_plan.steps
    ]
    manual_output = client.app.state.plan_repo.load_step_output(
        manual_plan.steps[-1].id
    )
    natural_output = client.app.state.plan_repo.load_step_output(
        natural_plan.steps[-1].id
    )
    assert manual_output["asset_id"] == natural_output["asset_id"]
    assert manual_output["asset_hash"] == natural_output["asset_hash"]
    assert {
        artifact["content_hash"] for artifact in manual_output["artifacts"]
    } == {
        artifact["content_hash"] for artifact in natural_output["artifacts"]
    }

    source_output = client.app.state.plan_repo.load_step_output(
        manual_plan.steps[0].id
    )
    candidate_id = source_output["candidate_id"]
    existing_inputs = {
        "feature": "score",
        "method": "manual",
        "source_candidate_id": candidate_id,
        "merge_groups": [["regular:0", "regular:1"]],
        "selection": {"source_bin_ids": ["regular:0", "regular:1"]},
        "selection_reason": "合并并保留已核验的两个风险箱",
    }
    existing_llm = _ParityLLM(
        existing_inputs,
        workflow="univariate_candidate_refinement",
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: existing_llm,
    )

    def unavailable_current_dataset(*args, **kwargs):
        del args, kwargs
        raise turn_handlers.StrategySetupError(
            "existing refinement must not require the current DataWorkspace"
        )

    monkeypatch.setattr(
        turn_handlers,
        "_strategy_dataset_preview",
        unavailable_current_dataset,
    )
    monkeypatch.setattr(
        turn_handlers,
        "_strategy_dataset_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "existing refinement must resolve its immutable source artifact"
            )
        ),
    )

    existing_manual = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "人工界面选择并合并已有候选箱",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "univariate_candidate_refinement",
                "workflow_inputs": existing_inputs,
            },
        },
    )
    assert existing_manual.status_code == 202, existing_manual.text
    assert existing_llm.calls == []

    existing_natural = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"在 {candidate_id} 中合并 regular:0 和 regular:1，"
                "并选择 regular:0 和 regular:1 的候选箱"
            )
        },
    )
    assert existing_natural.status_code == 202, existing_natural.text
    assert len(existing_llm.calls) == 1
    assert canonical_drafts[-2:] == [
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": existing_inputs,
        },
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": existing_inputs,
        },
    ]

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    existing_plans = [
        plan
        for plan in plans
        if plan["template_id"]
        == "strategy_univariate_candidate_refinement_existing"
    ]
    assert len(existing_plans) == 2
    assert all(plan["status"] == "done" for plan in existing_plans)
    stored_existing = [
        client.app.state.plan_repo.load_plan(plan["id"])
        for plan in existing_plans
    ]
    assert [step.tool_ref for step in stored_existing[0].steps] == [
        step.tool_ref for step in stored_existing[1].steps
    ]
    assert [step.inputs for step in stored_existing[0].steps] == [
        step.inputs for step in stored_existing[1].steps
    ]
    existing_outputs = [
        client.app.state.plan_repo.load_step_output(plan.steps[-1].id)
        for plan in stored_existing
    ]
    assert existing_outputs[0]["asset_id"] == existing_outputs[1]["asset_id"]
    assert existing_outputs[0]["asset_hash"] == existing_outputs[1]["asset_hash"]

    invalid_inputs = {
        "feature": "score",
        "method": "manual",
        "source_candidate_id": candidate_id,
        "selection": {"source_bin_ids": ["regular:999"]},
    }
    invalid_llm = _ParityLLM(
        invalid_inputs,
        workflow="univariate_candidate_refinement",
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: invalid_llm,
    )
    plans_before = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    artifacts_before = client.get(
        f"/api/tasks/{task_id}/task-artifacts"
    ).json()["artifacts"]
    invalid_natural = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"在 {candidate_id} 中选择 regular:999 的候选箱"
            )
        },
    )
    assert invalid_natural.status_code == 202, invalid_natural.text
    assert invalid_natural.json()["status"] == "clarification_required"
    assert len(invalid_llm.calls) == 1
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == plans_before
    assert client.get(f"/api/tasks/{task_id}/task-artifacts").json()[
        "artifacts"
    ] == artifacts_before
