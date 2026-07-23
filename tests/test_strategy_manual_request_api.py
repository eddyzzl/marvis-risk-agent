"""Typed Candidate Lab requests stay LLM-free and governed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent import turn_handlers
from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.orchestrator.contracts import Plan, PlanStatus
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


class _BombLLM:
    def complete(self, **kwargs):  # pragma: no cover - failure is the assertion
        del kwargs
        raise AssertionError("typed strategy_request must not call an LLM")


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(24)],
            "age": [20 + index * 2 for index in range(24)],
            "score": [360 + index * 20 for index in range(24)],
            "income": [3000 + (index % 8) * 800 for index in range(24)],
            "loan_amount": [1000.0 + index * 50 for index in range(24)],
            "overdue_amount": [
                50.0 + index if index < 12 else 0.0 for index in range(24)
            ],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    ).to_csv(source / "sample.csv", index=False)
    return source


def _task(
    client: TestClient,
    tmp_path: Path,
    *,
    task_type: str = "strategy",
    run_mode: str = "agent",
) -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "Candidate Lab typed request",
            "validator": "qa",
            "source_dir": str(_source(tmp_path)),
            "task_type": task_type,
            "run_mode": run_mode,
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _request(workflow: str, workflow_inputs: dict) -> dict:
    return {
        "content": f"人工界面执行 {workflow}",
        "strategy_request": {
            "request_kind": "standard_workflow",
            "workflow": workflow,
            "workflow_inputs": workflow_inputs,
        },
    }


@pytest.mark.slow
@pytest.mark.e2e
def test_three_typed_candidate_workflows_run_without_any_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    sample_ref = materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed request must not resolve an Agent gate LLM")
        ),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _BombLLM(),
    )

    requests = [
        _request(
            "univariate_candidate_analysis",
            {
                "features": ["score", "age"],
                "methods": ["equal_width"],
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
                "sentinel_values": [],
            },
        ),
        _request(
            "cross_matrix_analysis",
            {
                "x_feature": "age",
                "x_method": "equal_width",
                "y_feature": "score",
                "y_method": "equal_width",
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
                "sentinel_values": [],
            },
        ),
        _request(
            "automatic_tree_candidate_build",
            {
                "features": ["score", "income"],
                "directions": {
                    "score": "decreasing",
                    "income": "decreasing",
                },
                "max_depth": 2,
                "min_leaf_count": 2,
                "min_weight_fraction_leaf": 0.0,
                "seed": 20260724,
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
            },
        ),
    ]

    for body in requests:
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json=body,
        )
        assert response.status_code == 202, response.text

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design",
        "strategy_univariate_candidate_analysis",
        "strategy_cross_matrix_analysis",
        "strategy_automatic_tree_candidate_build",
    ]
    assert all(plan["status"] == "done" for plan in plans)
    for plan in plans[1:]:
        stored = client.app.state.plan_repo.load_plan(plan["id"])
        assert stored.steps[0].inputs["sample_design_ref"] == sample_ref

    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    typed_user_messages = [
        message
        for message in messages
        if (message.get("metadata") or {}).get("request_source") == "manual_ui"
    ]
    assert [message["metadata"]["workflow"] for message in typed_user_messages] == [
        "univariate_candidate_analysis",
        "cross_matrix_analysis",
        "automatic_tree_candidate_build",
    ]
    assert all(
        set(message["metadata"]) == {"intent", "request_source", "workflow"}
        for message in typed_user_messages
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_typed_refinement_runs_fresh_cutpoints_and_an_exact_existing_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path / "owner")
    sample_ref = materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed refinement must not resolve an Agent gate LLM")
        ),
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _BombLLM(),
    )

    fresh = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {
                "feature": "score",
                "method": "manual",
                "manual_breakpoints": {"score": [500, 700]},
                "selection": {
                    "risk_threshold": {"operator": ">=", "value": 0.5}
                },
                "selection_reason": "保留观测坏率达到 50% 的风险箱",
            },
        ),
    )
    assert fresh.status_code == 202, fresh.text
    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    fresh_plan = plans[-1]
    assert fresh_plan.template_id == "strategy_univariate_candidate_refinement"
    assert fresh_plan.status == PlanStatus.DONE
    assert fresh_plan.steps[0].inputs["sample_design_ref"] == sample_ref
    assert fresh_plan.steps[0].inputs["manual_breakpoints"] == {
        "score": [500.0, 700.0]
    }
    source = client.app.state.plan_repo.load_step_output(fresh_plan.steps[0].id)
    candidate_id = source["candidate_id"]

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

    existing = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "merge_groups": [["regular:0", "regular:1"]],
                "selection": {
                    "source_bin_ids": ["regular:0", "regular:1"],
                },
                "selection_reason": "合并并保留已经核验的两个风险箱",
            },
        ),
    )
    assert existing.status_code == 202, existing.text
    existing_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    assert (
        existing_plan.template_id
        == "strategy_univariate_candidate_refinement_existing"
    )
    assert existing_plan.status == PlanStatus.DONE
    assert len(existing_plan.steps) == 1
    output = client.app.state.plan_repo.load_step_output(existing_plan.steps[0].id)
    assert output["parent_candidate_id"] == candidate_id
    assert output["parent_evidence_hash"] == source["evidence_hash"]
    assert output["candidate_asset"]["refinement"]["merge_groups"] == [
        ["regular:0", "regular:1"]
    ]

    invalid_controls = [
        (
            "feature",
            {
                "feature": "age",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
        (
            "method",
            {
                "feature": "score",
                "method": "equal_width",
                "source_candidate_id": candidate_id,
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
        (
            "selection_bin",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "selection": {"source_bin_ids": ["regular:999"]},
            },
        ),
        (
            "merge_bin",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "merge_groups": [["regular:0", "regular:999"]],
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
    ]
    for label, workflow_inputs in invalid_controls:
        plans_before = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        artifacts_before = client.get(
            f"/api/tasks/{task_id}/task-artifacts"
        ).json()["artifacts"]
        rejected = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json=_request("univariate_candidate_refinement", workflow_inputs),
        )
        assert rejected.status_code == 202, (label, rejected.text)
        assert rejected.json()["status"] == "clarification_required", label
        assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == plans_before
        assert client.get(f"/api/tasks/{task_id}/task-artifacts").json()[
            "artifacts"
        ] == artifacts_before

    source_json = next(
        artifact
        for artifact in source["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    source_record = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).get_for_task(task_id, source_json["artifact_id"])
    assert source_record is not None
    Path(source_record["path"]).write_bytes(b"{}")
    plans_before_corruption = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    artifacts_before_corruption = client.get(
        f"/api/tasks/{task_id}/task-artifacts"
    ).json()["artifacts"]
    corrupted = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
    )
    assert corrupted.status_code == 202, corrupted.text
    assert corrupted.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == (
        plans_before_corruption
    )
    assert client.get(f"/api/tasks/{task_id}/task-artifacts").json()[
        "artifacts"
    ] == artifacts_before_corruption

    other_task_id = _task(client, tmp_path / "other")
    cross_task = client.post(
        f"/api/tasks/{other_task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": candidate_id,
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
    )
    assert cross_task.status_code == 202, cross_task.text
    assert client.get(f"/api/tasks/{other_task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{other_task_id}/task-artifacts").json()["artifacts"]
        == []
    )
    assert "当前任务没有候选证据" in " ".join(
        message.get("content", "") for message in cross_task.json()["messages"]
    )

    owner_plans_before = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    owner_artifacts_before = client.get(
        f"/api/tasks/{task_id}/task-artifacts"
    ).json()["artifacts"]
    forged = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {
                "feature": "score",
                "method": "manual",
                "source_candidate_id": "candidate-" + "f" * 32,
                "selection": {"source_bin_ids": ["regular:0"]},
            },
        ),
    )
    assert forged.status_code == 202, forged.text
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == (
        owner_plans_before
    )
    assert client.get(f"/api/tasks/{task_id}/task-artifacts").json()[
        "artifacts"
    ] == owner_artifacts_before
    assert "当前任务没有候选证据" in " ".join(
        message.get("content", "") for message in forged.json()["messages"]
    )


def test_typed_request_schema_and_platform_fields_fail_before_execution(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    cases = [
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_compile",
            "workflow_inputs": {},
        },
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {"features": ["score"]},
            "extra": "forbidden",
        },
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {"features": ["score"]},
            "schema_version": "invented.v1",
        },
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {
                "features": ["score"],
                "dataset_id": "forged",
            },
        },
    ]
    for strategy_request in cases:
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "执行", "strategy_request": strategy_request},
        )
        assert response.status_code == 422, response.text

    # Unknown user-control fields reach the authoritative workflow validator,
    # which returns a clarification but still creates no plan or artifact.
    rejected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_analysis",
            {"features": ["score"], "unknown_control": True},
        ),
    )
    assert rejected.status_code == 202, rejected.text
    assert rejected.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


@pytest.mark.parametrize(
    "platform_field",
    [
        "artifact_id",
        "expected_evidence_hash",
        "revision",
        "dataset_id",
        "target_col",
    ],
)
def test_typed_refinement_accepts_a_user_candidate_pointer_but_not_platform_bindings(
    tmp_path: Path,
    platform_field: str,
) -> None:
    client = TestClient(create_app(tmp_path / platform_field))
    task_id = _task(client, tmp_path / platform_field)
    workflow_inputs = {
        "feature": "score",
        "method": "equal_width",
        "source_candidate_id": "candidate-" + "a" * 32,
        "selection": {"source_bin_ids": ["regular:0"]},
    }

    accepted_shape = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request("univariate_candidate_refinement", workflow_inputs),
    )

    assert accepted_shape.status_code == 202, accepted_shape.text
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )

    rejected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_refinement",
            {**workflow_inputs, platform_field: "forged"},
        ),
    )
    assert rejected.status_code == 422, rejected.text


def test_new_typed_request_invalidates_an_obsolete_pending_strategy_request(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    pending_repo = PendingStrategyRequestRepository(
        client.app.state.settings.db_path
    )
    pending = pending_repo.create(
        task_id=task_id,
        validated_draft={
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {"features": ["score"]},
        },
        dataset_identity=None,
        target_col="bad",
    )
    TaskRepository(client.app.state.settings.db_path).add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="obsolete request",
        metadata={
            "intent": "strategy_request_confirmation",
            "strategy_request": pending.to_metadata_reference(),
        },
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_analysis",
            {"features": ["score"], "unknown_control": True},
        ),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    reloaded = pending_repo.get(task_id, pending.id)
    assert reloaded is not None
    assert reloaded.status == "invalidated"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "strategy_input",
            {
                "entry_mode": "strategy_analysis",
                "strategy_type": "approval",
            },
        ),
        ("selection", []),
        ("dedup_strategies", {}),
        ("adjust_params", {}),
        ("expected_step_id", "step-1"),
    ],
)
def test_typed_request_cannot_mix_with_other_structured_controls(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    client = TestClient(create_app(tmp_path / field))
    task_id = _task(client, tmp_path / field)
    body = _request(
        "univariate_candidate_analysis",
        {"features": ["score"]},
    )
    body[field] = value
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=body,
    )
    assert response.status_code == 422, response.text
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_typed_request_cannot_mix_with_stop_or_target_non_strategy_task(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    strategy_task_id = _task(client, tmp_path / "strategy")
    body = _request(
        "univariate_candidate_analysis",
        {"features": ["score"]},
    )
    body["content"] = "停止"
    stopped = client.post(
        f"/api/tasks/{strategy_task_id}/agent/messages",
        json=body,
    )
    assert stopped.status_code == 422, stopped.text

    modeling_task_id = _task(
        client,
        tmp_path / "modeling",
        task_type="modeling",
        run_mode="manual",
    )
    wrong_task = client.post(
        f"/api/tasks/{modeling_task_id}/agent/messages",
        json=_request(
            "univariate_candidate_analysis",
            {"features": ["score"]},
        ),
    )
    assert wrong_task.status_code == 422, wrong_task.text


@pytest.mark.parametrize("conflict_kind", ["active_plan", "open_gate"])
def test_typed_request_conflict_preserves_messages_gate_and_pending_request(
    tmp_path: Path,
    conflict_kind: str,
) -> None:
    client = TestClient(create_app(tmp_path / conflict_kind))
    task_id = _task(client, tmp_path / conflict_kind)
    pending_repo = PendingStrategyRequestRepository(
        client.app.state.settings.db_path
    )
    pending = pending_repo.create(
        task_id=task_id,
        validated_draft={
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {"features": ["score"]},
        },
        dataset_identity=None,
        target_col="bad",
    )
    metadata = {
        "intent": "strategy_request_confirmation",
        "strategy_request": pending.to_metadata_reference(),
    }
    if conflict_kind == "open_gate":
        metadata.update({"kind": "gate", "plan_id": "existing-plan"})
    task_repo = TaskRepository(client.app.state.settings.db_path)
    task_repo.add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="existing state",
        metadata=metadata,
    )
    if conflict_kind == "active_plan":
        client.app.state.plan_repo.create_plan(
            Plan(
                id="existing-plan",
                task_id=task_id,
                goal="existing work",
                source="template",
                template_id="existing",
                steps=[],
                autonomy_level=1,
                status=PlanStatus.VALIDATED,
            )
        )
    messages_before = task_repo.list_agent_messages(task_id)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "univariate_candidate_analysis",
            {"features": ["score"]},
        ),
    )

    assert response.status_code == 409, response.text
    assert task_repo.list_agent_messages(task_id) == messages_before
    reloaded = pending_repo.get(task_id, pending.id)
    assert reloaded is not None
    assert reloaded.status == "pending"
