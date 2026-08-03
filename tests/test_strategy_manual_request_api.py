"""Typed Candidate Lab requests stay LLM-free and governed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marvis.agent import turn_handlers
from marvis.api_schemas import ManualStrategyRequest
from marvis.app import create_app
from marvis.db import StrategyRepository, TaskRepository
from marvis.orchestrator.contracts import Plan, PlanStatus
from marvis.packs.strategy.strategy import build_strategy_from_spec
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


def _sample_design_v2_inputs() -> dict:
    return {
        "target_bad_value": 1,
        "drop_nan_labels": True,
        "relationship": "nested_same_cohort",
        "approval_population": {
            "inclusion": {
                "match": "all",
                "conditions": [
                    {"column": "age", "operator": "gte", "value": 20},
                ],
            },
            "exclusion": None,
        },
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": {
                    "op": "eq",
                    "left": {"column": "age"},
                    "right": {"literal": 20},
                },
                "validation": {
                    "op": "eq",
                    "left": {"column": "age"},
                    "right": {"literal": 22},
                },
                "oot": {
                    "op": "eq",
                    "left": {"column": "age"},
                    "right": {"literal": 24},
                },
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": 30,
            "cutoff_date": "2026-04-30",
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-04-30",
        },
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": None,
            "group_field": None,
            "month_field": None,
            "weight_field": None,
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "historical_score": {
            "status": "available",
            "column": "score",
            "direction": "higher_is_riskier",
            "reason": None,
        },
    }


def test_sample_design_v2_manual_request_accepts_only_fresh_user_dto() -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": _sample_design_v2_inputs(),
        },
        strict=True,
    )

    assert request.workflow == "strategy_sample_design_v2"
    assert request.workflow_inputs["relationship"] == "nested_same_cohort"


def test_manual_project_context_binds_typed_external_report_without_rewriting_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    report_path = tmp_path / "source" / "历史策略评审.xlsx"
    report_path.write_bytes(b"opaque reviewed strategy evidence")
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _BombLLM(),
    )
    body = _request(
        "strategy_project_context",
        {
            "as_of": "2026-07-27",
            "scope": "存量复借策略",
            "business_context": {"project.channel": "自营"},
            "explicit_unavailable": [
                "current.status_fields.volume",
                "current.status_fields.approval",
                "current.status_fields.risk",
                "current.status_fields.economics",
                "current.maturity_summary",
            ],
            "external_report_filenames": [report_path.name],
        },
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=body,
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_project_context"
    ], response.text
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    source_message = next(
        message
        for message in response.json()["messages"]
        if message["role"] == "user"
    )
    assert source_message["content"] == body["content"]
    request_sha256 = source_message["metadata"]["structured_request_sha256"]
    assert len(request_sha256) == 64
    assert stored.steps[0].inputs["user_message_ref"][
        "structured_request_sha256"
    ] == request_sha256
    assert stored.steps[0].inputs["explicit_unavailable"] == body[
        "strategy_request"
    ]["workflow_inputs"]["explicit_unavailable"]
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["external_artifacts"][0]["kind"] == (
        "strategy_history_external_source"
    )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "strategy_project_context",
            {
                "as_of": "2026-07-27",
                "scope": "存量复借策略",
                "business_context": {"project.channel": "自营"},
                "explicit_unavailable": ["historical_strategy_reviews"],
                "external_report_filenames": [],
            },
        ),
        (
            "strategy_pool_impact",
            {
                "strategy_type": "approval",
                "comparison_mode": "absolute",
                "drop_nan_labels": False,
            },
        ),
        (
            "strategy_impact_cube",
            {
                "strategy_type": "pricing",
                "partitions": ["development", "validation", "oot"],
            },
        ),
        (
            "strategy_dsl_delivery",
            {"strategy_id": "strategy-current-v2"},
        ),
        (
            "strategy_report_bundle_v2",
            {"title": "策略迭代评审报告", "status": "partial"},
        ),
    ],
)
def test_public_manual_workbench_spine_reaches_governed_preflight_without_llm(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    workflow_inputs: dict,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    reached = []

    def stop_at_preflight(runtime, task, draft):
        del runtime, task
        reached.append(draft.workflow)
        return ("test_preflight_reached", "governed preflight reached")

    monkeypatch.setattr(
        turn_handlers,
        "_strategy_request_preflight",
        stop_at_preflight,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _BombLLM(),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(workflow, workflow_inputs),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert reached == [workflow]


def test_typed_local_adoption_runs_backtest_and_stops_at_human_gate_without_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    strategy = build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "rules": [
                {
                    "rule_id": "low-score-reject",
                    "priority": 10,
                    "condition": {
                        "op": "compare",
                        "field": "score",
                        "operator": "<",
                        "value": 500,
                    },
                    "action": {
                        "type": "reject",
                        "reason_code": "low-score",
                    },
                }
            ],
        },
        description="Candidate Lab local adoption",
    )
    StrategyRepository(client.app.state.settings.db_path).create_strategy(
        task_id,
        strategy,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _BombLLM(),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "提交本地采纳复核",
            "strategy_request": {
                "request_kind": "strategy_lifecycle",
                "operation": "adopt",
                "strategy_type": "approval",
                "strategy_id": strategy.id,
                "adoption_reason": "已复核独立验证、影响测算和报告证据，同意本地采纳",
            },
        },
    )

    assert response.status_code == 202, response.text
    plans = [
        item
        for item in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        if item["template_id"] != "strategy_sample_design"
    ]
    assert [item["template_id"] for item in plans] == [
        "stored_strategy_adoption"
    ]
    assert plans[0]["status"] == "awaiting_confirm"
    assert plans[0]["steps"][0]["status"] == "done"
    assert plans[0]["steps"][1]["status"] == "awaiting_confirm"
    assert all(
        step["status"] == "pending"
        for step in plans[0]["steps"][2:]
    )
    meta = StrategyRepository(
        client.app.state.settings.db_path
    ).get_strategy_meta(strategy.id)
    assert meta is not None
    assert meta["asset_status"] == "draft"


def test_sample_design_v2_manual_request_accepts_flat_cross_column_selector() -> None:
    inputs = _sample_design_v2_inputs()
    inputs["partitioning"]["selectors"]["development"] = {
        "op": "and",
        "args": [
            {
                "op": "eq",
                "left": {"column": "age"},
                "right": {"literal": 20},
            },
            {
                "op": "eq",
                "left": {"column": "score"},
                "right": {"literal": 100},
            },
        ],
    }

    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design_v2",
            "workflow_inputs": inputs,
        },
        strict=True,
    )

    assert (
        request.workflow_inputs["partitioning"]["selectors"]["development"][
            "op"
        ]
        == "and"
    )


def test_sample_design_v2_manual_request_rejects_time_range_column_mismatch() -> None:
    inputs = _sample_design_v2_inputs()
    inputs["partitioning"] = {
        "method": "time_ranges",
        "column": "age",
        "ranges": {
            "development": {"start": "2026-01-01", "end": "2026-02-28"},
            "validation": {"start": "2026-03-01", "end": "2026-03-31"},
            "oot": {"start": "2026-04-01", "end": "2026-04-30"},
        },
    }
    inputs["field_bindings"]["time_field"] = "score"

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": inputs,
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "selector",
    [
        {
            "op": "or",
            "args": [
                {
                    "op": "and",
                    "args": [
                        {
                            "op": "eq",
                            "left": {"column": "age"},
                            "right": {"literal": 20},
                        },
                        {
                            "op": "eq",
                            "left": {"column": "score"},
                            "right": {"literal": 100},
                        },
                    ],
                },
                {
                    "op": "eq",
                    "left": {"column": "age"},
                    "right": {"literal": 22},
                },
            ],
        },
        {
            "op": "not",
            "arg": {
                "op": "eq",
                "left": {"column": "age"},
                "right": {"literal": 20},
            },
        },
        {
            "op": "eq",
            "left": {"column": "age"},
            "right": {"column": "score"},
        },
    ],
)
def test_sample_design_v2_manual_request_rejects_nonflat_partition_selector(
    selector: dict,
) -> None:
    inputs = _sample_design_v2_inputs()
    inputs["partitioning"]["selectors"]["development"] = selector

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": inputs,
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["approval_population"].update(
            {
                "inclusion": {
                    "op": "eq",
                    "left": {"column": "age"},
                    "right": {"literal": 20},
                }
            }
        ),
        lambda value: value.update({"source_mode": "native_active_dataset"}),
        lambda value: value.pop("relationship"),
        lambda value: value.update({"target_bad_value": True}),
        lambda value: value["approval_population"]["inclusion"]["conditions"][0].update(
            {"value": float("nan")}
        ),
    ],
)
def test_sample_design_v2_manual_request_rejects_raw_ast_platform_or_invalid_controls(
    mutate,
) -> None:
    inputs = _sample_design_v2_inputs()
    mutate(inputs)

    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design_v2",
                "workflow_inputs": inputs,
            },
            strict=True,
        )


@pytest.mark.parametrize(
    "workflow_inputs",
    [
        {"asset_id": "candidate-asset-" + "a" * 32},
        {
            "strategy_type": "approval",
            "entry_id": "pool-entry-" + "b" * 32,
        },
    ],
)
def test_candidate_stability_manual_request_accepts_visible_pointer_only(
    workflow_inputs: dict,
) -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "candidate_monthly_stability",
            "workflow_inputs": workflow_inputs,
        },
        strict=True,
    )

    assert request.workflow == "candidate_monthly_stability"
    assert request.workflow_inputs == workflow_inputs


@pytest.mark.parametrize(
    "workflow_inputs",
    [
        {"asset_id": "candidate-asset-" + "a" * 31},
        {"asset_id": "candidate-asset-" + "a" * 32, "strategy_type": "approval"},
        {
            "strategy_type": "unsupported",
            "entry_id": "pool-entry-" + "b" * 32,
        },
        {
            "strategy_type": "approval",
            "entry_id": "pool-entry-" + "b" * 32,
            "expected_pool_revision": 2,
        },
        {
            "strategy_type": "approval",
            "entry_id": "pool-entry-" + "b" * 32,
            "snapshot_hash": "f" * 64,
        },
    ],
)
def test_candidate_stability_manual_request_rejects_forged_or_invalid_inputs(
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "candidate_monthly_stability",
                "workflow_inputs": workflow_inputs,
            },
            strict=True,
        )


def test_candidate_stability_manual_pointer_reaches_governed_preflight_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed stability request must not resolve an LLM")
        ),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "candidate_monthly_stability",
            {
                "strategy_type": "approval",
                "entry_id": "pool-entry-" + "c" * 32,
            },
        ),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_pool_stability_manual_request_accepts_only_strategy_type(
    strategy_type: str,
) -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_stability",
            "workflow_inputs": {"strategy_type": strategy_type},
        },
        strict=True,
    )

    assert request.workflow == "strategy_pool_stability"
    assert request.workflow_inputs == {"strategy_type": strategy_type}


@pytest.mark.parametrize(
    "workflow_inputs",
    [
        {},
        {"strategy_type": None},
        {"strategy_type": "unsupported"},
        {"strategy_type": "approval", "partitions": ["oot"]},
        {"strategy_type": "approval", "artifact_id": "a" * 64},
        {"strategy_type": "approval", "psi_threshold": 0.25},
    ],
)
def test_pool_stability_manual_request_rejects_platform_controls(
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_pool_stability",
                "workflow_inputs": workflow_inputs,
            },
            strict=True,
        )


def test_pool_stability_manual_pointer_reaches_governed_preflight_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed Pool stability request must not resolve an LLM")
        ),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "strategy_pool_stability",
            {"strategy_type": "approval"},
        ),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


def test_pool_add_manual_pointer_reaches_governed_preflight_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed Pool add request must not resolve an LLM")
        ),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "strategy_pool_add_candidate",
            {
                "candidate_asset_id": "candidate-asset-" + "a" * 32,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        ),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


def test_voting_search_selection_manual_pointer_reaches_preflight_without_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("typed Voting request must not resolve an LLM")
        ),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_request(
            "voting_candidate_build_from_search",
            {
                "search_id": "voting-search-" + "c" * 32,
                "combo_id": "voting-combo-" + "d" * 32,
                "strategy_type": "approval",
            },
        ),
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "scorecard_model_score_evidence_build",
            {
                "features": ["age", "income"],
                "sample_weight_col": "weight",
                "seed": 23,
                "max_iter": 200,
                "scorecard_max_bins": 4,
            },
        ),
        ("scorecard_band_build", {}),
        ("scorecard_band_build", {"bin_count": 10}),
        (
            "scorecard_band_build",
            {"raw_pd_band_edges": [0, 0.25, 0.7, 1]},
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "reason": "选择经业务评审的观测切点",
            },
        ),
    ],
)
def test_scorecard_manual_request_schema_accepts_only_user_owned_controls(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    request = ManualStrategyRequest.model_validate(
        {
            "request_kind": "standard_workflow",
            "workflow": workflow,
            "workflow_inputs": workflow_inputs,
        },
        strict=True,
    )

    assert request.workflow == workflow
    assert request.workflow_inputs == workflow_inputs


@pytest.mark.parametrize(
    ("workflow", "workflow_inputs"),
    [
        (
            "scorecard_model_score_evidence_build",
            {
                "features": ["age", "age"],
                "seed": 23,
                "max_iter": 200,
                "scorecard_max_bins": 4,
            },
        ),
        (
            "scorecard_model_score_evidence_build",
            {
                "features": ["age"],
                "sample_weight_col": "age",
                "seed": 23,
                "max_iter": 200,
                "scorecard_max_bins": 4,
            },
        ),
        (
            "scorecard_model_score_evidence_build",
            {
                "features": ["age"],
                "seed": 23,
                "max_iter": 19,
                "scorecard_max_bins": 4,
            },
        ),
        (
            "scorecard_band_build",
            {"bin_count": 10, "raw_pd_band_edges": [0, 0.5, 1]},
        ),
        ("scorecard_band_build", {"bin_count": True}),
        ("scorecard_band_build", {"bin_count": 1}),
        ("scorecard_band_build", {"raw_pd_band_edges": [0, 0.5, 0.5, 1]}),
        ("scorecard_band_build", {"raw_pd_band_edges": [0.1, 0.5, 1]}),
        ("scorecard_band_build", {"raw_pd_band_edges": [0, 0.5, 0.9]}),
        ("scorecard_band_build", {"artifact_id": "forged"}),
        ("scorecard_band_build", {"unknown_control": True}),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "artifact_id": "forged",
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "asset_hash": "f" * 64,
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "not-a-scorecard-asset",
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "not-a-scorecard-cutoff",
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "reason": None,
            },
        ),
    ],
)
def test_scorecard_manual_request_schema_rejects_forged_or_unknown_inputs(
    workflow: str,
    workflow_inputs: dict,
) -> None:
    with pytest.raises(ValidationError):
        ManualStrategyRequest.model_validate(
            {
                "request_kind": "standard_workflow",
                "workflow": workflow,
                "workflow_inputs": workflow_inputs,
            },
            strict=True,
        )


def test_scorecard_forged_bindings_fail_at_http_schema_before_execution(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    forged_requests = [
        ("scorecard_band_build", {"artifact_id": "forged"}),
        ("scorecard_band_build", {"unknown_control": True}),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "source_artifact_id": "f" * 64,
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "expected_asset_hash": "f" * 64,
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "source_ref": {"artifact_id": "forged"},
            },
        ),
        (
            "scorecard_cutoff_selection",
            {
                "asset_id": "scorecard-band-asset-" + "a" * 32,
                "cutoff_id": "scorecard-cutoff-" + "b" * 32,
                "unknown_control": True,
            },
        ),
    ]

    for workflow, workflow_inputs in forged_requests:
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json=_request(workflow, workflow_inputs),
        )
        assert response.status_code == 422, response.text

    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert (
        client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"] == []
    )


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
