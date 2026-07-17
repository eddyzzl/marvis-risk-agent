"""HTTP contract for resuming strategy setup after structured clarification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.domain import StrategyTaskInput, TaskCreate


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _strategy_source(tmp_path: Path) -> Path:
    source = tmp_path / "strategy-materials"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bad": [1, 0, 0, 0, 1, 0],
            "score": [580, 620, 730, 760, 590, 800],
        }
    ).to_csv(source / "strategy.csv", index=False)
    return source


def _create_strategy_task(
    client: TestClient,
    tmp_path: Path,
    *,
    strategy_input: dict | None = None,
) -> dict:
    body = {
        "model_name": "额度准入策略",
        "validator": "qa",
        "source_dir": str(_strategy_source(tmp_path)),
        "task_type": "strategy",
        "run_mode": "manual",
        "target_col": "bad",
        "score_col": "score",
    }
    if strategy_input is not None:
        body["strategy_input"] = strategy_input
    response = client.post("/api/tasks", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_strategy_clarification_can_resume_with_structured_business_contract(
    tmp_path: Path,
):
    client = _client(tmp_path)
    created = _create_strategy_task(client, tmp_path)
    task_id = created["id"]
    assert created["strategy_input"] is None

    clarified = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert clarified.status_code == 202, clarified.text
    assert clarified.json()["status"] == "clarification_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    contract = {
        "entry_mode": "strategy_development",
        "objective": "max_approval",
        "max_bad_rate": 0.20,
    }
    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "补充策略业务口径",
            "strategy_input": contract,
        },
    )

    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["status"] == "ok"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_development"]
    loaded = client.get(f"/api/tasks/{task_id}")
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["strategy_input"] == {
        "entry_mode": "strategy_development",
        "objective": "max_approval",
        "max_bad_rate": 0.20,
        "min_approval_rate": None,
        "baseline_strategy_id": None,
        "profit": None,
    }
    assert loaded.json()["updated_at"] != created["updated_at"]


def test_strategy_input_on_non_strategy_agent_message_is_rejected(tmp_path: Path):
    client = _client(tmp_path)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "模型开发",
            "validator": "qa",
            "source_dir": str(tmp_path),
            "task_type": "modeling",
            "run_mode": "manual",
        },
    )
    assert created.status_code == 200, created.text

    response = client.post(
        f"/api/tasks/{created.json()['id']}/agent/messages",
        json={
            "content": "补充策略业务口径",
            "strategy_input": {
                "objective": "max_approval",
                "max_bad_rate": 0.20,
            },
        },
    )

    assert response.status_code == 422, response.text
    assert "strategy_input" in response.json()["detail"]


def test_repository_rejects_strategy_input_update_for_non_strategy_task(
    tmp_path: Path,
):
    client = _client(tmp_path)
    repo = TaskRepository(client.app.state.settings.db_path)
    task = repo.create_task(
        TaskCreate(
            task_type="modeling",
            model_name="模型开发",
            model_version="",
            validator="qa",
            source_dir=str(tmp_path),
        )
    )

    with pytest.raises(ValueError, match="strategy"):
        repo.update_strategy_input(
            task.id,
            StrategyTaskInput(objective="max_approval", max_bad_rate=0.20),
        )

    assert repo.get_task(task.id).strategy_input is None


def test_partial_profit_contract_is_rejected_without_mutating_task(tmp_path: Path):
    client = _client(tmp_path)
    created = _create_strategy_task(client, tmp_path)
    task_id = created["id"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "补充利润策略口径",
            "strategy_input": {
                "objective": "max_profit",
                "max_bad_rate": 0.20,
                "profit": {"ead_col": "ead"},
            },
        },
    )

    assert response.status_code == 422, response.text
    assert client.get(f"/api/tasks/{task_id}").json()["strategy_input"] is None
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


@pytest.mark.parametrize(
    ("contract", "expected_missing"),
    [
        (
            {"objective": "max_approval"},
            {"max_bad_rate_or_min_approval_rate"},
        ),
        (
            {"objective": "max_profit", "max_bad_rate": 0.20},
            {
                "profit.ead_col",
                "profit.pd_col",
                "profit.annual_rate",
                "profit.funding_rate",
                "profit.lgd",
                "profit.operating_cost_per_loan",
                "profit.term_months",
            },
        ),
    ],
)
def test_structured_continuation_requires_complete_contract_before_persisting(
    tmp_path: Path,
    contract: dict,
    expected_missing: set[str],
):
    client = _client(tmp_path)
    created = _create_strategy_task(client, tmp_path)
    task_id = created["id"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "补充策略业务口径", "strategy_input": contract},
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "strategy_business_inputs_required"
    assert set(detail["missing_fields"]) == expected_missing
    assert client.get(f"/api/tasks/{task_id}").json()["strategy_input"] is None
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_stop_intent_cannot_silently_discard_structured_strategy_input(tmp_path: Path):
    client = _client(tmp_path)
    created = _create_strategy_task(client, tmp_path)
    task_id = created["id"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "停止",
            "strategy_input": {
                "objective": "max_approval",
                "max_bad_rate": 0.20,
            },
        },
    )

    assert response.status_code == 422, response.text
    assert "停止" in response.json()["detail"]
    assert client.get(f"/api/tasks/{task_id}").json()["strategy_input"] is None
    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    assert messages == []


def test_losing_driver_job_request_does_not_persist_strategy_input(tmp_path: Path):
    client = _client(tmp_path)
    created = _create_strategy_task(client, tmp_path)
    task_id = created["id"]
    repo = TaskRepository(client.app.state.settings.db_path)
    held_job_id = repo.start_job(task_id, "driver")

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "补充策略业务口径",
            "strategy_input": {
                "objective": "max_approval",
                "max_bad_rate": 0.20,
            },
        },
    )

    assert response.status_code == 409, response.text
    assert client.get(f"/api/tasks/{task_id}").json()["strategy_input"] is None
    repo.finish_job(held_job_id, status="cancelled")


def test_nonterminal_plan_rejects_contract_replacement_without_mutation(
    tmp_path: Path,
):
    client = _client(tmp_path)
    original_contract = {
        "objective": "max_approval",
        "max_bad_rate": 0.20,
    }
    created = _create_strategy_task(
        client,
        tmp_path,
        strategy_input=original_contract,
    )
    task_id = created["id"]
    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert started.status_code == 202, started.text
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "修改策略业务口径",
            "strategy_input": {
                "objective": "max_approval",
                "max_bad_rate": 0.10,
            },
        },
    )

    assert response.status_code == 409, response.text
    loaded = client.get(f"/api/tasks/{task_id}").json()
    assert loaded["strategy_input"]["max_bad_rate"] == 0.20
    latest_job = client.get(
        f"/api/tasks/{task_id}/jobs/latest", params={"kind": "driver"}
    ).json()["job"]
    assert latest_job["status"] == "failed"
