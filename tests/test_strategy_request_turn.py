"""Natural-language strategy request -> validated, governed Workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.agent.plan_driver import DriverError
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository, TaskRepository
from marvis.files import sha256_file
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.pending_strategy_requests import (
    PendingStrategyRequestRepository,
)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _install_llm(monkeypatch, llm: _FakeLLM) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "strategy-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 0, 1, 0, 1, 1],
            "score": [800, 760, 720, 680, 640, 600],
            "ead": [1000, 1200, 1500, 1300, 1800, 2000],
            "pd": [0.02, 0.03, 0.08, 0.12, 0.20, 0.30],
        }
    ).to_csv(source / "sample.csv", index=False)
    return source


def _task(client: TestClient, tmp_path: Path) -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言策略任务",
            "validator": "qa",
            "source_dir": str(_source(tmp_path)),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _nan_target_task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "nan-target-strategy-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 0, 1, None, 1, 0],
            "score": [800, 760, 720, 680, 640, 600],
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "含空标签策略任务",
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


def _unlabeled_task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "unlabeled-strategy-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [0, 1, 2, 3]}).to_csv(
        source / "production.csv",
        index=False,
    )
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "无标签策略应用任务",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _standard_workflow_task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "standard-workflow-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "customer_id": ["a", "a", "b", "b"],
            "month": ["2026-01", "2026-02", "2026-01", "2026-02"],
            "status": ["M0", "M1", "M0", "M0"],
            "segment": ["new", "new", "old", "old"],
            "balance": [1000, 900, 2000, 1900],
            "score": [780, 760, 720, 700],
            "ead": [1000, 900, 2000, 1900],
            "pd": [0.02, 0.04, 0.08, 0.10],
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "标准策略分析任务",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _register_task_source(
    client: TestClient,
    task_id: str,
    source_path: Path,
) -> None:
    settings = client.app.state.settings
    root = settings.datasets_dir
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(root),
        root,
    )
    registry.register_from_upload(task_id, source_path, role="sample")


def _spec(strategy_type: str, *, field: str = "x") -> dict:
    default, matched = {
        "approval": ({"type": "approval"}, {"type": "reject"}),
        "reject": ({"type": "approval"}, {"type": "reject"}),
        "limit": (
            {"type": "limit", "value": 1000},
            {"type": "limit", "value": 2000},
        ),
        "pricing": (
            {"type": "pricing", "value": 0.10},
            {"type": "pricing", "value": 0.20},
        ),
        "segmentation": (
            {"type": "segment", "value": "base"},
            {"type": "segment", "value": "high"},
        ),
    }[strategy_type]
    return {
        "strategy_type": strategy_type,
        "default_action": default,
        "rules": [
            {
                "rule_id": "x-positive",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": field,
                    "operator": ">",
                    "value": 0,
                },
                "action": matched,
            }
        ],
    }


def _last_assistant(messages: list[dict]) -> dict:
    return next(
        message
        for message in reversed(messages)
        if message["role"] == "assistant"
    )


def _seed_legacy_pending(
    client: TestClient,
    task_id: str,
    draft: dict,
    *,
    dataset_identity: dict | None = None,
    target_col: str | None = None,
):
    settings = client.app.state.settings
    record = PendingStrategyRequestRepository(settings.db_path).create(
        task_id=task_id,
        validated_draft=draft,
        dataset_identity=dataset_identity,
        target_col=target_col,
    )
    TaskRepository(settings.db_path).add_agent_message(
        task_id,
        role="assistant",
        stage="chat",
        content="历史版本待确认策略草案。",
        metadata={
            "strategy_request": record.to_metadata_reference(),
            "intent": "strategy_request_confirmation",
            "kind": "confirmation",
        },
    )
    return record


def _saved_strategy(
    client: TestClient,
    task_id: str,
    *,
    threshold: int = 0,
    strategy_type: str = "approval",
) -> str:
    spec = _spec(strategy_type)
    spec["rules"][0]["condition"]["value"] = threshold
    strategy = build_strategy_from_spec(
        spec,
        description=f"{strategy_type}-{threshold}",
    )
    StrategyRepository(client.app.state.settings.db_path).create_strategy(
        task_id,
        strategy,
    )
    return strategy.id


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
@pytest.mark.slow
@pytest.mark.e2e
def test_five_typed_requests_auto_run_real_typed_workflow(
    tmp_path: Path,
    monkeypatch,
    strategy_type: str,
) -> None:
    client = TestClient(create_app(tmp_path / strategy_type))
    task_id = _task(client, tmp_path / strategy_type)
    payload = {
        "operation": "backtest",
        "strategy_type": strategy_type,
        "strategy_spec": _spec(strategy_type),
    }
    if strategy_type == "limit":
        payload["economics_inputs"] = {
            "pd_col": "pd",
            "lgd_value": 0.5,
            "utilization_value": 0.6,
        }
    elif strategy_type == "pricing":
        payload["economics_inputs"] = {
            "ead_col": "ead",
            "pd_col": "pd",
            "lgd_value": 0.5,
            "funding_rate_value": 0.03,
            "term_months_value": 12,
            "operating_cost_per_loan_value": 10,
        }
    llm = _FakeLLM(payload)
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"请回测这份 {strategy_type} 策略"},
    )

    assert opened.status_code == 202, opened.text
    assert len(llm.calls) == 1
    assert all(
        "strategy_request" not in message.get("metadata", {})
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["typed_strategy_evaluation"]
    plan = plans[0]
    assert plan["status"] == "done"
    assert [step["status"] for step in plan["steps"]] == ["done"] * 3
    strategies = StrategyRepository(client.app.state.settings.db_path).list_for_task(
        task_id
    )
    assert len(strategies) == 1
    assert strategies[0].strategy_type == strategy_type
    backtests = StrategyRepository(
        client.app.state.settings.db_path
    ).list_backtests(strategies[0].id)
    assert len(backtests) == 1
    assert backtests[0].schema_version == "strategy.backtest.v2"
    assert backtests[0].strategy_type == strategy_type
    if strategy_type in {"limit", "pricing"}:
        assert backtests[0].economics["expected_loss"] > 0
    artifacts = StrategyRepository(
        client.app.state.settings.db_path
    ).list_strategy_artifacts(strategies[0].id)
    assert [artifact["kind"] for artifact in artifacts] == ["strategy_doc_md"]
    assert len(llm.calls) == 1


def test_hallucinated_strategy_column_clarifies_without_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval", field="ghost_score"),
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "回测这份审批策略"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    last = _last_assistant(response.json()["messages"])
    assert "ghost_score" in last["content"]
    assert "strategy_request" not in last["metadata"]
    assert len(llm.calls) == 2
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


@pytest.mark.parametrize(
    ("content", "payload", "template_id"),
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
                    "states": ["M0", "M1"],
                    "balance_col": "balance",
                    "observation_semantics": "adjacent_observation",
                },
            },
            "strategy_roll_rate_analysis",
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
        ),
    ],
)
def test_standard_workflow_requests_auto_run_trusted_templates_without_target(
    tmp_path: Path,
    monkeypatch,
    content: str,
    payload: dict,
    template_id: str,
) -> None:
    client = TestClient(create_app(tmp_path / template_id))
    task_id = _standard_workflow_task(client, tmp_path / template_id)
    llm = _FakeLLM(payload)
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": content},
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [template_id]
    assert plans[0]["status"] == "done"
    assert all(step["status"] == "done" for step in plans[0]["steps"])
    assert all(
        "strategy_request" not in message.get("metadata", {})
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert len(llm.calls) == 1


def test_target_column_is_never_available_to_llm_authored_strategy_rules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval", field="bad"),
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "回测这份审批策略"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert "bad" in _last_assistant(response.json()["messages"])["content"]
    assert len(llm.calls) == 2
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_unrelated_turn_does_not_invoke_strategy_compiler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM({"operation": "backtest", "strategy_type": "approval"})
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "你好"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert llm.calls == []
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_data_question_is_not_hijacked_by_strategy_compiler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    _register_task_source(
        client,
        task_id,
        tmp_path / "strategy-source" / "sample.csv",
    )
    llm = _FakeLLM(
        {
            "group_by": ["x"],
            "metrics": [{"op": "bad_rate", "col": "bad"}],
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "按渠道看坏率"},
    )

    assert response.status_code == 202, response.text
    metadata = _last_assistant(response.json()["messages"])["metadata"]
    assert "adhoc_spec" in metadata
    assert "strategy_request" not in metadata
    assert len(llm.calls) == 1


def test_strategy_metric_question_wins_over_adhoc_routing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    _register_task_source(
        client,
        task_id,
        tmp_path / "strategy-source" / "sample.csv",
    )
    strategy_id = _saved_strategy(client, task_id)
    llm = _FakeLLM(
        {
            "operation": "analyze",
            "strategy_type": "approval",
            "strategy_id": strategy_id,
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "看一下这套策略的通过率"},
    )

    assert response.status_code == 202, response.text
    assert all(
        "adhoc_spec" not in message.get("metadata", {})
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    )
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "stored_strategy_evaluation"
    ]
    assert plans[0]["status"] == "done"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("content", ["分析失败原因", "这个规则什么意思"])
def test_action_or_subject_alone_does_not_invoke_strategy_compiler(
    tmp_path: Path,
    monkeypatch,
    content: str,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM({"operation": "backtest", "strategy_type": "approval"})
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": content},
    )

    assert response.status_code == 202, response.text
    assert llm.calls == []
    assert "strategy_request" not in _last_assistant(
        response.json()["messages"]
    )["metadata"]


def test_legacy_confirmation_releases_claim_after_driver_start_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)
    record = _seed_legacy_pending(
        client,
        task_id,
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        },
        target_col="bad",
    )
    calls = 0
    from marvis.agent import turn_handlers

    original_start = turn_handlers._start_confirmed_strategy_plan

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DriverError("simulated start failure")
        return original_start(*args, **kwargs)

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._start_confirmed_strategy_plan",
        fail_once,
    )

    first = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )
    assert first.status_code == 409, first.text
    pending_repo = PendingStrategyRequestRepository(client.app.state.settings.db_path)
    assert pending_repo.get(task_id, record.id).status == "pending"
    second = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )
    assert second.status_code == 202, second.text
    assert calls == 2
    assert pending_repo.get(task_id, record.id).status == "consumed"
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 1


def test_target_guard_is_explicitly_safe_when_preview_is_none(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "develop",
            "strategy_type": "approval",
            "objective": "max_approval",
            "max_bad_rate": 0.08,
        }
    )
    _install_llm(monkeypatch, llm)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_dataset_preview",
        lambda runtime, task: None,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_request_requires_dataset",
        lambda draft: False,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开发审批策略并最大化通过率"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == "strategy_target_context_required"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_cancel_discards_pending_draft_without_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)
    record = _seed_legacy_pending(
        client,
        task_id,
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        },
        target_col="bad",
    )
    assert DatasetRepository(
        client.app.state.settings.db_path
    ).list_datasets(task_id) == []
    pending_message = _last_assistant(
        TaskRepository(client.app.state.settings.db_path).list_agent_messages(task_id)
    )
    pending_ref = record.to_metadata_reference()
    assert set(pending_ref) == {"request_id", "payload_sha256"}
    assert "strategy_spec" not in json.dumps(
        pending_message["metadata"],
        ensure_ascii=False,
    )
    cancelled = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "先别执行"},
    )

    assert cancelled.status_code == 202, cancelled.text
    assert "没有创建计划" in _last_assistant(cancelled.json()["messages"])["content"]
    assert llm.calls == []
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert StrategyRepository(client.app.state.settings.db_path).list_for_task(task_id) == []
    assert PendingStrategyRequestRepository(
        client.app.state.settings.db_path
    ).get(task_id, pending_ref["request_id"]).status == "cancelled"


def test_rephrasing_invalidates_old_pending_strategy_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)
    record = _seed_legacy_pending(
        client,
        task_id,
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        },
        target_col="bad",
    )
    pending_ref = record.to_metadata_reference()

    replaced = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "先换个话题"},
    )

    assert replaced.status_code == 202, replaced.text
    record = PendingStrategyRequestRepository(
        client.app.state.settings.db_path
    ).get(task_id, pending_ref["request_id"])
    assert record.status == "invalidated"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_registered_dataset_mutation_invalidates_pending_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    source_path = tmp_path / "strategy-source" / "sample.csv"
    _register_task_source(client, task_id, source_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = DatasetRepository(settings.db_path).list_datasets(task_id)[0]
    registered_path = registry.resolve_path(dataset.id)
    record = _seed_legacy_pending(
        client,
        task_id,
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        },
        dataset_identity={
            "kind": "registered",
            "dataset_id": dataset.id,
            "content_hash": sha256_file(registered_path),
        },
        target_col="bad",
    )
    pending_ref = record.to_metadata_reference()
    mutated = DataBackend(settings.datasets_dir).read_frame(registered_path)
    mutated.loc[0, "x"] = 999
    mutated.to_parquet(registered_path, index=False)

    confirmed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )

    assert confirmed.status_code == 202, confirmed.text
    assert confirmed.json()["status"] == "clarification_required"
    assert confirmed.json()["code"] == "strategy_dataset_context_changed"
    record = PendingStrategyRequestRepository(settings.db_path).get(
        task_id,
        pending_ref["request_id"],
    )
    assert record.status == "invalidated"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_legacy_pending_same_schema_replacement_during_binding_is_not_consumed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    source_path = tmp_path / "strategy-source" / "sample.csv"
    _register_task_source(client, task_id, source_path)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = DatasetRepository(settings.db_path).list_datasets(task_id)[0]
    registered_path = registry.resolve_path(dataset.id)
    record = _seed_legacy_pending(
        client,
        task_id,
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        },
        dataset_identity={
            "kind": "registered",
            "dataset_id": dataset.id,
            "content_hash": sha256_file(registered_path),
        },
        target_col="bad",
    )
    llm = _FakeLLM({})
    _install_llm(monkeypatch, llm)

    from marvis.agent import turn_handlers

    original_context = turn_handlers._strategy_dataset_context
    mutated = False

    def replace_same_schema(runtime, task, *, require_target=True):
        nonlocal mutated
        if not mutated:
            frame = DataBackend(settings.datasets_dir).read_frame(registered_path)
            frame.loc[0, "x"] = 999
            frame.to_parquet(registered_path, index=False)
            mutated = True
        return original_context(runtime, task, require_target=require_target)

    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_dataset_context",
        replace_same_schema,
    )

    confirmed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )

    assert confirmed.status_code == 202, confirmed.text
    assert confirmed.json()["status"] == "clarification_required"
    assert confirmed.json()["code"] == "strategy_dataset_context_changed"
    stored = PendingStrategyRequestRepository(settings.db_path).get(
        task_id,
        record.id,
    )
    assert stored.status == "invalidated"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert llm.calls == []


def test_explicit_preview_only_strategy_request_never_compiles_or_executes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "请回测审批策略，但只预览不要执行"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "preview_only"
    assert response.json()["code"] == "strategy_execution_not_authorized"
    assert llm.calls == []
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert DatasetRepository(
        client.app.state.settings.db_path
    ).list_datasets(task_id) == []
    assert StrategyRepository(
        client.app.state.settings.db_path
    ).list_for_task(task_id) == []


def test_nan_target_requires_explicit_drop_then_threads_consent_into_workflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _nan_target_task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "回测这份审批策略"},
    )

    assert opened.status_code == 202, opened.text
    assert opened.json()["status"] == "clarification_required"
    assert opened.json()["code"] == (
        "strategy_drop_nan_labels_confirmation_required"
    )
    assert opened.json()["label_quality"] == {
        "target_col": "bad",
        "n_total": 6,
        "n_nan": 1,
    }
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    last = _last_assistant(opened.json()["messages"])
    assert "strategy_request" not in last["metadata"]
    assert "strategy_nan_label_confirmation" in last["metadata"]

    ambiguous = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )
    assert ambiguous.status_code == 202, ambiguous.text
    assert ambiguous.json()["code"] == (
        "strategy_drop_nan_labels_confirmation_required"
    )
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认丢弃空标签并继续"},
    )

    assert resumed.status_code == 202, resumed.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["typed_strategy_evaluation"]
    assert plans[0]["status"] == "done"
    backtest_step = next(
        step for step in plans[0]["steps"] if step["title"] == "回测类型化策略"
    )
    assert backtest_step["inputs"]["drop_nan_labels"] is True
    output = client.app.state.plan_repo.load_step_output(backtest_step["id"])
    assert output["nan_labels_dropped"] == 1
    assert output["label_coverage"] == pytest.approx(5 / 6)
    assert len(llm.calls) == 1


def test_nan_target_confirmation_is_invalidated_by_dataset_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _nan_target_task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "backtest",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)
    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "回测这份审批策略"},
    )
    assert opened.status_code == 202, opened.text
    assert opened.json()["code"] == (
        "strategy_drop_nan_labels_confirmation_required"
    )

    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = DatasetRepository(settings.db_path).list_datasets(task_id)[0]
    path = registry.resolve_path(dataset.id)
    mutated = DataBackend(settings.datasets_dir).read_frame(path)
    mutated.loc[0, "x"] = 999
    mutated.to_parquet(path, index=False)

    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认丢弃空标签并继续"},
    )

    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["status"] == "clarification_required"
    assert resumed.json()["code"] == "strategy_dataset_context_changed"
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert len(llm.calls) == 1


def test_approval_development_contract_persists_and_auto_runs_to_adoption_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "develop",
            "strategy_type": "approval",
            "objective": "max_approval",
            "max_bad_rate": 0.20,
        }
    )
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开发审批策略，最大坏账率 20%"},
    )
    assert opened.status_code == 202, opened.text
    strategy_input = client.get(f"/api/tasks/{task_id}").json()["strategy_input"]
    assert strategy_input["strategy_type"] == "approval"
    assert strategy_input["objective"] == "max_approval"
    assert strategy_input["max_bad_rate"] == 0.20
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[0]["template_id"] == "strategy_development"
    assert plans[0]["success_criteria"] == [
        {"metric": "approved_bad_rate", "max": 0.20}
    ]
    assert plans[0]["status"] == "awaiting_confirm"
    awaiting = [
        step for step in plans[0]["steps"] if step["status"] == "awaiting_confirm"
    ]
    assert [step["title"] for step in awaiting] == ["采纳策略"]


@pytest.mark.parametrize(
    ("operation", "content", "template_id"),
    [
        ("backtest", "回测已有审批策略", "stored_strategy_evaluation"),
        ("apply", "应用已有审批策略", "stored_strategy_apply"),
        ("report", "生成已有审批策略报告", "stored_strategy_report"),
        ("adopt", "采纳已有审批策略", "stored_strategy_adoption"),
    ],
)
def test_existing_strategy_operations_route_to_dedicated_workflows(
    tmp_path: Path,
    monkeypatch,
    operation: str,
    content: str,
    template_id: str,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    strategy_id = _saved_strategy(client, task_id)
    payload = {
        "operation": operation,
        "strategy_type": "approval",
        "strategy_id": strategy_id,
    }
    if operation == "adopt":
        payload["adoption_reason"] = "回测证据满足经营约束且已完成业务复核"
    llm = _FakeLLM(payload)
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": content},
    )
    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [template_id]
    if operation == "adopt":
        assert plans[0]["status"] == "awaiting_confirm"
        assert [
            step["title"]
            for step in plans[0]["steps"]
            if step["status"] == "awaiting_confirm"
        ] == ["采纳已有策略"]
    else:
        assert plans[0]["status"] == "done"
        assert all(step["status"] == "done" for step in plans[0]["steps"])
    assert len(
        StrategyRepository(client.app.state.settings.db_path).list_for_task(task_id)
    ) == 1


@pytest.mark.parametrize(
    ("strategy_type", "economics_inputs"),
    [
        (
            "limit",
            {
                "pd_col": "pd",
                "lgd_value": 0.5,
                "utilization_value": 0.6,
            },
        ),
        (
            "pricing",
            {
                "ead_col": "ead",
                "pd_col": "pd",
                "lgd_value": 0.5,
                "funding_rate_value": 0.03,
                "term_months_value": 12,
                "operating_cost_per_loan_value": 10,
            },
        ),
        ("segmentation", None),
    ],
)
def test_typed_stored_adoption_routes_type_specific_evidence_inputs(
    tmp_path: Path,
    monkeypatch,
    strategy_type: str,
    economics_inputs: dict | None,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    strategy_id = _saved_strategy(
        client,
        task_id,
        strategy_type=strategy_type,
    )
    payload = {
        "operation": "adopt",
        "strategy_type": strategy_type,
        "strategy_id": strategy_id,
        "adoption_reason": "业务委员会已复核类型化回测证据并批准本地采纳",
    }
    if economics_inputs is not None:
        payload["economics_inputs"] = economics_inputs
    llm = _FakeLLM(payload)
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"采纳这份 {strategy_type} 策略"},
    )
    assert opened.status_code == 202, opened.text
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][0]
    assert plan["template_id"] == "stored_strategy_adoption"
    assert plan["status"] == "awaiting_confirm"
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][1]["status"] == "awaiting_confirm"
    backtest_inputs = plan["steps"][0]["inputs"]
    if economics_inputs is None:
        assert "economics_inputs" not in backtest_inputs
    else:
        assert backtest_inputs["economics_inputs"] == economics_inputs


@pytest.mark.e2e
def test_apply_existing_strategy_to_unlabeled_sample_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _unlabeled_task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "apply",
            "strategy_type": "approval",
            "strategy_spec": _spec("approval"),
        }
    )
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "把这个审批策略应用到当前生产样本并输出逐行结果"},
    )
    assert opened.status_code == 202, opened.text
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][0]
    assert plan["template_id"] == "typed_strategy_apply"
    assert plan["status"] == "done"
    assert [step["status"] for step in plan["steps"]] == ["done"] * 3
    datasets = DatasetRepository(
        client.app.state.settings.db_path
    ).list_datasets(task_id)
    assert [dataset.role for dataset in datasets].count("strategy.applied") == 1
    assert len(
        StrategyRepository(client.app.state.settings.db_path).list_for_task(task_id)
    ) == 1
    assert len(llm.calls) == 1


def test_compare_existing_strategy_requires_owned_same_type_baseline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    strategy_id = _saved_strategy(client, task_id, threshold=0)
    baseline_id = _saved_strategy(client, task_id, threshold=2)
    llm = _FakeLLM(
        {
            "operation": "compare",
            "strategy_type": "approval",
            "strategy_id": strategy_id,
            "baseline_strategy_id": baseline_id,
        }
    )
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "对比候选审批策略和基线策略"},
    )
    assert opened.status_code == 202, opened.text
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][0]
    assert plan["template_id"] == "stored_strategy_evaluation"
    assert plan["status"] == "done"
    assert plan["steps"][0]["inputs"]["baseline_strategy_id"] == baseline_id
