from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient
import pytest

from marvis.agent.turn_handlers import is_portfolio_deterministic_turn
from marvis.app import create_app


def _portfolio_task(
    client: TestClient,
    tmp_path: Path,
    *,
    run_mode: str = "manual",
) -> str:
    source = tmp_path / "workspace" / "portfolio-source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "loan_id": ["a", "a", "b", "b"],
            "snapshot_month": ["2025-01", "2025-02", "2025-01", "2025-02"],
            "bucket": ["current", "M1", "M1", "charged_off"],
            "balance": [100.0, 90.0, 200.0, 180.0],
            "segment": ["new", "new", "existing", "existing"],
        }
    ).to_csv(source / "performance.csv", index=False)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "组合分析",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "portfolio",
            "run_mode": run_mode,
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_portfolio_agent_requires_typed_economics_then_reaches_human_state_gate(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _portfolio_task(client, tmp_path)

    started = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert started.status_code == 202, started.text
    assert started.json()["messages"][-1]["metadata"]["kind"] == (
        "portfolio_setup_required"
    )
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    configured = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "使用已确认的组合口径",
            "portfolio_request": {
                "id_col": "loan_id",
                "snapshot_col": "snapshot_month",
                "bucket_col": "bucket",
                "balance_col": "balance",
                "segment_col": "segment",
                "loss_state": "charged_off",
                "lgd": 0.45,
                "horizon_months": 18,
            },
        },
    )

    assert configured.status_code == 202, configured.text
    gate = configured.json()["messages"][-1]
    assert gate["metadata"]["kind"] == "gate"
    confirmed_content_hash = gate["metadata"]["portfolio_states"][
        "dataset_content_hash"
    ]
    assert len(confirmed_content_hash) == 64
    assert set(confirmed_content_hash) <= set("0123456789abcdef")
    assert gate["metadata"]["portfolio_states"]["loss_state"] == "charged_off"
    assert gate["metadata"]["portfolio_states"]["lgd"] == 0.45
    assert gate["metadata"]["portfolio_states"]["horizon_months"] == 18
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    confirmed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认"},
    )

    assert confirmed.status_code == 202, confirmed.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert plans[-1]["template_id"] == "portfolio_analysis_no_trend"
    assert plans[-1]["status"] == "validated"
    plan = client.get(f"/api/plans/{plans[-1]['id']}").json()["plan"]
    dataset_reader_steps = [
        step
        for step in plan["steps"]
        if step["tool_ref"]["tool"]
        in {
            "flow_rate",
            "bucket_migration",
            "segment_profile",
            "expected_loss_estimate",
        }
    ]
    assert len(dataset_reader_steps) == 4
    assert {
        step["inputs"]["expected_content_hash"] for step in dataset_reader_steps
    } == {confirmed_content_hash}


def test_portfolio_typed_request_is_task_scoped_and_strict(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _portfolio_task(client, tmp_path)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "配置组合分析",
            "portfolio_request": {
                "id_col": "loan_id",
                "snapshot_col": "snapshot_month",
                "bucket_col": "bucket",
                "balance_col": "balance",
                "segment_col": "segment",
                "loss_state": "charged_off",
                "lgd": 0.45,
                "horizon_months": 18,
                "unexpected": "must fail",
            },
        },
    )

    assert response.status_code == 422


def test_portfolio_execution_fails_closed_when_confirmed_dataset_bytes_change(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(tmp_path / "workspace")) as client:
        task_id = _portfolio_task(client, tmp_path)
        configured = _typed_portfolio_setup(client, task_id)
        assert configured.status_code == 202, configured.text
        confirmed_hash = configured.json()["messages"][-1]["metadata"][
            "portfolio_states"
        ]["dataset_content_hash"]

        confirmed = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "确认"},
        )
        assert confirmed.status_code == 202, confirmed.text
        plan_summary = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
        plan_id = plan_summary["id"]
        bound_plan = client.get(f"/api/plans/{plan_id}").json()["plan"]
        assert {
            step["inputs"]["expected_content_hash"]
            for step in bound_plan["steps"]
            if step["tool_ref"]["tool"]
            in {
                "flow_rate",
                "bucket_migration",
                "segment_profile",
                "expected_loss_estimate",
            }
        } == {confirmed_hash}

        registered_files = list(
            (tmp_path / "workspace" / "datasets").rglob("*.parquet")
        )
        assert len(registered_files) == 1
        pd.DataFrame(
            {
                "loan_id": ["replacement-a", "replacement-b"],
                "snapshot_month": ["2025-01", "2025-02"],
                "bucket": ["charged_off", "charged_off"],
                "balance": [999_999.0, 999_999.0],
                "segment": ["replacement", "replacement"],
            }
        ).to_parquet(registered_files[0], index=False)

        started = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "开始"},
        )
        assert started.status_code == 202, started.text
        failed_plan = client.get(f"/api/plans/{plan_id}").json()["plan"]

        assert failed_plan["status"] == "failed"
        failed_readers = [
            step
            for step in failed_plan["steps"]
            if step["tool_ref"]["tool"]
            in {
                "flow_rate",
                "bucket_migration",
                "segment_profile",
                "expected_loss_estimate",
            }
            and step["status"] == "failed"
        ]
        assert failed_readers
        assert all(step.get("output") is None for step in failed_readers)


def test_portfolio_human_confirmation_rejects_dataset_changed_after_gate(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(tmp_path / "workspace")) as client:
        task_id = _portfolio_task(client, tmp_path)
        configured = _typed_portfolio_setup(client, task_id)
        assert configured.status_code == 202, configured.text
        assert configured.json()["messages"][-1]["metadata"]["portfolio_states"]

        registered_files = list(
            (tmp_path / "workspace" / "datasets").rglob("*.parquet")
        )
        assert len(registered_files) == 1
        pd.DataFrame(
            {
                "loan_id": ["replacement"],
                "snapshot_month": ["2025-01"],
                "bucket": ["charged_off"],
                "balance": [999_999.0],
                "segment": ["replacement"],
            }
        ).to_parquet(registered_files[0], index=False)

        confirmed = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "确认"},
        )

        assert confirmed.status_code == 202, confirmed.text
        assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
        latest = confirmed.json()["messages"][-1]
        assert latest["metadata"]["error"] is True
        assert "内容" in latest["content"] or "快照" in latest["content"]


def _typed_portfolio_setup(client: TestClient, task_id: str):
    return client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "使用已确认的组合口径",
            "portfolio_request": {
                "id_col": "loan_id",
                "snapshot_col": "snapshot_month",
                "bucket_col": "bucket",
                "balance_col": "balance",
                "segment_col": "segment",
                "loss_state": "charged_off",
                "lgd": 0.45,
                "horizon_months": 18,
            },
        },
    )


@pytest.mark.parametrize("reply", ["确认", "current,M1,charged_off"])
def test_agent_mode_portfolio_gate_confirmation_does_not_require_llm(
    tmp_path: Path,
    reply: str,
) -> None:
    with TestClient(create_app(tmp_path / "workspace")) as client:
        task_id = _portfolio_task(
            client,
            tmp_path,
            run_mode="agent",
        )
        configured = _typed_portfolio_setup(client, task_id)
        assert configured.status_code == 202, configured.text
        assert configured.json()["messages"][-1]["metadata"]["portfolio_states"]

        confirmed = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": reply},
        )

        assert confirmed.status_code == 202, confirmed.text
        plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        assert [(item["template_id"], item["status"]) for item in plans] == [
            ("portfolio_analysis_no_trend", "validated")
        ]


@pytest.mark.parametrize(
    ("instruction", "review_overrides", "should_create"),
    [
        ("这个桶顺序没问题，就按它跑。", {}, True),
        ("这个桶顺序没问题吗？", {"is_question": True}, False),
        ("如果损失态没问题就按它跑。", {"is_conditional": True}, False),
    ],
)
def test_agent_mode_portfolio_accepts_only_two_pass_semantic_state_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
    review_overrides: dict,
    should_create: bool,
) -> None:
    class _SemanticClient:
        route_calls = 0
        review_calls = 0

        def complete(self, *_args, **kwargs):
            if kwargs.get("prompt_name") == "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS":
                self.review_calls += 1
                reviewed_instruction = json.loads(kwargs["user_prompt"])["instruction"]
                payload = {
                    "verdict": "authorize",
                    "evidence_quote": reviewed_instruction,
                    "reason": "用户明确接受当前完整桶顺序。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_authorization": False,
                }
                payload.update(review_overrides)
                return json.dumps(payload, ensure_ascii=False)
            self.route_calls += 1
            return json.dumps(
                {
                    "action": "confirm",
                    "params": {},
                    "constraint": "",
                    "reason": "用户明确接受当前完整桶顺序。",
                    "confidence": "high",
                    "explicit_authorization": True,
                },
                ensure_ascii=False,
            )

    semantic_client = _SemanticClient()
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: semantic_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: semantic_client,
    )
    with TestClient(create_app(tmp_path / "workspace")) as client:
        task_id = _portfolio_task(client, tmp_path, run_mode="agent")
        configured = _typed_portfolio_setup(client, task_id)
        assert configured.status_code == 202, configured.text

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": instruction},
        )

        assert response.status_code == 202, response.text
        plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        assert bool(plans) is should_create
        receipts = [
            message
            for message in response.json()["messages"]
            if message.get("metadata", {}).get("intent")
            == "portfolio_semantic_authorization"
        ]
        assert bool(receipts) is should_create
        if should_create:
            assert plans[-1]["template_id"] == "portfolio_analysis_no_trend"
            assert plans[-1]["status"] == "validated"
        else:
            assert response.json()["messages"][-1]["metadata"]["portfolio_states"]
    assert semantic_client.route_calls == 1
    assert semantic_client.review_calls == 1


def test_agent_mode_portfolio_free_question_at_setup_gate_still_requires_llm(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(tmp_path / "workspace")) as client:
        task_id = _portfolio_task(
            client,
            tmp_path,
            run_mode="agent",
        )
        configured = _typed_portfolio_setup(client, task_id)
        assert configured.status_code == 202, configured.text

        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "为什么要按这个桶顺序？"},
        )

        assert response.status_code == 409, response.text
        assert "大模型" in response.json()["detail"]
        assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_portfolio_plan_gate_bypass_is_limited_to_explicit_confirmation() -> None:
    task = SimpleNamespace(id="task-1", task_type="portfolio")
    repo = SimpleNamespace(
        list_agent_messages=lambda _task_id: [
            {
                "role": "assistant",
                "metadata": {"kind": "plan_overview"},
            }
        ]
    )
    plan_repo = SimpleNamespace(
        list_plans_for_task=lambda _task_id: [
            SimpleNamespace(
                id="plan-1",
                template_id="portfolio_analysis_no_trend",
                status="validated",
            )
        ]
    )

    assert is_portfolio_deterministic_turn(
        repo,
        plan_repo,
        task,
        "开始",
    )
    assert not is_portfolio_deterministic_turn(
        repo,
        plan_repo,
        task,
        "请解释一下这个计划为什么合理？",
    )
