from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app


class _WorkflowIntakeClient:
    """Expose the old generic-steal bug while supporting the owned workflow route."""

    def __init__(
        self,
        *,
        task_type: str,
        route: dict | None = None,
        mutation=None,
    ):
        self.task_type = task_type
        self.route = route
        self.mutation = mutation
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        caller = str(kwargs.get("caller") or "")
        if caller.startswith("semantic_intent_"):
            instruction = str(json.loads(kwargs["user_prompt"])["instruction"])
            return json.dumps(
                {
                    "intent": "dataset_analysis",
                    "evidence_quote": instruction,
                    "reason": "将整句误解为通用样本分析。",
                    "confidence": "high",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_action": False,
                },
                ensure_ascii=False,
            )
        if caller == "router":
            if self.mutation is not None:
                mutation, self.mutation = self.mutation, None
                mutation()
            if self.route is not None:
                return json.dumps(self.route, ensure_ascii=False)
            if self.task_type == "modeling":
                return json.dumps(
                    {
                        "action": "adjust",
                        "params": {
                            "target_type": "binary",
                            "recipes": ["lr"],
                            "n_trials": 1,
                        },
                        "constraint": "",
                        "reason": "用户明确要求进入建模并给出了首轮建模规格。",
                        "confidence": "high",
                        "explicit_authorization": False,
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "action": "confirm",
                    "params": {},
                    "constraint": "",
                    "reason": "用户明确要求进入当前特征分析工作流。",
                    "confidence": "high",
                    "explicit_authorization": True,
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected LLM caller: {caller}")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _source_dir(tmp_path: Path, *, task_type: str) -> Path:
    source = tmp_path / f"{task_type}_owned_workflow"
    source.mkdir()
    rows = 120
    pd.DataFrame(
        {
            "application_id": np.arange(rows),
            "signal": np.linspace(-1.0, 1.0, rows),
            "bad_flag": np.resize([0, 1], rows),
        }
    ).to_parquet(source / "sample.parquet")
    return source


def _create_task(
    client: TestClient,
    tmp_path: Path,
    *,
    task_type: str,
) -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": f"{task_type} 专属首轮路由",
            "validator": "qa",
            "source_dir": str(_source_dir(tmp_path, task_type=task_type)),
            "task_type": task_type,
            "run_mode": "agent",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    decision_client: _WorkflowIntakeClient,
) -> None:
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: decision_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: decision_client,
    )


@pytest.mark.parametrize(
    ("task_type", "instruction"),
    [
        (
            "modeling",
            "请分析当前样本并建立二分类模型，bad_flag 是目标列，使用逻辑回归。",
        ),
        (
            "feature_analysis",
            "请分析当前样本的特征表现，bad_flag 是目标列，进入特征分析工作流。",
        ),
    ],
)
def test_specialized_first_turn_cannot_be_stolen_by_generic_dataset_analysis(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    instruction: str,
):
    decision_client = _WorkflowIntakeClient(task_type=task_type)
    _install_client(monkeypatch, decision_client)
    task_id = _create_task(client, tmp_path, task_type=task_type)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": instruction, "acceptance_mode": "normal"},
    )

    assert response.status_code == 202, response.text
    assert client.app.state.plan_repo.list_plans_for_task(task_id)
    callers = [str(call.get("caller") or "") for call in decision_client.calls]
    assert "router" in callers
    assert not any(caller.startswith("semantic_intent_") for caller in callers)


@pytest.mark.parametrize("task_type", ["modeling", "feature_analysis"])
@pytest.mark.parametrize(
    ("instruction", "route"),
    [
        (
            "现在是不是应该进入当前工作流？",
            {
                "action": "confirm",
                "params": {},
                "constraint": "",
                "reason": "用户是在询问，没有明确授权进入工作流。",
                "confidence": "medium",
                "explicit_authorization": False,
            },
        ),
        (
            "如果字段检查没问题，再进入当前工作流。",
            {
                "action": "confirm",
                "params": {},
                "constraint": "先完成字段检查",
                "reason": "用户的继续要求依赖尚未满足的条件。",
                "confidence": "high",
                "explicit_authorization": True,
            },
        ),
        (
            "不要进入当前工作流。",
            {
                "action": "clarify",
                "params": {},
                "constraint": "",
                "reason": "用户明确拒绝进入当前工作流。",
                "confidence": "high",
                "explicit_authorization": False,
            },
        ),
    ],
)
def test_specialized_first_turn_questions_conditions_and_refusals_fail_closed(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    instruction: str,
    route: dict,
):
    decision_client = _WorkflowIntakeClient(task_type=task_type, route=route)
    _install_client(monkeypatch, decision_client)
    task_id = _create_task(client, tmp_path, task_type=task_type)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": instruction, "acceptance_mode": "normal"},
    )

    assert response.status_code == 202, response.text
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
    last = [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]
    assert last["metadata"]["code"] == "semantic_intent_clarification"
    callers = [str(call.get("caller") or "") for call in decision_client.calls]
    assert callers == ["router"]


def test_specialized_first_turn_discards_llm_route_when_source_changes(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _source_dir(tmp_path, task_type="modeling")
    source_file = source / "sample.parquet"

    def mutate_source() -> None:
        rows = 120
        pd.DataFrame(
            {
                "application_id": np.arange(rows),
                "signal": np.linspace(1.0, -1.0, rows),
                "bad_flag": np.resize([1, 0], rows),
            }
        ).to_parquet(source_file)

    decision_client = _WorkflowIntakeClient(
        task_type="modeling",
        mutation=mutate_source,
    )
    _install_client(monkeypatch, decision_client)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "建模首轮源材料漂移阻断",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "modeling",
            "run_mode": "agent",
        },
    )
    assert response.status_code == 200, response.text
    task_id = str(response.json()["id"])

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "请分析当前样本并建立二分类模型，bad_flag 是目标列。",
            "acceptance_mode": "normal",
        },
    )

    assert response.status_code == 202, response.text
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
    last = [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]
    assert last["metadata"]["code"] == "semantic_intent_clarification"
    assert "任务状态已变化" in last["metadata"]["reason"]
