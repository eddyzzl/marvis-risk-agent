from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.semantic_intent import (
    INTENT_CURRENT_WORKFLOW,
    INTENT_DATASET_JOIN,
    INTENT_NONE,
    route_semantic_intent,
)
from marvis.app import create_app


def _join_materials(root: Path, *, table_count: int = 2) -> Path:
    source_dir = root / "join-materials"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "application_id": ["A1", "A2", "A3"],
            "bad_flag": [0, 1, 0],
        }
    ).to_csv(source_dir / "applications.csv", index=False)
    if table_count >= 2:
        pd.DataFrame(
            {
                "application_id": ["A1", "A2", "A3"],
                "bureau_score": [710, 640, 688],
            }
        ).to_csv(source_dir / "features.csv", index=False)
    return source_dir


def _intent_reply(
    instruction: str,
    *,
    intent: str = INTENT_DATASET_JOIN,
    **overrides,
) -> str:
    payload = {
        "intent": intent,
        "evidence_quote": instruction,
        "reason": "用户明确要求进入当前数据拼接探索。",
        "confidence": "high",
        "is_question": False,
        "is_conditional": False,
        "requests_change": False,
        "withholds_action": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class _SemanticJoinClient:
    def __init__(
        self,
        *,
        mutation: Callable[[], None] | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.requests: list[dict] = []
        self._mutation = mutation

    def complete(self, **kwargs):
        caller = kwargs.get("caller")
        self.calls.append(kwargs)
        if caller == "router":
            return json.dumps(
                {
                    "action": "clarify",
                    "params": {},
                    "constraint": "",
                    "reason": "当前任务已选择特征分析，不能改走数据拼接。",
                    "confidence": "high",
                    "explicit_authorization": False,
                },
                ensure_ascii=False,
            )
        assert caller in {"semantic_intent_router", "semantic_intent_reviewer"}
        request = json.loads(kwargs["user_prompt"])
        self.requests.append(request)
        if caller == "semantic_intent_reviewer" and self._mutation is not None:
            self._mutation()
        return _intent_reply(request["instruction"])


class _StaticSemanticClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _install_semantic_client(monkeypatch, semantic_client) -> None:
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: semantic_client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: semantic_client,
    )


def _create_agent_task(
    client: TestClient,
    *,
    source_dir: Path,
    task_type: str,
) -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": f"semantic-{task_type}",
            "validator": "qa",
            "source_dir": str(source_dir),
            "task_type": task_type,
            "run_mode": "agent",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _last_assistant(response) -> dict:
    return [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]


@pytest.mark.parametrize(
    "instruction",
    [
        "请读取这两张表，先告诉我哪张是申请主表、哪张是特征表，再让我确认关联键和目标列。",
        "先检查两份数据各自承担什么角色，给出主数据与补充变量表的建议，等我核对后再往下走。",
    ],
)
def test_agent_semantically_enters_join_exploration_and_typed_ui_stays_llm_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instruction: str,
) -> None:
    source_dir = _join_materials(tmp_path)
    semantic_client = _SemanticJoinClient()
    _install_semantic_client(monkeypatch, semantic_client)

    with TestClient(create_app(tmp_path)) as client:
        task_id = _create_agent_task(
            client,
            source_dir=source_dir,
            task_type="data_join",
        )

        explored = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": instruction, "model_id": "semantic-test"},
        )

        assert explored.status_code == 202, explored.text
        c1 = _last_assistant(explored)
        assert c1["metadata"].get("code") != "semantic_intent_clarification"
        assert "样本主表" in c1["content"]
        assert "join_c1" in c1["metadata"]
        assert {
            item["name"]: item["proposed_role"]
            for item in c1["metadata"]["join_c1"]["files"]
        } == {
            "applications.csv": "anchor",
            "features.csv": "feature",
        }
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
        assert [call["caller"] for call in semantic_client.calls] == [
            "semantic_intent_router",
            "semantic_intent_reviewer",
        ]
        for request in semantic_client.requests:
            assert request["task_type"] == "data_join"
            assert request["allowed_intents"].count(INTENT_DATASET_JOIN) == 1
            assert request["current_context"]["available_join_table_count"] == 2
            assert request["current_context"]["join_exploration_phase"] == (
                "role_discovery"
            )
            assert set(request["current_context"]["available_join_tables"]) == {
                "applications.csv",
                "features.csv",
            }

        confirmed = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "确认", "ui_action": "confirm_roles"},
        )
        assert confirmed.status_code == 202, confirmed.text
        plans = client.app.state.plan_repo.list_plans_for_task(task_id)
        assert len(plans) == 1
        overview = _last_assistant(confirmed)
        assert overview["metadata"]["kind"] == "plan_overview"

        started = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "开始执行拼接诊断",
                "ui_action": "start_plan",
                "expected_plan_id": plans[0].id,
                **overview["metadata"]["confirmation_snapshot"],
            },
        )
        assert started.status_code == 202, started.text
        c2 = _last_assistant(started)
        assert "拼接诊断完成" in c2["content"]
        assert [call["caller"] for call in semantic_client.calls] == [
            "semantic_intent_router",
            "semantic_intent_reviewer",
        ]


@pytest.mark.parametrize(
    ("instruction", "first_overrides", "second_overrides"),
    [
        ("哪张表更像申请主表？", {"is_question": True}, {"is_question": True}),
        (
            "如果两张表的键完全一致，就把它们关联起来",
            {"is_conditional": True},
            {"is_conditional": True},
        ),
        (
            "先别读取或分析这两张表",
            {"withholds_action": True},
            {"withholds_action": True},
        ),
        (
            "把当前样本主表换成 features.csv",
            {"requests_change": True},
            {"requests_change": True},
        ),
        (
            "开始看一下这两份数据",
            {"confidence": "medium"},
            {"confidence": "medium"},
        ),
        (
            "开始看一下这两份数据",
            {},
            {"intent": INTENT_CURRENT_WORKFLOW},
        ),
    ],
)
def test_dataset_join_intent_safety_signals_still_fail_closed(
    instruction: str,
    first_overrides: dict,
    second_overrides: dict,
) -> None:
    client = _StaticSemanticClient(
        [
            _intent_reply(instruction, **first_overrides),
            _intent_reply(instruction, **second_overrides),
        ]
    )

    decision = route_semantic_intent(
        client,
        task_type="data_join",
        instruction=instruction,
        context={
            "available_join_table_count": 2,
            "available_join_tables": ["applications.csv", "features.csv"],
        },
        allowed_intents=(
            INTENT_DATASET_JOIN,
            INTENT_CURRENT_WORKFLOW,
            INTENT_NONE,
        ),
    )

    assert decision.accepted is False
    assert decision.intent == INTENT_NONE
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("task_type", "table_count"),
    [
        ("feature_analysis", 2),
        ("data_join", 1),
    ],
)
def test_dataset_join_route_is_bounded_by_task_and_available_materials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_type: str,
    table_count: int,
) -> None:
    source_dir = _join_materials(tmp_path, table_count=table_count)
    semantic_client = _SemanticJoinClient()
    _install_semantic_client(monkeypatch, semantic_client)

    with TestClient(create_app(tmp_path)) as client:
        task_id = _create_agent_task(
            client,
            source_dir=source_dir,
            task_type=task_type,
        )
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "请先识别这些表的角色，再进入数据关联。",
                "model_id": "semantic-test",
            },
        )

        assert response.status_code == 202, response.text
        last = _last_assistant(response)
        assert last["metadata"]["code"] == "semantic_intent_clarification"
        if task_type == "feature_analysis":
            assert [call["caller"] for call in semantic_client.calls] == ["router"]
            assert semantic_client.requests == []
        else:
            assert INTENT_DATASET_JOIN not in semantic_client.requests[0][
                "allowed_intents"
            ]
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


def test_join_source_material_change_during_two_pass_review_discards_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = _join_materials(tmp_path)
    feature_path = source_dir / "features.csv"
    mutated = False

    def mutate_source() -> None:
        nonlocal mutated
        if mutated:
            return
        feature_path.write_text(
            feature_path.read_text(encoding="utf-8") + "A4,699\n",
            encoding="utf-8",
        )
        mutated = True

    semantic_client = _SemanticJoinClient(mutation=mutate_source)
    _install_semantic_client(monkeypatch, semantic_client)

    with TestClient(create_app(tmp_path)) as client:
        task_id = _create_agent_task(
            client,
            source_dir=source_dir,
            task_type="data_join",
        )
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={
                "content": "请先识别两张表的角色并准备关联方案。",
                "model_id": "semantic-test",
            },
        )

        assert response.status_code == 202, response.text
        assert mutated is True
        last = _last_assistant(response)
        assert last["metadata"]["code"] == "semantic_intent_clarification"
        assert "状态已变化" in last["metadata"]["reason"]
        assert "join_c1" not in last["metadata"]
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
