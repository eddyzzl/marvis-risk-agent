"""V3 D-7: collections requests in conversation point to the strategy task's
collections workspace, say plainly what is out of reach, and create nothing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.turn_handlers.collections_guidance import (
    guidance_text,
    is_collections_request,
    maybe_handle_collections_request,
    unsupported_parts,
)
from marvis.app import create_app


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("collections guidance must not call the LLM")


def _install(monkeypatch, client) -> None:
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: client,
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda _request, _task: client,
    )


def _materials(root: Path) -> Path:
    source = root / "materials"
    source.mkdir()
    pd.DataFrame({"apply_id": ["A1", "A2"], "score": [600, 700], "bad_flag": [1, 0]}).to_csv(
        source / "sample.csv", index=False
    )
    return source


def _task(client: TestClient, source: Path, task_type: str, **extra) -> str:
    response = client.post(
        "/api/tasks",
        json={
            "model_name": f"collections-{task_type}",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": task_type,
            "run_mode": "agent",
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("给逾期客户做一版催收分案规则", True),
        ("Set up a collections strategy for M1 accounts", True),
        ("帮我把这批逾期客户做催收分案", True),
        ("优化一下催收", True),
        ("debt collection for M1 accounts", True),
        ("Set up a collections queue", True),
        # Bare 贷后管理 / "collection", and modeling, risk or feature work that
        # only mentions collections, keep their own routing.
        ("贷后管理怎么分案", False),
        ("训练一个预测催收回款率的模型", False),
        ("贷后管理口径下做vintage分析", False),
        ("feature collection for the model", False),
        ("data collection plan", False),
        ("催收策略里的特征怎么选", False),
        # A profit field and a C-card model keep their own workflows.
        ("收益测算，催收成本率 3%", False),
        ("做一个催收评分卡模型", False),
        ("看一下各月 Vintage 坏账曲线", False),
    ],
)
def test_collections_words_are_recognised_without_taking_profit_or_model_requests(text, expected):
    assert is_collections_request(text) is expected


def test_guidance_names_the_entry_and_says_plainly_what_is_out_of_reach():
    strategy = guidance_text("strategy", "做一套催收策略，自动外呼并预测回收率")
    assert "催收参考工作台" in strategy and "检查器" in strategy
    assert "现在做不到：联系客户（外呼、短信、话术）、自动优化催收动作或预测、保证回收效果。" in strategy
    assert "没有创建计划" in strategy
    other = guidance_text("modeling", "给逾期客户做一版催收分案规则")
    assert "按策略任务创建" in other and "现在做不到" not in other
    english = guidance_text("strategy", "Build a collections strategy and call customers automatically")
    assert "Collections reference workspace" in english
    assert "contacting customers" in english
    assert not any("一" <= char <= "鿿" for char in english)
    assert unsupported_parts("给逾期客户做一版催收分案规则") == []


def test_collections_request_in_a_strategy_task_points_to_its_workspace_without_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recording = _RecordingClient()
    _install(monkeypatch, recording)
    with TestClient(create_app(tmp_path)) as client:
        task_id = _task(client, _materials(tmp_path), "strategy")
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "开始策略分析。我的目标：给逾期客户做一版催收分案规则，自动发短信", "model_id": "t"},
        )
        assert response.status_code == 202, response.text
        reply = [m for m in response.json()["messages"] if m["role"] == "assistant"][-1]
        assert reply["metadata"]["code"] == "collections_entry_guidance"
        assert reply["metadata"]["entry"] == "collectionWorkspacePanel"
        # A display-only quick action opens the workspace; it carries no command.
        assert reply["metadata"]["inspector_actions"] == [
            {"label": "打开催收参考工作台", "item": "collectionWorkspacePanel"}
        ]
        assert reply["metadata"]["unsupported"] == ["联系客户（外呼、短信、话术）"]
        assert "催收参考工作台" in reply["content"]
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
        assert recording.calls == []
        english = client.get(
            f"/api/tasks/{task_id}/agent/messages", headers={"x-marvis-ui-locale": "en"}
        )
        assert english.status_code == 200, english.text
        payload = english.json()
        messages = payload["messages"] if isinstance(payload, dict) else payload
        last = [m for m in messages if m["role"] == "assistant"][-1]
        assert last["metadata"]["title"] == "Collections request"
        assert last["metadata"]["inspector_actions"][0]["label"] == "Open Collections reference workspace"


def test_collections_request_in_another_task_says_it_lives_under_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    recording = _RecordingClient()
    _install(monkeypatch, recording)
    with TestClient(create_app(tmp_path)) as client:
        task_id = _task(client, _materials(tmp_path), "data_join")
        response = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "帮我把这批逾期客户做催收分案", "model_id": "t"},
        )
        assert response.status_code == 202, response.text
        reply = [m for m in response.json()["messages"] if m["role"] == "assistant"][-1]
        assert reply["metadata"]["entry"] == "strategy_task"
        assert "inspector_actions" not in reply["metadata"]
        assert "按策略任务创建" in reply["content"]
        assert client.app.state.plan_repo.list_plans_for_task(task_id) == []
        assert recording.calls == []


class _Repo:
    def __init__(self) -> None:
        self.added: list[dict] = []

    def add_agent_message(self, task_id, **kwargs):
        self.added.append(kwargs)

    def list_agent_messages(self, task_id):
        return list(self.added)


@pytest.mark.parametrize(
    "task_type,answered",
    [("strategy", True), ("data_join", True), ("modeling", False), ("feature_analysis", False),
     ("vintage", False), ("portfolio", False), ("validation", False)],
)
def test_only_strategy_and_data_processing_tasks_take_collections_turns(task_type, answered):
    from types import SimpleNamespace

    repo = _Repo()
    task = SimpleNamespace(id="t1", task_type=task_type)
    result = maybe_handle_collections_request(repo, task, text="给逾期客户做一版催收分案规则")
    assert (result is not None) is answered
    assert bool(repo.added) is answered
