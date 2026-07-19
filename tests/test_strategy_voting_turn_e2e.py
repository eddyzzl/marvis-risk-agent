"""Natural-language Voting build and explicit Pool admission vertical."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from tests.test_strategy_voting_candidate_tool import _setup


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_voting_build_then_explicit_pool_add(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    client = TestClient(create_app(fx["settings"].workspace))
    task_id = fx["task"].id
    rule_ids = [entry["rule_id"] for entry in fx["pool"]["entries"]]

    build_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "voting_candidate_build",
            "workflow_inputs": {
                "strategy_type": "approval",
                "rule_ids": list(reversed(rule_ids)),
                "n": 1,
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: build_llm,
    )
    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "构建审批策略池的 Voting 候选："
                f"{rule_ids[0]}、{rule_ids[1]}，2 选 1；只生成候选。"
            )
        },
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_voting_candidate_build"
    ]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["n"] == 1
    assert output["k"] == 2
    assert [item["pool_position"] for item in output["selected_entries"]] == [0, 1]
    assert output["not_admitted"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert (
        StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
            task_id, "approval"
        )
        == fx["pool"]
    )
    build_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "仅生成候选" in build_text
    assert "尚未入池" in build_text
    assert "未采纳、未部署" in build_text

    asset_id = output["asset_id"]
    add_payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "candidate_asset_id": asset_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "review"},
        },
    }
    missing_placement_llm = _PayloadLLM(add_payload)
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: missing_placement_llm,
    )
    missing_placement = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把 {asset_id} 加入审批 Strategy Pool；默认动作 approval，"
                "命中动作 review 复核。"
            )
        },
    )
    assert missing_placement.status_code == 202, missing_placement.text
    assert len(client.get(f"/api/tasks/{task_id}/plans").json()["plans"]) == 1
    missing_text = "\n".join(
        message.get("content", "")
        for message in missing_placement.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "before_selected_members" in missing_text
    assert "replace_selected_members" in missing_text

    add_llm = _PayloadLLM(
        {
            **add_payload,
            "workflow_inputs": {
                **add_payload["workflow_inputs"],
                "placement_mode": "before_selected_members",
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: add_llm,
    )
    admitted = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把 {asset_id} 加入审批 Strategy Pool；默认动作 approval，"
                "命中动作 review 复核；放置方式: before_selected_members，"
                "保留成员作为未达 n 时的后续规则。"
            )
        },
    )

    assert admitted.status_code == 202, admitted.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_voting_candidate_build",
        "strategy_pool_add_candidate",
    ]
    assert plans[-1]["status"] == "done"
    current = StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
        task_id, "approval"
    )
    assert current is not None
    assert current["revision"] == fx["pool"]["revision"] + 1
    assert current["entries"][0]["source"]["asset_id"] == asset_id
    assert current["entries"][0]["source"]["asset_type"] == "voting_n_of_k"
    assert current["entries"][0]["execution"]["condition"]["op"] == "n_of_k"
