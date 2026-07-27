"""Natural-language Voting search reaches one artifact without Pool mutation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_voting_candidate_search_tools import _search_fixture


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_voting_search_publishes_artifact_without_pool_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _search_fixture(tmp_path)
    task_id = fixture["task"].id
    client = TestClient(create_app(fixture["settings"].workspace))
    before = StrategyCandidatePoolRepository(fixture["settings"].db_path).get_current(
        task_id, "approval"
    )
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "voting_candidate_search",
            "workflow_inputs": {
                "strategy_type": "approval",
                "member_count": 2,
                "n": 1,
                "objective": {
                    "metric": "bad_capture_rate",
                    "direction": "maximize",
                },
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "搜索审批 Strategy Pool 的 Voting 组合：K=2，n=1；"
                "目标最大化 bad_capture_rate；只搜索并保留聚合证据。"
            )
        },
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_voting_candidate_search"
    ]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert [step.tool_ref.tool for step in stored.steps] == ["search_voting_candidates"]
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["search_id"].startswith("voting-search-")
    assert output["search_space"] == 3
    assert output["evaluated"] == 3
    assert output["truncated"] is False
    assert output["not_mutated_pool"] is True
    assert output["not_selected"] is True
    assert output["not_admitted"] is True
    assert (
        StrategyCandidatePoolRepository(fixture["settings"].db_path).get_current(
            task_id,
            "approval",
        )
        == before
    )
    [descriptor] = output["artifacts"]
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        task_id, descriptor["artifact_id"]
    )
    assert record is not None
    persisted = json.loads(Path(record["path"]).read_text("utf-8"))
    assert persisted == output["search_result"]
    assert {"hit_matrix", "target", "weights", "amounts"}.isdisjoint(persisted)

    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert output["search_id"] in assistant_text
    assert "只读搜索完成" in assistant_text
    assert "未修改 Pool" in assistant_text
    assert "未选择任何组合" in assistant_text
    assert "未入池" in assistant_text
