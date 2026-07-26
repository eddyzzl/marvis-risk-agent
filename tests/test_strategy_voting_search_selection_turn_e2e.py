"""Natural language builds one Voting candidate from exact search pointers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.packs.strategy.voting_candidate_search_tools import (
    resolve_voting_candidate_search_inputs,
    run_search_voting_candidates,
)
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
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
def test_natural_language_builds_exact_search_combo_without_pool_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _search_fixture(tmp_path)
    task_id = fixture["task"].id
    search_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=task_id,
        user_controls=fixture["controls"],
    )
    search = run_search_voting_candidates(
        search_inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    exact_combo = search["search_result"]["combinations"][1]
    before = StrategyCandidatePoolRepository(fixture["settings"].db_path).get_current(
        task_id,
        "approval",
    )
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "voting_candidate_build_from_search",
            "workflow_inputs": {
                "search_id": search["search_id"],
                "combo_id": exact_combo["combo_id"],
                "strategy_type": "approval",
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    client = TestClient(create_app(fixture["settings"].workspace))

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "从 approval Strategy Pool 的 Voting 搜索结果精确构建候选："
                f"search_id={search['search_id']}，"
                f"combo_id={exact_combo['combo_id']}。"
                "只构建候选，不入池、不修改 Pool、不设置动作、不应用、"
                "不采纳、不部署、不写回。"
            )
        },
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_voting_candidate_build_from_search"
    ]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    [step] = stored.steps
    assert step.tool_ref.tool == "build_voting_candidate_from_search"
    assert step.inputs == {
        "search_id": search["search_id"],
        "combo_id": exact_combo["combo_id"],
        "strategy_type": "approval",
    }
    output = client.app.state.plan_repo.load_step_output(step.id)
    assert output["source_search_selection"] == {
        "search_id": search["search_id"],
        "combo_id": exact_combo["combo_id"],
        "strategy_type": "approval",
        "rank": exact_combo["rank"],
        "member_rule_ids": exact_combo["member_ids"],
        "n": exact_combo["n"],
        "eligible": exact_combo["eligible"],
        "constraint_failures": exact_combo["constraint_failures"],
    }
    assert output["voting_candidate"]["asset_id"].startswith("candidate-asset-")
    assert output["not_mutated_pool"] is True
    assert output["not_admitted"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert (
        StrategyCandidatePoolRepository(fixture["settings"].db_path).get_current(
            task_id,
            "approval",
        )
        == before
    )

    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert search["search_id"] in assistant_text
    assert exact_combo["combo_id"] in assistant_text
    assert output["voting_candidate"]["asset_id"] in assistant_text
    assert "精确点名" in assistant_text
    assert "尚未入池" in assistant_text
    assert "winner" not in assistant_text.casefold()
    assert "champion" not in assistant_text.casefold()
