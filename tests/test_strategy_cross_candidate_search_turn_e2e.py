"""Natural language search and manual exact build share one governed kernel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from tests.test_strategy_cross_matrix_candidate_tool import _setup


class _PayloadLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "cross_matrix_candidate_search",
                "workflow_inputs": {
                    "features": ["score", "age"],
                    "max_pairs": 1,
                },
            },
            ensure_ascii=False,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_cross_search_natural_then_manual_exact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    task_id = fixture["task"].id
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _PayloadLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    searched_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "搜索 Cross Matrix 候选组合："
                "features=[score, age]，max_pairs=1。"
                "只搜索，不构建候选、不入池。"
            )
        },
    )

    assert searched_response.status_code == 202, searched_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    search_plan = plans[-1]
    assert search_plan["template_id"] == "strategy_cross_matrix_candidate_search"
    assert search_plan["status"] == "done"
    stored_search = client.app.state.plan_repo.load_plan(search_plan["id"])
    assert [step.tool_ref.tool for step in stored_search.steps] == [
        "search_cross_matrix_candidates"
    ]
    searched = client.app.state.plan_repo.load_step_output(
        stored_search.steps[0].id
    )
    [pair] = searched["search_result"]["pairs"]
    assert searched["evaluated"] == 1
    assert searched["not_selected"] is True
    assert searched["not_admitted"] is True

    built_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "人工界面精确构建 Cross 搜索候选",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "cross_matrix_candidate_build_from_search",
                "workflow_inputs": {
                    "search_id": searched["search_id"],
                    "pair_id": pair["pair_id"],
                },
            },
        },
    )

    assert built_response.status_code == 202, built_response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    build_plan = plans[-1]
    assert (
        build_plan["template_id"]
        == "strategy_cross_matrix_candidate_build_from_search"
    )
    assert build_plan["status"] == "done"
    stored_build = client.app.state.plan_repo.load_plan(build_plan["id"])
    assert [step.tool_ref.tool for step in stored_build.steps] == [
        "build_cross_matrix_candidate_from_search"
    ]
    built = client.app.state.plan_repo.load_step_output(
        stored_build.steps[0].id
    )
    assert built["source_search_selection"] == {
        "search_id": searched["search_id"],
        "pair_id": pair["pair_id"],
        "rank": pair["rank"],
        "x_feature": pair["x_feature"],
        "x_method": pair["x_method"],
        "y_feature": pair["y_feature"],
        "y_method": pair["y_method"],
        "eligible": pair["eligible"],
    }
    assert built["cross_matrix_candidate"]["asset_hash"] == (
        pair["asset_fingerprint"]["asset_hash"]
    )
    assert len(llm.calls) == 1
