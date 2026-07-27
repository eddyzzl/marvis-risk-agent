"""Natural-language search and exact UI build share one governed Cross kernel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from tests.test_strategy_cross_matrix_candidate_tool import _setup


class _PayloadLLM:
    def complete(self, **_kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "cross_rule_search",
                "workflow_inputs": {
                    "features": ["score", "age"],
                    "dimension": 2,
                    "constraints": {
                        "min_lift": 0.0,
                        "min_bad_count": 0,
                        "max_hit_share": 1.0,
                        "min_amount_lift": None,
                    },
                    "max_trials": 4,
                },
            },
            ensure_ascii=False,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_cross_rule_natural_search_then_manual_exact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    task_id = fixture["task"].id
    client = TestClient(create_app(fixture["settings"].workspace))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _PayloadLLM(),
    )

    searched_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "搜索 2D Cross 阈值规则：features=[score, age]，"
                "dimension=2，min_lift=0，min_bad_count=0，"
                "max_hit_share=1，min_amount_lift=null，max_trials=4。"
                "只搜索，不构建候选、不入池。"
            )
        },
    )

    assert searched_response.status_code == 202, searched_response.text
    search_plan = client.get(
        f"/api/tasks/{task_id}/plans"
    ).json()["plans"][-1]
    assert search_plan["template_id"] == "strategy_cross_rule_search"
    assert search_plan["status"] == "done"
    stored_search = client.app.state.plan_repo.load_plan(search_plan["id"])
    assert [step.tool_ref.tool for step in stored_search.steps] == [
        "search_cross_threshold_rules"
    ]
    searched = client.app.state.plan_repo.load_step_output(
        stored_search.steps[0].id
    )
    [rule, *_] = searched["search_result"]["rules"]
    assert searched["evaluated"] == 4
    assert searched["not_selected"] is True
    assert searched["not_admitted"] is True

    built_response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "人工界面精确构建 Cross 阈值规则候选",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "cross_rule_candidate_build_from_search",
                "workflow_inputs": {
                    "search_id": searched["search_id"],
                    "rule_id": rule["rule_id"],
                    "selection_reason": "人工风险评审。",
                },
            },
        },
    )

    assert built_response.status_code == 202, built_response.text
    build_plan = client.get(
        f"/api/tasks/{task_id}/plans"
    ).json()["plans"][-1]
    assert build_plan["template_id"] == (
        "strategy_cross_rule_candidate_build_from_search"
    )
    assert build_plan["status"] == "done"
    stored_build = client.app.state.plan_repo.load_plan(build_plan["id"])
    assert [step.tool_ref.tool for step in stored_build.steps] == [
        "build_cross_rule_candidate_from_search"
    ]
    built = client.app.state.plan_repo.load_step_output(
        stored_build.steps[0].id
    )
    assert built["source_search_selection"]["search_id"] == (
        searched["search_id"]
    )
    assert built["source_search_selection"]["rule_id"] == rule["rule_id"]
    assert built["candidate"]["selection_reason"] == "人工风险评审。"
    assert built["not_admitted"] is True
