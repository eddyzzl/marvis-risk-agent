"""Structured-turn coverage for frontier OR-group materialization."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.plugins.manifest import ToolRef


def _request(
    revision_id: str,
    source_node_ids: list[str],
    *,
    selection_reason: str | None = None,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "revision_id": revision_id,
        "source_node_ids": source_node_ids,
    }
    if selection_reason is not None:
        inputs["selection_reason"] = selection_reason
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_frontier_group_materialization",
        "workflow_inputs": inputs,
    }


def test_frontier_group_without_dataset_reaches_tool_authentication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "无样本交互树前沿 OR 分组物化",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    revision_id = "interactive-tree-revision-" + "a" * 32
    source_node_ids = [
        "node-" + "b" * 20,
        "leaf-" + "c" * 20,
    ]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "按明确 revision 和两个前沿节点物化 OR group pointer。",
            "strategy_request": _request(
                revision_id,
                source_node_ids,
                selection_reason="业务评审确认任一成员命中",
            ),
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["template_id"] == (
        "strategy_interactive_tree_frontier_group_materialization"
    )
    assert plan["status"] == "failed"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].tool_ref == ToolRef(
        "strategy",
        "materialize_interactive_tree_frontier_group_selection",
    )
    assert stored.steps[0].inputs == {
        "revision_id": revision_id,
        "source_node_ids": source_node_ids,
        "selection_reason": "业务评审确认任一成员命中",
    }
    assistant_codes = {
        message.get("metadata", {}).get("code")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    }
    assert "strategy_dataset_context_required" not in assistant_codes
