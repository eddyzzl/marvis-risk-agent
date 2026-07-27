"""Structured-turn coverage for interactive-tree frontier materialization."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
)
from marvis.plugins.manifest import ToolRef


pytest_plugins = ("tests.test_strategy_interactive_tree_turn_e2e",)


def _request(
    revision_id: str,
    source_node_id: str,
    *,
    selection_reason: str | None = None,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "revision_id": revision_id,
        "source_node_id": source_node_id,
    }
    if selection_reason is not None:
        inputs["selection_reason"] = selection_reason
    return {
        "request_kind": "standard_workflow",
        "workflow": "interactive_tree_frontier_materialization",
        "workflow_inputs": inputs,
    }


def _materialize_revision(
    client: TestClient,
    *,
    task_id: str,
    source_tree_id: str,
    node_id: str,
) -> dict:
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "按指定节点创建不可变交互树修订。",
            "strategy_request": {
                "request_kind": "standard_workflow",
                "workflow": "interactive_tree_revision",
                "workflow_inputs": {
                    "source_tree_id": source_tree_id,
                    "node_id": node_id,
                    "operation": "prune_subtree",
                },
            },
        },
    )
    assert response.status_code == 202, response.text
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    assert plan["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    return client.app.state.plan_repo.load_step_output(stored.steps[0].id)


@pytest.mark.slow
@pytest.mark.e2e
def test_structured_turn_materializes_one_authenticated_revision_frontier(
    automatic_tree_turn: dict[str, object],
) -> None:
    client = automatic_tree_turn["client"]
    assert isinstance(client, TestClient)
    task_id = str(automatic_tree_turn["task_id"])
    node_id = str(automatic_tree_turn["root_node_id"])
    revision = _materialize_revision(
        client,
        task_id=task_id,
        source_tree_id=str(automatic_tree_turn["asset_id"]),
        node_id=node_id,
    )
    reason = "人工确认该前沿节点用于下一轮策略评审"
    workflow_inputs = {
        "revision_id": revision["revision_id"],
        "source_node_id": node_id,
        "selection_reason": reason,
    }

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "物化指定的交互树前沿节点。",
            "strategy_request": _request(
                revision["revision_id"],
                node_id,
                selection_reason=reason,
            ),
        },
    )

    assert response.status_code == 202, response.text
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    assert plan["template_id"] == (
        "strategy_interactive_tree_frontier_materialization"
    )
    assert plan["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].tool_ref == ToolRef(
        "strategy",
        "materialize_interactive_tree_frontier_selection",
    )
    assert stored.steps[0].inputs == workflow_inputs

    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["revision_id"] == revision["revision_id"]
    assert output["source_node_id"] == node_id
    assert output["selection_reason"] == reason
    descriptor = output["artifacts"][0]
    assert descriptor["kind"] == INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND

    artifacts = client.get(
        f"/api/tasks/{task_id}/task-artifacts"
    ).json()["artifacts"]
    registered = next(
        artifact
        for artifact in artifacts
        if artifact["id"] == descriptor["artifact_id"]
    )
    assert (
        registered["origin_tool"]
        == INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_frontier_materialization_without_dataset_reaches_tool_authentication(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "无样本交互树前沿物化",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "按明确 revision 和前沿节点物化 pointer。",
            "strategy_request": _request(
                "interactive-tree-revision-" + "a" * 32,
                "node-" + "b" * 20,
            ),
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["template_id"] == (
        "strategy_interactive_tree_frontier_materialization"
    )
    assert plan["status"] == "failed"
    assistant_codes = {
        message.get("metadata", {}).get("code")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    }
    assert "strategy_dataset_context_required" not in assistant_codes
