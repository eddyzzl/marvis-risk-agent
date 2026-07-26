"""Minimal structured-turn E2E coverage for interactive-tree revisions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
)
from marvis.packs.strategy.interactive_tree_tools import (
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
    INTERACTIVE_TREE_REVISION_ORIGIN_TOOL,
)
from marvis.plugins.manifest import ToolRef
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


def _standard_workflow_request(
    workflow: str,
    workflow_inputs: dict[str, object],
) -> dict[str, object]:
    return {
        "request_kind": "standard_workflow",
        "workflow": workflow,
        "workflow_inputs": workflow_inputs,
    }


@pytest.fixture
def automatic_tree_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(24)],
            "score": [360 + index * 20 for index in range(24)],
            "income": [3000 + (index % 8) * 800 for index in range(24)],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    ).to_csv(source / "sample.csv", index=False)

    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "交互式决策树修剪端到端",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)

    built = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "用 score 和 income 构建自动树候选。",
            "strategy_request": _standard_workflow_request(
                "automatic_tree_candidate_build",
                {
                    "features": ["score", "income"],
                    "max_depth": 2,
                    "min_leaf_count": 2,
                },
            ),
        },
    )
    assert built.status_code == 202, built.text

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    automatic_plan = plans[-1]
    assert automatic_plan["template_id"] == (
        "strategy_automatic_tree_candidate_build"
    )
    assert automatic_plan["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(automatic_plan["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    asset_descriptor = next(
        artifact
        for artifact in output["artifacts"]
        if artifact["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    downloaded = client.get(asset_descriptor["download_url"])
    assert downloaded.status_code == 200, downloaded.text
    asset = json.loads(downloaded.content)
    root_node_id = asset["tree_result"]["tree"]["root_node_id"]
    root = next(
        node
        for node in asset["tree_result"]["tree"]["nodes"]
        if node["node_id"] == root_node_id
    )
    assert root["kind"] == "split"
    return {
        "client": client,
        "task_id": task_id,
        "asset_id": output["summary"]["asset_id"],
        "root_node_id": root_node_id,
    }


@pytest.mark.slow
@pytest.mark.e2e
def test_structured_turn_executes_interactive_tree_revision_and_registers_artifact(
    automatic_tree_turn: dict[str, object],
) -> None:
    client = automatic_tree_turn["client"]
    assert isinstance(client, TestClient)
    task_id = str(automatic_tree_turn["task_id"])
    source_tree_id = str(automatic_tree_turn["asset_id"])
    node_id = str(automatic_tree_turn["root_node_id"])
    reason = "人工确认根节点以下颗粒度过细"
    workflow_inputs = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": "prune_subtree",
        "reason": reason,
    }

    revised = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "按指定节点创建一个不可变的交互式树修订。",
            "strategy_request": _standard_workflow_request(
                "interactive_tree_revision",
                workflow_inputs,
            ),
        },
    )

    assert revised.status_code == 202, revised.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    plan = plans[-1]
    assert plan["template_id"] == "strategy_interactive_tree_revision"
    assert plan["status"] == "done"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["needs_confirmation"] is False

    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].tool_ref == ToolRef(
        "strategy",
        "revise_interactive_tree",
    )
    assert stored.steps[0].inputs == workflow_inputs

    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["source_tree_id"] == source_tree_id
    assert output["edit"] == {
        "operation": "prune_subtree",
        "node_id": node_id,
        "reason": reason,
    }
    assert output["replay"]["exactly_once"] is True
    assert output["replay"]["metrics_matched"] is True
    assert len(output["artifacts"]) == 1
    descriptor = output["artifacts"][0]
    assert descriptor["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND

    artifacts = client.get(
        f"/api/tasks/{task_id}/task-artifacts"
    ).json()["artifacts"]
    registered = next(
        artifact
        for artifact in artifacts
        if artifact["id"] == descriptor["artifact_id"]
    )
    assert registered["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
    assert registered["origin_tool"] == INTERACTIVE_TREE_REVISION_ORIGIN_TOOL
    assert registered["content_hash"] == descriptor["content_hash"]
    assert registered["available"] is True
    downloaded = client.get(registered["download_url"])
    assert downloaded.status_code == 200, downloaded.text
    revision = downloaded.json()
    assert revision["revision_id"] == output["revision_id"]
    assert revision["revision_hash"] == output["revision_hash"]
    assert revision["edit"] == output["edit"]


@pytest.mark.slow
@pytest.mark.e2e
def test_structured_turn_without_dataset_reaches_tool_instead_of_preview_preflight(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "无样本交互树修剪",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    workflow_inputs = {
        "source_tree_id": "candidate-asset-" + "a" * 32,
        "node_id": "node-" + "b" * 20,
        "operation": "prune_subtree",
        "reason": "验证无 preview 时仍由 Tool 认证源资产",
    }

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "按指定源树和节点执行修剪。",
            "strategy_request": _standard_workflow_request(
                "interactive_tree_revision",
                workflow_inputs,
            ),
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["template_id"] == "strategy_interactive_tree_revision"
    assert plan["status"] == "failed"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].tool_ref == ToolRef(
        "strategy",
        "revise_interactive_tree",
    )
    assert stored.steps[0].inputs == workflow_inputs

    assistant_codes = {
        message.get("metadata", {}).get("code")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    }
    assert "strategy_dataset_context_required" not in assistant_codes
