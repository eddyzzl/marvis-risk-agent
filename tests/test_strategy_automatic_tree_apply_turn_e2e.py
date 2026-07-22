"""Natural-language full-tree writeback is exact, reversible and task-owned."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import _strategy_request_requires_dataset
from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import DatasetRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **_kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.fixture
def built_tree_for_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    source = tmp_path / "source"
    source.mkdir(parents=True)
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
            "model_name": "自动树全量写回",
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
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_from_upload(
        task_id,
        source / "sample.csv",
        role="strategy_sample",
    )
    workspaces = DataWorkspaceRepository(settings.db_path)
    selected = workspaces.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    workspaces.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={"bad": "target", "score": "score"},
            ),
        ),
        expected_revision=selected.revision,
    )
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_sample_design",
            "workflow_inputs": {
                "target_bad_value": 1,
                "performance_window_status": "provided",
                "performance_window_days": 90,
                "observation_window_status": "provided",
                "observation_start": "2025-01-01",
                "observation_end": "2025-12-31",
                "maturity_status": "confirmed_matured",
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    designed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "固化策略样本设计；表现窗 90 天；观察窗 2025-01-01 至 "
                "2025-12-31；成熟度已确认成熟；1 代表坏样本。"
            )
        },
    )
    assert designed.status_code == 202, designed.text
    designed_plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert [item.template_id for item in designed_plans] == [
        "strategy_sample_design"
    ], designed.json()
    assert designed_plans[0].status == "done", designed.json()
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_candidate_build",
        "workflow_inputs": {
            "features": ["score", "income"],
            "max_depth": 2,
            "min_leaf_count": 2,
        },
    }
    built = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "用 score 和 income 建树，max_depth 2，min_leaf_count 2。"},
    )
    assert built.status_code == 202, built.text
    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert [item.template_id for item in plans] == [
        "strategy_sample_design",
        "strategy_automatic_tree_candidate_build",
    ], json.dumps(built.json(), ensure_ascii=False)
    assert plans[-1].status == "done", json.dumps(
        built.json(), ensure_ascii=False
    )
    plan = plans[-1]
    output = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    workspace = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    return {
        "client": client,
        "task_id": task_id,
        "llm": llm,
        "asset_id": output["summary"]["asset_id"],
        "source_dataset_id": output["summary"]["dataset_id"],
        "workspace": workspace,
    }


def test_automatic_tree_apply_requires_current_dataset_context() -> None:
    draft = StandardWorkflowRequestDraft(
        workflow="automatic_tree_apply",
        workflow_inputs={"tree_asset_id": "candidate-asset-" + "a" * 32},
    )

    assert _strategy_request_requires_dataset(draft) is True


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_full_tree_apply_creates_inactive_derived_dataset(
    built_tree_for_apply: dict,
) -> None:
    client = built_tree_for_apply["client"]
    task_id = built_tree_for_apply["task_id"]
    asset_id = built_tree_for_apply["asset_id"]
    llm = built_tree_for_apply["llm"]
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_apply",
        "workflow_inputs": {
            "tree_asset_id": asset_id,
            "leaf_id_column": "tree_leaf_bucket",
            "rule_id_column": "tree_rule_bucket",
        },
    }

    applied = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把自动树资产 {asset_id} 应用到当前样本，"
                "叶节点输出列 tree_leaf_bucket，规则输出列 tree_rule_bucket。"
            )
        },
    )

    assert applied.status_code == 202, applied.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design",
        "strategy_automatic_tree_candidate_build",
        "strategy_automatic_tree_apply",
    ]
    plan = plans[-1]
    assert plan["status"] == "done"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["needs_confirmation"] is False
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["activated"] is False
    assert output["source"]["asset_id"] == asset_id
    assert output["source"]["dataset_id"] == built_tree_for_apply["source_dataset_id"]
    assert output["result"]["dataset_id"] != output["source"]["dataset_id"]
    assert output["result"]["row_count"] == output["source"]["row_count"] == 24
    assert output["columns"] == {
        "leaf_id": "tree_leaf_bucket",
        "rule_id": "tree_rule_bucket",
    }
    assert client.get(output["evidence"]["download_url"]).status_code == 200

    current = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    assert current == built_tree_for_apply["workspace"]
    assistant_text = "\n".join(
        message.get("content", "")
        for message in applied.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert asset_id in assistant_text
    assert "tree_leaf_bucket" in assistant_text
    assert "tree_rule_bucket" in assistant_text
    assert output["result"]["dataset_id"] in assistant_text
    assert output["evidence"]["artifact_id"] in assistant_text
    assert "development / unvalidated" in assistant_text
    assert "当前 workspace 未切换" in assistant_text
    assert "未入池、未采纳、未部署" in assistant_text


@pytest.mark.slow
@pytest.mark.e2e
def test_tree_apply_workspace_drift_fails_closed_without_new_state(
    built_tree_for_apply: dict,
) -> None:
    client = built_tree_for_apply["client"]
    task_id = built_tree_for_apply["task_id"]
    asset_id = built_tree_for_apply["asset_id"]
    llm = built_tree_for_apply["llm"]
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_apply",
        "workflow_inputs": {"tree_asset_id": asset_id},
    }
    workspaces = DataWorkspaceRepository(client.app.state.settings.db_path)
    current = workspaces.get_or_default(task_id)
    workspaces.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=current.active_dataset_id,
            active_dataset_content_hash=current.active_dataset_content_hash,
            page=current.page,
            selected_field=current.selected_field,
            semantic_mapping=DataSemanticMapping(
                target_col=current.semantic_mapping.target_col,
                field_roles=dict(current.semantic_mapping.field_roles),
                business_names={"score": "调整后的评分口径"},
            ),
        ),
        expected_revision=current.revision,
    )
    plan_repo = client.app.state.plan_repo
    artifacts = TaskArtifactRepository(client.app.state.settings.db_path)
    before_plan_ids = [plan.id for plan in plan_repo.list_plans_for_task(task_id)]
    before_artifact_ids = [item["id"] for item in artifacts.list_for_task(task_id)]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"把自动树资产 {asset_id} 应用到当前样本。"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == "automatic_tree_apply_binding_required"
    assert [plan.id for plan in plan_repo.list_plans_for_task(task_id)] == (
        before_plan_ids
    )
    assert [item["id"] for item in artifacts.list_for_task(task_id)] == (
        before_artifact_ids
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_tree_apply_unknown_task_asset_fails_before_plan_creation(
    built_tree_for_apply: dict,
) -> None:
    client = built_tree_for_apply["client"]
    task_id = built_tree_for_apply["task_id"]
    unknown_asset_id = "candidate-asset-" + "f" * 32
    llm = built_tree_for_apply["llm"]
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_apply",
        "workflow_inputs": {"tree_asset_id": unknown_asset_id},
    }
    plan_repo = client.app.state.plan_repo
    artifacts = TaskArtifactRepository(client.app.state.settings.db_path)
    before_plan_ids = [plan.id for plan in plan_repo.list_plans_for_task(task_id)]
    before_artifact_ids = [item["id"] for item in artifacts.list_for_task(task_id)]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": f"把自动树资产 {unknown_asset_id} 应用到当前样本。"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == "automatic_tree_apply_binding_required"
    assert [plan.id for plan in plan_repo.list_plans_for_task(task_id)] == (
        before_plan_ids
    )
    assert [item["id"] for item in artifacts.list_for_task(task_id)] == (
        before_artifact_ids
    )
