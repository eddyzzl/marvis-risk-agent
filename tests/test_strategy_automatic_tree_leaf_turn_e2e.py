"""Focused natural-language automatic-tree leaf materialization vertical."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _automatic_tree_leaf_materialization_slots,
    _strategy_request_requires_dataset,
)
from marvis.app import create_app
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.fixture
def built_automatic_tree(
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
            "model_name": "自动树精确叶节点物化",
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
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "automatic_tree_candidate_build",
            "workflow_inputs": {
                "features": ["score", "income"],
                "max_depth": 2,
                "min_leaf_count": 2,
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    built = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "用 score 和 income 建树，max_depth 2，min_leaf_count 2。"
            )
        },
    )
    assert built.status_code == 202, built.text
    plan = client.app.state.plan_repo.list_plans_for_task(task_id)[0]
    assert plan.template_id == "strategy_automatic_tree_candidate_build"
    output = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    return {
        "client": client,
        "task_id": task_id,
        "llm": llm,
        "asset_id": output["summary"]["asset_id"],
        "leaf_id": output["leaf_index"][0]["leaf_id"],
    }


def test_leaf_materialization_is_artifact_only_and_never_requires_dataset() -> None:
    draft = StandardWorkflowRequestDraft(
        workflow="automatic_tree_leaf_materialization",
        workflow_inputs={
            "tree_asset_id": "candidate-asset-" + "a" * 32,
            "leaf_id": "leaf-" + "b" * 20,
        },
    )

    assert _strategy_request_requires_dataset(draft) is False


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_exact_leaf_materialization_is_pointer_only(
    built_automatic_tree: dict,
) -> None:
    client = built_automatic_tree["client"]
    task_id = built_automatic_tree["task_id"]
    asset_id = built_automatic_tree["asset_id"]
    leaf_id = built_automatic_tree["leaf_id"]
    reason = "人工确认该叶节点用于后续候选评审"
    llm = built_automatic_tree["llm"]
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_leaf_materialization",
        "workflow_inputs": {
            "tree_asset_id": asset_id,
            "leaf_id": leaf_id,
            "selection_reason": reason,
        },
    }

    materialized = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"物化自动树资产 {asset_id} 的叶节点 {leaf_id}；"
                f"选择理由：{reason}。"
            )
        },
    )

    assert materialized.status_code == 202, materialized.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_automatic_tree_candidate_build",
        "strategy_automatic_tree_leaf_materialization",
    ]
    plan = plans[-1]
    assert plan["status"] == "done"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["needs_confirmation"] is False
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["tree_asset_id"] == asset_id
    assert output["leaf_id"] == leaf_id
    assert output["selection_reason"] == reason
    assert output["selection_id"].startswith("automatic-tree-leaf-selection-")
    assert output["fragment_id"].startswith("candidate-fragment-")
    assert output["rule_id"].startswith("rule-")
    assert output["effect_id"].startswith("candidate-effect-")
    assert len(output["artifacts"]) == 1
    assert client.get(output["artifacts"][0]["download_url"]).status_code == 200
    assert (
        StrategyCandidatePoolRepository(
            client.app.state.settings.db_path
        ).get_current(task_id, "approval")
        is None
    )
    assistant_text = "\n".join(
        message.get("content", "")
        for message in materialized.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "pointer-only" in assistant_text
    assert "没有复制规则、指标或动作" in assistant_text
    assert "未入池" in assistant_text
    assert "未采纳" in assistant_text
    assert "未部署" in assistant_text
    assert "最佳" not in assistant_text
    assert "推荐" not in assistant_text


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "不要物化自动树资产 {asset_id} 的叶节点 {leaf_id}",
            "automatic_tree_leaf_intent_negated",
        ),
        (
            "物化自动树资产 {asset_id} 的叶节点 {leaf_id}；选择理由：风险复核。",
            "automatic_tree_leaf_reason_not_grounded",
        ),
    ],
)
def test_leaf_materialization_compiler_failure_never_mutates_task_state(
    built_automatic_tree: dict,
    utterance: str,
    code: str,
) -> None:
    client = built_automatic_tree["client"]
    task_id = built_automatic_tree["task_id"]
    asset_id = built_automatic_tree["asset_id"]
    leaf_id = built_automatic_tree["leaf_id"]
    llm = built_automatic_tree["llm"]
    workflow_inputs = {
        "tree_asset_id": asset_id,
        "leaf_id": leaf_id,
    }
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "automatic_tree_leaf_materialization",
        "workflow_inputs": workflow_inputs,
    }
    plan_repo = client.app.state.plan_repo
    artifact_repo = TaskArtifactRepository(client.app.state.settings.db_path)
    before_plan_ids = [
        plan.id for plan in plan_repo.list_plans_for_task(task_id)
    ]
    before_artifact_ids = [
        artifact["id"] for artifact in artifact_repo.list_for_task(task_id)
    ]

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": utterance.format(asset_id=asset_id, leaf_id=leaf_id)},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == code
    assert [
        plan.id for plan in plan_repo.list_plans_for_task(task_id)
    ] == before_plan_ids
    assert [
        artifact["id"] for artifact in artifact_repo.list_for_task(task_id)
    ] == before_artifact_ids


def test_leaf_materialization_preflight_rejects_unknown_leaf(
    built_automatic_tree: dict,
) -> None:
    runtime = SimpleNamespace(settings=built_automatic_tree["client"].app.state.settings)
    draft = StandardWorkflowRequestDraft(
        workflow="automatic_tree_leaf_materialization",
        workflow_inputs={
            "tree_asset_id": built_automatic_tree["asset_id"],
            "leaf_id": "leaf-" + "f" * 20,
        },
    )

    with pytest.raises(StrategySetupError, match="叶节点"):
        _automatic_tree_leaf_materialization_slots(
            runtime,
            task_id=built_automatic_tree["task_id"],
            draft=draft,
        )


def test_leaf_materialization_preflight_rejects_drifted_tree_artifact(
    built_automatic_tree: dict,
) -> None:
    client = built_automatic_tree["client"]
    task_id = built_automatic_tree["task_id"]
    artifact = next(
        item
        for item in TaskArtifactRepository(
            client.app.state.settings.db_path
        ).list_for_task(task_id)
        if item["kind"] == "strategy_automatic_tree_asset_json"
    )
    Path(artifact["path"]).write_text("{}", encoding="utf-8")
    runtime = SimpleNamespace(settings=client.app.state.settings)
    draft = StandardWorkflowRequestDraft(
        workflow="automatic_tree_leaf_materialization",
        workflow_inputs={
            "tree_asset_id": built_automatic_tree["asset_id"],
            "leaf_id": built_automatic_tree["leaf_id"],
        },
    )

    with pytest.raises(StrategySetupError, match="完整性"):
        _automatic_tree_leaf_materialization_slots(
            runtime,
            task_id=task_id,
            draft=draft,
        )
