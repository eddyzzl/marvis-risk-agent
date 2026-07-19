"""Focused natural-language automatic-tree build vertical."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _ensure_automatic_tree_active_workspace,
    _is_strategy_request_intent,
)
from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.settings import build_settings


class _AutomaticTreeBuildLLM:
    def __init__(self, *, workflow_inputs: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.workflow_inputs = workflow_inputs or {
            "features": ["score", "income"],
            "directions": {
                "score": "decreasing",
                "income": "decreasing",
            },
            "max_depth": 2,
            "min_leaf_count": 2,
            "min_weight_fraction_leaf": 0.0,
            "seed": 20260719,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        }

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "automatic_tree_candidate_build",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


@pytest.mark.parametrize(
    "utterance",
    [
        "用 age 建树",
        "用 age 训练一棵树",
        "build a tree with age",
        "train an automatic tree with age",
    ],
)
def test_automatic_tree_build_shorthand_is_a_strategy_request(utterance: str) -> None:
    assert _is_strategy_request_intent(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    ["这棵树很好看", "建树状目录", "build a treehouse", "训练模型"],
)
def test_tree_words_without_build_intent_do_not_hijack_chat(utterance: str) -> None:
    assert _is_strategy_request_intent(utterance) is False


def test_explicitly_negated_tree_build_creates_no_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "score": [400, 500, 600, 700],
            "bad": [1, 1, 0, 0],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "明确否定自动树构建",
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
    llm = _AutomaticTreeBuildLLM(workflow_inputs={"features": ["score"]})
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "用 score，但不要建树。"},
    )

    assert response.status_code == 202, response.text
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_automatic_tree_build_is_one_unranked_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(24)],
            "score": [360 + index * 20 for index in range(24)],
            "income": [3000 + (index % 8) * 800 for index in range(24)],
            "loan_amount": [1000.0 + index * 50 for index in range(24)],
            "overdue_amount": [
                0.0 if index >= 12 else 50.0 + index for index in range(24)
            ],
            "bad": [1 if index < 12 else 0 for index in range(24)],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言自动树候选构建",
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
    llm = _AutomaticTreeBuildLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "用 score 和 income 建树；score 方向 decreasing，"
                "income 方向 decreasing；max_depth 2，min_leaf_count 2，"
                "min_weight_fraction_leaf 0.0，seed 20260719；"
                "放款金额列 loan_amount，逾期金额列 overdue_amount。"
            )
        },
    )

    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_automatic_tree_candidate_build"
    ]
    plan = plans[0]
    assert plan["status"] == "done"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["status"] == "done"
    assert plan["steps"][0]["needs_confirmation"] is False

    stored = client.app.state.plan_repo.load_plan(plan["id"])
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["summary"]["candidate_stage"] == "development"
    assert output["summary"]["observation_stage"] == "backtested"
    assert output["summary"]["validation_status"] == "unvalidated"
    assert output["leaf_index"]
    assert len(output["artifacts"]) == 6
    assert len(output["report_info_gaps"]) >= 1
    assert not ({"rank", "action", "recommendation"} & set(output))

    listed = client.get(f"/api/tasks/{task_id}/task-artifacts").json()["artifacts"]
    assert {artifact["id"] for artifact in listed} == {
        artifact["artifact_id"] for artifact in output["artifacts"]
    }
    assert all(
        client.get(artifact["download_url"]).status_code == 200
        for artifact in output["artifacts"]
    )

    assistant_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "development / backtested / unvalidated" in assistant_text
    assert "尚未选叶" in assistant_text
    assert "未入池" in assistant_text
    assert "最终报告留空" in assistant_text
    if output["red_flags"]:
        assert "风险方向红旗" in assistant_text
        assert "不是强制分裂约束" in assistant_text
    assert "最佳" not in assistant_text
    assert "最优" not in assistant_text
    assert len(llm.calls) == 1

    workspace = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    assert workspace.revision == 1
    assert workspace.analysis_generation == 1
    assert workspace.active_dataset_id == output["summary"]["dataset_id"]
    assert (
        workspace.active_dataset_content_hash
        == output["summary"]["dataset_content_hash"]
    )
    assert workspace.semantic_mapping.target_col == "bad"
    assert workspace.semantic_mapping.field_roles["bad"] == "target"


def test_automatic_tree_workspace_binding_never_guesses_between_datasets(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="ambiguous automatic tree sample",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    pd.DataFrame({"score": [1, 2], "bad": [0, 1]}).to_parquet(first_path, index=False)
    pd.DataFrame({"score": [3, 4], "bad": [0, 1]}).to_parquet(second_path, index=False)
    first = registry.register_existing(first_path, task_id=task.id, role="sample")
    registry.register_existing(second_path, task_id=task.id, role="sample")

    with pytest.raises(StrategySetupError, match="多个或不确定"):
        _ensure_automatic_tree_active_workspace(
            SimpleNamespace(settings=settings),
            task,
            preview=SimpleNamespace(),
            context=SimpleNamespace(
                dataset_id=first.id,
                dataset_content_hash=first.content_hash,
                target_col="bad",
            ),
        )

    assert (
        DataWorkspaceRepository(settings.db_path)
        .get_or_default(task.id)
        .active_dataset_id
        is None
    )
