from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import DatasetRepository
from marvis.db_schema import connect
from marvis.repositories.data_workspace import DataWorkspaceRepository


def _task(client: TestClient, tmp_path: Path) -> str:
    source = client.app.state.settings.workspace / f"source-{tmp_path.name}"
    source.mkdir(exist_ok=True)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "策略样本分析",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _register(client: TestClient, task_id: str, tmp_path: Path):
    source = tmp_path / f"{task_id}.csv"
    pd.DataFrame(
        {
            "score": [500.0, 600.0, 700.0, None],
            "income": [1000.0, 2000.0, 3000.0, 4000.0],
            "bad": [1, 0, 0, 1],
            "customer_name": ["张三", "李四", "王五", "张三"],
        }
    ).to_csv(source, index=False)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    return registry.register_from_upload(task_id, source, role="sample")


def _activate(client: TestClient, task_id: str, dataset, *, with_target: bool):
    repo = DataWorkspaceRepository(client.app.state.settings.db_path)
    activated = repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad" if with_target else None,
        field_roles={
            "score": "score",
            "customer_name": "name",
            **({"bad": "target"} if with_target else {}),
        },
        business_names={"score": "模型分", "income": "收入"},
    )
    return repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="statistics",
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )


def _post(client: TestClient, task_id: str, content: str):
    return client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": content},
    )


def _last_assistant(response) -> dict:
    return [
        message
        for message in response.json()["messages"]
        if message["role"] == "assistant"
    ][-1]


def test_natural_language_dataset_analysis_runs_bound_tool_and_renderer(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    dataset = _register(client, task_id, tmp_path)
    snapshot = _activate(client, task_id, dataset, with_target=True)

    response = _post(client, task_id, "看模型分和 income 的相关性与空值")

    assert response.status_code == 202, response.text
    last = _last_assistant(response)
    assert "计划已全部完成" in last["content"]
    assert "样本描述分析完成" in last["content"]
    assert {table["title"] for table in last["metadata"]["tables"]} == {
        "缺失分析",
        "相关矩阵",
    }
    assert "张三" not in str(last)

    with connect(client.app.state.settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT r.input_json
              FROM plan_step_runs r
              JOIN plans p ON p.id = r.plan_id
             WHERE p.task_id = ? AND r.tool_ref = 'data_ops.profile_dataset'
            """,
            (task_id,),
        ).fetchone()
    assert row is not None
    input_json = row["input_json"]
    assert f'"workspace_revision":{snapshot.revision}' in input_json
    assert f'"analysis_generation":{snapshot.analysis_generation}' in input_json
    assert f'"expected_content_hash":"{dataset.content_hash}"' in input_json
    assert '"sections":["missing","correlation"]' in input_json
    assert '"columns":["score","income"]' in input_json


def test_explicit_analysis_request_without_active_workspace_clarifies(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    _register(client, task_id, tmp_path)

    response = _post(client, task_id, "分析当前样本")

    assert response.status_code == 202, response.text
    last = _last_assistant(response)
    assert "数据工作区" in last["content"]
    assert last["metadata"]["intent"] == "dataset_analysis"
    with connect(client.app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM plans WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_target_distribution_without_target_mapping_clarifies(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    dataset = _register(client, task_id, tmp_path)
    _activate(client, task_id, dataset, with_target=False)

    response = _post(client, task_id, "查看目标分布")

    assert response.status_code == 202, response.text
    last = _last_assistant(response)
    assert "target" in last["content"].lower()
    assert last["metadata"]["intent"] == "dataset_analysis"
