from __future__ import annotations

import json
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
            "model_name": "策略样本加工",
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
            "customer_id": ["A", "B", "C"],
            "amount": [100.0, None, 300.0],
            "bad": [0, 1, 0],
            "mobile": ["13800000001", "13800000002", "13800000003"],
        }
    ).to_csv(source, index=False)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    return registry, registry.register_from_upload(
        task_id,
        source,
        role="strategy_sample",
    )


def _activate(client: TestClient, task_id: str, dataset):
    repo = DataWorkspaceRepository(client.app.state.settings.db_path)
    active = repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    return repo.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="semantics",
            selected_field="amount",
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={
                    "customer_id": "id",
                    "amount": "amount",
                    "bad": "target",
                    "mobile": "phone",
                },
                business_names={
                    "customer_id": "客户编号",
                    "amount": "申请金额",
                    "bad": "风险标签",
                    "mobile": "手机号",
                },
            ),
        ),
        expected_revision=active.revision,
    )


def _activate_without_semantics(client: TestClient, task_id: str, dataset):
    return DataWorkspaceRepository(client.app.state.settings.db_path).save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
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


def test_natural_language_transform_gets_first_refusal_and_activates_child(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    registry, dataset = _register(client, task_id, tmp_path)
    source_snapshot = _activate(client, task_id, dataset)

    response = _post(client, task_id, "用中位数填充申请金额缺失值")

    assert response.status_code == 202, response.text
    last = _last_assistant(response)
    assert "计划已全部完成" in last["content"]
    assert "数据加工完成" in last["content"]
    assert {table["title"] for table in last["metadata"]["tables"]} >= {
        "加工步骤影响",
        "数据血缘",
        "证据与下载",
    }
    active = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    assert active.active_dataset_id != dataset.id
    assert active.revision == source_snapshot.revision + 1
    assert active.analysis_generation == source_snapshot.analysis_generation + 1
    result = registry.get(active.active_dataset_id)
    frame = DataBackend(client.app.state.settings.datasets_dir).read_frame(
        registry.resolve_verified_path(result.id)
    )
    assert frame["amount"].tolist() == [100.0, 200.0, 300.0]

    with connect(client.app.state.settings.db_path) as conn:
        tool_refs = [
            str(row["tool_ref"])
            for row in conn.execute(
                """
                SELECT r.tool_ref
                 FROM plan_step_runs r
                  JOIN plans p ON p.id = r.plan_id
                 WHERE p.task_id = ?
                 ORDER BY r.id
                """,
                (task_id,),
            ).fetchall()
        ]
    assert tool_refs == ["data_ops.transform_dataset"]
    listed = client.get(f"/api/tasks/{task_id}/task-artifacts")
    assert listed.status_code == 200, listed.text
    evidence = next(
        artifact
        for artifact in listed.json()["artifacts"]
        if artifact["kind"] == "data_transform_evidence"
    )
    assert evidence["available"] is True
    assert evidence["download_url"]
    downloaded = client.get(evidence["download_url"])
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.headers["content-type"].startswith("application/json")


def test_transform_unknown_field_clarifies_without_creating_plan(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    _, dataset = _register(client, task_id, tmp_path)
    _activate(client, task_id, dataset)

    response = _post(client, task_id, "删除 ghost_field")

    assert response.status_code == 202, response.text
    last = _last_assistant(response)
    assert "ghost_field" in last["content"]
    assert last["metadata"]["intent"] == "dataset_transform"
    with connect(client.app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM plans WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0


def test_protected_drop_can_be_confirmed_in_the_following_turn(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    registry, dataset = _register(client, task_id, tmp_path)
    _activate(client, task_id, dataset)

    blocked = _post(client, task_id, "删除风险标签和手机号")

    assert blocked.status_code == 202, blocked.text
    clarification = _last_assistant(blocked)
    assert "确认" in clarification["content"]
    assert clarification["metadata"]["intent"] == "dataset_transform"
    assert clarification["metadata"]["kind"] == "clarification"
    assert clarification["metadata"]["pending_protected_drop"]["request_text"] == (
        "删除风险标签和手机号"
    )
    assert clarification["metadata"]["pending_protected_drop"]["operations"] == [
        {"op": "drop_columns", "columns": ["bad", "mobile"]}
    ]
    assert clarification["metadata"]["pending_protected_drop"][
        "protected_fields"
    ] == ["bad", "mobile"]

    confirmed = _post(client, task_id, "确认")

    assert confirmed.status_code == 202, confirmed.text
    last = _last_assistant(confirmed)
    assert "数据加工完成" in last["content"]
    active = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    result = registry.get(active.active_dataset_id)
    assert result.target_col is None
    assert [column.name for column in result.columns] == ["customer_id", "amount"]
    assert active.semantic_mapping.target_col is None
    assert "bad" not in active.semantic_mapping.field_roles
    assert "mobile" not in active.semantic_mapping.field_roles


def test_registry_target_is_protected_before_workspace_semantics_are_saved(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    registry, dataset = _register(client, task_id, tmp_path)
    assert dataset.target_col == "bad"
    _activate_without_semantics(client, task_id, dataset)

    blocked = _post(client, task_id, "删除 bad")

    assert blocked.status_code == 202, blocked.text
    clarification = _last_assistant(blocked)
    assert "确认" in clarification["content"]
    assert clarification["metadata"]["pending_protected_drop"][
        "protected_fields"
    ] == ["bad"]
    with connect(client.app.state.settings.db_path) as conn:
        assert conn.execute(
            "SELECT count(*) FROM plans WHERE task_id = ?",
            (task_id,),
        ).fetchone()[0] == 0

    confirmed = _post(client, task_id, "确认")

    assert confirmed.status_code == 202, confirmed.text
    assert "数据加工完成" in _last_assistant(confirmed)["content"]
    active = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    result = registry.get(active.active_dataset_id)
    assert result.target_col is None
    assert "bad" not in [column.name for column in result.columns]


def test_repeated_casts_run_as_two_ordered_tool_steps(tmp_path):
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _task(client, tmp_path)
    registry, dataset = _register(client, task_id, tmp_path)
    _activate(client, task_id, dataset)

    response = _post(
        client,
        task_id,
        "把申请金额严格转为 VARCHAR；再把申请金额严格转为 DOUBLE",
    )

    assert response.status_code == 202, response.text
    assert "数据加工完成" in _last_assistant(response)["content"]
    active = DataWorkspaceRepository(
        client.app.state.settings.db_path
    ).get_or_default(task_id)
    frame = DataBackend(client.app.state.settings.datasets_dir).read_frame(
        registry.resolve_verified_path(active.active_dataset_id)
    )
    assert frame["amount"].iloc[0] == 100.0
    assert pd.isna(frame["amount"].iloc[1])
    assert frame["amount"].iloc[2] == 300.0
    with connect(client.app.state.settings.db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM data_transform_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
    evidence = json.loads(str(row["result_json"]))
    assert [step["op"] for step in evidence["transform"]["steps"]] == [
        "cast_columns",
        "cast_columns",
    ]
