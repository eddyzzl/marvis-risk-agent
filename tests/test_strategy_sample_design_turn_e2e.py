"""Focused natural-language StrategySampleDesign vertical."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository


class _SampleDesignLLM:
    def __init__(self, workflow_inputs: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.workflow_inputs = workflow_inputs or {
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 90,
            "observation_window_status": "provided",
            "observation_start": "2025-01-01",
            "observation_end": "2025-12-31",
            "maturity_status": "confirmed_matured",
            "split_col": "sample_role",
            "development_values": ["dev"],
            "validation_values": ["validation"],
            "oot_values": ["oot"],
            "month_col": "month",
            "weight_col": "weight",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        }

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_sample_design",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


def _create_task(client: TestClient, tmp_path: Path) -> str:
    source_dir = client.app.state.settings.workspace / f"source-{tmp_path.name}"
    source_dir.mkdir(exist_ok=True)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言策略样本设计",
            "validator": "qa",
            "source_dir": str(source_dir),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _register_and_activate(
    client: TestClient,
    task_id: str,
    tmp_path: Path,
    *,
    nan_label: bool = False,
    bad_values: list[object] | None = None,
):
    source = tmp_path / f"{task_id}.parquet"
    bad: list[object] = list(bad_values or [0, 1, 0, 1, 0, 1])
    if nan_label:
        bad[1] = None
    pd.DataFrame(
        {
            "sample_role": ["dev", "dev", "validation", "validation", "oot", "oot"],
            "month": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
            "weight": [1.0, 1.0, 2.0, 2.0, 1.5, 1.5],
            "loan_amount": [100.0, 120.0, 130.0, 140.0, 150.0, 160.0],
            "overdue_amount": [0.0, 12.0, 0.0, 14.0, 0.0, 16.0],
            "bad": bad,
        }
    ).to_parquet(source, index=False)
    settings = client.app.state.settings
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_from_upload(task_id, source, role="strategy_sample")
    repository = DataWorkspaceRepository(settings.db_path)
    activated = repository.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "sample_role": "segment",
            "month": "month",
            "weight": "weight",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "bad": "target",
        },
    )
    workspace = repository.save(
        task_id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    return dataset, workspace, mapping


def _utterance() -> str:
    return (
        "固化策略样本设计；表现窗 90 天；观察窗 2025-01-01 至 2025-12-31；"
        "成熟度已确认成熟；1 代表坏样本；切分列 sample_role；开发值 dev；"
        "验证值 validation；"
        "OOT 值 oot；月份列 month；权重列 weight；放款金额列 loan_amount；"
        "逾期金额列 overdue_amount。"
    )


def test_natural_language_sample_design_binds_active_workspace_and_renders_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_task(client, tmp_path)
    dataset, workspace, mapping = _register_and_activate(
        client,
        task_id,
        tmp_path,
    )
    llm = _SampleDesignLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert response.status_code == 202, response.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design"]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert stored.steps[0].inputs == {
        "dataset_id": dataset.id,
        "expected_dataset_content_hash": dataset.content_hash,
        "workspace_revision": workspace.revision,
        "workspace_generation": workspace.analysis_generation,
        "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
        "target_col": "bad",
        "target_bad_value": 1,
        "performance_window_status": "provided",
        "performance_window_days": 90,
        "observation_window_status": "provided",
        "observation_window_start": "2025-01-01",
        "observation_window_end": "2025-12-31",
        "maturity_status": "confirmed_matured",
        "split_col": "sample_role",
        "development_values": ["dev"],
        "validation_values": ["validation"],
        "oot_values": ["oot"],
        "month_col": "month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "drop_nan_labels": False,
    }
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["development"] is True
    assert output["unvalidated"] is True
    assert output["not_created_strategy"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert output["bundle"]["sample_design"]["identity"]["dataset_ref"] == {
        "dataset_id": dataset.id,
        "content_hash": dataset.content_hash,
        "role": "active",
    }
    assert output["bundle"]["sample_design"]["target_definition"] == {
        "column": "bad",
        "good_value": 0,
        "bad_value": 1,
        "drop_nan_labels": False,
    }
    assert output["bundle"]["metric_observations"]
    assert client.get(output["artifact"]["download_url"]).status_code == 200
    assistant_text = "\n".join(
        message.get("content", "")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "策略样本设计已固化" in assistant_text
    assert "development / unvalidated" in assistant_text
    assert "未创建或修改策略" in assistant_text
    assert "未建模、未建树、未入池、未采纳、未部署" in assistant_text
    assert len(llm.calls) == 1


def test_sample_design_requires_confirmed_active_workspace_before_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_task(client, tmp_path)
    llm = _SampleDesignLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_sample_design_workspace_required"
    assert "DataWorkspace" in response.json()["messages"][-1]["content"]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert llm.calls == []


def test_sample_design_nan_labels_use_existing_snapshot_bound_confirmation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_task(client, tmp_path)
    _register_and_activate(client, task_id, tmp_path, nan_label=True)
    llm = _SampleDesignLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert opened.status_code == 202, opened.text
    assert opened.json()["code"] == "strategy_drop_nan_labels_confirmation_required"
    assert "保留在总体、金额和权重统计中" in opened.json()["messages"][-1][
        "content"
    ]
    assert "只从坏账率/风险率分母中排除" in opened.json()["messages"][-1][
        "content"
    ]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []

    resumed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "确认将空标签仅从风险分母排除并继续"},
    )

    assert resumed.status_code == 202, resumed.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == ["strategy_sample_design"]
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert stored.steps[0].inputs["drop_nan_labels"] is True
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["bundle"]["sample_design"]["target_definition"][
        "drop_nan_labels"
    ] is True
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "bad_values",
    [
        ["0", "1", "0", "1", "0", "1"],
        [0, 1, 0, 2, 0, 1],
        [0.0, 1.0, 0.0, math.inf, 0.0, 1.0],
    ],
)
def test_sample_design_rejects_non_numeric_non_binary_or_infinite_targets_before_llm(
    tmp_path: Path,
    monkeypatch,
    bad_values: list[object],
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    task_id = _create_task(client, tmp_path)
    _register_and_activate(
        client,
        task_id,
        tmp_path,
        bad_values=bad_values,
    )
    llm = _SampleDesignLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": _utterance()},
    )

    assert response.status_code == 202, response.text
    assert response.json()["code"] == "strategy_sample_design_target_invalid"
    assert "必须是数值 0/1 或真实空值" in response.json()["messages"][-1][
        "content"
    ]
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
    assert llm.calls == []
