"""Natural-language Cross build -> exact cells -> separate Pool add vertical."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import _strategy_request_requires_dataset
from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _create_task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "cross-cell-source"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "age": [20, 22, 24, 26, 40, 42, 44, 46, 60, 62, 64, 66],
            "score": [100, 110, 300, 310, 120, 130, 320, 330, 140, 150, 340, 350],
            "bad": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "Cross Matrix 精确单元格选择",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_cross_cell_selection_is_artifact_only_and_never_requires_dataset() -> None:
    draft = StandardWorkflowRequestDraft(
        workflow="cross_matrix_cell_selection",
        workflow_inputs={
            "cross_asset_id": "candidate-asset-" + "a" * 32,
            "cell_ids": ["cross-cell-" + "b" * 32],
        },
    )

    assert _strategy_request_requires_dataset(draft) is False


@pytest.mark.slow
@pytest.mark.e2e
def test_cross_matrix_exact_cells_enter_pool_only_on_separate_third_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _create_task(client, tmp_path)
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_matrix_analysis",
            "workflow_inputs": {
                "x_feature": "age",
                "x_method": "equal_width",
                "y_feature": "score",
                "y_method": "equal_width",
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    built = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "构建 age 等距 3 箱 × score 等距 3 箱的二维交叉矩阵"},
    )
    assert built.status_code == 202, built.text
    build_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[0]
    cross = client.app.state.plan_repo.load_step_output(build_plan.steps[1].id)
    asset_id = cross["asset_id"]
    source_cells = [
        cell["cell_id"] for cell in cross["cross_matrix_candidate"]["matrix"]["cells"]
    ]
    requested = [source_cells[2], source_cells[0]]
    reason = "人工确认用于风险复核"
    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_cell_selection",
        "workflow_inputs": {
            "cross_asset_id": asset_id,
            "cell_ids": requested,
            "selection_reason": reason,
        },
    }

    selected = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"从 {asset_id} 选择 Cross Matrix 单元格 "
                f"{requested[0]}、{requested[1]}，选择理由：{reason}。"
            )
        },
    )
    assert selected.status_code == 202, selected.text
    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert [plan.template_id for plan in plans] == [
        "strategy_cross_matrix_analysis",
        "strategy_cross_matrix_cell_selection",
    ]
    selection = client.app.state.plan_repo.load_step_output(plans[-1].steps[0].id)
    selection_id = selection["selection_id"]
    selection_artifact_id = selection["artifacts"][0]["artifact_id"]
    assert selection_id.startswith("cross-matrix-cell-selection-")
    assert selection["source_asset_id"] == asset_id
    assert selection["cell_ids"] == [source_cells[0], source_cells[2]]
    assert selection["selection_reason"] == reason
    assert selection["not_admitted"] is True
    assert selection["not_applied"] is True
    assert selection["not_adopted"] is True
    assert selection["not_deployed"] is True
    assert (
        StrategyCandidatePoolRepository(client.app.state.settings.db_path).get_current(
            task_id, "approval"
        )
        is None
    )

    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "selection_id": selection_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    }
    added = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把选择结果 {selection_id} 加入 Strategy Pool；"
                "策略池类型：approval；Pool 默认动作：approval；命中动作：reject。"
            )
        },
    )

    assert added.status_code == 202, added.text
    plans = client.app.state.plan_repo.list_plans_for_task(task_id)
    assert [plan.template_id for plan in plans] == [
        "strategy_cross_matrix_analysis",
        "strategy_cross_matrix_cell_selection",
        "strategy_pool_add_candidate",
    ]
    assert plans[-1].status == "done"
    pool = StrategyCandidatePoolRepository(
        client.app.state.settings.db_path
    ).get_current(task_id, "approval")
    assert pool is not None
    assert pool["status"] == "draft"
    assert pool["validation_status"] == "unvalidated"
    [entry] = pool["entries"]
    assert entry["source"]["artifact_id"] == selection_artifact_id
    assert entry["source"]["artifact_kind"] == (
        "strategy_cross_matrix_cell_selection_json"
    )
    assert entry["source"]["asset_id"] == asset_id
    assert entry["source"]["fragment_id"] == selection["fragment_id"]
    assert entry["source"]["effect_id"] == selection["effect_id"]
    assert entry["action"]["type"] == "reject"
    assert (
        StrategyRepository(client.app.state.settings.db_path).list_for_task(task_id)
        == []
    )
