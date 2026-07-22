"""Natural-language entry coverage for deterministic non-approval candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import StrategyRepository
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _install_llm(monkeypatch, llm: _FakeLLM) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


def _task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "candidate-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "income": [1000, 1200, 1800, 2200, 3000, 4000, 5000, 6500],
            "score": [500, 540, 580, 620, 660, 700, 740, 780],
            "bad": [1, 1, 0, 1, 0, 0, 0, 0],
            "ead": [800, 900, 1100, 1200, 1400, 1600, 1800, 2000],
            "pd": [0.30, 0.25, 0.18, 0.15, 0.10, 0.07, 0.04, 0.02],
        }
    ).to_csv(source / "sample.csv", index=False)
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "自然语言非审批候选策略",
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


def _latest_kind(messages: list[dict], kind: str) -> dict:
    return next(
        message
        for message in reversed(messages)
        if message.get("metadata", {}).get("kind") == kind
    )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_candidate_auto_runs_to_only_adoption_gate_and_rerenders_doc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    sample_design_ref = materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    llm = _FakeLLM(
        {
            "operation": "develop",
            "strategy_type": "segmentation",
            "candidate_design": {
                "method": "single_variable_segmentation",
                "feature_col": "income",
                "n_bands": 3,
            },
        }
    )
    _install_llm(monkeypatch, llm)

    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "按 income 开发三档风险分群候选策略"},
    )

    assert opened.status_code == 202, opened.text
    assert len(llm.calls) == 1
    assert all(
        "strategy_request" not in message.get("metadata", {})
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert not any(
        "请确认以上口径" in message.get("content", "")
        for message in opened.json()["messages"]
    )

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["template_id"] for plan in plans] == [
        "strategy_sample_design",
        "deterministic_strategy_candidate_development",
    ]
    plan = plans[1]
    assert plan["status"] == "awaiting_confirm"
    assert [step["status"] for step in plan["steps"]] == [
        "done",
        "done",
        "done",
        "done",
        "awaiting_confirm",
        "pending",
    ]
    stored_plan = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored_plan.steps[0].inputs["sample_design_ref"] == sample_design_ref
    assert stored_plan.steps[2].inputs["sample_design_ref"] == sample_design_ref
    gate = _latest_kind(opened.json()["messages"], "gate")
    assert gate["metadata"]["gate_source_tool"] == "adopt_strategy"

    strategy_repo = StrategyRepository(client.app.state.settings.db_path)
    before_meta = strategy_repo.list_meta_for_task(task_id)
    assert len(before_meta) == 1
    assert before_meta[0]["status"] == "draft"
    assert before_meta[0]["asset_status"] == "draft"
    strategy_id = before_meta[0]["id"]
    before_docs = [
        artifact
        for artifact in strategy_repo.list_strategy_artifacts(strategy_id)
        if artifact["kind"] == "strategy_doc_md"
    ]
    assert len(before_docs) == 1
    doc_path = Path(before_docs[0]["path"])
    assert doc_path.is_file()
    assert "- 状态：草稿" in doc_path.read_text(encoding="utf-8")

    reason = "策略委员会已复核确定性分群证据，批准本地采纳"
    completed = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认采纳",
            "expected_step_id": gate["metadata"]["step_id"],
            "adjust_params": {"adoption_reason": reason},
        },
    )

    assert completed.status_code == 202, completed.text
    final_plan = next(
        item
        for item in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        if item["id"] == plan["id"]
    )
    assert final_plan["status"] == "done"
    assert [step["status"] for step in final_plan["steps"]] == ["done"] * 6
    final_meta = strategy_repo.get_strategy_meta(strategy_id)
    assert final_meta["status"] == "adopted"
    assert final_meta["asset_status"] == "adopted_local"
    assert final_meta["adoption_reason"] == reason

    after_docs = [
        artifact
        for artifact in strategy_repo.list_strategy_artifacts(strategy_id)
        if artifact["kind"] == "strategy_doc_md"
    ]
    assert len(after_docs) == 2
    final_doc_path = Path(after_docs[-1]["path"])
    assert final_doc_path != doc_path
    assert "- 状态：草稿" in doc_path.read_text(encoding="utf-8")
    final_text = final_doc_path.read_text(encoding="utf-8")
    assert "- 状态：本地已采纳" in final_text
    assert "本地已采纳不代表生产环境已上线" in final_text
    assert reason in final_text


def test_missing_candidate_economics_stays_structured_and_does_not_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "develop",
            "strategy_type": "limit",
            "candidate_design": {
                "method": "score_band_limit",
                "score_col": "score",
                "n_bands": 3,
                "limit_grid": [1000, 2000, 4000],
                "max_expected_loss_per_account": 100,
            },
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "按 score 开发额度候选策略"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == "candidate_economics_incomplete"
    assert response.json()["fields"] == [
        "pd_col/pd_value",
        "lgd_col/lgd_value",
        "utilization_col/utilization_value",
    ]
    assert len(llm.calls) == 1
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []


def test_collection_development_fails_closed_before_llm_or_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _FakeLLM(
        {
            "operation": "develop",
            "strategy_type": "segmentation",
            "candidate_design": {
                "method": "single_variable_segmentation",
                "feature_col": "income",
                "n_bands": 3,
            },
        }
    )
    _install_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "帮我开发一个催收分案策略"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == "collection_strategy_unsupported"
    assert llm.calls == []
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == []
