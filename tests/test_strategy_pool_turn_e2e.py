"""Turn-boundary tests for task-owned Strategy Pool integrity inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import _strategy_pool_plan_slots
from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.repositories.strategy_pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository


ASSET_ID = "candidate-asset-" + "a" * 32
ASSET_HASH = "b" * 64
RULE_1 = "candidate-rule-" + "1" * 32
RULE_2 = "candidate-rule-" + "2" * 32
ENTRY_1 = "pool-entry-" + "3" * 32
ENTRY_2 = "pool-entry-" + "4" * 32


class _PoolRepository:
    def __init__(self, current) -> None:
        self.current = current

    def get_current(self, task_id: str, strategy_type: str):
        assert task_id == "task-1"
        assert strategy_type == "approval"
        return self.current


class _ArtifactRepository:
    def __init__(self, artifact: dict) -> None:
        self.artifact = artifact

    def list_for_task(self, task_id: str) -> list[dict]:
        assert task_id == "task-1"
        return [self.artifact]


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def _runtime(tmp_path: Path):
    return SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "marvis.sqlite"))


def test_add_turn_binds_asset_and_absent_pool_hash_from_platform_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {"asset_id": ASSET_ID, "asset_hash": ASSET_HASH}
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path = tmp_path / "candidate.json"
    path.write_bytes(content)
    content_hash = hashlib.sha256(content).hexdigest()
    artifact = {
        "id": "artifact-1",
        "kind": "strategy_candidate_asset_json",
        "path": str(path),
        "content_hash": content_hash,
        "origin_tool": "strategy.refine_univariate_candidate",
        "provenance": {"asset_id": ASSET_ID, "asset_hash": ASSET_HASH},
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(None),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.TaskArtifactRepository",
        lambda db_path: _ArtifactRepository(artifact),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.validate_candidate_asset", lambda value: value
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.canonical_candidate_asset_json",
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_add_candidate",
        workflow_inputs={
            "candidate_asset_id": ASSET_ID,
            "strategy_type": "approval",
            "default_action": {"type": "approval", "value": "approve"},
            "action": {"type": "reject", "value": "reject"},
        },
    )

    slots = _strategy_pool_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        draft,
    )

    assert slots["source_artifact_id"] == "artifact-1"
    assert slots["expected_artifact_content_hash"] == content_hash
    assert slots["expected_asset_id"] == ASSET_ID
    assert slots["expected_asset_hash"] == ASSET_HASH
    assert slots["expected_pool_revision"] == 0
    assert slots["expected_pool_snapshot_hash"] == ABSENT_POOL_SNAPSHOT_HASH


def test_reorder_turn_resolves_entry_ids_and_requires_the_complete_current_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current = {
        "revision": 7,
        "entries": [
            {"entry_id": ENTRY_1, "rule_id": RULE_1},
            {"entry_id": ENTRY_2, "rule_id": RULE_2},
        ],
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(current),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda snapshot: "c" * 64,
    )
    complete = StandardWorkflowRequestDraft(
        workflow="strategy_pool_reorder",
        workflow_inputs={
            "strategy_type": "approval",
            "ordered_ids": [ENTRY_2, RULE_1],
        },
    )

    slots = _strategy_pool_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        complete,
    )

    assert slots["ordered_rule_ids"] == [RULE_2, RULE_1]
    assert slots["expected_pool_revision"] == 7
    assert slots["expected_pool_snapshot_hash"] == "c" * 64

    partial = StandardWorkflowRequestDraft(
        workflow="strategy_pool_reorder",
        workflow_inputs={
            "strategy_type": "approval",
            "ordered_ids": [ENTRY_2],
        },
    )
    with pytest.raises(StrategySetupError, match="完整、无重复"):
        _strategy_pool_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            partial,
        )


def test_compile_turn_requires_an_existing_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda db_path: _PoolRepository(None),
    )
    draft = StandardWorkflowRequestDraft(
        workflow="strategy_pool_compile",
        workflow_inputs={"strategy_type": "approval"},
    )

    with pytest.raises(StrategySetupError, match="还没有.*Strategy Pool"):
        _strategy_pool_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            draft,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_add_and_read_only_compile_auto_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "score": [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
            "loan_amount": [100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
            "overdue_amount": [0, 0, 5, 0, 10, 0, 15, 0, 20, 0, 25, 30],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "Strategy Pool NL round trip",
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

    refinement_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": {
                "feature": "score",
                "method": "equal_width",
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "overdue_amount",
                "selection": {"risk_threshold": {"operator": ">=", "value": 0.5}},
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: refinement_llm,
    )
    refined = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "选择 score 等距分析中观测坏率大于等于 50% 的候选箱"},
    )
    assert refined.status_code == 202, refined.text
    refinement_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[0]
    asset_output = client.app.state.plan_repo.load_step_output(
        refinement_plan.steps[-1].id
    )
    asset_id = asset_output["asset_id"]

    add_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_add_candidate",
            "workflow_inputs": {
                "candidate_asset_id": asset_id,
                "strategy_type": "approval",
                "default_action": {"type": "approval"},
                "action": {"type": "reject"},
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: add_llm,
    )
    opened = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把 {asset_id} 加入审批 Strategy Pool；默认动作 approval，"
                "命中动作 reject 拒绝"
            )
        },
    )
    assert opened.status_code == 202, opened.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert len(plans) == 2, [
        (message.get("content"), message.get("metadata"))
        for message in opened.json()["messages"]
    ]
    add_plan = plans[-1]
    assert add_plan["template_id"] == "strategy_pool_add_candidate"
    assert add_plan["status"] == "done"
    assert add_plan["steps"][0]["status"] == "done"
    assert add_plan["steps"][0]["needs_confirmation"] is False
    assert not any(
        message.get("metadata", {}).get("kind") == "gate"
        for message in opened.json()["messages"]
    )
    assert StrategyRepository(client.app.state.settings.db_path).list_for_task(
        task_id
    ) == []
    pool = StrategyCandidatePoolRepository(
        client.app.state.settings.db_path
    ).get_current(task_id, "approval")
    assert pool is not None
    assert pool["status"] == "draft"
    assert pool["validation_status"] == "unvalidated"
    assert [entry["source"]["asset_id"] for entry in pool["entries"]] == [asset_id]
    add_text = "\n".join(
        message.get("content", "")
        for message in opened.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "Strategy Pool 已更新" in add_text
    assert "development / unvalidated" in add_text
    assert "未采纳、未部署" in add_text

    artifacts_before = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).list_for_task(task_id)
    compile_llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "strategy_pool_compile",
            "workflow_inputs": {"strategy_type": "approval"},
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: compile_llm,
    )
    compiled = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "只预览并编译审批 Strategy Pool 草案，不要采纳或部署"},
    )
    assert compiled.status_code == 202, compiled.text
    compile_plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    assert compile_plan["template_id"] == "strategy_pool_compile"
    assert compile_plan["status"] == "done"
    assert compile_plan["steps"][0]["needs_confirmation"] is False
    stored_compile = client.app.state.plan_repo.load_plan(compile_plan["id"])
    compile_output = client.app.state.plan_repo.load_step_output(
        stored_compile.steps[0].id
    )
    assert compile_output["design_hash"]
    assert compile_output["strategy_spec"]["rules"][0]["rule_id"] == pool[
        "entries"
    ][0]["rule_id"]
    assert TaskArtifactRepository(
        client.app.state.settings.db_path
    ).list_for_task(task_id) == artifacts_before
    compiled_text = "\n".join(
        message.get("content", "")
        for message in compiled.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert "Strategy Pool 编译完成" in compiled_text
    assert "只读草案" in compiled_text
    assert "未采纳、未部署" in compiled_text
