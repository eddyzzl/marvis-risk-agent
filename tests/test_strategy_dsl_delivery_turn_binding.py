"""Agent binding and real ToolRunner coverage for Strategy DSL delivery."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import _strategy_dsl_delivery_plan_slots
from marvis.app import create_app
from marvis.data.workspace import DataWorkspaceDraft, data_semantic_mapping_hash
from marvis.db import connect
from marvis.packs.strategy.dsl_delivery import MAX_EQUIVALENCE_ROWS
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.strategy import StrategyRepository
from tests.test_strategy_apply_tool import _runtime_fixture, _spec


def _turn_runtime(fixture: tuple) -> SimpleNamespace:
    return SimpleNamespace(settings=fixture[0])


def _context(fixture: tuple) -> SimpleNamespace:
    dataset = fixture[3]
    workspace = DataWorkspaceRepository(
        fixture[0].db_path
    ).get_or_default(fixture[1].id)
    return SimpleNamespace(
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        workspace_revision=workspace.revision,
        analysis_generation=workspace.analysis_generation,
        semantic_mapping_hash=data_semantic_mapping_hash(
            workspace.semantic_mapping
        ),
    )


def _draft(strategy_id: str | None = None) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_dsl_delivery",
        workflow_inputs=(
            {} if strategy_id is None else {"strategy_id": strategy_id}
        ),
    )


class _DeliveryLLM:
    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id

    def complete(self, **kwargs) -> str:
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_dsl_delivery",
                "workflow_inputs": {"strategy_id": self.strategy_id},
            },
            ensure_ascii=False,
        )


def test_delivery_turn_binds_exact_strategy_dataset_and_fixed_budget(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    strategy = fixture[4]

    slots = _strategy_dsl_delivery_plan_slots(
        _turn_runtime(fixture),
        fixture[1],
        _draft(strategy.id),
        context=_context(fixture),
    )

    snapshot = StrategyRepository(
        fixture[0].db_path
    ).get_strategy_snapshot(strategy.id)
    assert snapshot is not None
    context = _context(fixture)
    assert slots == {
        "strategy_ref": {
            "strategy_id": strategy.id,
            "expected_strategy_type": "approval",
            "expected_version": snapshot["metadata"]["version"],
            "expected_spec_hash": snapshot["strategy_spec_hash"],
        },
        "dataset_ref": {
            "dataset_id": fixture[3].id,
            "expected_content_hash": fixture[3].content_hash,
        },
        "workspace_ref": {
            "revision": context.workspace_revision,
            "analysis_generation": context.analysis_generation,
            "semantic_mapping_hash": context.semantic_mapping_hash,
            "active_dataset_id": None,
            "active_dataset_content_hash": None,
        },
        "maximum_equivalence_rows": MAX_EQUIVALENCE_ROWS,
    }


def test_delivery_turn_uniquely_binds_when_id_is_omitted(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path, "pricing")

    slots = _strategy_dsl_delivery_plan_slots(
        _turn_runtime(fixture),
        fixture[1],
        _draft(),
        context=_context(fixture),
    )

    assert slots["strategy_ref"]["strategy_id"] == fixture[4].id
    assert slots["strategy_ref"]["expected_strategy_type"] == "pricing"


def test_delivery_turn_requires_id_when_multiple_strategies_exist(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    repository = StrategyRepository(fixture[0].db_path)
    repository.create_strategy(
        fixture[1].id,
        build_strategy_from_spec(_spec("reject")),
    )

    with pytest.raises(StrategySetupError, match="多个可交付策略"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            fixture[1],
            _draft(),
            context=_context(fixture),
        )


def test_delivery_turn_rejects_cross_task_strategy(tmp_path: Path) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    other = _runtime_fixture(tmp_path / "other", "reject")

    with pytest.raises(StrategySetupError, match="属于当前任务"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            fixture[1],
            _draft(other[4].id),
            context=_context(fixture),
        )


def test_delivery_turn_rejects_strategy_change_before_plan_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    original = StrategyRepository.get_strategy_snapshot
    calls = 0

    def changed_snapshot(self, strategy_id):
        nonlocal calls
        calls += 1
        snapshot = original(self, strategy_id)
        if calls < 2 or snapshot is None:
            return snapshot
        changed = deepcopy(snapshot)
        changed["metadata"]["version"] += 1
        return changed

    monkeypatch.setattr(
        StrategyRepository,
        "get_strategy_snapshot",
        changed_snapshot,
    )

    with pytest.raises(StrategySetupError, match="计划创建前发生变化"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            fixture[1],
            _draft(fixture[4].id),
            context=_context(fixture),
        )


def test_delivery_turn_rechecks_unique_selection_before_plan_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    original = StrategyRepository.list_meta_for_task
    calls = 0

    def add_second_strategy_before_final_selection_check(self, task_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            self.create_strategy(
                task_id,
                build_strategy_from_spec(_spec("reject")),
            )
        return original(self, task_id)

    monkeypatch.setattr(
        StrategyRepository,
        "list_meta_for_task",
        add_second_strategy_before_final_selection_check,
    )

    with pytest.raises(StrategySetupError, match="计划创建前发生变化"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            fixture[1],
            _draft(),
            context=_context(fixture),
        )


def test_delivery_turn_rejects_unmigrated_historical_strategy_row(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    with connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET dsl_json = NULL,
                   dsl_schema_version = NULL,
                   dsl_content_hash = NULL
             WHERE id = ?
            """,
            (fixture[4].id,),
        )

    with pytest.raises(StrategySetupError, match="历史兼容行需先迁移"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            fixture[1],
            _draft(fixture[4].id),
            context=_context(fixture),
        )


def test_delivery_turn_implicit_selection_excludes_corrupt_strategy_row(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    repository = StrategyRepository(fixture[0].db_path)
    deliverable = build_strategy_from_spec(_spec("reject"))
    repository.create_strategy(fixture[1].id, deliverable)
    with connect(fixture[0].db_path) as conn:
        conn.execute(
            "UPDATE strategies SET dsl_json = ? WHERE id = ?",
            ("{not-canonical-json", fixture[4].id),
        )

    slots = _strategy_dsl_delivery_plan_slots(
        _turn_runtime(fixture),
        fixture[1],
        _draft(),
        context=_context(fixture),
    )

    assert slots["strategy_ref"]["strategy_id"] == deliverable.id
    assert slots["strategy_ref"]["expected_strategy_type"] == "reject"


def test_delivery_turn_rechecks_active_workspace_before_plan_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    settings, task, registry, dataset = fixture[:4]
    workspaces = DataWorkspaceRepository(settings.db_path)
    selected = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    context = _context(fixture)
    other_source = tmp_path / "other.parquet"
    frame = fixture[2]._backend.read_frame(
        fixture[2].resolve_verified_path(dataset.id)
    )
    frame.assign(x=frame["x"] + 10).to_parquet(other_source, index=False)
    other = registry.register_existing(
        other_source,
        task_id=task.id,
        role="strategy_sample",
    )
    original_get = type(registry).get
    calls = 0

    def switch_workspace_during_final_registry_read(self, dataset_id):
        nonlocal calls
        calls += 1
        result = original_get(self, dataset_id)
        if calls == 2:
            current = workspaces.get_or_default(task.id)
            workspaces.save(
                task.id,
                DataWorkspaceDraft(
                    active_dataset_id=other.id,
                    active_dataset_content_hash=other.content_hash,
                ),
                expected_revision=current.revision,
            )
        return result

    monkeypatch.setattr(
        type(registry),
        "get",
        switch_workspace_during_final_registry_read,
    )

    assert selected.revision == context.workspace_revision
    with pytest.raises(StrategySetupError, match="计划创建前发生变化"):
        _strategy_dsl_delivery_plan_slots(
            _turn_runtime(fixture),
            task,
            _draft(fixture[4].id),
            context=context,
        )


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_delivery_runs_real_toolrunner_and_publishes_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "segmentation")
    client = TestClient(create_app(fixture[0].workspace))
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: _DeliveryLLM(fixture[4].id),
    )
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: None,
    )

    response = client.post(
        f"/api/tasks/{fixture[1].id}/agent/messages",
        json={
            "content": (
                f"请导出 {fixture[4].id} 的策略代码，生成 Python、SQL、JSON "
                "和等价证据。"
            )
        },
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fixture[1].id}/plans"
    ).json()["plans"]
    assert plans
    assert plans[-1]["template_id"] == "strategy_dsl_delivery"
    assert plans[-1]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[-1]["id"])
    assert len(stored.steps) == 1
    output = client.app.state.plan_repo.load_step_output(stored.steps[0].id)
    assert output["strategy_type"] == "segmentation"
    assert output["equivalence"]["matched"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert len(output["artifacts"]) == 4
    assert all(
        "expected_content_hash=" in artifact["download_url"]
        for artifact in output["artifacts"]
    )
    final_message = response.json()["messages"][-1]["content"]
    assert "offline-only" in final_message
    assert "未应用、未采纳、未部署" in final_message
