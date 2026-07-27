"""Focused Agent routing for governed candidate monthly stability."""

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
    _candidate_monthly_stability_plan_slots,
    _strategy_request_preflight,
    _strategy_request_requires_dataset,
)
from marvis.app import create_app
from marvis.db import PluginRepository, init_db
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.packs.strategy.errors import StrategyError
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


ASSET_ID = "candidate-asset-" + "a" * 32
ENTRY_ID = "pool-entry-" + "b" * 32


class _PayloadLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete(self, **_kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


def _draft(**inputs: object) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="candidate_monthly_stability",
        workflow_inputs=inputs,
    )


def test_candidate_stability_is_artifact_driven_and_template_is_one_step() -> None:
    load_builtin_templates()
    draft = _draft(asset_id=ASSET_ID)
    template = get_template("strategy_candidate_monthly_stability")

    assert _strategy_request_requires_dataset(draft) is False
    assert len(template.steps) == 1
    assert template.steps[0].tool_ref == ToolRef(
        "strategy",
        "measure_candidate_monthly_stability",
    )
    assert template.steps[0].needs_confirmation is False


def test_candidate_stability_template_omits_the_unused_oneof_branch(
    tmp_path: Path,
) -> None:
    load_builtin_templates()
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(
        plugins,
        Path(__file__).parents[1] / "marvis" / "packs",
    )
    tools = ToolRegistry(plugins)
    validator = PlanValidator(tools)
    planner = Planner(tools, lambda: None, validator)
    template = get_template("strategy_candidate_monthly_stability")

    asset_plan = planner.from_template(
        template,
        {
            "source_kind": "univariate_asset",
            "source_artifact_id": "1" * 64,
            "expected_artifact_content_hash": "2" * 64,
            "expected_asset_id": ASSET_ID,
            "expected_asset_hash": "3" * 64,
        },
        task_id="task-1",
    )
    pool_plan = planner.from_template(
        template,
        {
            "source_kind": "pool_entry",
            "strategy_type": "approval",
            "expected_pool_revision": 2,
            "expected_pool_snapshot_hash": "4" * 64,
            "entry_id": ENTRY_ID,
        },
        task_id="task-1",
    )

    assert validator.validate(asset_plan) == []
    assert validator.validate(pool_plan) == []
    assert set(asset_plan.steps[0].inputs) == {
        "source_kind",
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
    }
    assert set(pool_plan.steps[0].inputs) == {
        "source_kind",
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "entry_id",
    }


@pytest.mark.parametrize(
    ("draft", "expected_pointer"),
    [
        (
            _draft(asset_id=ASSET_ID),
            {"source_kind": "univariate_asset", "asset_id": ASSET_ID},
        ),
        (
            _draft(strategy_type="approval", entry_id=ENTRY_ID),
            {
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": ENTRY_ID,
            },
        ),
    ],
)
def test_turn_preflight_injects_only_platform_resolved_tool_inputs(
    monkeypatch: pytest.MonkeyPatch,
    draft: StandardWorkflowRequestDraft,
    expected_pointer: dict[str, object],
) -> None:
    captured: dict[str, object] = {}
    resolved = (
        {
            "source_kind": "univariate_asset",
            "source_artifact_id": "1" * 64,
            "expected_artifact_content_hash": "2" * 64,
            "expected_asset_id": ASSET_ID,
            "expected_asset_hash": "3" * 64,
        }
        if "asset_id" in expected_pointer
        else {
            "source_kind": "pool_entry",
            "strategy_type": "approval",
            "expected_pool_revision": 2,
            "expected_pool_snapshot_hash": "4" * 64,
            "entry_id": ENTRY_ID,
        }
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: "read-runtime",
    )

    def resolve(runtime, *, task_id: str, user_pointer):
        captured.update(
            {
                "runtime": runtime,
                "task_id": task_id,
                "user_pointer": dict(user_pointer),
            }
        )
        return resolved

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_candidate_monthly_stability_inputs",
        resolve,
    )

    slots = _candidate_monthly_stability_plan_slots(
        SimpleNamespace(),
        SimpleNamespace(id="task-1"),
        draft,
    )

    assert slots == resolved
    assert captured == {
        "runtime": "read-runtime",
        "task_id": "task-1",
        "user_pointer": expected_pointer,
    }


def test_turn_preflight_maps_missing_month_to_plan_free_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_v2_read_runtime",
        lambda runtime: "read-runtime",
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_candidate_monthly_stability_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StrategyError(
                "candidate monthly stability requires a month field in the "
                "governed StrategySampleDesign"
            )
        ),
    )
    runtime = SimpleNamespace()
    task = SimpleNamespace(id="task-1")
    draft = _draft(asset_id=ASSET_ID)

    with pytest.raises(StrategySetupError, match="月份字段"):
        _candidate_monthly_stability_plan_slots(runtime, task, draft)
    assert _strategy_request_preflight(runtime, task, draft) == (
        "candidate_monthly_stability_month_required",
        (
            "当前受治理 StrategySampleDesign 没有唯一且非空的月份字段；"
            "请先补充并重新固化 month 口径，再测算候选逐月稳定性。"
        ),
    )


@pytest.mark.e2e
def test_natural_language_missing_month_never_creates_a_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame(
        {
            "score": [400, 500, 600, 700],
            "bad": [1, 0, 1, 0],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "候选逐月稳定性",
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
            "workflow": "candidate_monthly_stability",
            "workflow_inputs": {"asset_id": ASSET_ID},
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.resolve_candidate_monthly_stability_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            StrategyError(
                "candidate monthly stability requires a month field in the "
                "governed StrategySampleDesign"
            )
        ),
    )

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"请对已有单变量候选资产 {ASSET_ID} 做逐月稳定性和 PSI 分析。"
            )
        },
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert (
        response.json()["code"]
        == "candidate_monthly_stability_month_required"
    )
    assert client.app.state.plan_repo.list_plans_for_task(task_id) == []


@pytest.mark.slow
@pytest.mark.e2e
def test_natural_language_asset_stability_runs_the_pointer_bound_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    row_count = 90
    pd.DataFrame(
        {
            "month": [
                "202501" if index < 30 else "202502" if index < 60 else "202503"
                for index in range(row_count)
            ],
            "score": [350 + (index % 30) * 10 for index in range(row_count)],
            "bad": [1 if index % 4 == 0 else 0 for index in range(row_count)],
        }
    ).to_csv(source / "sample.csv", index=False)
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "候选逐月稳定性成功纵线",
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
    materialize_mature_strategy_sample_design(
        client,
        task_id,
        monkeypatch,
    )
    llm = _PayloadLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": {
                "feature": "score",
                "method": "equal_width",
                "bin_count": 3,
                "min_bin_pct": 0.02,
                "selection": {
                    "risk_threshold": {"operator": ">=", "value": 0.0}
                },
            },
        }
    )
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    refined = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                "选择 score 等距分析中观测坏率大于等于 0% 的全部候选箱。"
            )
        },
    )
    assert refined.status_code == 202, refined.text
    refinement_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    assert refinement_plan.template_id == (
        "strategy_univariate_candidate_refinement"
    )
    refinement_output = client.app.state.plan_repo.load_step_output(
        refinement_plan.steps[-1].id
    )
    asset_id = refinement_output["asset_id"]

    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "candidate_monthly_stability",
        "workflow_inputs": {"asset_id": asset_id},
    }
    measured = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"请对已有单变量候选资产 {asset_id} 做逐月稳定性和 PSI 分析。"
            )
        },
    )

    assert measured.status_code == 202, measured.text
    plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    assert plan.template_id == "strategy_candidate_monthly_stability"
    assert plan.status.value == "done"
    assert len(plan.steps) == 1
    assert set(plan.steps[0].inputs) == {
        "source_kind",
        "source_artifact_id",
        "expected_artifact_content_hash",
        "expected_asset_id",
        "expected_asset_hash",
    }
    output = client.app.state.plan_repo.load_step_output(plan.steps[0].id)
    assert output["source_kind"] == "univariate_asset"
    assert output["basis"] == "asset_rule_hit"
    assert output["month_col"] == "month"
    assert output["population_count"] == row_count
    assert output["month_count"] == 3
    assert len(output["artifacts"]) == 1
    assert client.get(output["artifacts"][0]["download_url"]).status_code == 200

    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "strategy_pool_add_candidate",
        "workflow_inputs": {
            "candidate_asset_id": asset_id,
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "action": {"type": "reject"},
        },
    }
    added = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"把 {asset_id} 加入审批 Strategy Pool；默认动作 approval，"
                "命中动作 reject 拒绝。"
            )
        },
    )
    assert added.status_code == 202, added.text
    pool = StrategyCandidatePoolRepository(
        client.app.state.settings.db_path
    ).get_current(task_id, "approval")
    assert pool is not None
    entry_id = pool["entries"][0]["entry_id"]

    llm.payload = {
        "request_kind": "standard_workflow",
        "workflow": "candidate_monthly_stability",
        "workflow_inputs": {
            "strategy_type": "approval",
            "entry_id": entry_id,
        },
    }
    pool_measured = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"测算审批策略池条目 {entry_id} 的逐月 PSI 和稳定性。"
            )
        },
    )

    assert pool_measured.status_code == 202, pool_measured.text
    pool_plan = client.app.state.plan_repo.list_plans_for_task(task_id)[-1]
    assert pool_plan.template_id == "strategy_candidate_monthly_stability"
    assert pool_plan.status.value == "done"
    assert set(pool_plan.steps[0].inputs) == {
        "source_kind",
        "strategy_type",
        "expected_pool_revision",
        "expected_pool_snapshot_hash",
        "entry_id",
    }
    pool_output = client.app.state.plan_repo.load_step_output(
        pool_plan.steps[0].id
    )
    assert pool_output["source_kind"] == "pool_entry"
    assert pool_output["basis"] == "pool_entry_incremental_first_match"
    assert pool_output["month_col"] == "month"
    assert pool_output["population_count"] == row_count
    assert pool_output["month_count"] == 3
