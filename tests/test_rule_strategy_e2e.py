"""End-to-end S4 journey with only the evidence-bound adoption gate.

Mining, deterministic keep-all selection, evaluation, construction and
backtesting are reversible and run after the overview is accepted. The driver
stops only before adoption; a human reason then authorizes adoption and export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.memory_bridge import capture_agent_memory_for_driver_done
from marvis.agent.plan_driver import PlanDriver
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import (
    DatasetRepository,
    PluginRepository,
    PlanRepository,
    TaskRepository,
    init_db,
)
from marvis.domain import TaskCreate
from marvis.governance.repository import GovernanceRepository
from marvis.governance.service import GovernanceService
from marvis.orchestrator.contracts import PlanStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.templates import load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _driver(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    tool_registry = ToolRegistry(plugin_registry)
    plan_repo = PlanRepository(settings.db_path)
    governance_repo = GovernanceRepository(settings.db_path)
    principal = governance_repo.create_local_principal(
        display_name="规则策略 E2E 操作员"
    )
    governance_service = GovernanceService(
        plan_repo=plan_repo,
        tool_registry=tool_registry,
        strategy_repo=StrategyRepository(settings.db_path),
        governance_repo=governance_repo,
    )
    runner = ToolRunner(
        tool_registry, plugin_repo, python_executable=sys.executable,
        datasets_root=settings.datasets_dir, workspace=settings.workspace,
        governance=governance_repo, binding_resolver=governance_service,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    executor = PlanExecutor(
        plan_repo,
        runner,
        Reviewer(lambda: FakeLLM()),
        None,
        FakeHooks(),
        HarnessState(plan_repo),
        authorizer=governance_service,
    )
    planner = Planner(tool_registry, lambda: FakeLLM(), PlanValidator(tool_registry))
    driver = PlanDriver(
        plan_repo,
        executor,
        planner=planner,
        validator=PlanValidator(tool_registry),
        governance_service=governance_service,
        local_principal=principal,
    )
    load_builtin_templates()
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="S4 端到端规则策略", model_version="dev", validator="qa",
            source_dir=str(tmp_path / "source"), algorithm="lr", run_mode="agent",
            task_type="strategy", target_col="bad",
        )
    )
    return driver, registry, plan_repo, settings, task


def _register(registry, tmp_path, task_id):
    # 40 rows: bad concentrated where f1 is low, giving several separable rules
    # with clear lift so mining returns a non-empty candidate set.
    f1 = list(range(10, 410, 10))
    f2 = [i % 3 for i in range(40)]
    bad = [1 if v <= 120 else 0 for v in f1]
    frame = pd.DataFrame({"f1": f1, "f2": f2, "bad": bad})
    path = tmp_path / "e2e_rules.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def _materialize_sample_ref(
    settings,
    task,
    dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    client = TestClient(create_app(settings))
    workspaces = DataWorkspaceRepository(settings.db_path)
    current = workspaces.get_or_default(task.id)
    activated = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=current.revision,
    )
    workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={"bad": "target"},
            ),
        ),
        expected_revision=activated.revision,
    )
    return materialize_mature_strategy_sample_design(
        client,
        task.id,
        monkeypatch,
    )


@pytest.mark.slow
def test_rule_strategy_runs_reversible_steps_to_only_adoption_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    driver, registry, plan_repo, settings, task = _driver(tmp_path)
    dataset = _register(registry, tmp_path, task.id)
    sample_design_ref = _materialize_sample_ref(
        settings,
        task,
        dataset,
        monkeypatch,
    )

    turn = driver.start(
        task_id=task.id, template_id="rule_strategy",
        slots={
            "dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1", "f2"],
            "min_support": 0.05, "min_lift": 1.2,
            "sample_design_ref": sample_design_ref,
        },
    )
    assert turn.status == PlanStatus.VALIDATED.value
    plan_id = turn.plan_id

    # One start confirmation runs every reversible step, then stops at adoption.
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert gate.metadata["gate_source_tool"] == "adopt_strategy"
    plan = plan_repo.load_plan(plan_id)
    mine_step = next(s for s in plan.steps if s.title == "挖掘规则")
    assert mine_step.status.value == "done"
    mine_out = plan_repo.load_step_output(mine_step.id)
    n_candidates = len(mine_out["candidate_rules"])
    assert n_candidates >= 2
    select_step = next(s for s in plan.steps if s.title == "选择规则集")
    assert select_step.status.value == "done"
    selected_out = plan_repo.load_step_output(select_step.id)
    assert selected_out["selected_count"] == n_candidates
    evaluate_step = next(s for s in plan.steps if s.title == "评估规则集")
    assert evaluate_step.status.value == "done"
    evaluate_out = plan_repo.load_step_output(evaluate_step.id)
    assert len(evaluate_out["waterfall"]) == n_candidates
    build_step = next(s for s in plan.steps if s.title == "构造策略")
    assert build_step.status.value == "done"
    backtest_step = next(s for s in plan.steps if s.title == "回测策略")
    assert backtest_step.status.value == "done"
    backtest_out = plan_repo.load_step_output(backtest_step.id)
    assert 0.0 <= backtest_out["approval_rate"] <= 1.0
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "awaiting_confirm"  # mandatory gate

    # Bind the final reason to the exact adoption gate instead of inheriting a
    # prefilled task/template value.
    turn = driver.resume(
        plan_id=plan_id,
        user_text="确认采纳",
        run_seq=2,
        adjust_params={"adoption_reason": "committee approved"},
        expected_step_id=adopt_step.id,
    )
    assert turn.status == PlanStatus.DONE.value
    done = turn.messages[-1]
    assert done.stage == "done"
    assert "策略文档已生成" in done.content

    plan = plan_repo.load_plan(plan_id)
    adopt_out = plan_repo.load_step_output(adopt_step.id)
    assert adopt_out["status"] == "adopted"
    assert {a["kind"] for a in adopt_out["artifacts"]} == {"decision_table_csv", "monitoring_plan_json"}
    for artifact in adopt_out["artifacts"]:
        assert Path(artifact["path"]).exists()
    doc_step = next(s for s in plan.steps if s.title == "策略文档")
    doc_out = plan_repo.load_step_output(doc_step.id)
    assert Path(doc_out["doc_path"]).exists()

    # MEM-1: strategy_experience capture reuses the S2 surface (no new kind); the
    # cutoff_summary carries the adopted rule conditions.
    capture_agent_memory_for_driver_done(
        settings, task,
        done_message_content=done.content, done_message_metadata=dict(done.metadata),
    )
    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="strategy_experience", limit=10)
    assert len(entries) == 1
    assert entries[0].payload["source_task_id"] == task.id
    strategy = StrategyRepository(settings.db_path).get_strategy(adopt_out["strategy_id"])
    assert entries[0].payload["cutoff_summary"] == "；".join(r.condition for r in strategy.rules)


@pytest.mark.slow
def test_rule_strategy_default_selection_keeps_all_candidates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The automatic reversible selection keeps every mined candidate by default."""
    driver, registry, plan_repo, settings, task = _driver(tmp_path)
    dataset = _register(registry, tmp_path, task.id)
    sample_design_ref = _materialize_sample_ref(
        settings,
        task,
        dataset,
        monkeypatch,
    )
    turn = driver.start(
        task_id=task.id, template_id="rule_strategy",
        slots={
            "dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1", "f2"],
            "min_support": 0.05, "min_lift": 1.2,
            "sample_design_ref": sample_design_ref,
        },
    )
    plan_id = turn.plan_id
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    select_step = next(s for s in plan.steps if s.title == "选择规则集")
    mine_step = next(s for s in plan.steps if s.title == "挖掘规则")
    select_out = plan_repo.load_step_output(select_step.id)
    mine_out = plan_repo.load_step_output(mine_step.id)
    assert select_out["selected_count"] == len(mine_out["candidate_rules"])
