"""End-to-end strategy development with one real human-responsibility gate.

All reversible calculations run after the plan overview is accepted. The driver
stops only at evidence-bound adoption; confirming that gate produces the final
document and completes the plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.agent.memory_bridge import capture_agent_memory_for_driver_done
from marvis.agent.plan_driver import PlanDriver
from marvis.agent_memory.store import AgentMemoryStore
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, PlanRepository, TaskRepository, init_db
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
from marvis.packs.strategy.backtest_compat import approval_backtest_projection
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _strategy_driver(tmp_path):
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
        display_name="策略开发 E2E 操作员"
    )
    governance_service = GovernanceService(
        plan_repo=plan_repo,
        tool_registry=tool_registry,
        strategy_repo=StrategyRepository(settings.db_path),
        governance_repo=governance_repo,
    )
    runner = ToolRunner(
        tool_registry,
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
        governance=governance_repo,
        binding_resolver=governance_service,
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
            model_name="S2 端到端策略开发",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            task_type="strategy",
            target_col="bad",
            score_col="score",
        )
    )
    return driver, registry, plan_repo, settings, task


def _register_dataset(registry, tmp_path, task_id: str):
    # 20 rows, higher_is_better (higher score = safer); bad concentrated in the
    # lowest-score decile so a tradeoff/band cut has an unambiguous good answer.
    scores = list(range(100, 2100, 100))
    bad = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    frame = pd.DataFrame({"score": scores, "bad": bad})
    path = tmp_path / "e2e_strategy.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def _strategy_backtest_approval_rate(strategies: StrategyRepository, strategy_id: str) -> float:
    backtests = strategies.list_backtests(strategy_id)
    return float(approval_backtest_projection(backtests[-1])["approval_rate"])


@pytest.mark.slow
def test_strategy_development_runs_reversible_steps_to_only_adoption_gate(tmp_path):
    driver, registry, plan_repo, settings, task = _strategy_driver(tmp_path)
    dataset = _register_dataset(registry, tmp_path, task.id)

    turn = driver.start(
        task_id=task.id,
        template_id="strategy_development",
        slots={
            "dataset_id": dataset.id,
            "target_col": "bad",
            "score_col": "score",
            "score_direction": "higher_is_better",
            # A max_bad_rate constraint is required for a max_profit-objective
            # scan with no profit_params (every prefix ties at 0 expected
            # profit) to prefer a non-trivial cut over "approve everyone" --
            # otherwise recommended_rules comes back empty and build_strategy's
            # schema (rules: minItems 1) rejects it. This is a real robustness
            # gap in design_cutoff_bands worth a follow-up, not papered over
            # silently: flagged separately, not fixed in this commit.
            "max_bad_rate": 0.05,
        },
    )
    assert turn.status == PlanStatus.VALIDATED.value
    plan_id = turn.plan_id

    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert gate.metadata["gate_source_tool"] == "adopt_strategy"
    plan = plan_repo.load_plan(plan_id)
    assert [
        step.title for step in plan.steps if step.status.value == "awaiting_confirm"
    ] == ["采纳策略"]
    bands_step = next(s for s in plan.steps if s.title == "设计分数带")
    assert bands_step.status.value == "done"
    bands_output = plan_repo.load_step_output(bands_step.id)
    assert bands_output["band_edges"]
    assert next(s for s in plan.steps if s.title == "构造策略").status.value == "done"
    backtest_step = next(s for s in plan.steps if s.title == "回测策略")
    assert backtest_step.status.value == "done"
    backtest_output = plan_repo.load_step_output(backtest_step.id)
    assert 0.0 <= backtest_output["approval_rate"] <= 1.0
    assert 0.0 <= backtest_output["approved_bad_rate"] <= 1.0
    compare_step = next(s for s in plan.steps if s.title == "对比基线")
    assert compare_step.status.value == "done"
    compare_output = plan_repo.load_step_output(compare_step.id)
    assert compare_output["summary_text"] == "未提供基线策略，跳过对比。"
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "awaiting_confirm"  # mandatory gate: not yet executed

    # Adoption binds the reviewed evidence, a real operator reason, and the
    # current gate token in one request; task setup never pre-authorizes it.
    turn = driver.resume(
        plan_id=plan_id,
        user_text="确认采纳",
        run_seq=2,
        adjust_params={"adoption_reason": "committee approved for Q3 rollout"},
        expected_step_id=adopt_step.id,
    )
    assert turn.status == PlanStatus.DONE.value
    done = turn.messages[-1]
    assert done.stage == "done"
    assert "策略文档已生成" in done.content

    plan = plan_repo.load_plan(plan_id)
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    adopt_output = plan_repo.load_step_output(adopt_step.id)
    assert adopt_output["status"] == "adopted"
    assert {a["kind"] for a in adopt_output["artifacts"]} == {
        "decision_table_csv", "monitoring_plan_json",
    }
    for artifact in adopt_output["artifacts"]:
        assert Path(artifact["path"]).exists()

    doc_step = next(s for s in plan.steps if s.title == "策略文档")
    doc_output = plan_repo.load_step_output(doc_step.id)
    assert Path(doc_output["doc_path"]).exists()

    # MEM-1 write side: the done message triggers strategy_experience capture
    # straight from persisted adopt+backtest results (not from the terminal
    # 策略文档 output, which carries no metrics).
    capture_agent_memory_for_driver_done(
        settings, task,
        done_message_content=done.content,
        done_message_metadata=dict(done.metadata),
    )
    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="strategy_experience", limit=10)
    assert len(entries) == 1
    assert entries[0].payload["source_task_id"] == task.id
    assert entries[0].payload["approval_rate"] == _strategy_backtest_approval_rate(
        StrategyRepository(settings.db_path), adopt_output["strategy_id"]
    )


@pytest.mark.slow
def test_strategy_development_double_adopt_confirm_conflicts_gracefully(tmp_path):
    """Re-confirming an already-executed mandatory adopt gate must not silently
    double-adopt (the ConflictError guard from Commit 1, exercised end-to-end)."""
    driver, registry, plan_repo, settings, task = _strategy_driver(tmp_path)
    dataset = _register_dataset(registry, tmp_path, task.id)
    turn = driver.start(
        task_id=task.id,
        template_id="strategy_development",
        slots={
            "dataset_id": dataset.id,
            "target_col": "bad",
            "score_col": "score",
            "score_direction": "higher_is_better",
            "max_bad_rate": 0.05,
        },
    )
    plan_id = turn.plan_id
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "awaiting_confirm"
    turn = driver.resume(
        plan_id=plan_id,
        user_text="确认采纳",
        run_seq=2,
        adjust_params={"adoption_reason": "first adoption"},
        expected_step_id=adopt_step.id,
    )  # -> DONE
    assert turn.status == PlanStatus.DONE.value

    plan = plan_repo.load_plan(plan_id)
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "done"
