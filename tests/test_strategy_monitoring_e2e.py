"""S5 end-to-end: adopt a strategy -> run the strategy_monitoring template through
the REAL PlanDriver on a drift-injected fresh sample -> pause at the red-light
alarm gate -> reply 「起新版本」 -> monitoring report lands on disk with a
next_action pointing at strategy_development.

The strategy here is pure-rule (no model), so monitoring reports only the
strategy-facing approval / approved-bad-rate drift; the fresh sample is engineered
so the approved bad rate blows well past the +-10pp red band, forcing a red
verdict at the alarm gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, PlanRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.orchestrator.contracts import PlanStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.templates import load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.strategy import StrategyRepository
from marvis.settings import build_settings

from marvis.agent.plan_driver import PlanDriver


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _monitoring_driver(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    tool_registry = ToolRegistry(plugin_registry)
    runner = ToolRunner(
        tool_registry,
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    plan_repo = PlanRepository(settings.db_path)
    executor = PlanExecutor(plan_repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(plan_repo))
    planner = Planner(tool_registry, lambda: FakeLLM(), PlanValidator(tool_registry))
    driver = PlanDriver(plan_repo, executor, planner=planner, validator=PlanValidator(tool_registry))
    load_builtin_templates()
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="S5 端到端策略监控",
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
    return driver, runner, registry, plan_repo, settings, task


def _register(registry, tmp_path, frame, name, task_id):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def _adopt_strategy(runner, registry, tmp_path, task):
    # Baseline: rule `score < 500` -> approval 0.80, approved bad rate 1/16 = 0.0625.
    scores = list(range(100, 2100, 100))
    bad = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    baseline = _register(registry, tmp_path, pd.DataFrame({"score": scores, "bad": bad}), "baseline", task.id)
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval",
         "rules": [{"condition": "score < 500", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id=task.id,
    )
    assert built.ok, built.error
    strategy_id = built.output["strategy_id"]
    bt = runner.invoke(
        ToolRef("strategy", "backtest_strategy"),
        {"dataset_id": baseline.id, "strategy_id": strategy_id, "target_col": "bad"},
        task_id=task.id,
    )
    assert bt.ok, bt.error
    adopted = runner.invoke(
        ToolRef("strategy", "adopt_strategy"),
        {"strategy_id": strategy_id, "backtest_id": bt.output["backtest_id"], "adoption_reason": "committee"},
        task_id=task.id,
    )
    assert adopted.ok, adopted.error
    return strategy_id


def _awaiting_step(plan_repo, plan_id):
    plan = plan_repo.load_plan(plan_id)
    from marvis.orchestrator.contracts import StepStatus
    for step in sorted(plan.steps, key=lambda s: (s.index, s.id)):
        if step.status == StepStatus.AWAITING_CONFIRM:
            return step
    return None


@pytest.mark.slow
def test_strategy_monitoring_e2e_red_then_new_version_then_report(tmp_path):
    driver, runner, registry, plan_repo, settings, task = _monitoring_driver(tmp_path)
    strategy_id = _adopt_strategy(runner, registry, tmp_path, task)

    # Drift-injected fresh sample: 100 rows, 30 rejected (score<500), 70 approved
    # with 25 bad -> approved bad rate 25/70=0.357, drift +0.294 (>0.10) -> RED.
    fresh_rows = []
    for _ in range(30):
        fresh_rows.append({"score": 100, "bad": 1})
    for _ in range(45):
        fresh_rows.append({"score": 900, "bad": 0})
    for _ in range(25):
        fresh_rows.append({"score": 900, "bad": 1})
    fresh = _register(registry, tmp_path, pd.DataFrame(fresh_rows), "fresh_drift", task.id)

    turn = driver.start(
        task_id=task.id,
        template_id="strategy_monitoring",
        slots={"strategy_id": strategy_id, "dataset_id": fresh.id, "target_col": "bad"},
    )
    # Confirm the plan-overview 开始 gate to run the monitoring step.
    turn = driver.resume(plan_id=turn.plan_id, user_text="开始")

    # The monitoring step ran; the plan paused at the report step's alarm gate,
    # which renders its run_strategy_monitoring dependency's output (red verdict +
    # red-light checklist).
    gate = _awaiting_step(plan_repo, turn.plan_id)
    assert gate is not None
    assert gate.tool_ref.tool == "render_monitoring_report"
    monitor_step = next(s for s in plan_repo.load_plan(turn.plan_id).steps
                        if s.tool_ref.tool == "run_strategy_monitoring")
    monitor_output = plan_repo.load_step_output(monitor_step.id)
    assert monitor_output["overall_level"] == "red"
    gate_text = "\n".join(m.content for m in turn.messages)
    assert "起新版本" in gate_text  # red-light checklist injected into the gate copy

    # Reply 「起新版本」 -> disposition recorded on the report step, gate confirmed,
    # report step runs.
    turn = driver.resume(plan_id=turn.plan_id, user_text="起新版本")

    plan = plan_repo.load_plan(turn.plan_id)
    assert plan.status == PlanStatus.DONE
    report_step = next(s for s in plan.steps if s.tool_ref.tool == "render_monitoring_report")
    report_output = plan_repo.load_step_output(report_step.id)
    report_path = Path(report_output["report_path"])
    assert report_path.exists()
    assert "策略监控报告" in report_path.read_text(encoding="utf-8")
    # next_action surfaced, pointing at the development template for a new version.
    assert report_output["next_action"]["template_id"] == "strategy_development"
    assert report_output["next_action"]["parent_strategy_id"] == strategy_id

    # A monitoring_report_md artifact is registered.
    kinds = [a["kind"] for a in StrategyRepository(settings.db_path).list_strategy_artifacts(strategy_id)]
    assert "monitoring_report_md" in kinds
