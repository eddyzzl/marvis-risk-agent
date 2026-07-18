"""S5 end-to-end: adopt a strategy -> run the strategy_monitoring template through
the REAL PlanDriver on a drift-injected fresh sample -> pause at the governed
red-light disposition gate -> reply 「起新版本」 -> create a real child task,
draft strategy, dataset reference and monitoring report.

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
    plan_repo = PlanRepository(settings.db_path)
    governance_repo = GovernanceRepository(settings.db_path)
    principal = governance_repo.create_local_principal(
        display_name="策略监控 E2E 操作员"
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
    return driver, registry, plan_repo, settings, task


def _register(registry, tmp_path, frame, name, task_id):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="strategy_sample")


def _adopt_strategy(driver, registry, plan_repo, tmp_path, task):
    # Baseline: rule `score < 500` -> approval 0.80, approved bad rate 1/16 = 0.0625.
    scores = list(range(100, 2100, 100))
    bad = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    baseline = _register(registry, tmp_path, pd.DataFrame({"score": scores, "bad": bad}), "baseline", task.id)
    turn = driver.start(
        task_id=task.id,
        template_id="strategy_development",
        slots={
            "dataset_id": baseline.id,
            "target_col": "bad",
            "score_col": "score",
            "score_direction": "higher_is_better",
            "strategy_type": "approval",
            "max_bad_rate": 0.1,
        },
    )
    plan_id = turn.plan_id
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value

    # Force the deterministic development path to reproduce the baseline rule,
    # while still reviewing and authorizing every canonical human/effect gate.
    turn = driver.resume(
        plan_id=plan_id,
        user_text="",
        run_seq=2,
        adjust_params={"band_edges": [0, 500, 2100]},
    )
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    adopt_step = next(step for step in plan.steps if step.tool_ref.tool == "adopt_strategy")
    turn = driver.resume(
        plan_id=plan_id,
        user_text="确认采纳",
        run_seq=3,
        adjust_params={"adoption_reason": "committee approved monitoring baseline"},
        expected_step_id=adopt_step.id,
    )
    assert turn.status == PlanStatus.DONE.value
    adopted = plan_repo.load_step_output(adopt_step.id)
    strategy_id = adopted["strategy_id"]
    strategy = StrategyRepository(plan_repo.db_path).get_strategy(strategy_id)
    assert [rule.condition for rule in strategy.rules] == ["score < 500"]
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
    driver, registry, plan_repo, settings, task = _monitoring_driver(tmp_path)
    strategy_id = _adopt_strategy(driver, registry, plan_repo, tmp_path, task)

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

    # The monitoring step ran; the plan paused at the governed disposition gate,
    # which is bound to immutable run/plan receipts rather than caller metrics.
    gate = _awaiting_step(plan_repo, turn.plan_id)
    assert gate is not None
    assert gate.tool_ref.tool == "apply_monitoring_disposition"
    monitor_step = next(s for s in plan_repo.load_plan(turn.plan_id).steps
                        if s.tool_ref.tool == "run_strategy_monitoring")
    monitor_output = plan_repo.load_step_output(monitor_step.id)
    assert monitor_output["overall_level"] == "red"
    gate_text = "\n".join(m.content for m in turn.messages)
    assert "起新版本" in gate_text  # red-light checklist injected into the gate copy

    # Reply 「起新版本」 -> the disposition tool creates real child records and the
    # final report renders those execution receipts.
    turn = driver.resume(plan_id=turn.plan_id, user_text="起新版本")

    plan = plan_repo.load_plan(turn.plan_id)
    assert plan.status == PlanStatus.DONE, {
        "turn_status": turn.status,
        "messages": [message.content for message in turn.messages],
        "awaiting_gate": (
            None
            if _awaiting_step(plan_repo, turn.plan_id) is None
            else {
                "tool": _awaiting_step(plan_repo, turn.plan_id).tool_ref.tool,
                "inputs": _awaiting_step(plan_repo, turn.plan_id).inputs,
            }
        ),
    }
    disposition_step = next(
        s for s in plan.steps if s.tool_ref.tool == "apply_monitoring_disposition"
    )
    disposition_output = plan_repo.load_step_output(disposition_step.id)
    assert disposition_output["disposition"] == "new_version"
    assert disposition_output["status"] == "new_version_created"
    assert disposition_output["source_monitoring_run_id"] == monitor_output["monitoring_run_id"]

    new_task_id = disposition_output["new_task_id"]
    new_strategy_id = disposition_output["new_strategy_id"]
    new_dataset_id = disposition_output["new_dataset_id"]
    assert new_task_id and new_strategy_id and new_dataset_id

    child_task = TaskRepository(settings.db_path).get_task(new_task_id)
    assert child_task.task_type == "strategy"
    assert child_task.strategy_input is not None
    assert child_task.strategy_input.baseline_strategy_id == strategy_id
    child_meta = StrategyRepository(settings.db_path).get_strategy_meta(new_strategy_id)
    assert child_meta["task_id"] == new_task_id
    assert child_meta["status"] == "draft"
    assert child_meta["parent_strategy_id"] == strategy_id
    child_dataset = DatasetRepository(settings.db_path).get_dataset(new_dataset_id)
    assert child_dataset is not None
    assert child_dataset.task_id == new_task_id
    assert child_dataset.role == "strategy.new_version_source"

    report_step = next(s for s in plan.steps if s.tool_ref.tool == "render_monitoring_report")
    report_output = plan_repo.load_step_output(report_step.id)
    report_path = Path(report_output["report_path"])
    assert report_path.exists()
    assert "策略监控报告" in report_path.read_text(encoding="utf-8")
    # next_action reports completed execution with actual identifiers, not a
    # suggestion that leaves the user to create another workflow manually.
    assert report_output["next_action"]["kind"] == "completed"
    assert report_output["next_action"]["action"] == "new_version"
    assert report_output["next_action"]["parent_strategy_id"] == strategy_id
    assert report_output["next_action"]["new_task_id"] == new_task_id
    assert report_output["next_action"]["new_strategy_id"] == new_strategy_id
    assert report_output["next_action"]["new_dataset_id"] == new_dataset_id
    assert "template_id" not in report_output["next_action"]

    # A monitoring_report_md artifact is registered.
    kinds = [a["kind"] for a in StrategyRepository(settings.db_path).list_strategy_artifacts(strategy_id)]
    assert "monitoring_report_md" in kinds
