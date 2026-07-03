"""End-to-end S4 journey: mine -> rule-set selection gate (incl. one 「选 …」 text
override round) -> evaluate -> build -> backtest gate -> mandatory adopt gate ->
doc, through the REAL PlanDriver + PlanExecutor + ToolRunner against the strategy
pack's real tools -- the spec's "从『给我挖拒绝规则』到『规则集采纳+文档导出』全程可跑"
acceptance bar (agent/manual dual mode).

Gate sequence (traced with a real driver run): 挖掘规则 (no gate) runs straight
into 规则集确认 (needs_confirmation) on the 开始 resume; the gate message renders
挖掘规则's output. A 「选 1,3」 reply is parsed to a selection and pushed through
apply_adjust (the band_edges precedent), re-arming the gate; confirming reruns
select_rule_set with the override, then 评估规则集 (decision_point, no confirm) +
构造策略 (no gate) run through to 回测策略's gate. Confirming 回测策略 runs it through
to the mandatory 采纳策略 gate; confirming that runs 采纳策略 + 策略文档 to DONE.
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
from marvis.db import (
    DatasetRepository,
    PluginRepository,
    PlanRepository,
    TaskRepository,
    init_db,
)
from marvis.domain import TaskCreate
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
    runner = ToolRunner(
        tool_registry, plugin_repo, python_executable=sys.executable,
        datasets_root=settings.datasets_dir, workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    plan_repo = PlanRepository(settings.db_path)
    executor = PlanExecutor(
        plan_repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(plan_repo)
    )
    planner = Planner(tool_registry, lambda: FakeLLM(), PlanValidator(tool_registry))
    driver = PlanDriver(plan_repo, executor, planner=planner, validator=PlanValidator(tool_registry))
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


@pytest.mark.slow
def test_rule_strategy_full_journey_with_text_selection_override(tmp_path):
    driver, registry, plan_repo, settings, task = _driver(tmp_path)
    dataset = _register(registry, tmp_path, task.id)

    turn = driver.start(
        task_id=task.id, template_id="rule_strategy",
        slots={
            "dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1", "f2"],
            "min_support": 0.05, "min_lift": 1.2, "adoption_reason": "committee approved",
        },
    )
    assert turn.status == PlanStatus.VALIDATED.value
    plan_id = turn.plan_id

    # 开始 -> runs 挖掘规则 straight into 规则集确认 gate; message renders mine output.
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "规则挖掘完成" in gate.content
    plan = plan_repo.load_plan(plan_id)
    mine_step = next(s for s in plan.steps if s.title == "挖掘规则")
    assert mine_step.status.value == "done"
    mine_out = plan_repo.load_step_output(mine_step.id)
    n_candidates = len(mine_out["candidate_rules"])
    assert n_candidates >= 2
    select_step = next(s for s in plan.steps if s.title == "规则集确认")
    assert select_step.status.value == "awaiting_confirm"

    # Text selection override: 「选 1」 keeps only the first candidate. Parsed to a
    # selection list and pushed through apply_adjust (band_edges precedent); the
    # gate re-arms (select_rule_set IS the reviewed step, so needs_confirmation
    # pauses again before it reruns with the override).
    turn = driver.resume(plan_id=plan_id, user_text="选 1", run_seq=2)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    select_step = next(s for s in plan.steps if s.title == "规则集确认")
    assert select_step.status.value == "awaiting_confirm"
    assert select_step.inputs["selection"] == [1]
    assert select_step.output_ref is None  # not yet recomputed with the override

    # Confirm -> reruns select_rule_set with selection=[1] (1 rule), then 评估规则集
    # (decision_point, no confirm) + 构造策略 (no gate) through to the 回测策略 gate;
    # message renders 构造策略's output.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=3)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "策略候选已生成" in gate.content
    plan = plan_repo.load_plan(plan_id)
    select_step = next(s for s in plan.steps if s.title == "规则集确认")
    assert select_step.status.value == "done"
    selected_out = plan_repo.load_step_output(select_step.id)
    assert selected_out["selected_count"] == 1
    evaluate_step = next(s for s in plan.steps if s.title == "评估规则集")
    assert evaluate_step.status.value == "done"
    evaluate_out = plan_repo.load_step_output(evaluate_step.id)
    assert len(evaluate_out["waterfall"]) == 1
    build_step = next(s for s in plan.steps if s.title == "构造策略")
    assert build_step.status.value == "done"
    backtest_step = next(s for s in plan.steps if s.title == "回测策略")
    assert backtest_step.status.value == "awaiting_confirm"

    # Confirm 回测策略 -> runs it through to the mandatory 采纳策略 gate.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=4)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "策略回测完成" in gate.content
    plan = plan_repo.load_plan(plan_id)
    backtest_out = plan_repo.load_step_output(backtest_step.id)
    assert 0.0 <= backtest_out["approval_rate"] <= 1.0
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "awaiting_confirm"  # mandatory gate

    # Confirm adoption -> runs 采纳策略 + 策略文档 -> DONE.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=5)
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
def test_rule_strategy_keep_all_via_confirm(tmp_path):
    """A plain 「确认」 at the rule-set gate (no selection) keeps every candidate
    (selection default None -> select_rule_set keeps all), so the journey runs
    the full mined set without a text override."""
    driver, registry, plan_repo, settings, task = _driver(tmp_path)
    dataset = _register(registry, tmp_path, task.id)
    turn = driver.start(
        task_id=task.id, template_id="rule_strategy",
        slots={
            "dataset_id": dataset.id, "target_col": "bad", "feature_cols": ["f1", "f2"],
            "min_support": 0.05, "min_lift": 1.2, "adoption_reason": "keep all",
        },
    )
    plan_id = turn.plan_id
    driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)  # -> 规则集确认 gate
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=2)  # keep-all -> 回测 gate
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    select_step = next(s for s in plan.steps if s.title == "规则集确认")
    mine_step = next(s for s in plan.steps if s.title == "挖掘规则")
    select_out = plan_repo.load_step_output(select_step.id)
    mine_out = plan_repo.load_step_output(mine_step.id)
    assert select_out["selected_count"] == len(mine_out["candidate_rules"])
