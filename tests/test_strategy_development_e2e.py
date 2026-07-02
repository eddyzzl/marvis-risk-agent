"""End-to-end S2 journey: scan -> band-design gate (incl. one manual band_edges
override round) -> backtest gate -> adopt gate (mandatory) -> doc, through the
REAL PlanDriver + PlanExecutor + ToolRunner against the strategy pack's real
tools -- the "agent/manual dual mode runs the full 分数->分段->回测->采纳->导出
journey" acceptance bar from the spec.

Gate sequence verified empirically (traced with a real driver run): 权衡扫描 and
回测策略/对比基线 are decision_point steps but that alone does not pause the
executor loop -- only needs_confirmation does (orchestrator/executor.py's
_next_ready_step / needs_confirmation check). So 权衡扫描 runs straight through
into 设计分数带 (needs_confirmation=True) on the same resume call, and the gate
message shown at that pause is 权衡扫描's rendered output (gate_message renders
the completed *dependency* step's output, not the paused step itself). Likewise
confirming 设计分数带 runs 构造策略 (no gate) through to 回测策略's gate, showing
构造策略's message; confirming 回测策略 runs it plus 对比基线 through to 采纳策略's
mandatory gate, showing both rendered; confirming 采纳策略 runs it plus 策略文档
straight to DONE.
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


def _strategy_driver(tmp_path):
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
    return backtests[-1].approval_rate


@pytest.mark.slow
def test_strategy_development_full_journey_with_manual_band_edges_override(tmp_path):
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
            "adoption_reason": "committee approved for Q3 rollout",
        },
    )
    assert turn.status == PlanStatus.VALIDATED.value
    plan_id = turn.plan_id

    # 开始 -> runs 权衡扫描 (decision_point, no needs_confirmation, so it does not
    # pause on its own) straight through to the 设计分数带 mandatory confirm gate.
    # The gate message renders 权衡扫描's output (its completed dependency).
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "策略权衡视图完成" in gate.content
    plan = plan_repo.load_plan(plan_id)
    assert next(s for s in plan.steps if s.title == "权衡扫描").status.value == "done"
    assert next(s for s in plan.steps if s.title == "设计分数带").status.value == "awaiting_confirm"

    # Manual mode: structured band_edges override via the generic adjust_params
    # gate-recompute channel (apply_adjust: agent/gate_execution_adapter.py).
    # 设计分数带 IS the reviewed computation (unlike confirm_join-style gates that
    # wrap a separate upstream step), so needs_confirmation pauses again *before*
    # rerunning it with the new inputs -- the override lands and re-arms the
    # gate, but the actual recompute happens on the next confirm.
    turn = driver.resume(
        plan_id=plan_id,
        user_text="",
        run_seq=2,
        adjust_params={"band_edges": [100, 400, 1200, 2000]},
    )
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(plan_id)
    bands_step = next(s for s in plan.steps if s.title == "设计分数带")
    assert bands_step.status.value == "awaiting_confirm"
    assert bands_step.inputs["band_edges"] == [100, 400, 1200, 2000]
    assert bands_step.output_ref is None  # not yet recomputed with the override

    # Confirm -> reruns 设计分数带 with the overridden band_edges, then 构造策略
    # (no gate) through to the 回测策略 mandatory gate; message renders 构造策略's
    # output.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=3)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "策略候选已生成" in gate.content
    plan = plan_repo.load_plan(plan_id)
    bands_step = next(s for s in plan.steps if s.title == "设计分数带")
    assert bands_step.status.value == "done"
    reran_output = plan_repo.load_step_output(bands_step.id)
    assert reran_output["band_edges"] == [100.0, 400.0, 1200.0, 2000.0]
    assert next(s for s in plan.steps if s.title == "构造策略").status.value == "done"
    assert next(s for s in plan.steps if s.title == "回测策略").status.value == "awaiting_confirm"

    # Confirm 回测策略 -> runs it plus 对比基线 (decision_point, no baseline slot
    # supplied -> degrades to a no-op instead of failing) through to the
    # mandatory 采纳策略 gate.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=4)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate = turn.messages[-1]
    assert "策略回测完成" in gate.content
    plan = plan_repo.load_plan(plan_id)
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

    # Confirm adoption (the forced gate) -> runs 采纳策略 + 策略文档 -> DONE.
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=5)
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
            "adoption_reason": "first adoption",
        },
    )
    plan_id = turn.plan_id
    driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)  # -> 设计分数带 gate
    driver.resume(plan_id=plan_id, user_text="确认", run_seq=2)  # -> 回测策略 gate
    driver.resume(plan_id=plan_id, user_text="确认", run_seq=3)  # -> 采纳策略 gate
    turn = driver.resume(plan_id=plan_id, user_text="确认", run_seq=4)  # -> DONE
    assert turn.status == PlanStatus.DONE.value

    plan = plan_repo.load_plan(plan_id)
    adopt_step = next(s for s in plan.steps if s.title == "采纳策略")
    assert adopt_step.status.value == "done"
