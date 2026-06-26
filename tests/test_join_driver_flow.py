"""End-to-end JOIN through the generic PlanDriver on REAL data_ops tools.

This is the gold-standard proof for the data-join entry: a synthetic anchor +
feature dataset are driven through the `data_join` template by the real
PlanDriver + PlanExecutor + ToolRunner (subprocess-isolated tools). It asserts
the driver pauses at the C2 forced-confirm gate (showing propose_join's
diagnostics), and that confirming runs confirm_join -> execute_join to produce a
1:1-anchored result. No LLM is required (manual-mode-compatible path).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

from marvis.agent.plan_driver import PlanDriver
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, PlanRepository, init_db
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
from marvis.settings import build_settings


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _join_driver(tmp_path):
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
    return driver, registry, plan_repo


def _register_csv(registry, tmp_path, name, frame, *, role):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-1", path, role=role)


def test_join_flow_pauses_at_forced_confirm_then_executes_1to1(tmp_path):
    driver, registry, plan_repo = _join_driver(tmp_path)
    phones = ["13800138000", "13900139000", "13700137000"]
    anchor = _register_csv(registry, tmp_path, "anchor", pd.DataFrame({"mobile": phones}), role="sample")
    feature = _register_csv(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [hashlib.md5(value.encode()).hexdigest() for value in phones],
            "balance": [10, 20, 30],
        }),
        role="feature",
    )

    turn = driver.start(task_id="task-1", template_id="data_join",
                        slots={"anchor_id": anchor.id, "feature_ids": [feature.id]})

    # start shows the plan overview and pauses at the plan-level 开始 gate (nothing run)
    assert turn.status == PlanStatus.VALIDATED.value
    assert turn.messages[0].stage == "plan_overview"
    plan_id = turn.plan_id

    # 开始 → run to the C2 forced-confirm gate (confirm_join), showing propose diagnostics
    turn = driver.resume(plan_id=plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    gate_msg = turn.messages[-1]
    assert gate_msg.stage == "gate"
    assert "拼接诊断完成" in gate_msg.content
    assert any(t["title"].startswith("拼接诊断") for t in gate_msg.metadata.get("tables", []))

    turn2 = driver.resume(plan_id=plan_id, user_text="确认", run_seq=2)

    # confirm -> confirm_join -> execute_join -> done, anchor preserved 1:1
    assert turn2.status == PlanStatus.DONE.value
    done = turn2.messages[-1]
    assert "拼接执行完成" in done.content
    assert "1:1 保持" in done.content


def test_join_flow_blocks_execute_until_confirmed(tmp_path):
    """The plan must never reach execute_join without passing the gate: before
    confirming, only propose_join has run."""
    driver, registry, plan_repo = _join_driver(tmp_path)
    phones = ["13800138000", "13900139000"]
    anchor = _register_csv(registry, tmp_path, "anchor", pd.DataFrame({"mobile": phones}), role="sample")
    feature = _register_csv(
        registry,
        tmp_path,
        "feature",
        pd.DataFrame({
            "phone_md5": [hashlib.md5(v.encode()).hexdigest() for v in phones],
            "balance": [10, 20],
        }),
        role="feature",
    )

    turn = driver.start(task_id="task-1", template_id="data_join",
                        slots={"anchor_id": anchor.id, "feature_ids": [feature.id]})
    # plan-overview gate: nothing has run before 开始
    assert turn.status == PlanStatus.VALIDATED.value
    before = plan_repo.load_plan(turn.plan_id)
    assert not {s.title for s in before.steps if s.status.value == "done"}

    # 开始 → run to the forced-confirm gate; only propose_join has run
    turn = driver.resume(plan_id=turn.plan_id, user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    plan = plan_repo.load_plan(turn.plan_id)
    titles_done = {s.title for s in plan.steps if s.status.value == "done"}
    assert "拼接诊断" in titles_done
    assert "执行拼接" not in titles_done  # execute has NOT run before confirmation
