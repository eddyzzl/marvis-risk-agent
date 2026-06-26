"""Generic PlanDriver loop: run -> pause at gate -> compose message from the
just-computed prior step -> confirm -> resume -> done. Driven against a REAL
PlanExecutor + PlanRepository with a fake tool runner returning canned outputs,
so the gate detection / dependency-output rendering / append-only messaging are
exercised deterministically without running real modeling tools.
"""

from __future__ import annotations

from dataclasses import replace as _dataclass_replace
from types import SimpleNamespace

from marvis.agent.plan_driver import PlanDriver, is_confirm, render_tool_output
from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.plugins.manifest import ToolRef
from marvis.plugins.runner import ToolResult


class FakeLLM:
    def complete(self, **kwargs):
        return '{"summary": "done", "open_items": [], "goal_doubt": false, "goal_met": true}'


class FakeTools:
    def resolve(self, ref):
        return SimpleNamespace(failure_policy="fail")


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self._tools = FakeTools()

    def invoke(self, ref, inputs, *, task_id):
        self.calls.append((ref.tool, inputs))
        return ToolResult(ok=True, output=self.outputs.pop(0), error=None, error_kind=None, duration_ms=1)


class FakeHooks:
    def dispatch(self, event, payload, *, task_id):
        return []


def _step(step_id, *, index, tool, depends_on=None, needs_confirmation=False, phase=None):
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=index,
        title=step_id,
        tool_ref=ToolRef("modeling", tool),
        inputs={},
        depends_on=depends_on or [],
        post_checks=[],  # empty => deterministic check trivially passes on canned output
        needs_confirmation=needs_confirmation,
        phase=phase,
    )


def _gated_modeling_plan() -> Plan:
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="modeling",
        source="template",
        template_id="modeling",
        autonomy_level=1,
        status=PlanStatus.VALIDATED,
        steps=[
            _step("screen", index=0, tool="screen_features", phase="特征"),
            _step("tune", index=1, tool="tune_hyperparameters", depends_on=["screen"], needs_confirmation=True, phase="建模"),
            _step("train", index=2, tool="train_model", depends_on=["tune"], phase="建模"),
        ],
    )


def _driver(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())
    runner = FakeRunner([
        {"selected": ["sig1", "sig2"], "leakage": [["leak_col", 0.55, 1]], "suspected": [["score_x", 0.3, 1]], "n_screened": 9,
         "ranked": [], "unusable": [], "scores": {}},
        {"best_params": {"num_leaves": 31}, "best_metrics": {"test_ks": 0.41}, "n_trials": 8},
        {"experiment_id": "exp-1", "artifact_id": "art-1", "metrics": {"oot_ks": 0.39, "oot_auc": 0.72}, "feature_importance": [["sig1", 120.0]]},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    return PlanDriver(repo, executor), repo


def test_driver_runs_to_first_gate_and_shows_prior_step(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")  # VALIDATED -> CONFIRMED (start() does this; here we drive the loop directly)

    turn = driver._run_and_handle("plan-1", run_seq=0)

    # paused before the gated 'tune' step, having just run 'screen'
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(turn.messages) == 1
    msg = turn.messages[0]
    assert msg.stage == "gate"
    # message is composed from the screen (dependency) output, not the gate step
    assert "特征筛选完成" in msg.content
    assert "保留 **2** 个" in msg.content
    assert "泄漏" in msg.content
    assert msg.metadata["step_id"] == "tune"
    assert msg.metadata["plan_id"] == "plan-1"
    assert msg.metadata["run_seq"] == 0
    assert any(t["title"].startswith("入选特征") for t in msg.metadata["tables"])
    # only screen has executed so far
    loaded = repo.load_plan("plan-1")
    statuses = {s.id: s.status for s in loaded.steps}
    assert statuses["screen"] == StepStatus.DONE
    assert statuses["tune"] == StepStatus.AWAITING_CONFIRM


def test_driver_resume_confirm_runs_to_done(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # pause at tune gate

    turn = driver.resume(plan_id="plan-1", user_text="确认", run_seq=1)

    assert turn.status == PlanStatus.DONE.value
    done = turn.messages[-1]
    assert done.stage == "done"
    assert "完成" in done.content
    loaded = repo.load_plan("plan-1")
    assert loaded.status == PlanStatus.DONE
    assert all(s.status == StepStatus.DONE for s in loaded.steps)


def test_driver_resume_non_confirm_holds_at_gate(tmp_path):
    driver, repo = _driver(tmp_path)
    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)

    turn = driver.resume(plan_id="plan-1", user_text="把阈值调高一点", run_seq=1)

    # stays awaiting; tune not yet executed (adjust is a later slice)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert "确认" in turn.messages[0].content
    loaded = repo.load_plan("plan-1")
    assert {s.id: s.status for s in loaded.steps}["tune"] == StepStatus.AWAITING_CONFIRM


def test_driver_plan_overview_gate_waits_for_kaishi(tmp_path):
    """A freshly-built (VALIDATED) plan does not run until 开始 is confirmed
    (spec §9 #2 plan-level overview gate)."""
    driver, repo = _driver(tmp_path)
    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED  # built, not started

    # non-confirm at the overview gate → the plan does NOT start
    turn = driver.resume(plan_id="plan-1", user_text="先看看", run_seq=0)
    assert turn.status == PlanStatus.VALIDATED.value
    assert repo.load_plan("plan-1").status == PlanStatus.VALIDATED
    assert {s.id: s.status for s in repo.load_plan("plan-1").steps}["screen"] != StepStatus.DONE

    # 开始 → confirm_plan + run to the first gate
    turn = driver.resume(plan_id="plan-1", user_text="开始", run_seq=1)
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert "特征筛选完成" in turn.messages[0].content
    assert {s.id: s.status for s in repo.load_plan("plan-1").steps}["screen"] == StepStatus.DONE


class FakeRouterLLM:
    """Returns a fixed instruction-route JSON (agent-mode gate instruction)."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


def test_driver_adjust_reruns_analysis_step_with_new_params(tmp_path):
    """An agent-mode 'adjust' instruction at a gate re-runs the gate's analysis
    dependency with overridden parameters and re-pauses at the gate (spec §3 调整)."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    plan = _gated_modeling_plan()
    # leakage_ks is a real declared input of screen_features; the template sets it, so it
    # is a legitimate override target (adjusting an UNDECLARED key would fail validation).
    plan.steps[0].inputs = {"leakage_ks": 0.4}
    repo.create_plan(plan)
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
        # re-run after the adjust keeps 3 features
        {"selected": ["sig1", "sig2", "sig3"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))
    llm = FakeRouterLLM('{"action":"adjust","params":{"leakage_ks":0.3},"constraint":"","reason":"放宽阈值重算"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # screen runs once, pause at the tune gate
    assert len(runner.calls) == 1

    turn = driver.resume(plan_id="plan-1", user_text="把泄漏阈值放宽到 0.3", run_seq=1)

    # screen re-ran with the overridden input and the plan re-paused at the same gate
    assert turn.status == PlanStatus.AWAITING_CONFIRM.value
    assert len(runner.calls) == 2
    assert runner.calls[1][1].get("leakage_ks") == 0.3  # the declared override reached the tool
    assert "保留 **3** 个" in turn.messages[-1].content  # the recomputed screen output is shown
    assert any("调整参数" in m.content for m in turn.messages)


def test_driver_replan_instruction_routes_to_structural_replan(tmp_path):
    """An agent-mode 'replan' instruction routes to the structural-replan path (no longer
    the canned stub); with no planner wired it fails gracefully with a recoverable hint."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())
    runner = FakeRunner([
        {"selected": ["sig1"], "leakage": [], "suspected": [], "n_screened": 9, "ranked": [], "unusable": [], "scores": {}},
    ])
    executor = PlanExecutor(repo, runner, Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo))  # planner=None
    llm = FakeRouterLLM('{"action":"replan","params":{},"constraint":"去掉调参步骤","reason":"改流程"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    repo.confirm_plan("plan-1")
    driver._run_and_handle("plan-1", run_seq=0)  # pause at the tune gate

    turn = driver.resume(plan_id="plan-1", user_text="把调参那步去掉", run_seq=1)

    assert turn.status == PlanStatus.AWAITING_CONFIRM.value  # holds at the gate
    assert "重规划未成功" in turn.messages[-1].content
    assert "仍在接入" not in turn.messages[-1].content  # not the retired stub


class _FakeReplanPlanner:
    """Returns a revised plan (one fresh step) for the structural-replan success path."""

    def __init__(self):
        self.last_instruction = None

    def replan(self, plan, *, completed_summaries, observation, reason, tier, instruction=None):
        self.last_instruction = instruction
        new_step = PlanStep(
            id="new-a", plan_id=plan.id, index=0, title="新步骤A",
            tool_ref=ToolRef("modeling", "screen_features"), inputs={}, depends_on=[],
            post_checks=[], needs_confirmation=False, phase="特征",
        )
        return _dataclass_replace(plan, steps=[new_step], replan_count=plan.replan_count + 1)


def test_driver_replan_success_at_overview_shows_new_plan_and_stays_validated(tmp_path):
    """A 'replan' instruction at the VALIDATED overview regenerates the plan, shows the new
    overview, and stays VALIDATED — nothing runs until 开始 (spec §3 replan + §9#2 gate)."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(_gated_modeling_plan())  # VALIDATED, not started
    planner = _FakeReplanPlanner()
    executor = PlanExecutor(
        repo, FakeRunner([]), Reviewer(lambda: FakeLLM()), None, FakeHooks(), HarnessState(repo), planner=planner
    )
    llm = FakeRouterLLM('{"action":"replan","params":{},"constraint":"只跑一步A","reason":"改流程"}')
    driver = PlanDriver(repo, executor, llm_client=llm)

    turn = driver.resume(plan_id="plan-1", user_text="把流程改成只跑一步", run_seq=0)

    assert turn.status == PlanStatus.VALIDATED.value  # not started — still awaits 开始
    assert planner.last_instruction == "把流程改成只跑一步"
    loaded = repo.load_plan("plan-1")
    assert [step.id for step in loaded.steps] == ["new-a"]  # remaining steps replaced
    assert loaded.status == PlanStatus.VALIDATED
    assert any("重规划" in message.content for message in turn.messages)


def test_is_confirm_matches_common_phrasings():
    assert is_confirm("确认")
    assert is_confirm("ok 继续")
    assert not is_confirm("把 age 去掉")


def test_render_registry_has_modeling_renderers_and_generic_fallback():
    text, tables = render_tool_output("screen_features", {"selected": ["a"], "leakage": [], "suspected": [], "n_screened": 3})
    assert "特征筛选完成" in text
    text2, _ = render_tool_output("some_unknown_tool", {"status": "ok", "rows": 10})
    assert "已完成" in text2


def test_render_tune_leaderboard_includes_auc_and_head_lift_columns():
    """The tune leaderboard surfaces each trial's test AUC + 头部lift10% (spec columns),
    not just KS."""
    _text, tables = render_tool_output("tune_hyperparameters", {
        "n_trials": 2, "best_params": {}, "best_metrics": {"test_ks": 0.4, "test_auc": 0.72},
        "trials": [
            {"train_ks": 0.45, "test_ks": 0.40, "oot_ks": 0.38, "score": 0.38, "test_auc": 0.72, "lift_head_10": 3.1},
            {"train_ks": 0.50, "test_ks": 0.35, "oot_ks": 0.33, "score": 0.30, "test_auc": 0.69, "lift_head_10": 2.8},
        ],
    })
    board = next(table for table in tables if table["title"].startswith("trials 排行"))
    assert "AUC" in board["columns"] and "头部lift10%" in board["columns"]
    assert any("0.72" in str(cell) for cell in board["rows"][0])  # AUC reached the top row
