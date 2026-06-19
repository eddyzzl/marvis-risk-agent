import json
from types import SimpleNamespace

from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import (
    AgentStatus,
    Plan,
    PlanStatus,
    PlanStep,
    PostCheck,
    StepStatus,
    SubAgent,
)
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.plugins.manifest import ToolRef
from marvis.plugins.runner import ToolResult


class FakeLLM:
    def __init__(self, response=None):
        self.response = response or json.dumps({"summary": "done"})
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeTools:
    def __init__(self, policies=None):
        self.policies = policies or {}

    def resolve(self, ref):
        return SimpleNamespace(failure_policy=self.policies.get(ref.tool, "fail"))


class FakeRunner:
    def __init__(self, outputs=None, policies=None):
        self.outputs = list(outputs or [])
        self.calls = []
        self._tools = FakeTools(policies)

    def invoke(self, ref, inputs, *, task_id):
        self.calls.append((ref, inputs, task_id))
        return self.outputs.pop(0)


class FakeHooks:
    def __init__(self):
        self.calls = []

    def dispatch(self, event, payload, *, task_id):
        self.calls.append((event, payload, task_id))
        return []


class FakeSubAgents:
    def __init__(self, result):
        self.result = result
        self.spawn_calls = []
        self.run_calls = []

    def spawn(self, step, *, parent_task_id):
        self.spawn_calls.append((step.id, parent_task_id))
        return SubAgent(
            id="sub-1",
            parent_task_id=parent_task_id,
            parent_step_id=step.id,
            scope=step.sub_agent_scope,
            granted_tools=step.granted_tools,
            context_budget=1024,
            status=AgentStatus.SPAWNED,
        )

    def run(self, sub, *, goal_inputs):
        self.run_calls.append((sub.id, goal_inputs))
        return self.result


class FakeAdaptivePlanner:
    def __init__(self, *, replanned_steps=None, explore_results=None):
        self.replanned_steps = replanned_steps or []
        self.explore_results = list(explore_results or [])
        self.replan_calls = []
        self.explore_calls = []

    def replan(self, plan, *, completed_summaries, observation, reason, tier):
        self.replan_calls.append((plan.id, completed_summaries, observation, reason, tier.name))
        steps = self.replanned_steps(plan) if callable(self.replanned_steps) else self.replanned_steps
        return _plan_like(plan, steps, replan_count=plan.replan_count + 1, tier=tier.name)

    def next_explore_segment(self, plan, *, completed_summaries, tier):
        self.explore_calls.append((plan.id, completed_summaries, tier.name))
        return self.explore_results.pop(0)


def _ok(output):
    return ToolResult(ok=True, output=output, error=None, error_kind=None, duration_ms=1)


def _fail(message="boom"):
    return ToolResult(
        ok=False,
        output=None,
        error=message,
        error_kind="execution",
        duration_ms=1,
    )


def _step(
    step_id,
    *,
    index=0,
    tool="echo",
    inputs=None,
    depends_on=None,
    post_checks=None,
    needs_confirmation=False,
    decision_point=False,
    sub_agent_scope=None,
    granted_tools=None,
    status=StepStatus.PENDING,
):
    return PlanStep(
        id=step_id,
        plan_id="plan-1",
        index=index,
        title=step_id,
        tool_ref=ToolRef("_sample", tool),
        inputs=inputs or {},
        depends_on=depends_on or [],
        post_checks=post_checks or [],
        needs_confirmation=needs_confirmation,
        decision_point=decision_point,
        sub_agent_scope=sub_agent_scope,
        granted_tools=granted_tools or [],
        status=status,
    )


def _plan(*steps, status=PlanStatus.CONFIRMED):
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="finish",
        source="template",
        template_id="test",
        steps=list(steps),
        autonomy_level=1,
        status=status,
    )


def _repo(tmp_path, plan):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PlanRepository(db_path)
    repo.create_plan(plan)
    return repo


def _executor(repo, runner, reviewer=None, subagents=None, hooks=None):
    return PlanExecutor(
        repo,
        runner,
        reviewer or Reviewer(lambda: FakeLLM()),
        subagents,
        hooks or FakeHooks(),
        HarnessState(repo),
    )


def _adaptive_executor(repo, runner, planner, reviewer=None, hooks=None):
    return PlanExecutor(
        repo,
        runner,
        reviewer or Reviewer(lambda: FakeLLM()),
        None,
        hooks or FakeHooks(),
        HarnessState(repo),
        planner=planner,
    )


def _plan_like(plan, steps, *, replan_count=None, tier=None):
    return Plan(
        id=plan.id,
        task_id=plan.task_id,
        goal=plan.goal,
        source=plan.source,
        template_id=plan.template_id,
        steps=list(steps),
        autonomy_level=plan.autonomy_level,
        status=plan.status,
        novel_mode=plan.novel_mode,
        tier=tier or plan.tier,
        replan_count=plan.replan_count if replan_count is None else replan_count,
    )


def test_plan_executor_runs_linear_plan_resolves_refs_and_finalizes(tmp_path):
    plan = _plan(
        _step("step-1", inputs={"message": "hi"}),
        _step(
            "step-2",
            index=1,
            inputs={"message": "$ref:step-1.output.echoed"},
            depends_on=["step-1"],
        ),
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "hi"}), _ok({"echoed": "again"})])
    hooks = FakeHooks()

    result = _executor(repo, runner, hooks=hooks).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert loaded.status == PlanStatus.DONE
    assert [step.status for step in loaded.steps] == [StepStatus.DONE, StepStatus.DONE]
    assert runner.calls[1][1] == {"message": "hi"}
    assert repo.load_plan_summary(result.summary_ref)["goal_met"] is True
    assert [call[0] for call in hooks.calls] == [
        "step.completed",
        "step.completed",
        "workflow.completed",
    ]


def test_plan_executor_keeps_goal_doubt_in_review(tmp_path):
    plan = _plan(_step("step-1"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "hi"})])
    hooks = FakeHooks()
    llm = FakeLLM(
        json.dumps(
            {
                "summary": "Needs human review.",
                "open_items": [],
                "goal_doubt": True,
            }
        )
    )

    result = _executor(
        repo,
        runner,
        reviewer=Reviewer(lambda: llm),
        hooks=hooks,
    ).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.REVIEW
    assert loaded.status == PlanStatus.REVIEW
    assert result.summary_ref is not None
    assert repo.load_plan_summary(result.summary_ref)["goal_doubt"] is True
    assert [call[0] for call in hooks.calls] == ["step.completed"]


def test_plan_executor_pauses_for_confirmation_and_resumes_from_db(tmp_path):
    repo = _repo(tmp_path, _plan(_step("step-1", needs_confirmation=True)))
    runner = FakeRunner([_ok({"echoed": "hi"})])

    first = _executor(repo, runner).run("plan-1")
    assert first.status == PlanStatus.AWAITING_CONFIRM
    assert runner.calls == []

    repo.confirm_step("step-1")
    second = _executor(repo, runner).run("plan-1")

    assert second.status == PlanStatus.DONE
    assert len(runner.calls) == 1
    assert repo.load_plan("plan-1").steps[0].status == StepStatus.DONE


def test_plan_executor_applies_retry_skip_and_fail_policies(tmp_path):
    retry_repo = _repo(tmp_path / "retry", _plan(_step("step-1", tool="retry_tool")))
    retry_runner = FakeRunner(
        [_fail("temporary"), _ok({"echoed": "ok"})],
        policies={"retry_tool": "retry"},
    )
    retry_result = _executor(retry_repo, retry_runner).run("plan-1")
    assert retry_result.status == PlanStatus.DONE
    assert len(retry_runner.calls) == 2

    skip_repo = _repo(
        tmp_path / "skip",
        _plan(
            _step("step-1", tool="skip_tool"),
            _step("step-2", index=1),
        ),
    )
    skip_runner = FakeRunner(
        [_fail("optional failed"), _ok({"echoed": "ok"})],
        policies={"skip_tool": "skip"},
    )
    skip_result = _executor(skip_repo, skip_runner).run("plan-1")
    assert skip_result.status == PlanStatus.DONE
    assert [step.status for step in skip_repo.load_plan("plan-1").steps] == [
        StepStatus.SKIPPED,
        StepStatus.DONE,
    ]

    fail_repo = _repo(tmp_path / "fail", _plan(_step("step-1")))
    fail_result = _executor(fail_repo, FakeRunner([_fail("fatal")])).run("plan-1")
    assert fail_result.status == PlanStatus.FAILED
    assert fail_repo.load_plan("plan-1").steps[0].status == StepStatus.FAILED


def test_plan_executor_blocks_deterministic_postcheck_failure_without_llm_rescue(tmp_path):
    llm = FakeLLM()
    repo = _repo(
        tmp_path,
        _plan(_step("step-1", post_checks=[PostCheck("range", {"field": "ks", "max": 1.0})])),
    )

    result = _executor(
        repo,
        FakeRunner([_ok({"ks": 1.2})]),
        reviewer=Reviewer(lambda: llm),
    ).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert loaded.steps[0].status == StepStatus.FAILED
    assert loaded.steps[0].review_verdicts[0].reviewer == "deterministic"
    assert llm.calls == []


def test_plan_executor_delegates_subagent_steps_and_stores_result_ref(tmp_path):
    step = _step(
        "step-1",
        sub_agent_scope="summarize table",
        granted_tools=[ToolRef("_sample", "echo")],
    )
    repo = _repo(tmp_path, _plan(step))
    subagents = FakeSubAgents(_ok({"result_ref": "artifact:sub-summary"}))
    runner = FakeRunner([])

    result = _executor(repo, runner, subagents=subagents).run("plan-1")

    loaded_step = repo.load_plan("plan-1").steps[0]
    assert result.status == PlanStatus.DONE
    assert runner.calls == []
    assert subagents.spawn_calls == [("step-1", "task-1")]
    assert subagents.run_calls == [("sub-1", {})]
    assert loaded_step.sub_agent_id == "sub-1"
    assert repo.load_step_output("step-1") == {"result_ref": "artifact:sub-summary"}


def test_plan_executor_replans_after_decision_point(tmp_path):
    plan = _plan(
        _step("step-1", decision_point=True),
        _step("step-2", index=1, depends_on=["step-1"]),
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "first"}), _ok({"echoed": "replanned"})])
    hooks = FakeHooks()
    planner = FakeAdaptivePlanner(
        replanned_steps=lambda loaded: [
            loaded.steps[0],
            _step("step-3", index=1, inputs={"message": "$ref:step-1.output.echoed"}, depends_on=["step-1"]),
        ]
    )

    result = _adaptive_executor(repo, runner, planner, hooks=hooks).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert [step.id for step in loaded.steps] == ["step-1", "step-3"]
    assert loaded.replan_count == 1
    assert planner.replan_calls[0][3] == "decision_point"
    assert [call[0] for call in hooks.calls] == [
        "step.completed",
        "plan.replanned",
        "step.completed",
        "workflow.completed",
    ]


def test_plan_executor_replans_execution_failure_and_continues(tmp_path):
    plan = _plan(_step("step-1", tool="fail_tool"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_fail("temporary missing column"), _ok({"echoed": "fixed"})])
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-2", tool="echo", inputs={"message": "fixed"})]
    )

    result = _adaptive_executor(repo, runner, planner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert [step.id for step in loaded.steps] == ["step-2"]
    assert loaded.replan_count == 1
    assert planner.replan_calls[0][3] == "failure"
    assert len(runner.calls) == 2


def test_plan_executor_appends_explore_segment_until_done(tmp_path):
    plan = _plan(_step("step-1"))
    plan.novel_mode = "explore"
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "first"}), _ok({"echoed": "next"})])
    planner = FakeAdaptivePlanner(
        explore_results=[
            ([_step("step-2", index=1, inputs={"message": "$ref:step-1.output.echoed"}, depends_on=["step-1"])], False),
            ([], True),
        ]
    )

    result = _adaptive_executor(repo, runner, planner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert [step.id for step in loaded.steps] == ["step-1", "step-2"]
    assert loaded.novel_mode == "explore"
    assert loaded.replan_count == 1
    assert len(planner.explore_calls) == 2
    assert len(runner.calls) == 2
