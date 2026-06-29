import json
from types import SimpleNamespace

from marvis.db import PlanRepository, init_db
from marvis.orchestrator.contracts import (
    AgentStatus,
    LoopEvent,
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
from marvis.plugins.manifest import PluginManifest, ToolRef, ToolSpec
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


class FakeManifestTools:
    def __init__(self):
        self.tool = ToolSpec(
            name="echo",
            summary="Echo",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            determinism="deterministic",
            timeout_seconds=5,
            failure_policy="fail",
            side_effects=(),
            entrypoint="echo",
        )
        self.manifest = PluginManifest(
            name="_sample",
            version="1.2.3",
            display_name="Sample",
            description="",
            module="sample",
            python_requires="",
            tools=(self.tool,),
            builtin=True,
        )

    def resolve(self, ref):
        return self.tool

    def resolve_with_manifest(self, ref):
        return self.manifest, self.tool


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

    def replan(self, plan, *, completed_summaries, observation, reason, tier, instruction=None):
        self.replan_calls.append((plan.id, completed_summaries, observation, reason, tier.name))
        self.last_instruction = instruction
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
    plugin="_sample",
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
        tool_ref=ToolRef(plugin, tool),
        inputs=inputs or {},
        depends_on=depends_on or [],
        post_checks=post_checks or [],
        needs_confirmation=needs_confirmation,
        decision_point=decision_point,
        sub_agent_scope=sub_agent_scope,
        granted_tools=granted_tools or [],
        status=status,
    )


def _plan(*steps, status=PlanStatus.CONFIRMED, success_criteria=None):
    return Plan(
        id="plan-1",
        task_id="task-1",
        goal="finish",
        source="template",
        template_id="test",
        steps=list(steps),
        autonomy_level=1,
        status=status,
        success_criteria=list(success_criteria or []),
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
        success_criteria=[dict(item) for item in plan.success_criteria],
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
    first_runs = repo.list_step_runs("step-1")
    second_runs = repo.list_step_runs("step-2")
    assert [run["status"] for run in first_runs] == ["succeeded"]
    assert [run["status"] for run in second_runs] == ["succeeded"]
    assert second_runs[0]["input"] == {"message": "hi"}
    assert second_runs[0]["output_ref"] == "metrics:step-2:v1"
    evidence = repo.load_step_evidence("step-2")
    assert evidence["tool_name"] == "_sample.echo"
    assert evidence["input_hash"].startswith("sha256:")
    assert evidence["input_summary"] == {"message": "hi"}
    assert evidence["parent_output_refs"] == ["metrics:step-1:v1"]


def test_plan_executor_evidence_records_tool_manifest_and_artifacts(tmp_path):
    plan = _plan(
        _step(
            "step-1",
            inputs={"dataset_id": "raw-1", "seed": 7},
        ),
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([
        _ok({
            "artifact_ref": "artifact:models/model.pkl",
            "report_path": "reports/model.xlsx",
        })
    ])
    runner._tools = FakeManifestTools()

    result = _executor(repo, runner).run("plan-1")

    assert result.status == PlanStatus.DONE
    evidence = repo.load_step_evidence("step-1")
    assert evidence["tool_name"] == "_sample.echo"
    assert evidence["tool_version"] == "1.2.3"
    assert evidence["manifest_hash"].startswith("sha256:")
    assert evidence["source_dataset_refs"] == ["dataset:raw-1"]
    assert evidence["artifact_refs"] == [
        "artifact:models/model.pkl",
        "artifact:reports/model.xlsx",
    ]
    assert evidence["random_seed"] == 7


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

    resumed = _executor(repo, FakeRunner([])).run("plan-1")
    assert resumed.status == PlanStatus.REVIEW
    assert resumed.summary_ref == result.summary_ref


def test_plan_executor_fails_when_final_reviewer_marks_goal_unmet(tmp_path):
    plan = _plan(_step("step-1"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "hi"})])
    hooks = FakeHooks()
    llm = FakeLLM(
        json.dumps(
            {
                "summary": "Needs final model selection.",
                "open_items": ["select production model"],
                "goal_doubt": False,
                "goal_met": False,
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
    summary = repo.load_plan_summary(result.summary_ref)
    assert result.status == PlanStatus.FAILED
    assert loaded.status == PlanStatus.FAILED
    assert summary["goal_met"] is False
    assert summary["llm_goal_met"] is False
    assert summary["open_items"] == ["select production model"]
    assert [call[0] for call in hooks.calls] == ["step.completed", "workflow.completed"]


def test_plan_executor_surfaces_llm_critique_warnings_without_blocking_step(tmp_path):
    plan = _plan(_step("step-1"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "hi"})])
    hooks = FakeHooks()
    llm = FakeLLM(json.dumps({"passed": False, "reasons": ["needs human review"]}))

    result = _executor(
        repo,
        runner,
        reviewer=Reviewer(lambda: llm),
        hooks=hooks,
    ).run("plan-1")

    loaded = repo.load_plan("plan-1")
    completed_payload = hooks.calls[0][1]
    assert result.status == PlanStatus.DONE
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].review_verdicts[-1].reviewer == "llm_critic"
    assert loaded.steps[0].review_verdicts[-1].passed is False
    assert completed_payload["review_warning_count"] == 1
    assert completed_payload["review_warnings"] == [
        {"reviewer": "llm_critic", "reasons": ["needs human review"]}
    ]


def test_plan_executor_fails_final_review_when_success_criteria_fail(tmp_path):
    plan = _plan(
        _step("step-1"),
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": 0.3331,
                "aggregate": "max",
                "label": "OOT KS",
                "target_type": "binary",
            }
        ],
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"target_type": "binary", "metrics": {"oot_ks": 0.2}})])
    hooks = FakeHooks()

    result = _executor(repo, runner, hooks=hooks).run("plan-1")

    loaded = repo.load_plan("plan-1")
    summary = repo.load_plan_summary(result.summary_ref)
    assert result.status == PlanStatus.FAILED
    assert loaded.status == PlanStatus.FAILED
    assert summary["goal_met"] is False
    assert "OOT KS=0.2 < 0.3331" in summary["open_items"]


def test_plan_executor_replans_after_failed_success_criteria_and_continues(tmp_path):
    plan = _plan(
        _step("step-1"),
        success_criteria=[
            {
                "metric": "oot_ks",
                "min": 0.3331,
                "label": "OOT KS",
                "target_type": "binary",
            }
        ],
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([
        _ok({"target_type": "binary", "metrics": {"oot_ks": 0.2}}),
        _ok({"target_type": "binary", "metrics": {"oot_ks": 0.45}}),
    ])
    hooks = FakeHooks()
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-2", index=1, inputs={"message": "try stronger model"})]
    )

    result = _adaptive_executor(repo, runner, planner, hooks=hooks).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert loaded.status == PlanStatus.DONE
    assert [step.id for step in loaded.steps] == ["step-1", "step-2"]
    assert loaded.replan_count == 1
    assert loaded.loop_events[0].reason == "final_review"
    assert planner.replan_calls[0][3] == "final_review"
    assert planner.replan_calls[0][2]["open_items"] == ["OOT KS=0.2 < 0.3331"]
    assert [call[0] for call in hooks.calls] == [
        "step.completed",
        "plan.replanned",
        "step.completed",
        "workflow.completed",
    ]


def test_plan_executor_does_not_replan_invalid_success_criterion_threshold(tmp_path):
    plan = _plan(
        _step("step-1"),
        success_criteria=[{"metric": "oot_ks", "min": "bad", "label": "OOT KS"}],
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"metrics": {"oot_ks": 0.45}})])
    hooks = FakeHooks()
    planner = FakeAdaptivePlanner(replanned_steps=[_step("step-2", index=1)])

    result = _adaptive_executor(repo, runner, planner, hooks=hooks).run("plan-1")

    summary = repo.load_plan_summary(result.summary_ref)
    assert result.status == PlanStatus.FAILED
    assert planner.replan_calls == []
    assert "OOT KS invalid min threshold: 'bad'" in summary["open_items"]


def test_plan_executor_dispatches_feature_computed_for_feature_pack_step(tmp_path):
    repo = _repo(
        tmp_path,
        _plan(_step("step-1", plugin="feature", tool="compute_feature_metrics")),
    )
    runner = FakeRunner([
        _ok({"dataset_id": "dataset-1", "features": ["income"], "metrics": []})
    ])
    hooks = FakeHooks()

    result = _executor(repo, runner, hooks=hooks).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert hooks.calls[:2] == [
        (
            "feature.computed",
            {
                "plan_id": "plan-1",
                "step_id": "step-1",
                "tool": "compute_feature_metrics",
                "output_ref": "metrics:step-1:v1",
                "dataset_id": "dataset-1",
                "features": ["income"],
            },
            "task-1",
        ),
        (
            "step.completed",
            {
                "plan_id": "plan-1",
                "step_id": "step-1",
                "review_warning_count": 0,
                "review_warnings": [],
            },
            "task-1",
        ),
    ]


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


def test_plan_executor_recovers_checking_step_from_persisted_output_without_rerun(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.CHECKING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})
    loaded = repo.load_plan("plan-1")
    loaded.steps[0].output_ref = output_ref
    repo.update_step(loaded.steps[0])
    runner = FakeRunner([])
    hooks = FakeHooks()

    result = _executor(repo, runner, hooks=hooks).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].output_ref == output_ref
    assert [verdict.reviewer for verdict in loaded.steps[0].review_verdicts] == [
        "deterministic",
        "llm_critic",
    ]
    assert [call[0] for call in hooks.calls] == ["step.completed", "workflow.completed"]


def test_plan_executor_recovers_checking_step_with_run_ledger_output_without_step_ref(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.CHECKING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "hi"},
    )
    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].output_ref == output_ref
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["output_ref"] == output_ref


def test_plan_executor_recovers_checking_step_from_succeeded_run_without_step_ref(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.CHECKING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "hi"},
    )
    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})
    repo.finish_step_run(run_id, status="succeeded", output_ref=output_ref, duration_ms=10)
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].output_ref == output_ref
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["output_ref"] == output_ref


def test_plan_executor_recovers_running_step_with_persisted_output_without_rerun(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.RUNNING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    first_run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "first"},
    )
    second_run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "second"},
    )
    output_ref = repo.store_step_output("step-1", {"echoed": "hi"})
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.DONE
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[0].output_ref == output_ref
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [first_run_id, second_run_id]
    assert [run["status"] for run in runs] == ["succeeded", "succeeded"]
    assert [run["output_ref"] for run in runs] == [output_ref, output_ref]


def test_plan_executor_does_not_recover_from_stale_output_version_after_reset(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.CHECKING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    repo.store_step_output("step-1", {"echoed": "old"})
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.FAILED
    assert loaded.steps[0].output_ref is None
    assert "before output was persisted" in loaded.steps[0].error


def test_plan_executor_recovers_running_step_without_output_as_failure_not_rerun(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.RUNNING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "hi"},
    )
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.FAILED
    assert "explicit retry required" in loaded.steps[0].error
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["status"] == "interrupted"
    assert runs[0]["error_kind"] == "ServerRestart"


def test_plan_executor_does_not_replan_recovered_running_failure(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.RUNNING), status=PlanStatus.RUNNING)
    plan.tier = "adaptive"
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([])
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-2", tool="echo", inputs={"message": "unsafe rerun"})]
    )

    result = _adaptive_executor(repo, runner, planner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    assert planner.replan_calls == []
    assert [step.id for step in loaded.steps] == ["step-1"]
    assert loaded.replan_count == 0
    assert "explicit retry required" in loaded.steps[0].error


def test_plan_executor_recovers_checking_step_without_output_as_failure_not_rerun(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.CHECKING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([])

    result = _executor(repo, runner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    assert loaded.steps[0].status == StepStatus.FAILED
    assert "before output was persisted" in loaded.steps[0].error


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


def test_plan_executor_replan_from_instruction(tmp_path):
    """A user-driven structural replan regenerates the remaining steps to satisfy a
    free-text instruction (driver §3 提指令→重规划), passing the instruction to the planner;
    with no planner it returns False so the caller keeps the current plan."""
    plan = _plan(
        _step("step-1"),
        _step("step-2", index=1, depends_on=["step-1"]),
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([])
    planner = FakeAdaptivePlanner(
        replanned_steps=lambda loaded: [
            _step("step-A"),
            _step("step-B", index=1, depends_on=["step-A"]),
        ]
    )

    ok = _adaptive_executor(repo, runner, planner).replan_from_instruction(
        "plan-1", "把流程改成只跑 A、B 两步"
    )

    assert ok is True
    assert planner.last_instruction == "把流程改成只跑 A、B 两步"
    loaded = repo.load_plan("plan-1")
    assert [step.id for step in loaded.steps] == ["step-A", "step-B"]
    assert loaded.replan_count == 1
    assert loaded.loop_events[-1].type == "replan"
    assert loaded.loop_events[-1].reason == "user_instruction"
    assert loaded.loop_events[-1].instruction == "把流程改成只跑 A、B 两步"  # audit trail kept

    # No planner → graceful False; the current plan is untouched.
    assert _executor(repo, FakeRunner([])).replan_from_instruction("plan-1", "x") is False


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
    assert len(loaded.loop_events) == 1
    assert loaded.loop_events[0].type == "replan"
    assert loaded.loop_events[0].reason == "decision_point"
    assert loaded.loop_events[0].trigger_step_id == "step-1"
    assert loaded.loop_events[0].at
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
    assert len(loaded.loop_events) == 1
    assert loaded.loop_events[0].type == "replan"
    assert loaded.loop_events[0].reason == "failure"
    assert loaded.loop_events[0].trigger_step_id == "step-1"
    assert loaded.loop_events[0].tool_ref == "_sample.fail_tool"
    assert planner.replan_calls[0][3] == "failure"
    assert len(runner.calls) == 2


def test_plan_executor_records_no_progress_when_repeated_failures_block_replan(tmp_path):
    plan = _plan(
        _step("step-1", tool="fail_tool", status=StepStatus.FAILED),
        _step("step-2", index=1, tool="fail_tool", status=StepStatus.FAILED),
    )
    repo = _repo(tmp_path, plan)
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-3", tool="echo", inputs={"message": "fixed"})]
    )

    result = _adaptive_executor(repo, FakeRunner([]), planner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert planner.replan_calls == []
    assert loaded.replan_count == 0
    assert len(loaded.loop_events) == 1
    assert loaded.loop_events[0].type == "no_progress"
    assert loaded.loop_events[0].reason == "failure"
    assert loaded.loop_events[0].trigger_step_id == "step-1"
    assert loaded.loop_events[0].tool_ref == "_sample.fail_tool"
    assert loaded.loop_events[0].at


def test_plan_executor_no_progress_uses_failure_history_after_replan_deleted_step(tmp_path):
    plan = _plan(
        _step("step-2", tool="fail_tool", status=StepStatus.FAILED),
    )
    plan.loop_events = [
        LoopEvent(
            type="replan",
            reason="failure",
            at="2026-01-01T00:00:00Z",
            trigger_step_id="step-1",
            tool_ref="_sample.fail_tool",
        )
    ]
    repo = _repo(tmp_path, plan)
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-3", tool="echo", inputs={"message": "fixed"})]
    )

    result = _adaptive_executor(repo, FakeRunner([]), planner).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert planner.replan_calls == []
    assert loaded.loop_events[-1].type == "no_progress"
    assert loaded.loop_events[-1].trigger_step_id == "step-2"
    assert loaded.loop_events[-1].tool_ref == "_sample.fail_tool"


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
    assert len(loaded.loop_events) == 1
    assert loaded.loop_events[0].type == "explore_segment"
    assert loaded.loop_events[0].reason == "explore_segment"
    assert loaded.loop_events[0].at
    assert len(planner.explore_calls) == 2
    assert len(runner.calls) == 2
