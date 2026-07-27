import json
from types import SimpleNamespace

import pytest

from marvis.db import PlanRepository, TaskRepository, connect, init_db
from marvis.domain import TASK_TYPE_MODELING, TaskCreate
from marvis.llm_client import LLMClientError
from marvis.llm_settings import LLMSettingsError
from marvis.job_cancellation import JobCancellationToken
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
from marvis.orchestrator.errors import OrchestratorError
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.plugins.manifest import (
    EffectTargetPolicy,
    GovernancePolicy,
    PluginManifest,
    ToolRef,
    ToolSpec,
)
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


class LLMFailingAdaptivePlanner:
    def replan(self, *args, **kwargs):
        raise LLMClientError("local model timed out")

    def next_explore_segment(self, *args, **kwargs):
        raise LLMClientError("local model timed out")


class LLMSettingsFailingAdaptivePlanner(LLMFailingAdaptivePlanner):
    def next_explore_segment(self, *args, **kwargs):
        raise LLMSettingsError("model disabled")


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
    policy=None,
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
        policy=policy or GovernancePolicy(),
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


def _repo_with_agent_task(tmp_path, plan):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = task_repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="建模人员",
            source_dir=str(tmp_path),
            task_type=TASK_TYPE_MODELING,
            run_mode="agent",
        )
    )
    plan.task_id = task.id
    repo = PlanRepository(db_path)
    repo.create_plan(plan)
    return repo, task_repo, task.id


def _force_confirmed_bit(repo, step_id):
    """Simulate a legacy/tampered raw confirmation without governance proof."""

    with connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE plan_steps SET confirmed = 1 WHERE id = ?",
            (step_id,),
        )


def _seed_run_for_checking_step(repo, step_id, *, inputs=None):
    """Seed a run-ledger row for a step that is now CHECKING, the way it really
    happens: the run is opened while the step is RUNNING (start_step_run only
    accepts RUNNING), then the step advances to CHECKING before the simulated
    crash. Returns the run id, leaving the step back in CHECKING."""
    plan = repo.load_plan("plan-1")
    step = next(s for s in plan.steps if s.id == step_id)
    original_status = step.status
    step.status = StepStatus.RUNNING
    repo.update_step(step)
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id=step_id,
        tool_ref="_sample.echo",
        inputs=inputs or {},
    )
    step.status = original_status
    repo.update_step(step)
    return run_id


def _executor(
    repo,
    runner,
    reviewer=None,
    subagents=None,
    hooks=None,
    authorizer=None,
    task_repo=None,
):
    return PlanExecutor(
        repo,
        runner,
        reviewer or Reviewer(lambda: FakeLLM()),
        subagents,
        hooks or FakeHooks(),
        HarnessState(repo),
        authorizer=authorizer,
        task_repo=task_repo,
    )


def _adaptive_executor(repo, runner, planner, reviewer=None, hooks=None, authorizer=None):
    return PlanExecutor(
        repo,
        runner,
        reviewer or Reviewer(lambda: FakeLLM()),
        None,
        hooks or FakeHooks(),
        HarnessState(repo),
        planner=planner,
        authorizer=authorizer,
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


def test_plan_executor_persists_tuning_progress_without_changing_step_result(tmp_path):
    class ProgressRunner:
        def __init__(self, task_repo):
            self._tools = FakeTools()
            self._task_repo = task_repo
            self.message_ids = []

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "xgb",
                    "trial": 4,
                    "trial_total": 40,
                    "completed_trials": 4,
                    "total_trials": 40,
                }
            )
            self.message_ids.append(
                self._task_repo.list_agent_messages(task_id)[0]["id"]
            )
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "xgb",
                    "trial": 5,
                    "trial_total": 40,
                    "completed_trials": 5,
                    "total_trials": 40,
                }
            )
            messages = self._task_repo.list_agent_messages(task_id)
            assert len(messages) == 1
            self.message_ids.append(messages[0]["id"])
            return _ok({"best_params": {"max_depth": 4}})

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )
    runner = ProgressRunner(task_repo)

    result = _executor(repo, runner, task_repo=task_repo).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert runner.message_ids[0] == runner.message_ids[1]
    runs = repo.list_step_runs("step-tune")
    assert runs[0]["status"] == "succeeded"
    assert runs[0]["progress"] == {
        "kind": "model_tuning",
        "algorithm": "xgb",
        "trial": 5,
        "trial_total": 40,
        "completed_trials": 5,
        "total_trials": 40,
    }
    messages = task_repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["id"] == runner.message_ids[0]
    assert messages[0]["metadata"]["kind"] == "tool_progress"
    assert messages[0]["metadata"]["plan_id"] == "plan-1"
    assert messages[0]["metadata"]["step_id"] == "step-tune"
    assert messages[0]["metadata"]["run_id"] == runs[0]["id"]
    assert messages[0]["metadata"]["status"] == "succeeded"
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["progress"] == runs[0]["progress"]


def test_plan_executor_closes_tuning_progress_message_on_failure(tmp_path):
    class ProgressRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "lgb",
                    "trial": 3,
                    "trial_total": 40,
                }
            )
            return _fail("tuning failed")

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )

    result = _executor(repo, ProgressRunner(), task_repo=task_repo).run("plan-1")

    assert result.status == PlanStatus.FAILED
    messages = task_repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["metadata"]["status"] == "failed"
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["progress"]["trial"] == 3


def test_plan_executor_closes_tuning_progress_message_on_cancellation(tmp_path):
    class TuningCancelled(BaseException):
        pass

    class ProgressRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "catboost",
                    "trial": 2,
                    "trial_total": 40,
                }
            )
            raise TuningCancelled("cancelled")

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )

    with pytest.raises(TuningCancelled):
        _executor(repo, ProgressRunner(), task_repo=task_repo).run("plan-1")

    messages = task_repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["metadata"]["status"] == "cancelled"
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["progress"]["trial"] == 2


def test_plan_executor_marks_progress_cancelled_for_cooperative_plan_cancel(tmp_path):
    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )

    class CancellingRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "xgb",
                    "trial": 7,
                    "trial_total": 40,
                }
            )
            repo.set_plan_status("plan-1", PlanStatus.CANCELLED)
            return _ok({"best_params": {"max_depth": 3}})

    result = _executor(repo, CancellingRunner(), task_repo=task_repo).run("plan-1")

    assert result.status == PlanStatus.CANCELLED
    messages = task_repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["metadata"]["status"] == "cancelled"
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["progress"]["trial"] == 7


def test_plan_executor_user_cancellation_interrupts_current_step_and_plan(tmp_path):
    token = JobCancellationToken(job_id="driver-job-1")

    class CancellableRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(
            self,
            ref,
            inputs,
            *,
            task_id,
            progress_callback=None,
            cancellation_check=None,
        ):
            assert progress_callback is not None
            assert cancellation_check is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "lgb",
                    "trial": 8,
                    "trial_total": 40,
                    "checkpoint_saved": True,
                }
            )
            token.cancel()
            cancellation_check()
            raise AssertionError("unreachable")

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )

    result = _executor(repo, CancellableRunner(), task_repo=task_repo).run(
        "plan-1",
        cancellation_check=token.raise_if_cancelled,
    )

    assert result.status == PlanStatus.CANCELLED
    plan = repo.load_plan("plan-1")
    assert plan.status == PlanStatus.CANCELLED
    assert plan.steps[0].status == StepStatus.FAILED
    assert "用户" in str(plan.steps[0].error)
    runs = repo.list_step_runs("step-tune")
    assert len(runs) == 1
    assert runs[0]["status"] == "interrupted"
    assert runs[0]["error_kind"] == "user_cancelled"
    assert runs[0]["progress"]["checkpoint_saved"] is True
    messages = task_repo.list_agent_messages(task_id)
    assert messages[0]["metadata"]["status"] == "cancelled"
    assert messages[0]["metadata"]["streaming"] is False


def test_plan_executor_commits_success_before_late_cancellation_boundary(tmp_path):
    """A stop arriving after a tool returns must not discard its side-effect evidence."""

    token = JobCancellationToken(job_id="driver-job-late-cancel")

    class LateCancellingRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(
            self,
            ref,
            inputs,
            *,
            task_id,
            cancellation_check=None,
        ):
            assert cancellation_check is not None
            token.cancel()
            return _ok({"artifact_id": "artifact-complete"})

    repo = _repo(tmp_path, _plan(_step("step-1")))

    result = _executor(repo, LateCancellingRunner()).run(
        "plan-1",
        cancellation_check=token.raise_if_cancelled,
    )

    assert result.status == PlanStatus.CANCELLED
    plan = repo.load_plan("plan-1")
    assert plan.steps[0].status == StepStatus.DONE
    assert repo.load_step_output("step-1") == {"artifact_id": "artifact-complete"}
    runs = repo.list_step_runs("step-1")
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"


def test_plan_executor_retries_transient_terminal_progress_message_update(tmp_path):
    class ProgressRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback(
                {
                    "kind": "model_tuning",
                    "algorithm": "lgb",
                    "trial": 40,
                    "trial_total": 40,
                }
            )
            return _ok({"best_params": {}})

    class FlakyTaskRepository:
        def __init__(self, delegate):
            self._delegate = delegate
            self.update_calls = 0

        def add_agent_message(self, *args, **kwargs):
            return self._delegate.add_agent_message(*args, **kwargs)

        def update_agent_message(self, *args, **kwargs):
            self.update_calls += 1
            if self.update_calls == 1:
                raise RuntimeError("database is temporarily locked")
            return self._delegate.update_agent_message(*args, **kwargs)

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )
    flaky_repo = FlakyTaskRepository(task_repo)

    result = _executor(repo, ProgressRunner(), task_repo=flaky_repo).run("plan-1")

    assert result.status == PlanStatus.DONE
    # The injected publisher fails once; the canonical TaskRepository fallback
    # closes the same message id without changing the tool result.
    assert flaky_repo.update_calls == 1
    messages = task_repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["metadata"]["status"] == "succeeded"
    assert messages[0]["metadata"]["streaming"] is False


def test_plan_executor_waits_for_delayed_progress_before_finalizing_run(
    tmp_path,
    monkeypatch,
):
    import threading
    import time

    progress_started = threading.Event()

    class DelayedProgressRunner:
        def __init__(self):
            self._tools = FakeTools()
            self.thread = None

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            payload = {
                "kind": "model_tuning",
                "algorithm": "xgb",
                "trial": 11,
                "trial_total": 40,
            }
            self.thread = threading.Thread(
                target=lambda: progress_callback(payload),
                daemon=True,
            )
            self.thread.start()
            assert progress_started.wait(timeout=1)
            return _ok({"best_params": {}})

    repo, task_repo, task_id = _repo_with_agent_task(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )
    original_update = repo.update_step_run_progress

    def delayed_update(run_id, payload):
        progress_started.set()
        time.sleep(0.05)
        return original_update(run_id, payload)

    monkeypatch.setattr(repo, "update_step_run_progress", delayed_update)
    runner = DelayedProgressRunner()

    result = _executor(repo, runner, task_repo=task_repo).run("plan-1")
    runner.thread.join(timeout=1)

    assert result.status == PlanStatus.DONE
    run = repo.list_step_runs("step-tune")[0]
    message = task_repo.list_agent_messages(task_id)[0]
    assert run["progress"]["trial"] == 11
    assert message["metadata"]["progress"] == run["progress"]
    assert message["metadata"]["streaming"] is False


def test_plan_executor_ignores_progress_persistence_failure(tmp_path, monkeypatch):
    class ProgressRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback({"kind": "model_tuning", "trial": 1})
            return _ok({"best_params": {}})

    repo = _repo(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )
    monkeypatch.setattr(
        repo,
        "update_step_run_progress",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable")),
    )

    result = _executor(repo, ProgressRunner()).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert repo.load_plan("plan-1").steps[0].status == StepStatus.DONE


def test_plan_executor_ignores_tool_progress_message_publisher_failure(tmp_path):
    class ProgressRunner:
        def __init__(self):
            self._tools = FakeTools()

        def invoke(self, ref, inputs, *, task_id, progress_callback=None):
            assert progress_callback is not None
            progress_callback({"kind": "model_tuning", "trial": 1})
            progress_callback({"kind": "model_tuning", "trial": 2})
            return _ok({"best_params": {}})

    class FailingTaskRepository:
        def add_agent_message(self, *_args, **_kwargs):
            raise RuntimeError("message database unavailable")

        def update_agent_message(self, *_args, **_kwargs):
            raise RuntimeError("message database unavailable")

    repo = _repo(
        tmp_path,
        _plan(_step("step-tune", plugin="modeling", tool="tune_hyperparameters")),
    )

    result = _executor(
        repo,
        ProgressRunner(),
        task_repo=FailingTaskRepository(),
    ).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert repo.load_plan("plan-1").steps[0].status == StepStatus.DONE
    assert repo.list_step_runs("step-tune")[0]["progress"]["trial"] == 2


def test_plan_executor_resolves_numeric_array_indices_in_ref_paths(tmp_path):
    plan = _plan(
        _step("step-1", inputs={"message": "hi"}),
        _step(
            "step-2",
            index=1,
            inputs={"message": "$ref:step-1.output.items.0.message"},
            depends_on=["step-1"],
        ),
    )
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([
        _ok({"items": [{"message": "from-list"}]}),
        _ok({"echoed": "from-list"}),
    ])

    result = _executor(repo, runner).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert runner.calls[1][1] == {"message": "from-list"}


def test_plan_executor_raises_typed_error_for_missing_ref_path(tmp_path):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    repo.store_step_output("step-1", {"items": []})
    executor = _executor(repo, FakeRunner([]))

    with pytest.raises(
        OrchestratorError,
        match=r"\$ref:step-1\.output\.items\.0\.message",
    ) as exc_info:
        executor._resolve_refs(
            {"message": "$ref:step-1.output.items.0.message"}
        )
    assert exc_info.type.__name__ == "RefResolutionError"


def test_plan_executor_run_stops_cleanly_when_plan_cancelled_between_steps(tmp_path):
    # REL-5: a cancel request (POST /api/plans/{id}/cancel) can land between
    # two step executions inside the same run() call — each _execute_step is
    # itself uninterruptible mid-tool-invocation, so the checkpoint has to be
    # the top of the main loop. Without it, the loop would try another
    # _set_plan_status transition on an already-CANCELLED plan and raise
    # IllegalPlanTransition instead of returning cleanly.
    plan = _plan(
        _step("step-1", inputs={"message": "hi"}),
        _step(
            "step-2",
            index=1,
            inputs={"message": "again"},
            depends_on=["step-1"],
        ),
    )
    repo = _repo(tmp_path, plan)

    class CancelAfterFirstStepRunner:
        def __init__(self):
            self._tools = FakeTools()
            self.calls = []

        def invoke(self, ref, inputs, *, task_id):
            self.calls.append((ref, inputs, task_id))
            if len(self.calls) == 1:
                repo.set_plan_status("plan-1", PlanStatus.CANCELLED)
            return _ok({"echoed": inputs.get("message")})

    runner = CancelAfterFirstStepRunner()

    result = _executor(repo, runner).run("plan-1")

    assert result.status == PlanStatus.CANCELLED
    assert len(runner.calls) == 1
    loaded = repo.load_plan("plan-1")
    assert loaded.status == PlanStatus.CANCELLED
    assert loaded.steps[0].status == StepStatus.DONE
    assert loaded.steps[1].status == StepStatus.PENDING


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


def test_plan_executor_routes_to_review_when_llm_alone_marks_goal_unmet(tmp_path):
    # AGT-3: with the step DONE and no configured/failed success_criteria, a bare
    # llm_goal_met=false is doubt, not a veto — the plan routes to REVIEW (human
    # re-check) instead of FAILED, and workflow.completed does NOT fire (that
    # event is reserved for a plan reaching a genuine terminal DONE/FAILED state).
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
    assert result.status == PlanStatus.REVIEW
    assert loaded.status == PlanStatus.REVIEW
    assert summary["goal_met"] is False
    assert summary["llm_goal_met"] is False
    assert summary["goal_doubt"] is True
    assert summary["open_items"] == ["select production model"]
    assert [call[0] for call in hooks.calls] == ["step.completed"]


def test_plan_executor_still_fails_when_success_criteria_fail_alongside_llm_doubt(tmp_path):
    # The narrowed LLM authority (AGT-3) does not touch the deterministic path:
    # a real success_criteria failure still fails the plan even if the LLM also
    # (redundantly) says goal_met=false.
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
    llm = FakeLLM(
        json.dumps(
            {
                "summary": "Needs final model selection.",
                "open_items": [],
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
    assert summary["goal_doubt"] is False
    assert "OOT KS=0.2 < 0.3331" in summary["open_items"]
    assert [call[0] for call in hooks.calls] == ["step.completed", "workflow.completed"]


def test_plan_executor_surfaces_llm_critique_warnings_without_blocking_step(tmp_path):
    # AGT-6: llm_critique is now scoped to decision_point/needs_confirmation steps
    # (or metric-bearing output) — use a decision_point step so this test still
    # exercises the "soft warning surfaces but does not block DONE" behavior.
    plan = _plan(_step("step-1", decision_point=True))
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


def test_plan_executor_skips_llm_critique_for_plain_steps_without_confirmation_or_metrics(tmp_path):
    # AGT-6: a plain read/profile-type step (no decision_point, no
    # needs_confirmation, no metric-shaped output) skips the LLM round-trip
    # entirely — no llm_critic verdict at all, since llm_critique is never called
    # for that step (final_review's own single end-of-plan LLM call is separate
    # and still happens, so we assert on the per-step verdict shape, not on
    # whether the shared FakeLLM was ever invoked).
    plan = _plan(_step("step-1"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"echoed": "hi"})])
    hooks = FakeHooks()
    llm = FakeLLM(json.dumps({"passed": False, "reasons": ["should not be called"]}))

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
    assert [verdict.reviewer for verdict in loaded.steps[0].review_verdicts] == ["deterministic"]
    assert completed_payload["review_warning_count"] == 0


def test_plan_executor_critiques_plain_steps_whose_output_carries_metrics(tmp_path):
    # AGT-6: even a step that is neither decision_point nor needs_confirmation
    # still gets critiqued when its output carries metric-shaped values (e.g. a
    # training step's KS/AUC) — the trigger surface is metric-aware, not just
    # gate-aware.
    plan = _plan(_step("step-1"))
    repo = _repo(tmp_path, plan)
    runner = FakeRunner([_ok({"metrics": {"oot_ks": 0.41}})])
    hooks = FakeHooks()
    llm = FakeLLM(json.dumps({"passed": True, "reasons": []}))

    _executor(
        repo,
        runner,
        reviewer=Reviewer(lambda: llm),
        hooks=hooks,
    ).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert [verdict.reviewer for verdict in loaded.steps[0].review_verdicts] == [
        "deterministic",
        "llm_critic",
    ]
    assert llm.calls


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


def _protected_effect_policy():
    return GovernancePolicy(
        human_decision_gate="required",
        effect_authorization="required",
        effect_target=EffectTargetPolicy(
            kind="strategy",
            id_input="strategy_id",
            expected_statuses=("draft",),
            result_status="adopted",
        ),
    )


def _protected_human_policy():
    return GovernancePolicy(human_decision_gate="required")


def test_executor_uses_canonical_human_policy_even_if_legacy_gate_bit_is_false(
    tmp_path,
):
    step = _step(
        "step-1",
        needs_confirmation=False,
        policy=GovernancePolicy(human_decision_gate="required"),
    )
    repo = _repo(tmp_path, _plan(step))
    runner = FakeRunner([_ok({"value": 1})])
    executor = PlanExecutor(
        repo,
        runner,
        Reviewer(lambda: FakeLLM()),
        None,
        FakeHooks(),
        HarnessState(repo),
    )

    result = executor.run("plan-1")

    assert result.status == PlanStatus.AWAITING_CONFIRM
    assert runner.calls == []
    loaded = repo.load_plan("plan-1")
    assert loaded.steps[0].status == StepStatus.AWAITING_CONFIRM


class ProtectedEffectRunner:
    def __init__(self, outputs, *, policy=None):
        self.outputs = list(outputs)
        self.calls = []
        effective_policy = policy or _protected_effect_policy()
        self._tools = SimpleNamespace(
            resolve=lambda _ref: SimpleNamespace(
                failure_policy="retry",
                policy=effective_policy,
            )
        )

    def invoke(self, ref, inputs, *, task_id, execution_context=None):
        self.calls.append((ref, inputs, task_id, execution_context))
        return self.outputs.pop(0)


def test_plan_executor_passes_effect_context_out_of_band_to_runner(tmp_path):
    step = _step(
        "step-adopt",
        tool="adopt",
        inputs={"strategy_id": "strategy-1"},
        needs_confirmation=True,
        policy=_protected_effect_policy(),
    )
    repo = _repo(tmp_path, _plan(step))
    runner = ProtectedEffectRunner([_ok({"status": "adopted"})])
    context = SimpleNamespace(
        plan_id="plan-1",
        plan_revision=0,
        step_id="step-adopt",
        decision_id="decision-1",
        approval_id="approval-1",
        runtime_generation="runtime-1",
        human_decision_required=True,
        effect_authorization_required=True,
    )
    authorizer_calls = []

    class Authorizer:
        def execution_context_for(self, *, plan, step, inputs):
            authorizer_calls.append((plan.id, step.id, inputs))
            return context

    executor = _executor(repo, runner, authorizer=Authorizer())
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-adopt")
    result = executor.run("plan-1")

    assert result.status == PlanStatus.DONE
    assert authorizer_calls == [
        ("plan-1", "step-adopt", {"strategy_id": "strategy-1"})
    ]
    assert runner.calls == [
        (
            ToolRef("_sample", "adopt"),
            {"strategy_id": "strategy-1"},
            "task-1",
            context,
        )
    ]


def test_plan_executor_never_retries_or_replans_protected_effect_failure(tmp_path):
    step = _step(
        "step-adopt",
        tool="adopt",
        inputs={"strategy_id": "strategy-1"},
        needs_confirmation=True,
        policy=_protected_effect_policy(),
    )
    plan = _plan(step)
    plan.tier = "adaptive"
    repo = _repo(tmp_path, plan)
    runner = ProtectedEffectRunner(
        [_fail("effect outcome uncertain"), _ok({"status": "adopted"})]
    )
    planner = FakeAdaptivePlanner(
        replanned_steps=[_step("step-retry", tool="adopt")]
    )

    executor = _adaptive_executor(
        repo,
        runner,
        planner,
        authorizer=lambda **kwargs: SimpleNamespace(
            plan_id="plan-1",
            plan_revision=0,
            step_id="step-adopt",
            approval_id="approval-1",
            runtime_generation="runtime-1",
        ),
    )
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-adopt")
    result = executor.run("plan-1")

    assert result.status == PlanStatus.FAILED
    assert len(runner.calls) == 1
    assert planner.replan_calls == []
    assert repo.load_plan("plan-1").steps[0].status == StepStatus.FAILED


def test_plan_executor_fails_closed_without_issued_effect_context(tmp_path):
    step = _step(
        "step-adopt",
        tool="adopt",
        inputs={"strategy_id": "strategy-1"},
        needs_confirmation=True,
        policy=_protected_effect_policy(),
    )
    repo = _repo(tmp_path, _plan(step))
    runner = ProtectedEffectRunner([_ok({"status": "adopted"})])
    executor = _executor(repo, runner)
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-adopt")

    result = executor.run("plan-1")

    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    assert "execution context unavailable" in repo.load_plan("plan-1").steps[0].error


def test_confirmed_bit_without_decision_record_never_executes_human_gate(tmp_path):
    step = _step(
        "step-review",
        tool="review",
        inputs={"candidate": "candidate-1"},
        needs_confirmation=True,
        policy=_protected_human_policy(),
    )
    repo = _repo(tmp_path, _plan(step))
    runner = ProtectedEffectRunner(
        [_ok({"status": "reviewed"})],
        policy=_protected_human_policy(),
    )
    executor = _executor(repo, runner)
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-review")
    result = executor.run("plan-1")

    assert result.status == PlanStatus.FAILED
    assert runner.calls == []
    loaded = repo.load_plan("plan-1").steps[0]
    assert loaded.status == StepStatus.FAILED
    assert "execution context unavailable" in loaded.error


def test_valid_decision_only_context_executes_human_gate(tmp_path):
    step = _step(
        "step-review",
        tool="review",
        inputs={"candidate": "candidate-1"},
        needs_confirmation=True,
        policy=_protected_human_policy(),
    )
    repo = _repo(tmp_path, _plan(step))
    runner = ProtectedEffectRunner(
        [_ok({"status": "reviewed"})],
        policy=_protected_human_policy(),
    )
    context = SimpleNamespace(
        plan_id="plan-1",
        plan_revision=0,
        step_id="step-review",
        decision_id="decision-1",
        approval_id=None,
        runtime_generation="runtime-1",
        human_decision_required=True,
        effect_authorization_required=False,
    )
    calls = []

    class Authorizer:
        def execution_context_for(self, *, plan, step, inputs):
            calls.append((plan.id, step.id, inputs))
            return context

    executor = _executor(repo, runner, authorizer=Authorizer())
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-review")
    result = executor.run("plan-1")

    assert result.status == PlanStatus.DONE
    assert calls == [
        ("plan-1", "step-review", {"candidate": "candidate-1"})
    ]
    assert runner.calls == [
        (
            ToolRef("_sample", "review"),
            {"candidate": "candidate-1"},
            "task-1",
            context,
        )
    ]


def test_plan_executor_never_retries_or_replans_human_gate_failure(tmp_path):
    step = _step(
        "step-review",
        tool="review",
        needs_confirmation=True,
        policy=_protected_human_policy(),
    )
    plan = _plan(step)
    plan.tier = "adaptive"
    repo = _repo(tmp_path, plan)
    runner = ProtectedEffectRunner(
        [_fail("decision proof rejected"), _ok({"status": "reviewed"})],
        policy=_protected_human_policy(),
    )
    planner = FakeAdaptivePlanner(replanned_steps=[_step("step-retry")])
    context = SimpleNamespace(
        plan_id="plan-1",
        plan_revision=0,
        step_id="step-review",
        decision_id="decision-1",
        approval_id=None,
        runtime_generation="runtime-1",
        human_decision_required=True,
        effect_authorization_required=False,
    )
    executor = _adaptive_executor(
        repo,
        runner,
        planner,
        authorizer=lambda **_kwargs: context,
    )
    assert executor.run("plan-1").status == PlanStatus.AWAITING_CONFIRM
    _force_confirmed_bit(repo, "step-review")
    result = executor.run("plan-1")

    assert result.status == PlanStatus.FAILED
    assert len(runner.calls) == 1
    assert planner.replan_calls == []


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
    run_id = _seed_run_for_checking_step(repo, "step-1", inputs={"message": "hi"})
    output_ref = repo.store_step_output(
        "step-1",
        {"echoed": "hi"},
        evidence={"step_run_id": run_id},
    )
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
    run_id = _seed_run_for_checking_step(repo, "step-1", inputs={"message": "hi"})
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
    output_ref = repo.store_step_output(
        "step-1",
        {"echoed": "hi"},
        evidence={"step_run_id": second_run_id},
    )
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


def test_plan_executor_does_not_recover_output_predating_running_attempt(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.RUNNING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    stale_output_ref = repo.store_step_output("step-1", {"echoed": "old attempt"})
    run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "retry"},
    )

    result = _executor(repo, FakeRunner([])).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert loaded.steps[0].status == StepStatus.FAILED
    assert loaded.steps[0].output_ref is None
    assert stale_output_ref != loaded.steps[0].output_ref
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [run_id]
    assert runs[0]["status"] == "interrupted"
    assert runs[0]["output_ref"] is None


def test_plan_executor_does_not_recover_late_output_from_previous_attempt(tmp_path):
    plan = _plan(_step("step-1", status=StepStatus.RUNNING), status=PlanStatus.RUNNING)
    repo = _repo(tmp_path, plan)
    current_run_id = repo.start_step_run(
        plan_id="plan-1",
        step_id="step-1",
        tool_ref="_sample.echo",
        inputs={"message": "current retry"},
    )
    repo.store_step_output(
        "step-1",
        {"echoed": "late previous attempt"},
        evidence={"step_run_id": "previous-run"},
    )

    result = _executor(repo, FakeRunner([])).run("plan-1")

    loaded = repo.load_plan("plan-1")
    assert result.status == PlanStatus.FAILED
    assert loaded.steps[0].status == StepStatus.FAILED
    assert loaded.steps[0].output_ref is None
    runs = repo.list_step_runs("step-1")
    assert [run["id"] for run in runs] == [current_run_id]
    assert runs[0]["status"] == "interrupted"


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


def test_plan_executor_llm_replan_failure_does_not_leave_plan_running(tmp_path):
    repo = _repo(tmp_path, _plan(_step("step-1", tool="fail_tool")))

    result = _adaptive_executor(
        repo,
        FakeRunner([_fail("temporary execution failure")]),
        LLMFailingAdaptivePlanner(),
    ).run("plan-1")

    assert result.status == PlanStatus.FAILED
    assert repo.load_plan("plan-1").status == PlanStatus.FAILED


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


def test_plan_executor_llm_explore_failure_finalizes_instead_of_staying_running(tmp_path):
    plan = _plan(_step("step-1"))
    plan.novel_mode = "explore"
    repo = _repo(tmp_path, plan)

    result = _adaptive_executor(
        repo,
        FakeRunner([_ok({"echoed": "first"})]),
        LLMFailingAdaptivePlanner(),
    ).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert repo.load_plan("plan-1").status == PlanStatus.DONE


def test_plan_executor_disabled_explore_model_finalizes_instead_of_staying_running(tmp_path):
    plan = _plan(_step("step-1"))
    plan.novel_mode = "explore"
    repo = _repo(tmp_path, plan)

    result = _adaptive_executor(
        repo,
        FakeRunner([_ok({"echoed": "first"})]),
        LLMSettingsFailingAdaptivePlanner(),
    ).run("plan-1")

    assert result.status == PlanStatus.DONE
    assert repo.load_plan("plan-1").status == PlanStatus.DONE
