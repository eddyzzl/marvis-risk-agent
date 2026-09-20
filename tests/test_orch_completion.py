import pytest

from marvis.db import PlanRepository, PluginRepository
from marvis.orchestrator.contracts import PlanStatus, ReviewVerdict, StepStatus
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.plan_recovery import PlanStepRecovery
from marvis.orchestrator.reviewer import FinalReview
from marvis.plugins.hooks import HookDispatcher
from marvis.plugins.registry import PluginRegistry
from marvis.recovery import reclaim_running_plans
from marvis.repositories.hook_deliveries import HookDeliveryRepository
from tests.test_orch_executor import (
    FakeHooks,
    FakeRunner,
    _executor,
    _ok,
    _plan,
    _repo,
    _repo_with_agent_task,
    _step,
)


class Crash(BaseException):
    pass


class CountingReviewer:
    def __init__(self):
        self.checks = 0
        self.critiques = 0

    def deterministic_check(self, _step, _output):
        self.checks += 1
        return ReviewVerdict("deterministic", True, [], "fixed")

    def llm_critique(self, _step, _output, _goal):
        self.critiques += 1
        return ReviewVerdict("llm_critic", False, ["soft warning"], "fixed")

    def final_review(self, _plan, _outputs, _goal):
        return FinalReview(True, "complete", [])


def _hooks(repo):
    plugins = PluginRepository(repo.db_path)
    dispatcher = HookDispatcher(PluginRegistry(plugins), FakeRunner(), plugins)
    dispatcher.rebuild_index()
    return dispatcher


@pytest.mark.parametrize(
    "decision,output,expected_critique",
    [
        (False, {"echo": "hello"}, 0),
        (True, {"echo": "hello"}, 1),
        (False, {"metrics": {"auc": 0.8}}, 1),
    ],
)
def test_execution_recovery_share_review_policy_and_event_binding(
    tmp_path, monkeypatch, decision, output, expected_critique
):
    repo = _repo(tmp_path, _plan(_step("step-1", decision_point=decision)))
    reviewer, hooks = CountingReviewer(), FakeHooks()
    runner = FakeRunner([_ok(output)])
    original = repo.update_step

    def crash_before_done(step):
        if step.status == StepStatus.DONE:
            raise Crash()
        return original(step)

    with monkeypatch.context() as patch:
        patch.setattr(repo, "update_step", crash_before_done)
        with pytest.raises(Crash):
            _executor(repo, runner, reviewer=reviewer, hooks=hooks).run("plan-1")
    before = hooks.calls[0]
    restored = PlanRepository(repo.db_path)
    plan = restored.load_plan("plan-1")
    recovered_hooks = FakeHooks()
    PlanStepRecovery(
        restored, reviewer, recovered_hooks, HarnessState(restored)
    ).recover_inflight_steps(plan)
    assert recovered_hooks.calls == [before]
    assert reviewer.critiques == expected_critique
    assert reviewer.checks == 1
    step = restored.load_plan("plan-1").steps[0]
    assert step.status == StepStatus.DONE
    payload = before[1]
    run = restored.list_step_runs(step.id)[0]
    assert payload["execution_id"] == run["id"]
    assert payload["output_ref"] == run["output_ref"]
    assert payload["output_hash"] == run["output_hash"]
    assert payload["review_warning_count"] == expected_critique
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "window,expected_done,effect_count",
    [
        ("before_output", False, 0),
        ("after_output", True, 1),
        ("after_review", True, 1),
        ("before_claim", True, 1),
        ("after_effect", False, 1),
        ("after_receipt", True, 1),
        ("before_done", True, 1),
    ],
)
def test_step_completion_crash_windows_recover_without_reexecuting_tool(
    tmp_path, monkeypatch, window, expected_done, effect_count
):
    repo = _repo(tmp_path, _plan(_step("step-1", decision_point=True)))
    reviewer, hooks = CountingReviewer(), _hooks(repo)
    runner, effects = FakeRunner([_ok({"echo": "hello"})]), []

    def business_effect(_event, payload):
        effects.append(payload["event_id"])

    hooks.register_listener("step.completed", business_effect, required=True)
    original_store = repo.store_step_output
    original_checkpoint = HookDeliveryRepository.store_checkpoint
    original_claim = HookDeliveryRepository.claim
    original_finish = HookDeliveryRepository.finish
    original_update = repo.update_step

    def store(*args, **kwargs):
        if window == "before_output":
            raise Crash()
        result = original_store(*args, **kwargs)
        if window == "after_output":
            raise Crash()
        return result

    def checkpoint(self, identity, *args, **kwargs):
        result = original_checkpoint(self, identity, *args, **kwargs)
        if window == "after_review" and identity.startswith("step:"):
            raise Crash()
        return result

    def claim(self, *args, **kwargs):
        if window == "before_claim":
            raise Crash()
        return original_claim(self, *args, **kwargs)

    def finish(self, *args, **kwargs):
        if window == "after_effect":
            raise Crash()
        result = original_finish(self, *args, **kwargs)
        if window == "after_receipt":
            raise Crash()
        return result

    def update(step):
        if window == "before_done" and step.status == StepStatus.DONE:
            raise Crash()
        return original_update(step)

    with monkeypatch.context() as patch:
        patch.setattr(repo, "store_step_output", store)
        patch.setattr(HookDeliveryRepository, "store_checkpoint", checkpoint)
        patch.setattr(HookDeliveryRepository, "claim", claim)
        patch.setattr(HookDeliveryRepository, "finish", finish)
        patch.setattr(repo, "update_step", update)
        with pytest.raises(Crash):
            _executor(repo, runner, reviewer=reviewer, hooks=hooks).run("plan-1")
    restored = PlanRepository(repo.db_path)
    recovered_hooks = _hooks(restored)
    recovered_hooks.register_listener("step.completed", business_effect, required=True)
    recovered = restored.load_plan("plan-1")
    PlanStepRecovery(
        restored, reviewer, recovered_hooks, HarnessState(restored)
    ).recover_inflight_steps(recovered)
    step = restored.load_plan("plan-1").steps[0]
    assert (step.status == StepStatus.DONE) is expected_done
    assert len(effects) == effect_count
    assert len(runner.calls) == 1
    assert reviewer.checks <= 1 and reviewer.critiques <= 1
    if window == "after_effect":
        assert "explicit reconciliation required" in step.error
        assert restored.list_step_runs(step.id)[0]["status"] == "succeeded"
    # A second startup cannot deliver again or replay the tool.
    PlanStepRecovery(
        restored, reviewer, recovered_hooks, HarnessState(restored)
    ).recover_inflight_steps(restored.load_plan("plan-1"))
    assert len(effects) == effect_count


def test_required_hook_failure_cannot_be_skipped_or_replanned(tmp_path):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    reviewer, hooks = CountingReviewer(), _hooks(repo)
    runner = FakeRunner([_ok({"echo": "hello"})], policies={"echo": "skip"})

    def broken(_event, _payload):
        raise RuntimeError("effect result unknown")

    hooks.register_listener("step.completed", broken, required=True)
    executor = _executor(repo, runner, reviewer=reviewer, hooks=hooks)
    executor._planner = object()  # Replanning this failure must never call it.
    result = executor.run("plan-1")
    assert result.status == PlanStatus.FAILED
    step = repo.load_plan("plan-1").steps[0]
    assert step.status == StepStatus.FAILED
    assert "required hook completion failed" in step.error
    assert repo.list_step_runs(step.id)[0]["status"] == "succeeded"
    assert len(runner.calls) == 1


def test_optional_hook_failure_is_a_visible_step_warning(tmp_path):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    hooks = _hooks(repo)

    def broken(_event, _payload):
        raise RuntimeError("optional failure")

    hooks.register_listener("step.completed", broken)
    result = _executor(
        repo,
        FakeRunner([_ok({"echo": "hello"})]),
        reviewer=CountingReviewer(),
        hooks=hooks,
    ).run("plan-1")
    assert result.status == PlanStatus.DONE
    assert (
        repo.load_plan("plan-1").steps[0].review_verdicts[-1].reviewer
        == "optional_hooks"
    )


@pytest.mark.parametrize(
    "window,expected_status",
    [("after_effect", PlanStatus.FAILED), ("before_done", PlanStatus.DONE)],
)
def test_startup_reconciles_workflow_completion_window(
    tmp_path, monkeypatch, window, expected_status
):
    repo, tasks, _task_id = _repo_with_agent_task(tmp_path, _plan(_step("step-1")))
    hooks, effects = _hooks(repo), []

    def business_effect(_event, payload):
        assert repo.load_plan("plan-1").status == PlanStatus.REVIEW
        effects.append(payload["event_id"])
        if window == "after_effect":
            raise Crash()

    hooks.register_listener("workflow.completed", business_effect, required=True)
    original_status = repo.set_plan_status

    def status(plan_id, value):
        if window == "before_done" and value == PlanStatus.DONE:
            raise Crash()
        return original_status(plan_id, value)

    with monkeypatch.context() as patch:
        patch.setattr(repo, "set_plan_status", status)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
                task_repo=tasks,
            ).run("plan-1")
    restored = PlanRepository(repo.db_path)
    recovered_hooks = _hooks(restored)
    recovered_hooks.register_listener(
        "workflow.completed", business_effect, required=True
    )
    assert (
        reclaim_running_plans(
            restored, CountingReviewer(), recovered_hooks, HarnessState(restored), tasks
        )
        == 1
    )
    assert restored.load_plan("plan-1").status == expected_status
    assert len(effects) == 1
    if window == "after_effect":
        notice = next(
            message for message in tasks.list_agent_messages(_task_id)
            if message["metadata"].get("plan_interrupted_by_restart")
        )
        assert notice["metadata"]["failure_envelope"]["retryable"] is False
        assert notice["metadata"]["error_diagnostic"]["auto_recoverable"] is False
        assert "不能直接重跑" in notice["content"]
    assert (
        reclaim_running_plans(
            restored, CountingReviewer(), recovered_hooks, HarnessState(restored), tasks
        )
        == 0
    )


@pytest.mark.parametrize(
    "effects", [("write:artifact",), ("network:optional",), ("process:spawn",), None]
)
@pytest.mark.parametrize("policy", ["retry", "skip"])
def test_unknown_tool_effect_never_retries_skips_or_replans(tmp_path, effects, policy):
    from types import SimpleNamespace
    from tests.test_orch_executor import _fail

    repo = _repo(tmp_path, _plan(_step("step-1")))
    runner = FakeRunner(
        [_fail("effect might have happened"), _ok({"echo": "must not run"})]
    )
    spec = SimpleNamespace(failure_policy=policy)
    if effects is not None:
        spec.side_effects = effects
    runner._tools = SimpleNamespace(resolve=lambda _ref: spec)
    executor = _executor(repo, runner, reviewer=CountingReviewer())
    executor._planner = object()
    result = executor.run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert len(runner.calls) == 1
    assert "explicit reconciliation required" in repo.load_plan("plan-1").steps[0].error
    assert repo.list_step_runs("step-1")[0]["error_kind"] == "unknown_effect"


def test_successful_tool_with_failed_output_persistence_cannot_replan(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    runner = FakeRunner([_ok({"echo": "finished"})])
    monkeypatch.setattr(
        repo,
        "store_step_output",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    executor = _executor(repo, runner, reviewer=CountingReviewer())
    executor._planner = object()
    result = executor.run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert len(runner.calls) == 1
    assert "explicit reconciliation required" in repo.load_plan("plan-1").steps[0].error


def test_output_schema_failure_does_not_prove_write_tool_had_no_effect(tmp_path):
    from types import SimpleNamespace
    from tests.test_orch_executor import _fail

    repo = _repo(tmp_path, _plan(_step("step-1")))
    failure = _fail("output did not match schema")
    failure.error_kind = "schema"
    runner = FakeRunner([failure, _ok({"echo": "must not run"})])
    runner._tools = SimpleNamespace(
        resolve=lambda _ref: SimpleNamespace(
            failure_policy="retry", side_effects=("write:artifact",)
        )
    )
    result = _executor(repo, runner, reviewer=CountingReviewer()).run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert len(runner.calls) == 1
    assert repo.list_step_runs("step-1")[0]["error_kind"] == "unknown_effect"


def test_recovery_without_dispatcher_cannot_drop_required_completion(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    hooks = _hooks(repo)
    effects = []

    def effect(_event, _payload):
        effects.append("ran")

    hooks.register_listener("step.completed", effect, required=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            HookDeliveryRepository,
            "claim",
            lambda *_a, **_k: (_ for _ in ()).throw(Crash()),
        )
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
            ).run("plan-1")
    restored = PlanRepository(repo.db_path)
    PlanStepRecovery(
        restored, CountingReviewer(), None, HarnessState(restored)
    ).recover_inflight_steps(restored.load_plan("plan-1"))
    step = restored.load_plan("plan-1").steps[0]
    assert step.status == StepStatus.FAILED
    assert "required hook dispatcher unavailable" in step.error
    assert effects == []


def test_changed_workflow_goal_cannot_reuse_completion_snapshot(tmp_path, monkeypatch):
    from marvis.db import connect

    repo, tasks, _task_id = _repo_with_agent_task(tmp_path, _plan(_step("step-1")))
    original = repo.set_plan_status

    def before_done(plan_id, status):
        if status == PlanStatus.DONE:
            raise Crash()
        return original(plan_id, status)

    with monkeypatch.context() as patch:
        patch.setattr(repo, "set_plan_status", before_done)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                task_repo=tasks,
            ).run("plan-1")
    with connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE plans SET goal = 'different business requirement' WHERE id = 'plan-1'"
        )
    assert (
        reclaim_running_plans(repo, CountingReviewer(), None, HarnessState(repo), tasks)
        == 1
    )
    plan = repo.load_plan("plan-1")
    assert plan.status == PlanStatus.FAILED
    assert "binding changed" in plan.loop_events[-1].reason


@pytest.mark.parametrize(
    "plugin,event",
    [
        ("_sample", "step.completed"),
        ("feature", "step.completed"),
        ("feature", "feature.computed"),
    ],
)
def test_required_recipients_frozen_before_review_checkpoint(
    tmp_path, monkeypatch, plugin, event
):
    repo = _repo(tmp_path, _plan(_step("step-1", plugin=plugin)))
    hooks = _hooks(repo)
    effects = []

    def required(_event, _payload):
        effects.append("ran")

    hooks.register_listener(event, required, required=True)
    original = HookDeliveryRepository.store_checkpoint

    def after_review(self, identity, *args, **kwargs):
        result = original(self, identity, *args, **kwargs)
        if identity.startswith("step:"):
            raise Crash()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(HookDeliveryRepository, "store_checkpoint", after_review)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
            ).run("plan-1")
    restored = PlanRepository(repo.db_path)
    # The required hook disappeared while the process was down.
    result = _executor(
        restored, FakeRunner([]), reviewer=CountingReviewer(), hooks=_hooks(restored)
    ).run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert (
        "required hook completion failed" in restored.load_plan("plan-1").steps[0].error
    )
    assert effects == []


def test_workflow_recipients_frozen_before_completion_checkpoint(tmp_path, monkeypatch):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    hooks = _hooks(repo)
    effects = []

    def required(_event, _payload):
        effects.append("ran")

    hooks.register_listener("workflow.completed", required, required=True)
    original = HookDeliveryRepository.store_checkpoint

    def after_checkpoint(self, identity, *args, **kwargs):
        result = original(self, identity, *args, **kwargs)
        if identity.startswith("workflow:"):
            raise Crash()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(HookDeliveryRepository, "store_checkpoint", after_checkpoint)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
            ).run("plan-1")
    restored = PlanRepository(repo.db_path)
    result = _executor(
        restored, FakeRunner([]), reviewer=CountingReviewer(), hooks=_hooks(restored)
    ).run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert effects == []
    assert "required hook" in restored.load_plan("plan-1").loop_events[-1].reason


def test_optional_workflow_warning_failure_cannot_rewrite_terminal_state(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    hooks = _hooks(repo)

    def optional(_event, _payload):
        raise RuntimeError("optional hook failed")

    hooks.register_listener("workflow.completed", optional)
    original = repo.append_loop_event

    def warning_failure(plan_id, event):
        if event["type"] == "hook_warning":
            raise OSError("warning persistence unavailable")
        return original(plan_id, event)

    monkeypatch.setattr(repo, "append_loop_event", warning_failure)
    result = _executor(
        repo,
        FakeRunner([_ok({"echo": "hello"})]),
        reviewer=CountingReviewer(),
        hooks=hooks,
    ).run("plan-1")
    # Infrastructure persistence failed before terminal state; the API returns a
    # consistent failure instead of raising an illegal DONE -> FAILED transition.
    assert result.status == repo.load_plan("plan-1").status == PlanStatus.FAILED


def test_workflow_prepare_before_checkpoint_cannot_abandon_obligation(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, _plan(_step("step-1")))
    hooks, calls = _hooks(repo), []

    def required(_event, _payload):
        calls.append("ran")

    hooks.register_listener("workflow.completed", required, required=True)
    original = HookDeliveryRepository.store_checkpoint

    def before_checkpoint(self, identity, *args, **kwargs):
        if identity.startswith("workflow:"):
            raise Crash()
        return original(self, identity, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(HookDeliveryRepository, "store_checkpoint", before_checkpoint)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
            ).run("plan-1")
    result = _executor(
        repo, FakeRunner([]), reviewer=CountingReviewer(), hooks=_hooks(repo)
    ).run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert calls == []


def test_feature_obligations_freeze_atomically_before_checkpoint(tmp_path, monkeypatch):
    repo = _repo(tmp_path, _plan(_step("step-1", plugin="feature")))
    hooks, calls = _hooks(repo), []

    def required(_event, _payload):
        calls.append("ran")

    hooks.register_listener("step.completed", required, required=True)
    original = HookDeliveryRepository.prepare_events

    def after_prepare(self, events):
        result = original(self, events)
        if len(events) == 2:
            raise Crash()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(HookDeliveryRepository, "prepare_events", after_prepare)
        with pytest.raises(Crash):
            _executor(
                repo,
                FakeRunner([_ok({"echo": "hello"})]),
                reviewer=CountingReviewer(),
                hooks=hooks,
            ).run("plan-1")
    result = _executor(
        repo, FakeRunner([]), reviewer=CountingReviewer(), hooks=_hooks(repo)
    ).run("plan-1")
    assert result.status == PlanStatus.FAILED
    assert calls == []


@pytest.mark.parametrize('event', ['step.completed', 'workflow.completed'])
def test_driver_exposes_required_hook_reconciliation_without_retry_action(tmp_path, event):
    from marvis.agent.plan_driver import PlanDriver

    repo = _repo(tmp_path, _plan(_step('step-1')))
    hooks = _hooks(repo)

    def broken(_event, _payload):
        raise RuntimeError('external result unknown')

    hooks.register_listener(event, broken, required=True)
    runner = FakeRunner([_ok({'echo': 'hello'})])
    driver = PlanDriver(repo, _executor(repo, runner, reviewer=CountingReviewer(), hooks=hooks))
    turn = driver._run_and_handle('plan-1', run_seq=1)
    message = turn.messages[0]
    assert turn.status == PlanStatus.FAILED.value
    assert message.metadata['failure_envelope']['retryable'] is False
    diagnostic = message.metadata['error_diagnostic']
    assert diagnostic['retryable'] is False
    assert diagnostic['recovery_actions'] == []
    assert diagnostic['code'] == 'workflow_reconciliation_required'
    assert '核对' in message.content
    assert '是否由 Agent' not in message.content
    if event == 'workflow.completed':
        assert '完成动作' in message.content


def test_repository_cannot_clear_unknown_effect_via_replacement_inputs(tmp_path):
    from types import SimpleNamespace
    from marvis.state_machine import ConflictError
    from tests.test_orch_executor import _fail

    repo = _repo(tmp_path, _plan(_step('step-1')))
    runner = FakeRunner([_fail('uncertain')])
    runner._tools = SimpleNamespace(resolve=lambda _ref: SimpleNamespace(failure_policy='retry', side_effects=('write:artifact',)))
    assert _executor(repo, runner).run('plan-1').status == PlanStatus.FAILED
    step = repo.load_plan('plan-1').steps[0]
    # Even a generic display error cannot hide the machine-owned run status.
    step.error = 'generic message'
    repo.update_step(step)
    with pytest.raises(ConflictError, match='尚未核对'):
        repo.retry_failed_step('plan-1', 'step-1', inputs={'different': 'input'})
    assert repo.load_plan('plan-1').steps[0].status == StepStatus.FAILED
