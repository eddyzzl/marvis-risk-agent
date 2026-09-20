"""Uncertain effects must survive every user-facing retry/revision route."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from marvis.agent.plan_driver import DriverError, PlanDriver
from marvis.agent.workflow_error_diagnostics import enrich_workflow_error_diagnostic
from marvis.agent.workflow_recovery import deterministic_workflow_recovery_reply
from marvis.db import connect
from marvis.orchestrator.contracts import (
    PlanStatus,
    StepStatus,
    plan_fingerprint,
    plan_step_confirmation_fingerprint,
)
from marvis.orchestrator.harness_state import HarnessState
from marvis.recovery import _add_plan_restart_notice, reclaim_running_plans
from marvis.state_machine import ConflictError
from tests.test_orch_api import _client
from tests.test_orch_completion import CountingReviewer, _hooks
from tests.test_orch_executor import (
    FakeRunner,
    _executor,
    _ok,
    _plan,
    _repo_with_agent_task,
    _step,
)
from tests.test_workflow_feature_rollback import EXCLUSIONS, _persist_failed_plan


MARKER = "effect may have happened; explicit reconciliation required"


def _snapshot(repo):
    # Include immutable results, executions, hook receipts and audit history.
    with connect(repo.db_path) as conn:
        return {
            table: sorted(tuple(row) for row in conn.execute(f"SELECT * FROM {table}"))
            for table in (
                "plans", "plan_steps", "plan_step_runs", "plan_step_output_versions",
                "hook_completion_checkpoints", "hook_events", "hook_deliveries", "audit",
            )
        }


def _block(repo, step_id, *, source):
    plan = repo.load_plan("plan-rollback")
    step = next(step for step in plan.steps if step.id == step_id)
    if source == "run":
        step.status = StepStatus.RUNNING
        repo.update_step(step)
        run_id = repo.start_step_run(
            plan_id=plan.id, step_id=step_id,
            tool_ref=step.tool_ref.label(), inputs=step.inputs,
        )
        repo.finish_step_run(run_id, status="failed", error="ordinary display error", error_kind="unknown_effect")
    step.status = StepStatus.FAILED
    step.error = MARKER if source == "marker" else "ordinary display error"
    repo.update_step(step)


def _adjust(repo, plan, ids):
    target = next(step for step in plan.steps if step.id == ids[0])
    repo.apply_gate_adjustment(
        plan.id, target_step_id=target.id, reset_step_ids=ids,
        replacement_inputs_by_step={target.id: {"changed": True}},
        expected_plan_status=plan.status, expected_plan_revision=plan.replan_count,
        expected_plan_fingerprint=plan_fingerprint(plan),
        expected_target_step_fingerprint=plan_step_confirmation_fingerprint(
            target, confirmed=repo.is_step_confirmed(target.id),
        ),
    )


@pytest.mark.parametrize("source", ["marker", "run"])
@pytest.mark.parametrize("operation", ["reset", "adjust", "retry_upstream", "rollback", "replace"])
def test_all_reset_routes_preserve_uncertain_effects_and_receipts(tmp_path, source, operation):
    repo, _ = _persist_failed_plan(tmp_path)
    _block(repo, "tune", source=source)
    if operation == "retry_upstream":
        upstream = repo.load_plan("plan-rollback").steps[2]
        upstream.status, upstream.error = StepStatus.FAILED, "safe failure"
        repo.update_step(upstream)
    plan = repo.load_plan("plan-rollback")
    before = _snapshot(repo)
    with pytest.raises(ConflictError, match="尚未核对"):
        if operation == "reset":
            repo.reset_step("tune", inputs={"changed": True})
        elif operation == "adjust":
            _adjust(repo, plan, ["screen", "select", "configure", "tune", "train"])
        elif operation == "retry_upstream":
            repo.retry_failed_step(plan.id, "screen", inputs={"changed": True})
        elif operation == "rollback":
            repo.rollback_failed_plan_from_step(
                plan.id, "screen", "tune", root_inputs={"features": ["safe_a"]},
                excluded_features=EXCLUSIONS, expected_plan_revision=plan.replan_count,
                expected_root_output_ref="metrics:screen:v1",
            )
        else:
            repo.replace_remaining_steps(plan.id, plan)
    assert _snapshot(repo) == before


@pytest.mark.parametrize("revision", ["features", "budget"])
def test_real_driver_upstream_revision_cannot_bypass_unknown_effect(tmp_path, revision):
    repo, plan = _persist_failed_plan(tmp_path)
    _block(repo, "tune", source="run")

    class NeverRun:
        def run(self, _plan_id):
            pytest.fail("unreconciled action must never reach executor")

    driver = PlanDriver(repo, NeverRun())
    before = _snapshot(repo)
    with pytest.raises(DriverError, match="尚未核对"):
        if revision == "features":
            driver.rollback_failed_plan_to_feature_screen(plan.id, "tune", excluded_features=EXCLUSIONS, run_seq=4)
        else:
            driver.rollback_failed_plan_to_tuning_config(
                plan.id, "tune", default_n_trials=1, n_trials_by_recipe={}, run_seq=4,
            )
    assert _snapshot(repo) == before


@pytest.mark.parametrize("operation", ["reset", "adjust", "replace", "append"])
def test_workflow_completion_effect_blocks_done_output_invalidation_and_new_steps(tmp_path, operation):
    repo, _tasks, _task_id = _repo_with_agent_task(tmp_path, _plan(_step("step-1")))
    hooks, effects = _hooks(repo), []

    def unknown_effect(_event, payload):
        effects.append(payload["event_id"])
        raise RuntimeError("lost receipt after applying effect")

    hooks.register_listener("workflow.completed", unknown_effect, required=True)
    _executor(repo, FakeRunner([_ok({"echo": "hello"})]), reviewer=CountingReviewer(), hooks=hooks).run("plan-1")
    plan = repo.load_plan("plan-1")
    assert plan.status == PlanStatus.FAILED
    assert plan.steps[0].status == StepStatus.DONE
    assert len(effects) == 1
    before = _snapshot(repo)
    with pytest.raises(ConflictError, match="尚未核对"):
        if operation == "reset":
            repo.reset_step("step-1", inputs={"changed": True})
        elif operation == "adjust":
            _adjust(repo, plan, ["step-1"])
        elif operation == "replace":
            # No remaining steps means the step-level closure is empty.
            plan.steps.append(_step("new-step", index=1))
            repo.replace_remaining_steps(plan.id, plan)
        else:
            repo.append_steps(plan.id, [_step("new-step", index=1)])
    assert _snapshot(repo) == before


def test_reset_waits_for_writer_and_observes_newly_published_stop(tmp_path, monkeypatch):
    repo, _ = _persist_failed_plan(tmp_path)
    entered = Event()

    @contextmanager
    def observed_connect(path):
        with connect(path) as conn:
            conn.set_trace_callback(lambda sql: entered.set() if sql == "BEGIN IMMEDIATE" else None)
            yield conn

    monkeypatch.setattr("marvis.repositories.plans.connect", observed_connect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with connect(repo.db_path) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE plan_steps SET error = ? WHERE id = 'tune'", (MARKER,))
            pending = pool.submit(repo.reset_step, "tune", inputs={"changed": True})
            assert entered.wait(5)
            assert not pending.done()
        with pytest.raises(ConflictError, match="尚未核对"):
            pending.result(timeout=5)
    assert repo.load_plan("plan-rollback").steps[5].error == MARKER
    assert repo.list_audit(kind="plan.step.reset") == []


@pytest.mark.parametrize("source", ["marker", "run", "downstream"])
def test_get_and_retry_endpoint_share_persisted_reconciliation_policy(tmp_path, source):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    first = _step("first", status=StepStatus.FAILED, inputs={"editable": "value"})
    last = _step("last", index=1, depends_on=[first.id], status=StepStatus.FAILED)
    first.error = "safe display error"
    last.error = MARKER if source == "downstream" else "safe display error"
    if source == "marker":
        first.error = MARKER
    plan = _plan(first, last, status=PlanStatus.FAILED)
    repo.create_plan(plan)
    if source == "run":
        first.status = StepStatus.RUNNING
        repo.update_step(first)
        run_id = repo.start_step_run(plan_id=plan.id, step_id=first.id, tool_ref=first.tool_ref.label(), inputs=first.inputs)
        repo.finish_step_run(run_id, status="failed", error_kind="unknown_effect")
        first.status = StepStatus.FAILED
        repo.update_step(first)
    envelope = client.get("/api/plans/plan-1").json()["plan"]["steps"][0]["failure_envelope"]
    assert envelope["retryable"] is False
    assert envelope["editable_input_schema"]["properties"] == {}
    assert envelope["downstream_reset_steps"] == []
    assert envelope["suggested_actions"] == ["halt"]
    response = client.post("/api/plans/plan-1/steps/first/retry", json={"inputs": {"changed": True}})
    assert response.status_code == 409
    assert client.app.state.plan_executor.calls == []
    assert repo.load_plan(plan.id).steps[0].inputs == {"editable": "value"}


@pytest.mark.parametrize("source", ["code", "marker", "kind"])
def test_diagnostic_upgrade_never_turns_unresolved_effect_into_retry(source):
    detail = "No match for FieldRef.Name(C0) in : int64"
    diagnostic = {"code": "workflow_step_failed", "technical_detail": detail, "retryable": True}
    if source == "code":
        diagnostic["code"] = "workflow_reconciliation_required"
    elif source == "marker":
        diagnostic["cause"] = MARKER
    else:
        diagnostic["error_kind"] = "unknown_effect"
    result = enrich_workflow_error_diagnostic(diagnostic)
    assert result["code"] == "workflow_reconciliation_required"
    assert result["retryable"] is False
    assert result["auto_recoverable"] is False
    assert result["recovery_actions"] == []
    assert result["technical_detail"] == detail
    reply = deterministic_workflow_recovery_reply(result)
    assert "请先核对动作的实际结果" in reply
    assert "需要先调整输入或重新建立计划" not in reply


def test_startup_notice_keeps_machine_unknown_effect_nonretryable(tmp_path):
    step = _step("step-1", status=StepStatus.RUNNING)
    repo, tasks, task_id = _repo_with_agent_task(tmp_path, _plan(step, status=PlanStatus.RUNNING))
    run_id = repo.start_step_run(plan_id="plan-1", step_id=step.id, tool_ref=step.tool_ref.label(), inputs={})
    repo.finish_step_run(run_id, status="failed", error_kind="unknown_effect")
    step.status, step.error = StepStatus.FAILED, "display error without marker"
    repo.update_step(step)
    assert reclaim_running_plans(repo, CountingReviewer(), _hooks(repo), HarnessState(repo), tasks) == 1
    notice = next(message for message in tasks.list_agent_messages(task_id) if message["metadata"].get("plan_interrupted_by_restart"))
    assert notice["metadata"]["failure_envelope"]["retryable"] is False
    diagnostic = notice["metadata"]["error_diagnostic"]
    assert diagnostic["auto_recoverable"] is False
    assert diagnostic["recovery_actions"] == []
    assert "请回复“重试" not in notice["content"]
    assert "不能直接重跑" in notice["content"]


def test_get_exposes_workflow_completion_stop_even_when_all_steps_are_done(tmp_path):
    client = _client(tmp_path)
    repo = client.app.state.plan_repo
    repo.create_plan(_plan(_step("step-1")))
    hooks = _hooks(repo)

    def effect(_event, _payload):
        raise RuntimeError("unknown workflow effect")

    hooks.register_listener("workflow.completed", effect, required=True)
    _executor(repo, FakeRunner([_ok({"echo": "hello"})]), reviewer=CountingReviewer(), hooks=hooks).run("plan-1")
    payload = client.get("/api/plans/plan-1").json()["plan"]
    assert payload["status"] == "failed"
    assert payload["steps"][0]["status"] == "done"
    assert payload["failure_envelope"]["retryable"] is False
    assert payload["failure_envelope"]["failed_step_id"] is None
    assert payload["failure_envelope"]["suggested_actions"] == ["halt"]


def test_composer_checks_downstream_receipt_before_suggesting_upstream_retry(tmp_path):
    repo, _ = _persist_failed_plan(tmp_path)
    _block(repo, "tune", source="run")
    plan = repo.load_plan("plan-rollback")
    plan.steps[2].status, plan.steps[2].error = StepStatus.FAILED, "safe failure"
    repo.update_step(plan.steps[2])
    message = PlanDriver(repo, None)._composer.failed_message(repo.load_plan(plan.id), run_seq=1)
    assert message.metadata["failure_envelope"]["retryable"] is False
    assert message.metadata["failure_envelope"]["suggested_actions"] == ["halt"]
    assert message.metadata["error_diagnostic"]["auto_recoverable"] is False


def test_later_unknown_effect_publishes_stop_over_prior_safe_restart_notice(tmp_path):
    step = _step("step-1", status=StepStatus.FAILED)
    repo, tasks, task_id = _repo_with_agent_task(tmp_path, _plan(step, status=PlanStatus.FAILED))
    _add_plan_restart_notice(tasks, task_id, "plan-1", run_mode="agent", failed_step=step)
    step.error = MARKER
    _add_plan_restart_notice(tasks, task_id, "plan-1", run_mode="agent", failed_step=step)
    _add_plan_restart_notice(tasks, task_id, "plan-1", run_mode="agent", failed_step=step)
    notices = [message for message in tasks.list_agent_messages(task_id) if message["metadata"].get("plan_interrupted_by_restart")]
    assert len(notices) == 2
    assert notices[0]["metadata"]["failure_envelope"]["retryable"] is True
    assert notices[-1]["metadata"]["failure_envelope"]["retryable"] is False
    assert "不能直接重跑" in notices[-1]["content"]
