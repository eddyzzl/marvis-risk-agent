"""Real native API/ledger results remain distinct from fault injection success."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

from fastapi.testclient import TestClient
import pytest

from marvis.orchestrator.eval.runtime_fault_launcher import (
    arm_process_fault, attached_launch, evaluation_fault_launcher,
)
from marvis.orchestrator.eval.runtime_process_faults import write_original
from test_event_runtime import runtime, prepare, gated_plan, approve


@pytest.mark.e2e
@pytest.mark.parametrize("kind", ["cancel_then_resume_once", "tool_timeout_once"])
def test_actual_plan_cancel_or_timeout_keeps_unknown_effect_blocked_on_real_retry(tmp_path, kind):
    rt = runtime.__wrapped__(tmp_path)
    plan = gated_plan(rt, prepare(rt))
    step_id = plan["steps"][0]["id"]
    directory = arm_process_fault(rt.settings.workspace, kind=kind, task_id=rt.task.id,
                                  plan_id=plan["id"], step_id=step_id, tool="risk_context.replay_events")
    with evaluation_fault_launcher(rt.settings.workspace, origin_ns=time.monotonic_ns()):
        with ThreadPoolExecutor() as pool:
            future = pool.submit(approve, rt, plan)
            launch = attached_launch(directory, deadline=time.monotonic() + 15)
            ready = launch.ready()
            assert ready["phase"] == "before_tool"
            if kind == "cancel_then_resume_once":
                control = TestClient(rt.app)
                control.cookies.update(rt.maker.cookies)
                stopped = control.post(f"/api/plans/{plan['id']}/cancel")
                assert stopped.status_code == 200, stopped.text
                assert stopped.json()["plan"]["status"] == "cancelled"
                write_original(launch.directory / "release", {"reason": "actual_plan_cancel_accepted"})
            settled = future.result(timeout=20)
        before = rt.app.state.plan_repo.list_step_runs(step_id)
        assert len(before) == 1
        assert settled["status"] in {"failed", "cancelled"}
        retry = rt.maker.post(f"/api/plans/{plan['id']}/steps/{step_id}/retry", json={})
        assert retry.status_code == 409, retry.text
        assert rt.app.state.plan_repo.list_step_runs(step_id) == before
    terminal = json.loads((launch.directory / "terminal.json").read_text())
    assert terminal["terminal"] == ("timeout" if kind == "tool_timeout_once" else "returned")
    if kind == "tool_timeout_once":
        assert terminal["observed_timeout_seconds"] == 2
        assert terminal["finished_elapsed_ns"] - terminal["started_elapsed_ns"] >= 2_000_000_000
        assert not (launch.directory / "worker-output.json").exists()


@pytest.mark.e2e
def test_wrong_step_binding_does_not_install_fault_in_other_native_invocation(tmp_path):
    rt = runtime.__wrapped__(tmp_path)
    plan = gated_plan(rt, prepare(rt))
    directory = arm_process_fault(rt.settings.workspace, kind="tool_timeout_once", task_id=rt.task.id,
                                  plan_id=plan["id"], step_id="unrelated-step", tool="risk_context.replay_events")
    with evaluation_fault_launcher(rt.settings.workspace, origin_ns=time.monotonic_ns()):
        settled = approve(rt, plan)
    assert settled["status"] == "done"
    assert not (directory / "claimed.json").exists()
    assert not (directory / "launch").exists()
