"""Actual search control records determine the public header and retry policy."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from marvis.orchestrator.contracts import StepStatus
from marvis.strategy_workbench.control import NativeSearchControl, SearchInterrupted
from tests.test_strategy_workbench_state import _action, _cross, _prepare


@pytest.mark.parametrize("action,kind,halt,label", [
    ("pause", "search_paused", "paused", "已暂停"),
    ("cancel", "search_cancelled", "cancelled", "已取消"),
])
def test_native_halt_public_header_and_generic_retry_remain_consistent(tmp_path, action, kind, halt, label):
    fixture, _, client, actor, inputs = _cross(tmp_path)
    repo, plan, step, run_id = _prepare(fixture, client, actor, inputs)
    control = NativeSearchControl(fixture["runtime"], replace(fixture["ctx"], invocation_id=run_id), inputs)
    control.bind_search("native-header-check")
    control.boundary([], 2)
    assert _action(client, fixture["task"].id, step.id, action).status_code == 202
    with pytest.raises(SearchInterrupted):
        control.boundary([{"measured": 1}], 2)
    repo.finish_step_run(run_id, status="failed", error_kind=kind, error="native search interrupted")
    step.status, step.error = StepStatus.FAILED, "native search interrupted"
    repo.update_step(step)
    response = client.get(f"/api/plans/{plan.id}")
    assert response.status_code == 200
    payload = response.json()["plan"]
    failed = next(s for s in payload["steps"] if s["id"] == step.id)
    assert failed["execution_halt"] == halt
    assert failed["failure_envelope"]["retryable"] is False
    assert failed["failure_envelope"]["suggested_actions"] == ["halt"]
    script = """
import {planWorkflowStatus} from './marvis/static/js/v2/plan_rail_controller.js';
let text='';for await (const chunk of process.stdin) text+=chunk;
process.stdout.write(JSON.stringify(planWorkflowStatus(JSON.parse(text))));
"""
    rendered = subprocess.run(["node", "--input-type=module", "-e", script],
                              input=json.dumps(payload), text=True, capture_output=True,
                              cwd=Path(__file__).resolve().parents[1], check=True)
    assert json.loads(rendered.stdout)["label"] == label


def test_failure_prose_does_not_establish_native_pause():
    from marvis.routers.plans import _attach_failure_envelopes
    payload = {"id": "plan", "steps": [{"id": "step", "status": "failed", "error": "search_paused 已暂停"}]}
    _attach_failure_envelopes(payload)
    assert "execution_halt" not in payload["steps"][0]
