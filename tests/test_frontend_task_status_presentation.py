"""Header and rail consume the same structured execution status."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_header_uses_real_plan_status_even_when_action_copy_disagrees():
    script = r'''
import assert from "node:assert/strict";
import { actionStatusPresentation } from "./marvis/static/js/task-status.js";
import { planWorkflowStatus } from "./marvis/static/js/v2/plan_rail_controller.js";

const task = { id: "join", task_type: "data_join", status: "created" };
const plans = [
  { status: "failed", steps: [{ status: "failed", failure_envelope: { retryable: false } }] },
  { status: "failed", steps: [{ status: "failed", failure_envelope: { retryable: true } }] },
  { status: "awaiting_confirm", steps: [] },
  { status: "running", steps: [] },
  { status: "done", steps: [] },
  { status: "review", steps: [] },
];
for (const plan of plans) {
  const snapshot = planWorkflowStatus(plan);
  for (const kind of ["info", "error", "success", "busy", "stopped"]) {
    const headline = actionStatusPresentation("操作已完成，也可能需复核", kind, {
      task, usesPlanWorkflow: true, workflowSnapshot: snapshot,
    });
    assert.equal(headline.label, snapshot.label, `${plan.status}/${kind}`);
  }
}
const snapshot = planWorkflowStatus(plans[0]);
assert.deepEqual(actionStatusPresentation(snapshot.message, snapshot.kind, {
  task, usesPlanWorkflow: true, workflowSnapshot: snapshot,
}), { label: "待核对", tone: "fail" });
assert.deepEqual(actionStatusPresentation("请求失败", "error", {
  task, usesPlanWorkflow: true,
}), { label: "执行失败", tone: "fail" });
assert.deepEqual(actionStatusPresentation("复核后仍失败", "error", {
  task: { status: "failed", task_type: "validation" },
}), { label: "验证失败", tone: "fail" });
assert.deepEqual(actionStatusPresentation("停止", "stopped", {
  task, stopped: true, usesPlanWorkflow: true, workflowSnapshot: snapshot,
}), { label: "停止", tone: "neutral" });
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
