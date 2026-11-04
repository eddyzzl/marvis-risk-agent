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
// Batch 4 (2026-10-14 redesign): the presentation also carries key/face/dot/checker; compare the word and tone.
const word = (presentation) => ({ label: presentation.label, tone: presentation.tone });
const snapshot = planWorkflowStatus(plans[0]);
assert.deepEqual(word(actionStatusPresentation(snapshot.message, snapshot.kind, {
  task, usesPlanWorkflow: true, workflowSnapshot: snapshot,
})), { label: "待核对", tone: "fail" });
// Batch 4 (2026-10-14 redesign): one failure word everywhere; stage wording stays in the detail line.
assert.deepEqual(word(actionStatusPresentation("请求失败", "error", {
  task, usesPlanWorkflow: true,
})), { label: "失败", tone: "fail" });
// Batch 4 (2026-10-14 redesign): 验证失败 -> the canonical 失败.
assert.deepEqual(word(actionStatusPresentation("复核后仍失败", "error", {
  task: { status: "failed", task_type: "validation" },
})), { label: "失败", tone: "fail" });
// Batch 4 (2026-10-14 redesign): 停止 -> the canonical 已停止.
assert.deepEqual(word(actionStatusPresentation("停止", "stopped", {
  task, stopped: true, usesPlanWorkflow: true, workflowSnapshot: snapshot,
})), { label: "已停止", tone: "neutral" });
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
