"""Exercise reconciliation safeguards through the imported plan controller."""

from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
import assert from "node:assert/strict";
import { createPlanRailController } from "./marvis/static/js/v2/plan_rail_controller.js";

function element() {
  const classes = new Set();
  return {
    innerHTML: "", dataset: {}, attributes: {},
    classList: {
      add(name) { classes.add(name); },
      remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
}
function failedStep(id, retryable) {
  return {
    id, title: `步骤 ${id}`, index: 1, status: "failed", phase: "执行",
    tool_ref: { plugin: id, tool: "train_model" },
    inputs: { limit: 3 },
    failure_envelope: {
      ...(retryable === undefined ? {} : { retryable }),
      editable_input_schema: {
        type: "object", properties: { limit: { type: "integer", default: 3 } },
      },
      downstream_reset_steps: [],
    },
  };
}
async function harness(steps, { agentMode = false, apiHandler = null, viewCallbacks = {} } = {}) {
  let plan = { id: "plan-1", status: "failed", steps };
  const elements = Object.fromEntries(
    ["progressRail", "workflowStepper", "planRetryPanel", "planDriverActions"]
      .map((id) => [id, element()]),
  );
  const requests = [], schemaRequests = [], statuses = [], busy = [];
  globalThis.document = { querySelector: () => null };
  globalThis.window = { setTimeout() {} };
  globalThis.fetch = async (url) => {
    assert.equal(url, "/api/tasks/task-1/plans");
    return { ok: true, json: async () => ({ plans: [structuredClone(plan)] }) };
  };
  const controller = createPlanRailController({
    ...viewCallbacks,
    $: (id) => elements[id] || null,
    getSelectedTask: () => ({ id: "task-1", task_type: "modeling" }),
    getSelectedTaskId: () => "task-1",
    getAgentMessages: () => [],
    isAgentMode: () => agentMode,
    setActionStatus: (message, kind) => statuses.push({ message, kind }),
    setDriverExecutionBusy: (value) => busy.push(value),
    apiClient: async (url, options) => {
      requests.push({ url, method: options.method, body: JSON.parse(options.body) });
      return apiHandler ? await apiHandler(url, options) : {};
    },
    listPluginToolsClient: async (name) => {
      schemaRequests.push(name);
      return { tools: [] };
    },
  });
  const renderSignatures = {};
  async function refresh(nextPlan = plan) {
    plan = structuredClone(nextPlan);
    controller.resetFetchThrottle();
    await controller.maybeFetchPlan();
    controller.render({ renderSignatures });
    await new Promise((resolve) => setImmediate(resolve));
  }
  async function click(stepId, { poisonForm = false } = {}) {
    const button = {
      dataset: { planRetryStep: stepId }, disabled: false,
      closest() {
        assert.equal(poisonForm, false, "blocked retry must not read editable inputs");
        return {
          querySelectorAll: () => [],
          querySelector: () => ({ value: '{"limit":5}', defaultValue: '{"limit":3}' }),
        };
      },
    };
    let prevented = false, stopped = false;
    assert.equal(controller.handleClick({
      target: { closest: (selector) => selector === "[data-plan-retry-step]" ? button : null },
      preventDefault() { prevented = true; },
      stopPropagation() { stopped = true; },
    }), true);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(prevented, true);
    assert.equal(stopped, true);
    return button;
  }
  await refresh();
  async function reconcile(targetId) {
    const button = { dataset: { planReconcile: targetId }, disabled: false };
    assert.equal(controller.handleClick({
      target: { closest: (selector) => selector === "[data-plan-reconcile]" ? button : null },
      preventDefault() {}, stopPropagation() {},
    }), true);
    await new Promise((resolve) => setImmediate(resolve));
    return button;
  }
  async function continuePlan(fingerprint) {
    const button = { dataset: { planContinue: fingerprint }, disabled: false };
    assert.equal(controller.handleClick({
      target: { closest: (selector) => selector === "[data-plan-continue]" ? button : null },
      preventDefault() {}, stopPropagation() {},
    }), true);
    await new Promise((resolve) => setImmediate(resolve));
    return button;
  }
  return { controller, elements, requests, schemaRequests, statuses, busy, refresh, click, reconcile, continuePlan };
}
'''


def run_controller_test(body: str) -> None:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", HARNESS + body],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_nonretryable_failure_renders_read_only_and_rejects_synthetic_action():
    run_controller_test(r'''
const step = failedStep("unknown-result", false);
step.title = '<img src=x onerror="unsafe()">';
const h = await harness([step]);
const panel = h.elements.planRetryPanel;
assert.equal(panel.attributes["aria-hidden"], "false");
assert.match(panel.innerHTML, /执行结果待核对/);
assert.match(panel.innerHTML, /已暂停重试/);
assert.match(panel.innerHTML, /data-plan-reconciliation-step="unknown-result"/);
assert.match(panel.innerHTML, /&lt;img/);
assert.doesNotMatch(panel.innerHTML, /<(?:input|select|textarea|button|img)\b/);
assert.doesNotMatch(panel.innerHTML, /data-plan-retry-step=/);
assert.deepEqual(h.schemaRequests, []);
assert.equal(h.controller.statusSnapshot().label, "待核对");
assert.match(h.elements.workflowStepper.innerHTML, /执行结果待核对/);
await h.click("unknown-result", { poisonForm: true });
assert.deepEqual(h.requests, []);
assert.deepEqual(h.busy, []);
assert.match(h.statuses.at(-1).message, /已暂停重试/);
''')


def test_mixed_failures_only_offer_and_execute_safe_step_retry():
    run_controller_test(r'''
const h = await harness([failedStep("unknown-result", false), failedStep("safe", true)]);
const html = h.elements.planRetryPanel.innerHTML;
assert.match(html, /处理失败步骤/);
assert.match(html, /data-plan-reconciliation-step="unknown-result"/);
assert.doesNotMatch(html, /data-plan-retry-step="unknown-result"/);
assert.match(html, /data-plan-retry-step="safe"/);
assert.deepEqual(h.schemaRequests, ["safe"]);
await h.click("unknown-result", { poisonForm: true });
assert.deepEqual(h.requests, []);
await h.click("safe");
assert.deepEqual(h.requests, [{
  url: "/api/plans/plan-1/steps/safe/retry", method: "POST", body: { inputs: { limit: 5 } },
}]);
''')


def test_policy_refresh_removes_old_form_and_rejects_its_stale_button():
    run_controller_test(r'''
const step = failedStep("step-1", true);
const h = await harness([step]);
const panel = h.elements.planRetryPanel;
assert.match(panel.innerHTML, /data-plan-retry-step="step-1"/);
const oldSignature = panel.dataset.planRetrySignature;
step.failure_envelope.retryable = false;
await h.refresh({ id: "plan-1", status: "failed", steps: [step] });
assert.notEqual(panel.dataset.planRetrySignature, oldSignature);
assert.doesNotMatch(panel.innerHTML, /<(?:input|select|textarea|button)\b/);
await h.click("step-1", { poisonForm: true });
assert.deepEqual(h.requests, []);
assert.deepEqual(h.busy, []);
step.failure_envelope.retryable = true;
await h.refresh({ id: "plan-1", status: "failed", steps: [step] });
assert.match(panel.innerHTML, /data-plan-retry-step="step-1"/);
assert.equal(h.controller.statusSnapshot().label, "失败");
''')


@pytest.mark.parametrize("retryable", ["undefined", "null", "true"])
def test_legacy_and_retryable_failures_remain_editable(retryable: str):
    run_controller_test(r'''
const step = failedStep("legacy", RETRYABLE);
// Narrative wording cannot turn a legacy failure into a blocked operation.
step.failure_envelope.message = "needs reconciliation; 执行结果待核对";
const h = await harness([step]);
assert.match(h.elements.planRetryPanel.innerHTML, /data-plan-retry-step="legacy"/);
assert.equal(h.controller.statusSnapshot().label, "失败");
await h.click("legacy");
assert.equal(h.requests.length, 1);
assert.deepEqual(h.requests[0].body.inputs, { limit: 5 });
'''.replace("RETRYABLE", retryable))


@pytest.mark.parametrize("status", ["done", "pending", "running", "missing"])
def test_stale_or_unknown_step_cannot_submit_retry(status: str):
    run_controller_test(r'''
const step = failedStep("step-1", true);
step.status = "STATUS";
const h = await harness(step.status === "missing" ? [] : [step]);
await h.click("step-1", { poisonForm: true });
assert.deepEqual(h.requests, []);
assert.deepEqual(h.busy, []);
assert.match(h.statuses.at(-1).message, /已不处于可重试状态/);
'''.replace("STATUS", status))


def test_agent_mode_retains_visible_reconciliation_state_without_manual_forms():
    run_controller_test(r'''
const h = await harness([failedStep("unknown-result", false)], { agentMode: true });
assert.equal(h.elements.planRetryPanel.innerHTML, "");
assert.equal(h.controller.statusSnapshot().label, "待核对");
assert.match(h.elements.workflowStepper.innerHTML, /执行结果待核对/);
await h.click("unknown-result", { poisonForm: true });
assert.deepEqual(h.requests, []);
''')


def test_agent_mode_does_not_expose_manual_forms_for_retryable_failures():
    run_controller_test(r'''
const h = await harness([failedStep("safe-step", true)], { agentMode: true });
assert.equal(h.elements.planRetryPanel.innerHTML, "");
assert.equal(h.elements.planRetryPanel.attributes["aria-hidden"], "true");
assert.deepEqual(h.schemaRequests, []);
assert.equal(h.controller.statusSnapshot().label, "失败");
''')


@pytest.mark.parametrize("agent_mode", ["false", "true"])
def test_failed_plan_completion_shows_read_only_state_with_done_steps(agent_mode: str):
    run_controller_test(r'''
const step = failedStep("completed-step", true);
step.status = "done";
const h = await harness([step], { agentMode: AGENT_MODE });
const panel = h.elements.planRetryPanel;
assert.equal(panel.innerHTML, "");
await h.refresh({
  id: "plan-1", status: "failed", steps: [step],
  failure_envelope: {
    retryable: false, error_kind: "completion_reconciliation", failed_step_id: null,
    message: "workflow completion requires reconciliation", suggested_actions: ["halt"],
  },
});
assert.equal(panel.attributes["aria-hidden"], "false");
assert.match(panel.innerHTML, /计划完成结果待核对/);
assert.match(panel.innerHTML, /data-plan-reconciliation-id="plan-1"/);
assert.doesNotMatch(panel.innerHTML, /<(?:input|select|textarea|button)\b/);
assert.equal(h.controller.statusSnapshot().label, "待核对");
assert.match(h.controller.statusSnapshot().detail, /计划完成结果待核对/);
assert.equal(h.controller.planStep({ step_id: step.id }).status, "done");
assert.match(h.elements.workflowStepper.innerHTML, /notebook-step succeeded/);
assert.deepEqual(h.schemaRequests, []);
await h.click(step.id, { poisonForm: true });
assert.deepEqual(h.requests, []);
assert.deepEqual(h.busy, []);
assert.match(h.statuses.at(-1).message, /已暂停重试/);
await h.refresh({ id: "plan-1", status: "done", steps: [step] });
assert.equal(panel.innerHTML, "");
assert.equal(panel.attributes["aria-hidden"], "true");
assert.equal(h.controller.statusSnapshot().label, "已完成");
'''.replace("AGENT_MODE", agent_mode))


def test_plan_reconciliation_refresh_removes_safe_step_forms_and_blocks_actions():
    run_controller_test(r'''
const step = failedStep("safe-step", true);
const h = await harness([step]);
const panel = h.elements.planRetryPanel;
const initialSignature = panel.dataset.planRetrySignature;
assert.match(panel.innerHTML, /data-plan-retry-step="safe-step"/);
await h.refresh({
  id: "plan-1", status: "failed", steps: [step],
  failure_envelope: { retryable: false, error_kind: "completion_reconciliation" },
});
assert.notEqual(panel.dataset.planRetrySignature, initialSignature);
assert.match(panel.innerHTML, /计划完成结果待核对/);
assert.doesNotMatch(panel.innerHTML, /data-plan-retry-step=/);
await h.click(step.id, { poisonForm: true });
assert.deepEqual(h.requests, []);
await h.refresh({
  id: "plan-1", status: "failed", steps: [step],
  failure_envelope: { retryable: true },
});
assert.match(panel.innerHTML, /data-plan-retry-step="safe-step"/);
await h.click(step.id);
assert.equal(h.requests.length, 1);
''')


@pytest.mark.parametrize("agent_mode", ["false", "true"])
def test_trusted_reconciliation_action_sends_only_identity_and_never_retries_tool(agent_mode):
    run_controller_test(r'''
const step = failedStep("step-1", false);
const target = { id: "opaque-original-binding", step_id: step.id, supported: true, reason: "原执行凭据可核对。" };
let release;
const waiting = new Promise((resolve) => { release = resolve; });
const h = await harness([step], { agentMode: AGENT_MODE, apiHandler: async () => {
  await waiting;
  return { plan: { id: "plan-1", status: "done", steps: [{ ...step, status: "done" }] }, reconciliation_result: { outcome: "applied", reason: "原生产者回执已核对。" } };
}});
await h.refresh({ id: "plan-1", status: "failed", steps: [step], reconciliation: { targets: [target] } });
assert.match(h.elements.planRetryPanel.innerHTML, /data-plan-reconcile="opaque-original-binding"/);
assert.doesNotMatch(h.elements.planRetryPanel.innerHTML, /data-plan-retry-step=/);
const button = await h.reconcile(target.id);
assert.equal(button.disabled, true);
assert.deepEqual(h.requests, [{ url: "/api/plans/plan-1/reconcile", method: "POST", body: { target_id: target.id } }]);
assert.deepEqual(h.busy, [true]);
release();
await new Promise((resolve) => setImmediate(resolve));
assert.equal(button.disabled, false);
assert.deepEqual(h.busy, [true, false]);
assert.equal(h.controller.statusSnapshot().label, "已完成");
assert.match(h.statuses.at(-1).message, /原生产者回执已核对/);
assert.equal(h.requests.length, 1);
'''.replace("AGENT_MODE", agent_mode))


def test_unsupported_or_stale_reconciliation_action_is_blocked_and_failure_is_visible():
    run_controller_test(r'''
const step = failedStep("step-1", false);
const target = { id: "original", step_id: step.id, supported: false, reason: "没有可信核对器。" };
const h = await harness([step], { apiHandler: async () => { throw new Error("原生产者暂不可用"); } });
await h.refresh({ id: "plan-1", status: "failed", steps: [step], reconciliation: { targets: [target] } });
assert.match(h.elements.planRetryPanel.innerHTML, /没有可信核对器/);
assert.doesNotMatch(h.elements.planRetryPanel.innerHTML, /data-plan-reconcile=/);
await h.reconcile("forged");
await h.reconcile(target.id);
assert.deepEqual(h.requests, []);
target.supported = true;
await h.refresh({ id: "plan-1", status: "failed", steps: [step], reconciliation: { targets: [target] } });
const button = await h.reconcile(target.id);
assert.equal(button.disabled, false);
assert.match(h.statuses.at(-1).message, /原生产者暂不可用/);
assert.equal(h.statuses.at(-1).kind, "error");
assert.equal(h.controller.statusSnapshot().label, "待核对");
assert.equal(h.requests.length, 1);
''')


def test_cancelled_completion_uses_explicit_resume_endpoint():
    run_controller_test(r'''
const step = failedStep("step-1", false);
const target = { id: "checked-original-output", step_id: step.id, supported: true, action: "resume_completion" };
const h = await harness([step], { agentMode: true });
await h.refresh({ id: "plan-1", status: "cancelled", steps: [step], reconciliation: { targets: [target] } });
assert.match(h.elements.planRetryPanel.innerHTML, /恢复完成步骤/);
assert.doesNotMatch(h.elements.planRetryPanel.innerHTML, /data-plan-retry-step=/);
await h.reconcile(target.id);
assert.deepEqual(h.requests, [{ url: "/api/plans/plan-1/resume-completion", method: "POST", body: { target_id: target.id } }]);
''')


@pytest.mark.parametrize("agent_mode", ["false", "true"])
def test_continue_remaining_uses_current_snapshot_and_rejects_repeated_action(agent_mode):
    run_controller_test(r'''
const h = await harness([], { agentMode: AGENT_MODE });
await h.refresh({ id: "plan-1", status: "running", steps: [
  { id: "done", status: "done" }, { id: "next", status: "pending" },
], reconciliation: { targets: [], continuation: {
  expected_plan_fingerprint: "current-snapshot", remaining_step_ids: ["next"],
} } });
assert.equal(h.controller.statusSnapshot().label, "待继续");
assert.match(h.elements.planRetryPanel.innerHTML, /继续剩余步骤/);
assert.match(h.elements.planRetryPanel.innerHTML, /data-plan-continue="current-snapshot"/);
await h.continuePlan("stale-snapshot");
assert.deepEqual(h.requests, []);
await h.continuePlan("current-snapshot");
assert.deepEqual(h.requests, [{ url: "/api/plans/plan-1/run", method: "POST",
  body: { expected_plan_fingerprint: "current-snapshot" } }]);
await h.continuePlan("current-snapshot");
assert.equal(h.requests.length, 1);
assert.deepEqual(h.busy, [true, false]);
'''.replace("AGENT_MODE", agent_mode))


@pytest.mark.parametrize("reject", ["false", "true"])
def test_late_recovery_response_cannot_repaint_a_new_visit_and_releases_only_its_lease(reject):
    run_controller_test(r'''
let visit = 1, release, ownedLease;
const released = [];
const wait = new Promise((resolve) => { release = resolve; });
const step = failedStep("step-1", false);
const target = { id: "original", step_id: step.id, supported: true };
const h = await harness([step], {
  viewCallbacks: {
    captureView: () => visit, isCurrentView: (view) => view === visit,
    beginActivity: () => ownedLease || (ownedLease = { operation: "reconcile" }),
    endActivity: (lease) => released.push(lease),
  },
  apiHandler: async () => {
    await wait;
    if (REJECT) throw new Error("old request failed");
    return { plan: { id: "plan-1", status: "done", steps: [] } };
  },
});
await h.refresh({ id: "plan-1", status: "failed", steps: [step], reconciliation: { targets: [target] } });
await h.reconcile("original");
const originalLease = ownedLease;
visit += 2; // A -> B -> A: same task id, different visit.
ownedLease = { operation: "new visit" };
release();
await new Promise((resolve) => setImmediate(resolve));
assert.deepEqual(h.statuses, []);
assert.equal(h.controller.statusSnapshot().label, "待核对");
assert.deepEqual(released, [originalLease]);
assert.notEqual(released[0], ownedLease);
'''.replace("REJECT", reject))
