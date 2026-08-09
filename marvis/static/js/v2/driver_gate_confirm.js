import { apiConflictKind } from "../api.js";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// UX-2: manual mode never calls this (its plain-gate confirm lives in the step
// rail, driven by the same data-driver-confirm handler — see
// plan_rail_controller.js). Agent mode's chat timeline is the only caller
// (agentMessageGateButtonHtml in app.js), and previously it short-circuited
// here whenever isAgentMode was true, so a gate with no structured widget
// (screen/dedup/modeling_setup/join_c1) rendered nothing but a chat bubble —
// forcing the user through free-text routing even for a plain "确认 to
// proceed" step. Gates that DO carry a structured widget already get their
// own primary action button from that widget (screen-confirm / dedup-confirm
// / modeling-weight-adjust / c1-confirm), so this plain button only renders
// when no widget is mounted, in either mode.
function gateHasStructuredWidget(message) {
  const meta = message?.metadata || {};
  return Boolean(
    meta.join_c1
    || meta.screen
    || meta.modeling_setup
    || meta.dedup
    || meta.join_keys
    || meta.feature_binning
    || meta.special_values
    || meta.editable_input_schema?.properties?.adoption_reason
  );
}

// UX-10: a bare "确认" button looks identical whether it writes artifacts to disk
// (execute_join) or just accepts a read-only screening result — map the gate step's
// own tool (it IS the step the gate is confirming, e.g. execute_join/screen_features/
// train_model) to copy that states the consequence. Falls back to the generic "确认"
// when the tool isn't in this table or isn't known (never blocks the button).
const GATE_CONFIRM_LABELS = {
  execute_join: "确认并执行拼接",
  confirm_join: "确认并执行拼接",
  screen_features: "确认所选特征",
  select_features: "确认所选特征",
  train_model: "确认并开始训练",
  configure_tuning: "确认调参配置",
  tune_hyperparameters: "确认并开始调参",
  select_experiment: "确认所选实验",
  generate_model_report: "确认并生成报告",
  generate_model_reports: "确认并生成报告",
  post_training_action: "确认模型交付",
  resolve_special_values: "确认治理策略并继续",
  adopt_strategy: "填写理由并采纳",
  apply_monitoring_disposition: "确认知悉",
};

export function gateConfirmLabel(toolName) {
  return GATE_CONFIRM_LABELS[toolName] || "确认";
}

export function confirmationSnapshotAttributes(snapshot = {}) {
  const status = String(snapshot?.expected_plan_status || "");
  const revision = snapshot?.expected_plan_revision;
  const planFingerprint = String(snapshot?.expected_plan_fingerprint || "");
  const stepFingerprint = String(snapshot?.expected_step_fingerprint || "");
  if (
    !status
    || revision === ""
    || revision == null
    || !Number.isInteger(Number(revision))
    || !planFingerprint
  ) return "";
  return [
    ` data-expected-plan-status="${escapeHtml(status)}"`,
    ` data-expected-plan-revision="${escapeHtml(String(revision))}"`,
    ` data-expected-plan-fingerprint="${escapeHtml(planFingerprint)}"`,
    stepFingerprint
      ? ` data-expected-step-fingerprint="${escapeHtml(stepFingerprint)}"`
      : "",
  ].join("");
}

export function confirmationSnapshotFromControl(control, { requireStep = false } = {}) {
  const status = control?.getAttribute?.("data-expected-plan-status")
    || control?.dataset?.expectedPlanStatus || "";
  const revisionText = control?.getAttribute?.("data-expected-plan-revision")
    || control?.dataset?.expectedPlanRevision || "";
  const planFingerprint = control?.getAttribute?.("data-expected-plan-fingerprint")
    || control?.dataset?.expectedPlanFingerprint || "";
  const stepFingerprint = control?.getAttribute?.("data-expected-step-fingerprint")
    || control?.dataset?.expectedStepFingerprint || "";
  const revision = Number(revisionText);
  if (!status || revisionText === "" || !Number.isInteger(revision) || !planFingerprint) return null;
  if (requireStep && !stepFingerprint) return null;
  return {
    expected_plan_status: status,
    expected_plan_revision: revision,
    expected_plan_fingerprint: planFingerprint,
    ...(stepFingerprint ? { expected_step_fingerprint: stepFingerprint } : {}),
  };
}

function requestJsonBody(body) {
  if (body && typeof body === "object" && !Array.isArray(body)) return body;
  if (typeof body !== "string") return null;
  try {
    const parsed = JSON.parse(body);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch (_error) {
    return null;
  }
}

export function driverGateRequestBinding(endpoint, options = {}) {
  if (String(options?.method || "GET").toUpperCase() !== "POST") return null;
  const match = String(endpoint || "").match(
    /(?:^|\/)api\/tasks\/([^/?#]+)\/agent\/messages(?:[?#]|$)/,
  );
  const body = requestJsonBody(options?.body);
  if (!match || !body?.ui_action || !body?.expected_plan_id) return null;
  const snapshot = {
    expected_plan_status: body.expected_plan_status,
    expected_plan_revision: body.expected_plan_revision,
    expected_plan_fingerprint: body.expected_plan_fingerprint,
    ...(body.expected_step_fingerprint
      ? { expected_step_fingerprint: body.expected_step_fingerprint }
      : {}),
  };
  if (!confirmationSnapshotAttributes(snapshot)) return null;
  const stepId = String(body.expected_step_id || "");
  if (stepId && !snapshot.expected_step_fingerprint) return null;
  return {
    taskId: decodeURIComponent(match[1]),
    planId: String(body.expected_plan_id),
    stepId,
    snapshot,
  };
}

const pendingDriverGateSubmissions = new Map();

function driverGateSubmissionKey({ taskId, planId, stepId = "" } = {}) {
  const task = String(taskId || "").trim();
  const plan = String(planId || "").trim();
  const step = String(stepId || "").trim();
  return task && plan ? `${task}\u0000${plan}\u0000${step}` : "";
}

function confirmationSnapshotKey(snapshot = {}) {
  return JSON.stringify([
    String(snapshot?.expected_plan_status || ""),
    Number.isInteger(Number(snapshot?.expected_plan_revision))
      ? Number(snapshot.expected_plan_revision)
      : null,
    String(snapshot?.expected_plan_fingerprint || ""),
    String(snapshot?.expected_step_fingerprint || ""),
  ]);
}

function confirmationSnapshotIsValid(snapshot = {}, { requireStep = false } = {}) {
  const status = String(snapshot?.expected_plan_status || "");
  const revision = snapshot?.expected_plan_revision;
  const planFingerprint = String(snapshot?.expected_plan_fingerprint || "");
  const stepFingerprint = String(snapshot?.expected_step_fingerprint || "");
  return Boolean(
    status
    && revision !== ""
    && revision != null
    && Number.isInteger(Number(revision))
    && planFingerprint
    && (!requireStep || stepFingerprint),
  );
}

export function confirmationSnapshotsEqual(
  left = {},
  right = {},
  { requireStep = false } = {},
) {
  return confirmationSnapshotIsValid(left, { requireStep })
    && confirmationSnapshotIsValid(right, { requireStep })
    && confirmationSnapshotKey(left) === confirmationSnapshotKey(right);
}

export function claimDriverGateSubmission({
  taskId,
  planId,
  stepId = "",
  snapshot = {},
  state = "submitting",
} = {}) {
  const key = driverGateSubmissionKey({ taskId, planId, stepId });
  if (!key) return "";
  pendingDriverGateSubmissions.set(key, {
    taskId: String(taskId),
    planId: String(planId),
    stepId: String(stepId || ""),
    snapshotKey: confirmationSnapshotKey(snapshot),
    state: String(state || "submitting"),
  });
  return key;
}

export function markDriverGateSubmission({
  taskId,
  planId,
  stepId = "",
  snapshot = {},
  state,
} = {}) {
  const key = driverGateSubmissionKey({ taskId, planId, stepId });
  if (!key) return "";
  const existing = pendingDriverGateSubmissions.get(key);
  if (!existing || existing.snapshotKey !== confirmationSnapshotKey(snapshot)) {
    return claimDriverGateSubmission({ taskId, planId, stepId, snapshot, state });
  }
  existing.state = String(state || existing.state || "submitting");
  return key;
}

export function releaseDriverGateSubmission({ taskId, planId, stepId = "" } = {}) {
  const key = driverGateSubmissionKey({ taskId, planId, stepId });
  return key ? pendingDriverGateSubmissions.delete(key) : false;
}

function currentDriverGateSubmission({
  taskId,
  planId,
  stepId = "",
  stepStatus,
  snapshot = {},
} = {}) {
  const key = driverGateSubmissionKey({ taskId, planId, stepId });
  if (!key) return null;
  const claim = pendingDriverGateSubmissions.get(key);
  if (!claim) return null;
  const expectedStatus = stepId ? "awaiting_confirm" : "validated";
  if (
    confirmationSnapshotKey(snapshot) !== claim.snapshotKey
    || String(stepStatus || "") !== expectedStatus
  ) {
    pendingDriverGateSubmissions.delete(key);
    return null;
  }
  return claim;
}

// An active-job 409 is different from a stale CAS snapshot: the submitted
// action was not accepted, so the same gate may become actionable again after
// an authoritative task refresh proves the already-running job is idle. Do
// not infer that transition from a render using an old task object; only the
// task refresh path calls this reconciler.
export function reconcileDriverGateSubmissions(taskId, { serverBusy = false } = {}) {
  const normalizedTaskId = String(taskId || "");
  if (!normalizedTaskId || serverBusy) return 0;
  let released = 0;
  for (const [key, claim] of pendingDriverGateSubmissions.entries()) {
    if (claim.taskId !== normalizedTaskId || claim.state !== "active_driver_job") continue;
    pendingDriverGateSubmissions.delete(key);
    released += 1;
  }
  return released;
}

export function driverGatePendingClaimSignature(options = {}) {
  const claim = currentDriverGateSubmission(options);
  return claim ? `${claim.state}:${claim.snapshotKey}` : "";
}

export function driverGateActionable({
  taskId,
  planId,
  stepId = "",
  stepStatus,
  snapshot = {},
  localBusy = false,
  serverBusy = false,
} = {}) {
  const expectedStatus = stepId ? "awaiting_confirm" : "validated";
  const pending = currentDriverGateSubmission({
    taskId,
    planId,
    stepId,
    stepStatus,
    snapshot,
    serverBusy,
  });
  return confirmationSnapshotIsValid(snapshot, { requireStep: Boolean(stepId) })
    && String(stepStatus || "") === expectedStatus
    && !localBusy
    && !serverBusy
    && !pending;
}

// All typed confirmation widgets use this one API wrapper in app.js. It owns
// the task+plan+step claim before the network call starts, so a DOM rebuild or
// a second widget for the same gate cannot revive the action while the first
// request is in flight. Successful requests keep the claim until a new CAS
// snapshot/step status arrives; unrelated failures release it immediately.
export function createDriverGateApi({
  api,
  getLocalBusyAction = () => "",
  setDriverExecutionBusy = () => {},
  onSubmissionStateChange = () => {},
} = {}) {
  return async function driverGateApi(endpoint, options = {}) {
    if (typeof api !== "function") return undefined;
    const binding = driverGateRequestBinding(endpoint, options);
    if (!binding) return api(endpoint, options);

    claimDriverGateSubmission({ ...binding, state: "submitting" });
    const ownsBusy = !String(getLocalBusyAction(binding.taskId) || "");
    if (ownsBusy) setDriverExecutionBusy(true, binding.taskId);
    onSubmissionStateChange(binding, "submitting");
    try {
      const result = await api(endpoint, options);
      markDriverGateSubmission({ ...binding, state: "accepted" });
      onSubmissionStateChange(binding, "accepted");
      return result;
    } catch (error) {
      const conflictKind = apiConflictKind(error);
      if (conflictKind === "active_driver_job") {
        markDriverGateSubmission({ ...binding, state: "active_driver_job" });
      } else if (conflictKind === "confirmation_snapshot_stale") {
        markDriverGateSubmission({ ...binding, state: "confirmation_snapshot_stale" });
      } else {
        releaseDriverGateSubmission(binding);
      }
      onSubmissionStateChange(binding, conflictKind);
      throw error;
    } finally {
      if (ownsBusy) setDriverExecutionBusy(false, binding.taskId);
      onSubmissionStateChange(binding, "settled");
    }
  };
}

export async function refreshAfterConfirmationConflict(error, context = {}) {
  const conflictKind = apiConflictKind(error);
  if (![
    "active_driver_job",
    "confirmation_snapshot_stale",
  ].includes(conflictKind)) return false;
  const taskId = context.taskId || "";
  const setActionStatus = context.setActionStatus || (() => {});
  const refreshAgentMessages = context.refreshAgentMessages;
  let refreshed = false;
  if (taskId && typeof refreshAgentMessages === "function") {
    try {
      await refreshAgentMessages(taskId);
      refreshed = true;
    } catch (_refreshError) {
      // Keep the stale control disabled even when recovery itself fails. A
      // second submission with the same obsolete snapshot can never succeed.
    }
  }
  if (conflictKind === "active_driver_job") {
    setActionStatus(
      refreshed
        ? "上一步仍在执行，本次重复操作未提交。"
        : "上一步仍在执行，本次重复操作未提交；状态刷新失败，请稍后重试。",
      "busy",
    );
    return true;
  }
  setActionStatus(
    refreshed
      ? "计划已更新，已加载最新待确认步骤，请重新检查后操作。"
      : "计划已更新，但最新待确认步骤加载失败，请刷新页面后重试。",
    "info",
  );
  return true;
}

export function renderDriverGateButton(message, options = {}) {
  if (message?.metadata?.kind !== "gate") return "";
  if (gateHasStructuredWidget(message)) return "";
  const expectedPlanId = message?.metadata?.plan_id ? String(message.metadata.plan_id) : "";
  const expectedStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const expectedPlanAttr = expectedPlanId
    ? ` data-expected-plan-id="${escapeHtml(expectedPlanId)}"`
    : "";
  const expectedAttr = expectedStepId
    ? ` data-expected-step-id="${escapeHtml(expectedStepId)}"`
    : "";
  const snapshotAttrs = confirmationSnapshotAttributes(
    message?.metadata?.confirmation_snapshot || {},
  );
  const gateStepTool = options.gateStepTool || message?.metadata?.gate_source_tool || "";
  const modelCandidateRequiredAttr = gateStepTool === "select_experiment"
    && Array.isArray(message?.metadata?.model_delivery?.candidates)
    && message.metadata.model_delivery.candidates.some((item) => String(item?.id || "").trim())
    ? ' data-model-candidate-required="1"'
    : "";
  const label = gateConfirmLabel(gateStepTool);
  const disabledAttrs = options.interactive === false
    ? ' disabled aria-disabled="true" title="上一步正在收尾，完成后可继续授权"'
    : "";
  return '<div class="driver-gate-actions gate-action-bar">'
    + `<button type="button" class="button compact primary driver-confirm" data-driver-confirm="1"${expectedPlanAttr}${expectedAttr}${snapshotAttrs}${modelCandidateRequiredAttr}${disabledAttrs}>${escapeHtml(label)}</button>`
    + "</div>";
}

function candidateControlBinding(control) {
  return {
    planId: control?.getAttribute?.("data-expected-plan-id") || "",
    stepId: control?.getAttribute?.("data-expected-step-id") || "",
  };
}

function controlsIn(root) {
  if (!root || typeof root.querySelectorAll !== "function") return [];
  return [...root.querySelectorAll("[data-model-candidate-choice]")];
}

function modelCandidateControls(button, expectedPlanId, expectedStepId) {
  const localRoot = button?.closest?.(".driver-analysis-section, .agent-message, section") || null;
  const boundControls = (controls) => controls.filter((control) => {
    const binding = candidateControlBinding(control);
    if (expectedPlanId && binding.planId !== expectedPlanId) return false;
    if (expectedStepId && binding.stepId !== expectedStepId) return false;
    return true;
  });
  const localControls = boundControls(controlsIn(localRoot));
  if (localControls.length) return localControls;
  const globalControls = typeof document !== "undefined" ? controlsIn(document) : [];
  return boundControls(globalControls);
}

function modelCandidateSelection(button, expectedPlanId, expectedStepId) {
  const explicitlyRequired = button?.getAttribute?.("data-model-candidate-required") === "1";
  if (!expectedStepId && !explicitlyRequired) {
    return { required: false, selectedExperimentId: "" };
  }
  const controls = modelCandidateControls(button, expectedPlanId, expectedStepId);
  const required = explicitlyRequired || controls.length > 0;
  // A recovered gate can leave an older read-only copy in the timeline. The
  // action panel lives outside the message, so prefer the last matching DOM
  // group (the current gate) when there is no local section to scope against.
  const selected = [...controls].reverse().find((control) => control?.checked === true);
  return {
    required,
    selectedExperimentId: String(selected?.value || selected?.getAttribute?.("value") || "").trim(),
  };
}

function driverConfirmContext(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function"
      ? context.getSelectedTaskId()
      : context.selectedTaskId,
    api: context.api,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
    setDriverExecutionBusy: context.setDriverExecutionBusy || (() => {}),
    refreshAgentMessages: context.refreshAgentMessages,
  };
}

// UX-1: the backend now runs the whole driver turn inside a task job (REL-1), so
// this click can be minutes long (tune_hyperparameters/train_model). Give
// immediate feedback (busy pill), keep polling agent messages so intermediate
// step messages appear as the turn runs, and force the plan rail to re-fetch on
// a short interval so the running step's ring/elapsed time stays live instead of
// looking frozen until the request finally resolves.
export async function submitDriverConfirm(button, context = {}) {
  const {
    taskId, api, setActionStatus, setAgentMessages, renderAgentConversation,
    pollAgentMessagesUntilSettled, resetFetchThrottle, renderWorkflowStepper,
    setDriverExecutionBusy, refreshAgentMessages,
  } = driverConfirmContext(context);
  if (!taskId || typeof api !== "function") return;
  const expectedPlanId = button?.getAttribute?.("data-expected-plan-id") || "";
  const expectedStepId = button?.getAttribute?.("data-expected-step-id") || "";
  const confirmationSnapshot = confirmationSnapshotFromControl(
    button,
    { requireStep: Boolean(expectedStepId) },
  );
  if (!expectedPlanId || !confirmationSnapshot) {
    setActionStatus("计划已变化，请刷新后重新确认。", "error");
    return;
  }
  const body = { content: "确认", ui_action: expectedStepId ? "confirm_gate" : "start_plan" };
  body.expected_plan_id = expectedPlanId;
  if (expectedStepId) body.expected_step_id = expectedStepId;
  Object.assign(body, confirmationSnapshot);
  const candidateSelection = modelCandidateSelection(button, expectedPlanId, expectedStepId);
  if (candidateSelection.required && !candidateSelection.selectedExperimentId) {
    setActionStatus("请先选择一个候选实验。", "error");
    return;
  }
  if (candidateSelection.selectedExperimentId) {
    body.adjust_params = {
      selected_experiment_id: candidateSelection.selectedExperimentId,
    };
  }
  button.disabled = true;
  setDriverExecutionBusy(true, taskId);
  setActionStatus("正在执行下一步…", "busy");
  let planRailTimer = null;
  if (typeof setInterval === "function") {
    planRailTimer = setInterval(() => {
      resetFetchThrottle(taskId);
      renderWorkflowStepper({ force: true });
    }, 1500);
  }
  try {
    const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
    const result = await requestPromise;
    await streamPollPromise;
    setAgentMessages(result.messages);
    renderAgentConversation();
  } catch (error) {
    const conflictHandled = await refreshAfterConfirmationConflict(error, {
      taskId,
      refreshAgentMessages,
      setActionStatus,
    });
    if (!conflictHandled) {
      button.disabled = false;
      setActionStatus(error?.message || "确认失败", "error");
    }
  } finally {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    setDriverExecutionBusy(false, taskId);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  }
}

export function handleDriverConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-driver-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitDriverConfirm(button, context);
  return true;
}
