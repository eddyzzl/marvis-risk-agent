function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderDriverGateButton(message, options = {}) {
  const isAgentMode = typeof options.isAgentMode === "function"
    ? options.isAgentMode()
    : Boolean(options.isAgentMode);
  if (message?.metadata?.kind !== "gate" || isAgentMode) return "";
  const expectedStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const expectedAttr = expectedStepId
    ? ` data-expected-step-id="${escapeHtml(expectedStepId)}"`
    : "";
  return '<div class="driver-gate-actions">'
    + `<button type="button" class="button compact primary driver-confirm" data-driver-confirm="1"${expectedAttr}>确认</button>`
    + "</div>";
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
  } = driverConfirmContext(context);
  if (!taskId || typeof api !== "function") return;
  const expectedStepId = button?.getAttribute?.("data-expected-step-id") || "";
  const body = { content: "确认" };
  if (expectedStepId) body.expected_step_id = expectedStepId;
  button.disabled = true;
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
    button.disabled = false;
    setActionStatus(error?.message || "确认失败", "error");
  } finally {
    if (planRailTimer !== null) clearInterval(planRailTimer);
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
