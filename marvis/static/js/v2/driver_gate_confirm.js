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
  };
}

export async function submitDriverConfirm(button, context = {}) {
  const { taskId, api, setActionStatus, setAgentMessages, renderAgentConversation } = driverConfirmContext(context);
  if (!taskId || typeof api !== "function") return;
  const expectedStepId = button?.getAttribute?.("data-expected-step-id") || "";
  const body = { content: "确认" };
  if (expectedStepId) body.expected_step_id = expectedStepId;
  button.disabled = true;
  try {
    const result = await api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    setAgentMessages(result.messages);
    renderAgentConversation();
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "确认失败", "error");
  }
}

export function handleDriverConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-driver-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitDriverConfirm(button, context);
  return true;
}
