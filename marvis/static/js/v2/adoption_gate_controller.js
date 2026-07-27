function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function adoptionReasonSchema(message) {
  const schema = message?.metadata?.editable_input_schema;
  const reason = schema?.properties?.adoption_reason;
  return reason && typeof reason === "object" ? reason : null;
}

export function isAdoptionGate(message) {
  return Boolean(adoptionReasonSchema(message));
}

export function renderAdoptionGate(message, options = {}) {
  if (!isAdoptionGate(message)) return "";
  const interactive = options.interactive !== false;
  const stepId = String(message?.metadata?.step_id || "");
  const disabled = interactive ? "" : " disabled";
  const readonly = interactive ? "false" : "true";
  return [
    `<div class="adoption-gate" data-adoption-step-id="${escapeHtml(stepId)}" data-adoption-readonly="${readonly}">`,
    '<label class="adoption-reason-field">',
    "<span>采纳理由</span>",
    `<textarea data-adoption-reason rows="3" maxlength="1000" placeholder="说明业务目标、验证证据和批准依据"${disabled}></textarea>`,
    "</label>",
    '<p class="adoption-gate-note">理由会与当前策略版本、回测结果和审计记录绑定，不能使用“待确认”或 TODO 占位。</p>',
    '<div class="adoption-gate-actions gate-action-bar">',
    `<button type="button" class="button compact primary adoption-confirm" data-adoption-confirm="1"${disabled}>填写理由并采纳</button>`,
    '</div>',
    "</div>",
  ].join("");
}

function contextValues(context = {}) {
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

export async function submitAdoption(button, context = {}) {
  const {
    taskId, api, setActionStatus, setAgentMessages, renderAgentConversation,
    pollAgentMessagesUntilSettled, resetFetchThrottle, renderWorkflowStepper,
  } = contextValues(context);
  const wrap = button?.closest?.("[data-adoption-step-id]");
  const reason = wrap?.querySelector?.("[data-adoption-reason]")?.value?.trim?.() || "";
  const expectedStepId = wrap?.dataset?.adoptionStepId || "";
  if (!taskId || typeof api !== "function") return;
  if (!expectedStepId) {
    setActionStatus("缺少当前采纳步骤，请刷新后重试。", "error");
    return;
  }
  if (reason.length < 2) {
    setActionStatus("请填写真实、可审计的采纳理由。", "error");
    return;
  }

  button.disabled = true;
  setActionStatus("正在绑定理由并采纳策略…", "busy");
  try {
    const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "确认采纳",
        ui_action: "confirm_adoption",
        adjust_params: { adoption_reason: reason },
        expected_step_id: expectedStepId,
      }),
    });
    const pollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
    const result = await requestPromise;
    await pollPromise;
    setAgentMessages(result.messages);
    renderAgentConversation();
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "策略采纳失败", "error");
  } finally {
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  }
}

export function handleAdoptionConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-adoption-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitAdoption(button, context);
  return true;
}
