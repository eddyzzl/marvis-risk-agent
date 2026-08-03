function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function renderFeatureBinningGate(message, options = {}) {
  const payload = message?.metadata?.feature_binning;
  if (!payload || !Array.isArray(payload.features)) return "";
  const interactive = options.interactive !== false;
  const disabled = interactive ? "" : " disabled";
  const stepId = String(message?.metadata?.step_id || "");
  const minBins = Number(payload.min_bins || 3);
  const maxBins = Number(payload.max_bins || 20);
  const defaultBins = Number(payload.default_bins || 10);
  const optionsHtml = payload.features.map((item) => {
    const feature = String(item?.feature || "");
    const recommendation = String(item?.recommendation || "");
    const reason = String(item?.recommendation_reason || "");
    return [
      '<label class="feature-binning-option">',
      `<input type="checkbox" data-feature-binning-pick value="${escapeHtml(feature)}"${disabled}>`,
      '<span class="feature-binning-option-main">',
      '<span class="feature-binning-option-title">',
      `<strong>${escapeHtml(feature)}</strong>`,
      recommendation ? `<span class="feature-binning-recommendation">${escapeHtml(recommendation)}</span>` : "",
      "</span>",
      reason ? `<small title="${escapeHtml(reason)}">${escapeHtml(reason)}</small>` : "",
      "</span>",
      "</label>",
    ].join("");
  }).join("");
  return [
    `<section class="feature-binning-gate" data-feature-binning-step-id="${escapeHtml(stepId)}">`,
    '<div class="feature-binning-heading"><div><h4>是否需要查看特征分箱？</h4><p>可多选；不选择也可以直接跳过并生成报告。</p></div>',
    `<label class="feature-binning-count"><span>分箱数</span><input type="number" data-feature-binning-count min="${minBins}" max="${maxBins}" value="${defaultBins}"${disabled}></label></div>`,
    `<div class="feature-binning-options">${optionsHtml}</div>`,
    '<div class="feature-binning-actions gate-action-bar">',
    `<button type="button" class="button compact secondary" data-feature-binning-submit="skip"${disabled}>跳过分箱并生成报告</button>`,
    `<button type="button" class="button compact primary" data-feature-binning-submit="selected"${disabled}>分析所选特征并生成报告</button>`,
    "</div>",
    "</section>",
  ].join("");
}

function contextValues(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function" ? context.getSelectedTaskId() : context.selectedTaskId,
    api: context.api,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
  };
}

export async function submitFeatureBinning(button, context = {}) {
  const values = contextValues(context);
  const wrap = button?.closest?.("[data-feature-binning-step-id]");
  if (!wrap || !values.taskId || typeof values.api !== "function") return;
  const expectedStepId = wrap.dataset.featureBinningStepId || "";
  const mode = button.dataset.featureBinningSubmit || "skip";
  const features = mode === "skip"
    ? []
    : [...wrap.querySelectorAll("[data-feature-binning-pick]:checked")].map((input) => input.value);
  const bins = Number(wrap.querySelector("[data-feature-binning-count]")?.value || 10);
  if (mode === "selected" && features.length === 0) {
    values.setActionStatus("请至少选择一个特征，或点击“跳过分箱并生成报告”。", "error");
    return;
  }
  if (!Number.isInteger(bins) || bins < 3 || bins > 20) {
    values.setActionStatus("分箱数必须是 3 到 20 之间的整数。", "error");
    return;
  }
  wrap.querySelectorAll("button, input").forEach((node) => { node.disabled = true; });
  values.setActionStatus(features.length ? "正在计算分箱并生成报告…" : "正在生成特征分析报告…", "busy");
  try {
    const request = values.api(`/api/tasks/${values.taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "确认",
        ui_action: "confirm_feature_binning",
        expected_step_id: expectedStepId,
        adjust_params: { features, bins },
      }),
    });
    const poll = values.pollAgentMessagesUntilSettled(values.taskId, request, { preserveOptimistic: true });
    const result = await request;
    await poll;
    values.setAgentMessages(result.messages);
    values.renderAgentConversation();
  } catch (error) {
    wrap.querySelectorAll("button, input").forEach((node) => { node.disabled = false; });
    values.setActionStatus(error?.message || "提交分箱设置失败", "error");
  } finally {
    values.resetFetchThrottle(values.taskId);
    values.renderWorkflowStepper({ force: true });
  }
}

export function handleFeatureBinningClick(event, context = {}) {
  const button = event.target?.closest?.("[data-feature-binning-submit]");
  if (!button) return false;
  event.preventDefault();
  void submitFeatureBinning(button, context);
  return true;
}
