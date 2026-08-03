function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatSpecialValue(value) {
  if (value === null) return "null";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function formatShare(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const ratio = number >= 0 && number <= 1 ? number * 100 : number;
  return `${ratio.toFixed(ratio >= 10 ? 1 : 2)}%`;
}

function valueEvidenceHtml(values = []) {
  return values.map((item) => {
    const value = item && typeof item === "object" && Object.prototype.hasOwnProperty.call(item, "value")
      ? item.value
      : item;
    const share = item && typeof item === "object" ? formatShare(item.share) : "";
    return [
      '<span class="special-value-chip">',
      `<strong>${escapeHtml(formatSpecialValue(value))}</strong>`,
      share ? `<span>${escapeHtml(share)}</span>` : "",
      "</span>",
    ].join("");
  }).join("");
}

export function renderSpecialValueGate(message, options = {}) {
  const payload = message?.metadata?.special_values;
  if (!payload || !Array.isArray(payload.columns) || payload.columns.length === 0) return "";
  const interactive = options.interactive !== false;
  const disabled = interactive ? "" : " disabled";
  const planId = String(message?.metadata?.plan_id || "");
  const stepId = String(message?.metadata?.step_id || payload.step_id || "");
  const rows = payload.columns.map((item) => {
    const column = String(item?.column || "");
    if (!column) return "";
    return [
      `<article class="special-value-row" data-special-value-row data-special-value-column="${escapeHtml(column)}">`,
      '<div class="special-value-evidence">',
      `<div class="special-value-column"><strong>${escapeHtml(column)}</strong><span>疑似特殊值</span></div>`,
      `<div class="special-value-chips">${valueEvidenceHtml(item?.values || [])}</div>`,
      "</div>",
      '<div class="special-value-decision">',
      `<label><span>治理方式</span><select data-special-value-action aria-label="${escapeHtml(column)} 治理方式"${disabled}>`,
      '<option value="">请选择</option>',
      '<option value="mask">转为空值，按缺失处理</option>',
      '<option value="retain">保留原值（需说明理由）</option>',
      '<option value="drop">删除该特征</option>',
      "</select></label>",
      `<label class="special-value-reason"><span>保留理由</span><input type="text" data-special-value-reason maxlength="500" placeholder="仅选择“保留原值”时必填"${disabled}></label>`,
      "</div>",
      "</article>",
    ].join("");
  }).join("");
  return [
    `<section class="special-value-gate" data-special-value-plan-id="${escapeHtml(planId)}" data-special-value-step-id="${escapeHtml(stepId)}">`,
    '<header class="special-value-gate-heading">',
    '<div><span class="special-value-kicker">Human in the loop</span><h4>确认特殊值治理策略</h4>',
    '<p>逐列选择处理方式。系统会使用检测结果中的完整值集合，界面不会回传或改写特殊值。</p></div>',
    `<span class="special-value-count">${payload.columns.length} 列待决策</span>`,
    "</header>",
    `<div class="special-value-list">${rows}</div>`,
    '<div class="special-value-actions gate-action-bar">',
    `<button type="button" class="button compact primary" data-special-value-submit${disabled}>确认治理策略并继续</button>`,
    "</div>",
    "</section>",
  ].join("");
}

export function collectSpecialValueDecisions(wrap) {
  const decisions = {};
  const errors = [];
  for (const row of wrap?.querySelectorAll?.("[data-special-value-row]") || []) {
    const column = String(row.dataset?.specialValueColumn || "").trim();
    const action = String(row.querySelector?.("[data-special-value-action]")?.value || "").trim();
    const reason = String(row.querySelector?.("[data-special-value-reason]")?.value || "").trim();
    if (!column) continue;
    if (!["mask", "retain", "drop"].includes(action)) {
      errors.push(`${column} 尚未选择治理方式`);
      continue;
    }
    const decision = { action };
    if (action === "retain") {
      if (!reason) {
        errors.push(`${column} 选择保留时必须填写理由`);
        continue;
      }
      decision.confirmed = true;
      decision.reason = reason;
    }
    decisions[column] = decision;
  }
  return { decisions, errors };
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

function setControlsDisabled(wrap, disabled) {
  wrap?.querySelectorAll?.("button, select, input").forEach((node) => {
    node.disabled = disabled;
  });
}

export async function submitSpecialValueDecisions(button, context = {}) {
  const values = contextValues(context);
  const wrap = button?.closest?.("[data-special-value-step-id]");
  if (!wrap || !values.taskId || typeof values.api !== "function") return;
  const expectedPlanId = wrap.dataset.specialValuePlanId || "";
  const expectedStepId = wrap.dataset.specialValueStepId || "";
  const { decisions, errors } = collectSpecialValueDecisions(wrap);
  if (errors.length) {
    values.setActionStatus(errors[0], "error");
    return;
  }
  const rowCount = wrap.querySelectorAll?.("[data-special-value-row]")?.length || 0;
  if (!rowCount || Object.keys(decisions).length !== rowCount) {
    values.setActionStatus("必须为每个特殊值特征选择治理方式。", "error");
    return;
  }
  setControlsDisabled(wrap, true);
  values.setActionStatus("正在应用特殊值治理策略…", "busy");
  try {
    const request = values.api(`/api/tasks/${values.taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "确认",
        ui_action: "confirm_gate",
        expected_plan_id: expectedPlanId,
        expected_step_id: expectedStepId,
        adjust_params: { decisions },
      }),
    });
    const poll = values.pollAgentMessagesUntilSettled(
      values.taskId,
      request,
      { preserveOptimistic: true },
    );
    const result = await request;
    await poll;
    values.setAgentMessages(result.messages);
    values.renderAgentConversation();
  } catch (error) {
    setControlsDisabled(wrap, false);
    values.setActionStatus(error?.message || "提交特殊值治理策略失败", "error");
  } finally {
    values.resetFetchThrottle(values.taskId);
    values.renderWorkflowStepper({ force: true });
  }
}

export function handleSpecialValueClick(event, context = {}) {
  const button = event.target?.closest?.("[data-special-value-submit]");
  if (!button) return false;
  event.preventDefault();
  void submitSpecialValueDecisions(button, context);
  return true;
}
