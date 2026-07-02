import { escapeHtml } from "../ui-utils.js";

export function screenNum(value) {
  const n = Number(value);
  return value === null || value === undefined || Number.isNaN(n) ? "n/a" : n.toFixed(4);
}

export function screenPct(value) {
  const n = Number(value);
  return value === null || value === undefined || Number.isNaN(n) ? "n/a" : (n * 100).toFixed(1) + "%";
}

export function renderScreenGateTable(message, options = {}) {
  const screen = message?.metadata?.screen;
  if (!screen || typeof screen !== "object") return "";
  const messageId = message?.id ? String(message.id) : "";
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const interactive = options.interactive !== false;
  const disabledAttr = interactive ? "" : " disabled aria-disabled=\"true\"";
  const scores = screen.scores && typeof screen.scores === "object" ? screen.scores : {};
  const selectedSet = new Set((screen.selected || []).map((value) => String(value)));
  const badges = {
    keep: '<span class="screen-badge keep">入选</span>',
    leakage: '<span class="screen-badge leak">泄漏</span>',
    suspected: '<span class="screen-badge susp">疑似</span>',
    unusable: '<span class="screen-badge unusable">不可用</span>',
  };
  const row = (feature, ks, category) => {
    const name = String(feature);
    const stats = scores[name] && typeof scores[name] === "object" ? scores[name] : {};
    const ksValue = ks === undefined ? stats.ks : ks;
    const checked = selectedSet.has(name);
    const disabled = category === "unusable" || !interactive; // constant/sparse: no signal to select
    return `<tr class="screen-row screen-${category}">
      <td class="screen-pick-cell"><input type="checkbox" class="screen-pick" value="${escapeHtml(name)}"${checked ? " checked" : ""}${disabled ? " disabled" : ""} /></td>
      <td class="screen-feat">${escapeHtml(name)}</td>
      <td class="screen-num">${screenNum(ksValue)}</td>
      <td class="screen-num">${screenNum(stats.iv)}</td>
      <td class="screen-num">${screenPct(stats.missing_rate)}</td>
      <td>${badges[category] || ""}</td>
    </tr>`;
  };
  const tuple = (item) => (Array.isArray(item) ? item : [item]);
  const rows = [];
  for (const feature of screen.selected || []) rows.push(row(feature, undefined, "keep"));
  for (const item of screen.leakage || []) rows.push(row(tuple(item)[0], tuple(item)[1], "leakage"));
  for (const item of screen.suspected || []) rows.push(row(tuple(item)[0], tuple(item)[1], "suspected"));
  for (const item of (screen.unusable || []).slice(0, 50)) rows.push(row(tuple(item)[0], null, "unusable"));
  const thresholds = screen.thresholds && typeof screen.thresholds === "object" ? screen.thresholds : {};
  const leakageKs = thresholds.leakage_ks ?? 0.4;
  const maxMissingRate = thresholds.max_missing_rate ?? 0.95;
  const note = interactive
    ? `共筛 ${screen.n_screened ?? rows.length} 列;泄漏阈值 KS≥${leakageKs}。勾选=入选,可硬选泄漏/疑似列;确认后用所选特征训练。`
    : `共筛 ${screen.n_screened ?? rows.length} 列;泄漏阈值 KS≥${leakageKs}。这是历史筛选结果,如需调整请使用最新待确认步骤。`;
  const thresholdControls = `<div class="screen-threshold-controls">
    <label>泄漏KS <input type="number" class="screen-threshold-input" data-screen-threshold="leakage_ks" min="0" max="1" step="0.01" value="${escapeHtml(String(leakageKs))}"${disabledAttr} required /></label>
    <label>最大缺失率 <input type="number" class="screen-threshold-input" data-screen-threshold="max_missing_rate" min="0" max="1" step="0.01" value="${escapeHtml(String(maxMissingRate))}"${disabledAttr} required /></label>
    <button type="button" class="button compact secondary screen-adjust"${interactive ? ` data-screen-adjust="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "重算" : "已归档"}</button>
  </div>`;
  return `<div class="screen-table-wrap" data-screen-form="${escapeHtml(messageId)}" data-screen-step-id="${escapeHtml(gateStepId)}"${interactive ? "" : ' data-screen-readonly="true"'}>
    ${thresholdControls}
    <div class="screen-table-scroll">
      <table class="screen-table">
        <thead><tr><th>选</th><th>特征</th><th>KS</th><th>IV</th><th>缺失率</th><th>类别</th></tr></thead>
        <tbody>${rows.join("")}</tbody>
      </table>
    </div>
    <div class="screen-table-foot">
      <span class="screen-note">${escapeHtml(note)}</span>
      <button type="button" class="button compact primary screen-confirm"${interactive ? ` data-screen-confirm="${escapeHtml(messageId)}"` : disabledAttr}>${interactive ? "确认所选特征" : "历史结果"}</button>
    </div>
  </div>`;
}

function screenGateContext(context = {}) {
  return {
    taskId: typeof context.getSelectedTaskId === "function"
      ? context.getSelectedTaskId()
      : context.selectedTaskId,
    api: context.api,
    acceptanceMode: typeof context.agentAcceptanceModeValue === "function"
      ? context.agentAcceptanceModeValue()
      : context.acceptanceMode,
    setActionStatus: context.setActionStatus || (() => {}),
    setAgentMessages: context.setAgentMessages || (() => {}),
    renderAgentConversation: context.renderAgentConversation || (() => {}),
    pollAgentMessagesUntilSettled: context.pollAgentMessagesUntilSettled || (() => Promise.resolve()),
    resetFetchThrottle: context.resetFetchThrottle || (() => {}),
    renderWorkflowStepper: context.renderWorkflowStepper || (() => {}),
  };
}

// UX-1: screen-gate submissions rerun the driver turn (now job-wrapped, REL-1)
// and can take a while (recompute screening / retrain downstream). Give
// immediate busy feedback, poll agent messages so intermediate step content
// streams in, and force the plan rail to re-fetch on a short interval.
function withDriverTurnBusyFeedback(taskId, context, run) {
  const { setActionStatus, pollAgentMessagesUntilSettled, resetFetchThrottle, renderWorkflowStepper } = context;
  setActionStatus("正在执行下一步…", "busy");
  let planRailTimer = null;
  if (typeof setInterval === "function") {
    planRailTimer = setInterval(() => {
      resetFetchThrottle(taskId);
      renderWorkflowStepper({ force: true });
    }, 1500);
  }
  const stopPlanRailTicker = () => {
    if (planRailTimer !== null) clearInterval(planRailTimer);
    resetFetchThrottle(taskId);
    renderWorkflowStepper({ force: true });
  };
  return run(pollAgentMessagesUntilSettled).finally(stopPlanRailTicker);
}

export async function submitScreenThresholdAdjust(button, rawContext = {}) {
  const wrap = button.closest(".screen-table-wrap");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = screenGateContext(rawContext);
  if (!wrap || !taskId || typeof api !== "function") return;
  if (wrap.dataset.screenReadonly === "true") {
    setActionStatus("这是历史筛选结果,请使用最新待确认步骤调整。", "error");
    return;
  }
  const adjustParams = {};
  for (const input of wrap.querySelectorAll(".screen-threshold-input")) {
    const key = input.getAttribute("data-screen-threshold");
    if (!key) continue;
    const rawValue = String(input.value || "").trim();
    if (!rawValue) {
      setActionStatus("阈值不能为空。", "error");
      return;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      setActionStatus("阈值需在 0 到 1 之间。", "error");
      return;
    }
    adjustParams[key] = value;
  }
  if (!Object.keys(adjustParams).length) return;
  const expectedStepId = wrap.dataset.screenStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息,请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  const context = screenGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "调整筛选阈值",
          adjust_params: adjustParams,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "重算特征筛选失败", "error");
  }
}

export async function submitScreenSelection(button, rawContext = {}) {
  const wrap = button.closest(".screen-table-wrap");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = screenGateContext(rawContext);
  if (!wrap || !taskId || typeof api !== "function") return;
  if (wrap.dataset.screenReadonly === "true") {
    setActionStatus("这是历史筛选结果,请使用最新待确认步骤确认。", "error");
    return;
  }
  const selection = [];
  for (const box of wrap.querySelectorAll(".screen-pick:checked")) {
    if (!box.disabled) selection.push(box.value);
  }
  if (!selection.length) {
    setActionStatus("请至少勾选一个特征。", "error");
    return;
  }
  const expectedStepId = wrap.dataset.screenStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息,请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  const context = screenGateContext(rawContext);
  try {
    await withDriverTurnBusyFeedback(taskId, context, async (pollAgentMessagesUntilSettled) => {
      const requestPromise = api(`/api/tasks/${taskId}/agent/messages`, {
        method: "POST",
        body: JSON.stringify({
          content: "确认",
          selection,
          expected_step_id: expectedStepId,
          acceptance_mode: acceptanceMode,
        }),
      });
      const streamPollPromise = pollAgentMessagesUntilSettled(taskId, requestPromise, { preserveOptimistic: true });
      const result = await requestPromise;
      await streamPollPromise;
      setAgentMessages(result.messages);
      renderAgentConversation();
    });
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "确认所选特征失败", "error");
  }
}

export function handleScreenAdjustClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-adjust]");
  if (!button) return false;
  event.preventDefault();
  void submitScreenThresholdAdjust(button, context);
  return true;
}

export function handleScreenConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-screen-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitScreenSelection(button, context);
  return true;
}
