import { escapeHtml } from "../ui-utils.js";

const DEDUP_STRATEGY_LABELS = { first: "保留首条 (first)", last: "保留末条 (last)" };

function joinGateContext(context = {}) {
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
  };
}

export function renderJoinC1Form(message) {
  const c1 = message?.metadata?.join_c1;
  if (!c1 || !Array.isArray(c1.files) || !c1.files.length) return "";
  const messageId = message?.id ? String(message.id) : "";
  const roleSelect = (datasetId, selected) => {
    const opt = (value, label) =>
      `<option value="${value}"${selected === value ? " selected" : ""}>${label}</option>`;
    return (
      `<select class="c1-role" data-c1-dataset="${escapeHtml(datasetId)}">`
      + opt("anchor", "样本主表")
      + opt("feature", "特征表")
      + opt("ignore", "忽略")
      + "</select>"
    );
  };
  const rows = c1.files
    .map(
      (file) => `<tr>
      <td class="c1-file">${escapeHtml(file.name || "")}</td>
      <td>${escapeHtml(String(file.row_count ?? ""))}</td>
      <td>${escapeHtml(String(file.n_cols ?? ""))}</td>
      <td>${file.has_target ? "✓" : ""}</td>
      <td>${roleSelect(file.dataset_id || "", file.proposed_role || "feature")}</td>
    </tr>`,
    )
    .join("");
  const columns = [];
  const seen = new Set();
  for (const file of c1.files) {
    for (const col of file.columns || []) {
      if (!seen.has(col)) {
        seen.add(col);
        columns.push(col);
      }
    }
  }
  const targetOptions = ['<option value="">（不指定）</option>']
    .concat(
      columns.map(
        (col) => `<option value="${escapeHtml(col)}"${col === c1.target_col ? " selected" : ""}>${escapeHtml(col)}</option>`,
      ),
    )
    .join("");
  return `<div class="c1-form" data-c1-form="${escapeHtml(messageId)}">
    <table class="c1-form-table">
      <thead><tr><th>文件</th><th>行数</th><th>列数</th><th>含目标</th><th>角色</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="c1-form-foot">
      <label class="c1-target-label">目标列 <select class="c1-target">${targetOptions}</select></label>
      <button type="button" class="button compact primary c1-confirm" data-c1-confirm="${escapeHtml(messageId)}">确认角色</button>
    </div>
  </div>`;
}

export async function submitC1Assignment(button, context = {}) {
  const form = button.closest(".c1-form");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = joinGateContext(context);
  if (!form || !taskId || typeof api !== "function") return;
  const anchorIds = [];
  const featureIds = [];
  for (const select of form.querySelectorAll(".c1-role")) {
    const datasetId = select.getAttribute("data-c1-dataset");
    if (select.value === "anchor") anchorIds.push(datasetId);
    else if (select.value === "feature") featureIds.push(datasetId);
  }
  if (!anchorIds.length) {
    setActionStatus("请先把一张表选为「样本主表」。", "error");
    return;
  }
  if (anchorIds.length > 1) {
    setActionStatus("只能有一张样本主表，请把其余表改为「特征表」或「忽略」。", "error");
    return;
  }
  const targetCol = form.querySelector(".c1-target")?.value || "";
  button.disabled = true;
  try {
    const result = await api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "[C1]" + JSON.stringify({ anchor_id: anchorIds[0], anchor_ids: anchorIds, feature_ids: featureIds, target_col: targetCol }),
        acceptance_mode: acceptanceMode,
      }),
    });
    setAgentMessages(result.messages);
    renderAgentConversation();
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "确认角色失败", "error");
  }
}

export function handleC1ConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-c1-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitC1Assignment(button, context);
  return true;
}

export function renderDedupPicker(message) {
  const dedup = message?.metadata?.dedup;
  if (!dedup || !Array.isArray(dedup.features) || !dedup.features.length) return "";
  const messageId = message?.id ? String(message.id) : "";
  const gateStepId = message?.metadata?.step_id ? String(message.metadata.step_id) : "";
  const strategies = Array.isArray(dedup.strategies) && dedup.strategies.length ? dedup.strategies : ["first", "last"];
  const rows = dedup.features
    .map((feature) => {
      const fid = String(feature.feature_id);
      const conflicts = feature.conflict_keys ? `${feature.conflict_keys} 个同键冲突` : "拼接键不唯一";
      const options = strategies
        .map((strategy) => {
          const value = String(strategy);
          return `<option value="${escapeHtml(value)}">${escapeHtml(DEDUP_STRATEGY_LABELS[value] || value)}</option>`;
        })
        .join("");
      return `<tr>
      <td class="dedup-feat">${escapeHtml(fid)}</td>
      <td>${escapeHtml(conflicts)}</td>
      <td><select class="dedup-strategy" data-dedup-feature="${escapeHtml(fid)}">${options}</select></td>
    </tr>`;
    })
    .join("");
  return `<div class="dedup-picker" data-dedup-form="${escapeHtml(messageId)}" data-dedup-gate-step-id="${escapeHtml(gateStepId)}">
    <p class="dedup-note">以下特征表的拼接键不唯一(同键多行),请选择去重策略后再拼接:</p>
    <table class="dedup-table">
      <thead><tr><th>特征表</th><th>冲突</th><th>去重策略</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="dedup-foot">
      <button type="button" class="button compact primary dedup-confirm" data-dedup-confirm="${escapeHtml(messageId)}">应用去重并确认</button>
    </div>
  </div>`;
}

export async function submitDedupStrategies(button, context = {}) {
  const form = button.closest(".dedup-picker");
  const { taskId, api, acceptanceMode, setActionStatus, setAgentMessages, renderAgentConversation } = joinGateContext(context);
  if (!form || !taskId || typeof api !== "function") return;
  const dedupStrategies = {};
  for (const select of form.querySelectorAll(".dedup-strategy")) {
    const featureId = select.getAttribute("data-dedup-feature");
    if (featureId) dedupStrategies[featureId] = select.value;
  }
  const expectedStepId = form.dataset.dedupGateStepId || "";
  if (!expectedStepId) {
    setActionStatus("缺少待确认步骤校验信息,请刷新后重试。", "error");
    return;
  }
  button.disabled = true;
  try {
    const result = await api(`/api/tasks/${taskId}/agent/messages`, {
      method: "POST",
      body: JSON.stringify({
        content: "确认",
        dedup_strategies: dedupStrategies,
        expected_step_id: expectedStepId,
        acceptance_mode: acceptanceMode,
      }),
    });
    setAgentMessages(result.messages);
    renderAgentConversation();
  } catch (error) {
    button.disabled = false;
    setActionStatus(error?.message || "应用去重失败", "error");
  }
}

export function handleDedupConfirmClick(event, context = {}) {
  const button = event.target?.closest?.("[data-dedup-confirm]");
  if (!button) return false;
  event.preventDefault();
  void submitDedupStrategies(button, context);
  return true;
}
