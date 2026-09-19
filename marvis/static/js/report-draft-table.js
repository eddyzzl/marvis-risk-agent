import { escapeHtml } from "./ui-utils.js";

export const REPORT_DRAFT_FIELDS = [
  {
    key: "TEXT:final_validation_conclusion",
    label: "最终验证结论",
    long: true,
  },
  {
    key: "TEXT:pressure_test_summary",
    label: "压力测试总结",
    long: true,
  },
  {
    key: "TEXT:pressure_impact_recommendation",
    label: "压力影响建议",
    long: true,
  },
  {
    key: "TEXT:model_overview",
    label: "模型概述",
    long: true,
  },
  {
    key: "TEXT:model_scope",
    label: "适用范围",
    long: false,
  },
  {
    key: "TEXT:sample_audience",
    label: "样本人群",
    long: false,
  },
  {
    key: "TEXT:bad_sample_definition",
    label: "坏样本定义",
    long: false,
  },
  {
    key: "TEXT:good_sample_definition",
    label: "好样本定义",
    long: false,
  },
  {
    key: "TEXT:model_training_description",
    label: "模型训练说明",
    long: true,
  },
];

const REQUIRED_REPORT_DRAFT_KEYS = [
  "TEXT:pressure_test_summary",
  "TEXT:pressure_impact_recommendation",
  "TEXT:final_validation_conclusion",
];

export function latestPendingReportDraftMessageId(messages = []) {
  if (!Array.isArray(messages)) return "";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.stage === "word_conclusion_confirmed") return "";
    if (message?.role !== "assistant") continue;
    if (message?.stage !== "word_conclusion_draft") continue;
    if (!hasReportDraftValues(message?.metadata?.draft_values)) continue;
    return String(message.id || "");
  }
  return "";
}

export function hasReportDraftValues(values) {
  return Boolean(values) && typeof values === "object" && !Array.isArray(values)
    && Object.keys(values).length > 0;
}

export function collectReportDraftValues(root) {
  const values = {};
  if (!root?.querySelectorAll) return values;
  root.querySelectorAll("[data-report-draft-key]").forEach((field) => {
    const key = field.getAttribute("data-report-draft-key");
    if (!key) return;
    values[key] = String(field.value || "");
  });
  return values;
}

export function reportDraftHasRequiredValues(values = {}) {
  return REQUIRED_REPORT_DRAFT_KEYS.every((key) => String(values[key] || "").trim());
}

export function reportDraftTableHtml(
  values = {},
  {
    editable = false,
    revision = 0,
    messageId = "",
    taskId = "",
    state = null,
  } = {},
) {
  const rows = REPORT_DRAFT_FIELDS.map((field) => reportDraftRowHtml(
    field,
    String(values[field.key] || ""),
    editable,
    Boolean(state?.confirming),
  )).join("");
  const revisionValue = Number.isFinite(Number(revision)) ? Number(revision) : 0;
  const messageAttr = messageId
    ? ` data-report-draft-message-id="${escapeHtml(String(messageId))}"`
    : "";
  const confirmHtml = editable
    ? [
      '<footer class="report-draft-actions">',
      `<button type="button" class="button compact secondary" data-report-draft-save${state?.confirming ? " disabled" : ""}>保存草稿</button>`,
      `<button type="button" class="button compact primary" data-report-draft-confirm${state?.conflict || state?.confirming ? " disabled" : ""}>`,
      "确认并生成报告",
      "</button>",
      "</footer>",
    ].join("")
    : "";
  return [
    `<section class="report-draft-panel" data-report-draft-table="true"${messageAttr}`,
    ` data-report-draft-task-id="${escapeHtml(taskId)}"`,
    ` data-report-revision="${escapeHtml(String(revisionValue))}"`,
    ` data-report-draft-editable="${editable ? "true" : "false"}">`,
    '<header class="report-draft-head">',
    "<h3>报告结论</h3>",
    editable
      ? '<p>逐项编辑，草稿自动保存。确认后生成 Word 和 Excel。</p><span class="report-draft-save-status" role="status" aria-live="polite" data-report-draft-save-status>' + escapeHtml(state?.status || "已保存草稿") + '</span>'
      : "<p>该草稿已确认或已被更新版本替代。</p>",
    "</header>",
    '<div data-report-draft-feedback>' + reportDraftFeedbackHtml(state) + '</div>',
    confirmHtml,
    '<div class="report-draft-table-wrap"><table class="report-draft-table">',
    "<thead><tr><th>字段</th><th>填写内容</th></tr></thead>",
    `<tbody>${rows}</tbody>`,
    "</table></div>",
    '<details class="report-draft-mapping"><summary>查看字段映射</summary>',
    REPORT_DRAFT_FIELDS.map((field) => `<div>${escapeHtml(field.label)} <code>${escapeHtml(field.key)}</code></div>`).join(""),
    '</details>',
    "</section>",
  ].join("");
}

function reportDraftRowHtml(field, value, editable, confirming) {
  const control = editable
    ? reportDraftControlHtml(field, value, confirming)
    : `<div class="report-draft-readonly">${escapeHtml(value) || "—"}</div>`;
  return [
    "<tr>",
    "<th scope=\"row\">",
    `<span class="report-draft-label">${escapeHtml(field.label)}</span>`,
    editable ? `<button type="button" class="report-draft-revise" data-report-draft-revise="${escapeHtml(field.label)}"${confirming ? " disabled" : ""}>请 Agent 修订</button>` : "",
    "</th>",
    `<td>${control}</td>`,
    "</tr>",
  ].join("");
}

export function reportDraftFeedbackHtml(state) {
  if (!state) return "";
  const error = state.error ? `<p role="alert">${escapeHtml(state.error)}</p>` : "";
  if (!state.conflict) return error;
  if (state.conflict.unavailable) return error + '<p>草稿已确认或撤回。请复制保留当前修改，并在对话中请求重新起草。</p>';
  const differences = REPORT_DRAFT_FIELDS.filter((field) =>
    String(state.values[field.key] || "") !== String(state.conflict.values[field.key] || ""));
  return error + '<details class="report-draft-conflict" open><summary>比较版本 · 当前修改尚未覆盖服务器</summary>'
    + differences.map((field) => `<div><strong>${escapeHtml(field.label)}</strong><p>当前修改：${escapeHtml(state.values[field.key] || "（空）")}</p><p>最新版本：${escapeHtml(state.conflict.values[field.key] || "（空）")}</p></div>`).join("")
    + '<button type="button" class="button compact secondary" data-report-draft-resolve="server">使用最新版本</button> '
    + '<button type="button" class="button compact secondary" data-report-draft-resolve="local">将当前修改应用到最新草稿</button></details>';
}

function reportDraftControlHtml(field, value, confirming) {
  const keyAttr = escapeHtml(field.key);
  const labelAttr = escapeHtml(field.label);
  const valueAttr = escapeHtml(value);
  if (field.long) {
    return [
      `<textarea class="report-draft-input" data-report-draft-key="${keyAttr}"`,
      ` aria-label="${labelAttr}" placeholder="待补充" rows="4"${confirming ? " readonly" : ""}>${valueAttr}</textarea>`,
    ].join("");
  }
  return [
    `<input class="report-draft-input" data-report-draft-key="${keyAttr}"`,
    ` aria-label="${labelAttr}" placeholder="待补充" type="text" value="${valueAttr}"${confirming ? " readonly" : ""}>`,
  ].join("");
}
