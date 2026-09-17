import { escapeHtml } from "./ui-utils.js";

export const REPORT_DRAFT_FIELDS = [
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
    key: "TEXT:final_validation_conclusion",
    label: "最终验证结论",
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
  } = {},
) {
  const rows = REPORT_DRAFT_FIELDS.map((field) => reportDraftRowHtml(
    field,
    String(values[field.key] || ""),
    editable,
  )).join("");
  const revisionValue = Number.isFinite(Number(revision)) ? Number(revision) : 0;
  const messageAttr = messageId
    ? ` data-report-draft-message-id="${escapeHtml(String(messageId))}"`
    : "";
  const confirmHtml = editable
    ? [
      '<footer class="report-draft-actions">',
      '<button type="button" class="button compact primary" data-report-draft-confirm>',
      "确认并生成报告",
      "</button>",
      "</footer>",
    ].join("")
    : "";
  return [
    `<section class="report-draft-panel" data-report-draft-table="true"${messageAttr}`,
    ` data-report-revision="${escapeHtml(String(revisionValue))}"`,
    ` data-report-draft-editable="${editable ? "true" : "false"}">`,
    '<header class="report-draft-head">',
    "<h3>报告结论</h3>",
    editable
      ? "<p>点击填写内容即可修改。确认后才生成 Word 和 Excel。</p>"
      : "<p>该草稿已确认或已被更新版本替代。</p>",
    "</header>",
    '<div class="report-draft-table-wrap"><table class="report-draft-table">',
    "<thead><tr><th>字段</th><th>填写内容</th></tr></thead>",
    `<tbody>${rows}</tbody>`,
    "</table></div>",
    confirmHtml,
    "</section>",
  ].join("");
}

function reportDraftRowHtml(field, value, editable) {
  const control = editable
    ? reportDraftControlHtml(field, value)
    : `<div class="report-draft-readonly">${escapeHtml(value) || "—"}</div>`;
  return [
    "<tr>",
    "<th scope=\"row\">",
    `<span class="report-draft-label">${escapeHtml(field.label)}</span>`,
    `<code class="report-draft-key">${escapeHtml(field.key)}</code>`,
    "</th>",
    `<td>${control}</td>`,
    "</tr>",
  ].join("");
}

function reportDraftControlHtml(field, value) {
  const keyAttr = escapeHtml(field.key);
  const labelAttr = escapeHtml(field.label);
  const valueAttr = escapeHtml(value);
  if (field.long) {
    return [
      `<textarea class="report-draft-input" data-report-draft-key="${keyAttr}"`,
      ` aria-label="${labelAttr}" rows="4">${valueAttr}</textarea>`,
    ].join("");
  }
  return [
    `<input class="report-draft-input" data-report-draft-key="${keyAttr}"`,
    ` aria-label="${labelAttr}" type="text" value="${valueAttr}">`,
  ].join("");
}
