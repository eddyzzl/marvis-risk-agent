import { escapeHtml } from "../ui-utils.js";

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function cleanText(value) {
  return value === undefined || value === null ? "" : String(value).trim();
}

function escapedText(value) {
  return escapeHtml(value === undefined || value === null ? "" : String(value));
}

function diagnosticFactsHtml(diagnostic) {
  const facts = [];
  const location = cleanText(diagnostic.location);
  if (location) facts.push({ label: "出错位置", value: location });
  if (Array.isArray(diagnostic.evidence)) {
    diagnostic.evidence.forEach((item) => {
      if (!isRecord(item)) return;
      const value = cleanText(item.value);
      if (!value) return;
      facts.push({ label: cleanText(item.label) || "诊断证据", value });
    });
  }
  if (!facts.length) return "";
  return [
    '<dl class="workflow-error-card__facts">',
    ...facts.map(({ label, value }) => [
      '<div class="workflow-error-card__fact">',
      `<dt>${escapeHtml(label)}</dt>`,
      `<dd>${escapeHtml(value)}</dd>`,
      "</div>",
    ].join("")),
    "</dl>",
  ].join("");
}

function diagnosticActionsHtml(actions) {
  const items = Array.isArray(actions)
    ? actions.map(cleanText).filter(Boolean)
    : [];
  if (!items.length) return "";
  return [
    '<div class="workflow-error-card__section">',
    "<h4>处理步骤</h4>",
    '<ol class="workflow-error-card__actions">',
    ...items.map((item) => `<li>${escapeHtml(item)}</li>`),
    "</ol>",
    "</div>",
  ].join("");
}

function diagnosticTechnicalHtml(diagnostic) {
  const code = cleanText(diagnostic.code);
  const workflow = cleanText(diagnostic.workflow);
  const detail = cleanText(diagnostic.technical_detail);
  const rows = [
    code ? `<div><dt>错误代码</dt><dd><code>${escapeHtml(code)}</code></dd></div>` : "",
    workflow ? `<div><dt>工作流</dt><dd>${escapeHtml(workflow)}</dd></div>` : "",
  ].filter(Boolean).join("");
  return [
    '<details class="workflow-error-card__technical">',
    "<summary>技术信息</summary>",
    rows ? `<dl>${rows}</dl>` : "",
    detail
      ? `<pre><code>${escapeHtml(detail)}</code></pre>`
      : '<p class="workflow-error-card__technical-empty">暂无更多技术信息。</p>',
    "</details>",
  ].join("");
}

export function hasWorkflowErrorDiagnostic(metadata = {}) {
  return isRecord(metadata?.error_diagnostic);
}

export function workflowErrorDiagnosticHtml(diagnostic) {
  if (!isRecord(diagnostic)) return "";
  const code = cleanText(diagnostic.code);
  const workflow = cleanText(diagnostic.workflow);
  const title = cleanText(diagnostic.title) || "工作流执行失败";
  const summary = cleanText(diagnostic.summary)
    || "当前步骤未能完成，请根据诊断信息处理后再试。";
  const cause = cleanText(diagnostic.cause);
  const retryable = diagnostic.retryable === true
    ? { value: "true", label: "可重试" }
    : diagnostic.retryable === false
      ? { value: "false", label: "需人工处理" }
      : null;
  const eyebrow = [workflow, code].filter(Boolean).map(escapeHtml).join(" · ");
  return [
    `<section class="workflow-error-card" role="alert" aria-label="${escapeHtml(title)}"`
      + `${code ? ` data-error-code="${escapeHtml(code)}"` : ""}`
      + `${retryable ? ` data-retryable="${retryable.value}"` : ""}>`,
    '<header class="workflow-error-card__header">',
    '<span class="workflow-error-card__icon" aria-hidden="true">!</span>',
    '<div class="workflow-error-card__heading">',
    eyebrow ? `<div class="workflow-error-card__eyebrow">${eyebrow}</div>` : "",
    `<h3>${escapeHtml(title)}</h3>`,
    "</div>",
    retryable
      ? `<span class="workflow-error-card__retry">${retryable.label}</span>`
      : "",
    "</header>",
    `<p class="workflow-error-card__summary">${escapeHtml(summary)}</p>`,
    cause
      ? `<div class="workflow-error-card__section"><h4>问题原因</h4><p>${escapeHtml(cause)}</p></div>`
      : "",
    diagnosticFactsHtml(diagnostic),
    diagnosticActionsHtml(diagnostic.actions),
    diagnosticTechnicalHtml(diagnostic),
    "</section>",
  ].join("");
}

function noticeSeverityLabel(value) {
  const severity = cleanText(value);
  if (!severity) return "";
  const normalized = severity.toLowerCase();
  if (normalized === "warning" || normalized === "warn") return "提示";
  if (normalized === "info" || normalized === "information") return "信息";
  if (normalized === "error" || normalized === "danger") return "异常已恢复";
  return severity;
}

function ingestNoticeItemHtml(notice) {
  const code = cleanText(notice.code);
  const severity = cleanText(notice.severity);
  const file = cleanText(notice.file);
  const declared = cleanText(notice.declared_format);
  const detected = cleanText(notice.detected_format);
  const message = cleanText(notice.message)
    || "系统已自动识别并处理材料格式。";
  const severityLabel = noticeSeverityLabel(severity);
  const formatHtml = declared || detected
    ? [
      '<span class="workflow-ingest-notice__format">',
      "格式：",
      `<code>${escapeHtml(declared || "未声明")}</code>`,
      '<span aria-hidden="true"> → </span><span class="visually-hidden">，实际识别为 </span>',
      `<code>${escapeHtml(detected || "未识别")}</code>`,
      "</span>",
    ].join("")
    : "";
  return [
    `<li class="workflow-ingest-notice__item"`
      + `${code ? ` data-notice-code="${escapeHtml(code)}"` : ""}`
      + `${severity ? ` data-notice-severity="${escapeHtml(severity)}"` : ""}>`,
    `<p>${escapeHtml(message)}</p>`,
    '<div class="workflow-ingest-notice__meta">',
    file ? `<span>文件：<code>${escapeHtml(file)}</code></span>` : "",
    formatHtml,
    code ? `<span>记录：<code>${escapeHtml(code)}</code></span>` : "",
    severityLabel ? `<span>${escapeHtml(severityLabel)}</span>` : "",
    "</div>",
    "</li>",
  ].join("");
}

export function ingestNoticesHtml(notices) {
  const items = Array.isArray(notices) ? notices.filter(isRecord) : [];
  if (!items.length) return "";
  return [
    '<aside class="workflow-ingest-notice" role="status" aria-label="材料已自动处理">',
    '<header class="workflow-ingest-notice__header">',
    '<span class="workflow-ingest-notice__icon" aria-hidden="true">✓</span>',
    "<strong>已自动处理</strong>",
    items.length > 1 ? `<span>${items.length} 项</span>` : "",
    "</header>",
    '<ul class="workflow-ingest-notice__list">',
    ...items.map(ingestNoticeItemHtml),
    "</ul>",
    "</aside>",
  ].join("");
}

// One shared presentation boundary for Agent chat and manual Driver analysis.
// The callback remains responsible for trusted Markdown rendering; the default
// fallback is escaped plain text. A structured diagnostic replaces the raw
// exception body, while non-error notices are prepended and preserve it.
export function workflowMessageContentHtml(message = {}, renderContent = escapedText) {
  const metadata = isRecord(message?.metadata) ? message.metadata : {};
  const noticesHtml = ingestNoticesHtml(metadata.ingest_notices);
  const diagnosticHtml = workflowErrorDiagnosticHtml(metadata.error_diagnostic);
  const content = message?.content === undefined || message?.content === null
    ? ""
    : String(message.content);
  const bodyHtml = diagnosticHtml || String(renderContent(content) ?? "");
  return `${noticesHtml}${bodyHtml}`;
}
