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

// What the platform checked: where it stopped plus each evidence item.
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

function diagnosticActionsHtml(actions, { hasRecovery = false } = {}) {
  const items = Array.isArray(actions)
    ? actions.map(cleanText).filter(Boolean)
    : [];
  if (!items.length) return "";
  return [
    '<div class="workflow-error-card__section">',
    `<h4>${hasRecovery ? "处理方式" : "你可以这样处理"}</h4>`,
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

function diagnosticRecoveryHtml(diagnostic) {
  const prompt = cleanText(diagnostic.agent_prompt);
  const actions = Array.isArray(diagnostic.recovery_actions)
    ? diagnostic.recovery_actions.filter(isRecord)
    : [];
  if (!prompt && !actions.length) return "";
  return [
    '<div class="workflow-error-card__recovery">',
    prompt ? `<p>${escapeHtml(prompt)}</p>` : "",
    actions.length ? '<div class="workflow-error-card__recovery-actions">' : "",
    ...actions.filter((action) => cleanText(action.command)).map((action, index) => {
      const label = cleanText(action.label) || "继续处理";
      const command = cleanText(action.command);
      // D14: an option's consequence is shown only when the platform states it.
      const effect = cleanText(action.consequence || action.effect || action.description);
      // B-line: an auto-tier option is reversible; say so next to the button.
      const reversible = cleanText(action.tier) === "auto";
      return [
        `<span class="workflow-error-card__option"${action.tier ? ` data-repair-tier="${escapeHtml(cleanText(action.tier))}"` : ""}>`,
        `<button type="button" class="button compact${index === 0 ? " primary" : ""}" data-workflow-recovery-command="${escapeHtml(command)}">${escapeHtml(label)}</button>`,
        // Read-only copy of the label for Agent mode, where the button is
        // stripped and the same command is offered as a quick reply.
        `<span class="workflow-error-card__option-label">${escapeHtml(label)}</span>`,
        reversible ? '<span class="workflow-error-card__tier">可撤销</span>' : "",
        effect ? `<small>${escapeHtml(effect)}</small>` : "",
        "</span>",
      ].join("");
    }),
    actions.length ? "</div>" : "",
    "</div>",
  ].join("");
}

export function hasWorkflowErrorDiagnostic(metadata = {}) {
  return isRecord(metadata?.error_diagnostic);
}

export function workflowErrorDiagnosticHtml(diagnostic, { role = "alert" } = {}) {
  if (!isRecord(diagnostic)) return "";
  const code = cleanText(diagnostic.code);
  const title = cleanText(diagnostic.title) || "工作流执行失败";
  const summary = cleanText(diagnostic.summary)
    || "当前步骤未能完成，请根据诊断信息处理后再试。";
  const cause = cleanText(diagnostic.cause);
  const impact = cleanText(diagnostic.impact);
  const facts = diagnosticFactsHtml(diagnostic);
  const recovery = diagnosticRecoveryHtml(diagnostic);
  const retryable = diagnostic.retryable === true
    ? { value: "true", label: "可重试" }
    : diagnostic.retryable === false
      ? { value: "false", label: "需人工处理" }
      : null;
  return [
    `<section class="workflow-error-card" role="${role === "status" ? "status" : "alert"}" aria-label="${escapeHtml(title)}"`
      + `${code ? ` data-error-code="${escapeHtml(code)}"` : ""}`
      + `${retryable ? ` data-retryable="${retryable.value}"` : ""}>`,
    '<header class="workflow-error-card__header">',
    '<span class="workflow-error-card__icon" aria-hidden="true">!</span>',
    '<div class="workflow-error-card__heading">',
    // Batch 3a: internal workflow / error codes stay in 技术信息 below, not in
    // the heading the user reads first.
    `<h3>${escapeHtml(title)}</h3>`,
    "</div>",
    retryable
      ? `<span class="workflow-error-card__retry">${retryable.label}</span>`
      : "",
    "</header>",
    // D14 (batch 6b): what failed, why, what the platform checked and what
    // it affects, then the ways forward. Rows appear only for fields the
    // diagnostic carries.
    '<dl class="workflow-error-card__chain">',
    `<div><dt>报错</dt><dd><p class="workflow-error-card__summary">${escapeHtml(summary)}</p></dd></div>`,
    cause ? `<div><dt>问题原因</dt><dd><p>${escapeHtml(cause)}</p></dd></div>` : "",
    facts ? `<div><dt>我查了</dt><dd>${facts}</dd></div>` : "",
    impact ? `<div><dt>影响</dt><dd><p>${escapeHtml(impact)}</p></dd></div>` : "",
    "</dl>",
    recovery,
    diagnosticActionsHtml(diagnostic.actions, { hasRecovery: Boolean(recovery) }),
    diagnosticTechnicalHtml(diagnostic),
    "</section>",
  ].join("");
}

// B-line (2026-10-16 spec): the "已处理好" card. Every line comes from the
// platform's execution record; a repair that failed its check says so.
export function repairReportCardHtml(report, { undone = false, undoable: undoableOption = true } = {}) {
  let undoable = undoableOption;
  if (!isRecord(report)) return "";
  const verified = report.verified === true;
  const isUndone = undone || report.undone === true;
  const title = isUndone
    ? "已撤销这次处理"
    : verified ? "遇到问题，已处理好" : "处理没成功，已还原";
  const rows = [
    ["发现", report.finding],
    ["处理", report.action],
    ["核对", report.verification],
  ].filter(([, value]) => cleanText(value));
  const undoCommand = cleanText(report.undo_command);
  const log = Array.isArray(report.log) ? report.log.map(cleanText).filter(Boolean) : [];
  // Data actions and environment overrides carry an audit entry: who, when,
  // what changed and how to undo it. It goes into the log, not new captions.
  if (isRecord(report.audit)) {
    const audit = report.audit;
    const who = cleanText(audit.actor?.label) || cleanText(audit.actor?.kind);
    if (who || cleanText(audit.at)) log.push(`操作人：${who || "—"}，时间：${cleanText(audit.at) || "—"}`);
    if (cleanText(audit.changed)) log.push(`改动：${cleanText(audit.changed)}`);
    if (cleanText(audit.undo)) log.push(`撤销方式：${cleanText(audit.undo)}`);
  }
  // V2 workflow reports: the backend decides on read whether undo still applies.
  if (report.undo_available === false) undoable = false;
  return [
    `<section class="workflow-repair-card${verified ? "" : " is-unverified"}${isUndone ? " is-undone" : ""}" role="status" aria-label="${escapeHtml(title)}"`
      + `${report.repair_id ? ` data-repair-id="${escapeHtml(cleanText(report.repair_id))}"` : ""}>`,
    '<header class="workflow-repair-card__header">',
    `<span class="workflow-repair-card__icon" aria-hidden="true">${verified && !isUndone ? "✓" : "↺"}</span>`,
    `<h3>${escapeHtml(title)}</h3>`,
    "</header>",
    '<dl class="workflow-repair-card__rows">',
    ...rows.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHtml(cleanText(value))}</dd></div>`),
    "</dl>",
    (undoCommand && verified && !isUndone && undoable) || log.length ? '<div class="workflow-repair-card__footer">' : "",
    undoCommand && verified && !isUndone && undoable
      ? `<button type="button" class="button compact" data-workflow-recovery-command="${escapeHtml(undoCommand)}">撤销</button>`
      : "",
    log.length
      ? [
        '<details class="workflow-repair-card__log">',
        "<summary>查看日志</summary>",
        "<ol>",
        ...log.map((line) => `<li>${escapeHtml(line)}</li>`),
        "</ol>",
        "</details>",
      ].join("")
      : "",
    (undoCommand && verified && !isUndone && undoable) || log.length ? "</div>" : "",
    "</section>",
  ].join("");
}

const PAST_CONFIRMATION_STAGES = new Set(["reproducibility", "metrics", "word_conclusion_draft"]);

// Mirrors latest_undoable_report in marvis/agent/validation_repair.py.
export function repairTimelineState(messages = []) {
  const list = Array.isArray(messages) ? messages : [];
  const undone = new Set(
    list.map((item) => cleanText(item?.metadata?.repair_undo?.repair_id)).filter(Boolean),
  );
  let undoableId = "";
  for (let index = list.length - 1; index >= 0; index -= 1) {
    const report = list[index]?.metadata?.repair_report;
    if (!isRecord(report)) continue;
    const repairId = cleanText(report.repair_id);
    const later = list.slice(index + 1).some((item) => item?.role === "assistant"
      && PAST_CONFIRMATION_STAGES.has(cleanText(item?.stage)));
    if (repairId && cleanText(report.undo_command) && report.verified === true
      && report.undo_available !== false && !undone.has(repairId) && !later) {
      undoableId = repairId;
    }
    break;
  }
  return { undone, undoableId };
}

export function repairUndoNoticeHtml(undo) {
  if (!isRecord(undo)) return "";
  const finding = cleanText(undo.finding) || "已撤销这次自动处理。";
  return `<p class="workflow-repair-undo" role="status"><span aria-hidden="true">↺</span>${escapeHtml(finding)}</p>`;
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
// ``repairState`` comes from the whole timeline: which reports were undone,
// and the one report that can still be undone (the newest, before a later
// stage started). Without it a lone card keeps its own undo button.
export function workflowMessageContentHtml(message = {}, renderContent = escapedText, { repairState = null } = {}) {
  const metadata = isRecord(message?.metadata) ? message.metadata : {};
  if (isRecord(metadata.repair_report)) {
    const repairId = cleanText(metadata.repair_report.repair_id);
    return repairReportCardHtml(metadata.repair_report, {
      undone: Boolean(repairId && repairState?.undone?.has?.(repairId)),
      undoable: repairState ? Boolean(repairId) && repairState.undoableId === repairId : true,
    });
  }
  if (isRecord(metadata.repair_undo)) return repairUndoNoticeHtml(metadata.repair_undo);
  const noticesHtml = ingestNoticesHtml(metadata.ingest_notices);
  // B-line extension: a strategy sample-design gap is a clarification, not a
  // failed step, so its card is a status card under its own key.
  const diagnosticHtml = workflowErrorDiagnosticHtml(metadata.error_diagnostic)
    || workflowErrorDiagnosticHtml(metadata.sample_design_diagnostic, { role: "status" });
  const content = message?.content === undefined || message?.content === null
    ? ""
    : String(message.content);
  const bodyHtml = diagnosticHtml || String(renderContent(content) ?? "");
  return `${noticesHtml}${bodyHtml}`;
}
