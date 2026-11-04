// V3 D-6 / D-8: conversation cards for SQL queries and monitoring schedules.
// The Agent only drafts (a query, a form); every button here posts an explicit
// confirmation to a platform endpoint, which runs the panel's own proposal,
// allow-list and audit path. Cards read metadata.conversation_action only.
// Pure module: no DOM access at import time.
import { escapeHtml } from "./ui-utils.js";

const FREQUENCY_LABELS = { daily: "每天", weekly: "每周", monthly: "每月" };
const NOTIFY_LABELS = { local_inbox: "本机收件箱" };
const FIELD_LABELS = {
  target_id: "监控对象",
  dataset_id: "监控数据",
  time_col: "时间列",
  frequency: "频率",
  interval: "间隔",
  start_date: "首次运行",
  notify: "通知",
  "advanced.schedule_id": "调度名称",
  "advanced.timezone": "时区",
  "advanced.complete_through": "数据完整覆盖到",
  "advanced.target_col": "标签列",
  "advanced.label_maturity_days": "标签成熟天数",
  "advanced.catch_up_budget": "单次最多补跑期数",
  "advanced.max_attempts": "最多尝试次数",
  "advanced.lease_seconds": "执行租约秒数",
};

function actionOf(message) {
  const action = message?.metadata?.conversation_action;
  return action && typeof action === "object" ? action : null;
}

// A proposal or form card is settled once a later message points back at it.
export function settledConversationActionIds(messages = []) {
  const ids = new Set();
  for (const message of Array.isArray(messages) ? messages : []) {
    const source = actionOf(message)?.source_message_id;
    if (source) ids.add(String(source));
  }
  return ids;
}

function head(title) {
  return `<div class="result-card-head"><b>${escapeHtml(title)}</b></div>`;
}

function button(label, attrs, { primary = true, disabled = false } = {}) {
  return `<button type="button" class="button compact ${primary ? "primary" : "secondary"}" ${attrs}${disabled ? " disabled" : ""}>${escapeHtml(label)}</button>`;
}

function option(value, label, selected, { raw = false } = {}) {
  const noTranslate = raw ? ' translate="no"' : "";
  return `<option value="${escapeHtml(value)}"${selected ? " selected" : ""}${noTranslate}>${escapeHtml(label)}</option>`;
}

// raw: the choices are column names (user data); only the placeholder translates.
function select(name, choices, value, { allowEmpty = true, raw = false } = {}) {
  const items = allowEmpty ? [option("", "请选择", value == null || value === "")] : [];
  for (const [itemValue, label] of choices) {
    items.push(option(itemValue, label, String(value ?? "") === String(itemValue), { raw }));
  }
  return `<select data-schedule-field="${escapeHtml(name)}">${items.join("")}</select>`;
}

function field(name, control, missing) {
  const invalid = missing.has(name) ? ' data-invalid="true"' : "";
  return `<label class="conversation-form-field"${invalid}><span>${escapeHtml(FIELD_LABELS[name] || name)}</span>${control}</label>`;
}

function input(name, value, type = "text", extra = "") {
  return `<input type="${type}" data-schedule-field="${escapeHtml(name)}" value="${escapeHtml(value ?? "")}"${extra}>`;
}

function sqlProposalHtml(message, action, settled) {
  const id = escapeHtml(String(message.id || ""));
  return [
    `<section class="result-card conversation-action-card" data-conversation-action="sql_query_proposal" data-message-id="${id}">`,
    head("待确认查询"),
    `<pre><code class="language-sql">${escapeHtml(action.query || "")}</code></pre>`,
    `<dl class="conversation-action-facts"><dt>数据源</dt><dd translate="no">${escapeHtml(action.source_name || action.source_id || "")}</dd>`,
    `<dt>行数上限</dt><dd>${escapeHtml(String(action.max_rows ?? ""))}</dd></dl>`,
    `<div class="history-actions">${button(settled ? "已执行" : "确认执行", `data-sql-conversation-confirm="${id}"`, { disabled: settled })}</div>`,
    '<p class="conversation-action-status" role="status"></p>',
    "</section>",
  ].join("");
}

function sqlSourceChoiceHtml(message, action, settled) {
  const id = escapeHtml(String(message.id || ""));
  const sources = Array.isArray(action.sources) ? action.sources : [];
  return [
    `<section class="result-card conversation-action-card" data-conversation-action="sql_source_choice" data-message-id="${id}">`,
    head("选择数据源"),
    '<div class="history-actions">',
    ...sources.map((source) => button(String(source.name || source.source_id), `data-sql-conversation-source="${escapeHtml(String(source.source_id))}" data-message-id="${id}" translate="no"`, { primary: false, disabled: settled })),
    "</div>",
    '<p class="conversation-action-status" role="status"></p>',
    "</section>",
  ].join("");
}

function sqlResultHtml(action) {
  const dataset = action.dataset || {};
  const columns = Array.isArray(dataset.column_names) ? dataset.column_names : [];
  return [
    '<section class="result-card conversation-action-card" data-conversation-action="sql_query_result">',
    head("取数结果"),
    `<dl class="conversation-action-facts"><dt>行数</dt><dd>${escapeHtml(String(dataset.row_count ?? ""))}</dd>`,
    `<dt>列数</dt><dd>${escapeHtml(String(dataset.column_count ?? columns.length))}</dd>`,
    `<dt>字段</dt><dd translate="no">${escapeHtml(columns.join("、"))}</dd></dl>`,
    "</section>",
  ].join("");
}

function scheduleFormHtml(message, action, settled) {
  const id = escapeHtml(String(message.id || ""));
  const form = action.form || {};
  const advanced = form.advanced || {};
  const options = action.options || {};
  const missing = new Set(Array.isArray(action.missing) ? action.missing : []);
  const datasets = Array.isArray(options.datasets) ? options.datasets : [];
  const dataset = datasets.find((item) => item.id === form.dataset_id);
  const columns = (dataset?.column_names || []).map((name) => [name, name]);
  const disabled = settled ? " disabled" : "";
  const business = [
    field("target_id", select("target_id", (options.targets || []).map((item) => [item.id, item.label]), form.target_id), missing),
    field("dataset_id", select("dataset_id", datasets.map((item) => [item.id, `${item.id} · ${item.row_count} 行`]), form.dataset_id), missing),
    field("time_col", select("time_col", columns, form.time_col, { raw: true }), missing),
    field("frequency", select("frequency", Object.entries(FREQUENCY_LABELS), form.frequency, { allowEmpty: false }), missing),
    field("interval", input("interval", form.interval, "number", ' min="1" max="12"'), missing),
    field("start_date", input("start_date", form.start_date, "date"), missing),
    field("notify", select("notify", Object.entries(NOTIFY_LABELS), form.notify, { allowEmpty: false }), missing),
  ];
  const raw = [
    field("advanced.schedule_id", input("advanced.schedule_id", advanced.schedule_id), missing),
    field("advanced.timezone", input("advanced.timezone", advanced.timezone), missing),
    field("advanced.complete_through", input("advanced.complete_through", advanced.complete_through), missing),
    field("advanced.target_col", select("advanced.target_col", columns, advanced.target_col, { raw: true }), missing),
    field("advanced.label_maturity_days", input("advanced.label_maturity_days", advanced.label_maturity_days, "number", ' min="0"'), missing),
    field("advanced.catch_up_budget", input("advanced.catch_up_budget", advanced.catch_up_budget, "number", ' min="1" max="100"'), missing),
    field("advanced.max_attempts", input("advanced.max_attempts", advanced.max_attempts, "number", ' min="1" max="10"'), missing),
    field("advanced.lease_seconds", input("advanced.lease_seconds", advanced.lease_seconds, "number", ' min="60"'), missing),
  ];
  return [
    `<section class="result-card conversation-action-card" data-conversation-action="monitoring_schedule_form" data-message-id="${id}">`,
    head("定时监控"),
    `<fieldset class="conversation-form-grid"${disabled}>${business.join("")}</fieldset>`,
    `<details class="conversation-form-advanced"><summary>高级设置</summary><fieldset class="conversation-form-grid"${disabled}>${raw.join("")}</fieldset></details>`,
    `<div class="history-actions">${button(settled ? "已创建" : "确认创建", `data-schedule-conversation-confirm="${id}"`, { disabled: settled })}</div>`,
    '<p class="conversation-action-status" role="status"></p>',
    "</section>",
  ].join("");
}

function scheduleCreatedHtml(action) {
  const summary = action.summary || {};
  const every = summary.interval > 1 ? `每 ${summary.interval} ${{ daily: "天", weekly: "周", monthly: "个月" }[summary.frequency] || ""}` : FREQUENCY_LABELS[summary.frequency] || "";
  return [
    '<section class="result-card conversation-action-card" data-conversation-action="monitoring_schedule_created">',
    head("定时监控已创建"),
    `<dl class="conversation-action-facts"><dt>调度名称</dt><dd translate="no">${escapeHtml(action.schedule_id || "")}</dd>`,
    `<dt>监控对象</dt><dd>${escapeHtml(summary.target_label || "")}</dd>`,
    `<dt>频率</dt><dd>${escapeHtml(every)}</dd>`,
    `<dt>首次运行</dt><dd>${escapeHtml(summary.start_date || "")}</dd>`,
    `<dt>通知</dt><dd>${escapeHtml(NOTIFY_LABELS[summary.notify] || summary.notify || "")}</dd></dl>`,
    "</section>",
  ].join("");
}

export function conversationActionCardHtml(message, messages = []) {
  if (message?.role !== "assistant") return "";
  const action = actionOf(message);
  if (!action) return "";
  const settled = settledConversationActionIds(messages).has(String(message.id || ""));
  if (action.kind === "sql_query_proposal") return sqlProposalHtml(message, action, settled);
  if (action.kind === "sql_source_choice") return sqlSourceChoiceHtml(message, action, settled);
  if (action.kind === "sql_query_result") return sqlResultHtml(action);
  if (action.kind === "monitoring_schedule_form") return scheduleFormHtml(message, action, settled);
  if (action.kind === "monitoring_schedule_created") return scheduleCreatedHtml(action);
  return "";
}

const NUMBER_FIELDS = new Set(["interval", "advanced.label_maturity_days", "advanced.catch_up_budget", "advanced.max_attempts", "advanced.lease_seconds"]);

// Read the card's current values into the form shape the backend validates.
export function readScheduleForm(card) {
  const form = { advanced: {} };
  for (const control of card.querySelectorAll("[data-schedule-field]")) {
    const name = control.getAttribute("data-schedule-field");
    const text = String(control.value ?? "").trim();
    const value = text === "" ? null : NUMBER_FIELDS.has(name) ? Number(text) : text;
    if (name.startsWith("advanced.")) form.advanced[name.slice("advanced.".length)] = value;
    else form[name] = value;
  }
  return form;
}

// Change only the dependent controls: re-rendering the whole card would lose
// the user's edits to the frequency, start date and advanced settings.
export function handleConversationActionChange(event, ctx) {
  const control = event?.target;
  const fieldName = control?.getAttribute?.("data-schedule-field");
  if (!fieldName) return false;
  const card = control.closest?.('[data-conversation-action="monitoring_schedule_form"]');
  if (!card || control.disabled || control.closest("fieldset")?.disabled) return false;
  const messageId = card.getAttribute("data-message-id");
  const messages = ctx.getMessages?.() || [];
  if (settledConversationActionIds(messages).has(messageId)) return false;
  if (fieldName !== "dataset_id") {
    control.closest(".conversation-form-field")?.removeAttribute("data-invalid");
    const status = card.querySelector(".conversation-action-status");
    if (status) status.textContent = "";
    return true;
  }
  const message = messages.find((item) => String(item.id) === messageId);
  const datasets = actionOf(message)?.options?.datasets;
  if (!Array.isArray(datasets)) return false;
  const dataset = datasets.find((item) => String(item.id) === String(control.value));
  const columns = Array.isArray(dataset?.column_names) ? dataset.column_names : [];
  for (const name of ["time_col", "advanced.target_col"]) {
    const dependent = card.querySelector(`[data-schedule-field="${name}"]`);
    if (!dependent) continue;
    const previous = dependent.value;
    const value = columns.includes(previous) ? previous : "";
    dependent.innerHTML = [option("", "请选择", !value),
      ...columns.map((column) => option(column, column, column === value, { raw: true }))].join("");
    dependent.value = value;
    const label = dependent.closest(".conversation-form-field");
    if (name === "time_col" && !value) label?.setAttribute("data-invalid", "true");
    else label?.removeAttribute("data-invalid");
    if (name === "advanced.target_col" && previous && !value) {
      const maturity = card.querySelector('[data-schedule-field="advanced.label_maturity_days"]');
      if (maturity) maturity.value = "";
    }
  }
  const label = control.closest(".conversation-form-field");
  if (dataset) label?.removeAttribute("data-invalid");
  else label?.setAttribute("data-invalid", "true");
  const status = card.querySelector(".conversation-action-status");
  if (status) status.textContent = "";
  return true;
}

function errorText(error) {
  const detail = error?.detail ?? error?.payload?.detail;
  if (Array.isArray(detail?.fields)) return `请检查：${detail.fields.map((name) => FIELD_LABELS[name] || name).join("、")}`;
  if (detail?.message) return String(detail.message);
  if (detail?.next_action) return String(detail.next_action);
  return error?.message || "请求未完成";
}

function markInvalid(card, error) {
  const fields = error?.detail?.fields ?? error?.payload?.detail?.fields;
  if (!Array.isArray(fields)) return;
  for (const label of card.querySelectorAll(".conversation-form-field")) label.removeAttribute("data-invalid");
  for (const name of fields) card.querySelector(`[data-schedule-field="${CSS.escape(name)}"]`)?.closest(".conversation-form-field")?.setAttribute("data-invalid", "true");
}

// ctx: { getSelectedTaskId, api, refreshAgentMessages }
export async function handleConversationActionClick(event, ctx) {
  const target = event?.target;
  const control = target?.closest?.("[data-sql-conversation-confirm], [data-sql-conversation-source], [data-schedule-conversation-confirm]");
  if (!control || control.disabled) return false;
  const card = control.closest(".conversation-action-card");
  const status = card?.querySelector(".conversation-action-status");
  const taskId = ctx.getSelectedTaskId();
  if (!taskId || !card) return false;
  event.preventDefault();
  const post = (url, body) => ctx.api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const task = encodeURIComponent(taskId);
  control.disabled = true;
  if (status) status.textContent = "";
  try {
    if (control.hasAttribute("data-sql-conversation-confirm")) {
      await post(`/api/tasks/${task}/sql/conversation/confirm`, { message_id: control.getAttribute("data-sql-conversation-confirm"), confirmed: true });
    } else if (control.hasAttribute("data-sql-conversation-source")) {
      const message = (ctx.getMessages?.() || []).find((item) => String(item.id) === control.getAttribute("data-message-id"));
      const question = actionOf(message)?.question;
      if (!question) throw new Error("找不到原始取数描述，请在对话里重新描述。");
      await post(`/api/tasks/${task}/sql/conversation/drafts`, { question, source_id: control.getAttribute("data-sql-conversation-source") });
    } else {
      await post("/api/operations/conversation/schedules", { task_id: taskId, message_id: control.getAttribute("data-schedule-conversation-confirm"), form: readScheduleForm(card) });
    }
    await ctx.refreshAgentMessages(taskId);
  } catch (error) {
    control.disabled = false;
    markInvalid(card, error);
    if (status) status.textContent = errorText(error);
  }
  return true;
}
