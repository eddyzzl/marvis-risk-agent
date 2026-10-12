import { api as defaultApi } from "./api.js";
import { createContractForm } from "./contract-form.js";

const path = encodeURIComponent;
const element = (document, tag, text = "") => { const node = document.createElement(tag); node.textContent = text; return node; };
const note = (document, text) => { const node = element(document, "p", text); node.className = "history-note"; return node; };
const titles = { expected_revision: "已审核的当前版本", instruction: "调度说明（可选）", schedule: "调度合同", schedule_id: "调度名称", revision: "新版本", monitoring_ref: "已安装执行能力", active_from: "首次生效时间", calendar: "执行日历", anchor_at: "窗口起点（含时区）", interval_seconds: "窗口秒数", interval_months: "间隔月数", timezone: "时间区域", catch_up_budget: "单次最多补跑期数", lease_seconds: "执行租约秒数", retry_policy: "失败重试规则", max_attempts: "最多尝试次数", initial_backoff_seconds: "首次等待秒数", multiplier: "等待倍数", max_backoff_seconds: "最长等待秒数", enabled: "启用调度", monitoring_binding: "模型或策略来源", portfolio_binding: "组合口径", feedback_binding: "在线决策反馈口径", active_until: "停止时间（可选）", source: "快照与时间声明", dataset_content_hash: "快照指纹", time_col: "事件时间列", complete_through: "来源声明完整覆盖到", target_id: "已登记模型或策略编号", target_col: "结果标签列", label_mode: "标签口径", label_maturity_seconds: "标签成熟等待秒数", score_col: "分数列（可选）", max_source_rows: "来源记录上限", id_col: "贷款编号列", bucket_col: "逾期状态列", states: "按风险顺序排列的状态", balance_col: "余额列", segment_col: "客群列", loss_state: "损失状态", lgd: "损失比例假设", horizon_months: "损失估计期限月数", lookback_seconds: "历史回看秒数", experiment_id: "已登记模型编号（可选）", generate_report: "生成原生组合报告", environment: "实际决策环境", package_hash: "实际已签名决策包指纹", min_labeled_count: "最少已成熟标签数", max_bad_rate: "坏账率红线", max_fallback_rate: "降级率红线", request_id_col: "决策请求编号列", output_hash_col: "决策输出指纹列", package_hash_col: "决策包指纹列", label_col: "实际二元结果列", event_at_col: "结果事件时间列", available_at_col: "结果可得时间列", period_key: "本期窗口编号", expected_outcome_hash: "本期已审核结果指纹", idempotency_key: "本次处置请求编号", expected_plan_id: "本期原生监控计划", expected_plan_revision: "原生监控计划版本", expected_plan_hash: "原生监控计划指纹", disposition: "人工处置决定", reason: "处置理由", threshold_patch: "明确调整的阈值", warn: "提醒阈值", fail: "红线阈值" };
Object.assign(titles, { schedules: "调度", feedback_packages: "决策反馈" });
function localizedSchema(schema) {
  const clone = JSON.parse(JSON.stringify(schema));
  function visit(value) { if (!value || typeof value !== "object") return; if (value.properties) for (const [key, spec] of Object.entries(value.properties)) { if (titles[key]) spec.title = titles[key]; } Object.values(value).forEach((v) => { if (Array.isArray(v)) v.forEach(visit); else if (typeof v === "object") visit(v); }); }
  visit(clone); return clone;
}
function taskOf(contract) { return contract.monitoring_binding?.task_id || contract.portfolio_binding?.source?.task_id || contract.feedback_binding?.task_id; }
function plainInputContract(contract) { const value = JSON.parse(JSON.stringify(contract)); delete value.schema_version; delete value.execution_actor_id; delete value.runtime_binding_hash; delete value.retry_policy.schema_version; delete value.calendar.schema_version; return value; }
function facts(document, object) {
  const table = element(document, "table"); table.className = "contract-review-table";
  function append(value, prefix = "") { for (const [key, item] of Object.entries(value || {})) { const label = prefix ? `${prefix} · ${titles[key] || key}` : titles[key] || key; if (item && typeof item === "object" && !Array.isArray(item)) append(item, label); else { const row = element(document, "tr"); row.append(element(document, "th", label), element(document, "td", item == null ? "未声明 / 未知" : Array.isArray(item) ? item.join("、") : typeof item === "boolean" ? item ? "是" : "否" : String(item))); table.append(row); } } }
  append(object); return table;
}

export function createOperationsExtensionsPanel({ getElement, getTask, captureView = () => null, isCurrentView = () => true, api = defaultApi, onSchedulePublished = async () => {} }) {
  let state = null;
  const current = (owner) => owner === state && owner.taskId === getTask()?.id && isCurrentView(owner.view);
  const post = (url, value = {}) => api(url, { method: "POST", body: JSON.stringify(value) });
  function status(owner, text) { if (current(owner)) owner.status.textContent = text; }
  function button(owner, text, action, parent = owner.actions) { const node = element(owner.document, "button", text); node.type = "button"; node.className = "button compact secondary"; parent.append(node); node.addEventListener("click", async () => { if (!current(owner) || owner.pending) return; owner.pending = true; node.disabled = true; try { await action(); } catch (e) { status(owner, e?.detail?.next_action || e?.detail?.code || e.message || "请求失败，已填写信息保留。"); } finally { owner.pending = false; if (current(owner)) node.disabled = false; } }); return node; }
  function render() {
    const root = getElement(), task = getTask(); if (!root) return;
    root.hidden = !task?.id; if (root.hidden) { state = null; root.replaceChildren(); return; }
    if (state && current(state)) return;
    const document = root.ownerDocument, details = element(document, "details"), body = element(document, "div"); body.className = "risk-context-body"; details.append(element(document, "summary", "定时分析与反馈处置"), body);
    body.append(note(document, "先校验来源、窗口和统计口径，再人工确认调度。红色结果需要另行明确处置；来源覆盖与标签成熟保留登记方声明。"));
    const actions = element(document, "div"), message = note(document, "准备当前身份和任务来源后，可配置调度或回传结果。"); actions.className = "history-actions"; message.setAttribute("role", "status"); const editor = element(document, "div"), result = element(document, "div"), listing = element(document, "div"); body.append(actions, message, editor, result, listing); root.replaceChildren(details);
    const owner = state = { taskId: task.id, view: captureView(), document, actions, status: message, editor, result, listing, pending: false, edit: 0 };
    button(owner, "准备与刷新", () => load(owner));
  }
  async function load(owner) {
    const [schemas, sources, principal, schedules, feedback] = await Promise.all([api("/api/operations/extensions/contracts"), api(`/api/operations/extensions/sources?task_id=${path(owner.taskId)}`), api("/api/production-governance/me"), api("/api/operations/schedules?include_disabled=true"), api(`/api/operations/decision-feedback?task_id=${path(owner.taskId)}`)]);
    if (!current(owner)) return;
    Object.assign(owner, { schemas, datasets: sources.datasets, principal, schedules: schedules.schedules.filter((s) => taskOf(s.contract) === owner.taskId) });
    owner.listing.replaceChildren(element(owner.document, "h4", "已登记状态"), facts(owner.document, { schedules: owner.schedules.map((s) => `${s.contract.schedule_id} · 版本 ${s.contract.revision} · ${s.contract.enabled ? "启用" : "停止"}`), feedback_packages: feedback.packages.map((p) => `${p.package_hash} · ${p.recorded_count} 条结果`) }));
    renderEditor(owner); status(owner, "来源与身份已读取。自然语言仅支持明确间隔和含时区起点，信息不足会要求补充。");
  }
  function schemaForm(owner, schema, initial, title, onSubmit, edit) {
    const choices = { dataset_id: owner.datasets.map((d) => ({ value: d.id, label: `${d.id} · ${d.row_count} 行` })), monitoring_ref: ["analysis.portfolio_run", "analysis.decision_ledger_feedback", "modeling.monitor_run", "strategy.run_strategy_monitoring"].map((value) => ({ value, label: { "analysis.portfolio_run": "组合分析", "analysis.decision_ledger_feedback": "在线决策反馈", "modeling.monitor_run": "模型监控", "strategy.run_strategy_monitoring": "策略监控" }[value] })) };
    const form = createContractForm({ document: owner.document, schema: localizedSchema(schema), initial, choices, title, submitLabel: title.includes("调度") ? "校验并准备审核" : "确认本次登记", onSubmit: async (payload) => { if (!current(owner) || owner.edit !== edit) throw new Error("任务或表单已变化，请重新准备。"); const answer = await onSubmit(payload); if (!current(owner) || owner.edit !== edit) return; owner.result.replaceChildren(facts(owner.document, answer)); } });
    owner.slot.replaceChildren(form.element); return form;
  }
  function renderEditor(owner) {
    const { document } = owner; owner.editor.replaceChildren(); owner.result.replaceChildren(); owner.edit++;
    if (!["maker", "admin"].includes(owner.principal.role)) { owner.editor.append(note(document, "当前身份可查看已授权证据；发布与处置需要人工发起人或管理员身份。")); return; }
    const select = element(document, "select"); select.setAttribute("aria-label", "调度与反馈操作");
    for (const [value, text] of [["", "请选择操作"], ["portfolio", "月度组合分析"], ["modeling", "定时模型监控"], ["strategy", "定时策略监控"], ["feedback-schedule", "定时在线决策反馈"], ["feedback-import", "回传已登记结果快照"], ["revise", "调整已有调度"], ["disposition", "处置本期红色结果"], ["reconcile", "核对未返回的处置证据"]]) { const option = element(document, "option", text); option.value = value; select.append(option); }
    owner.slot = element(document, "div"); owner.editor.append(select, owner.slot);
    select.addEventListener("change", () => { const edit = ++owner.edit; owner.result.replaceChildren(); owner.slot.replaceChildren(); if (select.value) void setup(owner, select.value, edit).catch((e) => status(owner, e.message)); });
  }
  async function setup(owner, mode, edit) {
    if (!current(owner) || edit !== owner.edit) return;
    if (mode === "feedback-import") { const d = owner.datasets[0]; schemaForm(owner, owner.schemas.feedback_import_schema, { task_id: owner.taskId, dataset_id: d?.id, dataset_content_hash: d?.content_hash, environment: "local-reference:production", timezone: "Asia/Shanghai" }, "登记真实决策结果", (v) => post("/api/operations/decision-feedback", v), edit); return; }
    if (["disposition", "reconcile", "revise"].includes(mode)) { await chooseExisting(owner, mode, edit); return; }
    const today = new Date(); today.setUTCDate(1); today.setUTCHours(0, 0, 0, 0); const anchor = today.toISOString();
    const d = owner.datasets[0], monthly = mode === "portfolio", source = { task_id: owner.taskId, dataset_id: d?.id, dataset_content_hash: d?.content_hash, timezone: "UTC", label_mode: monthly ? "not_applicable" : "required" };
    const initial = { expected_revision: 0, schedule: { schedule_id: `${mode}-${owner.taskId.slice(0, 8)}`, revision: 1, monitoring_ref: { portfolio: "analysis.portfolio_run", modeling: "modeling.monitor_run", strategy: "strategy.run_strategy_monitoring", "feedback-schedule": "analysis.decision_ledger_feedback" }[mode], active_from: anchor, calendar: monthly ? { anchor_at: anchor, interval_months: 1, timezone: "UTC" } : { anchor_at: anchor, interval_seconds: 86400 }, catch_up_budget: 1, lease_seconds: 600, retry_policy: { max_attempts: 1, initial_backoff_seconds: 0, multiplier: 1, max_backoff_seconds: 0 }, enabled: true } };
    if (monthly) initial.schedule.portfolio_binding = { source: { ...source, target_id: "portfolio_snapshots" }, horizon_months: 1, lookback_seconds: 90 * 86400, generate_report: true };
    else if (mode === "feedback-schedule") initial.schedule.feedback_binding = { task_id: owner.taskId, environment: "local-reference:production" };
    else initial.schedule.monitoring_binding = source;
    proposalForm(owner, initial, monthly, edit);
  }
  function proposalForm(owner, initial, monthly, edit) {
    initial = { ...initial, instruction: initial.instruction || undefined };
    const schema = JSON.parse(JSON.stringify(owner.schemas.proposal_schema));
    // Both branches are installed contracts. The explicit operation selects its
    // calendar form and never rewrites the server validation contract.
    schema.$defs.SchedulePayload.properties.calendar = { $ref: monthly ? "#/$defs/MonthlyCalendarPayload" : "#/$defs/CalendarPayload" };
    schemaForm(owner, schema, initial, "准备调度合同", async (value) => {
      const proposal = await post("/api/operations/proposals", value);
      if (!current(owner) || edit !== owner.edit) return {};
      owner.slot.replaceChildren();
      if (proposal.state === "clarification") { status(owner, `需要补充：${proposal.missing_fields.join("、")}。调度尚未发布。`); return proposal; }
      const actions = element(owner.document, "div"); actions.className = "history-actions";
      owner.slot.append(element(owner.document, "h4", "确认此份冻结合同"), facts(owner.document, proposal.contract), note(owner.document, `合同指纹 ${proposal.contract_hash} · 有效期至 ${proposal.expires_at}`), actions);
      button(owner, "确认发布这份调度", async () => { if (edit !== owner.edit || !current(owner)) return; const result = await post(`/api/operations/proposals/${path(proposal.proposal_id)}/confirm`, { expected_contract_hash: proposal.contract_hash }); if (!current(owner) || edit !== owner.edit) return; owner.edit++; owner.slot.replaceChildren(note(owner.document, `已发布 ${result.schedule_id} · 版本 ${result.revision}`)); owner.result.replaceChildren(facts(owner.document, result)); await onSchedulePublished(result); status(owner, "已发布。实际结果将在该窗口完成且来源就绪后生成。"); }, actions);
      return { state: proposal.state, published: false, automatic_action_permitted: false };
    }, edit);
  }
  async function chooseExisting(owner, mode, edit) {
    const select = element(owner.document, "select"); select.setAttribute("aria-label", "已登记调度"); select.append(element(owner.document, "option", "请选择已登记调度"));
    owner.schedules.forEach((s) => { const option = element(owner.document, "option", `${s.contract.schedule_id} · ${s.contract.revision}`); option.value = s.contract.schedule_id; select.append(option); }); owner.slot.append(select);
    select.addEventListener("change", async () => {
      if (!current(owner) || edit !== owner.edit) return;
      const schedule = owner.schedules.find((s) => s.contract.schedule_id === select.value); if (!schedule) return;
      const nextEdit = ++owner.edit;
      if (mode === "revise") { const contract = plainInputContract(schedule.contract); contract.revision++; proposalForm(owner, { expected_revision: schedule.contract.revision, schedule: contract }, !!contract.calendar.interval_months, nextEdit); return; }
      if (mode === "reconcile") { const input = element(owner.document, "input"), actions = element(owner.document, "div"); input.setAttribute("aria-label", "原处置请求编号"); actions.className = "history-actions"; owner.slot.replaceChildren(input, actions); button(owner, "只核对原生处置证据", async () => { const answer = await post(`/api/operations/dispositions/${path(input.value.trim())}/reconcile`); if (current(owner) && nextEdit === owner.edit) owner.result.replaceChildren(facts(owner.document, answer)); }, actions); return; }
      const periods = await api(`/api/operations/schedules/${path(schedule.contract.schedule_id)}/periods`);
      if (!current(owner) || nextEdit !== owner.edit) return;
      owner.slot.replaceChildren();
      for (const item of periods.periods) {
        const key = item.period.period.key;
        const period = await api(`/api/operations/schedules/${path(schedule.contract.schedule_id)}/period?period_key=${path(key)}`);
        if (!current(owner) || nextEdit !== owner.edit) return;
        if (period.outcome?.outcome.level !== "red") continue;
        const evidence = await api(`/api/operations/schedules/${path(schedule.contract.schedule_id)}/evidence?period_key=${path(key)}`);
        if (!current(owner) || nextEdit !== owner.edit) return;
        if (evidence.monitoring_ref !== "strategy.run_strategy_monitoring") continue;
        const output = evidence.tool_output;
        schemaForm(owner, owner.schemas.disposition_schema, { period_key: key, expected_outcome_hash: period.outcome.outcome_hash, expected_plan_id: output.monitoring_plan_id, expected_plan_revision: output.monitoring_plan_revision, expected_plan_hash: output.monitoring_plan_hash, disposition: "observe" }, "明确本期人工处置", (v) => post(`/api/operations/schedules/${path(schedule.contract.schedule_id)}/dispositions`, v), nextEdit);
        return;
      }
      owner.slot.append(note(owner.document, "当前没有可由此入口处置的策略红色窗口。模型、组合和在线反馈的红色提醒保留人工复核。"));
    });
  }
  function destroy() { state = null; getElement()?.replaceChildren(); }
  return { render, destroy, refresh: async () => { render(); if (state && current(state)) await load(state); } };
}
