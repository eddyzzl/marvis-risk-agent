import { api } from "./api.js";
import { escapeHtml } from "./ui-utils.js";
import { BUSINESS_OBJECTIVE_VERSION, EFFECT_STAGES, businessObjectiveFormHtml, collectBusinessObjective } from "./business-objective.js";

const STATUS = {
  passed: ["业务达标", "success"], failed: ["业务未达标", "danger"],
  insufficient_evidence: ["证据不足", "warning"], not_configured: ["未配置", "neutral"],
  not_applicable: ["不适用", "neutral"], pending: ["尚未评估", "neutral"],
  unsupported: ["结果版本不受支持", "warning"],
};
const show = value => escapeHtml(value == null || value === "" ? "未提供" : value);
const number = value => typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
const detail = (label, value) => `<div><dt>${label}</dt><dd>${show(value)}</dd></div>`;

export function businessAcceptanceProjection(plan, task = {}) {
  const configured = (Array.isArray(plan?.success_criteria) ? plan.success_criteria : []).find(item => item?.schema_version === BUSINESS_OBJECTIVE_VERSION)
    || task?.business_objective || task?.strategy_input?.business_objective || null;
  const raw = plan?.business_acceptance;
  const valid = raw?.schema_version === "business-acceptance.v1" && Object.hasOwn(STATUS, raw.status)
    && !["pending","unsupported"].includes(raw.status);
  const status = raw ? (valid ? raw.status : "unsupported") : configured ? "pending" : "not_configured";
  return {
    status, label: STATUS[status][0], tone: STATUS[status][1], result: valid ? raw : null,
    objective: valid ? raw.objective : configured,
    executionCompleted: typeof plan?.execution_completed === "boolean" ? plan.execution_completed : plan?.status === "done",
    planId: plan?.id || "", summaryRef: valid ? plan.summary_ref : "",
  };
}
export function businessAcceptanceHtml(plan, task = {}) {
  const view = businessAcceptanceProjection(plan, task);
  const { result, objective } = view;
  const reasons = result?.reasons || (view.status === "not_configured"
    ? ["尚未配置业务验收合同。执行完成只表示计划步骤完成，不代表业务达标。"]
    : view.status === "pending" ? ["合同已记录，等待实际采用对象的认证证据；不会使用候选最优值代替。"]
    : ["当前客户端无法识别此验收结果，请更新后重新查看；不会据此认定业务通过。"]);
  const criteria = result?.criteria?.length ? result.criteria : objective?.criteria || [];
  const rows = criteria.map(row => `<tr><th scope="row">${show(row.metric)}</th>
    <td>${number(row.value)}</td><td>${show(row.unit)}</td><td>${show(row.denominator)}</td>
    <td>${row.minimum == null ? "" : `≥ ${number(row.minimum)}`}${row.minimum != null && row.maximum != null ? " / " : ""}${row.maximum == null ? "" : `≤ ${number(row.maximum)}`}</td>
    <td>${row.comparison === "delta" ? `基线差值<br><small>${show(row.baseline_ref)} · 基线值 ${number(row.baseline_value)}</small>` : "实际值"}</td>
    <td>${show({passed:"数值达标",failed:"未达阈值",insufficient_evidence:"证据不足"}[row.status] || "待核对")}${row.reason ? `<small>${show(row.reason)}</small>` : ""}</td></tr>`).join("");
  const evidence = result?.evidence;
  const target = result?.target;
  const downloads = view.summaryRef && view.planId ? `<nav class="business-acceptance-downloads" aria-label="业务验收导出"><button type="button" class="button compact secondary" data-business-download="xlsx" data-business-plan="${show(view.planId)}" data-business-summary="${show(view.summaryRef)}">下载验收 Excel</button><button type="button" class="button compact secondary" data-business-download="docx" data-business-plan="${show(view.planId)}" data-business-summary="${show(view.summaryRef)}">下载验收 Word</button></nav>` : "";
  const editable = task?.business_objective_locked === false && !plan;
  const controls = editable ? `<button type="button" class="button compact secondary" data-business-edit>${objective ? "编辑业务合同" : "配置业务合同"}</button>` : '<small class="business-note">计划生成后合同已冻结；调整标准需要新任务。</small>';
  return `<header class="business-acceptance-head"><div><span>业务验收</span><strong class="business-verdict ${view.tone}" data-business-verdict="${view.status}">${view.label}</strong></div><small>计划执行：${view.executionCompleted ? "已完成" : "尚未完成"}</small></header>
    <p class="business-note">执行状态与业务达标分别判断；业务结论只取平台保存的认证验收结果。</p>
    ${reasons.length ? `<ul class="business-acceptance-reasons">${reasons.map(reason=>`<li>${show(reason)}</li>`).join("")}</ul>` : ""}
    ${controls}<div data-business-editor hidden></div><p data-business-action-error class="business-action-error" role="alert"></p>
    ${objective ? `<details class="business-objective-review"><summary>查看业务合同、指标与证据</summary>
      <dl class="business-evidence-grid">${detail("业务线 / 决策节点",`${objective.business_line} / ${objective.decision_node}`)}${detail("目标客群",objective.population)}${detail("约定时期",`${objective.period_start} — ${objective.period_end}`)}${detail("业务确认来源",objective.responsibility_source)}${detail("币种",objective.currency || "非货币口径")}${detail("最低证据阶段",EFFECT_STAGES[objective.minimum_effect_stage] || objective.minimum_effect_stage)}${detail("标签成熟要求",objective.require_mature_labels ? "必须为成熟的真实观察标签" : "合同未要求成熟标签")}${detail("结论边界",objective.claim === "causal" ? "因果增量（需因果识别证据）" : "观察性结论")}</dl>
      ${rows ? `<div class="business-criteria-scroll"><table class="business-criteria-table"><caption>业务指标核对（数值达标不替代证据完整性）</caption><thead><tr><th>指标</th><th>实际值</th><th>单位</th><th>分母 / 范围</th><th>阈值</th><th>比较方式</th><th>核对结果</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
      <dl class="business-evidence-grid">${detail("实际采用对象",target ? `${target.kind} · ${target.id}` : "尚无认证采纳结果")}${detail("实际采用版本",target?.version)}${detail("约定对象 / 版本",`${objective.target_id || "由实际采纳结果绑定"} / ${objective.target_version || "由实际采纳结果绑定"}`)}${detail("证据业务线 / 节点",evidence ? `${evidence.business_line || "未提供"} / ${evidence.decision_node || "未提供"}` : null)}${detail("证据观察时期",evidence?.period_start && evidence?.period_end ? `${evidence.period_start} — ${evidence.period_end}` : null)}${detail("实际证据阶段",EFFECT_STAGES[evidence?.effect_stage] || evidence?.effect_stage)}${detail("实际标签成熟度",evidence?.labels_mature === true ? "已成熟" : evidence?.labels_mature === false ? "未成熟" : "未认证")}${detail("标签来源",evidence?.label_origin)}${detail("证据引用",evidence?.source_ref)}${detail("证据内容指纹",evidence?.source_hash)}${detail("基线证据引用",evidence?.baseline_ref)}${detail("基线绑定指纹",evidence?.baseline_binding_hash)}${detail("合同内容指纹",result?.objective_hash)}${detail("合同 / 结果版本",`${objective.schema_version} / ${result?.schema_version || "尚无结果"}`)}</dl>
      ${downloads}</details>` : downloads}`;
}
export function renderBusinessAcceptance(element, plan, task) {
  if (!element) return;
  element.hidden = !task;
  if (!task) { element.innerHTML = ""; delete element.dataset.signature; return; }
  const signature = JSON.stringify([task.id, plan?.id, plan?.status, plan?.execution_completed, plan?.summary_ref, plan?.business_acceptance, plan?.success_criteria, task.business_objective, task.strategy_input?.business_objective, task.business_objective_locked]);
  if (element.dataset.signature === signature) return;
  const open = element.querySelector?.(".business-objective-review")?.open && element.dataset.taskId === task.id;
  element.innerHTML = businessAcceptanceHtml(plan, task);
  element.dataset.signature = signature; element.dataset.taskId = task.id;
  if (open && element.querySelector(".business-objective-review")) element.querySelector(".business-objective-review").open = true;
}

// The session owns task/plan identity. This controller only holds in-flight
// requests and form DOM; a response cannot update a later visit or model.
export function createBusinessAcceptanceController({
  getElement = () => document.getElementById("businessAcceptancePanel"),
  getTask, getPlan, captureView, isCurrentView,
  apiClient = api, onSaved = () => {}, setActionStatus = () => {},
  beginActivity = () => null, endActivity = () => {},
  downloadBlob = (blob, filename) => {
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href; anchor.download = filename;
    document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  },
} = {}) {
  const pending = new Set();
  function current(taskId, view) {
    return isCurrentView(view) && getTask()?.id === taskId;
  }
  function error(panel, message) {
    const target = panel?.querySelector("[data-business-action-error]");
    if (target) target.textContent = message;
  }
  function canEdit(task) {
    return task?.business_objective_locked === false && !getPlan();
  }
  async function save(button, panel) {
    const task = getTask();
    if (!task || pending.has(`save:${task.id}`)) return;
    if (!canEdit(task)) { error(panel, "合同已冻结，请刷新任务后查看。"); return; }
    let objective;
    try { objective = collectBusinessObjective(panel.querySelector("[data-business-editor] [data-business-objective]")); }
    catch (failure) { error(panel, failure.message); return; }
    const view = captureView();
    const key = `save:${task.id}`;
    pending.add(key); button.disabled = true; error(panel, "");
    const activity = beginActivity(key, task.id);
    try {
      const response = await apiClient(`/api/tasks/${encodeURIComponent(task.id)}/business-objective`, {
        method: "PUT", body: JSON.stringify({ business_objective: objective }),
      });
      if (!current(task.id, view)) return;
      await onSaved(response?.task || response, view);
      if (current(task.id, view)) setActionStatus("业务合同已保存，后续计划将使用这一版本。", "success");
    } catch (failure) {
      if (current(task.id, view)) {
        error(panel, failure.message || "业务合同保存失败。");
        setActionStatus(failure.message || "业务合同保存失败。", "error");
      }
    } finally {
      pending.delete(key); endActivity(activity);
      if (current(task.id, view)) button.disabled = false;
    }
  }
  async function download(button, panel) {
    const task = getTask(), plan = getPlan();
    const format = button.dataset.businessDownload;
    const planId = button.dataset.businessPlan;
    const summary = button.dataset.businessSummary;
    const matches = () => getPlan()?.id === planId && getPlan()?.summary_ref === summary;
    if (!task || !plan || !["xlsx", "docx"].includes(format) || !matches()) return;
    const key = `download:${planId}:${format}`;
    if (pending.has(key)) return;
    const view = captureView();
    pending.add(key); button.disabled = true; error(panel, "");
    try {
      const blob = await apiClient(`/api/plans/${encodeURIComponent(planId)}/business-acceptance/${format}?expected_summary_ref=${encodeURIComponent(summary)}`, { responseType: "blob" });
      if (current(task.id, view) && matches()) downloadBlob(blob, `business-acceptance-${planId}.${format}`);
    } catch (failure) {
      if (current(task.id, view) && matches()) error(panel, failure.message || "业务验收导出失败。");
    } finally {
      pending.delete(key);
      if (current(task.id, view) && matches()) button.disabled = false;
    }
  }
  function handleClick(event) {
    const button = event.target?.closest?.("[data-business-edit], [data-business-cancel], [data-business-save], [data-business-download]");
    const panel = getElement();
    if (!button || !panel?.contains(button)) return false;
    event.preventDefault();
    if (button.hasAttribute("data-business-download")) { void download(button, panel); return true; }
    if (button.hasAttribute("data-business-save")) { void save(button, panel); return true; }
    const editor = panel.querySelector("[data-business-editor]");
    if (!editor) return false;
    if (button.hasAttribute("data-business-cancel")) { editor.innerHTML = ""; editor.hidden = true; error(panel, ""); return true; }
    const task = getTask();
    if (!canEdit(task)) { error(panel, "合同已冻结，请刷新任务后查看。"); return true; }
    const targetKind = task.task_type === "modeling" ? "model" : task.task_type === "strategy" ? "strategy" : "workflow";
    editor.innerHTML = businessObjectiveFormHtml(task.business_objective || task.strategy_input?.business_objective, { targetKind })
      + '<div class="business-acceptance-downloads"><button type="button" class="button compact primary" data-business-save>保存业务合同</button><button type="button" class="button compact secondary" data-business-cancel>取消</button></div>';
    editor.hidden = false;
    const details = editor.querySelector("[data-business-objective]");
    if (details) details.open = true;
    error(panel, "");
    return true;
  }
  return Object.freeze({ handleClick });
}
