import { api } from "./api.js";
import { escapeHtml } from "./ui-utils.js";
const esc = (v) => escapeHtml(String(v ?? ""));
const states = { sealed: "已封存 · 开发中", frozen: "最终模型已冻结", prepared: "评估已预留", reading: "读取已开始 · 不可重复", committed: "单次评估已完成", uncertain: "评估中断 · 需保管人员处理" };
const errors = {
  blind_intake_must_precede_development: "该任务已开始开发。请新建空白建模任务，在注册任何开发材料前封存独立 OOT。",
  already_exposed_source_cannot_be_blind: "该原始材料曾进入普通开发路径，不能重新认定为盲 OOT。请使用从未暴露的新独立材料。",
  sealed_holdout_access_forbidden: "材料与已封存 OOT 的受保护身份重合，不能进入普通开发或预览。请使用独立的开发材料。",
  holdout_admission_identity_unverifiable: "当前来源无法在受保护范围内核对身份。为防止封存样本进入开发路径，已停止读取；请提供受支持且可验证的开发材料。",
  prior_exposure_provenance_unknown: "已有公开来源的身份记录不足以排除重合，不能宣称本批次为盲测。请使用可核对且从未暴露的新独立材料。",
  holdout_identity_already_reserved: "该材料或批次已封存。请打开原任务，原批次的评估次数不能重置。",
  native_model_receipt_missing_or_invalid: "模型缺少有效的原生训练证据。请在本任务使用内置训练工具生成模型后再冻结。",
  blind_development_requires_train_test_only: "开发材料只能包含 train/test。请移除 OOT 行，并保持独立封存材料不进入开发流程。",
  blind_development_requires_train_and_test: "请提供同时包含 train 和 test 的开发材料。",
  blind_development_schema_invalid: "开发材料缺少已声明的目标或切分字段。请检查字段后重新上传。",
  holdout_resource_profile_exceeded: "材料超过当前支持范围：64 MiB 文件、128 MiB 解码大小、100 万行、128 列。请由保管人员准备新的独立批次。",
  holdout_upload_too_large: "文件超过当前封存大小上限。请由保管人员准备符合支持范围的新批次。",
  holdout_role_forbidden: "当前身份缺少有效权限。请先绑定工作区身份，再由保管人员确认分工。",
  holdout_scope_forbidden: "当前身份未绑定此批次。请切换至该批次已指定的工作区身份。",
  holdout_evaluation_busy_or_recovery_required: "评估已预留或中断，次数不会因重复点击而增加。请刷新状态；中断时由保管人员审批安全恢复。",
  holdout_cannot_repeat_sample_read: "封存数据已读取，禁止重新评估。仅可恢复已保存且签名有效的结果；若无结果，本批次保持中断状态。",
  holdout_evaluation_failed_recovery_required: "评估中断，评估次数仍被保留。请由保管人员查看可用恢复动作。",
  final_model_freeze_immutable: "已冻结另一个最终模型，不能替换。请继续原模型的独立评估。",
  holdout_freeze_binding_failed: "模型冻结版本已变化。请刷新状态后继续。",
  holdout_result_integrity_failed: "结果完整性检查失败。请保留现场并联系保管人员；不可重新读取封存数据。",
};
export function holdoutError(error) {
  return errors[error?.detail?.code] || "操作未完成。请刷新状态，检查当前身份、材料契约及可用动作后继续。";
}
export function blindOotHtml({ summary, principal, participants = [], candidates = [], result, error = "", busy = false }) {
  const role = summary?.viewer_role || (principal?.role === "admin" ? "custody" : "unbound");
  const disabled = busy ? " disabled" : "";
  let body = "";
  if (!principal) body = '<p class="production-note">先绑定工作区身份，再由保管人员指定开发和复核人员。</p><button type="button" data-blind-action="identity" class="button compact secondary">绑定工作区身份</button>';
  else if (!summary && role === "custody") {
    const options = (r) => participants.filter(p => p.role === r).map(p => `<option value="${esc(p.id)}">${esc(p.display_name)}</option>`).join("");
    body = `<p class="production-note">仅适用于尚未注册材料、训练或生成计划的空白建模任务。独立 OOT 先封存，开发人员随后只上传 train/test。</p><form data-blind-form="seal"><div class="production-grid"><label>独立批次标识<input name="data_identity" required maxlength="160" pattern="[A-Za-z0-9_.:-]+" placeholder="机构已授权的批次标识" /></label><label>独立 OOT 文件<input name="file" type="file" required accept=".parquet" /></label><label>开发人员<select name="developer_id" required>${options("maker")}</select></label><label>复核人员<select name="evaluator_id" required>${options("checker")}</select></label><label>目标字段<input name="target_col" required value="y" /></label><label>开发集切分字段<input name="split_col" required value="split" /></label><label>目标类型<select name="target_type"><option value="binary">二分类</option><option value="continuous">回归</option><option value="multiclass">多分类</option></select></label><label>授权依据<input name="authority_basis" required maxlength="2000" placeholder="记录材料来源及授权依据" /></label></div><button class="button compact primary"${disabled}>封存独立 OOT</button></form>`;
  } else if (!summary) body = '<p class="production-note">该任务尚未启用盲 OOT。由保管人员在开发前封存新材料并绑定人员；现有建模流程保持可用。</p>';
  else if (role === "develop" && summary.state === "sealed") {
    const options = candidates.map(c => `<option value="${esc(c.artifact_id)}">${esc(c.algorithm)} · ${esc(String(c.created_at).slice(0, 19))}</option>`).join("");
    body = `<div class="production-grid"><section><h4>1. 只上传开发集</h4><form data-blind-form="development"><label>开发材料（train/test）<input name="file" type="file" accept=".parquet" required /></label><button class="button compact secondary"${disabled}>登记开发集</button></form><p class="production-note">登记后在当前任务与 Agent 完成特征选择、训练和调参。内置工具只使用开发集。</p></section><section><h4>2. 冻结最终模型</h4><form data-blind-form="freeze"><label>本任务原生训练模型<select name="artifact_id" required>${options || '<option value="">尚无可冻结模型</option>'}</select></label><p class="production-note">冻结将固定模型、参数、特征和预处理依赖；之后不可替换模型或继续训练。</p><button class="button compact primary"${disabled || (!options ? " disabled" : "")}>冻结最终模型</button></form></section></div>`;
  } else if (role === "evaluate") {
    if (summary.state === "frozen" || summary.recovery_id) body = `<p class="production-note">按冻结版本评估一次。重复请求只返回同一份结果。</p><button type="button" data-blind-action="evaluate" class="button compact primary"${disabled}>${summary.recovery_id ? "执行已批准的安全恢复" : "评估封存集一次"}</button>`;
    else if (summary.state === "sealed") body = '<p class="production-note">等待开发人员冻结最终模型，冻结前封存样本不会读取。</p>';
    else if (summary.state !== "committed") body = '<p class="production-note">请刷新状态。评估中断需保管人员审批；已读取的封存样本不可重新评估。</p>';
    if (result) body += `<h4>独立评估结果</h4><table class="blind-oot-metrics"><tbody>${Object.entries(result.metrics).map(([key, value]) => `<tr><th>${esc({rows:"样本量",ks:"KS",auc:"AUC",rmse:"RMSE",mae:"MAE",r2:"R²",logloss:"Logloss",macro_auc:"Macro AUC"}[key] || key)}</th><td>${esc(typeof value === "number" ? (key === "rows" ? value : value.toFixed(6)) : value)}</td></tr>`).join("")}</tbody></table><p class="production-note">这是确定性技术评估；历史可用性与标签成熟度未知，机构授权依据仅为保管人员声明。</p>`;
  } else if (role === "custody" && ["prepared", "reading", "uncertain"].includes(summary.state)) {
    const options = summary.recovery_options || [];
    body = options.length ? `<form data-blind-form="recover"><label>安全恢复方式<select name="mode">${options.map(m => `<option value="${m}">${m === "resume_before_read" ? "在尚未读取样本时继续" : "发布已签名保存的结果"}</option>`).join("")}</select></label><label>处理依据<input name="reason" required maxlength="2000" /></label><button class="button compact secondary"${disabled}>批准安全恢复</button></form><p class="production-note">审批有效期 15 分钟，随后由绑定复核人员执行；不能增加评估次数。</p>` : '<p class="production-note">没有安全恢复动作。封存数据已读取且未保存可验证结果时，本批次保持中断，不能再次评估。</p>';
  } else body = `<p class="production-note">${summary.state === "committed" ? "复核人员已完成单次评估。开发人员和 Agent 不接收封存样本或指标。" : summary.state === "frozen" ? "等待绑定复核人员完成独立评估。" : "等待开发人员完成开发并冻结最终模型。"}</p>`;
  const evidence = summary?.model_evidence ? `<p class="production-note">冻结证据：${esc(summary.model_evidence.algorithm)} · ${summary.model_evidence.feature_count} 个特征 · 原生训练凭据已验证 · 模型依赖已固定</p><details><summary>查看冻结指纹</summary><code>${esc(summary.freeze_hash)}</code></details>` : "";
  return `<header class="blind-oot-header"><div><strong>盲 OOT</strong><p class="production-state">${esc(summary ? states[summary.state] || summary.state : "可选 · 开发前启用")}</p></div><button type="button" data-blind-action="refresh" class="button compact secondary"${disabled}>刷新状态</button></header><p class="production-error" role="alert">${esc(error)}</p>${evidence}${body}<p class="production-boundary">保管入口仅为防重复封存核对材料身份，样本不进入 Agent、普通数据登记或 Notebook；每个独立批次只允许一次评估。本模式依赖可信本机账号；不能隔离同一系统账号直接访问磁盘。历史可用性、标签成熟度及外部身份核验仍未知。当前封存支持 64 MiB 文件、128 MiB 解码大小、100 万行、128 列。</p>`;
}
export function createBlindOotController({ root, getTask, openIdentity, onStateChange = async () => {} }) {
  let epoch = 0, data = {}, busy = false, taskId = "", refreshedAt = 0;
  const endpoint = (suffix = "") => `/api/tasks/${encodeURIComponent(taskId)}/blind-oot${suffix}`;
  const render = () => { if (root) root.innerHTML = blindOotHtml({ ...data, busy }); };
  async function refresh({ force = false } = {}) {
    const task = getTask();
    if (!root) return;
    root.hidden = task?.task_type !== "modeling";
    if (root.hidden) { epoch++; taskId = ""; return; }
    if (busy) {
      if (task.id !== taskId) root.innerHTML = '<p class="production-note">正在加载当前批次状态…</p>';
      return;
    }
    if (!force && taskId === task.id && Date.now() - refreshedAt < 5000) return;
    refreshedAt = Date.now();
    taskId = task.id; const captured = taskId, version = ++epoch;
    const next = { error: "" };
    try {
      try { next.principal = await api("/api/production-governance/me"); } catch { next.principal = null; }
      if (next.principal) {
        try { next.summary = await api(endpoint()); } catch (e) { if (e.status !== 404) throw e; }
        if (!next.summary && next.principal.role === "admin") next.participants = (await api(endpoint("/participants"))).participants;
        if (next.summary?.viewer_role === "develop" && next.summary.state === "sealed") next.candidates = (await api(endpoint("/candidates"))).candidates;
        if (next.summary?.viewer_role === "evaluate" && next.summary.state === "committed") next.result = await api(endpoint("/result"));
      }
    } catch (e) { next.error = holdoutError(e); }
    if (captured !== getTask()?.id || version !== epoch) return;
    data = next; render();
  }
  async function run(action) {
    const captured = taskId;
    busy = true; data.error = ""; render();
    try { await action(); await onStateChange(captured); } catch (error) { data.error = holdoutError(error); }
    busy = false;
    if (captured !== getTask()?.id) { await refresh({ force: true }); return; }
    if (data.error) render(); else await refresh({ force: true });
  }
  root?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-blind-action]");
    if (!button || busy) return;
    const action = button.dataset.blindAction;
    if (action === "identity") openIdentity();
    if (action === "refresh") refresh({ force: true });
    if (action === "evaluate") run(() => api(endpoint("/evaluate"), { method: "POST", body: JSON.stringify({ freeze_hash: data.summary.freeze_hash, recovery_id: data.summary.recovery_id, request_id: "workspace-evaluation" }) }));
  });
  root?.addEventListener("submit", (event) => {
    const form = event.target.closest("[data-blind-form]");
    if (!form) return;
    event.preventDefault(); if (busy || !form.reportValidity()) return;
    const values = new FormData(form), type = form.dataset.blindForm;
    const path = endpoint(type === "seal" ? "/cohort" : type === "recover" ? "/recoveries" : `/${type}`);
    if (type === "seal") {
      const payload = Object.fromEntries([...values.entries()].filter(([key]) => key !== "file"));
      values.set("contract", JSON.stringify(payload));
      [...values.keys()].filter(key => !["file", "contract"].includes(key)).forEach(key => values.delete(key));
    }
    const body = ["seal", "development"].includes(type) ? values : Object.fromEntries(values.entries());
    run(() => api(path, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body) }));
  });
  return { refresh };
}
