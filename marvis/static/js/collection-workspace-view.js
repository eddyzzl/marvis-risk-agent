import { escapeHtml as esc } from "./ui-utils.js";
import { safeSameOriginApiHref } from "./url-safety.js";
const states = {
  proposed: "待排队",
  queued: "已排队",
  completed: "已完成",
  cancelled: "已取消",
  eligible: "符合声明约束",
  blocked: "已拦截",
  required_review: "需复核",
  hold: "暂缓",
  contact: "参考触达",
  review: "人工复核",
  unknown_effect: "效果未知",
  held: "暂缓",
  eligible_for_approval: "待授权",
  manual_review: "需人工复核",
  insufficient_evidence: "证据不足",
  queue_batch: "排队",
  execute_reference: "参考执行",
  cancel_batch: "取消",
  declared_constraints_satisfied: "满足声明约束",
  contact_prohibited: "禁止触达",
  contact_permission_unknown: "触达许可未知",
  outside_contact_window: "不在允许时段",
  contact_history_incomplete: "历史覆盖不足",
  subject_frequency_limit: "达到主体频次上限",
  subject_minimum_interval: "未达到最小间隔",
  strategy_hold: "策略要求暂缓",
  policy_outside_validity: "政策不在有效期",
  queue_batch_capacity: "达到批次队列容量",
  batch_estimated_cost_limit: "超过预计成本上限",
};
export const status = (v) => states[v] || v || "未知";
export const details = (label, data) =>
  `<details class="collection-evidence"><summary>${esc(label)}</summary><pre>${esc(JSON.stringify(data, null, 2))}</pre></details>`;
export const button = (action, label, id = "") =>
  `<button type="button" class="button compact secondary" data-collection-action="${action}" data-collection-id="${esc(id)}">${esc(label)}</button>`;
export function money(value, unit) {
  if (value == null) return "未知";
  const exponent = unit?.minor_unit_exponent;
  if (!Number.isInteger(exponent)) return `${esc(value)} 最小货币单位`;
  return `${esc(unit.currency)} ${(value / 10 ** exponent).toLocaleString("zh-CN", { minimumFractionDigits: exponent, maximumFractionDigits: exponent })}`;
}
export function shellHtml() {
  return `<details data-collection-panel><summary>催收参考工作台 <span>核对批次、受治理执行、查看结果与证据</span></summary><div class="collection-body"><p class="collection-note">当前为本地参考执行，不联系客户。业务声明、历史覆盖与资金观测分别保留来源；排队及执行都需现有计划的人审授权。</p><div class="collection-actions">${button("load", "读取催收工作区")}${button("identity", "绑定会话身份")}</div><p data-collection-message class="collection-message" role="status"></p><div data-collection-identity></div><div class="collection-actions" data-collection-intake-actions></div><div data-collection-editor></div><div data-collection-gate></div><div data-collection-lists></div><div data-collection-detail></div><div data-collection-receipt></div></div></details>`;
}
export function listsHtml(cases, batches) {
  return `<h4>已登记案件</h4>${cases.length ? `<div class="collection-table-scroll"><table><thead><tr><th>案件</th><th>期初余额</th><th>开户时间</th><th>来源边界</th></tr></thead><tbody>${cases.map((c) => `<tr><td>${button("case", c.case_id, c.case_id)}</td><td>${money(c.opening_balance_minor, c.unit)}</td><td>${esc(c.opened_at)}</td><td>${c.source_assurance === "historical_import_unverified" ? "历史导入，未独立验证" : "本地参考记录"}</td></tr>`).join("")}</tbody></table></div>` : '<p class="collection-note">暂无本会话可读取的案件。</p>'}<h4>批次与执行</h4>${batches.length ? batches.map((b) => `<button type="button" class="collection-row" data-collection-action="batch" data-collection-id="${esc(b.batch_id)}"><strong>${esc(b.batch_id)}</strong><span>${esc(status(b.status))} · 版本 ${esc(b.revision)} · ${esc(b.created_at)}</span></button>`).join("") : '<p class="collection-note">暂无本会话可读取的批次。</p>'}`;
}
export function batchHtml(b, role) {
  const p = b.preview,
    unit = p.unit;
  return `<h4>${esc(b.batch_id)} · ${esc(status(b.status))}</h4><p class="collection-note">参考时点 ${esc(p.as_of)} · 知识截止 ${esc(p.knowledge_cutoff)}。${p.knowledge_mode === "retrospective_declared" ? "事后声明口径，不代表当时平台可见。" : "历史可见性为来源声明，未独立证明。"}</p><p>提案记录 ${(p.results || []).length} 条 · 预计触达成本 ${money(p.estimated_contact_cost_minor, unit)}</p><p class="collection-note">预览不等于执行授权或实时容量预留；预计成本不写入实际现金流。</p><div class="collection-actions">${role === "maker" ? (b.status === "proposed" ? button("queue", "生成排队计划") : b.status === "queued" ? button("execute", "生成参考执行计划") : "") : ""}${role === "maker" && ["proposed", "queued"].includes(b.status) ? button("cancel", "生成取消计划") : ""}${button("batch", "重新核对批次", b.batch_id)}</div><div class="collection-table-scroll"><table><thead><tr><th>案件</th><th>命中规则</th><th>规则动作</th><th>预览状态</th><th>原因</th></tr></thead><tbody>${(p.results || []).map((r) => `<tr><td>${esc(r.case_id)}</td><td>${esc(r.matched_rule_id || "默认动作")}</td><td>${esc(status(r.action?.value?.kind || r.action?.type))}</td><td>${esc(status(r.status))}</td><td>${esc(status(r.reason || "—"))}</td></tr>`).join("")}</tbody></table></div>${details("冻结策略、业务政策、历史覆盖与来源合同", b.request)}<h4>平台执行记录</h4>${b.actions.length ? `<div class="collection-table-scroll"><table><thead><tr><th>案件</th><th>队列</th><th>状态</th><th>预计成本</th><th>实际成本</th><th>客户联系</th></tr></thead><tbody>${b.actions.map((a) => `<tr><td>${esc(a.case_id)}</td><td>${esc(a.queue_id)}</td><td>${esc(status(a.state))}</td><td>${money(a.estimated_cost_minor, unit)}</td><td>${money(a.actual_cost_minor, unit)}</td><td>${a.customer_contacted === false ? "未联系" : "未知"}</td></tr>`).join("")}</tbody></table></div>` : '<p class="collection-note">尚无已记录的执行动作。</p>'}<h4>原生执行证据</h4>${b.effects.length ? b.effects.map((e) => button("evidence", `${status(e.operation || "执行")} · 查看原生回执`, safeSameOriginApiHref(e.evidence_url) ? e.evidence_url : "")).join(" ") : '<p class="collection-note">等待完成受治理的执行步骤。</p>'}${details("冻结批次指纹与执行身份", { batch_id: b.batch_id, request_hash: b.request_hash, preview_hash: b.preview_hash, actor_id: b.actor_id, revision: b.revision })}`;
}
export function gateHtml(plan) {
  const s = plan?.steps?.find(
    (s) =>
      s.status === "awaiting_confirm" && s.tool_ref?.plugin === "collection",
  );
  if (!s) return "";
  return `<section class="collection-gate"><h4>人工复核 · ${esc(s.title)}</h4><p class="collection-note">核对当前冻结批次、政策、历史覆盖和资源限制。仅授权本次计划快照；执行仍是本地参考模拟。</p>${details("查看待执行合同", s.inputs)}<label>本次复核理由<input name="collection_decision_reason" required maxlength="4000"></label><div class="collection-actions">${button("approve", "授权本次执行")}${button("reject", "拒绝本次执行")}</div></section>`;
}

export function collectionError(error) {
  const code = error?.detail?.code;
  const labels = {
    collection_idempotency_conflict: "该业务编号已登记且内容不同，请核对原记录",
    collection_material_identity_conflict:
      "该导入编号已绑定其他文件或映射，请核对来源或使用新编号",
    collection_case_actor_forbidden: "当前会话不属于这些案件的配置人员",
    collection_material_actor_forbidden: "当前会话没有登记该来源的权限",
    collection_contract_invalid:
      "资料不符合业务合同，请核对必填字段、单位、时点及编号",
  };
  return labels[code] || code || error?.message || "提交失败，请核对本次资料";
}

export const field = (name, label, type = "text", extra = "", value = "") =>
  `<label>${esc(label)}<input name="${name}" type="${type}" value="${esc(value)}" ${extra}></label>`;
export const select = (name, label, options, extra = "required") =>
  `<label>${esc(label)}<select name="${name}" ${extra}><option value="">请选择</option>${options.map(([value, text]) => `<option value="${value}">${esc(text)}</option>`).join("")}</select></label>`;
