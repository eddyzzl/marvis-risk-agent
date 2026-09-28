import { escapeHtml as esc } from "./ui-utils.js";

const field = (name, label, value = "", type = "text", attributes = "") => `<label>${esc(label)}<input name="${name}" type="${type}" value="${esc(value ?? "")}" ${attributes}></label>`;
export const selectOptions = (items, value = "", placeholder = "请选择") => `<option value="">${esc(placeholder)}</option>${items.map(item => `<option value="${esc(item.id)}"${item.id === value ? " selected" : ""}>${esc(item.label)}</option>`).join("")}`;
const select = (name, label, items, value = "", attributes = "") => `<label>${esc(label)}<select name="${name}" ${attributes}>${selectOptions(items, value)}</select></label>`;
const utcInput = value => value ? new Date(value).toISOString().slice(0, 19) : "";

export function operationsFormHtml({ tasks = [], refs = [], record = null, recheck = null } = {}) {
  const c = record?.contract || {}, b = c.monitoring_binding || {}, r = c.retry_policy || {};
  return `<form data-ops-form="schedule" class="ops-form">
    <div class="ops-section-title"><h3>${recheck ? "重新评估原周期" : record ? "修改监控配置" : "新建监控"}</h3><button type="button" class="secondary" data-ops-action="cancel-form">返回</button></div>
    <p class="ops-note">${recheck ? "保留原周期与目标，使用本次明确选择的数据和覆盖时间生成独立结果。原始证据不会被覆盖。" : "每个周期绑定已登记的数据快照。追加数据后需发布新配置，平台不会自行追踪本地文件变化。"}</p>
    ${recheck ? `<p class="ops-callout">原周期：${esc(recheck)} · 当前配置 v${esc(c.revision)}</p>` : `<div class="ops-grid">
      ${field("schedule_id", "监控名称 / 唯一标识", c.schedule_id, "text", `required maxlength="200" ${record ? "readonly" : ""}`)}
      ${select("monitoring_ref", "监控类型", refs.map(id => ({ id, label: id === "modeling.monitor_run" ? "模型监控" : id === "strategy.run_strategy_monitoring" ? "策略监控" : id })), c.monitoring_ref, "required")}
      ${field("anchor_at", "周期起点（UTC）", utcInput(c.calendar?.anchor_at), "datetime-local", 'required step="1"')}
      ${field("interval_seconds", "周期长度（秒）", c.calendar?.interval_seconds ?? 86400, "number", 'required min="1" step="1"')}
      ${field("active_from", "开始监控（UTC）", utcInput(c.active_from), "datetime-local", 'required step="1"')}
      ${field("active_until", "结束监控（UTC，可留空）", utcInput(c.active_until), "datetime-local", 'step="1"')}
    </div>`}
    <h4>监控对象与数据</h4><div class="ops-grid">
      ${select("task_id", "所属任务", tasks.map(t => ({ id: t.id, label: t.name || t.model_name || t.id })), b.task_id, `required ${recheck ? "disabled" : ""}`)}
      ${select("target_id", "监控目标", [], "", `required ${recheck ? "disabled" : ""}`)}
      ${select("dataset_id", "数据快照", [], "", "required")}
    </div>
    <div class="ops-upload"><label>上传新数据 <input type="file" data-ops-upload accept=".csv,.xlsx,.xls,.parquet" disabled></label><span data-ops-source-message class="ops-note">先选择任务与监控类型。</span></div>
    <p class="ops-note" data-ops-source-hash>尚未选择数据快照</p>
    <div class="ops-grid">
      ${select("time_col", "业务时间列", [], b.time_col, "required")}
      ${field("timezone", "无时区时间的来源时区（IANA）", b.timezone || "UTC", "text", 'required placeholder="Asia/Shanghai"')}
      ${field("complete_through", "声明数据完整覆盖至（UTC）", utcInput(b.complete_through), "datetime-local", 'required step="1"')}
      ${select("label_mode", "本次标签要求", [{id:"required",label:"需要真实标签"},{id:"not_applicable",label:"本次不适用标签"}], b.label_mode, "required")}
      ${select("target_col", "标签列（需要标签时必填）", [], b.target_col)}
      ${field("label_maturity_seconds", "标签成熟等待（秒，明确填写）", b.label_maturity_seconds, "number", 'min="0" max="315360000" step="1"')}
      ${select("score_col", "分数列（可选）", [], b.score_col)}
      ${field("max_source_rows", "允许读取的最大行数", b.max_source_rows ?? 200000, "number", 'required min="1" max="1000000" step="1"')}
    </div>
    <p class="ops-note">覆盖时间由发布者声明，并非平台独立认证。选择“不适用标签”不会生成标签效果结论；需要标签时必须明确填写成熟等待，0 表示标签即时成熟。</p>
    ${recheck ? "" : `<details><summary>运行与重试设置</summary><div class="ops-grid">
      ${field("catch_up_budget", "每次补跑周期上限", c.catch_up_budget ?? 1, "number", 'required min="1" max="100" step="1"')}
      ${field("lease_seconds", "执行租约（秒）", c.lease_seconds ?? 300, "number", 'required min="1" step="1"')}
      ${field("max_attempts", "最大尝试次数", r.max_attempts ?? 3, "number", 'required min="1" max="10" step="1"')}
      ${field("initial_backoff_seconds", "首次重试等待（秒）", r.initial_backoff_seconds ?? 30, "number", 'required min="0" step="any"')}
      ${field("multiplier", "重试等待倍数", r.multiplier ?? 2, "number", 'required min="0.001" step="any"')}
      ${field("max_backoff_seconds", "最长重试等待（秒）", r.max_backoff_seconds ?? 300, "number", 'required min="0" step="any"')}
    </div></details><label class="ops-check"><input name="enabled" type="checkbox" ${c.enabled !== false ? "checked" : ""}>启用监控（关闭后停止新增周期，保留历史）</label>`}
    <p data-ops-form-error class="ops-error" role="alert"></p>
    <button type="submit" class="primary" data-ops-publish disabled>${recheck ? "提交重新评估" : record ? `发布新版本（基于 v${esc(c.revision)}）` : "发布监控"}</button>
  </form>`;
}

function required(value, label) {
  if (!String(value ?? "").trim()) throw new Error(`请填写${label}`);
  return String(value).trim();
}
function number(value, label, min, max = Infinity, integer = true) {
  required(value, label);
  const n = Number(value);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n)) || n < min || n > max) throw new Error(`${label}数值无效`);
  return n;
}
function utc(value, label) {
  required(value, label);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$/.test(value)) throw new Error(`${label}必须是有效的 UTC 日期时间`);
  const normalized = value.length === 16 ? `${value}:00` : value;
  const date = new Date(`${normalized}Z`);
  if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0,19) !== normalized) throw new Error(`${label}无效`);
  return date.toISOString();
}

export function operationsPayload(values, { datasets = [], targets = [], record = null, recheck = null, idempotencyKey = "" } = {}) {
  const c = record?.contract;
  // datetime-local displays seconds; preserve finer server timestamps unless the
  // operator actually changes the displayed value (especially calendar anchors).
  const date = (value, label, previous) => {
    const validated = utc(value, label);
    return previous && value === utcInput(previous) ? previous : validated;
  };
  const taskId = required(values.task_id, "所属任务");
  const dataset = datasets.find(d => d.id === values.dataset_id && d.task_id === taskId);
  if (!dataset || !/^[a-f0-9]{64}$/.test(dataset.content_hash || "")) throw new Error("请选择当前任务中有内容指纹的数据快照");
  if (!targets.some(t => t.id === values.target_id)) throw new Error("请选择当前任务中的监控目标");
  const columns = dataset.columns.map(col => typeof col === "string" ? col : col.name);
  const column = (name, label, optional = false) => {
    if (!values[name] && optional) return null;
    if (!columns.includes(values[name])) throw new Error(`请从当前快照选择${label}`);
    return values[name];
  };
  if (!["required", "not_applicable"].includes(values.label_mode)) throw new Error("请明确选择本次标签要求");
  const timezone = required(values.timezone, "来源时区");
  try { new Intl.DateTimeFormat("en", { timeZone: timezone }); } catch { throw new Error("来源时区必须是有效的 IANA 时区"); }
  const binding = {task_id:taskId, dataset_id:dataset.id, dataset_content_hash:dataset.content_hash,
    time_col:column("time_col", "业务时间列"), timezone, complete_through:date(values.complete_through, "覆盖时间", c?.monitoring_binding?.complete_through),
    target_id:values.target_id, label_mode:values.label_mode,
    target_col:values.label_mode === "required" ? column("target_col", "标签列") : null,
    label_maturity_seconds:values.label_mode === "required" ? number(values.label_maturity_seconds, "标签成熟等待", 0, 315360000) : null,
    score_col:column("score_col", "分数列", true), max_source_rows:number(values.max_source_rows, "最大行数", 1, 1000000)};
  if (recheck) {
    if (taskId !== c.monitoring_binding.task_id || binding.target_id !== c.monitoring_binding.target_id) throw new Error("重新评估必须保留原任务和目标");
    return {period_key:recheck, expected_revision:c.revision, idempotency_key:idempotencyKey, monitoring_binding:binding};
  }
  const schedule = {schedule_id:required(values.schedule_id, "监控名称"), revision:(c?.revision || 0) + 1,
    monitoring_ref:required(values.monitoring_ref, "监控类型"),
    active_from:date(values.active_from, "开始时间", c?.active_from), active_until:values.active_until ? date(values.active_until, "结束时间", c?.active_until) : null,
    calendar:{anchor_at:date(values.anchor_at, "周期起点", c?.calendar?.anchor_at), interval_seconds:number(values.interval_seconds, "周期长度", 1)},
    catch_up_budget:number(values.catch_up_budget, "补跑上限", 1, 100), lease_seconds:number(values.lease_seconds, "执行租约", 1),
    retry_policy:{max_attempts:number(values.max_attempts, "尝试次数", 1, 10), initial_backoff_seconds:number(values.initial_backoff_seconds, "首次等待", 0, Infinity, false), multiplier:number(values.multiplier, "等待倍数", Number.MIN_VALUE, Infinity, false), max_backoff_seconds:number(values.max_backoff_seconds, "最长等待", 0, Infinity, false)},
    enabled:values.enabled === true, monitoring_binding:binding};
  if (c && schedule.schedule_id !== c.schedule_id) throw new Error("监控名称已锁定");
  if (schedule.active_until && new Date(schedule.active_until) <= new Date(schedule.active_from)) throw new Error("结束时间必须晚于开始时间");
  if (schedule.retry_policy.max_backoff_seconds < schedule.retry_policy.initial_backoff_seconds) throw new Error("最长等待不能小于首次等待");
  return {expected_revision:c?.revision || 0, schedule};
}
