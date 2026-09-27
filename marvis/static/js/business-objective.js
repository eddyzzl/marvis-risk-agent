import { escapeHtml } from "./ui-utils.js";

export const BUSINESS_OBJECTIVE_VERSION = "business-objective.v1";
export const EFFECT_STAGES = {
  estimated: "条件估算", backtested: "历史回测", oot_validated: "跨期验证", post_launch_observed: "上线后观察",
};
const METRICS = { approval_rate: "通过率", approved_bad_rate: "审批后坏率", profit: "利润（需认证成本口径）", oot_ks: "OOT KS", oot_auc: "OOT AUC" };
const UNITS = { ratio: "比例（0–1）", percent: "百分数（0–100）", currency: "货币元", currency_thousands: "货币千元", currency_millions: "货币百万元", count: "笔数" };
const DENOMINATORS = { "risk/development/rows": "开发样本 · 全部行", "risk/development/approved_labeled_rows": "开发样本 · 通过且有标签", "risk/oot/labeled_rows": "跨期样本 · 有标签行" };
const text = value => escapeHtml(value ?? "");
function selectOptions(options, selected) {
  const entries = { ...options };
  if (selected && !Object.hasOwn(entries, selected)) entries[selected] = selected;
  return Object.entries(entries).map(([value, label]) => `<option value="${text(value)}"${value === selected ? " selected" : ""}>${text(label)}</option>`).join("");
}
let suggestionsSequence = 0;
function suggestedInput(name, label, value, suggestions) {
  const id = `business-options-${++suggestionsSequence}`;
  return `<label><span>${label}</span><input data-business-field="${name}" value="${text(value)}" list="${id}" autocomplete="off"><datalist id="${id}">${Object.entries(suggestions).map(([key, description]) => `<option value="${text(key)}">${text(description)}</option>`).join("")}</datalist></label>`;
}
function input(name, label, value = "", type = "text", hint = "") {
  return `<label><span>${label}</span><input data-business-field="${name}" type="${type}"${type === "number" ? ' step="any"' : ""} value="${text(value)}" autocomplete="off"${hint ? ` placeholder="${text(hint)}"` : ""}></label>`;
}
export function businessCriterionHtml(value = {}) {
  return `<div class="business-criterion" data-business-criterion>
    <div class="business-objective-grid">
      ${suggestedInput("metric", "平台指标（输入或选择）", value.metric || "", METRICS)}
      ${suggestedInput("unit", "单位（输入或选择）", value.unit || "ratio", UNITS)}
      ${suggestedInput("denominator", "分母 / 统计范围", value.denominator || "", DENOMINATORS)}
      ${input("minimum", "下限（可留空）", value.minimum, "number")}
      ${input("maximum", "上限（可留空）", value.maximum, "number")}
      <label><span>比较方式</span><select data-business-field="comparison">${selectOptions({absolute:"实际值",delta:"相对基线的差值"},value.comparison || "absolute")}</select></label>
      ${input("baseline_ref", "基线证据引用（差值必填）", value.baseline_ref)}
      <button class="button compact secondary" type="button" data-business-remove>移除此指标</button>
    </div>
  </div>`;
}

export function businessObjectiveFormHtml(value = null, { targetKind = "strategy", readonly = false } = {}) {
  const objective = value || {};
  const unsupported = objective.schema_version && objective.schema_version !== BUSINESS_OBJECTIVE_VERSION;
  const locked = Boolean(readonly || unsupported);
  const rows = objective.criteria?.length ? objective.criteria : [{}];
  return `<details class="business-objective-editor" data-business-objective data-business-version="${text(objective.schema_version || BUSINESS_OBJECTIVE_VERSION)}" data-business-readonly="${locked}"${value ? " open" : ""}>
    <summary>业务验收合同 <span>先约定达标标准，再核对实际采用结果</span></summary>
    <label class="business-objective-enable"><input type="checkbox" data-business-enable${value ? " checked" : ""}${locked ? " disabled" : ""}>配置业务验收标准</label>
    <p class="business-note">${unsupported ? "此合同版本暂不支持编辑，请保留原合同并更新客户端。" : "未配置时仅报告执行结果，不认定业务达标。指标、单位和分母须与平台产物的测量口径一致。"}</p>
    <fieldset data-business-fields${!value || locked ? " disabled" : ""}>
      <div class="business-objective-grid">
        ${input("business_line", "业务线", objective.business_line, "text", "例如 consumer_credit")}
        ${input("decision_node", "决策节点", objective.decision_node, "text", "例如 approval / drawdown")}
        ${input("population", "目标客群", objective.population, "text", "明确适用样本与筛选条件")}
        ${input("period_start", "观察时期起点", objective.period_start, "date")}
        ${input("period_end", "观察时期终点", objective.period_end, "date")}
        ${input("responsibility_source", "业务确认来源", objective.responsibility_source, "text", "负责人 / 评审记录编号")}
        <label><span>验收对象</span><select data-business-field="target_kind">${selectOptions({strategy:"实际采用策略",model:"实际采用模型",workflow:"工作流"},objective.target_kind || targetKind)}</select></label>
        <label><span>最低证据阶段</span><select data-business-field="minimum_effect_stage">${selectOptions(EFFECT_STAGES,objective.minimum_effect_stage || "oot_validated")}</select></label>
        ${input("currency", "币种（货币指标必填）", objective.currency, "text", "如 CNY")}
      </div>
      <label class="business-objective-enable"><input type="checkbox" data-business-field="require_mature_labels"${objective.require_mature_labels !== false ? " checked" : ""}>验收必须具有成熟的真实观察标签</label>
      <p class="business-note">这是验收要求，不是标签已经成熟的声明。历史数据只能支持相应的回测或跨期结论；缺少认证口径会显示“证据不足”。</p>
      <details class="business-objective-binding"><summary>适用性声明</summary>
        <label class="business-objective-enable"><input type="checkbox" data-business-field="applicable"${objective.applicable !== false ? " checked" : ""}>此任务需要业务验收</label>
        <label class="business-objective-enable"><input type="checkbox" data-business-field="allow_not_applicable"${objective.allow_not_applicable === true ? " checked" : ""}>业务方明确允许不适用</label>
        ${input("not_applicable_reason", "不适用原因（需明确许可）", objective.not_applicable_reason)}
      </details>
      <div data-business-criteria>${rows.map(businessCriterionHtml).join("")}</div>
      <button class="button compact secondary" type="button" data-business-add>添加验收指标</button>
      <p class="business-note">指标分母须与实际认证产物一致；利润还须具有完整成本来源。比例 0.05 与百分数 5 的口径不同，平台不会自动换算。</p>
      <details class="business-objective-binding"><summary>对象版本与结论边界</summary><div class="business-objective-grid">
        ${input("target_id", "限定采用对象 ID（可选）", objective.target_id)}
        ${input("target_version", "限定采用版本（可选）", objective.target_version)}
        <label><span>结论类型</span><select data-business-field="claim">${selectOptions({observational:"观察性比较",causal:"因果增量结论"},objective.claim || "observational")}</select></label>
      </div><p class="business-note">未限定版本时，由认证采纳结果绑定实际版本；候选最优值不能代替。因果结论还须上线观察和因果识别证据，历史差异不能直接认定为增量收益。</p></details>
    </fieldset>
    <small class="business-contract-version">合同版本 ${text(objective.schema_version || BUSINESS_OBJECTIVE_VERSION)} · 生成计划时冻结</small>
  </details>`;
}
const valueOf = (root, name) => String(root?.querySelector?.(`[data-business-field="${name}"]`)?.value ?? "").trim();
export function collectBusinessObjective(root) {
  if (!root?.querySelector?.("[data-business-enable]")?.checked) return null;
  if (root.dataset.businessVersion !== BUSINESS_OBJECTIVE_VERSION) throw new Error("业务验收合同版本不受支持，请更新客户端。");
  const objective = {schema_version:BUSINESS_OBJECTIVE_VERSION};
  for (const key of ["business_line","decision_node","population","period_start","period_end","responsibility_source","target_kind","minimum_effect_stage","claim"]) objective[key]=valueOf(root,key);
  for (const key of ["target_id","target_version","currency"]) objective[key]=valueOf(root,key)||null;
  for (const key of ["require_mature_labels", "applicable", "allow_not_applicable"]) {
    objective[key] = root.querySelector(`[data-business-field="${key}"]`).checked;
  }
  objective.not_applicable_reason = valueOf(root, "not_applicable_reason") || null;
  objective.criteria=[...root.querySelectorAll("[data-business-criterion]")].filter(row => objective.applicable || ["metric", "denominator", "minimum", "maximum"].some(key => valueOf(row, key))).map(row=>({
    metric:valueOf(row,"metric"),unit:valueOf(row,"unit"),denominator:valueOf(row,"denominator"),
    minimum:valueOf(row,"minimum")===""?null:Number(valueOf(row,"minimum")),
    maximum:valueOf(row,"maximum")===""?null:Number(valueOf(row,"maximum")),
    comparison:valueOf(row,"comparison"),baseline_ref:valueOf(row,"baseline_ref")||null,
  }));
  const error = businessObjectiveError(objective);
  if (error) throw new Error(error);
  return objective;
}
export function businessObjectiveError(value) {
  if (!value) return "";
  if (value.schema_version !== BUSINESS_OBJECTIVE_VERSION) return "不支持此业务验收合同版本。";
  for (const [key,label] of Object.entries({business_line:"业务线",decision_node:"决策节点",population:"目标客群",responsibility_source:"业务确认来源"})) {
    if (typeof value[key] !== "string" || !value[key].trim()) return `请填写${label}。`;
  }
  for (const key of ["period_start","period_end"]) {
    const date=value[key];
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date || "") || !Number.isFinite(Date.parse(date)) || new Date(date).toISOString().slice(0,10)!==date) return "请填写有效的观察时期起止日期。";
  }
  if (value.period_start>value.period_end) return "观察时期终点不能早于起点。";
  if (!["model","strategy","workflow"].includes(value.target_kind)) return "请选择验收对象。";
  if (!Object.hasOwn(EFFECT_STAGES,value.minimum_effect_stage)) return "请选择最低证据阶段。";
  if (!["observational","causal"].includes(value.claim)) return "请选择结论类型。";
  if (value.applicable === false && (value.allow_not_applicable !== true || !value.not_applicable_reason?.trim())) return "不适用需要业务方明确许可和原因。";
  if (value.applicable !== false && !value.criteria?.length) return "请至少添加一项业务验收指标。";
  const seen=new Set();
  for (const row of value.criteria) {
    if (!row.metric?.trim() || !row.unit?.trim() || !row.denominator?.trim()) return "每项指标都要填写指标、单位和分母。";
    if (seen.has(row.metric)) return "同一指标不能重复，请合并上下限。";
    seen.add(row.metric);
    if (row.minimum==null && row.maximum==null) return "每项指标至少填写一个上限或下限。";
    if ([row.minimum,row.maximum].some(x=>x!=null && (typeof x!=="number" || !Number.isFinite(x)))) return "阈值必须是有限数字。";
    if (row.minimum!=null && row.maximum!=null && row.minimum>row.maximum) return "指标下限不能大于上限。";
    if (!["absolute","delta"].includes(row.comparison)) return "请选择比较方式。";
    if (row.comparison==="delta" && !row.baseline_ref?.trim()) return "相对基线的差值需要明确基线证据引用。";
    if (row.comparison==="absolute" && row.baseline_ref) return "实际值比较不应填写基线引用；需要对比基线时请选择差值。";
    if (row.unit.startsWith("currency") && !value.currency?.trim()) return "货币指标需要明确币种。";
  }
  return "";
}
export function handleBusinessObjectiveEvent(event) {
  const root=event.target?.closest?.("[data-business-objective]");
  if (!root || root.dataset.businessReadonly === "true") return false;
  if (event.type==="change" && event.target.matches?.("[data-business-enable]")) {
    root.querySelector("[data-business-fields]").disabled=!event.target.checked;
    return true;
  }
  if (event.type!=="click" || root.querySelector("[data-business-fields]")?.disabled) return false;
  if (event.target.closest?.("[data-business-add]")) {
    root.querySelector("[data-business-criteria]").insertAdjacentHTML("beforeend",businessCriterionHtml());
  } else if (event.target.closest?.("[data-business-remove]")) {
    if (root.querySelectorAll("[data-business-criterion]").length>1) event.target.closest("[data-business-criterion]").remove();
  } else return false;
  event.preventDefault();return true;
}
