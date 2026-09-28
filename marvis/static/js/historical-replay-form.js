import { escapeHtml as esc } from "./ui-utils.js";

export const REPLAY_VERSION = "decision_twin.batch_request.v2";
const attr = (value) => esc(value ?? "");
export const field = (name, label, type = "text", extra = "", value = "") =>
  `<label>${esc(label)}<input name="${name}" type="${type}" value="${attr(value)}" ${extra}></label>`;
export const options = (items, value = "") =>
  '<option value="">请选择</option>' +
  items
    .map(
      (item) =>
        `<option value="${attr(item.id)}"${item.id === value ? " selected" : ""}>${esc(item.label ?? item.id)}</option>`,
    )
    .join("");
export const select = (name, label, items, extra = "", value = "") =>
  `<label>${esc(label)}<select name="${name}" ${extra}>${options(items, value)}</select></label>`;
const note = (text) => `<p class="history-note">${text}</p>`;
const optional = (name, label, body) =>
  `<details class="history-option"><summary><label><input type="checkbox" data-history-enable="${name}">${esc(label)}</label></summary><fieldset data-history-fields="${name}" disabled>${body}</fieldset></details>`;
const req = "required";

export function replayFormHtml(datasets, packages) {
  const column = [];
  const packageOptions = packages.map((p) => ({
    id: p.package_hash,
    label: `${p.strategy_id} · v${p.strategy_version} · ${p.decision_node} · ${p.package_hash.slice(0, 12)}`,
  }));
  return `<form data-history-form="replay">
    ${note("选择历史申请快照与已冻结的基线、挑战方案。平台按逐笔决策时点验证特征可用性；回放不证明历史上线，也不证明因果收益。")}
    <div class="history-grid">${select(
      "dataset_id",
      "历史数据快照",
      datasets.map((d) => ({
        id: d.id,
        label: `${d.source_name || d.id} · ${d.row_count} 行`,
      })),
      req,
    )}${field("as_of", "数据截止时间（UTC）", "datetime-local", 'required step="1"')}${field("source_ref", "历史数据来源", "text", req)}${field("population", "目标申请人群", "text", req)}${select("record_id_col", "申请唯一编号列", column, req)}${select("decision_at_col", "原决策时间列", column, req)}</div>
    <p class="history-note" data-history-source></p>
    <h4>冻结方案</h4><div class="history-grid">${select("baseline", "基线方案", packageOptions, req)}${select("challenger", "挑战方案", packageOptions, req)}${select("counterfactual", "反事实方案（可选）", packageOptions)}</div>
    ${note("这里只列出已登记的冻结包。选择后会重新校验包内容，基线与挑战方案必须属于相同决策节点。")}
    <button type="button" class="button compact secondary" data-history-action="features">读取方案字段合同</button><p data-history-package-message class="history-note"></p>
    <div data-history-feature-fields></div>
    ${optional("observed_actions", "导入历史实际动作", `<div class="history-grid">${select("action_col", "历史动作列（approval/reject/review）", column, req)}${select("recorded_at_col", "动作记录时间列", column, req)}${field("actions_source_ref", "动作记录来源", "text", req)}</div>${note("导入的动作与平台回放分别保存，不会将外部历史记录标成 MARVIS 实际执行。")}`)}
    ${optional("economics", "估算风险与收益", `<div class="history-grid">${field("currency", "币种（例如 CNY）", "text", 'required pattern="[A-Z]{3}" maxlength="3"')}${select("ead_col", "风险暴露金额列", column, req)}${select("economics_available_at_col", "金额可用时间列", column, req)}${field("annual_rate", "年利率（0—1）", "number", 'required min="0" max="1" step="any"')}${field("funding_rate", "资金年成本（0—1）", "number", 'required min="0" max="1" step="any"')}${field("lgd", "违约损失率（0—1）", "number", 'required min="0" max="1" step="any"')}${field("term_months", "期限（月）", "number", 'required min="0.000001" step="any"')}${field("operating_cost_per_loan", "每笔运营成本", "number", 'required min="0" step="any"')}</div><div class="history-grid">${["ead", "pd", "annual_rate", "funding_rate", "lgd", "term_months", "operating_cost_per_loan"].map((name, index) => field(`source_${name}`, `${["金额", "PD", "利率", "资金成本", "损失率", "期限", "运营成本"][index]}假设来源`, "text", req)).join("")}</div>${note("收益按固定情景假设估算。缺失假设时保持未知，不用 0 代替。")}`)}
    ${optional("protected_group", "检查已授权的客群差异", `<div class="history-grid">${select("group_column", "分组列", column, req)}${select("group_available_at_col", "分组可用时间列", column, req)}${field("governance_ref", "分组使用授权依据", "text", req)}${field("minimum_group_size", "每组最少记录数", "number", 'required min="1" step="1"')}</div>`)}
    ${optional("capacity", "估算运营处理量", `<div class="history-grid">${field("capacity_unit", "处理量单位（英文标识）", "text", 'required pattern="[a-z][a-z0-9_]{0,79}"')}${field("capacity_source_ref", "处理成本假设来源", "text", req)}${["approval", "reject", "review"].map((name, i) => field(`capacity_${name}`, `${["批准", "拒绝", "转人工"][i]}每笔处理量`, "number", 'required min="0" step="any"')).join("")}</div>`)}
    ${optional("temporal_stability", "按时间窗口检查稳定性", `<div class="history-grid">${field("timezone", "IANA 时区（例如 Asia/Shanghai）", "text", req)}${field("policy_source_ref", "窗口与阈值口径来源", "text", req)}${field("minimum_reference_rows", "参考窗最少记录数", "number", 'required min="2" step="1"')}${field("minimum_comparison_rows", "对照窗最少记录数", "number", 'required min="2" step="1"')}${field("bin_count", "仅用参考窗拟合的分箱数", "number", 'required min="2" max="20" step="1"')}${field("max_score_psi", "分数 PSI 上限", "number", 'required min="0" step="any"')}${field("max_action_psi", "动作 PSI 上限", "number", 'required min="0" step="any"')}${field("max_absolute_approval_rate_delta", "批准率绝对变化上限（0—1）", "number", 'required min="0" max="1" step="any"')}</div><h4>参考窗口</h4>${temporalWindowHtml("reference")}<h4>对照窗口</h4><div data-history-windows>${temporalWindowHtml("comparison")}</div><button type="button" class="button compact secondary" data-history-action="add-window">添加对照窗口</button>${note("时间须包含时区偏移，如 2026-01-01T00:00:00+08:00；后端会校验其与 IANA 时区一致。窗口含起点、不含终点，互不重叠，参考窗先于对照窗，结束时间不得晚于数据截止时间。样本不足或分箱退化时保留未知，不自动推定阈值。")}`)}
    <details class="history-option"><summary>业务约束（可选）</summary><div data-history-constraints></div><button type="button" class="button compact secondary" data-history-action="add-constraint">添加约束</button></details>
    <div class="history-actions"><button type="submit" class="button compact primary">校验并生成提案</button></div>
  </form>`;
}
export function temporalWindowHtml(kind) {
  return `<div class="history-grid history-time-window" data-history-window="${kind}">${field("name", "窗口名称", "text", 'required maxlength="80"')}${field("start", "起点（含偏移）", "text", 'required placeholder="2026-01-01T00:00:00+08:00"')}${field("end", "终点（含偏移）", "text", 'required placeholder="2026-02-01T00:00:00+08:00"')}${kind === "comparison" ? '<button type="button" class="button compact secondary" data-history-action="remove-window">移除窗口</button>' : ""}</div>`;
}
export function featureFieldsHtml(names, columns) {
  return `<h4>特征与历史可用时点</h4>${note("逐个绑定输入字段、事件发生时间和信息真正可用的时间。仅有事件时间不能证明当时已经可用。")}${names.map((name) => `<fieldset class="history-feature" data-history-feature="${attr(name)}"><legend>${esc(name)}</legend><div class="history-grid">${select("value_col", "值所在列", columns, req, columns.some((c) => c.id === name) ? name : "")}${select("event_at_col", "事件发生时间列", columns, req)}${select("available_at_col", "信息可用时间列", columns, req)}</div></fieldset>`).join("")}`;
}
export function constraintHtml() {
  return `<div class="history-grid history-constraint" data-history-constraint>${select(
    "metric",
    "约束指标",
    [
      { id: "approval_rate", label: "批准率" },
      { id: "ead", label: "风险暴露金额" },
      { id: "expected_loss", label: "预期损失" },
      { id: "profit", label: "估算利润" },
      { id: "protected_group_disparity", label: "客群批准率差异" },
      { id: "operations_capacity", label: "运营处理量" },
    ],
    req,
  )}${select(
    "operator",
    "比较方式",
    [
      { id: ">=", label: "至少" },
      { id: "<=", label: "至多" },
    ],
    req,
  )}${field("threshold", "阈值", "number", 'required step="any"')}${field("unit", "单位", "text", req)}<button type="button" class="button compact secondary" data-history-action="remove-constraint">移除约束</button></div>`;
}
const text = (value, label) => {
  if (!String(value ?? "").trim()) throw new Error(`请填写${label}`);
  return String(value).trim();
};
const num = (
  value,
  label,
  min = -Infinity,
  max = Infinity,
  integer = false,
) => {
  text(value, label);
  const n = Number(value);
  if (
    !Number.isFinite(n) ||
    n < min ||
    n > max ||
    (integer && !Number.isInteger(n))
  )
    throw new Error(`${label}无效`);
  return n;
};
export function utcValue(value, label) {
  const input = text(value, label);
  const normalized = input.length === 16 ? `${input}:00` : input;
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(normalized))
    throw new Error(`${label}必须填写 UTC 日期时间`);
  const date = new Date(`${normalized}Z`);
  if (
    !Number.isFinite(date.getTime()) ||
    date.toISOString().slice(0, 19) !== normalized
  )
    throw new Error(`${label}无效`);
  return date.toISOString();
}
export function collectReplay(
  values,
  {
    taskId,
    datasets,
    packages,
    features,
    constraints = [],
    enabled = [],
    windows = [],
  },
) {
  const dataset = datasets.find(
    (d) => d.id === values.dataset_id && d.task_id === taskId,
  );
  if (!dataset || !/^[a-f0-9]{64}$/.test(dataset.content_hash || ""))
    throw new Error("请选择当前任务的认证数据快照");
  const columns = dataset.columns.map((c) =>
    typeof c === "string" ? c : c.name,
  );
  const col = (name, label) => {
    if (!columns.includes(name)) throw new Error(`请选择当前快照的${label}`);
    return name;
  };
  if (
    !features.length ||
    new Set(features.map((f) => f.name)).size !== features.length
  )
    throw new Error("请读取并绑定方案字段合同");
  const scenarios = ["baseline", "challenger", "counterfactual"]
    .filter((kind) => kind !== "counterfactual" || values[kind])
    .map((kind) => {
      if (!packages.some((p) => p.package_hash === values[kind]))
        throw new Error("请选择已登记的冻结方案");
      return {
        kind,
        name: {
          baseline: "基线冻结方案",
          challenger: "挑战冻结方案",
          counterfactual: "反事实冻结方案",
        }[kind],
        package_hash: values[kind],
      };
    });
  const result = {
    schema_version: REPLAY_VERSION,
    dataset_id: dataset.id,
    expected_content_hash: dataset.content_hash,
    record_id_col: col(values.record_id_col, "申请编号列"),
    decision_at_col: col(values.decision_at_col, "决策时间列"),
    as_of: utcValue(values.as_of, "数据截止时间"),
    source_ref: text(values.source_ref, "历史数据来源"),
    population: text(values.population, "目标人群"),
    scenarios,
    features: features.map((f) => ({
      name: f.name,
      value_col: col(f.value_col, "特征值列"),
      event_at_col: col(f.event_at_col, "事件时间列"),
      available_at_col: col(f.available_at_col, "信息可用时间列"),
    })),
    economics: null,
    protected_group: null,
    capacity: null,
    observed_actions: null,
    constraints: constraints.map((c) => ({
      metric: text(c.metric, "约束指标"),
      operator: text(c.operator, "比较方式"),
      threshold: num(c.threshold, "阈值"),
      unit: text(c.unit, "约束单位"),
    })),
  };
  if (enabled.includes("observed_actions"))
    result.observed_actions = {
      action_col: col(values.action_col, "历史动作列"),
      recorded_at_col: col(values.recorded_at_col, "动作记录时间列"),
      source_ref: text(values.actions_source_ref, "动作来源"),
    };
  if (enabled.includes("economics")) {
    const currency = text(values.currency, "币种");
    if (!/^[A-Z]{3}$/.test(currency)) throw new Error("币种须为三位大写代码");
    result.economics = {
      currency,
      ead_col: col(values.ead_col, "风险暴露列"),
      available_at_col: col(
        values.economics_available_at_col,
        "金额可用时间列",
      ),
      annual_rate: num(values.annual_rate, "年利率", 0, 1),
      funding_rate: num(values.funding_rate, "资金成本", 0, 1),
      lgd: num(values.lgd, "违约损失率", 0, 1),
      term_months: num(values.term_months, "期限", Number.MIN_VALUE),
      operating_cost_per_loan: num(
        values.operating_cost_per_loan,
        "运营成本",
        0,
      ),
      assumption_sources: Object.fromEntries(
        [
          "ead",
          "pd",
          "annual_rate",
          "funding_rate",
          "lgd",
          "term_months",
          "operating_cost_per_loan",
        ].map((key) => [key, text(values[`source_${key}`], `${key}假设来源`)]),
      ),
    };
  }
  if (enabled.includes("protected_group"))
    result.protected_group = {
      column: col(values.group_column, "分组列"),
      available_at_col: col(values.group_available_at_col, "分组可用时间列"),
      governance_ref: text(values.governance_ref, "授权依据"),
      minimum_group_size: num(
        values.minimum_group_size,
        "每组最少记录数",
        1,
        Infinity,
        true,
      ),
    };
  if (enabled.includes("capacity"))
    result.capacity = {
      unit: text(values.capacity_unit, "处理量单位"),
      source_ref: text(values.capacity_source_ref, "处理量来源"),
      per_action: Object.fromEntries(
        ["approval", "reject", "review"].map((kind) => [
          kind,
          num(values[`capacity_${kind}`], "动作处理量", 0),
        ]),
      ),
    };
  if (enabled.includes("temporal_stability")) {
    const timezone = text(values.timezone, "IANA 时区");
    try {
      new Intl.DateTimeFormat("en", { timeZone: timezone });
    } catch {
      throw new Error("IANA 时区无效");
    }
    const parsed = windows.map((w) => {
      const window = { name: text(w.name, "窗口名称") };
      for (const key of ["start", "end"]) {
        window[key] = text(w[key], "带时区偏移的窗口边界");
        if (
          !/T.*(?:Z|[+-]\d{2}:\d{2})$/.test(window[key]) ||
          !Number.isFinite(Date.parse(window[key]))
        )
          throw new Error("窗口边界必须为有效日期并包含时区偏移");
      }
      return { ...window, kind: w.kind };
    });
    const reference = parsed.filter((w) => w.kind === "reference"),
      comparisons = parsed.filter((w) => w.kind === "comparison");
    if (
      reference.length !== 1 ||
      !comparisons.length ||
      comparisons.length > 24
    )
      throw new Error("请提供一个参考窗和 1 至 24 个对照窗");
    if (new Set(parsed.map((w) => w.name)).size !== parsed.length)
      throw new Error("窗口名称不可重复");
    const sorted = [...parsed].sort(
      (a, b) => Date.parse(a.start) - Date.parse(b.start),
    );
    if (
      parsed.some(
        (w) =>
          Date.parse(w.start) >= Date.parse(w.end) ||
          Date.parse(w.end) > Date.parse(result.as_of),
      ) ||
      comparisons.some(
        (w) => Date.parse(w.start) < Date.parse(reference[0].end),
      ) ||
      sorted.some(
        (w, i) => i > 0 && Date.parse(sorted[i - 1].end) > Date.parse(w.start),
      )
    )
      throw new Error("窗口须有序、不重叠，且结束不晚于数据截止时间");
    const bare = ({ name, start, end }) => ({ name, start, end });
    result.temporal_stability = {
      timezone,
      reference_window: bare(reference[0]),
      comparison_windows: comparisons.map(bare),
      minimum_reference_rows: num(
        values.minimum_reference_rows,
        "参考窗最少记录数",
        2,
        Infinity,
        true,
      ),
      minimum_comparison_rows: num(
        values.minimum_comparison_rows,
        "对照窗最少记录数",
        2,
        Infinity,
        true,
      ),
      bin_count: num(values.bin_count, "分箱数", 2, 20, true),
      thresholds: {
        max_score_psi: num(values.max_score_psi, "分数 PSI 上限", 0),
        max_action_psi: num(values.max_action_psi, "动作 PSI 上限", 0),
        max_absolute_approval_rate_delta: num(
          values.max_absolute_approval_rate_delta,
          "批准率变化上限",
          0,
          1,
        ),
      },
      policy_source_ref: text(values.policy_source_ref, "窗口与阈值口径来源"),
    };
  }
  return result;
}
export function collectReconciliation(
  values,
  { taskId, datasets, artifactId },
) {
  const dataset = datasets.find(
    (d) => d.id === values.dataset_id && d.task_id === taskId,
  );
  if (!dataset || !/^[a-f0-9]{64}$/.test(dataset.content_hash || ""))
    throw new Error("请选择当前任务的认证数据快照");
  if (!/^[a-f0-9]{64}$/.test(artifactId || ""))
    throw new Error("请先选择认证回放回执");
  const columns = dataset.columns.map((c) =>
    typeof c === "string" ? c : c.name,
  );
  const result = {
    replay_artifact_id: artifactId,
    dataset_id: dataset.id,
    expected_content_hash: dataset.content_hash,
    currency: text(values.currency, "币种"),
    maturity_days: num(values.maturity_days, "成熟天数", 1, 36500, true),
    maturity_source_ref: text(values.maturity_source_ref, "成熟口径来源"),
    source_ref: text(values.source_ref, "现金流来源"),
    reconciled_at: utcValue(values.reconciled_at, "对账截止时间"),
  };
  if (!/^[A-Z]{3}$/.test(result.currency))
    throw new Error("币种须为三位大写代码");
  for (const key of [
    "record_id_col",
    "observed_at_col",
    "actual_loss_col",
    "actual_profit_col",
  ]) {
    if (!columns.includes(values[key]))
      throw new Error("请选择当前现金流快照中的字段");
    result[key] = values[key];
  }
  return result;
}
