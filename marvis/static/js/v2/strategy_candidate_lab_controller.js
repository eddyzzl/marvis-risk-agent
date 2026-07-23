import { escapeHtml } from "../ui-utils.js";
import {
  getStrategyCandidateLab,
  submitStrategyCandidateLabRequest,
} from "./api_v2.js";

export const STRATEGY_CANDIDATE_LAB_WORKFLOWS = Object.freeze([
  "univariate_candidate_analysis",
  "univariate_candidate_refinement",
  "cross_matrix_analysis",
  "automatic_tree_candidate_build",
]);

const WORKFLOW_LABELS = Object.freeze({
  univariate_candidate_analysis: "启动单变量候选分析",
  univariate_candidate_refinement: "启动单变量候选细化",
  cross_matrix_analysis: "启动二维 Cross Matrix",
  automatic_tree_candidate_build: "启动自动规则树",
});

const COLLECTION_DEFINITIONS = Object.freeze([
  {
    key: "univariate",
    title: "单变量候选",
    description: "分箱证据、候选排序与观测指标",
    pointerKey: "bins",
  },
  {
    key: "cross_matrix",
    title: "Cross Matrix",
    description: "二维交叉轴、单元格证据与风险观测",
    pointerKey: "cells",
  },
  {
    key: "automatic_tree",
    title: "自动规则树",
    description: "完整树结果、叶节点规则与现成效果证据",
    pointerKey: "leaves",
  },
]);

const FIELD_LABELS = Object.freeze({
  action: "动作",
  approval_rate: "通过率",
  artifact_id: "Artifact ID",
  asset_hash: "Asset Hash",
  asset_id: "Asset ID",
  bad: "坏样本",
  bad_rate: "坏率",
  bin_id: "Bin ID",
  candidate_id: "Candidate ID",
  candidate_stage: "候选阶段",
  cell_id: "Cell ID",
  column_bin_id: "列分箱",
  condition: "命中条件",
  count: "样本数",
  created_at: "创建时间",
  default_action: "默认动作",
  effect: "效果",
  effect_id: "Effect ID",
  enabled: "启用",
  evidence_hash: "Evidence Hash",
  feature: "字段",
  fragment_id: "Fragment ID",
  good: "好样本",
  iv: "IV",
  ks: "KS",
  lifecycle: "生命周期",
  lift: "Lift",
  method: "分箱方法",
  observation_stage: "观测阶段",
  pool_id: "Pool ID",
  position: "顺序",
  revision: "Revision",
  revision_id: "Revision ID",
  risk: "风险",
  row_bin_id: "行分箱",
  rule_id: "Rule ID",
  share: "占比",
  snapshot_hash: "Snapshot Hash",
  status: "状态",
  strategy_type: "策略类型",
  total: "总数",
  tree_id: "Tree ID",
  tree_result_hash: "Tree Result Hash",
  validation_status: "验证状态",
  value: "值",
  woe: "WOE",
});

const BLOCKED_REASON_COPY = Object.freeze({
  active_plan: "当前已有策略计划执行中。完成或停止该计划后，才能启动新的 Candidate Lab 分析。",
  open_gate: "当前策略任务有待处理确认门。请先完成确认，再启动新的 Candidate Lab 分析。",
  loading: "正在核验 Candidate Lab 状态，完成前暂不能启动新分析。",
  submitting: "Candidate Lab 请求正在提交，请等待当前动作完成。",
});

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function nonEmptyText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function fieldLabel(key) {
  const normalized = String(key || "");
  return FIELD_LABELS[normalized]
    || normalized
      .split("_")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
}

function stablePrimitiveText(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "-";
  return String(value);
}

function readableValue(value, depth = 0) {
  if (depth >= 3) return "…";
  if (Array.isArray(value)) {
    if (!value.length) return "-";
    const rendered = value.slice(0, 12).map((item) => readableValue(item, depth + 1));
    return rendered.join("；") + (value.length > 12 ? "；…" : "");
  }
  if (isRecord(value)) {
    const entries = Object.entries(value).slice(0, 16);
    if (!entries.length) return "-";
    const rendered = entries.map(
      ([key, item]) => `${fieldLabel(key)}：${readableValue(item, depth + 1)}`,
    );
    return rendered.join("；") + (Object.keys(value).length > 16 ? "；…" : "");
  }
  return stablePrimitiveText(value);
}

function safeDownloadUrl(value) {
  const url = nonEmptyText(value);
  return url.startsWith("/api/") ? url : "";
}

function collectionItems(collection) {
  if (!isRecord(collection)) return [];
  const items = Array.isArray(collection.all)
    ? collection.all.filter(isRecord)
    : [];
  const latest = isRecord(collection.latest) ? collection.latest : null;
  if (!latest) return items;
  const latestArtifactId = nonEmptyText(latest.artifact?.artifact_id);
  const alreadyIncluded = items.some((item) => (
    item === latest
    || (
      latestArtifactId
      && nonEmptyText(item.artifact?.artifact_id) === latestArtifactId
    )
  ));
  return alreadyIncluded ? items : [latest, ...items];
}

function collectionTotal(collection) {
  return Number.isInteger(collection?.total) && collection.total >= 0
    ? collection.total
    : null;
}

function evidenceIdentityHtml(item) {
  const artifact = isRecord(item?.artifact) ? item.artifact : {};
  const rows = [
    ["Candidate ID", item?.candidate_id],
    ["Evidence Hash", item?.evidence_hash],
    ["Artifact ID", artifact.artifact_id],
    ["Content Hash", artifact.content_hash],
    ["创建时间", artifact.created_at],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  const downloadUrl = safeDownloadUrl(artifact.download_url);
  return [
    '<dl class="candidate-lab-identity">',
    ...rows.map(([label, value]) => (
      `<div><dt>${escapeHtml(label)}</dt><dd><code>${escapeHtml(value)}</code></dd></div>`
    )),
    "</dl>",
    downloadUrl
      ? `<a class="button compact secondary candidate-lab-download" href="${escapeHtml(downloadUrl)}" download>下载受认证产物</a>`
      : "",
  ].join("");
}

function factRows(value, options = {}) {
  if (!isRecord(value)) return [];
  const excluded = new Set(options.exclude || []);
  return Object.entries(value)
    .filter(([key, item]) => !excluded.has(key) && item !== null && item !== undefined)
    .slice(0, options.limit || 40);
}

function factsTableHtml(value, options = {}) {
  const rows = factRows(value, options);
  if (!rows.length) return "";
  return [
    '<div class="candidate-lab-table-scroll">',
    '<table class="candidate-lab-table candidate-lab-facts"><tbody>',
    ...rows.map(([key, item]) => (
      `<tr><th>${escapeHtml(fieldLabel(key))}</th><td>${escapeHtml(readableValue(item))}</td></tr>`
    )),
    "</tbody></table>",
    "</div>",
  ].join("");
}

function lifecycleHtml(value) {
  if (!isRecord(value)) return "";
  const facts = factsTableHtml(value);
  return facts
    ? `<section class="candidate-lab-subsection"><h5>生命周期</h5>${facts}</section>`
    : "";
}

function riskHtml(value) {
  if (!isRecord(value)) return "";
  const redFlags = Array.isArray(value.red_flags) ? value.red_flags : [];
  const reportGaps = Array.isArray(value.report_info_gaps) ? value.report_info_gaps : [];
  if (!redFlags.length && !reportGaps.length) return "";
  const list = (items, label, tone) => {
    if (!items.length) return "";
    return [
      `<div class="candidate-lab-risk-group" data-tone="${escapeHtml(tone)}">`,
      `<strong>${escapeHtml(label)}</strong>`,
      "<ul>",
      ...items.slice(0, 24).map((item) => `<li>${escapeHtml(readableValue(item))}</li>`),
      "</ul>",
      items.length > 24 ? "<p>其余风险项已由服务端截断。</p>" : "",
      "</div>",
    ].join("");
  };
  return [
    '<section class="candidate-lab-subsection"><h5>风险与报告缺口</h5>',
    list(redFlags, "风险提示", "warn"),
    list(reportGaps, "报告信息缺口", "info"),
    "</section>",
  ].join("");
}

function pointerColumns(pointerKey) {
  if (pointerKey === "bins") {
    return ["feature", "method", "bin_id"];
  }
  if (pointerKey === "cells") {
    return ["cell_id", "row_bin_id", "column_bin_id", "effect"];
  }
  return [
    "leaf_id",
    "fragment_id",
    "rule_id",
    "effect_id",
    "condition",
    "metrics",
  ];
}

function pointerTableHtml(item, pointerKey) {
  const pointers = Array.isArray(item?.pointers?.[pointerKey])
    ? item.pointers[pointerKey].filter(isRecord)
    : [];
  if (!pointers.length) return "";
  const columns = pointerColumns(pointerKey);
  const header = columns.map((key) => `<th>${escapeHtml(fieldLabel(key))}</th>`).join("");
  const rows = pointers.map((pointer) => [
    "<tr>",
    ...columns.map((key) => `<td>${escapeHtml(readableValue(pointer[key]))}</td>`),
    "</tr>",
  ].join("")).join("");
  const total = Number.isInteger(item.total) && item.total >= 0 ? item.total : null;
  const truncation = item.truncated
    ? `<p class="candidate-lab-truncated">已显示 ${escapeHtml(pointers.length)}${total === null ? "" : ` / ${escapeHtml(total)}`} 条，剩余内容已由服务端截断。</p>`
    : "";
  return [
    '<section class="candidate-lab-subsection">',
    `<h5>${pointerKey === "bins" ? "候选分箱" : pointerKey === "cells" ? "矩阵单元格" : "叶节点"}</h5>`,
    '<div class="candidate-lab-table-scroll">',
    `<table class="candidate-lab-table"><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table>`,
    "</div>",
    truncation,
    "</section>",
  ].join("");
}

function candidateDetailHtml(item, pointerKey) {
  const detail = isRecord(item?.detail) ? item.detail : {};
  const title = nonEmptyText(detail.asset_id)
    || nonEmptyText(item?.candidate_id)
    || nonEmptyText(item?.artifact?.artifact_id)
    || "候选证据";
  const detailFacts = factsTableHtml(detail);
  return [
    '<details class="candidate-lab-evidence-card">',
    '<summary>',
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    `<small>${escapeHtml(nonEmptyText(item?.kind) || "已认证候选")}</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看证据</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml(item),
    lifecycleHtml(item.lifecycle),
    detailFacts
      ? `<section class="candidate-lab-subsection"><h5>结果摘要</h5>${detailFacts}</section>`
      : "",
    pointerTableHtml(item, pointerKey),
    riskHtml(item.risks),
    item?.truncated && !Array.isArray(item?.pointers?.[pointerKey])
      ? '<p class="candidate-lab-truncated">该候选明细已由服务端截断。</p>'
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function candidateCollectionHtml(candidates, definition) {
  const collection = isRecord(candidates?.[definition.key])
    ? candidates[definition.key]
    : {};
  const items = collectionItems(collection);
  const total = collectionTotal(collection);
  const countText = total === null ? "" : `${total} 个`;
  const list = items.length
    ? items.map((item) => candidateDetailHtml(item, definition.pointerKey)).join("")
    : '<p class="candidate-lab-empty">暂无受认证结果。先从左侧启动对应分析，完成后会在这里出现。</p>';
  return [
    '<section class="candidate-lab-result-group">',
    '<header class="candidate-lab-result-head">',
    "<div>",
    `<h4>${escapeHtml(definition.title)}</h4>`,
    `<p>${escapeHtml(definition.description)}</p>`,
    "</div>",
    countText ? `<strong>${escapeHtml(countText)}</strong>` : "",
    "</header>",
    collection?.truncated
      ? '<p class="candidate-lab-truncated">候选列表已由服务端截断，仅展示最新的受认证结果。</p>'
      : "",
    `<div class="candidate-lab-result-list">${list}</div>`,
    "</section>",
  ].join("");
}

function poolEntryTableHtml(entries) {
  const rows = Array.isArray(entries) ? entries.filter(isRecord) : [];
  if (!rows.length) return '<p class="candidate-lab-empty">当前 Pool 没有候选条目。</p>';
  const columns = ["position", "rule_id", "source", "action", "execution", "enabled"];
  return [
    '<div class="candidate-lab-table-scroll">',
    '<table class="candidate-lab-table"><thead><tr>',
    ...columns.map((key) => `<th>${escapeHtml(fieldLabel(key))}</th>`),
    "</tr></thead><tbody>",
    ...rows.map((entry) => [
      "<tr>",
      ...columns.map((key) => `<td>${escapeHtml(readableValue(entry[key]))}</td>`),
      "</tr>",
    ].join("")),
    "</tbody></table>",
    "</div>",
  ].join("");
}

function poolItemHtml(item) {
  const title = `${nonEmptyText(item.strategy_type) || "策略"} Pool · revision ${stablePrimitiveText(item.revision)}`;
  const facts = {
    pool_id: item.pool_id,
    strategy_type: item.strategy_type,
    revision: item.revision,
    revision_id: item.revision_id,
    snapshot_hash: item.snapshot_hash,
    status: item.status,
    validation_status: item.validation_status,
    default_action: item.default_action,
  };
  const visibleEntries = Array.isArray(item.entries) ? item.entries.length : 0;
  const total = Number.isInteger(item.total) && item.total >= 0 ? item.total : null;
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-pool-card">',
    '<summary>',
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    `<small>${escapeHtml(nonEmptyText(item.pool_id) || "task-scoped Pool")}</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看 Pool</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml({ artifact: item.artifact }),
    '<section class="candidate-lab-subsection"><h5>Pool 状态</h5>',
    factsTableHtml(facts),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>候选顺序与动作</h5>',
    poolEntryTableHtml(item.entries),
    "</section>",
    item.truncated
      ? `<p class="candidate-lab-truncated">已显示 ${escapeHtml(visibleEntries)}${total === null ? "" : ` / ${escapeHtml(total)}`} 条，剩余 Pool 条目已由服务端截断。</p>`
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function poolCollectionHtml(pools) {
  const collection = isRecord(pools) ? pools : {};
  const items = collectionItems(collection);
  const total = collectionTotal(collection);
  return [
    '<section class="candidate-lab-result-group">',
    '<header class="candidate-lab-result-head">',
    "<div><h4>Strategy Pool</h4><p>当前 task-scoped Pool revision、顺序、动作与来源证据</p></div>",
    total === null ? "" : `<strong>${escapeHtml(total)} 个</strong>`,
    "</header>",
    collection.truncated
      ? '<p class="candidate-lab-truncated">Pool 列表已由服务端截断。</p>'
      : "",
    '<div class="candidate-lab-result-list">',
    items.length
      ? items.map(poolItemHtml).join("")
      : '<p class="candidate-lab-empty">当前还没有 Strategy Pool。</p>',
    "</div>",
    "</section>",
  ].join("");
}

export function strategyCandidateLabResultsHtml(payload = {}) {
  const candidates = isRecord(payload.candidates) ? payload.candidates : {};
  return [
    ...COLLECTION_DEFINITIONS.map(
      (definition) => candidateCollectionHtml(candidates, definition),
    ),
    poolCollectionHtml(payload.pools),
  ].join("");
}

function formField(form, name) {
  return form?.querySelector?.(`[data-candidate-lab-field="${name}"]`) || null;
}

function formValue(form, name) {
  return String(formField(form, name)?.value ?? "").trim();
}

function splitValues(value) {
  return String(value || "")
    .split(/[\s,，、;；]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function uniqueValues(values, label) {
  if (new Set(values).size !== values.length) {
    throw new Error(`${label}不能包含重复值。`);
  }
  return values;
}

function optionalNumber(form, name, { integer = false } = {}) {
  const raw = formValue(form, name);
  if (!raw) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${fieldLabel(name)}必须是${integer ? "整数" : "有限数字"}。`);
  }
  return value;
}

function optionalText(inputs, key, value) {
  const normalized = nonEmptyText(value);
  if (normalized) inputs[key] = normalized;
}

function optionalValue(inputs, key, value) {
  if (value !== undefined) inputs[key] = value;
}

function checkedValues(form, name) {
  const fields = form?.querySelectorAll?.(
    `[data-candidate-lab-field="${name}"]:checked`,
  ) || [];
  return Array.from(fields)
    .map((field) => nonEmptyText(field.value))
    .filter(Boolean);
}

function sentinelValues(form) {
  const raw = formValue(form, "sentinel_values");
  if (!raw) return [];
  const entries = raw
    .split(/[,，、;；\n]+/)
    .map((item) => item.trim())
    .filter(Boolean);
  const values = entries.map((entry) => {
    const separator = entry.indexOf(":");
    if (separator < 1) {
      throw new Error(
        "特殊值必须显式标注类型，例如 text:001 或 number:-9999。",
      );
    }
    const type = entry.slice(0, separator).trim().toLowerCase();
    const value = entry.slice(separator + 1).trim();
    if (!value) throw new Error("特殊值的类型前缀后不能为空。");
    if (type === "text") return value;
    if (type !== "number") {
      throw new Error("特殊值类型只能是 text 或 number。");
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      throw new Error(`特殊值 ${entry} 必须包含有限数字。`);
    }
    return number;
  });
  return uniqueValues(values, "特殊值");
}

function parseFeatureNumberMapping(value, label) {
  const text = String(value || "").trim();
  if (!text) return {};
  const entries = text.split(/[;\n；]+/).map((item) => item.trim()).filter(Boolean);
  const mapping = {};
  for (const entry of entries) {
    const separator = entry.indexOf("=");
    if (separator < 1) {
      throw new Error(`${label}格式应为“字段=切点1,切点2；字段2=切点1”。`);
    }
    const feature = entry.slice(0, separator).trim();
    const points = splitValues(entry.slice(separator + 1)).map((item) => Number(item));
    if (!feature || !points.length || points.some((item) => !Number.isFinite(item))) {
      throw new Error(`${label}必须为字段到有限数字切点的映射。`);
    }
    if (Object.prototype.hasOwnProperty.call(mapping, feature)) {
      throw new Error(`${label}不能重复填写字段 ${feature}。`);
    }
    if (points.some((point, index) => index > 0 && point <= points[index - 1])) {
      throw new Error(`${label}.${feature} 的切点必须严格递增。`);
    }
    mapping[feature] = points;
  }
  return mapping;
}

function parseDirections(value) {
  const text = String(value || "").trim();
  if (!text) return {};
  const entries = text.split(/[;；,\n]+/).map((item) => item.trim()).filter(Boolean);
  const directions = {};
  for (const entry of entries) {
    const separator = entry.indexOf("=");
    if (separator < 1) {
      throw new Error("风险方向格式应为“字段=increasing/decreasing/unordered”。");
    }
    const feature = entry.slice(0, separator).trim();
    const direction = entry.slice(separator + 1).trim();
    if (!["increasing", "decreasing", "unordered"].includes(direction)) {
      throw new Error(`风险方向 ${feature} 只能是 increasing、decreasing 或 unordered。`);
    }
    if (Object.prototype.hasOwnProperty.call(directions, feature)) {
      throw new Error(`风险方向不能重复填写字段 ${feature}。`);
    }
    directions[feature] = direction;
  }
  return directions;
}

function collectUnivariateInputs(form) {
  const features = uniqueValues(splitValues(formValue(form, "features")), "分析字段");
  if (!features.length) throw new Error("请至少填写一个单变量分析字段。");
  const inputs = { features };
  const methods = uniqueValues(checkedValues(form, "methods"), "分箱方法");
  // Empty methods means the operator chose platform defaults. Omitting this
  // field is materially different from sending an invalid empty array.
  if (methods.length) inputs.methods = methods;
  optionalValue(inputs, "bin_count", optionalNumber(form, "bin_count", { integer: true }));
  optionalValue(inputs, "min_bin_pct", optionalNumber(form, "min_bin_pct"));
  optionalText(inputs, "loan_amount_col", formValue(form, "loan_amount_col"));
  optionalText(inputs, "overdue_amount_col", formValue(form, "overdue_amount_col"));
  const sentinels = sentinelValues(form);
  if (sentinels.length) inputs.sentinel_values = sentinels;
  const manualBreakpoints = parseFeatureNumberMapping(
    formValue(form, "manual_breakpoints"),
    "人工切点",
  );
  if (methods.includes("manual") && !Object.keys(manualBreakpoints).length) {
    throw new Error("选择人工切点分箱时，必须填写人工切点。");
  }
  if (!methods.includes("manual") && Object.keys(manualBreakpoints).length) {
    throw new Error("只有选择人工切点分箱时才能填写人工切点。");
  }
  if (Object.keys(manualBreakpoints).length) {
    inputs.manual_breakpoints = manualBreakpoints;
  }
  return inputs;
}

function selectedProjectionOption(form, name, label) {
  const field = formField(form, name);
  const selected = Array.from(field?.selectedOptions || []);
  const option = selected[0] || null;
  if (
    !option
    || option.dataset?.candidateLabProjection !== "1"
    || !nonEmptyText(option.value)
  ) {
    throw new Error(`请从当前任务的受认证投影中选择${label}。`);
  }
  return option;
}

function selectedProjectionValues(form, name, label) {
  const field = formField(form, name);
  const selected = Array.from(field?.selectedOptions || []);
  if (!selected.length) {
    throw new Error(`请从当前候选投影中至少选择一个${label}。`);
  }
  if (selected.some((option) => option.dataset?.candidateLabProjection !== "1")) {
    throw new Error(`${label}必须来自当前候选投影。`);
  }
  return uniqueValues(
    selected.map((option) => nonEmptyText(option.value)).filter(Boolean),
    label,
  );
}

function parseRefinementMergeGroups(value, allowedBinIds) {
  const text = String(value || "").trim();
  if (!text) return [];
  const groups = text
    .split(/[;；\n]+/)
    .map((group) => group.split("+").map((item) => item.trim()).filter(Boolean))
    .filter((group) => group.length);
  const seen = new Set();
  for (const group of groups) {
    if (group.length < 2) {
      throw new Error("每个合并组至少需要两个 Bin ID，并使用 + 连接。");
    }
    for (const binId of group) {
      if (!allowedBinIds.has(binId)) {
        throw new Error(`合并组中的 ${binId} 不属于当前投影候选。`);
      }
      if (seen.has(binId)) {
        throw new Error(`合并组不能重复使用 Bin ID ${binId}。`);
      }
      seen.add(binId);
    }
  }
  return groups;
}

function collectFreshRefinementInputs(form) {
  const feature = formValue(form, "refinement_feature");
  const method = formValue(form, "refinement_method");
  const operator = formValue(form, "risk_operator");
  const riskValue = optionalNumber(form, "risk_value");
  if (!feature) throw new Error("请填写需要细化的字段。");
  if (!method) throw new Error("请选择细化分箱方法。");
  if (![">=", ">", "<=", "<"].includes(operator)) {
    throw new Error("请选择有效的坏率阈值运算符。");
  }
  if (riskValue === undefined || riskValue < 0 || riskValue > 1) {
    throw new Error("坏率阈值必须是 0 到 1 之间的数字。");
  }
  const inputs = {
    feature,
    method,
    selection: { risk_threshold: { operator, value: riskValue } },
  };
  optionalValue(inputs, "bin_count", optionalNumber(form, "bin_count", { integer: true }));
  optionalValue(inputs, "min_bin_pct", optionalNumber(form, "min_bin_pct"));
  optionalText(inputs, "loan_amount_col", formValue(form, "loan_amount_col"));
  optionalText(inputs, "overdue_amount_col", formValue(form, "overdue_amount_col"));
  const sentinels = sentinelValues(form);
  if (sentinels.length) inputs.sentinel_values = sentinels;

  const rawBreakpoints = formValue(form, "refinement_manual_breakpoints");
  if (method !== "manual" && rawBreakpoints) {
    throw new Error("只有选择人工切点分箱时才能填写人工切点。");
  }
  if (method === "manual") {
    const points = splitValues(rawBreakpoints).map((value) => Number(value));
    if (!points.length || points.some((value) => !Number.isFinite(value))) {
      throw new Error("人工细化必须填写有限数字切点。");
    }
    if (points.some((point, index) => index > 0 && point <= points[index - 1])) {
      throw new Error("人工细化切点必须严格递增。");
    }
    inputs.manual_breakpoints = { [feature]: points };
  }
  optionalText(inputs, "selection_reason", formValue(form, "selection_reason"));
  return inputs;
}

function collectExistingRefinementInputs(form) {
  const source = selectedProjectionOption(
    form,
    "source_candidate_id",
    "已有 Candidate",
  );
  const pair = selectedProjectionOption(
    form,
    "source_feature_method",
    "字段与方法",
  );
  const sourceCandidateId = nonEmptyText(source.value);
  const feature = nonEmptyText(pair.dataset?.feature);
  const method = nonEmptyText(pair.dataset?.method);
  if (
    !feature
    || !method
    || pair.dataset?.sourceCandidateId !== sourceCandidateId
  ) {
    throw new Error("字段与方法必须属于当前选择的投影 Candidate。");
  }
  const sourceBinIds = selectedProjectionValues(
    form,
    "source_bin_ids",
    "源 Bin",
  );
  const binField = formField(form, "source_bin_ids");
  const allowedBinIds = new Set(
    Array.from(binField?.options || [])
      .filter((option) => (
        option.dataset?.candidateLabProjection === "1"
        && option.dataset?.sourceCandidateId === sourceCandidateId
        && option.dataset?.feature === feature
        && option.dataset?.method === method
      ))
      .map((option) => nonEmptyText(option.value))
      .filter(Boolean),
  );
  if (sourceBinIds.some((binId) => !allowedBinIds.has(binId))) {
    throw new Error("源 Bin 必须属于当前 Candidate 的已选字段与方法。");
  }
  const inputs = {
    feature,
    method,
    source_candidate_id: sourceCandidateId,
    selection: { source_bin_ids: sourceBinIds },
  };
  const mergeGroups = parseRefinementMergeGroups(
    formValue(form, "merge_groups"),
    allowedBinIds,
  );
  if (mergeGroups.length) inputs.merge_groups = mergeGroups;
  optionalText(inputs, "selection_reason", formValue(form, "selection_reason"));
  return inputs;
}

function collectRefinementInputs(form) {
  const mode = formValue(form, "refinement_mode");
  if (mode === "fresh") return collectFreshRefinementInputs(form);
  if (mode === "existing") return collectExistingRefinementInputs(form);
  throw new Error("请选择重新分析或细化已有 Candidate。");
}

function collectCrossInputs(form) {
  const xFeature = formValue(form, "x_feature");
  const yFeature = formValue(form, "y_feature");
  const xMethod = formValue(form, "x_method");
  const yMethod = formValue(form, "y_method");
  if (!xFeature || !yFeature) throw new Error("请完整填写 X 轴与 Y 轴字段。");
  if (xFeature === yFeature) throw new Error("X 轴与 Y 轴必须使用不同字段。");
  if (!xMethod || !yMethod) throw new Error("请为两个交叉轴选择分箱方法。");
  const inputs = {
    x_feature: xFeature,
    x_method: xMethod,
    y_feature: yFeature,
    y_method: yMethod,
  };
  optionalValue(inputs, "bin_count", optionalNumber(form, "bin_count", { integer: true }));
  optionalValue(inputs, "min_bin_pct", optionalNumber(form, "min_bin_pct"));
  optionalText(inputs, "loan_amount_col", formValue(form, "loan_amount_col"));
  optionalText(inputs, "overdue_amount_col", formValue(form, "overdue_amount_col"));
  const sentinels = sentinelValues(form);
  if (sentinels.length) inputs.sentinel_values = sentinels;

  const manualBreakpoints = {};
  for (const [feature, method, field] of [
    [xFeature, xMethod, "x_manual_breakpoints"],
    [yFeature, yMethod, "y_manual_breakpoints"],
  ]) {
    const raw = formValue(form, field);
    if (method !== "manual" && raw) {
      throw new Error(`${feature} 未选择人工切点分箱，不能填写人工切点。`);
    }
    if (method !== "manual") continue;
    const points = splitValues(raw).map((value) => Number(value));
    if (!points.length || points.some((value) => !Number.isFinite(value))) {
      throw new Error(`${feature} 选择人工切点分箱时必须填写有限数字切点。`);
    }
    if (points.some((point, index) => index > 0 && point <= points[index - 1])) {
      throw new Error(`${feature} 的人工切点必须严格递增。`);
    }
    manualBreakpoints[feature] = points;
  }
  if (Object.keys(manualBreakpoints).length) {
    inputs.manual_breakpoints = manualBreakpoints;
  }
  return inputs;
}

function collectTreeInputs(form) {
  const features = uniqueValues(splitValues(formValue(form, "features")), "建树字段");
  if (!features.length) throw new Error("请至少填写一个自动树建树字段。");
  const inputs = { features };
  const directions = parseDirections(formValue(form, "directions"));
  const unknownDirectionFeatures = Object.keys(directions)
    .filter((feature) => !features.includes(feature));
  if (unknownDirectionFeatures.length) {
    throw new Error(`风险方向引用了未选择字段：${unknownDirectionFeatures.join("、")}。`);
  }
  if (Object.keys(directions).length) inputs.directions = directions;
  optionalValue(inputs, "max_depth", optionalNumber(form, "max_depth", { integer: true }));
  optionalValue(
    inputs,
    "min_leaf_count",
    optionalNumber(form, "min_leaf_count", { integer: true }),
  );
  optionalValue(
    inputs,
    "min_weight_fraction_leaf",
    optionalNumber(form, "min_weight_fraction_leaf"),
  );
  optionalValue(inputs, "seed", optionalNumber(form, "seed", { integer: true }));
  optionalText(inputs, "sample_weight_col", formValue(form, "sample_weight_col"));
  optionalText(inputs, "loan_amount_col", formValue(form, "loan_amount_col"));
  optionalText(inputs, "overdue_amount_col", formValue(form, "overdue_amount_col"));
  return inputs;
}

export function collectStrategyCandidateLabRequest(form) {
  const workflow = nonEmptyText(form?.dataset?.candidateLabWorkflow);
  if (!STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(workflow)) {
    throw new Error("Candidate Lab 表单包含未开放的策略 workflow。");
  }
  const workflowInputs = {
    univariate_candidate_analysis: collectUnivariateInputs,
    univariate_candidate_refinement: collectRefinementInputs,
    cross_matrix_analysis: collectCrossInputs,
    automatic_tree_candidate_build: collectTreeInputs,
  }[workflow](form);
  return {
    request_kind: "standard_workflow",
    workflow,
    workflow_inputs: workflowInputs,
  };
}

function setFormError(form, message) {
  const target = form?.querySelector?.("[data-candidate-lab-form-error]");
  if (target) target.textContent = String(message || "");
}

function localBlockedReason(dependencies) {
  const value = dependencies.getBlockedReason?.();
  return ["active_plan", "open_gate"].includes(value) ? value : "";
}

function blockedReason(state, dependencies) {
  if (state.submitting) return "submitting";
  const local = localBlockedReason(dependencies);
  if (local) return local;
  if (state.loading && !state.payload) return "loading";
  if (state.payload?.can_start === false) {
    return nonEmptyText(state.payload.blocked_reason) || "active_plan";
  }
  return "";
}

function panelStatusText(state, dependencies) {
  if (state.error) return state.error;
  const reason = blockedReason(state, dependencies);
  if (reason) return BLOCKED_REASON_COPY[reason] || "当前 Candidate Lab 暂不可启动新分析。";
  if (state.loading) return "正在刷新受认证候选证据…";
  return "只展示平台已经登记并重新验真的候选；所有启动动作复用 Agent 的确定性治理链。";
}

function stateSnapshot(state) {
  return {
    taskId: state.taskId,
    payload: state.payload,
    loading: state.loading,
    submitting: state.submitting,
    error: state.error,
  };
}

function refinementForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="univariate_candidate_refinement"]',
  ) || null;
}

function setRefinementPanelVisible(panel, visible) {
  if (!panel) return;
  panel.classList?.toggle?.("hidden", !visible);
  panel.setAttribute?.("aria-hidden", visible ? "false" : "true");
  const controls = panel.querySelectorAll?.("input, select, textarea, button") || [];
  for (const control of controls) control.disabled = !visible;
}

function syncRefinementMode(form) {
  if (!form) return;
  const mode = formValue(form, "refinement_mode") || "fresh";
  const panels = form.querySelectorAll?.("[data-candidate-lab-refinement-panel]") || [];
  for (const modePanel of panels) {
    setRefinementPanelVisible(
      modePanel,
      modePanel.dataset?.candidateLabRefinementPanel === mode,
    );
  }
}

function univariateProjectionCandidates(payload) {
  const collection = isRecord(payload?.candidates?.univariate)
    ? payload.candidates.univariate
    : {};
  const seen = new Set();
  return collectionItems(collection).filter((item) => {
    const candidateId = nonEmptyText(item.candidate_id);
    if (!candidateId || seen.has(candidateId)) return false;
    seen.add(candidateId);
    return true;
  });
}

function univariateCandidatePairs(candidate) {
  const bins = Array.isArray(candidate?.pointers?.bins)
    ? candidate.pointers.bins.filter(isRecord)
    : [];
  const seen = new Set();
  return bins.reduce((pairs, bin) => {
    const feature = nonEmptyText(bin.feature);
    const method = nonEmptyText(bin.method);
    const key = `${feature}\u001f${method}`;
    if (!feature || !method || seen.has(key)) return pairs;
    seen.add(key);
    pairs.push({ feature, method });
    return pairs;
  }, []);
}

function projectionOptionHtml(value, label, data = {}) {
  const attributes = Object.entries(data)
    .map(([key, item]) => ` data-${key}="${escapeHtml(item)}"`)
    .join("");
  return `<option value="${escapeHtml(value)}"${attributes}>${escapeHtml(label)}</option>`;
}

function selectContainsValue(select, value) {
  return Array.from(select?.options || []).some((option) => option.value === value);
}

function selectedValues(select) {
  return Array.from(select?.selectedOptions || [])
    .map((option) => nonEmptyText(option.value))
    .filter(Boolean);
}

function syncRefinementCandidateControls(form, payload, { preserveBins = true } = {}) {
  if (!form) return;
  const candidates = univariateProjectionCandidates(payload);
  const sourceSelect = formField(form, "source_candidate_id");
  const pairSelect = formField(form, "source_feature_method");
  const binSelect = formField(form, "source_bin_ids");
  if (!sourceSelect || !pairSelect || !binSelect) return;

  const previousCandidateId = nonEmptyText(sourceSelect.value);
  sourceSelect.innerHTML = [
    '<option value="">请选择当前任务 Candidate</option>',
    ...candidates.map((candidate) => {
      const candidateId = nonEmptyText(candidate.candidate_id);
      const binCount = Array.isArray(candidate?.pointers?.bins)
        ? candidate.pointers.bins.length
        : 0;
      return projectionOptionHtml(
        candidateId,
        `${candidateId} · ${binCount} 个可见 Bin`,
        { "candidate-lab-projection": "1" },
      );
    }),
  ].join("");
  if (selectContainsValue(sourceSelect, previousCandidateId)) {
    sourceSelect.value = previousCandidateId;
  } else if (candidates.length) {
    sourceSelect.value = nonEmptyText(candidates[0].candidate_id);
  }

  const candidateId = nonEmptyText(sourceSelect.value);
  const candidate = candidates.find((item) => item.candidate_id === candidateId);
  const pairs = univariateCandidatePairs(candidate);
  const previousPairValue = nonEmptyText(pairSelect.value);
  pairSelect.innerHTML = [
    '<option value="">请选择字段与方法</option>',
    ...pairs.map(({ feature, method }, index) => projectionOptionHtml(
      `pair-${index}`,
      `${feature} · ${method}`,
      {
        "candidate-lab-projection": "1",
        "source-candidate-id": candidateId,
        feature,
        method,
      },
    )),
  ].join("");
  if (selectContainsValue(pairSelect, previousPairValue)) {
    pairSelect.value = previousPairValue;
  } else if (pairs.length) {
    pairSelect.value = "pair-0";
  }

  const pairOption = Array.from(pairSelect.selectedOptions || [])[0] || null;
  const feature = nonEmptyText(pairOption?.dataset?.feature);
  const method = nonEmptyText(pairOption?.dataset?.method);
  const previousBinIds = preserveBins ? new Set(selectedValues(binSelect)) : new Set();
  const bins = Array.isArray(candidate?.pointers?.bins)
    ? candidate.pointers.bins.filter((bin) => (
      isRecord(bin)
      && bin.feature === feature
      && bin.method === method
      && nonEmptyText(bin.bin_id)
    ))
    : [];
  binSelect.innerHTML = bins.map((bin) => projectionOptionHtml(
    nonEmptyText(bin.bin_id),
    `${nonEmptyText(bin.bin_id)} · ${readableValue(bin.condition)}`,
    {
      "candidate-lab-projection": "1",
      "source-candidate-id": candidateId,
      feature,
      method,
    },
  )).join("");
  for (const option of Array.from(binSelect.options || [])) {
    option.selected = previousBinIds.has(option.value);
  }

  const empty = form.querySelector?.("[data-candidate-lab-refinement-empty]");
  if (empty) {
    empty.textContent = candidates.length
      ? bins.length
        ? "按住 Command/Ctrl 可多选；这里只能选择投影中可见的 Bin。"
        : "该 Candidate 的当前字段/方法没有可选择的投影 Bin。"
      : "当前任务尚无单变量 Candidate，请先运行单变量分析或使用“重新分析”。";
  }
}

function syncRefinementForm(root, payload, options = {}) {
  const form = refinementForm(root);
  if (!form) return;
  syncRefinementMode(form);
  syncRefinementCandidateControls(form, payload, options);
}

function submissionClarificationText(result) {
  const direct = nonEmptyText(result?.message);
  if (direct) return direct;
  const messages = Array.isArray(result?.messages) ? result.messages : [];
  const assistant = [...messages].reverse().find((message) => (
    message?.role === "assistant" && nonEmptyText(message.content)
  ));
  if (assistant) return nonEmptyText(assistant.content);
  const code = nonEmptyText(result?.code);
  return code ? `平台需要补充信息（${code}）。` : "平台需要补充信息后才能启动该分析。";
}

function submissionWasAccepted(result) {
  return ["accepted", "ok", "plan_started"].includes(nonEmptyText(result?.status));
}

export function createStrategyCandidateLabController(dependencies = {}) {
  const $ = dependencies.$ || ((id) => document.getElementById(id));
  const fetchCandidateLab = dependencies.getStrategyCandidateLab || getStrategyCandidateLab;
  const submitCandidateLab = dependencies.submitStrategyCandidateLabRequest
    || submitStrategyCandidateLabRequest;
  const state = {
    taskId: "",
    payload: null,
    loading: false,
    submitting: false,
    error: "",
  };
  let operation = 0;
  let boundRoot = null;
  let activeRefresh = null;

  function selectedTask() {
    return dependencies.getSelectedTask?.() || null;
  }

  function selectedTaskId() {
    return nonEmptyText(dependencies.getSelectedTaskId?.());
  }

  function panel() {
    return $("strategyCandidateLabPanel");
  }

  function renderAvailability() {
    const root = panel();
    if (!root) return;
    const reason = blockedReason(state, dependencies);
    root.dataset.candidateLabBlockedReason = reason;
    const controls = root.querySelectorAll?.(
      "[data-candidate-lab-form] input, "
      + "[data-candidate-lab-form] select, "
      + "[data-candidate-lab-form] textarea, "
      + "[data-candidate-lab-form] button",
    ) || [];
    for (const control of controls) {
      const modePanel = control.closest?.("[data-candidate-lab-refinement-panel]");
      const hiddenByMode = Boolean(modePanel?.classList?.contains?.("hidden"));
      control.disabled = Boolean(reason) || hiddenByMode;
    }
    const status = $("strategyCandidateLabStatus");
    if (status) {
      status.textContent = panelStatusText(state, dependencies);
      status.dataset.tone = state.error ? "error" : reason ? "blocked" : state.loading ? "loading" : "ready";
    }
    const retries = root.querySelectorAll?.("[data-candidate-lab-retry]") || [];
    for (const retry of retries) retry.disabled = state.loading;
  }

  function render() {
    const root = panel();
    if (!root) return;
    const task = selectedTask();
    const visible = Boolean(task && task.task_type === "strategy" && state.taskId);
    root.classList.toggle("hidden", !visible);
    root.setAttribute("aria-hidden", visible ? "false" : "true");
    if (!visible) return;
    const results = $("strategyCandidateLabResults");
    if (results) {
      if (state.payload) {
        results.innerHTML = strategyCandidateLabResultsHtml(state.payload);
      } else if (state.error) {
        results.innerHTML = [
          '<div class="candidate-lab-load-state" data-tone="error">',
          "<strong>Candidate Lab 读取失败</strong>",
          `<p>${escapeHtml(state.error)}</p>`,
          '<button type="button" class="button compact secondary" data-candidate-lab-retry="1">重新读取</button>',
          "</div>",
        ].join("");
      } else {
        results.innerHTML = [
          '<div class="candidate-lab-load-state">',
          "<strong>正在核验候选证据</strong>",
          "<p>平台正在读取 task-owned artifact 与当前 Strategy Pool。</p>",
          "</div>",
        ].join("");
      }
    }
    syncRefinementForm(root, state.payload);
    renderAvailability();
  }

  function resetForms() {
    const root = panel();
    const forms = root?.querySelectorAll?.("[data-candidate-lab-form]") || [];
    for (const form of forms) {
      form.reset?.();
      setFormError(form, "");
    }
    syncRefinementForm(root, state.payload, { preserveBins: false });
  }

  function clear() {
    activeRefresh?.controller?.abort?.();
    activeRefresh = null;
    operation += 1;
    state.taskId = "";
    state.payload = null;
    state.loading = false;
    state.submitting = false;
    state.error = "";
    render();
    return stateSnapshot(state);
  }

  async function refresh(taskId = selectedTaskId(), { silent = false } = {}) {
    const requestedTaskId = nonEmptyText(taskId);
    if (
      !requestedTaskId
      || selectedTask()?.task_type !== "strategy"
      || selectedTaskId() !== requestedTaskId
    ) {
      return stateSnapshot(state);
    }
    if (activeRefresh?.taskId === requestedTaskId) {
      return activeRefresh.promise;
    }
    if (activeRefresh) {
      activeRefresh.controller?.abort?.();
      activeRefresh = null;
    }
    if (state.taskId !== requestedTaskId) {
      state.taskId = requestedTaskId;
      state.payload = null;
      state.error = "";
      resetForms();
    }
    const refreshOperation = ++operation;
    state.loading = true;
    if (!silent) state.error = "";
    render();
    const controller = typeof AbortController === "undefined"
      ? null
      : new AbortController();
    const refreshToken = {
      taskId: requestedTaskId,
      controller,
      promise: null,
    };
    const promise = (async () => {
      try {
        const payload = await fetchCandidateLab(
          requestedTaskId,
          controller ? { signal: controller.signal } : {},
        );
        if (refreshOperation !== operation || selectedTaskId() !== requestedTaskId) {
          return stateSnapshot(state);
        }
        if (!isRecord(payload) || payload.task_id !== requestedTaskId) {
          throw new Error("Candidate Lab 响应不属于当前任务。");
        }
        state.payload = payload;
        state.error = "";
        state.loading = false;
        render();
      } catch (error) {
        if (refreshOperation !== operation || selectedTaskId() !== requestedTaskId) {
          return stateSnapshot(state);
        }
        state.loading = false;
        if (controller?.signal?.aborted || error?.name === "AbortError") {
          render();
          return stateSnapshot(state);
        }
        state.error = error?.message || "Candidate Lab 读取失败。";
        render();
      } finally {
        if (activeRefresh === refreshToken) activeRefresh = null;
      }
      return stateSnapshot(state);
    })();
    refreshToken.promise = promise;
    activeRefresh = refreshToken;
    return promise;
  }

  async function selectTask(task) {
    const taskId = nonEmptyText(task?.id);
    if (!taskId || task?.task_type !== "strategy") {
      return clear();
    }
    if (state.taskId !== taskId) {
      activeRefresh?.controller?.abort?.();
      activeRefresh = null;
      operation += 1;
      state.taskId = taskId;
      state.payload = null;
      state.loading = true;
      state.submitting = false;
      state.error = "";
      resetForms();
      render();
    }
    return refresh(taskId);
  }

  async function submit(form) {
    const taskId = selectedTaskId();
    if (
      !taskId
      || selectedTask()?.task_type !== "strategy"
      || state.taskId !== taskId
    ) {
      const message = "缺少当前策略任务，请重新选择任务后再试。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      return null;
    }
    const reason = blockedReason(state, dependencies);
    if (reason) {
      const message = BLOCKED_REASON_COPY[reason] || "当前 Candidate Lab 暂不可启动新分析。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      renderAvailability();
      return null;
    }

    let strategyRequest;
    try {
      strategyRequest = collectStrategyCandidateLabRequest(form);
    } catch (error) {
      const message = error?.message || "Candidate Lab 表单输入无效。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      return null;
    }

    const workflow = strategyRequest.workflow;
    const content = WORKFLOW_LABELS[workflow] || "从 Candidate Lab 启动策略分析";
    setFormError(form, "");
    state.submitting = true;
    renderAvailability();
    dependencies.setActionStatus?.(`${content}…`, "busy");
    const requestTaskId = taskId;
    try {
      const requestPromise = submitCandidateLab(requestTaskId, strategyRequest, content);
      const pollPromise = Promise.resolve(
        dependencies.pollAgentMessagesUntilSettled?.(
          requestTaskId,
          requestPromise,
          { preserveOptimistic: false },
        ),
      ).catch(() => {});
      const result = await requestPromise;
      await pollPromise;
      if (selectedTaskId() !== requestTaskId) return result;
      if (Array.isArray(result?.messages)) {
        dependencies.setAgentMessages?.(result.messages);
      }
      dependencies.renderAgentConversation?.();
      dependencies.resetPlanFetchThrottle?.(requestTaskId);
      dependencies.renderWorkflowStepper?.({ force: true });
      if (result?.status === "clarification_required") {
        state.submitting = false;
        const message = submissionClarificationText(result);
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "info");
        renderAvailability();
        return result;
      }
      if (!submissionWasAccepted(result)) {
        throw new Error(
          submissionClarificationText(result)
          || `${content}未被平台接受。`,
        );
      }
      await dependencies.refreshAgentMessages?.(requestTaskId);
      state.submitting = false;
      await refresh(requestTaskId);
      dependencies.setActionStatus?.(`${content}已提交。`, "success");
      return result;
    } catch (error) {
      if (selectedTaskId() !== requestTaskId) return null;
      state.submitting = false;
      const message = error?.message || `${content}失败。`;
      // Do not reset or re-render the static form: every operator input stays
      // available for correction and retry after a failed request.
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      renderAvailability();
      return null;
    } finally {
      if (selectedTaskId() === requestTaskId) {
        state.submitting = false;
        dependencies.resetPlanFetchThrottle?.(requestTaskId);
        dependencies.renderWorkflowStepper?.({ force: true });
        renderAvailability();
      }
    }
  }

  function handleSubmit(event) {
    const form = event.target?.closest?.("[data-candidate-lab-form]");
    if (!form) return false;
    event.preventDefault?.();
    void submit(form);
    return true;
  }

  function handleClick(event) {
    const retry = event.target?.closest?.("[data-candidate-lab-retry]");
    if (!retry) return false;
    event.preventDefault?.();
    void refresh();
    return true;
  }

  function handleChange(event) {
    const field = event.target?.closest?.("[data-candidate-lab-field]");
    const form = field?.closest?.(
      '[data-candidate-lab-workflow="univariate_candidate_refinement"]',
    );
    if (!field || !form) return false;
    const fieldName = field.dataset?.candidateLabField;
    if (fieldName === "refinement_mode") {
      syncRefinementMode(form);
    } else if (
      fieldName === "source_candidate_id"
      || fieldName === "source_feature_method"
    ) {
      syncRefinementCandidateControls(form, state.payload, { preserveBins: false });
    } else {
      return false;
    }
    renderAvailability();
    return true;
  }

  function bind(root = document) {
    if (!root || boundRoot === root || typeof root.addEventListener !== "function") return;
    if (boundRoot) unbind();
    boundRoot = root;
    root.addEventListener("submit", handleSubmit);
    root.addEventListener("click", handleClick);
    root.addEventListener("change", handleChange);
  }

  function unbind() {
    if (!boundRoot) return;
    boundRoot.removeEventListener?.("submit", handleSubmit);
    boundRoot.removeEventListener?.("click", handleClick);
    boundRoot.removeEventListener?.("change", handleChange);
    boundRoot = null;
  }

  return {
    bind,
    clear,
    getState: () => stateSnapshot(state),
    handleChange,
    handleClick,
    handleSubmit,
    refresh,
    render,
    renderAvailability,
    selectTask,
    submit,
    unbind,
  };
}
