import { escapeHtml } from "../ui-utils.js";
import {
  getStrategyCandidateLab,
  submitStrategyCandidateLabRequest,
} from "./api_v2.js";

export const STRATEGY_CANDIDATE_LAB_WORKFLOWS = Object.freeze([
  "strategy_project_context",
  "strategy_sample_design_v2",
  "univariate_candidate_analysis",
  "univariate_candidate_refinement",
  "cross_matrix_analysis",
  "cross_matrix_candidate_search",
  "cross_matrix_candidate_build_from_search",
  "cross_rule_search",
  "cross_rule_candidate_build_from_search",
  "automatic_tree_candidate_build",
  "scorecard_band_build",
  "scorecard_cutoff_selection",
  "candidate_monthly_stability",
  "strategy_pool_add_candidate",
  "strategy_pool_compile",
  "strategy_pool_remove_entry",
  "strategy_pool_set_action",
  "strategy_pool_reorder",
  "strategy_pool_apply",
  "strategy_pool_validation",
  "strategy_pool_stability",
  "strategy_pool_impact",
  "strategy_impact_cube",
  "strategy_pool_materialize",
  "strategy_lifecycle_adopt",
  "strategy_dsl_delivery",
  "strategy_report_bundle_v2",
  "voting_candidate_search",
  "voting_candidate_build_from_search",
  "interactive_tree_split_search",
  "interactive_tree_auto_continuation",
  "interactive_tree_revision",
  "interactive_tree_frontier_group_materialization",
  "interactive_tree_frontier_materialization",
]);

const WORKFLOW_LABELS = Object.freeze({
  strategy_project_context: "固化当前项目现状与历史材料",
  strategy_sample_design_v2: "创建双人群 SampleDesign V2",
  univariate_candidate_analysis: "启动单变量候选分析",
  univariate_candidate_refinement: "启动单变量候选细化",
  cross_matrix_analysis: "启动二维 Cross Matrix",
  cross_matrix_candidate_search: "搜索 Cross Matrix 字段组合",
  cross_matrix_candidate_build_from_search: "从搜索结果构建 Cross Matrix 候选",
  cross_rule_search: "搜索 2D/3D Cross 阈值规则",
  cross_rule_candidate_build_from_search: "构建指定 Cross 阈值规则候选",
  automatic_tree_candidate_build: "启动自动规则树",
  scorecard_band_build: "生成评分卡分档证据",
  scorecard_cutoff_selection: "记录评分卡 Cutoff 选择",
  candidate_monthly_stability: "测算候选逐月稳定性",
  strategy_pool_add_candidate: "把已物化候选加入 Strategy Pool",
  strategy_pool_compile: "编译预览当前 Strategy Pool",
  strategy_pool_remove_entry: "从当前 Strategy Pool 移除条目",
  strategy_pool_set_action: "修改当前 Strategy Pool 条目动作",
  strategy_pool_reorder: "完整重排当前 Strategy Pool",
  strategy_pool_apply: "应用当前 Strategy Pool",
  strategy_pool_validation: "执行 Strategy Pool 独立样本回放验证",
  strategy_pool_stability: "测算 Strategy Pool 稳定性",
  strategy_pool_impact: "测算 Strategy Pool 影响",
  strategy_impact_cube: "生成统一策略 ImpactCube",
  strategy_pool_materialize: "把当前 Strategy Pool 物化为草稿策略",
  strategy_lifecycle_adopt: "提交策略本地采纳确认",
  strategy_dsl_delivery: "生成策略等价代码交付包",
  strategy_report_bundle_v2: "形成策略迭代评审报告",
  voting_candidate_search: "搜索 Voting 组合",
  voting_candidate_build_from_search: "从搜索结果构建 Voting 候选",
  interactive_tree_split_search: "搜索交互树节点分裂候选",
  interactive_tree_auto_continuation: "从明确候选受控续建交互树",
  interactive_tree_revision: "创建不可变交互式树修订",
  interactive_tree_frontier_group_materialization: "物化交互树前沿 OR 分组",
  interactive_tree_frontier_materialization: "物化交互树前沿节点",
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
    key: "cross_search",
    title: "Cross 自动搜索",
    description: "受认证单变量字段的两两组合搜索、交互增益与稀疏性证据",
    pointerKey: "",
  },
  {
    key: "cross_rule_search",
    title: "Cross 阈值规则搜索",
    description: "有预算的 2D/3D 阈值组合、约束结果与精确规则指针",
    pointerKey: "",
  },
  {
    key: "cross_rule_candidate",
    title: "Cross 阈值规则候选",
    description: "由用户精确点名规则后物化的可入池候选",
    pointerKey: "",
  },
  {
    key: "automatic_tree",
    title: "自动规则树",
    description: "完整拟合拓扑、可见节点、当前 frontier 与现成效果证据",
    pointerKey: "leaves",
  },
  {
    key: "interactive_tree_revision",
    title: "交互式树修订",
    description: "每条不可变分支各自保留完整拓扑、frontier、历史与回放证据",
    pointerKey: "frontier",
  },
  {
    key: "interactive_tree_split_search",
    title: "树节点分裂候选",
    description: "全特征或指定特征的有预算阈值试算，仅保留聚合风险证据",
    pointerKey: "",
  },
  {
    key: "scorecard_band",
    title: "评分卡分档",
    description: "原始 PD 分档、评分卡分数与 Cutoff 两侧观测效果",
    pointerKey: "bands",
  },
  {
    key: "scorecard_cutoff_selection",
    title: "Cutoff 选择记录",
    description: "人工选择的 Cutoff 指针及其受认证来源证据",
    pointerKey: "",
  },
  {
    key: "voting_search",
    title: "Voting 组合搜索",
    description: "确定性枚举结果、约束资格与精确组合指针",
    pointerKey: "",
  },
]);

const VOTING_SEARCH_METRICS = Object.freeze([
  "hit_count",
  "hit_share",
  "good_count",
  "bad_count",
  "bad_rate",
  "lift",
  "bad_capture_rate",
  "weighted_hit_total",
  "weighted_hit_share",
  "weighted_good_total",
  "weighted_bad_total",
  "weighted_bad_rate",
  "weighted_bad_capture_rate",
  "hit_amount",
  "hit_amount_share",
  "good_amount",
  "bad_amount",
  "bad_amount_rate",
  "bad_amount_capture_rate",
]);

const VOTING_RULE_ID_RE = /^candidate-rule-[0-9a-f]{32}$/;
const VOTING_SEARCH_ID_RE = /^voting-search-[0-9a-f]{32}$/;
const VOTING_COMBO_ID_RE = /^voting-combo-[0-9a-f]{32}$/;
const CROSS_SEARCH_ID_RE = /^cross-search-[0-9a-f]{32}$/;
const CROSS_PAIR_ID_RE = /^cross-pair-[0-9a-f]{32}$/;
const CROSS_RULE_SEARCH_ID_RE = /^cross-rule-search-[0-9a-f]{32}$/;
const CROSS_RULE_ID_RE = /^cross-rule-[0-9a-f]{32}$/;
const INTERACTIVE_TREE_SOURCE_ID_RE = /^(?:candidate-asset-[0-9a-f]{32}|interactive-tree-revision-[0-9a-f]{32})$/;
const INTERACTIVE_TREE_NODE_ID_RE = /^node-[0-9a-f]{20}$/;
const INTERACTIVE_TREE_REVISION_ID_RE = /^interactive-tree-revision-[0-9a-f]{32}$/;
const INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE = /^(?:node|leaf)-[0-9a-f]{20}$/;
const INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE = /^interactive-tree-split-search-[0-9a-f]{32}$/;
const INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE = /^interactive-tree-split-candidate-[0-9a-f]{32}$/;
const STRATEGY_POOL_TYPES = Object.freeze([
  "approval",
  "reject",
  "limit",
  "pricing",
  "segmentation",
]);
const STRATEGY_POOL_ACTION_TYPES = Object.freeze({
  approval: Object.freeze(["approval", "reject", "review"]),
  reject: Object.freeze(["approval", "reject", "review"]),
  limit: Object.freeze(["limit"]),
  pricing: Object.freeze(["pricing"]),
  segmentation: Object.freeze(["segment"]),
});
const STRATEGY_POOL_OPERATION_WORKFLOWS = Object.freeze([
  "strategy_pool_compile",
  "strategy_pool_remove_entry",
  "strategy_pool_set_action",
  "strategy_pool_reorder",
]);
const STRATEGY_POOL_ENTRY_ID_RE = /^pool-entry-[0-9a-f]{32}$/;
const STRATEGY_ID_RE = /^(?:strategy-[A-Za-z0-9][A-Za-z0-9_-]*|[0-9a-f]{32})$/;
const PROJECT_CONTEXT_FIELD_PATH_RE = /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,7}$/;
const PROJECT_CONTEXT_PLATFORM_FIELD_RE = /(?:^|\.)(?:artifact_id|content_hash|dataset_id|revision|revision_id|strategy_id|target_col)$/;
const STRATEGY_POOL_APPLY_PREFIX_RE = /^[A-Za-z_][A-Za-z0-9_]{0,47}$/;
const STRATEGY_POOL_CANDIDATE_ASSET_ID_RE = /^candidate-asset-[0-9a-f]{32}$/;
const STRATEGY_POOL_ADD_SELECTION_RE = /^(?:automatic-tree-leaf-selection|interactive-tree-frontier-selection|interactive-tree-frontier-group-selection|cross-matrix-cell-selection|scorecard-cutoff-selection)-[0-9a-f]{32}$/;
const STRATEGY_POOL_ADD_SOURCE_KINDS = Object.freeze({
  univariate_asset: "candidate_asset_id",
  automatic_tree_leaf_selection: "selection_id",
  interactive_tree_frontier_selection: "selection_id",
  interactive_tree_frontier_group_selection: "selection_id",
  cross_matrix_cell_selection: "selection_id",
  scorecard_cutoff_selection: "selection_id",
  voting_candidate: "candidate_asset_id",
});
const STRATEGY_POOL_VOTING_PLACEMENTS = Object.freeze([
  "before_selected_members",
  "replace_selected_members",
]);
const _MAX_STRATEGY_POOL_ADD_SOURCES = 140;

const FIELD_LABELS = Object.freeze({
  action: "动作",
  approval_rate: "通过率",
  artifact_id: "Artifact ID",
  asset_hash: "Asset Hash",
  asset_id: "Asset ID",
  bad: "坏样本",
  bad_rate: "坏率",
  bad_count: "坏样本",
  base_odds: "基准赔率",
  base_points: "基础分值",
  base_score: "基准分",
  average_pd: "平均原始 PD",
  artifact_id: "产物 ID",
  artifact_schema_version: "产物契约版本",
  bin_id: "Bin ID",
  bin_label: "分箱标签",
  candidate_id: "Candidate ID",
  candidate_stage: "候选阶段",
  cell_id: "Cell ID",
  column_bin_id: "列分箱",
  condition: "命中条件",
  confidence: "置信度",
  content_hash: "内容 Hash",
  count: "样本数",
  created_at: "创建时间",
  cutoff_id: "Cutoff ID",
  default_action: "默认动作",
  effect: "效果",
  effect_id: "Effect ID",
  eligible: "符合约束",
  empty_cell_count: "空单元格数",
  empty_cell_share: "空单元格占比",
  evaluated: "实际评估组合数",
  execution_pd: "执行原始 PD",
  enabled: "启用",
  evidence_hash: "Evidence Hash",
  factor: "Factor",
  feature: "字段",
  fragment_id: "Fragment ID",
  good: "好样本",
  good_count: "好样本",
  iv: "IV",
  iv_contribution: "IV 贡献",
  interaction_gain_iv: "Interaction Gain IV",
  input_binding_hash: "输入绑定 Hash",
  input_binding_status: "输入绑定口径",
  ks: "KS",
  lifecycle: "生命周期",
  lower_bound: "下界",
  lower_inclusive: "包含下界",
  lower_risk: "低风险侧",
  monotonic_direction: "单调方向",
  memory_id: "记忆 ID",
  memory_type: "记忆类别",
  node_id: "节点 ID",
  lift: "Lift",
  method: "分箱方法",
  max_pairs: "最大评估组合数",
  min_nonempty_cell_count: "最小非空单元格样本数",
  observation_stage: "观测阶段",
  origin_tool: "来源 Tool",
  pool_id: "Pool ID",
  position: "顺序",
  points: "分值",
  producer_version: "生成器版本",
  provenance_hash: "来源绑定 Hash",
  pair_id: "Pair ID",
  pdo: "PDO",
  display_points: "评分卡分数",
  revision: "Revision",
  revision_id: "Revision ID",
  risk: "风险",
  row_bin_id: "行分箱",
  rule_id: "Rule ID",
  share: "占比",
  snapshot_hash: "Snapshot Hash",
  status: "状态",
  search_id: "Search ID",
  search_space: "搜索空间",
  strategy_type: "策略类型",
  source_tree_id: "操作来源树",
  source_task_id: "来源任务",
  source_memory_count: "来源记忆数",
  support_count: "支持次数",
  total: "总数",
  upper_bound: "上界",
  upper_inclusive: "包含上界",
  higher_risk: "高风险侧",
  coefficient: "系数",
  offset: "Offset",
  tree_id: "Tree ID",
  tree_result_hash: "Tree Result Hash",
  x_axis_iv: "X Axis IV",
  x_feature: "X 轴字段",
  x_method: "X 轴方法",
  y_axis_iv: "Y Axis IV",
  y_feature: "Y 轴字段",
  y_method: "Y 轴方法",
  cross_total_iv: "Cross Total IV",
  cell_count: "单元格数",
  validation_status: "验证状态",
  value: "值",
  use_reason: "使用原因",
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
    `<h5>${pointerKey === "bins" ? "候选分箱" : pointerKey === "cells" ? "矩阵单元格" : pointerKey === "frontier" ? "Frontier 规则" : "叶节点"}</h5>`,
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

function scorecardDirectionNoteHtml() {
  return [
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>方向口径</strong>",
    "<p>原始 PD 越高表示风险越高；评分卡分数越高表示更安全。</p>",
    "<p>Cutoff 只记录观测边界，不等于通过或拒绝动作；平台不会推荐或自动选择某个 Cutoff，也不会自动进入 Strategy Pool。</p>",
    "</div>",
  ].join("");
}

function scorecardRowsTableHtml(rows, columns, emptyText) {
  const visible = Array.isArray(rows) ? rows.filter(isRecord) : [];
  if (!visible.length) {
    return `<p class="candidate-lab-empty">${escapeHtml(emptyText)}</p>`;
  }
  return [
    '<div class="candidate-lab-table-scroll">',
    '<table class="candidate-lab-table"><thead><tr>',
    ...columns.map((key) => `<th>${escapeHtml(fieldLabel(key))}</th>`),
    "</tr></thead><tbody>",
    ...visible.map((row) => [
      "<tr>",
      ...columns.map((key) => `<td>${escapeHtml(readableValue(row[key]))}</td>`),
      "</tr>",
    ].join("")),
    "</tbody></table>",
    "</div>",
  ].join("");
}

function scorecardPointIntervalText(row) {
  const lowerMissing = row.lower === null || row.lower === undefined || row.lower === "";
  const upperMissing = row.upper === null || row.upper === undefined || row.upper === "";
  if (lowerMissing && upperMissing) return "-";
  const lower = lowerMissing ? "-∞" : stablePrimitiveText(row.lower);
  const upper = upperMissing ? "+∞" : stablePrimitiveText(row.upper);
  return `${lower} ～ ${upper}`;
}

function scorecardPointValue(row, key) {
  if (key === "feature" && row.feature === "__base__") {
    return "基础分（Base Points）";
  }
  if (key === "interval") return scorecardPointIntervalText(row);
  return stablePrimitiveText(row[key]);
}

function scorecardScaleHtml(rows) {
  const base = rows.find((row) => row.feature === "__base__");
  if (!base) return "";
  const scale = factsTableHtml({
    base_points: base.points,
    base_score: base.base_score,
    pdo: base.pdo,
    base_odds: base.base_odds,
    factor: base.factor,
    offset: base.offset,
  });
  return scale
    ? `<section class="candidate-lab-subsection"><h6>基础分与刻度</h6>${scale}</section>`
    : "";
}

function scorecardPointsDetailHtml(item, rows) {
  const visible = Array.isArray(rows) ? rows.filter(isRecord) : [];
  const columns = [
    ["feature", "字段"],
    ["bin_label", "分箱标签"],
    ["interval", "区间"],
    ["count", "样本数"],
    ["good_count", "好样本"],
    ["bad_count", "坏样本"],
    ["bad_rate", "坏率"],
    ["woe", "WOE"],
    ["iv_contribution", "IV 贡献"],
    ["coefficient", "系数"],
    ["monotonic_direction", "单调方向"],
    ["points", "分值"],
  ];
  const table = visible.length
    ? [
      '<div class="candidate-lab-table-scroll">',
      '<table class="candidate-lab-table"><thead><tr>',
      ...columns.map(([, label]) => `<th>${escapeHtml(label)}</th>`),
      "</tr></thead><tbody>",
      ...visible.map((row) => [
        "<tr>",
        ...columns.map(([key]) => (
          `<td>${escapeHtml(scorecardPointValue(row, key))}</td>`
        )),
        "</tr>",
      ].join("")),
      "</tbody></table>",
      "</div>",
    ].join("")
    : '<p class="candidate-lab-empty">当前受认证投影没有可见评分卡分值明细。</p>';
  const truncation = item?.truncated
    ? `<p class="candidate-lab-truncated">评分卡分值明细已截断：当前仅显示前 ${escapeHtml(visible.length)} 行。</p>`
    : "";
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-scorecard-points" data-candidate-lab-scorecard-points>',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    "<strong>评分卡分值明细</strong>",
    `<small>${escapeHtml(visible.length)} 行受认证明细</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">展开分值表</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>分值方向</strong>",
    "<p>评分卡分值越高，代表风险越低（越安全）。</p>",
    "</div>",
    scorecardScaleHtml(visible),
    table,
    truncation,
    "</div>",
    "</details>",
  ].join("");
}

function scorecardBandDetailHtml(item) {
  const detail = isRecord(item?.detail) ? item.detail : {};
  const pointers = isRecord(item?.pointers) ? item.pointers : {};
  const title = nonEmptyText(detail.asset_id)
    || nonEmptyText(item?.candidate_id)
    || "评分卡分档";
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-scorecard-card">',
    '<summary>',
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    "<small>受认证评分卡分档证据</small>",
    "</span>",
    '<span class="candidate-lab-card-state">查看分档与 Cutoff</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml(item),
    lifecycleHtml(item.lifecycle),
    scorecardDirectionNoteHtml(),
    '<section class="candidate-lab-subsection"><h5>样本与性能</h5>',
    factsTableHtml({
      asset_id: detail.asset_id,
      sample: detail.sample,
      performance: detail.performance,
    }),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>分档证据</h5>',
    scorecardRowsTableHtml(
      pointers.bands,
      [
        "ordinal",
        "bin_id",
        "lower_bound",
        "upper_bound",
        "count",
        "share",
        "labeled_count",
        "bad_count",
        "bad_rate",
        "average_pd",
      ],
      "当前受认证投影没有可见分档。",
    ),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>Cutoff 观测</h5>',
    scorecardRowsTableHtml(
      pointers.cutoffs,
      [
        "ordinal",
        "cutoff_id",
        "execution_pd",
        "display_points",
        "lower_risk",
        "higher_risk",
      ],
      "当前受认证投影没有可见 Cutoff。",
    ),
    "</section>",
    scorecardPointsDetailHtml(item, pointers.scorecard_points),
    riskHtml(item.risks),
    "</div>",
    "</details>",
  ].join("");
}

function scorecardSelectionDetailHtml(item) {
  const detail = isRecord(item?.detail) ? item.detail : {};
  const effect = isRecord(detail.effect) ? [detail.effect] : [];
  const title = nonEmptyText(detail.selection_id)
    || nonEmptyText(item?.candidate_id)
    || "Cutoff 选择记录";
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-scorecard-card">',
    '<summary>',
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    "<small>人工 Cutoff 指针</small>",
    "</span>",
    '<span class="candidate-lab-card-state">查看选择证据</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml(item),
    lifecycleHtml(item.lifecycle),
    scorecardDirectionNoteHtml(),
    '<section class="candidate-lab-subsection"><h5>选择记录</h5>',
    factsTableHtml({
      selection_id: detail.selection_id,
      asset_id: detail.asset_id,
      cutoff_id: detail.cutoff_id,
      reason: detail.reason,
    }),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>Cutoff 观测</h5>',
    scorecardRowsTableHtml(
      effect,
      [
        "ordinal",
        "cutoff_id",
        "execution_pd",
        "display_points",
        "lower_risk",
        "higher_risk",
      ],
      "该选择记录没有可见的 Cutoff 观测。",
    ),
    "</section>",
    riskHtml(item.risks),
    "</div>",
    "</details>",
  ].join("");
}

function votingSearchDetailHtml(item) {
  const combinations = Array.isArray(item?.combinations)
    ? item.combinations.filter(isRecord)
    : [];
  const title = nonEmptyText(item?.search_id) || "Voting 组合搜索";
  const summary = {
    strategy_type: item?.strategy_type,
    pool_revision: item?.pool_revision,
    member_count: item?.member_count,
    n: item?.n,
    objective: item?.objective,
    constraints: item?.constraints,
    include_rule_ids: item?.include_rule_ids,
    exclude_rule_ids: item?.exclude_rule_ids,
    max_combinations: item?.max_combinations,
    search_space: item?.search_space,
    evaluated: item?.evaluated,
    eligible: item?.eligible,
    truncated: item?.truncated,
  };
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-voting-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    "<small>development search evidence · 未构建/未入池</small>",
    "</span>",
    '<span class="candidate-lab-card-state">查看组合证据</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml({ artifact: item?.artifact }),
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>候选边界</strong>",
    "<p>这里按确定性目标和约束展示已评估组合，不表达最佳、冠军或平台选择；搜索不会自动构建、选择、入池或部署。</p>",
    "</div>",
    '<section class="candidate-lab-subsection"><h5>搜索参数与计数</h5>',
    factsTableHtml(summary),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>精确组合指针</h5>',
    scorecardRowsTableHtml(
      combinations,
      ["combo_id", "members", "eligible", "failures", "metrics"],
      "当前受认证搜索没有可见组合。",
    ),
    "</section>",
    item?.truncated
      ? '<p class="candidate-lab-truncated">搜索空间或可见组合已按服务端预算截断；页面不会推断窗口外结果。</p>'
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function crossSearchDetailHtml(item) {
  const features = Array.isArray(item?.features)
    ? item.features.filter(isRecord)
    : [];
  const pairs = Array.isArray(item?.pairs)
    ? item.pairs.filter(isRecord)
    : [];
  const title = nonEmptyText(item?.search_id) || "Cross 自动搜索";
  const truncated = item?.truncated === true;
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-cross-search-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    `<small>${escapeHtml(stablePrimitiveText(item?.evaluated))} / ${escapeHtml(stablePrimitiveText(item?.search_space))} 个组合已评估 · 未构建</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看组合证据</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml({ artifact: item?.artifact }),
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>候选边界</strong>",
    "<p>Pair 的 rank、eligible 与指标只描述确定性搜索结果；页面不会自动构建、入池、采纳或部署，必须由用户明确选择完整 Pair。</p>",
    "</div>",
    '<section class="candidate-lab-subsection"><h5>搜索参数与预算</h5>',
    factsTableHtml({
      max_pairs: item?.max_pairs,
      search_space: item?.search_space,
      evaluated: item?.evaluated,
      eligible: item?.eligible,
      truncated,
    }),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>参与搜索的单变量字段</h5>',
    scorecardRowsTableHtml(
      features,
      ["feature", "method", "axis_iv", "bin_count"],
      "当前受认证搜索没有可见字段配置。",
    ),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>Top Pairs（仅展示，不代替选择）</h5>',
    scorecardRowsTableHtml(
      pairs,
      [
        "rank",
        "pair_id",
        "x_feature",
        "x_method",
        "y_feature",
        "y_method",
        "x_axis_iv",
        "y_axis_iv",
        "cross_total_iv",
        "interaction_gain_iv",
        "cell_count",
        "empty_cell_count",
        "empty_cell_share",
        "min_nonempty_cell_count",
        "eligible",
      ],
      "当前受认证搜索没有可见 Pair。",
    ),
    "</section>",
    truncated
      ? [
        '<div class="candidate-lab-risk-group" data-tone="warn">',
        "<strong>预算截断</strong>",
        `<p>搜索空间 ${escapeHtml(stablePrimitiveText(item?.search_space))} 个组合，本次预算最多评估 ${escapeHtml(stablePrimitiveText(item?.max_pairs))} 个，实际评估 ${escapeHtml(stablePrimitiveText(item?.evaluated))} 个；页面不会推断未评估组合。</p>`,
        "</div>",
      ].join("")
      : '<p class="candidate-lab-field-help">本次搜索未触发预算截断；可见 Pair 仍需用户逐项明确选择。</p>',
    "</div>",
    "</details>",
  ].join("");
}

function crossRuleSearchDetailHtml(item) {
  const features = Array.isArray(item?.features)
    ? item.features.filter(isRecord)
    : [];
  const rules = Array.isArray(item?.rules)
    ? item.rules.filter(isRecord)
    : [];
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-cross-search-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(nonEmptyText(item?.search_id) || "Cross 阈值规则搜索")}</strong>`,
    `<small>${escapeHtml(stablePrimitiveText(item?.dimension))}D · ${escapeHtml(stablePrimitiveText(item?.evaluated))} / ${escapeHtml(stablePrimitiveText(item?.search_space))} 条试验 · 未选择</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看规则证据</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml({ artifact: item?.artifact }),
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>人工选择边界</strong>",
    "<p>rank、eligible 和约束失败只描述确定性证据。页面不会把第一名当成冠军，也不会自动构建或入池；必须明确选择完整 rule_id。</p>",
    "</div>",
    '<section class="candidate-lab-subsection"><h5>搜索参数与预算</h5>',
    factsTableHtml({
      dimension: item?.dimension,
      constraints: item?.constraints,
      max_trials: item?.max_trials,
      search_space: item?.search_space,
      evaluated: item?.evaluated,
      eligible: item?.eligible,
      truncated: item?.truncated,
    }),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>字段阈值来源</h5>',
    scorecardRowsTableHtml(
      features,
      ["feature", "method", "risk_direction", "thresholds", "excluded_values", "missing_count", "missing_bad"],
      "当前搜索没有可见字段阈值配置。",
    ),
    "</section>",
    '<section class="candidate-lab-subsection"><h5>规则指针（仅展示，不代替选择）</h5>',
    scorecardRowsTableHtml(
      rules,
      ["rank", "rule_id", "conditions", "metrics", "eligible", "constraint_failures"],
      "当前搜索没有可见规则。",
    ),
    "</section>",
    item?.rules_truncated
      ? '<p class="candidate-lab-truncated">规则展示已按服务端预算截断；请下载完整搜索证据查看其余已评估规则。</p>'
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function crossRuleCandidateDetailHtml(item) {
  const detail = isRecord(item?.detail) ? item.detail : {};
  return [
    '<details class="candidate-lab-evidence-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(nonEmptyText(detail.asset_id) || "Cross 阈值规则候选")}</strong>`,
    `<small>${escapeHtml(stablePrimitiveText(detail.dimension))}D · development / ${escapeHtml(stablePrimitiveText(detail.validation_status))}</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看候选</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml({ artifact: item?.artifact }),
    '<section class="candidate-lab-subsection"><h5>精确来源与效果</h5>',
    factsTableHtml(detail),
    "</section>",
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>生命周期边界</strong>",
    "<p>候选已物化但尚未独立验证、入池、应用、采纳或部署；后续动作仍需单独确认。</p>",
    "</div>",
    riskHtml(item?.risks),
    "</div>",
    "</details>",
  ].join("");
}

function interactiveTreeEligiblePointers(item) {
  const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
  const nodes = new Map(
    (Array.isArray(item?.pointers?.nodes) ? item.pointers.nodes : [])
      .filter(isRecord)
      .map((node) => [nonEmptyText(node.node_id), node]),
  );
  const pointers = Array.isArray(item?.pointers?.eligible_prunes)
    ? item.pointers.eligible_prunes.filter(isRecord)
    : [];
  const seen = new Set();
  return pointers.filter((pointer) => {
    const pointerSource = nonEmptyText(pointer.source_tree_id);
    const nodeId = nonEmptyText(pointer.node_id);
    const node = nodes.get(nodeId);
    const key = `${pointerSource}\u001f${nodeId}`;
    if (
      pointerSource !== sourceTreeId
      || !INTERACTIVE_TREE_SOURCE_ID_RE.test(pointerSource)
      || !INTERACTIVE_TREE_NODE_ID_RE.test(nodeId)
      || pointer.operation !== "prune_subtree"
      || node?.kind !== "split"
      || node?.is_visible !== true
      || node?.is_frontier === true
      || node?.can_prune !== true
      || seen.has(key)
    ) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function interactiveTreeThresholdEligiblePointers(item) {
  const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
  const nodes = new Map(
    (Array.isArray(item?.pointers?.nodes) ? item.pointers.nodes : [])
      .filter(isRecord)
      .map((node) => [nonEmptyText(node.node_id), node]),
  );
  const pointers = Array.isArray(
    item?.pointers?.eligible_threshold_adjustments,
  )
    ? item.pointers.eligible_threshold_adjustments.filter(isRecord)
    : [];
  const seen = new Set();
  return pointers.filter((pointer) => {
    const pointerSource = nonEmptyText(pointer.source_tree_id);
    const nodeId = nonEmptyText(pointer.node_id);
    const feature = nonEmptyText(pointer.feature);
    const currentThreshold = Number(pointer.current_threshold);
    const node = nodes.get(nodeId);
    const nodeThreshold = Number(node?.threshold);
    const key = `${pointerSource}\u001f${nodeId}`;
    if (
      pointerSource !== sourceTreeId
      || !INTERACTIVE_TREE_SOURCE_ID_RE.test(pointerSource)
      || !INTERACTIVE_TREE_NODE_ID_RE.test(nodeId)
      || pointer.operation !== "adjust_split_threshold"
      || !feature
      || typeof pointer.current_threshold !== "number"
      || !Number.isFinite(currentThreshold)
      || node?.kind !== "split"
      || node?.is_visible !== true
      || node?.is_frontier === true
      || node?.can_prune !== true
      || nonEmptyText(node?.feature) !== feature
      || typeof node?.threshold !== "number"
      || !Number.isFinite(nodeThreshold)
      || nodeThreshold !== currentThreshold
      || seen.has(key)
    ) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function interactiveTreeFeatureEligiblePointers(item) {
  const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
  const featureUniverse = new Set(
    (Array.isArray(item?.pointers?.feature_universe)
      ? item.pointers.feature_universe
      : [])
      .map(nonEmptyText)
      .filter(Boolean),
  );
  const nodes = new Map(
    (Array.isArray(item?.pointers?.nodes) ? item.pointers.nodes : [])
      .filter(isRecord)
      .map((node) => [nonEmptyText(node.node_id), node]),
  );
  const pointers = Array.isArray(
    item?.pointers?.eligible_feature_replacements,
  )
    ? item.pointers.eligible_feature_replacements.filter(isRecord)
    : [];
  const seen = new Set();
  return pointers.filter((pointer) => {
    const pointerSource = nonEmptyText(pointer.source_tree_id);
    const nodeId = nonEmptyText(pointer.node_id);
    const currentFeature = nonEmptyText(pointer.current_feature);
    const currentThreshold = Number(pointer.current_threshold);
    const node = nodes.get(nodeId);
    const key = `${pointerSource}\u001f${nodeId}`;
    if (
      pointerSource !== sourceTreeId
      || !INTERACTIVE_TREE_SOURCE_ID_RE.test(pointerSource)
      || !INTERACTIVE_TREE_NODE_ID_RE.test(nodeId)
      || pointer.operation !== "replace_split_feature"
      || !featureUniverse.has(currentFeature)
      || featureUniverse.size < 2
      || typeof pointer.current_threshold !== "number"
      || !Number.isFinite(currentThreshold)
      || node?.kind !== "split"
      || node?.is_visible !== true
      || node?.is_frontier === true
      || node?.can_prune !== true
      || nonEmptyText(node?.feature) !== currentFeature
      || Number(node?.threshold) !== currentThreshold
      || seen.has(key)
    ) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function interactiveTreeFrontierEligiblePointers(item) {
  if (item?.kind !== "interactive_tree_revision") return [];
  const revisionId = nonEmptyText(item?.detail?.revision_id);
  if (!INTERACTIVE_TREE_REVISION_ID_RE.test(revisionId)) return [];
  const nodes = new Map(
    (Array.isArray(item?.pointers?.nodes) ? item.pointers.nodes : [])
      .filter(isRecord)
      .map((node) => [nonEmptyText(node.node_id), node]),
  );
  const frontierIds = new Set(
    (Array.isArray(item?.pointers?.frontier_node_ids)
      ? item.pointers.frontier_node_ids
      : [])
      .map(nonEmptyText)
      .filter((nodeId) => (
        INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE.test(nodeId)
      )),
  );
  const pointers = Array.isArray(item?.pointers?.frontier)
    ? item.pointers.frontier.filter(isRecord)
    : [];
  const seen = new Set();
  return pointers.filter((pointer) => {
    const sourceNodeId = nonEmptyText(pointer.source_node_id);
    const node = nodes.get(sourceNodeId);
    if (
      !INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE.test(sourceNodeId)
      || !frontierIds.has(sourceNodeId)
      || node?.is_visible !== true
      || node?.is_frontier !== true
      || seen.has(sourceNodeId)
    ) {
      return false;
    }
    seen.add(sourceNodeId);
    return true;
  });
}

function interactiveTreeNodesHtml(item) {
  const nodes = Array.isArray(item?.pointers?.nodes)
    ? item.pointers.nodes.filter(isRecord)
    : [];
  if (!nodes.length) {
    return '<p class="candidate-lab-empty">当前受认证树没有可见拓扑节点。</p>';
  }
  const eligible = new Set(
    interactiveTreeEligiblePointers(item).map(
      (pointer) => `${pointer.source_tree_id}\u001f${pointer.node_id}`,
    ),
  );
  const thresholdAdjustments = new Map(
    interactiveTreeThresholdEligiblePointers(item).map(
      (pointer) => [
        `${pointer.source_tree_id}\u001f${pointer.node_id}`,
        pointer,
      ],
    ),
  );
  const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
  const revisionId = item?.kind === "interactive_tree_revision"
    ? nonEmptyText(item?.detail?.revision_id)
    : "";
  const materializable = new Set(
    interactiveTreeFrontierEligiblePointers(item).map(
      (pointer) => pointer.source_node_id,
    ),
  );
  return [
    '<div class="candidate-lab-table-scroll candidate-lab-tree-scroll">',
    '<table class="candidate-lab-table candidate-lab-tree-table"><thead><tr>',
    "<th>深度</th><th>节点</th><th>分裂 / 条件</th><th>样本效果</th><th>状态</th><th>操作</th>",
    "</tr></thead><tbody>",
    ...nodes.map((node) => {
      const nodeId = nonEmptyText(node.node_id);
      const key = `${sourceTreeId}\u001f${nodeId}`;
      const split = node.kind === "split"
        ? `${stablePrimitiveText(node.feature)} ≤ ${stablePrimitiveText(node.threshold)}；缺失→${stablePrimitiveText(node.missing_child)}`
        : readableValue(node.condition);
      const state = [
        node.is_visible === true ? "可见" : "已隐藏",
        node.is_frontier === true ? "frontier" : "",
      ].filter(Boolean).join(" · ");
      const actions = [];
      if (node.can_prune === true && eligible.has(key)) {
        actions.push([
          '<button type="button" class="button compact secondary candidate-lab-tree-prune"',
          ' data-candidate-lab-interactive-tree-prune="1"',
          ` data-source-tree-id="${escapeHtml(sourceTreeId)}"`,
          ` data-node-id="${escapeHtml(nodeId)}">剪枝到此节点</button>`,
        ].join(""));
      }
      const thresholdAdjustment = thresholdAdjustments.get(key);
      if (thresholdAdjustment) {
        actions.push([
          '<button type="button" class="button compact secondary candidate-lab-tree-threshold"',
          ' data-candidate-lab-interactive-tree-threshold="1"',
          ` data-source-tree-id="${escapeHtml(sourceTreeId)}"`,
          ` data-node-id="${escapeHtml(nodeId)}"`,
          ` data-feature="${escapeHtml(thresholdAdjustment.feature)}"`,
          ` data-current-threshold="${escapeHtml(stablePrimitiveText(
            thresholdAdjustment.current_threshold,
          ))}">调整 ${escapeHtml(thresholdAdjustment.feature)} 阈值</button>`,
        ].join(""));
      }
      if (
        INTERACTIVE_TREE_REVISION_ID_RE.test(revisionId)
        && materializable.has(nodeId)
      ) {
        actions.push([
          '<button type="button" class="button compact secondary candidate-lab-tree-frontier-materialize"',
          ' data-candidate-lab-interactive-tree-frontier-materialize="1"',
          ` data-revision-id="${escapeHtml(revisionId)}"`,
          ` data-source-node-id="${escapeHtml(nodeId)}">物化前沿节点</button>`,
        ].join(""));
      }
      const action = actions.join(" ") || "—";
      return [
        "<tr>",
        `<td>${escapeHtml(stablePrimitiveText(node.depth))}</td>`,
        `<td><code>${escapeHtml(nodeId)}</code><small>${escapeHtml(stablePrimitiveText(node.kind))}</small></td>`,
        `<td>${escapeHtml(split)}</td>`,
        `<td>${escapeHtml(readableValue(node.metrics))}</td>`,
        `<td>${escapeHtml(state || "—")}</td>`,
        `<td>${action}</td>`,
        "</tr>",
      ].join("");
    }),
    "</tbody></table>",
    "</div>",
  ].join("");
}

function interactiveTreeDetailHtml(item) {
  const detail = isRecord(item?.detail) ? item.detail : {};
  const isRevision = item?.kind === "interactive_tree_revision";
  const identity = isRevision
    ? nonEmptyText(detail.revision_id)
    : nonEmptyText(detail.asset_id);
  const title = identity || (isRevision ? "交互式树修订" : "自动规则树");
  const history = Array.isArray(item?.history)
    ? item.history.filter(isRecord)
    : [];
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-tree-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(title)}</strong>`,
    `<small>${isRevision ? "immutable revision branch" : "verified automatic topology"}</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看完整拓扑</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml(item),
    lifecycleHtml(item.lifecycle),
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>不可变分支</strong>",
    "<p>每次剪枝或阈值调整都会创建新 revision；不会写回来源树、物化 frontier、入池、采纳或部署，页面也不会替你挑选节点。</p>",
    "</div>",
    '<section class="candidate-lab-subsection"><h5>树与修订身份</h5>',
    factsTableHtml({
      source_tree_id: detail.source_tree_id,
      derived_from_source_tree_id: detail.derived_from_source_tree_id,
      parent_revision_id: detail.parent_revision_id,
      base_asset_id: detail.base_asset_id || detail.asset_id,
      asset_hash: detail.asset_hash,
      tree_id: detail.tree_id,
      tree_result_hash: detail.tree_result_hash,
      semantic_tree_id: detail.semantic_tree_id,
      tree_hash: detail.tree_hash,
      edit: detail.edit,
      summary: detail.summary,
    }),
    "</section>",
    isRevision
      ? [
        '<section class="candidate-lab-subsection"><h5>当前分支历史（近到远）</h5>',
        scorecardRowsTableHtml(
          history,
          ["revision_id", "parent_revision_id", "edit", "semantic_tree_id"],
          "当前 revision 没有可见历史。",
        ),
        "</section>",
      ].join("")
      : "",
    pointerTableHtml(item, isRevision ? "frontier" : "leaves"),
    '<section class="candidate-lab-subsection"><h5>完整节点拓扑</h5>',
    interactiveTreeNodesHtml(item),
    "</section>",
    '<p class="candidate-lab-field-help">操作按钮仅来自服务端重新验真的 eligible_prunes；提交后 Tool 会再次按任务、父链和样本回放校验。</p>',
    riskHtml(item.risks),
    "</div>",
    "</details>",
  ].join("");
}

function interactiveTreeSplitSearchDetailHtml(item) {
  const searchId = nonEmptyText(item?.search_id);
  const sourceTreeId = nonEmptyText(item?.source_tree_id);
  const nodeId = nonEmptyText(item?.node_id);
  const sourceNode = isRecord(item?.source_node) ? item.source_node : {};
  const candidates = Array.isArray(item?.candidates)
    ? item.candidates.filter(isRecord)
    : [];
  const canPrefill = (
    sourceNode.kind === "split"
    && sourceNode.is_visible === true
    && sourceNode.is_frontier !== true
    && sourceNode.can_prune === true
  );
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-tree-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(searchId || "树节点候选搜索")}</strong>`,
    `<small>${escapeHtml(sourceTreeId)} · ${escapeHtml(nodeId)}</small>`,
    "</span>",
    `<span class="candidate-lab-card-state">${escapeHtml(stablePrimitiveText(
      candidates.length,
    ))} 个候选</span>`,
    "</summary>",
    '<div class="candidate-lab-card-body">',
    evidenceIdentityHtml(item),
    '<div class="candidate-lab-boundary-note" data-tone="info">',
    "<strong>排名仅用于浏览</strong>",
    "<p>搜索没有选择胜者，也没有修改树。按钮只会回填精确候选与后续控制项，仍需人工确认提交。</p>",
    "</div>",
    factsTableHtml({
      search_id: searchId,
      source_tree_id: sourceTreeId,
      node_id: nodeId,
      node_kind: item?.node_kind,
      mode: item?.mode,
      features: item?.features,
      population: item?.population,
      budget: item?.budget,
      claims: item?.claims,
    }),
    '<div class="candidate-lab-table-scroll">',
    '<table class="candidate-lab-table"><thead><tr>',
    "<th>排名</th><th>字段 / 阈值</th><th>左侧</th><th>右侧</th><th>增益 / 方向</th><th>资格</th><th>操作</th>",
    "</tr></thead><tbody>",
    ...candidates.map((candidate) => {
      const candidateId = nonEmptyText(candidate.candidate_id);
      const feature = nonEmptyText(candidate.feature);
      const threshold = Number(candidate.threshold);
      const eligible = candidate.eligible === true;
      const changesSplit = (
        feature !== nonEmptyText(sourceNode.feature)
        || threshold !== Number(sourceNode.threshold)
      );
      const revisionAction = (
        eligible
        && canPrefill
        && changesSplit
        && INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE.test(candidateId)
        && Number.isFinite(threshold)
      )
        ? [
          '<button type="button" class="button compact secondary"',
          ' data-candidate-lab-interactive-tree-split-candidate="1"',
          ` data-search-id="${escapeHtml(searchId)}"`,
          ` data-candidate-id="${escapeHtml(candidateId)}"`,
          ` data-source-tree-id="${escapeHtml(sourceTreeId)}"`,
          ` data-node-id="${escapeHtml(nodeId)}"`,
          ` data-feature="${escapeHtml(feature)}"`,
          ` data-threshold="${escapeHtml(stablePrimitiveText(threshold))}">`,
          "带入树修订</button>",
        ].join("")
        : "";
      const continuationAction = (
        eligible
        && sourceNode.is_visible === true
        && sourceNode.is_frontier === true
        && INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE.test(searchId)
        && INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE.test(candidateId)
      )
        ? [
          '<button type="button" class="button compact secondary"',
          ' data-candidate-lab-interactive-tree-auto-continuation="1"',
          ` data-search-id="${escapeHtml(searchId)}"`,
          ` data-candidate-id="${escapeHtml(candidateId)}">`,
          "带入受控续建</button>",
        ].join("")
        : "";
      const action = [revisionAction, continuationAction]
        .filter(Boolean)
        .join(" ") || "—";
      return [
        "<tr>",
        `<td>${escapeHtml(stablePrimitiveText(candidate.rank))}</td>`,
        `<td><strong>${escapeHtml(feature)}</strong><small>≤ ${escapeHtml(
          stablePrimitiveText(threshold),
        )} · 缺失→${escapeHtml(stablePrimitiveText(candidate.missing_child))}</small></td>`,
        `<td>${escapeHtml(readableValue(candidate.left))}</td>`,
        `<td>${escapeHtml(readableValue(candidate.right))}</td>`,
        `<td>${escapeHtml(stablePrimitiveText(candidate.gain))}<small>${escapeHtml(
          readableValue(candidate.direction),
        )}</small></td>`,
        `<td>${eligible ? "可用" : escapeHtml(readableValue(candidate.failures))}</td>`,
        `<td>${action}</td>`,
        "</tr>",
      ].join("");
    }),
    "</tbody></table>",
    "</div>",
    item?.truncated === true
      ? '<p class="candidate-lab-truncated">当前结果或阈值空间已按明确预算截断。</p>'
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function candidateItemHtml(item, definition) {
  if (
    definition.key === "automatic_tree"
    || definition.key === "interactive_tree_revision"
  ) {
    return interactiveTreeDetailHtml(item);
  }
  if (definition.key === "scorecard_band") {
    return scorecardBandDetailHtml(item);
  }
  if (definition.key === "interactive_tree_split_search") {
    return interactiveTreeSplitSearchDetailHtml(item);
  }
  if (definition.key === "scorecard_cutoff_selection") {
    return scorecardSelectionDetailHtml(item);
  }
  if (definition.key === "voting_search") {
    return votingSearchDetailHtml(item);
  }
  if (definition.key === "cross_search") {
    return crossSearchDetailHtml(item);
  }
  if (definition.key === "cross_rule_search") {
    return crossRuleSearchDetailHtml(item);
  }
  if (definition.key === "cross_rule_candidate") {
    return crossRuleCandidateDetailHtml(item);
  }
  return candidateDetailHtml(item, definition.pointerKey);
}

function candidateCollectionHtml(candidates, definition) {
  const collection = isRecord(candidates?.[definition.key])
    ? candidates[definition.key]
    : {};
  const items = collectionItems(collection);
  const total = collectionTotal(collection);
  const countText = total === null ? "" : `${total} 个`;
  const list = items.length
    ? items.map((item) => candidateItemHtml(item, definition)).join("")
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

const WORKFLOW_STAGE_STATUS_LABELS = Object.freeze({
  complete: "已完成",
  stale: "需刷新",
  missing: "待补充",
});

const REPORT_FIELD_AVAILABILITY_LABELS = Object.freeze({
  unavailable: "暂未提供",
  not_applicable: "不适用",
  not_matured: "样本未成熟",
});

function workflowStageSpineHtml(stages) {
  const rows = Array.isArray(stages) ? stages.filter(isRecord).slice(0, 7) : [];
  if (!rows.length) {
    return '<p class="candidate-lab-empty">七阶段状态尚未由平台返回。</p>';
  }
  return [
    '<ol class="candidate-lab-workflow-stages" aria-label="策略开发七阶段">',
    ...rows.map((stage, index) => {
      const status = ["complete", "stale", "missing"].includes(stage.status)
        ? stage.status
        : "missing";
      return [
        `<li class="candidate-lab-workflow-stage" data-workflow-stage="${escapeHtml(nonEmptyText(stage.id) || String(index + 1))}" data-status="${escapeHtml(status)}">`,
        `<span>${index + 1}</span>`,
        `<strong>${escapeHtml(nonEmptyText(stage.label) || `阶段 ${index + 1}`)}</strong>`,
        `<small>${escapeHtml(WORKFLOW_STAGE_STATUS_LABELS[status])}</small>`,
        "</li>",
      ].join("");
    }),
    "</ol>",
  ].join("");
}

function reportFieldReadableValue(field) {
  if (!isRecord(field)) return "暂未提供";
  if (field.availability === "present") return readableValue(field.value);
  return REPORT_FIELD_AVAILABILITY_LABELS[field.availability]
    || nonEmptyText(field.availability)
    || "暂未提供";
}

function projectContextHistoryHtml(histories) {
  const rows = Array.isArray(histories) ? histories.filter(isRecord) : [];
  if (!rows.length) {
    return '<p class="candidate-lab-empty">当前没有可展示的历史策略版本；如已明确暂缺，报告会保留为空。</p>';
  }
  return [
    '<div class="candidate-lab-result-list">',
    ...rows.map((history) => [
      '<article class="candidate-lab-evidence-card candidate-lab-project-history">',
      `<strong>${history.version === null || history.version === undefined ? "外部历史材料" : `版本 ${escapeHtml(history.version)}`}</strong>`,
      factsTableHtml({
        availability: history.availability,
        effective_period: reportFieldReadableValue(history.effective_period),
        asset_status: reportFieldReadableValue(history.asset_status),
        scope: reportFieldReadableValue(history.scope),
        traffic_allocation: reportFieldReadableValue(history.traffic_allocation),
        effect_stages: history.effect_stages,
        external_source_count: history.external_source_count,
      }),
      "</article>",
    ].join("")),
    "</div>",
  ].join("");
}

function projectContextMissingHtml(records) {
  const pending = Array.isArray(records)
    ? records.filter((item) => isRecord(item) && item.status === "pending")
    : [];
  if (!pending.length) return "";
  return [
    '<section class="candidate-lab-subsection candidate-lab-missing-information">',
    "<h5>还可补充的信息</h5>",
    "<ul>",
    ...pending.map((item) => (
      `<li><strong>${escapeHtml(fieldLabel(item.field_path))}</strong><span>${escapeHtml(item.question)}</span></li>`
    )),
    "</ul>",
    "</section>",
  ].join("");
}

function projectContextWorkflowHtml(project) {
  if (!isRecord(project)) {
    return [
      '<section class="candidate-lab-subsection candidate-lab-project-context" data-status="missing">',
      "<h5>项目现状与历史版本</h5>",
      '<p class="candidate-lab-empty">尚未固化当前项目状况和历史材料。可以直接告诉 Agent 已知信息；暂时没有的可明确说明暂缺。</p>',
      "</section>",
    ].join("");
  }
  const current = isRecord(project.current) ? project.current : {};
  const statusFields = isRecord(current.status_fields)
    ? current.status_fields
    : {};
  const downloadUrl = safeDownloadUrl(project.artifact?.download_url);
  return [
    '<section class="candidate-lab-subsection candidate-lab-project-context" data-status="complete">',
    "<header><div><h5>项目现状与历史版本</h5>",
    `<p>上下文 revision ${escapeHtml(stablePrimitiveText(project.revision))} · ${escapeHtml(stablePrimitiveText(project.as_of))}</p></div>`,
    downloadUrl
      ? `<a class="button compact secondary" href="${escapeHtml(downloadUrl)}" download>下载项目上下文</a>`
      : "",
    "</header>",
    factsTableHtml({
      scope: reportFieldReadableValue(project.scope),
      volume: reportFieldReadableValue(statusFields.volume),
      approval: reportFieldReadableValue(statusFields.approval),
      risk: reportFieldReadableValue(statusFields.risk),
      economics: reportFieldReadableValue(statusFields.economics),
      maturity: reportFieldReadableValue(current.maturity_summary),
      history_resolution: project.history_resolution,
    }),
    '<section class="candidate-lab-subsection"><h5>历史策略版本</h5>',
    projectContextHistoryHtml(project.historical_versions),
    "</section>",
    projectContextMissingHtml(project.missing_information),
    "</section>",
  ].join("");
}

function samplePopulationHtml(role, population) {
  const title = role === "approval" ? "审批人群" : "风险表现人群";
  if (!isRecord(population)) {
    return [
      `<article class="candidate-lab-evidence-card candidate-lab-sample-population" data-population-role="${escapeHtml(role)}" data-status="missing">`,
      `<strong>${title}</strong>`,
      "<p>尚未定义。</p>",
      "</article>",
    ].join("");
  }
  const maturity = isRecord(population.maturity) ? population.maturity : {};
  return [
    `<article class="candidate-lab-evidence-card candidate-lab-sample-population" data-population-role="${escapeHtml(role)}">`,
    `<strong>${title}</strong>`,
    factsTableHtml({
      total: population.total_count,
      partitions: population.partitions,
      maturity_status: maturity.status,
      performance_window_days: maturity.performance_window_days,
      cutoff_date: maturity.cutoff_date,
      eligible_count: maturity.eligible_count,
      labeled_count: maturity.labeled_count,
      reason: maturity.reason,
    }),
    "</article>",
  ].join("");
}

function sampleDesignWorkflowHtml(sample) {
  if (!isRecord(sample)) {
    return [
      '<section class="candidate-lab-subsection candidate-lab-sample-design" data-status="missing">',
      "<h5>双人群样本设计</h5>",
      '<p class="candidate-lab-empty">尚无当前受认证 SampleDesign V2；请先让 Agent 明确审批人群、风险表现人群、分区与成熟度。</p>',
      "</section>",
    ].join("");
  }
  const artifactUrl = safeDownloadUrl(sample.artifact?.download_url);
  return [
    `<section class="candidate-lab-subsection candidate-lab-sample-design" data-status="${escapeHtml(sample.freshness === "stale" ? "stale" : "complete")}">`,
    "<header><h5>双人群样本设计</h5>",
    artifactUrl
      ? `<a class="button compact secondary" href="${escapeHtml(artifactUrl)}" download>下载样本设计摘要</a>`
      : "",
    "</header>",
    factsTableHtml({
      source_mode: sample.source_mode,
      relationship: sample.relationship,
      analysis_universe_count: sample.analysis_universe_count,
      target: sample.target,
      relationship_counts: sample.relationship_counts,
      diagnostic_status: sample.diagnostics?.overall_status,
    }),
    '<div class="candidate-lab-result-list candidate-lab-dual-populations">',
    samplePopulationHtml("approval", sample.populations?.approval),
    samplePopulationHtml("risk", sample.populations?.risk),
    "</div>",
    "</section>",
  ].join("");
}

function workflowEvidenceItemHtml(label, item) {
  if (!isRecord(item)) {
    return [
      '<article class="candidate-lab-evidence-card" data-status="missing">',
      `<strong>${escapeHtml(label)}</strong>`,
      "<small>待生成</small>",
      "</article>",
    ].join("");
  }
  const downloadUrl = safeDownloadUrl(item.artifact?.download_url);
  const freshness = item.freshness === "stale" ? "stale" : "complete";
  return [
    `<article class="candidate-lab-evidence-card" data-status="${freshness}">`,
    `<strong>${escapeHtml(label)}</strong>`,
    `<small>${freshness === "stale" ? "基于旧 Pool revision，需刷新" : "与当前 Pool 一致"}</small>`,
    factsTableHtml({
      strategy_type: item.strategy_type,
      pool_revision: item.pool_revision,
      partitions: item.partitions || item.comparison_partitions || item.partition,
      population_count: item.population_count,
      labeled_count: item.labeled_count,
      lifecycle: item.lifecycle,
    }),
    downloadUrl
      ? `<a class="button compact secondary" href="${escapeHtml(downloadUrl)}" download>下载证据</a>`
      : "",
    "</article>",
  ].join("");
}

function workflowEvidenceHtml(latestEvidence) {
  const evidence = isRecord(latestEvidence) ? latestEvidence : {};
  const validations = isRecord(evidence.pool_validation)
    ? evidence.pool_validation
    : {};
  return [
    '<section class="candidate-lab-subsection candidate-lab-workflow-evidence">',
    "<h5>最新效果与稳定性证据</h5>",
    '<div class="candidate-lab-result-list">',
    workflowEvidenceItemHtml("Pool Stability", evidence.pool_stability),
    workflowEvidenceItemHtml("Pool Impact", evidence.pool_impact),
    workflowEvidenceItemHtml("ImpactCube", evidence.impact_cube),
    workflowEvidenceItemHtml("Validation", validations.validation),
    workflowEvidenceItemHtml("OOT", validations.oot),
    "</div>",
    "</section>",
  ].join("");
}

function workflowReportHtml(report) {
  if (!isRecord(report)) {
    return [
      '<section class="candidate-lab-subsection candidate-lab-workflow-report" data-status="missing">',
      "<h5>策略迭代评审报告</h5>",
      '<p class="candidate-lab-empty">尚未形成报告。缺失信息可以继续在对话里补充；暂时没有的字段会保留为空。</p>',
      "</section>",
    ].join("");
  }
  const artifacts = isRecord(report.artifacts) ? report.artifacts : {};
  const labels = {
    json: "JSON",
    markdown: "Markdown",
    xlsx: "Excel",
    docx: "Word",
  };
  const links = Object.entries(labels).map(([format, label]) => {
    const url = safeDownloadUrl(artifacts[format]?.download_url);
    return url
      ? `<a class="button compact secondary" href="${escapeHtml(url)}" download>${label}</a>`
      : `<span class="strategy-artifact-unavailable">${label} 不可用</span>`;
  }).join("");
  return [
    `<section class="candidate-lab-subsection candidate-lab-workflow-report" data-status="${escapeHtml(report.freshness === "stale" ? "stale" : "complete")}">`,
    "<h5>策略迭代评审报告</h5>",
    factsTableHtml({
      report_id: report.report_id,
      revision: report.revision,
      status: report.status,
      title: report.title,
      created_at: report.created_at,
    }),
    `<div class="candidate-lab-form-actions">${links}</div>`,
    "</section>",
  ].join("");
}

function strategyWorkflowSpineHtml(workflow) {
  const value = isRecord(workflow) ? workflow : {};
  return [
    '<section class="candidate-lab-result-group candidate-lab-workflow-spine">',
    '<header class="candidate-lab-result-head">',
    "<div><h4>策略开发全流程</h4><p>七阶段状态、双人群样本、最新效果证据和最终报告均来自结构化任务投影。</p></div>",
    "</header>",
    workflowStageSpineHtml(value.stages),
    projectContextWorkflowHtml(value.project_context),
    sampleDesignWorkflowHtml(value.sample_design),
    workflowEvidenceHtml(value.latest_evidence),
    workflowReportHtml(value.report),
    "</section>",
  ].join("");
}

const STRATEGY_TYPE_LABELS = Object.freeze({
  approval: "审批策略",
  reject: "拒绝策略",
  limit: "额度策略",
  pricing: "定价策略",
  segmentation: "分群策略",
});

const STRATEGY_ASSET_STATUS_LABELS = Object.freeze({
  draft: "draft",
  validated: "已验证",
  adopted_local: "本地已采纳",
});

function strategyMaterializationHtml(materialization) {
  if (!isRecord(materialization)) {
    return [
      '<section class="candidate-lab-subsection">',
      "<h5>物化与运行要求</h5>",
      '<p class="candidate-lab-empty">该版本没有 Strategy Pool 物化记录。</p>',
      "</section>",
    ].join("");
  }
  const blockers = Array.isArray(materialization.runtime_blockers)
    ? materialization.runtime_blockers
    : [];
  return [
    '<section class="candidate-lab-subsection">',
    "<h5>物化与运行要求</h5>",
    factsTableHtml({
      materialization_id: materialization.materialization_id,
      pool_id: materialization.pool_id,
      pool_revision_id: materialization.pool_revision_id,
      pool_revision: materialization.pool_revision,
      requirements_count: materialization.requirements_count,
    }),
    blockers.length
      ? [
        '<div class="candidate-lab-risk-group" data-tone="warn">',
        "<strong>运行阻塞</strong>",
        "<ul>",
        ...blockers.slice(0, 24).map(
          (blocker) => `<li>${escapeHtml(readableValue(blocker))}</li>`,
        ),
        "</ul>",
        blockers.length > 24
          ? "<p>其余运行阻塞已由服务端截断。</p>"
          : "",
        "</div>",
      ].join("")
      : '<p class="candidate-lab-boundary-note">当前投影未发现运行阻塞。</p>',
    "</section>",
  ].join("");
}

function strategyArtifactsHtml(artifacts) {
  const value = isRecord(artifacts) ? artifacts : {};
  const items = Array.isArray(value.all) ? value.all.filter(isRecord) : [];
  if (!items.length) {
    return [
      '<section class="candidate-lab-subsection">',
      "<h5>策略产物</h5>",
      '<p class="candidate-lab-empty">当前版本尚无已验证、可下载的策略产物。</p>',
      "</section>",
    ].join("");
  }
  return [
    '<section class="candidate-lab-subsection">',
    "<h5>策略产物</h5>",
    '<div class="candidate-lab-form-actions">',
    ...items.slice(0, 40).map((artifact) => {
      const filename = nonEmptyText(artifact.filename)
        || nonEmptyText(artifact.kind)
        || "策略产物";
      const url = safeDownloadUrl(artifact.download_url);
      return url
        ? `<a class="button compact secondary" href="${escapeHtml(url)}" download>下载 ${escapeHtml(filename)}</a>`
        : `<span class="strategy-artifact-unavailable">${escapeHtml(filename)} 不可下载</span>`;
    }),
    "</div>",
    value.truncated === true
      ? '<p class="candidate-lab-truncated">策略产物列表已由服务端截断。</p>'
      : "",
    "</section>",
  ].join("");
}

function strategyHistoryItemHtml(strategy, championIds) {
  const strategyId = nonEmptyText(strategy?.strategy_id) || "策略版本";
  const strategyType = nonEmptyText(strategy?.strategy_type);
  const assetStatus = nonEmptyText(strategy?.asset_status);
  const isChampion = championIds.has(strategyId);
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-strategy-history-card">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(strategyId)}</strong>`,
    `<small>${escapeHtml(STRATEGY_TYPE_LABELS[strategyType] || strategyType || "策略")} · v${escapeHtml(stablePrimitiveText(strategy?.version))} · ${escapeHtml(STRATEGY_ASSET_STATUS_LABELS[assetStatus] || assetStatus || nonEmptyText(strategy?.status) || "-")}</small>`,
    "</span>",
    isChampion
      ? '<span class="candidate-lab-card-state">当前本地策略</span>'
      : '<span class="candidate-lab-card-state">查看版本</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    factsTableHtml({
      status: strategy?.status,
      asset_status: STRATEGY_ASSET_STATUS_LABELS[assetStatus] || assetStatus,
      created_at: strategy?.created_at,
      adopted_at: strategy?.adopted_at,
      parent_strategy_id: strategy?.parent_strategy_id,
      rule_count: strategy?.rule_count,
    }),
    strategyMaterializationHtml(strategy?.materialization),
    strategyArtifactsHtml(strategy?.artifacts),
    "</div>",
    "</details>",
  ].join("");
}

function strategyHistoryHtml(collection) {
  const value = isRecord(collection) ? collection : {};
  const strategies = projectedStrategyItems({ strategies: value });
  const champions = Array.isArray(value.current_local_champions)
    ? value.current_local_champions.filter(isRecord)
    : [];
  const championIds = new Set(
    champions.map((champion) => nonEmptyText(champion.strategy_id)).filter(Boolean),
  );
  const championSummary = champions.length
    ? [
      '<div class="candidate-lab-boundary-note" data-tone="info">',
      "<strong>当前本地策略</strong>",
      `<p>${champions.slice(0, 5).map((champion) => {
        const type = nonEmptyText(champion.strategy_type);
        return `${escapeHtml(STRATEGY_TYPE_LABELS[type] || type)}：${escapeHtml(nonEmptyText(champion.strategy_id))}（v${escapeHtml(stablePrimitiveText(champion.version))}）`;
      }).join("；")}</p>`,
      "</div>",
    ].join("")
    : '<p class="candidate-lab-empty">当前任务尚无本地已采纳策略。</p>';
  return [
    '<section class="candidate-lab-result-group candidate-lab-strategy-history">',
    '<header class="candidate-lab-result-head">',
    "<div><h4>策略版本历史</h4><p>仅展示当前任务受认证的策略快照、物化关系、运行阻塞和已验证产物。</p></div>",
    "</header>",
    championSummary,
    '<p class="candidate-lab-boundary-note">本地采纳会先提交回测并等待人工确认，不是生产部署，也不会自动启动监控。</p>',
    strategies.length
      ? `<div class="candidate-lab-result-list">${strategies.map(
        (strategy) => strategyHistoryItemHtml(strategy, championIds),
      ).join("")}</div>`
      : '<p class="candidate-lab-empty">尚无策略版本；请先把完整 Strategy Pool 物化为 draft 草稿策略。</p>',
    value.truncated === true
      ? `<p class="candidate-lab-truncated">已显示 ${escapeHtml(strategies.length)} / ${escapeHtml(stablePrimitiveText(value.total))} 个最新策略版本，其余历史已由服务端截断。</p>`
      : "",
    "</section>",
  ].join("");
}

function evidenceDrawerArtifactHtml(artifact) {
  const value = isRecord(artifact) ? artifact : {};
  const inputStatus = nonEmptyText(value.input_binding_status);
  const datasets = Array.isArray(value.datasets)
    ? value.datasets.filter(isRecord)
    : [];
  const explicitInputs = Array.isArray(value.explicit_input_hashes)
    ? value.explicit_input_hashes.filter(isRecord)
    : [];
  const downloadUrl = safeDownloadUrl(value.download_url);
  return [
    '<details class="candidate-lab-evidence-card candidate-lab-drawer-artifact">',
    "<summary>",
    '<span class="candidate-lab-card-title">',
    `<strong>${escapeHtml(nonEmptyText(value.kind) || nonEmptyText(value.artifact_id) || "受认证产物")}</strong>`,
    `<small>${escapeHtml(nonEmptyText(value.origin_tool) || "未知 Tool")} · ${escapeHtml(nonEmptyText(value.producer_version) || nonEmptyText(value.artifact_schema_version) || "版本未单独记录")}</small>`,
    "</span>",
    '<span class="candidate-lab-card-state">查看绑定</span>',
    "</summary>",
    '<div class="candidate-lab-card-body">',
    factsTableHtml({
      artifact_id: value.artifact_id,
      artifact_schema_version: value.artifact_schema_version,
      producer_version: value.producer_version,
      origin_tool: value.origin_tool,
      created_at: value.created_at,
      content_hash: value.content_hash,
      provenance_hash: value.provenance_hash,
      input_binding_hash: value.input_binding_hash,
      input_binding_status: inputStatus === "explicit"
        ? "Tool 明确记录"
        : "由完整 provenance 规范化派生，未冒充原始 Tool 输入 hash",
    }),
    datasets.length
      ? [
        '<section class="candidate-lab-subsection">',
        "<h5>数据集绑定</h5>",
        '<div class="candidate-lab-table-scroll">',
        '<table class="candidate-lab-table"><thead><tr><th>角色</th><th>数据集 ID</th><th>内容 Hash</th></tr></thead><tbody>',
        ...datasets.map((dataset) => [
          "<tr>",
          `<td>${escapeHtml(nonEmptyText(dataset.role) || "-")}</td>`,
          `<td><code>${escapeHtml(nonEmptyText(dataset.dataset_id) || "-")}</code></td>`,
          `<td><code>${escapeHtml(nonEmptyText(dataset.content_hash) || "未单独记录")}</code></td>`,
          "</tr>",
        ].join("")),
        "</tbody></table>",
        "</div>",
        "</section>",
      ].join("")
      : '<p class="candidate-lab-boundary-note">该产物 provenance 未单独暴露 dataset 指针；不从文件内容猜测。</p>',
    explicitInputs.length
      ? [
        '<section class="candidate-lab-subsection">',
        "<h5>Tool 明确记录的输入 Hash</h5>",
        '<div class="candidate-lab-table-scroll">',
        '<table class="candidate-lab-table"><thead><tr><th>字段</th><th>Hash</th></tr></thead><tbody>',
        ...explicitInputs.map((input) => [
          "<tr>",
          `<td>${escapeHtml(nonEmptyText(input.field) || "-")}</td>`,
          `<td><code>${escapeHtml(nonEmptyText(input.hash) || "-")}</code></td>`,
          "</tr>",
        ].join("")),
        "</tbody></table>",
        "</div>",
        "</section>",
      ].join("")
      : "",
    downloadUrl
      ? `<a class="button compact secondary candidate-lab-download" href="${escapeHtml(downloadUrl)}" download>下载该受认证产物</a>`
      : "",
    "</div>",
    "</details>",
  ].join("");
}

function evidenceDrawerDatasetsHtml(collection) {
  const value = isRecord(collection) ? collection : {};
  const items = Array.isArray(value.all) ? value.all.filter(isRecord) : [];
  if (!items.length) {
    return '<p class="candidate-lab-empty">当前投影没有可展示的数据集指针。</p>';
  }
  return [
    '<div class="candidate-lab-table-scroll">',
    '<table class="candidate-lab-table"><thead><tr><th>数据集 ID</th><th>内容 Hash</th><th>关联产物</th></tr></thead><tbody>',
    ...items.map((dataset) => [
      "<tr>",
      `<td><code>${escapeHtml(nonEmptyText(dataset.dataset_id) || "-")}</code></td>`,
      `<td><code>${escapeHtml(nonEmptyText(dataset.content_hash) || "未单独记录")}</code></td>`,
      `<td>${escapeHtml(stablePrimitiveText(Array.isArray(dataset.artifact_ids) ? dataset.artifact_ids.length : 0))}</td>`,
      "</tr>",
    ].join("")),
    "</tbody></table>",
    "</div>",
    value.truncated === true
      ? '<p class="candidate-lab-truncated">数据集指针已由服务端按安全上限截断。</p>'
      : "",
  ].join("");
}

function evidenceDrawerRedFlagsHtml(collection) {
  const value = isRecord(collection) ? collection : {};
  const items = Array.isArray(value.all) ? value.all.filter(isRecord) : [];
  if (!items.length) {
    return '<p class="candidate-lab-empty">当前受认证投影没有红旗。</p>';
  }
  return [
    '<div class="candidate-lab-risk-group" data-tone="warn">',
    "<strong>跨证据红旗</strong>",
    "<ul>",
    ...items.map((flag) => (
      `<li><code>${escapeHtml(nonEmptyText(flag.code) || "risk")}</code> ${escapeHtml(nonEmptyText(flag.message) || "-")}</li>`
    )),
    "</ul>",
    "</div>",
    value.truncated === true
      ? '<p class="candidate-lab-truncated">红旗列表已由服务端截断。</p>'
      : "",
  ].join("");
}

function evidenceDrawerMemoryHtml(collection) {
  const value = isRecord(collection) ? collection : {};
  const items = Array.isArray(value.all) ? value.all.filter(isRecord) : [];
  if (!items.length) {
    return '<p class="candidate-lab-empty">最近一条 Agent 回复未引用受治理记忆。</p>';
  }
  return [
    '<div class="candidate-lab-result-list">',
    ...items.map((reference) => [
      '<article class="candidate-lab-evidence-card candidate-lab-memory-reference">',
      '<div class="candidate-lab-card-body">',
      factsTableHtml({
        memory_id: reference.id,
        kind: reference.kind,
        memory_type: reference.memory_type,
        source_task_id: reference.source_task_id,
        confidence: reference.confidence,
        use_reason: reference.use_reason,
        support_count: reference.support_count,
        source_memory_count: reference.source_memory_count,
      }),
      "</div>",
      "</article>",
    ].join("")),
    "</div>",
    value.truncated === true
      ? '<p class="candidate-lab-truncated">记忆引用已由服务端截断。</p>'
      : "",
    Number(value.omitted || 0) > 0
      ? `<p class="candidate-lab-truncated">${escapeHtml(stablePrimitiveText(value.omitted))} 条格式无效的引用未展示。</p>`
      : "",
  ].join("");
}

function strategyEvidenceDrawerHtml(drawer) {
  const value = isRecord(drawer) ? drawer : {};
  const artifacts = isRecord(value.artifacts) ? value.artifacts : {};
  const artifactItems = Array.isArray(artifacts.all)
    ? artifacts.all.filter(isRecord)
    : [];
  return [
    '<section class="candidate-lab-result-group candidate-lab-evidence-drawer">',
    '<header class="candidate-lab-result-head">',
    "<div><h4>Evidence Drawer</h4><p>统一查看当前页面已认证证据的数据集、产物、Tool/版本、内容与输入绑定 hash、红旗和最近 Agent 记忆引用。</p></div>",
    "</header>",
    '<p class="candidate-lab-boundary-note">仅展示当前任务且已经过对应领域 loader 重验的投影；不读取原始客户行，不把对话自由文本当成业务事实。缺少 Tool 原生输入 hash 时会明确标为 provenance 派生绑定摘要。</p>',
    '<details class="candidate-lab-evidence-card" open>',
    `<summary><span class="candidate-lab-card-title"><strong>受认证产物</strong><small>${escapeHtml(stablePrimitiveText(artifacts.total || 0))} 个当前任务产物</small></span><span class="candidate-lab-card-state">展开核验</span></summary>`,
    '<div class="candidate-lab-card-body">',
    artifactItems.length
      ? `<div class="candidate-lab-result-list">${artifactItems.map(
        (artifact) => evidenceDrawerArtifactHtml(artifact),
      ).join("")}</div>`
      : '<p class="candidate-lab-empty">当前页面尚无受认证产物。</p>',
    artifacts.truncated === true
      ? '<p class="candidate-lab-truncated">产物列表已由服务端按安全上限截断。</p>'
      : "",
    "</div>",
    "</details>",
    '<details class="candidate-lab-evidence-card">',
    "<summary><span class=\"candidate-lab-card-title\"><strong>数据集绑定与红旗</strong><small>跨产物去重后的 lineage</small></span><span class=\"candidate-lab-card-state\">查看</span></summary>",
    '<div class="candidate-lab-card-body">',
    evidenceDrawerDatasetsHtml(value.datasets),
    evidenceDrawerRedFlagsHtml(value.red_flags),
    "</div>",
    "</details>",
    '<details class="candidate-lab-evidence-card">',
    "<summary><span class=\"candidate-lab-card-title\"><strong>Agent 记忆引用</strong><small>最近一条回复的 metadata 审计指针</small></span><span class=\"candidate-lab-card-state\">查看</span></summary>",
    `<div class="candidate-lab-card-body">${evidenceDrawerMemoryHtml(value.memory_references)}</div>`,
    "</details>",
    "</section>",
  ].join("");
}

export function strategyCandidateLabResultsHtml(payload = {}) {
  const candidates = isRecord(payload.candidates) ? payload.candidates : {};
  return [
    strategyWorkflowSpineHtml(payload.workflow),
    strategyEvidenceDrawerHtml(payload.evidence_drawer),
    strategyHistoryHtml(payload.strategies),
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

function splitContextList(value) {
  return String(value || "")
    .split(/[\n,，;；]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseProjectBusinessContext(value) {
  const lines = String(value || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
  if (lines.length > 50) {
    throw new Error("项目业务信息最多填写 50 个字段。");
  }
  const context = {};
  for (const line of lines) {
    const separator = line.indexOf("=");
    if (separator <= 0 || separator === line.length - 1) {
      throw new Error("项目业务信息请按 field.path=内容 每行填写一项。");
    }
    const fieldPath = line.slice(0, separator).trim();
    const fieldValue = line.slice(separator + 1).trim();
    if (
      !PROJECT_CONTEXT_FIELD_PATH_RE.test(fieldPath)
      || PROJECT_CONTEXT_PLATFORM_FIELD_RE.test(fieldPath)
    ) {
      throw new Error(`项目业务信息字段路径不允许：${fieldPath || "-"}`);
    }
    if (Object.hasOwn(context, fieldPath)) {
      throw new Error(`项目业务信息字段路径重复：${fieldPath}`);
    }
    if (!fieldValue || fieldValue.length > 4000) {
      throw new Error(`项目业务信息 ${fieldPath} 必须是 1 到 4000 个字符。`);
    }
    context[fieldPath] = fieldValue;
  }
  return context;
}

function collectStrategyProjectContextInputs(form) {
  const asOf = formValue(form, "project_context_as_of");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(asOf)) {
    throw new Error("请填写项目现状截止日期（YYYY-MM-DD）。");
  }
  const inputs = {
    as_of: asOf,
    business_context: parseProjectBusinessContext(
      formValue(form, "project_context_business_context"),
    ),
    explicit_unavailable: checkedValues(form, "project_context_unavailable"),
    external_report_filenames: splitContextList(
      formValue(form, "project_context_external_reports"),
    ),
  };
  const scope = formValue(form, "project_context_scope");
  if (scope) inputs.scope = scope;
  if (inputs.external_report_filenames.length > 20) {
    throw new Error("外部历史材料文件名最多填写 20 个。");
  }
  if (
    inputs.external_report_filenames.some(
      (name) => !name || name.includes("/") || name.includes("\\") || name.includes("\0"),
    )
  ) {
    throw new Error("外部历史材料只能填写任务材料目录中的文件名。");
  }
  if (
    inputs.explicit_unavailable.some(
      (fieldPath) => !PROJECT_CONTEXT_FIELD_PATH_RE.test(fieldPath),
    )
  ) {
    throw new Error("暂缺信息字段路径无效。");
  }
  return inputs;
}

function collectStrategyPoolMaterializeInputs(form) {
  const select = formField(form, "pool_materialize_strategy_type");
  const strategyType = nonEmptyText(select?.value);
  const option = select?.selectedOptions?.[0];
  if (
    !STRATEGY_POOL_TYPES.includes(strategyType)
    || option?.dataset?.candidateLabProjection !== "1"
    || nonEmptyText(option?.dataset?.poolId) === ""
  ) {
    throw new Error("请选择当前受认证的非空 Strategy Pool。");
  }
  return { strategy_type: strategyType };
}

function collectStrategyDslDeliveryInputs(form) {
  const select = formField(form, "dsl_delivery_strategy_id");
  const strategyId = nonEmptyText(select?.value);
  const option = select?.selectedOptions?.[0];
  if (
    !STRATEGY_ID_RE.test(strategyId)
    || option?.dataset?.candidateLabProjection !== "1"
    || nonEmptyText(option?.dataset?.strategyId) !== strategyId
  ) {
    throw new Error("请选择当前任务受认证的策略版本。");
  }
  return { strategy_id: strategyId };
}

const STRATEGY_ADOPTION_ECONOMICS_COMPONENTS = Object.freeze({
  limit: Object.freeze(["pd", "lgd", "utilization"]),
  pricing: Object.freeze([
    "ead",
    "pd",
    "lgd",
    "funding_rate",
    "term_months",
    "operating_cost_per_loan",
  ]),
});

function collectStrategyLifecycleAdoptionRequest(form) {
  const option = selectedProjectionOption(
    form,
    "lifecycle_adopt_strategy_id",
    "待采纳 draft 策略",
  );
  const strategyId = nonEmptyText(option.value);
  const strategyType = nonEmptyText(option.dataset?.strategyType);
  if (
    !STRATEGY_ID_RE.test(strategyId)
    || nonEmptyText(option.dataset?.strategyId) !== strategyId
    || !STRATEGY_POOL_TYPES.includes(strategyType)
    || nonEmptyText(option.dataset?.assetStatus) !== "draft"
  ) {
    throw new Error("只能从当前受认证投影选择 draft 草稿策略。");
  }
  const adoptionReason = formValue(form, "lifecycle_adoption_reason");
  if (adoptionReason.length < 2 || adoptionReason.length > 1000) {
    throw new Error("本地采纳理由必须是 2 到 1000 个字符。");
  }
  const request = {
    request_kind: "strategy_lifecycle",
    operation: "adopt",
    strategy_type: strategyType,
    strategy_id: strategyId,
    adoption_reason: adoptionReason,
  };
  const components = STRATEGY_ADOPTION_ECONOMICS_COMPONENTS[strategyType];
  if (!components) return request;

  const economicsInputs = {};
  for (const component of components) {
    const mode = formValue(form, `lifecycle_adopt_${component}_mode`);
    if (mode === "column") {
      const column = formValue(
        form,
        `lifecycle_adopt_${component}_column`,
      );
      if (!column || column.length > 200) {
        throw new Error(`${fieldLabel(component)} 必须填写可用数据列。`);
      }
      economicsInputs[`${component}_col`] = column;
      continue;
    }
    if (mode !== "value") {
      throw new Error(
        `${fieldLabel(component)} 必须选择使用数据列或固定值。`,
      );
    }
    const rawValue = formValue(form, `lifecycle_adopt_${component}_value`);
    const value = Number(rawValue);
    if (
      !rawValue
      || !Number.isFinite(value)
      || value < 0
      || (component === "term_months" && value <= 0)
      || (
        ["pd", "lgd", "utilization", "funding_rate"].includes(component)
        && value > 1
      )
    ) {
      throw new Error(`${fieldLabel(component)} 固定值不符合经济口径范围。`);
    }
    economicsInputs[`${component}_value`] = value;
  }
  request.economics_inputs = economicsInputs;
  return request;
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

function parseRequiredInteger(raw, label, { min, max }) {
  const value = Number(raw);
  if (
    !nonEmptyText(raw)
    || !Number.isSafeInteger(value)
    || value < min
    || value > max
  ) {
    throw new Error(`${label}必须是 ${min} 到 ${max} 的整数。`);
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

function optionalProjectionValues(form, name, label) {
  const field = formField(form, name);
  const selected = Array.from(field?.selectedOptions || [])
    .filter((option) => nonEmptyText(option.value));
  if (selected.some((option) => option.dataset?.candidateLabProjection !== "1")) {
    throw new Error(`${label}必须来自当前 Strategy Pool 的受认证投影。`);
  }
  return uniqueValues(
    selected.map((option) => nonEmptyText(option.value)),
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

function collectCrossCandidateSearchInputs(form) {
  const select = formField(form, "cross_search_features");
  const options = Array.from(select?.selectedOptions || []);
  if (options.length < 2 || options.length > 20) {
    throw new Error("Cross 自动搜索必须明确选择 2 到 20 个独立字段。");
  }
  if (options.some((option) => (
    option.dataset?.candidateLabProjection !== "1"
    || nonEmptyText(option.value) !== nonEmptyText(option.dataset?.feature)
    || !nonEmptyText(option.value)
  ))) {
    throw new Error("Cross 自动搜索字段必须来自当前单变量受认证投影。");
  }
  const features = options.map((option) => nonEmptyText(option.value));
  if (new Set(features).size !== features.length) {
    throw new Error("Cross 自动搜索不能重复选择同一字段，必须使用独立字段。");
  }
  const maxPairs = optionalNumber(
    form,
    "cross_search_max_pairs",
    { integer: true },
  );
  if (maxPairs === undefined || maxPairs < 1 || maxPairs > 190) {
    throw new Error("Cross 自动搜索预算 max_pairs 必须是 1 到 190 的整数。");
  }
  return { features, max_pairs: maxPairs };
}

function collectCrossCandidateBuildFromSearchInputs(form) {
  const search = selectedProjectionOption(
    form,
    "cross_build_search_id",
    "Cross 搜索证据",
  );
  const pair = selectedProjectionOption(
    form,
    "cross_build_pair_id",
    "Cross 字段组合",
  );
  const searchId = nonEmptyText(search.value);
  const pairId = nonEmptyText(pair.value);
  if (
    !CROSS_SEARCH_ID_RE.test(searchId)
    || !CROSS_PAIR_ID_RE.test(pairId)
    || nonEmptyText(search.dataset?.searchId) !== searchId
    || nonEmptyText(pair.dataset?.searchId) !== searchId
    || nonEmptyText(pair.dataset?.pairId) !== pairId
  ) {
    throw new Error(
      "Cross 字段组合必须属于当前选择的受认证搜索证据。",
    );
  }
  return { search_id: searchId, pair_id: pairId };
}

function collectCrossRuleSearchInputs(form) {
  const select = formField(form, "cross_rule_features");
  const options = Array.from(select?.selectedOptions || []);
  if (options.length < 2 || options.length > 12) {
    throw new Error("Cross 阈值规则搜索必须明确选择 2 到 12 个独立字段。");
  }
  if (options.some((option) => (
    option.dataset?.candidateLabProjection !== "1"
    || nonEmptyText(option.value) !== nonEmptyText(option.dataset?.feature)
    || !nonEmptyText(option.value)
  ))) {
    throw new Error("Cross 阈值规则字段必须来自当前单变量受认证投影。");
  }
  const features = options.map((option) => nonEmptyText(option.value));
  if (new Set(features).size !== features.length) {
    throw new Error("Cross 阈值规则搜索不能重复选择同一字段。");
  }
  const dimension = optionalNumber(form, "cross_rule_dimension", {
    integer: true,
  });
  if (![2, 3].includes(dimension)) {
    throw new Error("Cross 阈值规则维度只能是 2D 或 3D。");
  }
  if (features.length < dimension) {
    throw new Error("参与搜索的字段数不能少于规则维度。");
  }
  const minLift = optionalNumber(form, "cross_rule_min_lift");
  const minBadCount = optionalNumber(form, "cross_rule_min_bad_count", {
    integer: true,
  });
  const maxHitShare = optionalNumber(form, "cross_rule_max_hit_share");
  const minAmountLift = optionalNumber(form, "cross_rule_min_amount_lift");
  const maxTrials = optionalNumber(form, "cross_rule_max_trials", {
    integer: true,
  });
  if (
    minLift === undefined || minLift < 0 || minLift > 1000
    || minBadCount === undefined || minBadCount < 0
    || maxHitShare === undefined || maxHitShare < 0 || maxHitShare > 1
    || (minAmountLift !== undefined
      && (minAmountLift < 0 || minAmountLift > 1000))
    || maxTrials === undefined || maxTrials < 1 || maxTrials > 5000
  ) {
    throw new Error("Cross 阈值规则约束或 max_trials 超出受控范围。");
  }
  return {
    features,
    dimension,
    constraints: {
      min_lift: minLift,
      min_bad_count: minBadCount,
      max_hit_share: maxHitShare,
      min_amount_lift: minAmountLift === undefined ? null : minAmountLift,
    },
    max_trials: maxTrials,
  };
}

function collectCrossRuleCandidateBuildInputs(form) {
  const search = selectedProjectionOption(
    form,
    "cross_rule_build_search_id",
    "Cross 阈值规则搜索证据",
  );
  const rule = selectedProjectionOption(
    form,
    "cross_rule_build_rule_id",
    "Cross 阈值规则",
  );
  const searchId = nonEmptyText(search.value);
  const ruleId = nonEmptyText(rule.value);
  if (
    !CROSS_RULE_SEARCH_ID_RE.test(searchId)
    || !CROSS_RULE_ID_RE.test(ruleId)
    || nonEmptyText(search.dataset?.searchId) !== searchId
    || nonEmptyText(rule.dataset?.searchId) !== searchId
    || nonEmptyText(rule.dataset?.ruleId) !== ruleId
  ) {
    throw new Error("Cross 阈值规则必须属于当前选择的受认证搜索证据。");
  }
  const inputs = { search_id: searchId, rule_id: ruleId };
  optionalText(
    inputs,
    "selection_reason",
    formValue(form, "cross_rule_selection_reason"),
  );
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

function parseRawPdBandEdges(value) {
  const edges = splitValues(value).map((item) => Number(item));
  if (
    edges.length < 3
    || edges.length > 21
    || edges.some((item) => !Number.isFinite(item))
  ) {
    throw new Error("原始 PD 边界必须包含 3 到 21 个有限数字。");
  }
  if (edges[0] !== 0 || edges.at(-1) !== 1) {
    throw new Error("原始 PD 边界必须从 0 开始并以 1 结束。");
  }
  if (edges.some((item, index) => (
    item < 0
    || item > 1
    || (index > 0 && item <= edges[index - 1])
  ))) {
    throw new Error("原始 PD 边界必须位于 0 到 1 且严格递增。");
  }
  return edges;
}

function collectScorecardBandInputs(form) {
  const mode = formValue(form, "scorecard_banding_mode");
  if (mode === "equal_frequency") {
    const binCount = optionalNumber(
      form,
      "scorecard_bin_count",
      { integer: true },
    );
    if (binCount === undefined) return {};
    if (binCount < 2 || binCount > 20) {
      throw new Error("评分卡分档数必须是 2 到 20 的整数。");
    }
    return { bin_count: binCount };
  }
  if (mode === "raw_pd_edges") {
    return {
      raw_pd_band_edges: parseRawPdBandEdges(
        formValue(form, "raw_pd_band_edges"),
      ),
    };
  }
  throw new Error("请选择等频分档或自定义原始 PD 边界。");
}

function collectScorecardCutoffSelectionInputs(form) {
  const asset = selectedProjectionOption(
    form,
    "scorecard_asset_id",
    "评分卡分档资产",
  );
  const cutoff = selectedProjectionOption(
    form,
    "scorecard_cutoff_id",
    "Cutoff",
  );
  const assetId = nonEmptyText(asset.value);
  const cutoffId = nonEmptyText(cutoff.value);
  if (cutoff.dataset?.sourceAssetId !== assetId) {
    throw new Error("Cutoff 必须属于当前选择的评分卡分档资产。");
  }
  const inputs = {
    asset_id: assetId,
    cutoff_id: cutoffId,
  };
  optionalText(
    inputs,
    "reason",
    formValue(form, "scorecard_selection_reason"),
  );
  return inputs;
}

function sampleDesignLiteral(rawValue) {
  const raw = nonEmptyText(rawValue);
  const separator = raw.indexOf(":");
  if (separator < 1) {
    throw new Error(
      "样本人群条件值必须标注类型，例如 text:APP 或 number:30。",
    );
  }
  const type = raw.slice(0, separator).trim().toLowerCase();
  const value = raw.slice(separator + 1).trim();
  if (!value) throw new Error("样本人群条件值不能为空。");
  if (type === "text") return value;
  if (type === "boolean") {
    if (!["true", "false"].includes(value.toLowerCase())) {
      throw new Error("boolean 条件值只能是 true 或 false。");
    }
    return value.toLowerCase() === "true";
  }
  if (type !== "number") {
    throw new Error("样本人群条件值类型只能是 text、number 或 boolean。");
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error("样本人群 number 条件值必须是有限数字。");
  }
  return number;
}

function collectSamplePopulation(form, prefix) {
  const column = formValue(form, `${prefix}_population_column`);
  const operator = formValue(form, `${prefix}_population_operator`);
  const rawValue = formValue(form, `${prefix}_population_value`);
  if (!column && !operator && !rawValue) {
    return { inclusion: null, exclusion: null };
  }
  if (!column || !operator) {
    throw new Error(`${prefix === "approval" ? "审批" : "风险"}人群条件必须完整填写字段与运算符。`);
  }
  const condition = { column, operator };
  if (!["is_null", "is_not_null"].includes(operator)) {
    if (!rawValue) {
      throw new Error(`${prefix === "approval" ? "审批" : "风险"}人群条件必须填写显式类型值。`);
    }
    condition.value = sampleDesignLiteral(rawValue);
  } else if (rawValue) {
    throw new Error("空值判断不需要填写条件值。");
  }
  return {
    inclusion: { match: "all", conditions: [condition] },
    exclusion: null,
  };
}

function nullableText(form, name) {
  return formValue(form, name) || null;
}

function nullableInteger(form, name) {
  const raw = formValue(form, name);
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${fieldLabel(name)}必须是正整数。`);
  }
  return value;
}

function sampleTimeRange(form, prefix, label) {
  const start = nullableText(form, `sample_${prefix}_start`);
  const end = nullableText(form, `sample_${prefix}_end`);
  if (!start && !end) {
    throw new Error(`${label}至少填写一个时间边界。`);
  }
  if (start && end && start > end) {
    throw new Error(`${label}开始日期不能晚于结束日期。`);
  }
  return { start, end };
}

function collectSampleDesignV2Inputs(form) {
  const targetBadValue = Number(formValue(form, "sample_target_bad_value"));
  if (![0, 1].includes(targetBadValue)) {
    throw new Error("坏样本值只能是 0 或 1。");
  }
  const relationship = formValue(form, "sample_relationship");
  if (!["nested_same_cohort", "parallel_time_cohorts"].includes(relationship)) {
    throw new Error("请选择审批与风险人群关系。");
  }
  const timeField = formValue(form, "sample_time_field");
  if (!timeField) throw new Error("请填写用于样本分区的时间字段。");
  const maturityStatus = formValue(form, "sample_maturity_status");
  const performanceStatus = formValue(form, "sample_performance_status");
  const observationStatus = formValue(form, "sample_observation_status");
  const historicalStatus = formValue(form, "sample_historical_score_status");
  const maturityDays = nullableInteger(form, "sample_maturity_days");
  const maturityCutoff = nullableText(form, "sample_maturity_cutoff");
  const maturityReason = nullableText(form, "sample_maturity_reason");
  const performanceDays = nullableInteger(form, "sample_performance_days");
  const observationStart = nullableText(form, "sample_observation_start");
  const observationEnd = nullableText(form, "sample_observation_end");
  const historicalColumn = nullableText(
    form,
    "sample_historical_score_column",
  );
  const historicalDirection = nullableText(
    form,
    "sample_historical_score_direction",
  );
  const historicalReason = nullableText(
    form,
    "sample_historical_score_reason",
  );
  if (
    ![
      "confirmed_matured",
      "not_matured",
      "unknown",
      "unavailable",
    ].includes(maturityStatus)
  ) {
    throw new Error("请选择有效的样本成熟度状态。");
  }
  if (!["provided", "unavailable"].includes(performanceStatus)) {
    throw new Error("请选择有效的表现窗口状态。");
  }
  if (!["provided", "unavailable"].includes(observationStatus)) {
    throw new Error("请选择有效的观察窗口状态。");
  }
  const evaluatedMaturity = [
    "confirmed_matured",
    "not_matured",
  ].includes(maturityStatus);
  if (evaluatedMaturity) {
    if (!maturityDays || !maturityCutoff) {
      throw new Error("已评估成熟度必须填写表现期天数和成熟截止日。");
    }
    if (performanceStatus !== "provided" || performanceDays !== maturityDays) {
      throw new Error("成熟度与表现窗口必须使用相同的已提供天数。");
    }
    if (maturityStatus === "not_matured" && !maturityReason) {
      throw new Error("尚未成熟时必须填写成熟度说明。");
    }
    if (maturityStatus === "confirmed_matured" && maturityReason) {
      throw new Error("已确认成熟时请清空成熟度说明。");
    }
  } else if (maturityDays || maturityCutoff || !maturityReason) {
    throw new Error(
      "成熟度未知或暂不可提供时，天数和截止日应留空并填写说明。",
    );
  }
  if (
    (performanceStatus === "provided" && !performanceDays)
    || (performanceStatus === "unavailable" && performanceDays)
  ) {
    throw new Error("表现窗口状态与天数不一致。");
  }
  if (observationStatus === "provided") {
    if (!observationStart || !observationEnd) {
      throw new Error("已提供观察窗口时必须填写开始和结束日期。");
    }
    if (observationStart > observationEnd) {
      throw new Error("观察窗口开始日期不能晚于结束日期。");
    }
  } else if (observationStart || observationEnd) {
    throw new Error("观察窗口暂不可提供时，开始和结束日期应留空。");
  }
  if (
    !["available", "unavailable", "not_applicable"].includes(historicalStatus)
  ) {
    throw new Error("请选择有效的历史分状态。");
  }
  if (historicalStatus === "available") {
    if (
      !historicalColumn
      || !["higher_is_riskier", "lower_is_riskier"].includes(
        historicalDirection,
      )
    ) {
      throw new Error("历史分可用时必须填写字段和风险方向。");
    }
    if (historicalReason) {
      throw new Error("历史分可用时请清空历史分说明。");
    }
  } else if (historicalColumn || historicalDirection || !historicalReason) {
    throw new Error("历史分不可用或不适用时，只填写说明。");
  }
  return {
    target_bad_value: targetBadValue,
    drop_nan_labels: Boolean(
      formField(form, "sample_drop_nan_labels")?.checked,
    ),
    relationship,
    approval_population: collectSamplePopulation(form, "approval"),
    risk_population: collectSamplePopulation(form, "risk"),
    partitioning: {
      method: "time_ranges",
      column: timeField,
      ranges: {
        development: sampleTimeRange(form, "development", "开发集"),
        validation: sampleTimeRange(form, "validation", "验证集"),
        oot: sampleTimeRange(form, "oot", "OOT"),
      },
    },
    maturity: {
      status: maturityStatus,
      performance_window_days: maturityDays,
      cutoff_date: maturityCutoff,
      reason: maturityReason,
    },
    performance_window: {
      status: performanceStatus,
      days: performanceDays,
    },
    observation_window: {
      status: observationStatus,
      start: observationStart,
      end: observationEnd,
    },
    field_bindings: {
      entity_field: nullableText(form, "sample_entity_field"),
      time_field: timeField,
      group_field: nullableText(form, "sample_group_field"),
      month_field: nullableText(form, "sample_month_field"),
      weight_field: nullableText(form, "sample_weight_field"),
      loan_amount_field: nullableText(form, "sample_loan_amount_field"),
      overdue_amount_field: nullableText(form, "sample_overdue_amount_field"),
    },
    historical_score: {
      status: historicalStatus,
      column: historicalColumn,
      direction: historicalDirection,
      reason: historicalReason,
    },
  };
}

function collectCandidateMonthlyStabilityInputs(form) {
  const mode = formValue(form, "stability_source_mode");
  if (mode === "pool_entry") {
    const entry = selectedProjectionOption(
      form,
      "stability_pool_entry",
      "当前 Strategy Pool 条目",
    );
    const strategyType = nonEmptyText(entry.dataset?.strategyType);
    if (!strategyType) {
      throw new Error("当前 Pool 条目缺少受认证策略类型。");
    }
    return {
      strategy_type: strategyType,
      entry_id: nonEmptyText(entry.value),
    };
  }
  if (mode === "univariate_asset") {
    const asset = selectedProjectionOption(
      form,
      "stability_asset_id",
      "单变量候选资产",
    );
    return { asset_id: nonEmptyText(asset.value) };
  }
  throw new Error("请选择当前 Pool 条目或单变量候选资产。");
}

function collectStrategyPoolValidationInputs(form) {
  const strategyType = selectedStrategyPoolType(
    form,
    "pool_validation_strategy_type",
  );
  const partition = formValue(form, "pool_validation_partition");
  if (!STRATEGY_POOL_TYPES.includes(strategyType)) {
    throw new Error(
      "独立样本回放验证需要选择受认证的 Strategy Pool 类型。",
    );
  }
  if (!["validation", "oot"].includes(partition)) {
    throw new Error(
      "独立样本回放验证分区只能是 validation 或 oot。",
    );
  }
  return {
    strategy_type: strategyType,
    partition,
  };
}

function collectStrategyPoolStabilityInputs(form) {
  return {
    strategy_type: selectedStrategyPoolType(
      form,
      "pool_stability_strategy_type",
    ),
  };
}

function collectStrategyPoolImpactInputs(form) {
  const strategyType = selectedStrategyPoolType(
    form,
    "pool_impact_strategy_type",
  );
  if (!["approval", "reject"].includes(strategyType)) {
    throw new Error("Pool Impact 只支持 approval 或 reject Pool。");
  }
  const comparisonMode = formValue(form, "pool_impact_comparison_mode")
    || "absolute";
  const inputs = {
    strategy_type: strategyType,
    comparison_mode: comparisonMode,
    drop_nan_labels: Boolean(
      formField(form, "pool_impact_drop_nan_labels")?.checked,
    ),
  };
  optionalText(
    inputs,
    "baseline_strategy_id",
    formValue(form, "pool_impact_baseline_strategy_id"),
  );
  optionalText(inputs, "month_col", formValue(form, "pool_impact_month_col"));
  optionalText(
    inputs,
    "loan_amount_col",
    formValue(form, "pool_impact_loan_amount_col"),
  );
  optionalText(
    inputs,
    "overdue_amount_col",
    formValue(form, "pool_impact_overdue_amount_col"),
  );
  if (
    comparisonMode === "vs_baseline"
    && !inputs.baseline_strategy_id
  ) {
    throw new Error("对比历史策略时必须填写完整 baseline_strategy_id。");
  }
  if (
    comparisonMode === "absolute"
    && inputs.baseline_strategy_id
  ) {
    throw new Error("绝对影响测算不能同时填写 baseline_strategy_id。");
  }
  return inputs;
}

function collectStrategyImpactCubeInputs(form) {
  const inputs = {
    strategy_type: selectedStrategyPoolType(
      form,
      "impact_cube_strategy_type",
    ),
  };
  const partitions = uniqueValues(
    checkedValues(form, "impact_cube_partitions"),
    "ImpactCube 分区",
  );
  if (partitions.length) inputs.partitions = partitions;
  optionalText(inputs, "month_col", formValue(form, "impact_cube_month_col"));
  optionalText(inputs, "group_col", formValue(form, "impact_cube_group_col"));
  optionalText(
    inputs,
    "segment_col",
    formValue(form, "impact_cube_segment_col"),
  );
  optionalText(
    inputs,
    "current_strategy_id",
    formValue(form, "impact_cube_current_strategy_id"),
  );
  const dimensions = [
    inputs.month_col,
    inputs.group_col,
    inputs.segment_col,
  ].filter(Boolean);
  if (new Set(dimensions).size !== dimensions.length) {
    throw new Error("月份、分组与分群维度必须使用不同字段。");
  }
  return inputs;
}

function collectStrategyReportBundleV2Inputs(form) {
  const title = formValue(form, "strategy_report_title")
    || "策略迭代评审报告";
  if (title.length > 200) throw new Error("报告标题最多 200 个字符。");
  const status = formValue(form, "strategy_report_status") || "partial";
  if (!["draft", "partial", "final"].includes(status)) {
    throw new Error("报告状态只能是 draft、partial 或 final。");
  }
  return { title, status };
}

function collectStrategyPoolApplyInputs(form) {
  const selected = selectedProjectionOption(
    form,
    "pool_apply_strategy_type",
    "当前非空 Strategy Pool",
  );
  const strategyType = nonEmptyText(selected.value);
  if (
    !STRATEGY_POOL_TYPES.includes(strategyType)
    || nonEmptyText(selected.dataset?.strategyType) !== strategyType
  ) {
    throw new Error(
      "Strategy Pool 必须来自当前任务受认证的非空 Pool 投影。",
    );
  }
  const inputs = { strategy_type: strategyType };
  const outputPrefix = formValue(form, "pool_apply_output_prefix");
  if (outputPrefix) {
    if (!STRATEGY_POOL_APPLY_PREFIX_RE.test(outputPrefix)) {
      throw new Error(
        "输出列前缀必须是最长 48 字符的安全 ASCII identifier prefix，且不能以数字开头。",
      );
    }
    inputs.output_prefix = outputPrefix;
  }
  return inputs;
}

function selectedStrategyPoolType(form, fieldName) {
  const selected = selectedProjectionOption(
    form,
    fieldName,
    "当前非空 Strategy Pool",
  );
  const strategyType = nonEmptyText(selected.value);
  if (
    !STRATEGY_POOL_TYPES.includes(strategyType)
    || nonEmptyText(selected.dataset?.strategyType) !== strategyType
  ) {
    throw new Error(
      "Strategy Pool 必须来自当前任务受认证的非空 Pool 投影。",
    );
  }
  return strategyType;
}

function selectedStrategyPoolEntry(form, fieldName, strategyType) {
  const selected = selectedProjectionOption(
    form,
    fieldName,
    "当前 Strategy Pool 条目",
  );
  const entryId = nonEmptyText(selected.value);
  if (
    !STRATEGY_POOL_ENTRY_ID_RE.test(entryId)
    || nonEmptyText(selected.dataset?.strategyType) !== strategyType
    || nonEmptyText(selected.dataset?.entryId) !== entryId
  ) {
    throw new Error(
      "Strategy Pool 条目必须来自所选 Pool 的当前受认证投影。",
    );
  }
  return entryId;
}

function optionalStrategyPoolReason(inputs, form, fieldName) {
  const reason = formValue(form, fieldName);
  if (!reason) return;
  if (reason.length > 500) {
    throw new Error("Strategy Pool 操作理由最多 500 个字符。");
  }
  inputs.reason = reason;
}

function collectStrategyPoolCompileInputs(form) {
  return {
    strategy_type: selectedStrategyPoolType(
      form,
      "pool_compile_strategy_type",
    ),
  };
}

function collectStrategyPoolRemoveEntryInputs(form) {
  const strategyType = selectedStrategyPoolType(
    form,
    "pool_remove_strategy_type",
  );
  const inputs = {
    strategy_type: strategyType,
    entry_id: selectedStrategyPoolEntry(
      form,
      "pool_remove_entry_id",
      strategyType,
    ),
  };
  optionalStrategyPoolReason(inputs, form, "pool_remove_reason");
  return inputs;
}

function strategyPoolTypedAction(
  form,
  strategyType,
  typeField,
  valueField,
) {
  const typeControl = formField(form, typeField);
  const actionType = formValue(form, typeField);
  const allowed = STRATEGY_POOL_ACTION_TYPES[strategyType] || [];
  if (!allowed.includes(actionType)) {
    throw new Error(
      `${actionType || "所选"}动作不适用于 ${strategyType} Strategy Pool。`,
    );
  }
  if (
    typeControl?.dataset?.candidateLabPoolAddLocked === "1"
    && typeControl.dataset.candidateLabPoolAddTypedAction
  ) {
    let projectedAction;
    try {
      projectedAction = JSON.parse(
        typeControl.dataset.candidateLabPoolAddTypedAction,
      );
    } catch {
      throw new Error("当前 Strategy Pool 默认动作投影无效，请刷新后重试。");
    }
    const exactAction = minimalProjectedPoolAction(
      projectedAction,
      strategyType,
    );
    if (!exactAction || exactAction.type !== actionType) {
      throw new Error("当前 Strategy Pool 默认动作投影已过期，请刷新后重试。");
    }
    return exactAction;
  }
  if (["approval", "reject", "review"].includes(actionType)) {
    return { type: actionType };
  }
  const rawValue = formValue(form, valueField);
  if (!rawValue) {
    throw new Error(`${actionType} 动作必须填写动作值。`);
  }
  if (actionType === "segment") {
    return { type: actionType, value: rawValue };
  }
  const value = Number(rawValue);
  if (!Number.isFinite(value)) {
    throw new Error(`${actionType} 动作值必须是有限数字。`);
  }
  if (actionType === "limit" && value < 0) {
    throw new Error("额度动作值必须是非负有限数字。");
  }
  if (actionType === "pricing" && (value < 0 || value > 1)) {
    throw new Error("定价动作值必须是 0 到 1 之间的有限数字。");
  }
  return { type: actionType, value };
}

function strategyPoolAction(form, strategyType) {
  return strategyPoolTypedAction(
    form,
    strategyType,
    "pool_action_type",
    "pool_action_value",
  );
}

function collectStrategyPoolAddCandidateInputs(form) {
  const strategyType = formValue(form, "pool_add_strategy_type");
  if (!STRATEGY_POOL_TYPES.includes(strategyType)) {
    throw new Error("请选择一个受支持的 Strategy Pool 类型。");
  }
  const selected = selectedProjectionOption(
    form,
    "pool_add_source_id",
    "已物化候选来源",
  );
  const sourceId = nonEmptyText(selected.value);
  const sourceKind = nonEmptyText(selected.dataset?.sourceKind);
  const pointerKind = nonEmptyText(selected.dataset?.pointerKind);
  const expectedPointer = STRATEGY_POOL_ADD_SOURCE_KINDS[sourceKind];
  if (
    !expectedPointer
    || pointerKind !== expectedPointer
    || nonEmptyText(selected.dataset?.sourceId) !== sourceId
    || (
      pointerKind === "candidate_asset_id"
        ? !STRATEGY_POOL_CANDIDATE_ASSET_ID_RE.test(sourceId)
        : !STRATEGY_POOL_ADD_SELECTION_RE.test(sourceId)
    )
  ) {
    throw new Error(
      "候选来源必须来自当前任务受认证、已物化的 Pool 入池投影。",
    );
  }
  const sourceStrategyType = nonEmptyText(
    selected.dataset?.strategyType,
  );
  if (
    sourceKind === "voting_candidate"
      ? sourceStrategyType !== strategyType
      : Boolean(sourceStrategyType)
  ) {
    throw new Error(
      "候选来源与所选 Strategy Pool 类型不兼容或投影绑定已漂移。",
    );
  }
  const inputs = {
    strategy_type: strategyType,
    [pointerKind]: sourceId,
    default_action: strategyPoolTypedAction(
      form,
      strategyType,
      "pool_add_default_action_type",
      "pool_add_default_action_value",
    ),
    action: strategyPoolTypedAction(
      form,
      strategyType,
      "pool_add_action_type",
      "pool_add_action_value",
    ),
  };
  const placementMode = formValue(form, "pool_add_placement_mode");
  if (sourceKind === "voting_candidate") {
    if (!STRATEGY_POOL_VOTING_PLACEMENTS.includes(placementMode)) {
      throw new Error("Voting 候选入池前必须明确选择一种放置方式。");
    }
    inputs.placement_mode = placementMode;
  } else if (placementMode) {
    throw new Error("普通候选不能提交 Voting placement_mode。");
  }
  optionalStrategyPoolReason(inputs, form, "pool_add_reason");
  return inputs;
}

function collectStrategyPoolSetActionInputs(form) {
  const strategyType = selectedStrategyPoolType(
    form,
    "pool_action_strategy_type",
  );
  const inputs = {
    strategy_type: strategyType,
    entry_id: selectedStrategyPoolEntry(
      form,
      "pool_action_entry_id",
      strategyType,
    ),
    action: strategyPoolAction(form, strategyType),
  };
  optionalStrategyPoolReason(inputs, form, "pool_action_reason");
  return inputs;
}

function collectStrategyPoolReorderInputs(form) {
  const strategyType = selectedStrategyPoolType(
    form,
    "pool_reorder_strategy_type",
  );
  const orderField = formField(form, "pool_reorder_ordered_ids");
  const options = Array.from(orderField?.options || []);
  if (options.length < 1 || options.length > 200) {
    throw new Error(
      "Strategy Pool 完整重排必须包含当前全部 1 到 200 个条目。",
    );
  }
  const orderedIds = options.map((option) => {
    const entryId = nonEmptyText(option.value);
    if (
      option.dataset?.candidateLabProjection !== "1"
      || !STRATEGY_POOL_ENTRY_ID_RE.test(entryId)
      || nonEmptyText(option.dataset?.strategyType) !== strategyType
      || nonEmptyText(option.dataset?.entryId) !== entryId
    ) {
      throw new Error(
        "Strategy Pool 完整重排只能使用当前受认证投影中的 Entry ID。",
      );
    }
    return entryId;
  });
  if (new Set(orderedIds).size !== orderedIds.length) {
    throw new Error("Strategy Pool 完整重排不能包含重复 Entry ID。");
  }
  const inputs = {
    strategy_type: strategyType,
    ordered_ids: orderedIds,
  };
  optionalStrategyPoolReason(inputs, form, "pool_reorder_reason");
  return inputs;
}

function parseVotingConstraints(value) {
  const text = String(value || "").trim();
  if (!text) return [];
  const rows = text
    .split(/[;；\n]+/)
    .map((row) => row.trim())
    .filter(Boolean);
  if (rows.length > 32) {
    throw new Error("Voting 资格约束最多填写 32 项。");
  }
  const seen = new Set();
  const constraints = rows.map((row) => {
    const match = row.match(
      /^([a-z_]+)\s*(>=|<=|gte|lte)\s*(\d+(?:\.\d+)?%?)$/i,
    );
    if (!match) {
      throw new Error(
        `Voting 资格约束“${row}”格式无效；请使用 metric >= value 或 metric <= value。`,
      );
    }
    const metric = match[1].toLowerCase();
    if (!VOTING_SEARCH_METRICS.includes(metric)) {
      throw new Error(`Voting 资格约束指标 ${metric} 不受支持。`);
    }
    const operator = [">=", "gte"].includes(match[2].toLowerCase())
      ? "gte"
      : "lte";
    const percent = match[3].endsWith("%");
    const number = Number(percent ? match[3].slice(0, -1) : match[3]);
    const normalized = percent ? number / 100 : number;
    if (!Number.isFinite(normalized) || normalized < 0) {
      throw new Error(`Voting 资格约束 ${metric} 必须使用非负有限数字。`);
    }
    const identity = `${metric}\u001f${operator}`;
    if (seen.has(identity)) {
      throw new Error(`Voting 资格约束不能重复 ${metric} ${operator}。`);
    }
    seen.add(identity);
    return { metric, operator, value: normalized };
  });
  return constraints.sort(
    (left, right) => (
      left.metric.localeCompare(right.metric)
      || left.operator.localeCompare(right.operator)
      || left.value - right.value
    ),
  );
}

function collectVotingCandidateSearchInputs(form) {
  const strategy = selectedProjectionOption(
    form,
    "voting_strategy_type",
    "当前 Strategy Pool",
  );
  const strategyType = nonEmptyText(strategy.value);
  const memberCount = optionalNumber(
    form,
    "voting_member_count",
    { integer: true },
  );
  const n = optionalNumber(form, "voting_n", { integer: true });
  const maxCombinations = optionalNumber(
    form,
    "voting_max_combinations",
    { integer: true },
  );
  if (memberCount === undefined || memberCount < 2 || memberCount > 50) {
    throw new Error("Voting 每个组合的成员数 K 必须是 2 到 50 的整数。");
  }
  if (n === undefined || n < 1 || n > memberCount) {
    throw new Error(`Voting 命中阈值 n 必须是 1 到 K=${memberCount} 的整数。`);
  }
  if (
    maxCombinations === undefined
    || maxCombinations < 1
    || maxCombinations > 10000
  ) {
    throw new Error("Voting 确定性搜索预算必须是 1 到 10000 的整数。");
  }
  const objectiveMetric = formValue(form, "voting_objective_metric");
  const objectiveDirection = formValue(form, "voting_objective_direction");
  if (!VOTING_SEARCH_METRICS.includes(objectiveMetric)) {
    throw new Error("请选择受支持的 Voting 排序指标。");
  }
  if (!["maximize", "minimize"].includes(objectiveDirection)) {
    throw new Error("Voting 排序方向只能是最大化或最小化。");
  }
  const includeRuleIds = optionalProjectionValues(
    form,
    "voting_include_rule_ids",
    "必须包含规则",
  ).sort();
  const excludeRuleIds = optionalProjectionValues(
    form,
    "voting_exclude_rule_ids",
    "排除规则",
  ).sort();
  if (
    [...includeRuleIds, ...excludeRuleIds].some(
      (ruleId) => !VOTING_RULE_ID_RE.test(ruleId),
    )
  ) {
    throw new Error("Voting 规则选择包含无效的 rule_id。");
  }
  const overlap = includeRuleIds.filter((ruleId) => excludeRuleIds.includes(ruleId));
  if (overlap.length) {
    throw new Error("Voting 必须包含规则与排除规则不能重叠。");
  }
  if (includeRuleIds.length > memberCount) {
    throw new Error("Voting 必须包含规则数量不能超过 K。");
  }
  const constraints = parseVotingConstraints(
    formValue(form, "voting_constraints"),
  );
  const minimumShareMetric = {
    bad_rate: "hit_share",
    lift: "hit_share",
    weighted_bad_rate: "weighted_hit_share",
    bad_amount_rate: "hit_amount_share",
  }[objectiveMetric];
  if (
    objectiveDirection === "minimize"
    && minimumShareMetric
    && !constraints.some((constraint) => (
      constraint.metric === minimumShareMetric
      && constraint.operator === "gte"
      && constraint.value > 0
    ))
  ) {
    throw new Error(
      `最小化 ${objectiveMetric} 时必须设置正数 ${minimumShareMetric} >= 约束，避免空命中组合排在最前。`,
    );
  }
  return {
    strategy_type: strategyType,
    member_count: memberCount,
    n,
    objective: {
      metric: objectiveMetric,
      direction: objectiveDirection,
    },
    constraints,
    include_rule_ids: includeRuleIds,
    exclude_rule_ids: excludeRuleIds,
    max_combinations: maxCombinations,
  };
}

function collectVotingCandidateBuildFromSearchInputs(form) {
  const search = selectedProjectionOption(
    form,
    "voting_search_id",
    "Voting 搜索证据",
  );
  const combo = selectedProjectionOption(
    form,
    "voting_combo_id",
    "Voting 组合",
  );
  const searchId = nonEmptyText(search.value);
  const comboId = nonEmptyText(combo.value);
  if (!VOTING_SEARCH_ID_RE.test(searchId) || !VOTING_COMBO_ID_RE.test(comboId)) {
    throw new Error("Voting 搜索或组合指针格式无效。");
  }
  if (nonEmptyText(combo.dataset?.sourceSearchId) !== searchId) {
    throw new Error("Voting 组合必须属于当前选择的受认证搜索证据。");
  }
  const inputs = {
    search_id: searchId,
    combo_id: comboId,
  };
  const strategyType = nonEmptyText(search.dataset?.strategyType);
  if (strategyType) inputs.strategy_type = strategyType;
  return inputs;
}

function collectInteractiveTreeSplitSearchInputs(form) {
  const source = selectedProjectionOption(
    form,
    "interactive_tree_search_source_id",
    "树或 revision",
  );
  const node = selectedProjectionOption(
    form,
    "interactive_tree_search_node_id",
    "当前可见节点",
  );
  const sourceTreeId = nonEmptyText(source.value);
  const nodeId = nonEmptyText(node.value);
  const mode = formValue(form, "interactive_tree_search_mode");
  if (
    !INTERACTIVE_TREE_SOURCE_ID_RE.test(sourceTreeId)
    || !INTERACTIVE_TREE_NODE_ID_RE.test(nodeId)
    || nonEmptyText(source.dataset?.sourceTreeId) !== sourceTreeId
    || nonEmptyText(node.dataset?.sourceTreeId) !== sourceTreeId
    || nonEmptyText(node.dataset?.nodeId) !== nodeId
    || !["all_features", "selected_features"].includes(mode)
  ) {
    throw new Error("树节点搜索必须来自当前任务的受认证可见拓扑。");
  }
  const maxThresholds = parseRequiredInteger(
    formValue(form, "interactive_tree_search_max_thresholds"),
    "每特征最大阈值数",
    { min: 1, max: 20 },
  );
  const maxRowEvaluations = parseRequiredInteger(
    formValue(form, "interactive_tree_search_max_row_evaluations"),
    "总行评估预算",
    { min: 1, max: 20000000 },
  );
  const inputs = {
    source_tree_id: sourceTreeId,
    node_id: nodeId,
    mode,
    max_thresholds_per_feature: maxThresholds,
    max_row_evaluations: maxRowEvaluations,
  };
  if (mode === "selected_features") {
    const features = splitValues(
      formValue(form, "interactive_tree_search_features"),
    );
    const universe = new Set(
      nonEmptyText(source.dataset?.featureUniverse)
        .split("\u001f")
        .map(nonEmptyText)
        .filter(Boolean),
    );
    if (
      !features.length
      || features.length > 50
      || new Set(features).size !== features.length
      || features.some((feature) => !universe.has(feature))
    ) {
      throw new Error("指定特征必须非空、唯一，并来自当前来源树的认证特征全集。");
    }
    inputs.features = features;
  }
  return inputs;
}

function collectInteractiveTreeAutoContinuationInputs(form) {
  const searchId = nonEmptyText(
    formValue(form, "interactive_tree_continuation_search_id"),
  );
  const candidateId = nonEmptyText(
    formValue(form, "interactive_tree_continuation_candidate_id"),
  );
  if (
    !INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE.test(searchId)
    || !INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE.test(candidateId)
  ) {
    throw new Error("请先从受认证搜索结果中明确带入一个 eligible 候选。");
  }
  const minimumGain = Number(
    formValue(form, "interactive_tree_continuation_min_gain"),
  );
  if (!Number.isFinite(minimumGain) || minimumGain < 0 || minimumGain > 0.5) {
    throw new Error("最小 Gini 增益必须是 0 到 0.5 的有限数值。");
  }
  const inputs = {
    search_id: searchId,
    candidate_id: candidateId,
    max_additional_depth: parseRequiredInteger(
      formValue(form, "interactive_tree_continuation_max_depth"),
      "最大追加深度",
      { min: 1, max: 6 },
    ),
    min_gini_gain: minimumGain,
    max_generated_nodes: parseRequiredInteger(
      formValue(form, "interactive_tree_continuation_max_nodes"),
      "最大生成节点数",
      { min: 3, max: 127 },
    ),
    max_thresholds_per_feature: parseRequiredInteger(
      formValue(form, "interactive_tree_continuation_max_thresholds"),
      "每特征最大阈值数",
      { min: 1, max: 20 },
    ),
    max_row_evaluations: parseRequiredInteger(
      formValue(form, "interactive_tree_continuation_max_row_evaluations"),
      "总行评估预算",
      { min: 1, max: 20000000 },
    ),
    objective: formValue(form, "interactive_tree_continuation_objective"),
    tie_break: formValue(form, "interactive_tree_continuation_tie_break"),
  };
  if (
    inputs.objective !== "max_gini_gain"
    || inputs.tie_break
      !== "eligible_gain_feature_threshold_candidate_id"
  ) {
    throw new Error("续建的固定目标或并列规则已改变，请刷新页面。");
  }
  optionalText(
    inputs,
    "reason",
    formValue(form, "interactive_tree_continuation_reason"),
  );
  return inputs;
}

function collectInteractiveTreeRevisionInputs(form) {
  const source = selectedProjectionOption(
    form,
    "interactive_tree_source_id",
    "树或 revision",
  );
  const node = selectedProjectionOption(
    form,
    "interactive_tree_node_id",
    "可剪枝节点",
  );
  const sourceTreeId = nonEmptyText(source.value);
  const nodeId = nonEmptyText(node.value);
  const operation = formValue(form, "interactive_tree_operation")
    || "prune_subtree";
  if (
    !INTERACTIVE_TREE_SOURCE_ID_RE.test(sourceTreeId)
    || !INTERACTIVE_TREE_NODE_ID_RE.test(nodeId)
    || nonEmptyText(source.dataset?.sourceTreeId) !== sourceTreeId
    || nonEmptyText(node.dataset?.sourceTreeId) !== sourceTreeId
    || nonEmptyText(node.dataset?.nodeId) !== nodeId
    || ![
      "prune_subtree",
      "adjust_split_threshold",
      "replace_split_feature",
    ].includes(operation)
    || node.dataset?.operation !== operation
  ) {
    throw new Error(
      "交互式树节点必须来自当前选择分支和操作的受认证投影。",
    );
  }
  const inputs = {
    source_tree_id: sourceTreeId,
    node_id: nodeId,
    operation,
  };
  if (
    operation === "adjust_split_threshold"
    || operation === "replace_split_feature"
  ) {
    const currentThreshold = Number(
      nonEmptyText(node.dataset?.currentThreshold),
    );
    const rawThreshold = formValue(form, "interactive_tree_threshold");
    const threshold = Number(rawThreshold);
    if (
      !nonEmptyText(node.dataset?.feature)
      || !Number.isFinite(currentThreshold)
    ) {
      throw new Error("阈值调整必须来自包含当前字段和阈值的受认证投影。");
    }
    if (
      !rawThreshold
      || !Number.isFinite(threshold)
      || (
        Number.isInteger(threshold)
        && !Number.isSafeInteger(threshold)
      )
    ) {
      throw new Error("新 threshold 必须是有限且可精确表达的数字。");
    }
    if (
      operation === "adjust_split_threshold"
      && threshold === currentThreshold
    ) {
      throw new Error("新 threshold 必须与当前阈值不同。");
    }
    inputs.threshold = threshold;
    if (operation === "replace_split_feature") {
      const featureOption = selectedProjectionOption(
        form,
        "interactive_tree_feature",
        "新分裂字段",
      );
      const feature = nonEmptyText(featureOption.value);
      if (
        !feature
        || nonEmptyText(featureOption.dataset?.sourceTreeId) !== sourceTreeId
        || featureOption.dataset?.candidateLabProjection !== "1"
        || feature === nonEmptyText(node.dataset?.feature)
      ) {
        throw new Error("新分裂字段必须从当前来源树的认证字段全集中明确选择。");
      }
      inputs.feature = feature;
    }
  }
  optionalText(inputs, "reason", formValue(form, "interactive_tree_reason"));
  return inputs;
}

function collectInteractiveTreeFrontierMaterializationInputs(form) {
  const revision = selectedProjectionOption(
    form,
    "interactive_tree_frontier_revision_id",
    "交互树 revision",
  );
  const frontier = selectedProjectionOption(
    form,
    "interactive_tree_frontier_source_node_id",
    "前沿节点",
  );
  const revisionId = nonEmptyText(revision.value);
  const sourceNodeId = nonEmptyText(frontier.value);
  if (
    !INTERACTIVE_TREE_REVISION_ID_RE.test(revisionId)
    || !INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE.test(sourceNodeId)
    || nonEmptyText(revision.dataset?.revisionId) !== revisionId
    || nonEmptyText(frontier.dataset?.revisionId) !== revisionId
    || nonEmptyText(frontier.dataset?.sourceNodeId) !== sourceNodeId
  ) {
    throw new Error(
      "交互树前沿节点必须来自当前 revision 的受认证 frontier 投影。",
    );
  }
  const inputs = {
    revision_id: revisionId,
    source_node_id: sourceNodeId,
  };
  optionalText(
    inputs,
    "selection_reason",
    formValue(form, "interactive_tree_frontier_selection_reason"),
  );
  return inputs;
}

function collectInteractiveTreeFrontierGroupMaterializationInputs(form) {
  const revision = selectedProjectionOption(
    form,
    "interactive_tree_frontier_group_revision_id",
    "交互树 revision",
  );
  const revisionId = nonEmptyText(revision.value);
  const nodeSelect = formField(
    form,
    "interactive_tree_frontier_group_source_node_ids",
  );
  const nodeOptions = Array.from(nodeSelect?.selectedOptions || []);
  if (nodeOptions.length < 2 || nodeOptions.length > 50) {
    throw new Error("交互树前沿 OR 分组必须选择 2 到 50 个节点。");
  }
  const sourceNodeIds = uniqueValues(
    nodeOptions.map((option) => nonEmptyText(option.value)),
    "交互树前沿 OR 分组节点",
  );
  if (
    !INTERACTIVE_TREE_REVISION_ID_RE.test(revisionId)
    || nonEmptyText(revision.dataset?.revisionId) !== revisionId
    || sourceNodeIds.some((sourceNodeId, index) => {
      const option = nodeOptions[index];
      return (
        option.dataset?.candidateLabProjection !== "1"
        || !INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE.test(sourceNodeId)
        || nonEmptyText(option.dataset?.revisionId) !== revisionId
        || nonEmptyText(option.dataset?.sourceNodeId) !== sourceNodeId
      );
    })
  ) {
    throw new Error(
      "交互树前沿 OR 分组节点必须来自当前 revision 的受认证 frontier 投影。",
    );
  }
  const inputs = {
    revision_id: revisionId,
    source_node_ids: sourceNodeIds,
  };
  optionalText(
    inputs,
    "selection_reason",
    formValue(form, "interactive_tree_frontier_group_selection_reason"),
  );
  return inputs;
}

export function collectStrategyCandidateLabRequest(form) {
  const workflow = nonEmptyText(form?.dataset?.candidateLabWorkflow);
  if (!STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(workflow)) {
    throw new Error("Candidate Lab 表单包含未开放的策略 workflow。");
  }
  if (workflow === "strategy_lifecycle_adopt") {
    return collectStrategyLifecycleAdoptionRequest(form);
  }
  const workflowInputs = {
    strategy_project_context: collectStrategyProjectContextInputs,
    strategy_sample_design_v2: collectSampleDesignV2Inputs,
    univariate_candidate_analysis: collectUnivariateInputs,
    univariate_candidate_refinement: collectRefinementInputs,
    cross_matrix_analysis: collectCrossInputs,
    cross_matrix_candidate_search: collectCrossCandidateSearchInputs,
    cross_matrix_candidate_build_from_search:
      collectCrossCandidateBuildFromSearchInputs,
    cross_rule_search: collectCrossRuleSearchInputs,
    cross_rule_candidate_build_from_search:
      collectCrossRuleCandidateBuildInputs,
    automatic_tree_candidate_build: collectTreeInputs,
    scorecard_band_build: collectScorecardBandInputs,
    scorecard_cutoff_selection: collectScorecardCutoffSelectionInputs,
    candidate_monthly_stability: collectCandidateMonthlyStabilityInputs,
    strategy_pool_add_candidate: collectStrategyPoolAddCandidateInputs,
    strategy_pool_compile: collectStrategyPoolCompileInputs,
    strategy_pool_remove_entry: collectStrategyPoolRemoveEntryInputs,
    strategy_pool_set_action: collectStrategyPoolSetActionInputs,
    strategy_pool_reorder: collectStrategyPoolReorderInputs,
    strategy_pool_apply: collectStrategyPoolApplyInputs,
    strategy_pool_validation: collectStrategyPoolValidationInputs,
    strategy_pool_stability: collectStrategyPoolStabilityInputs,
    strategy_pool_impact: collectStrategyPoolImpactInputs,
    strategy_impact_cube: collectStrategyImpactCubeInputs,
    strategy_pool_materialize: collectStrategyPoolMaterializeInputs,
    strategy_dsl_delivery: collectStrategyDslDeliveryInputs,
    strategy_report_bundle_v2: collectStrategyReportBundleV2Inputs,
    voting_candidate_search: collectVotingCandidateSearchInputs,
    voting_candidate_build_from_search:
      collectVotingCandidateBuildFromSearchInputs,
    interactive_tree_split_search: collectInteractiveTreeSplitSearchInputs,
    interactive_tree_auto_continuation:
      collectInteractiveTreeAutoContinuationInputs,
    interactive_tree_revision: collectInteractiveTreeRevisionInputs,
    interactive_tree_frontier_group_materialization:
      collectInteractiveTreeFrontierGroupMaterializationInputs,
    interactive_tree_frontier_materialization:
      collectInteractiveTreeFrontierMaterializationInputs,
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

function interactiveTreeForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="interactive_tree_revision"]',
  ) || null;
}

function interactiveTreeSplitSearchForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="interactive_tree_split_search"]',
  ) || null;
}

function interactiveTreeProjectionSources(payload) {
  const candidates = isRecord(payload?.candidates) ? payload.candidates : {};
  const collections = [
    candidates.automatic_tree,
    candidates.interactive_tree_revision,
  ];
  const seen = new Set();
  return collections.flatMap((collection) => collectionItems(collection))
    .filter((item) => {
      const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
      if (
        !INTERACTIVE_TREE_SOURCE_ID_RE.test(sourceTreeId)
        || seen.has(sourceTreeId)
      ) {
        return false;
      }
      seen.add(sourceTreeId);
      return true;
    });
}

function syncInteractiveTreeSplitSearchControls(
  form,
  payload,
  { preserveNode = true } = {},
) {
  if (!form) return;
  const sourceSelect = formField(
    form,
    "interactive_tree_search_source_id",
  );
  const nodeSelect = formField(form, "interactive_tree_search_node_id");
  if (!sourceSelect || !nodeSelect) return;
  const sources = interactiveTreeProjectionSources(payload);
  const previousSource = nonEmptyText(sourceSelect.value);
  const previousNode = preserveNode ? nonEmptyText(nodeSelect.value) : "";
  sourceSelect.innerHTML = [
    '<option value="">请选择自动树或不可变 revision</option>',
    ...sources.map((item) => {
      const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
      const featureUniverse = Array.isArray(
        item?.pointers?.feature_universe,
      )
        ? item.pointers.feature_universe.map(nonEmptyText).filter(Boolean)
        : [];
      const visibleCount = (
        Array.isArray(item?.pointers?.nodes)
          ? item.pointers.nodes
          : []
      ).filter((node) => node?.is_visible === true).length;
      return projectionOptionHtml(
        sourceTreeId,
        `${sourceTreeId} · ${visibleCount} 个可见节点 · ${featureUniverse.length} 个认证特征`,
        {
          "candidate-lab-projection": "1",
          "source-tree-id": sourceTreeId,
          "feature-universe": featureUniverse.join("\u001f"),
        },
      );
    }),
  ].join("");
  sourceSelect.value = selectContainsValue(sourceSelect, previousSource)
    ? previousSource
    : "";
  const selectedSourceId = nonEmptyText(sourceSelect.value);
  const selectedSource = sources.find(
    (item) => item?.detail?.source_tree_id === selectedSourceId,
  );
  const nodes = (
    Array.isArray(selectedSource?.pointers?.nodes)
      ? selectedSource.pointers.nodes
      : []
  ).filter((node) => (
    isRecord(node)
    && node.is_visible === true
    && INTERACTIVE_TREE_NODE_ID_RE.test(nonEmptyText(node.node_id))
  ));
  nodeSelect.innerHTML = [
    '<option value="">请选择要分析的当前可见节点</option>',
    ...nodes.map((node) => projectionOptionHtml(
      node.node_id,
      `${node.node_id} · ${node.kind}${node.feature ? ` · ${node.feature} ≤ ${stablePrimitiveText(node.threshold)}` : ""}`,
      {
        "candidate-lab-projection": "1",
        "source-tree-id": selectedSourceId,
        "node-id": node.node_id,
      },
    )),
  ].join("");
  nodeSelect.value = selectContainsValue(nodeSelect, previousNode)
    ? previousNode
    : "";
  const selectedSourceOption = Array.from(
    sourceSelect.selectedOptions || [],
  )[0] || null;
  const featurePanel = form.querySelector?.(
    "[data-candidate-lab-tree-search-features-panel]",
  );
  const selectedMode = formValue(form, "interactive_tree_search_mode")
    || "all_features";
  featurePanel?.classList?.toggle?.(
    "hidden",
    selectedMode !== "selected_features",
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-tree-search-help]",
  );
  if (help) {
    if (!sources.length) {
      help.textContent = "当前任务尚无受认证自动树，请先构建自动规则树。";
    } else if (!selectedSource) {
      help.textContent = "请明确选择来源树或 revision；页面不会自动代选。";
    } else if (!nodes.length) {
      help.textContent = "该来源树当前没有可见节点可供搜索。";
    } else if (!nodeSelect.value) {
      help.textContent = "请明确选择节点；页面不会按风险、样本量或排名自动代选。";
    } else {
      const featureCount = nonEmptyText(
        selectedSourceOption?.dataset?.featureUniverse,
      ).split("\u001f").filter(Boolean).length;
      help.textContent = (
        selectedMode === "all_features"
          ? `将搜索该树全部 ${featureCount} 个认证特征。`
          : `请从该树 ${featureCount} 个认证特征中填写明确子集。`
      ) + " 排名只用于浏览，不会修改树。";
    }
  }
}

function interactiveTreePointer(payload, sourceTreeId, nodeId) {
  return interactiveTreeRevisionPointer(
    payload,
    sourceTreeId,
    nodeId,
    "prune_subtree",
  );
}

function interactiveTreePointersForOperation(item, operation) {
  if (operation === "adjust_split_threshold") {
    return interactiveTreeThresholdEligiblePointers(item);
  }
  if (operation === "replace_split_feature") {
    return interactiveTreeFeatureEligiblePointers(item);
  }
  return interactiveTreeEligiblePointers(item);
}

function interactiveTreeRevisionPointer(
  payload,
  sourceTreeId,
  nodeId,
  operation,
) {
  const source = interactiveTreeProjectionSources(payload).find(
    (item) => item?.detail?.source_tree_id === sourceTreeId,
  );
  if (!source) return null;
  return interactiveTreePointersForOperation(source, operation).find(
    (pointer) => pointer.node_id === nodeId,
  ) || null;
}

function interactiveTreeRevisionRequestIsCurrent(payload, inputs) {
  if (!isRecord(inputs)) return false;
  const operation = nonEmptyText(inputs.operation);
  const pointer = interactiveTreeRevisionPointer(
    payload,
    nonEmptyText(inputs.source_tree_id),
    nonEmptyText(inputs.node_id),
    operation,
  );
  if (!pointer) return false;
  if (operation === "prune_subtree") {
    return (
      !Object.prototype.hasOwnProperty.call(inputs, "feature")
      && !Object.prototype.hasOwnProperty.call(inputs, "threshold")
    );
  }
  if (
    operation !== "adjust_split_threshold"
    && operation !== "replace_split_feature"
  ) return false;
  const threshold = Number(inputs.threshold);
  if (
    typeof inputs.threshold !== "number"
    || !Number.isFinite(threshold)
  ) return false;
  if (operation === "adjust_split_threshold") {
    return threshold !== Number(pointer.current_threshold);
  }
  const source = interactiveTreeProjectionSources(payload).find(
    (item) => item?.detail?.source_tree_id === inputs.source_tree_id,
  );
  const features = new Set(
    (Array.isArray(source?.pointers?.feature_universe)
      ? source.pointers.feature_universe
      : [])
      .map(nonEmptyText)
      .filter(Boolean),
  );
  return (
    typeof inputs.feature === "string"
    && features.has(inputs.feature)
    && inputs.feature !== pointer.current_feature
    && (
      threshold !== Number(pointer.current_threshold)
      || inputs.feature !== pointer.current_feature
    )
  );
}

function interactiveTreeSplitSearchRequestIsCurrent(payload, inputs) {
  if (!isRecord(inputs)) return false;
  const sourceTreeId = nonEmptyText(inputs.source_tree_id);
  const nodeId = nonEmptyText(inputs.node_id);
  const source = interactiveTreeProjectionSources(payload).find(
    (item) => item?.detail?.source_tree_id === sourceTreeId,
  );
  const node = (
    Array.isArray(source?.pointers?.nodes)
      ? source.pointers.nodes
      : []
  ).find((item) => (
    item?.node_id === nodeId && item?.is_visible === true
  ));
  const featureUniverse = new Set(
    (Array.isArray(source?.pointers?.feature_universe)
      ? source.pointers.feature_universe
      : [])
      .map(nonEmptyText)
      .filter(Boolean),
  );
  if (
    !source
    || !node
    || !Number.isInteger(inputs.max_thresholds_per_feature)
    || inputs.max_thresholds_per_feature < 1
    || inputs.max_thresholds_per_feature > 20
    || !Number.isInteger(inputs.max_row_evaluations)
    || inputs.max_row_evaluations < 1
    || inputs.max_row_evaluations > 20000000
  ) return false;
  if (inputs.mode === "all_features") {
    return !Object.prototype.hasOwnProperty.call(inputs, "features");
  }
  return (
    inputs.mode === "selected_features"
    && Array.isArray(inputs.features)
    && inputs.features.length > 0
    && inputs.features.length <= 50
    && new Set(inputs.features).size === inputs.features.length
    && inputs.features.every((feature) => featureUniverse.has(feature))
  );
}

function interactiveTreeSplitCandidatePointer(
  payload,
  searchId,
  candidateId,
) {
  const searches = collectionItems(
    payload?.candidates?.interactive_tree_split_search,
  );
  const search = searches.find((item) => (
    item?.kind === "interactive_tree_split_search"
    && item?.search_id === searchId
    && INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE.test(searchId)
  ));
  const candidate = (
    Array.isArray(search?.candidates) ? search.candidates : []
  ).find((item) => (
    item?.candidate_id === candidateId
    && item?.eligible === true
    && INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE.test(candidateId)
  ));
  return search && candidate ? { search, candidate } : null;
}

function syncInteractiveTreeRevisionControls(
  form,
  payload,
  {
    requestedOperation = "",
    requestedSourceTreeId = "",
    requestedNodeId = "",
    preserveNode = true,
  } = {},
) {
  if (!form) return;
  const operationField = formField(form, "interactive_tree_operation");
  const sourceSelect = formField(form, "interactive_tree_source_id");
  const nodeSelect = formField(form, "interactive_tree_node_id");
  if (!sourceSelect || !nodeSelect) return;
  const allowedOperations = [
    "prune_subtree",
    "adjust_split_threshold",
    "replace_split_feature",
  ];
  const requested = nonEmptyText(requestedOperation);
  if (operationField && allowedOperations.includes(requested)) {
    operationField.value = requested;
  }
  const operation = allowedOperations.includes(formValue(
    form,
    "interactive_tree_operation",
  ))
    ? formValue(form, "interactive_tree_operation")
    : "prune_subtree";
  const sources = interactiveTreeProjectionSources(payload);
  const previousSource = nonEmptyText(sourceSelect.value);
  const previousNode = preserveNode ? nonEmptyText(nodeSelect.value) : "";
  const previousNodeOption = Array.from(
    nodeSelect.selectedOptions || [],
  )[0] || null;
  const previousPointerIdentity = previousNodeOption
    ? [
      nonEmptyText(previousNodeOption.dataset?.sourceTreeId),
      nonEmptyText(previousNodeOption.dataset?.nodeId),
      nonEmptyText(previousNodeOption.dataset?.operation),
      nonEmptyText(previousNodeOption.dataset?.currentThreshold),
    ].join("\u001f")
    : "";
  const thresholdField = formField(form, "interactive_tree_threshold");
  const previousThreshold = String(thresholdField?.value ?? "");
  sourceSelect.innerHTML = [
    '<option value="">请选择自动树或不可变 revision</option>',
    ...sources.map((item) => {
      const sourceTreeId = nonEmptyText(item?.detail?.source_tree_id);
      const eligibleCount = interactiveTreePointersForOperation(
        item,
        operation,
      ).length;
      const type = item?.kind === "interactive_tree_revision"
        ? "revision"
        : "automatic";
      const pointerLabel = operation === "adjust_split_threshold"
        ? "个可调阈值节点"
        : (
          operation === "replace_split_feature"
            ? "个可换字段节点"
            : "个可剪枝节点"
        );
      return projectionOptionHtml(
        sourceTreeId,
        `${sourceTreeId} · ${type} · ${eligibleCount} ${pointerLabel}`,
        {
          "candidate-lab-projection": "1",
          "source-tree-id": sourceTreeId,
        },
      );
    }),
  ].join("");
  const preferredSource = nonEmptyText(requestedSourceTreeId) || previousSource;
  if (selectContainsValue(sourceSelect, preferredSource)) {
    sourceSelect.value = preferredSource;
  } else {
    sourceSelect.value = "";
  }

  const selectedSourceId = nonEmptyText(sourceSelect.value);
  const selectedSource = sources.find(
    (item) => item?.detail?.source_tree_id === selectedSourceId,
  );
  const pointers = selectedSource
    ? interactiveTreePointersForOperation(selectedSource, operation)
    : [];
  nodeSelect.innerHTML = [
    '<option value="">请选择当前分支可见 split 节点</option>',
    ...pointers.map((pointer) => {
      const node = selectedSource?.pointers?.nodes?.find?.(
        (item) => item?.node_id === pointer.node_id,
      );
      const label = node?.feature
        ? `${pointer.node_id} · ${node.feature} ≤ ${stablePrimitiveText(node.threshold)}`
        : pointer.node_id;
      return projectionOptionHtml(
        pointer.node_id,
        label,
        {
          "candidate-lab-projection": "1",
          "source-tree-id": pointer.source_tree_id,
          "node-id": pointer.node_id,
          operation,
          ...(operation !== "prune_subtree"
            ? {
              feature: pointer.feature || pointer.current_feature,
              "current-threshold": stablePrimitiveText(
                pointer.current_threshold,
              ),
            }
            : {}),
        },
      );
    }),
  ].join("");
  const preferredNode = nonEmptyText(requestedNodeId) || previousNode;
  if (selectContainsValue(nodeSelect, preferredNode)) {
    nodeSelect.value = preferredNode;
  } else {
    nodeSelect.value = "";
  }
  const selectedNodeOption = Array.from(
    nodeSelect.selectedOptions || [],
  )[0] || null;
  const selectedPointerIdentity = selectedNodeOption
    ? [
      nonEmptyText(selectedNodeOption.dataset?.sourceTreeId),
      nonEmptyText(selectedNodeOption.dataset?.nodeId),
      nonEmptyText(selectedNodeOption.dataset?.operation),
      nonEmptyText(selectedNodeOption.dataset?.currentThreshold),
    ].join("\u001f")
    : "";
  const isThresholdAdjustment = operation === "adjust_split_threshold";
  const isFeatureReplacement = operation === "replace_split_feature";
  const isSplitAdjustment = (
    isThresholdAdjustment || isFeatureReplacement
  );
  const thresholdPanel = form.querySelector?.(
    "[data-candidate-lab-tree-threshold-panel]",
  );
  thresholdPanel?.classList?.toggle?.("hidden", !isSplitAdjustment);
  const thresholdFeature = form.querySelector?.(
    "[data-candidate-lab-tree-threshold-feature]",
  );
  if (thresholdFeature) {
    thresholdFeature.textContent = isSplitAdjustment
      ? nonEmptyText(selectedNodeOption?.dataset?.feature) || "请先选择节点"
      : "—";
  }
  const currentThreshold = form.querySelector?.(
    "[data-candidate-lab-tree-current-threshold]",
  );
  if (currentThreshold) {
    currentThreshold.textContent = isSplitAdjustment
      ? nonEmptyText(selectedNodeOption?.dataset?.currentThreshold)
        || "请先选择节点"
      : "—";
  }
  if (thresholdField) {
    thresholdField.value = (
      isSplitAdjustment
      && previousPointerIdentity
      && previousPointerIdentity === selectedPointerIdentity
    )
      ? previousThreshold
      : "";
  }
  const featurePanel = form.querySelector?.(
    "[data-candidate-lab-tree-feature-panel]",
  );
  featurePanel?.classList?.toggle?.("hidden", !isFeatureReplacement);
  const featureField = formField(form, "interactive_tree_feature");
  if (featureField) {
    const currentFeature = nonEmptyText(
      selectedNodeOption?.dataset?.feature,
    );
    const featureUniverse = Array.isArray(
      selectedSource?.pointers?.feature_universe,
    )
      ? selectedSource.pointers.feature_universe
        .map(nonEmptyText)
        .filter((feature) => feature && feature !== currentFeature)
      : [];
    featureField.innerHTML = [
      '<option value="">请选择认证字段</option>',
      ...featureUniverse.map((feature) => projectionOptionHtml(
        feature,
        feature,
        {
          "candidate-lab-projection": "1",
          "source-tree-id": selectedSourceId,
        },
      )),
    ].join("");
  }
  const help = form.querySelector?.("[data-candidate-lab-tree-help]");
  if (help) {
    if (!sources.length) {
      help.textContent = "当前任务尚无受认证自动树，请先构建自动规则树。";
    } else if (!selectedSource) {
      help.textContent = "请明确选择来源树或 revision；即使只有一个，页面也不会自动代选。";
    } else if (!pointers.length) {
      help.textContent = isThresholdAdjustment
        ? "该分支当前没有受认证的可调阈值 split 节点。"
        : "该分支当前没有可继续剪枝的可见 split 节点。";
    } else if (!selectedNodeOption) {
      help.textContent = "请明确选择一个受认证 split 节点；页面不会按效果、排名或数量自动代选。";
    } else if (isFeatureReplacement) {
      help.textContent = "请明确选择一个认证新字段并填写有限阈值；页面不会按排名自动代选，提交只创建不可变 revision。";
    } else if (isThresholdAdjustment) {
      help.textContent = `当前 ${nonEmptyText(
        selectedNodeOption.dataset?.feature,
      )} 阈值为 ${nonEmptyText(
        selectedNodeOption.dataset?.currentThreshold,
      )}；新阈值必须不同，页面不会自动代选，提交只创建不可变 revision。`;
    } else {
      help.textContent = "确认后只创建不可变剪枝 revision；不会覆盖来源树，也不会自动入池。";
    }
  }
}

function interactiveTreeFrontierMaterializationForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="interactive_tree_frontier_materialization"]',
  ) || null;
}

function interactiveTreeFrontierGroupMaterializationForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="interactive_tree_frontier_group_materialization"]',
  ) || null;
}

function interactiveTreeFrontierProjectionSources(payload) {
  const collection = payload?.candidates?.interactive_tree_revision;
  const seen = new Set();
  return collectionItems(collection).filter((item) => {
    const revisionId = nonEmptyText(item?.detail?.revision_id);
    if (
      item?.kind !== "interactive_tree_revision"
      || !INTERACTIVE_TREE_REVISION_ID_RE.test(revisionId)
      || seen.has(revisionId)
    ) {
      return false;
    }
    seen.add(revisionId);
    return true;
  });
}

function interactiveTreeFrontierPointer(payload, revisionId, sourceNodeId) {
  const source = interactiveTreeFrontierProjectionSources(payload).find(
    (item) => item?.detail?.revision_id === revisionId,
  );
  if (!source) return null;
  return interactiveTreeFrontierEligiblePointers(source).find(
    (pointer) => pointer.source_node_id === sourceNodeId,
  ) || null;
}

function interactiveTreeFrontierGroupPointers(
  payload,
  revisionId,
  sourceNodeIds,
) {
  if (
    !Array.isArray(sourceNodeIds)
    || sourceNodeIds.length < 2
    || sourceNodeIds.length > 50
    || new Set(sourceNodeIds).size !== sourceNodeIds.length
  ) {
    return [];
  }
  const source = interactiveTreeFrontierProjectionSources(payload).find(
    (item) => item?.detail?.revision_id === revisionId,
  );
  if (!source) return [];
  const byId = new Map(
    interactiveTreeFrontierEligiblePointers(source).map(
      (pointer) => [pointer.source_node_id, pointer],
    ),
  );
  const pointers = sourceNodeIds.map((sourceNodeId) => byId.get(sourceNodeId));
  return pointers.every(Boolean) ? pointers : [];
}

function syncInteractiveTreeFrontierMaterializationControls(
  form,
  payload,
  {
    requestedRevisionId = "",
    requestedSourceNodeId = "",
    preserveNode = true,
  } = {},
) {
  if (!form) return;
  const revisionSelect = formField(
    form,
    "interactive_tree_frontier_revision_id",
  );
  const nodeSelect = formField(
    form,
    "interactive_tree_frontier_source_node_id",
  );
  if (!revisionSelect || !nodeSelect) return;
  const revisions = interactiveTreeFrontierProjectionSources(payload);
  const previousRevision = nonEmptyText(revisionSelect.value);
  revisionSelect.innerHTML = [
    '<option value="">请选择不可变 revision</option>',
    ...revisions.map((item) => {
      const revisionId = nonEmptyText(item?.detail?.revision_id);
      const frontierCount = interactiveTreeFrontierEligiblePointers(item).length;
      return projectionOptionHtml(
        revisionId,
        `${revisionId} · ${frontierCount} 个可物化前沿节点`,
        {
          "candidate-lab-projection": "1",
          "revision-id": revisionId,
        },
      );
    }),
  ].join("");
  const preferredRevision = (
    nonEmptyText(requestedRevisionId) || previousRevision
  );
  if (selectContainsValue(revisionSelect, preferredRevision)) {
    revisionSelect.value = preferredRevision;
  } else if (revisions.length) {
    revisionSelect.value = nonEmptyText(revisions[0]?.detail?.revision_id);
  }

  const revisionId = nonEmptyText(revisionSelect.value);
  const selectedRevision = revisions.find(
    (item) => item?.detail?.revision_id === revisionId,
  );
  const pointers = selectedRevision
    ? interactiveTreeFrontierEligiblePointers(selectedRevision)
    : [];
  const previousNode = preserveNode ? nonEmptyText(nodeSelect.value) : "";
  nodeSelect.innerHTML = [
    '<option value="">请选择当前 revision 的 frontier 节点</option>',
    ...pointers.map((pointer) => {
      const sourceNodeId = nonEmptyText(pointer.source_node_id);
      const node = selectedRevision?.pointers?.nodes?.find?.(
        (item) => item?.node_id === sourceNodeId,
      );
      const label = node?.condition
        ? `${sourceNodeId} · ${readableValue(node.condition)}`
        : sourceNodeId;
      return projectionOptionHtml(
        sourceNodeId,
        label,
        {
          "candidate-lab-projection": "1",
          "revision-id": revisionId,
          "source-node-id": sourceNodeId,
        },
      );
    }),
  ].join("");
  const preferredNode = nonEmptyText(requestedSourceNodeId) || previousNode;
  if (selectContainsValue(nodeSelect, preferredNode)) {
    nodeSelect.value = preferredNode;
  } else if (pointers.length) {
    nodeSelect.value = nonEmptyText(pointers[0]?.source_node_id);
  }
  const help = form.querySelector?.(
    "[data-candidate-lab-interactive-tree-frontier-help]",
  );
  if (help) {
    help.textContent = revisions.length
      ? pointers.length
        ? "每次只物化一个明确前沿节点；入池必须在后续单独请求中完成。"
        : "该 revision 当前没有可物化的受认证 frontier 节点。"
      : "当前任务尚无不可变交互树 revision，请先完成一次明确剪枝。";
  }
}

function syncInteractiveTreeFrontierGroupMaterializationControls(
  form,
  payload,
  { preserveNodes = true } = {},
) {
  if (!form) return;
  const revisionSelect = formField(
    form,
    "interactive_tree_frontier_group_revision_id",
  );
  const nodeSelect = formField(
    form,
    "interactive_tree_frontier_group_source_node_ids",
  );
  if (!revisionSelect || !nodeSelect) return;
  const revisions = interactiveTreeFrontierProjectionSources(payload);
  const previousRevision = nonEmptyText(revisionSelect.value);
  revisionSelect.innerHTML = [
    '<option value="">请选择不可变 revision</option>',
    ...revisions.map((item) => {
      const revisionId = nonEmptyText(item?.detail?.revision_id);
      const frontierCount = interactiveTreeFrontierEligiblePointers(item).length;
      return projectionOptionHtml(
        revisionId,
        `${revisionId} · ${frontierCount} 个可分组前沿节点`,
        {
          "candidate-lab-projection": "1",
          "revision-id": revisionId,
        },
      );
    }),
  ].join("");
  if (
    previousRevision
    && selectContainsValue(revisionSelect, previousRevision)
  ) {
    revisionSelect.value = previousRevision;
  } else if (revisions.length) {
    revisionSelect.value = nonEmptyText(revisions[0]?.detail?.revision_id);
  }

  const revisionId = nonEmptyText(revisionSelect.value);
  const selectedRevision = revisions.find(
    (item) => item?.detail?.revision_id === revisionId,
  );
  const pointers = selectedRevision
    ? interactiveTreeFrontierEligiblePointers(selectedRevision)
    : [];
  const previousNodeIds = preserveNodes
    ? new Set(selectedValues(nodeSelect))
    : new Set();
  nodeSelect.innerHTML = pointers.map((pointer) => {
    const sourceNodeId = nonEmptyText(pointer.source_node_id);
    const node = selectedRevision?.pointers?.nodes?.find?.(
      (item) => item?.node_id === sourceNodeId,
    );
    const label = node?.condition
      ? `${sourceNodeId} · ${readableValue(node.condition)}`
      : sourceNodeId;
    return projectionOptionHtml(
      sourceNodeId,
      label,
      {
        "candidate-lab-projection": "1",
        "revision-id": revisionId,
        "source-node-id": sourceNodeId,
      },
    );
  }).join("");
  for (const option of Array.from(nodeSelect.options || [])) {
    option.selected = previousNodeIds.has(option.value);
  }
  const help = form.querySelector?.(
    "[data-candidate-lab-interactive-tree-frontier-group-help]",
  );
  if (help) {
    help.textContent = revisions.length
      ? pointers.length >= 2
        ? "按住 Command/Ctrl 选择 2–50 个节点；只创建 pointer-only OR 分组，入池需后续单独请求。"
        : "该 revision 当前不足 2 个受认证 frontier 节点，不能创建 OR 分组。"
      : "当前任务尚无不可变交互树 revision，请先完成一次明确剪枝。";
  }
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

function scorecardBandBuildForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="scorecard_band_build"]',
  ) || null;
}

function scorecardCutoffSelectionForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="scorecard_cutoff_selection"]',
  ) || null;
}

function setScorecardBandingPanelVisible(panel, visible) {
  if (!panel) return;
  panel.classList?.toggle?.("hidden", !visible);
  panel.setAttribute?.("aria-hidden", visible ? "false" : "true");
  const controls = panel.querySelectorAll?.("input, select, textarea, button") || [];
  for (const control of controls) control.disabled = !visible;
}

function syncScorecardBandingMode(form) {
  if (!form) return;
  const mode = formValue(form, "scorecard_banding_mode") || "equal_frequency";
  const panels = form.querySelectorAll?.(
    "[data-candidate-lab-scorecard-banding-panel]",
  ) || [];
  for (const modePanel of panels) {
    setScorecardBandingPanelVisible(
      modePanel,
      modePanel.dataset?.candidateLabScorecardBandingPanel === mode,
    );
  }
}

function scorecardBandProjectionCandidates(payload) {
  const collection = isRecord(payload?.candidates?.scorecard_band)
    ? payload.candidates.scorecard_band
    : {};
  const seen = new Set();
  return collectionItems(collection).filter((item) => {
    const assetId = nonEmptyText(item?.detail?.asset_id);
    if (
      !/^scorecard-band-asset-[0-9a-f]{32}$/.test(assetId)
      || seen.has(assetId)
    ) {
      return false;
    }
    seen.add(assetId);
    return true;
  });
}

function scorecardProjectionCutoffs(candidate) {
  const rows = Array.isArray(candidate?.pointers?.cutoffs)
    ? candidate.pointers.cutoffs.filter(isRecord)
    : [];
  const seen = new Set();
  return rows.filter((row) => {
    const cutoffId = nonEmptyText(row.cutoff_id);
    if (
      !/^scorecard-cutoff-[0-9a-f]{32}$/.test(cutoffId)
      || seen.has(cutoffId)
    ) {
      return false;
    }
    seen.add(cutoffId);
    return true;
  });
}

function syncScorecardCutoffControls(
  form,
  payload,
  { preserveCutoff = true } = {},
) {
  if (!form) return;
  const candidates = scorecardBandProjectionCandidates(payload);
  const assetSelect = formField(form, "scorecard_asset_id");
  const cutoffSelect = formField(form, "scorecard_cutoff_id");
  if (!assetSelect || !cutoffSelect) return;

  const previousAssetId = nonEmptyText(assetSelect.value);
  assetSelect.innerHTML = [
    '<option value="">请选择当前任务的评分卡分档</option>',
    ...candidates.map((candidate) => {
      const assetId = nonEmptyText(candidate?.detail?.asset_id);
      const cutoffCount = scorecardProjectionCutoffs(candidate).length;
      return projectionOptionHtml(
        assetId,
        `${assetId} · ${cutoffCount} 个可见 Cutoff`,
        { "candidate-lab-projection": "1" },
      );
    }),
  ].join("");
  if (selectContainsValue(assetSelect, previousAssetId)) {
    assetSelect.value = previousAssetId;
  } else {
    assetSelect.value = "";
  }

  const assetId = nonEmptyText(assetSelect.value);
  const candidate = candidates.find(
    (item) => nonEmptyText(item?.detail?.asset_id) === assetId,
  );
  const cutoffs = scorecardProjectionCutoffs(candidate);
  const previousCutoffId = nonEmptyText(cutoffSelect.value);
  const previousCutoffSource = nonEmptyText(
    Array.from(cutoffSelect.selectedOptions || [])[0]?.dataset?.sourceAssetId,
  );
  cutoffSelect.innerHTML = [
    '<option value="">请选择该分档中的可见 Cutoff</option>',
    ...cutoffs.map((cutoff) => {
      const cutoffId = nonEmptyText(cutoff.cutoff_id);
      const pd = stablePrimitiveText(cutoff.execution_pd);
      const points = stablePrimitiveText(cutoff.display_points);
      return projectionOptionHtml(
        cutoffId,
        `${cutoffId} · PD ${pd} · ${points} 分`,
        {
          "candidate-lab-projection": "1",
          "source-asset-id": assetId,
        },
      );
    }),
  ].join("");
  if (
    preserveCutoff
    && previousCutoffSource === assetId
    && selectContainsValue(cutoffSelect, previousCutoffId)
  ) {
    cutoffSelect.value = previousCutoffId;
  } else {
    cutoffSelect.value = "";
  }

  const empty = form.querySelector?.("[data-candidate-lab-scorecard-empty]");
  if (empty) {
    empty.textContent = candidates.length
      ? assetId
        ? cutoffs.length
          ? "请明确选择一个 Cutoff；下拉仅包含当前任务受认证投影中的可见项。"
          : "该评分卡分档没有可选择的可见 Cutoff。"
        : "请先明确选择一个当前任务的评分卡分档资产。"
      : "当前任务尚无评分卡分档证据，请先生成分档。";
  }
}

function syncScorecardForms(root, payload) {
  syncScorecardBandingMode(scorecardBandBuildForm(root));
  syncScorecardCutoffControls(
    scorecardCutoffSelectionForm(root),
    payload,
  );
}

function candidateStabilityForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="candidate_monthly_stability"]',
  ) || null;
}

function setCandidateStabilityPanelVisible(panel, visible) {
  if (!panel) return;
  panel.classList?.toggle?.("hidden", !visible);
  panel.setAttribute?.("aria-hidden", visible ? "false" : "true");
  const controls = panel.querySelectorAll?.("input, select, textarea, button") || [];
  for (const control of controls) control.disabled = !visible;
}

function candidateStabilityPoolOptions(payload) {
  const pools = collectionItems(isRecord(payload?.pools) ? payload.pools : {});
  const result = [];
  for (const pool of pools) {
    const strategyType = nonEmptyText(pool?.strategy_type);
    if (!["approval", "reject", "limit", "pricing", "segmentation"].includes(
      strategyType,
    )) {
      continue;
    }
    for (const entry of Array.isArray(pool?.entries) ? pool.entries : []) {
      const entryId = nonEmptyText(entry?.entry_id);
      if (!/^pool-entry-[0-9a-f]{32}$/.test(entryId)) continue;
      result.push({
        entryId,
        strategyType,
        ruleId: nonEmptyText(entry.rule_id),
        assetId: nonEmptyText(entry?.source?.asset_id),
        assetType: nonEmptyText(entry?.source?.asset_type),
      });
    }
  }
  return result;
}

function syncCandidateStabilityControls(form, payload) {
  if (!form) return;
  const mode = formValue(form, "stability_source_mode") || "pool_entry";
  const panels = form.querySelectorAll?.(
    "[data-candidate-lab-stability-panel]",
  ) || [];
  for (const modePanel of panels) {
    setCandidateStabilityPanelVisible(
      modePanel,
      modePanel.dataset?.candidateLabStabilityPanel === mode,
    );
  }

  const entries = candidateStabilityPoolOptions(payload);
  const entrySelect = formField(form, "stability_pool_entry");
  if (entrySelect) {
    const previousEntry = nonEmptyText(entrySelect.value);
    entrySelect.innerHTML = [
      '<option value="">请选择当前 Pool 条目</option>',
      ...entries.map((entry) => projectionOptionHtml(
        entry.entryId,
        `${entry.strategyType} · ${entry.ruleId || entry.entryId} · ${entry.assetType || "candidate"}`,
        {
          "candidate-lab-projection": "1",
          "strategy-type": entry.strategyType,
        },
      )),
    ].join("");
    entrySelect.value = selectContainsValue(entrySelect, previousEntry)
      ? previousEntry
      : "";
  }

  const assets = [];
  const seenAssets = new Set();
  for (const entry of entries) {
    if (
      entry.assetType !== "univariate_refinement"
      || !/^candidate-asset-[0-9a-f]{32}$/.test(entry.assetId)
      || seenAssets.has(entry.assetId)
    ) {
      continue;
    }
    seenAssets.add(entry.assetId);
    assets.push(entry);
  }
  const assetSelect = formField(form, "stability_asset_id");
  if (assetSelect) {
    const previousAsset = nonEmptyText(assetSelect.value);
    assetSelect.innerHTML = [
      '<option value="">请选择 Pool 中可见的单变量候选资产</option>',
      ...assets.map((entry) => projectionOptionHtml(
        entry.assetId,
        `${entry.assetId} · ${entry.strategyType}`,
        { "candidate-lab-projection": "1" },
      )),
    ].join("");
    assetSelect.value = selectContainsValue(assetSelect, previousAsset)
      ? previousAsset
      : "";
  }

  const empty = form.querySelector?.("[data-candidate-lab-stability-empty]");
  if (empty) {
    empty.textContent = entries.length
      ? "必须明确选择当前受认证投影中的来源；平台会绑定完整 Pool revision、样本和月份口径。"
      : "当前任务尚无可测算的 Strategy Pool 条目。";
  }
}

function strategyPoolApplyForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_apply"]',
  ) || null;
}

function strategyPoolAddForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_add_candidate"]',
  ) || null;
}

function minimalProjectedPoolAction(value, strategyType) {
  if (!isRecord(value)) return null;
  const actionType = nonEmptyText(value.type);
  if (!(STRATEGY_POOL_ACTION_TYPES[strategyType] || []).includes(actionType)) {
    return null;
  }
  if (["approval", "reject", "review"].includes(actionType)) {
    return { type: actionType };
  }
  const actionValue = value.value;
  if (actionType === "segment") {
    if (
      !["string", "number"].includes(typeof actionValue)
      || (typeof actionValue === "string" && !actionValue.trim())
      || (typeof actionValue === "number" && !Number.isFinite(actionValue))
    ) {
      return null;
    }
    return { type: actionType, value: actionValue };
  }
  if (
    typeof actionValue !== "number"
    || !Number.isFinite(actionValue)
    || (actionType === "limit" && actionValue < 0)
    || (
      actionType === "pricing"
      && (actionValue < 0 || actionValue > 1)
    )
  ) {
    return null;
  }
  return { type: actionType, value: actionValue };
}

function strategyPoolAddCurrentPools(payload) {
  const operationPools = strategyPoolOperationPools(payload);
  const rawPools = collectionItems(
    isRecord(payload?.pools) ? payload.pools : {},
  );
  return operationPools.flatMap((pool) => {
    const matches = rawPools.filter((item) => (
      item?.kind === "candidate_pool"
      && nonEmptyText(item?.strategy_type) === pool.strategyType
    ));
    if (matches.length !== 1) return [];
    const defaultAction = minimalProjectedPoolAction(
      matches[0].default_action,
      pool.strategyType,
    );
    if (!defaultAction) return [];
    return [{ ...pool, defaultAction }];
  });
}

function strategyPoolAddSourcePointer(source) {
  if (!isRecord(source)) return null;
  const sourceKind = nonEmptyText(source.source_kind);
  const pointerKind = STRATEGY_POOL_ADD_SOURCE_KINDS[sourceKind];
  if (!pointerKind) return null;
  const keys = Object.keys(source).sort();
  const expectedKeys = [
    "candidate_stage",
    pointerKind,
    "source_kind",
    "strategy_type",
    "validation_status",
  ].sort();
  if (
    keys.length !== expectedKeys.length
    || keys.some((key, index) => key !== expectedKeys[index])
  ) {
    return null;
  }
  const sourceId = nonEmptyText(source[pointerKind]);
  const selectionPrefix = {
    automatic_tree_leaf_selection: "automatic-tree-leaf-selection-",
    interactive_tree_frontier_selection:
      "interactive-tree-frontier-selection-",
    interactive_tree_frontier_group_selection:
      "interactive-tree-frontier-group-selection-",
    cross_matrix_cell_selection: "cross-matrix-cell-selection-",
    scorecard_cutoff_selection: "scorecard-cutoff-selection-",
  }[sourceKind];
  if (
    pointerKind === "candidate_asset_id"
      ? !STRATEGY_POOL_CANDIDATE_ASSET_ID_RE.test(sourceId)
      : (
        !STRATEGY_POOL_ADD_SELECTION_RE.test(sourceId)
        || !sourceId.startsWith(selectionPrefix)
      )
  ) {
    return null;
  }
  const strategyType = source.strategy_type === null
    ? ""
    : nonEmptyText(source.strategy_type);
  if (
    sourceKind === "voting_candidate"
      ? !STRATEGY_POOL_TYPES.includes(strategyType)
      : strategyType
  ) {
    return null;
  }
  if (
    !nonEmptyText(source.candidate_stage)
    || !nonEmptyText(source.validation_status)
  ) {
    return null;
  }
  return {
    sourceKind,
    pointerKind,
    sourceId,
    strategyType,
    candidateStage: nonEmptyText(source.candidate_stage),
    validationStatus: nonEmptyText(source.validation_status),
  };
}

function strategyPoolAddSources(payload) {
  const collection = isRecord(payload?.pool_add_sources)
    ? payload.pool_add_sources
    : null;
  if (
    !collection
    || !Array.isArray(collection.all)
    || !Number.isInteger(collection.total)
    || collection.total < collection.all.length
    || collection.all.length > _MAX_STRATEGY_POOL_ADD_SOURCES
    || collection.truncated !== (collection.total > collection.all.length)
  ) {
    return [];
  }
  const projected = collection.all.map(strategyPoolAddSourcePointer);
  if (projected.some((source) => !source)) return [];
  const identities = projected.map((source) => (
    `${source.pointerKind}\u001f${source.sourceId}`
  ));
  if (new Set(identities).size !== identities.length) return [];
  const latest = collection.latest === null
    ? null
    : strategyPoolAddSourcePointer(collection.latest);
  if (
    projected.length
      ? (
        !latest
        || latest.pointerKind !== projected[0].pointerKind
        || latest.sourceId !== projected[0].sourceId
      )
      : collection.latest !== null
  ) {
    return [];
  }
  return projected;
}

function setStrategyPoolAddPanelVisible(form, selector, visible) {
  const panel = form?.querySelector?.(selector);
  if (!panel) return;
  panel.classList?.toggle?.("hidden", !visible);
  panel.setAttribute?.("aria-hidden", visible ? "false" : "true");
  const controls = panel.querySelectorAll?.("input, select, textarea") || [];
  for (const control of controls) control.disabled = !visible;
}

function setStrategyPoolAddLocked(control, locked) {
  if (!control) return;
  if (!control.dataset) control.dataset = {};
  if (locked) {
    control.dataset.candidateLabPoolAddLocked = "1";
  } else {
    delete control.dataset.candidateLabPoolAddLocked;
  }
}

function syncStrategyPoolAddAction(
  form,
  strategyType,
  {
    typeField,
    valueField,
    valuePanel,
    projectedAction = null,
    locked = false,
  },
) {
  const typeSelect = formField(form, typeField);
  const valueInput = formField(form, valueField);
  if (!typeSelect || !valueInput) return;
  const allowed = STRATEGY_POOL_ACTION_TYPES[strategyType] || [];
  const previousType = nonEmptyText(typeSelect.value);
  const previousValue = String(valueInput.value || "");
  typeSelect.innerHTML = [
    '<option value="">请选择兼容动作</option>',
    ...allowed.map((actionType) => projectionOptionHtml(
      actionType,
      actionType,
      { "candidate-lab-action-type": "1" },
    )),
  ].join("");
  const action = minimalProjectedPoolAction(projectedAction, strategyType);
  if (action) {
    typeSelect.value = action.type;
    valueInput.value = Object.hasOwn(action, "value")
      ? String(action.value)
      : "";
  } else {
    typeSelect.value = allowed.includes(previousType)
      ? previousType
      : allowed.length === 1
        ? allowed[0]
        : "";
    valueInput.value = previousValue;
  }
  if (locked && action) {
    typeSelect.dataset.candidateLabPoolAddTypedAction = JSON.stringify(action);
  } else {
    delete typeSelect.dataset.candidateLabPoolAddTypedAction;
  }
  const requiresValue = ["limit", "pricing", "segment"].includes(
    nonEmptyText(typeSelect.value),
  );
  setStrategyPoolAddPanelVisible(form, valuePanel, requiresValue);
  setStrategyPoolAddLocked(typeSelect, locked);
  setStrategyPoolAddLocked(valueInput, locked && requiresValue);
}

function syncStrategyPoolAddPlacement(form) {
  if (!form) return;
  const selected = Array.from(
    formField(form, "pool_add_source_id")?.selectedOptions || [],
  )[0];
  const voting = (
    selected?.dataset?.candidateLabProjection === "1"
    && selected?.dataset?.sourceKind === "voting_candidate"
  );
  if (!voting) {
    const placement = formField(form, "pool_add_placement_mode");
    if (placement) placement.value = "";
  }
  setStrategyPoolAddPanelVisible(
    form,
    "[data-candidate-lab-pool-add-placement-panel]",
    voting,
  );
}

function syncStrategyPoolAddControls(
  form,
  payload,
  { preserveSource = true } = {},
) {
  if (!form) return;
  const typeSelect = formField(form, "pool_add_strategy_type");
  const sourceSelect = formField(form, "pool_add_source_id");
  if (!typeSelect || !sourceSelect) return;
  const previousType = nonEmptyText(typeSelect.value);
  typeSelect.innerHTML = [
    '<option value="">请选择 Strategy Pool 类型</option>',
    ...STRATEGY_POOL_TYPES.map((strategyType) => projectionOptionHtml(
      strategyType,
      strategyType,
      { "candidate-lab-strategy-type": "1" },
    )),
  ].join("");
  typeSelect.value = STRATEGY_POOL_TYPES.includes(previousType)
    ? previousType
    : "";
  const strategyType = nonEmptyText(typeSelect.value);
  const currentPools = strategyPoolAddCurrentPools(payload);
  const currentPool = currentPools.find(
    (pool) => pool.strategyType === strategyType,
  );
  const eligible = strategyPoolAddSources(payload).filter((source) => (
    !source.strategyType
    || (
      source.strategyType === strategyType
      && currentPool
    )
  ));
  const previousSource = preserveSource
    ? nonEmptyText(sourceSelect.value)
    : "";
  sourceSelect.innerHTML = [
    '<option value="">请选择受认证、已物化候选</option>',
    ...eligible.map((source) => projectionOptionHtml(
      source.sourceId,
      `${source.sourceKind} · ${source.sourceId} · ${source.candidateStage}/${source.validationStatus}`,
      {
        "candidate-lab-projection": "1",
        "source-kind": source.sourceKind,
        "source-id": source.sourceId,
        "pointer-kind": source.pointerKind,
        "strategy-type": source.strategyType,
      },
    )),
  ].join("");
  if (selectContainsValue(sourceSelect, previousSource)) {
    sourceSelect.value = previousSource;
  } else if (eligible.length === 1) {
    sourceSelect.value = eligible[0].sourceId;
  } else {
    sourceSelect.value = "";
  }
  syncStrategyPoolAddAction(
    form,
    strategyType,
    {
      typeField: "pool_add_default_action_type",
      valueField: "pool_add_default_action_value",
      valuePanel: "[data-candidate-lab-pool-add-default-value-panel]",
      projectedAction: currentPool?.defaultAction || null,
      locked: Boolean(currentPool),
    },
  );
  syncStrategyPoolAddAction(
    form,
    strategyType,
    {
      typeField: "pool_add_action_type",
      valueField: "pool_add_action_value",
      valuePanel: "[data-candidate-lab-pool-add-action-value-panel]",
    },
  );
  syncStrategyPoolAddPlacement(form);
  const help = form.querySelector?.("[data-candidate-lab-pool-add-help]");
  if (help) {
    help.textContent = !strategyType
      ? "请先选择 Pool 类型；页面只允许选择受认证、已物化且属于当前任务的候选。"
      : !eligible.length
        ? "当前类型没有可入池的受认证已物化候选；请先完成候选选择或 Voting 构建。"
        : currentPool
          ? "当前 Pool 默认动作已从完整受认证投影恢复并锁定；本操作不会修改默认动作。"
          : "该类型当前没有 Pool；请明确设置首条入池的默认动作和命中动作。";
  }
}

function strategyPoolAddRequestIsCurrent(request, payload) {
  if (request?.workflow !== "strategy_pool_add_candidate") return true;
  const inputs = isRecord(request.workflow_inputs)
    ? request.workflow_inputs
    : {};
  const strategyType = nonEmptyText(inputs.strategy_type);
  if (!STRATEGY_POOL_TYPES.includes(strategyType)) return false;
  const pointerKind = Object.hasOwn(inputs, "candidate_asset_id")
    ? "candidate_asset_id"
    : Object.hasOwn(inputs, "selection_id")
      ? "selection_id"
      : "";
  const sourceId = nonEmptyText(inputs[pointerKind]);
  const source = strategyPoolAddSources(payload).find((item) => (
    item.pointerKind === pointerKind
    && item.sourceId === sourceId
    && (!item.strategyType || item.strategyType === strategyType)
  ));
  if (!source) return false;
  const rawMatches = collectionItems(
    isRecord(payload?.pools) ? payload.pools : {},
  ).filter((pool) => nonEmptyText(pool?.strategy_type) === strategyType);
  const currentPool = strategyPoolAddCurrentPools(payload).find(
    (pool) => pool.strategyType === strategyType,
  );
  if (rawMatches.length > 0 && (rawMatches.length !== 1 || !currentPool)) {
    return false;
  }
  const defaultAction = minimalProjectedPoolAction(
    inputs.default_action,
    strategyType,
  );
  if (!defaultAction) return false;
  if (
    currentPool
    && JSON.stringify(defaultAction) !== JSON.stringify(
      currentPool.defaultAction,
    )
  ) {
    return false;
  }
  if (source.sourceKind === "voting_candidate") {
    return Boolean(
      currentPool
      && STRATEGY_POOL_VOTING_PLACEMENTS.includes(inputs.placement_mode),
    );
  }
  return !Object.hasOwn(inputs, "placement_mode");
}

function strategyPoolApplyOptions(payload) {
  const pools = collectionItems(isRecord(payload?.pools) ? payload.pools : {});
  const byType = new Map();
  const duplicates = new Set();
  for (const pool of pools) {
    const strategyType = nonEmptyText(pool?.strategy_type);
    const entries = Array.isArray(pool?.entries)
      ? pool.entries.filter(isRecord)
      : [];
    if (
      pool?.kind !== "candidate_pool"
      || !STRATEGY_POOL_TYPES.includes(strategyType)
      || entries.length < 1
      || !Number.isInteger(pool?.total)
      || pool.total < entries.length
    ) {
      continue;
    }
    if (byType.has(strategyType)) {
      duplicates.add(strategyType);
      continue;
    }
    byType.set(strategyType, {
      strategyType,
      entryCount: entries.length,
    });
  }
  for (const strategyType of duplicates) byType.delete(strategyType);
  return STRATEGY_POOL_TYPES
    .filter((strategyType) => byType.has(strategyType))
    .map((strategyType) => byType.get(strategyType));
}

function syncStrategyPoolApplyControls(form, payload) {
  if (!form) return;
  const strategySelect = formField(form, "pool_apply_strategy_type");
  if (!strategySelect) return;
  const pools = strategyPoolApplyOptions(payload);
  const previousType = nonEmptyText(strategySelect.value);
  strategySelect.innerHTML = [
    '<option value="">请选择当前非空 Strategy Pool</option>',
    ...pools.map((pool) => projectionOptionHtml(
      pool.strategyType,
      `${pool.strategyType} · ${pool.entryCount} 条当前规则`,
      {
        "candidate-lab-projection": "1",
        "strategy-type": pool.strategyType,
      },
    )),
  ].join("");
  if (previousType && selectContainsValue(strategySelect, previousType)) {
    strategySelect.value = previousType;
  } else if (pools.length === 1) {
    strategySelect.value = pools[0].strategyType;
  } else {
    strategySelect.value = "";
  }
  const help = form.querySelector?.("[data-candidate-lab-pool-apply-empty]");
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可应用的受认证非空 Strategy Pool。"
      : pools.length === 1
        ? "已选择当前唯一的受认证非空 Strategy Pool；可直接应用或填写输出列前缀。"
        : "当前存在多个受认证非空 Strategy Pool，请明确选择要应用的策略类型。";
  }
}

function strategyPoolCompileForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_compile"]',
  ) || null;
}

function strategyProjectContextForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_project_context"]',
  ) || null;
}

function strategyPoolMaterializeForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_materialize"]',
  ) || null;
}

function strategyLifecycleAdoptionForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_lifecycle_adopt"]',
  ) || null;
}

function strategyDslDeliveryForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_dsl_delivery"]',
  ) || null;
}

function strategyPoolValidationForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_validation"]',
  ) || null;
}

function strategyPoolStabilityForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_stability"]',
  ) || null;
}

function strategyPoolImpactForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_impact"]',
  ) || null;
}

function strategyImpactCubeForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_impact_cube"]',
  ) || null;
}

function strategyPoolRemoveEntryForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_remove_entry"]',
  ) || null;
}

function strategyPoolSetActionForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_set_action"]',
  ) || null;
}

function strategyPoolReorderForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="strategy_pool_reorder"]',
  ) || null;
}

function strategyPoolOperationPools(payload) {
  const pools = collectionItems(isRecord(payload?.pools) ? payload.pools : {});
  const byType = new Map();
  const duplicates = new Set();
  for (const pool of pools) {
    const strategyType = nonEmptyText(pool?.strategy_type);
    const entries = Array.isArray(pool?.entries)
      ? pool.entries.filter(isRecord)
      : [];
    const entryIds = entries.map((entry) => nonEmptyText(entry.entry_id));
    const completeEntries = (
      entries.length > 0
      && entries.length <= 200
      && pool?.total === entries.length
      && pool?.truncated !== true
      && new Set(entryIds).size === entryIds.length
      && entries.every((entry, index) => (
        STRATEGY_POOL_ENTRY_ID_RE.test(nonEmptyText(entry.entry_id))
        && entry.position === index
        && isRecord(entry.action)
        && nonEmptyText(entry.action.type)
      ))
    );
    if (
      pool?.kind !== "candidate_pool"
      || !STRATEGY_POOL_TYPES.includes(strategyType)
      || !completeEntries
    ) {
      continue;
    }
    if (byType.has(strategyType)) {
      duplicates.add(strategyType);
      continue;
    }
    byType.set(strategyType, {
      strategyType,
      poolId: nonEmptyText(pool.pool_id),
      entries: entries.map((entry) => ({
        entryId: nonEmptyText(entry.entry_id),
        position: entry.position,
        action: { ...entry.action },
      })),
    });
  }
  for (const strategyType of duplicates) byType.delete(strategyType);
  return STRATEGY_POOL_TYPES
    .filter((strategyType) => byType.has(strategyType))
    .map((strategyType) => byType.get(strategyType));
}

function syncStrategyPoolTypeSelect(select, pools) {
  if (!select) return "";
  const previousType = nonEmptyText(select.value);
  select.innerHTML = [
    '<option value="">请选择当前非空 Strategy Pool</option>',
    ...pools.map((pool) => projectionOptionHtml(
      pool.strategyType,
      `${pool.strategyType} · ${pool.entries.length} 条当前规则`,
      {
        "candidate-lab-projection": "1",
        "strategy-type": pool.strategyType,
        "pool-id": pool.poolId,
      },
    )),
  ].join("");
  if (previousType && selectContainsValue(select, previousType)) {
    select.value = previousType;
  } else if (pools.length === 1) {
    select.value = pools[0].strategyType;
  } else {
    select.value = "";
  }
  return nonEmptyText(select.value);
}

function syncStrategyProjectContextControls(form) {
  if (!form) return;
  const asOf = formField(form, "project_context_as_of");
  if (!asOf || nonEmptyText(asOf.value)) return;
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  asOf.value = `${year}-${month}-${day}`;
}

function syncStrategyPoolMaterializeControls(form, payload) {
  if (!form) return;
  const pools = strategyPoolOperationPools(payload);
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_materialize_strategy_type"),
    pools,
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-pool-materialize-help]",
  );
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可物化的受认证非空 Strategy Pool。"
      : pools.length === 1
        ? `已选择当前唯一的 ${selectedType} Pool；物化后仍是 draft 策略。`
        : "当前存在多个受认证非空 Pool，请明确选择要物化的策略类型。";
  }
}

function projectedStrategyItems(payload) {
  const collection = isRecord(payload?.strategies) ? payload.strategies : {};
  const items = Array.isArray(collection.all)
    ? collection.all.filter(isRecord)
    : [];
  const seen = new Set();
  const unique = [];
  for (const strategy of items) {
    const strategyId = nonEmptyText(strategy.strategy_id);
    if (!strategyId || seen.has(strategyId)) continue;
    seen.add(strategyId);
    unique.push(strategy);
  }
  const latest = isRecord(collection.latest) ? collection.latest : null;
  const latestId = nonEmptyText(latest?.strategy_id);
  if (latestId && !seen.has(latestId)) unique.unshift(latest);
  return unique;
}

function strategyDeliveryOptions(payload) {
  return projectedStrategyItems(payload).filter((strategy) => (
    STRATEGY_ID_RE.test(nonEmptyText(strategy?.strategy_id))
    && STRATEGY_POOL_TYPES.includes(nonEmptyText(strategy?.strategy_type))
    && ["draft", "validated", "adopted_local"].includes(
      nonEmptyText(strategy?.asset_status),
    )
  )).map((strategy) => ({
    strategyId: nonEmptyText(strategy.strategy_id),
    strategyType: nonEmptyText(strategy.strategy_type),
    version: strategy.version,
    assetStatus: nonEmptyText(strategy.asset_status),
  }));
}

function syncStrategyDslDeliveryControls(form, payload) {
  if (!form) return;
  const select = formField(form, "dsl_delivery_strategy_id");
  if (!select) return;
  const strategies = strategyDeliveryOptions(payload);
  const previous = nonEmptyText(select.value);
  select.innerHTML = [
    '<option value="">请选择当前任务策略版本</option>',
    ...strategies.map((strategy) => projectionOptionHtml(
      strategy.strategyId,
      `${strategy.strategyType} · v${stablePrimitiveText(strategy.version)} · ${strategy.assetStatus}`,
      {
        "candidate-lab-projection": "1",
        "strategy-id": strategy.strategyId,
      },
    )),
  ].join("");
  if (previous && selectContainsValue(select, previous)) {
    select.value = previous;
  } else if (strategies.length === 1) {
    select.value = strategies[0].strategyId;
  } else {
    select.value = "";
  }
  const help = form.querySelector?.(
    "[data-candidate-lab-dsl-delivery-help]",
  );
  if (help) {
    help.textContent = strategies.length === 0
      ? "当前任务尚无可交付的 canonical Strategy；请先物化当前 Pool。"
      : strategies.length === 1
        ? "已选择当前唯一策略；平台会重新绑定活动数据并执行逐行等价验证。"
        : "当前有多个可交付策略版本，请明确选择完整版本。";
  }
}

function strategyLifecycleAdoptionOptions(payload) {
  return projectedStrategyItems(payload).filter((strategy) => (
    STRATEGY_ID_RE.test(nonEmptyText(strategy?.strategy_id))
    && STRATEGY_POOL_TYPES.includes(nonEmptyText(strategy?.strategy_type))
    && nonEmptyText(strategy?.asset_status) === "draft"
  )).map((strategy) => ({
    strategyId: nonEmptyText(strategy.strategy_id),
    strategyType: nonEmptyText(strategy.strategy_type),
    version: strategy.version,
    assetStatus: "draft",
    runtimeBlockers: Array.isArray(strategy?.materialization?.runtime_blockers)
      ? strategy.materialization.runtime_blockers.length
      : 0,
  }));
}

function strategyProjectionAvailableColumns(payload) {
  const workflow = isRecord(payload?.workflow) ? payload.workflow : {};
  const sample = isRecord(workflow.sample_design)
    ? workflow.sample_design
    : {};
  const sources = [
    payload?.available_columns,
    payload?.task?.available_columns,
    workflow.available_columns,
    sample.available_columns,
    sample.column_whitelist,
    sample.dataset?.available_columns,
  ];
  const columns = [];
  const seen = new Set();
  for (const source of sources) {
    if (!Array.isArray(source)) continue;
    for (const item of source) {
      const column = nonEmptyText(
        typeof item === "string" ? item : item?.name || item?.column,
      );
      if (!column || seen.has(column)) continue;
      seen.add(column);
      columns.push(column);
      if (columns.length >= 500) return columns;
    }
  }
  return columns;
}

function syncStrategyLifecycleAdoptionEconomics(form, strategyType) {
  const economics = form?.querySelector?.(
    "[data-candidate-lab-adoption-economics]",
  );
  const economicType = ["limit", "pricing"].includes(strategyType);
  economics?.classList?.toggle?.("hidden", !economicType);
  const components = form?.querySelectorAll?.(
    "[data-candidate-lab-adoption-component]",
  ) || [];
  for (const component of components) {
    const name = nonEmptyText(component.dataset?.candidateLabAdoptionComponent);
    const allowedTypes = nonEmptyText(component.dataset?.strategyTypes)
      .split(/\s+/)
      .filter(Boolean);
    const visible = economicType && allowedTypes.includes(strategyType);
    component.classList?.toggle?.("hidden", !visible);
    const mode = formField(form, `lifecycle_adopt_${name}_mode`);
    if (visible && !["column", "value"].includes(nonEmptyText(mode?.value))) {
      mode.value = "column";
    }
    const selectedMode = nonEmptyText(mode?.value) || "column";
    const bindings = component.querySelectorAll?.(
      "[data-candidate-lab-adoption-binding]",
    ) || [];
    for (const binding of bindings) {
      binding.classList?.toggle?.(
        "hidden",
        nonEmptyText(binding.dataset?.candidateLabAdoptionBinding)
          !== selectedMode,
      );
    }
  }
}

function syncStrategyLifecycleAdoptionControls(form, payload) {
  if (!form) return;
  const select = formField(form, "lifecycle_adopt_strategy_id");
  if (!select) return;
  const strategies = strategyLifecycleAdoptionOptions(payload);
  const previous = nonEmptyText(select.value);
  select.innerHTML = [
    '<option value="">请选择当前 draft 草稿策略</option>',
    ...strategies.map((strategy) => projectionOptionHtml(
      strategy.strategyId,
      `${STRATEGY_TYPE_LABELS[strategy.strategyType] || strategy.strategyType} · v${stablePrimitiveText(strategy.version)} · draft${strategy.runtimeBlockers ? ` · ${strategy.runtimeBlockers} 项运行阻塞` : ""}`,
      {
        "candidate-lab-projection": "1",
        "strategy-id": strategy.strategyId,
        "strategy-type": strategy.strategyType,
        "asset-status": strategy.assetStatus,
      },
    )),
  ].join("");
  if (previous && selectContainsValue(select, previous)) {
    select.value = previous;
  } else if (strategies.length === 1) {
    select.value = strategies[0].strategyId;
  } else {
    select.value = "";
  }
  const selected = Array.from(select.selectedOptions || [])[0] || null;
  const strategyType = nonEmptyText(selected?.dataset?.strategyType);
  syncStrategyLifecycleAdoptionEconomics(form, strategyType);

  const help = form.querySelector?.("[data-candidate-lab-adoption-help]");
  if (help) {
    help.textContent = strategies.length === 0
      ? "当前任务尚无可提交采纳的 draft 策略；请先物化完整 Strategy Pool。"
      : strategies.length === 1
        ? "已选择当前唯一 draft 策略；提交后平台会重新回测并等待人工确认。"
        : "当前有多个 draft 策略版本，请明确选择要重新回测并提交人工确认的版本。";
  }

  const columns = strategyProjectionAvailableColumns(payload);
  const datalist = form.querySelector?.(
    "[data-candidate-lab-adoption-available-columns]",
  );
  if (datalist) {
    datalist.innerHTML = columns.map(
      (column) => `<option value="${escapeHtml(column)}"></option>`,
    ).join("");
  }
  const columnsHelp = form.querySelector?.(
    "[data-candidate-lab-adoption-columns-help]",
  );
  if (columnsHelp) {
    columnsHelp.textContent = columns.length
      ? `当前投影提供 ${columns.length} 个可用列建议；也可输入列名，平台仍会按任务列白名单核验。`
      : "当前投影未提供列建议；可以输入列名，平台会按任务列白名单核验，不存在或不可用的列不会通过。";
  }
}

function strategyWorkbenchRequestIsCurrent(request, payload) {
  if (request?.request_kind === "strategy_lifecycle") {
    return strategyLifecycleAdoptionOptions(payload).some((strategy) => (
      strategy.strategyId === request.strategy_id
      && strategy.strategyType === request.strategy_type
      && strategy.assetStatus === "draft"
    ));
  }
  if (request?.workflow === "strategy_pool_materialize") {
    return strategyPoolOperationPools(payload).some(
      (pool) => pool.strategyType === request.workflow_inputs?.strategy_type,
    );
  }
  if (request?.workflow === "strategy_dsl_delivery") {
    return strategyDeliveryOptions(payload).some(
      (strategy) => (
        strategy.strategyId === request.workflow_inputs?.strategy_id
      ),
    );
  }
  return true;
}

function syncStrategyPoolValidationControls(form, payload) {
  if (!form) return;
  const pools = strategyPoolOperationPools(payload);
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_validation_strategy_type"),
    pools,
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-pool-validation-help]",
  );
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可验证的受认证非空 Strategy Pool。"
      : pools.length === 1
        ? `已选择当前唯一的 ${selectedType} Pool；请选择 validation 或 OOT 分区。`
        : "当前存在多个可验证的非空 Pool，请明确选择策略类型。";
  }
}

function syncStrategyPoolStabilityControls(form, payload) {
  if (!form) return;
  const pools = strategyPoolOperationPools(payload);
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_stability_strategy_type"),
    pools,
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-pool-stability-help]",
  );
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可测算的受认证非空 Strategy Pool。"
      : pools.length === 1
        ? `已选择当前唯一的 ${selectedType} Pool；平台将自动比较 development 与所有可用 validation / OOT。`
        : "当前存在多个受认证非空 Strategy Pool，请明确选择要测算稳定性的策略类型。";
  }
}

function syncStrategyPoolImpactControls(form, payload) {
  if (!form) return;
  const pools = strategyPoolOperationPools(payload).filter((pool) => (
    ["approval", "reject"].includes(pool.strategyType)
  ));
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_impact_strategy_type"),
    pools,
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-pool-impact-help]",
  );
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可测算的受认证非空 approval / reject Pool。"
      : pools.length === 1
        ? `已选择当前唯一的 ${selectedType} Pool；不会修改 Pool 或创建策略。`
        : "当前存在多个可测算 Pool，请明确选择 approval 或 reject。";
  }
}

function syncStrategyImpactCubeControls(form, payload) {
  if (!form) return;
  const pools = strategyPoolOperationPools(payload);
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "impact_cube_strategy_type"),
    pools,
  );
  const help = form.querySelector?.(
    "[data-candidate-lab-impact-cube-help]",
  );
  if (help) {
    help.textContent = pools.length === 0
      ? "当前任务尚无可测算的受认证非空 Strategy Pool。"
      : pools.length === 1
        ? `已选择当前唯一的 ${selectedType} Pool；分区留空时由平台选择全部非空可用分区。`
        : "当前存在多个受认证非空 Pool，请明确选择要测算的策略类型。";
  }
}

function strategyMeasurementRequestIsCurrent(request, payload) {
  if (
    !["strategy_pool_impact", "strategy_impact_cube"].includes(
      request?.workflow,
    )
  ) {
    return true;
  }
  const strategyType = nonEmptyText(
    request?.workflow_inputs?.strategy_type,
  );
  const pools = strategyPoolOperationPools(payload);
  return pools.some((pool) => pool.strategyType === strategyType)
    && (
      request.workflow !== "strategy_pool_impact"
      || ["approval", "reject"].includes(strategyType)
    );
}

function strategyPoolEntryLabel(entry, displayPosition = entry.position) {
  const actionType = nonEmptyText(entry?.action?.type) || "unknown";
  return `#${displayPosition + 1} · ${entry.entryId} · ${actionType}`;
}

function syncStrategyPoolEntrySelect(
  select,
  pool,
  { preserveEntry = true } = {},
) {
  if (!select) return;
  const previousEntryId = nonEmptyText(select.value);
  const previousType = nonEmptyText(
    Array.from(select.selectedOptions || [])[0]?.dataset?.strategyType,
  );
  const entries = pool?.entries || [];
  select.innerHTML = [
    '<option value="">请选择当前 Pool Entry</option>',
    ...entries.map((entry) => projectionOptionHtml(
      entry.entryId,
      strategyPoolEntryLabel(entry),
      {
        "candidate-lab-projection": "1",
        "strategy-type": pool.strategyType,
        "entry-id": entry.entryId,
      },
    )),
  ].join("");
  if (
    preserveEntry
    && previousType === pool?.strategyType
    && selectContainsValue(select, previousEntryId)
  ) {
    select.value = previousEntryId;
  } else {
    select.value = "";
  }
}

function setStrategyPoolActionValuePanelVisible(form, visible) {
  const panel = form?.querySelector?.(
    "[data-candidate-lab-pool-action-value-panel]",
  );
  if (!panel) return;
  panel.classList?.toggle?.("hidden", !visible);
  panel.setAttribute?.("aria-hidden", visible ? "false" : "true");
  const controls = panel.querySelectorAll?.("input, select, textarea") || [];
  for (const control of controls) control.disabled = !visible;
}

function syncStrategyPoolActionTypes(
  form,
  strategyType,
  { preserveAction = true } = {},
) {
  const select = formField(form, "pool_action_type");
  if (!select) return;
  const previousAction = nonEmptyText(select.value);
  const allowed = STRATEGY_POOL_ACTION_TYPES[strategyType] || [];
  select.innerHTML = [
    '<option value="">请选择兼容动作</option>',
    ...allowed.map((actionType) => projectionOptionHtml(
      actionType,
      actionType,
      { "candidate-lab-action-type": "1" },
    )),
  ].join("");
  if (
    preserveAction
    && previousAction
    && allowed.includes(previousAction)
    && selectContainsValue(select, previousAction)
  ) {
    select.value = previousAction;
  } else if (allowed.length === 1) {
    select.value = allowed[0];
  } else {
    select.value = "";
  }
  setStrategyPoolActionValuePanelVisible(
    form,
    ["limit", "pricing", "segment"].includes(nonEmptyText(select.value)),
  );
}

function syncStrategyPoolCompileControls(form, pools) {
  if (!form) return;
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_compile_strategy_type"),
    pools,
  );
  const help = form.querySelector?.("[data-candidate-lab-pool-compile-help]");
  if (help) {
    help.textContent = pools.length
      ? selectedType
        ? "将编译当前受认证非空 Pool 的完整规则瀑布。"
        : "当前存在多个非空 Pool，请明确选择要编译的策略类型。"
      : "当前没有可编译的非空 Strategy Pool；确定性内核不能编译空 Pool。";
  }
}

function syncStrategyPoolRemoveEntryControls(
  form,
  pools,
  { preserveEntry = true } = {},
) {
  if (!form) return;
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_remove_strategy_type"),
    pools,
  );
  const pool = pools.find((item) => item.strategyType === selectedType);
  syncStrategyPoolEntrySelect(
    formField(form, "pool_remove_entry_id"),
    pool,
    { preserveEntry },
  );
  const help = form.querySelector?.("[data-candidate-lab-pool-remove-help]");
  if (help) {
    help.textContent = pools.length
      ? selectedType
        ? "请明确选择一个当前受认证 Entry；平台不会按顺序自动代选。"
        : "当前存在多个非空 Pool，请先明确选择策略类型。"
      : "当前没有可移除条目的非空 Strategy Pool。";
  }
}

function syncStrategyPoolSetActionControls(
  form,
  pools,
  {
    preserveEntry = true,
    preserveAction = true,
  } = {},
) {
  if (!form) return;
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_action_strategy_type"),
    pools,
  );
  const pool = pools.find((item) => item.strategyType === selectedType);
  syncStrategyPoolEntrySelect(
    formField(form, "pool_action_entry_id"),
    pool,
    { preserveEntry },
  );
  syncStrategyPoolActionTypes(
    form,
    selectedType,
    { preserveAction },
  );
  const help = form.querySelector?.("[data-candidate-lab-pool-action-help]");
  if (help) {
    help.textContent = pools.length
      ? selectedType
        ? "请明确选择一个当前受认证 Entry，并设置与 Pool 类型兼容的动作。"
        : "当前存在多个非空 Pool，请先明确选择策略类型。"
      : "当前没有可修改动作的非空 Strategy Pool。";
  }
}

function syncStrategyPoolReorderControls(form, pools) {
  if (!form) return;
  const selectedType = syncStrategyPoolTypeSelect(
    formField(form, "pool_reorder_strategy_type"),
    pools,
  );
  const pool = pools.find((item) => item.strategyType === selectedType);
  renderStrategyPoolReorderOrder(form, pool);
  const help = form.querySelector?.("[data-candidate-lab-pool-reorder-help]");
  if (help) {
    help.textContent = pools.length
      ? selectedType
        ? "选择一个 Entry 后使用上移、下移；提交始终包含当前完整 Entry 集合。"
        : "当前存在多个非空 Pool，请先明确选择策略类型。"
      : "当前没有可重排的非空 Strategy Pool。";
  }
}

function strategyPoolOrderMatches(pool, orderedIds) {
  if (!pool || !Array.isArray(orderedIds)) return false;
  const currentIds = pool.entries.map((entry) => entry.entryId);
  return (
    orderedIds.length === currentIds.length
    && new Set(orderedIds).size === orderedIds.length
    && currentIds.every((entryId) => orderedIds.includes(entryId))
  );
}

function renderStrategyPoolReorderOrder(
  form,
  pool,
  orderedIds = null,
  selectedEntryId = "",
) {
  const orderSelect = formField(form, "pool_reorder_ordered_ids");
  if (orderSelect) {
    const order = (
      strategyPoolOrderMatches(pool, orderedIds)
        ? orderedIds
        : pool?.entries.map((entry) => entry.entryId)
    ) || [];
    const byId = new Map(
      (pool?.entries || []).map((entry) => [entry.entryId, entry]),
    );
    orderSelect.innerHTML = pool
      ? order.map((entryId, index) => {
        const entry = byId.get(entryId);
        return projectionOptionHtml(
          entry.entryId,
          strategyPoolEntryLabel(entry, index),
          {
            "candidate-lab-projection": "1",
            "strategy-type": pool.strategyType,
            "entry-id": entry.entryId,
          },
        );
      }).join("")
      : '<option value="">请先选择 Strategy Pool</option>';
    orderSelect.value = (
      selectedEntryId && selectContainsValue(orderSelect, selectedEntryId)
        ? selectedEntryId
        : ""
    );
  }
}

function syncStrategyPoolOperationForms(root, payload) {
  const pools = strategyPoolOperationPools(payload);
  syncStrategyPoolCompileControls(strategyPoolCompileForm(root), pools);
  syncStrategyPoolRemoveEntryControls(
    strategyPoolRemoveEntryForm(root),
    pools,
  );
  syncStrategyPoolSetActionControls(
    strategyPoolSetActionForm(root),
    pools,
  );
  syncStrategyPoolReorderControls(strategyPoolReorderForm(root), pools);
}

function strategyPoolOperationRequestIsCurrent(request, payload) {
  if (!STRATEGY_POOL_OPERATION_WORKFLOWS.includes(request?.workflow)) {
    return true;
  }
  const inputs = isRecord(request?.workflow_inputs)
    ? request.workflow_inputs
    : {};
  const pool = strategyPoolOperationPools(payload).find(
    (item) => item.strategyType === inputs.strategy_type,
  );
  if (!pool) return false;
  if (request.workflow === "strategy_pool_compile") return true;
  if (
    request.workflow === "strategy_pool_remove_entry"
    || request.workflow === "strategy_pool_set_action"
  ) {
    return pool.entries.some((entry) => entry.entryId === inputs.entry_id);
  }
  return strategyPoolOrderMatches(pool, inputs.ordered_ids);
}

function strategyPoolValidationRequestIsCurrent(request, payload) {
  if (request?.workflow !== "strategy_pool_validation") return true;
  const strategyType = nonEmptyText(
    request?.workflow_inputs?.strategy_type,
  );
  return (
    STRATEGY_POOL_TYPES.includes(strategyType)
    && strategyPoolOperationPools(payload).some(
      (pool) => pool.strategyType === strategyType,
    )
  );
}

function strategyPoolStabilityRequestIsCurrent(request, payload) {
  if (request?.workflow !== "strategy_pool_stability") return true;
  const strategyType = nonEmptyText(
    request?.workflow_inputs?.strategy_type,
  );
  return (
    STRATEGY_POOL_TYPES.includes(strategyType)
    && strategyPoolOperationPools(payload).some(
      (pool) => pool.strategyType === strategyType,
    )
  );
}

function crossCandidateSearchForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="cross_matrix_candidate_search"]',
  ) || null;
}

function crossCandidateBuildForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="cross_matrix_candidate_build_from_search"]',
  ) || null;
}

function crossSearchFeatureOptions(payload) {
  const byFeature = new Map();
  for (const candidate of univariateProjectionCandidates(payload)) {
    for (const pair of univariateCandidatePairs(candidate)) {
      if (!byFeature.has(pair.feature)) byFeature.set(pair.feature, new Set());
      byFeature.get(pair.feature).add(pair.method);
    }
  }
  return [...byFeature.entries()].map(([feature, methods]) => ({
    feature,
    methodCount: methods.size,
  }));
}

function syncCrossCandidateSearchControls(form, payload) {
  if (!form) return;
  const select = formField(form, "cross_search_features");
  if (!select) return;
  const features = crossSearchFeatureOptions(payload);
  const previous = new Set(selectedValues(select));
  select.innerHTML = features.length
    ? features.map((item) => projectionOptionHtml(
      item.feature,
      `${item.feature} · ${item.methodCount} 种受认证分箱方法`,
      {
        "candidate-lab-projection": "1",
        feature: item.feature,
      },
    )).join("")
    : '<option value="" disabled>当前没有可搜索的单变量字段</option>';
  for (const option of Array.from(select.options || [])) {
    option.selected = previous.has(option.value);
  }
  const selectedCount = selectedValues(select).length;
  const searchSpace = selectedCount * (selectedCount - 1) / 2;
  const help = form.querySelector?.(
    "[data-candidate-lab-cross-search-help]",
  );
  if (help) {
    help.textContent = features.length < 2
      ? "当前单变量受认证投影不足 2 个独立字段，请先完成更多单变量分析。"
      : selectedCount < 2
        ? `当前有 ${features.length} 个独立字段可选；请明确选择 2–20 个，页面不会自动代选。`
        : `已明确选择 ${selectedCount} 个字段，对应 ${searchSpace} 个两两组合；实际评估数量受 max_pairs 预算限制。`;
  }
}

function crossSearchProjectionCandidates(payload) {
  const collection = isRecord(payload?.candidates?.cross_search)
    ? payload.candidates.cross_search
    : {};
  const seen = new Set();
  return collectionItems(collection).filter((search) => {
    const searchId = nonEmptyText(search?.search_id);
    if (!CROSS_SEARCH_ID_RE.test(searchId) || seen.has(searchId)) return false;
    seen.add(searchId);
    return true;
  });
}

function crossSearchPairs(search) {
  const seen = new Set();
  return (Array.isArray(search?.pairs) ? search.pairs : []).filter((pair) => {
    const pairId = nonEmptyText(pair?.pair_id);
    if (!isRecord(pair) || !CROSS_PAIR_ID_RE.test(pairId) || seen.has(pairId)) {
      return false;
    }
    seen.add(pairId);
    return true;
  });
}

function crossPairOptionLabel(pair) {
  const eligibility = pair?.eligible === true
    ? "eligible"
    : "不符合稀疏性门槛";
  return [
    nonEmptyText(pair?.pair_id),
    `${nonEmptyText(pair?.x_feature)}/${nonEmptyText(pair?.x_method)}`,
    `× ${nonEmptyText(pair?.y_feature)}/${nonEmptyText(pair?.y_method)}`,
    eligibility,
    `空单元格 ${stablePrimitiveText(pair?.empty_cell_count)}/${stablePrimitiveText(pair?.cell_count)}`,
    `占比 ${stablePrimitiveText(pair?.empty_cell_share)}`,
    `rank ${stablePrimitiveText(pair?.rank)}`,
  ].join(" · ");
}

function syncCrossCandidateBuildControls(
  form,
  payload,
  { preservePair = true } = {},
) {
  if (!form) return;
  const searches = crossSearchProjectionCandidates(payload);
  const searchSelect = formField(form, "cross_build_search_id");
  const pairSelect = formField(form, "cross_build_pair_id");
  if (!searchSelect || !pairSelect) return;
  const previousSearchId = nonEmptyText(searchSelect.value);
  searchSelect.innerHTML = [
    '<option value="">请明确选择一份受认证 Cross 搜索</option>',
    ...searches.map((search) => projectionOptionHtml(
      search.search_id,
      `${search.search_id} · 已评估 ${stablePrimitiveText(search.evaluated)} / ${stablePrimitiveText(search.search_space)} · eligible ${stablePrimitiveText(search.eligible)}`,
      {
        "candidate-lab-projection": "1",
        "search-id": nonEmptyText(search.search_id),
      },
    )),
  ].join("");
  searchSelect.value = selectContainsValue(searchSelect, previousSearchId)
    ? previousSearchId
    : "";

  const searchId = nonEmptyText(searchSelect.value);
  const search = searches.find((item) => item.search_id === searchId);
  const pairs = crossSearchPairs(search);
  const previousPairId = nonEmptyText(pairSelect.value);
  const previousSource = nonEmptyText(
    Array.from(pairSelect.selectedOptions || [])[0]?.dataset?.searchId,
  );
  pairSelect.innerHTML = [
    `<option value="">${searchId ? "请明确选择该搜索中的完整 Pair" : "请先明确选择一份搜索证据"}</option>`,
    ...pairs.map((pair) => projectionOptionHtml(
      pair.pair_id,
      crossPairOptionLabel(pair),
      {
        "candidate-lab-projection": "1",
        "search-id": searchId,
        "pair-id": nonEmptyText(pair.pair_id),
        eligible: pair.eligible === true ? "1" : "0",
      },
    )),
  ].join("");
  if (
    preservePair
    && previousSource === searchId
    && selectContainsValue(pairSelect, previousPairId)
  ) {
    pairSelect.value = previousPairId;
  } else {
    pairSelect.value = "";
  }
  const help = form.querySelector?.("[data-candidate-lab-cross-build-help]");
  if (help) {
    help.textContent = searches.length === 0
      ? "当前任务尚无受认证 Cross 搜索，请先运行自动搜索。"
      : !searchId
        ? "请明确选择搜索；即使只有一个搜索，页面也不会把它当作推荐自动代选。"
        : pairs.length === 0
          ? "该搜索没有可见 Pair，无法从当前投影构建候选。"
          : "请逐项查看 eligible、空单元格与稀疏性后明确选择 Pair；即使只有一个也不会自动代选。";
  }
}

function crossCandidateRequestIsCurrent(request, payload) {
  if (request?.workflow === "cross_matrix_candidate_search") {
    const available = new Set(
      crossSearchFeatureOptions(payload).map((item) => item.feature),
    );
    const features = request?.workflow_inputs?.features;
    return (
      Array.isArray(features)
      && features.length >= 2
      && features.length <= 20
      && new Set(features).size === features.length
      && features.every((feature) => available.has(feature))
    );
  }
  if (
    request?.workflow !== "cross_matrix_candidate_build_from_search"
  ) {
    return true;
  }
  const search = crossSearchProjectionCandidates(payload).find(
    (item) => item.search_id === request?.workflow_inputs?.search_id,
  );
  return crossSearchPairs(search).some(
    (pair) => pair.pair_id === request?.workflow_inputs?.pair_id,
  );
}

function crossRuleSearchForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="cross_rule_search"]',
  ) || null;
}

function crossRuleBuildForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="cross_rule_candidate_build_from_search"]',
  ) || null;
}

function syncCrossRuleSearchControls(form, payload) {
  if (!form) return;
  const select = formField(form, "cross_rule_features");
  if (!select) return;
  const features = crossSearchFeatureOptions(payload);
  const previous = new Set(selectedValues(select));
  select.innerHTML = features.length
    ? features.map((item) => projectionOptionHtml(
      item.feature,
      `${item.feature} · ${item.methodCount} 种受认证分箱方法`,
      {
        "candidate-lab-projection": "1",
        feature: item.feature,
      },
    )).join("")
    : '<option value="" disabled>当前没有可搜索的单变量字段</option>';
  for (const option of Array.from(select.options || [])) {
    option.selected = previous.has(option.value);
  }
  const selectedCount = selectedValues(select).length;
  const dimension = Number(formValue(form, "cross_rule_dimension"));
  const help = form.querySelector?.("[data-candidate-lab-cross-rule-help]");
  if (help) {
    help.textContent = features.length < 2
      ? "当前受认证单变量字段不足 2 个，请先完成单变量分析。"
      : selectedCount < dimension
        ? `当前选择 ${selectedCount} 个字段；${dimension}D 搜索至少需要 ${dimension} 个字段，页面不会自动代选。`
        : `已明确选择 ${selectedCount} 个字段做 ${dimension}D 有预算搜索；阈值和风险方向由平台从认证证据恢复。`;
  }
}

function crossRuleSearchProjectionCandidates(payload) {
  const collection = isRecord(payload?.candidates?.cross_rule_search)
    ? payload.candidates.cross_rule_search
    : {};
  const seen = new Set();
  return collectionItems(collection).filter((search) => {
    const searchId = nonEmptyText(search?.search_id);
    if (
      !CROSS_RULE_SEARCH_ID_RE.test(searchId)
      || seen.has(searchId)
    ) return false;
    seen.add(searchId);
    return true;
  });
}

function crossRulePointers(search) {
  const seen = new Set();
  return (Array.isArray(search?.rules) ? search.rules : []).filter((rule) => {
    const ruleId = nonEmptyText(rule?.rule_id);
    if (!isRecord(rule) || !CROSS_RULE_ID_RE.test(ruleId) || seen.has(ruleId)) {
      return false;
    }
    seen.add(ruleId);
    return true;
  });
}

function syncCrossRuleBuildControls(
  form,
  payload,
  { preserveRule = true } = {},
) {
  if (!form) return;
  const searches = crossRuleSearchProjectionCandidates(payload);
  const searchSelect = formField(form, "cross_rule_build_search_id");
  const ruleSelect = formField(form, "cross_rule_build_rule_id");
  if (!searchSelect || !ruleSelect) return;
  const previousSearchId = nonEmptyText(searchSelect.value);
  searchSelect.innerHTML = [
    '<option value="">请明确选择一份 Cross 阈值规则搜索</option>',
    ...searches.map((search) => projectionOptionHtml(
      search.search_id,
      `${search.search_id} · ${stablePrimitiveText(search.dimension)}D · 已评估 ${stablePrimitiveText(search.evaluated)} · eligible ${stablePrimitiveText(search.eligible)}`,
      {
        "candidate-lab-projection": "1",
        "search-id": nonEmptyText(search.search_id),
      },
    )),
  ].join("");
  searchSelect.value = selectContainsValue(searchSelect, previousSearchId)
    ? previousSearchId
    : "";
  const searchId = nonEmptyText(searchSelect.value);
  const search = searches.find((item) => item.search_id === searchId);
  const rules = crossRulePointers(search);
  const previousRuleId = nonEmptyText(ruleSelect.value);
  const previousSource = nonEmptyText(
    Array.from(ruleSelect.selectedOptions || [])[0]?.dataset?.searchId,
  );
  ruleSelect.innerHTML = [
    `<option value="">${searchId ? "请明确选择该搜索中的完整 rule_id" : "请先明确选择搜索证据"}</option>`,
    ...rules.map((rule) => projectionOptionHtml(
      rule.rule_id,
      [
        rule.rule_id,
        `rank ${stablePrimitiveText(rule.rank)}`,
        rule.eligible === true ? "eligible" : "未满足约束",
        `lift ${stablePrimitiveText(rule?.metrics?.lift)}`,
        `命中率 ${stablePrimitiveText(rule?.metrics?.hit_share)}`,
      ].join(" · "),
      {
        "candidate-lab-projection": "1",
        "search-id": searchId,
        "rule-id": nonEmptyText(rule.rule_id),
        eligible: rule.eligible === true ? "1" : "0",
      },
    )),
  ].join("");
  if (
    preserveRule
    && previousSource === searchId
    && selectContainsValue(ruleSelect, previousRuleId)
  ) {
    ruleSelect.value = previousRuleId;
  } else {
    ruleSelect.value = "";
  }
  const help = form.querySelector?.("[data-candidate-lab-cross-rule-build-help]");
  if (help) {
    help.textContent = searches.length === 0
      ? "当前任务尚无受认证 Cross 阈值规则搜索。"
      : !searchId
        ? "请明确选择搜索；页面不会默认选择最新搜索。"
        : rules.length === 0
          ? "该搜索当前没有可见规则，请下载完整证据或重新搜索。"
          : "请逐项查看约束与指标后选择 rule_id；页面不会自动选择第一名。";
  }
}

function crossRuleRequestIsCurrent(request, payload) {
  if (request?.workflow === "cross_rule_search") {
    const available = new Set(
      crossSearchFeatureOptions(payload).map((item) => item.feature),
    );
    const features = request?.workflow_inputs?.features;
    return (
      Array.isArray(features)
      && features.length >= 2
      && features.length <= 12
      && new Set(features).size === features.length
      && features.every((feature) => available.has(feature))
    );
  }
  if (
    request?.workflow !== "cross_rule_candidate_build_from_search"
  ) return true;
  const search = crossRuleSearchProjectionCandidates(payload).find(
    (item) => item.search_id === request?.workflow_inputs?.search_id,
  );
  return crossRulePointers(search).some(
    (rule) => rule.rule_id === request?.workflow_inputs?.rule_id,
  );
}

function votingSearchForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="voting_candidate_search"]',
  ) || null;
}

function votingBuildForm(root) {
  return root?.querySelector?.(
    '[data-candidate-lab-workflow="voting_candidate_build_from_search"]',
  ) || null;
}

function votingPoolOptions(payload) {
  const pools = collectionItems(isRecord(payload?.pools) ? payload.pools : {});
  const seen = new Set();
  const result = [];
  for (const pool of pools) {
    const strategyType = nonEmptyText(pool?.strategy_type);
    if (
      !["approval", "reject", "limit", "pricing", "segmentation"].includes(
        strategyType,
      )
      || seen.has(strategyType)
    ) {
      continue;
    }
    const rules = (Array.isArray(pool?.entries) ? pool.entries : [])
      .filter((entry) => (
        isRecord(entry)
        && entry.enabled === true
        && nonEmptyText(entry?.source?.asset_type) !== "voting_n_of_k"
        && VOTING_RULE_ID_RE.test(nonEmptyText(entry.rule_id))
      ))
      .map((entry) => ({
        ruleId: nonEmptyText(entry.rule_id),
        assetType: nonEmptyText(entry?.source?.asset_type),
      }));
    if (rules.length < 2) continue;
    seen.add(strategyType);
    result.push({
      strategyType,
      revision: pool.revision,
      rules,
    });
  }
  return result;
}

function syncVotingSearchControls(form, payload) {
  if (!form) return;
  const pools = votingPoolOptions(payload);
  const strategySelect = formField(form, "voting_strategy_type");
  const includeSelect = formField(form, "voting_include_rule_ids");
  const excludeSelect = formField(form, "voting_exclude_rule_ids");
  if (!strategySelect || !includeSelect || !excludeSelect) return;
  const previousType = nonEmptyText(strategySelect.value);
  strategySelect.innerHTML = [
    '<option value="">请选择当前 Strategy Pool</option>',
    ...pools.map((pool) => projectionOptionHtml(
      pool.strategyType,
      `${pool.strategyType} · revision ${stablePrimitiveText(pool.revision)} · ${pool.rules.length} 条可搜索规则`,
      { "candidate-lab-projection": "1" },
    )),
  ].join("");
  if (selectContainsValue(strategySelect, previousType)) {
    strategySelect.value = previousType;
  } else if (pools.length === 1) {
    strategySelect.value = pools[0].strategyType;
  } else {
    strategySelect.value = "";
  }
  const selectedType = nonEmptyText(strategySelect.value);
  const selectedPool = pools.find((pool) => pool.strategyType === selectedType);
  const rules = selectedPool?.rules || [];
  for (const [select, placeholder] of [
    [includeSelect, "可选：必须包含的规则"],
    [excludeSelect, "可选：排除的规则"],
  ]) {
    const previous = new Set(selectedValues(select));
    select.innerHTML = [
      `<option value="" disabled>${escapeHtml(placeholder)}</option>`,
      ...rules.map((rule) => projectionOptionHtml(
        rule.ruleId,
        `${rule.ruleId} · ${rule.assetType || "candidate"}`,
        { "candidate-lab-projection": "1" },
      )),
    ].join("");
    for (const option of Array.from(select.options || [])) {
      option.selected = previous.has(option.value);
    }
  }
  const empty = form.querySelector?.("[data-candidate-lab-voting-pool-empty]");
  if (empty) {
    empty.textContent = pools.length
      ? selectedType
        ? "必须包含/排除项只能来自该 Pool 当前已启用且非 Voting 的受认证规则。"
        : "当前存在多个可搜索 Pool，请明确选择策略类型。"
      : "当前没有包含至少两条可搜索规则的 Strategy Pool。";
  }
}

function votingSearchProjectionCandidates(payload) {
  const collection = isRecord(payload?.candidates?.voting_search)
    ? payload.candidates.voting_search
    : {};
  const currentPoolRevisions = new Map(
    collectionItems(isRecord(payload?.pools) ? payload.pools : {})
      .map((pool) => [
        nonEmptyText(pool?.strategy_type),
        pool?.revision,
      ]),
  );
  const seen = new Set();
  return collectionItems(collection).filter((item) => {
    const searchId = nonEmptyText(item?.search_id);
    const strategyType = nonEmptyText(item?.strategy_type);
    if (
      !VOTING_SEARCH_ID_RE.test(searchId)
      || seen.has(searchId)
      || currentPoolRevisions.get(strategyType) !== item?.pool_revision
    ) {
      return false;
    }
    seen.add(searchId);
    return true;
  });
}

function votingSearchCombinations(search) {
  const seen = new Set();
  return (Array.isArray(search?.combinations) ? search.combinations : [])
    .filter((combo) => {
      const comboId = nonEmptyText(combo?.combo_id);
      if (!isRecord(combo) || !VOTING_COMBO_ID_RE.test(comboId) || seen.has(comboId)) {
        return false;
      }
      seen.add(comboId);
      return true;
    });
}

function syncVotingBuildControls(
  form,
  payload,
  { preserveCombo = true } = {},
) {
  if (!form) return;
  const searches = votingSearchProjectionCandidates(payload);
  const searchSelect = formField(form, "voting_search_id");
  const comboSelect = formField(form, "voting_combo_id");
  if (!searchSelect || !comboSelect) return;
  const previousSearchId = nonEmptyText(searchSelect.value);
  searchSelect.innerHTML = [
    '<option value="">请选择受认证 Voting 搜索</option>',
    ...searches.map((search) => projectionOptionHtml(
      search.search_id,
      `${search.search_id} · ${search.strategy_type} · K=${stablePrimitiveText(search.member_count)} / n=${stablePrimitiveText(search.n)}`,
      {
        "candidate-lab-projection": "1",
        "strategy-type": nonEmptyText(search.strategy_type),
      },
    )),
  ].join("");
  if (selectContainsValue(searchSelect, previousSearchId)) {
    searchSelect.value = previousSearchId;
  } else if (searches.length === 1) {
    searchSelect.value = searches[0].search_id;
  } else {
    searchSelect.value = "";
  }
  const searchId = nonEmptyText(searchSelect.value);
  const search = searches.find((item) => item.search_id === searchId);
  const combinations = votingSearchCombinations(search);
  const previousComboId = nonEmptyText(comboSelect.value);
  const previousSource = nonEmptyText(
    Array.from(comboSelect.selectedOptions || [])[0]?.dataset?.sourceSearchId,
  );
  comboSelect.innerHTML = [
    '<option value="">请选择该搜索中的精确组合</option>',
    ...combinations.map((combo) => projectionOptionHtml(
      combo.combo_id,
      `${combo.combo_id} · ${readableValue(combo.members)} · ${combo.eligible ? "约束通过" : "约束未通过"}`,
      {
        "candidate-lab-projection": "1",
        "source-search-id": searchId,
      },
    )),
  ].join("");
  if (
    preserveCombo
    && previousSource === searchId
    && selectContainsValue(comboSelect, previousComboId)
  ) {
    comboSelect.value = previousComboId;
  } else {
    comboSelect.value = "";
  }
  const empty = form.querySelector?.("[data-candidate-lab-voting-search-empty]");
  if (empty) {
    empty.textContent = searches.length
      ? searchId
        ? combinations.length
          ? "请明确选择组合；页面不会按名次、最好或冠军自动代选。"
          : "该受认证搜索没有可见的已评估组合。"
        : "请先选择一份受认证 Voting 搜索证据。"
      : "当前任务尚无受认证 Voting 搜索，请先运行组合搜索。";
  }
}

function syncVotingForms(root, payload) {
  syncVotingSearchControls(votingSearchForm(root), payload);
  syncVotingBuildControls(votingBuildForm(root), payload);
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
      + "[data-candidate-lab-form] button, "
      + "[data-candidate-lab-interactive-tree-prune], "
      + "[data-candidate-lab-interactive-tree-threshold], "
      + "[data-candidate-lab-interactive-tree-split-candidate], "
      + "[data-candidate-lab-interactive-tree-auto-continuation], "
      + "[data-candidate-lab-interactive-tree-frontier-materialize]",
    ) || [];
    for (const control of controls) {
      const refinementPanel = control.closest?.(
        "[data-candidate-lab-refinement-panel]",
      );
      const scorecardPanel = control.closest?.(
        "[data-candidate-lab-scorecard-banding-panel]",
      );
      const stabilityPanel = control.closest?.(
        "[data-candidate-lab-stability-panel]",
      );
      const poolActionValuePanel = control.closest?.(
        "[data-candidate-lab-pool-action-value-panel]",
      );
      const poolAddDefaultValuePanel = control.closest?.(
        "[data-candidate-lab-pool-add-default-value-panel]",
      );
      const poolAddActionValuePanel = control.closest?.(
        "[data-candidate-lab-pool-add-action-value-panel]",
      );
      const poolAddPlacementPanel = control.closest?.(
        "[data-candidate-lab-pool-add-placement-panel]",
      );
      const adoptionEconomicsPanel = control.closest?.(
        "[data-candidate-lab-adoption-economics]",
      );
      const adoptionComponent = control.closest?.(
        "[data-candidate-lab-adoption-component]",
      );
      const adoptionBinding = control.closest?.(
        "[data-candidate-lab-adoption-binding]",
      );
      const treeThresholdPanel = control.closest?.(
        "[data-candidate-lab-tree-threshold-panel]",
      );
      const hiddenByMode = Boolean(
        refinementPanel?.classList?.contains?.("hidden")
        || scorecardPanel?.classList?.contains?.("hidden")
        || stabilityPanel?.classList?.contains?.("hidden")
        || poolActionValuePanel?.classList?.contains?.("hidden")
        || poolAddDefaultValuePanel?.classList?.contains?.("hidden")
        || poolAddActionValuePanel?.classList?.contains?.("hidden")
        || poolAddPlacementPanel?.classList?.contains?.("hidden")
        || adoptionEconomicsPanel?.classList?.contains?.("hidden")
        || adoptionComponent?.classList?.contains?.("hidden")
        || adoptionBinding?.classList?.contains?.("hidden")
        || treeThresholdPanel?.classList?.contains?.("hidden"),
      );
      const locked = (
        control.dataset?.candidateLabPoolAddLocked === "1"
      );
      control.disabled = Boolean(reason) || hiddenByMode || locked;
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
    syncStrategyProjectContextControls(
      strategyProjectContextForm(root),
    );
    syncRefinementForm(root, state.payload);
    syncScorecardForms(root, state.payload);
    syncCandidateStabilityControls(
      candidateStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolAddControls(
      strategyPoolAddForm(root),
      state.payload,
    );
    syncStrategyPoolValidationControls(
      strategyPoolValidationForm(root),
      state.payload,
    );
    syncStrategyPoolStabilityControls(
      strategyPoolStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolImpactControls(
      strategyPoolImpactForm(root),
      state.payload,
    );
    syncStrategyImpactCubeControls(
      strategyImpactCubeForm(root),
      state.payload,
    );
    syncStrategyPoolMaterializeControls(
      strategyPoolMaterializeForm(root),
      state.payload,
    );
    syncStrategyLifecycleAdoptionControls(
      strategyLifecycleAdoptionForm(root),
      state.payload,
    );
    syncStrategyDslDeliveryControls(
      strategyDslDeliveryForm(root),
      state.payload,
    );
    syncStrategyPoolApplyControls(
      strategyPoolApplyForm(root),
      state.payload,
    );
    syncStrategyPoolOperationForms(root, state.payload);
    syncCrossCandidateSearchControls(
      crossCandidateSearchForm(root),
      state.payload,
    );
    syncCrossCandidateBuildControls(
      crossCandidateBuildForm(root),
      state.payload,
    );
    syncCrossRuleSearchControls(
      crossRuleSearchForm(root),
      state.payload,
    );
    syncCrossRuleBuildControls(
      crossRuleBuildForm(root),
      state.payload,
    );
    syncVotingForms(root, state.payload);
    syncInteractiveTreeSplitSearchControls(
      interactiveTreeSplitSearchForm(root),
      state.payload,
    );
    syncInteractiveTreeRevisionControls(
      interactiveTreeForm(root),
      state.payload,
    );
    syncInteractiveTreeFrontierGroupMaterializationControls(
      interactiveTreeFrontierGroupMaterializationForm(root),
      state.payload,
    );
    syncInteractiveTreeFrontierMaterializationControls(
      interactiveTreeFrontierMaterializationForm(root),
      state.payload,
    );
    renderAvailability();
  }

  function resetForms() {
    const root = panel();
    const forms = root?.querySelectorAll?.("[data-candidate-lab-form]") || [];
    for (const form of forms) {
      form.reset?.();
      setFormError(form, "");
    }
    syncStrategyProjectContextControls(
      strategyProjectContextForm(root),
    );
    syncRefinementForm(root, state.payload, { preserveBins: false });
    syncScorecardForms(root, state.payload);
    syncCandidateStabilityControls(
      candidateStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolAddControls(
      strategyPoolAddForm(root),
      state.payload,
      { preserveSource: false },
    );
    syncStrategyPoolValidationControls(
      strategyPoolValidationForm(root),
      state.payload,
    );
    syncStrategyPoolStabilityControls(
      strategyPoolStabilityForm(root),
      state.payload,
    );
    syncStrategyPoolImpactControls(
      strategyPoolImpactForm(root),
      state.payload,
    );
    syncStrategyImpactCubeControls(
      strategyImpactCubeForm(root),
      state.payload,
    );
    syncStrategyPoolMaterializeControls(
      strategyPoolMaterializeForm(root),
      state.payload,
    );
    syncStrategyLifecycleAdoptionControls(
      strategyLifecycleAdoptionForm(root),
      state.payload,
    );
    syncStrategyDslDeliveryControls(
      strategyDslDeliveryForm(root),
      state.payload,
    );
    syncStrategyPoolApplyControls(
      strategyPoolApplyForm(root),
      state.payload,
    );
    syncStrategyPoolOperationForms(root, state.payload);
    syncCrossCandidateSearchControls(
      crossCandidateSearchForm(root),
      state.payload,
    );
    syncCrossCandidateBuildControls(
      crossCandidateBuildForm(root),
      state.payload,
      { preservePair: false },
    );
    syncCrossRuleSearchControls(
      crossRuleSearchForm(root),
      state.payload,
    );
    syncCrossRuleBuildControls(
      crossRuleBuildForm(root),
      state.payload,
      { preserveRule: false },
    );
    syncVotingForms(root, state.payload);
    syncInteractiveTreeSplitSearchControls(
      interactiveTreeSplitSearchForm(root),
      state.payload,
      { preserveNode: false },
    );
    syncInteractiveTreeRevisionControls(
      interactiveTreeForm(root),
      state.payload,
      { preserveNode: false },
    );
    syncInteractiveTreeFrontierGroupMaterializationControls(
      interactiveTreeFrontierGroupMaterializationForm(root),
      state.payload,
      { preserveNodes: false },
    );
    syncInteractiveTreeFrontierMaterializationControls(
      interactiveTreeFrontierMaterializationForm(root),
      state.payload,
      { preserveNode: false },
    );
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
      if (!strategyPoolAddRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选候选来源或当前 Pool 默认动作已过期、不完整或不属于受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolOperationRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "当前 Strategy Pool 类型或 Entry 集合已过期、不完整或不属于受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolValidationRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyPoolStabilityRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyMeasurementRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "影响测算所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!strategyWorkbenchRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选 Pool 或策略版本已过期、不完整或不属于当前任务受认证投影，请刷新 Strategy Workbench 后重选。",
        );
      }
      if (!crossCandidateRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选单变量字段、Cross 搜索或 Pair 已过期或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (!crossRuleRequestIsCurrent(
        strategyRequest,
        state.payload,
      )) {
        throw new Error(
          "所选单变量字段、Cross 阈值搜索或规则已过期或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "strategy_pool_apply"
        && !strategyPoolApplyOptions(state.payload).some(
          (pool) => (
            pool.strategyType === strategyRequest.workflow_inputs.strategy_type
          ),
        )
      ) {
        throw new Error(
          "所选 Strategy Pool 已过期、为空或不属于当前任务受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_split_search"
        && !interactiveTreeSplitSearchRequestIsCurrent(
          state.payload,
          strategyRequest.workflow_inputs,
        )
      ) {
        throw new Error(
          "树节点搜索来源、可见节点或认证特征全集已过期，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_revision"
        && !interactiveTreeRevisionRequestIsCurrent(
          state.payload,
          strategyRequest.workflow_inputs,
        )
      ) {
        throw new Error(
          "交互式树修订指针或当前阈值已过期，或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow === "interactive_tree_auto_continuation"
        && (
          !interactiveTreeSplitCandidatePointer(
            state.payload,
            strategyRequest.workflow_inputs.search_id,
            strategyRequest.workflow_inputs.candidate_id,
          )
          || interactiveTreeSplitCandidatePointer(
            state.payload,
            strategyRequest.workflow_inputs.search_id,
            strategyRequest.workflow_inputs.candidate_id,
          )?.search?.source_node?.is_frontier !== true
        )
      ) {
        throw new Error(
          "自动续建的搜索或候选已过期，或不再指向受认证 frontier，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow
          === "interactive_tree_frontier_group_materialization"
        && interactiveTreeFrontierGroupPointers(
          state.payload,
          strategyRequest.workflow_inputs.revision_id,
          strategyRequest.workflow_inputs.source_node_ids,
        ).length !== strategyRequest.workflow_inputs.source_node_ids.length
      ) {
        throw new Error(
          "交互树前沿 OR 分组指针已过期或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
      if (
        strategyRequest.workflow
          === "interactive_tree_frontier_materialization"
        && !interactiveTreeFrontierPointer(
          state.payload,
          strategyRequest.workflow_inputs.revision_id,
          strategyRequest.workflow_inputs.source_node_id,
        )
      ) {
        throw new Error(
          "交互树前沿指针已过期或不属于当前任务的受认证投影，请刷新 Candidate Lab 后重选。",
        );
      }
    } catch (error) {
      const message = error?.message || "Candidate Lab 表单输入无效。";
      setFormError(form, message);
      dependencies.setActionStatus?.(message, "error");
      return null;
    }

    const workflow = strategyRequest.workflow;
    const content = strategyRequest.request_kind === "strategy_lifecycle"
      ? WORKFLOW_LABELS.strategy_lifecycle_adopt
      : WORKFLOW_LABELS[workflow] || "从 Candidate Lab 启动策略分析";
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
      await dependencies.settleCandidateLabSubmission?.(requestTaskId);
      if (selectedTaskId() !== requestTaskId) return result;
      if (
        strategyRequest.request_kind === "strategy_lifecycle"
        || strategyRequest.workflow === "cross_matrix_candidate_search"
        || strategyRequest.workflow
          === "cross_matrix_candidate_build_from_search"
        || strategyRequest.workflow === "strategy_pool_apply"
        || strategyRequest.workflow === "strategy_pool_stability"
        || strategyRequest.workflow === "strategy_pool_impact"
        || strategyRequest.workflow === "strategy_impact_cube"
        || strategyRequest.workflow === "strategy_project_context"
        || strategyRequest.workflow === "strategy_sample_design_v2"
        || strategyRequest.workflow === "strategy_pool_materialize"
        || strategyRequest.workflow === "strategy_dsl_delivery"
        || strategyRequest.workflow === "strategy_report_bundle_v2"
        || strategyRequest.workflow === "strategy_pool_add_candidate"
        || STRATEGY_POOL_OPERATION_WORKFLOWS.includes(strategyRequest.workflow)
      ) {
        await refresh(requestTaskId, { silent: true });
        if (selectedTaskId() !== requestTaskId) return result;
      }
      state.submitting = false;
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
    const reorderMove = event.target?.closest?.(
      "[data-candidate-lab-pool-reorder-move]",
    );
    if (reorderMove) {
      event.preventDefault?.();
      const form = strategyPoolReorderForm(panel());
      const reason = blockedReason(state, dependencies);
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可调整 Pool 顺序。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      try {
        const request = collectStrategyCandidateLabRequest(form);
        const strategyType = request.workflow_inputs.strategy_type;
        const orderedIds = request.workflow_inputs.ordered_ids;
        const pool = strategyPoolOperationPools(state.payload).find(
          (item) => item.strategyType === strategyType,
        );
        if (!strategyPoolOrderMatches(pool, orderedIds)) {
          throw new Error(
            "当前 Pool Entry 集合已过期，请刷新 Candidate Lab 后重试。",
          );
        }
        const orderSelect = formField(form, "pool_reorder_ordered_ids");
        const selected = Array.from(orderSelect?.selectedOptions || [])[0];
        const selectedEntryId = nonEmptyText(selected?.value);
        if (
          !selected
          || selected.dataset?.candidateLabProjection !== "1"
          || nonEmptyText(selected.dataset?.strategyType) !== strategyType
          || nonEmptyText(selected.dataset?.entryId) !== selectedEntryId
        ) {
          throw new Error("请先选择一个当前受认证 Pool Entry 再调整顺序。");
        }
        const direction = nonEmptyText(
          reorderMove.dataset?.candidateLabPoolReorderMove,
        );
        if (!["up", "down"].includes(direction)) {
          throw new Error("Pool Entry 调整方向无效。");
        }
        const selectedIndex = orderedIds.indexOf(selectedEntryId);
        const targetIndex = direction === "up"
          ? selectedIndex - 1
          : selectedIndex + 1;
        if (targetIndex >= 0 && targetIndex < orderedIds.length) {
          [orderedIds[selectedIndex], orderedIds[targetIndex]] = [
            orderedIds[targetIndex],
            orderedIds[selectedIndex],
          ];
          renderStrategyPoolReorderOrder(
            form,
            pool,
            orderedIds,
            selectedEntryId,
          );
        }
        setFormError(form, "");
        renderAvailability();
      } catch (error) {
        const message = error?.message || "无法调整当前 Pool Entry 顺序。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
      }
      return true;
    }
    const materialize = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-frontier-materialize]",
    );
    if (materialize) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeFrontierMaterializationForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const revisionId = nonEmptyText(materialize.dataset?.revisionId);
      const sourceNodeId = nonEmptyText(materialize.dataset?.sourceNodeId);
      const pointer = interactiveTreeFrontierPointer(
        state.payload,
        revisionId,
        sourceNodeId,
      );
      if (!pointer || !form) {
        const message = "该前沿指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeFrontierMaterializationControls(
        form,
        state.payload,
        {
          requestedRevisionId: revisionId,
          requestedSourceNodeId: sourceNodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证 revision 与前沿节点；确认后只物化该节点，不会自动入池。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const continuationCandidate = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-auto-continuation]",
    );
    if (continuationCandidate) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = panel()?.querySelector?.(
        '[data-candidate-lab-workflow="interactive_tree_auto_continuation"]',
      );
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchId = nonEmptyText(
        continuationCandidate.dataset?.searchId,
      );
      const candidateId = nonEmptyText(
        continuationCandidate.dataset?.candidateId,
      );
      const pointer = interactiveTreeSplitCandidatePointer(
        state.payload,
        searchId,
        candidateId,
      );
      if (
        !form
        || !pointer
        || pointer.search?.source_node?.is_frontier !== true
      ) {
        const message = "该候选不属于当前受认证 frontier 搜索，请刷新后重试。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchField = formField(
        form,
        "interactive_tree_continuation_search_id",
      );
      const candidateField = formField(
        form,
        "interactive_tree_continuation_candidate_id",
      );
      if (searchField) searchField.value = searchId;
      if (candidateField) candidateField.value = candidateId;
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入人工明确选择的 eligible 候选；请检查全部硬预算后再提交续建。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const splitCandidate = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-split-candidate]",
    );
    if (splitCandidate) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const searchId = nonEmptyText(splitCandidate.dataset?.searchId);
      const candidateId = nonEmptyText(splitCandidate.dataset?.candidateId);
      const pointer = interactiveTreeSplitCandidatePointer(
        state.payload,
        searchId,
        candidateId,
      );
      const sourceTreeId = nonEmptyText(
        splitCandidate.dataset?.sourceTreeId,
      );
      const nodeId = nonEmptyText(splitCandidate.dataset?.nodeId);
      const feature = nonEmptyText(splitCandidate.dataset?.feature);
      const threshold = Number(splitCandidate.dataset?.threshold);
      if (
        !pointer
        || !form
        || pointer.search.source_tree_id !== sourceTreeId
        || pointer.search.node_id !== nodeId
        || pointer.candidate.feature !== feature
        || Number(pointer.candidate.threshold) !== threshold
      ) {
        const message = "该分裂候选不属于当前任务的受认证搜索证据。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const currentFeature = nonEmptyText(
        pointer.search?.source_node?.feature,
      );
      const operation = feature === currentFeature
        ? "adjust_split_threshold"
        : "replace_split_feature";
      const revisionPointer = interactiveTreeRevisionPointer(
        state.payload,
        sourceTreeId,
        nodeId,
        operation,
      );
      if (!revisionPointer) {
        const message = "该搜索来源节点当前已不可编辑，请刷新树投影后重新搜索。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: operation,
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      const thresholdField = formField(form, "interactive_tree_threshold");
      if (thresholdField) thresholdField.value = stablePrimitiveText(threshold);
      if (operation === "replace_split_feature") {
        const featureField = formField(form, "interactive_tree_feature");
        if (!selectContainsValue(featureField, feature)) {
          const message = "候选字段已不在当前来源树的认证特征全集中。";
          setFormError(form, message);
          dependencies.setActionStatus?.(message, "error");
          return true;
        }
        featureField.value = feature;
      }
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已回填受认证候选字段与阈值；尚未修改树，请检查理由并手动确认创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const thresholdAdjustment = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-threshold]",
    );
    if (thresholdAdjustment) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const sourceTreeId = nonEmptyText(
        thresholdAdjustment.dataset?.sourceTreeId,
      );
      const nodeId = nonEmptyText(thresholdAdjustment.dataset?.nodeId);
      const pointer = interactiveTreeRevisionPointer(
        state.payload,
        sourceTreeId,
        nodeId,
        "adjust_split_threshold",
      );
      if (
        !pointer
        || !form
        || nonEmptyText(thresholdAdjustment.dataset?.feature)
          !== pointer.feature
        || nonEmptyText(thresholdAdjustment.dataset?.currentThreshold)
          !== stablePrimitiveText(pointer.current_threshold)
      ) {
        const message = "该阈值调整指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: "adjust_split_threshold",
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证分支、字段与当前阈值；请明确填写不同的新阈值后创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const prune = event.target?.closest?.(
      "[data-candidate-lab-interactive-tree-prune]",
    );
    if (prune) {
      event.preventDefault?.();
      const reason = blockedReason(state, dependencies);
      const form = interactiveTreeForm(panel());
      if (reason) {
        const message = BLOCKED_REASON_COPY[reason]
          || "当前 Candidate Lab 暂不可启动新分析。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      const sourceTreeId = nonEmptyText(prune.dataset?.sourceTreeId);
      const nodeId = nonEmptyText(prune.dataset?.nodeId);
      const pointer = interactiveTreePointer(
        state.payload,
        sourceTreeId,
        nodeId,
      );
      if (!pointer || !form) {
        const message = "该剪枝指针不属于当前任务的受认证 Candidate Lab 投影。";
        setFormError(form, message);
        dependencies.setActionStatus?.(message, "error");
        return true;
      }
      syncInteractiveTreeRevisionControls(
        form,
        state.payload,
        {
          requestedOperation: "prune_subtree",
          requestedSourceTreeId: sourceTreeId,
          requestedNodeId: nodeId,
          preserveNode: false,
        },
      );
      setFormError(form, "");
      const launcher = form.closest?.(".candidate-lab-launcher");
      if (launcher) launcher.open = true;
      dependencies.setActionStatus?.(
        "已带入受认证分支和节点；填写可选理由后确认创建不可变 revision。",
        "info",
      );
      renderAvailability();
      return true;
    }
    const retry = event.target?.closest?.("[data-candidate-lab-retry]");
    if (!retry) return false;
    event.preventDefault?.();
    void refresh();
    return true;
  }

  function handleChange(event) {
    const field = event.target?.closest?.("[data-candidate-lab-field]");
    if (!field) return false;
    const fieldName = field.dataset?.candidateLabField;
    const adoption = field.closest?.(
      '[data-candidate-lab-workflow="strategy_lifecycle_adopt"]',
    );
    if (adoption && fieldName === "lifecycle_adopt_strategy_id") {
      syncStrategyLifecycleAdoptionControls(adoption, state.payload);
      renderAvailability();
      return true;
    }
    if (
      adoption
      && /^lifecycle_adopt_[a-z_]+_mode$/.test(fieldName || "")
    ) {
      const selected = Array.from(
        formField(adoption, "lifecycle_adopt_strategy_id")
          ?.selectedOptions || [],
      )[0] || null;
      syncStrategyLifecycleAdoptionEconomics(
        adoption,
        nonEmptyText(selected?.dataset?.strategyType),
      );
      renderAvailability();
      return true;
    }
    const crossSearch = field.closest?.(
      '[data-candidate-lab-workflow="cross_matrix_candidate_search"]',
    );
    if (
      crossSearch
      && (
        fieldName === "cross_search_features"
        || fieldName === "cross_search_max_pairs"
      )
    ) {
      syncCrossCandidateSearchControls(crossSearch, state.payload);
      renderAvailability();
      return true;
    }
    const crossBuild = field.closest?.(
      '[data-candidate-lab-workflow="cross_matrix_candidate_build_from_search"]',
    );
    if (crossBuild && fieldName === "cross_build_search_id") {
      syncCrossCandidateBuildControls(
        crossBuild,
        state.payload,
        { preservePair: false },
      );
      renderAvailability();
      return true;
    }
    const crossRuleSearch = field.closest?.(
      '[data-candidate-lab-workflow="cross_rule_search"]',
    );
    if (
      crossRuleSearch
      && (
        fieldName === "cross_rule_features"
        || fieldName === "cross_rule_dimension"
        || fieldName === "cross_rule_max_trials"
      )
    ) {
      syncCrossRuleSearchControls(crossRuleSearch, state.payload);
      renderAvailability();
      return true;
    }
    const crossRuleBuild = field.closest?.(
      '[data-candidate-lab-workflow="cross_rule_candidate_build_from_search"]',
    );
    if (
      crossRuleBuild
      && fieldName === "cross_rule_build_search_id"
    ) {
      syncCrossRuleBuildControls(
        crossRuleBuild,
        state.payload,
        { preserveRule: false },
      );
      renderAvailability();
      return true;
    }
    const refinement = field.closest?.(
      '[data-candidate-lab-workflow="univariate_candidate_refinement"]',
    );
    if (refinement) {
      if (fieldName === "refinement_mode") {
        syncRefinementMode(refinement);
      } else if (
        fieldName === "source_candidate_id"
        || fieldName === "source_feature_method"
      ) {
        syncRefinementCandidateControls(
          refinement,
          state.payload,
          { preserveBins: false },
        );
      } else {
        return false;
      }
      renderAvailability();
      return true;
    }
    const scorecardBand = field.closest?.(
      '[data-candidate-lab-workflow="scorecard_band_build"]',
    );
    if (scorecardBand && fieldName === "scorecard_banding_mode") {
      syncScorecardBandingMode(scorecardBand);
      renderAvailability();
      return true;
    }
    const scorecardSelection = field.closest?.(
      '[data-candidate-lab-workflow="scorecard_cutoff_selection"]',
    );
    if (scorecardSelection && fieldName === "scorecard_asset_id") {
      syncScorecardCutoffControls(
        scorecardSelection,
        state.payload,
        { preserveCutoff: false },
      );
      renderAvailability();
      return true;
    }
    const candidateStability = field.closest?.(
      '[data-candidate-lab-workflow="candidate_monthly_stability"]',
    );
    if (
      candidateStability
      && fieldName === "stability_source_mode"
    ) {
      syncCandidateStabilityControls(
        candidateStability,
        state.payload,
      );
      renderAvailability();
      return true;
    }
    const poolAdd = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_add_candidate"]',
    );
    if (poolAdd && fieldName === "pool_add_strategy_type") {
      syncStrategyPoolAddControls(
        poolAdd,
        state.payload,
        { preserveSource: false },
      );
      renderAvailability();
      return true;
    }
    if (poolAdd && fieldName === "pool_add_source_id") {
      syncStrategyPoolAddPlacement(poolAdd);
      renderAvailability();
      return true;
    }
    if (
      poolAdd
      && (
        fieldName === "pool_add_default_action_type"
        || fieldName === "pool_add_action_type"
      )
    ) {
      const isDefault = fieldName === "pool_add_default_action_type";
      const actionType = formValue(poolAdd, fieldName);
      setStrategyPoolAddPanelVisible(
        poolAdd,
        isDefault
          ? "[data-candidate-lab-pool-add-default-value-panel]"
          : "[data-candidate-lab-pool-add-action-value-panel]",
        ["limit", "pricing", "segment"].includes(actionType),
      );
      renderAvailability();
      return true;
    }
    const poolRemove = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_remove_entry"]',
    );
    if (poolRemove && fieldName === "pool_remove_strategy_type") {
      syncStrategyPoolRemoveEntryControls(
        poolRemove,
        strategyPoolOperationPools(state.payload),
        { preserveEntry: false },
      );
      renderAvailability();
      return true;
    }
    const poolAction = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_set_action"]',
    );
    if (poolAction && fieldName === "pool_action_strategy_type") {
      syncStrategyPoolSetActionControls(
        poolAction,
        strategyPoolOperationPools(state.payload),
        { preserveEntry: false, preserveAction: false },
      );
      renderAvailability();
      return true;
    }
    if (poolAction && fieldName === "pool_action_type") {
      setStrategyPoolActionValuePanelVisible(
        poolAction,
        ["limit", "pricing", "segment"].includes(formValue(
          poolAction,
          "pool_action_type",
        )),
      );
      renderAvailability();
      return true;
    }
    const poolReorder = field.closest?.(
      '[data-candidate-lab-workflow="strategy_pool_reorder"]',
    );
    if (poolReorder && fieldName === "pool_reorder_strategy_type") {
      syncStrategyPoolReorderControls(
        poolReorder,
        strategyPoolOperationPools(state.payload),
      );
      renderAvailability();
      return true;
    }
    const votingSearch = field.closest?.(
      '[data-candidate-lab-workflow="voting_candidate_search"]',
    );
    if (votingSearch && fieldName === "voting_strategy_type") {
      syncVotingSearchControls(votingSearch, state.payload);
      renderAvailability();
      return true;
    }
    const votingBuild = field.closest?.(
      '[data-candidate-lab-workflow="voting_candidate_build_from_search"]',
    );
    if (votingBuild && fieldName === "voting_search_id") {
      syncVotingBuildControls(
        votingBuild,
        state.payload,
        { preserveCombo: false },
      );
      renderAvailability();
      return true;
    }
    const interactiveTree = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_revision"]',
    );
    const interactiveTreeSearch = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_split_search"]',
    );
    if (
      interactiveTreeSearch
      && [
        "interactive_tree_search_source_id",
        "interactive_tree_search_node_id",
        "interactive_tree_search_mode",
      ].includes(fieldName)
    ) {
      syncInteractiveTreeSplitSearchControls(
        interactiveTreeSearch,
        state.payload,
        {
          preserveNode: fieldName === "interactive_tree_search_node_id",
        },
      );
      renderAvailability();
      return true;
    }
    if (
      interactiveTree
      && [
        "interactive_tree_operation",
        "interactive_tree_source_id",
        "interactive_tree_node_id",
      ].includes(fieldName)
    ) {
      syncInteractiveTreeRevisionControls(
        interactiveTree,
        state.payload,
        {
          preserveNode: fieldName === "interactive_tree_node_id",
        },
      );
      renderAvailability();
      return true;
    }
    const interactiveTreeFrontier = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_frontier_materialization"]',
    );
    if (
      interactiveTreeFrontier
      && fieldName === "interactive_tree_frontier_revision_id"
    ) {
      syncInteractiveTreeFrontierMaterializationControls(
        interactiveTreeFrontier,
        state.payload,
        { preserveNode: false },
      );
      renderAvailability();
      return true;
    }
    const interactiveTreeFrontierGroup = field.closest?.(
      '[data-candidate-lab-workflow="interactive_tree_frontier_group_materialization"]',
    );
    if (
      interactiveTreeFrontierGroup
      && fieldName === "interactive_tree_frontier_group_revision_id"
    ) {
      syncInteractiveTreeFrontierGroupMaterializationControls(
        interactiveTreeFrontierGroup,
        state.payload,
        { preserveNodes: false },
      );
      renderAvailability();
      return true;
    }
    return false;
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
