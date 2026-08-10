/** Candidate Lab structured payload to escaped HTML presenters. */

import { escapeHtml } from "../ui-utils.js";

import {
  INTERACTIVE_TREE_REVISION_ID_RE,
  INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE,
  INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE,
  STRATEGY_TYPE_LABELS,
  collectionItems,
  fieldLabel,
  interactiveTreeEligiblePointers,
  interactiveTreeFrontierEligiblePointers,
  interactiveTreeThresholdEligiblePointers,
  isRecord,
  nonEmptyText,
  projectedStrategyItems,
  readableValue,
  stablePrimitiveText,
} from "./strategy_candidate_lab_contracts.js";

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

function safeDownloadUrl(value) {
  const url = nonEmptyText(value);
  return url.startsWith("/api/") ? url.slice(1) : "";
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
