/** Shared Candidate Lab identities and pure contract helpers. */

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
  "scorecard_model_score_evidence_build",
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

export const VOTING_SEARCH_METRICS = Object.freeze([
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

export const VOTING_RULE_ID_RE = /^candidate-rule-[0-9a-f]{32}$/;

export const VOTING_SEARCH_ID_RE = /^voting-search-[0-9a-f]{32}$/;

export const VOTING_COMBO_ID_RE = /^voting-combo-[0-9a-f]{32}$/;

export const CROSS_SEARCH_ID_RE = /^cross-search-[0-9a-f]{32}$/;

export const CROSS_PAIR_ID_RE = /^cross-pair-[0-9a-f]{32}$/;

export const CROSS_RULE_SEARCH_ID_RE = /^cross-rule-search-[0-9a-f]{32}$/;

export const CROSS_RULE_ID_RE = /^cross-rule-[0-9a-f]{32}$/;

export const INTERACTIVE_TREE_SOURCE_ID_RE = /^(?:candidate-asset-[0-9a-f]{32}|interactive-tree-revision-[0-9a-f]{32})$/;

export const INTERACTIVE_TREE_NODE_ID_RE = /^node-[0-9a-f]{20}$/;

export const INTERACTIVE_TREE_REVISION_ID_RE = /^interactive-tree-revision-[0-9a-f]{32}$/;

export const INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE = /^(?:node|leaf)-[0-9a-f]{20}$/;

export const INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE = /^interactive-tree-split-search-[0-9a-f]{32}$/;

export const INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE = /^interactive-tree-split-candidate-[0-9a-f]{32}$/;

export const STRATEGY_POOL_TYPES = Object.freeze([
  "approval",
  "reject",
  "limit",
  "pricing",
  "segmentation",
]);

export const STRATEGY_POOL_ACTION_TYPES = Object.freeze({
  approval: Object.freeze(["approval", "reject", "review"]),
  reject: Object.freeze(["approval", "reject", "review"]),
  limit: Object.freeze(["limit"]),
  pricing: Object.freeze(["pricing"]),
  segmentation: Object.freeze(["segment"]),
});

export const STRATEGY_POOL_OPERATION_WORKFLOWS = Object.freeze([
  "strategy_pool_compile",
  "strategy_pool_remove_entry",
  "strategy_pool_set_action",
  "strategy_pool_reorder",
]);

export const STRATEGY_POOL_ENTRY_ID_RE = /^pool-entry-[0-9a-f]{32}$/;

export const STRATEGY_ID_RE = /^(?:strategy-[A-Za-z0-9][A-Za-z0-9_-]*|[0-9a-f]{32})$/;

export const PROJECT_CONTEXT_FIELD_PATH_RE = /^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,7}$/;

export const PROJECT_CONTEXT_PLATFORM_FIELD_RE = /(?:^|\.)(?:artifact_id|content_hash|dataset_id|revision|revision_id|strategy_id|target_col)$/;

export const STRATEGY_POOL_APPLY_PREFIX_RE = /^[A-Za-z_][A-Za-z0-9_]{0,47}$/;

export const STRATEGY_POOL_CANDIDATE_ASSET_ID_RE = /^candidate-asset-[0-9a-f]{32}$/;

export const STRATEGY_POOL_ADD_SELECTION_RE = /^(?:automatic-tree-leaf-selection|interactive-tree-frontier-selection|interactive-tree-frontier-group-selection|cross-matrix-cell-selection|scorecard-cutoff-selection)-[0-9a-f]{32}$/;

export const STRATEGY_POOL_ADD_SOURCE_KINDS = Object.freeze({
  univariate_asset: "candidate_asset_id",
  automatic_tree_leaf_selection: "selection_id",
  interactive_tree_frontier_selection: "selection_id",
  interactive_tree_frontier_group_selection: "selection_id",
  cross_matrix_cell_selection: "selection_id",
  scorecard_cutoff_selection: "selection_id",
  voting_candidate: "candidate_asset_id",
});

export const STRATEGY_POOL_VOTING_PLACEMENTS = Object.freeze([
  "before_selected_members",
  "replace_selected_members",
]);

export const _MAX_STRATEGY_POOL_ADD_SOURCES = 140;

export const FIELD_LABELS = Object.freeze({
  action: "动作",
  approval_rate: "通过率",
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

export function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function nonEmptyText(value) {
  return typeof value === "string" ? value.trim() : "";
}

export function fieldLabel(key) {
  const normalized = String(key || "");
  return FIELD_LABELS[normalized]
    || normalized
      .split("_")
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
}

export function stablePrimitiveText(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "-";
  return String(value);
}

export function readableValue(value, depth = 0) {
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

export function collectionItems(collection) {
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

export function interactiveTreeEligiblePointers(item) {
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

export function interactiveTreeThresholdEligiblePointers(item) {
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

export function interactiveTreeFeatureEligiblePointers(item) {
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

export function interactiveTreeFrontierEligiblePointers(item) {
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

export const STRATEGY_TYPE_LABELS = Object.freeze({
  approval: "审批策略",
  reject: "拒绝策略",
  limit: "额度策略",
  pricing: "定价策略",
  segmentation: "分群策略",
});

export function formField(form, name) {
  return form?.querySelector?.(`[data-candidate-lab-field="${name}"]`) || null;
}

export function formValue(form, name) {
  return String(formField(form, name)?.value ?? "").trim();
}

export function minimalProjectedPoolAction(value, strategyType) {
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

export function projectedStrategyItems(payload) {
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
