/** Candidate Lab form collection and typed request validation. */

import {
  CROSS_PAIR_ID_RE,
  CROSS_RULE_ID_RE,
  CROSS_RULE_SEARCH_ID_RE,
  CROSS_SEARCH_ID_RE,
  INTERACTIVE_TREE_FRONTIER_SOURCE_NODE_ID_RE,
  INTERACTIVE_TREE_NODE_ID_RE,
  INTERACTIVE_TREE_REVISION_ID_RE,
  INTERACTIVE_TREE_SOURCE_ID_RE,
  INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_RE,
  INTERACTIVE_TREE_SPLIT_SEARCH_ID_RE,
  PROJECT_CONTEXT_FIELD_PATH_RE,
  PROJECT_CONTEXT_PLATFORM_FIELD_RE,
  STRATEGY_CANDIDATE_LAB_WORKFLOWS,
  STRATEGY_ID_RE,
  STRATEGY_POOL_ACTION_TYPES,
  STRATEGY_POOL_ADD_SELECTION_RE,
  STRATEGY_POOL_ADD_SOURCE_KINDS,
  STRATEGY_POOL_APPLY_PREFIX_RE,
  STRATEGY_POOL_CANDIDATE_ASSET_ID_RE,
  STRATEGY_POOL_ENTRY_ID_RE,
  STRATEGY_POOL_TYPES,
  STRATEGY_POOL_VOTING_PLACEMENTS,
  VOTING_COMBO_ID_RE,
  VOTING_RULE_ID_RE,
  VOTING_SEARCH_ID_RE,
  VOTING_SEARCH_METRICS,
  fieldLabel,
  formField,
  formValue,
  minimalProjectedPoolAction,
  nonEmptyText,
} from "./strategy_candidate_lab_contracts.js";

export { STRATEGY_CANDIDATE_LAB_WORKFLOWS };

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

function collectScorecardModelScoreEvidenceInputs(form) {
  const features = uniqueValues(
    splitValues(formValue(form, "scorecard_model_features")),
    "评分卡建模特征",
  );
  if (!features.length || features.length > 50) {
    throw new Error("评分卡建模特征必须包含 1 到 50 个独立字段。");
  }
  const inputs = {
    features,
    seed: parseRequiredInteger(
      formValue(form, "scorecard_model_seed"),
      "评分卡随机种子",
      { min: 0, max: 4_294_967_295 },
    ),
    max_iter: parseRequiredInteger(
      formValue(form, "scorecard_model_max_iter"),
      "评分卡最大迭代次数",
      { min: 20, max: 5_000 },
    ),
    scorecard_max_bins: parseRequiredInteger(
      formValue(form, "scorecard_model_max_bins"),
      "评分卡最大分箱数",
      { min: 2, max: 20 },
    ),
  };
  optionalText(
    inputs,
    "sample_weight_col",
    formValue(form, "scorecard_model_sample_weight_col"),
  );
  if (inputs.sample_weight_col && features.includes(inputs.sample_weight_col)) {
    throw new Error("评分卡样本权重列不能同时作为建模特征。");
  }
  return inputs;
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
    scorecard_model_score_evidence_build:
      collectScorecardModelScoreEvidenceInputs,
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