import { escapeHtml } from "../ui-utils.js";

// VD-6: dual-encode signal state (shape, not just color) so ready/warning/
// error/neutral read clearly for color-blind users scanning the delivery
// panel. Semantic shapes: check / triangle-exclaim / cross / open circle.
const SIGNAL_GLYPH_PATHS = {
  ready: '<path d="M4 8.5 L6.7 11 L12 4.5" />',
  warning: '<path d="M8 3 L14 13 L2 13 Z" /><path d="M8 6.5 V9.5" /><circle cx="8" cy="11.3" r="0.6" fill="currentColor" stroke="none" />',
  error: '<path d="M4 4 L12 12" /><path d="M12 4 L4 12" />',
  neutral: '<circle cx="8" cy="8" r="5" />',
};

function signalGlyph(kind) {
  const path = SIGNAL_GLYPH_PATHS[kind] || SIGNAL_GLYPH_PATHS.neutral;
  return `<span class="signal-glyph" aria-hidden="true"><svg viewBox="0 0 16 16">${path}</svg></span>`;
}

const METRIC_ORDER = [
  "oot_ks",
  "test_ks",
  "oot_auc",
  "test_auc",
  "oot_rmse",
  "test_rmse",
  "oot_mae",
  "test_mae",
  "oot_r2",
  "test_r2",
  "oot_macro_auc",
  "test_macro_auc",
  "oot_logloss",
  "test_logloss",
  "oot_accuracy",
  "test_accuracy",
  "feature_count",
  "n_features",
];

// 2026-10-19 eval: the panel reads in business words. Machine keys, ids and
// stored paths stay in the payload (radio values, downloads, tooltips); the
// visible text uses these labels. Display only; no value is recomputed.
const METRIC_LABELS = {
  feature_count: "特征数",
  n_features: "特征数",
  psi_test_vs_train: "PSI（测试集 vs 训练集）",
  psi_oot_vs_train: "PSI（OOT vs 训练集）",
  overfit_flag: "过拟合标记",
  overfit_train_test_gap: "训练/测试 KS 相对差距",
  ks_ci_n_boot: "置信区间抽样次数",
};
const SPLIT_LABELS = { train: "训练集", test: "测试集", oot: "OOT", cv: "交叉验证" };
const MEASURE_LABELS = {
  ks: "KS", auc: "AUC", rmse: "RMSE", mae: "MAE", r2: "R²", logloss: "Logloss",
  accuracy: "准确率", macro_auc: "Macro-AUC", rows: "样本数", count: "样本数",
  ks_ci_low: "KS 置信下限", ks_ci_high: "KS 置信上限", ks_ci_std: "KS 置信区间标准差",
  lift_head_5: "头部 5% Lift", lift_head_10: "头部 10% Lift",
  lift_tail_5: "尾部 5% Lift", lift_tail_10: "尾部 10% Lift",
};
const RECIPE_LABELS = {
  lr: "逻辑回归", lgb: "LightGBM", xgb: "XGBoost", catboost: "CatBoost", scorecard: "评分卡",
  mlp: "神经网络", lr_regressor: "线性回归", lgb_regressor: "LightGBM 回归",
  xgb_regressor: "XGBoost 回归", mlp_regressor: "神经网络回归", lr_multiclass: "逻辑回归多分类",
  lgb_multiclass: "LightGBM 多分类", xgb_multiclass: "XGBoost 多分类", mlp_multiclass: "神经网络多分类",
};
const TARGET_TYPE_LABELS = { binary: "二分类", continuous: "连续值", multiclass: "多分类" };
const SELECTION_METRIC_LABELS = {
  manual: "用户指定",
  "test_ks(overfit-penalized)": "测试集 KS（扣除过拟合惩罚）",
  "test_rmse(overfit-penalized)": "测试集 RMSE（扣除过拟合惩罚）",
  "test_logloss(overfit-penalized)": "测试集 Logloss（扣除过拟合惩罚）",
  cv_mean_ks_minus_std_penalty: "交叉验证 KS 均值减波动惩罚",
  ks_auc: "KS 与 AUC",
};
const POLICY_RULE_LABELS = {
  require_pmml: "要求可导出 PMML",
  require_handoff: "要求可移交验证",
  require_scorecard: "要求评分卡",
  require_monotonicity: "要求单调性约束",
  prefer_scorecard: "优先评分卡",
  allow_policy_override: "允许人工放行",
  disable_quality_warnings: "关闭质量提醒",
  max_feature_count: "特征数上限",
  max_oot_psi: "OOT PSI 上限",
  overfit_gap_warn_threshold: "过拟合差警戒阈值",
  min_feature_count_warning: "特征数下限提醒",
};
const FILE_KIND_LABELS = {
  md: "Markdown 文档", json: "JSON 数据", pmml: "PMML 文件", xlsx: "Excel 报告", docx: "Word 报告",
  joblib: "模型文件", pkl: "模型文件", csv: "CSV 文件", parquet: "数据文件",
};

export function metricLabel(key) {
  const text = String(key || "");
  if (METRIC_LABELS[text]) return METRIC_LABELS[text];
  const weighted = text.startsWith("weighted_");
  const body = weighted ? text.slice("weighted_".length) : text;
  const cut = body.indexOf("_");
  const split = cut > 0 ? body.slice(0, cut) : "";
  const measure = cut > 0 ? body.slice(cut + 1) : "";
  if (SPLIT_LABELS[split] && MEASURE_LABELS[measure]) {
    const label = `${SPLIT_LABELS[split]} ${MEASURE_LABELS[measure]}`;
    return weighted ? `加权${label}` : label;
  }
  return text;
}

function isCountMetric(key) {
  const text = String(key || "");
  return ["feature_count", "n_features", "ks_ci_n_boot"].includes(text)
    || /_(count|rows|n_boot)$/.test(text);
}

function recipeLabel(code) {
  const text = String(code || "").trim();
  return RECIPE_LABELS[text] || text || "-";
}

function selectionMetricLabel(code) {
  const text = String(code || "").trim();
  if (!text) return "-";
  return SELECTION_METRIC_LABELS[text] || metricLabel(text);
}

// "实验 2 · LightGBM": the candidate's position in this comparison plus its
// algorithm. The raw experiment id only rides in the tooltip / radio value.
function experimentLabel(id, delivery = {}, recipe = "") {
  const value = String(id || "");
  const candidates = Array.isArray(delivery?.candidates)
    ? delivery.candidates.filter((item) => item && typeof item === "object")
    : [];
  const index = value ? candidates.findIndex((item) => String(item.id || "") === value) : -1;
  const algorithm = recipe || (index >= 0 ? candidates[index].recipe : "") || delivery?.recipe || "";
  const algorithmText = algorithm ? ` · ${recipeLabel(algorithm)}` : "";
  if (index >= 0) return `实验 ${index + 1}${algorithmText}`;
  return value ? `所选实验${algorithmText}` : "-";
}

function fileName(value) {
  const raw = String(value || "").trim().replaceAll("\\", "/");
  return raw.split("/").filter(Boolean).pop() || "";
}

function fileKindLabel(value) {
  const name = fileName(value);
  const dot = name.lastIndexOf(".");
  if (dot < 0) return "";
  return FILE_KIND_LABELS[name.slice(dot + 1).toLowerCase()] || "文件";
}

export function renderModelDeliveryPanel(message, options = {}) {
  const delivery = message?.metadata?.model_delivery;
  if (!delivery || typeof delivery !== "object") return "";
  const taskId = String(
    delivery.task_id
    || message?.task_id
    || message?.metadata?.task_id
    || options.taskId
    || ""
  );
  const sourceTool = String(delivery.source_tool || "");
  const title = sourceTool === "post_training_action"
    ? "训练后交付"
    : sourceTool === "select_experiment"
      ? "最终模型选择"
      : "候选模型对比";
  const selectedExperimentId = String(delivery.selected_experiment_id || "");
  const artifactId = String(delivery.artifact_id || "");
  const recipe = String(delivery.recipe || "");
  const targetType = String(delivery.target_type || "");
  const selectionMetric = String(delivery.selection_metric || "");
  const reason = String(delivery.selection_reason || "");
  const chips = [
    ["实验", selectedExperimentId ? experimentLabel(selectedExperimentId, delivery, recipe) : "-", selectedExperimentId],
    ["算法", recipe ? recipeLabel(recipe) : "-"],
    ["目标", targetType ? (TARGET_TYPE_LABELS[targetType] || targetType) : "-"],
    ["指标", selectionMetricLabel(selectionMetric)],
    ["产物", artifactId ? "模型文件已生成" : "-", artifactId],
  // An empty chip only reads as missing data (the comparison card had five
  // 「-」 chips before anything was chosen), so empty facts are left out.
  ].filter(([, value]) => value !== "-").map(([label, value, detail = ""]) => (
    `<div class="model-delivery-chip"${detail ? ` title="${escapeHtml(detail)}"` : ""}><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
  )).join("");
  const readiness = visibleReadiness(delivery);
  const readinessHtml = readinessCards(readiness);
  const businessHtml = businessSignalSummary(delivery.business_signals);
  const policyHtml = policySignalSummary(delivery.policy_signals);
  const policyDecisionHtml = policyDecisionSummary(delivery.policy_decision);
  // After a champion refit these are the delivered model's fit-only numbers;
  // the hold-out evidence sits in its own table (review M5: two blocks both
  // titled 「最终模型指标」 disagreed).
  const refit = Boolean(delivery.evaluation_evidence?.refit_applied);
  const metricsHtml = metricsGrid(delivery.metrics, refit ? "交付模型指标（train+test 重新拟合）" : "最终模型指标");
  const expectedPlanId = String(message?.metadata?.plan_id || "");
  const expectedStepId = String(message?.metadata?.step_id || "");
  const candidateSelectionGate = message?.metadata?.kind === "gate"
    && String(message?.metadata?.gate_source_tool || "") === "select_experiment"
    && sourceTool === "compare_experiments";
  const candidatesHtml = candidateTable(delivery.candidates, {
    interactive: candidateSelectionGate,
    expectedPlanId,
    expectedStepId,
    recommendedExperimentId: trustedRecommendedExperimentId(delivery),
  });
  const actionsHtml = actionTable(delivery.actions);
  const reportHtml = reportSummary(delivery.report);
  const artifactsHtml = artifactList(delivery, { taskId });
  return `<div class="model-delivery-panel" data-model-delivery-source="${escapeHtml(sourceTool)}">
    <div class="model-delivery-head">
      <span>${escapeHtml(title)}</span>
      <small>${readinessHeadlineHtml(readiness)}</small>
    </div>
    ${chips ? `<div class="model-delivery-chip-grid">${chips}</div>` : ""}
    ${reason ? `<div class="model-delivery-reason">${escapeHtml(readableReason(reason))}</div>` : ""}
    ${readinessHtml}
    ${businessHtml}
    ${policyHtml}
    ${policyDecisionHtml}
    ${metricsHtml}
    ${candidatesHtml}
    ${actionsHtml}
    ${reportHtml}
    ${artifactsHtml}
  </div>`;
}

function visibleReadiness(delivery) {
  const items = Array.isArray(delivery.readiness) ? delivery.readiness.filter((item) => item && typeof item === "object") : [];
  const hasModel = delivery.native_model_path || delivery.artifact_id
    || delivery.model_card_path || Object.keys(delivery.model_card || {}).length || delivery.selected_experiment_id;
  if (!hasModel || delivery.package_kind === "rule_only") return items;
  // Persisted messages predate these cards. Missing evidence must stay visible
  // even when every artifact in the old message was marked ready.
  const missing = [
    ["feature_time_evidence", "字段时点"],
    ["preprocessing_evidence", "预处理范围"],
    ["parameter_time_evidence", "参数历史时间"],
  ].filter(([id]) => !items.some((item) => item.id === id)).map(([id, label]) => ({
    id, label, status: "unknown", reason: "旧交付记录未提供此项证据，不能据产物就绪推定通过。",
  }));
  return [...items, ...missing];
}

function readinessCards(items) {
  const rows = Array.isArray(items) ? items.filter((item) => item && typeof item === "object") : [];
  if (!rows.length) return "";
  return `<div class="model-delivery-readiness-grid">${rows.map((item) => {
    const status = String(item.status || "");
    const artifact = String(item.artifact || "");
    const reason = String(item.reason || "");
    const kind = statusKind(status);
    return `<div class="model-delivery-readiness-card" data-readiness-kind="${escapeHtml(kind)}" data-readiness-id="${escapeHtml(String(item.id || ""))}">
      <span>${escapeHtml(String(item.label || item.id || "交付项"))}</span>
      <strong>${signalGlyph(kind)}${escapeHtml(statusLabel(status))}</strong>
      ${artifact ? `<small title="${escapeHtml(fileName(artifact))}">${escapeHtml(artifactText(artifact))}</small>` : ""}
      ${reason ? `<small>${escapeHtml(reason)}</small>` : ""}
    </div>`;
  }).join("")}</div>`;
}

function metricsGrid(metrics, title = "最终模型指标") {
  const metricObject = metrics && typeof metrics === "object" ? metrics : {};
  const keys = sortedMetricKeys(Object.keys(metricObject)).slice(0, 10);
  if (!keys.length) return "";
  return `<div class="model-delivery-metrics">
    <div class="modeling-section-label">${escapeHtml(title)}</div>
    <div class="model-delivery-metric-grid">${keys.map((key) => (
      `<div class="model-delivery-metric" title="${escapeHtml(key)}"><span>${escapeHtml(metricLabel(key))}</span><strong>${escapeHtml(formatMetric(metricObject[key], key))}</strong></div>`
    )).join("")}</div>
  </div>`;
}

function trustedRecommendedExperimentId(delivery) {
  const candidates = Array.isArray(delivery?.candidates)
    ? delivery.candidates.filter((item) => item && typeof item === "object")
    : [];
  const ids = new Set(candidates.map((item) => String(item.id || "")).filter(Boolean));
  const declared = String(
    delivery?.recommended_experiment_id
    || delivery?.best_experiment_id
    || ""
  ).trim();
  if (declared && ids.has(declared)) return declared;
  const marked = candidates.find((item) => item.recommended === true);
  const markedId = String(marked?.id || "").trim();
  return ids.has(markedId) ? markedId : "";
}

function candidateTable(candidates, options = {}) {
  const rows = Array.isArray(candidates) ? candidates.filter((item) => item && typeof item === "object") : [];
  if (!rows.length) return "";
  const interactive = options.interactive === true;
  const expectedPlanId = String(options.expectedPlanId || "");
  const expectedStepId = String(options.expectedStepId || "");
  const recommendedExperimentId = String(options.recommendedExperimentId || "");
  const selectedRow = rows.find((row) => row.selected === true && String(row.id || ""));
  const firstCandidateId = String(rows.find((row) => String(row.id || ""))?.id || "");
  const defaultExperimentId = recommendedExperimentId
    || String(selectedRow?.id || "")
    || firstCandidateId;
  const radioName = `model-candidate-${expectedPlanId || "plan"}-${expectedStepId || "step"}`;
  const metricKeys = sortedMetricKeys([
    ...new Set(rows.flatMap((row) => Object.keys(row.metrics && typeof row.metrics === "object" ? row.metrics : {}))),
  ]).slice(0, 6);
  const headers = [
    ...(interactive ? ["选择"] : []),
    "算法", "实验", ...metricKeys, "稳定性", "特征数", "校准", "交付", "单调性", "审批",
  ];
  const selectionNote = interactive
    ? `<div class="model-delivery-selection-note">${recommendedExperimentId
      ? "已默认勾选平台推荐候选；请复核后确认。"
      : "平台未提供可追溯的推荐候选，已默认勾选首项；请复核后确认。"}</div>`
    : "";
  return `<div class="model-delivery-table-wrap">
    <div class="modeling-section-label">候选实验</div>
    ${selectionNote}
    <table class="model-delivery-table">
      <thead><tr>${headers.map((header) => `<th>${escapeHtml(metricLabel(header))}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => {
        const caps = row.capabilities && typeof row.capabilities === "object" ? row.capabilities : {};
        const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : {};
        const signals = row.business_signals && typeof row.business_signals === "object" ? row.business_signals : {};
        const policy = row.policy_signals && typeof row.policy_signals === "object" ? row.policy_signals : {};
        const candidateId = String(row.id || "");
        const selected = row.selected === true;
        const recommended = candidateId && candidateId === recommendedExperimentId;
        const checked = interactive && candidateId === defaultExperimentId;
        const choiceCell = interactive
          ? `<td><input type="radio" data-model-candidate-choice="1" data-expected-plan-id="${escapeHtml(expectedPlanId)}" data-expected-step-id="${escapeHtml(expectedStepId)}" name="${escapeHtml(radioName)}" value="${escapeHtml(candidateId)}" aria-label="选择候选${escapeHtml(experimentLabel(candidateId, { candidates: rows }, row.recipe))}"${checked ? " checked" : ""}${candidateId ? "" : " disabled"}></td>`
          : "";
        const badges = [
          selected ? '<span class="model-delivery-selected">已选</span>' : "",
          recommended ? '<span class="model-delivery-selected">平台推荐</span>' : "",
        ].filter(Boolean).join(" ");
        return `<tr${selected ? ' class="is-selected"' : ""}>
          ${choiceCell}
          <td>${escapeHtml(recipeLabel(row.recipe))}${badges ? ` ${badges}` : ""}</td>
          <td title="${escapeHtml(candidateId)}">${escapeHtml(experimentLabel(candidateId, { candidates: rows }, row.recipe))}</td>
          ${metricKeys.map((key) => `<td class="model-delivery-num">${escapeHtml(formatMetric(metrics[key], key))}</td>`).join("")}
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(signals.stability))}">${signalGlyph(signalKind(signals.stability))}${escapeHtml(String(signals.stability || "-"))}</span></td>
          <td class="model-delivery-num">${escapeHtml(formatMetric(signals.feature_count, "feature_count"))}</td>
          <td>${escapeHtml(String(signals.calibration || "-"))}</td>
          <td>${escapeHtml(String(signals.delivery || (caps.pmml_supported && caps.handoff_supported ? "可移交" : "仅原生")))}</td>
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(policy.monotonicity_status || policy.monotonicity))}">${signalGlyph(signalKind(policy.monotonicity_status || policy.monotonicity))}${escapeHtml(String(policy.monotonicity || "-"))}</span></td>
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(policy.approval_status || policy.approval))}">${signalGlyph(signalKind(policy.approval_status || policy.approval))}${escapeHtml(String(policy.approval || "-"))}</span></td>
        </tr>`;
      }).join("")}</tbody>
    </table>
  </div>`;
}

function businessSignalSummary(signals) {
  const data = signals && typeof signals === "object" ? signals : {};
  const items = [
    ["稳定性", data.stability || "待评估", signalKind(data.stability)],
    ["特征数", data.feature_count === null || data.feature_count === undefined ? "-" : formatMetric(data.feature_count, "feature_count"), "neutral"],
    ["校准", data.calibration || "未校准", data.calibration === "需说明" ? "warning" : "neutral"],
    ["交付", data.delivery || "待确认", data.delivery === "可移交" ? "ready" : "warning"],
  ];
  return `<div class="model-delivery-business-grid">${items.map(([label, value, kind]) => (
    `<div class="model-delivery-business-card" data-signal-kind="${escapeHtml(kind)}">
      <span>${escapeHtml(label)}</span>
      <strong>${signalGlyph(kind)}${escapeHtml(String(value))}</strong>
    </div>`
  )).join("")}</div>`;
}

function policySignalSummary(signals) {
  const data = signals && typeof signals === "object" ? signals : {};
  if (!Object.keys(data).length) return "";
  const items = [
    ["评分卡", data.scorecard || "待评估", signalKind(data.scorecard_status || data.scorecard)],
    ["单调性", data.monotonicity || "待评估", signalKind(data.monotonicity_status || data.monotonicity)],
    ["审批建议", data.approval || "待评估", signalKind(data.approval_status || data.approval)],
  ];
  const reasons = Array.isArray(data.reasons)
    ? data.reasons.map((item) => String(item)).filter(Boolean).slice(0, 3)
    : [];
  return `<div class="model-delivery-policy">
    <div class="modeling-section-label">模型策略</div>
    <div class="model-delivery-policy-grid">${items.map(([label, value, kind]) => (
      `<div class="model-delivery-policy-card" data-signal-kind="${escapeHtml(kind)}">
        <span>${escapeHtml(label)}</span>
        <strong>${signalGlyph(kind)}${escapeHtml(String(value))}</strong>
      </div>`
    )).join("")}</div>
    ${reasons.length ? `<div class="model-delivery-policy-reasons">${reasons.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
  </div>`;
}

function policyDecisionSummary(decision) {
  const data = decision && typeof decision === "object" ? decision : {};
  if (!Object.keys(data).length) return "";
  const status = String(data.status || "not_requested");
  const policy = data.policy && typeof data.policy === "object" ? data.policy : {};
  const profile = data.profile && typeof data.profile === "object" ? data.profile : {};
  const violations = Array.isArray(data.violations)
    ? data.violations.filter((item) => item && typeof item === "object")
    : [];
  const enabledRules = Object.entries(policy)
    .filter(([, value]) => value === true || typeof value === "number" || (typeof value === "string" && value.trim()))
    .map(([key, value]) => policyRuleText(key, value))
    .slice(0, 4);
  const items = [
    ["执行结果", policyDecisionLabel(status), policyDecisionKind(status)],
    ["PMML", profile.pmml_supported ? "满足" : "未满足", profile.pmml_supported ? "ready" : "warning"],
    ["移交", profile.handoff_supported ? "满足" : "未满足", profile.handoff_supported ? "ready" : "warning"],
    ["单调性", profile.monotonicity_declared ? "已声明" : "未声明", profile.monotonicity_declared ? "ready" : "warning"],
  ];
  const violationText = violations
    .map((item) => String(item.message || item.code || ""))
    .filter(Boolean)
    .slice(0, 3);
  return `<div class="model-delivery-policy" data-policy-decision-status="${escapeHtml(status)}">
    <div class="modeling-section-label">策略执行</div>
    <div class="model-delivery-policy-grid">${items.map(([label, value, kind]) => (
      `<div class="model-delivery-policy-card" data-signal-kind="${escapeHtml(kind)}">
        <span>${escapeHtml(label)}</span>
        <strong>${signalGlyph(kind)}${escapeHtml(String(value))}</strong>
      </div>`
    )).join("")}</div>
    ${enabledRules.length ? `<div class="model-delivery-policy-reasons">${enabledRules.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
    ${violationText.length ? `<div class="model-delivery-policy-reasons">${violationText.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
    ${data.override_reason ? `<div class="model-delivery-policy-reasons"><span>放行原因: ${escapeHtml(String(data.override_reason))}</span></div>` : ""}
  </div>`;
}

function actionTable(actions) {
  const rows = Array.isArray(actions) ? actions.filter((item) => item && typeof item === "object") : [];
  if (!rows.length) return "";
  return `<div class="model-delivery-table-wrap">
    <div class="modeling-section-label">交付动作</div>
    <table class="model-delivery-table">
      <thead><tr><th>动作</th><th>状态</th><th>产物/任务</th><th>说明</th></tr></thead>
      <tbody>${rows.map((row) => {
        const artifact = row.validation_task_id
          ? "已创建验证任务"
          : row.challenger_task_id
            ? "已创建挑战者回测任务"
            : fileKindLabel(row.pmml_path || row.markdown_path || row.package_path);
        const artifactName = fileName(row.pmml_path || row.markdown_path || row.package_path);
        return `<tr>
          <td>${escapeHtml(actionLabel(row.action))}</td>
          <td><span class="model-delivery-status" data-readiness-kind="${escapeHtml(statusKind(row.status))}">${signalGlyph(statusKind(row.status))}${escapeHtml(statusLabel(row.status))}</span></td>
          <td${artifactName ? ` title="${escapeHtml(artifactName)}"` : ""}>${escapeHtml(artifact || "-")}</td>
          <td>${escapeHtml(String(row.reason || ""))}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>
  </div>`;
}

function actionLabel(action) {
  const key = String(action || "");
  return {
    export_pmml: "导出 PMML",
    handoff_to_validation: "移交验证",
    create_challenger_backtest: "创建挑战者回测",
  }[key] || key || "-";
}

function artifactList(delivery, { taskId = "" } = {}) {
  const artifacts = [
    ["模型文件", delivery.native_model_path, "native_model"],
    ["PMML", delivery.pmml_path, "pmml"],
    ["模型卡", delivery.model_card_markdown_path || delivery.model_card_path, "model_card"],
    ["审批材料包", delivery.approval_package_markdown_path || delivery.approval_package_path, "approval_package"],
    ...(delivery.approval_package_markdown_path && delivery.approval_package_path
      ? [["审批材料包（JSON）", delivery.approval_package_path]]
      : []),
    ["模型报告", delivery.report?.report_path],
    ["监控策略", delivery.monitoring_policy_markdown_path || delivery.monitoring_policy_path],
    ["冠军/挑战者对比", delivery.challenger_comparison_markdown_path || delivery.challenger_comparison_path],
    ["挑战者回测任务", delivery.challenger_task_id, "", "已创建"],
    ["挑战者回测计划", delivery.challenger_package_markdown_path || delivery.challenger_package_path],
    ["验证任务", delivery.validation_task_id, "", "已创建"],
  ].filter(([, value]) => String(value || ""));
  if (!artifacts.length) return "";
  return `<div class="model-delivery-artifacts">${artifacts.map(([label, value, downloadKind, shown]) => {
    const href = downloadKind ? taskArtifactDownloadUrl(value, taskId) : "";
    const downloadLabel = label === "PMML" ? "下载 PMML" : `下载${label}`;
    const name = shown ? "" : fileName(value);
    return `<div><span>${escapeHtml(label)}</span><strong${name ? ` title="${escapeHtml(name)}"` : ""}>${escapeHtml(shown || fileKindLabel(value) || "已生成")}</strong>${href ? (
      `<a class="button compact secondary" data-model-delivery-download="${escapeHtml(downloadKind)}" href="${escapeHtml(href)}" download>${escapeHtml(downloadLabel)}</a>`
    ) : ""}</div>`;
  }).join("")}</div>`;
}

function taskArtifactDownloadUrl(value, taskId) {
  const relativePath = taskArtifactRelativePath(value, taskId);
  return relativePath ? `api/artifacts/${encodeURIComponent(relativePath)}` : "";
}

function taskArtifactRelativePath(value, taskId) {
  const safeTaskId = String(taskId || "").trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(safeTaskId)) return "";
  const raw = String(value || "").trim().replaceAll("\\", "/");
  if (!raw) return "";
  const taskPrefix = `tasks/${safeTaskId}/`;
  const absoluteMarker = `/${taskPrefix}`;
  const markerIndex = raw.lastIndexOf(absoluteMarker);
  let relativePath = "";
  if (markerIndex >= 0) {
    relativePath = raw.slice(markerIndex + 1);
  } else if (raw.startsWith(taskPrefix)) {
    relativePath = raw;
  } else if (raw.startsWith("modeling_artifacts/")) {
    relativePath = `${taskPrefix}${raw}`;
  } else if (!raw.includes("/")) {
    relativePath = `${taskPrefix}modeling_artifacts/${raw}`;
  }
  const segments = relativePath.split("/");
  if (
    !relativePath.startsWith(taskPrefix)
    || segments.some((segment) => !segment || segment === "." || segment === "..")
  ) return "";
  return relativePath;
}

// Section codes are the report contract; the reader sees the chapter names.
const REPORT_SECTION_LABELS = Object.freeze({
  sample_analysis: "样本分析",
  vintage: "Vintage 分析",
  amount_bin: "金额分箱",
  low_pricing: "低定价压力测试",
  product_list: "使用产品清单",
  development_refit_scope: "重新拟合说明",
});

function reportSummary(report) {
  if (!report || typeof report !== "object") return "";
  const total = Number(report.total_sections || 0);
  const available = Number(report.available_sections || 0);
  const skipped = Number(report.skipped_sections || 0);
  const sections = Array.isArray(report.sections)
    ? report.sections.filter((item) => item && typeof item === "object").slice(0, 8)
    : [];
  const sectionHtml = sections.map((item) => (
    `<div class="model-delivery-report-section" data-available="${item.available ? "true" : "false"}">
      <span>${escapeHtml(REPORT_SECTION_LABELS[item.section] || String(item.section || "未命名章节"))}</span>
      <strong>${item.available ? "可生成" : "缺输入/跳过"}</strong>
      ${item.reason ? `<small>${escapeHtml(String(item.reason))}</small>` : ""}
    </div>`
  )).join("");
  return `<div class="model-delivery-report-summary">
    <div class="modeling-section-label">报告就绪度</div>
    <div class="model-delivery-report-status">
      <strong>${escapeHtml(`${available}/${total || available} 章节可生成`)}</strong>
      <span>${skipped ? `${skipped} 个章节缺输入/跳过` : "报告章节完整"}</span>
    </div>
    ${sectionHtml ? `<div class="model-delivery-report-grid">${sectionHtml}</div>` : ""}
  </div>`;
}

function readinessHeadlineHtml(items) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) return "<span>等待交付状态</span>";
  const bad = rows.filter((item) => (
    ["unsupported", "skipped", "missing", "failed", "partial", "warn", "fail", "error", "needs_policy", "inferred", "exploration"].includes(String(item?.status || "").toLowerCase())
  )).length;
  const unknown = rows.filter((item) => String(item?.status || "").toLowerCase() === "unknown").length;
  return [unknown ? `${unknown} 项证据未知` : "", bad ? `${bad} 项需处理/不支持` : "", !unknown && !bad ? "交付产物已就绪" : "", "业务验收未在此确认"]
    .filter(Boolean).map((part) => `<span>${escapeHtml(part)}</span>`).join("；");
}

function sortedMetricKeys(keys) {
  return [...keys].sort((a, b) => {
    const ia = METRIC_ORDER.indexOf(a);
    const ib = METRIC_ORDER.indexOf(b);
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    return a.localeCompare(b);
  });
}

function formatMetric(value, key = "") {
  if (typeof value === "boolean") return value ? "是" : "否";
  const numeric = value === null || value === "" ? Number.NaN : Number(value);
  if (Number.isFinite(numeric)) {
    if (isCountMetric(key) && Number.isInteger(numeric)) return String(numeric);
    return Math.abs(numeric) >= 100 ? numeric.toFixed(2) : numeric.toFixed(4);
  }
  return value === undefined || value === null || value === "" ? "-" : String(value);
}

// A readiness artifact is either a stored file (named by its kind) or a
// created task (validation / backtest).
function artifactText(value) {
  return fileKindLabel(value) || "已创建";
}

function readableReason(text) {
  let value = String(text || "")
    .replace(/平台推荐为\s*`?experiment_[0-9a-f]{8,}`?/g, "平台推荐的是另一个实验")
    .replace(/\s*`?(?:experiment|artifact)_[0-9a-f]{8,}`?\s*/g, "");
  for (const [code, label] of Object.entries(SELECTION_METRIC_LABELS)) {
    if (code !== "manual") value = value.split(code).join(label);
  }
  return value.replace(/\b(?:weighted_)?(?:train|test|oot|cv)_[a-z0-9_]+\b/g, (code) => metricLabel(code));
}

function policyRuleText(key, value) {
  const label = POLICY_RULE_LABELS[key] || key;
  if (value === true) return label;
  return `${label}: ${value}`;
}

function statusKind(status) {
  const normalized = String(status || "").toLowerCase();
  if (["succeeded", "ready", "supported", "pass"].includes(normalized)) return "ready";
  if (["skipped", "unsupported", "missing", "partial", "warn", "warning", "needs_policy", "unknown", "inferred", "exploration"].includes(normalized)) return "warning";
  if (["failed", "error", "fail"].includes(normalized)) return "error";
  return "neutral";
}

function signalKind(value) {
  const text = String(value || "").toLowerCase();
  if (["ready", "supported"].includes(text)) return "ready";
  if (["warning", "partial", "missing", "unsupported"].includes(text)) return "warning";
  if (["error", "failed"].includes(text)) return "error";
  if (["稳定", "可移交"].includes(String(value || ""))) return "ready";
  if (["关注", "需复核", "高风险", "需说明", "仅原生", "不可交付", "需确认", "仅实验候选"].includes(String(value || ""))) return "warning";
  if (text.includes("risk") || text.includes("高")) return "warning";
  return "neutral";
}

function policyDecisionKind(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "accepted") return "ready";
  if (normalized === "overridden") return "warning";
  if (normalized === "blocked") return "error";
  return "neutral";
}

function policyDecisionLabel(status) {
  const labels = {
    accepted: "已通过",
    overridden: "已人工放行",
    blocked: "未通过",
    not_requested: "未启用",
  };
  return labels[String(status || "").toLowerCase()] || String(status || "待评估");
}

function statusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  const labels = {
    succeeded: "已完成",
    ready: "可交付",
    supported: "支持",
    pass: "通过",
    warn: "需复核",
    warning: "需复核",
    skipped: "已跳过",
    unsupported: "不支持",
    partial: "部分完成",
    missing: "缺失",
    failed: "失败",
    error: "失败",
    unknown: "未知",
    inferred: "仅推断",
    verified: "已核验记录时点",
    training_only: "仅外层训练成员",
    row_local: "已记录逐行变换",
    exploration: "探索范围",
  };
  return labels[normalized] || (status ? String(status) : "待确认");
}
