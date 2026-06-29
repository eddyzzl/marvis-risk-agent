import { escapeHtml } from "../ui-utils.js";

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

export function renderModelDeliveryPanel(message, options = {}) {
  const delivery = message?.metadata?.model_delivery;
  if (!delivery || typeof delivery !== "object") return "";
  const compact = options.compact === true;
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
    ["实验", selectedExperimentId || "-"],
    ["算法", recipe || "-"],
    ["目标", targetType || "-"],
    ["指标", selectionMetric || "-"],
    ["产物", artifactId || "-"],
  ].filter(([, value]) => value !== "-" || !compact).map(([label, value]) => (
    `<div class="model-delivery-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
  )).join("");
  const readinessHtml = readinessCards(delivery.readiness);
  const businessHtml = businessSignalSummary(delivery.business_signals);
  const policyHtml = policySignalSummary(delivery.policy_signals);
  const metricsHtml = metricsGrid(delivery.metrics);
  const candidatesHtml = candidateTable(delivery.candidates);
  const actionsHtml = actionTable(delivery.actions);
  const reportHtml = reportSummary(delivery.report);
  const artifactsHtml = artifactList(delivery);
  return `<div class="model-delivery-panel" data-model-delivery-source="${escapeHtml(sourceTool)}">
    <div class="model-delivery-head">
      <span>${escapeHtml(title)}</span>
      <small>${escapeHtml(readinessHeadline(delivery.readiness))}</small>
    </div>
    ${chips ? `<div class="model-delivery-chip-grid">${chips}</div>` : ""}
    ${reason ? `<div class="model-delivery-reason">${escapeHtml(reason)}</div>` : ""}
    ${readinessHtml}
    ${businessHtml}
    ${policyHtml}
    ${metricsHtml}
    ${candidatesHtml}
    ${actionsHtml}
    ${reportHtml}
    ${artifactsHtml}
  </div>`;
}

function readinessCards(items) {
  const rows = Array.isArray(items) ? items.filter((item) => item && typeof item === "object") : [];
  if (!rows.length) return "";
  return `<div class="model-delivery-readiness-grid">${rows.map((item) => {
    const status = String(item.status || "");
    const artifact = String(item.artifact || "");
    const reason = String(item.reason || "");
    return `<div class="model-delivery-readiness-card" data-readiness-kind="${escapeHtml(statusKind(status))}">
      <span>${escapeHtml(String(item.label || item.id || "交付项"))}</span>
      <strong>${escapeHtml(statusLabel(status))}</strong>
      ${artifact ? `<code>${escapeHtml(shortArtifact(artifact))}</code>` : ""}
      ${reason ? `<small>${escapeHtml(reason)}</small>` : ""}
    </div>`;
  }).join("")}</div>`;
}

function metricsGrid(metrics) {
  const metricObject = metrics && typeof metrics === "object" ? metrics : {};
  const keys = sortedMetricKeys(Object.keys(metricObject)).slice(0, 10);
  if (!keys.length) return "";
  return `<div class="model-delivery-metrics">
    <div class="modeling-section-label">最终模型指标</div>
    <div class="model-delivery-metric-grid">${keys.map((key) => (
      `<div class="model-delivery-metric"><span>${escapeHtml(key)}</span><strong>${escapeHtml(formatMetric(metricObject[key]))}</strong></div>`
    )).join("")}</div>
  </div>`;
}

function candidateTable(candidates) {
  const rows = Array.isArray(candidates) ? candidates.filter((item) => item && typeof item === "object") : [];
  if (!rows.length) return "";
  const metricKeys = sortedMetricKeys([
    ...new Set(rows.flatMap((row) => Object.keys(row.metrics && typeof row.metrics === "object" ? row.metrics : {}))),
  ]).slice(0, 6);
  const headers = ["算法", "实验", ...metricKeys, "稳定性", "特征数", "校准", "交付", "单调性", "审批"];
  return `<div class="model-delivery-table-wrap">
    <div class="modeling-section-label">候选实验</div>
    <table class="model-delivery-table">
      <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => {
        const caps = row.capabilities && typeof row.capabilities === "object" ? row.capabilities : {};
        const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : {};
        const signals = row.business_signals && typeof row.business_signals === "object" ? row.business_signals : {};
        const policy = row.policy_signals && typeof row.policy_signals === "object" ? row.policy_signals : {};
        const selected = row.selected === true;
        return `<tr${selected ? ' class="is-selected"' : ""}>
          <td>${escapeHtml(String(row.recipe || "-"))}${selected ? ' <span class="model-delivery-selected">已选</span>' : ""}</td>
          <td><code>${escapeHtml(String(row.id || "-"))}</code></td>
          ${metricKeys.map((key) => `<td class="model-delivery-num">${escapeHtml(formatMetric(metrics[key]))}</td>`).join("")}
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(signals.stability))}">${escapeHtml(String(signals.stability || "-"))}</span></td>
          <td class="model-delivery-num">${escapeHtml(formatMetric(signals.feature_count))}</td>
          <td>${escapeHtml(String(signals.calibration || "-"))}</td>
          <td>${escapeHtml(String(signals.delivery || (caps.pmml_supported && caps.handoff_supported ? "可移交" : "仅原生")))}</td>
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(policy.monotonicity_status || policy.monotonicity))}">${escapeHtml(String(policy.monotonicity || "-"))}</span></td>
          <td><span class="model-delivery-status" data-signal-kind="${escapeHtml(signalKind(policy.approval_status || policy.approval))}">${escapeHtml(String(policy.approval || "-"))}</span></td>
        </tr>`;
      }).join("")}</tbody>
    </table>
  </div>`;
}

function businessSignalSummary(signals) {
  const data = signals && typeof signals === "object" ? signals : {};
  const items = [
    ["稳定性", data.stability || "待评估", signalKind(data.stability)],
    ["特征数", data.feature_count === null || data.feature_count === undefined ? "-" : formatMetric(data.feature_count), "neutral"],
    ["校准", data.calibration || "未校准", data.calibration === "需说明" ? "warning" : "neutral"],
    ["交付", data.delivery || "待确认", data.delivery === "可移交" ? "ready" : "warning"],
  ];
  return `<div class="model-delivery-business-grid">${items.map(([label, value, kind]) => (
    `<div class="model-delivery-business-card" data-signal-kind="${escapeHtml(kind)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
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
        <strong>${escapeHtml(String(value))}</strong>
      </div>`
    )).join("")}</div>
    ${reasons.length ? `<div class="model-delivery-policy-reasons">${reasons.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
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
        const artifact = String(row.pmml_path || row.validation_task_id || "");
        return `<tr>
          <td>${escapeHtml(String(row.action || "-"))}</td>
          <td><span class="model-delivery-status" data-readiness-kind="${escapeHtml(statusKind(row.status))}">${escapeHtml(statusLabel(row.status))}</span></td>
          <td>${artifact ? `<code>${escapeHtml(shortArtifact(artifact))}</code>` : "-"}</td>
          <td>${escapeHtml(String(row.reason || ""))}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>
  </div>`;
}

function artifactList(delivery) {
  const artifacts = [
    ["原生模型", delivery.native_model_path],
    ["模型报告", delivery.report?.report_path],
    ["PMML", delivery.pmml_path],
    ["验证任务", delivery.validation_task_id],
  ].filter(([, value]) => String(value || ""));
  if (!artifacts.length) return "";
  return `<div class="model-delivery-artifacts">${artifacts.map(([label, value]) => (
    `<div><span>${escapeHtml(label)}</span><code>${escapeHtml(String(value))}</code></div>`
  )).join("")}</div>`;
}

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
      <span>${escapeHtml(String(item.section || "未命名章节"))}</span>
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

function readinessHeadline(items) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) return "等待交付状态";
  const bad = rows.filter((item) => ["unsupported", "skipped", "missing", "failed", "partial"].includes(String(item?.status || ""))).length;
  return bad ? `${bad} 项需处理/不支持` : "交付项已就绪";
}

function sortedMetricKeys(keys) {
  return [...keys].sort((a, b) => {
    const ia = METRIC_ORDER.indexOf(a);
    const ib = METRIC_ORDER.indexOf(b);
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    return a.localeCompare(b);
  });
}

function formatMetric(value) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return Math.abs(numeric) >= 100 ? numeric.toFixed(2) : numeric.toFixed(4);
  return value === undefined || value === null || value === "" ? "-" : String(value);
}

function shortArtifact(value) {
  const raw = String(value || "");
  if (raw.length <= 72) return raw;
  return `...${raw.slice(-69)}`;
}

function statusKind(status) {
  const normalized = String(status || "").toLowerCase();
  if (["succeeded", "ready", "supported"].includes(normalized)) return "ready";
  if (["skipped", "unsupported", "missing", "partial"].includes(normalized)) return "warning";
  if (["failed", "error"].includes(normalized)) return "error";
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

function statusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  const labels = {
    succeeded: "已完成",
    ready: "可交付",
    supported: "支持",
    skipped: "已跳过",
    unsupported: "不支持",
    partial: "部分完成",
    missing: "缺失",
    failed: "失败",
    error: "失败",
  };
  return labels[normalized] || (status ? String(status) : "待确认");
}
