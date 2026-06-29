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
  const metricsHtml = metricsGrid(delivery.metrics);
  const candidatesHtml = candidateTable(delivery.candidates);
  const actionsHtml = actionTable(delivery.actions);
  const artifactsHtml = artifactList(delivery);
  return `<div class="model-delivery-panel" data-model-delivery-source="${escapeHtml(sourceTool)}">
    <div class="model-delivery-head">
      <span>${escapeHtml(title)}</span>
      <small>${escapeHtml(readinessHeadline(delivery.readiness))}</small>
    </div>
    ${chips ? `<div class="model-delivery-chip-grid">${chips}</div>` : ""}
    ${reason ? `<div class="model-delivery-reason">${escapeHtml(reason)}</div>` : ""}
    ${readinessHtml}
    ${metricsHtml}
    ${candidatesHtml}
    ${actionsHtml}
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
  const headers = ["算法", "实验", ...metricKeys, "PMML", "验证移交"];
  return `<div class="model-delivery-table-wrap">
    <div class="modeling-section-label">候选实验</div>
    <table class="model-delivery-table">
      <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => {
        const caps = row.capabilities && typeof row.capabilities === "object" ? row.capabilities : {};
        const metrics = row.metrics && typeof row.metrics === "object" ? row.metrics : {};
        const selected = row.selected === true;
        return `<tr${selected ? ' class="is-selected"' : ""}>
          <td>${escapeHtml(String(row.recipe || "-"))}${selected ? ' <span class="model-delivery-selected">已选</span>' : ""}</td>
          <td><code>${escapeHtml(String(row.id || "-"))}</code></td>
          ${metricKeys.map((key) => `<td class="model-delivery-num">${escapeHtml(formatMetric(metrics[key]))}</td>`).join("")}
          <td>${caps.pmml_supported ? "是" : "否"}</td>
          <td>${caps.handoff_supported ? "是" : "否"}</td>
        </tr>`;
      }).join("")}</tbody>
    </table>
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
    ["PMML", delivery.pmml_path],
    ["验证任务", delivery.validation_task_id],
  ].filter(([, value]) => String(value || ""));
  if (!artifacts.length) return "";
  return `<div class="model-delivery-artifacts">${artifacts.map(([label, value]) => (
    `<div><span>${escapeHtml(label)}</span><code>${escapeHtml(String(value))}</code></div>`
  )).join("")}</div>`;
}

function readinessHeadline(items) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) return "等待交付状态";
  const bad = rows.filter((item) => ["unsupported", "skipped", "missing", "failed"].includes(String(item?.status || ""))).length;
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
  if (["skipped", "unsupported", "missing"].includes(normalized)) return "warning";
  if (["failed", "error"].includes(normalized)) return "error";
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
    missing: "缺失",
    failed: "失败",
    error: "失败",
  };
  return labels[normalized] || (status ? String(status) : "待确认");
}
