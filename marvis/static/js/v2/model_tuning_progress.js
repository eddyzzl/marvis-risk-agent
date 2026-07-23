import { escapeHtml } from "../ui-utils.js";

const ALGORITHM_LABELS = {
  cat: "CatBoost",
  catboost: "CatBoost",
  lgb: "LightGBM",
  lightgbm: "LightGBM",
  xgb: "XGBoost",
  xgboost: "XGBoost",
};

const STAGE_LABELS = {
  preparing: "准备训练数据",
  coarse: "粗搜",
  fine: "细搜",
};

const STATUS_ALIASES = {
  running: "running",
  succeeded: "succeeded",
  success: "succeeded",
  completed: "succeeded",
  failed: "failed",
  cancelled: "cancelled",
  canceled: "cancelled",
  interrupted: "interrupted",
};

const STATUS_LABELS = {
  running: "正在执行 · 调参",
  succeeded: "调参已完成",
  failed: "调参失败",
  cancelled: "调参已取消",
  interrupted: "调参已中断",
};

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function positiveInteger(value) {
  const parsed = finiteNumber(value);
  return parsed !== null && parsed > 0 ? Math.trunc(parsed) : null;
}

function nonNegativeInteger(value) {
  const parsed = finiteNumber(value);
  return parsed !== null && parsed >= 0 ? Math.trunc(parsed) : null;
}

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function progressPercent(payload) {
  const explicit = finiteNumber(payload?.percent);
  if (explicit !== null) return clamp(explicit, 0, 100);
  const completed = nonNegativeInteger(payload?.completed_trials);
  const total = positiveInteger(payload?.total_trials);
  if (completed === null || total === null) return 0;
  return clamp((completed / total) * 100, 0, 100);
}

function algorithmLabel(value) {
  const raw = String(value || "").trim();
  if (!raw) return "当前算法";
  return ALGORITHM_LABELS[raw.toLowerCase()] || raw;
}

function scoreText(value) {
  const parsed = finiteNumber(value);
  if (parsed === null) return "—";
  return parsed.toFixed(4);
}

function scoreFromBest(value) {
  if (value && typeof value === "object") {
    return value.selection_score ?? value.score ?? value.test_ks ?? null;
  }
  return value;
}

function normalizeBestByAlgorithm(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [];
  return Object.entries(value).map(([algorithm, result]) => ({
    algorithm: algorithmLabel(algorithm),
    score: scoreFromBest(result),
  }));
}

// Progress may arrive either as a plan step's `progress` payload or inside an
// Agent message (`metadata.kind=tool_progress`, `metadata.progress={...}`).
// Keep the compatibility logic here so the rail and the timeline cannot drift.
export function normalizeModelTuningProgress(value) {
  const metadata = value?.metadata && typeof value.metadata === "object"
    ? value.metadata
    : value;
  const nested = metadata?.progress && typeof metadata.progress === "object"
    ? metadata.progress
    : metadata;
  const isTuning = String(nested?.kind || "") === "model_tuning"
    || (
      String(metadata?.kind || "") === "tool_progress"
      && Boolean(nested?.algorithm)
      && (nested?.trial_total !== undefined || nested?.total_trials !== undefined)
    );
  if (!isTuning) return null;

  const trial = nonNegativeInteger(nested?.trial);
  const trialTotal = positiveInteger(nested?.trial_total);
  const completedTrials = nonNegativeInteger(nested?.completed_trials);
  const totalTrials = positiveInteger(nested?.total_trials);
  const algorithmIndex = positiveInteger(nested?.algorithm_index);
  const algorithmTotal = positiveInteger(nested?.algorithm_total);
  const rawStatus = String(metadata?.status ?? nested?.status ?? "").trim().toLowerCase();
  const status = STATUS_ALIASES[rawStatus] || "running";
  return {
    algorithm: algorithmLabel(nested?.algorithm),
    algorithmIndex,
    algorithmTotal,
    trial,
    trialTotal,
    stage: STAGE_LABELS[String(nested?.stage || "").trim()]
      || String(nested?.stage || "").trim(),
    completedTrials,
    totalTrials,
    percent: progressPercent(nested),
    selectionScore: nested?.selection_score,
    testKs: nested?.test_ks,
    bestSelectionScore: nested?.best_selection_score,
    bestTestKs: nested?.best_test_ks,
    bestByAlgorithm: normalizeBestByAlgorithm(nested?.best_by_algorithm),
    status,
    statusLabel: STATUS_LABELS[status],
    terminal: status !== "running",
  };
}

function countText(current, total, unit) {
  if (current === null && total === null) return `— ${unit}`;
  if (total === null) return `${current ?? 0} ${unit}`;
  return `${current ?? 0} / ${total} ${unit}`;
}

function algorithmPositionText(progress) {
  if (progress.algorithmIndex === null || progress.algorithmTotal === null) return "";
  return `算法 ${progress.algorithmIndex} / ${progress.algorithmTotal}`;
}

function bestScore(progress) {
  const direct = finiteNumber(progress.bestSelectionScore);
  if (direct !== null) return direct;
  const scores = progress.bestByAlgorithm
    .map((item) => finiteNumber(item.score))
    .filter((value) => value !== null);
  if (scores.length) return Math.max(...scores);
  // Backward compatibility for progress producers from before the explicit
  // best_selection_score field existed.
  return finiteNumber(progress.selectionScore);
}

function compactProgressHtml(progress) {
  const algorithmPosition = algorithmPositionText(progress);
  return [
    '<div class="model-tuning-progress model-tuning-progress--compact" data-model-tuning-progress="compact"',
    ` data-status="${escapeHtml(progress.status)}" role="progressbar" aria-label="${escapeHtml(progress.statusLabel)}"`,
    ` aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(progress.percent)}">`,
    '<div class="model-tuning-progress__compact-head">',
    `<strong title="${escapeHtml(progress.algorithm)}">${escapeHtml(progress.algorithm)}</strong>`,
    algorithmPosition ? `<span>${escapeHtml(algorithmPosition)}</span>` : "",
    "</div>",
    '<div class="model-tuning-progress__track" aria-hidden="true">',
    `<span style="width:${progress.percent.toFixed(2)}%"></span>`,
    "</div>",
    '<div class="model-tuning-progress__compact-foot">',
    `<span>${escapeHtml(countText(progress.trial, progress.trialTotal, "轮"))}</span>`,
    `<span>${escapeHtml(countText(progress.completedTrials, progress.totalTrials, "总轮次"))}</span>`,
    `<span>最佳 ${scoreText(bestScore(progress))}</span>`,
    "</div>",
    "</div>",
  ].join("");
}

function bestAlgorithmsHtml(progress) {
  if (!progress.bestByAlgorithm.length) return "";
  const items = progress.bestByAlgorithm.map((item) => [
    '<div class="model-tuning-progress__best-item">',
    `<span title="${escapeHtml(item.algorithm)}">${escapeHtml(item.algorithm)}</span>`,
    `<strong>${scoreText(item.score)}</strong>`,
    "</div>",
  ].join(""));
  return [
    '<div class="model-tuning-progress__best" aria-label="各算法当前最佳选择分">',
    '<span class="model-tuning-progress__best-label">各算法最佳</span>',
    ...items,
    "</div>",
  ].join("");
}

function cardProgressHtml(progress) {
  const algorithmPosition = algorithmPositionText(progress);
  const stage = progress.stage || "正在搜索更优参数";
  const algorithmState = algorithmPosition
    || (progress.status === "succeeded" ? "已完成" : progress.terminal ? "已停止" : "执行中");
  return [
    '<section class="model-tuning-progress model-tuning-progress--card" data-model-tuning-progress="card"',
    ` data-status="${escapeHtml(progress.status)}" role="progressbar" aria-label="${escapeHtml(progress.statusLabel)}"`,
    ` aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(progress.percent)}">`,
    '<header class="model-tuning-progress__head">',
    '<div class="model-tuning-progress__title">',
    `<span>${escapeHtml(progress.statusLabel)}</span>`,
    `<strong title="${escapeHtml(progress.algorithm)}">${escapeHtml(progress.algorithm)}</strong>`,
    "</div>",
    `<span class="model-tuning-progress__stage" title="${escapeHtml(stage)}">${escapeHtml(stage)}</span>`,
    "</header>",
    '<div class="model-tuning-progress__metrics">',
    '<div><span>当前算法</span>',
    `<strong>${escapeHtml(algorithmState)}</strong></div>`,
    '<div><span>当前轮次</span>',
    `<strong>${escapeHtml(countText(progress.trial, progress.trialTotal, "轮"))}</strong></div>`,
    '<div><span>总进度</span>',
    `<strong>${escapeHtml(countText(progress.completedTrials, progress.totalTrials, "轮"))}</strong></div>`,
    '<div><span>最佳选择分</span>',
    `<strong>${scoreText(bestScore(progress))}</strong></div>`,
    "</div>",
    '<div class="model-tuning-progress__bar-row">',
    '<div class="model-tuning-progress__track" aria-hidden="true">',
    `<span style="width:${progress.percent.toFixed(2)}%"></span>`,
    "</div>",
    `<strong>${progress.percent.toFixed(1)}%</strong>`,
    "</div>",
    progress.selectionScore === null || progress.selectionScore === undefined
      ? ""
      : `<p class="model-tuning-progress__test-ks">当前轮选择分 <strong>${scoreText(progress.selectionScore)}</strong>${progress.testKs === null || progress.testKs === undefined ? "" : ` · Test KS <strong>${scoreText(progress.testKs)}</strong>`}${progress.bestTestKs === null || progress.bestTestKs === undefined ? "" : ` · 最佳 Test KS <strong>${scoreText(progress.bestTestKs)}</strong>`}</p>`,
    bestAlgorithmsHtml(progress),
    "</section>",
  ].join("");
}

export function renderModelTuningProgress(value, { compact = false } = {}) {
  const progress = normalizeModelTuningProgress(value);
  if (!progress) return "";
  return compact ? compactProgressHtml(progress) : cardProgressHtml(progress);
}

export function renderModelTuningMessageProgress(message) {
  if (String(message?.metadata?.kind || "") !== "tool_progress") return "";
  return renderModelTuningProgress(message);
}

// Once a real execution-progress event has landed, the earlier optimistic
// empty streaming message no longer means "the Agent is composing a reply".
// Remove only placeholders that precede the latest tuning event; an older run's
// progress must never hide a new thinking placeholder from a later user turn.
export function hideSupersededTuningThinking(messages = []) {
  if (!Array.isArray(messages) || !messages.length) return [];
  let latestProgressIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (normalizeModelTuningProgress(messages[index])
        && String(messages[index]?.metadata?.kind || "") === "tool_progress") {
      latestProgressIndex = index;
      break;
    }
  }
  if (latestProgressIndex < 0) return messages;
  return messages.filter((message, index) => {
    if (index >= latestProgressIndex) return true;
    const metadata = message?.metadata || {};
    const optimisticThinking = message?.role !== "user"
      && metadata.optimistic === true
      && metadata.streaming === true
      && !String(message?.content || "").trim();
    return !optimisticThinking;
  });
}
