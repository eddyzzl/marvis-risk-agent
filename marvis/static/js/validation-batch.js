import { resolveBatchItemStatus, resolveBatchStatus, statusPresentation } from "./task-status.js";

export const validationBatchTaskType = "validation_batch";

const STATUS_WORDS = Object.fromEntries(
  ["running", "awaiting_confirm", "needs_review", "done", "failed"].map((key) => [key, statusPresentation(key).label]),
);

const terminalBatchStatuses = new Set([
  "completed",
  "failed",
  "cancelled",
]);

const stressLabels = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
};

// Stage suffix for the chip ("执行中 · PMML打分"); the status word itself comes
// from js/task-status.js.
const stageLabels = {
  queued: "排队中",
  created: "待开始",
  batch_created: "待开始",
  scan: "材料扫描",
  input_confirmation: "确认字段",
  reproducibility: "可复现性",
  pmml_scoring: "PMML打分",
  notebook: "Notebook 复现",
  metrics: "效果评估",
  word_conclusion_draft: "报告结论",
  word_conclusion_generated: "报告结论",
  report_conclusion: "报告结论",
  report: "生成报告",
  summary: "汇总",
  completed: "已完成",
  failure: "失败",
};

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function textValue(value) {
  return value === null || value === undefined ? "" : String(value).trim();
}

function firstText(...values) {
  for (const value of values) {
    const normalized = textValue(value);
    if (normalized) return normalized;
  }
  return "";
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const normalized = Number(value);
  return Number.isFinite(normalized) ? normalized : null;
}

function booleanValue(value) {
  return value === true || value === 1 || String(value).toLowerCase() === "true";
}

function safeLink(value) {
  const normalized = textValue(value);
  if (!normalized) return "";
  if (/^https?:\/\//i.test(normalized)) return normalized;
  if (/^\/(?!\/)/.test(normalized)) return normalized.slice(1);
  if (normalized.startsWith("?")) return normalized;
  return "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

// D16 (2026-10-14 redesign, batch 4): a batch item's word and tone come from
// the shared resolver. The overview counters keep their buckets; a cancelled
// item still counts in the 失败 column as before, while its chip is neutral.
function itemStatus(item, batchStatus = "") {
  return statusPresentation(resolveBatchItemStatus(item, { batchStatus }));
}

function statusBucket(item) {
  const { key } = itemStatus(item);
  if (["failed", "partial_failure", "needs_reconciliation", "cancelled"].includes(key)) return "failed";
  if (["awaiting_confirm", "awaiting_next", "needs_review"].includes(key)) return "awaitingConfirmation";
  if (key === "done") return "succeeded";
  if (key === "running") return "running";
  return "queued";
}

function normalizeBatchItem(rawItem, parentTaskId, fallbackOrdinal) {
  const item = objectValue(rawItem);
  const childTaskId = firstText(item.child_task_id, item.childTaskId, item.task_id, item.taskId);
  const itemId = firstText(item.id, item.item_id, item.itemId);
  const outcome = firstText(item.outcome, item.result).toLowerCase();
  const status = firstText(item.status, item.state, outcome || "queued").toLowerCase();
  const explicitManualUrl = safeLink(firstText(
    item.manual_review_url,
    item.manualReviewUrl,
    item.review_url,
  ));
  const manualReviewUrl = explicitManualUrl || (
    parentTaskId && childTaskId
      ? `?task=${encodeURIComponent(parentTaskId)}&item=${encodeURIComponent(childTaskId)}`
      : ""
  );
  return {
    id: itemId,
    parentTaskId,
    childTaskId,
    ordinal: Number.isInteger(Number(item.ordinal)) ? Number(item.ordinal) : fallbackOrdinal,
    modelName: firstText(item.model_name, item.modelName, item.name, `模型 ${fallbackOrdinal}`),
    modelVersion: firstText(item.model_version, item.modelVersion, item.version),
    status,
    stage: firstText(item.stage, item.current_stage, item.currentStage),
    inputContractStatus: firstText(
      item.input_contract_status,
      item.inputContractStatus,
    ).toLowerCase(),
    ootKs: finiteNumber(item.oot_ks ?? item.ootKs),
    ootPsi: finiteNumber(item.oot_psi ?? item.ootPsi),
    pmmlStatus: firstText(item.pmml_status, item.pmmlStatus).toLowerCase(),
    stressRisk: firstText(item.stress_risk, item.stressRisk).toLowerCase(),
    reportComplete: booleanValue(item.report_complete ?? item.reportComplete),
    outcome,
    errorCode: firstText(item.error_code, item.errorCode),
    errorMessage: firstText(
      item.error_message,
      item.errorMessage,
      item.failure_reason,
      item.failureReason,
      item.reason,
    ),
    contractUrl: safeLink(firstText(
      item.input_contract_url,
      item.inputContractUrl,
      item.validation_input_contract_url,
      item.contract_url,
      item.contractUrl,
    )),
    wordReportDownloadUrl: safeLink(firstText(
      item.word_report_download_url,
      item.wordReportDownloadUrl,
    )),
    analysisDownloadUrl: safeLink(firstText(
      item.analysis_download_url,
      item.analysisDownloadUrl,
    )),
    pendingReportDraft: booleanValue(item.pending_report_draft ?? item.pendingReportDraft),
    repairRecords: (Array.isArray(item.repair_records) ? item.repair_records : [])
      .map(objectValue)
      .map((record) => ({
        finding: textValue(record.finding),
        action: textValue(record.action),
        verified: record.verified === true,
        undone: record.undone === true,
      })),
    repairCount: Number.isInteger(Number(item.repair_count)) ? Number(item.repair_count) : 0,
    manualReviewUrl,
  };
}

// B-line extension (2026-10-23 spec §5.4): one line per child with what the
// platform handled, from the child's repair reports (never free text).
export function batchItemRepairHtml(item) {
  const records = Array.isArray(item?.repairRecords) ? item.repairRecords : [];
  if (!records.length) return "";
  const latest = records[records.length - 1];
  const state = latest.undone ? "已撤销" : latest.verified ? "已处理" : "处理没成功，已还原";
  const count = Number(item.repairCount) > 1 ? ` ${Number(item.repairCount)} 项` : "";
  const text = latest.verified && !latest.undone ? latest.action : latest.finding;
  return `<small class="validation-batch-repair" data-repair-state="${latest.undone ? "undone" : latest.verified ? "verified" : "rolled_back"}">`
    + `${escapeHtml(state + count)}：${escapeHtml(text)}</small>`;
}

export function parseTaskDeepLink(search = "") {
  const params = new URLSearchParams(String(search || "").replace(/^\?/, ""));
  return {
    taskId: textValue(params.get("task")),
    itemId: textValue(params.get("item")),
  };
}

export function preferredStartupTaskId(deepLink, storedTaskId = "") {
  return firstText(objectValue(deepLink).taskId, storedTaskId);
}

export function taskSelectionSearch(currentSearch, { taskId = "", itemId = "" } = {}) {
  const params = new URLSearchParams(String(currentSearch || "").replace(/^\?/, ""));
  const normalizedTaskId = textValue(taskId);
  const normalizedItemId = textValue(itemId);
  if (normalizedTaskId) {
    params.set("task", normalizedTaskId);
    if (normalizedItemId) params.set("item", normalizedItemId);
    else params.delete("item");
  } else {
    params.delete("task");
    params.delete("item");
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function syncTaskDeepLink(historyLike, locationLike, selection = {}) {
  if (!historyLike?.replaceState || !locationLike) return "";
  const nextSearch = taskSelectionSearch(locationLike.search, selection);
  const currentSearch = String(locationLike.search || "");
  if (nextSearch === currentSearch) return nextSearch;
  const path = locationLike.pathname || "/";
  const hash = locationLike.hash || "";
  try {
    historyLike.replaceState(historyLike.state, "", `${path}${nextSearch}${hash}`);
  } catch (_) {
    // Some embedded or sandboxed browsers block history mutation.
  }
  return nextSearch;
}

export function isValidationBatchTask(task) {
  return task?.task_type === validationBatchTaskType;
}

function formatLocalIsoDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatValidationBatchDate(value) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return formatLocalIsoDate(value);
  }
  const raw = textValue(value);
  const parsed = raw ? new Date(raw) : new Date();
  if (Number.isNaN(parsed.getTime())) return formatLocalIsoDate(new Date());
  return formatLocalIsoDate(parsed);
}

export function formatValidationBatchTaskName(createdAt, itemCount) {
  const dateText = formatValidationBatchDate(createdAt);
  const count = Number(itemCount);
  if (Number.isInteger(count) && count > 0) {
    return `${dateText} 模型验证批次 (${count}个模型)`;
  }
  return `${dateText} 模型验证批次`;
}

export function usesAgentValidationWorkbench(task) {
  return isValidationBatchTask(task) && textValue(task?.run_mode).toLowerCase() === "agent";
}

export function resolveProjectedChildTaskId(batchPayload, requestedId = "") {
  const batch = batchPayload?.counts && Array.isArray(batchPayload?.items)
    ? batchPayload
    : normalizeValidationBatchPayload(batchPayload);
  const requested = textValue(requestedId);
  const match = (batch.items || []).find((item) => (
    item.childTaskId === requested || item.id === requested
  ));
  return match?.childTaskId || batch.items?.[0]?.childTaskId || "";
}

export function normalizeValidationBatchPayload(rawPayload = {}, fallbackParentTaskId = "") {
  const payload = objectValue(rawPayload);
  const batch = objectValue(payload.batch);
  const batchSource = Object.keys(batch).length ? batch : payload;
  const parentTaskId = firstText(
    batchSource.parent_task_id,
    batchSource.parentTaskId,
    batchSource.task_id,
    batchSource.taskId,
    fallbackParentTaskId,
  );
  const rawItems = Array.isArray(payload.items)
    ? payload.items
    : Array.isArray(payload.models)
      ? payload.models
      : Array.isArray(batchSource.items)
        ? batchSource.items
        : [];
  const items = rawItems
    .map((item, index) => normalizeBatchItem(item, parentTaskId, index + 1))
    .sort((left, right) => left.ordinal - right.ordinal);
  const declaredCount = Number(batchSource.item_count ?? batchSource.itemCount);
  const total = Number.isInteger(declaredCount) && declaredCount >= 0
    ? Math.max(declaredCount, items.length)
    : items.length;
  const counts = {
    total,
    queued: Math.max(0, total - items.length),
    running: 0,
    awaitingConfirmation: 0,
    succeeded: 0,
    failed: 0,
  };
  items.forEach((item) => {
    counts[statusBucket(item)] += 1;
  });
  return {
    parentTaskId,
    status: firstText(batchSource.status, batchSource.state, "created").toLowerCase(),
    errorMessage: firstText(
      batchSource.error_message,
      batchSource.errorMessage,
      batchSource.failure_reason,
    ),
    summaryDownloadUrl: safeLink(firstText(
      batchSource.summary_download_url,
      batchSource.summaryDownloadUrl,
      batchSource.download_url,
      payload.summary_download_url,
      payload.summaryDownloadUrl,
      payload.download_url,
    )),
    summaryPath: firstText(batchSource.summary_path, batchSource.summaryPath),
    createdAt: firstText(batchSource.created_at, batchSource.createdAt),
    updatedAt: firstText(batchSource.updated_at, batchSource.updatedAt),
    counts,
    items,
  };
}

function batchStatus(batch) {
  return statusPresentation(resolveBatchStatus(batch));
}

function metricText(value, { percent = false } = {}) {
  if (!Number.isFinite(value)) return "—";
  return percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}



// Stages that name a status rather than a step; as a suffix they would put a
// second status word on the chip ("需复核 · 已完成", batch 4 review U1).
const STATUS_WORD_STAGES = new Set(["queued", "created", "batch_created", "completed", "failure"]);

// One status word per chip; the step suffix only says where a running or
// waiting model is ("执行中 · PMML打分", "待确认 · 确认字段").
function itemProgressLabel(item, batchStatusValue = "") {
  const status = itemStatus(item, batchStatusValue);
  const stage = firstText(item?.stage);
  const stepStage = stage && !STATUS_WORD_STAGES.has(stage) && stageLabels[stage] ? stageLabels[stage] : "";
  if (!stepStage || !["running", "awaiting_confirm"].includes(status.key)) return status.label;
  return `${status.label} · ${stepStage}`;
}

export function validationBatchItemNeedsContractConfirmation(item) {
  const contractStatus = firstText(
    item?.inputContractStatus,
    item?.input_contract_status,
  ).toLowerCase();
  if (contractStatus === "ready") return false;
  const status = textValue(item?.status).toLowerCase();
  return ["awaiting_confirmation", "pending", "pending_confirmation"].includes(status);
}

function remainingContractConfirmationCount(items, confirmedContractTaskIds) {
  return (items || []).filter((item) => (
    validationBatchItemNeedsContractConfirmation(item)
    && !confirmedContractTaskIds.has(item.childTaskId)
  )).length;
}

function pmmlLabel(value) {
  if (!value) return "—";
  if (value === "pass") return "通过";
  if (["fail", "failed"].includes(value)) return "失败";
  return value;
}

function reportLabel(complete) {
  return complete ? "完整" : "待生成";
}

export function validationBatchStartAction(
  status,
  { inFlight = false, awaitingConfirmationCount = 0 } = {},
) {
  const normalized = textValue(status).toLowerCase();
  if (inFlight) {
    return { visible: true, disabled: true, label: "正在提交…" };
  }
  if (normalized === "created") {
    return { visible: true, disabled: false, label: "启动批次" };
  }
  if (normalized === "awaiting_confirmation") {
    return {
      visible: true,
      disabled: awaitingConfirmationCount > 0,
      label: "确认合同后继续",
    };
  }
  if (normalized === "partial_failure") {
    return {
      visible: true,
      disabled: awaitingConfirmationCount > 0,
      label: "继续未完成项",
    };
  }
  if (normalized === "failed") {
    return {
      visible: true,
      disabled: awaitingConfirmationCount > 0,
      label: "修复后重试",
    };
  }
  if (normalized === "running") {
    return { visible: true, disabled: true, label: "批次运行中" };
  }
  return { visible: false, disabled: true, label: "" };
}

function itemLinksHtml(item, confirmedContractTaskIds = new Set()) {
  const links = [];
  const contractConfirmed = item.inputContractStatus === "ready"
    || confirmedContractTaskIds.has(item.childTaskId);
  if (contractConfirmed) {
    links.push('<span class="validation-batch-contract-confirmed">合同已确认</span>');
  } else if (item.childTaskId && validationBatchItemNeedsContractConfirmation(item)) {
    links.push(
      [
        '<button type="button" class="validation-batch-link validation-batch-contract-button"',
        ` data-batch-contract-task-id="${escapeHtml(item.childTaskId)}"`,
        ` data-batch-contract-item-id="${escapeHtml(item.id)}">确认合同</button>`,
      ].join(""),
    );
  }
  if (item.manualReviewUrl && statusBucket(item) !== "succeeded") {
    links.push(
      `<a class="validation-batch-link" href="${escapeHtml(item.manualReviewUrl)}">人工校验</a>`,
    );
  }
  if (item.wordReportDownloadUrl) {
    links.push(
      `<a class="validation-batch-link" href="${escapeHtml(item.wordReportDownloadUrl)}">Word</a>`,
    );
  }
  if (item.analysisDownloadUrl) {
    links.push(
      `<a class="validation-batch-link" href="${escapeHtml(item.analysisDownloadUrl)}">Excel</a>`,
    );
  }
  return links.length ? links.join("") : '<span class="validation-batch-muted">—</span>';
}

function itemRowHtml(item, requestedItemId, confirmedContractTaskIds, batchStatusValue = "") {
  const highlighted = requestedItemId
    && [item.childTaskId, item.id].includes(requestedItemId);
  const modelLabel = [item.modelName, item.modelVersion].filter(Boolean).join(" · ");
  const status = itemStatus(item, batchStatusValue);
  const statusDetail = item.stage && item.stage !== item.status
    ? `<small>${escapeHtml(item.stage)}</small>`
    : "";
  const failure = item.errorMessage
    ? `<div class="validation-batch-error">${escapeHtml(item.errorMessage)}</div>`
    : "";
  return [
    `<tr class="validation-batch-row${highlighted ? " is-deep-linked" : ""}"`,
    ` data-batch-item-id="${escapeHtml(item.id)}"`,
    ` data-batch-child-task-id="${escapeHtml(item.childTaskId)}">`,
    `<td><strong>${escapeHtml(modelLabel)}</strong>${failure}</td>`,
    `<td><span class="validation-batch-status ${status.tone}">${escapeHtml(status.label)}</span>${statusDetail}${batchItemRepairHtml(item)}</td>`,
    `<td class="numeric">${escapeHtml(metricText(item.ootKs, { percent: true }))}</td>`,
    `<td class="numeric">${escapeHtml(metricText(item.ootPsi))}</td>`,
    `<td>${escapeHtml(pmmlLabel(item.pmmlStatus))}</td>`,
    `<td>${escapeHtml(stressLabels[item.stressRisk] || item.stressRisk || "—")}</td>`,
    `<td>${escapeHtml(reportLabel(item.reportComplete))}</td>`,
    `<td><div class="validation-batch-links">${itemLinksHtml(item, confirmedContractTaskIds)}</div></td>`,
    "</tr>",
  ].join("");
}

export function validationBatchConfirmAllAction(
  batch,
  { inFlight = false } = {},
) {
  const items = batch?.items || [];
  const total = Number(batch?.counts?.total);
  if (!(total >= 2 || items.length >= 2)) {
    return { visible: false, disabled: true, label: "全部确认" };
  }
  const actionable = items.filter((item) => !["failed", "cancelled"].includes(item.status));
  const pending = actionable.filter((item) => item.pendingReportDraft);
  const notReady = actionable.filter(
    (item) => !item.pendingReportDraft && !item.wordReportDownloadUrl,
  );
  const failed = items.filter((item) => item.status === "failed").length;
  const cancelled = items.filter((item) => item.status === "cancelled").length;
  const scopeParts = [`${pending.length} 份待确认`, failed ? `${failed} 个失败` : "", cancelled ? `${cancelled} 个已取消` : "", notReady.length ? `${notReady.length} 个未就绪` : ""].filter(Boolean);
  const scope = scopeParts.join(" · ");
  if (inFlight) {
    return { visible: true, disabled: true, label: "正在生成全部报告…", scope, scopeParts, pendingCount: pending.length };
  }
  return {
    visible: true,
    disabled: pending.length === 0 || notReady.length > 0,
    label: "全部确认",
    scope,
    scopeParts,
    pendingCount: pending.length,
    title: notReady.length > 0
      ? "还有模型尚未完成报告结论草稿"
      : pending.length === 0
        ? "没有待确认的报告草稿"
        : `确认 ${pending.length} 份报告结论并生成文件；失败和取消的模型不在本次范围内`,
  };
}

export function renderValidationBatchSwitcher(
  batchPayload,
  { selectedChildTaskId = "", confirmAllInFlight = false } = {},
) {
  const batch = batchPayload?.counts && Array.isArray(batchPayload?.items)
    ? batchPayload
    : normalizeValidationBatchPayload(batchPayload);
  const selected = textValue(selectedChildTaskId) || batch.items[0]?.childTaskId || "";
  const buttons = (batch.items || []).map((item) => {
    const isSelected = item.childTaskId === selected;
    const tone = itemStatus(item, batch.status).tone;
    const progress = itemProgressLabel(item, batch.status);
    const version = item.modelVersion ? ` · ${item.modelVersion}` : "";
    return [
      `<button type="button" class="validation-batch-switcher-item${isSelected ? " is-selected" : ""}"`,
      ` data-tone="${escapeHtml(tone)}"`,
      ` data-batch-switch-child="${escapeHtml(item.childTaskId)}"`,
      ` title="${escapeHtml(`${item.modelName || `模型 ${item.ordinal}`}${version}：${progress}`)}"`,
      ` role="radio" aria-checked="${isSelected ? "true" : "false"}" tabindex="${isSelected ? "0" : "-1"}">`,
      `<span class="validation-batch-switcher-name">${escapeHtml(item.modelName || `模型 ${item.ordinal}`)}${escapeHtml(version)}</span>`,
      `<small class="validation-batch-switcher-progress">${escapeHtml(progress)}</small>`,
      "</button>",
    ].join("");
  }).join("");
  const showSummary = Number(batch.counts.total) >= 2;
  const confirmAll = validationBatchConfirmAllAction(batch, { inFlight: confirmAllInFlight });
  const confirmAllAction = confirmAll.visible
    ? [
      `<button type="button" class="button compact primary" data-batch-confirm-all="true"`,
      confirmAll.disabled ? ' disabled aria-disabled="true"' : "",
      confirmAll.title ? ` title="${escapeHtml(confirmAll.title)}"` : "",
      `>${escapeHtml(confirmAll.label)}`,
      // The header capsule shows only the pending count (mockup "全部确认 1");
      // the full scope stays in the status line below for assistive tech.
      confirmAll.pendingCount ? `<span class="validation-batch-confirm-count" aria-hidden="true">${escapeHtml(String(confirmAll.pendingCount))}</span>` : "",
      "</button>",
      // Separate platform phrases so each can translate and restore on its own.
      `<span class="validation-batch-confirm-scope" role="status">${(confirmAll.scopeParts || []).map((part) => `<span>${escapeHtml(part)}</span>`).join(" · ")}</span>`,
    ].join("")
    : "";
  const summaryAction = !showSummary
    ? ""
    : batch.summaryDownloadUrl
      ? `<a class="button compact validation-batch-download" href="${escapeHtml(batch.summaryDownloadUrl)}">下载汇总 Excel</a>`
      : "";
  const actions = confirmAllAction || summaryAction
    ? `<div class="validation-batch-switcher-actions">${confirmAllAction}${summaryAction}</div>`
    : "";
  return [
    '<div class="validation-batch-switcher" aria-label="当前验证模型">',
    `<div class="validation-batch-switcher-list" role="radiogroup" aria-label="选择验证模型">${buttons}</div>`,
    actions,
    "</div>",
  ].join("");
}

export function renderValidationBatchOverview(
  batchPayload,
  {
    requestedItemId = "",
    startInFlight = false,
    confirmedContractTaskIds = new Set(),
  } = {},
) {
  const batch = batchPayload?.counts && Array.isArray(batchPayload?.items)
    ? batchPayload
    : normalizeValidationBatchPayload(batchPayload);
  const awaitingConfirmationCount = remainingContractConfirmationCount(
    batch.items,
    confirmedContractTaskIds,
  );
  const startAction = validationBatchStartAction(batch.status, {
    inFlight: startInFlight,
    awaitingConfirmationCount,
  });
  const startActionHtml = startAction.visible
    ? [
      `<button type="button" class="button compact primary" data-batch-start="true"`,
      startAction.disabled ? ' disabled aria-disabled="true"' : "",
      `>${escapeHtml(startAction.label)}</button>`,
    ].join("")
    : "";
  const showSummary = Number(batch.counts.total) >= 2;
  const summaryAction = !showSummary
    ? ""
    : batch.summaryDownloadUrl
    ? `<a class="button compact validation-batch-download" href="${escapeHtml(batch.summaryDownloadUrl)}">下载汇总 Excel</a>`
    : '<span class="validation-batch-summary-pending" aria-disabled="true">汇总待生成</span>';
  const rows = batch.items.length
    ? batch.items
      .map((item) => itemRowHtml(item, requestedItemId, confirmedContractTaskIds, batch.status))
      .join("")
    : '<tr><td class="validation-batch-empty" colspan="8">批次项正在准备，点击刷新查看最新状态。</td></tr>';
  const batchError = batch.errorMessage
    ? `<p class="validation-batch-banner-error" role="alert">${escapeHtml(batch.errorMessage)}</p>`
    : "";
  const headStatus = batchStatus(batch);
  const contractGuidance = batch.status === "awaiting_confirmation"
    ? [
      '<p class="validation-batch-contract-guidance">',
      "无冲突时请在对话里回复「都按这个」按识别结果确认；某个模型要改列时指出模型名和字段。",
      "表单仍可展开核对，但不必逐项点选才能继续。",
      "</p>",
    ].join("")
    : "";
  return [
    '<div class="validation-batch-overview">',
    '<header class="validation-batch-head">',
    '<div><p class="validation-batch-eyebrow">MODEL VALIDATION BATCH</p>',
    '<div class="validation-batch-title-line"><h3>批次总览</h3>',
    `<span class="validation-batch-status ${headStatus.tone}">${escapeHtml(headStatus.label)}</span></div>`,
    `<p>${escapeHtml(batch.counts.total)} 个模型 · 独立执行，单项失败不阻塞其余模型</p></div>`,
    '<div class="validation-batch-head-actions">',
    '<button type="button" class="button compact secondary" data-batch-refresh="true">刷新批次</button>',
    startActionHtml,
    summaryAction,
    "</div></header>",
    batchError,
    contractGuidance,
    '<div class="validation-batch-stats" aria-label="批次状态汇总">',
    `<div><strong>${escapeHtml(batch.counts.total)}</strong><span>总模型</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.running)}</strong><span>${STATUS_WORDS.running}</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.awaitingConfirmation)}</strong><span>${STATUS_WORDS.awaiting_confirm} / ${STATUS_WORDS.needs_review}</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.succeeded)}</strong><span>${STATUS_WORDS.done}</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.failed)}</strong><span>${STATUS_WORDS.failed}</span></div>`,
    "</div>",
    '<div class="validation-batch-table-wrap"><table class="validation-batch-table">',
    '<thead><tr><th>模型</th><th>状态 / 阶段</th><th>OOT KS</th><th>OOT PSI</th><th>PMML</th><th>压力风险</th><th>报告</th><th>操作</th></tr></thead>',
    `<tbody>${rows}</tbody>`,
    "</table></div></div>",
  ].join("");
}

export function focusValidationBatchItem(panel, requestedItemId) {
  const normalizedItemId = textValue(requestedItemId);
  const rows = Array.from(panel?.querySelectorAll?.("[data-batch-child-task-id]") || []);
  let target = null;
  rows.forEach((row) => {
    const matches = Boolean(normalizedItemId) && (
      row.dataset?.batchChildTaskId === normalizedItemId
      || row.dataset?.batchItemId === normalizedItemId
    );
    row.classList?.toggle
      ? row.classList.toggle("is-deep-linked", matches)
      : matches
        ? row.classList?.add?.("is-deep-linked")
        : row.classList?.remove?.("is-deep-linked");
    if (matches) target = row;
  });
  if (!target) return false;
  target.scrollIntoView?.({ block: "center", behavior: "smooth" });
  return true;
}

export function createValidationBatchPanelController({
  api,
  getElementById,
  getSelectedTask,
  deepLink = {},
  confirmStart = async () => false,
  openContract = async () => {},
  refreshParentTask = async () => true,
  onError = () => {},
  onRecovered = () => {},
  onProjectedChildChange = () => {},
  onLayoutChange = () => {},
  onBatchMeta = () => {},
  onBatchStatus = () => {},
  confirmAllReportDrafts = async () => {},
  schedulePoll = (callback, delay) => setTimeout(callback, delay),
  cancelPoll = (handle) => clearTimeout(handle),
  pollIntervalMs = 1500,
  // Every unfinished model is waiting on the user (fields, drafts): polling
  // slows down; the user's own action refreshes at once (eval S5: ~49
  // requests per 10 s while idle).
  idlePollIntervalMs = 5000,
}) {
  let activeTaskId = "";
  let currentPayload = null;
  let requestVersion = 0;
  let startInFlight = false;
  let confirmAllInFlight = false;
  let pollHandle = null;
  let detailLoadErrorMessage = "";
  const confirmedContractTaskIds = new Set();
  const boundHosts = new Set();
  const initialTaskId = textValue(deepLink.taskId);
  const initialItemId = textValue(deepLink.itemId);
  let selectedChildTaskId = initialItemId;
  let lastNotifiedItemCount;
  let lastNotifiedStatus = "";

  function panelElement() {
    return getElementById("batchOverviewPanel");
  }

  function switcherHost() {
    const host = getElementById("validationBatchSwitcher");
    const panel = panelElement();
    return host && host !== panel ? host : null;
  }

  function usesHeroSwitcher() {
    return Boolean(switcherHost()) && usesAgentValidationWorkbench(getSelectedTask());
  }

  function notifyLayout() {
    try {
      onLayoutChange();
    } catch (_) {
      // Layout sync is presentational.
    }
  }

  // The parent's aggregated status feeds the sidebar row and the title pill
  // (D16); tell the app when it changes so both repaint from the same payload.
  function notifyBatchStatus() {
    if (!currentPayload) return;
    const signature = `${activeTaskId}:${resolveBatchStatus(currentPayload)}`;
    if (lastNotifiedStatus === signature) return;
    lastNotifiedStatus = signature;
    try {
      onBatchStatus({ parentTaskId: activeTaskId });
    } catch (_) {
      // Status repaint is presentational.
    }
  }

  function notifyBatchMeta() {
    if (!currentPayload) return;
    const itemCount = currentPayload.counts?.total;
    if (lastNotifiedItemCount === itemCount) return;
    lastNotifiedItemCount = itemCount;
    try {
      onBatchMeta({
        itemCount,
        createdAt: currentPayload.createdAt,
      });
    } catch (_) {
      // Title stamping is presentational.
    }
  }

  function setHostVisible(host, visible) {
    if (!host) return;
    host.classList?.toggle("hidden", !visible);
    host.toggleAttribute?.("hidden", !visible);
    host.setAttribute?.("aria-hidden", visible ? "false" : "true");
  }

  function requestedItemId() {
    return activeTaskId && activeTaskId === initialTaskId ? initialItemId : "";
  }

  function setVisible(visible) {
    const isBatch = Boolean(visible) && isValidationBatchTask(getSelectedTask());
    const heroSwitcher = usesHeroSwitcher();
    setHostVisible(panelElement(), isBatch && !heroSwitcher);
    setHostVisible(switcherHost(), isBatch && heroSwitcher);
    notifyLayout();
  }

  function renderLoading() {
    const loadingHtml = usesHeroSwitcher()
      ? '<div class="validation-batch-switcher-loading" role="status">正在读取模型状态…</div>'
      : '<div class="validation-batch-loading" role="status">正在读取批次状态…</div>';
    const host = usesHeroSwitcher() ? switcherHost() : panelElement();
    if (host) host.innerHTML = loadingHtml;
  }

  function renderError(message) {
    const errorHtml = [
      '<div class="validation-batch-load-error" role="alert">',
      `<strong>批次状态读取失败</strong><span>${escapeHtml(message || "请稍后重试。")}</span>`,
      '<button type="button" class="button compact secondary" data-batch-refresh="true">重新加载</button>',
      "</div>",
    ].join("");
    const host = usesHeroSwitcher() ? switcherHost() : panelElement();
    if (host) host.innerHTML = errorHtml;
  }

  function renderCurrent() {
    const panel = panelElement();
    const switcher = switcherHost();
    if (!currentPayload) return;
    if (usesAgentValidationWorkbench(getSelectedTask())) {
      selectedChildTaskId = resolveProjectedChildTaskId(currentPayload, selectedChildTaskId);
      const html = renderValidationBatchSwitcher(currentPayload, {
        selectedChildTaskId,
        confirmAllInFlight,
      });
      if (switcher) {
        switcher.innerHTML = html;
        if (panel) panel.innerHTML = "";
      } else if (panel) {
        panel.innerHTML = html;
      }
      notifyBatchMeta();
      notifyBatchStatus();
      notifyLayout();
      return;
    }
    if (switcher) switcher.innerHTML = "";
    if (!panel) return;
    panel.innerHTML = renderValidationBatchOverview(currentPayload, {
      requestedItemId: requestedItemId(),
      startInFlight,
      confirmedContractTaskIds,
    });
    notifyBatchMeta();
    notifyBatchStatus();
    notifyLayout();
  }

  function notifyProjectedChild({ force = false } = {}) {
    const childTaskId = resolveProjectedChildTaskId(currentPayload, selectedChildTaskId);
    selectedChildTaskId = childTaskId;
    if (!activeTaskId || !childTaskId) return;
    try {
      onProjectedChildChange({
        parentTaskId: activeTaskId,
        childTaskId,
        item: currentPayload?.items?.find((candidate) => candidate.childTaskId === childTaskId) || null,
        force,
      });
    } catch (_) {
      // Projection is presentational; batch detail remains authoritative.
    }
  }

  function clearScheduledPoll() {
    if (pollHandle === null) return;
    cancelPoll(pollHandle);
    pollHandle = null;
  }

  function scheduleParentRefreshRetry(taskId) {
    clearScheduledPoll();
    if (!activeTaskId || activeTaskId !== taskId) return;
    pollHandle = schedulePoll(async () => {
      pollHandle = null;
      if (!activeTaskId || activeTaskId !== taskId) return;
      try {
        const synchronized = await refreshParentTask({ parentTaskId: taskId });
        if (synchronized === false) scheduleParentRefreshRetry(taskId);
      } catch (_) {
        scheduleParentRefreshRetry(taskId);
      }
    }, pollIntervalMs);
  }

  function scheduleCurrentPoll() {
    clearScheduledPoll();
    if (!activeTaskId || !currentPayload) return;
    const workbench = usesAgentValidationWorkbench(getSelectedTask());
    const items = currentPayload.items || [];
    const unfinished = items.some((item) => (
      !["succeeded", "failed", "cancelled", "completed"].includes(item.status)
    ));
    if (workbench) {
      if (!unfinished) return;
    } else if (currentPayload.status !== "running") {
      return;
    }
    const taskId = activeTaskId;
    const active = items.some((item) => ["queued", "running"].includes(item.status));
    const interval = !workbench || active ? pollIntervalMs : Math.max(pollIntervalMs, idlePollIntervalMs);
    pollHandle = schedulePoll(async () => {
      pollHandle = null;
      if (!activeTaskId || activeTaskId !== taskId) return;
      const payload = await selectTask(getSelectedTask(), { force: true });
      if (!activeTaskId || activeTaskId !== taskId) return;
      let parentSynchronized = true;
      try {
        parentSynchronized = await refreshParentTask({ parentTaskId: taskId });
      } catch (_) {
        parentSynchronized = false;
      }
      if (
        payload
        && payload.status !== "running"
        && parentSynchronized === false
      ) {
        scheduleParentRefreshRetry(taskId);
      }
    }, interval);
  }

  async function selectTask(task, { force = false } = {}) {
    if (!isValidationBatchTask(task)) {
      clear();
      return null;
    }
    const taskId = textValue(task.id);
    if (!taskId) {
      clear();
      return null;
    }
    setVisible(true);
    if (!force && activeTaskId === taskId && currentPayload) {
      renderCurrent();
      scheduleCurrentPoll();
      notifyProjectedChild({ force: false });
      return currentPayload;
    }
    if (activeTaskId && activeTaskId !== taskId) {
      confirmedContractTaskIds.clear();
      detailLoadErrorMessage = "";
      selectedChildTaskId = taskId === initialTaskId ? initialItemId : "";
      lastNotifiedItemCount = undefined;
    }
    const previousPayload = activeTaskId === taskId ? currentPayload : null;
    activeTaskId = taskId;
    const version = ++requestVersion;
    if (!previousPayload) {
      currentPayload = null;
      renderLoading();
    }
    try {
      const raw = await api(`api/validation-batches/${encodeURIComponent(taskId)}`);
      if (version !== requestVersion || activeTaskId !== taskId) return null;
      const recoveredMessage = detailLoadErrorMessage;
      detailLoadErrorMessage = "";
      confirmedContractTaskIds.clear();
      currentPayload = normalizeValidationBatchPayload(raw, taskId);
      selectedChildTaskId = resolveProjectedChildTaskId(
        currentPayload,
        selectedChildTaskId || requestedItemId(),
      );
      renderCurrent();
      notifyProjectedChild({ force });
      focusRequestedItem();
      scheduleCurrentPoll();
      if (recoveredMessage) {
        try {
          onRecovered({ parentTaskId: taskId, message: recoveredMessage });
        } catch (_) {
          // Status restoration is presentational and must not restart polling.
        }
      }
      return currentPayload;
    } catch (error) {
      if (version !== requestVersion || activeTaskId !== taskId) return null;
      const message = error?.message || "请稍后重试。";
      const shouldRetry = currentPayload?.status === "running";
      detailLoadErrorMessage = message;
      renderError(message);
      onError(message);
      if (shouldRetry) scheduleCurrentPoll();
      return null;
    }
  }

  function renderVisibility(task = getSelectedTask()) {
    setVisible(isValidationBatchTask(task));
  }

  function focusRequestedItem() {
    return focusValidationBatchItem(panelElement(), requestedItemId());
  }

  async function startOrContinue() {
    const awaitingConfirmationCount = remainingContractConfirmations();
    const action = validationBatchStartAction(currentPayload?.status, {
      inFlight: startInFlight,
      awaitingConfirmationCount,
    });
    if (!activeTaskId || !currentPayload || !action.visible || action.disabled) return false;
    const parentTaskId = activeTaskId;
    const confirmed = await confirmStart({
      parentTaskId,
      status: currentPayload.status,
      itemCount: currentPayload.counts.total,
      awaitingConfirmationCount: remainingContractConfirmations(),
    });
    if (!confirmed || activeTaskId !== parentTaskId) return false;
    startInFlight = true;
    let startFailed = false;
    clearScheduledPoll();
    renderCurrent();
    try {
      try {
        await api(`api/validation-batches/${encodeURIComponent(parentTaskId)}/start`, {
          method: "POST",
          body: JSON.stringify({}),
        });
      } catch (error) {
        startFailed = true;
        const message = error?.message || "批次启动失败，请稍后重试。";
        renderError(message);
        onError(message);
        return false;
      }
      if (activeTaskId === parentTaskId && currentPayload) {
        currentPayload = { ...currentPayload, status: "running" };
        renderCurrent();
      }
      try {
        await refreshParentTask();
      } catch (_) {
        // The accepted start request is the commit boundary. Shared task-list
        // refresh is best-effort; batch detail polling remains authoritative.
      }
      if (activeTaskId !== parentTaskId) return true;
      const currentTask = getSelectedTask();
      await selectTask(
        isValidationBatchTask(currentTask)
          ? currentTask
          : { id: parentTaskId, task_type: validationBatchTaskType },
        { force: true },
      );
      return true;
    } finally {
      startInFlight = false;
      if (!startFailed && currentPayload && activeTaskId === parentTaskId) {
        renderCurrent();
        scheduleCurrentPoll();
      }
    }
  }

  async function openItemContract(button) {
    const parentTaskId = activeTaskId;
    const childTaskId = textValue(button?.dataset?.batchContractTaskId);
    const itemId = textValue(button?.dataset?.batchContractItemId);
    const item = currentPayload?.items?.find((candidate) => (
      candidate.childTaskId === childTaskId
      && (!itemId || candidate.id === itemId)
    ));
    if (
      !parentTaskId
      || !childTaskId
      || !item
      || !validationBatchItemNeedsContractConfirmation(item)
    ) return false;
    try {
      await openContract({
        parentTaskId,
        childTaskId,
        itemId: item.id,
        modelName: item.modelName,
        modelVersion: item.modelVersion,
      });
      return activeTaskId === parentTaskId;
    } catch (error) {
      onError(error?.message || "验证字段合同读取失败，请稍后重试。");
      return false;
    }
  }

  function remainingContractConfirmations() {
    return remainingContractConfirmationCount(
      currentPayload?.items,
      confirmedContractTaskIds,
    );
  }

  function markContractConfirmed(childTaskId) {
    const normalizedTaskId = textValue(childTaskId);
    if (normalizedTaskId) confirmedContractTaskIds.add(normalizedTaskId);
    renderCurrent();
    return remainingContractConfirmations();
  }

  function clear() {
    requestVersion += 1;
    clearScheduledPoll();
    activeTaskId = "";
    currentPayload = null;
    startInFlight = false;
    confirmAllInFlight = false;
    detailLoadErrorMessage = "";
    selectedChildTaskId = initialItemId;
    lastNotifiedItemCount = undefined;
    lastNotifiedStatus = "";
    confirmedContractTaskIds.clear();
    const panel = panelElement();
    const switcher = switcherHost();
    setVisible(false);
    if (panel) panel.innerHTML = "";
    if (switcher) switcher.innerHTML = "";
  }

  function handleBatchClick(event) {
    if (event.target?.closest?.("[data-batch-start]")) {
      void startOrContinue();
      return;
    }
    if (event.target?.closest?.("[data-batch-refresh]")) {
      void selectTask(getSelectedTask(), { force: true });
      return;
    }
    if (event.target?.closest?.("[data-batch-confirm-all]")) {
      void confirmAll();
      return;
    }
    const switchButton = event.target?.closest?.("[data-batch-switch-child]");
    if (switchButton) {
      const nextChild = textValue(switchButton.dataset.batchSwitchChild);
      if (nextChild && nextChild !== selectedChildTaskId) {
        selectedChildTaskId = nextChild;
        renderCurrent();
        notifyProjectedChild({ force: true });
      }
      return;
    }
    const contractButton = event.target?.closest?.("[data-batch-contract-task-id]");
    if (contractButton) {
      void openItemContract(contractButton);
    }
  }

  async function confirmAll() {
    const parentTaskId = currentPayload?.parentTaskId || activeTaskId;
    if (!parentTaskId || confirmAllInFlight) return;
    const action = validationBatchConfirmAllAction(currentPayload, { inFlight: false });
    if (!action.visible || action.disabled) return;
    confirmAllInFlight = true;
    renderCurrent();
    try {
      await confirmAllReportDrafts({ parentTaskId });
      await selectTask(getSelectedTask(), { force: true });
    } catch (error) {
      onError(error?.message || "全部确认失败，请稍后重试。");
    } finally {
      confirmAllInFlight = false;
      renderCurrent();
    }
  }

  function handleSwitcherWheel(event) {
    const list = event.target?.closest?.(".validation-batch-switcher-list");
    if (!list) return;
    if (list.scrollWidth <= list.clientWidth + 1) return;
    if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    event.preventDefault();
    list.scrollLeft += event.deltaY;
  }

  function bindHost(host) {
    if (!host || boundHosts.has(host)) return;
    boundHosts.add(host);
    host.addEventListener("click", handleBatchClick);
    host.addEventListener("wheel", handleSwitcherWheel, { passive: false });
    host.addEventListener("keydown", (event) => {
      const current = event.target?.closest?.("[data-batch-switch-child]");
      if (!current || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
      const buttons = [...host.querySelectorAll("[data-batch-switch-child]")];
      const index = buttons.indexOf(current);
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
        : (index + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) + buttons.length) % buttons.length;
      event.preventDefault();
      const childId = buttons[next].dataset.batchSwitchChild;
      selectChild(childId);
      [...host.querySelectorAll("[data-batch-switch-child]")].find((button) => button.dataset.batchSwitchChild === childId)?.focus();
    });
  }

  function selectChild(childTaskId) {
    const nextId = textValue(childTaskId);
    if (!nextId) return;
    selectedChildTaskId = currentPayload
      ? resolveProjectedChildTaskId(currentPayload, nextId)
      : nextId;
    if (currentPayload) renderCurrent();
    notifyProjectedChild({ force: true });
  }

  bindHost(panelElement());
  bindHost(switcherHost());

  return {
    // The loaded payload for the selected batch parent, so the sidebar row and
    // the title can aggregate the items instead of reading task.status alone.
    batchPayload: (taskId) => (taskId && taskId === activeTaskId ? currentPayload : null),
    clear,
    focusRequestedItem,
    markContractConfirmed,
    remainingContractConfirmations,
    renderVisibility,
    selectTask,
    selectChild,
    openItemContract,
    startOrContinue,
    statusIsTerminal: (status) => terminalBatchStatuses.has(textValue(status).toLowerCase()),
  };
}
