export const validationBatchTaskType = "validation_batch";

const terminalBatchStatuses = new Set([
  "completed",
  "failed",
  "cancelled",
]);

const statusLabels = {
  created: "待启动",
  queued: "排队中",
  running: "运行中",
  awaiting_confirmation: "待确认",
  review_required: "需人工复核",
  succeeded: "已通过",
  completed: "已完成",
  partial_failure: "部分失败",
  failed: "失败",
  cancelled: "已取消",
};

const stressLabels = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
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
  if (/^\/(?!\/)/.test(normalized) || normalized.startsWith("?")) return normalized;
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

function statusBucket(item) {
  const status = item.status;
  const outcome = item.outcome;
  if (["failed", "cancelled"].includes(status) || ["fail", "failed"].includes(outcome)) {
    return "failed";
  }
  if (["review_required", "awaiting_confirmation"].includes(status) || outcome === "manual_review") {
    return "awaitingConfirmation";
  }
  if (["succeeded", "completed"].includes(status) || outcome === "pass") return "succeeded";
  if (status === "running") return "running";
  return "queued";
}

function itemTone(item) {
  const bucket = statusBucket(item);
  if (bucket === "succeeded") return "success";
  if (bucket === "failed") return "danger";
  if (bucket === "running") return "running";
  if (bucket === "awaitingConfirmation") return "warning";
  return "neutral";
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
      ? `/?task=${encodeURIComponent(parentTaskId)}&item=${encodeURIComponent(childTaskId)}`
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
    manualReviewUrl,
  };
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

export function isValidationBatchTask(task) {
  return task?.task_type === validationBatchTaskType;
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

function batchStatusLabel(status) {
  return statusLabels[status] || status || "状态未知";
}

function metricText(value, { percent = false } = {}) {
  if (!Number.isFinite(value)) return "—";
  return percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(4);
}

function itemStatusLabel(item) {
  return statusLabels[item.status] || statusLabels[item.outcome] || item.status || "状态未知";
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
      label: "逐项确认合同后继续",
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

function itemRowHtml(item, requestedItemId, confirmedContractTaskIds) {
  const highlighted = requestedItemId
    && [item.childTaskId, item.id].includes(requestedItemId);
  const modelLabel = [item.modelName, item.modelVersion].filter(Boolean).join(" · ");
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
    `<td><span class="validation-batch-status ${itemTone(item)}">${escapeHtml(itemStatusLabel(item))}</span>${statusDetail}</td>`,
    `<td class="numeric">${escapeHtml(metricText(item.ootKs, { percent: true }))}</td>`,
    `<td class="numeric">${escapeHtml(metricText(item.ootPsi))}</td>`,
    `<td>${escapeHtml(pmmlLabel(item.pmmlStatus))}</td>`,
    `<td>${escapeHtml(stressLabels[item.stressRisk] || item.stressRisk || "—")}</td>`,
    `<td>${escapeHtml(reportLabel(item.reportComplete))}</td>`,
    `<td><div class="validation-batch-links">${itemLinksHtml(item, confirmedContractTaskIds)}</div></td>`,
    "</tr>",
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
  const summaryAction = batch.summaryDownloadUrl
    ? `<a class="button compact validation-batch-download" href="${escapeHtml(batch.summaryDownloadUrl)}">下载汇总 Excel</a>`
    : '<span class="validation-batch-summary-pending" aria-disabled="true">汇总待生成</span>';
  const rows = batch.items.length
    ? batch.items
      .map((item) => itemRowHtml(item, requestedItemId, confirmedContractTaskIds))
      .join("")
    : '<tr><td class="validation-batch-empty" colspan="8">批次项正在准备，点击刷新查看最新状态。</td></tr>';
  const batchError = batch.errorMessage
    ? `<p class="validation-batch-banner-error" role="alert">${escapeHtml(batch.errorMessage)}</p>`
    : "";
  const contractGuidance = batch.status === "awaiting_confirmation"
    ? [
      '<p class="validation-batch-contract-guidance">',
      "请先打开各待确认模型的“确认合同”逐项核对。",
      "平台不会自动确认合同；确认完成后再继续批次。",
      "</p>",
    ].join("")
    : "";
  return [
    '<div class="validation-batch-overview">',
    '<header class="validation-batch-head">',
    '<div><p class="validation-batch-eyebrow">MODEL VALIDATION BATCH</p>',
    '<div class="validation-batch-title-line"><h3>批次总览</h3>',
    `<span class="validation-batch-status ${escapeHtml(batch.status)}">${escapeHtml(batchStatusLabel(batch.status))}</span></div>`,
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
    `<div><strong>${escapeHtml(batch.counts.running)}</strong><span>运行中</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.awaitingConfirmation)}</strong><span>待确认 / 人工复核</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.succeeded)}</strong><span>已通过</span></div>`,
    `<div><strong>${escapeHtml(batch.counts.failed)}</strong><span>失败</span></div>`,
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
  schedulePoll = (callback, delay) => setTimeout(callback, delay),
  cancelPoll = (handle) => clearTimeout(handle),
  pollIntervalMs = 1500,
}) {
  let activeTaskId = "";
  let currentPayload = null;
  let requestVersion = 0;
  let startInFlight = false;
  let pollHandle = null;
  let detailLoadErrorMessage = "";
  const confirmedContractTaskIds = new Set();
  const initialTaskId = textValue(deepLink.taskId);
  const initialItemId = textValue(deepLink.itemId);

  function panelElement() {
    return getElementById("batchOverviewPanel");
  }

  function requestedItemId() {
    return activeTaskId && activeTaskId === initialTaskId ? initialItemId : "";
  }

  function setVisible(visible) {
    const panel = panelElement();
    panel?.classList.toggle("hidden", !visible);
    panel?.setAttribute("aria-hidden", visible ? "false" : "true");
  }

  function renderLoading() {
    const panel = panelElement();
    if (!panel) return;
    panel.innerHTML = '<div class="validation-batch-loading" role="status">正在读取批次状态…</div>';
  }

  function renderError(message) {
    const panel = panelElement();
    if (!panel) return;
    panel.innerHTML = [
      '<div class="validation-batch-load-error" role="alert">',
      `<strong>批次状态读取失败</strong><span>${escapeHtml(message || "请稍后重试。")}</span>`,
      '<button type="button" class="button compact secondary" data-batch-refresh="true">重新加载</button>',
      "</div>",
    ].join("");
  }

  function renderCurrent() {
    const panel = panelElement();
    if (!panel || !currentPayload) return;
    panel.innerHTML = renderValidationBatchOverview(currentPayload, {
      requestedItemId: requestedItemId(),
      startInFlight,
      confirmedContractTaskIds,
    });
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
    if (!activeTaskId || currentPayload?.status !== "running") return;
    const taskId = activeTaskId;
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
    }, pollIntervalMs);
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
      return currentPayload;
    }
    if (activeTaskId && activeTaskId !== taskId) {
      confirmedContractTaskIds.clear();
      detailLoadErrorMessage = "";
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
      renderCurrent();
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
    detailLoadErrorMessage = "";
    confirmedContractTaskIds.clear();
    const panel = panelElement();
    setVisible(false);
    if (panel) panel.innerHTML = "";
  }

  panelElement()?.addEventListener("click", (event) => {
    if (event.target?.closest?.("[data-batch-start]")) {
      void startOrContinue();
      return;
    }
    if (event.target?.closest?.("[data-batch-refresh]")) {
      void selectTask(getSelectedTask(), { force: true });
      return;
    }
    const contractButton = event.target?.closest?.("[data-batch-contract-task-id]");
    if (contractButton) {
      void openItemContract(contractButton);
    }
  });

  return {
    clear,
    focusRequestedItem,
    markContractConfirmed,
    remainingContractConfirmations,
    renderVisibility,
    selectTask,
    openItemContract,
    startOrContinue,
    statusIsTerminal: (status) => terminalBatchStatuses.has(textValue(status).toLowerCase()),
  };
}
