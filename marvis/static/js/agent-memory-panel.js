import { escapeHtml } from "./ui-utils.js";

export function formatMemoryConfidence(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value ?? "");
  if (number <= 1) return `${Math.round(number * 100)}%`;
  return `${Math.round(number)}%`;
}

function memoryTitle(memory = {}) {
  return memory.title || memory.summary || memory.key || memory.scope_key || memory.memory_type || memory.id || "未命名记忆";
}

function memorySummary(memory = {}) {
  return memory.summary || memory.content || memory.value || memory.use_reason || "";
}

function memoryMetaParts(memory = {}) {
  const mergedSourceTaskCount = Array.isArray(memory._merged_source_task_ids)
    ? memory._merged_source_task_ids.length
    : 0;
  return [
    memory.category || memory.memory_type,
    memory.status,
    memory.scope_key,
    Number(memory._merged_count || 0) > 1 ? `合并 ${memory._merged_count} 条原始记忆` : "",
    memory.support_count !== undefined ? `支持 ${memory.support_count}` : "",
    mergedSourceTaskCount > 1
      ? `来源任务 ${mergedSourceTaskCount} 个`
      : memory.source_task_id
        ? `来源 ${memory.source_task_id}`
        : "",
    memory.model_name,
    memory.confidence !== undefined ? `置信度 ${formatMemoryConfidence(memory.confidence)}` : "",
    memory.superseded_by ? "已被取代" : "",
  ].filter(Boolean);
}

function normalizeMemoryText(value) {
  return String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
}

function stableMemoryValue(value) {
  if (Array.isArray(value)) return value.map(stableMemoryValue);
  if (!value || typeof value !== "object") return value;
  return Object.keys(value)
    .sort()
    .reduce((out, key) => {
      out[key] = stableMemoryValue(value[key]);
      return out;
    }, {});
}

function stableMemoryJson(value) {
  try {
    return JSON.stringify(stableMemoryValue(value || {}));
  } catch (_error) {
    return String(value || "");
  }
}

function rawMemoryGroupKey(memory = {}) {
  const payload = memory.payload && typeof memory.payload === "object" ? memory.payload : {};
  return [
    normalizeMemoryText(memory.category || memory.memory_type),
    normalizeMemoryText(memory.status),
    normalizeMemoryText(memorySummary(memory) || memoryTitle(memory)),
    stableMemoryJson(payload),
  ].join("\u0001");
}

function mergeRawMemoryItems(sourceItems = []) {
  const groups = new Map();
  const order = [];
  for (const memory of sourceItems) {
    const key = rawMemoryGroupKey(memory);
    if (!groups.has(key)) {
      groups.set(key, []);
      order.push(key);
    }
    groups.get(key).push(memory);
  }
  return order.map((key) => {
    const group = groups.get(key) || [];
    if (group.length <= 1) return group[0];
    const representative = group[0] || {};
    const sourceTaskIds = Array.from(
      new Set(group.map((memory) => String(memory.source_task_id || "").trim()).filter(Boolean))
    );
    return {
      ...representative,
      _merged_count: group.length,
      _merged_ids: group.map((memory) => String(memory.id || memory.memory_id || "").trim()).filter(Boolean),
      _merged_source_task_ids: sourceTaskIds,
    };
  });
}

export function createAgentMemoryPanelController({
  $,
  api,
  runAction,
  showPlatformConfirm,
  openMemorySettings,
  openMemoryDetails,
} = {}) {
  let items = [];
  let viewMode = "raw";
  let selectedMemoryId = "";
  const pageLimit = 100;
  let nextOffset = 0;
  let hasMoreItems = false;

  function setStatus(message = "", kind = "") {
    const status = $("agentMemoryStatus");
    if (!status) return;
    status.textContent = message;
    status.className = ["status", kind].filter(Boolean).join(" ");
  }

  function syncViewControls() {
    const isDistillation = viewMode === "distillation";
    for (const tab of document.querySelectorAll("[data-agent-memory-view]")) {
      const selected = tab.dataset.agentMemoryView === viewMode;
      tab.classList.toggle("selected", selected);
      tab.setAttribute("aria-selected", selected ? "true" : "false");
    }
    const statusFilter = $("agentMemoryStatusFilter");
    if (statusFilter) {
      statusFilter.innerHTML = isDistillation
        ? [
            '<option value="active">当前</option>',
            '<option value="history">含历史</option>',
          ].join("")
        : [
            '<option value="active">启用</option>',
            '<option value="disabled">停用</option>',
            '<option value="deleted">已删除</option>',
            '<option value="rejected">已拒绝</option>',
          ].join("");
    }
    $("agentMemorySourceTaskFilterRow")?.classList.toggle("agent-memory-filter-hidden", isDistillation);
    $("agentMemoryModelFilterRow")?.classList.toggle("agent-memory-filter-hidden", isDistillation);
  }

  function filterParams({ offset = 0 } = {}) {
    const params = new URLSearchParams();
    params.set("limit", String(pageLimit));
    params.set("offset", String(offset));
    if (viewMode === "distillation") {
      const category = String($("agentMemoryTypeFilter")?.value || "").trim();
      const status = String($("agentMemoryStatusFilter")?.value || "").trim();
      if (category) params.set("category", category);
      if (status === "history") params.set("include_superseded", "true");
      return params.toString();
    }
    const filters = [
      ["memory_type", $("agentMemoryTypeFilter")?.value],
      ["status", $("agentMemoryStatusFilter")?.value],
      ["source_task_id", $("agentMemorySourceTaskFilter")?.value],
      ["model_name", $("agentMemoryModelFilter")?.value],
    ];
    for (const [key, value] of filters) {
      const normalized = String(value || "").trim();
      if (normalized) params.set(key, normalized);
    }
    return params.toString();
  }

  function renderItems() {
    const list = $("agentMemoryList");
    if (!list) return;
    const displayItems = viewMode === "distillation" ? items : mergeRawMemoryItems(items);
    if (!displayItems.length) {
      list.innerHTML = '<div class="agent-memory-empty">暂无匹配记忆。</div>';
      return;
    }
    const itemHtml = displayItems.map((memory) => {
      const memoryId = String(memory.id || memory.memory_id || "");
      const memoryStatus = String(memory.status || "").toLowerCase();
      const isDistillation = viewMode === "distillation" || memory.kind === "distillation";
      const isMergedRaw = !isDistillation && Number(memory._merged_count || 0) > 1;
      const isDisabled = memoryStatus === "disabled";
      const isTerminal = memoryStatus === "deleted" || memoryStatus === "rejected" || memoryStatus === "rolled_back";
      const title = memoryTitle(memory);
      const summary = memorySummary(memory);
      const meta = memoryMetaParts(memory).map(escapeHtml).join(" · ");
      return [
        `<article class="agent-memory-item${isMergedRaw ? " merged" : ""}" data-agent-memory-id="${escapeHtml(memoryId)}" data-agent-memory-merged-count="${escapeHtml(memory._merged_count || 1)}">`,
        '<div class="agent-memory-item-main">',
        `<strong>${escapeHtml(title)}</strong>`,
        meta ? `<span>${meta}</span>` : "",
        summary ? `<p>${escapeHtml(summary)}</p>` : "",
        "</div>",
        '<div class="agent-memory-actions">',
        `<button class="button compact secondary" type="button" data-agent-memory-action="inspect" data-agent-memory-id="${escapeHtml(memoryId)}">查看</button>`,
        isDistillation
          ? memoryStatus === "active" && !memory.superseded_by
            ? `<button class="button compact secondary danger" type="button" data-agent-memory-action="rollback" data-agent-memory-id="${escapeHtml(memoryId)}">回滚</button>`
            : ""
          : isMergedRaw
            ? ""
          : isTerminal
            ? ""
            : isDisabled
              ? `<button class="button compact secondary" type="button" data-agent-memory-action="enable" data-agent-memory-id="${escapeHtml(memoryId)}">启用</button>`
              : `<button class="button compact secondary" type="button" data-agent-memory-action="disable" data-agent-memory-id="${escapeHtml(memoryId)}">停用</button>`,
        isDistillation || isTerminal
          ? ""
          : isMergedRaw
            ? ""
          : `<button class="button compact secondary" type="button" data-agent-memory-action="not_useful" data-agent-memory-id="${escapeHtml(memoryId)}">这条没用/有误</button>`,
        isDistillation || isTerminal
          ? ""
          : isMergedRaw
            ? ""
          : `<button class="button compact secondary danger" type="button" data-agent-memory-action="delete" data-agent-memory-id="${escapeHtml(memoryId)}">删除</button>`,
        "</div>",
        "</article>",
      ].join("");
    }).join("");
    const loadMoreHtml = hasMoreItems
      ? [
          '<div class="agent-memory-load-more">',
          '<button class="button compact secondary" type="button" data-agent-memory-action="load_more">加载更多</button>',
          "</div>",
        ].join("")
      : "";
    list.innerHTML = itemHtml + loadMoreHtml;
  }

  function renderDetail(memory = null, events = [], detailOptions = {}) {
    const detail = $("agentMemoryDetail");
    if (!detail) return;
    if (!memory) {
      detail.innerHTML = "";
      selectedMemoryId = "";
      return;
    }
    selectedMemoryId = String(memory.id || memory.memory_id || "");
    const eventList = Array.isArray(events) ? events : [];
    const sourceMemories = Array.isArray(detailOptions.sourceMemories) ? detailOptions.sourceMemories : [];
    const predecessor = detailOptions.predecessor || null;
    const title = memoryTitle(memory);
    const summary = memorySummary(memory);
    const predecessorSummary = predecessor ? predecessor.summary || predecessor.scope_key || "" : "";
    const predecessorLabel = predecessor
      ? [predecessor.id || "", predecessorSummary].filter(Boolean).join(" · ")
      : "";
    const badges = [
      memory.category || memory.memory_type || memory.kind,
      memory.status,
      memory.confidence !== undefined ? `置信度 ${formatMemoryConfidence(memory.confidence)}` : "",
      memory.superseded_by ? "已被取代" : "",
    ].filter(Boolean);
    const evidenceRows = [
      ["范围", memory.scope_key],
      ["来源任务", memory.source_task_id],
      ["来源消息", memory.source_message_id],
      ["模型", memory.model_name],
      ["支持数", memory.support_count !== undefined ? memory.support_count : ""],
      ["创建时间", memory.created_at],
      ["更新时间", memory.updated_at],
      ["删除时间", memory.deleted_at],
      ["记忆 ID", selectedMemoryId],
      ["前驱", predecessorLabel],
    ].filter(([, value]) => String(value ?? "").trim());
    const evidenceHtml = evidenceRows.length
      ? evidenceRows
          .map(
            ([label, value]) =>
              `<div class="agent-memory-evidence-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`
          )
          .join("")
      : '<div class="agent-memory-empty compact">暂无来源字段。</div>';
    const sourceItemsHtml = sourceMemories
      .map(
        (item) =>
          `<li><span>${escapeHtml(item.id || "")}</span><p>${escapeHtml(item.summary || item.memory_type || "")}</p></li>`
      )
      .join("");
    const sourceMemoriesHtml = sourceMemories.length
      ? [
          '<div class="agent-memory-source-list">',
          '<strong>来源记忆</strong>',
          `<ul>${sourceItemsHtml}</ul>`,
          "</div>",
        ].join("")
      : "";
    const eventsHtml = eventList.length
      ? [
          '<ol class="agent-memory-audit-timeline">',
          eventList
            .map((event) => {
              const action = event.action || event.event_type || "event";
              const detailLine = [
                event.task_id ? `任务 ${event.task_id}` : "",
                event.memory_id ? `记忆 ${event.memory_id}` : "",
              ]
                .filter(Boolean)
                .join(" · ");
              return [
                '<li class="agent-memory-audit-event">',
                `<strong>${escapeHtml(action)}</strong>`,
                event.created_at ? `<time>${escapeHtml(event.created_at)}</time>` : "",
                detailLine ? `<span>${escapeHtml(detailLine)}</span>` : "",
                "</li>",
              ].join("");
            })
            .join(""),
          "</ol>",
        ].join("")
      : '<div class="agent-memory-empty compact">暂无审计事件。</div>';
    detail.innerHTML = [
      '<section class="agent-memory-detail-inner">',
      '<header class="agent-memory-detail-header">',
      '<div class="agent-memory-title-block">',
      '<span class="agent-memory-detail-eyebrow">当前记忆</span>',
      `<h3>${escapeHtml(title)}</h3>`,
      "</div>",
      badges.length
        ? `<div class="agent-memory-badges">${badges.map((badge) => `<span>${escapeHtml(badge)}</span>`).join("")}</div>`
        : "",
      "</header>",
      '<section class="agent-memory-summary-card">',
      '<strong>摘要</strong>',
      summary ? `<p>${escapeHtml(summary)}</p>` : '<p class="muted">暂无摘要。</p>',
      "</section>",
      '<div class="agent-memory-detail-grid">',
      '<section class="agent-memory-evidence-card">',
      "<h4>证据字段</h4>",
      evidenceHtml,
      sourceMemoriesHtml,
      "</section>",
      '<section class="agent-memory-audit-panel">',
      "<h4>审计事件</h4>",
      eventsHtml,
      "</section>",
      "</div>",
      "</section>",
    ].join("");
  }

  async function loadItems({ append = false } = {}) {
    const currentOffset = append ? nextOffset : 0;
    const query = filterParams({ offset: currentOffset });
    const isDistillation = viewMode === "distillation";
    setStatus(isDistillation ? "正在读取记忆沉淀..." : "正在读取记忆...");
    const endpoint = isDistillation ? "api/agent-memory/distillations" : "api/agent-memory";
    const payload = await api(endpoint + (query ? `?${query}` : ""));
    const incoming = Array.isArray(payload?.items) ? payload.items : [];
    items = append ? items.concat(incoming) : incoming;
    nextOffset = Number(payload?.offset ?? currentOffset) + incoming.length;
    hasMoreItems = Boolean(payload?.has_more);
    renderItems();
    if (!append) renderDetail(null);
    const displayCount = isDistillation ? items.length : mergeRawMemoryItems(items).length;
    const readMessage = isDistillation || displayCount === items.length
      ? `已读取 ${items.length} 条${isDistillation ? "沉淀" : "记忆"}${hasMoreItems ? "，可继续加载。" : "。"}`
      : `已读取 ${items.length} 条记忆，合并展示为 ${displayCount} 组${hasMoreItems ? "，可继续加载。" : "。"}`;
    setStatus(
      readMessage,
      "success"
    );
  }

  async function loadMoreItems() {
    if (!hasMoreItems) return;
    await loadItems({ append: true });
  }

  function setViewMode(mode, { reload = true } = {}) {
    viewMode = mode === "distillation" ? "distillation" : "raw";
    selectedMemoryId = "";
    items = [];
    nextOffset = 0;
    hasMoreItems = false;
    syncViewControls();
    renderItems();
    renderDetail(null);
    if (reload) {
      runAction(loadItems, { actionId: "agentMemory", busyText: "正在读取 Agent 记忆..." });
    }
  }

  async function inspect(memoryId) {
    if (!memoryId) return;
    const isDistillation = viewMode === "distillation";
    setStatus(isDistillation ? "正在读取沉淀详情..." : "正在读取记忆详情...");
    const payload = await api(
      isDistillation
        ? `api/agent-memory/distillations/${encodeURIComponent(memoryId)}`
        : `api/agent-memory/${encodeURIComponent(memoryId)}`
    );
    if (isDistillation) {
      renderDetail(payload?.distillation || null, payload?.events || [], {
        sourceMemories: payload?.source_memories || [],
        predecessor: payload?.predecessor || null,
      });
    } else {
      renderDetail(payload?.memory || null, payload?.events || []);
    }
    setStatus("记忆详情已更新。", "success");
  }

  async function disable(memoryId) {
    if (!memoryId) return;
    const payload = await api(`api/agent-memory/${encodeURIComponent(memoryId)}/disable`, { method: "POST" });
    renderDetail(payload?.memory || null, payload?.events || []);
    await loadItems();
  }

  async function enable(memoryId) {
    if (!memoryId) return;
    const payload = await api(`api/agent-memory/${encodeURIComponent(memoryId)}/enable`, { method: "POST" });
    renderDetail(payload?.memory || null, payload?.events || []);
    await loadItems();
  }

  async function reportNotUseful(memoryId) {
    if (!memoryId) return;
    const confirmed = await showPlatformConfirm({
      title: "反馈记忆质量",
      message: "标记这条记忆没用或有误？系统会降低它的可信度，减少后续被检索引用的机会。",
      confirmText: "确认反馈",
      cancelText: "取消",
      tone: "warning",
    });
    if (!confirmed) return;
    const payload = await api(
      `api/agent-memory/${encodeURIComponent(memoryId)}/negative-feedback`,
      { method: "POST" }
    );
    renderDetail(payload?.memory || null, payload?.events || []);
    await loadItems();
    setStatus("已记录反馈，该记忆可信度已下调。", "success");
  }

  async function remove(memoryId) {
    if (!memoryId) return;
    const confirmed = await showPlatformConfirm({
      title: "删除记忆",
      message: "删除后将从 Agent 记忆库移除，确定删除？",
      confirmText: "删除",
      cancelText: "取消",
      tone: "danger",
    });
    if (!confirmed) return;
    const payload = await api(`api/agent-memory/${encodeURIComponent(memoryId)}`, { method: "DELETE" });
    items = items.filter((memory) => String(memory.id || memory.memory_id || "") !== memoryId);
    nextOffset = items.length;
    renderItems();
    renderDetail(payload?.memory || null, payload?.events || []);
  }

  async function rollbackDistillation(memoryId) {
    if (!memoryId) return;
    const confirmed = await showPlatformConfirm({
      title: "回滚记忆沉淀",
      message: "回滚后该沉淀将不再用于 Agent 检索，确定回滚？",
      confirmText: "回滚",
      cancelText: "取消",
      tone: "warning",
    });
    if (!confirmed) return;
    await api(`api/agent-memory/distillations/${encodeURIComponent(memoryId)}/rollback`, { method: "POST" });
    await loadItems();
    await inspect(memoryId);
  }

  function handleListClick(event) {
    const button = event.target.closest("[data-agent-memory-action]");
    if (!button) return;
    event.preventDefault();
    const memoryId = button.dataset.agentMemoryId || selectedMemoryId;
    const action = button.dataset.agentMemoryAction;
    const actions = {
      inspect,
      disable,
      enable,
      not_useful: reportNotUseful,
      delete: remove,
      load_more: loadMoreItems,
      rollback: rollbackDistillation,
    };
    const handler = actions[action];
    if (handler) {
      runAction(() => handler(memoryId), { actionId: "agentMemory", busyText: "正在更新 Agent 记忆..." });
    }
  }

  function handleInlineInspect(event) {
    const button = event.target.closest("[data-agent-memory-inline-inspect]");
    if (!button) return;
    event.preventDefault();
    const memoryId = button.dataset.agentMemoryInlineInspect || "";
    if (!memoryId) return;
    const kind = button.dataset.agentMemoryInlineKind || "raw";
    openMemorySettings?.("memory-policy");
    openMemoryDetails?.();
    setViewMode(kind === "distillation" ? "distillation" : "raw", { reload: true });
    runAction(() => inspect(memoryId), { actionId: "agentMemory", busyText: "正在读取 Agent 记忆..." });
  }

  return {
    disable,
    enable,
    hasItems: () => items.length > 0,
    handleInlineInspect,
    handleListClick,
    inspect,
    loadItems,
    loadMoreItems,
    renderDetail,
    renderItems,
    reportNotUseful,
    remove,
    rollbackDistillation,
    setStatus,
    setViewMode,
    syncViewControls,
    viewMode: () => viewMode,
  };
}
