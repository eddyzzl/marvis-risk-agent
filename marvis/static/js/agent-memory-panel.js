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
  return [
    memory.category || memory.memory_type,
    memory.status,
    memory.scope_key,
    memory.support_count !== undefined ? `支持 ${memory.support_count}` : "",
    memory.source_task_id ? `来源 ${memory.source_task_id}` : "",
    memory.model_name,
    memory.confidence !== undefined ? `置信度 ${formatMemoryConfidence(memory.confidence)}` : "",
    memory.superseded_by ? "已被取代" : "",
  ].filter(Boolean);
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

  function filterParams() {
    const params = new URLSearchParams();
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
    if (!items.length) {
      list.innerHTML = '<div class="agent-memory-empty">暂无匹配记忆。</div>';
      return;
    }
    list.innerHTML = items.map((memory) => {
      const memoryId = String(memory.id || memory.memory_id || "");
      const memoryStatus = String(memory.status || "").toLowerCase();
      const isDistillation = viewMode === "distillation" || memory.kind === "distillation";
      const isDisabled = memoryStatus === "disabled";
      const isTerminal = memoryStatus === "deleted" || memoryStatus === "rejected" || memoryStatus === "rolled_back";
      const title = memoryTitle(memory);
      const summary = memorySummary(memory);
      const meta = memoryMetaParts(memory).map(escapeHtml).join(" · ");
      return [
        `<article class="agent-memory-item" data-agent-memory-id="${escapeHtml(memoryId)}">`,
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
          : isTerminal
            ? ""
            : isDisabled
              ? `<button class="button compact secondary" type="button" data-agent-memory-action="enable" data-agent-memory-id="${escapeHtml(memoryId)}">启用</button>`
              : `<button class="button compact secondary" type="button" data-agent-memory-action="disable" data-agent-memory-id="${escapeHtml(memoryId)}">停用</button>`,
        isDistillation || isTerminal
          ? ""
          : `<button class="button compact secondary danger" type="button" data-agent-memory-action="delete" data-agent-memory-id="${escapeHtml(memoryId)}">删除</button>`,
        "</div>",
        "</article>",
      ].join("");
    }).join("");
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
    const meta = memoryMetaParts(memory).map(escapeHtml).join(" · ");
    const eventList = Array.isArray(events) ? events : [];
    const sourceMemories = Array.isArray(detailOptions.sourceMemories) ? detailOptions.sourceMemories : [];
    const predecessor = detailOptions.predecessor || null;
    detail.innerHTML = [
      '<section class="agent-memory-detail-inner">',
      `<h3>${escapeHtml(memoryTitle(memory))}</h3>`,
      meta ? `<div class="agent-memory-detail-meta">${meta}</div>` : "",
      memorySummary(memory) ? `<p>${escapeHtml(memorySummary(memory))}</p>` : "",
      predecessor
        ? `<div class="agent-memory-detail-meta">前驱 ${escapeHtml(predecessor.id || "")} · ${escapeHtml(predecessor.summary || predecessor.scope_key || "")}</div>`
        : "",
      sourceMemories.length
        ? [
            '<div class="agent-memory-source-list">',
            '<strong>来源记忆</strong>',
            `<ul>${sourceMemories.map((item) => `<li>${escapeHtml(item.id || "")} · ${escapeHtml(item.summary || item.memory_type || "")}</li>`).join("")}</ul>`,
            "</div>",
          ].join("")
        : "",
      eventList.length
        ? `<ol>${eventList.map((event) => `<li>${escapeHtml(event.action || event.event_type || "event")} ${escapeHtml(event.created_at || "")}</li>`).join("")}</ol>`
        : '<div class="agent-memory-empty">暂无审计事件。</div>',
      "</section>",
    ].join("");
  }

  async function loadItems() {
    const query = filterParams();
    const isDistillation = viewMode === "distillation";
    setStatus(isDistillation ? "正在读取记忆沉淀..." : "正在读取记忆...");
    const endpoint = isDistillation ? "api/agent-memory/distillations" : "api/agent-memory";
    const payload = await api(endpoint + (query ? `?${query}` : ""));
    items = Array.isArray(payload?.items) ? payload.items : [];
    renderItems();
    renderDetail(null);
    setStatus(`已读取 ${items.length} 条${isDistillation ? "沉淀" : "记忆"}。`, "success");
  }

  function setViewMode(mode, { reload = true } = {}) {
    viewMode = mode === "distillation" ? "distillation" : "raw";
    selectedMemoryId = "";
    items = [];
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
      delete: remove,
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
    renderDetail,
    renderItems,
    remove,
    rollbackDistillation,
    setStatus,
    setViewMode,
    syncViewControls,
    viewMode: () => viewMode,
  };
}
