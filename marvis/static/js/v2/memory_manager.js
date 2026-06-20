import { escapeHtml } from "../ui-utils.js";
import {
  consolidateMemory as consolidateMemoryApi,
  getMemoryDistillation as getMemoryDistillationApi,
  listMemoryDistillations as listMemoryDistillationsApi,
  rollbackMemoryDistillation as rollbackMemoryDistillationApi,
} from "./api_v2.js";

const memoryCategories = [
  "",
  "user_preference",
  "field_convention",
  "validation_pitfall",
  "task_experience",
  "model_experience",
];

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function categoryOptionsHtml(selected = "") {
  return memoryCategories.map((category) => {
    const value = String(category);
    const label = value || "all categories";
    return `<option value="${escapeHtml(value)}"${value === selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
}

function distillationTitle(item = {}) {
  return item.summary || item.distilled_summary || item.scope_key || item.id || "Untitled distillation";
}

function distillationMeta(item = {}) {
  return [
    item.category || item.memory_type || "",
    item.confidence ? `confidence ${item.confidence}` : "",
    item.support_count !== undefined ? `support ${item.support_count}` : "",
    item.status || "",
    item.superseded_by ? `superseded by ${item.superseded_by}` : "",
  ].filter(Boolean).join(" · ");
}

function distillationRowHtml(item = {}) {
  const id = String(item.id || "");
  const meta = distillationMeta(item);
  return `<article class="memory-distillation-row" data-memory-row="${escapeHtml(id)}">
    <button type="button" data-memory-distillation-id="${escapeHtml(id)}">
      <strong>${escapeHtml(distillationTitle(item))}</strong>
      ${meta ? `<span>${escapeHtml(meta)}</span>` : ""}
    </button>
    <button type="button" data-rollback-memory-distillation="${escapeHtml(id)}">Rollback</button>
  </article>`;
}

export function memoryDistillationsHtml(data = {}, options = {}) {
  const selectedCategory = options.category || "";
  const items = data.items || data.distillations || [];
  const rows = items.length
    ? items.map(distillationRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="memory-distillations">No memory distillations</div>';
  return `<section class="memory-manager">
    <header class="memory-manager-toolbar">
      <label>
        Category
        <select data-memory-category>${categoryOptionsHtml(selectedCategory)}</select>
      </label>
      <button type="button" data-consolidate-memory>Consolidate</button>
    </header>
    <div class="memory-distillation-list">${rows}</div>
    <section data-memory-detail class="memory-distillation-detail"></section>
  </section>`;
}

function sourceMemoryHtml(memory = {}) {
  const label = [
    memory.id || "",
    memory.memory_type || "",
    memory.source_task_id ? `task ${memory.source_task_id}` : "",
  ].filter(Boolean).join(" · ");
  return `<li>
    <strong>${escapeHtml(label)}</strong>
    ${memory.summary ? `<span>${escapeHtml(memory.summary)}</span>` : ""}
  </li>`;
}

function eventHtml(event = {}) {
  const label = [event.event_type || event.type || "", event.created_at || ""]
    .filter(Boolean)
    .join(" · ");
  return `<li>${escapeHtml(label || JSON.stringify(event))}</li>`;
}

export function memoryDistillationDetailHtml(payload = {}) {
  const distillation = payload.distillation || payload.memory || {};
  if (!distillation.id) {
    return '<div class="v2-empty" data-v2-empty="memory-detail">Select a memory distillation</div>';
  }
  const meta = distillationMeta(distillation);
  const sources = payload.source_memories || [];
  const events = payload.events || [];
  const predecessor = payload.predecessor || payload.restored || null;
  return `<article class="memory-distillation-detail-inner">
    <header>
      <h4>${escapeHtml(distillationTitle(distillation))}</h4>
      ${meta ? `<span>${escapeHtml(meta)}</span>` : ""}
    </header>
    ${predecessor ? `<p>Restored predecessor ${escapeHtml(predecessor.id || "")}: ${escapeHtml(distillationTitle(predecessor))}</p>` : ""}
    <section>
      <strong>Source memories</strong>
      ${sources.length
        ? `<ul>${sources.map(sourceMemoryHtml).join("")}</ul>`
        : '<div class="v2-empty" data-v2-empty="memory-sources">No source memories</div>'}
    </section>
    <section>
      <strong>Audit events</strong>
      ${events.length
        ? `<ol>${events.map(eventHtml).join("")}</ol>`
        : '<div class="v2-empty" data-v2-empty="memory-events">No audit events</div>'}
    </section>
  </article>`;
}

export function renderMemoryManagerShell(container, data = {}) {
  if (!container) {
    throw new Error("renderMemoryManagerShell requires a container");
  }
  if (container.dataset) {
    container.dataset.v2MemoryManager = "true";
  }
  container.innerHTML = memoryDistillationsHtml(data);
  return () => {};
}

export async function renderMemoryManager(container, deps = {}) {
  if (!container) {
    throw new Error("renderMemoryManager requires a container");
  }
  if (container.dataset) {
    container.dataset.v2MemoryManager = "true";
  }
  const actions = {
    listMemoryDistillations: listMemoryDistillationsApi,
    ...deps,
  };
  const category = currentCategory(container);
  const data = await actions.listMemoryDistillations({ category });
  container.innerHTML = memoryDistillationsHtml(data, { category });
  return data;
}

function currentCategory(root) {
  return String(root?.querySelector?.("[data-memory-category]")?.value || "");
}

function defaultShowMessage(message) {
  console.info(message);
}

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

export function attachMemoryHandlers(root, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachMemoryHandlers requires a stable event root");
  }
  const actions = {
    consolidateMemory: consolidateMemoryApi,
    getMemoryDistillation: getMemoryDistillationApi,
    refreshMemories: async () => {},
    rollbackMemoryDistillation: rollbackMemoryDistillationApi,
    showError: defaultShowError,
    showMessage: defaultShowMessage,
    ...deps,
  };

  const refreshQuery = () => ({ category: currentCategory(root) });
  const detailSlot = () => root.querySelector?.("[data-memory-detail]");

  const renderDetail = (payload) => {
    const slot = detailSlot();
    if (slot) {
      slot.innerHTML = memoryDistillationDetailHtml(payload);
    }
  };

  const clickHandler = async (event) => {
    const target = event.target;
    const rollbackButton = closest(target, "[data-rollback-memory-distillation]");
    if (rollbackButton?.dataset?.rollbackMemoryDistillation) {
      event.preventDefault?.();
      try {
        await actions.rollbackMemoryDistillation(rollbackButton.dataset.rollbackMemoryDistillation);
        await actions.refreshMemories(refreshQuery());
      } catch (error) {
        actions.showError(error?.message || "memory rollback failed");
      }
      return;
    }

    const detailButton = closest(target, "[data-memory-distillation-id]");
    if (detailButton?.dataset?.memoryDistillationId) {
      event.preventDefault?.();
      try {
        renderDetail(await actions.getMemoryDistillation(detailButton.dataset.memoryDistillationId));
      } catch (error) {
        actions.showError(error?.message || "memory detail failed");
      }
      return;
    }

    const consolidateButton = closest(target, "[data-consolidate-memory]");
    if (consolidateButton) {
      event.preventDefault?.();
      const category = currentCategory(root);
      try {
        const payload = await actions.consolidateMemory(category);
        await actions.refreshMemories(refreshQuery());
        const count = Object.values(payload?.consolidated || {}).reduce(
          (total, value) => total + Number(value || 0),
          0,
        );
        actions.showMessage(`Consolidated ${count} memory distillations.`);
      } catch (error) {
        actions.showError(error?.message || "memory consolidation failed");
      }
    }
  };

  const changeHandler = async (event) => {
    const categorySelect = closest(event.target, "[data-memory-category]");
    if (!categorySelect) {
      return;
    }
    try {
      await actions.refreshMemories(refreshQuery());
    } catch (error) {
      actions.showError(error?.message || "memory refresh failed");
    }
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("change", changeHandler);
  return () => {
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("change", changeHandler);
  };
}
