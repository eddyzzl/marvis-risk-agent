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

const memoryCategoryLabels = {
  "": "全部类别",
  user_preference: "用户偏好",
  field_convention: "字段口径",
  validation_pitfall: "验证坑点",
  task_experience: "任务经验",
  model_experience: "模型经验",
};

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

function categoryOptionsHtml(selected = "") {
  return memoryCategories.map((category) => {
    const value = String(category);
    const label = memoryCategoryLabels[value] || value || "全部类别";
    return `<option value="${escapeHtml(value)}"${value === selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
  }).join("");
}

function distillationTitle(item = {}) {
  return item.summary || item.distilled_summary || item.scope_key || item.id || "未命名沉淀";
}

function distillationMeta(item = {}) {
  return [
    memoryCategoryLabels[item.category || item.memory_type] || item.category || item.memory_type || "",
    item.confidence ? `置信度 ${item.confidence}` : "",
    item.support_count !== undefined ? `支持证据 ${item.support_count}` : "",
    item.status || "",
    item.superseded_by ? `被 ${item.superseded_by} 替代` : "",
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
    <button type="button" data-rollback-memory-distillation="${escapeHtml(id)}">回滚</button>
  </article>`;
}

export function memoryDistillationsHtml(data = {}, options = {}) {
  const selectedCategory = options.category || "";
  const items = data.items || data.distillations || [];
  const rows = items.length
    ? items.map(distillationRowHtml).join("")
    : '<div class="v2-empty" data-v2-empty="memory-distillations">暂无记忆沉淀</div>';
  return `<section class="memory-manager">
    <header class="memory-manager-toolbar">
      <label>
        类别
        <select data-memory-category>${categoryOptionsHtml(selectedCategory)}</select>
      </label>
      <button type="button" data-consolidate-memory>合并沉淀</button>
    </header>
    <div class="memory-distillation-list">${rows}</div>
    <section data-memory-detail class="memory-distillation-detail"></section>
  </section>`;
}

function sourceMemoryHtml(memory = {}) {
  const label = [
    memory.id || "",
    memoryCategoryLabels[memory.memory_type] || memory.memory_type || "",
    memory.source_task_id ? `任务 ${memory.source_task_id}` : "",
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
    return '<div class="v2-empty" data-v2-empty="memory-detail">请选择一个记忆沉淀</div>';
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
    ${predecessor ? `<p>已恢复前序版本 ${escapeHtml(predecessor.id || "")}: ${escapeHtml(distillationTitle(predecessor))}</p>` : ""}
    <section>
      <strong>来源记忆</strong>
      ${sources.length
        ? `<ul>${sources.map(sourceMemoryHtml).join("")}</ul>`
        : '<div class="v2-empty" data-v2-empty="memory-sources">暂无来源记忆</div>'}
    </section>
    <section>
      <strong>审计事件</strong>
      ${events.length
        ? `<ol>${events.map(eventHtml).join("")}</ol>`
        : '<div class="v2-empty" data-v2-empty="memory-events">暂无审计事件</div>'}
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
        actions.showError(error?.message || "记忆回滚失败");
      }
      return;
    }

    const detailButton = closest(target, "[data-memory-distillation-id]");
    if (detailButton?.dataset?.memoryDistillationId) {
      event.preventDefault?.();
      try {
        renderDetail(await actions.getMemoryDistillation(detailButton.dataset.memoryDistillationId));
      } catch (error) {
        actions.showError(error?.message || "记忆详情读取失败");
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
        actions.showMessage(`已合并 ${count} 条记忆沉淀。`);
      } catch (error) {
        actions.showError(error?.message || "记忆沉淀合并失败");
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
      actions.showError(error?.message || "记忆列表刷新失败");
    }
  };

  root.addEventListener("click", clickHandler);
  root.addEventListener("change", changeHandler);
  return () => {
    root.removeEventListener?.("click", clickHandler);
    root.removeEventListener?.("change", changeHandler);
  };
}
