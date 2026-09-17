import { clamp, escapeHtml } from "./ui-utils.js";

const PREVIEW_GAP = 8;
const PREVIEW_MAX_WIDTH = 320;

const PERSON_ICON = [
  '<svg class="task-row-preview-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">',
  '<circle cx="12" cy="8" r="3.2"></circle>',
  '<path d="M5.5 19c0.9-3.5 3.2-5.4 6.5-5.4s5.6 1.9 6.5 5.4"></path>',
  "</svg>",
].join("");

const CALENDAR_ICON = [
  '<svg class="task-row-preview-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">',
  '<rect x="4.2" y="6.2" width="15.6" height="13.6" rx="2.2"></rect>',
  '<path d="M8 4.4v3.4M16 4.4v3.4M4.2 10.4h15.6"></path>',
  "</svg>",
].join("");

export function formatTaskRowPreviewTime(value) {
  if (!value) return "";
  try {
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(value));
  } catch (_) {
    return String(value);
  }
}

export function taskRowPreviewHtml({
  name = "",
  typeLabel = "",
  ownerLabel = "负责人",
  ownerName = "",
  createdAt = "",
} = {}) {
  const createdText = formatTaskRowPreviewTime(createdAt);
  const ownerText = String(ownerName || "-").trim() || "-";
  const rows = [
    `<div class="task-row-preview-row">${PERSON_ICON}<span><span class="task-row-preview-k">${escapeHtml(ownerLabel)}</span>${escapeHtml(ownerText)}</span></div>`,
  ];
  if (createdText) {
    rows.push(
      `<div class="task-row-preview-row">${CALENDAR_ICON}<span><span class="task-row-preview-k">创建时间</span>${escapeHtml(createdText)}</span></div>`
    );
  }
  return [
    `<strong class="task-row-preview-title">${escapeHtml(name)}</strong>`,
    typeLabel ? `<div class="task-row-preview-kind">${escapeHtml(typeLabel)}</div>` : "",
    `<div class="task-row-preview-meta">${rows.join("")}</div>`,
  ].join("");
}

export function positionTaskRowPreview(preview, anchor, sidebar) {
  if (!preview || !anchor) return;
  const rowRect = anchor.getBoundingClientRect();
  const sidebarRect = sidebar?.getBoundingClientRect?.();
  const preferredLeft = Math.round((sidebarRect?.right ?? rowRect.right) + PREVIEW_GAP);
  preview.style.width = `${PREVIEW_MAX_WIDTH}px`;
  preview.style.left = `${preferredLeft}px`;
  preview.style.top = `${Math.round(rowRect.top)}px`;
  const cardRect = preview.getBoundingClientRect();
  const maxLeft = window.innerWidth - cardRect.width - PREVIEW_GAP;
  const maxTop = window.innerHeight - cardRect.height - PREVIEW_GAP;
  preview.style.left = `${Math.round(clamp(preferredLeft, PREVIEW_GAP, Math.max(PREVIEW_GAP, maxLeft)))}px`;
  preview.style.top = `${Math.round(clamp(rowRect.top, PREVIEW_GAP, Math.max(PREVIEW_GAP, maxTop)))}px`;
}

function hoverMediaMatches() {
  return Boolean(window.matchMedia?.("(hover: hover) and (pointer: fine)")?.matches);
}

export function bindTaskRowPreview({
  list,
  preview,
  sidebar,
  getTask,
  displayName,
  typeLabel,
  ownerLabel,
} = {}) {
  let hoveredShell = null;
  let focusedShell = null;
  let suppressedShell = null;

  function hide() {
    if (!preview) return;
    preview.hidden = true;
    preview.innerHTML = "";
    preview.removeAttribute("data-task-id");
  }

  function dismiss(shell) {
    suppressedShell = shell || suppressedShell;
    hoveredShell = null;
    if (shell) focusedShell = shell;
    hide();
  }

  function show(shell) {
    if (!preview || !shell) {
      hide();
      return;
    }
    const taskId = shell.dataset.taskId || "";
    const task = typeof getTask === "function" ? getTask(taskId) : null;
    if (!task) {
      hide();
      return;
    }
    preview.innerHTML = taskRowPreviewHtml({
      name: typeof displayName === "function" ? displayName(task) : task.model_name || "",
      typeLabel: typeof typeLabel === "function" ? typeLabel(task) : "",
      ownerLabel: typeof ownerLabel === "function" ? ownerLabel(task) : "负责人",
      ownerName: task.validator || "-",
      createdAt: task.created_at || task.updated_at || "",
    });
    preview.dataset.taskId = taskId;
    preview.hidden = false;
    positionTaskRowPreview(preview, shell, sidebar);
  }

  function sync() {
    const shell = hoveredShell || focusedShell;
    if (shell?.isConnected && shell !== suppressedShell) show(shell);
    else hide();
  }

  function shellFromEvent(event) {
    const target = event.target;
    if (!(target instanceof Element)) return null;
    const shell = target.closest(".task-row-shell");
    if (!shell || !list?.contains(shell)) return null;
    return shell;
  }

  if (list) {
    list.addEventListener("pointerover", (event) => {
      if (!hoverMediaMatches()) return;
      const shell = shellFromEvent(event);
      if (!shell) return;
      if (shell === suppressedShell) return;
      suppressedShell = null;
      if (hoveredShell === shell) return;
      hoveredShell = shell;
      sync();
    });
    list.addEventListener("pointerout", (event) => {
      const next = event.relatedTarget;
      if (suppressedShell && (!(next instanceof Element) || !suppressedShell.contains(next))) {
        suppressedShell = null;
      }
      if (!hoveredShell) return;
      if (next instanceof Element && hoveredShell.contains(next)) return;
      hoveredShell = null;
      sync();
    });
    list.addEventListener("pointerdown", (event) => {
      const shell = shellFromEvent(event);
      if (shell) dismiss(shell);
    });
    list.addEventListener("click", (event) => {
      const shell = shellFromEvent(event);
      if (shell) dismiss(shell);
    });
    list.addEventListener("focusin", (event) => {
      const shell = shellFromEvent(event);
      if (!shell) return;
      if (shell === suppressedShell) {
        focusedShell = shell;
        return;
      }
      suppressedShell = null;
      focusedShell = shell;
      sync();
    });
    list.addEventListener("focusout", (event) => {
      if (!focusedShell) return;
      const next = event.relatedTarget;
      if (next instanceof Element && focusedShell.contains(next)) return;
      focusedShell = null;
      sync();
    });
    list.addEventListener("scroll", () => {
      hoveredShell = null;
      hide();
    }, { passive: true });
  }

  window.addEventListener("resize", sync);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hoveredShell = null;
      hide();
    }
  });

  return {
    hide() {
      hoveredShell = null;
      focusedShell = null;
      suppressedShell = null;
      hide();
    },
    refresh: sync,
  };
}
