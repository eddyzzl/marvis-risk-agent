import { workspaceGreetingForHour } from "./task-workspace-state.js";
import { escapeHtml } from "./ui-utils.js";

function defaultDocument() {
  return globalThis.document || null;
}

function elementGetter(root = defaultDocument()) {
  return (id) => root?.getElementById?.(id) || null;
}

function requestLayoutSync(callback, requestFrame = globalThis.requestAnimationFrame) {
  if (typeof callback !== "function") return;
  if (typeof requestFrame === "function") {
    requestFrame(callback);
  } else {
    callback();
  }
}

export function updateWorkspaceGreeting({ now = new Date(), getElementById } = {}) {
  const get = getElementById || elementGetter();
  const greeting = workspaceGreetingForHour(now.getHours());
  const target = get("workspaceGreetingText");
  if (target) target.textContent = greeting;
  return greeting;
}

function metaIcon(name) {
  const paths = {
    person: '<circle cx="12" cy="8" r="3.4"/><path d="M5.5 19a6.5 6.5 0 0 1 13 0"/>',
    type: '<path d="M4 6.5h7v7H4z"/><path d="M13 4h7v7h-7z"/><path d="M10 15h7v5h-7z"/>',
    mode: '<path d="M4 7h8M16 7h4M4 17h4M12 17h8"/><circle cx="14" cy="7" r="2.2"/><circle cx="10" cy="17" r="2.2"/>',
    folder: '<path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2.2h8A1.5 1.5 0 0 1 20 9.7V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  };
  return `<svg class="meta-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || ""}</svg>`;
}

function snapshotItem(icon, label, value, iconHtml, options = {}) {
  const valueHtml = options.copy
    ? [
        `<button class="task-snapshot-copy" type="button" data-copy="${escapeHtml(options.copy)}" aria-label="复制${escapeHtml(label)}路径" title="点击复制路径">`,
        `<strong>${escapeHtml(value)}</strong>`,
        "</button>",
      ].join("")
    : `<strong>${escapeHtml(value)}</strong>`;
  return [
    '<div class="task-snapshot-item task-meta-tile">',
    iconHtml || metaIcon(icon),
    '<div class="task-snapshot-text">',
    `<span>${escapeHtml(label)}</span>`,
    valueHtml,
    "</div>",
    "</div>",
  ].join("");
}

export function renderTaskSnapshot({
  selectedTask = null,
  getElementById,
  taskTypeLabel,
  taskKindIconHtml,
  runModeLabel,
} = {}) {
  const get = getElementById || elementGetter();
  const snapshot = get("taskSnapshot");
  if (!snapshot) return;
  if (!selectedTask) {
    snapshot.className = "workspace-task-meta empty";
    snapshot.textContent = "核心任务信息";
    return;
  }
  snapshot.className = "workspace-task-meta";
  snapshot.innerHTML = [
    '<div class="task-snapshot-list">',
    snapshotItem(
      "type",
      "任务类型",
      taskTypeLabel?.(selectedTask) || "",
      taskKindIconHtml?.(selectedTask, "meta-kind-ico"),
    ),
    snapshotItem("mode", "执行模式", runModeLabel?.(selectedTask.run_mode) || ""),
    snapshotItem("folder", "材料目录", selectedTask.source_dir, null, {
      copy: selectedTask.source_dir,
    }),
    "</div>",
  ].join("");
}

export function renderCurrentTaskWorkspace({
  selectedTask = null,
  selectedTaskId = "",
  getElementById,
  taskDisplayName,
  renderTaskSnapshot,
  setActionStatus,
  updateGreeting = updateWorkspaceGreeting,
  statusOverride = null,
  setTaskFailureActionStatus,
  taskActionStatusSnapshot,
  syncTaskHeroGlassLayout,
  requestFrame,
} = {}) {
  const get = getElementById || elementGetter();
  const hasTaskContext = Boolean(selectedTask || selectedTaskId);
  get("validationWorkspace")?.classList.toggle("is-empty", !hasTaskContext);
  const title = get("currentTaskTitle");
  const subtitle = get("currentTaskSubtitle");
  const renderSnapshot = () => {
    if (typeof renderTaskSnapshot === "function") renderTaskSnapshot();
  };
  const setStatus = (...args) => {
    if (typeof setActionStatus === "function") setActionStatus(...args);
  };
  const syncLayout = () => requestLayoutSync(syncTaskHeroGlassLayout, requestFrame);

  if (!selectedTask) {
    if (selectedTaskId) {
      if (title) title.textContent = "正在恢复任务";
      if (subtitle) subtitle.textContent = "正在加载任务内容";
      renderSnapshot();
      setStatus("");
      syncLayout();
      return;
    }
    updateGreeting({ getElementById: get });
    if (title) title.textContent = "验证任务";
    if (subtitle) subtitle.textContent = "创建任务或从左侧选择已有任务";
    renderSnapshot();
    setStatus("");
    syncLayout();
    return;
  }

  if (title) title.textContent = taskDisplayName?.(selectedTask) || "";
  if (subtitle) subtitle.textContent = "";
  renderSnapshot();
  if (statusOverride) {
    setStatus(statusOverride.message, statusOverride.kind, statusOverride.detail);
  } else if (!setTaskFailureActionStatus?.(selectedTask)) {
    const snapshot = taskActionStatusSnapshot?.(selectedTask) || { message: "", kind: "info" };
    setStatus(snapshot.message, snapshot.kind);
  }
  syncLayout();
}
