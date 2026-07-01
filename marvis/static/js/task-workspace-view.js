import { workspaceGreetingForHour } from "./task-workspace-state.js";

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
