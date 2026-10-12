import { api as defaultApi } from "./api.js";

export function createWorkerIsolationPanel({ container, api = defaultApi, onSettings = () => {} }) {
  if (!container) throw new Error("worker isolation container required");
  const document = container.ownerDocument;
  let current = null;
  let destroyed = false;
  let saving = false;
  let readRevision = 0;
  let reading = false;
  const select = document.createElement("select");
  select.dataset.workerIsolationField = "mode";
  for (const [value, label] of [["trusted_local", "可信本地兼容"], ["os_sandbox", "本机 OS 隔离计算"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  }
  const label = document.createElement("label");
  label.textContent = "第三方工具执行边界";
  label.append(select);
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "确认执行边界";
  button.dataset.workerIsolationAction = "save";
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  status.dataset.workerIsolationStatus = "true";
  const boundary = document.createElement("p");
  boundary.textContent = "强模式支持 macOS 上仅处理已提交输入的非内置计算工具，拒绝工作区、数据库、网络、IPC、另启进程及文件写入。内置平台工具仍以可信权限执行；Notebook 使用单独的执行设置。Linux / Windows 暂无已验证后端。";
  container.replaceChildren(label, button, status, boundary);
  function render(payload) {
    if (destroyed) return;
    current = payload.settings;
    onSettings({ ...current });
    select.value = current.worker_isolation_mode || "trusted_local";
    const capability = payload.validation?.worker_capability;
    if (select.value === "os_sandbox") {
      status.textContent = capability?.available
        ? "OS 启动预检通过。后续每次非内置计算执行都会重新核验边界；不支持的资源声明会拒绝启动。"
        : "OS 隔离当前不可用，非内置工具会拒绝启动。请检查系统后端和内存监测。";
    } else status.textContent = "可信本地模式保留现有能力，请仅运行可信材料。";
  }
  async function refresh() {
    const revision = ++readRevision;
    reading = true;
    select.disabled = true;
    button.disabled = true;
    try {
      const payload = await api("/api/settings/execution-environment");
      if (revision === readRevision) render(payload);
      return payload;
    } finally {
      if (!destroyed && revision === readRevision) {
        reading = false;
        select.disabled = saving;
        button.disabled = saving;
      }
    }
  }
  button.addEventListener("click", async () => {
    if (destroyed || saving || reading || !current) return;
    saving = true;
    button.disabled = true;
    const requestedMode = select.value;
    try {
      const payload = await api("/api/settings/execution-environment", { method: "PUT", body: JSON.stringify({
        execution_mode: current.execution_mode, kernel_name: current.kernel_name,
        conda_env_name: current.conda_env_name, python_executable: current.python_executable,
        notebook_network_mode: current.notebook_network_mode,
        notebook_memory_limit_mb: current.notebook_memory_limit_mb,
        worker_isolation_mode: requestedMode,
      }) });
      render(payload);
    } catch (_error) {
      if (!destroyed) {
        select.value = current.worker_isolation_mode || "trusted_local";
        status.textContent = "执行边界未保存。请检查系统后端、解释器和内存监测；原设置继续有效。";
      }
    } finally {
      saving = false;
      button.disabled = reading;
    }
  });
  const ready = refresh().catch(() => { if (!destroyed) status.textContent = "执行设置读取失败，请刷新后重试。"; });
  return { ready, refresh, destroy() { destroyed = true; ++readRevision; container.replaceChildren(); } };
}
