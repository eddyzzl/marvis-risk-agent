import { escapeHtml } from "../ui-utils.js";
import { getDataWorkspace, listDatasets, putDataWorkspace, uploadDataset } from "./api_v2.js";

export function createModelMonitoringPanel(dependencies = {}) {
  const get = dependencies.getElementById || (id => document.getElementById(id));
  const api = dependencies.api;
  const task = dependencies.getSelectedTask;
  const capture = dependencies.captureView || (() => task()?.id);
  const current = dependencies.isCurrentView || (view => view === task()?.id);
  const list = dependencies.listDatasets || listDatasets;
  const workspace = dependencies.getDataWorkspace || getDataWorkspace;
  const save = dependencies.putDataWorkspace || putDataWorkspace;
  const upload = dependencies.uploadDataset || uploadDataset;
  const el = Object.fromEntries([
    "Panel", "Form", "Model", "Dataset", "Target", "File", "Refresh", "Submit", "Status", "Limits",
  ].map(name => [name, get(`modelMonitoring${name}`)]));
  let taskId = "", version = 0, pending = false, snapshot = null, datasets = [], models = [], stale = false;

  function status(message, kind = "") {
    el.Status.textContent = message;
    el.Status.className = `model-monitoring-status ${kind}`;
  }

  function availability() {
    const selected = task();
    const visible = selected?.task_type === "modeling";
    el.Panel.hidden = !visible;
    if ((visible ? selected.id : "") !== taskId) {
      taskId = visible ? selected.id : "";
      version += 1;
      snapshot = null; datasets = []; models = []; pending = false; stale = false;
      el.Model.innerHTML = ""; el.Dataset.innerHTML = ""; el.Target.innerHTML = "";
      el.File.value = "";
      status(visible ? "展开后读取已选模型与本任务的数据。" : "");
      if (visible && el.Panel.open) void reload();
    }
    for (const name of ["Refresh", "File", "Model", "Dataset", "Target"]) el[name].disabled = pending || !visible;
    el.Submit.disabled = pending || stale || !snapshot || !models.length || !datasets.length || !visible;
    return visible;
  }

  function targets() {
    const selected = datasets.find(item => item.id === el.Dataset.value);
    el.Target.innerHTML = '<option value="">不提供标签，仅检查漂移</option>' + (selected?.columns || []).map(column => (
      `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)}</option>`
    )).join("");
    el.Target.value = "";
    limits();
  }

  function limits() {
    el.Limits.textContent = (el.Target.value
      ? "使用声明的标签列计算效果指标；标签成熟度仍未知。"
      : "缺少标签时仅检查分数和特征漂移，KS/AUC 不可用。")
      + " 默认技术阈值不代表业务验收目标；本次结果不能证明真实业务效果达标。";
  }

  async function reload(preferredDataset = "") {
    if (!taskId || pending) return false;
    const view = capture(), id = taskId, operation = ++version;
    pending = true; availability(); status("正在读取已选模型与数据…");
    try {
      const [experiments, data, state] = await Promise.all([
        api(`api/tasks/${encodeURIComponent(id)}/experiments`), list(id), workspace(id),
      ]);
      if (!current(view) || operation !== version) return false;
      models = (experiments.experiments || []).filter(item => item.task_id === id && item.status === "selected" && item.artifact_id);
      datasets = (data.datasets || []).filter(item => item.task_id === id && item.content_hash && item.role !== "modeling.scored");
      snapshot = state; stale = false;
      el.Model.innerHTML = models.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.recipe_id || item.recipe || "模型")} · ${escapeHtml(item.id.slice(-8))}（已选）</option>`).join("");
      el.Dataset.innerHTML = '<option value="">请选择本期数据</option>' + datasets.map(item => (
        `<option value="${escapeHtml(item.id)}">${escapeHtml(String(item.source_path || item.id).split(/[\\/]/).pop())} · ${item.row_count} 行</option>`
      )).join("");
      const preferred = preferredDataset || state.active_dataset_id;
      el.Dataset.value = datasets.some(item => item.id === preferred) ? preferred : "";
      targets();
      status(models.length ? "选择本期数据，生成计划后在对话中确认执行。" : "尚无已选模型。请先完成模型选择，再刷新此处。", models.length ? "" : "warning");
      return true;
    } catch (error) {
      if (current(view) && operation === version) { snapshot = null; status(error.message || "读取失败，请刷新。", "error"); }
      return false;
    } finally {
      if (operation === version) pending = false;
      availability();
    }
  }

  async function submit(event) {
    event?.preventDefault?.();
    if (pending || stale || !snapshot || !availability()) return false;
    const id = taskId, view = capture(), operation = ++version;
    const dataset = datasets.find(item => item.id === el.Dataset.value);
    const model = models.find(item => item.id === el.Model.value);
    if (!dataset || !model) { status("请选择已选模型与本期数据。", "error"); return false; }
    if (dependencies.workspaceController?.getState()?.dirty) {
      status("数据工作区有未保存修改，请先保存或撤销，再刷新监控选项。", "error"); return false;
    }
    const target = el.Target.value || null;
    const lease = dependencies.beginActivity ? dependencies.beginActivity("panel:model-monitoring", id) : {};
    if (!lease) return false;
    pending = true; availability(); status("正在核验模型和数据，生成监控计划…");
    try {
      if (snapshot.active_dataset_id !== dataset.id || snapshot.active_dataset_content_hash !== dataset.content_hash) {
        const next = await save(id, {
          active_dataset_id: dataset.id, active_dataset_content_hash: dataset.content_hash,
          page: "overview", selected_field: null,
          semantic_mapping: { target_col: null, field_roles: {}, business_names: {} },
        }, snapshot.revision);
        if (!current(view) || operation !== version) return false;
        snapshot = next;
      }
      const result = await api(`api/tasks/${encodeURIComponent(id)}/agent/messages`, {
        method: "POST", body: JSON.stringify({
          content: "为已选模型生成本期监控计划，待审阅计划后再执行。",
          model_monitoring_request: {
            experiment_id: model.id, dataset_id: dataset.id, expected_content_hash: dataset.content_hash,
            workspace_revision: snapshot.revision, analysis_generation: snapshot.analysis_generation,
            target_col: target,
          },
        }),
      });
      if (!current(view) || operation !== version) return false;
      if (Array.isArray(result.messages)) dependencies.onMessages?.(result.messages, view);
      status("监控计划已生成，尚未运行。请在对话中审阅并确认计划。", "success");
      await dependencies.onSubmitted?.(result, view);
      return true;
    } catch (error) {
      if (current(view) && operation === version) {
        stale = [409, 412].includes(Number(error.status));
        status(stale
          ? "当前任务或数据状态已变化。请刷新选项后重新选择，旧请求不会自动重试。"
          : error.message || "提交失败。", "error");
      }
      return false;
    } finally {
      if (operation === version) pending = false;
      dependencies.endActivity?.(lease);
      availability();
    }
  }

  async function uploadFile() {
    const file = el.File.files?.[0];
    if (!file || pending || !taskId) return false;
    const id = taskId, view = capture(), operation = ++version;
    const lease = dependencies.beginActivity ? dependencies.beginActivity("panel:monitoring-upload", id) : {};
    if (!lease) return false;
    pending = true; availability(); status("正在上传本期数据…");
    try {
      const result = await upload(id, file, { role: "unknown" });
      if (!current(view) || operation !== version) return false;
      pending = false;
      return await reload(result.datasets?.length === 1 ? result.datasets[0].id : "");
    } catch (error) {
      if (current(view) && operation === version) status(error.message || "上传失败。", "error");
      return false;
    } finally {
      if (operation === version) pending = false;
      if (current(view)) el.File.value = "";
      dependencies.endActivity?.(lease);
      availability();
    }
  }

  el.Form.addEventListener("submit", submit);
  el.Refresh.addEventListener("click", () => reload());
  el.Dataset.addEventListener("change", targets);
  el.Target.addEventListener("change", limits);
  el.File.addEventListener("change", uploadFile);
  el.Panel.addEventListener("toggle", () => { if (el.Panel.open && !snapshot) void reload(); });
  availability();
  return { renderAvailability: availability, reload, submit, uploadFile };
}
