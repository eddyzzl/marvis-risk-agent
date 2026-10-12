import { apiGet } from "../api.js";

const SHA256 = /^[0-9a-f]{64}$/;

export function sqlResultBinding(response, taskId) {
  const dataset = response?.dataset;
  const workspace = response?.workspace;
  const evidence = response?.evidence;
  if (response?.status !== "completed" || dataset?.task_id !== taskId
    || workspace?.task_id !== taskId || typeof dataset?.id !== "string" || !dataset.id
    || !Number.isSafeInteger(workspace?.revision) || workspace.revision < 1
    || !SHA256.test(evidence?.content_hash || "")
    || dataset.content_hash !== evidence.content_hash
    || workspace.active_dataset_id !== dataset.id
    || workspace.active_dataset_content_hash !== evidence.content_hash
    || !Number.isSafeInteger(evidence.row_count) || evidence.row_count < 0
    || dataset.row_count !== evidence.row_count) {
    throw new Error("导入结果与数据工作区身份不一致，请刷新当前任务后核对。");
  }
  return Object.freeze({ taskId, datasetId: dataset.id, revision: workspace.revision,
    contentHash: evidence.content_hash, rowCount: evidence.row_count });
}

export function matchesSqlResultWorkspace(binding, state) {
  const snapshot = state?.serverSnapshot;
  return Boolean(binding && state?.taskId === binding.taskId && !state.dirty
    && snapshot?.task_id === binding.taskId && snapshot.revision === binding.revision
    && snapshot.active_dataset_id === binding.datasetId
    && snapshot.active_dataset_content_hash === binding.contentHash);
}

function requireCurrent(binding, getState) {
  if (!matchesSqlResultWorkspace(binding, getState())) {
    throw new Error("当前数据或工作区版本已变化，请重新核对导入结果。");
  }
}

function route(binding) {
  return `/api/tasks/${encodeURIComponent(binding.taskId)}/datasets/${encodeURIComponent(binding.datasetId)}`;
}

async function verifyRemoteWorkspace(binding, { get, getState, signal }) {
  requireCurrent(binding, getState);
  const snapshot = await get(`/api/tasks/${encodeURIComponent(binding.taskId)}/data-workspace`, { signal });
  requireCurrent(binding, getState);
  if (!matchesSqlResultWorkspace(binding, { taskId: binding.taskId, serverSnapshot: snapshot, dirty: false })) {
    throw new Error("服务端当前数据或工作区版本已变化，请刷新后核对。");
  }
}

export async function readSqlResultPreview(binding, { get = apiGet, getState, signal }) {
  requireCurrent(binding, getState);
  const preview = await get(`${route(binding)}/preview?rows=50`, { signal });
  await verifyRemoteWorkspace(binding, { get, getState, signal });
  if (!Array.isArray(preview?.columns) || !preview.columns.length
    || preview.columns.some(column => typeof column !== "string" || !column)
    || new Set(preview.columns).size !== preview.columns.length
    || !Array.isArray(preview.rows) || preview.rows.length !== Math.min(50, binding.rowCount)
    || preview.truncated !== (binding.rowCount > 50)
    || preview.rows.some(row => !row || Array.isArray(row) || typeof row !== "object"
      || Object.keys(row).some(column => !preview.columns.includes(column)))) {
    throw new Error("结果预览格式无效，未显示未经核对的数据。");
  }
  return preview;
}

export async function readSqlResultDownload(binding, { get = apiGet, getState, signal }) {
  requireCurrent(binding, getState);
  const blob = await get(`${route(binding)}/download`, { signal, responseType: "blob" });
  if (!blob || typeof blob.arrayBuffer !== "function" || !globalThis.crypto?.subtle) {
    throw new Error("此环境无法核对下载指纹，未保存文件。");
  }
  const bytes = await blob.arrayBuffer();
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  const hash = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
  if (hash !== binding.contentHash) throw new Error("下载内容指纹与确认结果不一致，未保存文件。");
  await verifyRemoteWorkspace(binding, { get, getState, signal });
  if (signal?.aborted) throw new Error("结果读取已取消，未保存文件。");
  return blob;
}

export function saveSqlResultDownload(blob, binding) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `sql-result-${binding.datasetId.replace(/[^a-zA-Z0-9_-]/g, "_")}.parquet`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
