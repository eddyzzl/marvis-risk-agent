import { apiDelete, apiGet, apiPost } from "../api.js";

function pathPart(value) {
  return encodeURIComponent(String(value));
}

function queryPart(value) {
  return encodeURIComponent(String(value));
}

export const createPlan = (taskId, body) => apiPost(`/api/tasks/${pathPart(taskId)}/plans`, body);
export const getPlan = (planId) => apiGet(`/api/plans/${pathPart(planId)}`);
export const confirmPlan = (planId) => apiPost(`/api/plans/${pathPart(planId)}/confirm`, {});
export const runPlan = (planId) => apiPost(`/api/plans/${pathPart(planId)}/run`, {});
export const confirmStep = (planId, stepId) => (
  apiPost(`/api/plans/${pathPart(planId)}/steps/${pathPart(stepId)}/confirm`, {})
);
export const cancelPlan = (planId) => apiPost(`/api/plans/${pathPart(planId)}/cancel`, {});

export const listPlugins = (includeDisabled = false) => (
  apiGet(`/api/plugins?include_disabled=${queryPart(Boolean(includeDisabled))}`)
);

export function uploadPlugin(file) {
  const formData = new FormData();
  formData.append("file", file);
  return apiPost("/api/plugins", formData);
}

export const setPluginEnabled = (name, on) => (
  apiPost(`/api/plugins/${pathPart(name)}/${on ? "enable" : "disable"}`, {})
);
export const removePlugin = (name) => apiDelete(`/api/plugins/${pathPart(name)}`);
export const listPluginTools = (name) => apiGet(`/api/plugins/${pathPart(name)}/tools`);

export const listSkills = () => apiGet("/api/skills");
export const reloadSkills = () => apiPost("/api/skills/reload", {});
export const validateSkill = (skill) => apiPost("/api/skills/validate", { skill });

export const listDatasets = (taskId) => apiGet(`/api/tasks/${pathPart(taskId)}/datasets`);

export function uploadDataset(taskId, file, opts = {}) {
  const formData = new FormData();
  formData.append("file", file);
  if (opts.role) formData.append("role", opts.role);
  if (opts.sheet) formData.append("sheet", opts.sheet);
  return apiPost(`/api/tasks/${pathPart(taskId)}/datasets/upload`, formData);
}

export const previewDataset = (datasetId, rows = 50) => (
  apiGet(`/api/datasets/${pathPart(datasetId)}/preview?rows=${queryPart(rows)}`)
);
export const proposeJoin = (taskId, body) => (
  apiPost(`/api/tasks/${pathPart(taskId)}/joins/propose`, body)
);
export const getJoinPlan = (joinId) => apiGet(`/api/joins/${pathPart(joinId)}`);
export const confirmJoinSpec = (joinId, body) => apiPost(`/api/joins/${pathPart(joinId)}/confirm`, body);
export const executeJoin = (joinId) => apiPost(`/api/joins/${pathPart(joinId)}/execute`, {});

export const listCapabilityTiers = () => apiGet("/api/capability-tiers");

export function listMemoryDistillations({ category = "", includeSuperseded = false } = {}) {
  const params = new URLSearchParams();
  if (category) params.set("category", String(category));
  if (includeSuperseded) params.set("include_superseded", "true");
  const query = params.toString();
  return apiGet(`/api/agent-memory/distillations${query ? `?${query}` : ""}`);
}

export const getMemoryDistillation = (distillationId) => (
  apiGet(`/api/agent-memory/distillations/${pathPart(distillationId)}`)
);
export const rollbackMemoryDistillation = (distillationId) => (
  apiPost(`/api/agent-memory/distillations/${pathPart(distillationId)}/rollback`, {})
);
export function consolidateMemory(category = "") {
  const query = category ? `?category=${queryPart(category)}` : "";
  return apiPost(`/api/agent-memory/consolidate${query}`, {});
}

export function listDrafts({ taskId = "", status = "" } = {}) {
  const params = new URLSearchParams();
  if (taskId) params.set("task_id", String(taskId));
  if (status) params.set("status", String(status));
  const query = params.toString();
  return apiGet(`/api/drafts${query ? `?${query}` : ""}`);
}

export const getDraft = (draftId) => apiGet(`/api/drafts/${pathPart(draftId)}`);
export const runDraft = (draftId, inputs) => (
  apiPost(`/api/drafts/${pathPart(draftId)}/run`, { inputs })
);
export const promoteDraft = (draftId, testCases) => (
  apiPost(`/api/drafts/${pathPart(draftId)}/promote`, { test_cases: testCases })
);
export const rejectDraft = (draftId, reason) => (
  apiPost(`/api/drafts/${pathPart(draftId)}/reject`, { reason })
);
export const searchDraftWeb = (query, maxResults = 5) => (
  apiPost("/api/drafts/web-search", { query, max_results: maxResults })
);
export const fetchDraftUrl = (url, maxBytes = 500000) => (
  apiPost("/api/drafts/fetch-url", { url, max_bytes: maxBytes })
);
export const distillDraftLearning = (payload) => (
  apiPost("/api/drafts/learning-notes", payload)
);
export const authorDraftTool = (payload) => (
  apiPost("/api/drafts/author", payload)
);
