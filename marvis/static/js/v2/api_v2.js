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
