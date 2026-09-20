import { createTaskActivityOwner } from "./task-activity.js";
import { createTaskRequestScope } from "./task-request-scope.js";

// The workbench has one selection and one conversation. Panels consume these
// projections; asynchronous producers must present the visit that owns them.
export function createTaskSession() {
  const requests = createTaskRequestScope();
  const activities = createTaskActivityOwner();
  let taskId = null;
  let task = null;
  let modelId = "";
  let model = null;
  let messages = Object.freeze([]);
  let planIdentity = null;
  let plan = null;
  const agentRequests = new Map();

  function freeze(value) {
    if (value && typeof value === "object") {
      Object.values(value).forEach(freeze);
      Object.freeze(value);
    }
    return value;
  }
  const copy = value => value == null ? null : freeze(structuredClone(value));
  const workbenchId = () => modelId || taskId;

  function selectTask(value, id = value?.id || null) {
    if (value && value.id !== id) throw new TypeError("Task selection identity mismatch");
    if (taskId !== id) {
      modelId = "";
      model = null;
      messages = Object.freeze([]);
      planIdentity = null;
      plan = null;
    }
    taskId = id;
    task = copy(value);
    requests.select(taskId, modelId);
  }

  function refreshTask(value) {
    if (!value || value.id !== taskId) return false;
    task = copy(value);
    return true;
  }

  function selectModel(id = "") {
    if (id && !taskId) throw new TypeError("Select a parent task before its model");
    if (modelId !== id) {
      modelId = id;
      model = null;
      messages = Object.freeze([]);
      planIdentity = null;
      plan = null;
    }
    requests.select(taskId, modelId);
  }

  function projectModel(view, value) {
    if (!requests.current(view) || !value || value.id !== modelId) return false;
    model = copy(value);
    return true;
  }

  function replaceMessages(view, value, id = workbenchId()) {
    if (!requests.current(view) || id !== workbenchId() || !Array.isArray(value)) return false;
    messages = copy(value);
    return true;
  }

  function appendMessages(view, value) {
    return replaceMessages(view, [...messages, ...value]);
  }

  function removeMessage(view, id) {
    return replaceMessages(view, messages.filter(message => message.id !== id));
  }

  function acceptPlan(view, id, value) {
    if (!requests.current(view) || id !== workbenchId()) return false;
    if (value && value.task_id !== id) return false;
    const revision = Number(value?.replan_count || 0);
    if (planIdentity?.id === value?.id && revision < planIdentity.revision) return false;
    planIdentity = value ? Object.freeze({
      id: value.id, taskId: id, revision,
      fingerprint: value.reconciliation?.continuation?.expected_plan_fingerprint || "",
    }) : null;
    plan = copy(value);
    return true;
  }

  function beginAgentRequest(id) {
    agentRequests.get(id)?.controller.abort();
    const controller = new AbortController();
    const request = Object.freeze({ taskId: id, controller, signal: controller.signal });
    agentRequests.set(id, request);
    return request;
  }

  function finishAgentRequest(request) {
    if (agentRequests.get(request.taskId) === request) agentRequests.delete(request.taskId);
  }

  function abortAgentRequest(id) {
    agentRequests.get(id)?.controller.abort();
  }

  return Object.freeze({
    requests, activities, selectTask, refreshTask, selectModel, projectModel,
    replaceMessages, appendMessages, removeMessage, acceptPlan,
    beginAgentRequest, finishAgentRequest, abortAgentRequest,
    get taskId() { return taskId; },
    get task() { return task; },
    get modelId() { return modelId; },
    get model() { return model; },
    get messages() { return messages; },
    get planIdentity() { return planIdentity; },
    get plan() { return plan; },
  });
}
