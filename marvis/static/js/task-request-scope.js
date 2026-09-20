// Read requests belong to a particular visit to a task/model. Returning to the
// same ID must not revive a response from the previous visit.
export function createTaskRequestScope() {
  let identity = "";
  let generation = 0;
  let taskIdentity = "";
  let taskGeneration = 0;
  const pending = new Map();

  function select(taskId, modelId = "") {
    const next = JSON.stringify([taskId || "", modelId || ""]);
    if (next === identity) return;
    if (taskIdentity !== (taskId || "")) {
      taskIdentity = taskId || "";
      taskGeneration += 1;
    }
    identity = next;
    generation += 1;
    for (const request of pending.values()) request.controller.abort();
    pending.clear();
  }

  function capture({ includeModel = true } = {}) {
    return Object.freeze({ generation, taskGeneration, includeModel });
  }

  function current(scope) {
    return scope?.includeModel === false
      ? scope.taskGeneration === taskGeneration
      : scope?.generation === generation;
  }

  function begin(resource) {
    pending.get(resource)?.controller.abort();
    const controller = new AbortController();
    const request = Object.freeze({
      generation, resource, controller, signal: controller.signal,
    });
    pending.set(resource, request);
    return request;
  }

  function accepts(request) {
    return current(request)
      && !request.signal.aborted
      && pending.get(request.resource) === request;
  }

  function finish(request) {
    if (pending.get(request.resource) === request) pending.delete(request.resource);
  }

  return { select, capture, current, begin, accepts, finish };
}
