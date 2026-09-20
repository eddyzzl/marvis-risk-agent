// Read requests belong to a particular visit to a task/model. Returning to the
// same ID must not revive a response from the previous visit.
export function createTaskRequestScope() {
  let identity = "";
  let generation = 0;
  let taskIdentity = "";
  let taskGeneration = 0;
  const pending = new Map();
  const operations = new Map();

  function select(taskId, modelId = "") {
    const next = JSON.stringify([taskId || "", modelId || ""]);
    if (next === identity) return;
    if (taskIdentity !== (taskId || "")) {
      taskIdentity = taskId || "";
      taskGeneration += 1;
    }
    identity = next;
    generation += 1;
    for (const [resource, request] of pending) {
      if (request.includeSelection === false) continue;
      request.controller.abort();
      pending.delete(resource);
    }
  }

  function capture({ includeModel = true, includeSelection = true, operation = "" } = {}) {
    return Object.freeze({ generation, taskGeneration, includeModel, includeSelection, operation,
      operationVersion: operations.get(operation) || 0 });
  }

  function current(scope) {
    const sameVisit = scope?.includeSelection === false || (scope?.includeModel === false
      ? scope.taskGeneration === taskGeneration
      : scope?.generation === generation);
    return sameVisit && (!scope.operation
      || scope.operationVersion === (operations.get(scope.operation) || 0));
  }

  function advance(operation) {
    operations.set(operation, (operations.get(operation) || 0) + 1);
    for (const request of pending.values()) {
      if (request.operation === operation) request.controller.abort();
    }
    return capture({ operation });
  }

  function begin(resource, { operation = "", includeSelection = true } = {}) {
    pending.get(resource)?.controller.abort();
    const controller = new AbortController();
    const request = Object.freeze({
      ...capture({ operation, includeSelection }), resource, controller, signal: controller.signal,
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

  return { select, capture, current, advance, begin, accepts, finish };
}
