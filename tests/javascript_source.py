"""Select a top-level declaration for isolated JavaScript behavior tests."""


def slice_function(source: str, signature: str) -> str:
    """App declarations end with a column-zero brace, including default params."""
    start = source.index(signature)
    end = source.index("\n}", start)
    return source[start : end + 2]


def session_projection_fixture(script: str) -> str:
    """Explicit view double for pre-existing isolated app-consumer tests.

    Tests of task-session itself import the real module; this adapter supplies
    the same interface to renderer fixtures that intentionally own local state.
    """
    if "const taskSession =" in script or "const taskSession=" in script:
        return script
    return r'''
// Explicit session double for isolated rendering fixtures. Real session tests
// exercise identity, transition and message/plan acceptance separately.
const fixtureAgentRequests = new Map();
const taskSession = {
  beginAgentRequest(id) {
    const controller = new AbortController();
    const request = {taskId: id, controller, signal: controller.signal};
    fixtureAgentRequests.set(id, request);
    if (typeof agentRequestAbortControllers !== 'undefined') agentRequestAbortControllers.set(id, controller);
    return request;
  },
  finishAgentRequest(request) {
    if (fixtureAgentRequests.get(request.taskId) !== request) return;
    fixtureAgentRequests.delete(request.taskId);
    if (typeof agentRequestAbortControllers !== 'undefined') agentRequestAbortControllers.delete(request.taskId);
  },
  abortAgentRequest(id) { fixtureAgentRequests.get(id)?.controller.abort(); },
  get taskId() { return typeof selectedTaskId === 'undefined' ? null : selectedTaskId; },
  get task() { return typeof selectedTask === 'undefined' ? null : selectedTask; },
  get modelId() { return typeof projectedValidationChildTaskId === 'undefined' ? '' : projectedValidationChildTaskId; },
  get model() { return typeof projectedValidationChildTask === 'undefined' ? null : projectedValidationChildTask; },
  get messages() { return typeof agentMessages === 'undefined' ? [] : agentMessages; },
  selectTask(task, id = task?.id || null) {
    if (selectedTaskId !== id) { agentMessages = []; projectedValidationChildTaskId = ''; projectedValidationChildTask = null; }
    selectedTaskId = id; selectedTask = task; taskRequests.select(id, projectedValidationChildTaskId);
  },
  refreshTask(task) { selectedTask = task; },
  selectModel(id) { projectedValidationChildTaskId = id; projectedValidationChildTask = null; agentMessages = []; taskRequests.select(selectedTaskId, id); },
  projectModel(view, task) { if (taskRequests.current(view)) projectedValidationChildTask = task; },
  replaceMessages(view, messages) { if (taskRequests.current(view) && Array.isArray(messages)) agentMessages = messages; },
  appendMessages(view, messages) { if (taskRequests.current(view)) agentMessages = [...agentMessages, ...messages]; },
  removeMessage(view, id) { if (taskRequests.current(view)) agentMessages = agentMessages.filter(message => message.id !== id); },
};
''' + script
