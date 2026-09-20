"""Late responses cross the real request owner and app projection adapters."""

from pathlib import Path
import subprocess

import pytest

from tests.javascript_source import slice_function


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "marvis/static/app.js").read_text()


def run_node(body: str, *functions: str) -> None:
    helpers = "\n".join(slice_function(APP, name) for name in functions)
    script = '''
import assert from "node:assert/strict";
import { createTaskRequestScope } from "./marvis/static/js/task-request-scope.js";
const taskRequests = createTaskRequestScope();
let selectedTaskId = null;
let selectedTask = null;
let projectedValidationChildTaskId = "";
let projectedValidationChildTask = null;
let agentMessages = [];
const usesAgentValidationWorkbench = task => task?.task_type === "validation_batch";
const pending = [];
const painted = [];
function api(path, options = {}) {
  return new Promise((resolve, reject) => pending.push({ path, options, resolve, reject }));
}
''' + helpers + body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


IDENTITY = (
    "function setSelectedTask(", "function workbenchTask(",
    "function workbenchTaskId(", "function isWorkbenchTaskId(",
)


def test_scope_invalidates_visits_and_only_finishes_its_own_request():
    run_node('''
taskRequests.select("a", "m1");
const parent = taskRequests.capture({ includeModel: false });
const first = taskRequests.begin("messages");
const evidence = taskRequests.begin("evidence");
const second = taskRequests.begin("messages");
assert.equal(first.signal.aborted, true);
assert.equal(taskRequests.accepts(first), false);
taskRequests.finish(first);
assert.equal(taskRequests.accepts(second), true);
assert.equal(taskRequests.accepts(evidence), true);
taskRequests.select("a", "m2");
assert.equal(taskRequests.current(parent), true);
assert.equal(second.signal.aborted, true);
taskRequests.select("b");
taskRequests.select("a", "m1");
assert.equal(taskRequests.current(parent), false);
assert.equal(taskRequests.accepts(first), false);
const third = taskRequests.begin("messages");
taskRequests.select("a", "m1");
assert.equal(taskRequests.accepts(third), true);
taskRequests.finish(third);
assert.equal(taskRequests.accepts(third), false);
''')


@pytest.mark.parametrize("loader", ["loadTaskEvidence", "loadReportFields", "loadAgentMessages"])
def test_projection_rejects_reordered_reads_and_previous_visit(loader):
    run_node('''
const renderEvidence = value => painted.push(value.value);
const renderMetricPreview = value => painted.push(value.value);
const renderAgentConversation = () => painted.push(agentMessages[0]?.value);
const resetEvidenceSummaries = () => painted.push("empty");
const loadValidationInputContract = async () => {};
const notebookReproducibilityComplete = () => true;
const findTaskInCache = () => selectedTask;
const selectedTaskIsAgentMode = () => true;
const taskUsesPlanRail = () => false;
const agentMessageCanPollIncrementally = () => false;
const shouldPreserveOptimisticAgentMessages = () => false;
const mergeIncrementalAgentMessages = () => { throw Error("unexpected incremental"); };
const payload = value => ({ value, metric_values: { value }, messages: [{ id: value, value }] });
setSelectedTask({ id: "a" });
const first = LOADER("a");
const second = LOADER("a");
pending[1].resolve(payload("new"));
await second;
pending[0].resolve(payload("old"));
await first;
assert.deepEqual(painted, ["new"]);
assert.equal(pending[0].options.signal.aborted, true);
const previousVisit = LOADER("a");
setSelectedTask({ id: "b" });
setSelectedTask({ id: "a" });
const currentVisit = LOADER("a");
pending[2].resolve(payload("previous-visit"));
await previousVisit;
assert.deepEqual(painted, ["new"]);
pending[3].resolve(payload("current-visit"));
await currentVisit;
assert.deepEqual(painted, ["new", "current-visit"]);
const staleFailure = LOADER("a");
setSelectedTask({ id: "b" });
pending[4].reject(new Error("old failure"));
await staleFailure;
assert.deepEqual(painted, ["new", "current-visit"]);
'''.replace("LOADER", loader), *IDENTITY, f"async function {loader}(")


def test_model_projection_never_assigns_late_child_or_finishes_another_visit():
    run_node('''
let projectedChildContentLoadVersion = 0;
let suppressAgentAutoScrollTaskId = null;
let activeValidationView = "metrics";
const rememberValidationView = () => {};
const resetAgentTypingState = () => {};
const beginTaskContentLoad = () => {};
const finishTaskContentLoad = id => painted.push(id);
const rememberSelectedTaskId = () => {};
const loadTaskEvidence = async () => {};
const loadAgentMessages = async () => {};
const loadReportFields = async () => {};
const renderAll = () => {};
const nextAnimationFrame = async () => {};
const selectValidationView = () => {};
const $ = () => null;
setSelectedTask({ id: "batch", task_type: "validation_batch" });
const first = applyProjectedValidationChild("m1");
const second = applyProjectedValidationChild("m2");
pending[1].resolve({ id: "m2", marker: "current" });
assert.equal(await second, true);
pending[0].resolve({ id: "m1", marker: "late" });
assert.equal(await first, false);
assert.equal(projectedValidationChildTask.id, "m2");
assert.deepEqual(painted, ["m2"]);
const previous = applyProjectedValidationChild("m1");
setSelectedTask({ id: "other" });
setSelectedTask({ id: "batch", task_type: "validation_batch" });
const current = applyProjectedValidationChild("m1");
pending[2].resolve({ id: "m1", marker: "previous-visit" });
assert.equal(await previous, false);
assert.equal(projectedValidationChildTask, null);
pending[3].resolve({ id: "m1", marker: "new-visit" });
assert.equal(await current, true);
assert.equal(projectedValidationChildTask.marker, "new-visit");
assert.deepEqual(painted, ["m2", "m1"]);
''', *IDENTITY, "function isCurrentProjectedChildLoad(", "async function applyProjectedValidationChild(")


def test_placeholder_restoration_invalidates_previous_visit():
    run_node('''
let stored = "a";
const storedSelectedTaskId = () => stored;
restoreSelectedTaskPlaceholder();
assert.equal(selectedTaskId, "a");
assert.equal(selectedTask, null);
const old = taskRequests.begin("messages");
setSelectedTask(null);
restoreSelectedTaskPlaceholder();
assert.equal(taskRequests.accepts(old), false);
stored = "b";
restoreSelectedTaskPlaceholder();
assert.equal(selectedTaskId, "a");
''', *IDENTITY, "function restoreSelectedTaskPlaceholder(")


def test_contract_read_cannot_reopen_a_previous_model_visit():
    run_node('''
let validationInputContractLoadVersion = 0;
const usesPmmlScoringWorkflow = () => true;
const renderValidationInputContract = value => painted.push(value.id);
const selectedTaskIsValidationBatch = () => false;
const $ = () => null;
setSelectedTask({ id: "a" });
const previous = loadValidationInputContract("a");
setSelectedTask({ id: "b" });
setSelectedTask({ id: "a" });
pending[0].resolve({ id: "old-contract" });
await previous;
assert.deepEqual(painted, []);
const current = loadValidationInputContract("a");
pending[1].resolve({ id: "current-contract" });
await current;
assert.deepEqual(painted, ["current-contract"]);
''', *IDENTITY, "async function loadValidationInputContract(")


def test_action_failure_and_finally_cannot_repaint_a_later_visit():
    run_node('''
const busy = [];
let release;
const claimBusy = (action, _message, taskId) => { busy.push([action, taskId]); return {taskId}; };
const releaseBusy = lease => { if (lease) busy.push([null, lease.taskId]); };
const renderAll = () => painted.push("render");
const setActionStatus = () => painted.push("status");
const setCreateStatus = () => painted.push("create-error");
const renderActionError = () => painted.push("action-error");
const refreshTasks = async () => {};
setSelectedTask({ id: "a" });
const running = runAction(() => new Promise((_resolve, reject) => release = reject), { actionId: "metrics" });
setSelectedTask({ id: "b" });
setSelectedTask({ id: "a" });
release(new Error("previous visit failed"));
await running;
assert.deepEqual(painted, []);
assert.deepEqual(busy, [["metrics", "a"], [null, "a"]]);
await runAction(async () => {});
assert.deepEqual(painted, ["render"]);
''', *IDENTITY, "async function runAction(")


def test_task_list_is_global_but_only_latest_response_updates_cache():
    run_node('''
let taskCache=[];
const syncSelectedTaskFromCache=()=>painted.push(taskCache[0].id);
const ensureActiveTaskProgressPolling=()=>{};
setSelectedTask({id:"a"});
const old=refreshTasks();
setSelectedTask({id:"b"});
assert.equal(pending[0].options.signal.aborted,false);
const fresh=refreshTasks();
pending[1].resolve([{id:"latest"}]); await fresh;
pending[0].resolve([{id:"stale"}]); await old;
assert.deepEqual(taskCache,[{id:"latest"}]);
assert.deepEqual(painted,["latest"]);
''', *IDENTITY, "async function refreshTasks(")
