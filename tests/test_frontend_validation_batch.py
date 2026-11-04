from tests.javascript_source import session_projection_fixture
import json
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _run_batch_module(script_body: str) -> None:
    module_url = (STATIC_DIR / "js" / "validation-batch.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
const batch = await import({json.dumps(module_url)});
{script_body}
"""
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )




def test_batch_deep_link_prefers_query_parent_without_promoting_child():
    _run_batch_module(
        """
const link = batch.parseTaskDeepLink("?task=parent-1&item=child-2");
assert.deepEqual(link, { taskId: "parent-1", itemId: "child-2" });
assert.equal(batch.preferredStartupTaskId(link, "stored-task"), "parent-1");
assert.equal(batch.preferredStartupTaskId({ taskId: "", itemId: "child-only" }, "stored-task"), "stored-task");
assert.notEqual(batch.preferredStartupTaskId(link, "stored-task"), link.itemId);
assert.deepEqual(
  batch.parseTaskDeepLink(batch.taskSelectionSearch("?task=stale-batch&item=child-1", { taskId: "other-task" })),
  { taskId: "other-task", itemId: "" },
);
assert.deepEqual(
  batch.parseTaskDeepLink(batch.taskSelectionSearch("?task=stale-batch&item=child-1", { taskId: "parent-1", itemId: "child-2" })),
  { taskId: "parent-1", itemId: "child-2" },
);
assert.equal(batch.taskSelectionSearch("?task=stale-batch&item=child-1", { taskId: "" }), "");
const history = { state: { keep: true }, url: "", replaceState(state, _title, url) { this.state = state; this.url = url; } };
batch.syncTaskDeepLink(history, { pathname: "/", search: "?task=stale-batch&item=child-1", hash: "" }, { taskId: "stored-task" });
assert.equal(history.url, "/?task=stored-task");
assert.deepEqual(history.state, { keep: true });
batch.syncTaskDeepLink(history, { pathname: "/", search: "?task=stale-batch", hash: "#x" }, { taskId: "" });
assert.equal(history.url, "/#x");
"""
    )


def test_batch_overview_normalizes_current_and_future_payload_links():
    _run_batch_module(
        r"""
const payload = batch.normalizeValidationBatchPayload({
  batch: {
    parent_task_id: "parent-1",
    status: "partial_failure",
    item_count: 3,
    summary_download_url: "/api/validation-batches/parent-1/summary",
  },
  items: [
    {
      id: "item-1",
      child_task_id: "child-pass",
      ordinal: 1,
      model_name: "模型A",
      model_version: "v1",
      status: "succeeded",
      oot_ks: 0.31,
      oot_psi: 0.04,
      pmml_status: "pass",
      stress_risk: "low",
      report_complete: true,
      outcome: "pass",
      input_contract_url: "/api/tasks/child-pass/validation-input-contract",
      word_report_download_url: "/api/tasks/child-pass/report/download",
      analysis_download_url: "/api/tasks/child-pass/analysis/download",
    },
    {
      id: "item-2",
      child_task_id: "child-review",
      ordinal: 2,
      model_name: "模型B",
      status: "awaiting_confirmation",
      input_contract_status: "pending_confirmation",
      outcome: "manual_review",
      error_message: "PSI 达到预警阈值",
      manual_review_url: "/?task=parent-1&item=child-review",
      contract_url: "/api/tasks/child-review/validation-input-contract",
    },
    {
      id: "item-3",
      child_task_id: "child-failed",
      ordinal: 3,
      model_name: "模型C",
      status: "failed",
      failure_reason: "PMML 无法加载",
    },
  ],
});

assert.equal(payload.parentTaskId, "parent-1");
assert.deepEqual(payload.counts, {
  total: 3,
  queued: 0,
  running: 0,
  awaitingConfirmation: 1,
  succeeded: 1,
  failed: 1,
});
assert.equal(payload.items[1].contractUrl, "api/tasks/child-review/validation-input-contract");
assert.equal(payload.items[1].inputContractStatus, "pending_confirmation");
assert.equal(payload.items[0].wordReportDownloadUrl, "api/tasks/child-pass/report/download");
assert.equal(payload.items[0].analysisDownloadUrl, "api/tasks/child-pass/analysis/download");
assert.equal(payload.items[2].errorMessage, "PMML 无法加载");

const html = batch.renderValidationBatchOverview(payload, { requestedItemId: "child-review" });
assert.match(html, /批次总览/);
assert.match(html, /3 个模型/);
// Batch 4 (2026-10-14 redesign): overview counters use the canonical words.
assert.match(html, /待确认 \/ 需复核/);
assert.match(html, /PSI 达到预警阈值/);
assert.match(html, /确认合同/);
assert.match(html, /data-batch-contract-task-id="child-review"/);
assert.doesNotMatch(html, /target="_blank"/);
assert.doesNotMatch(html, /href="api\/tasks\/child-review\/validation-input-contract"/);
assert.match(html, /href="api\/tasks\/child-pass\/report\/download">Word<\/a>/);
assert.match(html, /href="api\/tasks\/child-pass\/analysis\/download">Excel<\/a>/);
assert.match(html, /\?task=parent-1&amp;item=child-review/);
assert.match(html, /下载汇总 Excel/);
assert.match(html, /data-batch-child-task-id="child-review"/);
assert.match(html, /is-deep-linked/);

const withoutDownload = batch.renderValidationBatchOverview(
  batch.normalizeValidationBatchPayload({ batch: { parent_task_id: "p", status: "created", item_count: 1 }, items: [] }),
);
assert.doesNotMatch(withoutDownload, /汇总待生成/);
assert.doesNotMatch(withoutDownload, /下载汇总 Excel/);
assert.match(withoutDownload, /data-batch-start="true"/);
assert.match(withoutDownload, />启动批次</);

const awaiting = batch.renderValidationBatchOverview(
  batch.normalizeValidationBatchPayload({
    batch: { parent_task_id: "p", status: "awaiting_confirmation", item_count: 1 },
    items: [{
      child_task_id: "child",
      status: "awaiting_confirmation",
      input_contract_status: "pending_confirmation",
      model_name: "模型D",
    }],
  }),
);
assert.match(awaiting, />确认合同后继续</);
assert.match(awaiting, /data-batch-start="true"[^>]*disabled/);
assert.match(awaiting, /都按这个/);

const awaitingWithReadyContract = batch.renderValidationBatchOverview(
  batch.normalizeValidationBatchPayload({
    batch: { parent_task_id: "p", status: "awaiting_confirmation", item_count: 1 },
    items: [{
      child_task_id: "child",
      status: "awaiting_confirmation",
      input_contract_status: "ready",
      model_name: "模型D",
    }],
  }),
);
assert.doesNotMatch(awaitingWithReadyContract, /data-batch-start="true"[^>]*disabled/);
assert.match(awaitingWithReadyContract, /合同已确认/);
assert.doesNotMatch(awaitingWithReadyContract, /data-batch-contract-task-id="child"/);

const running = batch.renderValidationBatchOverview(
  batch.normalizeValidationBatchPayload({ batch: { parent_task_id: "p", status: "running", item_count: 1 }, items: [] }),
);
assert.match(running, /data-batch-start="true"[^>]*disabled/);
assert.match(running, />批次运行中</);

const failed = batch.renderValidationBatchOverview(
  batch.normalizeValidationBatchPayload({ batch: { parent_task_id: "p", status: "failed", item_count: 1 }, items: [] }),
);
assert.match(failed, /data-batch-start="true"/);
assert.match(failed, />修复后重试</);

assert.equal(batch.validationBatchItemNeedsContractConfirmation({ status: "awaiting_confirmation" }), true);
assert.equal(batch.validationBatchItemNeedsContractConfirmation({
  status: "awaiting_confirmation",
  inputContractStatus: "ready",
}), false);
assert.equal(batch.validationBatchItemNeedsContractConfirmation({ status: "pending" }), true);
assert.equal(batch.validationBatchItemNeedsContractConfirmation({ status: "review_required" }), false);
assert.equal(batch.validationBatchItemNeedsContractConfirmation({ status: "succeeded" }), false);
assert.equal(batch.validationBatchStartAction(
  "partial_failure",
  { awaitingConfirmationCount: 1 },
).disabled, true);

const unsafeDownloads = batch.normalizeValidationBatchPayload({
  batch: { parent_task_id: "p", status: "completed", item_count: 1 },
  items: [{
    child_task_id: "child",
    status: "succeeded",
    word_report_download_url: "javascript:alert(1)",
    analysis_download_url: "//evil.example/report.xlsx",
  }],
});
const unsafeHtml = batch.renderValidationBatchOverview(unsafeDownloads);
assert.doesNotMatch(unsafeHtml, />Word<\/a>/);
assert.doesNotMatch(unsafeHtml, />Excel<\/a>/);
"""
    )


def test_batch_item_focus_marks_and_scrolls_only_the_matching_row():
    _run_batch_module(
        """
function row(childTaskId) {
  const classes = new Set();
  return {
    dataset: { batchChildTaskId: childTaskId },
    classList: {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
    },
    scrollCount: 0,
    scrollIntoView() { this.scrollCount += 1; },
  };
}
const first = row("child-1");
const target = row("child-2");
const panel = { querySelectorAll: () => [first, target] };

assert.equal(batch.focusValidationBatchItem(panel, "child-2"), true);
assert.equal(first.classList.contains("is-deep-linked"), false);
assert.equal(target.classList.contains("is-deep-linked"), true);
assert.equal(target.scrollCount, 1);
assert.equal(batch.focusValidationBatchItem(panel, "missing"), false);
"""
    )


def test_batch_controller_confirms_before_start_then_refreshes_parent_and_overview():
    _run_batch_module(
        """
const classes = new Set(["hidden"]);
const panel = {
  innerHTML: "",
  attrs: {},
  classList: {
    toggle(name, force) { force ? classes.add(name) : classes.delete(name); },
  },
  setAttribute(name, value) { this.attrs[name] = value; },
  addEventListener() {},
  querySelectorAll() { return []; },
};
const task = { id: "parent-1", task_type: "validation_batch" };
const calls = [];
let status = "created";
let confirms = 0;
let parentRefreshes = 0;
const controller = batch.createValidationBatchPanelController({
  api: async (url, options = {}) => {
    calls.push([url, options.method || "GET"]);
    if (options.method === "POST") {
      status = "running";
      return { status: "accepted" };
    }
    return { batch: { parent_task_id: "parent-1", status, item_count: 1 }, items: [] };
  },
  getElementById: () => panel,
  getSelectedTask: () => task,
  confirmStart: async (context) => {
    confirms += 1;
    assert.equal(context.parentTaskId, "parent-1");
    assert.equal(context.status, "created");
    return true;
  },
  refreshParentTask: async () => { parentRefreshes += 1; },
  schedulePoll: () => 1,
  cancelPoll: () => {},
});

await controller.selectTask(task);
assert.equal(await controller.startOrContinue(), true);
assert.equal(confirms, 1);
assert.equal(parentRefreshes, 1);
assert.deepEqual(calls, [
  ["api/validation-batches/parent-1", "GET"],
  ["api/validation-batches/parent-1/start", "POST"],
  ["api/validation-batches/parent-1", "GET"],
]);
assert.match(panel.innerHTML, /批次运行中/);
"""
    )


def test_batch_controller_keeps_running_when_parent_refresh_fails_after_start_acceptance():
    _run_batch_module(
        """
const panel = {
  innerHTML: "",
  classList: { toggle() {} },
  setAttribute() {},
  addEventListener() {},
  querySelectorAll() { return []; },
};
const task = { id: "parent-1", task_type: "validation_batch" };
const calls = [];
const errors = [];
const timers = new Map();
let nextTimerHandle = 1;
let status = "created";
const controller = batch.createValidationBatchPanelController({
  api: async (url, options = {}) => {
    calls.push([url, options.method || "GET"]);
    if (options.method === "POST") {
      status = "running";
      return { status: "accepted" };
    }
    return { batch: { parent_task_id: "parent-1", status, item_count: 1 }, items: [] };
  },
  getElementById: () => panel,
  getSelectedTask: () => task,
  confirmStart: async () => true,
  refreshParentTask: async () => { throw new Error("shared task refresh unavailable"); },
  onError: (message) => errors.push(message),
  schedulePoll: (callback) => {
    const handle = nextTimerHandle++;
    timers.set(handle, callback);
    return handle;
  },
  cancelPoll: (handle) => timers.delete(handle),
});

await controller.selectTask(task);
assert.equal(await controller.startOrContinue(), true);
assert.equal(calls.filter(([, method]) => method === "POST").length, 1);
assert.equal(errors.length, 0);
assert.match(panel.innerHTML, /批次运行中/);
assert.equal(timers.size, 1);
"""
    )


def test_batch_controller_retries_poll_after_transient_running_detail_failure():
    _run_batch_module(
        """
const panel = {
  innerHTML: "",
  classList: { toggle() {} },
  setAttribute() {},
  addEventListener() {},
  querySelectorAll() { return []; },
};
const task = { id: "parent-1", task_type: "validation_batch" };
const timers = new Map();
let nextTimerHandle = 1;
const errors = [];
const recoveries = [];
let failNextDetail = false;
const controller = batch.createValidationBatchPanelController({
  api: async () => {
    if (failNextDetail) {
      failNextDetail = false;
      throw new Error("temporary detail failure");
    }
    return {
      batch: { parent_task_id: "parent-1", status: "running", item_count: 1 },
      items: [{ child_task_id: "child-1", status: "running" }],
    };
  },
  getElementById: () => panel,
  getSelectedTask: () => task,
  refreshParentTask: async () => {},
  onError: (message) => errors.push(message),
  onRecovered: (context) => recoveries.push(context),
  schedulePoll: (callback) => {
    const handle = nextTimerHandle++;
    timers.set(handle, callback);
    return handle;
  },
  cancelPoll: (handle) => timers.delete(handle),
});

await controller.selectTask(task);
assert.equal(timers.size, 1);
const [firstHandle, firstPoll] = timers.entries().next().value;
timers.delete(firstHandle);
failNextDetail = true;
await firstPoll();
assert.deepEqual(errors, ["temporary detail failure"]);
assert.equal(recoveries.length, 0);
assert.equal(timers.size, 1);
const [secondHandle, secondPoll] = timers.entries().next().value;
timers.delete(secondHandle);
await secondPoll();
assert.match(panel.innerHTML, /批次运行中/);
assert.equal(timers.size, 1);
assert.deepEqual(recoveries, [{
  parentTaskId: "parent-1",
  message: "temporary detail failure",
}]);
"""
    )


def test_batch_poll_syncs_parent_after_detail_enters_confirmation_gate_and_retries_once():
    _run_batch_module(
        """
const panel = {
  innerHTML: "",
  classList: { toggle() {} },
  setAttribute() {},
  addEventListener() {},
  querySelectorAll() { return []; },
};
const task = {
  id: "parent-1",
  task_type: "validation_batch",
  active_job_kind: "validation_batch",
};
const timers = new Map();
let nextTimerHandle = 1;
let detailStatus = "running";
let gateRefreshAttempts = 0;
const controller = batch.createValidationBatchPanelController({
  api: async () => ({
    batch: { parent_task_id: "parent-1", status: detailStatus, item_count: 1 },
    items: [{
      child_task_id: "child-1",
      status: detailStatus === "running" ? "running" : "awaiting_confirmation",
      input_contract_status: "pending_confirmation",
    }],
  }),
  getElementById: () => panel,
  getSelectedTask: () => task,
  refreshParentTask: async () => {
    if (detailStatus === "running") return false;
    gateRefreshAttempts += 1;
    if (gateRefreshAttempts === 1) return false;
    task.active_job_kind = null;
    return true;
  },
  schedulePoll: (callback) => {
    const handle = nextTimerHandle++;
    timers.set(handle, callback);
    return handle;
  },
  cancelPoll: (handle) => timers.delete(handle),
});

await controller.selectTask(task);
assert.equal(timers.size, 1);
const [runningHandle, runningPoll] = timers.entries().next().value;
timers.delete(runningHandle);
detailStatus = "awaiting_confirmation";
await runningPoll();
assert.match(panel.innerHTML, /确认合同后继续/);
assert.equal(task.active_job_kind, "validation_batch");
assert.equal(gateRefreshAttempts, 1);
assert.equal(timers.size, 1);

const [syncHandle, syncParent] = timers.entries().next().value;
timers.delete(syncHandle);
await syncParent();
assert.equal(task.active_job_kind, null);
assert.equal(gateRefreshAttempts, 2);
assert.equal(timers.size, 0);
"""
    )

    app = _read_static("app.js")
    assert "refreshParentTask: async ({ parentTaskId } = {}) => {" in app
    assert 'return taskServerBusyAction(taskSession.task) !== "validation_batch";' in app


def test_batch_detail_recovery_only_clears_the_matching_transient_status():
    app = _read_static("app.js")

    assert "onRecovered: restoreValidationBatchActionStatusAfterRecovery" in app
    function_start = app.index("function restoreValidationBatchActionStatusAfterRecovery")
    function_end = app.index("\n\nconst validationBatchPanelController", function_start)
    function_body = app[function_start:function_end]
    script = "\n".join(
        [
            "const validationBatchErrorTitle = '批次操作失败。';",
            "const selectedTaskId = 'parent-1';",
            "const selectedTask = { id: 'parent-1', active_job_kind: 'validation_batch' };",
            "const renderSignatures = { actionStatus: '' };",
            "const updates = [];",
            "function signatureFromParts(parts) { return JSON.stringify(parts); }",
            "function taskActionStatusSnapshot() {",
            "  return { message: '批次模型验证进行中。', kind: 'busy', detail: '顺序处理中。' };",
            "}",
            "function setActionStatus(message, kind, detail) { updates.push({ message, kind, detail }); }",
            function_body,
            "const errorMessage = 'temporary detail failure';",
            "renderSignatures.actionStatus = signatureFromParts([",
            "  selectedTaskId, validationBatchErrorTitle, 'error', errorMessage,",
            "]);",
            "assert.equal(restoreValidationBatchActionStatusAfterRecovery({",
            "  parentTaskId: 'parent-1', message: errorMessage,",
            "}), true);",
            "assert.deepEqual(updates, [{",
            "  message: '批次模型验证进行中。', kind: 'busy', detail: '顺序处理中。',",
            "}]);",
            "renderSignatures.actionStatus = 'another-real-status';",
            "assert.equal(restoreValidationBatchActionStatusAfterRecovery({",
            "  parentTaskId: 'parent-1', message: errorMessage,",
            "}), false);",
            "assert.equal(updates.length, 1);",
        ]
    )
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", 'import assert from "node:assert/strict";\n' + script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_validation_batch_uses_only_its_dedicated_progress_poll():
    app = _read_static("app.js")
    function_start = app.index("function ensureActiveTaskProgressPolling")
    function_end = app.index("\nfunction runModeLabel", function_start)
    function_body = app[function_start:function_end]
    script = "\n".join(
        [
            "let selectedTaskId = 'parent-1';",
            "let selectedTask = null;",
            "const progressPolls = new Map();",
            "const terminalTaskStatuses = new Set();",
            "let genericPolls = 0;",
            "function taskServerBusyAction(task) { return task.active_job_kind || null; }",
            "function isValidationBatchTask(task) { return task.task_type === 'validation_batch'; }",
            "function usesAgentValidationWorkbench(task) {",
            "  return task?.task_type === 'validation_batch' && task?.run_mode === 'agent';",
            "}",
            "let projectedValidationChildTaskId = null;",
            "let projectedValidationChildTask = null;",
            "async function pollValidationProgress() { genericPolls += 1; }",
            function_body,
            "const batchTask = {",
            "  id: 'parent-1', task_type: 'validation_batch', active_job_kind: 'validation_batch',",
            "};",
            "ensureActiveTaskProgressPolling(batchTask);",
            "assert.equal(genericPolls, 0);",
            "const reportTask = { id: 'single-1', task_type: 'validation', active_job_kind: 'report' };",
            "ensureActiveTaskProgressPolling(reportTask);",
            "assert.equal(genericPolls, 1);",
        ]
    )
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", 'import assert from "node:assert/strict";\n' + script],
        check=True,
        capture_output=True,
        text=True,
    )


def test_validation_batch_server_busy_state_has_independent_copy_and_blocks_delete():
    app = _read_static("app.js")
    busy_start = app.index("function taskBusyAction")
    busy_end = app.index("function selectedTaskIsBusy", busy_start)
    delete_start = app.index("async function deleteTask")
    delete_end = app.index("async function runAction", delete_start)
    delete_body = app[delete_start:delete_end]
    script = "\n".join(
        [
            "const selectedTaskId = 'parent-1';",
            "const selectedTask = { id: 'parent-1', status: 'running', active_job_kind: 'validation_batch' };",
            "const globalBusyAction = null;",
            f"import {{ createTaskActivityOwner }} from {json.dumps((STATIC_DIR / 'js/task-activity.js').as_uri())};",
            "const taskActivities = createTaskActivityOwner();",
            "function taskStopped() { return false; }",
            "function taskPlanWorkflowStatusSnapshot() { return null; }",
            "function usesPmmlScoringWorkflow() { return false; }",
            "function taskFailureStage() { return ''; }",
            app[busy_start:busy_end],
            slice_function(app, "function taskActionStatusSnapshot("),
            "assert.equal(taskServerBusyAction(selectedTask), 'validation_batch');",
            "assert.equal(taskBusyAction('parent-1'), 'validation_batch');",
            # Batch 4 (2026-10-14 redesign): kind info so the pill resolves the batch's
            # worst-first status like the sidebar (review B1); the copy is unchanged.
            "assert.deepEqual(taskActionStatusSnapshot(selectedTask), {",
            "  message: '批次模型验证进行中。',",
            "  kind: 'info',",
            "  detail: '平台正在按顺序处理批次中的模型。',",
            "});",
        ]
    )
    script = session_projection_fixture(script)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", 'import assert from "node:assert/strict";\n' + script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    assert delete_body.index("taskBusyAction(targetTask.id)") < delete_body.index(
        "await showPlatformConfirm({"
    )
    assert delete_body.index("taskServerBusyAction(targetTask)") < delete_body.index(
        "await showPlatformConfirm({"
    )




def test_validation_batch_delete_preview_surfaces_full_cascade_and_fails_closed():
    app = _read_static("app.js")
    summary_start = app.index("const PURGE_SUMMARY_FIELDS")
    summary_end = app.index("async function reconcileTaskBeforeDelete", summary_start)
    summary_body = app[summary_start:summary_end]
    delete_start = app.index("async function deleteTask")
    delete_end = app.index("async function runAction", delete_start)
    delete_body = app[delete_start:delete_end]
    script = "\n".join(
        [
            "let fail = false;",
            "let response = {",
            "  task_count: 3, child_task_count: 2, active_job_count: 0,",
            "  owned_batch_material_tree_count: 1,",
            "  purge_summary: { datasets: 4, child_tasks: 2, owned_batch_material_trees: 1 },",
            "};",
            "function isValidationBatchTask(task) { return task.task_type === 'validation_batch'; }",
            "async function api() {",
            "  if (fail) throw new Error('preview unavailable');",
            "  return response;",
            "}",
            summary_body,
            "const task = { id: 'parent-1', task_type: 'validation_batch' };",
            "const items = await loadTaskPurgeSummary(task);",
            "assert.deepEqual(items, [",
            "  { key: 'datasets', label: '数据集', count: 4 },",
            "  { key: 'child_tasks', label: '子验证任务', count: 2 },",
            "  { key: 'owned_batch_material_trees', label: '平台持有的批次上传材料', count: 1 },",
            "]);",
            "fail = true;",
            "await assert.rejects(() => loadTaskPurgeSummary(task), /preview unavailable/);",
            "fail = false;",
            "for (const malformed of [",
            "  {},",
            "  { purge_summary: {} },",
            "  { task_count: 3, child_task_count: 2, active_job_count: 0,",
            "    owned_batch_material_tree_count: 1,",
            "    purge_summary: { child_tasks: -1, owned_batch_material_trees: 1 } },",
            "  { task_count: 3, child_task_count: 1, active_job_count: 0,",
            "    owned_batch_material_tree_count: 1,",
            "    purge_summary: { child_tasks: 2, owned_batch_material_trees: 1 } },",
            "  { task_count: 1, child_task_count: 0, active_job_count: 0,",
            "    owned_batch_material_tree_count: 0,",
            "    purge_summary: { child_tasks: 0, owned_batch_material_trees: 0 } },",
            "  { task_count: 12, child_task_count: 11, active_job_count: 0,",
            "    owned_batch_material_tree_count: 0,",
            "    purge_summary: { child_tasks: 11, owned_batch_material_trees: 0 } },",
            "  { task_count: 3, child_task_count: 2, active_job_count: 0,",
            "    owned_batch_material_tree_count: 1,",
            "    purge_summary: { purge_summary: { child_tasks: 2, owned_batch_material_trees: 1 } } },",
            "]) {",
            "  response = malformed;",
            "  await assert.rejects(",
            "    () => loadTaskPurgeSummary(task),",
            "    /invalid validation batch purge preview/,",
            "  );",
            "}",
        ]
    )
    script = session_projection_fixture(script)
    subprocess.run(
        ["node", "--input-type=module", "-e", 'import assert from "node:assert/strict";\n' + script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "无法读取批次删除范围，已停止删除" in delete_body
    assert "将级联删除" in delete_body
    assert "子验证任务" in delete_body
    assert "平台持有的批次上传材料" in delete_body
    assert "外部手动材料路径不会删除" in delete_body


def test_batch_controller_blocks_start_until_server_reports_every_contract_ready():
    _run_batch_module(
        """
const classes = new Set(["hidden"]);
const panel = {
  innerHTML: "",
  attrs: {},
  classList: {
    toggle(name, force) { force ? classes.add(name) : classes.delete(name); },
  },
  setAttribute(name, value) { this.attrs[name] = value; },
  addEventListener() {},
  querySelectorAll() { return []; },
};
const task = { id: "parent-1", task_type: "validation_batch" };
const calls = [];
let batchStatus = "awaiting_confirmation";
let contractStatus = "pending_confirmation";
let confirms = 0;
let confirmedContext = null;
const controller = batch.createValidationBatchPanelController({
  api: async (url, options = {}) => {
    calls.push([url, options.method || "GET"]);
    if (options.method === "POST") {
      batchStatus = "running";
      return { status: "accepted" };
    }
    return {
      batch: { parent_task_id: "parent-1", status: batchStatus, item_count: 1 },
      items: [{
        child_task_id: "child-1",
        status: "awaiting_confirmation",
        input_contract_status: contractStatus,
      }],
    };
  },
  getElementById: () => panel,
  getSelectedTask: () => task,
  confirmStart: async (context) => {
    confirms += 1;
    confirmedContext = context;
    return true;
  },
  refreshParentTask: async () => {},
  schedulePoll: () => 1,
  cancelPoll: () => {},
});

await controller.selectTask(task);
assert.equal(controller.remainingContractConfirmations(), 1);
assert.match(panel.innerHTML, /data-batch-start="true"[^>]*disabled/);
assert.equal(await controller.startOrContinue(), false);
assert.equal(confirms, 0);
assert.equal(calls.filter(([, method]) => method === "POST").length, 0);

assert.equal(controller.markContractConfirmed("child-1"), 0);
assert.doesNotMatch(panel.innerHTML, /data-batch-start="true"[^>]*disabled/);

await controller.selectTask(task, { force: true });
assert.equal(controller.remainingContractConfirmations(), 1);
assert.match(panel.innerHTML, /data-batch-start="true"[^>]*disabled/);
assert.equal(await controller.startOrContinue(), false);
assert.equal(confirms, 0);
assert.equal(calls.filter(([, method]) => method === "POST").length, 0);

contractStatus = "ready";
await controller.selectTask(task, { force: true });
assert.equal(controller.remainingContractConfirmations(), 0);
assert.doesNotMatch(panel.innerHTML, /data-batch-start="true"[^>]*disabled/);
assert.match(panel.innerHTML, /合同已确认/);
assert.equal(await controller.startOrContinue(), true);
assert.equal(confirms, 1);
assert.equal(confirmedContext.awaitingConfirmationCount, 0);
assert.equal(calls.filter(([, method]) => method === "POST").length, 1);
"""
    )


def test_batch_contract_button_opens_child_contract_inside_parent_workspace():
    _run_batch_module(
        """
let clickHandler = null;
const panel = {
  innerHTML: "",
  classList: { toggle() {} },
  setAttribute() {},
  addEventListener(name, handler) { if (name === "click") clickHandler = handler; },
  querySelectorAll() { return []; },
};
const parent = { id: "parent-1", task_type: "validation_batch" };
const opened = [];
const controller = batch.createValidationBatchPanelController({
  api: async () => ({
    batch: { parent_task_id: "parent-1", status: "awaiting_confirmation", item_count: 1 },
    items: [{
      id: "item-1",
      child_task_id: "child-1",
      model_name: "模型 A",
      model_version: "v1",
      status: "awaiting_confirmation",
      input_contract_url: "/api/tasks/child-1/validation-input-contract",
    }],
  }),
  getElementById: () => panel,
  getSelectedTask: () => parent,
  openContract: (context) => { opened.push(context); },
});
await controller.selectTask(parent);
assert.ok(clickHandler);
clickHandler({
  target: {
    closest(selector) {
      if (selector === "[data-batch-contract-task-id]") {
        return {
          dataset: {
            batchContractTaskId: "child-1",
            batchContractItemId: "item-1",
          },
        };
      }
      return null;
    },
  },
});
assert.deepEqual(opened, [{
  parentTaskId: "parent-1",
  childTaskId: "child-1",
  itemId: "item-1",
  modelName: "模型 A",
  modelVersion: "v1",
}]);
assert.equal(parent.id, "parent-1");
assert.equal(controller.markContractConfirmed("child-1"), 0);
assert.match(panel.innerHTML, /合同已确认/);
assert.doesNotMatch(panel.innerHTML, /data-batch-contract-task-id="child-1"/);
clickHandler({
  target: {
    closest(selector) {
      return selector === "[data-batch-contract-task-id]"
        ? { dataset: { batchContractTaskId: "forged-child", batchContractItemId: "item-1" } }
        : null;
    },
  },
});
assert.equal(opened.length, 1);
"""
    )


def test_batch_contract_submission_is_explicitly_bound_to_child_and_refreshes_parent():
    index_html = _read_static('index.html')
    assert 'id="batchValidationContractPanel"' in index_html


def test_batch_frontend_is_wired_without_reusing_single_task_creation_dialog():
    index_html = _read_static('index.html')
    assert 'id="batchOverviewPanel"' in index_html
    assert 'static/css/validation-batch.css' in index_html


def test_agent_batch_uses_compact_switcher_instead_of_old_overview():
    batch_js = _read_static("js/validation-batch.js")
    app_js = _read_static("app.js")

    assert "export function usesAgentValidationWorkbench" in batch_js
    assert "export function renderValidationBatchSwitcher" in batch_js
    assert "export function validationBatchConfirmAllAction" in batch_js
    assert "data-batch-confirm-all" in batch_js
    assert "全部确认" in batch_js
    assert "export function formatValidationBatchTaskName" in batch_js
    assert "validation-batch-switcher" in batch_js
    assert "data-batch-switch-child" in batch_js
    assert "data-tone" in batch_js
    assert "itemProgressLabel" in batch_js
    assert "onProjectedChildChange" in batch_js
    assert "function selectChild" in batch_js
    assert "onLayoutChange" in batch_js
    assert "validationBatchSwitcher" in batch_js
    assert "usesAgentValidationWorkbench(task)" in batch_js
    assert "renderValidationBatchOverview" in batch_js
    assert 'id="validationBatchSwitcher"' in _read_static("index.html")
    assert "stampValidationBatchItemCount" in app_js
    stamp_fn = app_js.split("function stampValidationBatchItemCount", 1)[1].split(
        "function reportTitleForTask",
        1,
    )[0]
    assert "const nextName = formatValidationBatchTaskName" in stamp_fn
    assert "currentName === nextName" in stamp_fn
    assert "model_name: nextName" in stamp_fn
    row_html = app_js.split("function taskRowInnerHtml", 1)[1].split(
        "function createTaskRowShell",
        1,
    )[0]
    assert "const displayName = taskDisplayName(task)" in row_html
    assert "${escapeHtml(displayName)}" in row_html
    assert "escapeHtml(task.model_name)" not in row_html
    row_signature = app_js.split("function taskRowContentSignature", 1)[1].split(
        "function taskRowInnerHtml",
        1,
    )[0]
    assert "taskDisplayName(task)" in row_signature
    list_signature = app_js.split("function taskListSignature", 1)[1].split(
        "function metricPreviewSignature",
        1,
    )[0]
    assert "task.item_count || \"\"" in list_signature
    assert "task.model_name || \"\"" in list_signature
    assert "lastNotifiedItemCount === itemCount" in batch_js
    assert "formatValidationBatchTaskName(new Date(), rows.length)" in _read_static(
        "js/create-task-dialog.js"
    )
    batch_css = _read_static("css/validation-batch.css")
    assert "overflow-x: auto" in batch_css
    # Batch 4 (2026-10-14 redesign): chip tones are the shared set ok|fail|run|review|neutral.
    assert 'data-tone="ok"' in batch_css or ".validation-batch-switcher-item[data-tone=\"ok\"]" in batch_css
    switcher_item_css = batch_css.split(".validation-batch-switcher-item {", 1)[1].split(
        ".validation-batch-switcher-actions {",
        1,
    )[0]
    assert "background: var(--surface)" in switcher_item_css
    assert "border-radius: var(--radius-control)" in switcher_item_css
    # Batch 4 (2026-10-14 redesign): no coloured stripe/outline or brand tint on
    # chips (D3); the selected chip is raised like a pressed segment.
    assert "--switcher-tone" not in switcher_item_css
    assert "var(--button-primary-border)" not in switcher_item_css
    assert "var(--brand-primary)" not in switcher_item_css
    assert ".validation-batch-switcher-item.is-selected," in switcher_item_css
    assert "#3b6dff" not in switcher_item_css
    assert "background: var(--warning-soft)" not in switcher_item_css
    assert "background: var(--success-soft)" not in switcher_item_css
    assert "background: var(--accent-soft)" not in switcher_item_css
    assert "background: var(--danger-soft)" not in switcher_item_css
    # Batch 4 (2026-10-14 redesign): success -> ok in the shared tone set.
    assert ".validation-batch-switcher-item[data-tone=\"ok\"] .validation-batch-switcher-progress" in switcher_item_css

    assert "function workbenchTask(" in app_js
    assert "function workbenchTaskId(" in app_js
    assert "taskSession.modelId" in app_js
    assert "onProjectedChildChange:" in app_js
    apply_start = app_js.index("async function applyProjectedValidationChild")
    apply_end = app_js.index("function selectedTaskIsRiskAnalysisAgent", apply_start)
    apply_body = app_js[apply_start:apply_end]
    # Messages are cleared only when model identity changes; race tests assert
    # that a late child response cannot restore the previous model.
    assert "const childChanged = taskSession.modelId !== normalizedChildId;" in apply_body
    assert "resetAgentTypingState();" in apply_body
    assert "beginTaskContentLoad(normalizedChildId);" in apply_body
    assert "finishTaskContentLoad(normalizedChildId);" in apply_body
    assert "beginProjectedChildContentLoad" not in apply_body
    assert "is-projected-child-loading" not in apply_body
    assert "await nextAnimationFrame();" in apply_body
    assert "isCurrentProjectedChildLoad(loadVersion, childChanged)" in apply_body
    assert "scrollContent.scrollTop = 0;" in apply_body
    assert "renderAll();" in apply_body
    assert "function beginProjectedChildContentLoad" not in app_js
    assert 'classList.add("is-projected-child-loading")' not in app_js
    batch_css = _read_static("css/app-shell.css")
    assert ".validation-workspace.is-projected-child-loading" not in batch_css
    assert ".validation-workspace.is-task-content-loading:has(#validationBatchSwitcher:not([hidden])) :is(.workspace-head)" in batch_css
    assert "top: var(--workspace-head-space);" in batch_css.split(
        ".validation-workspace.is-task-content-loading:has(#validationBatchSwitcher:not([hidden])) .result-workspace::after",
        1,
    )[1].split("}", 1)[0]
    assert "beginTaskContentLoad(task.id);" in app_js

    notebook_vis = app_js.split("function renderReproducibilitySectionVisibility", 1)[1].split(
        "function metricOverviewComplete",
        1,
    )[0]
    metric_vis = app_js.split("function renderMetricSectionVisibility", 1)[1].split(
        "function workflowIndex",
        1,
    )[0]
    scan_vis = app_js.split("function updateAgentScanSectionVisibility", 1)[1].split(
        "function updateAgentReportSectionVisibility",
        1,
    )[0]
    assert "selectedTaskIsValidationBatch" not in notebook_vis
    assert "selectedTaskIsValidationBatch" not in metric_vis
    assert "selectedTaskIsValidationBatch" not in scan_vis

    _run_batch_module(
        r"""
assert.equal(
  batch.usesAgentValidationWorkbench({ task_type: "validation_batch", run_mode: "agent" }),
  true,
);
assert.equal(
  batch.usesAgentValidationWorkbench({ task_type: "validation_batch", run_mode: "manual" }),
  false,
);
assert.equal(
  batch.usesAgentValidationWorkbench({ task_type: "validation", run_mode: "agent" }),
  false,
);
const html = batch.renderValidationBatchSwitcher(
  {
    batch: { parent_task_id: "parent-1", status: "running", item_count: 2 },
    items: [
      {
        child_task_id: "child-a",
        model_name: "模型A",
        model_version: "v1",
        ordinal: 1,
        status: "succeeded",
        stage: "completed",
      },
      {
        child_task_id: "child-b",
        model_name: "模型B",
        model_version: "v1",
        ordinal: 2,
        status: "running",
        stage: "pmml_scoring",
      },
    ],
  },
  { selectedChildTaskId: "child-a" },
);
assert.match(html, /class="validation-batch-switcher"/);
assert.match(html, /data-batch-switch-child="child-a"/);
assert.match(html, /data-batch-switch-child="child-b"/);
// Batch 4 (2026-10-14 redesign): chip tones are the shared set (success -> ok, running -> run).
assert.match(html, /data-tone="ok"/);
assert.match(html, /data-tone="run"/);
assert.match(html, /is-selected/);
assert.match(html, /模型A/);
assert.match(html, /模型B/);
// Batch 4 (2026-10-14 redesign): an item that finished reads 已完成; the 通过 verdict lives in result cards.
assert.match(html, /已完成/);
assert.doesNotMatch(html, /已通过/);
assert.match(html, /PMML打分/);
assert.match(html, /全部确认/);
assert.match(html, /data-batch-confirm-all="true"/);
assert.match(html, /disabled/);
assert.doesNotMatch(html, /正在验证/);
assert.doesNotMatch(html, /批次总览/);
assert.doesNotMatch(html, /MODEL VALIDATION BATCH/);
assert.doesNotMatch(html, /启动批次/);
assert.equal(
  batch.formatValidationBatchTaskName(new Date(2026, 7, 20), 2),
  "2026-08-20 模型验证批次 (2个模型)",
);
assert.equal(
  batch.formatValidationBatchTaskName(new Date(2026, 7, 20), 0),
  "2026-08-20 模型验证批次",
);
const readyHtml = batch.renderValidationBatchSwitcher(
  {
    batch: { parent_task_id: "parent-1", status: "running", item_count: 2 },
    items: [
      {
        child_task_id: "child-a",
        model_name: "模型A",
        ordinal: 1,
        status: "writing_artifacts",
        pending_report_draft: true,
      },
      {
        child_task_id: "child-b",
        model_name: "模型B",
        ordinal: 2,
        status: "writing_artifacts",
        pending_report_draft: true,
      },
    ],
  },
  { selectedChildTaskId: "child-a" },
);
assert.match(readyHtml, /全部确认/);
assert.doesNotMatch(readyHtml, /disabled/);
const action = batch.validationBatchConfirmAllAction(
  batch.normalizeValidationBatchPayload({
    batch: { parent_task_id: "parent-1", status: "running", item_count: 2 },
    items: [
      { child_task_id: "child-a", pending_report_draft: true, status: "writing_artifacts" },
      { child_task_id: "child-b", pending_report_draft: true, status: "writing_artifacts" },
    ],
  }),
);
assert.equal(action.visible, true);
assert.equal(action.disabled, false);
assert.equal(action.label, "全部确认");
"""
    )


def test_validation_batch_sidebar_icon_matches_model_validation():
    assert _read_static('css/validation-batch.css').count('.task-kind-icon[data-kind="validation_batch"]') == 0
