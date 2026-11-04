"""One status vocabulary for every surface (2026-10-14 redesign, D16 / batch 4).

``js/task-status.js`` is the single source: the sidebar row, the title pill,
the multi-model chip and the right-rail capsule all resolve a record there.
The first tests pin the table and the resolvers; the acceptance test renders
the same state through the four real surface functions and requires one word
and one tone.
"""

import json
import subprocess
from pathlib import Path

from tests.javascript_source import session_projection_fixture
from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"

CANONICAL = {
    "not_started": ("待开始", "neutral", "idle", "none"),
    "queued": ("排队中", "neutral", "idle", "none"),
    "planning": ("规划中", "run", "busy", "run"),
    "running": ("执行中", "run", "busy", "run"),
    "awaiting_confirm": ("待确认", "review", "wait", "wait"),
    "awaiting_next": ("待继续", "review", "wait", "wait"),
    "needs_review": ("需复核", "review", "wait", "wait"),
    "done": ("已完成", "ok", "done", "none"),
    "partial_failure": ("部分失败", "fail", "fail", "bad"),
    "failed": ("失败", "fail", "fail", "bad"),
    "needs_reconciliation": ("待核对", "fail", "fail", "bad"),
    "paused": ("已暂停", "neutral", "idle", "none"),
    "stopped": ("已停止", "neutral", "idle", "none"),
    "cancelled": ("已取消", "neutral", "idle", "none"),
    "skipped": ("已跳过", "neutral", "done", "none"),
    "unknown": ("状态未知", "neutral", "idle", "none"),
}

CHECKERS = {
    "not_started": "pending",
    "queued": "pending",
    "planning": "running",
    "running": "running",
    "awaiting_confirm": "review",
    "awaiting_next": "review",
    "needs_review": "review",
    "done": "succeeded",
    "partial_failure": "failed",
    "failed": "failed",
    "needs_reconciliation": "failed",
    "paused": "stopped",
    "stopped": "stopped",
    "cancelled": "stopped",
    "skipped": "skipped",
    "unknown": "pending",
}


def _run_module(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def test_status_table_matches_the_canonical_vocabulary():
    data = _run_module(
        """
import { STATUS, statusPresentation } from "./marvis/static/js/task-status.js";
process.stdout.write(JSON.stringify({
  table: STATUS,
  missing: statusPresentation("no_such_code"),
  empty: statusPresentation(undefined),
}));
"""
    )
    assert set(data["table"]) == set(CANONICAL)
    for key, (label, tone, face, dot) in CANONICAL.items():
        assert data["table"][key] == {
            "label": label,
            "tone": tone,
            "face": face,
            "dot": dot,
            "checker": CHECKERS[key],
        }, key
    # The raw code never reaches the UI.
    assert data["missing"]["key"] == "unknown"
    assert data["missing"]["label"] == "状态未知"
    assert data["empty"]["key"] == "unknown"


def test_resolvers_map_every_raw_value_to_one_key():
    data = _run_module(
        r"""
import {
  resolveBatchItemStatus,
  resolveBatchStatus,
  resolvePlanStatus,
  resolvePlanStepStatus,
  resolveTaskStatus,
  resolveWorkflowStatus,
} from "./marvis/static/js/task-status.js";

const v1 = {};
for (const status of ["created", "scanned", "configured", "running", "executed", "computing_metrics",
  "writing_artifacts", "succeeded", "failed", "review_required", "mystery"]) {
  v1[status] = resolveTaskStatus({ status, task_type: "validation" }).key;
}
const workflow = {};
for (const status of ["draft", "validated", "confirmed", "running", "awaiting_confirm", "review", "done",
  "failed", "cancelled", "mystery"]) {
  workflow[status] = resolveWorkflowStatus(status);
}
const step = {};
for (const status of ["pending", "blocked", "awaiting_confirm", "running", "checking", "done", "skipped",
  "failed", "mystery", ""]) {
  step[status || "(empty)"] = resolvePlanStepStatus({ status });
}
const item = {};
for (const status of ["created", "queued", "running", "awaiting_confirmation", "pending_confirmation",
  "review_required", "succeeded", "completed", "failed", "cancelled", "mystery"]) {
  item[status] = resolveBatchItemStatus({ status });
}
const batch = {};
for (const status of ["created", "queued", "running", "awaiting_confirmation", "partial_failure",
  "completed", "failed", "cancelled", "mystery"]) {
  batch[status] = resolveBatchStatus({ status, items: [] });
}
process.stdout.write(JSON.stringify({
  v1,
  workflow,
  step,
  item,
  batch,
  writingReport: resolveTaskStatus({ status: "writing_artifacts", active_job_kind: "report" }).key,
  anyActiveJob: resolveTaskStatus({ status: "scanned", active_job_kind: "agent" }).key,
  localBusy: resolveTaskStatus({ status: "succeeded" }, { busy: true }).key,
  createdInRunningBatch: resolveBatchItemStatus({ status: "created" }, { batchStatus: "running" }),
  verdictOnly: ["pass", "manual_review", "fail"].map((outcome) => resolveBatchItemStatus({ outcome })),
  halts: [
    resolvePlanStepStatus({ status: "failed", execution_halt: "paused", failure_envelope: { error_kind: "search_paused", retryable: false } }),
    resolvePlanStepStatus({ status: "failed", execution_halt: "cancelled", failure_envelope: { error_kind: "search_cancelled", retryable: false } }),
    resolvePlanStepStatus({ status: "failed", failure_envelope: { retryable: false } }),
  ],
  noPlan: resolvePlanStatus(null),
}));
"""
    )
    assert data["v1"] == {
        "created": "not_started",
        "scanned": "awaiting_next",
        "configured": "awaiting_next",
        "running": "running",
        "executed": "awaiting_next",
        "computing_metrics": "running",
        "writing_artifacts": "awaiting_next",
        "succeeded": "done",
        "failed": "failed",
        "review_required": "needs_review",
        "mystery": "unknown",
    }
    assert data["writingReport"] == "running"
    assert data["anyActiveJob"] == "running"
    assert data["localBusy"] == "running"
    assert data["workflow"] == {
        "draft": "planning",
        "validated": "not_started",
        "confirmed": "running",
        "running": "running",
        "awaiting_confirm": "awaiting_confirm",
        "review": "needs_review",
        "done": "done",
        "failed": "failed",
        "cancelled": "stopped",
        "mystery": None,
    }
    assert data["step"] == {
        "pending": "not_started",
        "blocked": "stopped",
        "awaiting_confirm": "awaiting_confirm",
        "running": "running",
        "checking": "running",
        "done": "done",
        "skipped": "skipped",
        "failed": "failed",
        "mystery": "unknown",
        "(empty)": "not_started",
    }
    assert data["halts"] == ["paused", "cancelled", "needs_reconciliation"]
    assert data["item"] == {
        "created": "not_started",
        "queued": "queued",
        "running": "running",
        "awaiting_confirmation": "awaiting_confirm",
        "pending_confirmation": "awaiting_confirm",
        "review_required": "needs_review",
        "succeeded": "done",
        "completed": "done",
        "failed": "failed",
        "cancelled": "cancelled",
        "mystery": "unknown",
    }
    assert data["createdInRunningBatch"] == "queued"
    assert data["verdictOnly"] == ["done", "needs_review", "failed"]
    assert data["batch"] == {
        "created": "not_started",
        "queued": "queued",
        "running": "running",
        "awaiting_confirmation": "awaiting_confirm",
        "partial_failure": "partial_failure",
        "completed": "done",
        "failed": "failed",
        "cancelled": "cancelled",
        "mystery": "unknown",
    }
    assert data["noPlan"] is None


def test_resolvers_share_one_precedence():
    data = _run_module(
        r"""
import { resolvePlanStatus, resolveTaskStatus, rollupStatus } from "./marvis/static/js/task-status.js";

const failedStep = { title: "调参", status: "failed" };
process.stdout.write(JSON.stringify({
  // Task level: stopped beats every structured state.
  stoppedOverPlan: resolveTaskStatus({ stopped: true, status: "failed" }, { planStatus: "failed" }).key,
  stoppedOverBatch: resolveTaskStatus({ stopped: true }, { batch: { status: "failed" } }).key,
  // Plan / workflow state beats the shared task record and an active job.
  planOverRecord: resolveTaskStatus({ status: "succeeded", active_job_kind: "driver" }, { usesPlanWorkflow: true, planStatus: { key: "awaiting_confirm" } }).key,
  // Plan-rail task without a plan: job > gate > unproven completion > not started.
  planningWithJob: resolveTaskStatus({ status: "created", active_job_kind: "driver" }, { usesPlanWorkflow: true, openGate: true }).key,
  gateBeforePlan: resolveTaskStatus({ status: "created" }, { usesPlanWorkflow: true, openGate: true }).key,
  unprovenCompletion: resolveTaskStatus({ status: "succeeded" }, { usesPlanWorkflow: true }).key,
  planNotStarted: resolveTaskStatus({ status: "created" }, { usesPlanWorkflow: true }).key,
  // Plan level: reconciliation > stopped > halt > failed > awaiting_confirm > continuation > running.
  reconciliation: resolvePlanStatus({ status: "cancelled", steps: [{ status: "failed", failure_envelope: { retryable: false } }] }),
  planCancelled: resolvePlanStatus({ status: "cancelled", steps: [failedStep] }),
  halt: resolvePlanStatus({ status: "failed", steps: [{ status: "failed", execution_halt: "paused" }, { status: "awaiting_confirm" }] }),
  failedUnderRetryQuestion: resolvePlanStatus({ status: "awaiting_confirm", steps: [failedStep, { status: "pending" }] }),
  stepAwaitingUnderRunningPlan: resolvePlanStatus({ status: "running", steps: [{ status: "done" }, { status: "awaiting_confirm" }] }),
  continuation: resolvePlanStatus({ status: "running", steps: [], reconciliation: { continuation: { remaining_step_ids: ["x"] } } }),
  running: resolvePlanStatus({ status: "running", steps: [{ status: "running" }] }),
  // Phase roll-up uses the same order.
  rollups: [
    rollupStatus(["done", "failed", "awaiting_confirm"]),
    rollupStatus(["done", "paused", "pending"]),
    rollupStatus(["done", "skipped"]),
    rollupStatus(["done", "stopped"]),
    rollupStatus(["done", "not_started"]),
    rollupStatus([]),
  ],
}));
"""
    )
    assert data == {
        "stoppedOverPlan": "stopped",
        "stoppedOverBatch": "stopped",
        "planOverRecord": "awaiting_confirm",
        "planningWithJob": "planning",
        "gateBeforePlan": "awaiting_confirm",
        "unprovenCompletion": "awaiting_confirm",
        "planNotStarted": "not_started",
        "reconciliation": "needs_reconciliation",
        "planCancelled": "stopped",
        "halt": "paused",
        "failedUnderRetryQuestion": "failed",
        "stepAwaitingUnderRunningPlan": "awaiting_confirm",
        "continuation": "awaiting_next",
        "running": "running",
        "rollups": ["failed", "paused", "done", "stopped", "not_started", "not_started"],
    }


def test_batch_parent_aggregates_items_worst_first():
    data = _run_module(
        r"""
import { resolveBatchStatus } from "./marvis/static/js/task-status.js";
const batch = (status, ...items) => ({ status, items: items.map((status) => ({ status })) });
process.stdout.write(JSON.stringify({
  partial: resolveBatchStatus(batch("running", "failed", "running")),
  allFailed: resolveBatchStatus(batch("running", "failed", "failed")),
  declaredTotal: resolveBatchStatus({ status: "running", counts: { total: 3 }, items: [{ status: "failed" }] }),
  confirmOverReview: resolveBatchStatus(batch("running", "review_required", "awaiting_confirmation", "running")),
  reviewOverRunning: resolveBatchStatus(batch("running", "review_required", "running")),
  runningOverDone: resolveBatchStatus(batch("running", "succeeded", "queued")),
  completedWithReview: resolveBatchStatus(batch("completed", "succeeded", "review_required")),
  completedClean: resolveBatchStatus(batch("completed", "succeeded", "succeeded")),
  backendPartial: resolveBatchStatus(batch("partial_failure", "succeeded", "failed")),
  cancelledNeutral: resolveBatchStatus(batch("cancelled", "cancelled")),
  created: resolveBatchStatus(batch("created", "created", "created")),
}));
"""
    )
    assert data == {
        "partial": "partial_failure",
        "allFailed": "failed",
        "declaredTotal": "partial_failure",
        "confirmOverReview": "awaiting_confirm",
        "reviewOverRunning": "needs_review",
        "runningOverDone": "running",
        "completedWithReview": "needs_review",
        "completedClean": "done",
        "backendPartial": "partial_failure",
        "cancelledNeutral": "cancelled",
        "created": "not_started",
    }


def test_title_pill_overlay_never_parses_message_text():
    data = _run_module(
        r"""
import { actionStatusPresentation, planWorkflowStatus } from "./marvis/static/js/task-status.js";
const word = (value) => value && [value.label, value.tone];
const v1 = (status) => ({ status, task_type: "validation" });
process.stdout.write(JSON.stringify({
  // Free text mentioning confirmation no longer picks the word.
  scannedConfirmCopy: word(actionStatusPresentation("等待你的确认。", "info", { task: v1("scanned") })),
  createdInfo: word(actionStatusPresentation("任务已创建。", "info", { task: v1("created") })),
  reviewKind: word(actionStatusPresentation("验证已完成，需复核报告。", "review", { task: v1("review_required") })),
  reviewError: word(actionStatusPresentation("报告需要复核。", "error", { task: v1("review_required") })),
  stageFailure: word(actionStatusPresentation("PMML打分测试失败。", "error", { task: v1("failed") })),
  busy: word(actionStatusPresentation("正在扫描材料…", "busy", { task: v1("created") })),
  cancelled: word(actionStatusPresentation("Notebook 已停止，可重新运行。", "stopped", { task: v1("scanned") })),
  // A resolved record status supplied by the app wins for non-overlay kinds.
  supplied: word(actionStatusPresentation("自动审查已完成全部模型。", "success", {
    task: v1("review_required"), status: { label: "部分失败", tone: "fail", key: "partial_failure" },
  })),
  planWins: word(actionStatusPresentation("操作失败", "error", {
    task: { status: "created", task_type: "data_join" }, usesPlanWorkflow: true,
    workflowSnapshot: planWorkflowStatus({ status: "awaiting_confirm", steps: [] }),
  })),
  empty: actionStatusPresentation("", "success", { task: v1("succeeded") }),
}));
"""
    )
    assert data == {
        "scannedConfirmCopy": ["待继续", "review"],
        "createdInfo": ["待开始", "neutral"],
        "reviewKind": ["需复核", "review"],
        "reviewError": ["需复核", "review"],
        "stageFailure": ["失败", "fail"],
        "busy": ["执行中", "run"],
        "cancelled": ["已停止", "neutral"],
        "supplied": ["部分失败", "fail"],
        "planWins": ["待确认", "review"],
        "empty": None,
    }
    source = (STATIC / "js" / "task-status.js").read_text(encoding="utf-8")
    assert ".test(message)" not in source


# ---------- Acceptance: one state, four surfaces, one word ----------

def _surface_harness() -> str:
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    pill_start = app_js.index("function actionStatusPill")
    pill_end = app_js.index("function describeActionStatus", pill_start)
    status_start = app_js.index("function taskPlanWorkflowStatusSnapshot")
    status_end = app_js.index("function notebookReproducibilityComplete", status_start)
    failure_start = app_js.index("function taskFailureActionStatusMessage")
    failure_end = app_js.index("function actionFailureStatusTitle", failure_start)
    return "\n".join(
        [
            'import { normalizeValidationBatchPayload, renderValidationBatchSwitcher, usesAgentValidationWorkbench } from "./marvis/static/js/validation-batch.js";',
            'import { createWorkbenchChrome } from "./marvis/static/js/workbench-chrome.js";',
            'import { actionStatusPresentation, planWorkflowStatus, resolveTaskStatus, workflowStatusSnapshot } from "./marvis/static/js/task-status.js";',
            'import { taskUsesPlanRail } from "./marvis/static/js/v2/plan_rail_controller.js";',
            'import { escapeHtml } from "./marvis/static/js/ui-utils.js";',
            'import { localizeText } from "./marvis/static/js/i18n.js";',
            "let selectedTaskId = null;",
            "let selectedTask = null;",
            "let projectedValidationChildTaskId = '';",
            "let projectedValidationChildTask = null;",
            "let agentMessages = [];",
            "let plans = {};",
            "let batches = {};",
            "const planRailController = { statusSnapshot: (id) => (plans[id] ? planWorkflowStatus(plans[id]) : null) };",
            "const validationBatchPanelController = { batchPayload: (id) => batches[id] || null };",
            "const stepper = { dataset: {}, querySelectorAll: () => [] };",
            "const capsule = { dataset: {}, hidden: false, setAttribute() {} };",
            "const capsuleLabel = { textContent: '' };",
            "const elements = { workflowStepper: stepper, progressRailExpand: capsule, progressRailExpandLabel: capsuleLabel };",
            "function $(id) { return elements[id] || null; }",
            "const chrome = createWorkbenchChrome({ doc: { getElementById: (id) => elements[id] || null } });",
            "function usesPmmlScoringWorkflow() { return false; }",
            "function notebookStepsForRail() { return []; }",
            "function metricStepsForRail() { return []; }",
            "function notebookStepTone() { return 'pending'; }",
            "function taskKindIconHtml() { return ''; }",
            "function taskDisplayName(task) { return task?.name || ''; }",
            "function taskTypeDefinition() { return { validatorLabel: '负责人' }; }",
            "function formatDate() { return ''; }",
            "let captured = null;",
            "function setActionStatus(message, kind, detail) { captured = { message, kind, detail }; }",
            *(slice_function(app_js, f"function {name}(") for name in (
                "taskStopped",
                "latestOpenGateMessage",
                "workbenchTask",
                "taskActionStatusSnapshot",
                "taskRowAriaLabel",
                "taskRowInnerHtml",
                "syncRailStatus",
            )),
            app_js[pill_start:pill_end],
            app_js[status_start:status_end],
            app_js[failure_start:failure_end],
            # The title follows renderCurrentTaskWorkspace: failure path, else the snapshot.
            "function titlePill(task) {",
            "  captured = null;",
            "  if (!setTaskFailureActionStatus(task)) {",
            "    const snapshot = taskActionStatusSnapshot(task);",
            "    captured = { message: snapshot.message, kind: snapshot.kind };",
            "  }",
            "  const pill = actionStatusPill(captured.message, captured.kind, task);",
            "  return pill && { label: pill.label, tone: pill.tone };",
            "}",
            "function sidebarRow(task) {",
            "  const html = taskRowInnerHtml(task);",
            "  const dot = (html.match(/task-status-dot (\\w+)/) || [])[1] || 'none';",
            "  return { label: taskStatusLabel(task), tone: taskStatusTone(task), dot, aria: taskRowAriaLabel(task) };",
            "}",
            "function railCapsule() {",
            "  syncRailStatus();",
            "  chrome.syncRailCapsule();",
            "  return { label: capsuleLabel.textContent.split(' · ')[0], text: capsuleLabel.textContent, tone: capsule.dataset.railState };",
            "}",
            "function chip(batchPayload, childId) {",
            "  const html = renderValidationBatchSwitcher(batchPayload, { selectedChildTaskId: childId });",
            "  const button = html.split('<button').find((part) => part.includes('data-batch-switch-child=\"' + childId + '\"')) || '';",
            "  const tone = (button.match(/data-tone=\"(\\w+)\"/) || [])[1];",
            "  const progress = (button.match(/switcher-progress\">([^<]*)</) || [])[1] || '';",
            "  return { label: progress.split(' · ')[0], text: progress, tone };",
            "}",
        ]
    )


def test_one_batch_state_reads_the_same_on_all_four_surfaces():
    # A one-model agent batch: the sidebar row and the title show the parent,
    # the chip shows the item and the rail shows the projected child. Each
    # fixture describes one state family consistently across the records.
    families = {
        "not_started": {
            "parent": {"status": "created"}, "batch": "created",
            "item": {"status": "created", "stage": "batch_created"}, "child": {"status": "created"},
            "word": "待开始", "tone": "neutral", "dot": "none",
        },
        "running": {
            "parent": {"status": "created", "active_job_kind": "validation_batch"}, "batch": "running",
            "item": {"status": "running", "stage": "pmml_scoring"},
            "child": {"status": "running", "active_job_kind": "pipeline"},
            "word": "执行中", "tone": "run", "dot": "run",
        },
        "awaiting_confirm": {
            "parent": {"status": "created"}, "batch": "awaiting_confirmation",
            "item": {"status": "awaiting_confirmation", "stage": "input_confirmation"},
            "child": {"status": "scanned"},
            "word": "待确认", "tone": "review", "dot": "wait",
        },
        "needs_review": {
            "parent": {"status": "review_required"}, "batch": "completed",
            "item": {"status": "review_required", "outcome": "manual_review", "stage": "report"},
            "child": {"status": "review_required"},
            "word": "需复核", "tone": "review", "dot": "wait",
        },
        "done": {
            "parent": {"status": "succeeded"}, "batch": "completed",
            "item": {"status": "succeeded", "outcome": "pass", "stage": "completed"},
            "child": {"status": "succeeded"},
            "word": "已完成", "tone": "ok", "dot": "none",
        },
        "failed": {
            "parent": {"status": "failed"}, "batch": "failed",
            "item": {"status": "failed", "outcome": "failed", "stage": "failure"},
            "child": {"status": "failed", "failure_stage": "notebook"},
            "word": "失败", "tone": "fail", "dot": "bad",
        },
        "cancelled": {
            "parent": {"status": "created"}, "batch": "cancelled",
            "item": {"status": "cancelled", "stage": "scan"}, "child": {"status": "created"},
            "word": "已取消", "tone": "neutral", "dot": "none",
        },
    }
    script = _surface_harness() + "\n" + "\n".join(
        [
            f"const families = {json.dumps(families, ensure_ascii=False)};",
            "const out = {};",
            "for (const [name, f] of Object.entries(families)) {",
            "  const parent = { id: 'parent', task_type: 'validation_batch', run_mode: 'agent', name: '批次', ...f.parent };",
            "  const child = { id: 'child', task_type: 'validation', ...f.child };",
            "  const payload = normalizeValidationBatchPayload({",
            "    batch: { parent_task_id: 'parent', status: f.batch, item_count: 1 },",
            "    items: [{ id: 'item', child_task_id: 'child', ordinal: 1, model_name: '模型A', ...f.item }],",
            "  });",
            "  batches = { parent: payload };",
            "  selectedTaskId = 'parent'; selectedTask = parent;",
            "  projectedValidationChildTaskId = 'child'; projectedValidationChildTask = child;",
            "  out[name] = { sidebar: sidebarRow(parent), title: titlePill(parent), chip: chip(payload, 'child'), rail: railCapsule() };",
            "}",
            "process.stdout.write(JSON.stringify(out));",
        ]
    )
    data = _run_module(session_projection_fixture(script))
    for name, family in families.items():
        surfaces = data[name]
        words = {surface: value["label"] for surface, value in surfaces.items()}
        tones = {surface: value["tone"] for surface, value in surfaces.items()}
        assert set(words.values()) == {family["word"]}, (name, surfaces)
        assert set(tones.values()) == {family["tone"]}, (name, surfaces)
        assert surfaces["sidebar"]["dot"] == family["dot"], (name, surfaces)
        assert family["word"] in surfaces["sidebar"]["aria"], (name, surfaces)
    # The chip keeps its stage suffix and the capsule never says 计划.
    assert data["running"]["chip"]["text"] == "执行中 · PMML打分"
    assert data["awaiting_confirm"]["chip"]["text"] == "待确认 · 确认字段"
    assert data["done"]["rail"]["text"] == "已完成"
    assert all("计划" not in value["rail"]["text"] for value in data.values())
    # The batch-of-four-words regression: 已出报告 / 已通过 / 计划已完成 are gone.
    rendered = json.dumps(data, ensure_ascii=False)
    for stale in ("已出报告", "已通过", "计划已完成", "待复核", "运行中", "进行中"):
        assert stale not in rendered


def test_partial_failure_parent_agrees_between_sidebar_and_title():
    script = _surface_harness() + "\n" + "\n".join(
        [
            "const parent = { id: 'parent', task_type: 'validation_batch', run_mode: 'agent', status: 'review_required' };",
            "const payload = normalizeValidationBatchPayload({",
            "  batch: { parent_task_id: 'parent', status: 'partial_failure', item_count: 2 },",
            "  items: [",
            "    { id: 'a', child_task_id: 'child-a', ordinal: 1, status: 'succeeded', outcome: 'pass' },",
            "    { id: 'b', child_task_id: 'child-b', ordinal: 2, status: 'failed', outcome: 'failed' },",
            "  ],",
            "});",
            "batches = { parent: payload };",
            "selectedTaskId = 'parent'; selectedTask = parent;",
            "projectedValidationChildTaskId = 'child-b';",
            "projectedValidationChildTask = { id: 'child-b', task_type: 'validation', status: 'failed' };",
            "process.stdout.write(JSON.stringify({",
            "  sidebar: sidebarRow(parent), title: titlePill(parent),",
            "  chipA: chip(payload, 'child-a'), chipB: chip(payload, 'child-b'), rail: railCapsule(),",
            "  unloaded: (() => { batches = {}; return sidebarRow(parent); })(),",
            "  summarised: (() => { batches = {}; return sidebarRow({ ...parent, batch_summary: {",
            "    status: 'partial_failure', items: [{ status: 'succeeded', outcome: 'pass' }, { status: 'failed', outcome: 'failed' }],",
            "  } }); })(),",
            "  summarisedRunning: (() => { batches = {}; return sidebarRow({ ...parent, status: 'running', active_job_kind: 'validation_batch', batch_summary: {",
            "    status: 'running', items: [{ status: 'awaiting_confirmation', outcome: '' }, { status: 'running', outcome: '' }],",
            "  } }); })(),",
            "}));",
        ]
    )
    data = _run_module(session_projection_fixture(script))
    assert data["sidebar"]["label"] == data["title"]["label"] == "部分失败"
    assert data["sidebar"]["tone"] == data["title"]["tone"] == "fail"
    assert data["sidebar"]["dot"] == "bad"
    assert [data["chipA"]["label"], data["chipA"]["tone"]] == ["已完成", "ok"]
    # The selected (failed) model reads the same on its chip and on the rail.
    assert [data["chipB"]["label"], data["chipB"]["tone"]] == ["失败", "fail"]
    assert [data["rail"]["label"], data["rail"]["tone"]] == ["失败", "fail"]
    # Without any batch data the row falls back to the parent record.
    assert [data["unloaded"]["label"], data["unloaded"]["tone"]] == ["需复核", "review"]
    # Unselected rows read the task list's batch_summary, so the dot does not
    # change when the row is clicked (batch 4 review B2).
    assert [data["summarised"]["label"], data["summarised"]["dot"]] == ["部分失败", "bad"]
    assert [data["summarisedRunning"]["label"], data["summarisedRunning"]["tone"]] == ["待确认", "review"]


def test_single_tasks_read_the_same_on_sidebar_title_and_rail():
    # Single tasks have no model chip; the other three surfaces read the same
    # record. Plan families resolve through the cached plan.
    fixtures = {
        "v1_created": ({"task_type": "validation", "status": "created"}, None, "待开始", "neutral"),
        "v1_running": ({"task_type": "validation", "status": "running", "active_job_kind": "pipeline"}, None, "执行中", "run"),
        "v1_scanned": ({"task_type": "validation", "status": "scanned"}, None, "待继续", "review"),
        "v1_writing_idle": ({"task_type": "validation", "status": "writing_artifacts"}, None, "待继续", "review"),
        "v1_field_check": ({"task_type": "validation", "status": "scanned", "awaiting_input_confirmation": True}, None, "待确认", "review"),
        "v1_field_check_running": ({"task_type": "validation", "status": "scanned", "awaiting_input_confirmation": True, "active_job_kind": "agent"}, None, "执行中", "run"),
        "v1_writing_report": ({"task_type": "validation", "status": "writing_artifacts", "active_job_kind": "report"}, None, "执行中", "run"),
        "v1_review": ({"task_type": "validation", "status": "review_required"}, None, "需复核", "review"),
        "v1_done": ({"task_type": "validation", "status": "succeeded"}, None, "已完成", "ok"),
        "v1_failed": ({"task_type": "validation", "status": "failed", "failure_stage": "metrics"}, None, "失败", "fail"),
        "v1_stopped": ({"task_type": "validation", "status": "scanned", "stopped": True}, None, "已停止", "neutral"),
        "plan_drafting": ({"task_type": "modeling", "status": "created", "active_job_kind": "driver"}, None, "规划中", "run"),
        "plan_validated": ({"task_type": "modeling", "status": "created"}, {"status": "validated", "steps": [{"status": "pending"}]}, "待开始", "neutral"),
        "plan_running": ({"task_type": "modeling", "status": "created"}, {"status": "running", "steps": [{"status": "done"}, {"status": "running"}]}, "执行中", "run"),
        "plan_gate": ({"task_type": "modeling", "status": "created"}, {"status": "awaiting_confirm", "steps": [{"status": "awaiting_confirm"}]}, "待确认", "review"),
        "plan_continuation": ({"task_type": "modeling", "status": "created"}, {"status": "running", "steps": [{"status": "done"}, {"status": "pending"}], "reconciliation": {"continuation": {"remaining_step_ids": ["x"]}}}, "待继续", "review"),
        "plan_review": ({"task_type": "modeling", "status": "created"}, {"status": "review", "steps": [{"status": "done"}]}, "需复核", "review"),
        "plan_done": ({"task_type": "modeling", "status": "created"}, {"status": "done", "steps": [{"status": "done"}]}, "已完成", "ok"),
        "plan_failed": ({"task_type": "modeling", "status": "created"}, {"status": "failed", "steps": [{"status": "failed"}]}, "失败", "fail"),
        "plan_reconcile": ({"task_type": "modeling", "status": "created"}, {"status": "failed", "steps": [{"status": "failed", "failure_envelope": {"retryable": False}}]}, "待核对", "fail"),
        "plan_paused": ({"task_type": "strategy", "status": "created"}, {"status": "failed", "steps": [{"status": "failed", "execution_halt": "paused", "failure_envelope": {"retryable": False, "error_kind": "search_paused"}}]}, "已暂停", "neutral"),
        "plan_search_cancelled": ({"task_type": "strategy", "status": "created"}, {"status": "failed", "steps": [{"status": "failed", "execution_halt": "cancelled", "failure_envelope": {"retryable": False, "error_kind": "search_cancelled"}}]}, "已取消", "neutral"),
        "plan_stopped": ({"task_type": "modeling", "status": "created"}, {"status": "cancelled", "steps": [{"status": "done"}]}, "已停止", "neutral"),
        "workflow_fallback": ({"task_type": "modeling", "status": "created", "workflow_status": "review"}, None, "需复核", "review"),
    }
    script = _surface_harness() + "\n" + "\n".join(
        [
            f"const fixtures = {json.dumps(fixtures, ensure_ascii=False)};",
            "const out = {};",
            "for (const [name, [record, plan]] of Object.entries(fixtures)) {",
            "  const task = { id: name, name, ...record };",
            "  plans = plan ? { [name]: plan } : {};",
            "  batches = {};",
            "  selectedTaskId = name; selectedTask = task;",
            "  projectedValidationChildTaskId = ''; projectedValidationChildTask = null;",
            "  out[name] = { sidebar: sidebarRow(task), title: titlePill(task), rail: railCapsule() };",
            "}",
            "process.stdout.write(JSON.stringify(out));",
        ]
    )
    data = _run_module(session_projection_fixture(script))
    for name, (_record, _plan, word, tone) in fixtures.items():
        surfaces = data[name]
        assert {value["label"] for value in surfaces.values()} == {word}, (name, surfaces)
        assert {value["tone"] for value in surfaces.values()} == {tone}, (name, surfaces)


def test_running_batch_title_uses_worst_first_like_the_sidebar():
    # Batch 4 review B1: while the runner is active the title must not fall
    # back to 执行中 when an item already failed or waits for the user.
    cases = {
        "partial": [("failed", "failed"), ("running", ""), ("queued", "")],
        "waiting": [("awaiting_confirmation", ""), ("running", "")],
        "review": [("review_required", "manual_review"), ("running", "")],
        "allRunning": [("running", ""), ("queued", "")],
    }
    lines = [_surface_harness()]
    lines.append("const out = {};")
    for name, items in cases.items():
        item_js = ", ".join(
            f"{{ id: '{name}-{i}', child_task_id: '{name}-c{i}', ordinal: {i + 1}, status: '{status}', outcome: '{outcome}' }}"
            for i, (status, outcome) in enumerate(items)
        )
        lines.extend([
            f"{{ const parent = {{ id: 'p-{name}', task_type: 'validation_batch', run_mode: 'agent', status: 'running', active_job_kind: 'validation_batch' }};",
            f"  const payload = normalizeValidationBatchPayload({{ batch: {{ parent_task_id: 'p-{name}', status: 'running', item_count: {len(items)} }}, items: [{item_js}] }});",
            f"  batches = {{ 'p-{name}': payload }}; selectedTaskId = 'p-{name}'; selectedTask = parent;",
            f"  out['{name}'] = {{ sidebar: sidebarRow(parent), title: titlePill(parent) }}; }}",
        ])
    lines.append("process.stdout.write(JSON.stringify(out));")
    data = _run_module(session_projection_fixture("\n".join(lines)))
    expected = {"partial": "部分失败", "waiting": "待确认", "review": "需复核", "allRunning": "执行中"}
    for name, word in expected.items():
        assert data[name]["sidebar"]["label"] == data[name]["title"]["label"] == word, (name, data[name])
        assert data[name]["sidebar"]["tone"] == data[name]["title"]["tone"], (name, data[name])


def test_batch_chip_carries_exactly_one_status_word():
    # Batch 4 review U1: a status-named stage must not become a second word.
    script = _surface_harness() + "\n" + "\n".join(
        [
            "const parent = { id: 'parent', task_type: 'validation_batch', run_mode: 'agent', status: 'review_required' };",
            "const payload = normalizeValidationBatchPayload({",
            "  batch: { parent_task_id: 'parent', status: 'completed', item_count: 4 },",
            "  items: [",
            "    { id: 'a', child_task_id: 'c-a', ordinal: 1, status: 'review_required', stage: 'completed', outcome: 'manual_review' },",
            "    { id: 'b', child_task_id: 'c-b', ordinal: 2, status: 'succeeded', stage: 'completed', outcome: 'pass' },",
            "    { id: 'c', child_task_id: 'c-c', ordinal: 3, status: 'running', stage: 'pmml_scoring', outcome: '' },",
            "    { id: 'd', child_task_id: 'c-d', ordinal: 4, status: 'running', stage: 'some_new_stage', outcome: '' },",
            "  ],",
            "});",
            "batches = { parent: payload }; selectedTaskId = 'parent'; selectedTask = parent;",
            "process.stdout.write(JSON.stringify(['c-a', 'c-b', 'c-c', 'c-d'].map((id) => chip(payload, id).text)));",
        ]
    )
    texts = _run_module(session_projection_fixture(script))
    assert texts == ["需复核", "已完成", "执行中 · PMML打分", "执行中"]
