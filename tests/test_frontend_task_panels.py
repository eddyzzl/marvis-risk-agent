"""F17: per-task panel state lives in the task session, exercised through app.js code.

Each test slices the real app.js functions and runs them in node against the
real ``createTaskSession()``; DOM and rendering collaborators are recorded
doubles so the assertions are about state ownership and render decisions.
"""

import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")


def run_node(body: str, *functions: str) -> None:
    helpers = "\n".join(slice_function(APP, name) for name in functions)
    script = '''
import assert from "node:assert/strict";
import { createTaskSession } from "./marvis/static/js/task-session.js";
import { signatureFromParts } from "./marvis/static/js/ui-utils.js";
const taskSession = createTaskSession();
const taskRequests = taskSession.requests;
const calls = [];
''' + helpers + body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    check=False)
    assert result.returncode == 0, result.stderr


def test_panel_groups_are_sealed_and_owned_per_session():
    run_node('''
const other = createTaskSession();
taskSession.panels.metrics.values = { ks: 0.3 };
assert.deepEqual(other.panels.metrics.values, {});
assert.throws(() => { taskSession.panels.metrics.valuez = 1; }, TypeError);
assert.throws(() => { taskSession.panels.extra = {}; }, TypeError);
assert.deepEqual(Object.keys(taskSession.panels).sort(),
  ["contract", "metrics", "notebook", "reportDraft", "scroll"]);
''')


METRIC_PREVIEW = (
    "function metricPreviewSignature(",
    "function renderMetricPreview(\n",
    "function currentMetricPreviewHasValues(",
    "function rememberScoreEvidence(",
)
METRIC_DOUBLES = '''
let modelName = "A卡";
const renderSignatures = {};
const renderMetricSectionVisibility = () => {};
const metricPreviewEmptyMessage = () => "";
const renderMetricPreviewEmpty = message => calls.push(["empty", message]);
const workbenchTask = () => ({ model_name: modelName });
const workbenchTaskId = () => taskSession.taskId;
const scoreEvidenceTile = evidence => ({ key: "score", kind: evidence.kind });
const renderMetricTableSection = section => section.title;
const validationMetricCardModel = sections => sections.length;
const validationMetricCardHtml = (model, options) => {
  calls.push(["card", options.title, options.scoreTile?.kind || null]);
  return "card";
};
const element = { innerHTML: "" };
const $ = () => element;
const attachRocInteractions = () => {};
const attachMetricTooltip = () => {};
const attachResultCardInteractions = () => {};
const sections = [{ title: "分箱", tables: [{ layout: "bin_table" }] }];
'''


def test_metric_preview_records_owner_and_skips_identical_repaints():
    run_node(METRIC_DOUBLES + '''
taskSession.selectTask({ id: "a" });
renderMetricPreview({ ks: 0.31 }, null, sections);
assert.equal(taskSession.panels.metrics.valuesTaskId, "a");
assert.equal(currentMetricPreviewHasValues("a"), true);
assert.equal(currentMetricPreviewHasValues("b"), false);
// A poll with the same payload leaves the DOM alone.
renderMetricPreview({ ks: 0.31 }, null, sections);
assert.equal(calls.filter(([kind]) => kind === "card").length, 1);
// The model name and the score tile are part of the signature.
modelName = "B卡";
renderMetricPreview();
taskSession.panels.metrics.scoreEvidence = { kind: "pmml", taskId: "a", data: { status: "pass" } };
renderMetricPreview();
assert.deepEqual(calls.filter(([kind]) => kind === "card"),
  [["card", "A卡", null], ["card", "B卡", null], ["card", "B卡", "pmml"]]);
''', *METRIC_PREVIEW)


def test_score_evidence_keeps_tile_on_empty_poll_and_repaints_only_its_owner():
    run_node(METRIC_DOUBLES + '''
taskSession.selectTask({ id: "a" });
renderMetricPreview({ ks: 0.31 }, null, sections);
const cards = () => calls.filter(([kind]) => kind === "card").length;
taskSession.panels.metrics.sectionsOwnerId = "b";
rememberScoreEvidence("pmml", { status: "pass" }, "a");
// The sections on screen belong to another model: remember, do not repaint.
assert.equal(taskSession.panels.metrics.scoreEvidence.kind, "pmml");
assert.equal(cards(), 1);
// An empty poll for the same model keeps the tile.
rememberScoreEvidence("pmml", {}, "a");
assert.equal(taskSession.panels.metrics.scoreEvidence.data.status, "pass");
taskSession.panels.metrics.sectionsOwnerId = "a";
rememberScoreEvidence("reproducibility", { match_count: 3 }, "a");
assert.equal(cards(), 2);
// Unchanged evidence never repaints.
rememberScoreEvidence("reproducibility", { match_count: 3 }, "a");
assert.equal(cards(), 2);
''', *METRIC_PREVIEW)


CONTRACT_DOUBLES = '''
let pmml = true;
let current = "m1";
const usesPmmlScoringWorkflow = () => pmml;
const workbenchTaskId = () => current;
const taskFailureStage = () => "";
const taskStopped = () => false;
const taskBusyAction = () => "";
const stepFingerprint = steps => steps.length;
const notebookStepsForRail = () => [];
const metricStepsForRail = () => [];
const contract = taskSession.panels.contract;
'''


def test_pmml_run_waits_only_for_the_current_models_unconfirmed_contract():
    run_node(CONTRACT_DOUBLES + '''
const WAIT = "请先确认上面的验证字段";
assert.equal(stepActionBlockedReason("notebook"), "");
contract.value = { status: "pending_confirmation" };
contract.taskId = "m2";
assert.equal(stepActionBlockedReason("notebook"), "", "another model's contract never blocks");
contract.taskId = "m1";
assert.equal(stepActionBlockedReason("notebook"), WAIT);
assert.equal(stepActionBlockedReason("report"), "");
pmml = false;
assert.equal(stepActionBlockedReason("notebook"), "");
pmml = true;
contract.value = { status: "ready" };
assert.equal(stepActionBlockedReason("notebook"), "");
''', "function stepActionBlockedReason(")


def test_workflow_rail_signature_follows_the_current_contract_status():
    run_node(CONTRACT_DOUBLES + '''
const task = { id: "m1", status: "scanned" };
const before = workflowStepperSignature(task);
contract.taskId = "m1";
contract.value = { status: "pending_confirmation" };
const pending = workflowStepperSignature(task);
assert.notEqual(pending, before);
contract.value = { status: "ready" };
assert.notEqual(workflowStepperSignature(task), pending);
// Another model's contract does not churn this rail.
contract.taskId = "m2";
assert.equal(workflowStepperSignature(task), before);
''', "function workflowStepperSignature(")


NOTEBOOK_DOUBLES = '''
const v2WorkflowSubstepPlans = {
  notebook: [{ id: "system-score", title: "PMML 打分" }, { id: "system-check", title: "一致性" }],
};
const notebookStepTone = status => status === "ok" ? "succeeded" : (status === "error" ? "failed" : "pending");
'''


def test_logical_substeps_read_evidence_from_the_sessions_notebook_steps():
    run_node(NOTEBOOK_DOUBLES + '''
taskSession.panels.notebook.steps = [{ id: "system-score", status: "ok", detail: "kept" }];
const running = v2WorkflowSubsteps("notebook", "running");
assert.deepEqual(running.map(step => [step.id, step.status]),
  [["system-score", "ok"], ["system-check", "pending"]]);
assert.equal(running[0].detail, "kept");
assert.equal(running[0].system, true);
assert.deepEqual(v2WorkflowSubsteps("notebook", "failed").map(step => step.status), ["succeeded", "pending"]);
assert.deepEqual(v2WorkflowSubsteps("notebook", "succeeded").map(step => step.status), ["succeeded", "succeeded"]);
assert.deepEqual(v2WorkflowSubsteps("notebook", "stopped").map(step => step.status), ["succeeded", "stopped"]);
taskSession.panels.notebook.steps = [];
assert.deepEqual(v2WorkflowSubsteps("notebook", "failed").map(step => step.status), ["failed", "pending"]);
''', "function v2WorkflowSubsteps(")


def test_notebook_renderer_stores_every_normalized_step_and_repaints_the_rail():
    run_node('''
const normalizeNotebookSteps = (steps, cells) => [...steps, ...cells.map(id => ({ id }))];
const mergePendingSystemSteps = steps => [...steps, { id: "system-pending" }];
const renderReproducibilitySectionVisibility = () => calls.push("repro");
const renderMetricSectionVisibility = () => calls.push("metrics");
const renderWorkflowStepper = () => calls.push(["rail", taskSession.panels.notebook.steps.length]);
const steps = Array.from({ length: 12 }, (_, index) => ({ id: `cell-${index}` }));
renderNotebookSteps(steps, ["extra"]);
// No cap: all 12 cells, the extra cell and the pending system step are kept.
assert.equal(taskSession.panels.notebook.steps.length, 14);
assert.deepEqual(calls, ["repro", "metrics", ["rail", 14]]);
''', "function renderNotebookSteps(")


SCROLL_DOUBLES = '''
let agentMode = true;
let agentAutoScrollFollows = true;
let agentAutoScrollFrame = null;
const scrolls = [];
const scrollContent = { scrollHeight: 900, scrollTo: options => scrolls.push(options.top) };
const $ = () => scrollContent;
globalThis.window = {
  requestAnimationFrame: callback => { callback(); return 1; },
  cancelAnimationFrame: () => {},
};
const selectedTaskIsAgentMode = () => agentMode;
const scheduleTaskHeroGlassState = () => calls.push("glass");
const rememberResultScrollPosition = () => calls.push(["remember", taskSession.taskId]);
const recomputeAgentAutoScrollFollow = () => calls.push("follow");
const scroll = taskSession.panels.scroll;
'''


def test_agent_conversation_does_not_jump_while_its_scroll_is_being_restored():
    run_node(SCROLL_DOUBLES + '''
taskSession.selectTask({ id: "a" });
requestAgentConversationScrollToLatest();
assert.deepEqual(scrolls, [900]);
scroll.suppressAutoScrollTaskId = "a";
requestAgentConversationScrollToLatest();
assert.deepEqual(scrolls, [900], "restoring task a's scroll suppresses the jump");
scroll.suppressAutoScrollTaskId = "b";
requestAgentConversationScrollToLatest();
assert.deepEqual(scrolls, [900, 900], "another task's restore does not");
taskSession.panels.reportDraft.expanded = true;
requestAgentConversationScrollToLatest();
assert.equal(scrolls.length, 2, "an open report layer keeps its place");
taskSession.panels.reportDraft.expanded = false;
agentAutoScrollFollows = false;
requestAgentConversationScrollToLatest();
assert.equal(scrolls.length, 2, "disengaged follow-mode never jumps, regardless of distance");
agentAutoScrollFollows = true;
requestAgentConversationScrollToLatest();
assert.equal(scrolls.length, 3, "only explicit follow-mode restores scrolling");
''', "function requestAgentConversationScrollToLatest(")


def test_result_scroll_is_remembered_except_while_this_task_restores_it():
    run_node(SCROLL_DOUBLES + '''
taskSession.selectTask({ id: "a" });
handleResultScroll();
scroll.pendingRestoreTaskId = "a";
handleResultScroll();
scroll.pendingRestoreTaskId = "b";
handleResultScroll();
assert.deepEqual(calls.filter(call => Array.isArray(call)), [["remember", "a"], ["remember", "a"]]);
assert.equal(calls.filter(call => call === "glass").length, 3);
''', "function handleResultScroll(")
