"""F13: behaviour tests for app.js scroll follow-mode and render signatures.

These replace source-substring assertions from test_frontend_static_v2.py.
Each test slices the real app.js function and runs it in node against the
real task session, with recorded doubles for the DOM.
"""

from pathlib import Path
import subprocess

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
const calls = [];
const frames = [];
globalThis.window = { requestAnimationFrame: callback => frames.push(callback) };
globalThis.requestAnimationFrame = callback => frames.push(callback);
const flushFrames = () => frames.splice(0).forEach(callback => callback());
''' + helpers + body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


FOLLOW_DOUBLES = '''
const AGENT_USER_SCROLL_INPUT_WINDOW_MS = 250;
const AGENT_AUTO_SCROLL_BOTTOM_TOLERANCE_PX = 2;
let agentAutoScrollFollows = true;
let lastUserScrollInputAt = performance.now();
let agentMode = true;
const selectedTaskIsAgentMode = () => agentMode;
const panel = { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 };
const $ = () => panel;
const at = (scrollTop, overrides = {}) => {
  Object.assign(panel, { scrollHeight: 1000, clientHeight: 400, scrollTop }, overrides);
  lastUserScrollInputAt = performance.now();
  recomputeAgentAutoScrollFollow();
  return agentAutoScrollFollows;
};
'''


def test_follow_mode_tracks_user_scrolls_and_ignores_impossible_distances():
    run_node(FOLLOW_DOUBLES + '''
assert.equal(at(300), false, "scrolling up disengages follow-mode");
assert.equal(at(599), true, "within the 2px tolerance re-engages it");
at(300);
assert.equal(at(700), false, "overscroll bounce (negative distance) leaves it alone");
assert.equal(at(0, { scrollHeight: 300 }), false, "a panel that cannot scroll leaves it alone");
// Programmatic scrolls (no recent wheel/touch input) never flip the flag.
lastUserScrollInputAt = performance.now() - 10_000;
Object.assign(panel, { scrollHeight: 1000, clientHeight: 400, scrollTop: 600 });
recomputeAgentAutoScrollFollow();
assert.equal(agentAutoScrollFollows, false);
agentMode = false;
assert.equal(at(600), false, "manual mode does not own follow-mode");
''', "function recomputeAgentAutoScrollFollow(")


def test_result_scroll_recomputes_follow_mode_on_every_scroll():
    run_node('''
const rememberResultScrollPosition = () => calls.push("remember");
const scheduleTaskHeroGlassState = () => calls.push("glass");
const recomputeAgentAutoScrollFollow = () => calls.push("follow");
taskSession.selectTask({ id: "a" });
handleResultScroll();
taskSession.panels.scroll.pendingRestoreTaskId = "a";
handleResultScroll();
assert.deepEqual(calls, ["remember", "glass", "follow", "glass", "follow"]);
''', "function handleResultScroll(")


def test_step_fingerprint_keys_on_backend_progress_not_the_clock():
    run_node('''
const step = { id: "s1", status: "running", started_at: "t0", elapsed_seconds: 3, cell_count: 2, title: "x" };
const first = JSON.stringify(stepFingerprint([step]));
await new Promise(resolve => setTimeout(resolve, 5));
assert.equal(JSON.stringify(stepFingerprint([step])), first, "a clock tick alone never changes it");
assert.equal(JSON.stringify(stepFingerprint([{ ...step, title: "renamed" }])), first);
assert.notEqual(JSON.stringify(stepFingerprint([{ ...step, status: "succeeded" }])), first);
assert.notEqual(JSON.stringify(stepFingerprint([{ ...step, elapsed_seconds: 4 }])), first);
assert.deepEqual(stepFingerprint(null), []);
''', "function stepFingerprint(")


def test_manual_section_scroll_is_dropped_after_a_task_switch():
    run_node('''
let agentMode = false;
const selectedTaskIsAgentMode = () => agentMode;
const workflowSteps = [{ id: "metrics", target: "metricsSection" }, { id: "scan" }];
const scrollStepTarget = target => calls.push([target, taskSession.taskId]);
taskSession.selectTask({ id: "a" });
scrollToManualWorkflowSection("metrics");
flushFrames();
scrollToManualWorkflowSection("metrics");
taskSession.selectTask({ id: "b" });
flushFrames();
scrollToManualWorkflowSection("scan");
agentMode = true;
scrollToManualWorkflowSection("metrics");
flushFrames();
assert.deepEqual(calls, [["metricsSection", "a"]]);
''', "function scrollToManualWorkflowSection(")


def test_action_status_repaints_when_the_same_copy_belongs_to_another_task():
    run_node('''
const renderSignatures = {};
const pill = { textContent: "", className: "", dataset: {}, closest: () => null };
const $ = id => (id === "actionStatus" ? pill : null);
const actionStatusPill = message => ({ label: message, tone: "info", key: "k" });
const agentFaceForStatus = () => "idle";
let liveAgentFace = "idle";
const syncLiveAgentFaces = () => {};
const describeActionStatus = message => message;
const setActionErrorDetail = () => calls.push("detail");
const syncTaskHeroGlassLayout = () => {};
taskSession.selectTask({ id: "a" });
setActionStatus("运行中");
setActionStatus("运行中");
assert.equal(calls.length, 1, "an identical status for the same task is not repainted");
taskSession.selectTask({ id: "b" });
setActionStatus("运行中");
assert.equal(calls.length, 2, "the same copy on a newly selected task repaints");
assert.equal(pill.textContent, "运行中");
''', "function setActionStatus(")


def test_pet_mood_treats_idle_writing_artifacts_as_complete():
    run_node('''
let busy = false;
const taskStopped = task => task?.stopped === true;
const taskUsesPlanRail = () => false;
const selectedTaskIsBusy = () => busy;
const mood = status => { taskSession.selectTask({ id: "a", status }); return basePetMoodFromTask(); };
assert.equal(mood("writing_artifacts"), "complete", "no report job yet: stage 3 is done");
assert.equal(mood("scanned"), "complete");
assert.equal(mood("computing_metrics"), "running");
busy = true;
assert.equal(mood("writing_artifacts"), "running", "an active job is what makes it run");
busy = false;
assert.equal(mood("succeeded"), "success");
assert.equal(mood("failed"), "failed");
taskSession.selectTask({ id: "a", status: "running", stopped: true });
assert.equal(basePetMoodFromTask(), "idle");
''', "function basePetMoodFromTask(")


def test_reproducibility_scores_render_missing_values_as_a_dash():
    run_node('''
assert.equal(formatScoreValue(null), "-");
assert.equal(formatScoreValue(undefined), "-");
assert.equal(formatScoreValue(""), "-");
assert.equal(formatScoreValue("abc"), "-");
assert.equal(formatScoreValue(0), "0.000000");
assert.equal(formatScoreValue("0.1234567"), "0.123457");
''', "function formatScoreValue(")


def test_sparkline_raw_html_is_only_trusted_with_the_local_marker():
    run_node('''
const { renderCellByKind } = await import("./marvis/static/js/metric-tables.js");
const svg = "<svg><script>x</script></svg>";
const context = { headerLabel: "", fractions: new Map(), rowIndex: 0 };
assert.equal(renderCellByKind({ kind: "trend-spark", __localHtml: true }, svg, context).html, svg);
const untrusted = renderCellByKind({ kind: "trend-spark" }, svg, context).html;
assert.ok(!untrusted.includes("<script>"), "server-supplied specs never inject raw HTML");
''')


def test_delete_refuses_tasks_with_a_server_job_but_not_a_stale_status():
    run_node('''
let serverBusy = true;
const statuses = [];
const reconcileTaskBeforeDelete = async task => task;
const taskBusyAction = () => null;
const taskServerBusyAction = () => serverBusy;
const setActionStatus = (message, kind) => statuses.push([message, kind || "info"]);
const loadTaskPurgeSummary = async () => [];
const taskDisplayName = task => task.id;
const isValidationBatchTask = () => false;
const showPlatformConfirm = async () => false;
await deleteTask({ id: "a", status: "succeeded", active_job_kind: "metrics" });
assert.deepEqual(statuses.at(-1), ["运行中的任务不能删除。", "error"]);
serverBusy = false;
// A stale "running" status without a server job goes on to the confirmation.
await deleteTask({ id: "a", status: "running" });
assert.deepEqual(statuses.at(-1), ["已取消删除。", "info"]);
''', "async function deleteTask(")


def test_agent_markdown_keeps_ordered_section_numbers_across_blank_lines():
    run_node('''
const { renderAgentMarkdown } = await import("./marvis/static/js/render-agent.js");
const html = renderAgentMarkdown("1. 第一节 **粗**\\n\\n2. 第二节\\n\\n5. 第五节 <b>x</b>");
assert.ok(html.includes("<ol><li>第一节 <strong>粗</strong></li></ol>"), html);
assert.ok(html.includes('<ol start="2"><li>第二节</li></ol>'), html);
assert.ok(html.includes('<ol start="5">'), html);
assert.ok(!html.includes("<b>x</b>"), "list items are still escaped");
''')


def test_computing_metrics_status_phrase_comes_from_the_snapshot():
    run_node('''
const taskStopped = () => false;
const taskUsesPlanRail = () => false;
const taskPlanWorkflowStatusSnapshot = () => null;
const snapshot = taskActionStatusSnapshot({ id: "a", status: "computing_metrics" });
assert.ok(snapshot.message.includes("指标概览进行中"), snapshot.message);
assert.equal(snapshot.kind, "busy");
''', "function taskActionStatusSnapshot(")


def test_step_checker_marks_by_state_and_stopped_has_no_square():
    run_node('''
const { stepCheckerHtml } = await import("./marvis/static/js/step-checker.js");
assert.equal(stepCheckerHtml("stopped"), '<span class="check-icon stopped" aria-hidden="true"></span>');
assert.ok(stepCheckerHtml("succeeded").includes("<svg"));
assert.ok(stepCheckerHtml("failed").includes('class="check-icon failed"'));
assert.ok(!stepCheckerHtml("stopped").includes("<rect"));
''')
