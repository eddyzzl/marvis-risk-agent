"""A gate keeps its controls while the user asks questions (eval round 1).

Two questions at the feature-binning gate removed 「跳过分箱并生成报告」 and
「分析所选特征并生成报告」 from the DOM: only a gate that was the very last
Agent message counted as the latest gate. Whether the controls can act is
still decided against the plan step and confirmation snapshot.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "marvis" / "static" / "js" / "agent-conversation-mount.js").as_uri()


def _latest(messages):
    script = "\n".join([
        f"const m = await import({json.dumps(MODULE)});",
        f"process.stdout.write(JSON.stringify(m.lastGateMessageId({json.dumps(messages)})));",
    ])
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


GATE = {"id": "gate", "role": "assistant", "metadata": {"kind": "gate", "plan_id": "p", "step_id": "s1"}}


def _user(text):
    return {"id": text, "role": "user", "metadata": {"intent": "feature_analysis"}, "content": text}


def test_questions_on_the_gate_step_keep_it_latest():
    answer = {"id": "a1", "role": "assistant", "metadata": {"plan_id": "p", "step_id": "s1", "confirmation_snapshot": {}}}
    chat = {"id": "a2", "role": "assistant", "metadata": {"intent": "chat"}}
    replan = {"id": "a3", "role": "assistant", "metadata": {"plan_id": "p", "step_id": "s1", "action_outcome": "replan_failed"}}
    messages = [GATE, _user("为什么没算稳定性？"), answer, _user("还有呢"), chat, _user("加上 PSI"), replan]
    assert _latest(messages) == "gate"


def test_acks_progress_and_other_steps_end_the_gate():
    ack = {"id": "ack", "role": "assistant", "metadata": {"intent": "ui_action_ack", "ui_action": {"type": "confirm"}, "expected_plan_id": "p"}}
    other_step = {"id": "next", "role": "assistant", "metadata": {"plan_id": "p", "step_id": "s2"}}
    done = {"id": "done", "role": "assistant", "metadata": {"plan_id": "p", "execution_completed": True}}
    progress = {"id": "prog", "role": "assistant", "metadata": {"kind": "tool_progress", "plan_id": "p", "step_id": "s1"}}
    for later in (ack, other_step, done, progress):
        assert _latest([GATE, _user("确认"), later]) == "", later["id"]
    # The newest of two gates wins.
    newer = {"id": "gate2", "role": "assistant", "metadata": {"kind": "gate", "plan_id": "p", "step_id": "s2"}}
    assert _latest([GATE, newer]) == "gate2"







def test_gate_widgets_rebuild_when_the_gate_becomes_actionable():
    # Eval S6: the join key picker was drawn while the diagnosis job was still
    # finishing; when the job ended only its actionable state changed and the
    # content-only fast path kept it read-only until a reload. Runs the real
    # renderAgentConversation over stubs (review M6).
    from tests.javascript_source import slice_function

    app_js = (ROOT / "marvis" / "static" / "app.js").read_text(encoding="utf-8")
    render = slice_function(app_js, "function renderAgentConversation(")
    script = """
import assert from "node:assert/strict";
import { latestSuggestedReply } from "./marvis/static/js/composer-suggestion.js";
const noop = () => {};
const element = { dataset: {}, classList: { toggle: noop, remove: noop }, setAttribute: noop, placeholder: "" };
const $ = () => element;
let actionable = false;
let rebuilds = 0, patches = 0;
const gate = { id: "gate", role: "assistant", content: "诊断完成", metadata: { kind: "gate", plan_id: "p", step_id: "s" } };
const taskSession = { taskId: "t", task: { id: "t", task_type: "data_join" }, messages: [gate] };
const renderReportDraftWorkspace = noop, syncRiskMaterialUploadControl = noop;
const renderAgentAcceptanceModePreference = noop, renderAgentModelOptions = noop;
const renderAgentEffortPreference = noop, renderAgentQuickReplies = noop;
const strategyCandidateLabController = { renderAvailability: noop };
const workbenchChrome = { setThreadFilterAvailable: noop };
const requestAnimationFrame = noop, syncAgentComposerClearance = noop;
const selectedTaskIsAgentMode = () => true, selectedTaskNeedsManualRiskIntake = () => false;
const taskUsesPlanRail = () => true;
const planRailController = { planId: () => "p", resetFetchThrottle: noop };
const renderWorkflowStepper = noop, renderDriverManualAnalysis = noop;
let lastDriverManualAnalysisSignature = null, lastAgentRenderSignature = null, lastAgentStructuralSignature = null;
const agentTimelineVisibleStages = () => ["chat"];
const agentReportMessagesForDisplay = (messages) => messages;
const hideSupersededTuningThinking = (messages) => messages;
const filterRecoveredWorkflowFailures = (messages) => messages;
const driverManualMessageStepStatus = () => "awaiting_confirm";
const driverGateMessageInteractionState = () => ({ actionable, localBusy: !actionable, serverBusy: false, pendingClaim: "", stepStatus: "awaiting_confirm" });
const completedReportReadyForDownloads = () => false;
const agentStructuralSignature = () => "layout";
const updateAgentMessageContentsInPlace = () => { patches += 1; return true; };
const requestAgentConversationScrollToLatest = noop, freezeAgentSectionSnapshotsForReruns = noop;
const clearAgentStageMessages = noop, parkReportDraftCard = noop, placeReportDraftCard = noop;
const renderAgentTimeline = () => { rebuilds += 1; };
const syncLiveAgentFaces = noop, liveAgentFace = null, attachCalibrationInteractions = noop;
""" + render + """
renderAgentConversation();
assert.equal(rebuilds, 1);
renderAgentConversation();
assert.equal(rebuilds, 1, "an unchanged poll does not rebuild");
actionable = true;
renderAgentConversation();
assert.equal(rebuilds, 2, "the gate turning actionable rebuilds its widget markup");
assert.equal(patches, 0);
gate.content = "诊断完成（流式补全）";
renderAgentConversation();
assert.equal(rebuilds, 2, "a content-only change still takes the fast path");
assert.equal(patches, 1);
gate.metadata.suggested_reply = "确认继续";
renderAgentConversation();
assert.equal(element.placeholder, "确认继续");
assert.equal(element.dataset.suggestion, "确认继续");
taskSession.messages.push({id:"user",role:"user",content:"我的问题"});
renderAgentConversation();
assert.equal(element.dataset.suggestion, "");
assert.equal(element.placeholder, "输入执行步骤或问题");
process.stdout.write("ok");
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script], check=False, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"
