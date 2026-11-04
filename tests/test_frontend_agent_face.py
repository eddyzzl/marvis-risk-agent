"""Mascot avatar (2026-10-14 redesign, D6 / batch 4).

The Agent avatar is the mascot head in two real-logo layers. Its expression is
the ``face`` field of the unified status vocabulary (``js/task-status.js``);
it appears on the first message of each Agent run (newest live, history idle),
next to the title status, and on the rail step that waits on the user.
"""

import json
import re
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function
from tests.test_frontend_static_v2 import _BROWSER_STUBS
from tests.test_frontend_static_v2 import _MINI_DOM_JS
from tests.test_frontend_static_v2 import _prepare_app_module_for_node
from tests.test_frontend_static_v2 import _run_node_capture_json

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


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


def test_agent_face_markup_defaults_and_escapes_the_speaker_label():
    data = _run_module(
        """
import { AGENT_FACES, agentFaceHtml } from "./marvis/static/js/agent-face.js";
process.stdout.write(JSON.stringify({
  faces: AGENT_FACES,
  plain: agentFaceHtml(),
  waitSmall: agentFaceHtml({ face: "wait", size: "sm" }),
  unknown: agentFaceHtml({ face: "angry", size: "xl" }),
  labelled: agentFaceHtml({ face: "busy", label: '<b>蛋黄</b>"' }),
}));
"""
    )
    assert data["faces"] == ["idle", "busy", "wait", "done", "fail"]
    assert data["plain"] == '<span class="agent-face" data-face="idle" data-size="md" aria-hidden="true"></span>'
    assert data["waitSmall"] == '<span class="agent-face" data-face="wait" data-size="sm" aria-hidden="true"></span>'
    assert data["unknown"] == '<span class="agent-face" data-face="idle" data-size="md" aria-hidden="true"></span>'
    assert 'role="img"' in data["labelled"]
    assert "aria-hidden" not in data["labelled"]
    assert "<b>" not in data["labelled"]
    assert "&lt;b&gt;蛋黄&lt;/b&gt;" in data["labelled"]


def test_agent_face_comes_from_the_status_vocabulary_face_field():
    data = _run_module(
        """
import { agentFaceForStatus } from "./marvis/static/js/agent-face.js";
import { STATUS, actionStatusPresentation, planWorkflowStatus, statusPresentation } from "./marvis/static/js/task-status.js";
const byKey = Object.fromEntries(Object.keys(STATUS).map((key) => [key, agentFaceForStatus(statusPresentation(key))]));
const tableFaces = Object.fromEntries(Object.entries(STATUS).map(([key, value]) => [key, value.face]));
const planGate = planWorkflowStatus({ status: "awaiting_confirm", steps: [{ status: "awaiting_confirm", title: "确认拼接" }] });
const pill = actionStatusPresentation("等待确认", "info", {
  task: { id: "t1", task_type: "feature_analysis" },
  usesPlanWorkflow: true,
  workflowSnapshot: planGate,
});
process.stdout.write(JSON.stringify({
  byKey,
  tableFaces,
  pill: agentFaceForStatus(pill),
  bareKey: agentFaceForStatus("failed"),
  none: agentFaceForStatus(null),
  // A word without a key or face is never parsed into an expression.
  wordOnly: agentFaceForStatus({ label: "待确认", tone: "neutral" }),
}));
"""
    )
    assert data["byKey"] == data["tableFaces"]
    assert data["pill"] == "wait"
    assert data["bareKey"] == "fail"
    assert data["none"] == "idle"
    assert data["wordOnly"] == "idle"


def test_sync_live_agent_faces_sets_only_the_newest_avatar():
    data = _run_module(
        """
import { syncLiveAgentFaces } from "./marvis/static/js/agent-face.js";
const nodes = [{ dataset: { face: "idle" } }, { dataset: { face: "busy" } }, { dataset: { face: "idle" } }];
let selector = "";
const root = { querySelectorAll(value) { selector = value; return nodes; } };
syncLiveAgentFaces(root, "wait");
const afterWait = nodes.map((node) => node.dataset.face);
syncLiveAgentFaces(root, "nonsense");
const afterUnknown = nodes.map((node) => node.dataset.face);
syncLiveAgentFaces(null, "done");
syncLiveAgentFaces({ querySelectorAll: () => [] }, "done");
process.stdout.write(JSON.stringify({ selector, afterWait, afterUnknown }));
"""
    )
    # Batch 4 review U4: the live face is the newest run-start avatar.
    assert data["selector"] == ".agent-message.assistant[data-run-start] > .agent-face"
    assert data["afterWait"] == ["idle", "idle", "wait"]
    assert data["afterUnknown"] == ["idle", "idle", "idle"]


def test_agent_messages_show_one_avatar_per_agent_run_without_the_speaker_word():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    boot_marker = 'document.addEventListener(\n  "mousedown"'
    app_js = _prepare_app_module_for_node(app_js[: app_js.index(boot_marker)])
    view_url = (STATIC / "js" / "agent-conversation-view.js").as_uri()
    messages = [
        {"id": "a1", "role": "assistant", "stage": "chat", "content": "收到。",
         "metadata": {"phase": "特征", "step_title": "筛选特征"}},
        {"id": "a2", "role": "assistant", "stage": "chat", "content": "筛选完成。"},
        {"id": "u1", "role": "user", "stage": "chat", "content": "好"},
        {"id": "a3", "role": "assistant", "stage": "chat", "content": "继续。"},
    ]
    driver = "\n".join(
        [
            f"const {{ agentMessagesHtml }} = await import({json.dumps(view_url)});",
            f"const messages = {json.dumps(messages, ensure_ascii=False)};",
            "taskSession.selectTask({ id: 'task-face', task_type: 'modeling', run_mode: 'agent', validator: '' });",
            "const deps = { agentStageLabel, agentMessageHtml: (m, s, o = {}) => agentMessageHtml(m, s, { ...o, conversationOnly: true }) };",
            "const plain = agentMessagesHtml(messages, undefined, deps);",
            "agentValidatorAliases = { '于添': '蛋黄' };",
            "taskSession.refreshTask({ ...taskSession.task, validator: '于添' });",
            "const aliased = agentMessagesHtml(messages.slice(0, 1), undefined, deps);",
            "process.stdout.write(JSON.stringify({ plain, aliased }));",
        ]
    )
    payload = _run_node_capture_json(_BROWSER_STUBS + "\n" + app_js + "\n" + driver)
    plain = payload["plain"]
    # Two Agent runs (the user message splits them): one avatar each, idle as
    # rendered; the live expression is applied afterwards from the status.
    # Batch 4 review U4: every Agent message carries a face (CSS shows only
    # the run-starts, plus visible results under 只看结果); two runs -> two
    # labelled run-start avatars.
    assert plain.count('class="agent-face"') == 3
    assert plain.count(' data-run-start="true"') == 2
    assert plain.count('data-face="idle" data-size="md" role="img" aria-label="Agent"') == 2
    # The meta line renders one span per phrase (so each phrase translates on
    # its own); its visible text is what the user reads.
    def meta_texts(html):
        return [
            re.sub(r"<[^>]+>", "", inner)
            for inner in re.findall(r'<div class="agent-message-meta">(.*?)</div>', html)
        ]

    assert meta_texts(plain) == ["特征 · 筛选特征"]
    assert ">Agent<" not in plain
    # A brand.json validator alias stays visible as the speaker word, and is
    # user data, so the UI language switch never translates it.
    assert meta_texts(payload["aliased"]) == ["蛋黄 · 特征 · 筛选特征"]
    assert '<span translate="no">蛋黄</span>' in payload["aliased"]
    assert 'aria-label="蛋黄"' in payload["aliased"]


def test_title_status_face_tracks_the_resolved_status_after_the_signature_guard():
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    body = slice_function(app_js, "function setActionStatus(")
    assert body.index("renderSignatures.actionStatus === nextSignature") < body.index("agentFaceForStatus(info)")
    script = "\n".join(
        [
            'import { agentFaceForStatus, syncLiveAgentFaces } from "./marvis/static/js/agent-face.js";',
            'import { statusPresentation } from "./marvis/static/js/task-status.js";',
            "let liveAgentFace = 'idle';",
            "const renderSignatures = {};",
            "const taskSession = { taskId: 't1', task: { id: 't1' } };",
            "let next = statusPresentation('awaiting_confirm');",
            "function actionStatusPill() { return next; }",
            "function signatureFromParts(parts) { return parts.join('|'); }",
            "function setActionErrorDetail() {}",
            "function describeActionStatus() { return ''; }",
            "function requestAnimationFrame() {}",
            "function syncTaskHeroGlassLayout() {}",
            "const avatars = [{ dataset: { face: 'idle' } }, { dataset: { face: 'idle' } }];",
            "const els = {",
            "  actionStatus: { dataset: {}, closest: () => null },",
            "  taskStatusFace: { hidden: true, dataset: { face: 'idle' } },",
            "  resultScrollContent: { querySelectorAll: () => avatars },",
            "};",
            "function $(id) { return els[id] || null; }",
            body,
            "const seen = [];",
            "const snap = () => seen.push([els.taskStatusFace.hidden, els.taskStatusFace.dataset.face, avatars.map((a) => a.dataset.face).join(',')]);",
            "setActionStatus('等待确认', 'info'); snap();",
            "next = statusPresentation('running'); setActionStatus('执行中', 'busy'); snap();",
            "next = statusPresentation('failed'); setActionStatus('失败', 'error'); snap();",
            "next = null; setActionStatus('', 'info'); snap();",
            "process.stdout.write(JSON.stringify(seen));",
        ]
    )
    assert _run_module(script) == [
        [False, "wait", "idle,wait"],
        [False, "busy", "idle,busy"],
        [False, "fail", "idle,fail"],
        [True, "idle", "idle,idle"],
    ]

    index_html = (STATIC / "index.html").read_text(encoding="utf-8")
    # 2026-10-19 (mockup parity): face and status word sit in the title capsule.
    right = index_html[index_html.index('<span class="task-hero-state">'):]
    assert right.index('id="taskStatusFace"') < right.index('id="actionStatus"')
    assert '<span id="taskStatusFace" class="agent-face" data-face="idle" data-size="sm" aria-hidden="true" hidden></span>' in right
    # The full timeline rebuild re-applies the live face; the streaming fast
    # path only patches message content, so faces survive it untouched.
    render = slice_function(app_js, "function renderAgentConversation(")
    rebuild = render[render.index("renderAgentTimeline(displayedMessages);"):]
    assert rebuild.index('syncLiveAgentFaces($("resultScrollContent"), liveAgentFace);') < rebuild.index(
        'attachCalibrationInteractions($("agentMessages"));'
    )


def test_waiting_rail_steps_show_the_review_mark_and_keep_the_review_class():
    module_url = (STATIC / "js" / "v2" / "plan_rail_controller.js").as_uri()
    script = _MINI_DOM_JS + "\n" + "\n".join(
        [
            f"const {{ createPlanRailController }} = await import({json.dumps(module_url)});",
            "const __doc = makeDocument();",
            "const workflowStepper = __doc.createElement('div');",
            "const elements = { progressRail: { setAttribute() {} }, workflowStepper };",
            "function $(id) { return elements[id] || null; }",
            "const plan = { id: 'plan-1', status: 'awaiting_confirm', steps: [",
            "  { id: 's1', index: 0, phase: '数据准备', title: '拼接诊断', status: 'done', tool_ref: { plugin: 'data_ops', tool: 'propose_join' }, depends_on: [] },",
            "  { id: 's2', index: 1, phase: '数据准备', title: '确认拼接', status: 'awaiting_confirm', tool_ref: { plugin: 'data_ops', tool: 'confirm_join' }, depends_on: ['s1'] },",
            "  { id: 's3', index: 2, phase: '特征分析', title: '特征指标', status: 'pending', tool_ref: { plugin: 'modeling', tool: 'screen_features' }, depends_on: ['s2'] },",
            "] };",
            "globalThis.document = { createElement: (t) => __doc.createElement(t), querySelector() { return { textContent: '' }; } };",
            "globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({ plans: [plan] }) });",
            "const controller = createPlanRailController({ $, stepCheckerHtml: (s) => `<i class=\"check-icon ${s}\"></i>`, getSelectedTask: () => ({ task_type: 'feature_analysis', active_job_kind: null }), getTaskBusyAction: () => '', getSelectedTaskId: () => 'task-A', getAgentMessages: () => [], isAgentMode: () => true, renderWorkflowStepper: () => {}, setActionStatus: () => {} });",
            "const rs = {};",
            "controller.render({ force: true, renderSignatures: rs });",
            "await new Promise((r) => setTimeout(r, 20));",
            "controller.render({ force: true, renderSignatures: rs });",
            "const steps = Array.from(workflowStepper.querySelectorAll(':scope > .plan-rail-step'));",
            "process.stdout.write(JSON.stringify({",
            "  classes: steps.map((node) => node.className),",
            "  heads: steps.map((node) => {",
            "    const head = node.querySelector('.step-head');",
            "    const face = head.querySelector('.agent-face');",
            "    return { face: face ? { ...face.dataset } : null, checker: head.querySelector('.check-icon')?.className || '' };",
            "  }),",
            "  substepCheckers: Array.from(workflowStepper.querySelectorAll('.check-icon')).map((node) => node.className),",
            "}));",
        ]
    )
    data = _run_node_capture_json(script)
    assert data["classes"] == ["step plan-rail-step review", "step plan-rail-step pending"], data
    # 2026-10-19 user feedback: the rail carries no mascot; the step waiting
    # on the user shows the amber review mark.
    assert data["heads"] == [
        {"face": None, "checker": "check-icon review"},
        {"face": None, "checker": "check-icon pending"},
    ], data
    assert "check-icon review" in data["substepCheckers"], data

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    stepper = app_js[app_js.index("function renderWorkflowStepper"):app_js.index("function formatDate")]
    assert "agentFaceHtml" not in stepper
    assert "stepCheckerHtml(stepStatus)," in stepper
    # The shared review glyph stays for substeps and other callers.
    assert "check-icon review" in (STATIC / "js" / "step-checker.js").read_text(encoding="utf-8")


def test_agent_face_css_uses_the_two_logo_layers_namespaced_motion_and_reduced_motion():
    css = (STATIC / "css" / "agent-face.css").read_text(encoding="utf-8")
    assert 'url("../brand/marvis-face-base.png")' in css
    assert 'url("../brand/marvis-face-eyes.png")' in css
    assert (STATIC / "brand" / "marvis-face-base.png").is_file()
    assert (STATIC / "brand" / "marvis-face-eyes.png").is_file()
    keyframes = re.findall(r"@keyframes ([\w-]+)", css)
    assert keyframes and all(name.startswith("agent-face-") for name in keyframes), keyframes
    for face in ("busy", "wait", "done", "fail"):
        assert f'.agent-face[data-face="{face}"]' in css
    reduced = re.sub(r"/\*.*?\*/", "", css[css.index("@media (prefers-reduced-motion: reduce)"):], flags=re.S)
    # Scoped to the avatar (no global animation kill); waiting keeps wide eyes.
    assert "*" not in reduced
    assert ".agent-face[data-face]::after" in reduced
    assert re.search(r'\.agent-face\[data-face="wait"\]::after\s*\{\s*transform: scale\(1\.12\);', reduced)
    # 2026-10-19 user feedback: expressions change the face only; the head
    # element itself never moves (no nod, tilt, hop or shake).
    head_rules = re.findall(r'\.agent-face\[data-face="\w+"\]\s*\{([^}]*)\}', css)
    assert not any("animation" in rule or "transform" in rule for rule in head_rules), head_rules
    assert "!important" not in css
    conversation_css = (STATIC / "css" / "agent-conversation.css").read_text(encoding="utf-8")
    meta = conversation_css[conversation_css.index(".agent-message-meta {"):]
    meta = meta[: meta.index("}")]
    assert "uppercase" not in meta
    assert ".agent-message-meta::before" not in conversation_css



def test_validation_rail_marks_the_step_waiting_on_the_user():
    # Batch 4 review B3: workflowStepStatus must be able to return "review" so
    # the validation rail shows the 等你 face on the step the user starts next.
    import json
    import subprocess

    app_js = (STATIC / "app.js").read_text(encoding="utf-8")

    def slice_fn(name):
        start = app_js.index(f"function {name}(")
        end = app_js.index("\n}\n", start) + 3
        return app_js[start:end]

    cases = {
        "scanned": ["succeeded", "review", "pending", "pending"],
        "executed": ["succeeded", "succeeded", "review", "pending"],
        "writing_artifacts": ["succeeded", "succeeded", "succeeded", "review"],
        "review_required": ["succeeded", "succeeded", "succeeded", "succeeded"],
    }
    results = {}
    for status in cases:
        script = "\n".join([
            "const taskSession = { taskId: 'task-1' };",
            f"const task = {{ id: 'task-1', task_type: 'validation', status: '{status}' }};",
            "const workflowSteps = [{id:'scan'}, {id:'notebook'}, {id:'metrics'}, {id:'report'}];",
            "function taskRunningStepId() { return ''; }",
            "function taskFailureWasRestartReclaim() { return false; }",
            "function taskFailureStepId() { return ''; }",
            "function taskStopped() { return false; }",
            "function taskServerBusyAction() { return null; }",
            "function taskResolvedStatus(t) { return { key: ({ scanned: 'awaiting_next', executed: 'awaiting_next', writing_artifacts: 'awaiting_next', review_required: 'needs_review' })[t.status] }; }",
            slice_fn("workflowStepStatus"),
            "process.stdout.write(JSON.stringify(workflowSteps.map((_, i) => workflowStepStatus(i, -1, task))));",
        ])
        out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
        results[status] = json.loads(out.stdout)
    assert results == cases
