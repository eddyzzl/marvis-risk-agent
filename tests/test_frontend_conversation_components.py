"""Conversation components (2026-10-14 redesign, batch 3a): quick replies, latest report draft, result marking."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"


def node(body):
    prelude = r'''
import assert from "node:assert/strict";
import { agentQuickReplies, agentQuickRepliesHtml } from "./marvis/static/js/agent-quick-replies.js";
import { latestReportDraftMessageId, latestPendingReportDraftMessageId } from "./marvis/static/js/report-draft-table.js";
import { agentMessagesHtml } from "./marvis/static/js/agent-conversation-view.js";
'''
    result = subprocess.run(["node", "--input-type=module", "-e", prelude + body], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_quick_replies_come_from_structured_metadata_of_the_latest_agent_turn():
    node(r'''
const failure = {
  id: "f1", role: "assistant", content: "❌ 失败 strategy not found",
  metadata: { error_diagnostic: { recovery_actions: [
    { label: "由 Agent 重试当前步骤", command: "重试当前步骤" },
    { label: "", command: "" },
    { label: "重复", command: "重试当前步骤" },
  ] } },
};
assert.deepEqual(agentQuickReplies([failure]), [
  { label: "由 Agent 重试当前步骤", command: "重试当前步骤", kind: "recovery" },
]);
// Backend-declared replies come first and keep their exact command.
assert.deepEqual(agentQuickReplies([{ role: "assistant", metadata: { quick_replies: [
  { label: "都按识别结果确认", command: "都按这个" },
  { label: "确认将空标签仅从风险分母排除并继续", command: "确认将空标签仅从风险分母排除并继续" },
], awaiting_confirmation: true } }]), [
  { label: "都按识别结果确认", command: "都按这个", kind: "reply" },
  { label: "确认将空标签仅从风险分母排除并继续", command: "确认将空标签仅从风险分母排除并继续", kind: "reply" },
]);
// V1 hand-offs stored before quick_replies existed still get 继续; a bare
// awaiting_confirmation flag is not guessed into a command.
assert.deepEqual(agentQuickReplies([{ role: "assistant", metadata: { awaiting_next_stage: "metrics" } }]).map((r) => r.command), ["继续"]);
assert.deepEqual(agentQuickReplies([{ role: "assistant", metadata: { awaiting_confirmation: true } }]), []);
assert.deepEqual(agentQuickReplies([{ role: "assistant", metadata: { awaiting_next_stage: "metrics", quick_replies: [{ label: "继续", command: "继续" }] } }]).map((r) => r.kind), ["reply"]);
// Free text is never parsed: an instruction inside the body yields nothing.
assert.deepEqual(agentQuickReplies([{ role: "assistant", content: "请明确回复“继续”", metadata: {} }]), []);
// A newer user turn, a streaming reply or an empty timeline yields nothing.
assert.deepEqual(agentQuickReplies([failure, { role: "user", content: "重试当前步骤" }]), []);
assert.deepEqual(agentQuickReplies([{ ...failure, metadata: { ...failure.metadata, streaming: true } }]), []);
assert.deepEqual(agentQuickReplies([]), []);
// Hidden ui_action echoes do not count as the latest turn.
assert.equal(agentQuickReplies([failure, { role: "user", content: "确认", metadata: { display_in_timeline: false } }]).length, 1);
const html = agentQuickRepliesHtml([{ label: "<b>重试</b>", command: "重试\"当前\"", kind: "recovery" }]);
assert.ok(html.includes('data-agent-quick-reply="重试&quot;当前&quot;"'));
assert.ok(html.includes("&lt;b&gt;重试&lt;/b&gt;"));
assert.ok(html.includes('class="agent-quick-reply ui-glass"'));
''')


def test_only_the_newest_report_draft_is_marked_latest():
    node(r'''
const draft = (id) => ({ id, role: "assistant", stage: "word_conclusion_draft", metadata: { draft_values: { a: "1" } } });
const messages = [draft("d1"), { id: "u", role: "user", content: "改一下" }, draft("d2"), { id: "c", role: "assistant", stage: "word_conclusion_confirmed" }];
assert.equal(latestReportDraftMessageId(messages), "d2");
// Pending ignores confirmed drafts; latest does not.
assert.equal(latestPendingReportDraftMessageId(messages), "");
assert.equal(latestReportDraftMessageId([{ role: "assistant", stage: "word_conclusion_draft", metadata: {} }]), "");
const seen = [];
agentMessagesHtml(messages, undefined, {
  latestReportDraftMessageId: "d2",
  agentMessageHtml: (message, _stage, options) => { seen.push([message.id, options.isLatestReportDraft]); return ""; },
});
assert.deepEqual(seen.filter(([id]) => id === "d1" || id === "d2"), [["d1", false], ["d2", true]]);
// Callers that do not resolve the latest draft keep the single-draft behaviour.
const legacy = [];
agentMessagesHtml([draft("d1")], undefined, { agentMessageHtml: (m, s, o) => { legacy.push(o.isLatestReportDraft); return ""; } });
assert.deepEqual(legacy, [undefined]);
''')


def test_app_wires_quick_replies_superseded_drafts_and_result_marking():
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    composer = html[html.index('<div id="agentComposer"'):html.index('<div class="agent-composer-bar">')]
    assert '<div id="agentQuickReplies" class="agent-quick-replies" role="group" aria-label="快捷回复" hidden></div>' in composer
    assert 'id="threadFilter" class="task-thread-filter" role="radiogroup"' in html


def test_conversation_styles_keep_reading_width_and_flat_file_cards():
    css = (STATIC / 'css' / 'conversation-components.css').read_text(encoding='utf-8')
    assert '--thread-max-width: 980px;' in css
    assert 'left: var(--thread-inline-inset);' in css
    assert '.result-dataset-download {\n  border: 0;\n  background: var(--card);\n}' in css
    assert '#resultWorkspace[data-thread-filter="results"] .agent-message:not([data-agent-result="true"])' in css


def test_activity_card_lists_current_phase_with_recorded_durations():
    script = r'''
import assert from "node:assert/strict";
import { planActivityCardHtml, formatStepDuration } from "./marvis/static/js/v2/plan_activity_card.js";
assert.equal(formatStepDuration(800), "不到 1 秒");
assert.equal(formatStepDuration(1234), "1.2 秒");
assert.equal(formatStepDuration(42000), "42 秒");
assert.equal(formatStepDuration(185000), "3 分 5 秒");
assert.equal(formatStepDuration(-1), "");
const now = Date.parse("2026-10-15T08:00:43Z");
const plan = { steps: [
  { id: "a", title: "读取样本", phase: "数据准备", status: "done", last_run_duration_ms: 800 },
  { id: "b", title: "切分样本", phase: "特征", status: "done", last_run_duration_ms: 2100 },
  { id: "c", title: "筛选<特征>", phase: "特征", status: "running", started_at: "2026-10-15T08:00:00Z" },
  { id: "d", title: "精选特征", phase: "特征", status: "pending" },
] };
const html = planActivityCardHtml(plan, { now });
assert.ok(html.includes("正在执行「特征」"));
assert.ok(!html.includes("读取样本"), "only the running phase is listed");
assert.ok(html.includes('data-activity-state="done"') && html.includes("2.1 秒"));
// Batch 4 (2026-10-14 redesign): the running row uses the canonical 执行中.
assert.ok(html.includes("执行中 · 已用 43 秒"));
assert.ok(html.includes("筛选&lt;特征&gt;"));
assert.ok(html.includes('data-activity-state="pending"'));
// Nothing running: no activity card (the gate or report card speaks instead).
assert.equal(planActivityCardHtml({ steps: plan.steps.map((s) => ({ ...s, status: "done" })) }), "");
// A duration the platform did not record is never invented.
const unknown = planActivityCardHtml({ steps: [
  { id: "x", title: "旧步骤", phase: "P", status: "done" },
  { id: "y", title: "运行中", phase: "P", status: "checking" },
] }, { now });
assert.ok(unknown.includes('<span class="plan-activity-title">旧步骤</span></div>'));
// Batch 4 (2026-10-14 redesign): checking reads 执行中, the canonical running word.
assert.ok(unknown.includes(">执行中<"));
'''
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
