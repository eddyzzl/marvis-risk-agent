"""V3 D-6/D-8 conversation cards: render from metadata, confirm only by explicit click."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).parents[1]

SCRIPT = r"""
import assert from 'node:assert/strict';
import { conversationActionCardHtml, handleConversationActionClick } from './marvis/static/js/conversation-action-cards.js';
const proposal = { id: 'm1', role: 'assistant', metadata: { conversation_action: {
  kind: 'sql_query_proposal', query: 'SELECT "channel" FROM "loans" LIMIT 100000', source_name: 'biz', max_rows: 100000 } } };
const result = { id: 'm2', role: 'assistant', metadata: { conversation_action: {
  kind: 'sql_query_result', source_message_id: 'm1', dataset: { row_count: 2, column_count: 1, column_names: ['channel'] } } } };
const open = conversationActionCardHtml(proposal, [proposal]);
assert.match(open, /确认执行/); assert.doesNotMatch(open, /disabled/);
const settled = conversationActionCardHtml(proposal, [proposal, result]);
assert.match(settled, /已执行/); assert.match(settled, /disabled/);
assert.match(conversationActionCardHtml(result, [proposal, result]), /取数结果/);
assert.equal(conversationActionCardHtml({ role: 'user', metadata: proposal.metadata }, []), '');
const form = { id: 'f1', role: 'assistant', metadata: { conversation_action: { kind: 'monitoring_schedule_form', missing: ['time_col'],
  form: { target_id: 'exp-1', dataset_id: 'ds', time_col: null, frequency: 'daily', interval: 1, start_date: '2025-04-02', notify: 'local_inbox',
    advanced: { schedule_id: 's', timezone: 'Asia/Shanghai', complete_through: '2025-04-01T00:00:00+00:00', catch_up_budget: 1, max_attempts: 1, lease_seconds: 600 } },
  options: { targets: [{ id: 'exp-1', label: '模型 lgbm' }], datasets: [{ id: 'ds', row_count: 3, column_names: ['applied_at'] }] } } } };
const formHtml = conversationActionCardHtml(form, [form]);
assert.match(formHtml, /<details class="conversation-form-advanced"><summary>高级设置<\/summary>/);
assert.match(formHtml, /data-schedule-field="advanced.lease_seconds"/);
assert.match(formHtml, /data-invalid="true"><span>时间列/);
// A click elsewhere never calls the API; a confirm click posts the explicit flag.
const calls = [];
const ctx = { getSelectedTaskId: () => 't1', api: async (url, init) => { calls.push([url, JSON.parse(init.body)]); return {}; }, refreshAgentMessages: async () => {} };
const card = { querySelector: () => null };
const button = { disabled: false, hasAttribute: (n) => n === 'data-sql-conversation-confirm', getAttribute: () => 'm1', closest: (s) => s.includes('[data-') ? button : card };
assert.equal(await handleConversationActionClick({ target: { closest: () => null } }, ctx), false);
assert.equal(calls.length, 0);
assert.equal(await handleConversationActionClick({ target: button, preventDefault() {} }, ctx), true);
assert.deepEqual(calls, [['/api/tasks/t1/sql/conversation/confirm', { message_id: 'm1', confirmed: true }]]);
console.log('ok');
"""


def test_conversation_action_cards_render_and_confirm_explicitly():
    completed = subprocess.run(["node", "--input-type=module", "-e", SCRIPT], cwd=ROOT,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
