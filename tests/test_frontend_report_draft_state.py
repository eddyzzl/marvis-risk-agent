import json
from pathlib import Path
import subprocess


def test_cross_model_saves_inflight_edits_conflicts_and_retry():
    module = (Path(__file__).resolve().parents[1] / 'marvis/static/js/report-draft-state.js').as_uri()
    script = r'''
import assert from 'node:assert/strict';
const { createReportDraftState } = await import(MODULE);
const message = (id, revision = 0, value = 'server') => ({ id, metadata: {
  report_revision: 0, draft_edit_revision: revision, draft_values: { 'TEXT:final_validation_conclusion': value },
}});
const requests = [];
let release;
let revision = 0;
const state = createReportDraftState({ delay: 100000, api: async (url, options) => {
  const body = JSON.parse(options.body);
  requests.push([url, body]);
  if (requests.length === 1) await new Promise((r) => { release = r; });
  return { message: message(body.draft_message_id, ++revision) };
}});
state.receive('a', message('draft-a'));
state.edit('a', { 'TEXT:final_validation_conclusion': 'edited a' });
state.receive('b', message('draft-b'));
state.edit('b', { 'TEXT:final_validation_conclusion': 'edited b' });
assert.equal(state.receive('a', message('draft-a')).values['TEXT:final_validation_conclusion'], 'edited a');
const firstSave = state.save('a');
state.edit('a', { 'TEXT:final_validation_conclusion': 'typed during save' });
release();
await firstSave;
await state.flush(['a', 'b']);
assert.equal(requests.length, 3);
assert.equal(requests[1][1].text_values['TEXT:final_validation_conclusion'], 'typed during save');
assert.equal(state.payload('b').text_values['TEXT:final_validation_conclusion'], 'edited b');
assert.equal(state.hasUnsaved(), false);
state.edit('a', { 'TEXT:final_validation_conclusion': 'local conflict' });
state.receive('a', message('draft-a', 99, 'other window'));
await assert.rejects(state.save('a'), /冲突/);
assert.equal(state.get('a').values['TEXT:final_validation_conclusion'], 'local conflict');
state.resolve('a', 'local');
assert.equal(state.payload('a').draft_edit_revision, 99);
assert.equal(state.payload('a').text_values['TEXT:final_validation_conclusion'], 'local conflict');
state.discard('a'); state.discard('b');
let fail = true;
const retry = createReportDraftState({ delay: 100000, api: async () => {
  if (fail) throw new Error('offline');
  return { message: message('r', 1) };
}});
retry.receive('r', message('r'));
retry.edit('r', {'TEXT:final_validation_conclusion': 'keep me'});
await assert.rejects(retry.save('r'), /offline/);
assert.equal(retry.hasUnsaved(), true);
assert.equal(retry.payload('r').text_values['TEXT:final_validation_conclusion'], 'keep me');
fail = false; await retry.save('r');
assert.equal(retry.hasUnsaved(), false);
retry.discard('r');
'''.replace('MODULE', json.dumps(module))
    result = subprocess.run(['node', '--input-type=module', '-e', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
