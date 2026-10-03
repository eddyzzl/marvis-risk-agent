from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HARNESS = """
import assert from 'node:assert/strict';
import { createModelMonitoringPanel } from './marvis/static/js/v2/model_monitoring_panel.js';
function element() {
  const events = {};
  return { hidden: false, disabled: false, value: '', open: false, files: [], innerHTML: '',
    addEventListener: (name, fn) => { events[name] = fn; }, events };
}
const names = ['Panel','Form','Model','Dataset','Target','File','Refresh','Submit','Status','Limits'];
const elements = Object.fromEntries(names.map(name => ['modelMonitoring'+name, element()]));
const el = name => elements['modelMonitoring'+name];
let task = {id: 'task-a', task_type: 'modeling'}, dirty = false;
const hash = 'a'.repeat(64), calls = [];
let state = {revision: 3, analysis_generation: 2, active_dataset_id: 'sample', active_dataset_content_hash: hash};
let serverPlans = [], planReadFails = false;
let apiImpl = async (path, options) => {
  calls.push({path, options});
  if (path.endsWith('/plans')) {
    if (planReadFails) throw new Error('plan read unavailable');
    return {plans: serverPlans};
  }
  return options ? {messages: [{content:'overview'}]} : {experiments: [{id:'exp',task_id:'task-a',status:'selected',artifact_id:'model',recipe_id:'lr'}]};
};
let saveImpl = async (id, value, revision) => { calls.push({save: id, value, revision}); return {...value, revision:revision+1, analysis_generation:3}; };
let listImpl = async () => ({datasets: [{id:'sample',task_id:'task-a',content_hash:hash,role:'monitoring.input',source_path:'new.csv',row_count:12,columns:[{name:'label'}]}]});
let delivered = [], lease = {}, uploads = 0;
const panel = createModelMonitoringPanel({
  getElementById: id => elements[id], getSelectedTask: () => task,
  api: (...args) => apiImpl(...args), listDatasets: (...args) => listImpl(...args),
  getDataWorkspace: async () => state, putDataWorkspace: (...args) => saveImpl(...args),
  workspaceController: {getState: () => ({dirty})},
  beginActivity: () => lease,
  uploadDataset: async (id, file, options) => { uploads += 1; calls.push({upload:id,options}); return {datasets: []}; },
  onMessages: (messages, view) => delivered.push({messages, view}),
});
async function ready() { await panel.reload(); el('Model').value='exp'; el('Dataset').value='sample'; }
"""


def run(script):
    subprocess.run(["node", "--input-type=module", "-e", HARNESS + script],
                   cwd=ROOT, check=True, capture_output=True, text=True)


def test_monitoring_panel_sends_only_frozen_typed_intake_and_preserves_gate():
    run("""
await ready();
assert.equal(await panel.submit(), true);
const posts = calls.filter(c => c.options);
assert.equal(posts.length, 1);
assert.equal(posts[0].path, 'api/tasks/task-a/agent/messages');
const body = JSON.parse(posts[0].options.body);
assert.deepEqual(body.model_monitoring_request, {experiment_id:'exp',dataset_id:'sample',expected_content_hash:hash,workspace_revision:3,analysis_generation:2,target_col:null});
assert.equal('ui_action' in body, false);
assert.equal(calls.some(c=>c.save), false);
assert.match(el('Status').textContent, /尚未运行/);
assert.match(el('Limits').textContent, /KS\\/AUC 不可用/);
assert.equal(delivered.length, 1);
""")


def test_monitoring_panel_rebinds_selected_data_with_displayed_revision():
    run("""
state={...state,active_dataset_id:'old',active_dataset_content_hash:'b'.repeat(64)};
await ready(); el('Target').value='label';
assert.equal(await panel.submit(), true);
const saved=calls.find(c=>c.save);
assert.equal(saved.revision, 3);
assert.equal(saved.value.active_dataset_id, 'sample');
assert.deepEqual(saved.value.semantic_mapping, {target_col:null,field_roles:{},business_names:{}});
const body=JSON.parse(calls.find(c=>c.options).options.body);
assert.equal(body.model_monitoring_request.workspace_revision,4);
assert.equal(body.model_monitoring_request.analysis_generation,3);
assert.equal(body.model_monitoring_request.target_col,'label');
""")


def test_stale_workspace_does_not_refresh_or_retry_without_user_action():
    run("""
await ready();
apiImpl=async ()=>{calls.push('rejected');throw Object.assign(new Error('监控数据选择已变化'),{status:409});};
assert.equal(await panel.submit(),false);
assert.equal(el('Submit').disabled,true);
assert.equal(el('Status').textContent,'监控数据选择已变化');
assert.equal(await panel.submit(),false);
assert.equal(calls.filter(c=>c==='rejected').length,1);
""")


def test_dirty_workspace_or_busy_lease_cannot_submit_or_upload():
    run("""
await ready(); dirty=true;
assert.equal(await panel.submit(),false);
assert.match(el('Status').textContent,/未保存/);
dirty=false;lease=null;
assert.equal(await panel.submit(),false);
el('File').files=[{name:'new.csv'}];
assert.equal(await panel.uploadFile(),false);
assert.equal(uploads,0);
assert.equal(calls.filter(c=>c.options).length,0);
""")


def test_late_reply_cannot_change_another_tasks_panel_or_messages():
    run("""
await ready();
let resolve;
apiImpl=()=>new Promise(r=>{resolve=r;});
const pending=panel.submit();
task={id:'task-b',task_type:'validation'};
panel.renderAvailability();
resolve({messages:[{content:'old task'}]});
assert.equal(await pending,false);
assert.equal(delivered.length,0);
assert.equal(el('Panel').hidden,true);
assert.equal(el('Status').textContent,'');
""")


def test_monitor_upload_uses_existing_public_dataset_role():
    run("""
await ready();el('File').files=[{name:'new.csv'}];
await panel.uploadFile();
assert.equal(uploads,1);
assert.deepEqual(calls.find(c=>c.upload).options,{role:'unknown'});
""")


def test_all_task_plans_block_until_authoritative_projection_finishes_them():
    run("""
serverPlans=[{id:'older',task_id:'task-a',status:'validated'},
 {id:'latest',task_id:'task-a',status:'done'},
 {id:'foreign',task_id:'task-b',status:'running'}];
await ready();
assert.equal(el('Submit').disabled,true);
assert.match(el('Status').textContent,/确认并完成，或取消/);
assert.equal(await panel.submit(),false);
await panel.reload();
assert.equal(el('Submit').disabled,true);
panel.acceptPlan({id:'older',task_id:'task-b',status:'done'});
assert.equal(el('Submit').disabled,true);
panel.acceptPlan({id:'older',task_id:'task-a',status:'cancelled'});
assert.equal(el('Submit').disabled,false);
assert.match(el('Status').textContent,/当前计划已结束/);
assert.equal(calls.filter(c=>c.options).length,0);
""")


def test_plan_status_uses_the_backend_terminal_set_and_fails_closed():
    run("""
for (const status of ['draft','validated','confirmed','running','awaiting_confirm','new_status']) {
 serverPlans=[{id:'plan',task_id:'task-a',status}];
 await ready();assert.equal(el('Submit').disabled,true,status);
}
for (const status of ['done','failed','cancelled']) {
 serverPlans=[{id:'plan',task_id:'task-a',status}];
 await ready();assert.equal(el('Submit').disabled,false,status);
}
planReadFails=true;
await panel.reload();
assert.equal(el('Submit').disabled,true);
assert.equal(await panel.submit(),false);
""")


def test_created_plan_blocks_repetition_and_survives_refresh():
    run("""
await ready();
const base=apiImpl;
apiImpl=async(path,options)=>{
 if(options) serverPlans=[{id:'monitor',task_id:'task-a',status:'validated'}];
 return base(path,options);
};
assert.equal(await panel.submit(),true);
assert.equal(el('Submit').disabled,true);
assert.equal(await panel.submit(),false);
await panel.reload();
assert.equal(el('Submit').disabled,true);
assert.match(el('Status').textContent,/取消计划/);
assert.equal(calls.filter(c=>c.options).length,1);
""")


def test_failed_post_creation_read_preserves_success_and_disables_new_requests():
    run("""
await ready();
const base=apiImpl;
apiImpl=async(path,options)=>{
 if(options) planReadFails=true;
 return base(path,options);
};
assert.equal(await panel.submit(),true);
assert.match(el('Status').textContent,/计划已生成/);
assert.match(el('Status').textContent,/暂时无法读取/);
assert.equal(el('Submit').disabled,true);
assert.equal(await panel.submit(),false);
panel.acceptPlan({id:'one',task_id:'task-a',status:'done'});
assert.equal(el('Submit').disabled,true);
assert.equal(calls.filter(c=>c.options).length,1);
""")


def test_conflicting_request_preserves_the_servers_real_recovery_advice():
    run("""
await ready();
const reason='当前任务已有进行中的计划，请完成或取消后再创建模型监控。';
apiImpl=async()=>{throw Object.assign(new Error(reason),{status:409});};
assert.equal(await panel.submit(),false);
assert.equal(el('Status').textContent,reason);
assert.equal(el('Submit').disabled,true);
""")
