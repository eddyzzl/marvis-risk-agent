"""Collection UI keeps proposals, effect authority and observed money distinct."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PRELUDE=r'''
import assert from 'node:assert/strict';
import {batchPlanPayload,createCollectionWorkspaceController} from './marvis/static/js/collection-workspace-controller.js';
import {batchHtml,gateHtml,money} from './marvis/static/js/collection-workspace-view.js';
'''

def node(code):
    result=subprocess.run(['node','--input-type=module','-e',PRELUDE+code],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_plan_slots_are_exact_frozen_batch_and_each_operation_checks_state():
    node(r'''
const b={batch_id:'b',status:'proposed',request_hash:'r',preview_hash:'p'};
assert.deepEqual(batchPlanPayload(b,'queue'),{goal:'催收参考排队',slots:{collection_batch_id:'b',collection_request_hash:'r',collection_preview_hash:'p'}});
assert.throws(()=>batchPlanPayload(b,'execute'));
assert.equal(batchPlanPayload({...b,status:'queued'},'execute').goal,'催收参考执行');
assert.equal(batchPlanPayload(b,'cancel').goal,'取消催收参考批次');
for(const status of ['completed','cancelled'])for(const op of ['queue','execute','cancel'])assert.throws(()=>batchPlanPayload({...b,status},op));
''')


def test_reference_costs_and_unknown_actual_cost_never_become_observed_money():
    node(r'''
const unit={currency:'CNY',minor_unit_exponent:2};assert.equal(money(null,unit),'未知');assert.equal(money(0,unit),'CNY 0.00');assert.equal(money(123,unit),'CNY 1.23');
const b={batch_id:'b',status:'queued',request:{},preview:{unit,as_of:'now',knowledge_cutoff:'later',knowledge_mode:'retrospective_declared',estimated_contact_cost_minor:5,results:[]},actions:[{case_id:'case',queue_id:'q',state:'completed',estimated_cost_minor:5,actual_cost_minor:null,customer_contacted:false}],effects:[]};
const html=batchHtml(b,'maker');assert.match(html,/事后声明/);assert.match(html,/未知/);assert.match(html,/未联系/);assert.match(html,/data-collection-action="execute"/);assert.doesNotMatch(batchHtml(b,'checker'),/data-collection-action="execute"/);
assert.equal(gateHtml({steps:[{status:'awaiting_confirm',tool_ref:{plugin:'decision_twin'}}]}),'');
''')


HARNESS=r'''
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let visit=1,task={id:'A',task_type:'strategy'},plan=null;const els=new Map();
const element=()=>({innerHTML:'',textContent:'',querySelectorAll:()=>[]});
const panel={hidden:false,innerHTML:'',contains:()=>true,querySelectorAll:()=>[],querySelector:s=>{if(!els.has(s))els.set(s,element());return els.get(s)}};
const calls=[];const c=createCollectionWorkspaceController({getElement:()=>panel,getTask:()=>task,getPlan:()=>plan,captureView:()=>visit,isCurrentView:v=>v===visit,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject}))});
const click=(action,id)=>c.handle({type:'click',target:{closest:()=>({dataset:{collectionAction:action,collectionId:id}})},preventDefault(){}});
'''


def test_a_b_a_late_identity_never_replaces_current_principal():
    node(HARNESS+r'''
c.render();const old=c.load(),stale=calls.at(-1);task={id:'B',task_type:'strategy'};visit++;c.render();task={id:'A',task_type:'strategy'};visit++;c.render();
const fresh=c.load();calls.at(-1).resolve({role:'checker',display_name:'Current'});await tick();
for(const call of calls.slice(2))call.resolve(call.url.endsWith('cases')?{cases:[]}:call.url.endsWith('batches')?{batches:[]}:{});await fresh;
stale.resolve({role:'maker',display_name:'Old'});await old;assert.match(panel.querySelector('[data-collection-identity]').textContent,/Current/);assert.equal(calls.length,5);
''')


def test_late_batch_read_and_cross_task_evidence_cannot_replace_new_selection():
    node(HARNESS+r'''
c.render();click('case','old');const old=calls.at(-1);click('case','new');calls.at(-1).resolve({case_id:'new'});await tick();old.resolve({case_id:'old'});await tick();assert.match(panel.querySelector('[data-collection-detail]').innerHTML,/new/);assert.doesNotMatch(panel.querySelector('[data-collection-detail]').innerHTML,/old/);
const count=calls.length;click('evidence','/api/tasks/B/collection/batches/b/effects/e');assert.equal(calls.length,count);assert.match(panel.querySelector('[data-collection-message]').textContent,/不属于当前任务/);
''')
