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


INTAKE=r'''
import {collectIntake,intakeHtml} from './marvis/static/js/collection-intake-form.js';
import {reconciliationHtml} from './marvis/static/js/collection-intake-controller.js';
const refs={source_artifact_id:'native-source',source_artifact_hash:'a'.repeat(64)};
const unit={currency:'CNY',minor_unit_exponent:2,definition_source:'explicit'};
const record={case_id:'c',subject_namespace:'bank',subject_token:'b'.repeat(64),unit};
'''


def test_cashflow_intake_uses_exact_minor_units_and_unknown_availability_and_reversals():
    node(INTAKE+r'''
const v={source_id:'bank',event_id:'e',kind:'payment',amount_minor:'100',event_at:'2026-08-01T12:00:00+08:00',available_at:''};
const result=collectIntake('cashflow',v,{record,refs}).events[0];assert.equal(result.amount_minor,100);assert.equal(result.available_at,null);assert.deepEqual(result.unit,unit);assert.equal(result.source_assurance,'historical_import_unverified');
assert.throws(()=>collectIntake('cashflow',{...v,amount_minor:'1.2'},{record,refs}),/整数/);
assert.throws(()=>collectIntake('cashflow',{...v,event_at:'2026-08-01T12:00:00'},{record,refs}),/明确时区/);
assert.throws(()=>collectIntake('cashflow',{...v,kind:'payment_reversal'},{record,refs}),/原资金来源/);
const reversed=collectIntake('cashflow',{...v,kind:'payment_reversal',reversal_source:'bank',reversal_event:'e'},{record,refs}).events[0];assert.deepEqual(reversed.reverses,{source_id:'bank',event_id:'e'});
''')


def test_reconciliation_never_fills_missing_coverage_or_maturity_with_positive_claims():
    node(INTAKE+r'''
const v={as_of:'2026-08-30T00:00:00Z',knowledge_cutoff:'2026-09-01T00:00:00Z',expected_sources:'bank,cost'};
const r=collectIntake('reconcile',v,{record,refs});assert.deepEqual(r.coverage,[]);assert.equal(Object.hasOwn(r,'maturity'),false);
const html=reconciliationHtml({case_id:'c',status:'insufficient_evidence',unit,amounts:null,observed_subtotals:{net_payments_minor:10},maturity:{status:'not_declared'}});assert.match(html,/净回款：未知/);assert.doesNotMatch(html,/净回款：CNY 0.10/);
''')


def test_business_batch_declares_inputs_and_does_not_compute_policy_hash_or_strategy_in_browser():
    node(INTAKE+r'''
const v={batch_id:'batch',policy_id:'p',revision:'1',valid_from:'2026-09-01T00:00:00Z',valid_until:'2026-10-01T00:00:00Z',timezone:'Asia/Shanghai',queue_id:'q',action_kind:'contact',channel:'sms',priority:'10',estimated_cost_minor:'0',max_batch_actions:'10',max_active_actions:'20',frequency_window_seconds:'86400',max_contacts_per_subject_window:'2',min_contact_interval_seconds:'0',max_estimated_batch_cost_minor:'100',max_estimated_active_cost_minor:'200',as_of:'2026-09-28T04:00:00Z',knowledge_cutoff:'2026-09-28T04:00:00Z'};
const context={refs,windows:[{weekdays:'1,2,3,4,5',start:'09:00',end:'18:00'}],cases:[{record,values:{contact_permission:'unknown',history_coverage:'unknown'},attempts:[]}]};
const batch=collectIntake('batch',v,context);assert.equal(Object.hasOwn(batch,'strategy'),false);assert.equal(Object.hasOwn(batch.action,'policy_hash'),false);assert.equal(batch.action.estimated_cost_minor,0);assert.deepEqual(batch.histories,[]);assert.equal(batch.cases[0].contact_permission,'unknown');
assert.throws(()=>collectIntake('batch',{...v,estimated_cost_minor:''},context),/预计单位成本/);
const hold=collectIntake('batch',{...v,action_kind:'hold',estimated_cost_minor:'',priority:'',channel:''},context);assert.deepEqual(hold.action,{kind:'hold',queue_id:null,priority:0,channel:null,estimated_cost_minor:null});
assert.match(intakeHtml('batch',[record]),/政策队列/);
''')


INTAKE_ASYNC = r'''
import {createCollectionIntakeController} from './marvis/static/js/collection-intake-controller.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const elements=new Map();const root={querySelector:s=>{if(!elements.has(s))elements.set(s,{innerHTML:'',textContent:''});return elements.get(s)}};
let owner={taskId:'A',principal:{id:'maker-A',role:'maker'},cases:[],pending:false};let messages=[],calls=[],refreshes=0;
const intake=createCollectionIntakeController({getRoot:()=>root,getOwner:()=>owner,isCurrent:o=>o===owner,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject})),message:s=>messages.push(s),setPending:(o,p)=>o.pending=p,refresh:async()=>{refreshes++},onBatch:async()=>{},downloadBlob:()=>{}});
const formOf=values=>({querySelectorAll:s=>s==='[name]'?Object.entries(values).map(([name,value])=>({name,value})):[]});
const submit=form=>intake.handle({type:'submit',target:{closest:()=>form},preventDefault(){}});
const caseValues={case_id:'case-1',subject_namespace:'bank',subject_token:'a'.repeat(64),currency:'CNY',minor_unit_exponent:'2',definition_source:'Explicit source',opening_balance_minor:'10000',opened_at:'2026-08-01T00:00:00Z',source_description:'declared'};
'''


def test_late_material_cannot_create_a_case_after_task_visit_changes():
    node(INTAKE_ASYNC + r'''
intake.open('case');const old=owner;submit(formOf(caseValues));await tick();assert.equal(calls.length,1);assert.match(calls[0].url,/tasks\/A\/collection\/materials$/);
owner={taskId:'B',principal:{id:'maker-B',role:'maker'},cases:[],pending:false};intake.open('case');const currentHtml=root.querySelector('[data-collection-editor]').innerHTML;const count=messages.length;
calls[0].resolve({source_artifact_id:'old',source_artifact_hash:'a'.repeat(64)});await tick();assert.equal(calls.length,1);assert.equal(refreshes,0);assert.equal(messages.length,count);assert.equal(root.querySelector('[data-collection-editor]').innerHTML,currentHtml);assert.equal(owner.pending,false);assert.equal(old.pending,false);
''')


def test_late_case_read_cannot_create_material_after_principal_changes():
    node(INTAKE_ASYNC + r'''
intake.open('cashflow');submit(formOf({case_id:'case-1',source_description:'source'}));await tick();assert.equal(calls.length,1);
owner.principal={id:'different-maker',role:'maker'};calls[0].resolve({case_id:'case-1'});await tick();assert.equal(calls.length,1);assert.equal(refreshes,0);assert.equal(owner.pending,false);
''')


def test_material_permission_failure_keeps_form_values_and_releases_pending():
    node(INTAKE_ASYNC + r'''
intake.open('case');const form=formOf(caseValues),html=root.querySelector('[data-collection-editor]').innerHTML;submit(form);await tick();calls[0].reject({status:403,detail:{code:'collection_owner_required'}});await tick();assert.match(messages.at(-1),/collection_owner_required/);assert.equal(root.querySelector('[data-collection-editor]').innerHTML,html);assert.equal(form.querySelectorAll('[name]').find(x=>x.name==='opening_balance_minor').value,'10000');assert.equal(owner.pending,false);assert.equal(calls.length,1);assert.equal(refreshes,0);
''')
