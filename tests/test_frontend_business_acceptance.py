"""Business contracts and verdicts use the real browser modules and session fences."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = r'''
import assert from "node:assert/strict";
import {businessObjectiveError, businessObjectiveFormHtml, collectBusinessObjective, handleBusinessObjectiveEvent} from "./marvis/static/js/business-objective.js";
import {businessAcceptanceProjection, businessAcceptanceHtml, createBusinessAcceptanceController} from "./marvis/static/js/business-acceptance.js";
import {createTaskSession} from "./marvis/static/js/task-session.js";
const objective = {
 schema_version:"business-objective.v1", business_line:"cash", decision_node:"approval",
 population:"all applications", period_start:"2025-01-01", period_end:"2025-03-31",
 target_kind:"strategy", responsibility_source:"review-7", minimum_effect_stage:"oot_validated",
 require_mature_labels:true, claim:"observational", applicable:true, allow_not_applicable:false,
 not_applicable_reason:null, currency:null, target_id:null, target_version:null,
 criteria:[{metric:"approval_rate",unit:"ratio",denominator:"risk/development/rows",minimum:0.3,maximum:null,comparison:"absolute",baseline_ref:null}],
};
const tick = () => new Promise(resolve => setImmediate(resolve));
function form(value) {
 const fields = Object.fromEntries(Object.entries(value).map(([key,value])=>[key,{value:value??"",checked:value===true}]));
 return {
  dataset:{businessVersion:value.schema_version},
  querySelector: selector => selector === "[data-business-enable]" ? {checked:true}
    : fields[selector.match(/data-business-field="(.*?)"/)?.[1]],
  querySelectorAll: () => value.criteria.map(row=>({querySelector:selector=>({value:row[selector.match(/data-business-field="(.*?)"/)[1]]??""})})),
 };
}
'''


def run_node(body):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PRELUDE + body], cwd=ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_contract_roundtrip_preserves_zero_thresholds_baseline_and_not_applicable():
    run_node(r'''
assert.equal(businessObjectiveError(objective),"");
assert.deepEqual(collectBusinessObjective(form(objective)),objective);
const delta=structuredClone(objective);delta.criteria[0]={...delta.criteria[0],minimum:0,comparison:"delta",baseline_ref:"baseline:v3"};
assert.deepEqual(collectBusinessObjective(form(delta)),delta);
const na={...objective,applicable:false,allow_not_applicable:true,not_applicable_reason:"data-only task",criteria:[]};
assert.deepEqual(collectBusinessObjective(form(na)),na);
assert.match(businessObjectiveError({...na,allow_not_applicable:false}),/明确许可/);
assert.match(businessObjectiveError({...objective,period_start:"2025-02-30"}),/有效/);
assert.match(businessObjectiveError({...delta,criteria:[{...delta.criteria[0],baseline_ref:null}]}),/基线/);
assert.match(businessObjectiveError({...objective,criteria:[{...objective.criteria[0],minimum:Infinity}]}),/有限/);
assert.match(businessObjectiveError({...objective,criteria:[{...objective.criteria[0],unit:"currency"}]}),/币种/);
''')


def test_form_does_not_invent_contract_and_preserves_readonly_unsupported_values():
    run_node(r'''
const empty=businessObjectiveFormHtml();
assert.doesNotMatch(empty,/data-business-enable checked/);
assert.match(empty,/data-business-fields disabled/);
const html=businessObjectiveFormHtml({...objective,business_line:'<img src=x onerror="bad()">',schema_version:"future.v2"});
assert.doesNotMatch(html,/<img/);assert.match(html,/&lt;img/);
assert.match(html,/data-business-readonly="true"/);
assert.match(html,/data-business-enable checked disabled/);
assert.throws(()=>collectBusinessObjective(form({...objective,schema_version:"future.v2"})),/版本/);
const fields={disabled:true};const root={dataset:{businessReadonly:"true"},querySelector:()=>fields};
assert.equal(handleBusinessObjectiveEvent({type:"change",target:{closest:()=>root,matches:()=>true,checked:true}}),false);
assert.equal(fields.disabled,true);
''')


@pytest.mark.parametrize("status", ["passed", "failed", "insufficient_evidence", "not_configured", "not_applicable"])
def test_verdict_uses_stored_business_status_independent_of_done(status):
    run_node(r'''
const status="STATUS";
const plan={id:"p",status:"done",execution_completed:false,summary_ref:"summary:1",
 business_acceptance:{schema_version:"business-acceptance.v1",status,objective,target:{kind:"strategy",id:"adopted-b",version:"v2"},
 evidence:{source_ref:"output:7",labels_mature:false,label_origin:"observed",effect_stage:"backtested"},reasons:["reason"],criteria:[{...objective.criteria[0],value:0.9,status:"passed"}]}};
const view=businessAcceptanceProjection(plan,{});
assert.equal(view.status,status);assert.equal(view.executionCompleted,false);
const html=businessAcceptanceHtml(plan,{});
assert.match(html,new RegExp(`data-business-verdict="${status}"`));
assert.match(html,/adopted-b/);assert.match(html,/未成熟/);assert.match(html,/output:7/);
assert.match(html,/data-business-download="xlsx"/);assert.doesNotMatch(html,/download href/);
assert.equal(businessAcceptanceProjection({...plan,business_acceptance:{...plan.business_acceptance,schema_version:"v2"}},{}).status,"unsupported");
assert.equal(businessAcceptanceProjection({...plan,business_acceptance:{...plan.business_acceptance,status:"success"}},{}).status,"unsupported");
assert.equal(businessAcceptanceProjection({status:"done"},{}).status,"not_configured");
assert.equal(businessAcceptanceProjection({status:"done"},{business_objective:objective}).status,"pending");
const missing=businessAcceptanceHtml({...plan,business_acceptance:{...plan.business_acceptance,status:"insufficient_evidence",criteria:[]}},{});
assert.match(missing,/approval_rate/);assert.match(missing,/≥ 0.3/);
'''.replace("STATUS", status))


CONTROLLER = r'''
const session=createTaskSession();session.selectTask({id:"a",task_type:"strategy",business_objective_locked:false,business_objective:objective});
const editor={hidden:true,innerHTML:"",querySelector:()=>({open:false})}, error={textContent:""};
const panel={contains:()=>true,querySelector:selector=>selector.includes("action-error")?error:selector.includes(" [data-business-objective]")?form(objective):editor};
const requests=[],saved=[],statuses=[],downloads=[];
const controller=createBusinessAcceptanceController({
 getElement:()=>panel,getTask:()=>session.task,getPlan:()=>session.plan,
 captureView:()=>session.requests.capture(),isCurrentView:view=>session.requests.current(view),
 apiClient:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject})),
 onSaved:(task,view)=>{saved.push(task);if(session.requests.current(view))session.refreshTask(task);},
 setActionStatus:(...args)=>statuses.push(args),downloadBlob:(...args)=>downloads.push(args),
 beginActivity:(operation,id)=>session.activities.claim(id,"business_objective",operation),endActivity:session.activities.release,
});
function click(action,dataset={}) {
 const button={disabled:false,dataset,hasAttribute:name=>name===`data-business-${action}`};
 assert.equal(controller.handleClick({target:{closest:()=>button},preventDefault(){}}),true);
 return button;
}
'''


def test_save_deduplicates_and_previous_visit_cannot_update_session_or_status():
    run_node(CONTROLLER + r'''
const first=click("save");click("save");assert.equal(requests.length,1);assert.equal(first.disabled,true);
assert.equal(requests[0].url,"/api/tasks/a/business-objective");
assert.deepEqual(JSON.parse(requests[0].options.body),{business_objective:objective});
session.selectTask({id:"b"});session.selectTask({id:"a",business_objective_locked:false});
requests[0].resolve({task:{id:"a",business_objective:objective}});await tick();
assert.deepEqual(saved,[]);assert.deepEqual(statuses,[]);assert.equal(session.task.business_objective,undefined);
assert.equal(session.activities.action("a"),null);
''')


def test_locked_task_or_plan_never_submits_editor_and_server_rejection_remains_visible():
    run_node(CONTROLLER + r'''
session.refreshTask({...session.task,business_objective_locked:true});click("save");assert.equal(requests.length,0);assert.match(error.textContent,/冻结/);
session.refreshTask({...session.task,business_objective_locked:false});session.acceptPlan(session.requests.capture(),"a",{id:"p",task_id:"a"});click("save");assert.equal(requests.length,0);
session.acceptPlan(session.requests.capture(),"a",null);const button=click("save");
requests[0].reject(new Error("contract locked"));await tick();assert.equal(button.disabled,false);assert.equal(error.textContent,"contract locked");assert.deepEqual(saved,[]);
''')


def test_download_is_bound_to_plan_summary_and_visit_and_uses_authenticated_blob_api():
    run_node(CONTROLLER + r'''
session.acceptPlan(session.requests.capture(),"a",{id:"p",task_id:"a",summary_ref:"s1"});
click("download",{businessDownload:"xlsx",businessPlan:"p",businessSummary:"old"});assert.equal(requests.length,0);
click("download",{businessDownload:"xlsx",businessPlan:"p",businessSummary:"s1"});
assert.equal(requests[0].options.responseType,"blob");assert.match(requests[0].url,/expected_summary_ref=s1$/);
session.acceptPlan(session.requests.capture(),"a",{id:"p",task_id:"a",summary_ref:"s2"});requests[0].resolve("old blob");await tick();assert.deepEqual(downloads,[]);
click("download",{businessDownload:"docx",businessPlan:"p",businessSummary:"s2"});requests[1].resolve("new blob");await tick();assert.deepEqual(downloads,[["new blob","business-acceptance-p.docx"]]);
const {api}=await import("./marvis/static/js/api.js");
globalThis.document={baseURI:"http://localhost/proxy/9010/",body:{dataset:{marvisLocalToken:"fixture-token"}}};
globalThis.fetch=async(url,options)=>{assert.equal(url,"http://localhost/proxy/9010/api/plans/p/business-acceptance/xlsx");assert.equal(options.headers["X-Marvis-Token"],"fixture-token");assert.equal(options.responseType,undefined);return {ok:true,status:200,blob:async()=>"bytes"};};
assert.equal(await api("/api/plans/p/business-acceptance/xlsx",{responseType:"blob"}),"bytes");
''')


def test_strategy_clarification_keeps_frozen_contract_read_only():
    run_node(r'''
const {renderStrategyClarification}=await import("./marvis/static/js/v2/strategy_clarification_controller.js");
const message={metadata:{kind:"clarification",clarification:{code:"strategy_business_inputs_required",business_objective_locked:true,current_input:{objective:"max_approval",business_objective:objective}}}};
const html=renderStrategyClarification(message,{interactive:true});
assert.match(html,/data-strategy-clarification-readonly="false"/);
assert.match(html,/data-business-readonly="true"/);
assert.match(html,/data-business-enable checked disabled/);
assert.match(html,/value="cash"/);
assert.match(html,/data-business-fields disabled/);
''')


def test_task_session_accepts_first_empty_plan_response_without_a_false_fetch_error():
    run_node(r'''
const session=createTaskSession();session.selectTask({id:"new-task",task_type:"strategy"});
const view=session.requests.capture();
assert.equal(session.acceptPlan(view,"new-task",null),true);
assert.equal(session.plan,null);assert.equal(session.planIdentity,null);
assert.equal(session.acceptPlan(view,"new-task",null),true);
''')
