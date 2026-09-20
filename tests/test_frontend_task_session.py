"""Real session and plan controller behavior under task/model/request races."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = '''
import assert from "node:assert/strict";
import { createTaskSession } from "./marvis/static/js/task-session.js";
import { createPlanRailController } from "./marvis/static/js/v2/plan_rail_controller.js";
const session = createTaskSession();
const tick = () => new Promise(resolve => setImmediate(resolve));
'''


def run_node(body):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PRELUDE + body], cwd=ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_session_owns_snapshots_and_rejects_wrong_model_or_previous_visit():
    run_node('''
const task = {id:"a", metadata:{title:"original"}};
session.selectTask(task);
task.metadata.title="external";
assert.equal(session.task.metadata.title,"original");
assert.throws(()=>{session.task.metadata.title="bypass"},TypeError);
assert.throws(()=>{session.taskId="bypass"},TypeError);
const a = session.requests.capture();
session.replaceMessages(a,[{id:"a-message",metadata:{key:1}}]);
assert.throws(()=>session.messages[0].metadata.key=2,TypeError);
session.selectModel("m1");
assert.deepEqual(session.messages,[]);
assert.equal(session.replaceMessages(a,[{id:"old-parent"}],"a"),false);
const m1 = session.requests.capture();
assert.equal(session.projectModel(m1,{id:"m2"}),false);
assert.equal(session.projectModel(m1,{id:"m1"}),true);
session.selectModel("m2");session.selectModel("m1");
assert.equal(session.projectModel(m1,{id:"m1",stale:true}),false);
assert.equal(session.replaceMessages(m1,[{id:"stale"}]),false);
session.selectTask({id:"b"});session.selectTask({id:"a"});
assert.equal(session.replaceMessages(a,[{id:"stale"}]),false);
assert.equal(session.refreshTask({id:"b"}),false);
assert.equal(session.model,null);
''')


def test_session_keeps_plan_revision_and_agent_request_release_identity():
    run_node('''
session.selectTask({id:"a"});
const view = session.requests.capture();
const plan = {id:"p",task_id:"a",replan_count:2,reconciliation:{continuation:{expected_plan_fingerprint:"fp"}}};
assert.equal(session.acceptPlan(view,"a",plan),true);
plan.reconciliation.continuation.expected_plan_fingerprint="tampered";
assert.equal(session.planIdentity.fingerprint,"fp");
assert.equal(session.plan.reconciliation.continuation.expected_plan_fingerprint,"fp");
assert.equal(session.acceptPlan(view,"a",{...plan,replan_count:1}),false);
assert.equal(session.acceptPlan(view,"a",{...plan,task_id:"b"}),false);
const old = session.beginAgentRequest("a");
const next = session.beginAgentRequest("a");
assert.equal(old.signal.aborted,true);
session.finishAgentRequest(old);session.abortAgentRequest("a");
assert.equal(next.signal.aborted,true);
const fresh = session.beginAgentRequest("a");
session.finishAgentRequest(fresh);session.abortAgentRequest("a");
assert.equal(fresh.signal.aborted,false);
session.selectTask({id:"b"});
assert.equal(session.planIdentity,null);assert.equal(session.plan,null);
''')


PLAN_HARNESS = '''
session.selectTask({id:"a",task_type:"modeling"});
const calls=[], statuses=[], timers=[], projections=[];
const elements=new Map();
const element=()=>({innerHTML:"",dataset:{},classList:{add(){},remove(){}},setAttribute(){}});
for(const id of ["progressRail","workflowStepper","planRetryPanel","planDriverActions"]) elements.set(id,element());
globalThis.document={querySelector:()=>null};
globalThis.window={setTimeout:fn=>timers.push(fn)};
const controller=createPlanRailController({
 $:id=>elements.get(id),getSelectedTask:()=>session.task,getSelectedTaskId:()=>session.taskId,
 getAgentMessages:()=>session.messages,isAgentMode:()=>true,
 requestOwner:session.requests,captureView:()=>session.requests.capture(),isCurrentView:session.requests.current,
 onPlanProjection:(view,id,plan)=>{const accepted=session.acceptPlan(view,id,plan);if(accepted)projections.push(plan);return accepted;},
 beginActivity:(operation,id)=>session.activities.claim(id,"driver_execute",operation),endActivity:session.activities.release,
 setActionStatus:(...args)=>statuses.push(args),listPluginToolsClient:async()=>({tools:[]}),
 apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject})),
});
const plan=(revision=0)=>({id:"p",task_id:"a",replan_count:revision,status:"failed",steps:[],
 failure_envelope:{retryable:false},reconciliation:{targets:[{id:"target",supported:true,action:"reconcile"}]}});
const seed=controller.maybeFetchPlan();calls.at(-1).resolve({plans:[plan()]});await seed;
const clickReconcile=()=>controller.handleClick({
 target:{closest:selector=>selector==="[data-plan-reconcile]"?{dataset:{planReconcile:"target"},disabled:false}:null},
 preventDefault(){},stopPropagation(){},
});
'''


def test_plan_get_cannot_survive_a_b_a_or_throttle_the_new_visit():
    run_node(PLAN_HARNESS + '''
controller.resetFetchThrottle();
const previous=controller.maybeFetchPlan();const old=calls.at(-1);
session.selectTask({id:"b",task_type:"modeling"});session.selectTask({id:"a",task_type:"modeling"});
assert.equal(controller.statusSnapshot("a"),null);
const fresh=controller.maybeFetchPlan();const current=calls.at(-1);
assert.notEqual(current,old);assert.equal(old.options.signal.aborted,true);
current.resolve({plans:[plan(3)]});await fresh;
old.resolve({plans:[plan(1)]});await previous;
assert.equal(session.planIdentity.revision,3);
assert.equal(controller.planId("a"),"p");
''')


def test_reconcile_fences_reads_before_and_during_mutation_and_deduplicates():
    run_node(PLAN_HARNESS + '''
controller.resetFetchThrottle("a",{invalidate:false});
const early=controller.maybeFetchPlan();const earlyRead=calls.at(-1);
assert.equal(clickReconcile(),true);const mutation=calls.at(-1);
assert.match(mutation.url,/reconcile$/);
assert.equal(earlyRead.options.signal.aborted,true);
clickReconcile();assert.equal(calls.at(-1),mutation);
const mid=controller.maybeFetchPlan();const midRead=calls.at(-1);
mutation.resolve({plan:{...plan(2),status:"done",failure_envelope:null,reconciliation:{targets:[]}}});await tick();
assert.equal(midRead.options.signal.aborted,true);
midRead.resolve({plans:[plan(1)]});await mid;
earlyRead.resolve({plans:[plan(0)]});await early;
assert.equal(session.planIdentity.revision,2);assert.equal(session.plan.status,"done");
assert.equal(controller.statusSnapshot("a").label,"已完成");
assert.equal(session.activities.action("a"),null);
''')


def test_retry_timer_and_old_error_cannot_enter_another_visit():
    run_node(PLAN_HARNESS + '''
const pending=controller.retryFetch("a");const first=calls.at(-1);
first.resolve({plans:[{...plan(1),status:"running",failure_envelope:null}]});await pending;
assert.equal(timers.length,1);
session.selectTask({id:"b"});session.selectTask({id:"a"});
const before=calls.length;timers[0]();await tick();assert.equal(calls.length,before);
const errorRead=controller.maybeFetchPlan();const errorCall=calls.at(-1);
session.selectTask({id:"b"});errorCall.reject(new Error("stale"));await errorRead;
assert.equal(session.plan,null);assert.deepEqual(statuses,[]);
''')


def test_portfolio_submission_leases_and_late_view_callbacks_are_isolated():
    run_node('''
const {createPortfolioSetupPanel}=await import("./marvis/static/js/v2/portfolio_setup_panel.js");
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:"",addEventListener(){}});return elements.get(id);};
for(const [id,value] of Object.entries({portfolioIdCol:"loan",portfolioSnapshotCol:"date",portfolioBucketCol:"bucket",portfolioBalanceCol:"balance",portfolioSegmentCol:"segment",portfolioLossState:"loss",portfolioLgd:"0.5",portfolioHorizonMonths:"12"}))$(id).value=value;
const requests=[],submitted=[],errors=[];
const panel=createPortfolioSetupPanel({getElementById:$,getSelectedTask:()=>session.task,getAgentMessages:()=>session.messages,
 captureView:()=>session.requests.capture(),isCurrentView:session.requests.current,
 beginActivity:(operation,id)=>session.activities.claim(id,"driver_execute",operation),endActivity:session.activities.release,
 onMessages:(messages,view)=>session.replaceMessages(view,messages),onSubmitted:()=>submitted.push(true),onError:error=>errors.push(error),
 api:()=>new Promise((resolve,reject)=>requests.push({resolve,reject})),
});
session.selectTask({id:"a",task_type:"portfolio"});
const old=panel.submit();await panel.submit();assert.equal(requests.length,1);
session.selectTask({id:"b",task_type:"portfolio"});panel.renderAvailability();
const fresh=panel.submit();assert.equal(requests.length,2);
requests[0].resolve({messages:[{id:"wrong-task"}]});await old;
assert.deepEqual(session.messages,[]);assert.deepEqual(submitted,[]);
assert.equal(session.activities.action("b"),"driver_execute");
assert.equal($("portfolioSetupSubmit").disabled,true);
requests[1].resolve({messages:[{id:"current-task"}]});await fresh;
assert.equal(session.messages[0].id,"current-task");assert.equal(session.activities.action("b"),null);
const staleError=panel.submit();session.selectTask({id:"a"});session.selectTask({id:"b"});
requests[2].reject(new Error("stale"));await staleError;
assert.deepEqual(errors,[]);assert.deepEqual(session.messages,[]);
''')
