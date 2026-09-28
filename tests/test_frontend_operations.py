"""Operation forms keep source identity, label semantics and revisions explicit."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = r'''
import assert from "node:assert/strict";
import {operationsPayload,operationsFormHtml} from "./marvis/static/js/operations-form.js";
import {inboxHtml,scheduleHtml,periodHtml,evidenceHtml,runtimeHtml} from "./marvis/static/js/operations-view.js";
const data = {id:"d",task_id:"a",content_hash:"a".repeat(64),columns:[{name:"time"},{name:"y"},{name:"score"}],row_count:100};
const ctx = {datasets:[data],targets:[{id:"m"}]};
const v = {schedule_id:"monthly",monitoring_ref:"modeling.monitor_run",task_id:"a",dataset_id:"d",target_id:"m",time_col:"time",timezone:"Asia/Shanghai",complete_through:"2026-08-31T16:00",label_mode:"required",target_col:"y",label_maturity_seconds:"0",score_col:"",max_source_rows:"100000",anchor_at:"2026-08-01T00:00",interval_seconds:"86400",active_from:"2026-08-01T00:00",active_until:"2026-09-01T00:00",catch_up_budget:"1",lease_seconds:"300",max_attempts:"3",initial_backoff_seconds:"0",multiplier:"2",max_backoff_seconds:"300",enabled:true};
'''


def run_node(script):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", PRELUDE + script],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_publish_uses_registered_hash_explicit_zero_and_utc():
    run_node(r'''
const p=operationsPayload(v,ctx);
assert.equal(p.expected_revision,0);assert.equal(p.schedule.revision,1);
assert.equal(p.schedule.monitoring_binding.dataset_content_hash,data.content_hash);
assert.equal(p.schedule.monitoring_binding.label_maturity_seconds,0);
assert.equal(p.schedule.monitoring_binding.complete_through,"2026-08-31T16:00:00.000Z");
assert.equal(p.schedule.retry_policy.initial_backoff_seconds,0);
const na=operationsPayload({...v,label_mode:"not_applicable",target_col:"y",label_maturity_seconds:"999"},ctx);
assert.equal(na.schedule.monitoring_binding.target_col,null);
assert.equal(na.schedule.monitoring_binding.label_maturity_seconds,null);
''')


@pytest.mark.parametrize("change,expected", [
    ('task_id:"other"', "当前任务"),
    ('target_id:"old-model"', "监控目标"),
    ('time_col:"old-column"', "业务时间列"),
    ('label_mode:""', "标签要求"),
    ('label_maturity_seconds:""', "成熟等待"),
    ('label_maturity_seconds:"NaN"', "成熟等待"),
    ('label_maturity_seconds:"0.5"', "成熟等待"),
    ('complete_through:"2026-02-30T00:00"', "无效"),
    ('timezone:"imaginary/zone"', "IANA"),
    ('active_until:"2026-07-01T00:00"', "晚于"),
    ('max_backoff_seconds:"0",initial_backoff_seconds:"1"', "最长等待"),
])
def test_invalid_or_stale_source_cannot_publish(change, expected):
    run_node(f'assert.throws(()=>operationsPayload({{...v,{change}}},ctx),/{expected}/);')


def test_update_and_recheck_preserve_revision_and_original_boundary():
    run_node(r'''
const initial=operationsPayload(v,ctx).schedule;
const record={contract:{...initial,revision:5,schema_version:"operations.schedule.v1",calendar:{...initial.calendar,schema_version:"calendar.v1"},recheck_of:{schedule_id:"original"}}};
const p=operationsPayload({...v,enabled:false},{...ctx,record});
assert.equal(p.expected_revision,5);assert.equal(p.schedule.revision,6);assert.equal(p.schedule.enabled,false);
assert.equal(p.schedule.schema_version,undefined);assert.equal(p.schedule.calendar.schema_version,undefined);assert.equal(p.schedule.recheck_of,undefined);
const recheck=operationsPayload(v,{...ctx,record,recheck:"period1",idempotencyKey:"dedup1"});
assert.deepEqual(Object.keys(recheck).sort(),["expected_revision","idempotency_key","monitoring_binding","period_key"]);
assert.equal(recheck.expected_revision,5);assert.equal(recheck.idempotency_key,"dedup1");
assert.throws(()=>operationsPayload({...v,task_id:"b"},{datasets:[{...data,task_id:"b"}],targets:ctx.targets,record,recheck:"p"}),/原任务/);
assert.throws(()=>operationsPayload({...v,target_id:"m2"},{...ctx,targets:[{id:"m2"}],record,recheck:"p"}),/原任务/);
assert.throws(()=>operationsPayload({...v,schedule_id:"new-name"},{...ctx,record}),/锁定/);
const fine={...record,contract:{...record.contract,calendar:{...record.contract.calendar,anchor_at:"2026-08-01T00:00:00.123456Z"}}};
assert.equal(operationsPayload({...v,anchor_at:"2026-08-01T00:00:00"},{...ctx,record:fine}).schedule.calendar.anchor_at,"2026-08-01T00:00:00.123456Z");
assert.equal(operationsPayload({...v,anchor_at:"2026-08-01T00:01:00"},{...ctx,record:fine}).schedule.calendar.anchor_at,"2026-08-01T00:01:00.000Z");
''')


def test_receipt_and_evidence_never_turn_delivery_or_missing_metrics_into_success():
    run_node(r'''
const receipt={notification_id:"n",schedule_id:'<script>alert(1)</script>',period_key:"p",payload:{level:"not_available"},delivered_at:"2026-09-01T00:00:00Z",read_at:null};
const html=inboxHtml([receipt]);
assert.match(html,/已投递/);assert.match(html,/未读/);assert.match(html,/无可用结论/);assert.match(html,/data-ops-action="ack"/);assert.doesNotMatch(html,/<script>/);
const read=inboxHtml([{...receipt,read_at:"now",read_by:"maker"}]);
assert.match(read,/已读/);assert.doesNotMatch(read,/data-ops-action="ack"/);
const missing=evidenceHtml({coverage_assurance:"publisher_declared",label_mode:"not_applicable",label_maturity_seconds:null});
assert.match(missing,/未独立认证/);assert.match(missing,/无监控工具输出/);assert.doesNotMatch(missing,/0 风险/);
assert.match(runtimeHtml({runtime:{alive:false}},null),/后台未运行/);
assert.match(runtimeHtml({runtime:{alive:true,last_error_code:"failed"}},null),/后台运行异常/);
const output=evidenceHtml({tool_output:{row_count:0,overall_level:"not_available",checks:[{label:"AUC",value:null,level:"not_available",message:"labels missing"},{label:"PSI",value:0,level:"green"}]}});
assert.match(output,/0 行/);assert.match(output,/无可用值/);assert.match(output,/AUC/);assert.match(output,/>0<\/td>/);
''')


def test_form_does_not_infer_label_policy_or_source_and_exposes_readonly_recheck():
    run_node(r'''
const html=operationsFormHtml({tasks:[{id:"a",model_name:"业务线"}],refs:["modeling.monitor_run"]});
assert.match(html,/业务线/);assert.match(html,/data-ops-publish disabled/);
assert.doesNotMatch(html,/value="required" selected/);
assert.match(html,/name="label_maturity_seconds" type="number" value=""/);
const recheck=operationsFormHtml({record:{contract:operationsPayload(v,ctx).schedule},recheck:"p"});
assert.match(recheck,/name="task_id" required disabled/);assert.match(recheck,/name="target_id" required disabled/);
assert.doesNotMatch(recheck,/name="anchor_at"/);assert.match(recheck,/原始证据不会被覆盖/);
''')


CONTROLLER_HARNESS = r'''
import {createOperationsController} from "./marvis/static/js/operations-controller.js";
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const elements=new Map();
const element=()=>({innerHTML:"",textContent:"",hidden:false,disabled:false,classList:{toggle(){}},setAttribute(){}});
const events=new Map();
const root={open:false,showModal(){this.open=true},querySelector(selector){if(!elements.has(selector))elements.set(selector,element());return elements.get(selector)},querySelectorAll(){return []},addEventListener(name,fn){events.set(name,fn)}};
const openButton={querySelector(){return root.querySelector("badge")},addEventListener(){}};
const calls=[],timers=[];
const controller=createOperationsController({root,openButton,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject})),setIntervalFn:fn=>timers.push(fn),clearIntervalFn(){}});
const resolveView=()=>{
 for(const call of calls.filter(c=>!c.settled && !c.url.endsWith("/me"))){
  call.settled=true;
  call.resolve(call.url.includes("capabilities")?{runtime:{alive:true},monitoring_refs:[]}:call.url.includes("inbox")?{notifications:[]}:{schedules:[]});
 }
};
'''


def test_closed_view_identity_response_cannot_change_reopened_role():
    run_node(CONTROLLER_HARNESS + r'''
const first=controller.open(), stale=calls.at(-1);
root.open=false;
const second=controller.open(), fresh=calls.at(-1);
fresh.resolve({role:"checker",display_name:"current checker"});await tick();resolveView();await second;
assert.equal(root.querySelector('[data-ops-action="new"]').hidden,true);
const count=calls.length;
stale.resolve({role:"maker",display_name:"old maker"});await tick();
assert.equal(calls.length,count,"closed view must not start fresh reads after late identity response");
await first;
assert.match(root.querySelector("[data-ops-runtime]").innerHTML,/current checker/);
assert.equal(root.querySelector('[data-ops-action="new"]').hidden,true);
''')


def test_bootstrap_identity_does_not_replace_newer_opened_session():
    run_node(CONTROLLER_HARNESS + r'''
controller.bind();const bootstrap=calls.at(-1);
const opened=controller.open();calls.at(-1).resolve({role:"checker",display_name:"current checker"});
await tick();resolveView();await opened;
const count=calls.length;
bootstrap.resolve({role:"admin",display_name:"old admin"});await tick();
assert.equal(calls.length,count,"obsolete bootstrap identity must not poll as the old role");
const poll=timers[0]();await tick();resolveView();await tick();resolveView();await poll;
assert.equal(root.querySelector('[data-ops-action="new"]').hidden,true);
assert.match(root.querySelector("[data-ops-runtime]").innerHTML,/current checker/);
''')


POLL_HARNESS = r'''
controller.bind();calls.at(-1).resolve({role:"maker",display_name:"original maker"});
await tick();resolveView();await tick();
const opened=controller.open();calls.at(-1).resolve({role:"maker",display_name:"original maker"});
await tick();resolveView();await opened;
const reopen=async()=>{
 root.open=false;events.get("close")();
 const next=controller.open();calls.at(-1).resolve({role:"checker",display_name:"current checker"});
 await tick();resolveView();await next;
};
'''


def test_poll_unauthorized_response_cannot_clear_reopened_identity():
    run_node(CONTROLLER_HARNESS + POLL_HARNESS + r'''
const oldPoll=timers[0]();const stale=calls.at(-1);stale.settled=true;
await reopen();
stale.reject(Object.assign(new Error("old session expired"),{status:403}));await oldPoll;
assert.equal(root.querySelector("[data-ops-error]").textContent,"");
const count=calls.length;
const freshPoll=timers[0]();assert.equal(calls.length,count+1,"new identity must still poll after obsolete 403");
await tick();resolveView();await tick();resolveView();await freshPoll;
assert.match(root.querySelector("[data-ops-runtime]").innerHTML,/current checker/);
assert.equal(root.querySelector('[data-ops-action="new"]').hidden,true);
''')


def test_late_poll_capabilities_cannot_overwrite_reopened_runtime_state():
    run_node(CONTROLLER_HARNESS + POLL_HARNESS + r'''
const oldPoll=timers[0]();resolveView();await tick();
const stale=calls.at(-1);assert.match(stale.url,/capabilities$/);stale.settled=true;
root.open=false;events.get("close")();
const fresh=controller.open();calls.at(-1).resolve({role:"checker",display_name:"current checker"});await tick();
const latestCaps=calls.at(-3);assert.match(latestCaps.url,/capabilities$/);
latestCaps.settled=true;latestCaps.resolve({runtime:{alive:false},monitoring_refs:[]});resolveView();await fresh;
stale.resolve({runtime:{alive:true},monitoring_refs:[]});await oldPoll;
assert.match(root.querySelector("[data-ops-runtime]").innerHTML,/后台未运行/);
assert.match(root.querySelector("[data-ops-runtime]").innerHTML,/current checker/);
''')


def test_closed_dialog_can_refresh_unread_badge_without_reopening():
    run_node(CONTROLLER_HARNESS + r'''
controller.bind();calls.at(-1).resolve({role:"maker",display_name:"maker"});await tick();
calls.at(-1).resolve({notifications:[{notification_id:"one",read_at:null}]});await tick();
assert.equal(root.open,false);assert.equal(root.querySelector("badge").textContent,"1");
assert.equal(root.querySelector("badge").hidden,false);
const poll=timers[0]();calls.at(-1).resolve({notifications:[{notification_id:"one",read_at:"ack"}]});await poll;
assert.equal(root.open,false);assert.equal(root.querySelector("badge").hidden,true);
''')
