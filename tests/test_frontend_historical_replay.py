"""UI contracts preserve data identity, unknown evidence and plan CAS authority."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = r'''
import assert from "node:assert/strict";
import {collectReplay,collectReconciliation,replayFormHtml,utcValue} from "./marvis/static/js/historical-replay-form.js";
import {receiptHtml} from "./marvis/static/js/historical-replay-view.js";
import {submitDriverConfirm,driverGateActionable} from "./marvis/static/js/v2/driver_gate_confirm.js";
const dataset={id:"d",task_id:"t",content_hash:"a".repeat(64),columns:["id","decision","x","event","available","y","actual_loss","actual_profit"],row_count:2};
const values={dataset_id:"d",record_id_col:"id",decision_at_col:"decision",as_of:"2026-08-01T00:00",source_ref:"historical-export",population:"all",baseline:"b".repeat(64),challenger:"c".repeat(64),counterfactual:""};
const context={taskId:"t",datasets:[dataset],packages:[{package_hash:values.baseline},{package_hash:values.challenger}],features:[{name:"x",value_col:"x",event_at_col:"event",available_at_col:"available"}]};
'''


def node(body):
    result = subprocess.run(["node", "--input-type=module", "-e", PRELUDE + body], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_replay_contract_keeps_unknowns_and_current_source_hash():
    node(r'''
const result=collectReplay(values,context);
assert.equal(result.expected_content_hash,dataset.content_hash);
assert.equal(result.economics,null);assert.equal(result.protected_group,null);assert.equal(result.observed_actions,null);
assert.equal(result.as_of,"2026-08-01T00:00:00.000Z");
assert.deepEqual(result.features,context.features);
assert.throws(()=>collectReplay(values,{...context,taskId:"another"}),/当前任务/);
assert.throws(()=>collectReplay(values,{...context,features:[]}),/字段合同/);
assert.throws(()=>collectReplay({...values,challenger:"z"},context),/已登记/);
assert.throws(()=>collectReplay(values,{...context,features:[{...context.features[0],available_at_col:"wrong"}]}),/可用时间/);
assert.throws(()=>utcValue("2026-02-30T00:00","截止时间"),/无效/);
''')


def test_optional_assumptions_preserve_zero_require_sources_and_never_infer():
    node(r'''
const sources=Object.fromEntries(["ead","pd","annual_rate","funding_rate","lgd","term_months","operating_cost_per_loan"].map(k=>[`source_${k}`,"explicit"]));
const v={...values,...sources,currency:"CNY",ead_col:"x",economics_available_at_col:"available",annual_rate:"0",funding_rate:"0",lgd:"0",term_months:"12",operating_cost_per_loan:"0"};
const result=collectReplay(v,{...context,enabled:["economics"],constraints:[{metric:"profit",operator:">=",threshold:"0",unit:"CNY"}]});
assert.equal(result.economics.lgd,0);assert.equal(result.economics.annual_rate,0);assert.equal(result.constraints[0].threshold,0);
assert.throws(()=>collectReplay({...v,source_pd:""},{...context,enabled:["economics"]}),/假设来源/);
assert.throws(()=>collectReplay({...v,lgd:""},{...context,enabled:["economics"]}),/损失率/);
assert.equal(collectReplay(v,context).economics,null);
''')


def test_reconciliation_requires_bound_receipt_and_maturity_source():
    node(r'''
const v={dataset_id:"d",record_id_col:"id",observed_at_col:"available",actual_loss_col:"actual_loss",actual_profit_col:"actual_profit",currency:"CNY",maturity_days:"90",maturity_source_ref:"document",source_ref:"cash-flow-export",reconciled_at:"2026-08-01T00:00"};
const ctx={taskId:"t",datasets:[dataset],artifactId:"d".repeat(64)};
assert.equal(collectReconciliation(v,ctx).maturity_days,90);
assert.equal(collectReconciliation(v,ctx).expected_content_hash,dataset.content_hash);
assert.throws(()=>collectReconciliation({...v,maturity_days:"0"},ctx),/成熟天数/);
assert.throws(()=>collectReconciliation({...v,maturity_source_ref:""},ctx),/成熟口径来源/);
assert.throws(()=>collectReconciliation(v,{...ctx,artifactId:"wrong"}),/认证回放/);
''')


def test_rendering_keeps_missing_economics_unknown_and_escapes_labels():
    node(r'''
const html=receiptHtml({artifact_id:"d".repeat(64),kind:"decision_twin_batch_replay",payload:{schema_version:"decision_twin.batch_receipt.v2",source:{population:"<script>bad()</script>",population_count:2},observed_actions:{status:"unknown"},scenarios:[{name:"candidate",package_hash:"c".repeat(64),metrics:{count:2,approval_count:0,approval_rate:0,economics:{status:"unknown",value:null,reason:"missing"},protected_groups:{value:null},operations_capacity:{value:null},stability:{status:"unknown",reason:"missing"}},comparison:{approval_rate_delta:0,decision_change_count:0},constraints:{status:"unknown"}}]}});
assert.doesNotMatch(html,/<script>/);assert.match(html,/0 \/ 2/);assert.match(html,/未知/);assert.match(html,/证据不足/);assert.match(html,/未导入/);assert.match(html,/尚未证明方案当时已上线/);assert.match(html,/因果关系也未识别/);
assert.match(html,/data-history-format="xlsx"/);assert.match(html,/data-history-format="docx"/);
''')


MANUAL = r'''
const calls=[],messages=[],busy=[];
const attrs={"data-expected-plan-id":"plan", "data-expected-plan-status":"validated","data-expected-plan-revision":"0","data-expected-plan-fingerprint":"before"};
const button={getAttribute:name=>attrs[name],disabled:false};
const config={getSelectedTaskId:()=>"t",getSelectedTask:()=>({id:"t",run_mode:"manual"}),setActionStatus:m=>messages.push(m),setDriverExecutionBusy:value=>busy.push(value),refreshAgentMessages:async()=>{},api:async(url,options)=>{calls.push({url,body:JSON.parse(options.body)});return {plan:{confirmation_snapshot:{expected_plan_fingerprint:"after"}}};}};
'''


def test_manual_start_uses_current_confirmed_fingerprint_without_agent_route():
    node(MANUAL + r'''
await submitDriverConfirm(button,config);
assert.deepEqual(calls.map(c=>c.url),["/api/plans/plan/confirm","/api/plans/plan/run"]);
assert.equal(calls[0].body.expected_plan_fingerprint,"before");assert.equal(calls[1].body.expected_plan_fingerprint,"after");assert.deepEqual(busy,[true,false]);
''')


def test_manual_start_cannot_run_after_confirmation_conflict():
    node(MANUAL + r'''
config.api=async(url)=>{calls.push({url});throw new Error("stale confirmation")};
await submitDriverConfirm(button,config);
assert.equal(calls.length,1);assert.match(messages.at(-1),/stale/);assert.deepEqual(busy,[true,false]);
''')


def test_confirmed_manual_plan_can_retry_run_without_confirming_again():
    node(MANUAL + r'''
attrs["data-expected-plan-status"]="confirmed";
await submitDriverConfirm(button,config);
assert.deepEqual(calls.map(c=>c.url),["/api/plans/plan/run"]);
assert.equal(calls[0].body.expected_plan_fingerprint,"before");
const binding={taskId:"t",planId:"p",stepStatus:"confirmed",snapshot:{expected_plan_status:"confirmed",expected_plan_revision:0,expected_plan_fingerprint:"fp"}};
assert.equal(driverGateActionable(binding),false);
assert.equal(driverGateActionable({...binding,startStatuses:["validated","confirmed"]}),true);
''')


TEMPORAL = r'''
const temporalValues={...values,timezone:"UTC",policy_source_ref:"human policy",minimum_reference_rows:"4",minimum_comparison_rows:"4",bin_count:"2",max_score_psi:"0",max_action_psi:"0.25",max_absolute_approval_rate_delta:"0.1"};
const windows=[{kind:"reference",name:"reference",start:"2026-01-01T00:00:00Z",end:"2026-02-01T00:00:00Z"},{kind:"comparison",name:"comparison",start:"2026-02-01T00:00:00Z",end:"2026-03-01T00:00:00Z"}];
const temporalContext={...context,enabled:["temporal_stability"],windows};
'''


def test_temporal_requires_explicit_policy_and_nonoverlapping_offset_windows():
    node(TEMPORAL + r'''
const t=collectReplay(temporalValues,temporalContext).temporal_stability;
assert.equal(t.thresholds.max_score_psi,0);assert.equal(t.reference_window.start,windows[0].start);assert.equal(t.bin_count,2);
assert.equal(collectReplay(temporalValues,context).temporal_stability,undefined,"absent temporal settings preserve legacy contract");
assert.throws(()=>collectReplay({...temporalValues,policy_source_ref:""},temporalContext),/口径来源/);
assert.throws(()=>collectReplay({...temporalValues,max_score_psi:""},temporalContext),/PSI/);
assert.throws(()=>collectReplay({...temporalValues,timezone:"bad/zone"},temporalContext),/IANA/);
assert.throws(()=>collectReplay(temporalValues,{...temporalContext,windows:[]}),/参考窗/);
assert.throws(()=>collectReplay(temporalValues,{...temporalContext,windows:[windows[0],{...windows[1],start:"2026-01-20T00:00:00Z"}]}),/重叠/);
assert.throws(()=>collectReplay(temporalValues,{...temporalContext,windows:[windows[0],{...windows[1],start:"2026-02-01T00:00:00"}]}),/偏移/);
''')


def test_temporal_views_display_memberships_and_unknown_checks_without_calculating():
    node(r'''
import {temporalPopulationHtml,temporalResultHtml} from "./marvis/static/js/historical-replay-view.js";
const window={name:"window<",start:"2026-01-01",end:"2026-02-01",sample_count:3,members_hash:"frozen"};
const proposal=temporalPopulationHtml({source_count:7,excluded_count:1,reference:window,comparisons:[window]});
assert.match(proposal,/窗口外排除 1/);assert.match(proposal,/frozen/);assert.doesNotMatch(proposal,/window</);
const html=temporalResultHtml({schema_version:"decision_twin.temporal_stability.v1",verdict:"insufficient_evidence",comparisons:[{window,checks:[{metric:"score_psi",value:null,threshold:0.25,passed:null,reason:"too few rows"},{metric:"action_psi",value:0,threshold:0,passed:true}]}],reference:{},policy:{}});
assert.match(html,/window&lt;/);assert.match(html,/证据不足/);assert.match(html,/未知：too few rows/);assert.match(html,/动作 PSI/);assert.match(html,/>0<\/td>/);
''')


CONTROLLER = r'''
import {createHistoricalReplayController} from "./marvis/static/js/historical-replay-controller.js";
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let task={id:"a",task_type:"modeling",run_mode:"manual"},visit=1,plan=null,busy=false;
const elements=new Map(),buttons=[{disabled:false},{disabled:false}];
const element=()=>({innerHTML:"",textContent:"",querySelectorAll:()=>buttons});
const panel={hidden:false,innerHTML:"",contains:()=>true,querySelector:s=>{if(!elements.has(s))elements.set(s,element());return elements.get(s)}};
const calls=[];
const controller=createHistoricalReplayController({getElement:()=>panel,getTask:()=>task,getPlan:()=>plan,captureView:()=>visit,isCurrentView:v=>v===visit,isBusy:()=>busy,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject}))});
const click=action=>controller.handle({type:"click",preventDefault(){},target:{closest:()=>({dataset:{historyAction:action}})}});
controller.render();
'''


def test_late_receipt_response_after_a_b_a_does_not_replace_current_view():
    node(CONTROLLER + r'''
click("refresh");const old=calls.at(-1);
visit++;task={...task,id:"b"};controller.render();visit++;task={...task,id:"a"};controller.render();
click("refresh");calls.at(-1).resolve({artifacts:[{artifact_id:"fresh",status:"available",kind:"decision_twin_batch_replay"}]});await tick();
old.resolve({artifacts:[{artifact_id:"stale",status:"available",kind:"decision_twin_batch_replay"}]});await tick();
assert.match(panel.querySelector("[data-history-receipts]").innerHTML,/fresh/);assert.doesNotMatch(panel.querySelector("[data-history-receipts]").innerHTML,/stale/);
''')


def test_replay_gate_waits_for_task_lease_without_replacing_reason():
    node(CONTROLLER + r'''
plan={id:"p",status:"running",steps:[{id:"s",title:"review",status:"awaiting_confirm",tool_ref:{plugin:"decision_twin"},confirmation_snapshot:{expected_step_fingerprint:"fp"}}]};
controller.render();const gate=panel.querySelector("[data-history-gate]");const initial=gate.innerHTML;
busy=true;controller.render();assert.equal(buttons[0].disabled,true);assert.equal(gate.innerHTML,initial);
click("approve");assert.equal(calls.length,0);assert.match(panel.querySelector("[data-history-error]").textContent,/理由会保留/);
busy=false;controller.render();assert.equal(buttons[0].disabled,false);assert.equal(gate.innerHTML,initial);
''')
