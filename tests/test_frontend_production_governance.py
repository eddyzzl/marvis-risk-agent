"""Local decision UI binds native packages, role stages and exact deployment heads."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = r'''
import assert from "node:assert/strict";
import {collectProbe,promotionPayload,createProductionGovernanceController} from "./marvis/static/js/production-governance-controller.js";
import {collectPackage,readinessHtml} from "./marvis/static/js/production-package-form.js";
import {requestHtml,headsHtml,decisionHtml} from "./marvis/static/js/production-governance-view.js";
'''


def node(body):
    result = subprocess.run(["node", "--input-type=module", "-e", PRELUDE + body], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_probe_preserves_false_zero_null_and_cannot_guess_missing_values():
    node(r'''
const fields=[{name:"pd_input",type:"number",nullable:false},{name:"count",type:"integer",nullable:false},{name:"flag",type:"boolean",nullable:false},{name:"note",type:"string",nullable:true}];
assert.deepEqual(collectProbe({feature_0:"0",feature_1:"2",feature_2:"false",feature_3:""},fields),{pd_input:0,count:2,flag:false,note:null});
assert.throws(()=>collectProbe({feature_0:"",feature_1:"2",feature_2:"false",feature_3:""},fields),/pd_input/);
assert.throws(()=>collectProbe({feature_0:"0",feature_1:"2.5",feature_2:"false",feature_3:""},fields),/数值类型/);
assert.throws(()=>collectProbe({feature_0:"0",feature_1:"2",feature_2:"unknown",feature_3:""},fields),/true 或 false/);
''')


def test_promotion_binds_verified_package_strategy_version_and_explicit_expiry():
    node(r'''
const record={package_hash:"a".repeat(64),manifest:{configuration:{strategy_id:"s",strategy_version:4}}};
const v={deployment_slot:"shadow",expires_in_seconds:"120",reason:"reviewed"};
assert.deepEqual(promotionPayload(v,record),{environment:"local-reference",deployment_slot:"shadow",strategy_id:"s",strategy_version:4,decision_package_hash:record.package_hash,reason:"reviewed",expires_in_seconds:120});
assert.throws(()=>promotionPayload({...v,expires_in_seconds:""},record),/有效期/);
assert.throws(()=>promotionPayload({...v,deployment_slot:"unknown"},record),/服务位置/);
''')


BUILD = r'''
const readiness={state:"authenticated",raw_requirements:[{name:"raw_x",type:null,nullable:null,declaration_required:true}],score_products:[{value:"raw_pd",available:true},{value:"calibrated_pd",available:false,reason_code:"no_calibration"}],score_field_candidates:["pd"]};
const values={model_artifact_id:"m",strategy_id:"s",score_product:"raw_pd",score_field:"pd",decision_node:"approval",timeout_seconds:"10",failure_action:"review",raw_type_0:"number",raw_nullable_0:"false"};
const context={readiness,strategies:[{strategy_id:"s",version:4}],checkedSignature:JSON.stringify(["m","s",4])};
'''


def test_build_requires_raw_contract_declaration_and_current_readiness_identity():
    node(BUILD + r'''
assert.deepEqual(collectPackage(values,context).raw_schema,[{name:"raw_x",type:"number",nullable:false}]);
assert.throws(()=>collectPackage({...values,raw_type_0:""},context),/类型和空值/);
assert.throws(()=>collectPackage({...values,raw_nullable_0:""},context),/类型和空值/);
assert.throws(()=>collectPackage({...values,model_artifact_id:"new"},context),/来源已变化/);
assert.throws(()=>collectPackage({...values,score_product:"calibrated_pd"},context),/已认证/);
assert.throws(()=>collectPackage(values,{...context,readiness:{state:"blocked"}}),/校验构包来源/);
const html=readinessHtml(readiness);assert.match(html,/请选择/);assert.doesNotMatch(html,/selected/);
assert.doesNotMatch(readinessHtml({state:"blocked",reason_codes:["missing_provenance"]}),/type="submit"/);
''')


def test_only_independent_matching_role_stage_gets_approval_control():
    node(r'''
const r={id:"p",strategy_id:"s",strategy_version:4,status:"pending_checker",maker_principal_id:"maker",approvals:[],reason:"safe<script>",expires_at:"later"};
assert.match(requestHtml(r,{id:"checker",role:"checker"}),/data-production-form="approve"/);
assert.doesNotMatch(requestHtml(r,{id:"admin",role:"admin"}),/data-production-form="approve"/);
assert.doesNotMatch(requestHtml(r,{id:"maker",role:"checker"}),/data-production-form="approve"/);
assert.match(requestHtml({...r,status:"awaiting_admin"},{id:"admin",role:"admin"}),/data-production-form="approve"/);
assert.doesNotMatch(requestHtml({...r,status:"awaiting_admin"},{id:"checker",role:"checker"}),/data-production-form="approve"/);
assert.doesNotMatch(requestHtml(r,{id:"checker",role:"checker"}),/<script>/);
assert.doesNotMatch(requestHtml({...r,status:"approved"},{id:"admin",role:"admin"}),/data-production-form="activate"/);
assert.match(requestHtml({...r,status:"approved"},{id:"admin",role:"admin"},{activation_evidence:{receipt_id:"read-back"}}),/data-production-form="activate"/);
''')


def test_first_deployment_cannot_claim_rollback_and_absent_decision_is_unknown():
    node(r'''
const head={state:"serving",revision:1,package_hash:"a".repeat(64),deployment_id:"d"};
assert.doesNotMatch(headsHtml([head,{state:"unavailable"}],{active:{predecessor_deployment_id:null}},"admin"),/data-production-action="rollback"/);
assert.match(headsHtml([head,{state:"unavailable"}],{active:{predecessor_deployment_id:"previous"}},"admin"),/data-production-action="rollback"/);
assert.match(decisionHtml({score:null}),/未知/);assert.match(decisionHtml({status:"fallback",score:null,error_code:"worker_timeout",next_action:"retry"}),/执行不可用/);assert.match(decisionHtml({score:0,action:{type:"review"}}),/分数：0/);
''')


CONTROLLER = r'''
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let visible=true;
const elements=new Map();const element=()=>({innerHTML:"",textContent:"",hidden:false,dataset:{}});
const root={dataset:{},innerHTML:"",contains:()=>true,querySelector:s=>{if(!elements.has(s))elements.set(s,element());return elements.get(s)},querySelectorAll:()=>[]};
const calls=[];
const c=createProductionGovernanceController({root,isVisible:()=>visible,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject}))});
const settle=()=>{for(const call of calls.filter(c=>!c.settled&&!c.url.endsWith('/me'))){call.settled=true;call.resolve(call.url.includes('promotion-requests')?{requests:[],next_offset:null}:call.url.includes('installations')?{installations:[],next_offset:null}:call.url.includes('packages')?{packages:[],next_offset:null}:call.url.includes('status')?{state:'unavailable'}:{active:null,shadow:null});}};
const click=(action,id)=>c.handle({type:'click',target:{closest:()=>({dataset:{productionAction:action,productionId:id}})},preventDefault(){}});
'''


def test_closed_view_late_identity_does_not_replace_reopened_principal():
    node(CONTROLLER + r'''
const old=c.refresh(),stale=calls.at(-1);c.leave();
const fresh=c.refresh();calls.at(-1).resolve({id:"new",display_name:"Current checker",role:"checker"});await tick();settle();await fresh;
const count=calls.length;stale.resolve({id:"old",display_name:"Old maker",role:"maker"});await old;
assert.equal(calls.length,count);assert.match(root.querySelector('[data-production-identity]').innerHTML,/Current checker/);assert.equal(root.querySelector('[data-production-action="build"]').hidden,true);
''')


def test_late_package_detail_cannot_bind_newer_selection():
    node(CONTROLLER + r'''
const loaded=c.refresh();calls.at(-1).resolve({id:"maker",role:"maker",display_name:"Maker"});await tick();settle();await loaded;
click('package','old');const old=calls.at(-1);click('package','new');const fresh=calls.at(-1);
const record=id=>({package_hash:id,manifest:{configuration:{strategy_id:id,strategy_version:1}}});
fresh.resolve(record('new'));await tick();old.resolve(record('old'));await tick();
const html=root.querySelector('[data-production-detail]').innerHTML;assert.match(html,/new/);assert.doesNotMatch(html,/old/);
''')


def test_unbound_strategy_inputs_require_their_own_explicit_contract():
    node(BUILD + r'''
const extra={...readiness,unbound_strategy_fields:["pd","channel"]};
assert.throws(()=>collectPackage(values,{...context,readiness:extra}),/channel/);
const result=collectPackage({...values,raw_type_1:"string",raw_nullable_1:"false"},{...context,readiness:extra});
assert.deepEqual(result.raw_schema,[{name:"raw_x",type:"number",nullable:false},{name:"channel",type:"string",nullable:false}]);
assert.ok(!result.raw_schema.some(f=>f.name==="pd"),"computed model score must not be client input");
assert.throws(()=>collectPackage({...values,score_field:"raw_x"},context),/已认证/);
''')


def test_post_mutation_refresh_cannot_replace_a_newer_detail_selection():
    node(CONTROLLER + r'''
const load=c.refresh();calls.at(-1).resolve({id:"maker",role:"maker",display_name:"Maker"});await tick();settle();await load;
const record=id=>({package_hash:id,manifest:{configuration:{strategy_id:id,strategy_version:1}}});
click('package','old');calls.at(-1).resolve(record('old'));await tick();
const form={dataset:{productionForm:'promotion'},querySelectorAll:()=>[{name:'deployment_slot',value:'production'},{name:'reason',value:'reviewed'},{name:'expires_in_seconds',value:'100'}]};
c.handle({type:'submit',target:{closest:()=>form},preventDefault(){}});
calls.at(-1).resolve({id:'accepted-request'});await tick();
assert.match(calls.at(-1).url,/\/me$/);
click('package','new');const detail=calls.at(-1);detail.settled=true;detail.resolve(record('new'));await tick();
calls.findLast(c=>c.url.endsWith('/me')).resolve({id:'maker',role:'maker',display_name:'Maker'});await tick();settle();await tick();
assert.match(root.querySelector('[data-production-detail]').innerHTML,/new/);
assert.equal(calls.some(c=>c.url.endsWith('/promotion-requests/accepted-request')),false,'older accepted action must not navigate over a newer user selection');
''')


def test_rejection_is_explicit_terminal_and_escaped_with_no_install_or_activation():
    node(r'''
const r={id:"p",strategy_id:"s",strategy_version:1,status:"pending_checker",maker_principal_id:"maker",approvals:[],reason:"request",expires_at:"later"};
const html=requestHtml(r,{id:"checker",role:"checker"});
assert.match(html,/name="decision" required/);assert.match(html,/value="approve"/);assert.match(html,/value="reject"/);assert.doesNotMatch(html,/ selected/);
const rejected=requestHtml({...r,status:"rejected",rejection:{role:"checker",reason:"missing <policy>"}},{id:"admin",role:"admin"},{activation_evidence:{receipt_id:"old"}});
assert.match(rejected,/missing &lt;policy&gt;/);assert.doesNotMatch(rejected,/data-production-form="(approve|activate)"|data-production-action="prepare-install"/);
const h={state:"serving",revision:3,package_hash:"a".repeat(64),deployment_id:"shadow-new"};
assert.match(headsHtml([{state:"unavailable"},h],{shadow:{predecessor_deployment_id:"shadow-old"}},"admin"),/data-production-action="rollback" data-production-id="shadow"/);
''')


def test_shadow_rollback_submit_binds_shadow_head_and_slot():
    node(CONTROLLER + r'''
const loaded=c.refresh();calls.at(-1).resolve({id:"admin",role:"admin",display_name:"Admin"});await tick();
for(const call of calls.filter(c=>c.url.includes('/status?'))){call.settled=true;const shadow=call.url.endsWith('shadow');call.resolve({state:'serving',revision:4,package_hash:'a'.repeat(64),deployment_id:shadow?'shadow-current':'production-current'});}settle();await loaded;
click('rollback','shadow');
const form={dataset:{productionForm:'rollback'},querySelectorAll:()=>[{name:'reason',value:'verified rollback'}]};c.handle({type:'submit',target:{closest:()=>form},preventDefault(){}});
const request=calls.at(-1);assert.match(request.url,/shadow-current\/rollback$/);assert.equal(JSON.parse(request.options.body).deployment_slot,'shadow');assert.equal(JSON.parse(request.options.body).expected_active_deployment_id,'shadow-current');
''')
