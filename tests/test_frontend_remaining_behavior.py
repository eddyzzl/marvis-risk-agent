"""Execute remaining app consumers; service and DOM boundaries are explicit doubles."""
import subprocess
from pathlib import Path

from tests.javascript_source import slice_function

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / 'marvis/static/app.js').read_text(encoding='utf-8')


def app_node(setup, body, *signatures):
    code = '\n'.join(['import assert from "node:assert/strict";',
                      'import { escapeHtml } from "./marvis/static/js/ui-utils.js";',
                      setup, *(slice_function(APP, signature) for signature in signatures), body])
    result = subprocess.run(['node', '--input-type=module', '-e', code], check=False, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr or result.stdout


def test_validation_clears_data_workspace_and_other_tasks_reload_only_the_owner():
    app_node('''
const taskSession = {taskId: 't', task: {id:'t'}};
const calls = [];
const blindCustodyOnlyTask = task => Boolean(task.custody);
const dataWorkspacePanel = {clear: () => calls.push('clear'), reload: async (...args) => {calls.push(args); return true;}};
''', '''
for (const type of ['validation','validation_batch']) {
 taskSession.task.task_type = type;
 assert.equal(await reloadDataWorkspace('t'), false);
}
assert.deepEqual(calls, ['clear','clear']); calls.length=0;
for (const type of ['data_join','feature_analysis','modeling','strategy','vintage','portfolio']) {
 taskSession.task.task_type=type;
 assert.equal(await reloadDataWorkspace('stale'), false);
 assert.equal(await reloadDataWorkspace('t', {silent:true}), true);
}
assert.equal(calls.length, 6);
assert.ok(calls.every(([id,opts]) => id==='t' && opts.silent));
taskSession.task.custody=true;
assert.equal(await reloadDataWorkspace('t'),false);
assert.equal(calls.at(-1),'clear');
''', 'function taskUsesDataWorkspace(', 'async function reloadDataWorkspace(')


def test_gate_signature_preserves_question_gate_and_revokes_acknowledged_gate():
    app_node('''
import { lastGateMessageId } from './marvis/static/js/agent-conversation-mount.js';
import { latestPendingReportDraftMessageId } from './marvis/static/js/report-draft-table.js';
const taskSession={messages:[]};
const agentStageLabel=()=>'', agentMessageIsStreaming=()=>false, agentMessageIsThinking=()=>false;
const hasWorkflowErrorDiagnostic=()=>false, driverGateMessageIsActionable=()=>true;
''', '''
const gate={id:'g',role:'assistant',metadata:{kind:'gate',plan_id:'p',step_id:'s'}};
const question={id:'q',role:'user',content:'为什么？'};
const answer={id:'a',role:'assistant',metadata:{intent:'chat'}};
const read=messages=>JSON.parse(agentStructuralSignature(messages,[])).skeleton.find(row=>row.id==='g').flags.is_latest_gate;
assert.equal(read([gate]),true);
assert.equal(read([gate,question,answer]),true);
assert.equal(read([gate,question,{id:'ack',role:'assistant',metadata:{intent:'ui_action_ack',ui_action:{type:'confirm'},expected_plan_id:'p'}}]),false);
''', 'function agentStructuralSignature(')


def test_vintage_creation_sends_home_choice_before_start_and_releases_lock():
    app_node('''
let createTaskInFlight=false;
const calls=[]; const taskSession={taskId:'t'};
const task={id:'t',task_type:'vintage',run_mode:'agent'};
const setCreateTaskSubmitting=value=>calls.push(['submitting',value]);
const createTask=async ({onCreated})=>{onCreated?.(task);return task;};
const ensureValidationMaterialSelection=async task=>task;
const createTaskDialog={activeTaskType:()=> 'vintage'};
const usesAgentValidationWorkbench=()=>false, taskTypeDefinition=()=>({});
const loadAgentMessages=async id=>calls.push(['messages',id]);
const renderAll=()=>{}, setActionStatus=()=>{};
const sendFirstAgentMessage=async (_task,text)=>{calls.push(['send',text]);return Boolean(text);};
const dispatchDriverStart=async id=>calls.push(['start',id]);
''', '''
await createTaskAndScan({firstMessage:'标准 Vintage'});
assert.deepEqual(calls,[['submitting',true],['messages','t'],['send','标准 Vintage'],['submitting',false]]);
assert.equal(createTaskInFlight,false);
calls.length=0;
await createTaskAndScan();
assert.deepEqual(calls,[['submitting',true],['messages','t'],['send',''],['start','t'],['submitting',false]]);
''', 'async function createTaskAndScan(')


def test_home_creation_resolves_on_task_creation_and_reports_later_failure():
    app_node('''
const calls=[];let callbacks, fail;
const createTaskDialog={prefillTaskDialog:(...args)=>calls.push(['prefill',...args])};
const homeDraftFirstMessage=draft=>draft.text;
const createTaskAndScan=options=>{callbacks=options;return new Promise((_resolve,reject)=>{fail=reject;});};
const setActionStatus=(...args)=>calls.push(['status',...args]),renderAll=()=>calls.push(['render']);
const $=()=>({textContent:'材料不足'});
''', '''
const draft={taskType:'vintage',text:'标准 Vintage',files:[]};
const promise=createTaskFromHome(draft);
assert.deepEqual(calls,[['prefill',draft]]);
assert.equal(callbacks.firstMessage,'标准 Vintage');
callbacks.onCreated({id:'t'});
assert.equal(await promise,true);
fail(new Error('后续失败'));
await new Promise(resolve=>setImmediate(resolve));
assert.deepEqual(calls.slice(1),[['status','任务已创建，但后续步骤失败。','error','后续失败'],['render']]);
const rejected=createTaskFromHome(draft);fail(new Error('创建失败'));
await assert.rejects(rejected,/创建失败/);
''', 'function createTaskFromHome(')


def test_manual_step_button_blocks_unconfirmed_contract_and_labels_finished_step():
    app_node('''
const taskSession={taskId:'t',panels:{contract:{taskId:'t',value:{status:'pending_confirmation'}}}};
let runnable=true;
const selectedTaskIsAgentMode=()=>false,completedReportReadyForDownloads=()=>false;
const selectedTaskIsBusy=()=>false,stepStopAction=()=>null,canRunStepAction=()=>runnable;
const recommendedAction=()=>'',usesPmmlScoringWorkflow=()=>true,workbenchTaskId=()=> 't';
''', '''
let html=stepActionButtonHtml({action:'notebook'});
assert.match(html,/ disabled/);assert.match(html,/请先确认上面的验证字段/);
taskSession.panels.contract.value.status='ready';
assert.doesNotMatch(stepActionButtonHtml({action:'notebook'}),/ disabled/);
runnable=false;
assert.match(stepActionButtonHtml({action:'notebook'},'succeeded'),/这一步已完成/);
assert.match(stepActionButtonHtml({action:'notebook'},'pending'),/请先完成上一步/);
''', 'function stepActionBlockedReason(', 'function stepActionButtonHtml(')


def test_failed_manual_action_toasts_reason_and_always_releases_busy():
    # Load the real translation table as well as the function consuming it.
    constants = APP[APP.index('const ACTION_ERROR_TEXT ='):APP.index('\n];', APP.index('const ACTION_ERROR_TEXT ='))+3]
    app_node(constants + '''
const calls=[];const taskSession={taskId:'t'};
let current=true;
const taskRequests={capture:()=>1,current:()=>current};
const claimBusy=()=>({id:1}),releaseBusy=token=>calls.push(['released',token?.id]);
const refreshTasks=async()=>{throw Error('refresh unavailable');};
const renderActionError=()=>{},setActionStatus=()=>{},renderAll=()=>calls.push(['render']);
const showComingSoonToast=(message,options)=>calls.push(['toast',message,options.durationMs]);
const usesPmmlScoringWorkflow=()=>true;
''', '''
for (const actionId of ['scan','notebook','metrics','report']) {
 calls.length=0;
 await runAction(async()=>{throw Error('validation input contract requires confirmation');},{actionId});
 assert.equal(calls[0][0],'toast');assert.match(calls[0][1],/请先确认上面的验证字段，再运行。/);
 assert.equal(calls[0][2],5200);assert.deepEqual(calls.slice(1),[['released',1],['render']]);
}
current=false;calls.length=0;
await runAction(async()=>{throw Error('stale task');},{actionId:'notebook'});
assert.deepEqual(calls,[['released',1]]);
''', 'function humanizeActionError(', 'function actionFailureStatusTitle(', 'async function runAction(')


def test_scan_contract_text_updates_only_ready_owner_and_preserves_errors():
    app_node('''
const taskSession={taskId:'t',panels:{contract:{taskId:'t',value:{status:'ready'}}}};
const workbenchTaskId=()=> 't';
const SCAN_CONTRACT_CONFIRMED_TEXT='验证字段已确认';
const line={textContent:'等待确认'};let error=false;
const document={querySelector:()=>({querySelector:()=>line,classList:{contains:()=>error}})};
const setTextIfChanged=(element,text)=>element.textContent=text;
''', '''
const check={id:'validation_input_contract',status:'warning',message:'等待确认'};
assert.equal(scanCheckMessage(check),'验证字段已确认');syncScanContractCheck();
assert.equal(line.textContent,'验证字段已确认');
assert.equal(scanCheckMessage({...check,status:'error',message:'字段损坏'}),'字段损坏');
line.textContent='字段损坏';error=true;syncScanContractCheck();assert.equal(line.textContent,'字段损坏');
error=false;taskSession.panels.contract.taskId='other';line.textContent='等待确认';
assert.equal(scanCheckMessage(check),'等待确认');syncScanContractCheck();assert.equal(line.textContent,'等待确认');
''', 'function scanContractConfirmed(', 'function scanCheckMessage(', 'function syncScanContractCheck(')


def test_field_confirmation_refreshes_header_before_next_step_and_ignores_switch():
    app_node('''
const calls=[];let selected='t';let switchOnRefresh=false;
const taskSession={taskId:'t',panels:{contract:{taskId:'t',value:{status:'pending_confirmation',revision:7,contract:{}}}}};
const isWorkbenchTaskId=id=>id===selected,workbenchTaskId=()=>selected;
const validationContractCandidateValue=(_form,_contract,key)=>({target_col:'bad',split_col:'split',time_col:'date',pmml_output_field:'p',model_params:{},feature_metadata_selection:{feature_col:'feature'}}[key]);
const validationContractScalar=value=>value;
const button={disabled:false},status={textContent:''};
const form={dataset:{validationContractTaskId:'t'},querySelector:s=>s.includes('submit')?button:status,elements:Object.fromEntries(['positive_label','negative_label','split_train','split_test','split_oot','time_granularity'].map(key=>[key,{value:key}]))};
const api=async(url,options)=>{calls.push(['put',url,JSON.parse(options.body)]);return {status:'ready'};};
const renderValidationInputContract=record=>calls.push(['contract',record.status]);
const refreshTasks=async()=>{calls.push(['refresh']);if(switchOnRefresh)selected='other';};
const selectedTaskIsAgentMode=()=>false,setActionStatus=(...args)=>calls.push(['status',...args]);
''', '''
await submitValidationInputContract(form);
assert.equal(calls[0][0],'put');assert.equal(calls[0][1],'/api/tasks/t/validation-input-contract');
assert.equal(calls[0][2].revision,7);assert.equal(calls[0][2].target_col,'bad');
assert.deepEqual(calls.slice(1,3),[['contract','ready'],['refresh']]);
assert.match(calls[3][1],/下一步点右侧「运行」做 PMML 打分测试。/);
calls.length=0;switchOnRefresh=true;await submitValidationInputContract(form);
assert.deepEqual(calls.map(row=>row[0]),['put','contract','refresh']);
''', 'async function submitValidationInputContract(')


def test_reply_hint_is_escaped_assistant_metadata_and_hidden_during_thinking():
    app_node('''
const gateMessageForCurrentTool=message=>message;
const hasWorkflowErrorDiagnostic=()=>false,isStrategyClarificationMessageController=()=>false,agentGateUsesConversationOnly=()=>false;
const agentMessageIsStreaming=()=>false,agentMessageIsThinking=message=>Boolean(message.metadata?.thinking);
const hasReportDraftValues=()=>false,agentVisibleContent=message=>message.content||'';
const formatAgentMessageContent=escapeHtml,workflowMessageContentHtml=(_message,fallback)=>fallback();
const empty=()=>'';
const agentMemoryReferencesHtml=empty,agentThinkingHtml=empty,agentMessageIsResult=()=>false;
const planStepMessageAttrs=empty,renderModelTuningMessageProgress=empty,agentMessageModelDeliveryHtml=empty;
const agentMessageTablesHtml=empty,agentMessageResultDatasetHtml=empty,conversationActionCardHtml=empty;
const agentMessageReportDownloadHtml=empty,agentMessageValidationArtifactsHtml=empty,agentMessageGateButtonHtml=empty;
const agentFaceHtml=empty,agentStageLabel=empty,agentMessageMetaHtml=empty,agentMessageMetaLabel=empty;
const taskSession={messages:[]};
''', r'''
const message={id:'m',role:'assistant',content:'正文',metadata:{reply_hint:' <b>继续</b> '}};
assert.match(agentMessageHtml(message),/<p class="agent-reply-hint">&lt;b&gt;继续&lt;\/b&gt;<\/p>/);
assert.doesNotMatch(agentMessageHtml({...message,role:'user'}),/agent-reply-hint/);
assert.doesNotMatch(agentMessageHtml({...message,metadata:{...message.metadata,thinking:true}}),/agent-reply-hint/);
assert.doesNotMatch(agentMessageHtml({...message,metadata:{}}),/agent-reply-hint/);
''', 'function agentMessageHtml(')


def test_plan_controller_places_activity_for_running_phase_and_clears_on_completion():
    app_node('''
import {createPlanRailController} from './marvis/static/js/v2/plan_rail_controller.js';
const elements={};
const $=id=>elements[id] ||= {innerHTML:'',hidden:false,dataset:{},classList:{toggle(){},add(){},remove(){}},setAttribute(){},querySelector(){return null;}};
globalThis.document={querySelector:()=>null};
let plan={id:'p',status:'running',steps:[{id:'s',title:'训练模型',phase:'建模',status:'running',started_at:new Date().toISOString(),tool_ref:{plugin:'modeling',tool:'train_model'}}]};
const task={id:'t',task_type:'modeling',run_mode:'agent'};
const controller=createPlanRailController({$,stepCheckerHtml:()=>'',getSelectedTask:()=>task,getSelectedTaskId:()=> 't',getAgentMessages:()=>[],isAgentMode:()=>true,apiClient:async()=>({plans:[plan]})});
''', '''
await controller.maybeFetchPlan();controller.render({force:true});
assert.match(elements.planDriverActions.innerHTML,/正在执行「建模」/);
assert.match(elements.planDriverActions.innerHTML,/训练模型/);
plan={...plan,status:'done',steps:plan.steps.map(step=>({...step,status:'done'}))};
controller.resetFetchThrottle('t');await controller.maybeFetchPlan();controller.render({force:true});
assert.doesNotMatch(elements.planDriverActions.innerHTML,/正在执行「建模」/);
''')


def test_inspector_action_opens_named_panel_without_sending_a_message():
    app_node('''
const calls=[];
const workbenchChrome={openInspectorItem:id=>calls.push(['open',id])};
const startAgentValidation=()=>{throw Error('unexpected send');};
''', '''
handleAgentInspectorActionClick({target:{closest:()=>null},preventDefault(){throw Error('unrelated click prevented');}});
handleAgentInspectorActionClick({target:{closest:()=>({dataset:{agentInspectorItem:'collectionWorkspacePanel'}})},preventDefault:()=>calls.push(['prevent'])});
assert.deepEqual(calls,[['prevent'],['open','collectionWorkspacePanel']]);
''', 'function handleAgentInspectorActionClick(')


def test_material_selection_resumes_agent_only_after_busy_claim_and_keeps_task_scope():
    app_node('''
const calls=[],timers=[];let busy=true;const input={value:''};
const taskSession={taskId:'t',task:{id:'t',run_mode:'agent'}};
const taskRequests={capture:()=>1,current:()=>true};
const findTaskInCache=()=>taskSession.task;
const materialBindingDialog={ensureMaterialSelection:async task=>task};
const setSelectedTask=()=>{},rememberSelectedTaskId=()=>{},refreshTasks=async()=>{},reloadDataWorkspace=async()=>{};
const setActionStatus=()=>{},workbenchTaskId=()=>taskSession.taskId;
const setTimeout=(callback,delay)=>timers.push([callback,delay]);
const taskBusyAction=()=>busy,agentSendIsStopMode=()=>false,$=()=>input;
const autoGrowComposerInput=()=>{},updateAgentSendDisabled=()=>{};
const startAgentValidation=()=>{};
const runAction=(action,options)=>calls.push({action,options,value:input.value});
''', '''
assert.equal(await handleAgentMaterialSelectionRequest('t'),true);
assert.equal(timers.length,1);timers.shift()[0]();assert.equal(calls.length,0);
assert.equal(timers[0][1],120);busy=false;timers.shift()[0]();
assert.equal(calls.length,1);assert.equal(calls[0].action,startAgentValidation);
assert.equal(calls[0].value,'继续验证');assert.equal(calls[0].options.taskId,'t');
continueAgentAfterMaterialSelection('t');taskSession.taskId='other';timers.shift()[0]();assert.equal(calls.length,1);
taskSession.taskId='t';taskSession.task.run_mode='manual';
await handleAgentMaterialSelectionRequest('t');assert.equal(timers.length,0);
''', 'async function handleAgentMaterialSelectionRequest(', 'function continueAgentAfterMaterialSelection(')


def test_feature_binning_click_posts_selected_fields_and_snapshot():
    app_node('''
import {handleFeatureBinningClick as handleFeatureBinningClickController} from './marvis/static/js/v2/feature_binning_gate.js';
const posts=[];let rendered=false;
const input={value:'7',disabled:false};
const button={dataset:{featureBinningSubmit:'selected'},disabled:false};
const wrap={dataset:{featureBinningPlanId:'p',featureBinningStepId:'s',expectedPlanStatus:'awaiting_confirm',expectedPlanRevision:'2',expectedPlanFingerprint:'a'.repeat(64),expectedStepFingerprint:'b'.repeat(64)},querySelector:()=>input,querySelectorAll:selector=>selector==='button, input'?[input,button]:[{value:'age'},{value:'income'}]};
button.closest=()=>wrap;
const driverConfirmControllerContext=()=>({selectedTaskId:'t',api:async(url,options)=>{posts.push([url,JSON.parse(options.body)]);return {messages:[]};},renderAgentConversation:()=>{rendered=true;}});
''', '''
assert.equal(handleFeatureBinningClick({target:{closest:()=>null},preventDefault(){}}),false);
let prevented=false;
assert.equal(handleFeatureBinningClick({target:{closest:()=>button},preventDefault(){prevented=true;}}),true);
await new Promise(resolve=>setImmediate(resolve));
assert.equal(prevented,true);assert.equal(rendered,true);assert.equal(posts.length,1);
assert.equal(posts[0][0],'/api/tasks/t/agent/messages');
assert.equal(posts[0][1].ui_action,'confirm_feature_binning');
assert.deepEqual(posts[0][1].adjust_params,{features:['age','income'],bins:7});
assert.equal(posts[0][1].expected_plan_revision,2);
''', 'function handleFeatureBinningClick(')
