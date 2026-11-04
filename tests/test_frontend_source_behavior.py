"""F13 behavior replacements: drive production functions and observe effects."""

from tests.test_frontend_scroll_signature_behaviour import run_node
from tests.test_frontend_task_session import PLAN_HARNESS
from tests.test_frontend_task_session import run_node as run_session_node


def test_agent_message_cursor_respects_optimistic_streaming_and_report_draft_state():
    run_node(r'''
import {latestPendingReportDraftMessageId} from './marvis/static/js/report-draft-table.js';
const taskRequests = taskSession.requests;
taskSession.selectTask({id:'task-1',task_type:'modeling',run_mode:'agent'});
const set = messages => taskSession.replaceMessages(taskRequests.capture(),messages);
const stable = {id:'message / 1',role:'assistant',content:'已记录',metadata:{}};
assert.equal(agentMessageCanPollIncrementally(),false);
set([stable]);
assert.equal(agentMessageCanPollIncrementally(),true);
assert.equal(agentMessageCanPollIncrementally({preserveOptimistic:true}),false);
for (const key of ['optimistic','streaming']) {
  set([{...stable,metadata:{[key]:true}}]);
  assert.equal(agentMessageCanPollIncrementally(),false);
}
set([{...stable,stage:'word_conclusion_draft',metadata:{draft_values:{'TEXT:summary':'待确认'}}}]);
assert.equal(agentMessageCanPollIncrementally(),false);
set([stable]);
const urls=[];let response={incremental:true,messages:[stable,{id:'second',role:'assistant',metadata:{}}]};
const api=async url=>{urls.push(url);return response;};
const isWorkbenchTaskId=id=>id===taskSession.taskId;
const findTaskInCache=()=>taskSession.task;
const selectedTaskIsAgentMode=()=>true;
const taskUsesPlanRail=()=>true;
let renders=0;const renderAgentConversation=()=>{renders++;};
await loadAgentMessages('task-1');
assert.equal(urls[0],'api/tasks/task-1/agent/messages?after_id=message%20%2F%201');
assert.deepEqual(taskSession.messages.map(m=>m.id),['message / 1','second']);
assert.equal(renders,1);
await loadAgentMessages('task-1');
assert.equal(renders,1,'an unchanged incremental response never repaints');
set([{...stable,metadata:{streaming:true}}]);
response={incremental:false,messages:[{id:'final',role:'assistant',metadata:{}}]};
await loadAgentMessages('task-1');
assert.equal(urls.at(-1),'api/tasks/task-1/agent/messages');
assert.deepEqual(taskSession.messages.map(m=>m.id),['final']);
''', 'function agentMessageCanPollIncrementally(', 'function mergeIncrementalAgentMessages(', 'async function loadAgentMessages(')


def test_plan_read_failure_is_visible_and_retry_replaces_it_with_authoritative_empty_state():
    run_session_node(PLAN_HARNESS + r'''
controller.resetFetchThrottle();
const pending=controller.maybeFetchPlan();
calls.at(-1).reject(new Error('network unavailable'));
await pending;
controller.render({force:true,renderSignatures:{}});
const failed=elements.get('workflowStepper').innerHTML;
assert.match(failed,/计划读取失败/);
assert.doesNotMatch(failed,/data-plan-rail-retry|上次缓存|skeleton/);
controller.resetFetchThrottle();
const recovered=controller.maybeFetchPlan();
calls.at(-1).resolve({plans:[]});await recovered;
controller.render({force:true,renderSignatures:{}});
assert.doesNotMatch(elements.get('workflowStepper').innerHTML,/计划读取失败|skeleton/);
assert.equal(session.plan,null);
''')


def test_composer_model_options_keep_existing_nodes_and_focused_choice_during_poll():
    run_node(r'''
const llmSettings={default_model_id:'a',enabled_models:[{model_id:'a'},{model_id:'b'}]};
let agentSelectedModelId='b';let writes=0;
const select={dataset:{},options:[],value:'',appendChild(option){this.options.push(option);if(option.selected)this.value=option.value;},
  set innerHTML(value){writes++;this.options=[];}};
const configure={hidden:false};
const $=id=>id==='agentModelSelect'?select:id==='configureAgentModelButton'?configure:null;
globalThis.document={activeElement:null,createElement:()=>({})};
renderAgentModelOptions();const original=[...select.options];
assert.equal(writes,1);assert.equal(select.value,'b');assert.equal(configure.hidden,true);
renderAgentModelOptions();assert.equal(writes,1);assert.equal(select.options[0],original[0]);
document.activeElement=select;select.value='a';renderAgentModelOptions();
assert.equal(select.value,'a','polling cannot replace a selection while it has focus');
document.activeElement=null;renderAgentModelOptions();assert.equal(select.value,'b');
llmSettings.enabled_models=[{model_id:'a'}];renderAgentModelOptions();
assert.equal(writes,2);assert.equal(select.value,'a','disabled saved model falls back to an enabled model');
llmSettings.enabled_models=[];renderAgentModelOptions();
assert.equal(select.disabled,true);assert.equal(configure.hidden,false);assert.equal(select.options[0].value,'');
''', 'function renderAgentModelOptions(', 'function agentPreferredModelId(')


def test_task_labels_and_acceptance_chips_use_current_task_values():
    run_node(r'''
import {formatValidationBatchTaskName,isValidationBatchTask} from './marvis/static/js/validation-batch.js';
assert.equal(taskDisplayName({model_name:' 用户模型 ',model_version:''}),'用户模型');
assert.equal(taskDisplayName({model_name:'用户模型',model_version:'v2'}),'用户模型 · v2');
assert.equal(taskDisplayName({task_type:'validation_batch',created_at:'2026-11-04',item_count:3}),formatValidationBatchTaskName('2026-11-04',3));
const chip={dataset:{}},option={};const select={value:'',closest:()=>chip,querySelector:()=>option};
const $=()=>select;let agentAcceptanceMode='auto_accept';const normalizeAgentAcceptanceMode=value=>value;
for(const [type,label] of [['data_join','自动拼接'],['feature_analysis','自动分析'],['modeling','自动建模'],['portfolio','自动组合分析'],['validation','自动审查']]) {
  taskSession.selectTask({id:type,task_type:type});renderAgentAcceptanceModePreference();
  assert.equal(option.textContent,label);assert.equal(chip.dataset.acceptanceMode,'auto_accept');
}
const planRailController={planStep:(meta,id)=>{assert.equal(id,'validation');return {phase:'登记阶段',title:'原始步骤'};}};
const agentStageLabel=()=> 'Agent';
assert.equal(agentMessageMetaLabel({metadata:{step_id:'s',run_seq:2}}),'Agent · 登记阶段 · 原始步骤 · 第 2 轮');
assert.equal(agentMessageMetaLabel({metadata:{phase:'用户阶段',step_title:'用户步骤',run_seq:0}}),'Agent · 用户阶段 · 用户步骤');
''', 'function taskDisplayName(', 'function displayModelVersion(', 'function autoAcceptLabel(',
             'function renderAgentAcceptanceModePreference(', 'function agentMessageMetaLabel(', 'function agentMessagePlanStep(')


def test_pet_reaction_expiry_cannot_reset_a_newer_task_reaction():
    run_node(r'''
const PET_REACTION_DURATION_MS=6500,petReactionMoods=new Set(['success','failed','complete','review']);
let petReactionMood=null,petReactionKey='',petReactionTimer=null,mood='success';
const timers=[];globalThis.setTimeout=(callback,delay)=>{timers.push({callback,delay});return timers.length;};
globalThis.clearTimeout=id=>calls.push(['cancel',id]);
const basePetMoodFromTask=()=>mood,renderPetState=()=>calls.push('render');
taskSession.selectTask({id:'a',status:'succeeded',updated_at:'one'});
assert.equal(petMoodFromTask(),'success');assert.equal(timers.length,1);assert.equal(timers[0].delay,6500);
assert.equal(petMoodFromTask(),'success');assert.equal(timers.length,1);
taskSession.refreshTask({id:'a',status:'succeeded',updated_at:'two'});petMoodFromTask();
timers[0].callback();assert.equal(petReactionMood,'success','an earlier timer cannot clear a newer reaction');
timers[1].callback();assert.equal(petMoodFromTask(),'idle');
mood='running';assert.equal(petMoodFromTask(),'running');assert.equal(petReactionKey,'');
''', 'function clearPetReactionTimer(', 'function petReactionKeyForMood(', 'function schedulePetReactionReset(', 'function petMoodFromTask(')


def test_pet_restore_uses_workspace_offset_and_clamps_stale_coordinates():
    run_node(r'''
const petPositionStorageKey='position';let stored=JSON.stringify({left:-999,top:9999,workspaceOffsetLeft:-999});
const localStorage={getItem:()=>stored,setItem:(key,value)=>{assert.equal(key,'position');stored=value;}};
const workspace={getBoundingClientRect:()=>({left:220,top:20})};
const properties={};const pet={offsetWidth:104,offsetHeight:116,style:{setProperty:(key,value)=>properties[key]=value}};
const $=id=>id==='validationWorkspace'?workspace:pet;
Object.assign(window,{innerWidth:1600,innerHeight:1000});
const petCssPx=()=>14,clamp=(value,min,max)=>Math.max(min,Math.min(value,max));
restorePetPosition();
assert.deepEqual(JSON.parse(stored),{left:234,top:870,workspaceOffsetLeft:14});
assert.equal(properties['--pet-offset-left'],'14px');assert.equal(pet.style.left,'');assert.equal(pet.style.top,'870px');
stored=JSON.stringify({left:999,top:50,workspaceOffsetLeft:100});restorePetPosition();
assert.equal(JSON.parse(stored).left,320,'saved workspace offset takes precedence over obsolete absolute left');
stored='malformed';assert.doesNotThrow(()=>restorePetPosition());
''', 'function applyPetPosition(', 'function savePetPosition(', 'function restorePetPosition(', 'function petDragBounds(', 'function clampPetPosition(')


def test_composer_preferences_persist_by_task_and_preserve_sibling_updates():
    run_node(r'''
const storage=new Map();const localStorage={getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)};
const agentComposerPreferenceStorageKey='global',agentTaskComposerStorageKey='tasks';
storage.set('global',JSON.stringify({model_id:'seed',effort:'low',acceptance_mode:'normal'}));
const agentComposerPreferences=restoreAgentComposerPreferences();
let agentSelectedModelId='',agentSelectedEffort='',agentAcceptanceMode='';
let agentTaskComposerOverrides=loadAgentTaskComposerOverrides();
const usesAgentValidationWorkbench=()=>false,findTaskInCache=()=>null;
taskSession.selectTask({id:'a'});applyAgentTaskComposerPreferences('a');
assert.deepEqual([agentSelectedModelId,agentSelectedEffort],['seed','low']);
persistCurrentAgentComposerPreference({model_id:'A',effort:'medium',acceptance_mode:'auto_accept'});
storage.set('tasks',JSON.stringify({...JSON.parse(storage.get('tasks')),b:{model_id:'B',effort:'high'}}));
persistCurrentAgentComposerPreference({effort:'low'});
agentTaskComposerOverrides=loadAgentTaskComposerOverrides();
applyAgentTaskComposerPreferences('b');assert.deepEqual([agentSelectedModelId,agentSelectedEffort],['B','high']);
applyAgentTaskComposerPreferences('a');assert.deepEqual([agentSelectedModelId,agentSelectedEffort,agentAcceptanceMode],['A','low','auto_accept']);
resetAgentComposerToGlobalDefaults();assert.deepEqual([agentSelectedModelId,agentSelectedEffort],['seed','low']);
assert.equal(JSON.parse(storage.get('global')).model_id,'seed');
storage.set('tasks','broken json');assert.deepEqual(loadAgentTaskComposerOverrides(),{});
storage.set('global','broken json');assert.deepEqual(restoreAgentComposerPreferences(),{model_id:'',effort:'high',acceptance_mode:'normal'});
''', 'function normalizeAgentEffort(', 'function normalizeAgentAcceptanceMode(', 'function restoreAgentComposerPreferences(', 'function loadAgentTaskComposerOverrides(', 'function persistAgentTaskComposerOverrides(', 'function getAgentTaskComposerOverride(', 'function updateAgentTaskComposerOverride(', 'function persistCurrentAgentComposerPreference(', 'function applyAgentTaskComposerPreferences(', 'function resetAgentComposerToGlobalDefaults(')


def test_status_snapshots_keep_stage_detail_and_failures_do_not_clear_success():
    run_node(r'''
const taskStopped=()=>false,taskUsesPlanRail=()=>false,taskPlanWorkflowStatusSnapshot=()=>null;
const usesPmmlScoringWorkflow=task=>task.workflow_version==='v2',taskFailureStage=task=>task.failure_stage;
const setActionStatus=(...args)=>calls.push(args);
for(const [status,kind,detail] of [['created','info','等待材料识别'],['scanned','success','下一步：PMML打分测试'],['running','busy','PMML解析与全量评分'],['executed','success','下一步：模型效果'],['computing_metrics','busy','模型效果、稳定性与压力测试'],['writing_artifacts','success','下一步：报告输出'],['succeeded','success','已完成：材料识别、PMML打分、模型验证与报告输出'],['review_required','review','等待人工复核']]){
 const snapshot=taskActionStatusSnapshot({status,workflow_version:'v2'});assert.equal(snapshot.kind,kind);assert.ok(snapshot.detail.includes(detail));
}
assert.equal(taskActionStatusSnapshot({status:'writing_artifacts',active_job_kind:'report'}).kind,'busy');
assert.equal(taskActionStatusSnapshot({status:'running',workflow_version:'v1'}).detail,'当前步骤：执行 Notebook 与分数一致性验证。');
assert.equal(setTaskFailureActionStatus({status:'succeeded'}),false);assert.equal(calls.length,0);
setTaskFailureActionStatus({status:'failed',failure_stage:'metrics',status_message:'bad input: 用户字段'});
assert.deepEqual(calls.at(-1),['模型效果&稳定性验证失败。','error','bad input: 用户字段']);
setTaskFailureActionStatus({status:'review_required'});assert.equal(calls.at(-1)[1],'review');
''','function taskActionStatusSnapshot(','function taskFailureActionStatusMessage(','function taskFailureActionStatusTitle(','function setTaskFailureActionStatus(')


def test_create_flow_routes_manual_scan_agent_composer_and_preserves_typed_goal():
    run_node(r'''
let createTaskInFlight=false;let task={id:'a',task_type:'validation',run_mode:'manual'};
const createTask=async()=>{calls.push('create');taskSession.selectTask(task);return task;};
const ensureValidationMaterialSelection=async value=>value, usesAgentValidationWorkbench=()=>false;
const createTaskDialog={activeTaskType:()=>task.task_type};const defaultTaskType='validation';
const setCreateTaskSubmitting=value=>calls.push(['submit',value]);const setCreateStatus=(...args)=>calls.push(args);
const setActionStatus=(...args)=>calls.push(['status',...args]);
const claimBusy=()=>({id:'busy'}),releaseBusy=()=>calls.push('released');
const scanCurrentTask=async()=>calls.push('scan'),loadTaskEvidence=async id=>calls.push(['evidence',id]);
const loadAgentMessages=async id=>calls.push(['messages',id]),renderAll=()=>calls.push('render');
const taskUsesPlanRail=value=>value.task_type!=='validation';
const taskTypeDefinition=()=>({initialGoal:'先分析模型',label:'建模'});
const sendFirstAgentMessage=async()=>false,dispatchDriverStart=async id=>calls.push(['driver',id]);
const input={value:'',focus:()=>calls.push('focus')};const $=()=>input;
const autoGrowComposerInput=()=>calls.push('grow'),updateAgentSendDisabled=()=>{};
await createTaskAndScan();assert.ok(calls.indexOf('create')<calls.indexOf('scan'));assert.ok(calls.indexOf('scan')<calls.findIndex(x=>Array.isArray(x)&&x[0]==='evidence'));assert.equal(createTaskInFlight,false);
calls.length=0;task={id:'b',task_type:'modeling',run_mode:'agent'};
prefillAgentTaskInstruction(task);assert.equal(input.value,'先分析模型');
input.value='用户目标';prefillAgentTaskInstruction(task);assert.equal(input.value,'用户目标');
await createTaskAndScan();assert.ok(calls.includes('focus'));assert.ok(!calls.includes('scan'));assert.ok(!calls.some(x=>Array.isArray(x)&&x[0]==='driver'));
calls.length=0;task={id:'c',task_type:'modeling',run_mode:'manual'};
await createTaskAndScan();assert.ok(calls.some(x=>Array.isArray(x)&&x[0]==='driver'&&x[1]==='c'));assert.ok(!calls.includes('scan'));
''','async function createTaskAndScan(','function prefillAgentTaskInstruction(')


def test_initialization_restores_evidence_and_awaits_each_context_panel_once():
    run_node(r'''
const refreshTasks=async()=>{calls.push('refresh');taskSession.selectTask({id:'saved',task_type:'strategy'});};
const reloadDataWorkspace=async id=>calls.push(['workspace',id]);
const validationBatchPanelController={selectTask:async task=>calls.push(['batch',task.id]),focusRequestedItem:()=>calls.push('focus')};
const strategyCandidateLabController={selectTask:async task=>calls.push(['lab',task.id])};
const strategyHistoryMappingPanel={selectTask:async()=>{}};
const renderStoredStateSummaries=()=>calls.push('stored');const loadReportFields=async()=>calls.push('fields');
const loadTaskEvidence=async()=>calls.push('evidence'),loadAgentMessages=async()=>calls.push('messages');
const renderAll=()=>calls.push('render'),restoreResultScrollPositionAfterRender=async id=>calls.push(['scroll',id]);
const finishAppBoot=()=>calls.push('boot'),maybeResumeAgentValidationBatch=()=>calls.push('resume');
const setActionStatus=()=>{throw new Error('unexpected initialization error');},setCreateStatus=()=>{};
await initializeApp();assert.equal(calls[0],'refresh');assert.deepEqual(calls.filter(x=>Array.isArray(x)&&x[0]==='lab'),[['lab','saved']]);
assert.ok(calls.indexOf('evidence')<calls.indexOf('messages'));assert.ok(calls.indexOf('messages')<calls.indexOf('render'));assert.equal(calls.at(-1),'resume');
''','async function initializeApp(')


def test_selection_restore_uses_saved_identity_and_same_task_keeps_live_state():
    run_node(r'''
const taskRequests=taskSession.requests;const taskCache=[{id:'a',task_type:'modeling'},{id:'saved',task_type:'modeling'}];
let saved='saved';const storedSelectedTaskId=()=>saved,rememberSelectedTaskId=id=>{saved=id;};
const setSelectedTask=task=>taskSession.selectTask(task);const applyAgentTaskComposerPreferences=id=>calls.push(['preferences',id]);
const prepareResultScrollRestoreForTask=id=>calls.push(['restore',id]);
syncSelectedTaskFromCache();assert.equal(taskSession.taskId,'saved');assert.deepEqual(calls,[['preferences','saved'],['restore','saved']]);
const visit=taskRequests.capture();taskSession.panels.metrics.values={ks:0.4};
const rememberResultScrollPosition=()=>{},renderCurrentTask=()=>{},renderTaskList=()=>{};
const validationBatchPanelController={selectTask:()=>calls.push('batch')};
const strategyCandidateLabController={renderAvailability:()=>{},refresh:()=>{}};
const labelingSetupPanel={renderAvailability:()=>{}},portfolioSetupPanel=labelingSetupPanel,modelMonitoringPanel=labelingSetupPanel;
selectTask({...taskCache[1],status:'executed'});
assert.equal(taskRequests.current(visit),true);assert.equal(taskSession.task.status,'executed');assert.deepEqual(taskSession.panels.metrics.values,{ks:0.4});
taskSession.selectTask(null);saved='missing';syncSelectedTaskFromCache();assert.equal(taskSession.taskId,null);assert.equal(saved,null);
taskSession.selectTask(null);saved=null;syncSelectedTaskFromCache();assert.equal(taskSession.taskId,null,'never pick the first unrelated cached task');
''','function syncSelectedTaskFromCache(','function selectTask(')


def test_result_dataset_download_requires_exact_task_and_complete_provenance():
    run_node(r'''
import {safeSameOriginApiHref} from './marvis/static/js/url-safety.js';
const escapeHtml=value=>String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;');
window.location={origin:'http://localhost',href:'http://localhost/'};
taskSession.selectTask({id:'task'});
const make=url=>({metadata:{result_dataset:{dataset_id:'ds',download_url:url,title:'<unsafe>'}}});
const url='/api/tasks/task/datasets/ds/download?plan_id=p&step_id=s&output_ref=o&expected_content_hash=hash';
const html=agentMessageResultDatasetHtml(make(url));assert.ok(html.includes('data-result-dataset-download'));assert.ok(html.includes('&lt;unsafe>'));assert.ok(html.includes('expected_content_hash=hash'));
for(const invalid of [url.replace('task/datasets','other/datasets'),url.replace('ds/download','other/download'),url.replace('expected_content_hash=','missing='),'https://evil.invalid'+url,'javascript:alert(1)']) assert.equal(agentMessageResultDatasetHtml(make(invalid)),'');
''','function agentMessageResultDatasetHtml(')


def test_execution_environment_reads_saves_and_reverts_failed_optimistic_selection():
    run_node(r'''
const defaultExecutionEnvironment={execution_mode:'local',kernel_name:'python3',conda_env_name:'',python_executable:''};
let executionEnvironmentSettings={...defaultExecutionEnvironment},executionEnvironmentOptions=[{id:'first'}];
const classes=new Set();const $=()=>({classList:{add:value=>classes.add(value),remove:value=>classes.delete(value)}});
const setExecutionEnvironmentStatus=(...args)=>calls.push(['status',...args]);
const populateExecutionEnvironmentForm=(settings,options)=>calls.push(['form',settings,options]);
const renderExecutionEnvironmentValidation=value=>calls.push(['validation',value]);
const workerIsolationPanel={refresh:async()=>calls.push('isolation')};
let fail=false;const api=async(url,options)=>{calls.push(['api',url,options]);if(fail)throw new Error('保存失败');return {settings:{execution_mode:'conda_env',conda_env_name:'py_313'},options:[{id:'env'}],validation:{valid:true}};};
await loadExecutionEnvironmentSettings();assert.equal(calls[0][1],'/api/settings/execution-environment/options');assert.equal(calls[1][1].conda_env_name,'py_313');
await saveExecutionEnvironmentSettings({execution_mode:'python_executable',python_executable:'/usr/bin/python3'});
const request=calls.find(x=>Array.isArray(x)&&x[0]==='api'&&x[2]?.method==='PUT');
assert.equal(request[1],'/api/settings/execution-environment');assert.deepEqual(JSON.parse(request[2].body),{...defaultExecutionEnvironment,execution_mode:'python_executable',python_executable:'/usr/bin/python3'});assert.ok(calls.includes('isolation'));assert.equal(classes.size,0);
calls.length=0;fail=true;await saveExecutionEnvironmentSettings({execution_mode:'conda_env'});
assert.deepEqual(calls.find(x=>Array.isArray(x)&&x[0]==='form')[1],executionEnvironmentSettings);assert.equal(calls.at(-1)[2],'error');assert.equal(classes.size,0);
''','async function loadExecutionEnvironmentSettings(','async function saveExecutionEnvironmentSettings(')


def test_api_requests_stay_below_mount_and_multipart_keeps_browser_content_type():
    run_node(r'''
globalThis.document={baseURI:'https://marvis.test/proxy/9000/',querySelector:()=>null,documentElement:{lang:'zh'}};
const {api}=await import('./marvis/static/js/api.js');
globalThis.fetch=async(url,options)=>{calls.push({url,options});return {ok:true,status:200,json:async()=>({ok:true})};};
await api('/api/tasks',{method:'POST',body:JSON.stringify({name:'user'})});
assert.equal(calls[0].url,'https://marvis.test/proxy/9000/api/tasks');assert.equal(calls[0].options.headers['Content-Type'],'application/json');
const form=new FormData();form.append('files',new Blob(['x']),'data.csv');await api('api/material-uploads',{method:'POST',body:form});
assert.equal(calls[1].url,'https://marvis.test/proxy/9000/api/material-uploads');assert.equal(calls[1].options.body,form);assert.equal(calls[1].options.headers['Content-Type'],undefined);
await assert.rejects(api('//evil.invalid/api'),/invalid API endpoint/);assert.equal(calls.length,2);
''')


def test_trend_renderer_generates_local_svg_and_escapes_untrusted_labels():
    run_node(r'''
const {renderTrendTable}=await import('./marvis/static/js/metric-tables.js');
const html=renderTrendTable({headers:['月份','KS'],rows:[['<script>x</script>',0.1],['2026-02',0.5]],column_specs:[{kind:'text'},{kind:'number'}]});
assert.ok(html.includes('<svg'));assert.ok(!html.includes('<script>'));assert.ok(html.includes('&lt;script&gt;'));
''')


def test_task_switch_clears_typing_cancels_old_follow_frame_and_loads_new_evidence():
    run_node(r'''
const taskRequests=taskSession.requests;taskSession.selectTask({id:'a'});
let agentAutoScrollFollows=false,agentAutoScrollFrame=7,agentTypingTimer=9;
const agentTypingState=new Map([['old',{text:'old'}]]),agentTypingCompleted=new Set(['old']);
window.cancelAnimationFrame=id=>calls.push(['cancelFrame',id]);window.clearTimeout=id=>calls.push(['clearTimer',id]);
const rememberResultScrollPosition=()=>{},rememberSelectedTaskId=()=>{},applyAgentTaskComposerPreferences=()=>{};
const setSelectedTask=value=>taskSession.selectTask(value),invalidateAgentBatchAutoRun=()=>{};
const beginTaskContentLoad=()=>{},ensureActiveTaskProgressPolling=()=>{},renderMetricPreview=()=>{},renderStoredStateSummaries=()=>{};
const validationBatchPanelController={selectTask:async()=>{},focusRequestedItem:()=>{}};
const strategyCandidateLabController={selectTask:async()=>{}};const strategyHistoryMappingPanel={selectTask:async()=>{}};
const pending=[];const runAction=callback=>pending.push(callback());const renderTaskList=()=>{},usesAgentValidationWorkbench=()=>false;
const loadTaskEvidence=async id=>calls.push(['evidence',id]),loadReportFields=async()=>{},loadAgentMessages=async()=>{};
const renderAll=()=>{},restoreResultScrollPositionAfterRender=async()=>{},finishTaskContentLoad=()=>{},maybeResumeAgentValidationBatch=()=>{};
selectTask({id:'b'});await Promise.all(pending);
assert.equal(taskSession.taskId,'b');assert.equal(agentAutoScrollFollows,true);assert.equal(agentAutoScrollFrame,null);assert.equal(agentTypingTimer,null);
assert.equal(agentTypingState.size,0);assert.equal(agentTypingCompleted.size,0);assert.equal(taskSession.panels.scroll.pendingRestoreTaskId,'b');
assert.deepEqual(calls,[['clearTimer',9],['cancelFrame',7],['evidence','b']]);
''','function selectTask(','function resetAgentTypingState(','function prepareResultScrollRestoreForTask(')


def test_manual_history_omits_overview_and_recovery_controls_follow_messages():
    run_node(r'''
const {driverManualAnalysisHtml}=await import('./marvis/static/js/v2/driver_manual_analysis.js');
const html=driverManualAnalysisHtml([{id:'o',role:'assistant',content:'OVER',metadata:{kind:'overview'}},{id:'p',role:'assistant',content:'PLAN',metadata:{kind:'plan_overview'}},{id:'ok',role:'assistant',content:'RESULT',metadata:{}}],{renderAgentMarkdown:value=>`<p>${value}</p>`});
assert.ok(!html.includes('OVER'));assert.ok(!html.includes('PLAN'));assert.ok(html.includes('RESULT'));
const order=agentPersistentTimelineElementIds();assert.ok(order.indexOf('agentConversationPanel')<order.indexOf('planRetryPanel'));assert.ok(order.indexOf('planRetryPanel')<order.indexOf('planDriverActions'));
''','function agentPersistentTimelineElementIds(')


def test_agent_dispatch_requires_task_identity_before_request_or_busy_claim():
    run_node(r'''
const claimBusy=()=>{throw new Error('must not claim without a task');};
await assert.rejects(dispatchAgentValidation('  '),/Agent 初始化缺少任务 ID/);
assert.equal(requireTaskId(' task '),'task');
''','async function dispatchAgentValidation(','function requireTaskId(')


def test_agent_wait_keeps_busy_lease_until_job_and_message_stream_settle():
    run_node(r'''
const taskRequests=taskSession.requests;taskSession.selectTask({id:'a'});
const claimBusy=()=>({id:'lease'}),releaseBusy=()=>calls.push('release');
const isWorkbenchTaskId=id=>id==='a',setActionStatus=(...args)=>calls.push(['status',...args]);
let finish;const completion=new Promise(resolve=>{finish=resolve;});
const pollValidationProgress=(statuses,id,options)=>{assert.equal(id,'a');assert.ok(statuses.has('executed'));assert.deepEqual(options,{stopping:true,settleWhenServerIdle:true});return completion;};
const pollAgentMessagesUntilSettled=async(_id,promise)=>{await promise;calls.push('stream');};
const loadAgentMessages=async()=>calls.push('messages'),loadReportFields=async()=>calls.push('fields');
const wait=waitForAgentValidation('a',{stopping:true});await Promise.resolve();assert.ok(!calls.includes('release'));
finish({id:'a',status:'executed',active_job_kind:null});await wait;
assert.equal(calls.at(-1),'release');assert.ok(calls.indexOf('stream')<calls.indexOf('messages'));assert.ok(calls.some(item=>Array.isArray(item)&&item[2]==='stopped'));
''','async function waitForAgentValidation(')


def test_scan_and_report_visibility_follow_current_workflow_and_real_evidence():
    run_node(r'''
const classes=new Set(['empty']);const hidden=new Set();const attributes={};
const elements={scanSummary:{textContent:'',classList:{contains:value=>classes.has(value)}},scanSection:{classList:{add:value=>hidden.add(value),remove:value=>hidden.delete(value),toggle:(value,on)=>on?hidden.add(value):hidden.delete(value)}},reportSection:{setAttribute:(name,value)=>attributes[name]=value},agentReportLeadMessages:{children:[]},agentReportMessages:{children:[]}};
const $=id=>elements[id];const workbenchTask=()=>taskSession.task;
const taskUsesPlanRail=task=>task.task_type==='modeling',selectedTaskIsAgentMode=task=>task.run_mode==='agent';
taskSession.selectTask({id:'a',task_type:'validation',run_mode:'agent'});updateAgentScanSectionVisibility();assert.ok(hidden.has('hidden'));
elements.scanSummary.textContent='已识别用户材料';classes.clear();updateAgentScanSectionVisibility();assert.ok(!hidden.has('hidden'));
taskSession.selectTask({id:'b',task_type:'modeling',run_mode:'manual'});updateAgentScanSectionVisibility();assert.ok(hidden.has('hidden'));
taskSession.selectTask({id:'c',task_type:'validation',run_mode:'manual'});updateAgentScanSectionVisibility();assert.ok(!hidden.has('hidden'));
updateAgentReportSectionVisibility();assert.equal(attributes['aria-hidden'],'true');elements.agentReportMessages.children.push({});updateAgentReportSectionVisibility();assert.equal(attributes['aria-hidden'],'false');
''','function updateAgentScanSectionVisibility(','function scanSummaryHasResult(','function updateAgentReportSectionVisibility(')
