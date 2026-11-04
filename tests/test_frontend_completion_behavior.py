"""F13 app-consumer regressions: execute installed functions with observable boundaries."""

from pathlib import Path
import subprocess

from tests.javascript_source import slice_function


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
CREATE = (ROOT / "marvis/static/js/create-task-dialog.js").read_text(encoding="utf-8")


def _nested(signature):
    start = CREATE.index("  " + signature)
    return CREATE[start:CREATE.index("\n  }", start) + 4]


def _callback(start, end):
    return APP.split(start, 1)[1].split(end, 1)[0]


def _run(body, *functions, declarations=""):
    script = """
import assert from 'node:assert/strict';
import {createTaskSession} from './marvis/static/js/task-session.js';
import {escapeHtml} from './marvis/static/js/ui-utils.js';
const taskSession=createTaskSession();
const calls=[];
""" + "\n".join(slice_function(APP, name) for name in functions) + declarations + body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_app_branding_load_applies_safe_assets_titles_aliases_and_failure_defaults():
    _run(r'''
await import('./marvis/static/js/pet_catalog.js');
const {applyBranding,normalizeBranding}=await import('./marvis/static/js/branding.js');
const {defaultBranding}=await import('./marvis/static/js/state.js');
const nodes=new Map();const colors={};
const element=()=>({textContent:'',children:[],append(...children){this.children.push(...children);}});
for(const id of ['platformName','brandLogo','workspaceBrandLogo','brandFavicon','brandFaviconDark','brandAppleTouchIcon','brandAppleTouchIconDark']) nodes.set(id,element());
globalThis.document={title:'',getElementById:id=>nodes.get(id),createElement:element,
  documentElement:{style:{setProperty:(name,value)=>colors[name]=value}}};
let agentValidatorAliases={};let fail=false;
const api=async url=>{assert.equal(url,'api/branding');if(fail)throw Error('offline');return {
  platformName:'客户—原始名称',browserTitle:'客户标题',primaryColor:'#123456',
  logoUrl:'/assets/logo.svg',workspaceLogoUrl:'/assets/hero.png',faviconUrl:'javascript:bad',
  validatorAliases:{'验收人员':'本地别名'}};};
await loadBranding();
assert.equal(document.title,'客户标题');assert.equal(nodes.get('brandLogo').src,'/assets/logo.svg');
assert.equal(nodes.get('workspaceBrandLogo').src,'/assets/hero.png');
assert.equal(nodes.get('brandFavicon').href,defaultBranding.faviconUrl);
assert.deepEqual(nodes.get('platformName').children.map(node=>node.textContent),['客户','原始名称']);
assert.deepEqual(agentValidatorAliases,{'验收人员':'本地别名'});assert.equal(colors['--brand-primary'],'#123456');
fail=true;await loadBranding();assert.equal(document.title,defaultBranding.browserTitle);
assert.equal(nodes.get('brandLogo').src,defaultBranding.logoUrl);
''', 'async function loadBranding(')


def test_report_message_actions_follow_actual_readiness_and_scoped_click_dispatch():
    _run(r'''
let busy='';const task={id:'child',status:'succeeded',report_available:true};
const workbenchTask=()=>task;const taskBusyAction=()=>busy;
const message={stage:'word_report_ready',metadata:{report_ready:true}};
assert.equal(agentMessageValidationArtifactsHtml({stage:'scan'}),'');
assert.equal(agentMessageValidationArtifactsHtml({...message,metadata:{report_ready:false}}),'');
busy='report';assert.equal(agentMessageValidationArtifactsHtml(message),'');busy='';
task.report_available=false;assert.equal(agentMessageValidationArtifactsHtml(message),'');task.report_available=true;
task.status='running';assert.equal(agentMessageValidationArtifactsHtml(message),'');task.status='succeeded';
const html=agentMessageValidationArtifactsHtml(message);
const actionIds=[...html.matchAll(/data-artifact-action="([^"]+)"/g)].map(match=>match[1]);
assert.deepEqual(actionIds,['previewWordReport','downloadWordReport','downloadExcelAnalysis']);
let inside=true;const $=()=>({contains:()=>inside});
const workflowActionConfig=id=>({action:id,busyText:'等待'});
const runAction=(action,options)=>calls.push([action,options]);
for(const id of actionIds) handleArtifactActionClick({target:{closest:()=>({dataset:{artifactAction:id}})},preventDefault(){}});
assert.deepEqual(calls.map(call=>call[1].actionId),actionIds);
inside=false;handleArtifactActionClick({target:{closest:()=>({dataset:{artifactAction:'downloadWordReport'}})},preventDefault(){throw Error('outside');}});
assert.equal(calls.length,3);
''', 'function completedReportReadyForDownloads(', 'function artifactCardHtml(',
         'function agentMessageValidationArtifactsHtml(', 'function handleArtifactActionClick(')


def test_report_readiness_change_rebuilds_the_same_message_with_real_artifact_markup():
    _run(r'''
const empty=()=>'',noop=()=>{};
const task={id:'single',task_type:'validation',run_mode:'agent',status:'succeeded',report_available:false};
taskSession.selectTask(task);taskSession.replaceMessages(taskSession.requests.capture(),[
 {id:'ready-message',role:'assistant',stage:'word_report_ready',metadata:{report_ready:true},content:'完成'}]);
const workbenchTask=()=>task,taskBusyAction=()=>'';
const node={dataset:{},classList:{toggle(){},remove(){}},setAttribute(){}};const $=()=>node;
const renderReportDraftWorkspace=noop,strategyCandidateLabController={renderAvailability:noop};
const selectedTaskIsAgentMode=()=>true,selectedTaskNeedsManualRiskIntake=()=>false,taskUsesPlanRail=()=>false;
const syncRiskMaterialUploadControl=noop,renderAgentAcceptanceModePreference=noop,renderAgentModelOptions=noop;
const renderAgentEffortPreference=noop,renderAgentQuickReplies=noop,syncAgentComposerClearance=noop;
const filterStates=[];const workbenchChrome={setThreadFilterAvailable:(...args)=>filterStates.push(args)};const requestAnimationFrame=noop;
let lastDriverManualAnalysisSignature=null,lastAgentRenderSignature=null,lastAgentStructuralSignature=null;
const agentTimelineVisibleStages=()=>['report'],agentReportMessagesForDisplay=value=>value,hideSupersededTuningThinking=value=>value;
const agentStructuralSignature=messages=>messages.map(message=>message.id).join('|');let fastPatches=0;
const updateAgentMessageContentsInPlace=()=>{fastPatches++;return true;};
const layout=[];const freezeAgentSectionSnapshotsForReruns=noop,clearAgentStageMessages=noop;
const parkReportDraftCard=()=>layout.push('park'),placeReportDraftCard=()=>layout.push('place');
const syncLiveAgentFaces=noop,liveAgentFace=null,attachCalibrationInteractions=noop,requestAgentConversationScrollToLatest=noop;
const gateMessageForCurrentTool=message=>message;
const hasWorkflowErrorDiagnostic=()=>false,isStrategyClarificationMessageController=()=>false,agentGateUsesConversationOnly=()=>false;
const agentMessageIsStreaming=()=>false,agentMessageIsThinking=()=>false,hasReportDraftValues=value=>Boolean(value&&Object.keys(value).length);
const agentVisibleContent=message=>message.content||'',formatAgentMessageContent=escapeHtml;
const workflowMessageContentHtml=(_message,fallback)=>fallback();
const agentMemoryReferencesHtml=empty,agentThinkingHtml=empty,agentMessageIsResult=()=>true;
const planStepMessageAttrs=empty,renderModelTuningMessageProgress=empty,agentMessageModelDeliveryHtml=empty;
const agentMessageTablesHtml=empty,agentMessageResultDatasetHtml=empty,conversationActionCardHtml=empty;
const agentMessageReportDownloadHtml=empty,agentMessageGateButtonHtml=empty;
const agentFaceHtml=empty,agentStageLabel=empty,agentMessageMetaHtml=empty,agentMessageMetaLabel=empty;
const rendered=[];const renderAgentTimeline=messages=>{layout.push('timeline');rendered.push(messages.map(message=>agentMessageHtml(message)).join(''));};
renderAgentConversation();assert.equal(rendered.length,1);assert.doesNotMatch(rendered[0],/data-artifact-action/);
renderAgentConversation();assert.equal(rendered.length,1,'identical readiness should preserve the existing DOM');
task.report_available=true;renderAgentConversation();
assert.equal(rendered.length,2);assert.match(rendered[1],/data-artifact-action="downloadWordReport"/);
assert.equal(fastPatches,0,'readiness changes require structural rebuild, not a text-only patch');
assert.deepEqual(layout,['park','timeline','place','park','timeline','place']);
assert.deepEqual(filterStates,[[true,'single'],[true,'single'],[true,'single']]);
const draft={id:'draft',role:'assistant',stage:'word_conclusion_draft',metadata:{draft_values:{x:'draft text'}}};
assert.match(agentMessageHtml(draft,undefined,{isLatestReportDraft:false}),/这一版报告草稿已被后面的新版本取代/);
assert.doesNotMatch(agentMessageHtml(draft,undefined,{isLatestReportDraft:false}),/data-report-draft-slot/);
assert.match(agentMessageHtml(draft,undefined,{isLatestReportDraft:true}),/data-report-draft-slot/);
''', 'function completedReportReadyForDownloads(', 'function artifactCardHtml(',
         'function agentMessageValidationArtifactsHtml(', 'function agentMessageHtml(', 'function renderAgentConversation(')


def test_app_table_consumers_use_presented_cells_and_field_lists_without_relabeling_raw_names():
    _run(r'''
import {presentedCellHtml,fieldListModel,fieldListHtml,displayNumberText} from './marvis/static/js/table-presentation.js';
import {renderWorkflowDataWidget} from './marvis/static/js/v2/workflow_widgets.js';
import {resultSummaryHtml} from './marvis/static/js/result-summary.js';
const metricHeaderShouldRightAlign=()=>false;const platformLabelSource=value=>value;
const driverColumnKindFromHeader=()=> 'text';const driverTableChartHtml=()=>'';
assert.match(driverTableCellHtml('candidate',0,1,'状态','text',new Map()).html,/候选/);
assert.equal(driverTableCellHtml('candidate',0,1,'字段名','text',new Map()).html,'candidate');
const html=agentMessageTablesHtml({id:'message',metadata:{tables:[{
  title:'验证字段候选',columns:['验证字段','候选值'],rows:[['algorithm','"lr"'],['model_params','{"algorithm":"lr","intercept":0}']]
}]}});
assert.match(html,/算法/);assert.match(html,/模型参数/);assert.match(html,/截距 0/);
assert.doesNotMatch(html,/<table>|&quot;algorithm&quot;/);
const result={id:'result',metadata:{tables:[{title:'策略回测摘要',columns:['指标','值'],rows:[['审批率','80.0%'],['拒绝人数','48']]}]}};
const summary=agentMessageTablesHtml(result);assert.match(summary,/result-headline/);assert.match(summary,/<details/);
assert.ok(summary.indexOf('result-headline')<summary.indexOf('<details'));
assert.doesNotMatch(agentMessageTablesHtml({...result,metadata:{...result.metadata,kind:'gate'}}),/<details/);
''', 'function driverTableCellHtml(', 'function agentMessageTablesHtml(')


def test_app_batch_poll_refreshes_shared_views_and_ready_copy_without_claiming_other_models():
    refresh = _callback('refreshParentTask: ', ',\n  onRecovered:')
    confirm = _callback('confirmStart: ', ',\n  openContract:')
    _run(r'''
taskSession.selectTask({id:'parent',task_type:'validation_batch'});
let busy='validation_batch';const refreshTasks=async()=>calls.push('tasks');
const renderChangedValidationViews=()=>calls.push('views');const taskServerBusyAction=()=>busy;
assert.equal(await refreshParentTask({parentTaskId:'parent'}),false);
assert.deepEqual(calls,['tasks','views']);busy='';assert.equal(await refreshParentTask({parentTaskId:'parent'}),true);
let confirmation;const showPlatformConfirm=async options=>{confirmation=options;return true;};
await confirmStart({status:'awaiting_confirmation',itemCount:2});
assert.match(confirmation.message,/所有模型输入合同已确认/);assert.match(confirmation.message,/不会自动启动/);
const panel={dataset:{},innerHTML:'',classList:{contains:()=>false,remove(){},add(){}}};
const $=()=>panel;const selectedTaskIsValidationBatch=()=>true;
const validationBatchPanelController={markContractConfirmed:id=>calls.push(['confirmed',id])};
renderValidationInputContract({status:'ready',revision:2},{taskId:'child',parentTaskId:'parent'});
assert.match(panel.innerHTML,/此模型的验证字段已经确认/);
assert.doesNotMatch(panel.innerHTML,/请继续确认其他待确认模型|仍有.*等待/);
assert.deepEqual(calls.at(-1),['confirmed','child']);
''', 'function renderValidationInputContract(',
         declarations=f"const refreshParentTask={refresh};\nconst confirmStart={confirm};\n")


def test_app_batch_delete_routes_preview_and_delete_together_and_fails_closed():
    _run(r'''
const isValidationBatchTask=task=>task.task_type==='validation_batch';
const PURGE_SUMMARY_FIELDS=[['child_tasks','子任务'],['owned_batch_material_trees','上传材料']];
const taskBusyAction=()=>'';const taskServerBusyAction=()=>'';const taskDisplayName=task=>task.id;
const statuses=[];const setActionStatus=(...args)=>statuses.push(args);let fail=false;
const api=async(url,options={})=>{calls.push([url,options.method||'GET']);
  if(fail)throw Error('preview unavailable');
  return {task_count:3,child_task_count:2,active_job_count:0,owned_batch_material_tree_count:1,
    purge_summary:{child_tasks:2,owned_batch_material_trees:1}};};
const showPlatformConfirm=async options=>{assert.match(options.message,/2 个子验证任务/);return true;};
const claimBusy=()=>({});const releaseBusy=()=>{};const renderAll=()=>{};
const resultScrollPositionsByTask=new Map();const persistResultScrollPositions=()=>{};
const refreshTasks=async()=>{};const renderStoredStateSummaries=()=>{};const loadReportFields=async()=>{};
await deleteTask({id:'batch',task_type:'validation_batch'});
assert.deepEqual(calls,[['api/validation-batches/batch/purge-preview','GET'],['api/validation-batches/batch','DELETE']]);
calls.length=0;fail=true;await deleteTask({id:'batch',task_type:'validation_batch'});
assert.deepEqual(calls,[['api/validation-batches/batch/purge-preview','GET']]);
assert.match(statuses.at(-1)[0],/已停止删除/);
assert.equal(taskPurgeApiBase({id:'single',task_type:'validation'}),'api/tasks/single');
''', 'function taskPurgeApiBase(', 'function validatedValidationBatchPurgeSummary(',
         'async function loadTaskPurgeSummary(', 'async function reconcileTaskBeforeDelete(', 'async function deleteTask(')


def test_app_candidate_submit_callback_settles_real_poller_and_refreshes_exactly_once():
    callback = _callback('settleCandidateLabSubmission: ', ',\n  refreshAgentMessages:')
    _run(r'''
import {claimProgressPoll,releaseProgressPoll} from './marvis/static/js/polling.js';
taskSession.selectTask({id:'strategy',task_type:'strategy'});const taskRequests=taskSession.requests;
const progressPolls=new Map();const terminalTaskStatuses=new Set(['succeeded','failed']);
const sleep=async()=>{};let polls=0;const refreshTasks=async()=>{polls++;};
const findTaskInCache=()=>({id:'strategy',task_type:'strategy',status:'created',active_job_kind:polls<2?'agent':''});
const isWorkbenchTaskId=id=>id==='strategy';const loadTaskEvidence=async()=>{};
const metricOverviewComplete=()=>false;const selectedTaskIsAgentMode=()=>false;
const usesAgentValidationWorkbench=()=>false;const renderChangedValidationViews=()=>{};
const taskServerBusyAction=task=>task.active_job_kind;
const strategyCandidateLabController={refresh:async(id,options)=>calls.push([id,options])};
const final=await settle('strategy');assert.equal(final.status,'created');
assert.equal(polls,2,'app wiring must settle when the server is idle even before terminal status');
assert.deepEqual(calls,[['strategy',{silent:true}]]);assert.equal(progressPolls.size,0);
''', 'async function pollValidationProgress(', 'async function refreshStrategyCandidateLabAfterSettled(',
         declarations=f"const settle={callback};\n")


def test_agent_batch_creation_enters_selected_workbench_then_dispatches_continuation():
    _run(r'''
let createTaskInFlight=false;const setCreateTaskSubmitting=value=>calls.push(['submitting',value]);
const setCreateStatus=()=>{};const setActionStatus=(...args)=>calls.push(['status',...args]);
const task={id:'batch',task_type:'validation_batch',run_mode:'agent'};
const createTask=async()=>task;const ensureValidationMaterialSelection=async value=>value;
const createTaskDialog={activeTaskType:()=> 'validation'};const defaultTaskType='validation';
const usesAgentValidationWorkbench=value=>value.task_type==='validation_batch'&&value.run_mode==='agent';
const requestTaskSelection=async value=>calls.push(['selected',value.id]);
const continueAgentValidationBatch=async()=>calls.push(['continue']);
const loadAgentMessages=()=>{throw Error('parent messages must not be requested');};
await createTaskAndScan();
assert.deepEqual(calls.filter(call=>['selected','continue'].includes(call[0])),[['selected','batch'],['continue']]);
assert.equal(createTaskInFlight,false);assert.deepEqual(calls.at(-1),['submitting',false]);
''', 'async function createTaskAndScan(')


def test_batch_stepper_and_report_urls_use_the_projected_child_identity():
    _run(r'''
taskSession.selectTask({id:'parent',task_type:'validation_batch',status:'created',validation_workflow_version:1});
taskSession.selectModel('child');taskSession.projectModel(taskSession.requests.capture(),
 {id:'child',task_type:'validation',status:'succeeded',validation_workflow_version:2});
const usesAgentValidationWorkbench=task=>task?.task_type==='validation_batch';
const element=()=>({dataset:{},childNodes:[],scrollTop:0,innerHTML:'',classList:{remove(){}},
 setAttribute(){},getAttribute(){return '验证步骤';},appendChild(node){this.childNodes.push(node);this.innerHTML+=node.innerHTML;},
 replaceChildren(...nodes){this.childNodes=nodes;this.innerHTML=nodes.map(node=>node.innerHTML).join('');}});
const nodes=new Map(['progressRail','workflowStepper','wordPreviewFrame','wordPreviewTitle','wordPreviewDialog'].map(id=>[id,element()]));
const $=id=>nodes.get(id);globalThis.document={querySelector:()=>({}),createElement:element};
globalThis.window={location:{href:''}};
const planRailController={render:()=>false,clearRetryPanel(){},clearDriverActionsPanel(){}};
const renderSignatures={};const renderTaskArtifacts=()=>{};const syncRailStatus=()=>{};const setTextIfChanged=()=>{};
const workflowStepperSignature=task=>{assert.equal(task.id,'child');return task.id+task.status;};
const workflowIndex=(status,task)=>{assert.equal(task.id,'child');assert.equal(status,'succeeded');return 1;};
const workflowStepStatus=(_index,_active,task)=>{assert.equal(task.id,'child');return 'succeeded';};
const workflowSteps=[{id:'notebook',action:'notebook',title:'Notebook',hint:'执行',target:'notebook'}];
const v2WorkflowSubsteps=()=>[];const stepCheckerHtml=()=>'';const stepActionButtonHtml=()=>'';
const renderNotebookStepRail=()=>'';const localizeHtml=value=>value;const refreshWorkflowStepperElapsedTimes=()=>{};
renderWorkflowStepper();assert.equal(nodes.get('workflowStepper').dataset.taskId,'child');
assert.match(nodes.get('workflowStepper').innerHTML,/PMML打分测试/);
downloadWordReport();assert.equal(window.location.href,'api/tasks/child/report/download');
downloadExcelAnalysis();assert.equal(window.location.href,'api/tasks/child/analysis/download');
const reportTitleForTask=task=>{assert.equal(task.id,'child');return '子模型';};
const getUiLocale=()=> 'zh';const setActionStatus=()=>{};let opened=false;
nodes.get('wordPreviewDialog').showModal=()=>{opened=true;};openWordPreviewDialog();
assert.equal(opened,true);assert.match(nodes.get('wordPreviewFrame').src,/api\/tasks\/child\/report\/preview/);
assert.match(nodes.get('wordPreviewTitle').textContent,/子模型/);
''', 'function workbenchTask(', 'function workbenchTaskId(', 'function usesPmmlScoringWorkflow(',
         'function workflowStepForTask(', 'function renderWorkflowStepper(', 'function downloadWordReport(',
         'function downloadExcelAnalysis(', 'function openWordPreviewDialog(')


def test_primary_creation_classifies_own_uploads_and_returns_parent_identity_or_single_task():
    declarations = "\n".join(_nested(name) for name in (
        'async function createTask()', 'function collectPrimaryValidationModel(',
        'function classifiedFilesFromSelection(', 'function validationModelMaterialsReady(',
        'async function createMultiModelValidationTask(',
    ))
    _run(r'''
import {buildValidationBatchUploadFormData,classifyValidationBatchFiles,submitValidationBatchDraft} from './marvis/static/js/validation-batch-create.js';
import {formatValidationBatchTaskName} from './marvis/static/js/validation-batch.js';
const fields={taskType:{value:'validation'},modelName:{value:'Primary'},validator:{value:'QA'},sourceDir:{value:'/primary'}};
const $=id=>fields[id];globalThis.document={querySelector:()=>({value:'manual'})};
const activeTaskType='validation',defaultTaskType='validation';const taskTypeDefinition=()=>({});
const statuses=[];const setCreateStatus=(...args)=>statuses.push(args);const focusCreateField=()=>{};
let extras=[{id:'extra',modelName:'Extra',modelVersion:'v1',sourceDir:'/extra',files:{}}];
const collectValidationExtraModels=()=>extras;let upload=true;
const files=['primary.ipynb','sample.parquet','score.pmml','数据字典.xlsx'].map(name=>new File(['fixture'],name));
const materialSourceController={mode:()=>upload?'upload':'path',selectedFiles:()=>files};
let response={batch:{parent_task_id:'parent'}};
const api=async(url,options)=>{calls.push([url,options]);
 if(url.endsWith('material-uploads'))return {source_dir:'/uploaded-primary',upload_token:'token-primary'};
 if(url==='api/tasks')return {id:'single',task_type:'validation'};
 return response;};
const created=await createTask();assert.deepEqual(created,{id:'parent',task_type:'validation_batch',run_mode:'agent'});
assert.deepEqual(calls.map(call=>call[0]),['/api/validation-batches/material-uploads','/api/validation-batches']);
const uploaded=[...calls[0][1].body.values()].filter(value=>value instanceof File).map(file=>file.name);
assert.deepEqual(uploaded.sort(),files.map(file=>file.name).sort());
const body=JSON.parse(calls[1][1].body);assert.equal(body.items[0].model_name,'Primary');
assert.equal(body.items[0].source_dir,'/uploaded-primary');assert.equal(body.items[1].source_dir,'/extra');
calls.length=0;response={parent_task:{id:'rich-parent',task_type:'validation_batch',extra:'kept'}};
assert.deepEqual(await createTask(),response.parent_task);
calls.length=0;extras=[];upload=false;
assert.deepEqual(await createTask(),{id:'single',task_type:'validation'});
assert.deepEqual(calls.map(call=>call[0]),['api/tasks']);
assert.equal(JSON.parse(calls[0][1].body).source_dir,'/primary');
''', declarations=declarations)
