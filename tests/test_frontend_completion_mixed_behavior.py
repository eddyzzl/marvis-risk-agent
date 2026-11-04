"""Business interactions previously mixed into HTML/CSS source-contract tests."""

from tests.javascript_source import slice_function
from tests.test_frontend_completion_behavior import APP, ROOT, _run


def test_quick_reply_click_uses_composer_send_and_structured_metadata_marks_results():
    _run(r'''
taskSession.selectTask({id:'task'});let stop=false;const input={value:''};const $=()=>input;
const agentSendIsStopMode=()=>stop,autoGrowComposerInput=()=>{},updateAgentSendDisabled=()=>{};
const workbenchTaskId=()=> 'child';const startAgentValidation=()=>{};
const runAction=(action,options)=>calls.push([action,options]);
const event={target:{closest:()=>({dataset:{agentQuickReply:'  继续验证  '}})},preventDefault(){}};
handleAgentQuickReplyClick(event);assert.equal(input.value,'继续验证');
assert.equal(calls[0][0],startAgentValidation);assert.equal(calls[0][1].taskId,'child');
stop=true;input.value='用户输入';handleAgentQuickReplyClick(event);assert.equal(calls.length,1);assert.equal(input.value,'用户输入');
const hasReportDraftValues=value=>Boolean(value&&Object.keys(value).length);
assert.equal(agentMessageIsResult({role:'assistant',content:'报告已生成',metadata:{}}),false);
assert.equal(agentMessageIsResult({role:'assistant',metadata:{tables:[{}]}}),true);
assert.equal(agentMessageIsResult({role:'user',metadata:{tables:[{}]}}),false);
assert.equal(agentMessageIsResult({role:'assistant',stage:'word_conclusion_draft',metadata:{draft_values:{x:'y'}}}),true);
''', 'function handleAgentQuickReplyClick(', 'function agentMessageIsResult(')


def test_both_app_render_paths_refresh_labeling_portfolio_and_candidate_visibility():
    _run(r'''
const noop=()=>{};
const renderCurrentTask=noop,renderReproducibilitySectionVisibility=noop,renderMetricSectionVisibility=noop;
const renderWorkflowStepper=noop,renderTaskList=noop,renderSettingsState=noop,renderAgentConversation=noop;
const renderPetState=noop,updateAgentSendDisabled=noop;
const validationBatchPanelController={renderVisibility:()=>calls.push('batch')};
const strategyCandidateLabController={renderAvailability:()=>calls.push('candidate')};
const labelingSetupPanel={renderAvailability:()=>calls.push('labeling')};
const portfolioSetupPanel={renderAvailability:()=>calls.push('portfolio')};
const modelMonitoringPanel={renderAvailability:()=>calls.push('monitoring')};
const historicalReplayController={render:noop},riskContextPanel=historicalReplayController;
const institutionAdapterPanel=historicalReplayController,operationsExtensionsPanel=historicalReplayController;
const collectionWorkspaceController=historicalReplayController;
renderAll();renderChangedValidationViews();
assert.deepEqual(calls,['batch','candidate','labeling','portfolio','monitoring','batch','candidate','labeling','portfolio','monitoring']);
''', 'function renderAll(', 'function renderChangedValidationViews(')


def test_welcome_actions_deselect_current_task_or_fill_composer_without_opening_dialog():
    _run(r'''
const $=()=>({open:false});const closeTaskDialog=()=>{throw Error('unexpected dialog');};
const validationBatchCreateController={close(){throw Error('unexpected batch dialog');}};
const deselectCurrentTask=()=>calls.push('deselect'),rememberSelectedTaskId=id=>calls.push(['remember',id]);
const setActionStatus=()=>{},renderCurrentTask=()=>calls.push('render'),renderTaskList=()=>{};
taskSession.selectTask({id:'current'});openTaskTypeWelcome();assert.deepEqual(calls,['deselect']);
taskSession.selectTask(null);openTaskTypeWelcome();assert.deepEqual(calls.slice(1),[['remember',null],'render']);
const defaultTaskType='validation';const taskTypeDefinition=()=>({available:true});
const homeComposer={pickTaskType:event=>calls.push(['pick',event.target.kind])};
pickWelcomeTaskType({target:{kind:'portfolio',closest:()=>({dataset:{taskKind:'portfolio'}})}});
assert.deepEqual(calls.at(-1),['pick','portfolio']);
''', 'function openTaskTypeWelcome(', 'function pickWelcomeTaskType(')


def test_greeting_app_delegation_observes_all_hour_boundaries():
    _run(r'''
import {updateWorkspaceGreeting as updateWorkspaceGreetingView} from './marvis/static/js/task-workspace-view.js';
const node={textContent:''};const $=()=>node;
for(const [hour,expected]of [[4,'晚上好'],[5,'早上好'],[8,'早上好'],[9,'上午好'],[11,'上午好'],[12,'下午好'],[17,'下午好'],[18,'晚上好']]){
 updateWorkspaceGreeting(new Date(2026,10,4,hour));assert.equal(node.textContent,expected);
}
''', 'function updateWorkspaceGreeting(')


def test_report_draft_confirmation_posts_current_payload_and_ignores_wrong_task():
    _run(r'''
taskSession.selectTask({id:'current'});const taskRequests=taskSession.requests,workbenchTaskId=()=> 'current';
const table={dataset:{reportDraftTaskId:'other'}};const button={closest:()=>table,disabled:false,textContent:'确认'};
let confirming=false;const payload={message_id:'draft',revision:3,text_values:{'TEXT:final_validation_conclusion':'approved'}};
const reportDraftState={isConfirming:()=>confirming,setConfirming:(_ids,value)=>{confirming=value;},
 save:async id=>calls.push(['save',id]),payload:()=>payload,discard:id=>calls.push(['discard',id])};
const claimBusy=()=>({}),releaseBusy=()=>calls.push(['released']);
const api=async(url,options)=>{calls.push(['post',url,JSON.parse(options.body)]);return {messages:[]};};
const renderAgentConversation=()=>{},pollAgentMessagesUntilSettled=async()=>{},refreshTasks=async()=>{};
const setActionStatus=()=>{};
await submitVisibleReportDraft(button);assert.equal(calls.length,0);
table.dataset.reportDraftTaskId='current';await submitVisibleReportDraft(button);
assert.deepEqual(calls[0],['save','current']);
assert.deepEqual(calls[1],['post','api/tasks/current/agent/report-draft/confirm',payload]);
assert.deepEqual(calls[2],['discard','current']);assert.equal(confirming,false);assert.deepEqual(calls.at(-1),['released']);
''', 'async function submitVisibleReportDraft(')


def test_report_workspace_renders_editable_native_draft_and_hides_when_absent():
    _run(r'''
import {reportDraftCardHtml,latestPendingReportDraftMessageId,hasReportDraftValues} from './marvis/static/js/report-draft-table.js';
import {createReportDraftState} from './marvis/static/js/report-draft-state.js';
taskSession.selectTask({id:'task',task_type:'validation',model_name:'用户模型'});
const workbenchTask=()=>taskSession.task,workbenchTaskId=()=>taskSession.taskId,selectedTaskIsAgentMode=()=>true;
const panel={hidden:false,innerHTML:'',querySelectorAll:()=>[]};const $=()=>panel;
const reportDraftState=createReportDraftState({api:async()=>{throw Error('no save expected');}});
const updateReportDraftSaveStatus=()=>{};renderReportDraftWorkspace();assert.equal(panel.hidden,true);
taskSession.replaceMessages(taskSession.requests.capture(),[{id:'draft',role:'assistant',stage:'word_conclusion_draft',metadata:{
 report_revision:0,draft_edit_revision:0,draft_values:{'TEXT:final_validation_conclusion':'确认文本'}}}]);
renderReportDraftWorkspace();assert.equal(panel.hidden,false);assert.match(panel.innerHTML,/确认文本/);
assert.match(panel.innerHTML,/data-report-draft-task-id="task"/);assert.match(panel.innerHTML,/data-report-draft-confirm/);
''', 'function renderReportDraftWorkspace(')


def test_batch_upload_file_roles_and_bounds_come_from_the_rendered_controller():
    source = (ROOT / 'marvis/static/js/validation-batch-create.js').read_text(encoding='utf-8')
    declarations = slice_function(source, 'function fileInputMarkup(') + slice_function(source, 'function rowMarkup(')
    _run(r'''
import {MIN_VALIDATION_BATCH_ROWS,MAX_VALIDATION_BATCH_ROWS,VALIDATION_BATCH_FILE_RULES} from './marvis/static/js/validation-batch-create.js';
assert.equal(MIN_VALIDATION_BATCH_ROWS,2);assert.equal(MAX_VALIDATION_BATCH_ROWS,10);
assert.deepEqual(Object.keys(VALIDATION_BATCH_FILE_RULES),['notebook','sample','pmml','dictionary']);
assert.equal(VALIDATION_BATCH_FILE_RULES.notebook.accept,'.ipynb');
assert.equal(VALIDATION_BATCH_FILE_RULES.pmml.accept,'.pmml');
assert.match(VALIDATION_BATCH_FILE_RULES.sample.accept,/\.parquet/);
assert.match(VALIDATION_BATCH_FILE_RULES.dictionary.accept,/\.xlsx/);
const html=rowMarkup('row');
for(const role of ['notebook','sample','pmml','dictionary'])assert.match(html,new RegExp('data-batch-file-role="'+role+'"'));
assert.equal((html.match(/type="file" required/g)||[]).length,4);
''', declarations=declarations)


def test_input_contract_failure_loads_single_task_diagnostic_and_batch_success_keeps_child_binding():
    _run(r'''
taskSession.selectTask({id:'parent',task_type:'validation_batch'});
const taskId='child';taskSession.panels.contract.taskId=taskId;
taskSession.panels.contract.value={status:'pending_confirmation',revision:7,contract:{}};
taskSession.panels.contract.context={item:{modelName:'用户模型'}};
const button={disabled:false},status={textContent:''};
const form={dataset:{validationContractTaskId:'child',validationContractParentTaskId:'parent'},
 querySelector:s=>s.includes('submit')?button:status,
 elements:Object.fromEntries(['positive_label','negative_label','split_train','split_test','split_oot','time_granularity'].map(key=>[key,{value:'month'}]))};
const validationContractCandidateValue=(_form,_contract,key)=>({target_col:'bad',split_col:'split',time_col:'date',pmml_output_field:'p',model_params:{},feature_metadata_selection:{feature_col:'feature'}}[key]);
const validationContractScalar=value=>value,workbenchTaskId=()=> 'child',isWorkbenchTaskId=id=>id==='child';
const selectedTaskIsValidationBatch=()=>true,selectedTaskIsAgentMode=()=>true;
let fail=false;const api=async(url,options)=>{calls.push(['put',url,JSON.parse(options.body)]);if(fail)throw Error('invalid choice');return {status:'ready'};};
const validationBatchPanelController={selectTask:async task=>calls.push(['parent',task.id]),markContractConfirmed:id=>{calls.push(['confirmed',id]);return 0;}};
const renderValidationInputContract=()=>{},setActionStatus=()=>{};
const continueAgentValidationBatch=async options=>calls.push(['continue',options]);
const startAgentValidation=()=>{throw Error('must not send parent conversation');};
await submitValidationInputContract(form);
assert.equal(calls[0][1],'/api/tasks/child/validation-input-contract');
assert.deepEqual(calls.slice(1),[['parent','parent'],['confirmed','child'],['continue',{resumeChildId:'child'}]]);
calls.length=0;form.dataset.validationContractParentTaskId='';fail=true;
const loadAgentMessages=async id=>calls.push(['diagnostic',id]);
await assert.rejects(submitValidationInputContract(form),/invalid choice/);
assert.deepEqual(calls.at(-1),['diagnostic','child']);assert.equal(button.disabled,false);assert.equal(status.textContent,'invalid choice');
''', 'async function submitValidationInputContract(')


def test_batch_and_single_validation_share_icon_and_rows_preserve_raw_name():
    glyphs = APP[APP.index('const TASK_KIND_GLYPHS ='):APP.index('function taskKindIconKind(')]
    _run(r'''
const defaultTaskType='validation';const taskStatusDot=()=> 'wait',taskDisplayName=task=>task.model_name;
assert.equal(taskKindIconHtml('validation_batch'),taskKindIconHtml('validation'));
const row=taskRowInnerHtml({task_type:'validation_batch',model_name:'<用户原始名称>'});
assert.match(row,/class="task-row-name" translate="no">&lt;用户原始名称&gt;/);
assert.doesNotMatch(row,/task-row-meta|task-row-date/);assert.match(row,/data-kind="validation"/);
''', 'function taskKindIconKind(', 'function taskKindIconHtml(', 'function taskRowInnerHtml(', declarations=glyphs)


def test_batch_dialog_creation_callback_selects_only_the_returned_parent():
    source = APP.split('const validationBatchCreateController =',1)[1].split('const dataWorkspaceController =',1)[0]
    callback = source.split('onCreated: ',1)[1].rsplit(',\n});',1)[0]
    _run(r'''
const refreshTasks=async()=>calls.push('refresh');const findTaskInCache=id=>id==='parent'?{id}:null;
const selectTask=task=>calls.push(['selected',task.id]);
await created({batch:{parent_task_id:'parent'}});assert.deepEqual(calls,['refresh',['selected','parent']]);
await assert.rejects(created({}),/父任务 ID/);
await assert.rejects(created({batch:{parent_task_id:'missing'}}),/尚未找到父任务/);
''', declarations=f"const created={callback};\n")


def test_metric_preview_consumer_keeps_original_detail_table_inside_native_summary_card():
    _run(r'''
import {validationMetricCardHtml,validationMetricCardModel} from './marvis/static/js/result-cards.js';
import {renderMetricTableSection} from './marvis/static/js/metric-tables.js';
taskSession.selectTask({id:'t',model_name:'原始模型'});
const workbenchTask=()=>taskSession.task,workbenchTaskId=()=> 't';
const renderMetricSectionVisibility=()=>{},metricPreviewEmptyMessage=()=>'';
const metricPreviewSignature=(...args)=>JSON.stringify(args),renderSignatures={};
const panel={innerHTML:''},$=()=>panel;const attachRocInteractions=()=>{},attachMetricTooltip=()=>{},attachResultCardInteractions=()=>{};
renderMetricPreview({},null,[{title:'样本情况',tables:[{title:'用户明细',layout:'table',headers:['字段','值'],rows:[['user_field','123']]}]}]);
assert.match(panel.innerHTML,/原始模型/);assert.match(panel.innerHTML,/样本情况/);assert.match(panel.innerHTML,/user_field/);assert.match(panel.innerHTML,/123/);
assert.match(panel.innerHTML,/result-card/);
''', 'function renderMetricPreview(')


def test_batch_switcher_tooltip_and_visible_progress_describe_the_same_child():
    _run(r'''
import {renderValidationBatchSwitcher} from './marvis/static/js/validation-batch.js';
const html=renderValidationBatchSwitcher({batch:{parent_task_id:'p',status:'running',item_count:1},items:[{
 child_task_id:'child',model_name:'用户名称',model_version:'v1',ordinal:1,status:'succeeded',stage:'completed'}]});
const progress=html.match(/validation-batch-switcher-progress">([^<]+)</)[1];
assert.ok(progress);assert.match(html,new RegExp('title="用户名称 · v1：'+progress+'"'));
''')


def test_app_portfolio_route_delegates_current_task_and_messages_to_real_gate_classifier():
    _run(r'''
import {portfolioTurnUsesDeterministicRoute} from './marvis/static/js/v2/portfolio_setup_panel.js';
for(const mode of ['manual','agent']){
 const task={id:'portfolio',task_type:'portfolio',run_mode:mode};
 for(const [messages,expected]of [[[],false],[[{role:'assistant',metadata:{kind:'portfolio_setup_required'}}],true],[[{role:'assistant',metadata:{kind:'gate'}}],true],[[{role:'assistant',metadata:{kind:'result'}}],false]]){
  assert.equal(selectedTaskNeedsDeterministicPortfolioTurn(task,messages),expected);
 }
}
assert.equal(selectedTaskNeedsDeterministicPortfolioTurn({task_type:'modeling'},[]),false);
''', 'function selectedTaskNeedsDeterministicPortfolioTurn(')


def test_real_api_attaches_current_ui_language_to_json_and_multipart_requests():
    _run(r'''
globalThis.document={baseURI:'http://localhost/',querySelector:()=>null,documentElement:{lang:'zh-CN'}};
const {api}=await import('./marvis/static/js/api.js');
const {__setI18nStateForTests}=await import('./marvis/static/js/i18n.js');
globalThis.fetch=async(url,options)=>{calls.push(options);return {ok:true,json:async()=>({})};};
__setI18nStateForTests({locale:'en'});await api('api/test',{method:'POST',body:JSON.stringify({content:'hello'})});
assert.equal(calls[0].headers['X-MARVIS-UI-Locale'],'en');
__setI18nStateForTests({locale:'zh'});const form=new FormData();await api('api/test',{method:'POST',body:form});
assert.equal(calls[1].headers['X-MARVIS-UI-Locale'],'zh');assert.equal(calls[1].headers['Content-Type'],undefined);
''')
