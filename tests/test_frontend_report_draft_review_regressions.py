from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_agent_message_waits_for_draft_save_and_retains_prompt_on_failure():
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
const app=fs.readFileSync('./marvis/static/app.js','utf8');
const source=app.slice(app.indexOf('async function startAgentValidation()'),app.indexOf('async function uploadRiskAnalysisMaterials('));
function setup(save) {
  const input={value:'请只修订结论'};
  const calls=[];
  const context={workbenchTaskId:()=> 'a',$:(id)=>id==='agentComposerInput'?input:{value:'model'},
    selectedTaskNeedsManualRiskIntake:()=>false,selectedTaskNeedsDeterministicPortfolioTurn:()=>false,
    agentModelUnavailableMessage:()=>'',showAgentModelGuidance:()=>false,setAgentComposerNotice:()=>{},
    autoGrowComposerInput:()=>{},updateAgentSendDisabled:()=>{},appendOptimisticAgentUserMessage:()=>({id:'user'}),
    appendOptimisticAgentThinkingMessage:()=>({id:'thinking'}),agentRequestAbortControllers:new Map(),AbortController,
    reportDraftState:{get:()=>({dirty:true}),save},agentEffort:()=> 'high',agentAcceptanceModeValue:()=> 'auto_accept',
    api:async(url,options)=>{calls.push([url,JSON.parse(options.body)]);return {messages:[],status:'done'};},
    pollAgentMessagesUntilSettled:async()=>{},removeOptimisticAgentMessage:()=>{},agentModelConfigurationErrorMessage:()=>'',
    agentMessages:[],renderAgentConversation:()=>{},setActionStatus:()=>{}};
  return {input,calls,start:new Function('ctx',`with(ctx){${source};return startAgentValidation;}`)(context)};
}
let finishSave;
const success=setup(()=>new Promise(resolve=>{finishSave=resolve;}));
const pending=success.start();
assert.equal(success.calls.length,0);
finishSave();await pending;
assert.equal(success.calls.length,1);
assert.equal(success.calls[0][1].content,'请只修订结论');
const failure=setup(async()=>{throw new Error('保存失败');});
await assert.rejects(failure.start(),/保存失败/);
assert.equal(failure.calls.length,0);
assert.equal(failure.input.value,'请只修订结论');
'''
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_report_draft_switch_poll_and_confirmation_boundaries():
    script = r'''
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createReportDraftState } from './marvis/static/js/report-draft-state.js';
import { latestPendingReportDraftMessageId, hasReportDraftValues, reportDraftTableHtml } from './marvis/static/js/report-draft-table.js';
const app = fs.readFileSync('./marvis/static/app.js', 'utf8');
const values = (text) => ({ 'TEXT:final_validation_conclusion': text });
const message = (task, revision=0) => ({ task_id:task, id:`draft-${task}`,role:'assistant',stage:'word_conclusion_draft',metadata:{report_revision:0,draft_edit_revision:revision,draft_values:values('server')}});
function extract(context, name, nextName) {
  const start = app.indexOf(`function ${name}(`);
  const end = app.indexOf(`function ${nextName}(`, start);
  assert.ok(start >= 0 && end > start);
  const source = app.slice(app.slice(start - 6, start) === 'async ' ? start - 6 : start, end).replace(/async\s*$/, '');
  return new Function('ctx', `with (ctx) { ${source}; return ${name}; }`)(context);
}
function setup(api) {
  const fields = [{dataset:{reportDraftKey:'TEXT:final_validation_conclusion'},value:'server',readOnly:false}];
  const status = {textContent:''};
  const confirm = {disabled:false};
  const panel = {hidden:false,dataset:{},innerHTML:'',querySelector(selector) {
    if (selector.includes('save-status')) return status;
    if (selector.includes('feedback')) return {innerHTML:''};
    if (selector.includes('confirm')) return confirm;
    return null;
  }, querySelectorAll(selector) {return selector === '[data-report-draft-key]' ? fields : [];}};
  const context = {workbenchTask:()=>({task_type:'validation',model_name:'A'}),selectedTaskIsAgentMode:()=>true,$:()=>panel,
    workbenchTaskId:()=> 'a',agentMessages:[message('a')],latestPendingReportDraftMessageId,hasReportDraftValues,
    renderedReportDraftSignature:'',escapeHtml:(x)=>x,reportDraftTableHtml,reportDraftFeedbackHtml:()=>'',
    activeValidationViewTaskId:'a',activeValidationView:'report',validationViewState:new Map(),selectValidationView:()=>{},
    pendingTaskContentLoadTaskId:null,setBusy:()=>{},api,renderAgentConversation:()=>{},pollAgentMessagesUntilSettled:async()=>{},
    refreshTasks:async()=>{},setActionStatus:()=>{},loadAgentMessages:async()=>{}};
  context.updateReportDraftSaveStatus = extract(context, 'updateReportDraftSaveStatus', 'rememberValidationView');
  const state = createReportDraftState({delay:100000,api,onChange:context.updateReportDraftSaveStatus});
  context.reportDraftState = state;
  return {context,state,fields,status,confirm,panel};
}

// Evidence renders before the newly selected model's messages. A dirty cache
// must survive that transient empty projection, including after an offline save.
const switched = setup(async()=>({message:message('a',1)}));
switched.state.receive('a',message('a')); switched.state.edit('a',values('unsaved A'));
const render = extract(switched.context,'renderReportDraftWorkspace','renderAgentConversation');
switched.context.agentMessages=[message('b')]; render();
assert.equal(switched.state.get('a').conflict,undefined);
switched.context.agentMessages=[message('a')]; render();
await switched.state.save('a'); assert.equal(switched.state.hasUnsaved(),false);
// A delayed pre-save poll cannot turn the next local edit into a false conflict.
switched.state.edit('a',values('next edit'));
switched.state.receive('a',message('a',0));
assert.equal(switched.state.get('a').conflict,undefined);
assert.equal(switched.state.payload('a').draft_edit_revision,1);
// A genuinely newer revision still blocks, with the user's edit retained.
switched.state.receive('a',message('a',2));
await assert.rejects(switched.state.save('a'),/冲突/);
assert.equal(switched.state.payload('a').text_values['TEXT:final_validation_conclusion'],'next edit');
switched.state.resolve('a','local');
// Explicit server confirmation continues to retire the draft safely.
switched.context.agentMessages=[message('a'),{task_id:'a',id:'confirmed',stage:'word_conclusion_confirmed'}];
render(); assert.equal(switched.state.get('a').conflict.unavailable,true);
switched.state.discard('a');

// A saved empty required conclusion remains the editable current draft after
// reload. The backend rejects confirmation until the user completes it.
const incomplete = setup(async()=>({message:message('a',1)}));
const emptyDraft = message('a',1);
emptyDraft.metadata.draft_values = values('');
incomplete.context.agentMessages = [message('a'),emptyDraft];
extract(incomplete.context,'renderReportDraftWorkspace','renderAgentConversation')();
assert.equal(incomplete.state.get('a').values['TEXT:final_validation_conclusion'],'');
assert.match(incomplete.panel.innerHTML,/data-report-draft-editable="true"/);
assert.equal(incomplete.fields[0].readOnly,false);
assert.equal(incomplete.fields[0].value,'');
incomplete.state.edit('a',values('completed conclusion'));
await incomplete.state.save('a');
assert.equal(incomplete.state.hasUnsaved(),false);
incomplete.state.discard('a');

// Single-model confirmation locks immediately, stays locked through autosave,
// and restores editable, retained values if the confirmation request fails.
let rejectConfirm;
const single=setup(async(url)=>url.endsWith('/confirm')
  ? new Promise((_,reject)=>{rejectConfirm=reject;}) : {message:message('a',1)});
single.state.receive('a',message('a'));single.state.edit('a',values('reviewed text'));
const submit=extract(single.context,'submitVisibleReportDraft','confirmAllValidationBatchReportDrafts');
const button={closest:()=>({dataset:{reportDraftTaskId:'a'}}),disabled:false,textContent:'确认'};
const pending=submit(button);
assert.equal(single.fields[0].readOnly,true);
while (!rejectConfirm) await Promise.resolve();
assert.equal(single.confirm.disabled,true);
assert.match(single.status.textContent,/编辑已暂停/);
single.state.edit('a',values('must not replace the accepted snapshot'));
assert.equal(single.state.payload('a').text_values['TEXT:final_validation_conclusion'],'reviewed text');
rejectConfirm(new Error('network failed'));await pending;
assert.equal(single.fields[0].readOnly,false);
assert.equal(single.state.payload('a').text_values['TEXT:final_validation_conclusion'],'reviewed text');
single.state.edit('a',values('retry text'));
assert.equal(single.state.hasUnsaved(),true);single.state.discard('a');

// Batch confirmation locks every pending model, including an unvisited model
// opened while the request is pending; unrelated models remain editable.
let acceptBatch;let sent;
const batch=setup(async(url,options)=>!options ? {items:[{childTaskId:'a',pendingReportDraft:true,status:'review_required'},
  {childTaskId:'b',pendingReportDraft:true,status:'review_required'}]}
  : url.endsWith('/confirm-all') ? new Promise((resolve)=>{acceptBatch=resolve;sent=JSON.parse(options.body);})
  : {message:message('a',1)});
batch.context.normalizeValidationBatchPayload=(payload)=>payload;
batch.state.receive('a',message('a'));batch.state.edit('a',values('batch A'));
const confirmBatch=extract(batch.context,'confirmAllValidationBatchReportDrafts','handleDriverReportDownloadClick');
const pendingBatch=confirmBatch({parentTaskId:'parent'});
while (!acceptBatch) await Promise.resolve();
batch.state.receive('b',message('b'));
assert.equal(batch.state.get('b').confirming,true);
assert.match(reportDraftTableHtml(values('B'),{editable:true,state:batch.state.get('b')}),/readonly/);
batch.state.edit('b',values('late B'));
assert.equal(batch.state.payload('b').text_values['TEXT:final_validation_conclusion'],'server');
assert.equal(sent.overrides.a.text_values['TEXT:final_validation_conclusion'],'batch A');
batch.state.receive('c',message('c'));batch.state.edit('c',values('other task'));
assert.equal(batch.state.get('c').dirty,true);
acceptBatch({});await pendingBatch;
assert.equal(batch.state.get('a'),undefined);assert.equal(batch.state.get('b'),undefined);
assert.equal(batch.state.isConfirming('b'),false);
assert.equal(batch.state.get('c').dirty,true);batch.state.discard('c');
'''
    result = subprocess.run(
        ['node', '--input-type=module', '-e', script],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_strategy_context_action_reveals_already_open_filtered_tool():
    script = r'''
import assert from 'node:assert/strict';
import { updateStrategyToolDirectory, revealStrategyToolLauncher } from './marvis/static/js/v2/strategy-tool-directory.js';
const select={value:''};const search={value:''};const count={textContent:''};const listeners={};
const directory={querySelector:(s)=>s==='select'?select:s==='input'?search:count,
  addEventListener:(name,listener)=>{listeners[name]=listener;},innerHTML:''};
const tool=(workflow)=>({open:false,hidden:false,textContent:workflow,
  querySelector:()=>({dataset:{candidateLabWorkflow:workflow}})});
const target=tool('interactive_tree_revision');const context=tool('strategy_project_context');
const launchers={before:()=>{},querySelectorAll:()=>[target,context],addEventListener:()=>{}};
const root={ownerDocument:{createElement:()=>directory},querySelector:()=>launchers};
updateStrategyToolDirectory(root,{workflow:{stages:[{id:'candidate_analysis',status:'pending'}]}},'task');
target.open=true;
select.value='current_context';listeners.input();
assert.equal(target.hidden,true);assert.equal(target.open,true);
// Existing contextual handlers can select this tool again without a toggle event.
revealStrategyToolLauncher(root,target);
assert.equal(target.hidden,false);assert.equal(target.open,true);assert.equal(select.value,'all');
// Subsequent polling respects the deliberate reveal rather than hiding it again.
updateStrategyToolDirectory(root,{workflow:{stages:[{id:'current_context',status:'pending'}]}},'task');
assert.equal(target.hidden,false);
'''
    result = subprocess.run(
        ['node', '--input-type=module', '-e', script],
        cwd=ROOT, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
