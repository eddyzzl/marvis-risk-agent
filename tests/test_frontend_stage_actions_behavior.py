"""Execute actual validation actions and polling; recorded doubles are effects."""

import json

import pytest

from tests.test_frontend_scroll_signature_behaviour import run_node


@pytest.mark.parametrize("stage,action,final_status", [
    ("scan", "scanCurrentTask", "scanned"),
    ("notebook", "validateCurrentTask", "executed"),
    ("metrics", "generateMetrics", "writing_artifacts"),
    ("report", "generateReport", "succeeded"),
])
@pytest.mark.parametrize("outcome", ["success", "failed", "switch"])
def test_stage_actions_bind_task_prime_progress_and_settle_before_revealing_results(stage, action, final_status, outcome):
    run_node(f"const stage={json.dumps(stage)}, finalStatus={json.dumps(final_status)}, outcome={json.dumps(outcome)};" + r'''
taskSession.selectTask({id:'a',status:'created'});
let scanAbortController=null;
const terminalTaskStatuses=new Set(['failed','succeeded']);
const record=(name,...args)=>calls.push([name,...args]);
const ensureValidationMaterialSelection=async task=>task;
const selectedTaskIsAgentMode=()=>false;
const renderScanResult=result=>record('scan-result',result);
const renderValidationResult=result=>record('notebook-result',result);
const resetReproducibilityRenderSignatures=()=>record('reset-signature');
const appendPendingReproducibilitySteps=()=>record('prime-notebook');
const appendPendingMetricSteps=()=>record('prime-metrics');
const elements=new Map();const $=id=>{if(!elements.has(id))elements.set(id,{});return elements.get(id);};
const setActionStatus=(message,tone)=>record('status',tone);
const setTaskFailureActionStatus=task=>record('failure',task.id);
const scrollToManualWorkflowSection=value=>record('scroll',value);
const loadReportFields=async id=>record('fields',id);
const loadTaskEvidence=async id=>record('evidence',id);
const finish=()=>{if(outcome!=='switch')taskSession.refreshTask({id:'a',status:outcome==='failed'?'failed':finalStatus});};
const refreshTasks=async()=>{record('refresh');finish();};
const pollValidationProgress=async (_states,id)=>{record('poll',id);finish();return taskSession.task;};
const api=async (url,options)=>{
  record('post',url,options.method);
  if(outcome==='switch')taskSession.selectTask({id:'b',status:'created'});
  return {ok:true};
};
''' + f"await {action}();" + r'''
assert.deepEqual(calls.find(c=>c[0]==='post'),['post',`api/tasks/a/${stage}`, 'POST']);
if(stage!=='scan')assert.deepEqual(calls.find(c=>c[0]==='poll'),['poll','a']);
if(outcome==='switch') {
  assert.equal(taskSession.taskId,'b');
  assert.equal(calls.some(c=>['status','failure','fields','evidence','scroll'].includes(c[0])),false);
} else if(outcome==='failed') {
  assert.deepEqual(calls.find(c=>c[0]==='failure'),['failure','a']);
  assert.equal(calls.some(c=>c[0]==='status'||c[0]==='scroll'),false);
} else {
  assert.deepEqual(calls.at(-1),['scroll',stage]);
  assert.ok(calls.some(c=>c[0]==='status'&&c[1]==='success'));
  if(stage==='notebook'||stage==='metrics') {
    assert.ok(calls.findIndex(c=>c[0]===`prime-${stage}`)<calls.findIndex(c=>c[0]==='poll'));
    assert.ok(calls.findIndex(c=>c[0]==='evidence')>calls.findIndex(c=>c[0]==='poll'));
  }
}
assert.equal(scanAbortController,null);
''', f"async function {action}(")


@pytest.mark.parametrize("stopping,background", [(False, False), (True, False), (True, True)])
def test_validation_poller_observes_job_liveness_and_refreshes_evidence_before_messages(stopping, background):
    run_node(f"const stop={json.dumps(stopping)}, background={json.dumps(background)};" + r'''
import {createProgressPollRegistry,claimProgressPoll,releaseProgressPoll} from './marvis/static/js/polling.js';
const progressPolls=createProgressPollRegistry(),taskRequests=taskSession.requests;
const terminalTaskStatuses=new Set(['succeeded','failed']);
taskSession.selectTask({id:'a',status:'running'});
const snapshots=[{status:'running',active_job_kind:'agent'},
  {status:'succeeded',active_job_kind:'agent'},
  {status:stop?'running':'succeeded',active_job_kind:null}];
let current,reads=0;const refreshTasks=async()=>{current={id:'a',...snapshots[reads++]};taskSession.refreshTask(current);};
const findTaskInCache=()=>current;
const isWorkbenchTaskId=id=>id===taskSession.taskId;
const sleep=async()=>{};
const loadTaskEvidence=async()=>calls.push('evidence');
const loadReportFields=async()=>calls.push('fields');
const metricOverviewComplete=()=>true,currentMetricPreviewHasValues=()=>false;
const selectedTaskIsAgentMode=()=>true,usesAgentValidationWorkbench=()=>false;
const loadAgentMessages=async()=>calls.push('messages');
const renderChangedValidationViews=()=>calls.push('render');
const renderTaskList=()=>calls.push('list');
const renderAll=()=>{throw Error('poll must not rebuild unrelated views');};
const taskServerBusyAction=task=>task.active_job_kind;
const setActionStatus=(_message,tone)=>calls.push(tone);
const setTaskFailureActionStatus=()=>calls.push('failed');
const result=await pollValidationProgress(terminalTaskStatuses,'a',{stopping:stop,background});
assert.equal(reads,3,'a terminal status alone does not settle a live job');
assert.equal(result.active_job_kind,null);assert.equal(progressPolls.size,0);
assert.deepEqual(calls.slice(0,4),['evidence','fields','messages','render']);
assert.deepEqual(calls.slice(4,8),calls.slice(0,4));
assert.equal(calls.filter(c=>['success','stopped'].includes(c)).length,background?0:1);
if(!background)assert.equal(calls.at(-1),stop?'stopped':'success');
''', 'async function pollValidationProgress(')


@pytest.mark.parametrize("completed", [False, True])
def test_evidence_read_failure_preserves_only_completed_reproducibility(completed):
    run_node(f"const completed={json.dumps(completed)};" + r'''
const taskRequests=taskSession.requests;taskSession.selectTask({id:'a'});
const workbenchTaskId=()=>taskSession.taskId,workbenchTask=()=>taskSession.task;
const isWorkbenchTaskId=id=>id===taskSession.taskId;
const blindCustodyOnlyTask=()=>false,usesAgentValidationWorkbench=()=>false;
const notebookReproducibilityComplete=()=>completed;
const resetEvidenceSummaries=()=>calls.push('reset');
let fail=true;const api=async()=>{if(fail)throw Error('offline');return {scan:{status:'recorded'}};};
const renderEvidence=value=>calls.push(value);
const loadValidationInputContract=async id=>calls.push(['contract',id]);
await loadTaskEvidence('a');assert.deepEqual(calls,completed?[]:['reset']);
calls.length=0;fail=false;await loadTaskEvidence('a');
assert.deepEqual(calls,[{scan:{status:'recorded'}},['contract','a']]);
''', 'async function loadTaskEvidence(')


def test_action_buttons_follow_completed_stages_busy_ownership_and_cancel_semantics():
    run_node(r'''
import {escapeHtml} from './marvis/static/js/ui-utils.js';
const taskActivities=taskSession.activities;
const workbenchTask=()=>taskSession.task;
const taskStopped=()=>false,notebookStepStageFailure=()=>null;
const failureStageRank=stage=>['scan','notebook','metrics','report'].indexOf(stage);
let agent=false;const selectedTaskIsAgentMode=()=>agent;
const completedReportReadyForDownloads=()=>false,stepActionBlockedReason=()=>'';
const actions=['scan','notebook','metrics','report'];
for(const [status,expected,next] of [
  ['created',[true,false,false,false],'scan'],['scanned',[true,true,false,false],'notebook'],
  ['executed',[true,true,true,false],'metrics'],['writing_artifacts',[true,true,false,true],'report']]) {
  taskSession.selectTask({id:'a',status});
  assert.deepEqual(actions.map(canRunStepAction),expected);assert.equal(recommendedAction(),next);
  for(const [index,action] of actions.entries())assert.equal(stepActionButtonHtml({action}).includes(' disabled'),!expected[index]);
}
for(const stage of actions) {
  taskSession.selectTask({id:'a',status:'failed',failure_stage:stage,status_message:'misleading report failure'});
  assert.equal(recommendedAction(),stage);assert.equal(canRunStepAction(stage),true);
}
taskSession.selectTask({id:'a',status:'created'});
const lease=taskActivities.claim('a','notebook','start');assert.equal(selectedTaskIsBusy(),true);
taskSession.selectTask({id:'b',status:'created'});assert.equal(selectedTaskIsBusy(),false);
taskActivities.release(lease);
for(const [status,action,job,cancel] of [['running','notebook','notebook','cancelNotebook'],['computing_metrics','metrics','metrics','cancelMetrics'],['writing_artifacts','report','report','cancelReport']]) {
  taskSession.selectTask({id:'a',status,active_job_kind:job});
  const html=stepActionButtonHtml({action});assert.ok(html.includes(`data-step-action="${cancel}"`));
  assert.equal(html.includes(' disabled'),false);assert.ok(html.includes('停止'));
}
agent=true;assert.equal(stepActionButtonHtml({action:'notebook'}),'');
''', 'function taskBusyAction(', 'function taskServerBusyAction(', 'function selectedTaskIsBusy(',
             'function taskFailedDuringScan(', 'function taskFailedDuringNotebook(', 'function taskFailedDuringMetrics(',
             'function taskFailedDuringReport(', 'function taskFailureStage(', 'function normalizedFailureStage(',
             'function earliestFailureStage(', 'function recommendedAction(', 'function canRunStepAction(',
             'function stepStopAction(', 'function stepActionButtonHtml(')


def test_evidence_renderer_dispatches_retained_scan_or_notebook_and_pmml_by_workflow():
    run_node(r'''
let pmml=true,agent=true,lastAgentRenderSignature='old';
const record=(name,...values)=>calls.push([name,...values]);
const renderReproducibilitySectionVisibility=()=>record('visibility');
const renderScanResult=(...args)=>record('scan',...args),renderNotebookSteps=(...args)=>record('steps',...args);
const usesPmmlScoringWorkflow=()=>pmml,selectedTaskIsAgentMode=()=>agent;
const renderPmmlScoringEvidence=value=>record('pmml',value),renderReproducibilityEvidence=value=>record('notebook',value);
const renderAgentConversation=()=>record('conversation');
renderEvidence({scan:{files:['user.csv']},notebook_cells:[{id:1}],pmml_scoring:{rows:3}});
assert.deepEqual(calls,[['visibility'],['scan',{files:['user.csv']},[{id:1}]],['pmml',{rows:3}],['conversation']]);
assert.equal(lastAgentRenderSignature,null);calls.length=0;pmml=false;agent=false;
renderEvidence({notebook_steps:[{id:'s'}],reproducibility:{pass:true}});
assert.deepEqual(calls,[['visibility'],['steps',[{id:'s'}],[]],['notebook',{pass:true}]]);
''', 'function renderEvidence(')


def test_active_task_polling_starts_once_and_does_not_start_for_idle_tasks():
    run_node(r'''
const progressPolls=new Map(),terminalTaskStatuses=new Set(['succeeded']);
const usesAgentValidationWorkbench=()=>false,isValidationBatchTask=()=>false;
const taskServerBusyAction=task=>task?.active_job_kind;
const pollValidationProgress=async (states,id,options)=>{assert.equal(states,terminalTaskStatuses);calls.push([id,options]);progressPolls.set(id,{});};
taskSession.selectTask({id:'a',active_job_kind:'notebook'});
ensureActiveTaskProgressPolling();ensureActiveTaskProgressPolling();
assert.deepEqual(calls,[['a',{background:true}]]);
taskSession.selectTask({id:'b'});ensureActiveTaskProgressPolling();assert.equal(calls.length,1);
''', 'function ensureActiveTaskProgressPolling(')
