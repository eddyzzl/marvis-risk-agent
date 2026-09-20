"""Execute the real activity owner and its app adapters under reordered replies."""

from pathlib import Path
import subprocess

from tests.javascript_source import slice_function


ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "marvis/static/app.js").read_text()


def run_node(body, *functions):
    source = "\n".join(slice_function(APP, name) for name in functions)
    script = '''
import assert from "node:assert/strict";
import { createTaskActivityOwner } from "./marvis/static/js/task-activity.js";
import { createTaskRequestScope } from "./marvis/static/js/task-request-scope.js";
const taskActivities = createTaskActivityOwner();
const taskRequests = createTaskRequestScope();
let selectedTaskId = "a";
taskRequests.select("a");
const painted = [];
const isWorkbenchTaskId = id => id === selectedTaskId;
const setActionStatus = (...args) => painted.push(args);
const renderWorkflowStepper = () => {};
const renderPetState = () => {};
const updateAgentSendDisabled = () => {};
const renderAll = () => {};
''' + source + body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=ROOT,
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


BUSY = ("function renderBusyActivity(", "function claimBusy(", "function releaseBusy(")


def test_only_operation_owner_can_release_and_stop_has_separate_identity():
    run_node('''
const first = taskActivities.claim("a", "agent", "start");
assert.equal(taskActivities.claim("a", "agent", "start"), null);
const stop = taskActivities.claim("a", "agent", "stop");
assert.ok(stop);
assert.equal(taskActivities.release({...stop}), false);
taskActivities.release(first);
assert.equal(taskActivities.action("a"), "agent");
const next = taskActivities.claim("a", "metrics", "start");
assert.equal(taskActivities.release(first), false);
taskActivities.release(stop);
assert.equal(taskActivities.action("a"), "metrics");
taskActivities.release(next);
assert.equal(taskActivities.action("a"), null);
const oldController = taskActivities.channel("driver_execute");
const newController = taskActivities.channel("driver_execute");
oldController(true, "a"); newController(true, "a");
oldController(false, "a"); oldController(false, "a");
assert.equal(taskActivities.action("a"), "driver_execute");
newController(false, "a");
assert.equal(taskActivities.action("a"), null);
''')


def test_run_action_deduplicates_start_but_stop_and_old_finally_are_independent():
    run_node('''
let startCount=0, stopCount=0, endStart, endStop;
const start = () => { startCount++; return new Promise(r => endStart=r); };
const stop = () => { stopCount++; return new Promise(r => endStop=r); };
const first=runAction(start,{actionId:"agent"});
await runAction(start,{actionId:"agent"});
assert.equal(startCount,1);
const stopping=runAction(stop,{actionId:"agent",operation:"agent:stop"});
await runAction(stop,{actionId:"agent",operation:"agent:stop"});
assert.equal(stopCount,1);
endStart(); await first;
assert.equal(taskActivities.action("a"),"agent");
selectedTaskId="b"; taskRequests.select("b");
const newer=taskActivities.claim("b","metrics");
endStop(); await stopping;
assert.equal(taskActivities.action("a"),null);
assert.equal(taskActivities.action("b"),"metrics");
taskActivities.release(newer);
''', *BUSY, "async function runAction(")


def test_stop_invalidates_start_reply_and_reads_without_losing_current_visit():
    run_node('''
const visit=taskRequests.capture();
const start=taskRequests.advance("agent");
const read=taskRequests.begin("messages",{operation:"agent"});
const stop=taskRequests.advance("agent");
assert.equal(taskRequests.current(start),false);
assert.equal(taskRequests.accepts(read),false);
assert.equal(read.signal.aborted,true);
assert.equal(taskRequests.current(stop),true);
assert.equal(taskRequests.current(visit),true);
taskRequests.select("b"); taskRequests.select("a");
assert.equal(taskRequests.current(stop),false);
''')


def test_real_start_and_stop_adapters_do_not_accept_late_start_messages():
    run_node('''
const composer={value:"original instruction"};
const $=id => id==="agentComposerInput" ? composer : {value:"model"};
const workbenchTaskId=()=>selectedTaskId;
const requireTaskId=id=>id;
const selectedTaskNeedsManualRiskIntake=()=>false;
const selectedTaskNeedsDeterministicPortfolioTurn=()=>false;
const agentModelUnavailableMessage=()=>"";
const showAgentModelGuidance=()=>false;
const setAgentComposerNotice=()=>{};
const autoGrowComposerInput=()=>{};
const appendOptimisticAgentUserMessage=()=>({id:"user"});
const appendOptimisticAgentThinkingMessage=()=>({id:"thinking"});
const agentRequestAbortControllers=new Map();
const agentEffort=()=>"high";
const agentAcceptanceModeValue=()=>"normal";
const reportDraftState={get:()=>null};
const pollAgentMessagesUntilSettled=async()=>{};
const invalidateAgentBatchAutoRun=()=>{};
let agentMessages=[];
const renderAgentConversation=()=>{};
const requests=[];
const api=(path,options)=>new Promise(resolve=>requests.push({path,options,resolve}));
const start=startAgentValidation();
composer.value="should not submit twice";
await startAgentValidation();
assert.equal(requests.length,1);
const stop=stopAgentValidation();
await stopAgentValidation();
assert.equal(requests.length,2);
requests[1].resolve({messages:[{id:"stop"}],status:"stopped"}); await stop;
requests[0].resolve({messages:[{id:"late-start"}],status:"done"}); await start;
assert.deepEqual(agentMessages,[{id:"stop"}]);
assert.equal(taskActivities.action("a"),null);
assert.equal(composer.value,"should not submit twice");
''', *BUSY, "async function startAgentValidation(", "async function stopAgentValidation(")


def test_deterministic_manual_intake_exposes_stop_while_its_request_is_pending():
    run_node('''
const selectedTaskIsAgentMode=()=>false;
const selectedTaskNeedsManualRiskIntake=()=>true;
const workbenchTaskId=()=>"a";
const taskBusyAction=id=>taskActivities.action(id);
assert.equal(agentSendIsStopMode(),false);
const lease=taskActivities.claim("a","agent","agent:request");
assert.equal(agentSendIsStopMode(),true);
taskActivities.release(lease);
assert.equal(agentSendIsStopMode(),false);
''', "function agentSendIsStopMode(")
