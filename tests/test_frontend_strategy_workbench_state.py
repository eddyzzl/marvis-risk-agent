from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).parents[1]


def _node(body):
    subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            textwrap.dedent("""
        import assert from 'node:assert/strict';
        import {captureWorkbenchDraft,restoreWorkbenchDraft,createStrategyWorkbenchStateController}
          from './marvis/static/js/v2/strategy_workbench_state_controller.js';
    """)
            + textwrap.dedent(body),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_full_form_capture_restore_prose_views_and_current_options():
    _node("""
      const reason = {type:'text',tagName:'INPUT',value:'合成理由',dataset:{candidateLabField:'reason'}};
      const scope = {type:'textarea',tagName:'TEXTAREA',value:'project.channel=合成',dataset:{candidateLabField:'scope'}};
      const pointer = {type:'text',tagName:'INPUT',readOnly:true,value:'native-old',dataset:{candidateLabField:'id'}};
      const secret = {type:'password',tagName:'INPUT',value:'do-not-save',dataset:{candidateLabField:'credential'}};
      const choice = {type:'select-one',tagName:'SELECT',value:'old',options:[{value:'old'},{value:'new'}],dataset:{candidateLabField:'candidate'}};
      const launcher = {open:true};
      const form = {dataset:{candidateLabWorkflow:'synthetic'},closest:()=>launcher,querySelectorAll:()=>[reason,scope,pointer,secret,choice]};
      const root = {scrollTop:42,querySelectorAll:()=>[form]};
      const captured = captureWorkbenchDraft(root);
      assert.equal(captured.fields.length,4); assert.equal(captured.fields[0].kind,'narrative');
      assert.equal(JSON.stringify(captured).includes('do-not-save'),false);
      reason.value='';scope.value='';pointer.value='native-current';choice.options=[{value:'new'}];choice.value='new';launcher.open=false;root.scrollTop=0;
      assert.equal(restoreWorkbenchDraft(root,captured),true);
      assert.equal(reason.value,'合成理由');assert.equal(scope.value,'project.channel=合成');
      assert.equal(pointer.value,'native-current');assert.equal(choice.value,'');
      assert.equal(launcher.open,true);assert.equal(root.scrollTop,42);
    """)


def test_actual_api_wire_json_cas_and_private_state_without_local_storage():
    _node("""
      const wire=[]; let handler; let task='task-a';
      const field={type:'text',tagName:'INPUT',value:'合成项目',dataset:{candidateLabField:'project_context_scope'}};
      const form={dataset:{candidateLabWorkflow:'strategy_project_context'},closest:()=>({open:true}),querySelectorAll:()=>[field]};
      const panel={scrollTop:12,querySelectorAll:()=>[form]};
      const slot={innerHTML:'',addEventListener:(type,fn)=>handler=fn,contains:()=>true};
      let revision=0; const evidence='a'.repeat(64);
      const binding=()=>({task_id:task,actor_id:'server-session-owner',revision,head_revision:revision,evidence_hash:evidence,history:[],draft:null,invalidated_selections:[]});
      globalThis.fetch=async(url,options={})=>{
        wire.push({url,options});
        let payload;
        if(options.method==='PUT'){
          assert.equal(typeof options.body,'string');
          const body=JSON.parse(options.body);
          assert.equal(body.expected_revision,revision);assert.equal(body.evidence_hash,evidence);
          assert.equal(body.draft.fields[0].values[0],'合成项目');
          assert.equal('actor_id' in body,false);assert.equal('checkpoint' in body,false);
          revision++;payload=binding();
        }else if(url.endsWith('/executions')) payload={evidence_hash:evidence,items:[],supported_tools:[]};
        else payload=binding();
        return new Response(JSON.stringify(payload),{status:200,headers:{'content-type':'application/json'}});
      };
      globalThis.localStorage={setItem(){throw new Error('private server draft must not enter localStorage');},getItem(){throw new Error('legacy draft must not be silently imported');}};
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=>task,schedule:()=>1,cancelSchedule:()=>{}});
      await controller.load(task);controller.persist(task,panel);await controller.save(task);
      assert.equal(revision,1);assert.equal(wire.filter(w=>w.options.method==='PUT').length,1);
      assert.match(slot.innerHTML,/草稿已保存/);controller.clear();
    """)


def test_state_conflict_has_explicit_choice_and_never_retries_overwrite():
    _node("""
      let task='a', putCalls=0;
      const field={type:'text',tagName:'INPUT',value:'本页草稿',dataset:{candidateLabField:'project_context_scope'}};
      const panel={querySelectorAll:()=>[{dataset:{candidateLabWorkflow:'strategy_project_context'},querySelectorAll:()=>[field],closest:()=>({open:true})}]};
      const slot={innerHTML:'',addEventListener(){}};
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=>task,schedule:()=>1,cancelSchedule(){},
        get:async(url)=>url.endsWith('/executions')?{items:[],evidence_hash:'a'.repeat(64)}:{actor_id:'owner',head_revision:2,evidence_hash:'a'.repeat(64),draft:null,history:[],invalidated_selections:[]},
        put:async()=>{putCalls++;const e=new Error('conflict');e.status=409;throw e;}});
      await controller.load(task);controller.persist(task,panel);await controller.save(task);await controller.save(task);
      assert.equal(putCalls,1);assert.match(slot.innerHTML,/另一页面修改/);assert.match(slot.innerHTML,/保留本页/);controller.clear();
    """)


def test_late_task_response_never_restores_previous_task_draft():
    _node("""
      let task='a', resolveA;
      const slot={innerHTML:'',addEventListener(){}};
      const bind=(id)=>({task_id:id,actor_id:'same-server-owner',head_revision:1,evidence_hash:'a'.repeat(64),history:[],draft:{schema_version:'strategy.workbench-draft.v1',fields:[],open_workflows:[],scroll_top:0},invalidated_selections:[]});
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>({}),getTaskId:()=>task,schedule:()=>1,cancelSchedule(){},get:(url)=>{
        if(url.includes('/a/')&&!url.endsWith('/executions')) return new Promise(r=>resolveA=r);
        return Promise.resolve(url.endsWith('/executions')?{items:[],evidence_hash:'a'.repeat(64)}:bind('b'));}});
      const pending=controller.load('a');task='b';await controller.load('b');resolveA(bind('a'));
      assert.equal(await pending,null);assert.equal(controller.snapshot('a'),null);assert.equal(controller.snapshot('b').schema_version,'strategy.workbench-draft.v1');controller.clear();
    """)


def test_legacy_task_view_import_is_explicit_and_removes_invalid_options():
    _node("""
      let click; const field={type:'select-one',tagName:'SELECT',value:'current',options:[{value:'current'}],dataset:{candidateLabField:'source'}};
      const form={dataset:{candidateLabWorkflow:'synthetic'},querySelectorAll:()=>[field],closest:()=>({open:false})};
      const panel={querySelectorAll:()=>[form]};
      const slot={innerHTML:'',addEventListener:(kind,handler)=>click=handler,contains:()=>true};
      let imported=null, reads=0;
      globalThis.localStorage={getItem(key){reads++;return JSON.stringify({schema_version:'strategy.candidate-lab-view-state.v1',fields:[{workflow:'synthetic',field:'source',kind:'select',values:['stale']}],open_workflows:['synthetic']});}};
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=> 'task',schedule:()=>1,cancelSchedule(){},
        get:async(url)=>url.endsWith('/executions')?{items:[],evidence_hash:'a'.repeat(64)}:{head_revision:0,evidence_hash:'a'.repeat(64),history:[],draft:null,invalidated_selections:[]},
        put:async(url,body)=>{imported=body.draft;return {head_revision:1,evidence_hash:'a'.repeat(64),history:[]};}});
      await controller.load('task');assert.equal(reads,0);assert.equal(field.value,'current');
      const button={hasAttribute:name=>name==='data-workbench-import-local'};
      await click({target:{closest:()=>button}});
      assert.equal(reads,1);assert.equal(field.value,'');assert.equal(imported.fields[0].values[0],'');
      assert.match(slot.innerHTML,/旧版参数和选择已导入/);controller.clear();
    """)


def test_actual_evidence_conflict_wire_preserves_intent_and_requires_explicit_rebind():
    _node("""
      let click, revision=3, evidence='a'.repeat(64); const puts=[];
      const reason={type:'text',tagName:'INPUT',value:'本页未保存理由',dataset:{candidateLabField:'reason'}};
      const candidate={type:'select-one',tagName:'SELECT',value:'old',options:[{value:'old'},{value:'new'}],dataset:{candidateLabField:'candidate'}};
      const mode={type:'select-one',tagName:'SELECT',value:'business',options:[{value:'physical'},{value:'business'},{value:'both'}],dataset:{candidateLabField:'field_display_mode'}};
      const method={type:'select-one',tagName:'SELECT',value:'all_features',options:[{value:'all_features'}],dataset:{candidateLabField:'method'}};
      const check={type:'checkbox',tagName:'INPUT',value:'equal_frequency',checked:true,dataset:{candidateLabField:'methods'}};
      const empty={type:'select-one',tagName:'SELECT',value:'',options:[{value:''}],dataset:{candidateLabField:'empty'}};
      const form=(workflow,fields)=>({dataset:{candidateLabWorkflow:workflow},closest:()=>({open:true}),querySelectorAll:()=>fields});
      const panel={querySelectorAll:()=>[form('synthetic',[reason,candidate,method,check,empty]),form('strategy_workbench_preferences',[mode])]};
      const slot={innerHTML:'',contains:()=>true,addEventListener:(type,fn)=>click=fn};
      const binding=()=>({head_revision:revision,evidence_hash:evidence,history:[],draft:null,invalidated_selections:[],static_choices:[
        {workflow:'synthetic',field:'method',values:['all_features']},{workflow:'synthetic',field:'methods',values:['equal_frequency','tree']}
      ]});
      globalThis.fetch=async(url,options={})=>{
        if(options.method==='PUT'){
          const body=JSON.parse(options.body);puts.push(body);
          if(body.evidence_hash!==evidence) return new Response(JSON.stringify({detail:{code:'workbench_evidence_changed'}}),{status:409,headers:{'content-type':'application/json'}});
          assert.equal(body.expected_revision,revision);revision++;
        }
        return new Response(JSON.stringify(url.endsWith('/executions')?{items:[],evidence_hash:evidence}:binding()),{status:200,headers:{'content-type':'application/json'}});
      };
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=> 'task',schedule:()=>1,cancelSchedule(){}});
      await controller.load('task');controller.persist('task',panel);evidence='b'.repeat(64);
      await controller.save('task');await controller.save('task');
      assert.equal(puts.length,1);assert.match(slot.innerHTML,/证据已更新/);assert.doesNotMatch(slot.innerHTML,/另一页面修改/);
      assert.equal(controller.snapshot('task').fields[0].values[0],'本页未保存理由');
      await click({target:{closest:()=>({hasAttribute:n=>n==='data-workbench-read-head'})}});
      assert.equal(puts.length,1);assert.equal(reason.value,'本页未保存理由');assert.equal(candidate.value,'');assert.equal(mode.value,'business');
      assert.equal(method.value,'all_features');assert.equal(check.checked,true);assert.equal(empty.value,'');
      assert.match(slot.innerHTML,/1 项旧证据选择已清除/);
      candidate.value='new';controller.persist('task',panel);await controller.save('task');
      assert.equal(puts.length,2);assert.equal(puts[1].evidence_hash,evidence);
      assert.equal(puts[1].draft.fields.find(f=>f.field==='candidate').values[0],'new');
      assert.equal(puts[1].draft.fields.find(f=>f.field==='field_display_mode').values[0],'business');controller.clear();
    """)


def test_same_task_reload_cancels_delayed_save_and_keeps_unsaved_input():
    _node("""
      const timers=new Map();let next=0,puts=0;
      const field={type:'text',tagName:'INPUT',value:'新输入',dataset:{candidateLabField:'reason'}};
      const panel={querySelectorAll:()=>[{dataset:{candidateLabWorkflow:'synthetic'},closest:()=>({open:true}),querySelectorAll:()=>[field]}]};
      const slot={innerHTML:'',addEventListener(){}};
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=> 'task',
        schedule:fn=>{timers.set(++next,fn);return next;},cancelSchedule:id=>timers.delete(id),
        get:async url=>url.endsWith('/executions')?{items:[],evidence_hash:'a'.repeat(64)}:{head_revision:0,evidence_hash:'a'.repeat(64),history:[],draft:null,invalidated_selections:[]},
        put:async()=>{puts++;throw new Error('old timer must not save');}});
      await controller.load('task');controller.persist('task',panel);
      const delayed=timers.get(next);await controller.load('task');delayed();await Promise.resolve();
      assert.equal(puts,0);assert.equal(controller.snapshot('task').fields[0].values[0],'新输入');
      assert.match(slot.innerHTML,/未覆盖本页/);controller.clear();
    """)


def test_same_task_reload_waits_for_own_save_and_latest_pending_edit_before_reading_head():
    _node("""
      let revision=0,draftReads=0,committed=null;const pending=[];
      const field={type:'text',tagName:'INPUT',value:'第一版',dataset:{candidateLabField:'reason'}};
      const panel={querySelectorAll:()=>[{dataset:{candidateLabWorkflow:'synthetic'},closest:()=>({open:true}),querySelectorAll:()=>[field]}]};
      const slot={innerHTML:'',addEventListener(){}};
      const binding=()=>({head_revision:revision,evidence_hash:'a'.repeat(64),history:[],draft:committed,invalidated_selections:[]});
      const controller=createStrategyWorkbenchStateController({getElementById:()=>slot,getPanel:()=>panel,getTaskId:()=> 'task',schedule:()=>1,cancelSchedule(){},
        get:async url=>{if(url.endsWith('/executions'))return {items:[],evidence_hash:'a'.repeat(64)};draftReads++;return binding();},
        put:(url,body)=>new Promise(resolve=>pending.push(()=>{assert.equal(body.expected_revision,revision);revision++;committed=body.draft;resolve(binding());}))});
      await controller.load('task');controller.persist('task',panel);const saving=controller.save('task');
      field.value='第二版';controller.persist('task',panel);const loading=controller.load('task');
      assert.equal(draftReads,1);pending[0]();
      for(let i=0;i<5;i++)await Promise.resolve();
      assert.equal(draftReads,1);assert.equal(pending.length,2);pending[1]();await saving;await loading;
      assert.equal(draftReads,2);assert.equal(revision,2);assert.doesNotMatch(slot.innerHTML,/另一页面修改/);
      assert.equal(controller.snapshot('task').fields[0].values[0],'第二版');
      controller.persist('task',panel);await controller.save('task');assert.equal(pending.length,2);
      controller.clear();
    """)
