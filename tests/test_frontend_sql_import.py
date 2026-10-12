from tests.test_frontend_data_workspace_panel import run_node


def test_sql_panel_preserves_confirmation_gate_and_ignores_stale_task_responses():
    run_node("""
      import assert from 'node:assert/strict';
      import { createSqlImportPanel } from './marvis/static/js/v2/sql_import_panel.js';
      function element() {
        const listeners = new Map();
        return {value: '', innerHTML: '', textContent: '', disabled: false, hidden: false,
          addEventListener(kind, callback) {listeners.set(kind, callback);},
          fire(kind) {return listeners.get(kind)?.();}};
      }
      const selectors = ['source','mode','input','preview','confirm','propose','status','schema','refresh',
        'result','result-meta','result-table','result-status','result-refresh','download','cancel-read'];
      const fields = Object.fromEntries(selectors.map(key => [`[data-sql-${key}]`, element()]));
      const root = {innerHTML:'', querySelector: selector => fields[selector]};
      const listeners = [];
      let state = {taskId:'', serverSnapshot:null, dirty:false};
      const controller = {getState:() => state, subscribe:fn => listeners.push(fn),
        load:async task => {state = {taskId:task,serverSnapshot:{task_id:task,revision:1,
          active_dataset_id:'result-one',active_dataset_content_hash:'a'.repeat(64)},dirty:false}; listeners.forEach(fn=>fn(state));}};
      const source = {source_id:'s1', name:'source',status:'ready',tables:[{table:'loans',columns:[{name:'amount',dtype:'integer'}]}]};
      const postCalls = [];
      const getCalls = [];
      let resolveProposal;
      let failConfirmation = false;
      const panel = createSqlImportPanel({controller, getElementById:() => root,
        get: async(url) => {
          getCalls.push(url);
          return url.includes('/preview?') ? {columns:['amount'],rows:[{amount:100},{amount:200}],truncated:false}
            : url.endsWith('/data-workspace') ? state.serverSnapshot : {sources:[source]};
        },
        post: async(url, body) => {
          postCalls.push([url,body]);
          if (url.endsWith('/proposals')) return new Promise(resolve => {resolveProposal=resolve;});
          if (failConfirmation) throw {detail:{code:'sql_import_failed_consumed',message:'本次导入失败。',execution_consumed:true,next_action:'请重新生成待确认查询，不会重复运行这次查询。'}};
          return {status:'completed',dataset:{id:'result-one',task_id:state.taskId,row_count:2,content_hash:'a'.repeat(64)},
            workspace:{task_id:state.taskId,revision:1,active_dataset_id:'result-one',active_dataset_content_hash:'a'.repeat(64)},
            evidence:{row_count:2,content_hash:'a'.repeat(64)}};
        }});
      const tick = () => new Promise(resolve=>setTimeout(resolve,0));
      function select(task, revision=0) {state={taskId:task, serverSnapshot:{revision},dirty:false}; listeners.forEach(fn=>fn(state));}
      select('task-one'); await tick();
      fields['[data-sql-source]'].value='s1'; fields['[data-sql-source]'].fire('change');
      fields['[data-sql-mode]'].value='query'; fields['[data-sql-input]'].value='SELECT amount FROM loans';
      const pending = fields['[data-sql-propose]'].fire('click'); await tick();
      assert.equal(postCalls.length,1); assert.equal(postCalls[0][1].query,'SELECT amount FROM loans');
      assert.equal(fields['[data-sql-confirm]'].disabled,true);
      select('task-two'); await tick();
      resolveProposal({query:'SELECT amount FROM loans',proposal_token:'stale',workspace_revision:0,max_rows:100000,timeout_seconds:30});
      await pending;
      assert.equal(fields['[data-sql-confirm]'].disabled,true);
      assert.equal(fields['[data-sql-preview]'].hidden,true);
      fields['[data-sql-source]'].value='s1'; fields['[data-sql-source]'].fire('change');
      fields['[data-sql-input]'].value='SELECT amount FROM loans WHERE amount > 10';
      const current = fields['[data-sql-propose]'].fire('click'); await tick();
      const beforeRefresh = getCalls.length;
      assert.equal(fields['[data-sql-refresh]'].disabled,true);
      await fields['[data-sql-refresh]'].fire('click');
      await panel.refresh(); // programmatic refresh cannot invalidate a live proposal either
      assert.equal(getCalls.length,beforeRefresh);
      resolveProposal({query:'SELECT amount FROM loans WHERE amount > 10',proposal_token:'frozen-current',workspace_revision:0,max_rows:100000,timeout_seconds:30});
      await current;
      assert.equal(fields['[data-sql-refresh]'].disabled,false);
      assert.equal(fields['[data-sql-input]'].disabled,false);
      assert.equal(fields['[data-sql-confirm]'].disabled,false);
      assert.equal(postCalls.length,2); // generating a proposal executes nothing
      await fields['[data-sql-confirm]'].fire('click');
      assert.equal(fields['[data-sql-result]'].hidden,false);
      assert.ok(fields['[data-sql-result-table]'].innerHTML.includes('>100</td>'));
      assert.equal(fields['[data-sql-download]'].disabled,false);
      assert.deepEqual(postCalls[2][1], {proposal_token:'frozen-current'});
      assert.ok(postCalls[2][0].includes('task-two'));
      assert.match(fields['[data-sql-status]'].textContent,/已导入 2 行/);
      assert.equal(fields['[data-sql-confirm]'].disabled,true);
      assert.equal(controller.getState().serverSnapshot.revision,1);
      failConfirmation=true;
      fields['[data-sql-input]'].value='SELECT amount FROM loans';
      const retryProposal = fields['[data-sql-propose]'].fire('click'); await tick();
      resolveProposal({query:'SELECT amount FROM loans',proposal_token:'failed-proposal',workspace_revision:1,max_rows:100000,timeout_seconds:30});
      await retryProposal;
      assert.equal(fields['[data-sql-confirm]'].disabled,false);
      await fields['[data-sql-confirm]'].fire('click');
      assert.match(fields['[data-sql-status]'].textContent,/重新生成待确认查询/);
      assert.equal(fields['[data-sql-confirm]'].disabled,true);
      assert.equal(fields['[data-sql-propose]'].disabled,false);
      assert.equal(postCalls.length,5); // failed confirmation never retries automatically
      assert.ok(panel.refresh);
    """)


def test_sql_result_download_authenticates_exact_bytes_and_current_workspace():
    run_node("""
      import assert from 'node:assert/strict';
      import { sqlResultBinding, matchesSqlResultWorkspace, readSqlResultPreview, readSqlResultDownload }
        from './marvis/static/js/v2/sql_import_controller.js';
      const blob = new Blob(['native-registered-parquet-bytes']);
      const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
      const hash = [...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,'0')).join('');
      const workspace = {task_id:'task-one',revision:1,active_dataset_id:'result-one',active_dataset_content_hash:hash};
      const response = {status:'completed',dataset:{id:'result-one',task_id:'task-one',row_count:2,content_hash:hash},
        workspace,evidence:{row_count:2,content_hash:hash},download_url:'https://untrusted.invalid/exfiltrate'};
      const binding = sqlResultBinding(response,'task-one');
      let state = {taskId:'task-one',serverSnapshot:workspace,dirty:false};
      const calls = [];
      const get = async (url,options) => {
        calls.push([url,options]);
        if (url.endsWith('/download')) return blob;
        if (url.includes('/preview?')) return {columns:['amount'],rows:[{amount:3},{amount:4}],truncated:false};
        return workspace;
      };
      const getState = () => state;
      assert.equal((await readSqlResultPreview(binding,{get,getState})).rows.length,2);
      assert.equal(await readSqlResultDownload(binding,{get,getState}),blob);
      assert.ok(calls.every(([url])=>url.startsWith('/api/tasks/task-one/')));
      assert.equal(calls.find(([url])=>url.endsWith('/download'))[1].responseType,'blob');
      assert.throws(()=>sqlResultBinding({...response,dataset:{...response.dataset,task_id:'another'}},'task-one'),/身份/);
      assert.throws(()=>sqlResultBinding({...response,workspace:{...workspace,active_dataset_content_hash:'b'.repeat(64)}},'task-one'),/身份/);
      await assert.rejects(()=>readSqlResultDownload(binding,{getState,get:async()=>new Blob(['tampered'])}),/指纹/);
      await assert.rejects(()=>readSqlResultPreview(binding,{getState,get:async url => url.endsWith('/data-workspace')
        ? {...workspace,revision:2} : {columns:['amount'],rows:[{amount:3}],truncated:false}}),/服务端/);
      const denied = {status:403,detail:{message:'原生授权已撤销'}};
      await assert.rejects(()=>readSqlResultDownload(binding,{getState,get:async()=>{throw denied;}}),error=>error===denied);
      state={...state,taskId:'task-two'};
      assert.equal(matchesSqlResultWorkspace(binding,state),false);
      const before=calls.length;
      await assert.rejects(()=>readSqlResultDownload(binding,{get,getState}),/版本已变化/);
      assert.equal(calls.length,before);
      state={taskId:'task-one',serverSnapshot:workspace,dirty:true};
      assert.equal(matchesSqlResultWorkspace(binding,state),false);
    """)


def test_sql_panel_late_preview_cancel_and_workspace_change_never_save_wrong_result():
    run_node("""
      import assert from 'node:assert/strict';
      import { createSqlImportPanel } from './marvis/static/js/v2/sql_import_panel.js';
      const fields = {};
      function field(selector) { return fields[selector] ??= {value:'',innerHTML:'',textContent:'',hidden:false,disabled:false,
        listeners:new Map(),addEventListener(k,f){this.listeners.set(k,f);},fire(k){return this.listeners.get(k)?.();}}; }
      const root = {innerHTML:'',querySelector:field};
      const listeners=[]; let state={taskId:'',serverSnapshot:null,dirty:false};
      const hash='a'.repeat(64);
      const snapshot = task => ({task_id:task,revision:1,active_dataset_id:'registered',active_dataset_content_hash:hash});
      const controller={getState:()=>state,subscribe:f=>{listeners.push(f);return ()=>{};},
        load:async task=>{state={taskId:task,serverSnapshot:snapshot(task),dirty:false};listeners.forEach(f=>f(state));}};
      const select = (task,revision=0) => {state={taskId:task,serverSnapshot:{...snapshot(task),revision},dirty:false};listeners.forEach(f=>f(state));};
      const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
      let resolvePreview, saves=0, denyDownload=false;
      const panel=createSqlImportPanel({controller,getElementById:()=>root,saveDownload:()=>saves++,
        get:async url => {
          if (url.endsWith('/download') && denyDownload) throw {status:403,detail:{message:'原生授权已撤销'}};
          return url.includes('/preview?') ? new Promise(resolve=>resolvePreview=resolve)
            : url.endsWith('/data-workspace') ? state.serverSnapshot
            : {sources:[{source_id:'s',name:'synthetic',status:'ready',tables:[]}]};
        },
        post:async(url)=>url.endsWith('/proposals') ? {query:'SELECT amount FROM loans',proposal_token:'confirmed-token',workspace_revision:0,max_rows:10,timeout_seconds:30}
          : {status:'completed',dataset:{id:'registered',task_id:state.taskId,row_count:2,content_hash:hash},
            workspace:snapshot(state.taskId),evidence:{row_count:2,content_hash:hash}}});
      async function start(task) {
        select(task);await tick();field('[data-sql-source]').value='s';field('[data-sql-source]').fire('change');
        field('[data-sql-mode]').value='query';field('[data-sql-input]').value='SELECT amount FROM loans';
        await field('[data-sql-propose]').fire('click');
        const pending=field('[data-sql-confirm]').fire('click');await tick();return {pending};
      }
      let {pending}=await start('one');select('two');await tick();
      resolvePreview({columns:['amount'],rows:[{amount:3}],truncated:true});await pending;
      assert.equal(field('[data-sql-result]').hidden,true);
      assert.equal(field('[data-sql-download]').disabled,true);assert.equal(saves,0);
      ({pending}=await start('three'));field('[data-sql-cancel-read]').fire('click');
      resolvePreview({columns:['amount'],rows:[{amount:3}],truncated:true});await pending;
      assert.equal(field('[data-sql-result]').hidden,true);assert.equal(saves,0);
      ({pending}=await start('four'));state={...state,serverSnapshot:{...state.serverSnapshot,revision:2}};
      listeners.forEach(f=>f(state));resolvePreview({columns:['amount'],rows:[{amount:3}],truncated:true});await pending;
      assert.equal(field('[data-sql-result]').hidden,true);assert.equal(saves,0);
      ({pending}=await start('five'));resolvePreview({columns:['amount'],rows:[{amount:3},{amount:4}],truncated:false});await pending;
      assert.equal(field('[data-sql-download]').disabled,false);
      denyDownload=true;await field('[data-sql-download]').fire('click');
      assert.equal(field('[data-sql-result]').hidden,true);assert.equal(saves,0);
      assert.equal(field('[data-sql-download]').disabled,true);
      assert.equal(field('[data-sql-status]').textContent,'原生授权已撤销');
      panel.destroy();
    """)
