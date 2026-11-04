from tests.test_frontend_data_workspace_panel import run_node


def test_history_mapping_panel_confirms_exact_preview_and_clears_stale_task_state():
    run_node("""
      import assert from 'node:assert/strict';
      import {createStrategyHistoryMappingPanel} from './marvis/static/js/v2/strategy_history_mapping_panel.js';
      function element() {
        const listeners = new Map();
        return {value:'',innerHTML:'',textContent:'',hidden:false,disabled:false,parentElement:{hidden:false},
          addEventListener(kind,fn){listeners.set(kind,fn);},fire(kind,event={}){return listeners.get(kind)?.(event);}};
      }
      const names=['source','refresh','mode','form','cell','more','kind','name','metric','unit','format','period','version','add','instruction-label','instruction','draft','propose','preview','confirm','status'];
      const fields=Object.fromEntries(names.map(name=>[`[data-hm-${name}]`,element()]));
      const root={innerHTML:'',querySelector:key=>fields[key]};
      const source={artifact_id:'s1',content_hash:'a'.repeat(64),name:'CSV',status:'ready'};
      const cell={locator:{format:'csv',row:2,column:3},state:'present',value:'2.5%',cell_hash:'b'.repeat(64)};
      const calls=[];
      let resolveProposal;
      const panel=createStrategyHistoryMappingPanel({root,get:async url=>url.endsWith('/sources')?{sources:[source]}:url.includes('/cells')?{source_hash:source.content_hash,cells:[cell],next_offset:null}:{head:{revision:0,artifact_id:null,content_hash:null},mapping:null},
        post:async(url,body)=>{
          calls.push([url,body]);
          if(url.endsWith('/proposals'))return new Promise(resolve=>{resolveProposal=resolve;});
          return {head:{revision:1,artifact_id:'mapping',content_hash:'c'.repeat(64)},mapping:{bindings:calls[0][1].bindings},next_action:'已保存。请刷新项目现状与历史材料，再生成报告。'};
        }});
      fields['[data-hm-source]'].value='s1';
      await panel.selectTask({id:'task-one',task_type:'strategy'});
      fields['[data-hm-mode]'].value='form';fields['[data-hm-kind]'].value='metric';
      fields['[data-hm-metric]'].value='bad_rate';fields['[data-hm-unit]'].value='percent';fields['[data-hm-cell]'].value='0';
      fields['[data-hm-kind]'].fire('change');fields['[data-hm-add]'].fire('click');
      assert.equal(calls.length,0);assert.equal(fields['[data-hm-confirm]'].disabled,true);
      const pending=fields['[data-hm-propose]'].fire('click');await new Promise(resolve=>setTimeout(resolve,0));
      assert.equal(calls.length,1);
      assert.deepEqual(calls[0][1].bindings[0].locator,{format:'csv',row:2,column:3});
      assert.equal(calls[0][1].bindings[0].source_hash,'a'.repeat(64));
      assert.equal(calls[0][1].bindings[0].cell_hash,'b'.repeat(64));
      assert.equal(calls[0][1].bindings[0].period_key,null);
      const typed={...calls[0][1].bindings[0],value:2.5,availability:'present',validation_status:'historical_reported_unvalidated'};
      resolveProposal({proposal_token:'exact-frozen',bindings:[typed],historical_declaration:'历史报告值，未经复算验证。'});await pending;
      assert.equal(fields['[data-hm-confirm]'].disabled,false);assert.match(fields['[data-hm-preview]'].innerHTML,/2.5/);
      assert.equal(calls.length,1);
      await fields['[data-hm-confirm]'].fire('click');
      assert.deepEqual(calls[1][1],{proposal_token:'exact-frozen'});
      assert.match(fields['[data-hm-status]'].textContent,/刷新项目现状/);
      assert.equal(fields['[data-hm-confirm]'].disabled,true);
      const stale=fields['[data-hm-propose]'].fire('click');await new Promise(resolve=>setTimeout(resolve,0));
      const finishOld=resolveProposal;
      const switching=panel.selectTask({id:'task-two',task_type:'strategy'});
      assert.equal(fields['[data-hm-preview]'].innerHTML,'');
      assert.equal(fields['[data-hm-cell]'].innerHTML,'');
      finishOld({proposal_token:'old-task',bindings:[typed],historical_declaration:'old'});
      await stale;await switching;
      assert.equal(fields['[data-hm-confirm]'].disabled,true);
      assert.equal(fields['[data-hm-preview]'].hidden,true);
      assert.equal(calls.length,3); // no automatic resubmission or report execution
      await panel.selectTask(null);
      assert.equal(fields['[data-hm-draft]'].innerHTML,'尚未选择映射。');
    """)


def test_data_processing_panel_registers_a_task_material_and_hides_elsewhere():
    run_node("""
      import assert from 'node:assert/strict';
      import {createStrategyHistoryMappingPanel} from './marvis/static/js/v2/strategy_history_mapping_panel.js';
      function element() {
        const listeners = new Map();
        return {value:'',innerHTML:'',textContent:'',hidden:false,disabled:false,parentElement:{hidden:false},
          addEventListener(kind,fn){listeners.set(kind,fn);},fire(kind,event={}){return listeners.get(kind)?.(event);}};
      }
      const names=['source','refresh','mode','form','cell','more','kind','name','metric','unit','format','period','version','add','instruction-label','instruction','draft','propose','preview','confirm','status','candidate','register'];
      const fields=Object.fromEntries(names.map(name=>[`[data-hm-${name}]`,element()]));
      const root={innerHTML:'',hidden:true,querySelector:key=>fields[key]};
      let registered=false;const gets=[],posts=[];
      const panel=createStrategyHistoryMappingPanel({root,taskTypes:['data_join'],registerSources:true,hideOutsideTaskTypes:true,
        get:async url=>{gets.push(url);
          if(url.endsWith('/sources/candidates'))return {filenames:['历史报告.csv']};
          if(url.endsWith('/sources'))return {sources:registered?[{artifact_id:'s1',name:'CSV',status:'ready'}]:[]};
          if(url.includes('/cells'))return {source_hash:'a'.repeat(64),cells:[],next_offset:null};
          return {head:{revision:0,artifact_id:null,content_hash:null},mapping:null};},
        post:async(url,body)=>{posts.push([url,body]);registered=true;return {artifact_id:'s1',status:'registered'};}});
      assert.match(root.innerHTML,/data-hm-register/);
      await panel.selectTask({id:'join-1',task_type:'data_join'});
      assert.equal(root.hidden,false);
      assert.ok(gets.some(url=>url==='/api/tasks/join-1/strategy/history-mapping/sources/candidates'));
      assert.match(fields['[data-hm-candidate]'].innerHTML,/历史报告.csv/);
      fields['[data-hm-candidate]'].value='历史报告.csv';fields['[data-hm-candidate]'].fire('change');
      assert.equal(fields['[data-hm-register]'].disabled,false);
      await fields['[data-hm-register]'].fire('click');
      assert.deepEqual(posts,[['/api/tasks/join-1/strategy/history-mapping/sources',{filename:'历史报告.csv'}]]);
      assert.match(fields['[data-hm-source]'].innerHTML,/s1/);
      await panel.selectTask({id:'s-1',task_type:'strategy'});
      assert.equal(root.hidden,true);
    """)
