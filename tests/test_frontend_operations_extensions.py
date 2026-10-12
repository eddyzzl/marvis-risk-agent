"""Actual ES module typed forms, exact JSON confirmation and stale task fences."""

from pathlib import Path
import subprocess
import textwrap

from marvis.operations.extensions_router import ProposalRequest
from marvis.operations.extension_bindings import FeedbackImport
from marvis.operations.dispositions import DispositionRequest
import json

ROOT = Path(__file__).parents[1]
HARNESS = r"""
import assert from 'node:assert/strict';
import {createOperationsExtensionsPanel} from './marvis/static/js/operations-extensions-panel.js';
class Node {
 constructor(tag,document){this.tagName=tag.toUpperCase();this.ownerDocument=document;this.children=[];this.dataset={};this.listeners={};this.attributes={};this._value='';this.checked=false;this.disabled=false;this._text='';this.classList={add(){}};this.isConnected=true;}
 append(...nodes){for(const n of nodes){n.parentNode=this;this.children.push(n);}if(this.tagName==='SELECT'&&this.children.length===1)this._value=this.children[0].value;}
 prepend(...nodes){this.children.unshift(...nodes);}
 replaceChildren(...nodes){this.children.forEach(n=>n.isConnected=false);this.children=[];this.append(...nodes);}
 set textContent(t){this._text=String(t);this.children=[];}get textContent(){return this._text+this.children.map(n=>n.textContent).join('');}
 set value(v){this._value=String(v);}get value(){return this._value;}
 get options(){return this.children;}setAttribute(k,v){this.attributes[k]=v;}
 addEventListener(t,f){(this.listeners[t]||=[]).push(f);}async emit(t){for(const f of this.listeners[t]||[])await f({target:this,preventDefault(){}});}
 reportValidity(){return true;}remove(){this.parentNode.children=this.parentNode.children.filter(n=>n!==this);}
}
const document={createElement(tag){return new Node(tag,this);}};globalThis.document=document;
const root=new Node('div',document);const all=(n=root)=>[n,...n.children.flatMap(all)];
const find=text=>all().find(n=>n.tagName==='BUTTON'&&n.textContent===text);
const control=p=>{const w=all().find(n=>n.dataset.contractPath===p);assert(w,p);return all(w).find(n=>['INPUT','SELECT'].includes(n.tagName)&&!n.dataset.contractEnabled);};
const flush=()=>new Promise(r=>setTimeout(r,0));
let task={id:'task-one'},epoch=1;const wire=[];
const contract={schedule_id:'monthly',revision:1,monitoring_ref:'analysis.portfolio_run',active_from:'2025-03-01T00:00:00Z',calendar:{anchor_at:'2025-03-01T00:00:00Z',interval_months:1,timezone:'UTC'},catch_up_budget:1,lease_seconds:600,retry_policy:{max_attempts:1,initial_backoff_seconds:0,multiplier:1,max_backoff_seconds:0},portfolio_binding:{source:{task_id:'task-one',dataset_id:'ds-one',dataset_content_hash:'a'.repeat(64),time_col:'observed_at',timezone:'UTC',complete_through:'2025-04-01T00:00:00Z',target_id:'portfolio',label_mode:'not_applicable'},id_col:'loan',bucket_col:'bucket',states:['current','bad'],balance_col:'balance',segment_col:'segment',loss_state:'bad',lgd:.5,horizon_months:1,lookback_seconds:86400,generate_report:true}};
const proposal={state:'needs_confirmation',proposal_id:'proposal-one',contract_hash:'b'.repeat(64),expires_at:'2026-12-01T00:00:00Z',contract};
"""


def _node(body):
    schema = dict(
        proposal_schema=ProposalRequest.model_json_schema(),
        feedback_import_schema=FeedbackImport.model_json_schema(),
        disposition_schema=DispositionRequest.model_json_schema(),
    )
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            HARNESS
            + "\nconst schemas="
            + json.dumps(schema)
            + ";\n"
            + textwrap.dedent(body),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_actual_typed_form_json_wire_human_confirmation_and_no_double_publish():
    _node(r"""
    const api=async(url,options={})=>{wire.push({url,options});if(url.endsWith('/contracts'))return schemas;if(url.includes('/sources?'))return {datasets:[{id:'ds-one',content_hash:'a'.repeat(64),row_count:12,columns:['observed_at','loan','balance','bucket','segment']}]};if(url.endsWith('/me'))return {role:'admin'};if(url.includes('/schedules?'))return {schedules:[]};if(url.includes('/decision-feedback?'))return {packages:[]};if(url==='/api/operations/proposals'){const v=JSON.parse(options.body);assert.equal(v.expected_revision,0);assert.equal(v.schedule.monitoring_ref,'analysis.portfolio_run');assert.equal(v.schedule.calendar.interval_months,1);assert.equal('actor' in v,false);assert.equal('execution_actor_id' in v.schedule,false);return proposal;}if(url.endsWith('/confirm')){assert.deepEqual(JSON.parse(options.body),{expected_contract_hash:'b'.repeat(64)});return {state:'published',schedule_id:'monthly',revision:1};}throw Error(url);};
    const panel=createOperationsExtensionsPanel({getElement:()=>root,getTask:()=>task,captureView:()=>epoch,isCurrentView:e=>e===epoch,api});await panel.refresh();
    const select=all().find(n=>n.attributes['aria-label']==='调度与反馈操作');select.value='portfolio';await select.emit('change');await flush();
    const form=all().find(n=>n.tagName==='FORM');assert(form);assert.equal(all().some(n=>n.tagName==='TEXTAREA'),false);
    // Fill explicit source time/physical fields; submit installed complete DTO.
    control('schedule.portfolio_binding.source.time_col').value='observed_at';control('schedule.portfolio_binding.source.complete_through').value='2025-04-01T00:00:00Z';
    for(const [k,v] of Object.entries({id_col:'loan',bucket_col:'bucket',balance_col:'balance',segment_col:'segment',loss_state:'bad',lgd:'0.5'}))control('schedule.portfolio_binding.'+k).value=v;
    const add=all().filter(n=>n.tagName==='BUTTON'&&n.textContent==='添加一项');
    // Generic array form is real; add two ordered state literals.
    const array=all().find(n=>n.dataset.contractPath==='schedule.portfolio_binding.states');
    const arrayAdd=all(array).find(n=>n.tagName==='BUTTON'&&n.textContent.startsWith('添加'));await arrayAdd.emit('click');await arrayAdd.emit('click');
    control('schedule.portfolio_binding.states.0').value='current';control('schedule.portfolio_binding.states.1').value='bad';
    await form.emit('submit');assert(find('确认发布这份调度'),root.textContent);assert.equal(wire.filter(x=>x.url.endsWith('/confirm')).length,0);await find('确认发布这份调度').emit('click');assert.equal(wire.filter(x=>x.url.endsWith('/confirm')).length,1);assert.match(root.textContent,/已发布/);assert.equal(find('确认发布这份调度'),undefined);
    """)


def test_stale_async_task_and_stale_confirmation_are_inert():
    _node(r"""
    let release;const deferred=new Promise(r=>release=r);const api=async(url)=>{wire.push({url});if(url.endsWith('/contracts'))return deferred;if(url.includes('/sources?'))return {datasets:[]};if(url.endsWith('/me'))return {role:'admin'};if(url.includes('/schedules?'))return {schedules:[]};if(url.includes('/decision-feedback?'))return {packages:[]};throw Error(url);};
    const panel=createOperationsExtensionsPanel({getElement:()=>root,getTask:()=>task,captureView:()=>epoch,isCurrentView:e=>e===epoch,api});const loading=panel.refresh();task={id:'task-two'};epoch++;panel.render();release(schemas);await loading;assert.equal(all().some(n=>n.attributes['aria-label']==='调度与反馈操作'),false);panel.destroy();assert.equal(root.children.length,0);
    """)
