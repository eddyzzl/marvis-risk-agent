"""Actual ES module and shared wire, with a small explicit DOM carrier."""

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).parents[1]
HARNESS = r"""
import assert from 'node:assert/strict';
import {createGovernedCapabilitiesPanel} from './marvis/static/js/governed_capabilities_panel.js';
class Node {
  constructor(tag,document){this.tagName=tag.toUpperCase();this.ownerDocument=document;this.children=[];this.dataset={};this.listeners={};this.attributes={};this._value='';this.checked=false;this.disabled=false;this._text='';}
  append(...nodes){for(const node of nodes){node.parentNode=this;this.children.push(node);}if(this.tagName==='SELECT'&&this.children.length===1)this._value=this.children[0].value;}
  replaceChildren(...nodes){this.children=[];this.append(...nodes);}
  set textContent(text){this._text=String(text);this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
  set value(value){this._value=String(value);}get value(){return this._value;}
  get options(){return this.children;}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(type,handler){(this.listeners[type]||=[]).push(handler);}
  async emit(type){for(const handler of this.listeners[type]||[])await handler({target:this});}
}
const document={baseURI:'http://127.0.0.1/',body:{dataset:{}},createElement(tag){return new Node(tag,this);}};
globalThis.document=document;
globalThis.localStorage={getItem(){throw Error('no private state storage');},setItem(){throw Error('no private state storage');}};
const root=new Node('div',document);
const all=(node=root)=>[node,...node.children.flatMap(child=>all(child))];
const field=name=>all().find(node=>node.dataset.capabilityField===name);
const action=name=>all().find(node=>node.dataset.capabilityAction===name);
const status=()=>all().find(node=>node.dataset.capabilityStatus).textContent;
const initial=()=>({settings:{mode:'trusted_local',revision:0},publishers:[],versions:[]});
const proposed=()=>({id:'proposal-exact',kind:'plugin',name:'synthetic_pack',version:'1.0.0',publisher_id:'vendor',
  content_hash:'a'.repeat(64),declaration_hash:'b'.repeat(64),proposal_hash:'c'.repeat(64),expected_revision:0,
  permissions:['read:artifacts'],delta:{added_permissions:['read:artifacts'],removed_permissions:[]},declaration:{tools:[{name:'echo'}],hooks:[],permissions:['read:artifacts']}});
const reply=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'content-type':'application/json'}});
"""


def _node(body):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", HARNESS + textwrap.dedent(body)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_signed_upload_is_multipart_and_permission_confirmation_is_exact_actual_json_wire():
    _node(r"""
      const wire=[];let registered=false;let callback;
      globalThis.fetch=async(url,options={})=>{
        wire.push({url,options});
        if(url.includes('/proposals?')){
          assert(options.body instanceof FormData);assert(options.body.get('file') instanceof Blob);
          assert.equal(options.headers['x-marvis-plugin-admin'],'synthetic-admin');
          assert.equal(Object.keys(options.headers).some(x=>x.toLowerCase()==='content-type'),false);
          assert.match(url,/expected_revision=0/);return reply(proposed(),201);
        }
        if(url.endsWith('/confirm')){
          assert.equal(typeof options.body,'string');const body=JSON.parse(options.body);
          assert.deepEqual(body,{proposal_hash:'c'.repeat(64),expected_revision:0,approved_permissions:['read:artifacts']});
          assert.equal('actor' in body,false);registered=true;
          return reply({name:'synthetic_pack',version:'1.0.0',revision:1},201);
        }
        const overview=initial();if(registered)overview.versions=[{kind:'plugin',name:'synthetic_pack',version:'1.0.0',active:1,enabled:1,head_revision:1,content_hash:'a'.repeat(64)}];
        return reply(overview);
      };
      const panel=createGovernedCapabilitiesPanel({container:root,onChanged:value=>callback=value});await panel.ready;
      field('admin').value='synthetic-admin';field('file').files=[new Blob(['signed synthetic bytes'])];
      await action('propose').emit('click');assert.equal(registered,false);
      assert.equal(action('confirm').disabled,true);
      field('reviewed').checked=true;await field('reviewed').emit('change');assert.equal(action('confirm').disabled,true);
      field('permission:read:artifacts').checked=true;await field('permission:read:artifacts').emit('change');
      assert.equal(action('confirm').disabled,false);await action('confirm').emit('click');
      assert.equal(registered,true);assert.equal(callback.version,'1.0.0');assert.match(status(),/既有计划保留/);
      assert.equal(wire.filter(call=>call.url.endsWith('/confirm')).length,1);
      const secret=field('admin');panel.destroy();assert.equal(secret.value,'');assert.equal(root.children.length,0);
    """)


def test_head_conflict_keeps_reviewed_proposal_without_automatic_retry():
    _node(r"""
      let confirmations=0;
      globalThis.fetch=async(url,options={})=>{
        if(url.includes('/proposals?'))return reply(proposed(),201);
        if(url.endsWith('/confirm')){confirmations++;return reply({detail:'capability_head_conflict'},409);}
        return reply(initial());
      };
      const panel=createGovernedCapabilitiesPanel({container:root,getAdminToken:()=> 'synthetic'});await panel.ready;
      field('file').files=[new Blob(['signed synthetic'])];await action('propose').emit('click');
      field('reviewed').checked=true;field('permission:read:artifacts').checked=true;await field('reviewed').emit('change');
      await action('confirm').emit('click');assert.equal(confirmations,1);assert.match(status(),/版本已变化/);
      assert(field('reviewed').checked);assert(field('permission:read:artifacts').checked);assert(action('confirm'));
      await panel.refresh();assert.equal(confirmations,1);panel.destroy();
    """)


def test_global_mode_wire_is_cas_bound_and_destroy_discards_late_response():
    _node(r"""
      const calls=[];let later;
      globalThis.fetch=async(url,options={})=>{
        calls.push({url,options});
        if(options.method==='PUT'){
          assert.deepEqual(JSON.parse(options.body),{mode:'governed',expected_revision:0,reason:'synthetic reason'});
          return reply({mode:'governed',revision:1});
        }
        if(calls.length>2)return new Promise(resolve=>later=resolve);
        return reply(initial());
      };
      const panel=createGovernedCapabilitiesPanel({container:root,getAdminToken:()=> 'synthetic'});await panel.ready;
      field('mode').value='governed';field('modeReason').value='synthetic reason';
      const pending=action('mode').emit('click');
      for(let i=0;i<20&&!later;i++)await Promise.resolve();
      assert(later);panel.destroy();later(reply(initial()));await pending;assert.equal(root.children.length,0);
    """)
