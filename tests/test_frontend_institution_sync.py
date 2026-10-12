"""Dedicated panels use actual shared API serialization and mounted fetch URLs."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
import assert from 'node:assert/strict';
import {createInstitutionPanel} from './marvis/static/js/institution_panel.js';
import {createInstanceSynchronizationPanel} from './marvis/static/js/instance_sync_panel.js';
globalThis.document={baseURI:'http://127.0.0.1/proxy/8123/',body:{dataset:{marvisLocalToken:'synthetic-local-boundary'}}};
const reply=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'content-type':'application/json'}});
class Element {
 constructor(){this.value='';this.checked=false;this.textContent='';this.dataset={};this.disabled=false;this.files=[];}
 closest(){return this;}
}
class Container {
 constructor(){this.fields=new Map;this.parts=new Map;this.listeners=new Map;this.buttons=[];}
 set innerHTML(html){this.html=html;for(const m of html.matchAll(/data-input="([^"]+)"/g))this.fields.set(m[1],new Element);this.fields.get('revision')&&(this.fields.get('revision').value='0');}
 querySelector(selector){let m=selector.match(/^\[data-input="([^"]+)"\]$/);if(m)return this.fields.get(m[1]);if(!this.parts.has(selector))this.parts.set(selector,new Element);return this.parts.get(selector);}
 querySelectorAll(selector){if(selector==='button')return this.buttons;if(selector==='[data-actions] input:checked')return [{value:'read'},{value:'execute'}];return [];}
 addEventListener(type,handler){this.listeners.set(type,handler);}
 removeEventListener(type){this.listeners.delete(type);}
 contains(){return true;}
 async click(action,id){const b=new Element;b.dataset={action,id};this.listeners.get('click')?.({target:b});await new Promise(r=>setTimeout(r,25));}
 async file(file){this.fields.get('file').files=[file];this.listeners.get('change')?.({target:this.fields.get('file')});await new Promise(r=>setTimeout(r,25));}
}
const wire=[];
function record(url,options){const parsed=new URL(url);assert.equal(parsed.origin,'http://127.0.0.1');assert.ok(parsed.pathname.startsWith('/proxy/8123/api/'),parsed.pathname);assert.equal(options.headers['X-Marvis-Token'],'synthetic-local-boundary');const path=parsed.pathname.slice('/proxy/8123'.length);let body;if(options.body!==undefined){assert.equal(typeof options.body,'string');body=JSON.parse(options.body);assert.equal(options.headers['Content-Type'],'application/json');}wire.push({path,body,method:options.method||'GET'});return {path,body};}
'''


def _run(script):
    result = subprocess.run(['node','--input-type=module','-e',HARNESS+script],cwd=ROOT,
                            capture_output=True,text=True)
    assert result.returncode == 0,result.stderr


def test_identity_real_wire_mode_login_grants_and_cache_invalidation():
    _run(r'''
let mode='trusted_local',revision=0;let changed=0;
globalThis.fetch=async(url,options={})=>{const {path,body}=record(url,options);
 if(path==='/api/institution/status')return reply({mode,revision,audience:'synthetic',protocol:'marvis.identity-assertion.v1',algorithm:'Ed25519'});
 if(path==='/api/institution/me')return reply({id:'institution-admin',display_name:'Synthetic admin',role:'admin'});
 if(path==='/api/institution/mode'){assert.equal(body.expected_revision,0);mode=body.mode;++revision;return reply({revision});}
 if(path==='/api/institution/login'){assert.deepEqual(body,{assertion:{nonce:'synthetic'},signature:'synthetic'});return reply({id:'institution-admin'});}
 if(path.startsWith('/api/institution/'))return reply({id:'synthetic',revision:1});
 throw Error('unexpected route '+path);
};
const root=new Container,panel=createInstitutionPanel({container:root,onIdentityChange:()=>{++changed;}});await panel.ready;
const field=n=>root.fields.get(n);
for(const [n,v]of Object.entries({admin:'synthetic-admin',issuer:'synthetic-idp','key-id':'key', 'public-key':'public',subject:'admin','display-name':'Admin',role:'admin',audience:'synthetic'}))field(n).value=v;
field('login-issuer').value='synthetic-idp';field('login-subject').value='admin';await root.click('challenge');assert.equal(wire.find(x=>x.path==='/api/institution/challenge').body.subject,'admin');
await root.click('trust');await root.click('provision');field('mode').value='institutional';await root.click('mode');
assert.equal(changed,1);field('assertion').value=JSON.stringify({assertion:{nonce:'synthetic'},signature:'synthetic'});await root.click('login');assert.equal(field('assertion').value,'');await root.click('logout');assert.equal(changed,3);
for(const [n,v]of Object.entries({kind:'task',resource:'task-a',target:'actor-b'}))field(n).value=v;
await root.click('grant');await root.click('revoke-grant');await root.click('owner');
const grant=wire.find(x=>x.path==='/api/institution/grants');assert.deepEqual(grant.body.actions,['read','execute']);assert.equal(grant.body.expected_revision,0);assert.equal(grant.body.actor_id,undefined);
assert.equal(wire.find(x=>x.path==='/api/institution/grants/revoke').body.expected_revision,1);
panel.destroy();assert.equal(field('admin').value,'');
''')


def test_sync_real_wire_reviewed_heads_permissions_and_task_change():
    _run(r'''
const root=new Container;let task='task-a',holdImport=null,importResolve=null;
const heads=[{kind:'workflow',resource_name:'flow',version_id:'version-new',revision:2,enabled:1}];
const catalog={node:{id:'node-local',public_key:'public'},versions:[{id:'version-old',kind:'workflow',resource_name:'flow',version:'1.0.0',revision:2,active:false,node_revoked:false,permissions:['filesystem.read']}]};
globalThis.fetch=async(url,options={})=>{const {path,body}=record(url,options);
 if(path==='/api/institution/status')return reply({mode:'trusted_local'});
 if(path==='/api/instance-sync')return reply(catalog);
 if(path.endsWith('/revocation-proposal'))return reply({node_id:'node-peer',heads,proposal_hash:'synthetic-proposal'});
 if(path.endsWith('/revoke')){assert.deepEqual(body,{expected_heads:heads,proposal_hash:'synthetic-proposal'});return reply({revoked:true});}
 if(path==='/api/instance-sync/rollback'){assert.deepEqual(body.approved_permissions,['filesystem.read']);assert.equal(body.expected_revision,2);return reply({native_ref:'workflow:old',revision:3});}
 if(path==='/api/instance-sync/import'){assert.equal(body.task_id,'task-a');assert.deepEqual(body.approved_permissions,['filesystem.read']);if(holdImport)await new Promise(r=>importResolve=r);return reply({native_ref:'workflow:new',revision:7});}
 if(path==='/api/instance-sync/trust')return reply({trusted:true});
 if(path.endsWith('/readback'))return reply({verified:true});
 throw Error('unexpected route '+path);
};
const panel=createInstanceSynchronizationPanel({container:root,getTaskId:()=>task});await panel.ready;
assert.equal(wire.some(x=>x.path==='/api/instance-sync'),false);root.fields.get('admin').value='synthetic';await root.click('refresh');
root.fields.get('node-id').value='node-peer';root.fields.get('public-key').value='public';await root.click('trust');await root.click('revoke');assert.equal(wire.some(x=>x.path.endsWith('/revoke')),false);
await root.click('review-revoke');root.fields.get('revoke-approved').checked=true;await root.click('revoke');
await root.click('rollback','version-old');assert.equal(wire.some(x=>x.path==='/api/instance-sync/rollback'),false);
root.querySelector('[data-version-approval="version-old"]').checked=true;await root.click('rollback','version-old');await root.click('readback','version-old');
const bundle={envelope:{body:{kind:'workflow',name:'flow',version:'2.0.0',origin_node:'node-peer',permissions:['filesystem.read']}},material:{archive:'synthetic'}};
await root.file({size:20,text:async()=>JSON.stringify(bundle)});root.fields.get('approved').checked=true;
holdImport=true;const pending=root.click('import');await new Promise(r=>setTimeout(r,5));task='task-b';importResolve();await pending;
assert.equal(root.fields.get('revision').value,'0');panel.destroy();assert.equal(root.fields.get('admin').value,'');
''')
