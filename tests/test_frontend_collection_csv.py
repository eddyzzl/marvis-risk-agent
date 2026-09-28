"""Bulk collection mapping preserves literal source values and native write boundaries."""
from tests.test_frontend_collection import node


CSV = r'''
import {parseCsv,mapCsv,csvOperations} from './marvis/static/js/collection-csv-form.js';
const unit={currency:'CNY',minor_unit_exponent:2,definition_source:'Explicit'};
const cases=new Map([['c',{case_id:'c',unit}]]);
'''


def test_csv_preserves_quoted_values_leading_zeroes_and_refuses_ambiguous_rows():
    node(CSV + r'''
assert.deepEqual(parseCsv('\ufeffid,label\r\n0001,"a,b"\r\n0002,"line\nwith ""quote"""\r\n'),{headers:['id','label'],rows:[['0001','a,b'],['0002','line\nwith "quote"']]});
for(const text of ['id,id\n1,2','id,label\n1','id\n"open','id\n"closed"extra','id,\n1,2'])assert.throws(()=>parseCsv(text));
assert.throws(()=>parseCsv('id\n'+'x\n'.repeat(1001)),/1000/);
''')


def test_mapping_requires_explicit_columns_or_constants_and_blank_availability_is_unknown():
    node(CSV + r'''
const data=parseCsv('case,event,amount,time\nc,0001,100,2026-08-01T00:00:00Z');
const mapping={case_id:{source:'column:0'},event_id:{source:'column:1'},amount_minor:{source:'column:2'},event_at:{source:'column:3'},source_id:{source:'constant',value:'bank'},kind:{source:'constant',value:'payment'}};
const rows=mapCsv('cashflow',data,mapping);const ops=csvOperations('cashflow',rows,cases);assert.equal(ops.length,1);const e=ops[0].value.events[0];assert.equal(e.event_id,'0001');assert.equal(e.amount_minor,100);assert.equal(e.available_at,null);assert.deepEqual(e.unit,unit);
assert.throws(()=>mapCsv('cashflow',data,{...mapping,kind:{source:''}}),/流水类型/);
assert.throws(()=>csvOperations('cashflow',[{...rows[0],amount_minor:'1.50'}],cases),/整数/);
assert.throws(()=>csvOperations('cashflow',[rows[0],rows[0]],cases),/重复/);
assert.throws(()=>csvOperations('cashflow',[{...rows[0],available_at:'2026-07-31T00:00:00Z'}],cases),/不得早于/);
''')


def test_schedule_groups_by_case_and_schedule_and_never_reorders_or_invents_periods():
    node(CSV + r'''
const rows=[{case_id:'c',schedule_id:'s',installment_id:'p1',due_at:'2026-08-01T00:00:00Z',amount_minor:'100'},{case_id:'c',schedule_id:'s',installment_id:'p2',due_at:'2026-09-01T00:00:00Z',amount_minor:'200'}];
const ops=csvOperations('schedule',rows,cases);assert.equal(ops.length,1);assert.equal(ops[0].value.installments.length,2);assert.deepEqual(ops[0].value.unit,unit);
assert.throws(()=>csvOperations('schedule',rows.toReversed(),cases),/应还时间排序/);
assert.throws(()=>csvOperations('schedule',[rows[0],rows[0]],cases),/重复/);
assert.throws(()=>csvOperations('schedule',rows,new Map()),/已登记案件/);
''')


ASYNC = CSV + r'''
import {createCollectionCsvController} from './marvis/static/js/collection-csv-controller.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let owner={taskId:'A',principal:{id:'maker',role:'maker'},pending:false},calls=[],messages=[],refreshes=0;
const nodes=new Map();const blank=()=>({innerHTML:'',querySelectorAll:()=>[],querySelector:()=>null});
const root={querySelector:s=>{if(!nodes.has(s))nodes.set(s,blank());return nodes.get(s)}};
const bulk=createCollectionCsvController({getRoot:()=>root,getOwner:()=>owner,isCurrent:o=>o===owner,apiClient:(url,options)=>new Promise((resolve,reject)=>calls.push({url,options,resolve,reject})),message:s=>messages.push(s),setPending:(o,p)=>o.pending=p,refresh:async()=>refreshes++,onOpen(){}});
const click=action=>bulk.handle({type:'click',target:{closest:s=>s==='[data-collection-action]'?{dataset:{collectionAction:action}}:null},preventDefault(){}});
const fields=['case_id','subject_namespace','subject_token','currency','minor_unit_exponent','definition_source','opening_balance_minor','opened_at'];
const mappings=fields.map((name,i)=>({dataset:{csvField:name},value:'column:'+i}));
const form={elements:{csv_import_id:{value:'import-1'},csv_description:{value:'explicit declarations'}},querySelectorAll:s=>s==='[data-csv-field]'?mappings:[],querySelector:s=>({value:''})};
const change=el=>bulk.handle({type:'change',target:{...el,closest:s=>s==='[data-collection-bulk]'?form:null,dataset:el.dataset||{}},preventDefault(){}});
const text='case,namespace,token,currency,scale,definition,balance,opened\nc1,bank,'+'a'.repeat(64)+',CNY,2,Explicit,100,2026-08-01T00:00:00Z\nc2,bank,'+'b'.repeat(64)+',CNY,2,Explicit,200,2026-08-01T00:00:00Z';
async function prepared(){click('import-csv');change({name:'csv_kind',value:'case'});change({name:'csv_file',files:[{size:text.length,name:'cases.csv',arrayBuffer:async()=>new TextEncoder().encode(text).buffer}]});await tick();bulk.handle({type:'submit',target:{closest:()=>form},preventDefault(){}});await tick();}
'''


def test_csv_preview_writes_nothing_and_late_material_never_continues_after_task_switch():
    node(ASYNC + r'''
await prepared();assert.equal(calls.length,0);assert.match(root.querySelector('[data-collection-csv-preview]').innerHTML,/2 次业务登记/);click('csv-commit');assert.equal(calls.length,1);
owner={taskId:'B',principal:{id:'maker',role:'maker'},pending:false};click('import-csv');const count=messages.length;calls[0].resolve({source_artifact_id:'old',source_artifact_hash:'a'.repeat(64)});await tick();assert.equal(calls.length,1);assert.equal(messages.length,count);assert.equal(owner.pending,false);assert.equal(refreshes,0);
''')


def test_csv_partial_failure_retains_exact_source_and_retries_only_unconfirmed_item():
    node(ASYNC + r'''
await prepared();click('csv-commit');calls[0].resolve({source_artifact_id:'native',source_artifact_hash:'a'.repeat(64)});await tick();assert.equal(calls.length,2);const first=JSON.parse(calls[1].options.body);assert.equal(first.case_id,'c1');calls[1].resolve(first);await tick();const second=JSON.parse(calls[2].options.body);assert.equal(second.case_id,'c2');calls[2].reject({status:409,detail:{code:'conflict'}});await tick();assert.match(messages.at(-1),/已登记 1 \/ 2/);assert.equal(owner.pending,false);
click('csv-commit');assert.equal(calls.length,4);assert.equal(calls[3].url,calls[2].url);assert.equal(calls[3].options.body,calls[2].options.body);calls[3].resolve(second);await tick();assert.equal(calls.length,4);assert.equal(refreshes,1);assert.match(messages.at(-1),/全部登记/);
''')
