"""Signed transfer through real native stores; no remote pickle deserialization."""

import base64
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.institution.synchronization import SynchronizationStore
from marvis.institution.policy import actor_context,claim_resource
from marvis.institution.store import IdentityStore
from marvis.packs.modeling.train_tools import tool_train_model
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.plugins.contracts import ToolContext
from marvis.plugins.manifest import manifest_to_dict
from marvis.plugins.publisher_trust import digest,canonical
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.tasks import TaskRepository
from marvis.packs.strategy.contracts import Strategy,StrategyRule
from tests.test_governed_capabilities import _archive,_propose,_confirm


def _app(tmp):
    app=create_app(tmp/'workspace')
    client=TestClient(app)
    client.get('/api/tasks')
    headers={'x-marvis-plugin-admin':app.state.plugin_admin_token}
    task=TaskRepository(app.state.settings.db_path).create_task(TaskCreate(
        model_name='Synthetic synchronization',model_version='reference',validator='Synthetic',source_dir=str(app.state.settings.workspace),
        task_type='modeling',run_mode='manual'))
    return app,client,headers,task.id


def _train(app,task_id,tmp):
    rng=np.random.default_rng(41)
    n=120
    frame=pd.DataFrame({'x':rng.normal(size=n),'z':rng.normal(size=n),'y':np.arange(n)%2,
                        'split':['train']*80+['test']*20+['oot']*20})
    path=tmp/'synthetic.parquet'
    frame.to_parquet(path,index=False)
    registry=DatasetRegistry(DatasetRepository(app.state.settings.db_path),
        DataBackend(app.state.settings.datasets_dir),app.state.settings.datasets_dir)
    dataset=registry.register_from_upload(task_id,path,role='sample')
    ctx=ToolContext(task_id,19,app.state.settings.datasets_dir,app.state.settings.workspace)
    output=tool_train_model({'dataset_id':dataset.id,'features':['x','z'],'target_col':'y',
        'split_col':'split','split_values':{'train':'train','test':'test','oot':'oot'},'recipe':'lr','seed':19},ctx)
    repo=ModelingRepository(app.state.settings.db_path)
    experiment=repo.get_experiment(output['experiment_id'])
    return experiment.artifact_id,frame


def _signed_workflow(app,client,headers,tmp,key,release_version='1.0.0',revision=0):
    public=base64.b64encode(key.public_key().public_bytes_raw()).decode()
    if revision==0:
        response=client.post('/api/plugins/governance/publishers',headers=headers,json={
            'id':'synthetic_vendor','display_name':'Synthetic Vendor','public_key':public})
        assert response.status_code==201,response.text
    manifest=app.state.plugin_registry.get('_sample')
    declaration={'workflow':{'id':'sync_flow','title':'Synthetic transferred workflow','goal_patterns':['transfer workflow'],
        'slots':[],'steps':[{'title':'Echo','tool':{'plugin':'_sample','tool':'echo','version':manifest.version},
        'inputs':{'message':release_version},'depends_on':[],'post_checks':[]}]},
        'tool_bindings':[{'plugin':'_sample','tool':'echo','version':manifest.version,
                         'declaration_hash':digest(manifest_to_dict(manifest))}],
        'permissions':sorted(manifest.permissions)}
    rt=(app,client,headers,key,tmp)
    _confirm(rt,_propose(rt,_archive(rt,version=release_version,workflow=declaration),revision=revision,kind='workflow'))
    return declaration


@pytest.fixture
def pair(tmp_path):
    source=_app(tmp_path/'source')
    destination=_app(tmp_path/'destination')
    stores=[SynchronizationStore(item[0].state.settings) for item in (source,destination)]
    for own,peer in ((0,1),(1,0)):
        node=stores[peer].node()
        response=(source,destination)[own][1].post('/api/instance-sync/trust',headers=(source,destination)[own][2],
                                                json={k:node[k] for k in ('id','public_key')})
        assert response.status_code==201,response.text
    return source,destination,stores,tmp_path


def _transfer(pair,kind,native_ref,name,release_version='1.0.0',revision=0):
    source,destination,stores,_=pair
    response=source[1].post('/api/instance-sync/export',headers=source[2],json={
        'kind':kind,'native_ref':native_ref,'name':name,'release_version':release_version,'receiver_node':stores[1].node()['id']})
    assert response.status_code==200,response.text
    bundle=response.json()
    payload={**bundle,'task_id':destination[3],'expected_revision':revision,
             'approved_permissions':bundle['envelope']['body']['permissions']}
    imported=destination[1].post('/api/instance-sync/import',headers=destination[2],json=payload)
    assert imported.status_code==201,imported.text
    return payload,imported.json()


def test_native_lr_pipeline_real_prediction_and_replay(pair):
    source,destination,_,tmp=pair
    artifact_id,frame=_train(source[0],source[3],tmp)
    payload,receipt=_transfer(pair,'model',artifact_id,'synthetic_lr')
    artifact=ModelingRepository(destination[0].state.settings.db_path).get_model_artifact(receipt['native_ref'])
    source_artifact=ModelingRepository(source[0].state.settings.db_path).get_model_artifact(artifact_id)
    def scores(item,model):
        return _ModelArtifactScorer(model,base_dir=item[0].state.settings.tasks_dir/item[3]/'modeling_artifacts').score(frame)
    assert scores(destination,artifact)==pytest.approx(scores(source,source_artifact),rel=1e-13,abs=1e-13)
    assert ModelingRepository(destination[0].state.settings.db_path).get_experiment(artifact.experiment_id).metrics is None
    replay=destination[1].post('/api/instance-sync/import',headers=destination[2],json=payload)
    assert replay.status_code==201 and replay.json()['replayed'] is True,replay.text
    assert replay.json()['native_ref']==receipt['native_ref']
    readback=destination[1].get('/api/instance-sync/versions/'+receipt['version_id']+'/readback',headers=destination[2])
    assert readback.status_code==200 and readback.json()['verified']


def test_strategy_two_versions_conflict_rollback_and_node_revocation(pair):
    source,destination,stores,_=pair
    for release_version,cutoff,revision in [('1.0.0',0.,0),('2.0.0',1.,1)]:
        strategy=Strategy('synthetic-'+release_version,'approval',(StrategyRule('x > '+str(cutoff),'approve',None),),
                          'x','reject','Synthetic transfer')
        StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],strategy)
        payload,receipt=_transfer(pair,'strategy',strategy.id,'synthetic_strategy',release_version,revision)
        if revision==0:
            first,first_payload=receipt,payload
    stale=destination[1].post('/api/instance-sync/import',headers=destination[2],json=first_payload)
    assert stale.status_code==409,stale.text
    rollback=destination[1].post('/api/instance-sync/rollback',headers=destination[2],json={
        'version_id':first['version_id'],'expected_revision':2,'approved_permissions':[]})
    assert rollback.status_code==200 and rollback.json()['revision']==3,rollback.text
    native=StrategyRepository(destination[0].state.settings.db_path).get_strategy(first['native_ref'])
    assert native.spec.rules[0].condition is not None
    node_id=stores[0].node()['id']
    proposal=destination[1].get('/api/instance-sync/nodes/'+node_id+'/revocation-proposal',headers=destination[2]).json()
    revoke=destination[1].post('/api/instance-sync/nodes/'+node_id+'/revoke',headers=destination[2],
                              json={'expected_heads':proposal['heads'],'proposal_hash':proposal['proposal_hash']})
    assert revoke.status_code==200
    catalog=destination[1].get('/api/instance-sync',headers=destination[2]).json()
    assert all(v['node_revoked'] and not v['active'] and not v['enabled'] for v in catalog['versions'])
    with pytest.raises(PermissionError,match='sync_node_untrusted'):
        StrategyRepository(destination[0].state.settings.db_path).get_strategy(first['native_ref'])


def test_signed_workflow_exact_dependencies_two_versions_and_rollback(pair):
    source,destination,stores,tmp=pair
    key=Ed25519PrivateKey.generate()
    public=base64.b64encode(key.public_key().public_bytes_raw()).decode()
    response=destination[1].post('/api/plugins/governance/publishers',headers=destination[2],json={
        'id':'synthetic_vendor','display_name':'Synthetic Vendor','public_key':public})
    assert response.status_code==201,response.text
    for release_version,revision in [('1.0.0',0),('2.0.0',1)]:
        _signed_workflow(source[0],source[1],source[2],tmp,key,release_version,revision)
        payload,receipt=_transfer(pair,'workflow','sync_flow','sync_flow',release_version,revision)
        if revision==0:
            first=receipt
        assert 'sync_flow' in destination[0].state.skill_report.active
    result=destination[1].post('/api/instance-sync/rollback',headers=destination[2],json={
        'version_id':first['version_id'],'expected_revision':2,'approved_permissions':payload['approved_permissions']})
    assert result.status_code==200,result.text
    from marvis.plugins.capability_versions import CapabilityStore
    row=CapabilityStore(destination[0].state.settings.db_path).resolve('workflow','sync_flow')
    assert row['version']=='1.0.0'
    with connect(destination[0].state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM capability_versions').fetchone()[0]==2
        assert conn.execute('SELECT COUNT(*) FROM instance_resource_versions').fetchone()[0]==2
        plans_before=conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0]
    origin=stores[0].node()['id']
    proposal=destination[1].get('/api/instance-sync/nodes/'+origin+'/revocation-proposal',headers=destination[2]).json()
    revoked=destination[1].post('/api/instance-sync/nodes/'+origin+'/revoke',headers=destination[2],json={
        'expected_heads':proposal['heads'],'proposal_hash':proposal['proposal_hash']})
    assert revoked.status_code==200
    refused=destination[1].post('/api/tasks/'+destination[3]+'/plans',json={'goal':'transfer workflow'})
    assert refused.status_code==403 and refused.json()['detail']['code']=='sync_node_untrusted',refused.text
    with connect(destination[0].state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM plans').fetchone()[0]==plans_before


def test_tamper_unknown_node_permission_failure_and_concurrent_claim(pair):
    source,destination,stores,_=pair
    strategy=Strategy('simple','approval',(StrategyRule('x > 0','approve',None),),'x','reject','Synthetic')
    StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],strategy)
    exported=source[1].post('/api/instance-sync/export',headers=source[2],json={'kind':'strategy','native_ref':'simple',
        'name':'synthetic_strategy','release_version':'1.0.0','receiver_node':stores[1].node()['id']}).json()
    payload={**exported,'task_id':destination[3],'expected_revision':0,'approved_permissions':[]}
    for variant in ('material','signature','permission'):
        bad=json.loads(json.dumps(payload))
        if variant=='material':
            bad['material']['payload']=base64.b64encode(b'{"tampered":true}').decode()
        elif variant=='signature':
            bad['envelope']['signature']=base64.b64encode(b'X'*64).decode()
        else:
            bad['approved_permissions']=['network']
        response=destination[1].post('/api/instance-sync/import',headers=destination[2],json=bad)
        assert response.status_code in {403,422},response.text
    with connect(destination[0].state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM instance_resource_versions').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM strategies').fetchone()[0]==0
    def transfer(_):
        client=TestClient(destination[0])
        client.get('/api/tasks')
        return client.post('/api/instance-sync/import',headers=destination[2],json=payload)
    with ThreadPoolExecutor(2) as pool:
        responses=list(pool.map(transfer,range(2)))
    assert sorted(r.status_code for r in responses) in ([201,201],[201,409]),[r.text for r in responses]
    with connect(destination[0].state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM instance_resource_versions').fetchone()[0]==1
        assert conn.execute('SELECT COUNT(*) FROM strategies').fetchone()[0]==1


def _port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def test_actual_two_workspace_processes_transfer_model_strategy_workflow(tmp_path):
    import httpx
    source,destination=_app(tmp_path/'process-source'),_app(tmp_path/'process-destination')
    artifact_id,frame=_train(source[0],source[3],tmp_path)
    strategy=Strategy('process-strategy','approval',(StrategyRule('x > 0','approve',None),),'x','reject','Synthetic')
    StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],strategy)
    key=Ed25519PrivateKey.generate()
    _signed_workflow(source[0],source[1],source[2],tmp_path,key)
    destination[1].post('/api/plugins/governance/publishers',headers=destination[2],json={
        'id':'synthetic_vendor','display_name':'Synthetic Vendor',
        'public_key':base64.b64encode(key.public_key().public_bytes_raw()).decode()})
    idp_key=Ed25519PrivateKey.generate()
    actors=[]
    for item in (source,destination):
        store=IdentityStore(item[0].state.settings.db_path)
        store.trust_issuer('synthetic-reference-idp','idp-key',
            base64.b64encode(idp_key.public_key().public_bytes_raw()).decode())
        actor=store.provision_subject('synthetic-reference-idp','reference-admin','Reference Admin','admin')
        actors.append(actor)
        claim_resource(store.db_path,actor['id'],'task',item[3],actor['id'],0)
        store.set_mode('institutional','reference-'+item[0].state.settings.workspace.parent.name,0)
    processes=[]
    clients=[]
    code='from marvis.app import create_app; import uvicorn,sys; uvicorn.run(create_app(sys.argv[1]),host="127.0.0.1",port=int(sys.argv[2]),log_level="error")'
    try:
        for item in (source,destination):
            port=_port()
            process=subprocess.Popen([sys.executable,'-c',code,str(item[0].state.settings.workspace),str(port)],
                cwd=Path(__file__).resolve().parents[1],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            processes.append(process)
            client=httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=30)
            clients.append(client)
            deadline=time.monotonic()+20
            while True:
                try:
                    response=client.get('/api/institution/status')
                    if response.status_code==200:
                        break
                except httpx.ConnectError:
                    pass
                assert time.monotonic()<deadline,'reference app process startup timeout'
                time.sleep(.1)
        for client in clients:
            assert client.get('/api/tasks').status_code==401
            challenge=client.post('/api/institution/challenge',json={
                'issuer':'synthetic-reference-idp','subject':'reference-admin'}).json()
            assertion={k:challenge[k] for k in ('issuer','subject','audience','nonce')}
            assertion.update(schema_version='marvis.identity-assertion.v1',algorithm='Ed25519',key_id='idp-key',
                             issued_at=time.time(),expires_at=time.time()+240)
            response=client.post('/api/institution/login',json={'assertion':assertion,
                'signature':base64.b64encode(idp_key.sign(canonical(assertion))).decode()})
            assert response.status_code==200,response.text
            assert client.get('/api/tasks').status_code==200
        nodes=[clients[i].get('/api/instance-sync/node',headers=item[2]).json() for i,item in enumerate((source,destination))]
        for own,peer in ((0,1),(1,0)):
            response=clients[own].post('/api/instance-sync/trust',headers=(source,destination)[own][2],
                json={k:nodes[peer][k] for k in ('id','public_key')})
            assert response.status_code==201,response.text
        receipts={}
        for kind,native_ref,name in [('model',artifact_id,'process_lr'),('strategy','process-strategy','process_strategy'),
                                     ('workflow','sync_flow','sync_flow')]:
            response=clients[0].post('/api/instance-sync/export',headers=source[2],json={
                'kind':kind,'native_ref':native_ref,'name':name,'release_version':'1.0.0','receiver_node':nodes[1]['id']})
            assert response.status_code==200,(kind,response.text)
            payload={**response.json(),'task_id':destination[3],'expected_revision':0,
                     'approved_permissions':response.json()['envelope']['body']['permissions']}
            imported=clients[1].post('/api/instance-sync/import',headers=destination[2],json=payload)
            assert imported.status_code==201,imported.text
            receipts[kind]=imported.json()
            readback=clients[1].get('/api/instance-sync/versions/'+imported.json()['version_id']+'/readback',headers=destination[2])
            assert readback.status_code==200 and readback.json()['verified'],readback.text
            replay=clients[1].post('/api/instance-sync/import',headers=destination[2],json=payload)
            assert replay.status_code==201 and replay.json()['replayed'],replay.text
        created=clients[1].post('/api/tasks/'+destination[3]+'/plans',json={'goal':'transfer workflow'})
        assert created.status_code==201,created.text
        plan=created.json()['plan']
        assert plan['template_id']=='sync_flow'
        confirmation=clients[1].post('/api/plans/'+plan['id']+'/confirm',json=plan['confirmation_snapshot'])
        assert confirmation.status_code==200,confirmation.text
        run=clients[1].post('/api/plans/'+plan['id']+'/run')
        assert run.status_code==202,run.text
        deadline=time.monotonic()+20
        while True:
            finished=clients[1].get('/api/plans/'+plan['id'])
            if finished.json()['plan']['status'] in {'done','failed','cancelled'}:
                break
            assert time.monotonic()<deadline,'native transferred Workflow did not settle'
            time.sleep(.1)
        assert finished.status_code==200 and finished.json()['plan']['status']=='done',finished.text
        output=clients[1].get('/api/step-outputs/'+plan['steps'][0]['id'])
        assert output.status_code==200 and output.json()=={'echoed':'1.0.0'},output.text
        assert processes[0].pid!=processes[1].pid
        with actor_context(destination[0].state.settings.db_path,actors[1]['id']):
            artifact=ModelingRepository(destination[0].state.settings.db_path).get_model_artifact(receipts['model']['native_ref'])
            predicted=_ModelArtifactScorer(artifact,base_dir=destination[0].state.settings.tasks_dir/destination[3]/'modeling_artifacts').score(frame)
        with actor_context(source[0].state.settings.db_path,actors[0]['id']):
            original=ModelingRepository(source[0].state.settings.db_path).get_model_artifact(artifact_id)
            expected=_ModelArtifactScorer(original,base_dir=source[0].state.settings.tasks_dir/source[3]/'modeling_artifacts').score(frame)
        assert predicted==pytest.approx(expected,abs=1e-13)
    finally:
        for client in clients:
            client.close()
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=10)
            (tmp_path/('process-'+str(process.pid)+'-stderr.log')).write_bytes(process.stderr.read())
            process.stderr.close()


def test_node_key_publication_concurrent_and_reuse_of_export_version(pair,tmp_path):
    from marvis.institution.synchronization import node_identity
    workspace=tmp_path/'concurrent-node'
    workspace.mkdir()
    with ThreadPoolExecutor(8) as pool:
        nodes=list(pool.map(lambda _:node_identity(workspace)[0],range(16)))
    assert len({node['id'] for node in nodes})==1
    assert (workspace/'.institution/node-signing.key').stat().st_mode&0o077==0
    source,destination,stores,_=pair
    for native_ref,cutoff in [('same-origin-a',0.),('same-origin-b',1.)]:
        StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],Strategy(
            native_ref,'approval',(StrategyRule('x > '+str(cutoff),'approve',None),),'x','reject','Synthetic'))
    first=source[1].post('/api/instance-sync/export',headers=source[2],json={
        'kind':'strategy','native_ref':'same-origin-a','name':'fixed_strategy','release_version':'1.0.0',
        'receiver_node':stores[1].node()['id']})
    assert first.status_code==200
    second=source[1].post('/api/instance-sync/export',headers=source[2],json={
        'kind':'strategy','native_ref':'same-origin-b','name':'fixed_strategy','release_version':'1.0.0',
        'receiver_node':stores[1].node()['id']})
    assert second.status_code==409
    assert second.json()['detail']['code']=='sync_export_version_reused'


def test_native_model_bytes_are_verified_before_local_load_and_readback(pair,monkeypatch):
    source,destination,stores,tmp=pair
    artifact_id,_=_train(source[0],source[3],tmp)
    payload,receipt=_transfer(pair,'model',artifact_id,'tamper_reference_lr')
    source_repo=ModelingRepository(source[0].state.settings.db_path)
    artifact=source_repo.get_model_artifact(artifact_id)
    path=source[0].state.settings.tasks_dir/source[3]/'modeling_artifacts'/artifact.model_path
    path.write_bytes(path.read_bytes()+b'synthetic-tamper')
    from marvis.packs.modeling import artifact as artifact_module
    calls=[]
    def no_load(*args,**kwargs):
        calls.append(True)
        raise AssertionError('unverified native bytes must not be loaded')
    monkeypatch.setattr(artifact_module,'load_model',no_load)
    response=source[1].post('/api/instance-sync/export',headers=source[2],json={
        'kind':'model','native_ref':artifact_id,'name':'tamper_reference_lr','release_version':'2.0.0',
        'receiver_node':stores[1].node()['id']})
    assert response.status_code==409 and response.json()['detail']['code']=='sync_native_source_changed',response.text
    assert not calls
    imported=ModelingRepository(destination[0].state.settings.db_path).get_model_artifact(receipt['native_ref'])
    target=destination[0].state.settings.tasks_dir/destination[3]/'modeling_artifacts'/imported.model_path
    target.write_bytes(target.read_bytes()+b'synthetic-tamper')
    response=destination[1].get('/api/instance-sync/versions/'+receipt['version_id']+'/readback',headers=destination[2])
    assert response.status_code==409 and response.json()['detail']['code']=='sync_native_readback_changed',response.text
    replay=destination[1].post('/api/instance-sync/import',headers=destination[2],json=payload)
    assert replay.status_code==409,replay.text
    assert not calls


def test_actual_browser_json_round_trip_preserves_signed_strategy_bytes(pair):
    source,destination,stores,_=pair
    StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],Strategy(
        'browser-float-strategy','approval',(StrategyRule('x > 0.0','approve',None),),'x','reject','Synthetic'))
    exported=source[1].post('/api/instance-sync/export',headers=source[2],json={
        'kind':'strategy','native_ref':'browser-float-strategy','name':'browser_float_strategy',
        'release_version':'1.0.0','receiver_node':stores[1].node()['id']})
    assert exported.status_code==200,exported.text
    original=exported.json()
    # This is the real JavaScript parse/stringify operation of the file form.
    result=subprocess.run(['node','-e','let s="";process.stdin.on("data",x=>s+=x);process.stdin.on("end",()=>process.stdout.write(JSON.stringify(JSON.parse(s))));'],
                          input=exported.text,capture_output=True,text=True,check=True)
    browser=json.loads(result.stdout)
    assert browser['material']==original['material']
    assert browser['envelope']['body']['schema_version']=='marvis.instance-resource.v2'
    raw=base64.b64decode(browser['material']['payload'])
    assert canonical(json.loads(raw))==raw
    response=destination[1].post('/api/instance-sync/import',headers=destination[2],json={
        **browser,'task_id':destination[3],'expected_revision':0,'approved_permissions':[]})
    assert response.status_code==201,response.text
    checked=destination[1].get('/api/instance-sync/versions/'+response.json()['version_id']+'/readback',headers=destination[2])
    assert checked.status_code==200 and checked.json()['verified'],checked.text


def test_wire_material_preserves_float_and_large_integer_custody():
    from marvis.institution.synchronization import _encode_material,_decode_material
    values={'zero':-0.0,'exponent':1e-7,'large':2**60,'bool':True}
    wire=_encode_material(values)
    result=subprocess.run(['node','-e','let s="";process.stdin.on("data",x=>s+=x);process.stdin.on("end",()=>process.stdout.write(JSON.stringify(JSON.parse(s))));'],
                          input=json.dumps(wire),capture_output=True,text=True,check=True)
    restored=_decode_material(json.loads(result.stdout))
    assert canonical(restored)==canonical(values)
    assert type(restored['zero']) is float and type(restored['large']) is int
    assert type(restored['bool']) is bool


@pytest.mark.parametrize('raw',[b'{"x":1,"x":2}',b'{"x":NaN}',b'[]'])
def test_valid_signature_cannot_admit_noncanonical_or_duplicate_material(pair,raw):
    from marvis.institution.synchronization import node_identity,SCHEMA
    source,destination,stores,_=pair
    node,key=node_identity(source[0].state.settings.workspace)
    material={'encoding':'base64.canonical-json.v1','payload':base64.b64encode(raw).decode()}
    body={'schema_version':SCHEMA,'origin_node':node['id'],'receiver_node':stores[1].node()['id'],
          'kind':'strategy','name':'invalid_bytes','version':'1.0.0','material_hash':digest(material),
          'permissions':[],'expires_at':time.time()+900}
    envelope={'body':body,'signature':base64.b64encode(key.sign(canonical(body))).decode()}
    response=destination[1].post('/api/instance-sync/import',headers=destination[2],json={
        'envelope':envelope,'material':material,'task_id':destination[3],'expected_revision':0,'approved_permissions':[]})
    assert response.status_code==422 and response.json()['detail']['code']=='sync_material_contract',response.text
    with connect(destination[0].state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM instance_resource_versions').fetchone()[0]==0
        assert conn.execute('SELECT COUNT(*) FROM strategies').fetchone()[0]==0
