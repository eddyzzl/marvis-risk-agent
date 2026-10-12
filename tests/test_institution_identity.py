"""Real HTTP cookies, signature admission, native lookups and current grants."""

import base64
from concurrent.futures import ThreadPoolExecutor
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.db_schema import connect
from marvis.institution.policy import actor_context, authorize, grant_resource
from marvis.institution.store import IdentityError, IdentityStore
from marvis.plugins.publisher_trust import canonical
from marvis.repositories.tasks import TaskRepository


@pytest.fixture
def institution(tmp_path):
    app = create_app(tmp_path/'workspace')
    bootstrap = TestClient(app)
    bootstrap.get('/api/tasks')
    key = Ed25519PrivateKey.generate()
    headers = {'x-marvis-plugin-admin':app.state.plugin_admin_token}
    issuer = {'issuer':'synthetic-reference-idp','key_id':'reference-key',
              'public_key':base64.b64encode(key.public_key().public_bytes_raw()).decode()}
    response = bootstrap.post('/api/institution/issuers',json=issuer,headers=headers)
    assert response.status_code == 201,response.text
    subjects = {}
    for subject,role in [('admin','admin'),('alice','maker'),('bob','checker'),('automation','service')]:
        response = bootstrap.post('/api/institution/subjects',headers=headers,json={
            'issuer':issuer['issuer'],'subject':subject,'display_name':'Synthetic '+subject,'role':role})
        assert response.status_code == 201,response.text
        subjects[subject] = response.json()
    response = bootstrap.put('/api/institution/mode',headers=headers,json={
        'mode':'institutional','audience':'synthetic-instance','expected_revision':0})
    assert response.status_code == 200,response.text
    return app,key,subjects,bootstrap


def _assertion(institution, subject, client=None):
    app,key,subjects,_ = institution
    client = client or TestClient(app)
    challenge = client.post('/api/institution/challenge',json={
        'issuer':subjects[subject]['issuer'],'subject':subject}).json()
    assertion = {k:challenge[k] for k in ('issuer','subject','audience','nonce')}
    assertion.update(schema_version='marvis.identity-assertion.v1',key_id='reference-key',
                     algorithm='Ed25519',issued_at=time.time(),expires_at=time.time()+240)
    return {'assertion':assertion,'signature':base64.b64encode(key.sign(canonical(assertion))).decode()}


def _login(institution, subject):
    client = TestClient(institution[0])
    response = client.post('/api/institution/login',json=_assertion(institution,subject,client))
    assert response.status_code == 200,response.text
    return client


def _task(client, name='Synthetic owned strategy'):
    response = client.post('/api/tasks',json={'task_type':'strategy','model_name':name,
                                             'validator':'Synthetic','run_mode':'manual'})
    assert response.status_code == 200,response.text
    return response.json()['id']


def test_no_local_cookie_fallback_and_server_roles(institution):
    app,key,subjects,old_local = institution
    assert old_local.get('/api/tasks').status_code == 401
    alice = _login(institution,'alice')
    assert alice.get('/api/institution/me').json()['role']=='maker'
    assert alice.post('/api/production-governance/principals/claim',json={
        'display_name':'Spoof','role':'admin'},headers={'x-marvis-governance-admin':app.state.plugin_admin_token}).status_code==403
    assert alice.post('/api/institution/subjects',json={
        'issuer':'synthetic-reference-idp','subject':'spoof','display_name':'Spoof','role':'admin'}).status_code==403
    admin = _login(institution,'admin')
    assert admin.post('/api/production-governance/principals/claim',json={
        'display_name':'Spoof','role':'admin'},headers={'x-marvis-governance-admin':app.state.plugin_admin_token}).status_code==403
    body=_assertion(institution,'alice')
    body['assertion']['role']='admin'
    assert alice.post('/api/institution/login',json=body).status_code==401
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM institution_sessions').fetchone()[0]==2
        assert conn.execute('SELECT mode FROM capability_settings').fetchone()[0]=='governed'


@pytest.mark.parametrize('change',['signature','algorithm','issuer','audience','expiry','key','subject'])
def test_exact_assertion_rejections(institution,change):
    body=_assertion(institution,'alice')
    app,key,_,_=institution
    if change=='signature':
        body['signature']=base64.b64encode(b'X'*64).decode()
    else:
        field={'algorithm':'algorithm','issuer':'issuer','audience':'audience','expiry':'expires_at',
               'key':'key_id','subject':'subject'}[change]
        body['assertion'][field]=time.time()-1 if change=='expiry' else 'unknown'
        body['signature']=base64.b64encode(key.sign(canonical(body['assertion']))).decode()
    response=TestClient(app).post('/api/institution/login',json=body)
    assert response.status_code==401,response.text
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM institution_sessions').fetchone()[0]==0


def test_nonce_once_concurrent_and_cookie_custody(institution):
    body=_assertion(institution,'alice')
    def login(_):
        return TestClient(institution[0]).post('/api/institution/login',json=body)
    with ThreadPoolExecutor(2) as pool:
        responses=list(pool.map(login,range(2)))
    assert sorted(r.status_code for r in responses)==[200,401]
    success=next(r for r in responses if r.status_code==200)
    assert 'HttpOnly' in success.headers['set-cookie'] and 'SameSite=strict' in success.headers['set-cookie']
    token=success.cookies['marvis_institution_session']
    with connect(institution[0].state.settings.db_path) as conn:
        row=dict(conn.execute('SELECT * FROM institution_sessions').fetchone())
        assert token not in str(row) and len(row['token_hash'])==64


def test_task_owner_grants_cas_aliases_and_native_backend(institution,tmp_path):
    app,_,subjects,_=institution
    alice,bob=_login(institution,'alice'),_login(institution,'bob')
    task=_task(alice)
    assert alice.get('/api/tasks/'+task).status_code==200
    assert bob.get('/api/tasks/'+task).status_code==403
    assert bob.get('/api/tasks').json()==[]
    assert [r['id'] for r in alice.get('/api/tasks?limit=1').json()]==[task]
    denied=alice.post('/api/tasks/'+task+'/datasets/register-path',json={'path':str(tmp_path/'x.csv'),'role':'sample'})
    assert denied.status_code==403
    upload=alice.post('/api/tasks/'+task+'/datasets/upload',data={'role':'sample'},
        files={'file':('synthetic.csv',b'x,y\n1,0\n2,1\n','text/csv')})
    assert upload.status_code==201,upload.text
    dataset=upload.json()['datasets'][0]['id']
    assert bob.get('/api/datasets/'+dataset+'/preview').status_code==403
    assert alice.get('/api/datasets/'+dataset+'/preview').status_code==200
    with actor_context(app.state.settings.db_path,subjects['alice']['id'],task):
        from marvis.data.registry import DatasetRegistry
        from marvis.repositories.datasets import DatasetRepository
        registry=DatasetRegistry(DatasetRepository(app.state.settings.db_path),DataBackend(app.state.settings.datasets_dir),app.state.settings.datasets_dir)
        registered=registry.get(dataset)
    with actor_context(app.state.settings.db_path,subjects['bob']['id']):
        with pytest.raises(IdentityError):
            TaskRepository(app.state.settings.db_path).update_algorithm(task,'lr')
        with pytest.raises(IdentityError):
            DataBackend(app.state.settings.datasets_dir).read_frame(registered.source_path)
    grant={'kind':'task','resource_id':task,'subject_id':subjects['bob']['id'],'actions':['read'],
           'expected_revision':0}
    response=alice.post('/api/institution/grants',json=grant)
    assert response.status_code==200,response.text
    assert alice.post('/api/institution/grants',json=grant).status_code==409
    assert bob.get('/api/tasks/'+task).status_code==200
    assert bob.get('/api/datasets/'+dataset+'/preview').status_code==200
    assert bob.post('/api/tasks/'+task+'/datasets/register-path',json={}).status_code==403
    revoked=alice.post('/api/institution/grants/revoke',json={k:v for k,v in {**grant,'expected_revision':1}.items() if k!='actions'})
    assert revoked.status_code==200,revoked.text
    assert bob.get('/api/datasets/'+dataset+'/preview').status_code==403


def test_service_expiring_scope_and_current_revocation(institution):
    app,_,subjects,_=institution
    alice,service,admin=(_login(institution,s) for s in ('alice','automation','admin'))
    task=_task(alice)
    grant={'kind':'task','resource_id':task,'subject_id':subjects['automation']['id'],
           'actions':['read','execute'],'expected_revision':0}
    assert alice.post('/api/institution/grants',json=grant).status_code==200
    assert service.get('/api/tasks/'+task).status_code==200
    assert service.post('/api/plans/missing/confirm',json={}).status_code==403
    with pytest.raises(IdentityError,match='service_scope'):
        grant_resource(app.state.settings.db_path,subjects['alice']['id'],'task',task,
                       subjects['automation']['id'],['approve'],1)
    assert admin.post('/api/institution/revoke',json={'kind':'subject','resource_id':subjects['automation']['id']}).status_code==200
    assert service.get('/api/tasks/'+task).status_code==401
    assert admin.post('/api/institution/revoke',json={'kind':'issuer','resource_id':'synthetic-reference-idp','key_id':'reference-key'}).status_code==200
    assert alice.get('/api/tasks/'+task).status_code==401


def test_no_actor_native_lookup_or_hidden_source(institution,tmp_path):
    app,_,subjects,_=institution
    alice=_login(institution,'alice')
    task=_task(alice)
    path=app.state.settings.datasets_dir/'unregistered.csv'
    pd.DataFrame({'x':[1,2],'y':[0,1]}).to_csv(path,index=False)
    with pytest.raises(IdentityError):
        TaskRepository(app.state.settings.db_path).get_task(task)
    with actor_context(app.state.settings.db_path,subjects['alice']['id']):
        with pytest.raises(IdentityError,match='unregistered'):
            DataBackend(app.state.settings.datasets_dir).read_frame(path)
    assert alice.get('/api/experiments').json()==[]
    assert alice.get('/api/unmounted-private-surface').status_code==403
    assert alice.get('/api/artifacts/unknown/preview').status_code in {403,404}
    assert alice.post('/api/tasks',json={'task_type':'strategy','model_name':'No paths','validator':'Synthetic',
                                        'source_dir':str(tmp_path)}).status_code==403


def test_subject_and_key_bindings_immutable(institution):
    app,_,subjects,_=institution
    with connect(app.state.settings.db_path) as conn:
        with pytest.raises(Exception,match='identity binding immutable'):
            conn.execute("UPDATE institution_subjects SET role='admin' WHERE id=?",(subjects['alice']['id'],))
        with pytest.raises(Exception,match='identity binding immutable'):
            conn.execute("UPDATE institution_issuer_keys SET public_key='changed'")
    store=IdentityStore(app.state.settings.db_path)
    store.revoke('subject',subjects['bob']['id'])
    with pytest.raises(IdentityError):
        authorize(app.state.settings.db_path,'task','missing',actor_id=subjects['bob']['id'])


def test_existing_native_plan_http_execution_and_current_subject(institution):
    from marvis.orchestrator.templates import get_template
    from marvis.orchestrator.contracts import PlanStatus,plan_fingerprint
    app,_,subjects,_=institution
    alice=_login(institution,'alice')
    task=_task(alice)
    with actor_context(app.state.settings.db_path,subjects['alice']['id'],task):
        plan=app.state.planner.from_template(get_template('sample_echo'),{'message':'synthetic identity carrier'},task)
        plan.status=PlanStatus.VALIDATED
        app.state.plan_repo.create_plan(plan)
        plan=app.state.plan_repo.load_plan(plan.id)
        fingerprint=plan_fingerprint(plan)
    confirmed=alice.post('/api/plans/'+plan.id+'/confirm',json={
        'expected_plan_status':plan.status.value,'expected_plan_revision':plan.replan_count,
        'expected_plan_fingerprint':fingerprint})
    assert confirmed.status_code==200,confirmed.text
    started=alice.post('/api/plans/'+plan.id+'/run')
    assert started.status_code==202,started.text
    finished=alice.get('/api/plans/'+plan.id)
    assert finished.status_code==200 and finished.json()['plan']['status']=='done',finished.text
    with actor_context(app.state.settings.db_path,subjects['alice']['id'],task):
        assert app.state.plan_repo.load_plan(plan.id).status==PlanStatus.DONE
        assert app.state.plan_repo.load_plan(plan.id).steps[0].tool_ref.plugin=='_sample'
    IdentityStore(app.state.settings.db_path).revoke('subject',subjects['alice']['id'])
    assert alice.get('/api/plans/'+plan.id).status_code==401


def test_existing_production_maker_checker_approval_and_list_resource_binding(institution):
    from marvis.repositories.strategy import StrategyRepository
    from marvis.packs.strategy.contracts import Strategy,StrategyRule
    from tests.test_production_governance_api import _promotion_payload
    app,_,subjects,_=institution
    alice,bob,admin=(_login(institution,s) for s in ('alice','bob','admin'))
    task=_task(alice)
    with actor_context(app.state.settings.db_path,subjects['alice']['id']):
        StrategyRepository(app.state.settings.db_path).create_strategy(task,Strategy(
            'institution-owned-strategy','approval',(StrategyRule('x > 0','approve',True),),
            'x','reject','Synthetic governed strategy'))
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE strategies SET asset_status='adopted_local',status='adopted' WHERE id='institution-owned-strategy'")
    response=alice.post('/api/production-governance/promotion-requests',json=_promotion_payload('institution-owned-strategy'))
    assert response.status_code==201,response.text
    request_id=response.json()['id']
    assert alice.get('/api/production-governance/me').status_code==200
    assert bob.get('/api/production-governance/me').status_code==200
    assert alice.get('/api/production-governance/promotion-requests?limit=1').json()['count']==1
    assert bob.get('/api/production-governance/promotion-requests?limit=1').json()['count']==0
    assert bob.get('/api/production-governance/promotion-requests/'+request_id).status_code==403
    assert bob.post('/api/production-governance/promotion-requests/'+request_id+'/approvals',json={
        'reason':'Synthetic review','decision':'approve'}).status_code==403
    grant_resource(app.state.settings.db_path,subjects['alice']['id'],'task',task,subjects['bob']['id'],['read','approve'],0)
    response=bob.post('/api/production-governance/promotion-requests/'+request_id+'/approvals',json={
        'reason':'Synthetic checker review','decision':'approve'})
    assert response.status_code==200,response.text
    assert admin.get('/api/production-governance/promotion-requests/'+request_id).status_code==403
    grant_resource(app.state.settings.db_path,subjects['alice']['id'],'task',task,subjects['admin']['id'],['read','approve'],0)
    response=admin.post('/api/production-governance/promotion-requests/'+request_id+'/approvals',json={
        'reason':'Synthetic administrator review','decision':'approve'})
    assert response.status_code==200,response.text
    assert response.json()['status']=='approved'


def test_content_hash_lookup_and_object_write_do_not_bypass_resource_policy(institution,tmp_path):
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository
    from marvis.repositories.modeling import ModelingRepository
    from marvis.packs.modeling.contracts import ModelArtifact
    app,_,subjects,_=institution
    task=_task(_login(institution,'alice'))
    source=app.state.settings.datasets_dir/task/'synthetic.parquet'
    source.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'x':[1,2,3]}).to_parquet(source,index=False)
    repo=DatasetRepository(app.state.settings.db_path)
    with actor_context(app.state.settings.db_path,subjects['alice']['id'],task):
        dataset=DatasetRegistry(repo,DataBackend(app.state.settings.datasets_dir),app.state.settings.datasets_dir).register_from_upload(task,source,role='sample')
        assert repo.find_dataset_by_content_hash(dataset.content_hash).id==dataset.id
    with actor_context(app.state.settings.db_path,subjects['bob']['id']):
        assert repo.find_dataset_by_content_hash(dataset.content_hash) is None
        with pytest.raises(IdentityError,match='identity_resource_forbidden'):
            ModelingRepository(app.state.settings.db_path).create_model_artifact(ModelArtifact(
                id='synthetic-forged',experiment_id='non-owned-experiment',algorithm='lr',model_path='synthetic.pkl',
                pmml_path=None,feature_list=('x',),feature_importance=None,params={},woe_maps=None,scorecard_table=None,
                created_at='2026-10-10T00:00:00+00:00'))


@pytest.mark.parametrize('revoked',[False,True])
def test_real_restart_recovery_retains_frozen_actor_and_stops_revoked_actor(institution,revoked,monkeypatch):
    from marvis.orchestrator.templates import get_template
    from marvis.orchestrator.contracts import PlanStatus
    app,_,subjects,_=institution
    task=_task(_login(institution,'alice'))
    with actor_context(app.state.settings.db_path,subjects['alice']['id'],task):
        plan=app.state.planner.from_template(get_template('sample_echo'),{'message':'synthetic restart'},task)
        plan.status=PlanStatus.VALIDATED
        app.state.plan_repo.create_plan(plan)
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE plans SET status='running' WHERE id=?",(plan.id,))
        conn.execute("UPDATE plan_steps SET status='running' WHERE plan_id=?",(plan.id,))
    if revoked:
        IdentityStore(app.state.settings.db_path).revoke('subject',subjects['alice']['id'])
    # Neither branch is authorized to re-run the interrupted native Tool.
    from marvis.plugins.runner import ToolRunner
    def no_dispatch(*args,**kwargs):
        raise AssertionError('restart must not dispatch an interrupted Tool')
    monkeypatch.setattr(ToolRunner,'invoke',no_dispatch)
    restarted=create_app(app.state.settings.workspace)
    with connect(restarted.state.settings.db_path) as conn:
        row=conn.execute('SELECT status FROM plans WHERE id=?',(plan.id,)).fetchone()
        assert row['status']=='failed'
        if revoked:
            step=conn.execute('SELECT status,error FROM plan_steps WHERE plan_id=?',(plan.id,)).fetchone()
            assert step['status']=='failed' and 'identity runtime authorization revoked' in step['error']
        assert conn.execute('SELECT subject_id FROM institution_plan_actors WHERE plan_id=?',(plan.id,)).fetchone()[0]==subjects['alice']['id']


def test_institutional_mode_forces_signed_admission_and_service_cannot_confirm(institution):
    app,_,subjects,_=institution
    admin=_login(institution,'admin')
    response=admin.put('/api/plugins/governance/settings',headers={'x-marvis-plugin-admin':app.state.plugin_admin_token},
                       json={'mode':'trusted_local','expected_revision':1,'reason':'Synthetic mode downgrade probe'})
    assert response.status_code==403,response.text
    assert response.json()['detail']=='institution_requires_signed_capabilities'
    service=_login(institution,'automation')
    task=_task(service,'Synthetic service-owned strategy')
    from marvis.orchestrator.templates import get_template
    from marvis.orchestrator.contracts import PlanStatus
    with actor_context(app.state.settings.db_path,subjects['automation']['id'],task):
        plan=app.state.planner.from_template(get_template('sample_echo'),{'message':'Synthetic service'},task)
        plan.status=PlanStatus.VALIDATED
        app.state.plan_repo.create_plan(plan)
        with pytest.raises(IdentityError,match='identity_resource_forbidden'):
            app.state.plan_repo.confirm_plan(plan.id)
    assert service.post('/api/plans/'+plan.id+'/confirm',json={}).status_code==403


def test_native_browser_metadata_and_sync_catalog_admission(institution):
    # The browser reads the catalog's exact root, then sends explicit actions to
    # child routes. Admission must retain the router's human-admin check.
    admin=_login(institution,'admin')
    assert admin.get('/api/branding').status_code==200
    catalog=admin.get('/api/instance-sync')
    assert catalog.status_code==200 and 'versions' in catalog.json(),catalog.text
    maker=_login(institution,'alice')
    assert maker.get('/api/branding').status_code==200
    refusal=maker.get('/api/instance-sync')
    assert refusal.status_code==403
    assert refusal.json()['detail']['code']=='identity_admin_required'
    assert maker.get('/api/unknown-browser-metadata').status_code==403
